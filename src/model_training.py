"""
Phase 4: Model training & comparison.

Phase 3 showed class weighting beats SMOTE/ADASYN for this data, so every
model here uses class weighting (no resampling) rather than a modified
training set.

We train three models:
  1. XGBoost        - primary candidate, gradient-boosted trees
  2. LightGBM       - faster gradient-boosted trees, often competitive with XGBoost
  3. Isolation Forest - unsupervised anomaly detector; doesn't use fraud labels
                        to learn, just isolates points that look "weird".
                        Included as a sanity-check baseline: if a label-free
                        method gets close to the supervised models, it tells
                        you fraud is structurally very different from normal
                        transactions. It usually underperforms supervised
                        models when labels are available, which is expected.

We compare all three on the SAME held-out test set using:
  - Precision-Recall curve (most informative under heavy imbalance)
  - ROC-AUC, PR-AUC, F1
Then we tune the decision threshold of the winning model: the default 0.5
cutoff is rarely optimal for fraud, since the cost of a missed fraud
(false negative) and a wrongly-blocked legit customer (false positive) are
not equal from a business standpoint.
"""

import json
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, precision_score, recall_score, confusion_matrix
)
from sklearn.ensemble import IsolationForest
import xgboost as xgb

# LightGBM is optional: on some Windows setups it crashes with a native
# "access violation" during training due to a broken local install (not a
# code/data issue). We try to import it, and skip it gracefully if it's
# unavailable or broken, rather than blocking the whole pipeline on it.
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

RANDOM_STATE = 42


def load_data(path="../data/processed/creditcard_features.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=[c for c in ["Class", "pseudo_user"] if c in df.columns])
    y = df["Class"]
    return X, y


def get_sample_weight(y):
    fraud_ratio = (y == 0).sum() / (y == 1).sum()
    return np.where(y == 1, fraud_ratio, 1.0)


def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=get_sample_weight(y_train))
    return model


def train_lightgbm(X_train, y_train):
    # LightGBM on Windows can throw an "access violation" OSError if it's
    # handed a pandas Series with a non-contiguous index (exactly what
    # train_test_split produces) or non-float64 label dtypes. Converting to
    # plain numpy arrays with a clean index avoids this reliably.
    X_train_np = X_train.reset_index(drop=True).to_numpy()
    y_train_np = y_train.reset_index(drop=True).to_numpy().astype(np.float64)
    weights = get_sample_weight(y_train).astype(np.float64)

    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train_np, y_train_np, sample_weight=weights)
    return model


def train_isolation_forest(X_train, y_train):
    # Unsupervised: trained WITHOUT labels. contamination is set to roughly
    # the known fraud rate as a prior on how many points should be flagged.
    fraud_rate = y_train.mean()
    model = IsolationForest(
        n_estimators=200, contamination=fraud_rate, random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train)
    return model


def get_scores(model, X_test, is_isolation_forest=False):
    if is_isolation_forest:
        # decision_function: higher = more normal, lower = more anomalous.
        # Flip and rescale to 0-1 so higher = more fraud-like, matching the
        # other models' predict_proba convention.
        raw = model.decision_function(X_test)
        scores = (raw.max() - raw) / (raw.max() - raw.min())
        return scores
    return model.predict_proba(X_test)[:, 1]


def evaluate(name, y_test, scores, threshold=0.5):
    preds = (scores >= threshold).astype(int)
    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)
    f1 = f1_score(y_test, preds)
    print(f"{name:20s} | ROC-AUC {roc_auc:.4f} | PR-AUC {pr_auc:.4f} | F1 {f1:.4f}")
    return {"model": name, "roc_auc": roc_auc, "pr_auc": pr_auc, "f1": f1}


def plot_pr_curves(results_scores, y_test, out_path="../reports/pr_curve_comparison.png"):
    plt.figure(figsize=(7, 6))
    for name, scores in results_scores.items():
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_auc = average_precision_score(y_test, scores)
        plt.plot(recall, precision, label=f"{name} (AP={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved PR curve comparison to {out_path}")


def tune_threshold(y_test, scores, out_path="../reports/threshold_tuning.png"):
    """
    Sweep thresholds and report precision/recall/F1 at each, so the business
    can choose a threshold based on their tolerance for false positives
    (blocked legit customers) vs false negatives (missed fraud), rather than
    blindly using 0.5.
    """
    thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        rows.append({"threshold": round(t, 2), "precision": p, "recall": r, "f1": f1})

    df = pd.DataFrame(rows)
    best_f1_row = df.loc[df["f1"].idxmax()]

    plt.figure(figsize=(8, 5))
    plt.plot(df["threshold"], df["precision"], label="Precision", marker="o")
    plt.plot(df["threshold"], df["recall"], label="Recall", marker="o")
    plt.plot(df["threshold"], df["f1"], label="F1", marker="o")
    plt.axvline(best_f1_row["threshold"], color="gray", linestyle="--",
                label=f"Best F1 @ {best_f1_row['threshold']}")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Tuning: Precision / Recall / F1 Trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved threshold tuning plot to {out_path}")
    print(f"\nBest F1 threshold: {best_f1_row['threshold']} "
          f"(Precision={best_f1_row['precision']:.3f}, Recall={best_f1_row['recall']:.3f}, "
          f"F1={best_f1_row['f1']:.3f})")
    df.to_csv("../reports/threshold_sweep.csv", index=False)
    return best_f1_row["threshold"], df


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)

    models = {"XGBoost": xgb_model}
    scores = {"XGBoost": get_scores(xgb_model, X_test)}

    if LIGHTGBM_AVAILABLE:
        try:
            print("Training LightGBM...")
            lgb_model = train_lightgbm(X_train, y_train)
            models["LightGBM"] = lgb_model
            scores["LightGBM"] = get_scores(lgb_model, X_test)
        except OSError as e:
            print(f"LightGBM failed to train on this machine ({e}). "
                  f"Skipping it and continuing with XGBoost + Isolation Forest only.")
    else:
        print("LightGBM not installed/available. Skipping it.")

    print("Training Isolation Forest (unsupervised)...")
    iso_model = train_isolation_forest(X_train, y_train)
    models["IsolationForest"] = iso_model
    scores["IsolationForest"] = get_scores(iso_model, X_test, is_isolation_forest=True)

    print("\n=== Model Comparison (threshold=0.5) ===")
    results = [evaluate(name, y_test, s) for name, s in scores.items()]
    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    print("\n" + results_df.to_string(index=False))
    results_df.to_csv("../reports/model_comparison.csv", index=False)

    plot_pr_curves(scores, y_test)

    best_model_name = results_df.iloc[0]["model"]
    print(f"\nBest model by PR-AUC: {best_model_name}")

    best_model = models[best_model_name]
    best_scores = scores[best_model_name]

    # Threshold tuning only makes sense for the supervised winner
    best_threshold, threshold_df = tune_threshold(y_test, best_scores)

    # Confusion matrix at tuned threshold
    final_preds = (best_scores >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, final_preds)
    print(f"\nConfusion matrix at threshold={best_threshold}:")
    print(f"              Predicted Legit  Predicted Fraud")
    print(f"Actual Legit  {cm[0][0]:>15d}  {cm[0][1]:>15d}")
    print(f"Actual Fraud  {cm[1][0]:>15d}  {cm[1][1]:>15d}")

    # Save the best model + metadata for Phase 5 (SHAP) and Phase 6 (API)
    with open("../models/fraud_model.pkl", "wb") as f:
        pickle.dump({
            "model": best_model,
            "model_name": best_model_name,
            "threshold": float(best_threshold),
            "feature_names": list(X.columns),
        }, f)
    print(f"\nSaved best model ({best_model_name}) to ../models/fraud_model.pkl")

    with open("../reports/model_training_summary.json", "w") as f:
        json.dump({
            "best_model": best_model_name,
            "best_threshold": float(best_threshold),
            "metrics_at_0.5": results,
        }, f, indent=2)


if __name__ == "__main__":
    main()