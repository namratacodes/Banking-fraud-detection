"""
Phase 3: Handling class imbalance.

Fraud is 0.17% of transactions. A model trained naively will just learn to
predict "legit" every time and still score 99.8% accuracy — useless.

This script compares three standard strategies for imbalanced classification,
using a held-out test set that is NEVER resampled (resampling the test set
would let the model "cheat" by evaluating on synthetic data):

  1. SMOTE            - synthesize new minority (fraud) samples by interpolating
                         between real fraud examples and their neighbors
  2. ADASYN            - like SMOTE, but generates more synthetic samples in
                         regions where fraud is harder to separate from legit
  3. class_weight      - no resampling; instead, tell the model to penalize
                         mistakes on fraud examples more heavily during training

For each strategy we train the SAME base model (XGBoost) and compare:
  - ROC-AUC            - good general ranking metric, but can look
                         over-optimistic under heavy imbalance
  - PR-AUC              - the metric that matters most here, since it focuses
                         on precision/recall for the rare positive class
  - F1 (at 0.5 threshold) - harmonic mean of precision & recall

Why not just use raw accuracy? See eda.py — predicting "all legit" already
gets 99.8% accuracy while catching zero fraud. Accuracy cannot distinguish
a useless model from a useful one here.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE, ADASYN
import xgboost as xgb

RANDOM_STATE = 42


def load_data(path="../data/processed/creditcard_features.csv"):
    df = pd.read_csv(path)
    drop_cols = ["Class", "pseudo_user"]  # pseudo_user is an ID, not a predictive feature
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["Class"]
    return X, y


def train_and_eval(X_train, y_train, X_test, y_test, strategy_name, sample_weight=None):
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    f1 = f1_score(y_test, preds)

    print(f"\n=== {strategy_name} ===")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")
    print(f"F1      : {f1:.4f}")
    print(classification_report(y_test, preds, target_names=["Legit", "Fraud"], digits=4))

    return {"strategy": strategy_name, "roc_auc": roc_auc, "pr_auc": pr_auc, "f1": f1}


def main():
    X, y = load_data()

    # Single held-out test set shared across all strategies for a fair comparison.
    # stratify=y keeps the same ~0.17% fraud rate in both train and test splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results = []

    # Baseline: no imbalance handling at all, for reference
    results.append(train_and_eval(X_train, y_train, X_test, y_test, "Baseline (no handling)"))

    # Strategy 1: SMOTE oversampling (train set only)
    X_sm, y_sm = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
    results.append(train_and_eval(X_sm, y_sm, X_test, y_test, "SMOTE"))

    # Strategy 2: ADASYN oversampling (train set only)
    X_ada, y_ada = ADASYN(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
    results.append(train_and_eval(X_ada, y_ada, X_test, y_test, "ADASYN"))

    # Strategy 3: class weighting (no resampling, just reweight the loss)
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    sample_weight = np.where(y_train == 1, fraud_ratio, 1.0)
    results.append(
        train_and_eval(X_train, y_train, X_test, y_test, "Class Weighting", sample_weight=sample_weight)
    )

    # Summary table
    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    print("\n\n=== STRATEGY COMPARISON (sorted by PR-AUC) ===")
    print(results_df.to_string(index=False))
    results_df.to_csv("../reports/imbalance_strategy_comparison.csv", index=False)
    print("\nSaved comparison to reports/imbalance_strategy_comparison.csv")

    # 5-fold stratified cross-validation on the winning strategy, to confirm
    # the result isn't a fluke of one particular train/test split
    best_strategy = results_df.iloc[0]["strategy"]
    print(f"\nRunning 5-fold StratifiedKFold CV to validate best strategy: {best_strategy}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        if best_strategy == "Class Weighting":
            fr = (y_tr == 0).sum() / (y_tr == 1).sum()
            sw = np.where(y_tr == 1, fr, 1.0)
            model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                       eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1)
            model.fit(X_tr, y_tr, sample_weight=sw)
        else:
            resampler = SMOTE(random_state=RANDOM_STATE) if best_strategy == "SMOTE" else ADASYN(random_state=RANDOM_STATE)
            X_res, y_res = resampler.fit_resample(X_tr, y_tr)
            model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                       eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1)
            model.fit(X_res, y_res)
        probs = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, probs)
        cv_scores.append(pr_auc)
        print(f"  Fold {fold}: PR-AUC = {pr_auc:.4f}")

    print(f"\nMean PR-AUC across folds: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")


if __name__ == "__main__":
    main()