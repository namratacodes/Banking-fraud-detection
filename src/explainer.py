"""
Phase 5: Explainability with SHAP.

A fraud model that just outputs "0.87 probability of fraud" is not useful to
a risk analyst who has to decide whether to call the customer or release the
transaction. This script adds two layers of explanation on top of the model
saved in Phase 4:

  1. GLOBAL SHAP   - across ALL transactions, which features drive fraud
                      predictions the most, on average? Answers: "what does
                      our model generally key off of?"

  2. LOCAL SHAP    - for ONE specific transaction, why did the model flag
                      (or clear) it? Answers: "why THIS transaction?"

The local explanation is then translated from raw feature names (V14, V10...)
into a plain-English business reason, e.g.:
  "Flagged because: unusual transaction pattern (V14) + amount 4x higher
   than typical (V4) + high transaction velocity in the last hour"

This business-translation step is what separates a notebook demo from
something a real fraud ops team could actually use.
"""

import pickle
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# Human-readable labels for the PCA components that SHAP found most
# predictive in Phase 1's correlation analysis. For a real (non-PCA) dataset
# these would map to actual business features (e.g. "transaction_amount",
# "merchant_risk_score"); here we label them based on what they correlate
# with, so the explanation reads naturally to a risk analyst.
FEATURE_LABELS = {
    "V17": "unusual spending pattern",
    "V14": "irregular transaction signature",
    "V12": "atypical merchant/category pattern",
    "V10": "deviation from normal account behavior",
    "V16": "uncommon transaction characteristics",
    "V7": "unusual merchant/transaction signal",
    "V3": "atypical account usage pattern",
    "V1": "irregular transaction indicator",
    "V5": "unusual spending signature",
    "V6": "atypical transaction behavior",
    "V9": "unusual account signal",
    "V13": "irregular usage pattern",
    "V15": "atypical transaction marker",
    "V18": "uncommon account activity",
    "V19": "unusual spending signal",
    "V20": "irregular account pattern",
    "V21": "atypical merchant signal",
    "V22": "unusual transaction marker",
    "V23": "irregular spending indicator",
    "V24": "atypical account behavior",
    "V25": "unusual account marker",
    "V26": "irregular transaction indicator",
    "V27": "atypical spending signal",
    "V28": "unusual account pattern",
    "V11": "elevated risk pattern",
    "V4": "transaction pattern anomaly",
    "V2": "unusual account activity signal",
    "txn_velocity_1h": "high transaction velocity in the last hour",
    "amount_dev_from_user_mean": "amount far from this account's typical spend",
    "amount_zscore_user": "statistically unusual transaction amount",
    "rolling_avg_amount_5": "deviation from recent spending average",
    "is_night": "transaction occurred during unusual hours",
    "Amount": "transaction amount",
}


def humanize_feature(feat_name: str) -> str:
    return FEATURE_LABELS.get(feat_name, feat_name)


def load_model(path="../models/fraud_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_global_shap(model, X_sample, feature_names, out_path="../reports/shap_summary.png"):
    """
    Global explainability: averaged SHAP values across a sample of
    transactions, showing which features matter most to the model overall.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # TreeExplainer on a binary XGBoost/LightGBM classifier returns shap
    # values for the positive (fraud) class directly, or a list [class0, class1]
    # depending on model type/version - handle both.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved global SHAP summary plot to {out_path}")

    # Also save a ranked list of mean |SHAP value| per feature, for the
    # dashboard's "Model Performance" page
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "business_label": [humanize_feature(f) for f in feature_names],
    }).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv("../reports/shap_feature_importance.csv", index=False)
    print("Top 10 globally important features:")
    print(importance_df.head(10).to_string(index=False))

    return explainer, shap_values, importance_df


def explain_transaction(explainer, transaction_row, feature_names, top_n=3):
    """
    Local explainability: for ONE transaction (a single-row DataFrame or
    array matching feature_names), return the top N features pushing the
    prediction toward fraud, translated into plain business language.
    """
    shap_values = explainer.shap_values(transaction_row)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.array(shap_values).flatten()

    # Rank by the SIGNED shap value - we only want reasons pushing TOWARD
    # fraud (positive contribution), not away from it
    contributions = list(zip(feature_names, shap_values))
    fraud_pushing = [c for c in contributions if c[1] > 0]
    fraud_pushing.sort(key=lambda x: x[1], reverse=True)

    top_reasons = fraud_pushing[:top_n]
    reasons_text = [humanize_feature(feat) for feat, _ in top_reasons]

    return {
        "top_reasons": reasons_text,
        "raw_contributions": [
            {"feature": feat, "shap_value": round(float(val), 4), "label": humanize_feature(feat)}
            for feat, val in top_reasons
        ],
    }


def main():
    saved = load_model()
    model = saved["model"]
    feature_names = saved["feature_names"]
    threshold = saved["threshold"]

    df = pd.read_csv("../data/processed/creditcard_features.csv")
    X = df[feature_names]
    y = df["Class"]

    # Use a sample for the global SHAP summary - computing SHAP on all 284K
    # rows is slow and unnecessary for a stable average; 2000 rows (with all
    # known frauds included) gives a representative picture.
    fraud_idx = df[df["Class"] == 1].index
    legit_sample_idx = df[df["Class"] == 0].sample(n=1500, random_state=42).index
    sample_idx = fraud_idx.union(legit_sample_idx)
    X_sample = X.loc[sample_idx]

    print(f"Computing global SHAP values on {len(X_sample)} transactions "
          f"({len(fraud_idx)} fraud + 1500 legit sample)...")
    explainer, shap_values, importance_df = compute_global_shap(model, X_sample, feature_names)

    # Demonstrate local explanation on a few real flagged fraud transactions
    print("\n=== Example local explanations (real fraud cases) ===")
    example_frauds = df[df["Class"] == 1].sample(n=3, random_state=1)
    examples_out = []
    for idx, row in example_frauds.iterrows():
        txn = X.loc[[idx]]
        prob = model.predict_proba(txn)[0, 1]
        explanation = explain_transaction(explainer, txn, feature_names)
        reason_str = " + ".join(explanation["top_reasons"])
        print(f"\nTransaction #{idx} | Amount: {df.loc[idx, 'Amount']:.2f} | "
              f"Fraud probability: {prob:.3f} | Flagged: {prob >= threshold}")
        print(f"  Reasons: {reason_str}")
        examples_out.append({
            "transaction_id": int(idx),
            "amount": float(df.loc[idx, "Amount"]),
            "fraud_probability": round(float(prob), 4),
            "flagged": bool(prob >= threshold),
            "top_reasons": explanation["top_reasons"],
            "raw_contributions": explanation["raw_contributions"],
        })

    with open("../reports/shap_example_explanations.json", "w") as f:
        json.dump(examples_out, f, indent=2)
    print("\nSaved example explanations to reports/shap_example_explanations.json")


if __name__ == "__main__":
    main()