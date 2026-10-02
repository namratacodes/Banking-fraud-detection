"""
Inference logic for the real-time fraud API.

Loads the model saved in Phase 4 (models/fraud_model.pkl) and the same
business-label translation used in Phase 5's SHAP explainer, so a single
prediction returns:
  - a fraud probability
  - a risk level bucket (LOW / MEDIUM / HIGH)
  - a flagged boolean (using the threshold tuned in Phase 4)
  - the top 3 plain-English reasons it was flagged
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap

# --- Business-friendly labels for PCA components (same mapping as Phase 5) ---
FEATURE_LABELS = {
    "V1": "irregular transaction indicator", "V2": "unusual account activity signal",
    "V3": "atypical account usage pattern", "V4": "transaction pattern anomaly",
    "V5": "unusual spending signature", "V6": "atypical transaction behavior",
    "V7": "unusual merchant/transaction signal", "V8": "atypical transaction marker",
    "V9": "unusual account signal", "V10": "deviation from normal account behavior",
    "V11": "elevated risk pattern", "V12": "atypical merchant/category pattern",
    "V13": "irregular usage pattern", "V14": "irregular transaction signature",
    "V15": "atypical transaction marker", "V16": "uncommon transaction characteristics",
    "V17": "unusual spending pattern", "V18": "uncommon account activity",
    "V19": "unusual spending signal", "V20": "irregular account pattern",
    "V21": "atypical merchant signal", "V22": "unusual transaction marker",
    "V23": "irregular spending indicator", "V24": "atypical account behavior",
    "V25": "unusual account marker", "V26": "irregular transaction indicator",
    "V27": "atypical spending signal", "V28": "unusual account pattern",
    "Amount": "transaction amount", "txn_velocity_1h": "high transaction velocity in the last hour",
    "amount_dev_from_user_mean": "amount far from this account's typical spend",
    "amount_zscore_user": "statistically unusual transaction amount",
    "rolling_avg_amount_5": "deviation from recent spending average",
    "is_night": "transaction occurred during unusual hours",
}

MODEL_PATH = Path(__file__).parent.parent / "models" / "fraud_model.pkl"


class FraudPredictor:
    """Loads the trained model once and serves predictions + explanations."""

    def __init__(self, model_path: Path = MODEL_PATH):
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        self.model = saved["model"]
        self.model_name = saved["model_name"]
        self.threshold = saved["threshold"]
        self.feature_names = saved["feature_names"]
        # TreeExplainer is built once at startup, not per-request, since
        # constructing it has fixed overhead independent of the input.
        self.explainer = shap.TreeExplainer(self.model)

    def _engineer_features(self, raw: dict) -> pd.DataFrame:
        """
        Build a single-row DataFrame matching the model's expected feature
        columns. Velocity/rolling-average features need transaction HISTORY
        for a given user, which a single API call doesn't have, so for a
        standalone request we default those to neutral values (0). In a real
        deployment, these would be looked up from a live feature store keyed
        by card/account ID.
        """
        row = {k: raw[k] for k in [
            "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
            "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
            "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
        ]}

        seconds_in_day = 24 * 60 * 60
        hour_of_day = (row["Time"] % seconds_in_day) // 3600
        row["hour_of_day"] = hour_of_day
        row["day_of_week"] = (row["Time"] // seconds_in_day) % 7
        row["is_night"] = 1 if (hour_of_day < 6 or hour_of_day >= 22) else 0

        # No transaction history available in a stateless single-request API,
        # so these default to neutral (0 = "no deviation detected").
        row["txn_velocity_1h"] = 0
        row["amount_dev_from_user_mean"] = 0
        row["amount_zscore_user"] = 0
        row["rolling_avg_amount_5"] = row["Amount"]

        df = pd.DataFrame([row])
        return df[self.feature_names]  # enforce exact column order the model expects

    def _risk_level(self, probability: float) -> str:
        if probability < 0.3:
            return "LOW"
        elif probability < 0.7:
            return "MEDIUM"
        return "HIGH"

    def predict(self, raw_transaction: dict, top_n: int = 3) -> dict:
        X = self._engineer_features(raw_transaction)

        probability = float(self.model.predict_proba(X)[0, 1])
        flagged = probability >= self.threshold
        risk_level = self._risk_level(probability)

        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_values = np.array(shap_values).flatten()

        contributions = list(zip(self.feature_names, shap_values))
        fraud_pushing = [c for c in contributions if c[1] > 0]
        fraud_pushing.sort(key=lambda x: x[1], reverse=True)
        top = fraud_pushing[:top_n]

        reason_details = [
            {"feature": feat, "label": FEATURE_LABELS.get(feat, feat), "shap_value": round(float(val), 4)}
            for feat, val in top
        ]
        top_reasons = [r["label"] for r in reason_details]

        return {
            "fraud_probability": round(probability, 4),
            "risk_level": risk_level,
            "flagged": bool(flagged),
            "top_reasons": top_reasons,
            "reason_details": reason_details,
            "model_used": self.model_name,
            "threshold_used": self.threshold,
        }