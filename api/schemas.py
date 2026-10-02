"""
Request/response schemas for the fraud detection API.

The request mirrors the raw Kaggle creditcard.csv columns (Time, V1-V28,
Amount) since that's what the model was trained on. In a real bank's system,
an upstream service would compute these PCA features and the engineered
features (velocity, deviation) in real time before calling this endpoint;
here we recompute the engineered features inside the API itself so the
endpoint is usable standalone with just the raw transaction fields.
"""

from pydantic import BaseModel, Field
from typing import List


class TransactionRequest(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since the first transaction in the dataset")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount")

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 406, "V1": -2.3122, "V2": 1.9519, "V3": -1.6098, "V4": 3.9979,
                "V5": -0.5221, "V6": -1.4265, "V7": -2.5373, "V8": 1.3916, "V9": -2.7700,
                "V10": -2.7722, "V11": 3.2020, "V12": -2.8999, "V13": -0.5952, "V14": -4.2892,
                "V15": 0.3897, "V16": -1.1407, "V17": -2.8301, "V18": -0.0168, "V19": 0.4169,
                "V20": 0.1267, "V21": 0.5172, "V22": -0.0350, "V23": -0.4652, "V24": 0.3202,
                "V25": 0.0445, "V26": 0.1778, "V27": 0.2611, "V28": -0.1433, "Amount": 0.0
            }
        }


class ReasonDetail(BaseModel):
    feature: str
    label: str
    shap_value: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    risk_level: str  # LOW, MEDIUM, HIGH
    flagged: bool
    top_reasons: List[str]
    reason_details: List[ReasonDetail]
    model_used: str
    threshold_used: float