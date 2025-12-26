import json
import joblib
import shap
import pandas as pd
import logging
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

FEATURE_ORDER = [
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

# --------------------------------------------------
# App setup
# --------------------------------------------------
app = FastAPI(title="Fraud Detection Service", version="1.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud-api")

# --------------------------------------------------
# Load artifacts at startup
# --------------------------------------------------
MODEL_PATH = "artifacts/xgb_model.pkl"
THRESHOLD_PATH = "artifacts/decision_threshold.json"

model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as f:
    threshold_cfg = json.load(f)

THRESHOLD = threshold_cfg["threshold"]

# SHAP explainer (initialized once)
explainer = shap.TreeExplainer(model)

# --------------------------------------------------
# Input schema
# --------------------------------------------------
class Transaction(BaseModel):
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
    Amount: float = Field(..., gt=0)

# --------------------------------------------------
# Inference endpoint
# --------------------------------------------------
@app.post("/predict")
def predict(txn: Transaction):
    try:
        txn_dict = txn.model_dump()

        X = pd.DataFrame(
            [[txn_dict[col] for col in FEATURE_ORDER]],
            columns=FEATURE_ORDER
        )

        prob = float(model.predict_proba(X)[0][1])
        is_fraud = prob >= THRESHOLD

        logger.info({
            "event": "prediction",
            "fraud_probability": round(prob, 6),
            "decision": bool(is_fraud),
            "threshold": THRESHOLD
        })

        return {
            "fraud_probability": round(prob, 6),
            "is_fraud": bool(is_fraud),
            "threshold": THRESHOLD
        }

    except Exception as e:
        logger.exception("Inference failed")
        return {"error": str(e)}

# --------------------------------------------------# 
# Explanation endpoint
# --------------------------------------------------
@app.post("/explain")
def explain(txn: Transaction):
    try:
        txn_dict = txn.model_dump()

        X = pd.DataFrame(
            [[txn_dict[col] for col in FEATURE_ORDER]],
            columns=FEATURE_ORDER
        )

        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            fraud_shap = shap_values[1][0]
        else:
            fraud_shap = shap_values[0]

        top_idx = abs(fraud_shap).argsort()[-5:][::-1]

        explanation = {
            FEATURE_ORDER[i]: round(float(fraud_shap[i]), 6)
            for i in top_idx
        }

        prob = float(model.predict_proba(X)[0][1])

        return {
            "fraud_probability": round(prob, 6),
            "threshold": THRESHOLD,
            "decision": prob >= THRESHOLD,
            "top_contributing_features": explanation
        }

    except Exception as e:
        logger.exception("Explanation failed")
        return {"error": str(e)}

# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}