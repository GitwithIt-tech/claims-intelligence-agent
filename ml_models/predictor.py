"""
ml_models/predictor.py
───────────────────────
Loads trained models ONCE at module import time.
Avoids repeated disk reads that cause memory issues on Mac.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "saved_models"

TYPE_MAP   = {"vehicle":0,"property":1,"liability":2,"health":3,"travel":4}
STATUS_MAP = {"open":0,"closed":1,"in_review":2,"litigated":3}
REGION_MAP = {"London":0,"Manchester":1,"Birmingham":2,"Leeds":3,
              "Belfast":4,"Edinburgh":5,"Bristol":6,"Cardiff":7,
              "Liverpool":8,"Sheffield":9}


def _load(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Run: python3 ml_models/train_models.py"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Load once at import time ──────────────────────────────────────────────────
print("  Loading ML models into memory...")
_FRAUD_MODEL      = _load("fraud_detector.pkl")
_LITIGATION_MODEL = _load("litigation_predictor.pkl")
_RESOLUTION_MODEL = _load("resolution_forecaster.pkl")
print("  ✓ All models loaded")


def _build_row(claim: dict) -> pd.DataFrame:
    claim_date    = pd.to_datetime(claim.get("claim_date", "2023-01-01"))
    incident_date = pd.to_datetime(claim.get("incident_date", "2023-01-01"))
    claim_amount  = float(claim.get("claim_amount", 10000))
    res_days      = float(claim.get("resolution_days", 90))

    row = {
        "log_claim_amount": np.log1p(claim_amount),
        "amount_per_day":   claim_amount / (res_days + 1),
        "resolution_days":  res_days,
        "days_to_report":   max((claim_date - incident_date).days, 0),
        "claim_month":      claim_date.month,
        "claim_quarter":    claim_date.quarter,
        "claim_dow":        claim_date.dayofweek,
        "claim_type_enc":   TYPE_MAP.get(claim.get("claim_type", "vehicle"), 0),
        "status_enc":       STATUS_MAP.get(claim.get("status", "open"), 0),
        "region_enc":       REGION_MAP.get(claim.get("region", "London"), 0),
        "adjuster_id_enc":  int(str(claim.get("adjuster_id", "ADJ0001")).replace("ADJ","")) % 40,
    }
    return pd.DataFrame([row])


def predict_claim(claim: dict) -> dict:
    """
    Input:  any claim dict
    Output: fraud score, litigation score, resolution forecast, risk tier
    """
    X = _build_row(claim)

    fraud_prob = float(_FRAUD_MODEL.predict_proba(X)[0][1])
    lit_prob   = float(_LITIGATION_MODEL.predict_proba(X)[0][1])
    res_days   = float(_RESOLUTION_MODEL.predict(X)[0])

    risk = (
        "high"   if fraud_prob > 0.6 or lit_prob > 0.6 else
        "medium" if fraud_prob > 0.3 or lit_prob > 0.3 else
        "low"
    )

    return {
        "fraud_score":              round(fraud_prob, 4),
        "fraud_flag":               fraud_prob > 0.5,
        "litigation_score":         round(lit_prob, 4),
        "litigation_flag":          lit_prob > 0.5,
        "resolution_days_forecast": int(round(res_days)),
        "risk_tier":                risk,
    }


if __name__ == "__main__":
    test_claim = {
        "claim_type":    "liability",
        "claim_amount":  85000,
        "status":        "open",
        "region":        "London",
        "adjuster_id":   "ADJ0001",
        "claim_date":    "2024-03-15",
        "incident_date": "2024-03-01",
        "resolution_days": 120,
    }
    result = predict_claim(test_claim)
    print("\nTest prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")