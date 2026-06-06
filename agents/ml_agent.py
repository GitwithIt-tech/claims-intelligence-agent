"""
agents/ml_agent.py
───────────────────
ML Agent — runs fraud, litigation, and resolution
predictions for a given claim.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml_models.predictor import predict_claim
from sqlalchemy import create_engine, text
from config.setting import db_settings

engine = create_engine(db_settings.url)


def get_claim_from_db(claim_id: str) -> dict | None:
    """Fetch a single claim row from PostgreSQL by claim_id."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM claims WHERE claim_id = :id"),
            {"id": claim_id}
        ).fetchone()
    if row is None:
        return None
    return dict(row._mapping)


def run_ml_agent(claim_id: str = None, claim_data: dict = None) -> dict:
    """
    Run ML scoring on a claim.
    Accepts either a claim_id (looks up DB) or raw claim_data dict.
    """
    try:
        if claim_id:
            claim = get_claim_from_db(claim_id)
            if claim is None:
                return {
                    "success": False,
                    "agent":   "ml",
                    "answer":  f"Claim {claim_id} not found in database.",
                    "scores":  {},
                }
        elif claim_data:
            claim = claim_data
        else:
            return {
                "success": False,
                "agent":   "ml",
                "answer":  "No claim_id or claim_data provided.",
                "scores":  {},
            }

        scores = predict_claim(claim)

        answer = (
            f"ML Scoring Results:\n"
            f"  Fraud Score:       {scores['fraud_score']:.2%} "
            f"({'⚠ FLAGGED' if scores['fraud_flag'] else '✓ OK'})\n"
            f"  Litigation Score:  {scores['litigation_score']:.2%} "
            f"({'⚠ FLAGGED' if scores['litigation_flag'] else '✓ OK'})\n"
            f"  Resolution Forecast: {scores['resolution_days_forecast']} days\n"
            f"  Risk Tier:         {scores['risk_tier'].upper()}"
        )

        return {
            "success": True,
            "agent":   "ml",
            "answer":  answer,
            "scores":  scores,
            "claim":   claim,
        }

    except Exception as e:
        return {
            "success": False,
            "agent":   "ml",
            "answer":  f"ML agent error: {str(e)}",
            "scores":  {},
        }


if __name__ == "__main__":
    # Test with first claim in DB
    result = run_ml_agent(claim_id="CLM0000001")
    print(result["answer"])