"""
agents/ml_agent.py
Uses the already-loaded models from predictor.py — never reloads.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, text
from config.setting import db_settings

engine = create_engine(db_settings.url, pool_pre_ping=True, pool_recycle=300)

# Import predict_claim — models already loaded at module level in predictor.py
from ml_models.predictor import predict_claim


def get_claim_from_db(claim_id: str) -> dict | None:
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM claims WHERE claim_id = :id"),
                {"id": claim_id}
            ).fetchone()
        return dict(row._mapping) if row else None
    except Exception:
        return None


def run_ml_agent(claim_id: str = None, claim_data: dict = None) -> dict:
    try:
        if claim_id:
            claim = get_claim_from_db(claim_id)
            if claim is None:
                return {
                    "success": False, "agent": "ml",
                    "answer": f"Claim {claim_id} not found.",
                    "scores": {},
                }
        elif claim_data:
            claim = claim_data
        else:
            return {
                "success": False, "agent": "ml",
                "answer": "No claim_id or claim_data provided.",
                "scores": {},
            }

        scores = predict_claim(claim)

        answer = (
            f"ML Scoring Results:\n"
            f"  Fraud Score:         {scores['fraud_score']:.2%} "
            f"({'⚠ FLAGGED' if scores['fraud_flag'] else '✓ OK'})\n"
            f"  Litigation Score:    {scores['litigation_score']:.2%} "
            f"({'⚠ HIGH RISK' if scores['litigation_flag'] else '✓ LOW RISK'})\n"
            f"  Resolution Forecast: {scores['resolution_days_forecast']} days\n"
            f"  Risk Tier:           {scores['risk_tier'].upper()}"
        )

        return {
            "success": True, "agent": "ml",
            "answer": answer, "scores": scores, "claim": claim,
        }

    except Exception as e:
        return {
            "success": False, "agent": "ml",
            "answer": f"ML scoring error: {str(e)}",
            "scores": {},
        }