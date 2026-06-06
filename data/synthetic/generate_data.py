"""
data/synthetic/generate_data.py
────────────────────────────────
Generates 10,000 realistic synthetic insurance claims rows.

Run:
    python3 data/synthetic/generate_data.py
"""

import random
import pandas as pd
import numpy as np
from faker import Faker
from datetime import date, timedelta
from pathlib import Path

fake = Faker("en_GB")
random.seed(42)
np.random.seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────

CLAIM_TYPES = ["vehicle", "property", "liability", "health", "travel"]
REGIONS     = ["London", "Manchester", "Birmingham", "Leeds", "Belfast",
               "Edinburgh", "Bristol", "Cardiff", "Liverpool", "Sheffield"]
STATUSES    = ["open", "closed", "in_review", "litigated"]
RISK_TIERS  = ["low", "medium", "high"]
SPECIALISMS = ["vehicle", "property", "liability", "health", "fraud"]

N_ADJUSTERS = 40
N_POLICIES  = 3000
N_CLAIMS    = 10_000

OUT_DIR = Path(__file__).parent


# ── Helpers ───────────────────────────────────────────────────────────────────

def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def fraud_probability(claim_type: str, amount: float, region: str) -> float:
    base = {"vehicle": 0.12, "property": 0.08, "liability": 0.22,
            "health": 0.06, "travel": 0.15}.get(claim_type, 0.10)
    amount_factor  = min(amount / 50_000, 1.0) * 0.15
    region_risk    = {"London": 0.05, "Birmingham": 0.04, "Manchester": 0.03}.get(region, 0.0)
    score = base + amount_factor + region_risk + np.random.normal(0, 0.05)
    return float(np.clip(score, 0.01, 0.99))


def litigation_probability(claim_type: str, amount: float,
                            fraud_score: float, resolution_days: int) -> float:
    base = {"vehicle": 0.08, "property": 0.06, "liability": 0.28,
            "health": 0.10, "travel": 0.04}.get(claim_type, 0.08)
    amount_factor = min(amount / 80_000, 1.0) * 0.20
    fraud_factor  = fraud_score * 0.15
    delay_factor  = min(resolution_days / 365, 1.0) * 0.12
    score = base + amount_factor + fraud_factor + delay_factor + np.random.normal(0, 0.04)
    return float(np.clip(score, 0.01, 0.99))


# ── Generators ────────────────────────────────────────────────────────────────

def generate_adjusters() -> pd.DataFrame:
    rows = []
    for i in range(N_ADJUSTERS):
        rows.append({
            "adjuster_id":    f"ADJ{i+1:04d}",
            "adjuster_name":  fake.name(),
            "region":         random.choice(REGIONS),
            "specialisation": random.choice(SPECIALISMS),
            "active":         random.random() > 0.1,
            "claims_handled": random.randint(50, 800),
        })
    return pd.DataFrame(rows)


def generate_policies() -> pd.DataFrame:
    rows = []
    for i in range(N_POLICIES):
        start = random_date(date(2018, 1, 1), date(2023, 12, 31))
        end   = start + timedelta(days=365 * random.randint(1, 3))
        rows.append({
            "policy_number":  f"POL{i+1:06d}",
            "policy_type":    random.choice(CLAIM_TYPES),
            "holder_name":    fake.name(),
            "holder_age":     random.randint(18, 80),
            "start_date":     start.isoformat(),
            "end_date":       end.isoformat(),
            "premium_amount": round(random.uniform(300, 5000), 2),
            "coverage_limit": round(random.uniform(10_000, 500_000), 2),
            "region":         random.choice(REGIONS),
            "risk_tier":      random.choices(RISK_TIERS, weights=[50, 35, 15])[0],
        })
    return pd.DataFrame(rows)


def generate_claims(adjusters: pd.DataFrame, policies: pd.DataFrame) -> pd.DataFrame:
    adj_ids  = adjusters["adjuster_id"].tolist()
    pol_nums = policies["policy_number"].tolist()

    notes_options = [
        "Claimant reported incident. Initial assessment pending.",
        "Documentation received. Claim under review by adjuster.",
        "Third party involved. Liability assessment required. Amount disputed.",
        "Supporting evidence submitted. Fast-track review requested.",
        "Complex claim. Legal representation indicated. Escalated.",
        "Routine claim. No anomalies detected at intake stage.",
        "High-value claim flagged for senior review. Coverage verification in progress.",
    ]

    rows = []
    for i in range(N_CLAIMS):
        claim_type      = random.choice(CLAIM_TYPES)
        region          = random.choice(REGIONS)
        incident_date   = random_date(date(2020, 1, 1), date(2024, 6, 30))
        claim_date      = incident_date + timedelta(days=random.randint(1, 30))
        claim_amount    = round(float(np.random.lognormal(mean=9.5, sigma=1.2)), 2)
        resolution_days = random.randint(5, 500)
        status          = random.choices(STATUSES, weights=[25, 55, 15, 5])[0]
        settled_amount  = (
            round(claim_amount * random.uniform(0.4, 1.1), 2)
            if status == "closed" else None
        )

        fraud_score = fraud_probability(claim_type, claim_amount, region)
        fraud_flag  = fraud_score > 0.28

        lit_score   = litigation_probability(claim_type, claim_amount, fraud_score, resolution_days)
        lit_flag    = lit_score > 0.20 or status == "litigated"

        rows.append({
            "claim_id":         f"CLM{i+1:07d}",
            "policy_number":    random.choice(pol_nums),
            "claim_type":       claim_type,
            "claim_date":       claim_date.isoformat(),
            "incident_date":    incident_date.isoformat(),
            "claim_amount":     claim_amount,
            "settled_amount":   settled_amount,
            "status":           status,
            "resolution_days":  resolution_days,
            "adjuster_id":      random.choice(adj_ids),
            "region":           region,
            "fraud_flag":       fraud_flag,
            "fraud_score":      round(fraud_score, 4),
            "litigation_flag":  lit_flag,
            "litigation_score": round(lit_score, 4),
            "notes":            random.choice(notes_options),
        })

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating adjusters...")
    adjusters = generate_adjusters()
    adjusters.to_csv(OUT_DIR / "adjusters.csv", index=False)
    print(f"  ✓ {len(adjusters)} adjusters saved")

    print("Generating policies...")
    policies = generate_policies()
    policies.to_csv(OUT_DIR / "policies.csv", index=False)
    print(f"  ✓ {len(policies)} policies saved")

    print("Generating claims...")
    claims = generate_claims(adjusters, policies)
    claims.to_csv(OUT_DIR / "claims.csv", index=False)
    print(f"  ✓ {len(claims)} claims saved")

    print("\nDataset summary:")
    print(f"  Fraud cases:      {claims['fraud_flag'].sum():,}  ({claims['fraud_flag'].mean()*100:.1f}%)")
    print(f"  Litigation cases: {claims['litigation_flag'].sum():,}  ({claims['litigation_flag'].mean()*100:.1f}%)")
    print(f"  Avg claim amount: £{claims['claim_amount'].mean():,.2f}")
    print(f"  Claim types:\n{claims['claim_type'].value_counts().to_string()}")
    print("\nAll files saved to data/synthetic/")


if __name__ == "__main__":
    main()