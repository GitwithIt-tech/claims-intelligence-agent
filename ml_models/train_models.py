"""
ml_models/train_models.py
──────────────────────────
Trains 3 models on the synthetic claims data:
  1. Fraud detector        (XGBoost classifier)
  2. Litigation predictor  (XGBoost classifier)
  3. Resolution forecaster (XGBoost regressor)

Run:
    python3 ml_models/train_models.py
"""

import sys
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    mean_absolute_error, r2_score
)
from sklearn.preprocessing import LabelEncoder

ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "data/synthetic/claims.csv"
MODELS_DIR = ROOT / "ml_models/saved_models"
MODELS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
from config.setting import ml_settings

mlflow.set_tracking_uri(ml_settings.MLFLOW_TRACKING_URI)
mlflow.set_experiment("claims-intelligence-agent")


# ── Feature engineering ───────────────────────────────────────────────────────

TYPE_MAP   = {"vehicle":0,"property":1,"liability":2,"health":3,"travel":4}
STATUS_MAP = {"open":0,"closed":1,"in_review":2,"litigated":3}
REGION_MAP = {"London":0,"Manchester":1,"Birmingham":2,"Leeds":3,
              "Belfast":4,"Edinburgh":5,"Bristol":6,"Cardiff":7,
              "Liverpool":8,"Sheffield":9}

FEATURE_COLS = [
    "log_claim_amount",
    "amount_per_day",
    "resolution_days",
    "days_to_report",
    "claim_month",
    "claim_quarter",
    "claim_dow",
    "claim_type_enc",
    "status_enc",
    "region_enc",
    "adjuster_id_enc",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()

    f["claim_date"]    = pd.to_datetime(f["claim_date"])
    f["incident_date"] = pd.to_datetime(f["incident_date"])

    f["days_to_report"]    = (f["claim_date"] - f["incident_date"]).dt.days.clip(lower=0)
    f["claim_month"]       = f["claim_date"].dt.month
    f["claim_quarter"]     = f["claim_date"].dt.quarter
    f["claim_dow"]         = f["claim_date"].dt.dayofweek
    f["log_claim_amount"]  = np.log1p(f["claim_amount"])
    f["amount_per_day"]    = f["claim_amount"] / (f["resolution_days"] + 1)

    f["claim_type_enc"]  = f["claim_type"].map(TYPE_MAP).fillna(0).astype(int)
    f["status_enc"]      = f["status"].map(STATUS_MAP).fillna(0).astype(int)
    f["region_enc"]      = f["region"].map(REGION_MAP).fillna(0).astype(int)
    f["adjuster_id_enc"] = f["adjuster_id"].str.replace("ADJ","").astype(int) % 40

    return f[FEATURE_COLS]


# ── Model 1 — Fraud Detector ──────────────────────────────────────────────────

def train_fraud_model(df: pd.DataFrame):
    print("\n" + "="*50)
    print("Training Model 1: Fraud Detector")
    print("="*50)

    X = build_features(df)
    y = df["fraud_flag"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train==0).sum() / (y_train==1).sum(),
        random_state=42,
        eval_metric="auc",
        verbosity=0,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds      = model.predict(X_test)
    preds_prob = model.predict_proba(X_test)[:, 1]
    auc        = roc_auc_score(y_test, preds_prob)
    f1         = f1_score(y_test, preds)

    print(f"  AUC-ROC : {auc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(classification_report(y_test, preds, target_names=["Not Fraud","Fraud"]))

    with mlflow.start_run(run_name="fraud_detector"):
        mlflow.log_params({"n_estimators":200,"max_depth":6,"learning_rate":0.05})
        mlflow.log_metrics({"auc_roc":auc,"f1_score":f1})
        mlflow.sklearn.log_model(model, "fraud_detector")

    path = MODELS_DIR / "fraud_detector.pkl"
    with open(path, "wb") as f_out:
        pickle.dump(model, f_out)
    print(f"  ✓ Saved to {path}")

    return model


# ── Model 2 — Litigation Predictor ───────────────────────────────────────────

def train_litigation_model(df: pd.DataFrame):
    print("\n" + "="*50)
    print("Training Model 2: Litigation Predictor")
    print("="*50)

    X = build_features(df)
    y = df["litigation_flag"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train==0).sum() / (y_train==1).sum(),
        random_state=42,
        eval_metric="auc",
        verbosity=0,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds      = model.predict(X_test)
    preds_prob = model.predict_proba(X_test)[:, 1]
    auc        = roc_auc_score(y_test, preds_prob)
    f1         = f1_score(y_test, preds)

    print(f"  AUC-ROC : {auc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(classification_report(y_test, preds, target_names=["No Litigation","Litigation"]))

    with mlflow.start_run(run_name="litigation_predictor"):
        mlflow.log_params({"n_estimators":200,"max_depth":5,"learning_rate":0.05})
        mlflow.log_metrics({"auc_roc":auc,"f1_score":f1})
        mlflow.sklearn.log_model(model, "litigation_predictor")

    path = MODELS_DIR / "litigation_predictor.pkl"
    with open(path, "wb") as f_out:
        pickle.dump(model, f_out)
    print(f"  ✓ Saved to {path}")

    return model


# ── Model 3 — Resolution Forecaster ──────────────────────────────────────────

def train_resolution_model(df: pd.DataFrame):
    print("\n" + "="*50)
    print("Training Model 3: Resolution Forecaster")
    print("="*50)

    closed = df[df["status"] == "closed"].copy()
    print(f"  Using {len(closed):,} closed claims for training")

    X = build_features(closed)
    y = closed["resolution_days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)

    print(f"  MAE (days): {mae:.1f}")
    print(f"  R² Score  : {r2:.4f}")

    with mlflow.start_run(run_name="resolution_forecaster"):
        mlflow.log_params({"n_estimators":200,"max_depth":5,"learning_rate":0.05})
        mlflow.log_metrics({"mae_days":mae,"r2_score":r2})
        mlflow.sklearn.log_model(model, "resolution_forecaster")

    path = MODELS_DIR / "resolution_forecaster.pkl"
    with open(path, "wb") as f_out:
        pickle.dump(model, f_out)
    print(f"  ✓ Saved to {path}")

    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading claims data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  ✓ {len(df):,} claims loaded")

    train_fraud_model(df)
    train_litigation_model(df)
    train_resolution_model(df)

    with open(MODELS_DIR / "feature_cols.pkl", "wb") as f:
        pickle.dump(FEATURE_COLS, f)

    print("\n" + "="*50)
    print("✓ All 3 models trained and saved")
    print(f"  Location: {MODELS_DIR}")
    print("\nTo view MLflow UI run:")
    print("  mlflow ui --port 5000")
    print("  Then open: http://localhost:5000")
    print("="*50)


if __name__ == "__main__":
    main()