"""
ML model eval — beyond AUC.

For the two classifiers (fraud, litigation):
  - AUC-ROC and PR-AUC (PR matters more under class imbalance)
  - Brier score + calibration curve: a model that says "80%" should be wrong
    ~1 time in 5. Calibration is what makes a probability trustworthy as a
    business signal, and almost nobody checks it.
  - Cost-weighted threshold tuning: a missed fraud case costs far more than an
    unnecessary review, so 0.5 is rarely the right operating point.

For the regressor (resolution): MAE, RMSE, R^2, and residual sanity.

Outputs PNGs (calibration curves) to evals/artifacts/.

Install:  pip install scikit-learn matplotlib numpy
Run:      python -m evals.run_ml_eval
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from evals.adapter import load_ml_models, load_ml_test_data

ART = Path(__file__).resolve().parent / "artifacts"
ART.mkdir(exist_ok=True)


def _proba(model, X):
    """Get positive-class probabilities from an sklearn/XGBoost classifier."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # XGBoost Booster fallback
    import xgboost as xgb
    return model.predict(xgb.DMatrix(X))


def eval_classifier(name: str, model, X, y, fn_cost: float, fp_cost: float) -> None:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, brier_score_loss,
        precision_score, recall_score, f1_score,
    )
    from sklearn.calibration import calibration_curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.asarray(y)
    p = np.asarray(_proba(model, X))

    auc = roc_auc_score(y, p)
    pr_auc = average_precision_score(y, p)
    brier = brier_score_loss(y, p)

    print(f"\n=== {name.upper()} CLASSIFIER ===")
    print(f"  AUC-ROC : {auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}   (more honest under imbalance)")
    print(f"  Brier   : {brier:.4f}   (lower = better calibrated)")

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    plt.plot(mean_pred, frac_pos, "o-", label=name)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration — {name}")
    plt.legend()
    out = ART / f"calibration_{name}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  calibration curve -> {out}")

    # Cost-weighted threshold tuning
    best_t, best_cost = 0.5, float("inf")
    for t in np.linspace(0.05, 0.95, 19):
        pred = (p >= t).astype(int)
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        cost = fn * fn_cost + fp * fp_cost
        if cost < best_cost:
            best_cost, best_t = cost, t
    pred = (p >= best_t).astype(int)
    print(f"  cost-optimal threshold = {best_t:.2f}  "
          f"(FN cost £{fn_cost:.0f}, FP cost £{fp_cost:.0f})")
    print(f"    at that threshold: precision={precision_score(y, pred):.2f} "
          f"recall={recall_score(y, pred):.2f} f1={f1_score(y, pred):.2f}")


def eval_regressor(name: str, model, X, y) -> None:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y = np.asarray(y)
    pred = np.asarray(model.predict(X))
    mae = mean_absolute_error(y, pred)
    rmse = mean_squared_error(y, pred) ** 0.5
    r2 = r2_score(y, pred)
    print(f"\n=== {name.upper()} REGRESSOR ===")
    print(f"  MAE  : {mae:.2f} days")
    print(f"  RMSE : {rmse:.2f} days")
    print(f"  R^2  : {r2:.4f}")
    if r2 > 0.999:
        print("  ! R^2 this high is suspicious — check for target leakage in features.")


def main() -> None:
    models = load_ml_models()
    data = load_ml_test_data()

    # Asymmetric costs: tune these to your business. A missed fraud (FN) is
    # far more expensive than an extra manual review (FP).
    Xf, yf = data["fraud"]
    eval_classifier("fraud", models["fraud"], Xf, yf, fn_cost=5000, fp_cost=50)

    Xl, yl = data["litigation"]
    eval_classifier("litigation", models["litigation"], Xl, yl, fn_cost=8000, fp_cost=100)

    Xr, yr = data["resolution"]
    eval_regressor("resolution", models["resolution"], Xr, yr)


if __name__ == "__main__":
    main()