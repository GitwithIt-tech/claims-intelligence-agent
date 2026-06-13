"""
adapter.py — wired to InsuredAI's real function signatures.
Every eval script imports from here.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# 1. ROUTER — ask() runs the full graph and returns the chosen route.
#    We map your "both_sql_rag" label to the eval's "hybrid".
# 1. ROUTER — call the router node DIRECTLY, not the full ask() pipeline.
#    Unit-tests routing in isolation: one LLM call, no SQL/RAG/ML execution,
#    so it's fast AND avoids the in-process XGBoost segfault.
def route_query(query: str) -> str:
    from agents.orchestrator import router_node
    state = {
        "question": query, "route": "", "claim_id": None,
        "sql_result": {}, "rag_result": {}, "ml_result": {}, "final_answer": "",
    }
    out = router_node(state)
    route = out.get("route", "sql")
    return "hybrid" if route == "both_sql_rag" else route


# 2. SQL — query() returns a pandas DataFrame; we score on rows, so convert.
def run_sql_agent(query: str) -> dict:
    from agents.sql_agent import query as sql_query
    out = sql_query(query)
    df = out.get("dataframe")
    rows = df.to_dict("records") if df is not None and hasattr(df, "to_dict") else []
    return {"sql": out.get("sql"), "rows": rows}


# 3. RAG — your run_rag_agent returns sources as "AXA (p.3)" strings; parse them.
_SRC_RE = re.compile(r"^(.*?)\s*\(p\.?\s*(\d+)\)\s*$", re.IGNORECASE)


def _parse_sources(sources: list) -> list[dict]:
    out = []
    for s in sources:
        m = _SRC_RE.match(str(s))
        if m:
            out.append({"doc": m.group(1).strip(), "page": int(m.group(2))})
        else:
            out.append({"doc": str(s), "page": None})
    return out


def run_rag_agent(query: str) -> dict:
    from agents.rag_agent import run_rag_agent as _rag
    out = _rag(query)
    return {
        "answer": out.get("answer", ""),
        "source_type": out.get("source_type", "ai_knowledge"),
        "confidence": float(out.get("confidence", 0.0)),  # exposed after the patch below
        "contexts": out.get("contexts", []),              # exposed after the patch below
        "sources": _parse_sources(out.get("sources", [])),
    }


# 4. ML — your globals already match; no change needed.
def load_ml_models() -> dict:
    from ml_models.predictor import _FRAUD_MODEL, _LITIGATION_MODEL, _RESOLUTION_MODEL
    return {
        "fraud": _FRAUD_MODEL,
        "litigation": _LITIGATION_MODEL,
        "resolution": _RESOLUTION_MODEL,
    }


def load_ml_test_data():
    """Reconstruct the EXACT held-out splits the models never trained on, by
    repeating train_models.py's splits with the same seed (42) on the same CSV.
    Note: resolution trains only on CLOSED claims, so we filter identically."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from ml_models.train_models import build_features

    csv = REPO_ROOT / "data" / "synthetic" / "claims.csv"
    df = pd.read_csv(csv)

    # Fraud + litigation: full dataset, stratified, seed 42
    X_all = build_features(df)
    yf = df["fraud_flag"].astype(int)
    _, Xtf, _, ytf = train_test_split(X_all, yf, test_size=0.2, random_state=42, stratify=yf)

    yl = df["litigation_flag"].astype(int)
    _, Xtl, _, ytl = train_test_split(X_all, yl, test_size=0.2, random_state=42, stratify=yl)

    # Resolution: CLOSED claims only (matches train_models.py line 191), not stratified
    closed = df[df["status"] == "closed"].copy()
    X_closed = build_features(closed)
    yr = closed["resolution_days"].astype(float)
    _, Xtr, _, ytr = train_test_split(X_closed, yr, test_size=0.2, random_state=42)

    return {"fraud": (Xtf, ytf), "litigation": (Xtl, ytl), "resolution": (Xtr, ytr)}