"""
api/main.py
────────────
FastAPI backend — exposes the claims intelligence agent as a REST API.

Endpoints:
  GET  /health              — health check
  POST /query               — ask any question to the orchestrator
  GET  /claim/{claim_id}    — full analysis of a specific claim
  GET  /claims/stats        — summary statistics
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
import pandas as pd
from sqlalchemy import create_engine, text

from agents.orchestrator import ask as orchestrator_ask
from agents.ml_agent import run_ml_agent
from agents.rag_agent import run_rag_agent
from agents.sql_agent import query as sql_query
from config.setting import db_settings

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Claims Intelligence Agent API",
    description="Autonomous multi-agent AI system for insurance claims analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = create_engine(db_settings.url)


# ── Request / Response models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question:  str
    answer:    str
    route:     str
    sources:   list[str] = []


class ClaimAnalysis(BaseModel):
    claim_id:               str
    claim_type:             str
    claim_amount:           float
    status:                 str
    region:                 str
    fraud_score:            float
    fraud_flag:             bool
    litigation_score:       float
    litigation_flag:        bool
    resolution_days_forecast: int
    risk_tier:              str
    policy_context:         str


class StatsResponse(BaseModel):
    total_claims:       int
    open_claims:        int
    fraud_flagged:      int
    litigation_flagged: int
    avg_claim_amount:   float
    avg_resolution_days: float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check endpoint."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return {
        "status":    "ok",
        "db":        db_status,
        "version":   "1.0.0",
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Ask any question in natural language.
    The orchestrator routes it to the right agent(s).
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info(f"Query: {request.question}")

    result = orchestrator_ask(request.question)

    sources = []
    if result.get("rag_result", {}).get("sources"):
        sources = result["rag_result"]["sources"]

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        route=result["route"],
        sources=sources,
    )


@app.get("/claim/{claim_id}", response_model=ClaimAnalysis)
def analyse_claim(claim_id: str):
    """
    Full analysis of a specific claim:
    - Fetches claim from database
    - Runs ML scoring (fraud, litigation, resolution)
    - Gets relevant policy context via RAG
    """
    claim_id = claim_id.upper()
    logger.info(f"Analysing claim: {claim_id}")

    # Fetch from DB
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM claims WHERE claim_id = :id"),
            {"id": claim_id}
        ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Claim {claim_id} not found"
        )

    claim = dict(row._mapping)

    # ML scoring
    ml_result = run_ml_agent(claim_data=claim)
    if not ml_result["success"]:
        raise HTTPException(
            status_code=500,
            detail=f"ML scoring failed: {ml_result['answer']}"
        )

    scores = ml_result["scores"]

    # RAG — get relevant policy context for this claim type
    rag_question = (
        f"What are the key procedures and requirements for "
        f"{claim['claim_type']} insurance claims?"
    )
    rag_result = run_rag_agent(rag_question)
    policy_context = rag_result.get("answer", "No policy context available.")

    return ClaimAnalysis(
        claim_id=claim_id,
        claim_type=claim["claim_type"],
        claim_amount=float(claim["claim_amount"]),
        status=claim["status"],
        region=claim["region"],
        fraud_score=scores["fraud_score"],
        fraud_flag=scores["fraud_flag"],
        litigation_score=scores["litigation_score"],
        litigation_flag=scores["litigation_flag"],
        resolution_days_forecast=scores["resolution_days_forecast"],
        risk_tier=scores["risk_tier"],
        policy_context=policy_context,
    )


@app.get("/claims/stats", response_model=StatsResponse)
def get_stats():
    """Summary statistics across all claims."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT
                COUNT(*)                              AS total_claims,
                SUM(CASE WHEN status = 'open'
                    THEN 1 ELSE 0 END)               AS open_claims,
                SUM(CASE WHEN fraud_flag
                    THEN 1 ELSE 0 END)               AS fraud_flagged,
                SUM(CASE WHEN litigation_flag
                    THEN 1 ELSE 0 END)               AS litigation_flagged,
                AVG(claim_amount)                    AS avg_claim_amount,
                AVG(resolution_days)                 AS avg_resolution_days
            FROM claims
        """)).fetchone()

    return StatsResponse(
        total_claims=row[0],
        open_claims=row[1],
        fraud_flagged=row[2],
        litigation_flagged=row[3],
        avg_claim_amount=round(float(row[4]), 2),
        avg_resolution_days=round(float(row[5]), 1),
    )