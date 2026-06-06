"""
api/main.py
Fixes:
- ML runs in ProcessPoolExecutor (avoids XGBoost/OpenMP segfault on Mac)
- All LLM/RAG/SQL runs in ThreadPoolExecutor
- source_type passed through from RAG agent
- Timeout middleware on all requests
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger
from sqlalchemy import create_engine, text

from config.setting import db_settings

thread_executor  = ThreadPoolExecutor(max_workers=4)
process_executor = ProcessPoolExecutor(max_workers=2)

engine = create_engine(
    db_settings.url,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=300,
)


# ── Functions that run in subprocess (XGBoost safe) ───────────────────────────

def _score_in_process(claim: dict) -> dict:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ml_models.predictor import predict_claim
    return predict_claim(claim)


def _get_rag_context_in_thread(claim_type: str) -> str:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agents.rag_agent import run_rag_agent
    q      = f"What are the key procedures and requirements for {claim_type} insurance claims?"
    result = run_rag_agent(q)
    return result.get("answer", "Policy context unavailable.")


def _run_orchestrator(question: str) -> dict:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agents.orchestrator import ask
    return ask(question)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting InsuredAI API — warming up...")
    loop = asyncio.get_event_loop()

    # Warm up process pool with a dummy ML call
    try:
        dummy = {
            "claim_type": "vehicle", "claim_amount": 10000.0,
            "status": "open", "region": "London",
            "adjuster_id": "ADJ0001", "claim_date": "2024-01-01",
            "incident_date": "2024-01-01", "resolution_days": 90.0,
        }
        await loop.run_in_executor(process_executor, _score_in_process, dummy)
        logger.info("ML process pool ready")
    except Exception as e:
        logger.warning(f"ML warmup failed (non-fatal): {e}")

    logger.info("API ready")
    yield
    logger.info("Shutting down InsuredAI API")
    thread_executor.shutdown(wait=False)
    process_executor.shutdown(wait=False)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="InsuredAI — Claims Intelligence API",
    description="Autonomous multi-agent AI system for insurance claims analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=120.0)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"detail": "Request timed out after 120 seconds"}
        )


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question:    str
    answer:      str
    route:       str
    sources:     list[str] = []
    source_type: str | None = None

class ClaimAnalysis(BaseModel):
    claim_id:                 str
    claim_type:               str
    claim_amount:             float
    status:                   str
    region:                   str
    fraud_score:              float
    fraud_flag:               bool
    litigation_score:         float
    litigation_flag:          bool
    resolution_days_forecast: int
    risk_tier:                str
    policy_context:           str

class StatsResponse(BaseModel):
    total_claims:        int
    open_claims:         int
    fraud_flagged:       int
    litigation_flagged:  int
    avg_claim_amount:    float
    avg_resolution_days: float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"
    return {"status": "ok", "db": db_status, "version": "1.0.0"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info(f"Query: {request.question[:80]}")
    start = time.time()
    loop  = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(
            thread_executor,
            _run_orchestrator,
            request.question
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = round(time.time() - start, 2)
    logger.info(f"Query done in {elapsed}s route={result.get('route')}")

    rag_result  = result.get("rag_result", {})
    sources     = rag_result.get("sources", [])
    source_type = rag_result.get("source_type", None)

    return QueryResponse(
        question=request.question,
        answer=result.get("answer", "No answer returned"),
        route=result.get("route", "sql"),
        sources=sources,
        source_type=source_type,
    )


@app.get("/claim/{claim_id}", response_model=ClaimAnalysis)
async def analyse_claim(claim_id: str):
    claim_id = claim_id.upper().strip()
    logger.info(f"Analysing claim: {claim_id}")
    loop = asyncio.get_event_loop()

    # Step 1 — fetch from DB
    def _fetch():
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM claims WHERE claim_id = :id"),
                {"id": claim_id}
            ).fetchone()
        return dict(row._mapping) if row else None

    claim = await loop.run_in_executor(thread_executor, _fetch)

    if claim is None:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")

    # Clean dict — only JSON-serialisable types for process pool
    clean_claim = {
        "claim_type":     str(claim.get("claim_type", "vehicle")),
        "claim_amount":   float(claim.get("claim_amount", 0)),
        "status":         str(claim.get("status", "open")),
        "region":         str(claim.get("region", "London")),
        "adjuster_id":    str(claim.get("adjuster_id", "ADJ0001")),
        "claim_date":     str(claim.get("claim_date", "2024-01-01")),
        "incident_date":  str(claim.get("incident_date", "2024-01-01")),
        "resolution_days": float(claim.get("resolution_days", 90) or 90),
    }

    # Step 2 — ML scoring in PROCESS pool (avoids Mac segfault)
    try:
        scores = await loop.run_in_executor(
            process_executor,
            _score_in_process,
            clean_claim
        )
    except Exception as e:
        logger.error(f"ML scoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"ML scoring failed: {e}")

    # Step 3 — RAG policy context in thread pool
    try:
        policy_context = await loop.run_in_executor(
            thread_executor,
            _get_rag_context_in_thread,
            clean_claim["claim_type"]
        )
    except Exception as e:
        logger.warning(f"RAG context failed: {e}")
        policy_context = "Policy context unavailable at this time."

    return ClaimAnalysis(
        claim_id=claim_id,
        claim_type=clean_claim["claim_type"],
        claim_amount=clean_claim["claim_amount"],
        status=clean_claim["status"],
        region=clean_claim["region"],
        fraud_score=scores["fraud_score"],
        fraud_flag=scores["fraud_flag"],
        litigation_score=scores["litigation_score"],
        litigation_flag=scores["litigation_flag"],
        resolution_days_forecast=scores["resolution_days_forecast"],
        risk_tier=scores["risk_tier"],
        policy_context=policy_context,
    )


@app.get("/claims/stats", response_model=StatsResponse)
async def get_stats():
    loop = asyncio.get_event_loop()

    def _fetch():
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN fraud_flag THEN 1 ELSE 0 END),
                    SUM(CASE WHEN litigation_flag THEN 1 ELSE 0 END),
                    AVG(claim_amount),
                    AVG(resolution_days)
                FROM claims
            """)).fetchone()
        return row

    try:
        row = await loop.run_in_executor(thread_executor, _fetch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    return StatsResponse(
        total_claims=int(row[0]),
        open_claims=int(row[1]),
        fraud_flagged=int(row[2]),
        litigation_flagged=int(row[3]),
        avg_claim_amount=round(float(row[4]), 2),
        avg_resolution_days=round(float(row[5]), 1),
    )