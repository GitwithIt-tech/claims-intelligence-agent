# InsuredAI — Technical Architecture

## System Overview

InsuredAI is a multi-agent AI system built on the LangGraph framework. It routes natural language queries to specialised agents — SQL, RAG, or ML — based on intent classification, then synthesises results into a single coherent response.

## Agent Routing Logic

```
User Query
    │
    ▼
Intent Classification (Groq LLaMA 3.3)
    │
    ├── "how many claims" / "average fraud" / "which region"
    │        → SQL Agent (text-to-SQL → PostgreSQL)
    │
    ├── "what does policy say" / "procedure for" / "AXA terms"
    │        → RAG Agent (vector search → ChromaDB → LLM)
    │
    ├── "score claim CLM..." / "fraud risk for"
    │        → ML Agent (XGBoost inference)
    │
    └── "litigation rates AND policy say"
             → Both SQL + RAG (parallel execution)
```

## Data Flow — RAG Pipeline

```
7 Insurance PDFs (AXA, Direct Line, Admiral)
    │
    ▼ PyMuPDF — text extraction page by page
    │
    ▼ Chunking — 512 words, 50-word overlap
    │
    ▼ HuggingFace all-MiniLM-L6-v2 — 384-dim embeddings
    │
    ▼ ChromaDB — persisted vector store (cosine similarity)
    │
    ▼ Top-5 retrieval → Context with [Source, Page] citations
    │
    ▼ Groq LLaMA 3.3 — generates cited answer
```

## Data Flow — ML Pipeline

```
10,000 Synthetic Claims (PostgreSQL)
    │
    ▼ Feature Engineering
      - log(claim_amount), amount_per_day
      - days_to_report, claim_month, quarter, day_of_week
      - label-encoded: claim_type, status, region, adjuster_id
    │
    ▼ XGBoost Training (scale_pos_weight for class imbalance)
      - Fraud Detector:       200 trees, max_depth=6, lr=0.05
      - Litigation Predictor: 200 trees, max_depth=5, lr=0.05
      - Resolution Forecaster:200 trees, max_depth=5, lr=0.05
    │
    ▼ MLflow — parameters, metrics, artifacts logged
    │
    ▼ Serialised .pkl models loaded once at API startup
    │
    ▼ predict_claim(dict) → fraud_score, litigation_score, days
```

## API Contract

### POST /query
```json
Request:  { "question": "string" }
Response: { "question": "string", "answer": "string", "route": "sql|rag|ml|both_sql_rag", "sources": ["string"] }
```

### GET /claim/{claim_id}
```json
Response: {
  "claim_id": "CLM0000042",
  "claim_type": "liability",
  "claim_amount": 45000.00,
  "status": "open",
  "region": "London",
  "fraud_score": 0.3241,
  "fraud_flag": false,
  "litigation_score": 0.5812,
  "litigation_flag": true,
  "resolution_days_forecast": 187,
  "risk_tier": "medium",
  "policy_context": "string"
}
```

### GET /claims/stats
```json
Response: {
  "total_claims": 10000,
  "open_claims": 2500,
  "fraud_flagged": 1879,
  "litigation_flagged": 6943,
  "avg_claim_amount": 27433.77,
  "avg_resolution_days": 252.4
}
```

## Database Schema

```sql
claims (
  claim_id, policy_number, claim_type, claim_date,
  incident_date, claim_amount, settled_amount, status,
  resolution_days, adjuster_id, region,
  fraud_flag, fraud_score, litigation_flag, litigation_score, notes
)

policies (
  policy_number, policy_type, holder_name, holder_age,
  start_date, end_date, premium_amount, coverage_limit,
  region, risk_tier
)

adjusters (
  adjuster_id, adjuster_name, region,
  specialisation, active, claims_handled
)
```

## LangGraph State

```python
class AgentState(TypedDict):
    question:     str
    route:        str          # sql | rag | ml | both_sql_rag
    claim_id:     str | None   # extracted CLM ID if present
    sql_result:   dict         # { success, sql, dataframe, summary }
    rag_result:   dict         # { success, answer, sources }
    ml_result:    dict         # { success, answer, scores }
    final_answer: str
```

## Technology Decisions

| Decision | Rationale |
|---|---|
| Groq over OpenAI | Free tier, sub-1s inference, LLaMA 3.3 70B quality |
| ChromaDB over Pinecone | Local, no API key, persistent, free |
| XGBoost over neural nets | Tabular data, interpretable, fast inference, SHAP-compatible |
| FastAPI over Flask/Django | Async, automatic OpenAPI docs, Pydantic validation |
| React+Vite over Streamlit | No lag, real SPA, smooth animations, production-grade |
| LangGraph over raw LangChain | Stateful graph, explicit routing, easier debugging |