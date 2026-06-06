<div align="center">

<img src="https://img.shields.io/badge/InsuredAI-Claims%20Intelligence-2563eb?style=for-the-badge&logo=shield&logoColor=white" />

# InsuredAI — Autonomous Claims Intelligence Platform

**An end-to-end multi-agent AI system for insurance claims analysis**

Built with LangGraph · Groq LLaMA 3.3 · RAG · XGBoost · FastAPI · React

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.14-FF6B35?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-F55036?style=flat-square)](https://groq.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=flat-square&logo=postgresql&logoColor=white)](https://postgresql.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-AUC%200.94-FF6600?style=flat-square)](https://xgboost.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<br/>

> Ask InsuredAI: *"Which open claims are at highest litigation risk this week and what does the AXA policy say about fast-tracking them?"*
> It routes to SQL + RAG simultaneously, queries 10,000 claims, retrieves real policy PDFs, and synthesises a cited answer in under 3 seconds.

<br/>

[🚀 Live Demo](#demo) · [📖 Architecture](#architecture) · [⚡ Quick Start](#quick-start) · [📊 Results](#results)

</div>

---

## 🎯 What This Project Does

InsuredAI is a production-grade autonomous AI platform that lets insurance teams query claims data, extract policy insights, and score individual claims for fraud and litigation risk — all through natural language.

It demonstrates three cutting-edge AI paradigms working together:

| Paradigm | Implementation |
|---|---|
| **Agentic AI** | LangGraph orchestrator with ReAct loop — plans, acts, observes, synthesises |
| **RAG** | 7 real insurance PDFs (AXA, Direct Line, Admiral) chunked, embedded, retrieved with citations |
| **Predictive ML** | XGBoost fraud detector (AUC 0.91) + litigation predictor (AUC 0.94) + resolution forecaster (R² 0.9998) |

---

## 🏗️ Architecture

```
User Query (natural language)
           │
           ▼
  ┌─────────────────┐
  │  LangGraph      │   ← Intent router — classifies query type
  │  Orchestrator   │     using Groq LLaMA 3.3 70B
  └────────┬────────┘
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
┌───────┐ ┌────┐ ┌──────┐
│  SQL  │ │RAG │ │  ML  │
│ Agent │ │    │ │Agent │
└───┬───┘ └─┬──┘ └──┬───┘
    │       │        │
    │  Text │ Vector │ XGBoost
    │  to   │ Search │ Inference
    │  SQL  │ + LLM  │
    ▼       ▼        ▼
PostgreSQL ChromaDB Saved
10k Claims 7 PDFs   Models
    │       │        │
    └───────┴────────┘
                │
                ▼
     ┌─────────────────┐
     │   Synthesiser   │  ← Combines all agent results
     │   + Guardrails  │    into one coherent response
     └────────┬────────┘
              │
    ┌─────────┴──────────┐
    ▼                    ▼
FastAPI REST          React SPA
/query /claim         InsuredAI UI
/stats /health        Dashboard + Chat
```

---

## ✨ Key Features

- **🤖 Multi-agent orchestration** — LangGraph routes questions to SQL, RAG, or ML agents based on intent
- **💬 Natural language queries** — ask anything in plain English, get cited answers
- **📄 Real policy RAG** — retrieves from actual AXA, Direct Line, Admiral policy PDFs with page citations
- **🔍 Claim risk scoring** — instant fraud score, litigation probability, and resolution forecast per claim
- **📊 Live analytics dashboard** — KPI tiles, fraud heatmaps, trend charts, status breakdowns
- **🛡️ No hallucination** — RAG answers grounded in real documents, SQL answers verified against live DB
- **⚡ Fast** — Groq inference, sub-3s end-to-end response time

---

## 📊 Model Results

| Model | Algorithm | Key Metric | Score |
|---|---|---|---|
| Fraud Detector | XGBoost Classifier | AUC-ROC | **0.9143** |
| Litigation Predictor | XGBoost Classifier | AUC-ROC | **0.9399** |
| Resolution Forecaster | XGBoost Regressor | R² Score | **0.9998** |
| Resolution Forecaster | XGBoost Regressor | MAE | **1.6 days** |
| SQL Analytics Agent | Groq LLaMA 3.3 | Query Accuracy | **~92%** |

All experiments tracked in MLflow with parameters, metrics, and model artifacts.

---

## 🗂️ Project Structure

```
claims-intelligence-agent/
├── agents/
│   ├── orchestrator.py      # LangGraph multi-agent graph
│   ├── sql_agent.py         # Text-to-SQL with safety guardrails
│   ├── rag_agent.py         # Policy document Q&A
│   └── ml_agent.py          # Fraud/litigation scoring
│
├── rag/
│   ├── ingest.py            # PDF → ChromaDB ingestion pipeline
│   ├── retriever.py         # Vector similarity search + citations
│   └── rag_chain.py         # LangChain + Groq answer generation
│
├── ml_models/
│   ├── train_models.py      # XGBoost training + MLflow logging
│   ├── predictor.py         # Inference interface
│   └── saved_models/        # Serialised .pkl model files
│
├── api/
│   └── main.py              # FastAPI app — 4 endpoints
│
├── insuredai/               # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx          # All pages + components
│   │   └── index.css        # Design system
│   └── package.json
│
├── data/
│   ├── raw/                 # 7 real insurance PDFs + SQL schema
│   └── synthetic/
│       ├── generate_data.py # 10,000 claims data generator
│       └── load_to_db.py    # PostgreSQL loader
│
├── config/
│   └── settings.py          # Central config — all env vars
│
├── docker-compose.yml        # PostgreSQL + Redis containers
├── requirements.txt
└── .env.example
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker Desktop
- Groq API key (free at [console.groq.com](https://console.groq.com))

### 1. Clone and configure

```bash
git clone https://github.com/GitwithIt-tech/claims-intelligence-agent.git
cd claims-intelligence-agent
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### 2. Start database

```bash
docker-compose up postgres -d
```

### 3. Install Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
brew install libomp         # Mac only — required for XGBoost
```

### 4. Generate data and train models

```bash
python3 data/synthetic/generate_data.py
python3 data/synthetic/load_to_db.py
python3 rag/ingest.py
python3 ml_models/train_models.py
```

### 5. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

### 6. Start the frontend

```bash
cd insuredai
npm install
npm run dev
```

Open **http://localhost:5173**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health + DB status |
| `POST` | `/query` | Natural language query → agent response |
| `GET` | `/claim/{id}` | Full AI analysis of a specific claim |
| `GET` | `/claims/stats` | Aggregate KPI statistics |

### Example API call

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which regions have the highest fraud rates?"}'
```

```json
{
  "question": "Which regions have the highest fraud rates?",
  "answer": "London leads with an average fraud score of 23.7%, followed by Birmingham at 21.7% and Manchester at 21.1%...",
  "route": "sql",
  "sources": []
}
```

---

## 💬 Example Queries

These demonstrate all three agents working:

```
# SQL Agent
"How many open claims are there by region?"
"Which adjusters have handled the most high-risk claims?"
"What is the average resolution time for litigated vs non-litigated claims?"

# RAG Agent
"What does the AXA policy say about submitting a claim after an incident?"
"What are the requirements for high-value property claims?"
"When does a claim get referred to the Special Investigations Unit?"

# ML Agent
"Score claim CLM0000042"
"What is the fraud risk for claim CLM0001234?"

# Both SQL + RAG
"Which regions have highest litigation rates and what does policy say about litigation procedures?"
```

---

## 🛠️ Tech Stack

### AI & ML
- **LangGraph** — multi-agent orchestration with typed state graph
- **LangChain** — RAG chains, SQL agent, prompt templates
- **Groq API** — LLaMA 3.3 70B inference (sub-1s latency)
- **ChromaDB** — vector store with cosine similarity search
- **HuggingFace** — `all-MiniLM-L6-v2` sentence embeddings (local, free)
- **XGBoost** — gradient boosted trees for classification and regression
- **scikit-learn** — preprocessing, evaluation metrics, train/test split
- **MLflow** — experiment tracking, model registry, artifact storage

### Backend
- **FastAPI** — async REST API with Pydantic validation
- **PostgreSQL 16** — relational database via Docker
- **SQLAlchemy** — ORM and connection pooling
- **PyMuPDF** — PDF text extraction

### Frontend
- **React 18** — component-based SPA
- **Vite** — fast dev server with API proxy
- **Recharts** — charts and data visualisation
- **Lucide React** — icon library
- **Inter + Outfit** — premium typography

### DevOps
- **Docker + Docker Compose** — containerised PostgreSQL and Redis
- **MLflow** — model tracking UI at `localhost:5000`

---

## 📈 MLflow Experiment Tracking

After training, view all experiments:

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

You'll see all 3 model runs with parameters, metrics, and artifacts logged.

---

## 🔑 Environment Variables

```bash
# .env.example
GROQ_API_KEY=your-groq-key-here
GROQ_MODEL=llama-3.3-70b-versatile

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=claims_db
POSTGRES_USER=claims_user
POSTGRES_PASSWORD=claims_pass

CHROMA_PERSIST_DIR=./rag/chroma_db
MLFLOW_TRACKING_URI=./mlruns
API_SECRET_KEY=your-secret-key
API_PORT=8000
```

---

## 🗺️ Roadmap

- [ ] PDF report generation per claim (ReportLab)
- [ ] Authentication layer (JWT)
- [ ] Real-time claim ingestion via Kafka
- [ ] Azure Container Apps deployment
- [ ] Power BI embedded dashboard
- [ ] SHAP explainability panel in UI

---

## 👤 Author

**Sumedh Wani** — Senior Data Analyst Consultant
- 💼 [LinkedIn](https://linkedin.com/in/sumedh-wani)
- 🐙 [GitHub](https://github.com/GitwithIt-tech)
- 📧 wanisumedh833@gmail.com

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**If this project helped you, please ⭐ star the repo**

Built with ❤️ by Sumedh Wani · Belfast, UK

</div>