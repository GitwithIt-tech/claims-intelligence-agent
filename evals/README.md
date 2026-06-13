# InsuredAI — Evaluation Suite

> In agentic systems, evaluation *is* the architecture. The confidence
> threshold, the fallback path, the source badges — these came out of eval
> results, not design intuition. This folder is how every claim about
> InsuredAI is verified.

A multi-agent system fails in **compound** ways, so nothing here is tested
"end-to-end only". Each layer is evaluated in isolation, then the whole thing
is regression-tested under concurrent load.

## Setup (one-time)

```bash
pip install -r evals/requirements-eval.txt
```

Then open **`evals/adapter.py`** — the only file you edit. Wire the five
functions to your real code in `agents/` and `ml_models/`. Every script
imports from the adapter, so this is the single integration point.

For the ML eval you also need to dump your held-out test splits once (see the
docstring in `adapter.load_ml_test_data`).

Run everything from the **repo root** so `agents`/`ml_models` import cleanly.

## The five layers

| # | Layer | Script | Key metric | Why it matters |
|---|-------|--------|-----------|----------------|
| 1 | Router | `run_router_eval.py` | Routing accuracy + confusion matrix | A policy question hitting the SQL agent returns confident nonsense. Misroutes are silent. |
| 2 | SQL agent | `run_sql_eval.py` | **Execution** accuracy vs golden set | Two different queries can both be correct — score results, not SQL strings. |
| 3 | RAG retrieval | `run_rag_retrieval_eval.py` | recall@k, MRR, **threshold sweep** | Empirically justifies the 0.55 confidence cutoff instead of guessing it. |
| 4 | RAG generation | `run_rag_faithfulness_eval.py` | Faithfulness, answer relevancy (RAGAS) | Catches hallucination — is every claim grounded in retrieved context? |
| 5 | ML models | `run_ml_eval.py` | AUC, PR-AUC, **calibration**, cost-tuned threshold | A "80%" fraud score should be wrong 1 in 5 times. Threshold tuned to asymmetric cost. |
| — | Regression | `test_regression.py` | pytest, concurrent + sequential | Every fixed bug locked in so it can never silently return. |

## Run it

```bash
# from repo root
python -m evals.run_router_eval
python -m evals.run_sql_eval
python -m evals.run_rag_retrieval_eval
python -m evals.run_rag_faithfulness_eval
python -m evals.run_ml_eval            # writes calibration PNGs to evals/artifacts/
pytest evals/test_regression.py -v
```

## Test data

All under `data/` as JSONL (with `//` comments, stripped by the loader):

- `router_testset.jsonl` — labelled queries across sql / rag / ml / hybrid, incl. adversarial phrasings. **Ready to run.**
- `sql_golden.jsonl` — question → declarative result check. **Fill `expected` from your DB.**
- `rag_retrieval.jsonl` — question → correct {doc, page}. **Fill `page` from your 7 PDFs.**
- `rag_qa.jsonl` — question → ground-truth answer for faithfulness. **Write answers from sources.**

Aim for 40–60 router queries, 25–30 SQL, 20–30 RAG retrieval, 15–20 RAG QA.
More cases = a tighter number you can defend.

## Results

## Results

| Layer | Metric | Result | Date |
|-------|--------|--------|------|
| Router | accuracy (23 queries) | 100% | 2026-06-13 |
| SQL | execution accuracy (10 questions) | 100% | 2026-06-13 |
| RAG | source routing (6 policy / 4 general) | 6/6 → docs, 4/4 → fallback | 2026-06-13 |
| Fraud | AUC-ROC / PR-AUC / Brier | 0.9143 / 0.7187 / 0.104 | 2026-06-13 |
| Litigation | AUC-ROC / PR-AUC / Brier | 0.9399 / 0.9741 / 0.100 | 2026-06-13 |
| Resolution | R² / MAE | 0.9998 / 1.6d — ⚠️ TARGET LEAKAGE, not a valid result | 2026-06-13 |

Cost-weighted operating points (asymmetric FN/FP costs):
- Fraud: threshold 0.05 → precision 0.35, recall 0.98 (missing fraud >> cost of review)
- Litigation: threshold 0.05 → precision 0.76, recall 1.00

Calibration curves: evals/artifacts/calibration_fraud.png, calibration_litigation.png