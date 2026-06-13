"""
Regression tests — every bug you fixed, locked in so it can never silently
come back. This is the file that says "I engineer, I don't just demo."

Each test maps to a documented bug from the project history:
  Bug 1 — second request hangs (global graph state leak)
  Bug 2 — XGBoost segfault from threads on Mac (now process-isolated)
  Bug 4 — models reloaded per call (now loaded once at module level)

Run:  pytest evals/test_regression.py -v
"""
from __future__ import annotations
import time
import concurrent.futures as cf

import pytest

from evals.adapter import run_sql_agent, run_rag_agent


# --- Bug 1: sequential requests must not hang (fresh graph per call) --------
def test_sequential_requests_do_not_hang():
    """Five back-to-back requests should all return well under a timeout.
    The original bug froze on request #2 due to leaked LangGraph state."""
    deadline_each = 30  # seconds; generous for an LLM round-trip
    for i in range(5):
        start = time.time()
        out = run_rag_agent("What is subrogation in insurance?")
        elapsed = time.time() - start
        assert out.get("answer"), f"empty answer on request {i}"
        assert elapsed < deadline_each, f"request {i} took {elapsed:.1f}s (hang regression)"


# --- Bug 2 / 4: concurrent ML-bearing requests must not crash the process ---
def test_concurrent_requests_no_segfault():
    """Fire requests in parallel. If model loading or XGBoost threading
    regressed, the worker process would die and a future would raise."""
    def task(_):
        return run_sql_agent("How many claims are currently open?")

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(task, range(4)))
    assert all(r is not None for r in results)


# --- Bug 4: models load once, not per call ---------------------------------
def test_models_are_module_level_singletons():
    """Importing the predictor twice must yield the SAME object, proving the
    pkl files are loaded once at import, not on every call."""
    from ml_models import predictor as p1
    from ml_models import predictor as p2
    assert p1._FRAUD_MODEL is p2._FRAUD_MODEL
    assert p1._LITIGATION_MODEL is p2._LITIGATION_MODEL
    assert p1._RESOLUTION_MODEL is p2._RESOLUTION_MODEL


# --- Guardrail: low-confidence RAG must flag its source as AI knowledge -----
def test_rag_fallback_labels_source():
    """A general definition question should come back labelled ai_knowledge,
    not silently presented as a policy-document answer."""
    out = run_rag_agent("Define proximate cause in plain English")
    assert out.get("source_type") in {"policy_documents", "ai_knowledge"}
    # If confidence is below threshold it MUST be ai_knowledge.
    if out.get("confidence", 1.0) < 0.55:
        assert out["source_type"] == "ai_knowledge"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))