"""
agents/rag_agent.py
────────────────────
RAG Agent — wraps the RAG chain for use inside LangGraph.
Answers questions about insurance policy documents.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.rag_chain import ask


def run_rag_agent(question: str) -> dict:
    """
    Run the RAG agent on a policy question.
    Returns answer with citations.
    """
    try:
        result = ask(question)
        return {
            "success": True,
            "agent":   "rag",
            "answer":  result["answer"],
            "sources": result["sources"],
        }
    except Exception as e:
        return {
            "success": False,
            "agent":   "rag",
            "answer":  f"RAG agent error: {str(e)}",
            "sources": [],
        }


if __name__ == "__main__":
    result = run_rag_agent(
        "What is the time limit for submitting a vehicle insurance claim?"
    )
    print(result["answer"])
    print("Sources:", result["sources"])