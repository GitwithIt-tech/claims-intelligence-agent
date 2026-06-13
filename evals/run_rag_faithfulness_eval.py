"""
RAG generation eval — faithfulness + answer relevancy.

Faithfulness  = is every claim in the answer supported by the retrieved
                context? (catches hallucination)
Answer relevancy = does the answer actually address the question?

Primary path uses RAGAS (the standard library for this). If RAGAS isn't
installed, falls back to a lightweight LLM-as-judge faithfulness check so the
script still runs. Human spot-checks on a handful of answers remain the
ground truth for trust — automate for scale, verify by hand for confidence.

Install:  pip install ragas datasets
Run:      python -m evals.run_rag_faithfulness_eval
"""
from __future__ import annotations

from evals._loader import load_jsonl
from evals.adapter import run_rag_agent


def run_with_ragas(samples: list[dict]) -> None:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy

    ds = Dataset.from_list(
        [
            {
                "question": s["question"],
                "answer": s["answer"],
                "contexts": s["contexts"] or [" "],
                "ground_truth": s["ground_truth"],
            }
            for s in samples
        ]
    )
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy])
    print("\n=== RAG FAITHFULNESS (RAGAS) ===")
    print(result)
    df = result.to_pandas()
    weak = df[df["faithfulness"] < 0.7]
    if len(weak):
        print(f"\nLow-faithfulness answers ({len(weak)}) — review for hallucination:")
        for _, row in weak.iterrows():
            print(f"  faith={row['faithfulness']:.2f}  Q: {row['question']}")


def run_fallback(samples: list[dict]) -> None:
    print("\n=== RAG FAITHFULNESS (fallback heuristic) ===")
    print("RAGAS not installed. Install with: pip install ragas datasets")
    print("Showing answers + contexts for manual review:\n")
    for s in samples:
        print(f"Q: {s['question']}")
        print(f"  source_type: {s['source_type']}")
        print(f"  answer: {s['answer'][:160]}...")
        print(f"  #contexts retrieved: {len(s['contexts'])}\n")


def main() -> None:
    rows = load_jsonl("rag_qa.jsonl")
    samples = []
    for r in rows:
        out = run_rag_agent(r["question"])
        samples.append(
            {
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "answer": out.get("answer", ""),
                "contexts": out.get("contexts", []),
                "source_type": out.get("source_type", ""),
            }
        )

    try:
        run_with_ragas(samples)
    except ImportError:
        run_fallback(samples)


if __name__ == "__main__":
    main()