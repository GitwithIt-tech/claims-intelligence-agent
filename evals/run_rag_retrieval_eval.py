"""
RAG retrieval eval — recall@k, MRR, and a threshold sweep.

Two things measured:
1. Retrieval quality: for policy questions, does the correct PDF page appear
   in the top-k retrieved chunks? -> recall@k and MRR.
2. Threshold justification: sweep the confidence cutoff and show that 0.55
   (your chosen value) sits where policy questions stay above and general
   questions fall below. This is how you DEFEND the number in an interview
   instead of saying "it felt right".

Run:  python -m evals.run_rag_retrieval_eval
"""
from __future__ import annotations

from evals._loader import load_jsonl
from evals.adapter import run_rag_agent

K = 3


def _page_match(sources: list[dict], relevant: list[dict]) -> int | None:
    """Return the 1-based rank of the first retrieved source that matches a
    relevant {doc, page}, or None if not in the retrieved set."""
    rel = {(r["doc"].lower(), r["page"]) for r in relevant}
    for rank, s in enumerate(sources, start=1):
        if (str(s.get("doc", "")).lower(), s.get("page")) in rel:
            return rank
    return None


def main() -> None:
    rows = load_jsonl("rag_retrieval.jsonl")
    policy_rows = [r for r in rows if r["is_policy"]]
    general_rows = [r for r in rows if not r["is_policy"]]

    # ---- Retrieval quality on policy questions ----
    hits, recip_ranks = 0, []
    for r in policy_rows:
        out = run_rag_agent(r["question"])
        rank = _page_match(out.get("sources", [])[:K], r["relevant"])
        if rank:
            hits += 1
            recip_ranks.append(1.0 / rank)
        else:
            recip_ranks.append(0.0)

    n_pol = len(policy_rows) or 1
    print(f"\n=== RAG RETRIEVAL EVAL ===")
    print(f"Policy questions: {len(policy_rows)} | General questions: {len(general_rows)}")
    print(f"Recall@{K}: {hits}/{len(policy_rows)} = {hits / n_pol:.1%}")
    print(f"MRR@{K}:    {sum(recip_ranks) / n_pol:.3f}")

    # ---- Threshold sweep: do confidences separate the two groups? ----
    print("\n--- Confidence threshold analysis ---")
    pol_conf = [run_rag_agent(r["question"]).get("confidence", 0.0) for r in policy_rows]
    gen_conf = [run_rag_agent(r["question"]).get("confidence", 0.0) for r in general_rows]

    def summary(name, vals):
        if not vals:
            return
        vals = sorted(vals)
        print(f"  {name}: min={vals[0]:.2f} median={vals[len(vals)//2]:.2f} max={vals[-1]:.2f}")

    summary("policy  ", pol_conf)
    summary("general ", gen_conf)

    print("\n  threshold | policy kept | general (correctly) dropped")
    for t in [0.40, 0.50, 0.55, 0.60, 0.70]:
        kept = sum(c >= t for c in pol_conf) / (len(pol_conf) or 1)
        dropped = sum(c < t for c in gen_conf) / (len(gen_conf) or 1)
        marker = "  <- current" if abs(t - 0.55) < 1e-9 else ""
        print(f"     {t:.2f}   |   {kept:5.0%}     |        {dropped:5.0%}{marker}")
    print("\nThe best threshold maximises policy-kept AND general-dropped together.")


if __name__ == "__main__":
    main()