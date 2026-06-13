"""
Router eval — does the orchestrator send each query to the right agent?

Metric: overall accuracy + per-class precision/recall + confusion matrix.
Why it matters: the expensive failure is a policy question routed to the SQL
agent (or vice versa), which returns a confident, wrong answer. Misroutes are
silent — only an eval catches them.

Run:  python -m evals.run_router_eval
"""
from __future__ import annotations
from collections import defaultdict

from evals._loader import load_jsonl
from evals.adapter import route_query

CLASSES = ["sql", "rag", "ml", "hybrid"]


def main() -> None:
    rows = load_jsonl("router_testset.jsonl")
    y_true, y_pred = [], []
    errors = []

    for r in rows:
        expected = r["expected"]
        try:
            pred = route_query(r["query"])
        except Exception as e:  # a crash is a routing failure too
            pred = f"ERROR:{type(e).__name__}"
        y_true.append(expected)
        y_pred.append(pred)
        if pred != expected:
            errors.append((r["query"], expected, pred))

    n = len(rows)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    print(f"\n=== ROUTER EVAL ===  ({n} queries)")
    print(f"Overall accuracy: {correct}/{n} = {correct / n:.1%}\n")

    # Per-class precision / recall
    for c in CLASSES:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"  {c:8s}  precision={prec:.2f}  recall={rec:.2f}  (support={sum(t == c for t in y_true)})")

    # Confusion matrix
    print("\nConfusion matrix (rows=true, cols=pred):")
    labels = sorted(set(y_true) | set(y_pred))
    cm = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    header = "true\\pred".ljust(12) + "".join(l[:10].ljust(11) for l in labels)
    print(header)
    for t in labels:
        line = t.ljust(12) + "".join(str(cm[t][p]).ljust(11) for p in labels)
        print(line)

    if errors:
        print(f"\nMisroutes ({len(errors)}):")
        for q, exp, pred in errors:
            print(f"  [{exp} -> {pred}]  {q}")

    # Gate: fail CI if accuracy drops below target.
    TARGET = 0.90
    print(f"\n{'PASS' if correct / n >= TARGET else 'FAIL'} (target {TARGET:.0%})")


if __name__ == "__main__":
    main()