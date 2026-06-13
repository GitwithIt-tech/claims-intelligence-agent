"""
SQL agent eval — EXECUTION accuracy, not string match.

We run each question through the SQL agent, take the returned rows, and check
them against a declarative expectation. Two different SQL strings can both be
correct, so we never compare SQL text — we compare results.

Run:  python -m evals.run_sql_eval
"""
from __future__ import annotations

from evals._loader import load_jsonl
from evals.adapter import run_sql_agent


def _first_scalar(rows: list[dict]):
    """Return the single value from a 1x1-ish result set."""
    if not rows:
        return None
    first = rows[0]
    if isinstance(first, dict):
        return list(first.values())[0]
    return first


def check(rows: list[dict], spec: dict) -> tuple[bool, str]:
    kind = spec["type"]
    exp = spec.get("expected")

    if kind == "row_count":
        return len(rows) == exp, f"rows={len(rows)} expected={exp}"

    if kind == "scalar":
        got = _first_scalar(rows)
        tol = spec.get("tolerance")
        if tol is not None and isinstance(got, (int, float)) and isinstance(exp, (int, float)):
            ok = abs(got - exp) <= tol
            return ok, f"got={got} expected={exp}±{tol}"
        return got == exp, f"got={got!r} expected={exp!r}"

    if kind == "top_k_ids":
        got_ids = [r.get("claim_id") or r.get("id") for r in rows][: len(exp)]
        return got_ids == exp, f"got={got_ids} expected={exp}"

    if kind == "contains_keys":
        if not rows:
            return False, "no rows returned"
        keys = set(rows[0].keys())
        missing = [k for k in exp if k not in keys]
        return not missing, f"missing keys={missing}" if missing else "ok"

    return False, f"unknown check type {kind}"


def main() -> None:
    rows = load_jsonl("sql_golden.jsonl")
    passed, results = 0, []

    for r in rows:
        try:
            out = run_sql_agent(r["question"])
            ok, detail = check(out.get("rows", []), r["check"])
        except Exception as e:
            ok, detail = False, f"ERROR {type(e).__name__}: {e}"
            out = {"sql": None}
        passed += ok
        results.append((r["id"], ok, detail, out.get("sql")))

    n = len(rows)
    print(f"\n=== SQL EXECUTION EVAL ===  ({n} questions)")
    print(f"Execution accuracy: {passed}/{n} = {passed / n:.1%}\n")
    for cid, ok, detail, sql in results:
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {cid}: {detail}")
        if not ok and sql:
            print(f"          SQL: {sql}")

    TARGET = 0.90
    print(f"\n{'PASS' if passed / n >= TARGET else 'FAIL'} (target {TARGET:.0%})")


if __name__ == "__main__":
    main()