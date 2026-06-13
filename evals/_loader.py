"""Shared helpers for loading the JSONL test sets (with // comment support)."""
from __future__ import annotations
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def load_jsonl(name: str) -> list[dict]:
    """Load a .jsonl file, skipping blank lines and // comment lines."""
    path = DATA / name
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            rows.append(json.loads(line))
    return rows