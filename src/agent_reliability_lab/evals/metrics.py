from __future__ import annotations
from collections import Counter, defaultdict
from typing import Any, Dict, List

def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    statuses = Counter(r["actual"] for r in rows)
    failures = Counter(r.get("failure_type", "none") for r in rows if not r["correct"])
    by_category: Dict[str, Dict[str, int | float]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in rows:
        bucket = by_category[r["category"]]
        bucket["n"] += 1
        bucket["correct"] += int(r["correct"])
    category_accuracy = {
        category: {
            "n": int(values["n"]),
            "accuracy": round(int(values["correct"]) / int(values["n"]), 3),
        }
        for category, values in by_category.items()
    }
    return {
        "n": n,
        "accuracy": round(correct / n, 3) if n else 0.0,
        "status_counts": dict(statuses),
        "failure_counts": dict(failures),
        "category_accuracy": category_accuracy,
        "avg_tool_calls": round(sum(r.get("tool_call_count", 0) for r in rows) / n, 2) if n else 0.0,
        "tool_failures": sum(r.get("tool_failure_count", 0) for r in rows),
    }
