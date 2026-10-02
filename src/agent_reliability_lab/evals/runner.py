from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
from .metrics import summarize

def load_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def run_cases(runtime, cases: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for case in cases:
        out = runtime.run(case["input"], task_id=case["id"])
        actual = out.get("result", {}).get("action") if out["status"] == "COMPLETED" else "HUMAN_REVIEW"
        expected = case["expected_action"]
        correct = actual == expected
        events = out["trace"]["events"]
        tool_calls = [e for e in events if e["kind"] == "tool_call"]
        tool_failures = [e for e in events if e["kind"] == "tool_result" and not e["payload"]["result"]["ok"]]
        rows.append({
            "id": case["id"],
            "category": case.get("category", "unknown"),
            "expected": expected,
            "actual": actual,
            "correct": correct,
            "tool_call_count": len(tool_calls),
            "tool_failure_count": len(tool_failures),
            "trace_event_count": len(events),
            "failure_type": None if correct else "wrong_final_action",
            "trace": out["trace"],
        })
    return {"summary": summarize(rows), "cases": rows}
