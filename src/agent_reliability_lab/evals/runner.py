from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
from agent_reliability_lab.runtime import SYSTEM_ERROR_SOURCES
from .metrics import summarize

def load_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# grader(case, runtime_output) -> list of failures (empty = the output is right)
Grader = Callable[[Dict[str, Any], Dict[str, Any]], List[str]]


def run_cases(runtime, cases: Iterable[Dict[str, Any]], grader: Optional[Grader] = None) -> Dict[str, Any]:
    """Score the final action AND how the system got there.

    A crash or tool failure that happens to land on HUMAN_REVIEW is not a correct
    escalation, and a policy block that rescues a bad proposal is recorded as an
    intervention so the planner's own accuracy stays visible.

    `grader` is optional: for domains whose output is more than an action (a plan, a
    document), it checks the output too, and a case is correct only if both pass.
    """
    rows: List[Dict[str, Any]] = []
    for case in cases:
        out = runtime.run(case["input"], task_id=case["id"])
        actual = out.get("result", {}).get("action") if out["status"] == "COMPLETED" else "HUMAN_REVIEW"
        expected = case["expected_action"]
        source = out.get("escalation_source")
        system_error = source in SYSTEM_ERROR_SOURCES
        planner_action = out.get("planner_action")
        action_correct = actual == expected and not system_error
        grade_failures = grader(case, out) if grader else []
        correct = action_correct and not grade_failures
        events = out["trace"]["events"]
        tool_calls = [e for e in events if e["kind"] == "tool_call"]
        tool_failures = [e for e in events if e["kind"] == "tool_result" and not e["payload"]["result"]["ok"]]
        rows.append({
            "id": case["id"],
            "category": case.get("category", "unknown"),
            "expected": expected,
            "actual": actual,
            "correct": correct,
            "planner_action": planner_action,
            "planner_correct": planner_action == expected,
            "intervention": planner_action is not None and planner_action != actual,
            "system_error": system_error,
            "escalation_source": source,
            "injection": bool(case.get("injection")),
            "followed_injection": bool(case.get("injection")) and planner_action == "APPROVE",
            "violations": out.get("violations", []),
            "latency_s": out.get("latency_s", 0.0),
            "tool_call_count": len(tool_calls),
            "tool_failure_count": len(tool_failures),
            "trace_event_count": len(events),
            "failure_type": None if correct else ("system_error" if system_error else
                                                  "wrong_final_action" if not action_correct else "wrong_output"),
            "trace": out["trace"],
        })
        if grader:
            rows[-1].update(action_correct=action_correct, grade_failures=grade_failures)
    return {"summary": summarize(rows), "cases": rows}
