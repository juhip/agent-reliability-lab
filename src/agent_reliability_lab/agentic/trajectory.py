from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List
from agent_reliability_lab.types import ToolResult


@dataclass
class Event:
    step: int
    kind: str            # tool_call | denied | gate_passed | unknown_tool | invalid_args
    name: str
    arguments: Dict[str, Any]
    ok: bool = True
    result: Any = None
    note: str = ""


@dataclass
class Trajectory:
    goal: str
    depth: int = 0
    events: List[Event] = field(default_factory=list)
    children: List["Trajectory"] = field(default_factory=list)
    final_text: str = ""
    stop_reason: str = ""            # final | max_steps | budget | refusal | max_tokens | error
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    truncated: int = 0               # tool results cut down before the model saw them

    def tool_results(self) -> List[ToolResult]:
        """Successful and failed tool executions as ToolResults, for invariants to inspect."""
        return [ToolResult(e.name, e.ok, e.result if e.ok else None, None if e.ok else str(e.result), e.arguments)
                for e in self.events if e.kind == "tool_call"]

    def executed(self) -> List[Event]:
        return [e for e in self.events if e.kind == "tool_call"]

    def metrics(self) -> Dict[str, Any]:
        """Totals over this run and every sub-agent it spawned."""
        runs = [self] + [d for c in self.children for d in c._flatten()]
        events = [e for r in runs for e in r.events]
        calls = [e for e in events if e.kind == "tool_call"]
        redundant = 0                     # per agent: two sub-agents each looking up the same PO is not waste
        for r in runs:
            seen = set()
            for e in (x for x in r.events if x.kind == "tool_call"):
                key = (e.name, json.dumps(e.arguments, sort_keys=True, default=str))
                if e.ok and key in seen:
                    redundant += 1
                seen.add(key)
        return {
            "steps": sum(r.steps for r in runs),
            "tool_calls": len(calls),
            "tool_errors": sum(not e.ok for e in calls),
            "denied": sum(e.kind == "denied" for e in events),
            "unknown_tools": sum(e.kind == "unknown_tool" for e in events),
            "invalid_args": sum(e.kind == "invalid_args" for e in events),
            "redundant_calls": redundant,
            "truncated_results": sum(r.truncated for r in runs),
            "subagents": len(runs) - 1,
            "input_tokens": sum(r.input_tokens for r in runs),
            "output_tokens": sum(r.output_tokens for r in runs),
            "latency_s": round(self.latency_s, 4),
            "stop_reason": self.stop_reason,
        }

    def _flatten(self) -> List["Trajectory"]:
        return [self] + [d for c in self.children for d in c._flatten()]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal, "depth": self.depth, "stop_reason": self.stop_reason, "final_text": self.final_text,
            "events": [e.__dict__ for e in self.events],
            "children": [c.to_dict() for c in self.children], "metrics": self.metrics(),
        }
