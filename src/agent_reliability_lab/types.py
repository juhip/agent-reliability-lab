from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]

@dataclass
class ToolResult:
    name: str
    ok: bool
    output: Any = None
    error: Optional[str] = None
    # The arguments the tool was called with, so a gate can tell which case a result is about.
    arguments: Optional[Dict[str, Any]] = None

@dataclass
class AgentDecision:
    action: str
    rationale: str
    confidence: float
    tool_calls: List[ToolCall] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    final_answer: Optional[Dict[str, Any]] = None
    # False means "run these tool_calls, then ask me again with the results".
    # True (the default) keeps the original one-shot behaviour.
    is_final: bool = True

@dataclass
class TraceEvent:
    kind: str
    payload: Dict[str, Any]

@dataclass
class AgentTrace:
    task_id: str
    events: List[TraceEvent] = field(default_factory=list)

    def add(self, kind: str, **payload: Any) -> None:
        self.events.append(TraceEvent(kind=kind, payload=payload))

    def to_dict(self) -> Dict[str, Any]:
        return {"task_id": self.task_id, "events": [asdict(e) for e in self.events]}

ToolFn = Callable[..., Any]
