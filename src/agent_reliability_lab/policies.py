from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set
from .types import AgentDecision, ToolResult

# An invariant inspects the proposed final decision against what the tools
# actually returned, and names the violated rule (or returns None).
Invariant = Callable[[AgentDecision, List[ToolResult], Dict[str, Any]], Optional[str]]

INJECTION_PATTERNS = [
    r"ignore (all |any )?(prior|previous) instructions",
    r"disregard .{0,40}(rules|instructions)",
    r"you are now",
    r"approve (this|the) (invoice|payment) (immediately|without)",
    r"approve (everything|all)",
]

def looks_like_injection(text: str) -> bool:
    return any(re.search(p, text, re.I) for p in INJECTION_PATTERNS)

@dataclass
class PolicyResult:
    allowed: bool
    reason: str

class PolicyEngine:
    """Deterministic guardrails around model-proposed actions."""
    def __init__(
        self,
        allowed_actions: Iterable[str],
        approval_actions: Iterable[str] = (),
        invariants: Iterable[Invariant] = (),
    ) -> None:
        self.allowed_actions: Set[str] = set(allowed_actions)
        self.approval_actions: Set[str] = set(approval_actions)
        self.invariants: List[Invariant] = list(invariants)

    def validate(self, decision: AgentDecision) -> PolicyResult:
        """Pre-execution check: is this action even permitted?"""
        if decision.action not in self.allowed_actions:
            return PolicyResult(False, "action_not_allowlisted")
        if decision.action in self.approval_actions:
            return PolicyResult(False, "human_approval_required")
        if not (0.0 <= decision.confidence <= 1.0):
            return PolicyResult(False, "invalid_confidence")
        return PolicyResult(True, "ok")

    def screen_input(self, task: Dict[str, Any]) -> List[str]:
        """Flag free text that reads like instructions. Task fields are data."""
        return [
            f"possible_instruction_in_{key}"
            for key, value in task.items()
            if isinstance(value, str) and looks_like_injection(value)
        ]

    def verify(self, decision: AgentDecision, observations: List[ToolResult], task: Dict[str, Any]) -> List[str]:
        """Post-execution check: do the tool results actually support the decision?"""
        violations: List[str] = []
        for invariant in self.invariants:
            try:
                violation = invariant(decision, observations, task)
            except Exception as exc:
                violation = f"invariant_error:{type(exc).__name__}"
            if violation:
                violations.append(violation)
        return violations
