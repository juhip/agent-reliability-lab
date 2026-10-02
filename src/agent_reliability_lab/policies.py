from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Set
from .types import AgentDecision

@dataclass
class PolicyResult:
    allowed: bool
    reason: str

class PolicyEngine:
    """Deterministic guardrails around model-proposed actions."""
    def __init__(self, allowed_actions: Iterable[str], approval_actions: Iterable[str] = ()) -> None:
        self.allowed_actions: Set[str] = set(allowed_actions)
        self.approval_actions: Set[str] = set(approval_actions)

    def validate(self, decision: AgentDecision) -> PolicyResult:
        if decision.action not in self.allowed_actions:
            return PolicyResult(False, "action_not_allowlisted")
        if decision.action in self.approval_actions:
            return PolicyResult(False, "human_approval_required")
        if not (0.0 <= decision.confidence <= 1.0):
            return PolicyResult(False, "invalid_confidence")
        return PolicyResult(True, "ok")
