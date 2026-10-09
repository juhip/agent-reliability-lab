from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Set
from .domain import Invariant  # noqa: F401  (re-exported: domains import it from here)
from .screening import looks_like_injection  # noqa: F401  (re-exported for callers)
from .types import AgentDecision


@dataclass
class PolicyResult:
    allowed: bool
    reason: str


class PolicyEngine:
    """Pre-execution check on every proposed decision: is this action permitted at all?
    The post-decision checks (screen, triggers, invariants, verifier) live in domain.run_gate."""
    def __init__(self, allowed_actions: Iterable[str], blocked_actions: Iterable[str] = ()) -> None:
        self.allowed_actions: Set[str] = set(allowed_actions)
        self.blocked_actions: Set[str] = set(blocked_actions)   # allowed in principle, but always need a person

    def validate(self, decision: AgentDecision) -> PolicyResult:
        if decision.action not in self.allowed_actions:
            return PolicyResult(False, "action_not_allowlisted")
        if decision.action in self.blocked_actions:
            return PolicyResult(False, "human_approval_required")
        if not (0.0 <= decision.confidence <= 1.0):
            return PolicyResult(False, "invalid_confidence")
        return PolicyResult(True, "ok")
