"""Trivial planners. A metric that cannot separate these from the real agent is not measuring anything.
They take the domain's action names, so they work for any domain."""
from __future__ import annotations
from typing import Any, Dict
from agent_reliability_lab.types import AgentDecision


class AlwaysEscalate:
    def __init__(self, action: str = "HUMAN_REVIEW") -> None:
        self.action = action

    def __call__(self, task: Dict[str, Any], tools: Dict[str, str]) -> AgentDecision:
        return AgentDecision(self.action, "baseline: escalate everything", 1.0)


class AlwaysApprove:
    """Proposes the gated action without looking at anything; the gate should catch it."""
    def __init__(self, action: str = "APPROVE") -> None:
        self.action = action

    def __call__(self, task: Dict[str, Any], tools: Dict[str, str]) -> AgentDecision:
        return AgentDecision(self.action, "baseline: approve everything", 1.0)


def baselines_for(domain: Any) -> Dict[str, Any]:
    return {"always-escalate": AlwaysEscalate(domain.fail_safe), "always-approve": AlwaysApprove(domain.gated_actions[0])}
