"""Trivial planners. A metric that cannot separate these from the real agent is not measuring anything."""
from __future__ import annotations
from typing import Any, Dict
from agent_reliability_lab.types import AgentDecision


class AlwaysEscalate:
    def __call__(self, task: Dict[str, Any], tools: Dict[str, str]) -> AgentDecision:
        return AgentDecision("HUMAN_REVIEW", "baseline: escalate everything", 1.0)


class AlwaysApprove:
    """Approves without looking at anything; the policy invariants should catch it."""
    def __call__(self, task: Dict[str, Any], tools: Dict[str, str]) -> AgentDecision:
        return AgentDecision("APPROVE", "baseline: approve everything", 1.0)
