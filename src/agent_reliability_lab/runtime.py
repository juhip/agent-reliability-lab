from __future__ import annotations
import inspect
import time
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional
from .types import AgentDecision, AgentTrace, ToolResult
from .tool_registry import ToolRegistry
from .policies import PolicyEngine

Planner = Callable[..., AgentDecision]

# Escalations caused by the system misbehaving, not by the task being exceptional.
SYSTEM_ERROR_SOURCES = {"planner_failure", "invalid_planner_output", "tool_failure", "max_steps"}


def _accepts_observations(planner: Planner) -> bool:
    """Original planners take (task, tools); observing planners take (task, tools, observations)."""
    try:
        params = [
            p for p in inspect.signature(planner).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        return False
    return len(params) >= 3


class AgentRuntime:
    """plan -> policy -> act -> observe loop.

    A planner may return a non-final decision (is_final=False) with tool calls;
    the runtime runs them and calls the planner again with the results. The first
    final decision ends the loop, is checked against the tool results, and is
    downgraded to HUMAN_REVIEW if the evidence does not support it.
    """

    def __init__(self, planner: Planner, tools: ToolRegistry, policy: PolicyEngine, max_steps: int = 6) -> None:
        self.planner = planner
        self.tools = tools
        self.policy = policy
        self.max_steps = max_steps

    def run(self, task: Dict[str, Any], task_id: str | None = None) -> Dict[str, Any]:
        trace = AgentTrace(task_id=task_id or str(uuid.uuid4())[:8])
        started = time.perf_counter()
        out = self._run(task, trace)
        out["latency_s"] = round(time.perf_counter() - started, 4)
        out["trace"] = trace.to_dict()
        return out

    def _plan(self, task: Dict[str, Any], observations: List[ToolResult]) -> AgentDecision:
        if _accepts_observations(self.planner):
            return self.planner(task, self.tools.schemas, list(observations))
        return self.planner(task, self.tools.descriptions)

    @staticmethod
    def _escalate(source: str, reason: str, decision: Optional[AgentDecision] = None, **extra: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "status": "HUMAN_REVIEW",
            "reason": reason,
            "escalation_source": source,
            "planner_action": decision.action if decision else None,
        }
        if decision:
            out["decision"] = asdict(decision)
        out.update(extra)
        return out

    def _run(self, task: Dict[str, Any], trace: AgentTrace) -> Dict[str, Any]:
        trace.add("task_received", task=task)
        flags = self.policy.screen_input(task)
        if flags:
            trace.add("input_flagged", flags=flags)

        observations: List[ToolResult] = []
        tool_outputs: List[Dict[str, Any]] = []
        decision: Optional[AgentDecision] = None

        for step in range(self.max_steps):
            try:
                decision = self._plan(task, observations)
            except Exception as exc:
                trace.add("planner_failure", error=f"{type(exc).__name__}: {exc}")
                return self._escalate("planner_failure", "planner_failure")
            if not isinstance(decision, AgentDecision):
                trace.add("planner_failure", error="planner_returned_invalid_type")
                return self._escalate("invalid_planner_output", "invalid_planner_output")
            trace.add("decision_proposed", step=step, decision=asdict(decision))

            check = self.policy.validate(decision)
            trace.add("policy_check", allowed=check.allowed, reason=check.reason)
            if not check.allowed:
                return self._escalate("policy", check.reason, decision, tool_outputs=tool_outputs)

            for index, call in enumerate(decision.tool_calls):
                trace.add("tool_call", index=index, name=call.name, arguments=call.arguments)
                result = self.tools.call(call.name, call.arguments)
                observations.append(result)
                tool_outputs.append(asdict(result))
                trace.add("tool_result", index=index, result=asdict(result))
                if not result.ok:
                    if decision.is_final:
                        return self._escalate("tool_failure", "tool_failure", decision, tool_outputs=tool_outputs)
                    break  # observing mode: the planner sees the failure and decides
            if decision.is_final:
                break
        else:
            trace.add("planner_failure", error="max_steps_exceeded")
            return self._escalate("max_steps", "max_steps_exceeded", decision, tool_outputs=tool_outputs)

        violations = self.policy.verify(decision, observations, task)
        if flags and decision.action == "APPROVE":
            violations.append("approve_with_flagged_input")
        if violations:
            trace.add("policy_violation", violations=violations)
            return self._escalate(
                "policy_invariant", "; ".join(violations), decision,
                violations=violations, tool_outputs=tool_outputs,
            )

        if decision.action == "HUMAN_REVIEW":
            trace.add("escalated", reason=decision.rationale)
            return self._escalate("planner", decision.rationale, decision, tool_outputs=tool_outputs)

        final = dict(decision.final_answer or {})
        final.setdefault("action", decision.action)
        final.setdefault("confidence", decision.confidence)
        final.setdefault("evidence", decision.evidence)
        trace.add("completed", final=final)
        return {
            "status": "COMPLETED",
            "result": final,
            "decision": asdict(decision),
            "tool_outputs": tool_outputs,
            "planner_action": decision.action,
        }
