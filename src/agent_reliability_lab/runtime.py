from __future__ import annotations
import inspect
import time
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional
from .agentic.types import READ
from .domain import Domain, run_gate
from .policies import PolicyEngine
from .screening import screen_value
from .tool_registry import ToolRegistry
from .types import AgentDecision, AgentTrace, ToolResult

Planner = Callable[..., AgentDecision]

# Fail-safe outcomes caused by the system misbehaving, not by the task being exceptional.
SYSTEM_ERROR_SOURCES = {"planner_failure", "invalid_planner_output", "tool_failure", "max_steps", "executor_failure"}


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
    """plan -> policy -> act -> observe loop, then the gate.

    A planner may return a non-final decision (is_final=False) with tool calls; the runtime runs
    them and calls the planner again with the results. The first final decision ends the loop.
    A gated action (domain.gated_actions) then has to pass domain.run_gate, and only then is the
    domain's executor called. Anything blocked, and any system failure, ends on domain.fail_safe.
    """

    def __init__(self, planner: Planner, tools: ToolRegistry, policy: PolicyEngine, domain: Domain,
                 max_steps: int | None = None) -> None:
        self.planner = planner
        self.tools = tools
        self.policy = policy
        self.domain = domain
        self.max_steps = max_steps or domain.limits["max_steps"]

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

    def _fetch(self, name: str, arguments: Dict[str, Any]) -> Any:
        """The verifier's own read access: READ tools only, outside the planner's observations."""
        if not self.tools.is_read_only(name):
            raise PermissionError(f"verifier may only fetch read-only tools, not {name!r}")
        result = self.tools.call(name, arguments)
        if not result.ok:
            raise LookupError(result.error)
        return result.output

    def _fail_safe(self, source: str, reason: str, decision: Optional[AgentDecision] = None, **extra: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "status": self.domain.fail_safe,
            "outcome": self.domain.fail_safe,
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
        flags = screen_value(task, "task", self.domain.patterns)
        if flags:
            trace.add("input_flagged", flags=[h.path for h in flags])

        observations: List[ToolResult] = []
        tool_outputs: List[Dict[str, Any]] = []
        decision: Optional[AgentDecision] = None

        for step in range(self.max_steps):
            try:
                decision = self._plan(task, observations)
            except Exception as exc:
                trace.add("planner_failure", error=f"{type(exc).__name__}: {exc}")
                return self._fail_safe("planner_failure", "planner_failure")
            if not isinstance(decision, AgentDecision):
                trace.add("planner_failure", error="planner_returned_invalid_type")
                return self._fail_safe("invalid_planner_output", "invalid_planner_output")
            trace.add("decision_proposed", step=step, decision=asdict(decision))

            check = self.policy.validate(decision)
            trace.add("policy_check", allowed=check.allowed, reason=check.reason)
            if not check.allowed:
                return self._fail_safe("policy", check.reason, decision, tool_outputs=tool_outputs)

            for index, call in enumerate(decision.tool_calls):
                trace.add("tool_call", index=index, name=call.name, arguments=call.arguments)
                result = self.tools.call(call.name, call.arguments)
                observations.append(result)
                tool_outputs.append(asdict(result))
                trace.add("tool_result", index=index, result=asdict(result))
                if result.ok:
                    hits = screen_value(result.output, f"{call.name}[{len(observations) - 1}]", self.domain.patterns)
                    if hits:
                        trace.add("tool_result_flagged", flags=[h.path for h in hits])
                if not result.ok:
                    if decision.is_final:
                        return self._fail_safe("tool_failure", "tool_failure", decision, tool_outputs=tool_outputs)
                    break  # observing mode: the planner sees the failure and decides
            if decision.is_final:
                break
        else:
            trace.add("planner_failure", error="max_steps_exceeded")
            return self._fail_safe("max_steps", "max_steps_exceeded", decision, tool_outputs=tool_outputs)

        if decision.action in self.domain.gated_actions:
            gate = run_gate(self.domain, decision, task, observations, self._fetch)
            trace.add("gate", **gate.details())
            if not gate.passed:
                return self._fail_safe(
                    "gate", "; ".join(gate.violations), decision, violations=gate.violations,
                    triggers_fired=[h.name for h in gate.triggers_fired], tool_outputs=tool_outputs,
                )

        if decision.action == self.domain.fail_safe:
            trace.add("escalated", reason=decision.rationale)
            return self._fail_safe("planner", decision.rationale, decision, tool_outputs=tool_outputs)

        final = dict(decision.final_answer or {})
        final.setdefault("action", decision.action)
        final.setdefault("confidence", decision.confidence)
        final.setdefault("evidence", decision.evidence)
        execution = None
        if decision.action in self.domain.gated_actions:
            try:
                execution = self.domain.executor(decision.action, task, final)
            except Exception as exc:
                trace.add("executor_failure", error=f"{type(exc).__name__}: {exc}")
                return self._fail_safe("executor_failure", "executor_failure", decision, tool_outputs=tool_outputs)
            trace.add("executed", result=execution)
        trace.add("completed", final=final)
        return {
            "status": "COMPLETED",
            "outcome": decision.action,
            "result": final,
            "execution": execution,
            "decision": asdict(decision),
            "tool_outputs": tool_outputs,
            "planner_action": decision.action,
        }


def build_runtime(domain: Domain, planner: Planner, max_steps: int | None = None) -> AgentRuntime:
    """The pipeline runtime for any domain: its tools in an allowlisted registry, its actions allowlisted."""
    tools = ToolRegistry()
    for t in domain.tools:
        tools.register(t.name, t.description, t.handler, parameters=t.parameters, read_only=t.tier == READ)
    return AgentRuntime(planner, tools, PolicyEngine(allowed_actions=domain.actions), domain, max_steps)
