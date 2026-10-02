from __future__ import annotations
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict
from .types import AgentDecision, AgentTrace
from .tool_registry import ToolRegistry
from .policies import PolicyEngine

Planner = Callable[[Dict[str, Any], Dict[str, str]], AgentDecision]

class AgentRuntime:
    def __init__(self, planner: Planner, tools: ToolRegistry, policy: PolicyEngine) -> None:
        self.planner = planner
        self.tools = tools
        self.policy = policy

    def run(self, task: Dict[str, Any], task_id: str | None = None) -> Dict[str, Any]:
        task_id = task_id or str(uuid.uuid4())[:8]
        trace = AgentTrace(task_id=task_id)
        trace.add("task_received", task=task)

        try:
            decision = self.planner(task, self.tools.descriptions)
        except Exception as exc:
            trace.add("planner_failure", error=f"{type(exc).__name__}: {exc}")
            return {
                "status": "HUMAN_REVIEW",
                "reason": "planner_failure",
                "trace": trace.to_dict(),
            }
        if not isinstance(decision, AgentDecision):
            trace.add("planner_failure", error="planner_returned_invalid_type")
            return {"status": "HUMAN_REVIEW", "reason": "invalid_planner_output", "trace": trace.to_dict()}
        trace.add("decision_proposed", decision=asdict(decision))

        policy = self.policy.validate(decision)
        trace.add("policy_check", allowed=policy.allowed, reason=policy.reason)
        if not policy.allowed:
            return {
                "status": "HUMAN_REVIEW",
                "reason": policy.reason,
                "decision": asdict(decision),
                "trace": trace.to_dict(),
            }

        tool_outputs = []
        for index, call in enumerate(decision.tool_calls):
            trace.add("tool_call", index=index, name=call.name, arguments=call.arguments)
            result = self.tools.call(call.name, call.arguments)
            tool_outputs.append(asdict(result))
            trace.add("tool_result", index=index, result=asdict(result))
            if not result.ok:
                return {
                    "status": "HUMAN_REVIEW",
                    "reason": "tool_failure",
                    "decision": asdict(decision),
                    "tool_outputs": tool_outputs,
                    "trace": trace.to_dict(),
                }

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
            "trace": trace.to_dict(),
        }
