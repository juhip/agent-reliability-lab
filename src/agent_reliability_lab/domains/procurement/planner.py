"""One-shot reference planner: reads the scenario directly (like InvoicePlanner reads module data),
then lists the tool calls that justify its answer. It does not screen free text; core's screen (task and
tool results) and the invariants are what stop it approving a case whose data carries instructions."""
from __future__ import annotations
from typing import Any, Dict

from agent_reliability_lab.types import AgentDecision, ToolCall
from .session import Session, line_dict


class ProcurementPlanner:
    def __init__(self, tools) -> None:
        self.tools = tools            # for the scenario allowlist and rulebook only

    def __call__(self, task: Dict[str, Any], tools: Dict[str, str]) -> AgentDecision:
        sc = str(task.get("scenario", "")).strip()
        path = self.tools.scenarios.get(sc)
        if not path:
            return AgentDecision("HUMAN_REVIEW", "Unknown scenario", 1.0)
        try:
            s = Session(path, self.tools.rules)
        except Exception as exc:  # unreadable scenario: a person has to look, not a crash
            return AgentDecision("HUMAN_REVIEW", f"Scenario unreadable: {type(exc).__name__}", 1.0)
        short = sorted(s.shortfalls())
        for c in short:
            s.plan_component(c)
        gate = s.verify()
        checks = [s.release_check(l) for l in gate["lines"]]
        alerts = s.alerts()
        plan = []
        for po, c in zip(s.pos, checks):
            plan.append({**line_dict(po), "release": "HOLD" if c["needs_approval"] else "RELEASE",
                         "approver": c["approver"], "reasons": c["reasons"], "rationale": c["rationale"]})
        calls = [ToolCall("load_scenario", {"scenario": sc}), ToolCall("net_requirements", {"scenario": sc})]
        calls += [ToolCall("plan_component", {"scenario": sc, "component_id": c}) for c in short]
        calls += [ToolCall("check_hard_rules", {"scenario": sc})]
        calls += [ToolCall("release_check", {"scenario": sc, "line_id": l}) for l in gate["lines"]]
        calls += [ToolCall("draft_alerts", {"scenario": sc})]
        final = {"scenario": sc, "plan": plan, "alerts": alerts, "blocked": gate["blocked"]}
        planned = {p["component_id"] for p in plan}
        if gate["blocked"] or any(c not in planned for c in short) or any(p["release"] == "HOLD" for p in plan):
            return AgentDecision("HUMAN_REVIEW", "Plan needs a person", 0.98, tool_calls=calls, final_answer=final)
        return AgentDecision("APPROVE", "Every line clears the gate and needs no approver", 0.95,
                             tool_calls=calls, final_answer=final)
