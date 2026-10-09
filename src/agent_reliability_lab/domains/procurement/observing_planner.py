"""Deterministic loop planner: decides from tool results only, never from the scenario file.
This is the shape a model-backed procurement planner would have to follow."""
from __future__ import annotations
from typing import Any, Dict, List

from agent_reliability_lab.types import AgentDecision, ToolCall, ToolResult
from .patterns import flagged_request_fields, flagged_scenario_text


def _review(reason: str, final: Dict[str, Any] | None = None) -> AgentDecision:
    return AgentDecision("HUMAN_REVIEW", reason, 0.99, final_answer=final)


def _calls(why: str, *calls: ToolCall) -> AgentDecision:
    return AgentDecision("HUMAN_REVIEW", why, 0.5, tool_calls=list(calls), is_final=False)


class ObservingProcurementPlanner:
    """load -> net -> plan every short part -> hard-rule gate -> release check per line -> alerts -> decide.
    Batches calls per stage so a 20-part scenario still finishes in six planner turns."""

    def __call__(self, task: Dict[str, Any], tools: Dict[str, Any], observations: List[ToolResult]) -> AgentDecision:
        sc = str(task.get("scenario", "")).strip()
        if not sc:
            return _review("No scenario given")
        failed = [r for r in observations if not r.ok]
        if failed:
            return _review(f"Tool {failed[0].name} failed: {failed[0].error}")
        seen: Dict[str, List[ToolResult]] = {}
        for r in observations:
            seen.setdefault(r.name, []).append(r)
        one = lambda name: seen[name][-1].output if name in seen else None

        if "load_scenario" not in seen:
            return _calls("Need the scenario and the shortfall",
                          ToolCall("load_scenario", {"scenario": sc}), ToolCall("net_requirements", {"scenario": sc}))
        short = one("net_requirements")
        if "plan_component" not in seen and short:
            return _calls("Plan each short part", *[ToolCall("plan_component", {"scenario": sc, "component_id": c})
                                                     for c in sorted(short)])
        if "check_hard_rules" not in seen:
            return _calls("Gate the plan", ToolCall("check_hard_rules", {"scenario": sc}))
        gate = one("check_hard_rules")
        if "release_check" not in seen and gate["lines"]:
            return _calls("Who has to approve each line",
                          *[ToolCall("release_check", {"scenario": sc, "line_id": l}) for l in gate["lines"]])
        if "draft_alerts" not in seen:
            return _calls("Draft alerts", ToolCall("draft_alerts", {"scenario": sc}))

        lines = {l["line_id"]: l for r in seen.get("plan_component", []) for l in r.output["lines"]}
        checks = {r.output["line_id"]: r.output for r in seen.get("release_check", [])}
        plan = []
        for lid in gate["lines"]:
            line = dict(lines[lid])
            c = checks[lid]
            line.update(release="HOLD" if c["needs_approval"] else "RELEASE", approver=c["approver"],
                        reasons=c["reasons"], rationale=c["rationale"])
            plan.append(line)
        final = {"scenario": sc, "plan": plan, "alerts": one("draft_alerts"), "blocked": gate["blocked"]}

        flagged = flagged_request_fields(task) + flagged_scenario_text(one("load_scenario")["untrusted_text"])
        if flagged:
            return _review(f"Free text reads like instructions ({', '.join(flagged)}); a person must look", final)
        if gate["blocked"]:
            return _review("Some planned lines broke a hard rule and were removed", final)
        missing = [c for c in (short or {}) if not any(l["component_id"] == c for l in plan)]
        if missing:
            return _review(f"No compliant supplier for {', '.join(missing)}", final)
        held = [l for l in plan if l["release"] == "HOLD"]
        if held:
            approvers = sorted({l["approver"] for l in held})
            return _review(f"{len(held)} of {len(plan)} lines need approval ({', '.join(approvers)})", final)
        return AgentDecision("APPROVE", "Every line passes the hard-rule gate and needs no approver", 0.95,
                             evidence=[f"lines:{len(plan)}"], final_answer=final)
