"""A deterministic stand-in for a model that follows the procurement system prompt exactly.
It tests the loop and the gate; it says nothing about how well a real model orchestrates.
The rubber-stamp and always-escalate baselines are core's approve_on_sight / escalate_on_sight."""
from __future__ import annotations
import json
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from agent_reliability_lab.agentic.standins import case_id_from_goal
from agent_reliability_lab.agentic.types import ModelTurn, ToolCallRequest
from .patterns import flagged_request_fields, flagged_scenario_text
from .queue import QUEUE

_GOAL = SimpleNamespace(queue=QUEUE)        # case_id_from_goal only needs the queue spec


def _all_results(messages) -> Dict[str, List[Any]]:
    """Every result per tool name (plan_component and release_check are called once per part / line)."""
    out: Dict[str, List[Any]] = {}
    for m in messages:
        if m["role"] == "tool":
            for r in m["results"]:
                try:
                    val = json.loads(r["content"])
                except json.JSONDecodeError:
                    val = {}
                if r["is_error"]:
                    val = {"error": True, **val} if isinstance(val, dict) else {"error": True}
                out.setdefault(r["name"], []).append(val)
    return out


def _turn(*calls: Tuple[str, Dict[str, Any]]) -> ModelTurn:
    return ModelTurn(tool_calls=[ToolCallRequest(f"c{i}_{n}", n, a) for i, (n, a) in enumerate(calls)], stop="tool_use")


def careful_model(messages) -> ModelTurn:
    """Follows the system prompt exactly."""
    sc = case_id_from_goal(_GOAL, messages[0]["content"])
    r = _all_results(messages)
    if "hold_plan" in r or ("release_plan" in r and not r["release_plan"][-1].get("error")):
        return ModelTurn(text=f"{sc} decided.")
    hold = lambda why: _turn(("hold_plan", {"scenario": sc, "reason": why}))
    if "release_plan" in r:                                   # refused: do not argue, hold
        return hold("release refused: " + str(r["release_plan"][-1].get("reason")))
    if any(isinstance(v, dict) and v.get("error") for vs in r.values() for v in vs):
        return hold("a tool failed")
    if "load_scenario" not in r:
        return _turn(("read_request", {"scenario": sc}), ("load_scenario", {"scenario": sc}),
                     ("net_requirements", {"scenario": sc}))
    short = r["net_requirements"][-1]
    if "plan_component" not in r and short:
        return _turn(*[("plan_component", {"scenario": sc, "component_id": c}) for c in sorted(short)])
    if "check_hard_rules" not in r:
        return _turn(("check_hard_rules", {"scenario": sc}))
    gate = r["check_hard_rules"][-1]
    if "release_check" not in r and gate["lines"]:
        return _turn(*[("release_check", {"scenario": sc, "line_id": l}) for l in gate["lines"]])
    if flagged_request_fields(r["read_request"][-1]["request"]) or \
            flagged_scenario_text(r["load_scenario"][-1].get("untrusted_text", [])):
        return hold("free text reads like instructions")
    if gate["blocked"]:
        return hold("a line broke a hard rule")
    planned = {l["component_id"] for x in r.get("plan_component", []) for l in x["lines"]}
    if any(c not in planned for c in short):
        return hold("no compliant supplier for a short part")
    if any(x["needs_approval"] for x in r.get("release_check", [])):
        return hold("a line needs an approver")
    return _turn(("release_plan", {"scenario": sc, "reason": "gate passed and every line cleared"}))
