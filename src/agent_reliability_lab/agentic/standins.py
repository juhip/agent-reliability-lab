"""Deterministic stand-in models for orchestrated mode. They test the loop and the gate and give
the evals their baselines; they say nothing about how any real model behaves.

approve_on_sight / escalate_on_sight work for any domain with a queue spec. A domain that wants
a stand-in that follows its rules ("careful") writes its own, using the helpers here."""
from __future__ import annotations
import json
import re
from typing import Any, Callable, Dict, List
from agent_reliability_lab.domain import Domain
from .types import ModelTurn, ToolCallRequest

Fn = Callable[[List[Dict[str, Any]]], ModelTurn]


def tool_results(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The latest result per tool name; errors are marked with "error": True."""
    out: Dict[str, Any] = {}
    for m in messages:
        if m["role"] == "tool":
            for r in m["results"]:
                try:
                    body = json.loads(r["content"])
                    out[r["name"]] = {"error": True, **body} if r["is_error"] else body
                except json.JSONDecodeError:
                    out[r["name"]] = {"error": True}
    return out


def call(name: str, **args: Any) -> ModelTurn:
    return ModelTurn(tool_calls=[ToolCallRequest(f"c_{name}", name, args)], stop="tool_use")


def case_id_from_goal(domain: Domain, text: str) -> str:
    prefix, _, suffix = domain.queue.goal_template.partition("{id}")
    m = re.search(re.escape(prefix) + r"(.+?)" + (re.escape(suffix[:1]) if suffix else r"$"), text)
    return m.group(1) if m else ""


def queue_parent(domain: Domain) -> Fn:
    """Queue mode: list the queue, delegate one case at a time, then confirm nothing is left."""
    spec = domain.queue
    child_tools = [f"read_{spec.case_noun}", *domain.read_tools, *spec.decision_tools.values()]

    def fn(messages: List[Dict[str, Any]]) -> ModelTurn:
        r = tool_results(messages)
        calls = [c["name"] for m in messages if m["role"] == "assistant" for c in m["tool_calls"]]
        if "list_queue" not in r:
            return call("list_queue")
        ids = [c["id"] for c in r["list_queue"]["cases"]]
        n = calls.count("delegate")
        if n < len(ids):
            return call("delegate", task=spec.goal_for(ids[n]), tools=child_tools)
        if calls.count("list_queue") < 2:
            return call("list_queue")
        return ModelTurn(text="Queue cleared.")
    return fn


def with_queue_parent(domain: Domain, per_case: Fn) -> Fn:
    parent = queue_parent(domain)
    return lambda messages: parent(messages) if messages[0]["content"] == domain.queue.queue_goal else per_case(messages)


def approve_on_sight(domain: Domain) -> Fn:
    """Calls the first gated decision tool without looking at anything; if refused, takes the fail-safe."""
    spec = domain.queue
    gated_tool, safe_tool, idf = spec.decision_tools[domain.gated_actions[0]], spec.decision_tools[domain.fail_safe], domain.case_id_field

    def fn(messages: List[Dict[str, Any]]) -> ModelTurn:
        case_id = case_id_from_goal(domain, messages[0]["content"])
        r = tool_results(messages)
        if (gated_tool in r and not r[gated_tool].get("error")) or safe_tool in r:
            return ModelTurn(text="done")
        if gated_tool in r:
            return call(safe_tool, **{idf: case_id, "reason": "approval refused"})
        return call(gated_tool, **{idf: case_id, "reason": "looks fine"})
    return fn


def escalate_on_sight(domain: Domain) -> Fn:
    spec = domain.queue
    safe_tool, idf = spec.decision_tools[domain.fail_safe], domain.case_id_field

    def fn(messages: List[Dict[str, Any]]) -> ModelTurn:
        if safe_tool in tool_results(messages):
            return ModelTurn(text="done")
        return call(safe_tool, **{idf: case_id_from_goal(domain, messages[0]["content"]), "reason": "baseline: escalate"})
    return fn
