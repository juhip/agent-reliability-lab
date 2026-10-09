"""A deterministic stand-in for a model that follows the refund system prompt exactly.
It tests the loop; it says nothing about how well a real model orchestrates."""
from __future__ import annotations
from typing import Any, Dict, List
from agent_reliability_lab.agentic.standins import call, case_id_from_goal, tool_results, with_queue_parent
from agent_reliability_lab.agentic.types import ModelTurn
from agent_reliability_lab.screening import screen_value
from .domain import REFUND


def _careful(messages: List[Dict[str, Any]]) -> ModelTurn:
    rid = case_id_from_goal(REFUND, messages[0]["content"])
    r = tool_results(messages)
    if "issue_refund" in r or "send_to_manual_review" in r:
        return ModelTurn(text=f"{rid} decided.")

    def review(why: str) -> ModelTurn:
        return call("send_to_manual_review", refund_id=rid, reason=why)

    if "read_refund" not in r:
        return call("read_refund", refund_id=rid)
    req = r["read_refund"].get("refund", {})
    order_id, customer_id, amount = str(req.get("order_id", "")).strip(), str(req.get("customer_id", "")).strip(), req.get("amount")
    if not order_id or not customer_id or isinstance(amount, bool) or not isinstance(amount, (int, float)) or not amount > 0:
        return review("missing or invalid fields")
    for tool, args in (("lookup_order", {"order_id": order_id}), ("lookup_return", {"order_id": order_id}),
                       ("lookup_customer", {"customer_id": customer_id})):
        if tool not in r:
            return call(tool, **args)
        if r[tool].get("error"):
            return review(f"{tool} failed")
    order, ret, cust = r["lookup_order"], r["lookup_return"], r["lookup_customer"]
    if any(screen_value(v, k, REFUND.patterns) for k, v in r.items()):
        return review("instruction-like text in the request or the records")
    if order["customer_id"] != customer_id or not ret["received"]:
        return review("not the customer's order, or return not received")
    if cust["status"] != "active" or cust["flags"]["chargeback_open"]:
        return review("account needs a person")
    if any(i["final_sale"] for i in order["items"]) or amount > 1000:
        return review("final-sale item or large refund")
    if order["delivered_days_ago"] > 30 or round(amount * 100) + round(order["refunded_so_far"] * 100) > round(order["total"] * 100):
        return review("outside the window or over the balance")
    return call("issue_refund", refund_id=rid, reason="eligible and no escalation condition")


careful_model = with_queue_parent(REFUND, _careful)
