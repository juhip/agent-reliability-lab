"""Deterministic refund planners for the pipeline. They decide from tool results only."""
from __future__ import annotations
import math
from typing import Any, Dict, List
from agent_reliability_lab.screening import screen
from agent_reliability_lab.types import AgentDecision, ToolCall, ToolResult
from .domain import REFUND

REFUND_WINDOW_DAYS = 30
LARGE_REFUND = 1000


def _review(reason: str) -> AgentDecision:
    return AgentDecision("MANUAL_REVIEW", reason, 0.95)


def _call(name: str, **arguments: Any) -> AgentDecision:
    return AgentDecision("MANUAL_REVIEW", f"need {name}", 0.5, [ToolCall(name, arguments)], is_final=False)


class ObservingRefundPlanner:
    """Follows the written policy, including the escalation conditions."""
    window_days = REFUND_WINDOW_DAYS
    checks_account = True
    screens_text = True

    def __call__(self, task: Dict[str, Any], tools: Dict[str, Any], observations: List[ToolResult]) -> AgentDecision:
        order_id, customer_id, amount = task.get("order_id"), task.get("customer_id"), task.get("amount")
        if not all(isinstance(v, str) and v.strip() for v in (task.get("refund_id"), order_id, customer_id)):
            return _review("missing request fields")
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return _review("amount is not numeric")
        if not math.isfinite(amount) or amount <= 0:
            return _review("amount must be positive")

        seen = {r.name: r for r in observations}
        if "lookup_order" not in seen:
            return _call("lookup_order", order_id=order_id.strip())
        if not seen["lookup_order"].ok:
            return _review("order not found")
        order = seen["lookup_order"].output
        if order["customer_id"] != customer_id.strip():
            return _review("order belongs to another customer")
        if "lookup_return" not in seen:
            return _call("lookup_return", order_id=order_id.strip())
        if not seen["lookup_return"].ok or not seen["lookup_return"].output["received"]:
            return _review("return not received")
        if "lookup_customer" not in seen:
            return _call("lookup_customer", customer_id=customer_id.strip())
        customer = seen["lookup_customer"].output if seen["lookup_customer"].ok else None

        if self.screens_text and screen(task, observations, REFUND.patterns):
            return _review("instruction-like text in the request or the records")
        if self.checks_account:
            if customer is None or customer["status"] != "active" or customer["flags"]["chargeback_open"]:
                return _review("account needs a person")
            if any(item["final_sale"] for item in order["items"]):
                return _review("final-sale item")
            if amount > LARGE_REFUND:
                return _review("large refund")
        if order["delivered_days_ago"] > self.window_days:
            return _review("outside the refund window")
        if round(amount * 100) + round(order["refunded_so_far"] * 100) > round(order["total"] * 100):
            return _review("more than the unrefunded balance")
        return AgentDecision("ISSUE_REFUND", "eligible", 0.9, evidence=[f"order:{order_id}"],
                             final_answer={"refund_id": task["refund_id"], "amount": amount})


class PlantedBugRefundPlanner(ObservingRefundPlanner):
    """The same planner with three planted mistakes, each one a different core mechanism must catch:
    - reads the refund window as 45 days           -> the verifier (it has its own 30)
    - ignores account status, flags, final sale and size -> the declared triggers
    - does not look for instructions in free text   -> the screen"""
    window_days = 45
    checks_account = False
    screens_text = False
