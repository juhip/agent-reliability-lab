"""Independent check of a refund. Written from the policy text below, not from the planner: it
imports nothing from planner.py, standin.py, queue.py or records.py, keeps its own constants,
works in integer cents, and fetches its own records through `fetch`.

Eligibility policy: the request names a refund, an order and a customer and has a positive
amount; the order exists and belongs to that customer; the return has been received; the order
was delivered no more than 30 days ago; and the amount, added to what was already refunded on
the order, does not exceed the order total. Escalation conditions (chargebacks, account status,
final-sale items, large amounts) are not checked here: they are declared as triggers.
"""
from __future__ import annotations
import math
from typing import Any, Callable, Dict
from agent_reliability_lab.domain import Verdict

REFUND, REVIEW = "ISSUE_REFUND", "MANUAL_REVIEW"
WINDOW_DAYS = 30


def _cents(x: float) -> int:
    return int(round(x * 100))


def verify_refund(task: Dict[str, Any], fetch: Callable[[str, Dict[str, Any]], Any]) -> Verdict:
    order_id, customer_id, amount = task.get("order_id"), task.get("customer_id"), task.get("amount")
    if not all(isinstance(v, str) and v.strip() for v in (task.get("refund_id"), order_id, customer_id)):
        return Verdict(REVIEW, "refund, order or customer reference missing")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount <= 0:
        return Verdict(REVIEW, "amount is not a positive number")
    try:
        order = fetch("lookup_order", {"order_id": order_id.strip()})
        ret = fetch("lookup_return", {"order_id": order_id.strip()})
    except Exception as exc:
        return Verdict(REVIEW, f"record unavailable: {exc}")
    if order["customer_id"] != customer_id.strip():
        return Verdict(REVIEW, "order belongs to someone else")
    if ret["received"] is not True:
        return Verdict(REVIEW, "return not received")
    if order["delivered_days_ago"] > WINDOW_DAYS:
        return Verdict(REVIEW, "outside the refund window")
    if _cents(amount) + _cents(order["refunded_so_far"]) > _cents(order["total"]):
        return Verdict(REVIEW, "more than the unrefunded balance")
    return Verdict(REFUND, "eligible")
