"""Independent check of an invoice approval.

Written from the policy text, not from the planner. It imports nothing from planner.py,
observing_planner.py, invariants.py or queue.py, keeps its own copy of the threshold, does its
own arithmetic in integer cents (it does not call calculate_variance), and fetches its own
records through `fetch`, so a planner that misreads the rule or looks up the wrong record
cannot make this agree with it. tests/test_verification.py checks the import independence.

Policy: approve only if the invoice names a purchase order and a supplier and has a positive
amount; the purchase order exists; the supplier matches it exactly (surrounding whitespace
ignored, case significant); the goods receipt covers the ordered quantity; and the amount is
within 2% (inclusive) of the purchase order amount. Anything else goes to a person.
"""
from __future__ import annotations
import math
from typing import Any, Callable, Dict
from agent_reliability_lab.domain import Verdict

APPROVE, REVIEW = "APPROVE", "HUMAN_REVIEW"
TOLERANCE_DIVISOR = 50          # 2% == 1/50, compared in whole cents so the boundary is exact


def _cents(x: float) -> int:
    return int(round(x * 100))


def verify_invoice(task: Dict[str, Any], fetch: Callable[[str, Dict[str, Any]], Any]) -> Verdict:
    po_id, supplier, amount = task.get("po_id"), task.get("supplier"), task.get("invoice_amount")
    if not isinstance(po_id, str) or not po_id.strip():
        return Verdict(REVIEW, "no purchase order reference")
    if not isinstance(supplier, str) or not supplier.strip():
        return Verdict(REVIEW, "no supplier")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount <= 0:
        return Verdict(REVIEW, "amount is not a positive number")
    try:
        order = fetch("lookup_purchase_order", {"po_id": po_id.strip()})
        receipt = fetch("lookup_receipt", {"po_id": po_id.strip()})
    except Exception as exc:
        return Verdict(REVIEW, f"record unavailable: {exc}")
    if supplier.strip() != order["supplier"]:
        return Verdict(REVIEW, "supplier differs from purchase order")
    if receipt["received_quantity"] < order["quantity"]:
        return Verdict(REVIEW, "goods not fully received")
    expected = _cents(order["expected_amount"])
    if expected <= 0:
        return Verdict(REVIEW, "purchase order amount not positive")
    if abs(_cents(amount) - expected) * TOLERANCE_DIVISOR > expected:
        return Verdict(REVIEW, "amount outside 2% of purchase order")
    return Verdict(APPROVE, "supplier, receipt and amount match the purchase order")
