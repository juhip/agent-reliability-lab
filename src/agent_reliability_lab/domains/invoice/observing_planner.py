from __future__ import annotations
import math
from typing import Any, Dict, List
from agent_reliability_lab.types import AgentDecision, ToolCall, ToolResult
from .planner import AUTO_APPROVE_VARIANCE


def _review(reason: str, evidence: List[str] | None = None) -> AgentDecision:
    return AgentDecision("HUMAN_REVIEW", reason, 0.99, evidence=evidence or [])


def _call(name: str, arguments: Dict[str, Any], why: str) -> AgentDecision:
    return AgentDecision("HUMAN_REVIEW", why, 0.5, tool_calls=[ToolCall(name, arguments)], is_final=False)


class ObservingInvoicePlanner:
    """Deterministic reference for the loop: decides from tool results only, never from
    module data. This is the shape a model-backed planner has to follow."""

    def __call__(self, task: Dict[str, Any], tools: Dict[str, Any], observations: List[ToolResult]) -> AgentDecision:
        po_id = str(task.get("po_id", "")).strip()
        supplier = str(task.get("supplier", "")).strip()
        amount = task.get("invoice_amount")
        if not po_id or amount is None or not supplier:
            return _review("Missing required invoice fields")
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return _review("Invoice amount is not numeric")
        if not math.isfinite(amount) or amount <= 0:
            return _review("Invoice amount must be a positive finite value")

        seen = {r.name: r for r in observations}

        po_res = seen.get("lookup_purchase_order")
        if po_res is None:
            return _call("lookup_purchase_order", {"po_id": po_id}, "Need the purchase order")
        if not po_res.ok:
            return _review(f"Purchase order lookup failed: {po_res.error}", [f"po_id={po_id}"])
        po = po_res.output
        evidence = [f"PO:{po_id}", f"supplier:{po['supplier']}", f"expected:{po['expected_amount']}"]
        if supplier != po["supplier"]:
            return _review("Supplier does not match purchase order", evidence)

        rc_res = seen.get("lookup_receipt")
        if rc_res is None:
            return _call("lookup_receipt", {"po_id": po_id}, "Need the goods receipt")
        if not rc_res.ok:
            return _review(f"Goods receipt lookup failed: {rc_res.error}", evidence)
        if rc_res.output["received_quantity"] < po["quantity"]:
            return _review("Goods receipt incomplete", evidence)

        var_res = seen.get("calculate_variance")
        if var_res is None:
            return _call(
                "calculate_variance",
                {"invoice_amount": amount, "expected_amount": po["expected_amount"]},
                "Need the variance",
            )
        if not var_res.ok:
            return _review(f"Variance calculation failed: {var_res.error}", evidence)

        pct = var_res.output["pct"]
        if abs(pct) <= AUTO_APPROVE_VARIANCE:
            return AgentDecision(
                "APPROVE",
                "Three-way match passes and amount variance is within policy threshold",
                0.97,
                evidence=evidence,
                final_answer={"invoice_id": task.get("invoice_id"), "po_id": po_id, "variance_pct": pct},
            )
        return _review("Invoice amount exceeds auto-approval variance threshold", evidence)
