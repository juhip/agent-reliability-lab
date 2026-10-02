from __future__ import annotations
import math
from typing import Any, Dict
from agent_reliability_lab.types import AgentDecision, ToolCall
from .tools import PURCHASE_ORDERS, RECEIPTS

AUTO_APPROVE_VARIANCE = 0.02

class InvoicePlanner:
    """Deterministic reference planner; replace with LMStudioDecisionModel for Liquid."""
    def __call__(self, task: Dict[str, Any], tools: Dict[str, str]) -> AgentDecision:
        po_id = str(task.get("po_id", "")).strip()
        supplier = str(task.get("supplier", "")).strip()
        amount = task.get("invoice_amount")
        if not po_id or amount is None or not supplier:
            return AgentDecision("HUMAN_REVIEW", "Missing required invoice fields", 1.0)
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return AgentDecision("HUMAN_REVIEW", "Invoice amount is not numeric", 1.0)
        if not math.isfinite(amount) or amount <= 0:
            return AgentDecision("HUMAN_REVIEW", "Invoice amount must be a positive finite value", 1.0)
        if po_id not in PURCHASE_ORDERS:
            return AgentDecision("HUMAN_REVIEW", "Purchase order not found", 1.0, evidence=[f"po_id={po_id}"])
        po = PURCHASE_ORDERS[po_id]
        receipt = RECEIPTS.get(po_id)
        evidence = [f"PO:{po_id}", f"supplier:{po['supplier']}", f"expected:{po['expected_amount']}"]
        if supplier != po["supplier"]:
            return AgentDecision("HUMAN_REVIEW", "Supplier does not match purchase order", 0.99, evidence=evidence)
        if receipt is None or receipt["received_quantity"] < po["quantity"]:
            return AgentDecision("HUMAN_REVIEW", "Goods receipt incomplete", 0.99, evidence=evidence)
        variance_pct = (amount - po["expected_amount"]) / po["expected_amount"]
        calls = [
            ToolCall("lookup_purchase_order", {"po_id": po_id}),
            ToolCall("lookup_receipt", {"po_id": po_id}),
            ToolCall("calculate_variance", {"invoice_amount": amount, "expected_amount": po["expected_amount"]}),
        ]
        if abs(variance_pct) <= AUTO_APPROVE_VARIANCE:
            return AgentDecision(
                "APPROVE",
                "Three-way match passes and amount variance is within policy threshold",
                0.97,
                tool_calls=calls,
                evidence=evidence,
                final_answer={"invoice_id": task.get("invoice_id"), "po_id": po_id, "variance_pct": round(variance_pct, 4)},
            )
        return AgentDecision(
            "HUMAN_REVIEW",
            "Invoice amount exceeds auto-approval variance threshold",
            0.98,
            tool_calls=calls,
            evidence=evidence,
        )
