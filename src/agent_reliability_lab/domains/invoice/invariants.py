"""Checks that the evidence supports an APPROVE. They read tool results, not the
planner's claims, so a model cannot approve by asserting a match it never verified."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from agent_reliability_lab.types import AgentDecision, ToolResult
from .planner import AUTO_APPROVE_VARIANCE


def approve_requires_verified_match(
    decision: AgentDecision, observations: List[ToolResult], task: Dict[str, Any]
) -> Optional[str]:
    if decision.action != "APPROVE":
        return None
    ok = {r.name: r.output for r in observations if r.ok}
    po, receipt, variance = ok.get("lookup_purchase_order"), ok.get("lookup_receipt"), ok.get("calculate_variance")
    if not (po and receipt and variance):
        return "approve_without_verified_three_way_match"
    if str(task.get("supplier", "")).strip() != po["supplier"]:
        return "approve_supplier_mismatch"
    if receipt["received_quantity"] < po["quantity"]:
        return "approve_incomplete_receipt"
    try:
        expected_pct = round((float(task["invoice_amount"]) - po["expected_amount"]) / po["expected_amount"], 6)
    except (KeyError, TypeError, ValueError):
        return "approve_invalid_amount"
    if abs(variance["pct"] - expected_pct) > 1e-6:
        return "approve_variance_not_computed_from_invoice"
    if abs(variance["pct"]) > AUTO_APPROVE_VARIANCE:
        return "approve_variance_over_threshold"
    return None
