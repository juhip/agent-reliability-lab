"""A deterministic stand-in for a model that follows the invoice system prompt exactly.
It tests the loop; it says nothing about how well a real model orchestrates."""
from __future__ import annotations
from typing import Any, Dict, List
from agent_reliability_lab.agentic.standins import call, case_id_from_goal, tool_results, with_queue_parent
from agent_reliability_lab.agentic.types import ModelTurn
from agent_reliability_lab.screening import screen_value
from .domain import INVOICE


def _s(v: Any) -> str:
    return str(v).strip()


def _careful(messages: List[Dict[str, Any]]) -> ModelTurn:
    inv_id = case_id_from_goal(INVOICE, messages[0]["content"])
    r = tool_results(messages)
    if "approve_invoice" in r or "escalate_invoice" in r:
        return ModelTurn(text=f"{inv_id} decided.")

    def escalate(why: str) -> ModelTurn:
        return call("escalate_invoice", invoice_id=inv_id, reason=why)

    if "read_invoice" not in r:
        return call("read_invoice", invoice_id=inv_id)
    inv = r["read_invoice"].get("invoice", {})
    po_id, supplier, amount = _s(inv.get("po_id", "")), _s(inv.get("supplier", "")), inv.get("invoice_amount")
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return escalate("amount missing or not numeric")
    if not po_id or not supplier or amount <= 0 or amount != amount or amount in (float("inf"),):
        return escalate("missing or invalid fields")
    if screen_value(inv, "invoice", INVOICE.patterns):
        return escalate("free text contains instructions")
    if "lookup_purchase_order" not in r:
        return call("lookup_purchase_order", po_id=po_id)
    po = r["lookup_purchase_order"]
    if po.get("error"):
        return escalate("purchase order not found")
    if supplier != po["supplier"]:
        return escalate("supplier does not match PO")
    if "lookup_receipt" not in r:
        return call("lookup_receipt", po_id=po_id)
    rc = r["lookup_receipt"]
    if rc.get("error") or rc["received_quantity"] < po["quantity"]:
        return escalate("goods receipt incomplete")
    if "calculate_variance" not in r:
        return call("calculate_variance", invoice_amount=amount, expected_amount=po["expected_amount"])
    if abs(r["calculate_variance"]["pct"]) <= 0.02:
        return call("approve_invoice", invoice_id=inv_id, reason="three-way match within 2%")
    return escalate("variance over 2%")


careful_model = with_queue_parent(INVOICE, _careful)
