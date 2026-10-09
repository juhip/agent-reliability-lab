from __future__ import annotations
from typing import Dict, Any

PURCHASE_ORDERS = {
    "PO-1001": {"supplier": "Northwind Components", "expected_amount": 10000.0, "currency": "USD", "quantity": 100},
    "PO-1002": {"supplier": "Vector Metals", "expected_amount": 7500.0, "currency": "USD", "quantity": 50},
    "PO-1003": {"supplier": "Northwind Components", "expected_amount": 4200.0, "currency": "USD", "quantity": 40},
}

RECEIPTS = {
    "PO-1001": {"received_quantity": 100},
    "PO-1002": {"received_quantity": 45},
    "PO-1003": {"received_quantity": 40},
}

def lookup_purchase_order(po_id: str) -> Dict[str, Any]:
    if po_id not in PURCHASE_ORDERS:
        raise KeyError(f"unknown PO {po_id}")
    return PURCHASE_ORDERS[po_id]

def lookup_receipt(po_id: str) -> Dict[str, Any]:
    if po_id not in RECEIPTS:
        raise KeyError(f"missing receipt {po_id}")
    return RECEIPTS[po_id]

def calculate_variance(invoice_amount: float, expected_amount: float) -> Dict[str, float]:
    delta = round(invoice_amount - expected_amount, 2)
    pct = round(delta / expected_amount, 4) if expected_amount else 0.0
    return {"delta": delta, "pct": pct}
