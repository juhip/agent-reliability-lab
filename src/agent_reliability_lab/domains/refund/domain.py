"""The refund domain, declared against core's contract using only declarations: triggers, extra
screen patterns, evidence bindings, a fail-safe outcome, limits, and an independent verifier.
It adds no invariants and no code to core."""
from __future__ import annotations
from agent_reliability_lab.agentic.types import READ, Tool
from agent_reliability_lab.domain import Domain
from .queue import QUEUE
from .records import lookup_customer, lookup_order, lookup_return
from .verifier import verify_refund


def _by(key: str):
    return {"type": "object", "properties": {key: {"type": "string"}}, "required": [key]}


TOOLS = [
    Tool("lookup_order", "Fetch an order: owner, total, items, days since delivery, amount already refunded.",
         _by("order_id"), lookup_order, READ),
    Tool("lookup_return", "Fetch the return for an order: received or not, condition, carrier note.",
         _by("order_id"), lookup_return, READ),
    Tool("lookup_customer", "Fetch a customer account: status and flags.", _by("customer_id"), lookup_customer, READ),
]

# Conditions under which a person decides, whatever the eligibility rules say.
TRIGGERS = [
    {"name": "chargeback_open", "source": "lookup_customer", "path": "flags.chargeback_open", "op": "eq", "value": True},
    # required: no refund goes out unless the customer's account status was actually looked up
    {"name": "account_not_active", "source": "lookup_customer", "path": "status", "op": "ne", "value": "active",
     "required": True},
    {"name": "final_sale_item", "source": "lookup_order", "path": "items.final_sale", "op": "eq", "value": True},
    {"name": "large_refund", "source": "task", "path": "amount", "op": "gt", "value": 1000},
]

INJECTION_PATTERNS = [r"\b(issue|process|send) (the |this |my )?refund\b.{0,30}\b(now|immediately|without)\b"]

EVIDENCE_BINDINGS = {
    "lookup_order": {"order_id": "order_id"},
    "lookup_return": {"order_id": "order_id"},
    "lookup_customer": {"customer_id": "customer_id"},
    "read_refund": {"refund_id": "refund_id"},
}

REFUND = Domain(
    name="refund",
    actions=["ISSUE_REFUND", "MANUAL_REVIEW"],
    gated_actions=["ISSUE_REFUND"],
    fail_safe="MANUAL_REVIEW",
    verifier=verify_refund,
    tools=TOOLS,
    case_id_field="refund_id",
    triggers=TRIGGERS,
    injection_patterns=INJECTION_PATTERNS,
    evidence_bindings=EVIDENCE_BINDINGS,
    limits={"max_steps": 8, "max_result_chars": 8_000},
    queue=QUEUE,
)
