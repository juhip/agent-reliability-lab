"""The invoice domain, declared against core's contract (see agent_reliability_lab/domain.py)."""
from __future__ import annotations
from agent_reliability_lab.agentic.types import READ, Tool
from agent_reliability_lab.domain import Domain
from .invariants import approve_requires_verified_match
from .queue import QUEUE
from .tools import calculate_variance, lookup_purchase_order, lookup_receipt
from .verifier import verify_invoice

_PO = {"type": "object", "properties": {"po_id": {"type": "string"}}, "required": ["po_id"]}

TOOLS = [
    Tool("lookup_purchase_order", "Fetch a purchase order by po_id.", _PO, lookup_purchase_order, READ),
    Tool("lookup_receipt", "Fetch the goods receipt for a po_id.", _PO, lookup_receipt, READ),
    Tool("calculate_variance", "Deterministically compute the amount delta and percent between an invoice and its PO.",
         {"type": "object", "properties": {"invoice_amount": {"type": "number"}, "expected_amount": {"type": "number"}},
          "required": ["invoice_amount", "expected_amount"]}, calculate_variance, READ),
]

# A person signs off on anything this large, whatever the match says.
TRIGGERS = [
    {"name": "large_invoice", "source": "task", "path": "invoice_amount", "op": "gt", "value": 50_000},
]

# Domain wording on top of core's neutral patterns.
INJECTION_PATTERNS = [r"approve (this|the) (invoice|payment) (immediately|without)"]

# A lookup counts as evidence for an invoice only if it was made with that invoice's values.
EVIDENCE_BINDINGS = {
    "lookup_purchase_order": {"po_id": "po_id"},
    "lookup_receipt": {"po_id": "po_id"},
    "calculate_variance": {"invoice_amount": "invoice_amount"},
    "read_invoice": {"invoice_id": "invoice_id"},
}

INVOICE = Domain(
    name="invoice",
    actions=["APPROVE", "HUMAN_REVIEW"],
    gated_actions=["APPROVE"],
    fail_safe="HUMAN_REVIEW",
    verifier=verify_invoice,
    tools=TOOLS,
    case_id_field="invoice_id",
    triggers=TRIGGERS,
    injection_patterns=INJECTION_PATTERNS,
    invariants=[approve_requires_verified_match],
    evidence_bindings=EVIDENCE_BINDINGS,
    limits={"max_steps": 6, "max_result_chars": 20_000},
    queue=QUEUE,
)
