from __future__ import annotations
from agent_reliability_lab.runtime import AgentRuntime, Planner, build_runtime
from .domain import INVOICE
from .planner import InvoicePlanner


def build_invoice_runtime(planner: Planner | None = None) -> AgentRuntime:
    return build_runtime(INVOICE, planner or InvoicePlanner())
