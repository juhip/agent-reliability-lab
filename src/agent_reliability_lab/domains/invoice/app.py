from __future__ import annotations
from agent_reliability_lab.runtime import AgentRuntime
from agent_reliability_lab.tool_registry import ToolRegistry
from agent_reliability_lab.policies import PolicyEngine
from .planner import InvoicePlanner
from .tools import lookup_purchase_order, lookup_receipt, calculate_variance

def build_invoice_runtime() -> AgentRuntime:
    tools = ToolRegistry()
    tools.register("lookup_purchase_order", "Fetch purchase order by po_id", lookup_purchase_order)
    tools.register("lookup_receipt", "Fetch goods receipt by po_id", lookup_receipt)
    tools.register("calculate_variance", "Deterministically compute amount delta and percent", calculate_variance)
    policy = PolicyEngine(allowed_actions={"APPROVE", "HUMAN_REVIEW"}, approval_actions={"HUMAN_REVIEW"})
    return AgentRuntime(InvoicePlanner(), tools, policy)
