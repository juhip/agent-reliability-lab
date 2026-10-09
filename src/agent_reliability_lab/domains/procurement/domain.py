"""The procurement domain, declared against core's contract (see agent_reliability_lab/domain.py).

Tools are bound to one run's scenario allowlist and rulebook, so the Domain is built per run:
`build_domain(tools)`. Everything else is declarations.
"""
from __future__ import annotations
from agent_reliability_lab.domain import Domain
from .invariants import PlanInvariants
from .patterns import INJECTION_PATTERNS
from .queue import QUEUE
from .tools import ProcurementTools
from .verifier import verify_release

ACTIONS = ["APPROVE", "HUMAN_REVIEW"]

# Conditions under which a person decides, whatever the release settings say. Read from tool results,
# never from the agent's rulebook, so a rulebook that drops one of them does not drop it here.
TRIGGERS = [
    {"name": "hazardous_material", "source": "plan_component", "path": "hazardous", "op": "eq", "value": True},
    {"name": "air_freight", "source": "plan_component", "path": "lines.mode", "op": "eq", "value": "air"},
    {"name": "no_compliant_supplier", "source": "plan_component", "path": "no_eligible_supplier", "op": "eq", "value": True},
    {"name": "blocked_by_hard_rule", "source": "check_hard_rules", "path": "blocked.line_id", "op": "exists"},
    # fail closed: nothing is released unless the hard-rule gate ran on this scenario
    {"name": "hard_rule_gate_not_run", "source": "check_hard_rules", "path": "lines", "op": "missing"},
    # the agent's own release check said a person must approve this line
    {"name": "line_needs_approver", "source": "release_check", "path": "needs_approval", "op": "eq", "value": True},
]

# A tool result counts as evidence for a case only if it was called with that case's scenario.
EVIDENCE_BINDINGS = {name: {"scenario": "scenario"} for name in (
    "load_scenario", "net_requirements", "plan_component", "check_hard_rules", "release_check", "draft_alerts",
    "read_records", "read_request")}

LIMITS = {"max_steps": 8, "orchestrator_max_steps": 30, "max_result_chars": 200_000}


def build_domain(tools: ProcurementTools) -> Domain:
    return Domain(
        name="procurement",
        actions=ACTIONS,
        gated_actions=["APPROVE"],
        fail_safe="HUMAN_REVIEW",
        verifier=verify_release,
        tools=tools.tools(),
        case_id_field="scenario",
        triggers=TRIGGERS,
        injection_patterns=INJECTION_PATTERNS,
        invariants=PlanInvariants(tools).all(),
        evidence_bindings=EVIDENCE_BINDINGS,
        limits=LIMITS,
        queue=QUEUE,
    )
