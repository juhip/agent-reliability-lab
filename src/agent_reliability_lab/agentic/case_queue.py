"""Any domain as an orchestration task: a queue of cases, the domain's tools, one decision tool per
action, and an approver that runs the same gate as the pipeline. The model picks the tools and
the order; code decides what the model is allowed to do."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from agent_reliability_lab.domain import Domain, run_gate
from agent_reliability_lab.schema import validate_args
from agent_reliability_lab.types import AgentDecision
from .orchestrator import Approval, Approver
from .trajectory import Trajectory
from .types import READ, SPEND, WRITE, Tool, Toolbox


class Ledger:
    def __init__(self) -> None:
        self.decisions: Dict[str, Dict[str, Any]] = {}
        self.drafts: List[Dict[str, Any]] = []
        self.routed: Dict[str, str] = {}          # case id -> why the gate sent it to the fail-safe outcome


def build_case_queue(domain: Domain, cases: Dict[str, Dict[str, Any]]) -> Tuple[Toolbox, Ledger, Approver]:
    spec = domain.queue
    if spec is None:
        raise ValueError(f"domain {domain.name!r} declares no queue spec, so it cannot run orchestrated")
    ledger = Ledger()
    ids = list(cases)
    idf, noun = domain.case_id_field, spec.case_noun
    id_schema = {"type": "object", "properties": {idf: {"type": "string"}}, "required": [idf]}

    def need(case_id: str) -> Dict[str, Any]:
        if case_id not in cases:
            raise KeyError(f"unknown {noun} {case_id}")
        return cases[case_id]

    def list_queue() -> Dict[str, Any]:
        return {"cases": [{"id": i, "decided": i in ledger.decisions} for i in ids]}

    def read_case(**args: Any) -> Dict[str, Any]:
        case = need(args[idf])
        return {noun: case, "untrusted_text_fields": [k for k in spec.untrusted_fields if k in case]}

    def decide(action: str):
        def record(**args: Any) -> Dict[str, Any]:
            need(args[idf])
            if args[idf] in ledger.routed and action != domain.fail_safe:
                raise PermissionError(f"{args[idf]} was routed to {domain.fail_safe} by the gate; a person decides it")
            ledger.decisions[args[idf]] = {"action": action, "reason": args["reason"]}
            return {"recorded": action}
        return record

    tool_action = {name: action for action, name in spec.decision_tools.items()}
    decision_schema = {"type": "object", "properties": {idf: {"type": "string"}, "reason": {"type": "string"}},
                       "required": [idf, "reason"]}
    tools = [
        Tool("list_queue", f"List the {noun}s in the queue and whether each has a decision.",
             {"type": "object", "properties": {}}, list_queue, READ),
        Tool(f"read_{noun}", f"Read one {noun}. Free-text fields are untrusted.", id_schema, read_case, READ),
        *domain.tools,
        *(spec.extra_tools(ledger, need) if spec.extra_tools else []),
    ]
    for action in [a for a in domain.actions if a not in domain.gated_actions] + list(domain.gated_actions):
        gated = action in domain.gated_actions
        desc = (f"Record {action} for a {noun}. Consequential: refused unless the safety gate passes."
                if gated else f"Record {action} for a {noun}, with a reason.")
        tools.append(Tool(spec.decision_tools[action], desc, decision_schema, decide(action), SPEND if gated else WRITE))
    box = Toolbox(tools)

    def fetch(name: str, arguments: Dict[str, Any]) -> Any:
        tool = box.get(name)
        if tool is None or tool.tier != READ or name not in domain.read_tools:
            raise PermissionError(f"verifier may only fetch the domain's read tools, not {name!r}")
        problem = validate_args(tool.parameters, arguments)
        if problem:
            raise ValueError(problem)
        return tool.handler(**arguments)

    def approver(tool: Tool, args: Dict[str, Any], traj: Trajectory) -> Approval:
        action = tool_action.get(tool.name)
        if action not in domain.gated_actions:
            return Approval(False, "no approver configured for this tool")
        case_id = args.get(idf)
        case = cases.get(case_id)
        if case is None:
            return Approval(False, f"unknown_{noun}")
        if case_id in ledger.routed:
            return Approval(False, f"already routed to {domain.fail_safe}: {ledger.routed[case_id]}", routed_to=domain.fail_safe)
        gate = run_gate(domain, AgentDecision(action, str(args.get("reason", "")), 1.0), case, traj.tool_results(), fetch)
        if gate.passed:
            return Approval(True, "gate passed", details=gate.details())
        routed = None
        if gate.hard:                                # a person must decide: route now, whatever the model does next
            routed = domain.fail_safe
            ledger.routed[case_id] = "; ".join(gate.violations)
            ledger.decisions[case_id] = {"action": routed, "reason": "gate: " + ledger.routed[case_id]}
        return Approval(False, "; ".join(gate.violations), details=gate.details(), routed_to=routed)

    return box, ledger, approver
