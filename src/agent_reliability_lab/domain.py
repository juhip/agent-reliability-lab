"""The domain contract, and the gate every gated action must pass.

A domain supplies declarations and domain functions; core owns every safety mechanism.

Must provide
    actions         every outcome the agent may choose
    gated_actions   the subset with consequences (approve, pay, refund, ...). Only these are gated.
    fail_safe       the outcome core routes to when anything is blocked or breaks. Must not be gated.
    verifier        verifier(task, fetch) -> outcome | Verdict. Recomputes the expected outcome from
                    raw tool data, fetched through `fetch(tool_name, arguments)` (READ tools only).
                    It must not share code, constants or data paths with the planner: a rule the
                    planner misreads must not be misread the same way here.
    tools           the domain's tools, each with a JSON-schema for its arguments and a tier

May provide
    triggers            declarative human-review conditions (see triggers.py)
    injection_patterns  extra regexes for the untrusted-text screen (see screening.py)
    invariants          checks that the evidence the planner gathered supports the action
    evidence_bindings   {tool: {argument: task_path}}: a tool result counts as evidence for a case
                        only if its call arguments match the case. Unbound tools are never evidence.
    executor            called once a gated action has passed the gate. The default, `record_only`,
                        performs nothing and says so.
    limits              step and size limits (see DEFAULT_LIMITS)
    queue               presentation for orchestrated mode (system prompt, tool names, goals)

Core guarantees, for every domain, in both the pipeline and the orchestrated runtime
    A gated action runs only if ALL of these hold; otherwise the case goes to `fail_safe`:
      1. no screen hit in the task or in any tool result the deciding agent saw
      2. no declared trigger fired over the task and the in-scope tool results
      3. every invariant passes on the in-scope tool results
      4. the verifier ran without error and returned the same outcome
    The trace records which screen paths, triggers, invariants and verifier verdict were involved.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from .agentic.types import READ, Tool
from .screening import ScreenHit, compile_patterns, screen
from .triggers import TriggerHit, compile_triggers, evaluate, resolve
from .types import AgentDecision, ToolResult

Fetch = Callable[[str, Dict[str, Any]], Any]
Invariant = Callable[[AgentDecision, List[ToolResult], Dict[str, Any]], Optional[str]]

DEFAULT_LIMITS: Dict[str, int] = {
    "max_steps": 6,                 # pipeline: planner turns per case
    "orchestrator_max_steps": 30,   # orchestrated: model turns per agent
    "child_max_steps": 15,          # orchestrated: model turns per sub-agent
    "max_result_chars": 20_000,     # orchestrated: longest tool result shown to the model (the gate always sees all)
}


@dataclass
class Verdict:
    outcome: str
    reason: str = ""


Verifier = Callable[[Dict[str, Any], Fetch], Union[str, Verdict]]


def record_only(action: str, task: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
    """Default executor: performs nothing. The decision is returned to the caller, not carried out."""
    return {"executed": False, "note": "no executor configured; decision recorded only"}


@dataclass
class QueueSpec:
    """How the domain looks to a model in orchestrated mode. Presentation only: no safety logic here."""
    system: str
    case_noun: str                                  # "invoice" -> read_invoice(invoice_id)
    decision_tools: Dict[str, str]                  # action -> tool name
    goal_template: str                              # must contain "{id}"
    queue_goal: str
    untrusted_fields: Tuple[str, ...] = ()          # hint shown to the model; the screen checks every field anyway
    extra_tools: Optional[Callable[[Any, Callable[[str], Dict[str, Any]]], List[Tool]]] = None

    def goal_for(self, case_id: str) -> str:
        return self.goal_template.format(id=case_id)


@dataclass
class Domain:
    name: str
    actions: Sequence[str]
    gated_actions: Sequence[str]
    fail_safe: str
    verifier: Verifier
    tools: Sequence[Tool]
    case_id_field: str = "id"
    triggers: Sequence[Dict[str, Any]] = ()
    injection_patterns: Sequence[str] = ()
    invariants: Sequence[Invariant] = ()
    evidence_bindings: Dict[str, Dict[str, str]] = field(default_factory=dict)
    executor: Callable[[str, Dict[str, Any], Dict[str, Any]], Any] = record_only
    limits: Dict[str, int] = field(default_factory=dict)
    queue: Optional[QueueSpec] = None

    def __post_init__(self) -> None:
        self.actions, self.gated_actions = list(self.actions), list(self.gated_actions)
        if not self.gated_actions:
            raise ValueError(f"{self.name}: declare at least one gated action")
        if not set(self.gated_actions) <= set(self.actions):
            raise ValueError(f"{self.name}: gated actions must be among the actions")
        if self.fail_safe not in self.actions or self.fail_safe in self.gated_actions:
            raise ValueError(f"{self.name}: fail_safe must be an action and must not be gated")
        if not callable(self.verifier):
            raise ValueError(f"{self.name}: a verifier is required")
        unknown_limits = set(self.limits) - set(DEFAULT_LIMITS)
        if unknown_limits:
            raise ValueError(f"{self.name}: unknown limits {sorted(unknown_limits)}")
        self.limits = {**DEFAULT_LIMITS, **self.limits}
        tool_names = {t.name for t in self.tools}
        known_sources = tool_names | ({f"read_{self.queue.case_noun}"} if self.queue else set())
        self.compiled_triggers = compile_triggers(self.triggers)
        for t in self.compiled_triggers:
            if t.source != "task" and t.source not in known_sources:
                raise ValueError(f"{self.name}: trigger {t.name!r} reads unknown source {t.source!r}")
        for tool in self.evidence_bindings:
            if tool not in known_sources:
                raise ValueError(f"{self.name}: evidence binding for unknown tool {tool!r}")
        self.patterns = compile_patterns(self.injection_patterns)
        if self.queue:
            missing = set(self.actions) - set(self.queue.decision_tools)
            if missing or "{id}" not in self.queue.goal_template:
                raise ValueError(f"{self.name}: queue needs a decision tool per action and an {{id}} goal template")

    @property
    def read_tools(self) -> List[str]:
        return [t.name for t in self.tools if t.tier == READ]


# ---- evidence scoping ---------------------------------------------------------------------------------
def _same(a: Any, b: Any) -> bool:
    """Argument-to-task match: numbers compare as numbers, everything else as trimmed strings."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    try:
        if isinstance(a, (int, float)) or isinstance(b, (int, float)):
            return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-9)
        return str(a).strip() == str(b).strip()
    except (TypeError, ValueError):
        return False


def scope_evidence(domain: Domain, task: Dict[str, Any], results: Sequence[ToolResult]) -> List[ToolResult]:
    """Only the successful results whose call arguments match this case, per the domain's bindings."""
    keep: List[ToolResult] = []
    for r in results:
        binding = domain.evidence_bindings.get(r.name)
        if not r.ok or binding is None:
            continue
        args = r.arguments or {}
        ok = True
        for arg, task_path in binding.items():
            want = resolve(task, task_path.split("."))
            if arg not in args or not want or not _same(args[arg], want[0]):
                ok = False
                break
        if ok:
            keep.append(r)
    return keep


# ---- the gate -------------------------------------------------------------------------------------------
@dataclass
class GateResult:
    passed: bool
    hard: bool                      # True: route to fail_safe. False: only evidence is missing; the agent may gather it.
    violations: List[str]
    screen_hits: List[ScreenHit]
    triggers_fired: List[TriggerHit]
    triggers_unresolved: List[TriggerHit]
    verifier: Dict[str, Any]

    def details(self) -> Dict[str, Any]:
        return {
            "passed": self.passed, "hard": self.hard, "violations": self.violations,
            "screen_hits": [h.__dict__ for h in self.screen_hits],
            "triggers_fired": [h.__dict__ for h in self.triggers_fired],
            "triggers_unresolved_not_required": [h.__dict__ for h in self.triggers_unresolved],
            "verifier": self.verifier,
        }


def run_gate(domain: Domain, decision: AgentDecision, task: Dict[str, Any],
             observed: Sequence[ToolResult], fetch: Fetch) -> GateResult:
    """Check a proposed gated action. `observed` is every tool result the deciding agent saw."""
    evidence = scope_evidence(domain, task, observed)
    violations: List[str] = []

    hits = screen(task, observed, domain.patterns)
    if hits:
        violations.append(f"{decision.action.lower()}_with_flagged_input")

    fired, soft = evaluate(domain.compiled_triggers, task, evidence)
    violations += [f"trigger:{h.name}" for h in fired]

    invariant_violations: List[str] = []
    for invariant in domain.invariants:
        try:
            v = invariant(decision, list(evidence), task)
        except Exception as exc:
            v = f"invariant_error:{type(exc).__name__}"
        if v:
            invariant_violations.append(v)
    violations += invariant_violations

    fetches: List[Dict[str, Any]] = []

    def logged_fetch(name: str, arguments: Dict[str, Any]) -> Any:
        try:
            out = fetch(name, arguments)
        except Exception as exc:
            fetches.append({"tool": name, "arguments": arguments, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            raise
        fetches.append({"tool": name, "arguments": arguments, "ok": True})
        return out

    verifier_failed = True
    try:
        verdict = domain.verifier(task, logged_fetch)
        verdict = verdict if isinstance(verdict, Verdict) else Verdict(str(verdict))
        verifier_info: Dict[str, Any] = {"outcome": verdict.outcome, "reason": verdict.reason, "fetches": fetches}
        if verdict.outcome == decision.action:
            verifier_failed = False
        else:
            violations.append(f"verifier_disagrees:{verdict.outcome}")
    except Exception as exc:
        verifier_info = {"error": f"{type(exc).__name__}: {exc}", "fetches": fetches}
        violations.append(f"verifier_error:{type(exc).__name__}")

    hard = bool(hits or fired or verifier_failed)
    return GateResult(not violations, hard, violations, hits, fired, soft, verifier_info)
