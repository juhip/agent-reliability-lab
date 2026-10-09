"""One planning session over one scenario, computed stage by stage with the agent's own modules.

The agent's pipeline (procurement/pipeline.py) runs interpret -> decide -> verify -> escalate in one call.
Here the same stages are exposed one at a time so a harness planner has to request each of them as a
tool. The glue below mirrors pipeline.run() without the optional model steps; tests check that the
result is identical to pipeline.run(model=None) line for line.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from . import agent_loader

RELEASE_MODES = ("approve_by_exception", "approve_every_order", "policy_thresholds_only")


def rulebook(raw: Optional[dict] = None, release_mode: Optional[str] = None):
    """The agent's rulebook (or a variant of it), optionally with a different release mode. In memory only."""
    import copy
    import json
    a = agent_loader.load()
    data = copy.deepcopy(raw) if raw is not None else json.loads(a.rules_path.read_text(encoding="utf-8"))
    if release_mode:
        if release_mode not in RELEASE_MODES:
            raise ValueError(f"unknown release mode {release_mode}")
        data.setdefault("settings", {})["order_release"] = release_mode
    return a.rules.Rulebook(data)


def line_id(po) -> str:
    return f"{po.component_id}:{po.supplier_id}:{po.mode}"


def line_dict(po) -> Dict[str, Any]:
    return {"line_id": line_id(po), "component_id": po.component_id, "supplier_id": po.supplier_id,
            "quantity": po.quantity, "unit_price": po.unit_price, "total": po.total,
            "order_date": po.order_date.isoformat(), "expected_delivery_date": po.expected_delivery_date.isoformat(),
            "mode": po.mode, "serves": [list(s) for s in po.serves]}


def planned_po(d: Dict[str, Any]):
    """Rebuild the agent's PlannedPO from a plan line (used by invariants and the grader)."""
    a = agent_loader.load()
    return a.decider.PlannedPO(str(d["component_id"]), str(d["supplier_id"]), int(d["quantity"]),
                               float(d["unit_price"]), date.fromisoformat(d["order_date"]),
                               date.fromisoformat(d["expected_delivery_date"]), str(d["mode"]),
                               [tuple(s) for s in d.get("serves", [])], d.get("rationale", ""))


@dataclass
class Session:
    path: str
    rules: Any                                    # the configured Rulebook (before date adjustment)
    state: Any = None
    eff_rules: Any = None
    policy_note: Optional[str] = None
    part_labels: Any = None
    tags: Dict[str, Any] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict)
    decider: Any = None
    decisions: Dict[str, Any] = field(default_factory=dict)
    planned: Dict[str, List[Any]] = field(default_factory=dict)   # component -> PlannedPO list
    pos: Optional[List[Any]] = None                               # after the hard-rule gate, sorted
    violations: List[Any] = field(default_factory=list)
    statuses: List[Any] = field(default_factory=list)
    flags: Dict[int, List[str]] = field(default_factory=dict)
    releases: Dict[int, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        a = agent_loader.load()
        self.state = a.db.load_state(self.path)                    # opens the file read-only
        today = self.state.current_date
        self.eff_rules, self.policy_note = self.rules.base_policy_in_force(today)
        self.part_labels = a.part_labels.label_parts(None, self.state, self.eff_rules, None)   # no model: empty
        self.eff_rules = self.eff_rules.with_extra_scope(self.part_labels.scope)
        self.tags = a.classifier.classify_all(self.state.components, self.eff_rules, today, self.part_labels.critical)
        self.requirements = a.planner.net_requirements(self.state)
        self.decider = a.decider.Decider(self.state, self.eff_rules, self.tags)

    # ---- stage 1: what does the world look like ---------------------------------------------
    def overview(self) -> Dict[str, Any]:
        s = self.state
        text = [{"where": f"supplier {k} notes", "text": v.notes} for k, v in sorted(s.suppliers.items()) if v.notes.strip()]
        text += [{"where": f"catalog {e.supplier_id}/{e.component_id} notes", "text": e.notes} for e in s.catalog if e.notes.strip()]
        text += [{"where": f"component {k} description", "text": v.description}
                 for k, v in sorted(s.components.items()) if v.description.strip()]
        return {"current_date": s.current_date.isoformat(), "components": sorted(s.components),
                "production_orders": len(s.schedule), "data_issues": list(s.issues),
                "policy_note": self.policy_note, "untrusted_text": text}

    # ---- stage 2: what is short, by when ----------------------------------------------------
    def shortfalls(self) -> Dict[str, Any]:
        return {c: {"to_buy": r.to_buy, "tranches": [{"need_by": t.need_by.isoformat(), "qty": t.qty,
                                                       "orders": [list(o) for o in t.orders]} for t in r.tranches]}
                for c, r in sorted(self.requirements.items()) if r.tranches}

    # ---- stage 3: choose suppliers for one part ---------------------------------------------
    def plan_component(self, component_id: str) -> Dict[str, Any]:
        req = self.requirements.get(component_id)
        if req is None or not req.tranches:
            raise KeyError(f"nothing to buy for component {component_id}")
        if self.pos is not None:
            raise RuntimeError("plan is already verified; start a new session to re-plan")
        dec = self.decision(component_id)
        self.planned[component_id] = self.decider.purchase_orders(dec)
        return {"component_id": component_id, "hazardous": self.state.components[component_id].is_hazardous,
                "lines": [line_dict(p) for p in self.planned[component_id]],
                "no_eligible_supplier": dec.no_eligible_supplier, "conflicts": list(dec.conflicts),
                "eligible_suppliers": sorted({o.supplier.supplier_id for o in dec.eligible}),
                "excluded": {o.supplier.supplier_id: list(o.excluded_because) for o in dec.options if not o.eligible}}

    def decision(self, component_id: str):
        if component_id not in self.decisions:
            self.decisions[component_id] = self.decider.decide(self.requirements[component_id])
        return self.decisions[component_id]

    # ---- stage 4: independent hard-rule gate and calendar replay -----------------------------
    def verify(self) -> Dict[str, Any]:
        a = agent_loader.load()
        missing = [c for c, r in self.requirements.items() if r.tranches and c not in self.planned]
        if missing:
            raise RuntimeError(f"components not planned yet: {sorted(missing)}")
        if self.pos is None:
            # same order as pipeline.run: decisions are made in sorted component order
            pos = [p for c in sorted(self.planned) for p in self.planned[c]]
            self.violations = a.validator.hard_violations(self.state, self.eff_rules, self.tags, pos)
            blocked = {id(p) for p, _ in self.violations}
            pos = [p for p in pos if id(p) not in blocked]
            pos.sort(key=lambda p: (p.expected_delivery_date, p.component_id, p.supplier_id))
            self.pos = pos
            buffers = {c: self.decider.buffer_days(c) for c in self.state.components}
            self.statuses = a.validator.replay(self.state, pos, buffers)
        return {"lines": [line_id(p) for p in self.pos],
                "planned_lines": [{k: line_dict(p)[k] for k in ("line_id", "component_id", "supplier_id", "quantity", "mode")}
                           for p in self.pos],
                "blocked": [{"line_id": line_id(p), "reason": why} for p, why in self.violations],
                "late_orders": [{"order_id": st.order.order_id, "late_days": st.late_days} for st in self.statuses
                                if not st.on_time]}

    # ---- stage 5: release or hold one order --------------------------------------------------
    def release_check(self, lid: str) -> Dict[str, Any]:
        a = agent_loader.load()
        if self.pos is None:
            raise RuntimeError("run check_hard_rules before release_check")
        idx = next((i for i, p in enumerate(self.pos) if line_id(p) == lid), None)
        if idx is None:
            raise KeyError(f"unknown or blocked line {lid}")
        po, dec = self.pos[idx], self.decisions[self.pos[idx].component_id]
        tags = self.tags[po.component_id]
        if idx not in self.releases:
            served = {oid for oid, _ in po.serves}
            urgent = any((o.materials_needed_by - po.expected_delivery_date).days <= 7
                         for o in self.state.schedule if o.order_id in served)
            self.flags[idx] = a.alerts.po_flags(po, dec, self.state, self.eff_rules, tags, urgent)
            same = [p for p in self.pos if p.component_id == po.component_id]
            body = a.explain.rationale(po, dec, same, self.state, self.eff_rules, tags, self.flags[idx])
            rel = a.release.decide_release(po, dec, self.state, self.eff_rules, tags, list(self.part_labels.holds.get(po.component_id, [])))
            self.releases[idx] = rel
            po.rationale = ((f"DRAFT, awaiting approval by {rel.approver} before release to the supplier "
                             f"(because: {'; '.join(rel.reasons)}). ") if rel.needs_approval else
                            "RELEASED automatically: routine order within the approval guardrails. ") + body
        rel = self.releases[idx]
        return {"line_id": lid, "needs_approval": rel.needs_approval, "approver": rel.approver,
                "reasons": list(rel.reasons), "rationale": po.rationale, "total": po.total}

    # ---- stage 6: alerts for people ----------------------------------------------------------
    def alerts(self) -> List[Dict[str, str]]:
        a = agent_loader.load()
        if self.pos is None or len(self.releases) != len(self.pos):
            raise RuntimeError("every line needs a release_check before alerts can be drafted")
        releases = [self.releases[i] for i in range(len(self.pos))]
        out = a.alerts.build_alerts(self.state, self.eff_rules, self.tags, self.decisions, self.pos, self.flags,
                                    self.statuses, self.violations, releases)
        if self.policy_note:
            out.insert(0, a.alerts.Alert("WARNING", self.policy_note))
        order = a.alerts.ORDER
        if self.part_labels.alerts:
            summary = out.pop()
            out = sorted(out + self.part_labels.alerts, key=lambda x: order[x.severity]) + [summary]
        if self.eff_rules.raw.get("settings", {}).get("recovery_options"):
            result = a.pipeline.RunResult(self.pos, out, self.statuses, self.decisions, self.tags, releases, self.part_labels)
            try:
                exceptions = a.exceptions.recovery_options(self.state, self.eff_rules, result)
            except Exception as e:  # noqa: BLE001  (pipeline.run treats advice failures the same way)
                exceptions = []
                out.insert(len(out) - 1, a.alerts.Alert("WARNING", f"Recovery options could not be worked out "
                                                                   f"({type(e).__name__}: {str(e)[:120]}). The plan itself is unaffected."))
            summary = out.pop()
            out = out + [a.exceptions.as_alert(ex, self.eff_rules, "") for ex in exceptions]
            out = sorted(out, key=lambda x: order[x.severity]) + [summary]
        return [{"severity": x.severity, "text": x.text} for x in out]
