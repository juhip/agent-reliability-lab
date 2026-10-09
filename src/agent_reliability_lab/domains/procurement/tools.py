"""Allowlisted procurement tools: thin, validated wrappers over the agent's deterministic stages.

Every tool takes a `scenario` handle. Handles are registered by whoever builds the runtime (the eval
harness), so a planner cannot point a tool at an arbitrary file path. All tools are read-only: the
scenario database is opened read-only and nothing is written; writing a plan is the caller's job.

`read_records` is different from the others: it returns raw rows straight from the scenario file
(plain SQL, none of the agent's code), so the independent verifier has data the agent did not shape.
"""
from __future__ import annotations
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from agent_reliability_lab.agentic.types import READ, Tool
from .session import Session

LINE_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,64}:[A-Za-z0-9_.\-]{1,64}:(standard|air)$")
COMPONENT_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


class ArgumentError(ValueError):
    pass


def _str(name: str, value: Any, pattern: re.Pattern | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArgumentError(f"{name} must be a non-empty string")
    value = value.strip()
    if pattern is not None and not pattern.match(value):
        raise ArgumentError(f"{name} has an invalid format")
    return value


def _schema(**props: str) -> Dict[str, Any]:
    return {"type": "object", "properties": {k: {"type": "string"} for k in props}, "required": list(props)}


class ProcurementTools:
    """Holds the scenario allowlist and one Session per scenario for the life of a run."""

    def __init__(self, rules, scenarios: Dict[str, str | Path] | None = None) -> None:
        self.rules = rules
        self.scenarios: Dict[str, str] = {k: str(v) for k, v in (scenarios or {}).items()}
        self.sessions: Dict[str, Session] = {}

    def register_scenario(self, handle: str, path: str | Path) -> None:
        self.scenarios[handle] = str(path)

    def reset(self, handle: str | None = None) -> None:
        if handle is None:
            self.sessions.clear()
        else:
            self.sessions.pop(handle, None)

    def path(self, scenario: Any) -> str:
        handle = _str("scenario", scenario)
        if handle not in self.scenarios:
            raise ArgumentError("scenario is not registered for this run")
        return self.scenarios[handle]

    def session(self, scenario: Any) -> Session:
        handle = _str("scenario", scenario)
        if handle not in self.sessions:
            self.sessions[handle] = Session(self.path(handle), self.rules)
        return self.sessions[handle]

    # ---- the tools (keyword arguments only; the registry binds them by name) ------------------
    def load_scenario(self, scenario: str) -> Dict[str, Any]:
        return self.session(scenario).overview()

    def net_requirements(self, scenario: str) -> Dict[str, Any]:
        return self.session(scenario).shortfalls()

    def plan_component(self, scenario: str, component_id: str) -> Dict[str, Any]:
        return self.session(scenario).plan_component(_str("component_id", component_id, COMPONENT_ID))

    def check_hard_rules(self, scenario: str) -> Dict[str, Any]:
        return self.session(scenario).verify()

    def release_check(self, scenario: str, line_id: str) -> Dict[str, Any]:
        return self.session(scenario).release_check(_str("line_id", line_id, LINE_ID))

    def draft_alerts(self, scenario: str) -> List[Dict[str, str]]:
        return self.session(scenario).alerts()

    def read_records(self, scenario: str) -> Dict[str, Any]:
        """Raw structured rows (no free text) from the scenario file, read-only."""
        con = sqlite3.connect(f"file:{self.path(scenario)}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            q = lambda sql: [dict(r) for r in con.execute(sql)]
            return {
                "components": q("SELECT component_id, name, category, is_hazardous, requires_certification FROM components"),
                "suppliers": q("SELECT supplier_id, name, country, certifications, on_approved_list FROM suppliers"),
                "catalog": q("SELECT supplier_id, component_id, unit_price, lead_time_days, minimum_order_qty "
                             "FROM supplier_catalog"),
            }
        finally:
            con.close()

    def tools(self) -> List[Tool]:
        sc = _schema(scenario="")
        return [
            Tool("load_scenario", "Read the scenario: date, parts, production orders, untrusted free text.", sc,
                 self.load_scenario, READ),
            Tool("net_requirements", "Net demand against stock and open orders: what is short, by when.", sc,
                 self.net_requirements, READ),
            Tool("plan_component", "Choose suppliers and quantities for one short component under the rulebook.",
                 _schema(scenario="", component_id=""), self.plan_component, READ),
            Tool("check_hard_rules", "Independent hard-rule gate over every planned line, plus a calendar replay.", sc,
                 self.check_hard_rules, READ),
            Tool("release_check", "Does this line need a named approver before release? Returns the rationale.",
                 _schema(scenario="", line_id=""), self.release_check, READ),
            Tool("draft_alerts", "Alerts for people (late orders, approvals, recovery options).", sc,
                 self.draft_alerts, READ),
            Tool("read_records", "Raw component, supplier and catalog rows from the scenario file.", sc,
                 self.read_records, READ),
        ]
