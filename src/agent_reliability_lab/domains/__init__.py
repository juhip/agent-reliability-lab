"""Bundled domains. Each package exposes a `suite` module with DOMAIN, CASE_FILES, PLANNERS,
REFERENCE, PIPELINE_PLANNER and STAND_INS, or, for a domain whose cases are generated at run time
(procurement), a `main(argv)` that the runners hand over to. Core never imports from here."""
import importlib
from types import ModuleType

SUITES = {
    "invoice": "agent_reliability_lab.domains.invoice.suite",
    "refund": "agent_reliability_lab.domains.refund.suite",
    "procurement": "agent_reliability_lab.domains.procurement.suite",
}


def load_suite(name: str) -> ModuleType:
    if name not in SUITES:
        raise ValueError(f"unknown domain {name!r}; known: {sorted(SUITES)}")
    return importlib.import_module(SUITES[name])
