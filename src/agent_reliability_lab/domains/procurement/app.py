from __future__ import annotations
from typing import Dict, Optional

from agent_reliability_lab.runtime import AgentRuntime, Planner, build_runtime
from .domain import build_domain
from .observing_planner import ObservingProcurementPlanner
from .tools import ProcurementTools


def build_procurement_runtime(planner: Planner | str | None = None, rules=None, scenarios: Optional[Dict[str, str]] = None,
                              max_steps: int | None = None) -> AgentRuntime:
    """`rules` is the agent's Rulebook (see session.rulebook); `scenarios` maps handles to .sqlite paths.
    The runtime exposes `.procurement_tools` so callers can register more scenarios."""
    from .session import rulebook
    from .planner import ProcurementPlanner
    tools = ProcurementTools(rules if rules is not None else rulebook(), scenarios)
    if planner is None or planner == "observing":
        planner = ObservingProcurementPlanner()
    elif planner == "oneshot":
        planner = ProcurementPlanner(tools)
    runtime = build_runtime(build_domain(tools), planner, max_steps)
    runtime.procurement_tools = tools
    return runtime
