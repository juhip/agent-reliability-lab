"""Procurement wording for the untrusted-text screen, on top of core's neutral patterns."""
from __future__ import annotations
from typing import Any, Dict, List

from agent_reliability_lab.screening import compile_patterns, screen_value

INJECTION_PATTERNS = [
    r"(release|approve|issue|place) (all|every|this|these|the) (orders?|pos?|purchase orders?)\b.{0,40}(now|immediately|without)",
    r"(skip|bypass|waive|override) (the )?(approval|review|policy|rules?|checks?)",
    r"auto-?release mode",
    r"\bsystem\s*:",
]

PATTERNS = compile_patterns(INJECTION_PATTERNS)


def flagged_request_fields(task: Dict[str, Any]) -> List[str]:
    """Paths of request fields (everything but the scenario handle) that read like instructions."""
    return [h.path for h in screen_value({k: v for k, v in task.items() if k != "scenario"}, "task", PATTERNS)]


def flagged_scenario_text(untrusted: List[Dict[str, str]]) -> List[str]:
    """`where` labels of scenario free text (supplier/catalog notes, descriptions) that read like instructions."""
    return [t["where"] for t in untrusted if screen_value(t["text"], "text", PATTERNS)]
