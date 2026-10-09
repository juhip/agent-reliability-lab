"""Untrusted-text screen: flag strings that read like instructions to the agent.

The core patterns are domain-free (no business nouns). A domain may add its own patterns as data.
The screen walks every string in nested dicts and lists, in the task and in tool results, and
reports the path of each hit, e.g. "task.customer_message.body" or "lookup_return[0].carrier_note".

This is a tripwire, not a defence: a regex list misses paraphrases. What it guarantees is that a
flagged case cannot reach a gated action; it does not make an unflagged case safe."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Pattern, Sequence, Tuple
from .types import ToolResult

CORE_PATTERNS: Tuple[str, ...] = (
    # override the agent's instructions
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions|prompts?|rules|guidelines|polic(y|ies)|guardrails)\b",
    r"\bnew (system )?instructions\b",
    # re-cast the agent's role
    r"\byou are now\b",
    r"\b(act|behave|respond) as (an? )?(admin|administrator|approver|developer|system|unrestricted)\b",
    r"(^|[\n>\]])\s*(system|assistant|developer)\s*(prompt|message|override)?\s*:",
    r"<\s*/?\s*(system|instructions?)\s*>",
    # ask the agent to skip its controls
    r"\bwithout (any |further |additional )?(review|verification|checks?|approval|escalation)\b",
    r"\b(bypass|skip|disable|turn off)\b.{0,20}\b(review|verification|checks?|controls?|approvals?|safeguards?|screening)\b",
    r"\bdo not (escalate|flag|review|verify|report)\b",
    r"\b(approve|accept|authori[sz]e|release)\b (everything|all of (them|these)|all requests|all pending)\b",
    # exfiltrate the prompt
    r"\b(reveal|print|repeat|show)\b.{0,20}\b(system prompt|hidden instructions|your instructions)\b",
)


@dataclass
class ScreenHit:
    path: str
    pattern: str
    excerpt: str


def compile_patterns(extra: Iterable[str] = ()) -> List[Pattern[str]]:
    """Core patterns plus a domain's extras. A bad extra pattern raises re.error at build time."""
    return [re.compile(p, re.I) for p in (*CORE_PATTERNS, *extra)]


def _strings(obj: Any, path: str) -> Iterator[Tuple[str, str]]:
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _strings(v, f"{path}[{i}]")


def screen_value(obj: Any, root: str, patterns: Sequence[Pattern[str]]) -> List[ScreenHit]:
    hits: List[ScreenHit] = []
    for path, text in _strings(obj, root):
        for p in patterns:
            m = p.search(text)
            if m:
                start = max(0, m.start() - 20)
                hits.append(ScreenHit(path, p.pattern, text[start:m.end() + 20]))
                break
    return hits


def screen(task: Any, results: Sequence[ToolResult], patterns: Sequence[Pattern[str]]) -> List[ScreenHit]:
    """Screen the task and every successful tool result. Paths name the tool and its call index."""
    hits = screen_value(task, "task", patterns)
    for i, r in enumerate(results):
        if r.ok:
            hits += screen_value(r.output, f"{r.name}[{i}]", patterns)
    return hits


_DEFAULT = compile_patterns()


def looks_like_injection(text: str, patterns: Sequence[Pattern[str]] | None = None) -> bool:
    return any(p.search(text) for p in (patterns or _DEFAULT))
