"""Declarative human-review triggers.

A domain declares, as data, conditions under which a person must decide, whatever the
planner proposed. Core evaluates them over the task and the tool results; a match blocks
every gated action and routes the case to the domain's fail-safe outcome.

Spec (a dict):
    name      required, unique label recorded in the trace when the trigger fires
    source    "task" or the name of a tool; for a tool, every successful result of that tool
              that is in scope for the case is checked
    path      dotted path into the source, e.g. "flags.chargeback_open" or "items.final_sale".
              A list along the path fans out (matches if ANY element matches); a numeric
              segment indexes a list ("items.0.sku").
    op        eq | ne | gt | ge | lt | le | in | not_in | contains | exists | missing
    value     operand (not used by exists / missing)
    required  default False. See "unresolved" below.

Unresolved (the source tool was never called successfully, the path does not exist, or the
comparison raised, e.g. "gt" between a string and a number):
    required=True   -> the trigger FIRES (fail closed), reason "unresolved"
    required=False  -> the trigger does not fire; the miss is reported so it lands in the trace
`exists` and `missing` never count as unresolved: answering that question is their job.

Malformed specs (unknown op, missing name/path/source, value of the wrong shape for in/not_in)
raise ValueError when the domain is built, so a typo can never silently disable a trigger.
"""
from __future__ import annotations
import operator
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from .types import ToolResult

_MISSING = object()

def _eq(a: Any, b: Any) -> bool:
    """Equality that does not treat True as 1: a flag declared `eq true` must really be a boolean."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return a == b


_OPS = {
    "eq": _eq, "ne": lambda a, b: not _eq(a, b), "gt": operator.gt, "ge": operator.ge, "lt": operator.lt, "le": operator.le,
    "in": lambda a, b: a in b, "not_in": lambda a, b: a not in b, "contains": lambda a, b: b in a,
}
_PRESENCE = {"exists", "missing"}
_KEYS = {"name", "source", "path", "op", "value", "required"}


@dataclass(frozen=True)
class Trigger:
    name: str
    source: str
    path: Tuple[str, ...]
    op: str
    value: Any = None
    required: bool = False


@dataclass
class TriggerHit:
    name: str
    reason: str          # "matched" | "unresolved"
    detail: str


def compile_trigger(spec: Dict[str, Any]) -> Trigger:
    if not isinstance(spec, dict):
        raise ValueError(f"trigger spec must be a dict, got {type(spec).__name__}")
    unknown = set(spec) - _KEYS
    if unknown:
        raise ValueError(f"trigger {spec.get('name')!r}: unknown keys {sorted(unknown)}")
    for key in ("name", "source", "path", "op"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise ValueError(f"trigger {spec.get('name')!r}: '{key}' must be a non-empty string")
    op = spec["op"]
    if op not in _OPS and op not in _PRESENCE:
        raise ValueError(f"trigger {spec['name']!r}: unknown op {op!r}")
    if op in ("in", "not_in") and not isinstance(spec.get("value"), (list, tuple, set, frozenset)):
        raise ValueError(f"trigger {spec['name']!r}: op {op!r} needs a list value")
    if op not in _PRESENCE and "value" not in spec:
        raise ValueError(f"trigger {spec['name']!r}: op {op!r} needs a value")
    if not isinstance(spec.get("required", False), bool):
        raise ValueError(f"trigger {spec['name']!r}: 'required' must be true or false")
    value = spec.get("value")
    if isinstance(value, list):
        value = tuple(value)
    return Trigger(spec["name"], spec["source"], tuple(spec["path"].split(".")), op, value, spec.get("required", False))


def compile_triggers(specs: Iterable[Dict[str, Any]]) -> List[Trigger]:
    triggers = [compile_trigger(s) for s in specs]
    names = [t.name for t in triggers]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate trigger names: {dupes}")
    return triggers


def resolve(obj: Any, path: Sequence[str]) -> List[Any]:
    """All values at `path`. Lists fan out; a missing key yields nothing."""
    if not path:
        return [obj]
    head, rest = path[0], path[1:]
    if isinstance(obj, dict):
        return resolve(obj[head], rest) if head in obj else []
    if isinstance(obj, (list, tuple)):
        if head.isdigit():
            i = int(head)
            return resolve(obj[i], rest) if i < len(obj) else []
        return [v for item in obj for v in resolve(item, path)]
    return []


def _sources(trigger: Trigger, task: Dict[str, Any], results: Sequence[ToolResult]) -> List[Any]:
    if trigger.source == "task":
        return [task]
    return [r.output for r in results if r.ok and r.name == trigger.source]


def evaluate(triggers: Sequence[Trigger], task: Dict[str, Any],
             results: Sequence[ToolResult]) -> Tuple[List[TriggerHit], List[TriggerHit]]:
    """Returns (fired, unresolved_but_not_required). Only `fired` blocks."""
    fired: List[TriggerHit] = []
    soft: List[TriggerHit] = []
    for t in triggers:
        where = f"{t.source}.{'.'.join(t.path)}"
        values = [v for src in _sources(t, task, results) for v in resolve(src, t.path)]
        if t.op == "exists":
            if values:
                fired.append(TriggerHit(t.name, "matched", f"{where} exists"))
            continue
        if t.op == "missing":
            if not values:
                fired.append(TriggerHit(t.name, "matched", f"{where} missing"))
            continue
        matched, errors = False, []
        for v in values:
            try:
                if _OPS[t.op](v, t.value):
                    matched = True
                    break
            except TypeError as exc:
                errors.append(str(exc))
        if matched:
            fired.append(TriggerHit(t.name, "matched", f"{where} {t.op} {t.value!r}"))
        elif not values or errors:
            why = f"{where} could not be evaluated" + (f" ({errors[0]})" if errors else " (no value)")
            (fired if t.required else soft).append(TriggerHit(t.name, "unresolved", why))
    return fired, soft
