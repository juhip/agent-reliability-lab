"""Minimal JSON-schema argument checking, shared by the pipeline registry and the orchestrator.

Checks object shape, required keys, unknown keys and primitive types. It does not check
nested structure, formats, ranges or enums."""
from __future__ import annotations
import inspect
import typing
from typing import Any, Callable, Dict

_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def validate_args(schema: Dict[str, Any], args: Any) -> str | None:
    if not isinstance(args, dict):
        return "arguments must be an object"
    props = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        return f"missing required arguments: {missing}"
    extra = [k for k in args if k not in props]
    if extra:
        return f"unexpected arguments: {extra}"
    for key, value in args.items():
        want = props[key].get("type")
        if want in _JSON_TYPES:
            ok = isinstance(value, _JSON_TYPES[want]) and not (want in ("integer", "number", "string") and isinstance(value, bool))
            if not ok:
                return f"argument '{key}' must be {want}"
    return None


def schema_from_signature(fn: Callable[..., Any]) -> Dict[str, Any]:
    """Derive an argument schema from a function's signature and type hints.
    Parameters without a recognised hint are accepted with any type."""
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    props: Dict[str, Any] = {}
    required = []
    for p in inspect.signature(fn).parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        json_type = _PY_TO_JSON.get(hints.get(p.name))
        props[p.name] = {"type": json_type} if json_type else {}
        if p.default is inspect.Parameter.empty:
            required.append(p.name)
    return {"type": "object", "properties": props, "required": required}
