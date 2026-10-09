from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Mapping
from .schema import schema_from_signature, validate_args
from .types import ToolFn, ToolResult

@dataclass
class ToolSpec:
    name: str
    description: str
    fn: ToolFn
    parameters: Dict[str, Any]
    read_only: bool = True

class ToolRegistry:
    """Allowlisted tool execution with argument-schema enforcement: required and unknown
    argument names, and primitive JSON types (string, number, integer, boolean, array, object).
    The schema is the one given at registration, or is derived from the function's type hints."""
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, name: str, description: str, fn: ToolFn,
                 parameters: Dict[str, Any] | None = None, read_only: bool = True) -> None:
        if name in self._tools:
            raise ValueError(f"duplicate tool: {name}")
        self._tools[name] = ToolSpec(name, description, fn, parameters or schema_from_signature(fn), read_only)

    @property
    def descriptions(self) -> Dict[str, str]:
        return {k: v.description for k, v in self._tools.items()}

    @property
    def schemas(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for name, spec in self._tools.items():
            required = set(spec.parameters.get("required", []))
            out[name] = {
                "description": spec.description,
                "arguments": {
                    arg: {"required": arg in required, "type": prop.get("type", "any")}
                    for arg, prop in spec.parameters.get("properties", {}).items()
                },
            }
        return out

    def is_read_only(self, name: str) -> bool:
        spec = self._tools.get(name)
        return bool(spec and spec.read_only)

    def call(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        args = dict(arguments) if isinstance(arguments, Mapping) else arguments
        if name not in self._tools:
            return ToolResult(name=name, ok=False, error="tool_not_allowlisted", arguments=args if isinstance(args, dict) else None)
        spec = self._tools[name]
        problem = validate_args(spec.parameters, args)
        if problem:
            return ToolResult(name=name, ok=False, error=f"invalid_arguments: {problem}",
                              arguments=args if isinstance(args, dict) else None)
        try:
            return ToolResult(name=name, ok=True, output=spec.fn(**args), arguments=args)
        except Exception as exc:
            return ToolResult(name=name, ok=False, error=f"{type(exc).__name__}: {exc}", arguments=args)
