from __future__ import annotations
from dataclasses import dataclass
import inspect
from typing import Any, Dict, Mapping
from .types import ToolFn, ToolResult

@dataclass
class ToolSpec:
    name: str
    description: str
    fn: ToolFn

class ToolRegistry:
    """Allowlisted tool execution with basic argument-schema enforcement."""
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, name: str, description: str, fn: ToolFn) -> None:
        if name in self._tools:
            raise ValueError(f"duplicate tool: {name}")
        self._tools[name] = ToolSpec(name, description, fn)

    @property
    def descriptions(self) -> Dict[str, str]:
        return {k: v.description for k, v in self._tools.items()}

    @property
    def schemas(self) -> Dict[str, Dict[str, Any]]:
        schemas: Dict[str, Dict[str, Any]] = {}
        for name, spec in self._tools.items():
            sig = inspect.signature(spec.fn)
            schemas[name] = {
                "description": spec.description,
                "arguments": {
                    p.name: {
                        "required": p.default is inspect._empty,
                        "type": getattr(p.annotation, "__name__", str(p.annotation)),
                    }
                    for p in sig.parameters.values()
                },
            }
        return schemas

    def _validate_arguments(self, spec: ToolSpec, arguments: Mapping[str, Any]) -> str | None:
        if not isinstance(arguments, Mapping):
            return "arguments_must_be_object"
        sig = inspect.signature(spec.fn)
        try:
            sig.bind(**dict(arguments))
        except TypeError as exc:
            return f"invalid_arguments: {exc}"
        return None

    def call(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        if name not in self._tools:
            return ToolResult(name=name, ok=False, error="tool_not_allowlisted")
        spec = self._tools[name]
        error = self._validate_arguments(spec, arguments)
        if error:
            return ToolResult(name=name, ok=False, error=error)
        try:
            return ToolResult(name=name, ok=True, output=spec.fn(**arguments))
        except Exception as exc:
            return ToolResult(name=name, ok=False, error=f"{type(exc).__name__}: {exc}")
