from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List
from agent_reliability_lab.schema import validate_args  # noqa: F401  (re-exported for callers)

READ, WRITE, SPEND = 0, 1, 2   # authority tiers: read freely, write drafts/logs, move money or leave the building


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ModelTurn:
    """One model response, provider-neutral. `raw` is the provider's own assistant payload,
    kept so adapters can echo it back verbatim (needed for thinking blocks)."""
    text: str = ""
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    stop: str = "end_turn"          # end_turn | tool_use | refusal | max_tokens
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]      # JSON schema, type "object"
    handler: Callable[..., Any]
    tier: int = READ


class Toolbox:
    def __init__(self, tools: List[Tool] | None = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def specs(self) -> List[Dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "parameters": t.parameters, "tier": t.tier}
                for t in self._tools.values()]

    def subset(self, names: List[str]) -> "Toolbox":
        return Toolbox([self._tools[n] for n in names if n in self._tools])

