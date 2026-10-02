from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from agent_reliability_lab.types import AgentDecision, ToolResult

class DecisionModel(ABC):
    @abstractmethod
    def decide(
        self, task: Dict[str, Any], tools: Dict[str, Any], observations: Optional[List[ToolResult]] = None
    ) -> AgentDecision:
        raise NotImplementedError

    def __call__(
        self, task: Dict[str, Any], tools: Dict[str, Any], observations: Optional[List[ToolResult]] = None
    ) -> AgentDecision:
        """Lets a model be passed straight to AgentRuntime as a planner."""
        return self.decide(task, tools, observations)
