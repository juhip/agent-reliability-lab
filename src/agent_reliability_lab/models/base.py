from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict
from agent_reliability_lab.types import AgentDecision

class DecisionModel(ABC):
    @abstractmethod
    def decide(self, task: Dict[str, Any], tools: Dict[str, str]) -> AgentDecision:
        raise NotImplementedError
