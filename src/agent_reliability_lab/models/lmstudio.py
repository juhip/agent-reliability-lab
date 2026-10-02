from __future__ import annotations
import json
from typing import Any, Dict
from urllib import request
from agent_reliability_lab.models.base import DecisionModel
from agent_reliability_lab.types import AgentDecision, ToolCall

class LMStudioDecisionModel(DecisionModel):
    """OpenAI-compatible local endpoint adapter. Defaults to LM Studio localhost."""
    def __init__(self, model: str = "local-model", base_url: str = "http://localhost:1234/v1") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def decide(self, task: Dict[str, Any], tools: Dict[str, str]) -> AgentDecision:
        system = (
            "You are a constrained enterprise agent. Return JSON only with keys: "
            "action, rationale, confidence, tool_calls, evidence, final_answer. "
            "Never invent tool outputs. Use only listed tools. If evidence is insufficient, "
            "set action to HUMAN_REVIEW."
        )
        user = json.dumps({"task": task, "tools": tools})
        payload = json.dumps({
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }).encode()
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
        content = body["choices"][0]["message"]["content"]
        data = json.loads(content)
        return AgentDecision(
            action=data["action"],
            rationale=data.get("rationale", ""),
            confidence=float(data.get("confidence", 0)),
            tool_calls=[ToolCall(name=x["name"], arguments=x.get("arguments", {})) for x in data.get("tool_calls", [])],
            evidence=list(data.get("evidence", [])),
            final_answer=data.get("final_answer"),
        )
