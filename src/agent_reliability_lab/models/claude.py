from __future__ import annotations
import json
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional
from agent_reliability_lab.models.base import DecisionModel
from agent_reliability_lab.models.lmstudio import SYSTEM
from agent_reliability_lab.models.parsing import extract_json_object, parse_decision
from agent_reliability_lab.types import AgentDecision, ToolResult

# USD per million tokens, for the run's cost estimate only.
PRICES = {"claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50)}


class ClaudeDecisionModel(DecisionModel):
    """Claude through the official `anthropic` SDK, as a pipeline-mode planner.

    Same contract as the LM Studio adapter: one JSON decision per call, parsed tolerantly,
    one retry on unusable output. A safety-classifier refusal raises, so the runtime fails safe.
    With `use_fallbacks`, a refusal is first retried server-side on a fallback model.
    `stats` accumulates calls, tokens, latency and an estimated cost."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", max_tokens: int = 16000,
                 max_retries: int = 1, use_fallbacks: bool = True,
                 actions: tuple = ("APPROVE", "HUMAN_REVIEW"), fail_safe: str = "HUMAN_REVIEW",
                 client: Any = None) -> None:
        if client is None:
            import anthropic                          # optional dependency: pip install anthropic
            client = anthropic.Anthropic()
        self.client, self.model, self.effort, self.max_tokens = client, model, effort, max_tokens
        self.max_retries, self.use_fallbacks = max_retries, use_fallbacks
        self.actions, self.fail_safe = tuple(actions), fail_safe
        self.stats: Dict[str, Any] = {"calls": 0, "retries": 0, "refusals": 0, "input_tokens": 0,
                                      "output_tokens": 0, "latency_s": [], "est_cost_usd": 0.0}

    def _create(self, system: str, messages: List[Dict[str, str]]) -> str:
        kwargs: Dict[str, Any] = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                                      messages=messages, output_config={"effort": self.effort})
        started = time.perf_counter()
        if self.use_fallbacks:
            resp = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                    fallbacks="default", **kwargs)
        else:
            resp = self.client.messages.create(**kwargs)
        self.stats["calls"] += 1
        self.stats["latency_s"].append(round(time.perf_counter() - started, 3))
        self.stats["input_tokens"] += resp.usage.input_tokens
        self.stats["output_tokens"] += resp.usage.output_tokens
        price_in, price_out = PRICES.get(self.model, (0.0, 0.0))
        self.stats["est_cost_usd"] = round((self.stats["input_tokens"] * price_in
                                            + self.stats["output_tokens"] * price_out) / 1e6, 4)
        if resp.stop_reason == "refusal":
            self.stats["refusals"] += 1
            raise RuntimeError(f"model refused ({getattr(resp.stop_details, 'category', None)})")
        if resp.stop_reason == "max_tokens":
            raise ValueError("output hit max_tokens")
        return "".join(b.text for b in resp.content if b.type == "text")

    def decide(self, task: Dict[str, Any], tools: Dict[str, Any],
               observations: Optional[List[ToolResult]] = None) -> AgentDecision:
        system = SYSTEM.format(tools=json.dumps(tools), actions=" or ".join(self.actions), fail_safe=self.fail_safe)
        messages = [{"role": "user", "content": json.dumps(
            {"task": task, "observations": [asdict(o) for o in observations or []]})}]
        last_error: Optional[Exception] = None
        for _ in range(self.max_retries + 1):
            text = self._create(system, messages)
            try:
                return parse_decision(extract_json_object(text))
            except ValueError as exc:
                last_error = exc
                self.stats["retries"] += 1
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"That was not usable ({exc}). Reply with ONE JSON object only."},
                ]
        raise ValueError(f"model output unusable after {self.max_retries + 1} attempts: {last_error}")
