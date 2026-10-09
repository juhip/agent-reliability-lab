"""Provider-neutral chat models for the orchestrator.

Neutral messages:
  {"role": "user", "content": str}
  {"role": "assistant", "text": str, "tool_calls": [{id, name, arguments}], "raw": provider payload | None}
  {"role": "tool", "results": [{id, name, content, is_error}]}
"""
from __future__ import annotations
import json
import urllib.request
from typing import Any, Callable, Dict, List
from .types import ModelTurn, ToolCallRequest


class ScriptedModel:
    """Replays fixed turns, or calls `fn(messages) -> ModelTurn`. For tests and baselines."""

    def __init__(self, turns: List[ModelTurn] | Callable[[List[Dict[str, Any]]], ModelTurn]) -> None:
        self._turns, self.calls = turns, 0

    def complete(self, system: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> ModelTurn:
        self.calls += 1
        if callable(self._turns):
            return self._turns(messages)
        if self.calls > len(self._turns):
            return ModelTurn(text="(script exhausted)")
        return self._turns[self.calls - 1]


class OpenAIChatModel:
    """Any OpenAI-compatible server with function calling (LM Studio, vLLM, llama.cpp, SGLang)."""

    def __init__(self, model: str, base_url: str = "http://localhost:1234/v1", timeout: float = 300.0,
                 max_tokens: int = 2048) -> None:
        self.model, self.base_url, self.timeout, self.max_tokens = model, base_url.rstrip("/"), timeout, max_tokens

    @staticmethod
    def _tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                   "parameters": t["parameters"]}} for t in tools]

    @staticmethod
    def _messages(system: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg: Dict[str, Any] = {"role": "assistant", "content": m["text"] or None}
                if m["tool_calls"]:
                    msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                          "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                         for c in m["tool_calls"]]
                out.append(msg)
            else:
                out += [{"role": "tool", "tool_call_id": r["id"], "content": r["content"]} for r in m["results"]]
        return out

    def complete(self, system: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> ModelTurn:
        body = {"model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
                "messages": self._messages(system, messages), "tools": self._tools(tools)}
        req = urllib.request.Request(f"{self.base_url}/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        msg, usage = data["choices"][0]["message"], data.get("usage") or {}
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            raw_args = c["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {"__unparseable__": raw_args}      # surfaces as invalid_args, not a crash
            calls.append(ToolCallRequest(c.get("id") or f"call_{i}", c["function"]["name"], args))
        finish = data["choices"][0].get("finish_reason")
        return ModelTurn(text=msg.get("content") or "", tool_calls=calls,
                         stop="tool_use" if calls else ("max_tokens" if finish == "length" else "end_turn"),
                         input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0))


class AnthropicChatModel:
    """Claude through the official `anthropic` SDK.

    NOT yet run against the live API (no credentials were available when this was written).
    Thinking blocks are echoed back verbatim by keeping the response content in `raw`.
    With `use_fallbacks`, a safety-classifier decline is retried server-side on a fallback model.
    """

    def __init__(self, model: str = "claude-opus-5-5", max_tokens: int = 16000, effort: str = "medium",
                 use_fallbacks: bool = True, client: Any = None) -> None:
        if client is None:
            import anthropic                          # optional dependency: pip install anthropic
            client = anthropic.Anthropic()
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self.effort, self.use_fallbacks = effort, use_fallbacks

    @staticmethod
    def _tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]

    @staticmethod
    def _messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                if m.get("raw") is not None:
                    out.append({"role": "assistant", "content": m["raw"]})   # includes any thinking blocks
                else:
                    blocks: List[Dict[str, Any]] = ([{"type": "text", "text": m["text"]}] if m["text"] else [])
                    blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]}
                               for c in m["tool_calls"]]
                    out.append({"role": "assistant", "content": blocks})
            else:   # all results for one assistant turn go back in a single user message
                out.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"], "is_error": r["is_error"]}
                    for r in m["results"]]})
        return out

    def complete(self, system: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> ModelTurn:
        kwargs: Dict[str, Any] = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                                      tools=self._tools(tools), messages=self._messages(messages),
                                      output_config={"effort": self.effort})
        if self.use_fallbacks:
            resp = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                    fallbacks="default", **kwargs)
        else:
            resp = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCallRequest(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        stop = "refusal" if resp.stop_reason == "refusal" else ("tool_use" if calls else
                ("max_tokens" if resp.stop_reason == "max_tokens" else "end_turn"))
        return ModelTurn(text=text, tool_calls=calls, stop=stop, raw=resp.content,
                         input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens)
