from __future__ import annotations
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List
from .trajectory import Event, Trajectory
from .types import SPEND, ModelTurn, Tool, Toolbox, validate_args


@dataclass
class Approval:
    allowed: bool
    reason: str = ""


Approver = Callable[[Tool, Dict[str, Any], Trajectory], Approval]


def deny_all(tool: Tool, args: Dict[str, Any], traj: Trajectory) -> Approval:
    return Approval(False, "approval_required")


class Orchestrator:
    """The model chooses every step. Code owns authority.

    Each turn the model may request tools; the runtime validates the call, enforces the
    permission tier, runs it, and feeds the result back. SPEND-tier tools only run if the
    approver says so, and the approver sees the whole trajectory, so it can demand evidence
    the model actually gathered. A model can talk its way into nothing.

    With `allow_delegate`, the model gets a `delegate` tool: it hands a bounded task to a
    fresh sub-agent with a restricted toolbox and receives only a short summary back, which
    keeps the parent's context small.
    """

    def __init__(self, model, toolbox: Toolbox, system: str, approver: Approver = deny_all,
                 max_steps: int = 30, max_total_tokens: int = 400_000, max_result_chars: int = 4000,
                 allow_delegate: bool = False, child_max_steps: int = 15, depth: int = 0) -> None:
        self.model, self.system, self.approver = model, system, approver
        self.max_steps, self.max_total_tokens, self.max_result_chars = max_steps, max_total_tokens, max_result_chars
        self.child_max_steps, self.depth = child_max_steps, depth
        self.base_toolbox = toolbox
        self.toolbox = Toolbox(list(toolbox._tools.values()))
        self.traj: Trajectory | None = None
        if allow_delegate and depth == 0:
            self.toolbox.add(Tool(
                "delegate",
                "Hand one bounded task to a sub-agent with only the tools you name. It works in a fresh context "
                "and returns a short summary. Use it to keep your own context small.",
                {"type": "object", "properties": {"task": {"type": "string"},
                                                  "tools": {"type": "array"}}, "required": ["task", "tools"]},
                self._delegate, tier=1))

    # ---- sub-agents -------------------------------------------------------------------
    def _delegate(self, task: str, tools: List[str]) -> Dict[str, Any]:
        allowed = [t for t in tools if t in self.base_toolbox.names() and t != "delegate"]
        child = Orchestrator(self.model, self.base_toolbox.subset(allowed), self.system, self.approver,
                             max_steps=self.child_max_steps, max_total_tokens=self.max_total_tokens,
                             max_result_chars=self.max_result_chars, depth=self.depth + 1)
        traj = child.run(task)
        assert self.traj is not None
        self.traj.children.append(traj)
        return {"summary": traj.final_text, "steps": traj.steps, "stop_reason": traj.stop_reason,
                "tools_granted": allowed}

    # ---- the loop ---------------------------------------------------------------------
    def run(self, goal: str) -> Trajectory:
        traj = self.traj = Trajectory(goal=goal, depth=self.depth)
        messages: List[Dict[str, Any]] = [{"role": "user", "content": goal}]
        started = time.perf_counter()
        try:
            for step in range(self.max_steps):
                if traj.input_tokens + traj.output_tokens > self.max_total_tokens:
                    traj.stop_reason = "budget"
                    break
                turn: ModelTurn = self.model.complete(self.system, messages, self.toolbox.specs())
                traj.steps += 1
                traj.input_tokens += turn.input_tokens
                traj.output_tokens += turn.output_tokens
                messages.append({"role": "assistant", "text": turn.text, "raw": turn.raw,
                                 "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments}
                                                for c in turn.tool_calls]})
                if turn.stop == "refusal":
                    traj.stop_reason, traj.final_text = "refusal", turn.text
                    break
                if not turn.tool_calls:
                    traj.stop_reason, traj.final_text = ("max_tokens" if turn.stop == "max_tokens" else "final"), turn.text
                    break
                results = [self._execute(step, c.id, c.name, c.arguments) for c in turn.tool_calls]
                messages.append({"role": "tool", "results": results})
            else:
                traj.stop_reason = "max_steps"
        except Exception as exc:                     # a model/transport failure ends the run; never crashes the caller
            traj.stop_reason, traj.final_text = "error", f"{type(exc).__name__}: {exc}"
        traj.latency_s = time.perf_counter() - started
        return traj

    def _execute(self, step: int, call_id: str, name: str, args: Any) -> Dict[str, Any]:
        traj = self.traj
        assert traj is not None

        def reply(content: Any, is_error: bool) -> Dict[str, Any]:
            text = json.dumps(content, default=str)
            if len(text) > self.max_result_chars:
                text = text[: self.max_result_chars] + '..."(truncated)"'
            return {"id": call_id, "name": name, "content": text, "is_error": is_error}

        tool = self.toolbox.get(name)
        if tool is None:
            traj.events.append(Event(step, "unknown_tool", name, args if isinstance(args, dict) else {}, False))
            return reply({"error": f"unknown tool '{name}'", "available": self.toolbox.names()}, True)
        problem = validate_args(tool.parameters, args)
        if problem:
            traj.events.append(Event(step, "invalid_args", name, args if isinstance(args, dict) else {}, False, note=problem))
            return reply({"error": problem}, True)
        if tool.tier >= SPEND:
            verdict = self.approver(tool, args, traj)
            if not verdict.allowed:
                traj.events.append(Event(step, "denied", name, args, False, note=verdict.reason))
                return reply({"error": "denied", "reason": verdict.reason}, True)
        try:
            result = tool.handler(**args)
            traj.events.append(Event(step, "tool_call", name, args, True, result))
            return reply(result, False)
        except Exception as exc:
            traj.events.append(Event(step, "tool_call", name, args, False, f"{type(exc).__name__}: {exc}"))
            return reply({"error": f"{type(exc).__name__}: {exc}"}, True)
