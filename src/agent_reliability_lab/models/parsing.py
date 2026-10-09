"""Turn raw model text into an AgentDecision. Small models wrap JSON in fences or prose;
anything unrecoverable raises ValueError so the caller can retry or fail closed."""
from __future__ import annotations
import ast
import json
import re
from typing import Any, Dict, List
from agent_reliability_lab.types import AgentDecision, ToolCall

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_NATIVE_CALLS = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.S)


def extract_native_tool_calls(text: str) -> List[ToolCall]:
    """LFM models answer in their trained tool-call format, e.g.
    <|tool_call_start|>[lookup_po(po_id='PO-1')]<|tool_call_end|>, instead of the requested JSON.
    Returns the calls, or [] if the text holds none. Only literal keyword arguments are accepted."""
    calls: List[ToolCall] = []
    for block in _NATIVE_CALLS.findall(text):
        try:
            tree = ast.parse(block.strip(), mode="eval").body
        except SyntaxError:
            return []
        for node in (tree.elts if isinstance(tree, ast.List) else [tree]):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)) or node.args:
                return []
            try:
                args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
            except ValueError:
                return []
            calls.append(ToolCall(node.func.id, args))
    return calls


def extract_json_object(text: str) -> Dict[str, Any]:
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    depth, in_str, escaped = 0, False, False
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(candidate[start:i + 1])
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON: {exc}") from exc
                if not isinstance(data, dict):
                    raise ValueError("top-level JSON must be an object")
                return data
    raise ValueError("unterminated JSON object in model output")


def parse_decision(data: Dict[str, Any]) -> AgentDecision:
    action = data.get("action")
    if not isinstance(action, str) or not action:
        raise ValueError("'action' must be a non-empty string")
    raw_calls = data.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise ValueError("'tool_calls' must be a list")
    calls = []
    for item in raw_calls:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("each tool call needs a string 'name'")
        args = item.get("arguments", {})
        if not isinstance(args, dict):
            raise ValueError("tool call 'arguments' must be an object")
        calls.append(ToolCall(item["name"], args))
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("'confidence' must be a number") from exc
    final = data.get("final", True)
    if not isinstance(final, bool):
        raise ValueError("'final' must be true or false")
    answer = data.get("final_answer")
    return AgentDecision(
        action=action,
        rationale=str(data.get("rationale", "")),
        confidence=confidence,
        tool_calls=calls,
        evidence=[str(e) for e in (data.get("evidence") or [])],
        final_answer=answer if isinstance(answer, dict) else None,
        is_final=final,
    )
