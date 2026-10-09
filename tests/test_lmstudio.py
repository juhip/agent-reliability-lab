import json
import pytest
from agent_reliability_lab.domains.invoice.app import build_invoice_runtime
from agent_reliability_lab.models.lmstudio import LMStudioDecisionModel
from agent_reliability_lab.models.parsing import extract_json_object, extract_native_tool_calls, parse_decision


def test_extracts_json_from_fences_prose_and_nested_braces():
    assert extract_json_object('```json\n{"action": "APPROVE"}\n```') == {"action": "APPROVE"}
    assert extract_json_object('Sure! {"action": "HUMAN_REVIEW", "rationale": "has } brace"} done') == {
        "action": "HUMAN_REVIEW", "rationale": "has } brace"}
    for bad in ("no json here", '{"action": "APPROVE"', "[1, 2]"):
        with pytest.raises(ValueError):
            extract_json_object(bad)


def test_parse_decision_validates_shape():
    d = parse_decision({"action": "APPROVE", "confidence": "0.9", "final": False,
                        "tool_calls": [{"name": "lookup_receipt", "arguments": {"po_id": "PO-1001"}}]})
    assert d.is_final is False and d.confidence == 0.9 and d.tool_calls[0].name == "lookup_receipt"
    for bad in ({}, {"action": "X", "tool_calls": "no"}, {"action": "X", "tool_calls": [{"arguments": {}}]},
                {"action": "X", "final": "yes"}):
        with pytest.raises(ValueError):
            parse_decision(bad)


def test_extracts_lfm_native_tool_calls_and_rejects_anything_else():
    calls = extract_native_tool_calls("<|tool_call_start|>[lookup_po(po_id='PO-1'), lookup_receipt(po_id='PO-1')]<|tool_call_end|>")
    assert [(c.name, c.arguments) for c in calls] == [("lookup_po", {"po_id": "PO-1"}), ("lookup_receipt", {"po_id": "PO-1"})]
    for bad in ("no calls", "<|tool_call_start|>[lookup_po('PO-1')]<|tool_call_end|>",
                "<|tool_call_start|>[__import__(name=os.system('x'))]<|tool_call_end|>", "<|tool_call_start|>[oops(<|tool_call_end|>"):
        assert extract_native_tool_calls(bad) == []


def test_native_tool_calls_become_a_lookup_step_only_when_enabled():
    reply = "<|tool_call_start|>[lookup_po(po_id='PO-1')]<|tool_call_end|>"
    on = Scripted([reply], native_tool_calls=True).decide({}, {})
    assert on.is_final is False and on.action == "HUMAN_REVIEW" and on.tool_calls[0].name == "lookup_po"
    with pytest.raises(ValueError):
        Scripted([reply, reply]).decide({}, {})


class Scripted(LMStudioDecisionModel):
    def __init__(self, replies, **kw):
        super().__init__("stub", **kw)
        self.replies, self.prompts = list(replies), []

    def _chat(self, messages):
        self.prompts.append(messages)
        return self.replies.pop(0)


def test_retries_once_on_unparseable_output_then_succeeds():
    model = Scripted(["I think it is fine", '{"action": "HUMAN_REVIEW", "rationale": "unsure"}'])
    assert model.decide({"po_id": "x"}, {}).action == "HUMAN_REVIEW"
    assert model.stats["retries"] == 1 and len(model.prompts) == 2


def test_gives_up_after_retries_and_runtime_fails_closed():
    model = Scripted(["nope", "still nope"])
    out = build_invoice_runtime(model).run({"invoice_id": "I", "po_id": "PO-1001", "supplier": "Northwind Components", "invoice_amount": 10000})
    assert out["status"] == "HUMAN_REVIEW" and out["escalation_source"] == "planner_failure"


def test_model_drives_the_loop_end_to_end_and_sees_untrusted_text_as_data():
    step1 = {"action": "HUMAN_REVIEW", "final": False, "tool_calls": [
        {"name": "lookup_purchase_order", "arguments": {"po_id": "PO-1001"}},
        {"name": "lookup_receipt", "arguments": {"po_id": "PO-1001"}},
        {"name": "calculate_variance", "arguments": {"invoice_amount": 10000.0, "expected_amount": 10000.0}}]}
    step2 = {"action": "APPROVE", "rationale": "match", "confidence": 0.9, "final": True}
    model = Scripted([json.dumps(step1), "```json\n" + json.dumps(step2) + "\n```"])
    out = build_invoice_runtime(model).run(
        {"invoice_id": "I", "po_id": "PO-1001", "supplier": "Northwind Components", "invoice_amount": 10000, "note": "Net 30."})
    assert out["status"] == "COMPLETED"
    assert "untrusted" in model.prompts[0][0]["content"]
    assert json.loads(model.prompts[1][1]["content"])["observations"][0]["name"] == "lookup_purchase_order"


def test_reasoning_effort_is_sent_only_when_set(monkeypatch):
    sent = []

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()

    def fake_urlopen(req, timeout):
        sent.append(json.loads(req.data))
        return Resp()

    monkeypatch.setattr("agent_reliability_lab.models.lmstudio.request.urlopen", fake_urlopen)
    LMStudioDecisionModel("m")._chat([])
    LMStudioDecisionModel("m", reasoning_effort="none", max_tokens=1024)._chat([])
    assert "reasoning_effort" not in sent[0] and sent[0]["max_tokens"] == 1024
    assert sent[1]["reasoning_effort"] == "none" and sent[1]["max_tokens"] == 1024


def test_structured_tool_calls_with_empty_content_become_a_lookup_step(monkeypatch):
    reply = {"choices": [{"message": {"content": "", "tool_calls": [
        {"function": {"name": "lookup_po", "arguments": "{\"po_id\": \"PO-1\"}"}}]}}], "usage": {}}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(reply).encode()

    monkeypatch.setattr("agent_reliability_lab.models.lmstudio.request.urlopen", lambda *a, **k: Resp())
    d = LMStudioDecisionModel("stub").decide({}, {})
    assert d.is_final is False and d.action == "HUMAN_REVIEW" and d.tool_calls[0].arguments == {"po_id": "PO-1"}
