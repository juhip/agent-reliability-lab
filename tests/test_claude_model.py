"""ClaudeDecisionModel against a fake SDK client: no network, no spend."""
import json
from types import SimpleNamespace
from agent_reliability_lab.domains.invoice.app import build_invoice_runtime
from agent_reliability_lab.models.claude import ClaudeDecisionModel

CASE = {"invoice_id": "I", "po_id": "PO-1001", "supplier": "Northwind Components", "invoice_amount": 10000}


class FakeClient:
    """Returns scripted replies; records every request. `beta.messages` and `messages` share the script."""
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        stop = "refusal" if reply is None else "end_turn"
        content = [] if reply is None else [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=reply)]
        return SimpleNamespace(content=content, stop_reason=stop, stop_details=SimpleNamespace(category="cyber"),
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=200))


def test_parses_decision_and_sends_effort_and_fallbacks():
    client = FakeClient(['{"action": "HUMAN_REVIEW", "rationale": "unsure", "confidence": 0.4}'])
    model = ClaudeDecisionModel(client=client)
    assert model.decide({"po_id": "x"}, {}).action == "HUMAN_REVIEW"
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5" and req["output_config"] == {"effort": "medium"}
    assert req["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in req["betas"]
    assert "untrusted" in req["system"]
    assert model.stats["calls"] == 1 and model.stats["est_cost_usd"] == round((1000 * 4 + 200 * 20) / 1e6, 4)


def test_retries_once_on_unusable_output():
    client = FakeClient(["Sure, approving.", '{"action": "APPROVE", "rationale": "match", "confidence": 0.9}'])
    model = ClaudeDecisionModel(client=client, use_fallbacks=False)
    assert model.decide({"po_id": "x"}, {}).action == "APPROVE"
    assert model.stats["retries"] == 1 and client.requests[1]["messages"][1]["role"] == "assistant"
    assert "fallbacks" not in client.requests[0]


def test_refusal_makes_the_runtime_fail_safe():
    model = ClaudeDecisionModel(client=FakeClient([None]))
    out = build_invoice_runtime(model).run(dict(CASE))
    assert out["status"] == "HUMAN_REVIEW" and out["escalation_source"] == "planner_failure"
    assert model.stats["refusals"] == 1


def test_model_proposal_still_goes_through_the_gate():
    step1 = {"action": "HUMAN_REVIEW", "final": False, "tool_calls": [
        {"name": "lookup_purchase_order", "arguments": {"po_id": "PO-1001"}},
        {"name": "lookup_receipt", "arguments": {"po_id": "PO-1001"}},
        {"name": "calculate_variance", "arguments": {"invoice_amount": 10000.0, "expected_amount": 10000.0}}]}
    step2 = {"action": "APPROVE", "rationale": "match", "confidence": 0.9, "final": True}
    client = FakeClient([json.dumps(step1), json.dumps(step2)])
    out = build_invoice_runtime(ClaudeDecisionModel(client=client)).run(dict(CASE))
    assert out["status"] == "COMPLETED"
    assert json.loads(client.requests[1]["messages"][0]["content"])["observations"][0]["name"] == "lookup_purchase_order"


def test_run_eval_refuses_claude_without_confirm_spend(capsys):
    import run_eval
    assert run_eval.main(["--planner", "claude"]) == 2
    assert "--confirm-spend" in capsys.readouterr().out
