from agent_reliability_lab.domains.invoice.app import build_invoice_runtime
from agent_reliability_lab.evals.runner import load_jsonl, run_cases

def test_invoice_eval_suite():
    report = run_cases(build_invoice_runtime(), load_jsonl("data/invoice_cases.jsonl"))
    assert report["summary"]["accuracy"] == 1.0
    assert report["summary"]["n"] == 20

def test_unknown_tool_is_blocked():
    runtime = build_invoice_runtime()
    result = runtime.tools.call("wire_money", {"amount": 1000})
    assert result.ok is False
    assert result.error == "tool_not_allowlisted"

def test_bad_tool_arguments_are_blocked():
    runtime = build_invoice_runtime()
    result = runtime.tools.call("lookup_purchase_order", {"wrong": "PO-1001"})
    assert result.ok is False
    assert result.error.startswith("invalid_arguments")

def test_planner_failure_fails_closed():
    runtime = build_invoice_runtime()
    runtime.planner = lambda task, tools: (_ for _ in ()).throw(ValueError("bad model output"))
    out = runtime.run({"invoice_id": "X"})
    assert out["status"] == "HUMAN_REVIEW"
    assert out["reason"] == "planner_failure"

def test_trace_includes_tool_calls():
    runtime = build_invoice_runtime()
    out = runtime.run({"invoice_id":"INV","po_id":"PO-1001","supplier":"Apex Components","invoice_amount":10000})
    kinds = [e["kind"] for e in out["trace"]["events"]]
    assert kinds.count("tool_call") == 3
    assert kinds.count("tool_result") == 3
