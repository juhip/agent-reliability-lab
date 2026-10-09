"""The verifier is the independent check: it must catch a planner that misreads a rule, even when the
invariants share the planner's reading of that rule."""
import ast
import pathlib
from agent_reliability_lab.domains.invoice import invariants as inv_mod, observing_planner as obs_mod, planner as plan_mod
from agent_reliability_lab.domains.invoice.app import build_invoice_runtime
from agent_reliability_lab.domains.invoice.observing_planner import ObservingInvoicePlanner

BASE = {"invoice_id": "T", "po_id": "PO-1001", "supplier": "Northwind Components"}
DOMAIN_DIR = pathlib.Path("src/agent_reliability_lab/domains")


def imported_modules(path: pathlib.Path):
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add(("." * node.level) + (node.module or ""))
            names.update(f"{'.' * node.level}{node.module or ''}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    return names


def assert_independent(verifier_path, forbidden):
    mods = imported_modules(verifier_path)
    leaks = [m for m in mods for f in forbidden if m.rstrip(".").split(".")[-1] == f or f".{f}." in f"{m}."]
    assert not leaks, f"{verifier_path} imports planner-side code: {leaks}"


def test_invoice_verifier_shares_no_code_with_the_planner_side():
    assert_independent(DOMAIN_DIR / "invoice/verifier.py",
                       ["planner", "observing_planner", "invariants", "queue", "standin", "tools"])


def test_planted_rule_misreading_is_caught_by_the_verifier_not_the_invariants(monkeypatch):
    # The planner reads the variance limit as 5% instead of 2%. The invariants read the limit from the
    # same place, so they inherit the misreading. Only the verifier, with its own copy, catches it.
    for mod in (plan_mod, obs_mod, inv_mod):
        monkeypatch.setattr(mod, "AUTO_APPROVE_VARIANCE", 0.05)
    out = build_invoice_runtime(ObservingInvoicePlanner()).run(dict(BASE, invoice_amount=10300))
    assert out["planner_action"] == "APPROVE"
    assert out["status"] == "HUMAN_REVIEW" and out["escalation_source"] == "gate"
    assert out["violations"] == ["verifier_disagrees:HUMAN_REVIEW"]
    gate = next(e for e in out["trace"]["events"] if e["kind"] == "gate")["payload"]
    assert gate["verifier"]["outcome"] == "HUMAN_REVIEW"
    assert [f["tool"] for f in gate["verifier"]["fetches"]] == ["lookup_purchase_order", "lookup_receipt"]


def test_verifier_agrees_with_every_reference_approval():
    out = build_invoice_runtime(ObservingInvoicePlanner()).run(dict(BASE, invoice_amount=10200))
    assert out["status"] == "COMPLETED" and out["outcome"] == "APPROVE"
    assert out["execution"] == {"executed": False, "note": "no executor configured; decision recorded only"}
