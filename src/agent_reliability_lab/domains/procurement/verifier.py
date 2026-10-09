"""Independent check of a release. Written from the purchasing policy text below, not from the agent
or the planner: it imports nothing from the agent, session.py, tools.py, invariants.py, the planners
or the stand-in; it keeps its own constants; it prices every line itself in integer cents from the
raw catalog rows; and it fetches everything through `fetch`.

What the policy says, as this module reads it (the agent reads the same text into its rulebook; a
misreading there must not be repeated here):
  - Only suppliers on the Approved Supplier List may be used.
  - A supplier must sell the part, and an order may not be below the supplier's minimum order quantity.
  - An order whose value is above $40,000 needs Sourcing Manager approval (above $120,000, VP of
    Supply Chain). Either way a person approves it: it cannot be released automatically.
  - Air freight needs approval for every request.
  - Hazardous materials need a procurement review before ordering.
  - Certifications a part's own record lists must be held by the supplier.
  - A line the hard-rule gate blocked was not placed; a plan with blocked lines goes to a person.

Not checked here: whether the plan covers the whole shortfall (the invariants recompute that with the
agent's netting), supplier choice (price, domestic and sustainability preferences, memo splits: the
plan-content grader covers those), and the Strategic-supplier rule (it needs the full option set).
"""
from __future__ import annotations
from typing import Any, Callable, Dict

from agent_reliability_lab.domain import Verdict

RELEASE, REVIEW = "APPROVE", "HUMAN_REVIEW"
APPROVAL_ABOVE_CENTS = 40_000 * 100


def _cents(x: float) -> int:
    return int(round(float(x) * 100))


def _codes(value: Any) -> set:
    return {c.strip().upper() for c in str(value or "").replace(";", ",").split(",") if c.strip()}


def verify_release(task: Dict[str, Any], fetch: Callable[[str, Dict[str, Any]], Any]) -> Verdict:
    sc = task.get("scenario")
    if not isinstance(sc, str) or not sc.strip():
        return Verdict(REVIEW, "no scenario")
    sc = sc.strip()
    try:
        gate = fetch("check_hard_rules", {"scenario": sc})
        records = fetch("read_records", {"scenario": sc})
    except Exception as exc:
        return Verdict(REVIEW, f"plan or records unavailable: {exc}")
    if gate.get("blocked"):
        return Verdict(REVIEW, "the hard-rule gate blocked a line")
    parts = {r["component_id"]: r for r in records["components"]}
    suppliers = {r["supplier_id"]: r for r in records["suppliers"]}
    catalog = {(r["supplier_id"], r["component_id"]): r for r in records["catalog"]}
    for line in gate.get("planned_lines", []):
        cid, sid, qty, mode = line["component_id"], line["supplier_id"], int(line["quantity"]), line["mode"]
        part, sup, offer = parts.get(cid), suppliers.get(sid), catalog.get((sid, cid))
        if part is None or sup is None or offer is None:
            return Verdict(REVIEW, f"{cid} from {sid}: no such part, supplier or catalog entry")
        if mode != "standard":
            return Verdict(REVIEW, f"{cid}: air freight needs approval")
        if part["is_hazardous"]:
            return Verdict(REVIEW, f"{cid}: hazardous material needs a procurement review")
        if not sup["on_approved_list"]:
            return Verdict(REVIEW, f"{sid} is not on the Approved Supplier List")
        missing = _codes(part["requires_certification"]) - _codes(sup["certifications"])
        if missing:
            return Verdict(REVIEW, f"{sid} lacks {sorted(missing)} required by {cid}")
        if qty < int(offer["minimum_order_qty"] or 1):
            return Verdict(REVIEW, f"{cid}: {qty} is below {sid}'s minimum order")
        if qty * _cents(offer["unit_price"]) > APPROVAL_ABOVE_CENTS:
            return Verdict(REVIEW, f"{cid}: order value above $40,000 needs approval")
    return Verdict(RELEASE, "every line is releasable under the policy text")
