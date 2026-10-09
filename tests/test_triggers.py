import pytest
from agent_reliability_lab.triggers import compile_trigger, compile_triggers, evaluate, resolve
from agent_reliability_lab.types import ToolResult

ITEM = {"item": {"flags": {"restricted": True}, "lines": [{"sku": "a", "final": False}, {"sku": "b", "final": True}]},
        "status": "suspended", "amount": 60_000}


def fire(spec, task=None, results=()):
    fired, soft = evaluate([compile_trigger(spec)], task or {}, list(results))
    return [h.reason for h in fired], [h.reason for h in soft]


def out(output, name="lookup"):
    return ToolResult(name, True, output, arguments={})


def test_paths_resolve_nested_keys_list_fan_out_and_indexes():
    assert resolve(ITEM, ["item", "flags", "restricted"]) == [True]
    assert resolve(ITEM, ["item", "lines", "final"]) == [False, True]
    assert resolve(ITEM, ["item", "lines", "1", "sku"]) == ["b"]
    assert resolve(ITEM, ["item", "nope"]) == [] and resolve(ITEM, ["item", "lines", "9"]) == []


@pytest.mark.parametrize("spec", [
    {"path": "item.flags.restricted", "op": "eq", "value": True},
    {"path": "amount", "op": "gt", "value": 50_000},
    {"path": "amount", "op": "ge", "value": 60_000},
    {"path": "status", "op": "in", "value": ["suspended", "closed"]},
    {"path": "status", "op": "ne", "value": "active"},
    {"path": "status", "op": "contains", "value": "susp"},
    {"path": "item.lines.final", "op": "eq", "value": True},      # any element of a list
    {"path": "item.flags", "op": "exists"},
    {"path": "item.owner", "op": "missing"},
])
def test_each_operator_matches_on_tool_results(spec):
    assert fire({"name": "t", "source": "lookup", **spec}, results=[out(ITEM)]) == (["matched"], [])


def test_non_matching_values_do_not_fire():
    assert fire({"name": "t", "source": "lookup", "path": "amount", "op": "lt", "value": 10}, results=[out(ITEM)]) == ([], [])
    assert fire({"name": "t", "source": "lookup", "path": "status", "op": "not_in", "value": ["suspended"]},
                results=[out(ITEM)]) == ([], [])


def test_task_source_and_only_the_named_tool_is_read():
    spec = {"name": "big", "source": "task", "path": "amount", "op": "gt", "value": 50_000}
    assert fire(spec, task={"amount": 50_001}) == (["matched"], [])
    other = {"name": "t", "source": "lookup", "path": "amount", "op": "gt", "value": 1}
    assert fire(other, results=[out(ITEM, name="something_else")]) == ([], ["unresolved"])


def test_true_is_not_one():
    assert fire({"name": "t", "source": "task", "path": "f", "op": "eq", "value": True}, task={"f": 1}) == ([], [])


def test_unresolved_fails_closed_only_when_required():
    spec = {"name": "t", "source": "lookup", "path": "item.flags.restricted", "op": "eq", "value": True}
    assert fire(spec) == ([], ["unresolved"])                                  # tool never called
    assert fire({**spec, "required": True}) == (["unresolved"], [])
    assert fire({**spec, "required": True}, results=[out({"item": {}})]) == (["unresolved"], [])   # path missing
    assert fire({**spec, "required": True}, results=[ToolResult("lookup", False, None, "boom")]) == (["unresolved"], [])


def test_comparison_type_errors_are_unresolved_not_crashes():
    spec = {"name": "t", "source": "task", "path": "amount", "op": "gt", "value": 50_000}
    assert fire(spec, task={"amount": "ten thousand"}) == ([], ["unresolved"])
    assert fire({**spec, "required": True}, task={"amount": "ten thousand"}) == (["unresolved"], [])


@pytest.mark.parametrize("bad", [
    {"name": "t", "source": "task", "path": "a", "op": "approx", "value": 1},
    {"name": "t", "source": "task", "path": "", "op": "eq", "value": 1},
    {"source": "task", "path": "a", "op": "eq", "value": 1},
    {"name": "t", "source": "task", "path": "a", "op": "in", "value": "abc"},
    {"name": "t", "source": "task", "path": "a", "op": "gt"},
    {"name": "t", "source": "task", "path": "a", "op": "eq", "value": 1, "requird": True},
    {"name": "t", "source": "task", "path": "a", "op": "eq", "value": 1, "required": "yes"},
    "amount > 5",
])
def test_malformed_specs_are_rejected_when_the_domain_is_built(bad):
    with pytest.raises(ValueError):
        compile_trigger(bad)


def test_duplicate_names_rejected():
    spec = {"name": "t", "source": "task", "path": "a", "op": "exists"}
    with pytest.raises(ValueError):
        compile_triggers([spec, spec])
