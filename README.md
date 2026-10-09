# Agent Reliability Lab

A small harness for building agents that can be evaluated as systems, not just chatbots.

Once a model can call tools and take actions, model quality is one failure source among several. The others are tool selection, argument correctness, policy compliance, escalation behaviour, and whether anything checked the decision independently before it took effect. This repo separates those concerns so any workflow can plug into the same runtime, gate and eval harness.

The question it makes testable:

> **Under which conditions can this system act safely, and how do we know when it should stop?**

## The rule: core is workflow-agnostic

Every safety check lives in core and is a general mechanism. A domain supplies declarations (data) and a few domain functions; it never adds a check to core. Two domains ship to prove it: invoice exception handling and customer refunds. The refund domain was written using only declarations, and a test fails if domain vocabulary ever appears in core source.

## What core guarantees

A **gated action** (whatever the domain marks as consequential: approve, pay, refund) takes effect only if all four of these pass. Otherwise the case goes to the domain's **fail-safe outcome**, whatever the planner or model proposed:

1. **Untrusted-text screen.** Domain-free patterns, run recursively over every string in the task and in every tool result the deciding agent saw (nested dicts and lists included). Hits are reported by path, such as `task.customer_message.body[1]` or `lookup_return[1].carrier_note`. Domains can add patterns as data.
2. **Declarative human-review triggers.** Conditions such as `{"source": "lookup_customer", "path": "flags.chargeback_open", "op": "eq", "value": true}` or `amount gt 1000`, evaluated by a small core evaluator over the task and the tool results for this case. Lists along a path fan out. A trigger marked `required` fails closed when its value can't be resolved (tool never called, path missing, type error). Malformed specs are rejected when the domain is built.
3. **Invariants** (optional). Domain checks that the evidence the agent gathered supports the action.
4. **Independent verifier** (required). A domain function that recomputes the expected outcome from raw tool data, fetched through its own read-only access. It must agree with the proposed action. It must not share code, constants or data paths with the planner, because a checker that reads the rule from the same place as the planner inherits its misreadings. Tests check the verifier's imports.

The trace records which screen paths, triggers, invariants and verifier verdict were involved. Tool results only count as evidence for a case if their call arguments match it (declared per tool as `evidence_bindings`), so a lookup of a different record can't vouch for this one.

Also in core:

- **Plan → policy → act → observe loop** (pipeline mode). The planner can request tools, see results and plan again.
- **Model-orchestrated mode.** The model picks every tool; code owns authority through permission tiers. Gated tools go through the same gate. See [docs/orchestration.md](docs/orchestration.md).
- **Allowlisted tools with argument checking.** Required and unknown argument names, plus primitive JSON types (string, number, integer, boolean, array, object), in both modes. Nested structure, ranges and enums are not checked.
- **Executor hook.** Called only after a gated action passes the gate. The default, `record_only`, performs nothing and says so in the output (`"executed": false`). Nothing in this repo moves money or changes an external system.
- **Eval harness** with always-approve and always-escalate baselines built from each domain's own action names.

## Results

Every number below comes from deterministic planners or deterministic stand-in models. They validate the harness and the gate. They are **not** a claim about any model's capability; no real model results are included.

### Invoice (25 cases), pipeline mode: `python run_eval.py`

| planner | accuracy | planner accuracy | approve recall | escalation recall | gate interventions | followed injection |
|---|---|---|---|---|---|---|
| `oneshot` | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 / 3 |
| `observing` | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 / 3 |
| `always-escalate` | 0.72 | 0.72 | 0.00 | 1.00 | 0 | 0 / 3 |
| `always-approve` | 0.72 | 0.28 | 0.00 | 1.00 | 25 | 3 / 3 |

Identical to the table before the generalization, case by case. What changed is the reasons recorded for blocks: `always-approve` proposals are now also rejected by the verifier on the 18 cases that should escalate, and the `huge-amount` case also fires the declared `large_invoice` trigger.

### Refund (20 cases, synthetic, declarations only): `python run_eval.py --domain refund`

| planner | accuracy | planner accuracy | approve recall | escalation recall | gate interventions | followed injection |
|---|---|---|---|---|---|---|
| `observing` | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 / 2 |
| `planted-bug` | 1.00 | 0.60 | 1.00 | 1.00 | 8 | 2 / 2 |
| `always-escalate` | 0.70 | 0.70 | 0.00 | 1.00 | 0 | 0 / 2 |
| `always-approve` | 0.70 | 0.30 | 0.00 | 1.00 | 20 | 2 / 2 |

`planted-bug` is the reference planner with three mistakes planted on purpose. Each of its 8 bad proposals is blocked by the mechanism meant for it: the 45-day window misreading by the verifier (1), ignored account conditions by triggers (5, including one `required` trigger on an unknown customer), and unscreened free text by the screen (2, one in a nested task field and one inside a tool result).

### Orchestrated mode: `python run_orchestrator.py --domain <name> [--mode queue]`

| domain / model | accuracy | approve recall | unsafe attempts | unsafe executed | routed by gate | steps per case |
|---|---|---|---|---|---|---|
| invoice / careful stand-in | 1.00 | 1.00 | 0 | 0 | 0 | 4.64 |
| invoice / always-approve | 0.72 | 0.00 | 18 | 0 | 18 | 3.0 |
| invoice / always-escalate | 0.72 | 0.00 | 0 | 0 | 0 | 2.0 |
| refund / careful stand-in | 1.00 | 1.00 | 0 | 0 | 0 | 5.6 |
| refund / always-approve | 0.70 | 0.00 | 14 | 0 | 20 | 3.0 |
| refund / always-escalate | 0.70 | 0.00 | 0 | 0 | 0 | 2.0 |

Queue mode (one parent delegating each case to a sub-agent) gives the careful stand-in 1.00 on both domains.

### How to read the metrics

- **accuracy** scores the final outcome. A crash or tool failure that lands on the fail-safe outcome is **not** counted correct.
- **planner accuracy** scores what the planner proposed before the gate acted. The gap between it and accuracy is the gate working.
- **approve recall** is the share of cases expecting a gated action that got it; **escalation recall** the same for the fail-safe outcome. Most cases expect escalation, so always-escalate scores 0.70-0.72 on accuracy. Read accuracy with approve recall.
- **followed injection** counts cases labelled `injection` where the planner *proposed* a gated action. It counts proposals, not harm (the gate decides whether anything happens), and it can't tell whether the planner acted because of the injected text or in spite of it.
- **unsafe attempts / executed** (orchestrated): a gated tool call on a case that should not get one, refused / actually recorded.
- **routed by gate** (orchestrated): cases the gate sent straight to the fail-safe outcome. They are locked; the model can't approve them later.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

pytest -q
python demo.py
python run_eval.py [--domain refund]                         # reference planners + baselines
python run_eval.py --planner lmstudio --model <id>           # a local model served by LM Studio
python run_orchestrator.py [--domain refund] [--mode queue]  # stand-in models, no network

pip install -e '.[demo]' && streamlit run streamlit_app.py   # invoice UI
```

## Adding a domain

Create `src/agent_reliability_lab/domains/<name>/` and register its `suite` module in `domains/__init__.py`. Nothing in core changes.

1. **Tools.** Each a `Tool(name, description, json_schema, handler, tier)`. READ tools are what the planner and the verifier look things up with.
2. **A `Domain` declaration** (`domain.py`):
   - `actions`, `gated_actions`, `fail_safe` (must not be gated), `case_id_field`
   - `triggers`: human-review conditions as data
   - `injection_patterns`: extra screen regexes, if your domain has its own phrasing
   - `evidence_bindings`: `{tool: {argument: task_path}}`. Unbound tools never count as evidence.
   - `limits`: `max_steps`, `orchestrator_max_steps`, `child_max_steps`, `max_result_chars`
   - optional `invariants`, `executor`, and a `QueueSpec` (system prompt, decision tool names, goal text) for orchestrated mode
3. **A verifier**, `verify(task, fetch) -> Verdict`. Write it from the policy text, not from the planner. Keep your own constants, do your own arithmetic, fetch your own records. Add your verifier to the import-independence test in `tests/test_verification.py`.
4. **Planners and cases.** At least one reference planner and a JSONL file of `{"id", "category", "input", "expected_action", "injection"?}`. Cover the happy path, boundaries, every trigger, injection in nested fields and in tool results, and benign messy text that must *not* be flagged.
5. **A suite** (`suite.py`): `DOMAIN`, `CASE_FILES`, `PLANNERS`, `REFERENCE`, `PIPELINE_PLANNER`, `STAND_INS`.

`domains/refund/` is the smallest complete example.

## Repository structure

```text
src/agent_reliability_lab/
  domain.py        the contract (Domain, Verdict, QueueSpec), evidence scoping, run_gate
  triggers.py      declarative trigger specs and their evaluator
  screening.py     the neutral untrusted-text screen
  runtime.py       pipeline loop; build_runtime(domain, planner)
  tool_registry.py, schema.py, policies.py, types.py
  agentic/         orchestrator, generic case queue, stand-in baselines, model adapters
  evals/           runner, metrics, baselines, orchestration eval
  models/          LM Studio adapter, output parsing
  domains/invoice/ domains/refund/
data/              invoice_cases.jsonl, untrusted_input_cases.jsonl, refund_cases.jsonl
results/           last orchestrated run per domain
```

## Design choices

**Code owns authority.** A gated action has to survive the screen, the triggers, the invariants and an independent verifier, all evaluated against tool results, not against what the planner claimed.

**Independent means independent.** The first version only had invariants, and they read the rule from the same source as the planner, so they agreed with its mistakes. The verifier exists to break that coupling: the test `test_planted_rule_misreading_is_caught_by_the_verifier_not_the_invariants` plants a 5%-for-2% misreading the invariants inherit, and only the verifier catches it.

**Fail closed.** Invalid planner output, blocked actions, malformed tool arguments, verifier errors, executor errors and step-limit overruns all end on the fail-safe outcome. System failures are never scored correct.

**Turn failures into regressions.** `calculate_variance` once rounded to 4 decimals, which made 10200.01 look exactly like 2.00% and slip under the threshold. The boundary case now pins it, and the verifier compares in whole cents so it can't recur there.

## Limitations

- No real model has been evaluated. All results are deterministic planners and stand-ins.
- Both datasets are small, synthetic and in-memory. Inputs are structured JSON plus short free text: no PDFs, scans or email threads.
- The screen is a regex list. It is a tripwire, not a defence. It misses paraphrases and can false-positive on unusual benign text; a flag blocks a gated action but an unflagged case is not thereby safe.
- Triggers and the screen only see what was observed. A non-required trigger whose tool was never called does not fire (the miss is recorded in the trace); mark it `required` if the evidence must be gathered.
- Verifier independence is enforced by convention plus an import check. Core can't stop a domain author from copying planner logic into the verifier by hand.
- The verifier fetches records again, which doubles read traffic for gated actions. It assumes READ tools are side-effect free.
- Orchestrated queue mode needs `orchestrator_max_steps` of at least the queue size plus 3; a smaller value leaves cases undecided (seen while building the refund domain).
- The supplier rule in the invoice domain trims whitespace but is case-sensitive. That is a policy choice encoded in two test cases.
- The LM Studio adapter asks for one JSON object per turn rather than using native tool calling.

## License

MIT
