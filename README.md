# Agent Reliability Lab

A small, reusable framework for building **agents that can be evaluated as systems, not just chatbots**.

Once an LLM can call tools and take actions, model quality is only one source of failure. Reliability also depends on tool selection, argument correctness, deterministic execution, policy compliance, escalation behavior, and end-to-end task success.

Agent Reliability Lab separates those concerns so a new domain can plug into the same runtime and evaluation harness.

## Why this exists

Most agent demos ask:

> **Can the model do the task?**

Production systems need a stricter question:

> **Under which conditions can this system act safely, and how do we know when it should stop?**

This repo makes that question testable.

## What it contains

- **Agent runtime** — a plan → policy → act → observe loop. A planner can request tools, see the results, and plan again; the first final decision ends the run.
- **Allowlisted tool registry** — unknown tools cannot execute
- **Argument validation** — malformed tool calls fail closed; in loop mode the planner sees the failure and can recover
- **Deterministic policy layer** — action allowlist plus *invariants* that check a proposed `APPROVE` against what the tools actually returned, not against what the planner claimed
- **Untrusted-input screening** — free-text fields (notes, email bodies) are treated as data; text that reads like instructions blocks auto-approval
- **Trace capture** — decisions, tool calls, tool results, policy checks, and final state are inspectable
- **Evaluation harness** — JSONL cases across happy path, boundary, edge, adversarial (including prompt injection), messy-input, and invalid-input categories
- **Baselines** — always-escalate and always-approve planners, so the metrics have to separate them from a real agent
- **Reference implementation** — invoice-exception handling for a fictional automotive supplier
- **Local-model adapter** — OpenAI-compatible LM Studio client with tolerant JSON parsing, one retry, and latency / token accounting

## Architecture

```text
Task ──► screen input (flag instruction-like text)
  │
  ▼
Planner / model ◄─────────────── tool results (observations)
  │  proposes decision (+ tool calls)          ▲
  ▼                                            │
Policy: is the action allowed? ──► HUMAN_REVIEW│
  │                                            │
  ▼                                            │
Allowlisted tool registry ──► deterministic tools
  │
  ▼  (final decision)
Invariants: do the tool results support it? ──► HUMAN_REVIEW
  │
  ▼
Final state + execution trace ──► evaluation harness
```

More detail: [docs/architecture.md](docs/architecture.md)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

pytest -q
python demo.py
python run_eval.py                                  # reference planners + baselines
python run_eval.py --planner lmstudio --model <id>  # a local model served by LM Studio
```

For the Streamlit UI:

```bash
pip install -e '.[demo]'
streamlit run streamlit_app.py
```

## Current results

25 synthetic cases: 20 in `data/invoice_cases.jsonl` and 5 in `data/untrusted_input_cases.jsonl` (three prompt-injection cases, two messy-input cases).

| planner | accuracy | planner accuracy | approve recall | policy interventions | followed injection |
|---|---|---|---|---|---|
| `oneshot` (deterministic) | 1.00 | 1.00 | 1.00 | 0 | 0 / 3 |
| `observing` (deterministic, loop) | 1.00 | 1.00 | 1.00 | 0 | 0 / 3 |
| `always-escalate` | 0.72 | 0.72 | 0.00 | 0 | 0 / 3 |
| `always-approve` | 0.72 | 0.28 | 0.00 | 25 | 3 / 3 |

How to read it:

- *accuracy* scores the final outcome. A crash or tool failure that happens to land on `HUMAN_REVIEW` is **not** counted correct.
- *planner accuracy* scores what the planner proposed before the policy layer acted. `always-approve` is right only 28% of the time, but the invariants rescue it to 72%. The gap is the safety net working.
- 18 of 25 cases expect escalation, so a trivial always-escalate planner scores 0.72. Accuracy alone is therefore not enough; use approve recall alongside it.

These numbers validate the harness and the deterministic reference planners. They are **not** a claim about any model's capability. No LM Studio results are included yet.

## Reference domain: invoice exception handling

The reference agent evaluates invoices against a purchase order, receipt status, supplier identity, and an auto-approval variance policy (±2%).

Two deterministic planners ship so the harness runs without external services:

- `InvoicePlanner` is a one-shot oracle that reads module data directly.
- `ObservingInvoicePlanner` decides only from tool results, which is the shape a model-backed planner has to follow.

A local or hosted model can replace either planner (`LMStudioDecisionModel`) while keeping the same tool allowlist, policy, and eval cases.

## Adapt it to another domain

Implement three things:

1. **Tools** — the read/action primitives available to the agent.
2. **Planner** — a deterministic policy or model that proposes an action and tool calls.
3. **Cases and invariants** — a JSONL eval suite with expected outcomes, and the checks that must hold before an action is accepted.

The runtime, policy enforcement, tracing, execution, and reporting remain reusable.

## Repository structure

```text
src/agent_reliability_lab/
  runtime.py
  tool_registry.py
  policies.py
  types.py
  evals/        runner, metrics, baselines
  models/       LM Studio adapter, output parsing
  domains/invoice/
data/           invoice_cases.jsonl, untrusted_input_cases.jsonl
tests/
docs/
run_eval.py
```

## Design choices

**Let code own invariants.** Authorization, tool allowlists, and hard policy constraints do not depend on probabilistic model behavior. An `APPROVE` must be backed by tool results (a verified three-way match and a variance computed from the real invoice amount), whatever the planner says.

**Fail closed.** Invalid planner output, blocked actions, malformed tool arguments, and step-limit overruns route to human review rather than silently continuing.

**Score the path, not just the answer.** The runner records who escalated (planner, policy, invariant, or a system failure) and counts system failures as wrong even when they land on the safe action.

**Turn failures into regressions.** The long-term loop is: production trace → failure taxonomy → new eval case → fix → regression test. One example already happened here: `calculate_variance` rounded to 4 decimals, which made 10200.01 look exactly like 2.00% and slip under the threshold. It surfaced while building the observing planner, and the boundary test now covers it.

## Roadmap

- real-model results table (LFM2.5 on LM Studio, one frontier model) with latency and memory
- rebalance cases so approvals are not a minority class
- trajectory-level scoring for tool order, retries, and redundant calls
- richer type validation for model-generated arguments
- semantic graders only where deterministic checks are insufficient
- automatic eval generation from reviewed production failures

## Limitations

- Both bundled planners are deterministic and intentionally simple; no model has been evaluated yet.
- The invoice dataset is synthetic, in-memory, and small (three POs).
- Inputs are structured JSON plus free-text notes. There are no PDFs, scans, or email threads, so no step in the reference app requires language understanding yet.
- Injection screening is a regex list. It is a tripwire, not a defence; the invariants are what hold the line.
- The supplier rule trims whitespace but is case-sensitive. That is a policy choice, encoded in two test cases, not a discovery.
- The LM Studio adapter asks for one JSON object per turn rather than using native tool-calling.

## License

MIT
