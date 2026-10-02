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

- **Agent runtime** — plan → policy check → tool execution → trace → output
- **Allowlisted tool registry** — unknown tools cannot execute
- **Argument validation** — malformed tool calls fail closed
- **Deterministic policy layer** — approval and safety constraints live outside the model
- **Trace capture** — decisions, tool calls, tool results, policy checks, and final state are inspectable
- **Evaluation harness** — JSONL cases across happy paths, boundary cases, edge cases, adversarial cases, and invalid inputs
- **Reference implementation** — invoice-exception handling for a fictional automotive supplier
- **Local-model adapter** — OpenAI-compatible LM Studio client for swapping in a local model

## Architecture

```text
User / task
   |
   v
Planner / model
   |
   v
Policy engine --------> HUMAN_REVIEW
   |
   v
Allowlisted tool registry
   |
   v
Deterministic tools / APIs
   |
   v
Final state + execution trace
   |
   v
Evaluation harness
   +--> task success
   +--> tool correctness
   +--> policy adherence
   +--> escalation behavior
   +--> failure taxonomy
```

More detail: [`docs/architecture.md`](docs/architecture.md)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

pytest -q
python demo.py
python run_eval.py
```

For the Streamlit UI:

```bash
pip install -e '.[demo]'
streamlit run streamlit_app.py
```

## Current results

The included regression suite contains **20 cases across five categories**:

- happy path
- boundary
- edge
- adversarial
- invalid input

Current reference implementation result:

```text
20 / 20 correct final actions
5 / 5 code tests passing
unknown tools blocked
malformed arguments blocked
planner failures fail closed to human review
```

These numbers validate the deterministic reference implementation and reliability scaffolding; they are **not a claim about frontier-model capability**.

## Reference domain: invoice exception handling

The reference agent evaluates invoices against a purchase order, receipt status, supplier identity, and an auto-approval variance policy.

The default planner is deliberately deterministic so the harness runs without external services. A local or hosted model can replace the planner while keeping the same execution contract and tool allowlist.

## Adapt it to another domain

Implement three things:

1. **Tools** — the read/action primitives available to the agent.
2. **Planner** — a deterministic policy or model that proposes an action and tool calls.
3. **Cases** — a JSONL eval suite with expected outcomes.

The runtime, policy enforcement, tracing, execution, and reporting remain reusable.

## Repository structure

```text
src/agent_reliability_lab/
  runtime.py
  tool_registry.py
  policies.py
  types.py
  evals/
  models/
  domains/invoice/

data/
  invoice_cases.jsonl

tests/
docs/
```

## Design choices

**Let code own invariants.** Arithmetic, authorization, tool allowlists, and hard policy constraints should not depend on probabilistic model behavior.

**Fail closed.** Invalid planner output, blocked actions, malformed tool arguments, and tool failures route to human review rather than silently continuing.

**Trace the trajectory.** A correct final answer can hide a bad execution path. The runtime records intermediate decisions and tool use so trajectory-level evaluation can be added.

**Turn failures into regressions.** The long-term loop is: production trace → failure taxonomy → new eval case → fix → regression test.

## Roadmap

- richer schema/type validation for model-generated arguments
- trajectory-level scoring for tool order, retries, and redundant calls
- latency and resource instrumentation
- semantic graders only where deterministic checks are insufficient
- automatic eval generation from reviewed production failures
- human-review feedback loop into regression datasets

## Limitations

- The bundled planner is deterministic and intentionally simple.
- The invoice dataset is synthetic and in-memory.
- The current evaluator emphasizes final-state correctness and basic trajectory statistics.
- The LM Studio adapter assumes structured output and needs stronger production-grade parsing/retry logic.

## License

MIT
