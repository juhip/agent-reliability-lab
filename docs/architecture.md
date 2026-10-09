# Architecture

Agent Reliability Lab separates model judgment from system enforcement, and separates the
enforcement mechanisms (core) from the workflow they enforce (a domain).

```text
Task
  |
  v
Planner / model <------------- tool results (allowlisted, argument-checked tools)
  |                                 ^
  v                                 |
Policy: action allowlisted? --------+--> fail-safe outcome
  |
  v   (final decision)
Gated action? --no--> outcome as proposed (escalations are never upgraded)
  |
  yes
  v
Gate (core, the same in pipeline and orchestrated mode)
  1. screen: instruction-like text anywhere in the task or any observed tool result
  2. triggers: declared human-review conditions over the task and in-scope tool results
  3. invariants: does the gathered evidence support the action? (optional)
  4. verifier: recompute the outcome from raw tool data; must agree
  |                    \
  pass                  any failure --> fail-safe outcome (+ which check, recorded in the trace)
  v
Executor hook (default record_only: performs nothing)
  |
  v
Final state + trace --> evaluation harness
```

## The domain contract

`src/agent_reliability_lab/domain.py` defines it. A domain provides:

| field | required | what it is |
|---|---|---|
| `actions`, `gated_actions`, `fail_safe` | yes | outcomes; which ones have consequences; where anything blocked goes (must not be gated) |
| `verifier(task, fetch)` | yes | recomputes the expected outcome from raw tool data; must not share code, constants or data paths with the planner |
| `tools` | yes | `Tool`s with JSON-schema arguments and a tier (READ / WRITE / SPEND) |
| `case_id_field` | yes in practice | the task field that identifies a case |
| `triggers` | no | declarative human-review conditions (`triggers.py` documents the spec) |
| `injection_patterns` | no | extra screen regexes on top of core's neutral ones |
| `invariants` | no | evidence-gathered checks over the in-scope tool results |
| `evidence_bindings` | no | `{tool: {arg: task_path}}`; only matching results count as evidence for a case |
| `executor` | no | called after a gated action passes; default `record_only` |
| `limits` | no | `max_steps`, `orchestrator_max_steps`, `child_max_steps`, `max_result_chars` |
| `queue` | for orchestrated mode | system prompt, decision tool names, goal text; presentation only |

Bad declarations fail when the domain is built: a gated fail-safe, a missing verifier, an
unknown trigger op or source, a binding for an unknown tool, an unknown limit, a regex that
does not compile.

What core guarantees for any domain:

- A gated action takes effect only if the screen, every trigger, every invariant and the verifier pass.
- Anything blocked, and any system failure, ends on the domain's `fail_safe`.
- The trace names the screen paths, triggers, invariant violations and verifier verdict involved.
- In orchestrated mode, a hard block (screen, trigger, verifier) records the fail-safe outcome
  immediately and locks the case. Missing evidence alone (an invariant) is a refusal the model
  can recover from by gathering it.

What core does **not** guarantee: that the verifier is actually independent (checked by
convention and an import test, not enforced), that unscreened text is safe, or that a trigger
whose evidence was never gathered fires (only `required` triggers fail closed).

### Why a verifier and not just invariants

Invariants check that the planner's evidence supports its action. When they read the rule
from the same place the planner does, a misread rule passes both. The verifier recomputes the
answer from the written policy with its own constants, its own arithmetic and its own reads,
so the two can only agree by both being right. `tests/test_verification.py` plants a
misreading that the invariants inherit and shows only the verifier catches it.

### Triggers vs verifier

The verifier answers "is this eligible?". Triggers answer "does a person have to look at this
regardless?" (an open chargeback, a restricted item, a large amount). Keeping them apart means
a planner that knows the eligibility rule but not the escalation policy is still caught, and
the escalation policy is data a reviewer can read without reading code.

## Escalation sources

Every fail-safe outcome carries an `escalation_source`:

| source | meaning | counted as |
|---|---|---|
| `planner` | the planner chose the fail-safe outcome | a normal outcome |
| `policy` | the action is not allowlisted | an intervention |
| `gate` | a gated action failed the screen, a trigger, an invariant or the verifier | an intervention |
| `planner_failure`, `invalid_planner_output` | the planner crashed or returned junk | a system error |
| `tool_failure` | a tool failed in one-shot mode | a system error |
| `max_steps` | the planner never reached a final decision | a system error |
| `executor_failure` | the executor raised after the gate passed | a system error |

System errors are never scored correct, even when they land on the fail-safe outcome.

## Design principles

1. **The model proposes; the runtime executes.** Model output never directly invokes arbitrary code.
2. **Core owns every check; domains own data and domain functions.** A new workflow should
   need tools, a declaration, a verifier, planners and cases, never a change to core.
3. **Observe, then decide.** Planners can request tools, read the results and plan again.
4. **Failures become test cases.** Traces are meant to feed a regression loop.
