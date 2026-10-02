# Architecture

Agent Reliability Lab separates model judgment from system enforcement.

```text
Task
  |
  v
Screen input (flag instruction-like text)
  |
  v
Planner / model <------------- tool results
  |                                 ^
  v                                 |
Policy: action allowed? --> human review
  |                                 |
  v                                 |
Allowlisted tool registry --> deterministic tools / APIs
  |
  v   (final decision)
Invariants: do the results support it? --> human review
  |
  v
Final state + trace
  |
  v
Evaluation harness
```

## Design principles

1. **The model proposes; the runtime executes.** Model output never directly invokes arbitrary code.
2. **Invariants live outside the model.** Authorization, tool allowlists, required approvals, and deterministic calculations are enforced in code, and a final decision is checked against the tool results, not against the planner's claims.
3. **Observe, then decide.** Planners can request tools, read the results, and plan again. A failed tool call is an observation the planner can react to; a one-shot planner that returns a final decision keeps the original behaviour.
4. **Failures become test cases.** Traces are intended to feed a regression loop rather than exist only for debugging.
5. **Domains are adapters.** A new workflow should require new tools, planner logic, invariants, and eval cases, not a new runtime.

## Escalation sources

Every `HUMAN_REVIEW` carries an `escalation_source`:

| source | meaning | counted as |
|---|---|---|
| `planner` | the planner decided the case needs a person | a normal outcome |
| `policy` | the action is not allowlisted or needs approval | an intervention |
| `policy_invariant` | the tool results did not support an `APPROVE`, or the input was flagged | an intervention |
| `planner_failure`, `invalid_planner_output` | the planner crashed or returned junk | a system error |
| `tool_failure` | a tool failed in one-shot mode | a system error |
| `max_steps` | the planner never reached a final decision | a system error |

System errors are never scored correct, even when they land on `HUMAN_REVIEW`.

## Reliability dimensions

- end-to-end task success
- planner accuracy versus final accuracy (how much the policy layer rescued)
- tool selection and argument validity
- policy compliance and injection resistance
- escalation behavior
- latency and token use (recorded by the LM Studio adapter)
- trajectory quality (future)
