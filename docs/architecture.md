# Architecture

Agent Reliability Lab separates model judgment from system enforcement.

```text
Task
  |
  v
Planner / model
  |
  v
Policy engine -----> human review
  |
  v
Allowlisted tool registry
  |
  v
Deterministic tools / APIs
  |
  v
Final state + trace
  |
  v
Evaluation harness
```

## Design principles

1. **The model proposes; the runtime executes.** Model output never directly invokes arbitrary code.
2. **Invariants live outside the model.** Authorization, tool allowlists, required approvals, and deterministic calculations are enforced in code.
3. **Failures become test cases.** Traces are intended to feed a regression loop rather than exist only for debugging.
4. **Domains are adapters.** A new workflow should require new tools, planner logic, and eval cases—not a new runtime.

## Reliability dimensions

- end-to-end task success
- tool selection and argument validity
- policy compliance
- escalation behavior
- trajectory quality
- latency / resource use (future)
