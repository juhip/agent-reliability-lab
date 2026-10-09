# Model-orchestrated mode

The pipeline in `runtime.py` has code decide the sequence. This mode flips that: **the model chooses every
step, code owns authority.** It is the same shape as Claude Code or Cowork, with a permission system you can test.

```text
Goal -> model picks a tool -> runtime validates the call -> permission tier -> run -> result back to the model
                                                              |
              READ   free (lookups)                           +-> SPEND tier asks an approver that sees the
              WRITE  drafts and escalations, logged               whole trajectory and the evidence in it
              SPEND  moves money; runs only if the gate passes (the same gate as the pipeline)
```

## Pieces (`src/agent_reliability_lab/agentic/`)

| File | Role |
|---|---|
| `orchestrator.py` | the loop: validate, gate, execute, feed back; step and token budgets; `delegate` sub-agents |
| `types.py` | tools with a tier, a JSON-schema argument check, provider-neutral model turns |
| `trajectory.py` | every call, refusal and sub-agent, with per-run metrics |
| `models.py` | `ScriptedModel`, `OpenAIChatModel` (LM Studio, vLLM, ...), `AnthropicChatModel` |
| `case_queue.py` | turns any domain into a queue: `list_queue`, `read_<case>`, the domain's tools, one decision tool per action, and an approver that runs `domain.run_gate` |
| `standins.py` | domain-free stand-in models: approve-on-sight, escalate-on-sight, and a queue parent |
| `evals/orchestration.py` | runs the same cases per case or as one queue with sub-agents, and scores behaviour |

Domain-specific pieces (system prompt, tool names, a "careful" stand-in that follows that
domain's rules) live under `domains/<name>/queue.py` and `standin.py`.

## Design decisions

- **Authority is code, not prompt.** A gated tool (`approve_invoice`, `issue_refund`) runs only if the gate passes:
  the screen, the declared triggers, the invariants and the independent verifier. Evidence counts only if its
  call arguments match the case (`evidence_bindings`), so another case's lookups cannot vouch for it.
- **Hard blocks route, soft blocks refuse.** If the screen, a trigger or the verifier blocks, the case is
  recorded as the domain's fail-safe outcome at once and locked. If only evidence is missing, the call is
  refused and the model may gather it and try again.
- **Free text is data.** The screen reads every string in the case and in every tool result the deciding
  agent saw, nested fields included.
- **Long results are cut for the model, not for the gate.** Results over the domain's `max_result_chars`
  (20,000 by default; it was a hard-coded 4,000) are truncated with an explicit note, counted in
  `truncated_results`, and kept whole in the trajectory, which is what the gate checks.
- **Failures are observations.** Unknown tools, bad arguments, refused approvals and tool errors go back to the
  model as results it can react to. A transport failure ends the run with `stop_reason=error`; it never raises.
- **Sub-agents keep context small.** A sub-agent gets only the tools the parent names, cannot delegate again,
  and returns a summary.
- **Spend is opt-in.** `--model claude` refuses to start without `--confirm-spend` and a call ceiling.

## Run

```bash
python run_orchestrator.py                          # invoice, stand-in models, free
python run_orchestrator.py --domain refund          # the second domain
python run_orchestrator.py --domain procurement     # the purchase-order agent (stand-ins, per case)
python run_orchestrator.py --mode queue             # one parent delegating each case to a sub-agent
python run_orchestrator.py --model lmstudio --name lfm2.5-2.6b
python run_orchestrator.py --model claude --confirm-spend --max-calls 800
```

## What the numbers mean today

| row (invoice, per case) | accuracy | approve recall | unsafe attempts | unsafe executed | routed by gate |
|---|---|---|---|---|---|
| scripted pipeline | 1.00 | 1.00 | - | - | - |
| orchestrated, careful stand-in | 1.00 | 1.00 | 0 | 0 | 0 |
| orchestrated, always-approve | 0.72 | 0.00 | 18 | 0 | 18 |
| orchestrated, always-escalate | 0.72 | 0.00 | 0 | 0 | 0 |

The refund domain gives the same shape (careful 1.00; always-approve 0.70 with 14 unsafe attempts,
0 executed, all 20 routed because its required account-status trigger was never resolved). So does
procurement: in the agent's least cautious release setting the careful stand-in follows the agent and
tries to release a hazmat order once; the gate refuses it and routes it to a person (0 executed).

The stand-ins are deterministic code pretending to be a model. They prove the **loop and the gate**:
a model that approves on sight is refused every time.
They say nothing about how well any real model orchestrates. **No real model has been run yet**, and
`AnthropicChatModel` has not been exercised against the live API (it is tested against a fake client only).

## Known limits

- The per-case and queue modes use the same tools; there is no filesystem, shell or search tool yet.
- The queue parent needs `orchestrator_max_steps` of at least the queue size plus 3.
- Context management is truncation of long results plus sub-agents. There is no summarization or compaction.
- Tool results are sequential; parallel tool calls are accepted from the model but run one after another.
