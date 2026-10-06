# Model-orchestrated mode

The pipeline in `runtime.py` has code decide the sequence. This mode flips that: **the model chooses every
step, code owns authority.** It is the same shape as Claude Code or Cowork, with a permission system you can test.

```text
Goal -> model picks a tool -> runtime validates the call -> permission tier -> run -> result back to the model
                                                              |
              READ   free (lookups)                           +-> SPEND tier asks an approver that sees the
              WRITE  drafts and escalations, logged               whole trajectory and the evidence in it
              SPEND  moves money; runs only if the evidence supports it
```

## Pieces (`src/agent_reliability_lab/agentic/`)

| File | Role |
|---|---|
| `orchestrator.py` | the loop: validate, gate, execute, feed back; step and token budgets; `delegate` sub-agents |
| `types.py` | tools with a tier, a JSON-schema argument check, provider-neutral model turns |
| `trajectory.py` | every call, refusal and sub-agent, with per-run metrics |
| `models.py` | `ScriptedModel`, `OpenAIChatModel` (LM Studio, vLLM, ...), `AnthropicChatModel` |
| `invoice_queue.py` | the domain: eight tools, the approver, and two deterministic stand-in models |
| `evals/orchestration.py` | runs the same cases per invoice or as one queue with sub-agents, and scores behaviour |

## Design decisions

- **Authority is code, not prompt.** `approve_invoice` is refused unless the trajectory contains a verified
  match for *that* invoice. Evidence about another invoice or another PO does not count (two tests pin this).
- **Free-text fields are data.** An invoice whose note reads like an instruction cannot be approved, whatever the model did.
- **Failures are observations.** Unknown tools, bad arguments, refused approvals and tool errors go back to the
  model as results it can react to. A transport failure ends the run with `stop_reason=error`; it never raises.
- **Sub-agents keep context small.** A sub-agent gets only the tools the parent names, cannot delegate again,
  and returns a summary.
- **Spend is opt-in.** `--model claude` refuses to start without `--confirm-spend` and a call ceiling.

## Run

```bash
python run_orchestrator.py                          # stand-in models, free
python run_orchestrator.py --mode queue             # one parent delegating each invoice to a sub-agent
python run_orchestrator.py --model lmstudio --name lfm2.5-2.6b
python run_orchestrator.py --model claude --confirm-spend --max-calls 800
```

## What the numbers mean today

| row | accuracy | approve recall | unsafe attempts | unsafe executed |
|---|---|---|---|---|
| scripted pipeline | 1.00 | 1.00 | - | - |
| orchestrated, careful stand-in | 1.00 | 1.00 | 0 | 0 |
| orchestrated, gullible stand-in | 0.72 | 0.00 | 18 | 0 |

The stand-ins are deterministic code pretending to be a model. They prove the **loop and the gate**:
a model that approves on sight is refused every time (18 attempts on invoices that should escalate, 0 executed).
They say nothing about how well any real model orchestrates. **No real model has been run yet**, and
`AnthropicChatModel` has not been exercised against the live API (it is tested against a fake client only).

## Known limits

- The per-invoice and queue modes use the same eight tools; there is no filesystem, shell or search tool yet.
- Context management is truncation of long results plus sub-agents. There is no summarization or compaction.
- Tool results are sequential; parallel tool calls are accepted from the model but run one after another.
