"""Model-orchestrated invoice queue vs the scripted pipeline.

python run_orchestrator.py                         # stand-in models only (free, no network)
python run_orchestrator.py --model lmstudio --name lfm2.5-2.6b
python run_orchestrator.py --model claude --confirm-spend --max-calls 400   # real API; costs money
"""
import argparse
import json
import sys
from pathlib import Path
from agent_reliability_lab.agentic import invoice_queue as iq
from agent_reliability_lab.agentic.models import AnthropicChatModel, OpenAIChatModel
from agent_reliability_lab.domains.invoice.app import build_invoice_runtime
from agent_reliability_lab.domains.invoice.observing_planner import ObservingInvoicePlanner
from agent_reliability_lab.evals.orchestration import run_orchestrated
from agent_reliability_lab.evals.runner import load_jsonl, run_cases

DATA = ["data/invoice_cases.jsonl", "data/untrusted_input_cases.jsonl"]
COLS = ["accuracy", "approve_recall", "escalation_recall", "no_decision", "unsafe_attempts", "unsafe_executed",
        "steps_per_invoice", "redundant_calls", "tool_errors", "denied", "input_tokens", "output_tokens"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["standins", "lmstudio", "claude"], default="standins")
    ap.add_argument("--name", default="local-model", help="model id for lmstudio / claude")
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--mode", choices=["per-invoice", "queue"], default="per-invoice")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--confirm-spend", action="store_true", help="required for --model claude (real API calls)")
    ap.add_argument("--max-calls", type=int, default=400, help="refuse to start if worst-case model calls exceed this")
    ap.add_argument("--out", default="results/orchestration_last.json")
    a = ap.parse_args(argv)

    cases = [c for p in DATA for c in load_jsonl(p)]
    rows = {}
    pipe = run_cases(build_invoice_runtime(ObservingInvoicePlanner()), cases)["summary"]
    rows["scripted-pipeline"] = {"accuracy": pipe["accuracy"], "approve_recall": pipe["approve_recall"],
                                 "escalation_recall": pipe["escalation_recall"], "steps_per_invoice": "-"}
    runs = {}
    if a.model == "standins":
        stand_ins = [("orchestrated:careful", iq.careful_model)]
        if a.mode == "per-invoice":       # the gullible stand-in only knows how to handle one invoice
            stand_ins.append(("orchestrated:gullible", iq.gullible_model))
        for name, fn in stand_ins:
            runs[name] = run_orchestrated(fn, cases, a.mode, a.max_steps)
    else:
        worst = len(cases) * a.max_steps
        if a.model == "claude":
            if not a.confirm_spend:
                print(f"Refusing: --model claude makes real API calls (worst case ~{worst} model calls). Add --confirm-spend.")
                return 2
            if worst > a.max_calls:
                print(f"Refusing: worst case {worst} calls exceeds --max-calls {a.max_calls}. Lower --max-steps or raise --max-calls.")
                return 2
            model = AnthropicChatModel(a.name if a.name != "local-model" else "claude-opus-5-5")
        else:
            model = OpenAIChatModel(a.name, a.base_url)
        runs[f"orchestrated:{a.model}"] = run_orchestrated(model, cases, a.mode, a.max_steps)
    for name, rep in runs.items():
        rows[name] = rep["summary"]

    print(f"{len(cases)} cases, mode={a.mode}\n")
    print(f"{'':<24}" + "".join(f"{c[:15]:>16}" for c in COLS))
    for name, s in rows.items():
        print(f"{name:<24}" + "".join(f"{str(s.get(c, '-')):>16}" for c in COLS))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": rows, "runs": {k: {"cases": v["cases"], "trajectories": v["trajectories"]}
                                                                  for k, v in runs.items()}}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
