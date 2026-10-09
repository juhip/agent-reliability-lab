"""Model-orchestrated queue vs the scripted pipeline, for any bundled domain.

python run_orchestrator.py                         # invoice, stand-in models only (free, no network)
python run_orchestrator.py --domain refund --mode queue
python run_orchestrator.py --model lmstudio --name lfm2.5-2.6b
python run_orchestrator.py --model claude --confirm-spend --max-calls 400   # real API; costs money
"""
import argparse
import json
import sys
from pathlib import Path
from agent_reliability_lab.agentic.models import AnthropicChatModel, OpenAIChatModel
from agent_reliability_lab.agentic.standins import approve_on_sight, escalate_on_sight
from agent_reliability_lab.domains import SUITES, load_suite
from agent_reliability_lab.evals.orchestration import run_orchestrated
from agent_reliability_lab.evals.runner import load_jsonl, run_cases
from agent_reliability_lab.runtime import build_runtime

COLS = ["accuracy", "approve_recall", "escalation_recall", "no_decision", "unsafe_attempts", "unsafe_executed",
        "routed_by_gate", "steps_per_case", "redundant_calls", "tool_errors", "denied", "input_tokens", "output_tokens"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=sorted(SUITES), default="invoice")
    ap.add_argument("--model", choices=["standins", "lmstudio", "claude"], default="standins")
    ap.add_argument("--name", default="local-model", help="model id for lmstudio / claude")
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--mode", choices=["per-case", "per-invoice", "queue"], default="per-case")
    ap.add_argument("--max-steps", type=int, default=None, help="default: the domain's orchestrator_max_steps")
    ap.add_argument("--confirm-spend", action="store_true", help="required for --model claude (real API calls)")
    ap.add_argument("--max-calls", type=int, default=400, help="refuse to start if worst-case model calls exceed this")
    ap.add_argument("--out", default=None, help="default: results/orchestration_<domain>.json")
    a = ap.parse_args(argv)

    suite = load_suite(a.domain)
    domain = suite.DOMAIN
    max_steps = a.max_steps or domain.limits["orchestrator_max_steps"]
    cases = [c for p in suite.CASE_FILES for c in load_jsonl(p)]
    rows = {}
    pipe = run_cases(build_runtime(domain, suite.PLANNERS[suite.PIPELINE_PLANNER]()), cases)["summary"]
    rows["scripted-pipeline"] = {"accuracy": pipe["accuracy"], "approve_recall": pipe["approve_recall"],
                                 "escalation_recall": pipe["escalation_recall"]}
    runs = {}
    if a.model == "standins":
        stand_ins = [(f"orchestrated:{k}", fn) for k, fn in suite.STAND_INS.items()]
        if a.mode != "queue":        # the baselines only know how to handle one case at a time
            stand_ins += [("orchestrated:always-approve", approve_on_sight(domain)),
                          ("orchestrated:always-escalate", escalate_on_sight(domain))]
        for name, fn in stand_ins:
            runs[name] = run_orchestrated(domain, fn, cases, a.mode, max_steps)
    else:
        worst = len(cases) * max_steps
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
        runs[f"orchestrated:{a.model}"] = run_orchestrated(domain, model, cases, a.mode, max_steps)
    for name, rep in runs.items():
        rows[name] = rep["summary"]

    print(f"domain={a.domain}, {len(cases)} cases, mode={a.mode}\n")
    print(f"{'':<30}" + "".join(f"{c[:15]:>16}" for c in COLS))
    for name, s in rows.items():
        print(f"{name:<30}" + "".join(f"{str(s.get(c, '-')):>16}" for c in COLS))
    out = Path(a.out or f"results/orchestration_{a.domain}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": rows, "runs": {k: {"cases": v["cases"], "trajectories": v["trajectories"]}
                                                         for k, v in runs.items()}}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
