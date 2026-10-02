"""Run the eval suite for one or more planners and print a comparison table.

python run_eval.py                              # reference planners + baselines
python run_eval.py --planner lmstudio --model <id> [--base-url http://localhost:1234/v1]

Exits non-zero if a reference planner drops below 100%, so CI catches regressions.
"""
import argparse
import sys
from agent_reliability_lab.domains.invoice.app import build_invoice_runtime
from agent_reliability_lab.domains.invoice.observing_planner import ObservingInvoicePlanner
from agent_reliability_lab.domains.invoice.planner import InvoicePlanner
from agent_reliability_lab.evals.baselines import AlwaysApprove, AlwaysEscalate
from agent_reliability_lab.evals.runner import load_jsonl, run_cases

DATA = ["data/invoice_cases.jsonl", "data/untrusted_input_cases.jsonl"]
REFERENCE = {"oneshot", "observing"}
COLUMNS = ["accuracy", "planner_accuracy", "approve_recall", "escalation_recall",
           "policy_interventions", "system_errors", "followed_injection", "mean_latency_s"]


def load_cases():
    return [case for path in DATA for case in load_jsonl(path)]


def build_planner(name, args):
    if name == "oneshot":
        return InvoicePlanner()
    if name == "observing":
        return ObservingInvoicePlanner()
    if name == "always-escalate":
        return AlwaysEscalate()
    if name == "always-approve":
        return AlwaysApprove()
    from agent_reliability_lab.models.lmstudio import LMStudioDecisionModel
    return LMStudioDecisionModel(args.model, args.base_url)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=["oneshot", "observing", "always-escalate", "always-approve", "lmstudio"])
    ap.add_argument("--model", default="local-model")
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    args = ap.parse_args(argv)

    names = [args.planner] if args.planner else ["oneshot", "observing", "always-escalate", "always-approve"]
    cases = load_cases()
    reports = {}
    for name in names:
        planner = build_planner(name, args)
        reports[name] = run_cases(build_invoice_runtime(planner), cases)
        if name == "lmstudio":
            reports[name]["model_stats"] = planner.stats

    print(f"{len(cases)} cases\n")
    print(f"{'planner':<16}" + "".join(f"{c[:14]:>16}" for c in COLUMNS))
    for name, rep in reports.items():
        s = rep["summary"]
        print(f"{name:<16}" + "".join(f"{str(s[c]):>16}" for c in COLUMNS))
    for name, rep in reports.items():
        misses = [r for r in rep["cases"] if not r["correct"]]
        if misses and name not in ("always-escalate", "always-approve"):
            print(f"\n{name}: {len(misses)} wrong")
            for r in misses:
                print(f"  {r['id']}: expected {r['expected']}, got {r['actual']} ({r['failure_type']}, {r['escalation_source']})")
        if "model_stats" in reports[name]:
            print(f"\nmodel stats: {reports[name]['model_stats']}")

    failed = [n for n in names if n in REFERENCE and reports[n]["summary"]["accuracy"] < 1.0]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
