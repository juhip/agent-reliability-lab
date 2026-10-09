"""Run the eval suite for one domain's planners plus the two baselines, and print a comparison table.

python run_eval.py                                   # invoice: reference planners + baselines
python run_eval.py --domain refund
python run_eval.py --planner lmstudio --model <id> [--base-url http://localhost:1234/v1] [--reasoning-effort none]
python run_eval.py --planner claude --confirm-spend [--model claude-opus-5-5] [--effort medium]   # real API; costs money

Exits non-zero if a reference planner's final accuracy drops below 100%, so CI catches regressions.
"""
import argparse
import sys
from agent_reliability_lab.domains import SUITES, load_suite
from agent_reliability_lab.evals.baselines import baselines_for
from agent_reliability_lab.evals.runner import load_jsonl, run_cases
from agent_reliability_lab.runtime import build_runtime

COLUMNS = ["accuracy", "planner_accuracy", "approve_recall", "escalation_recall",
           "policy_interventions", "system_errors", "followed_injection", "mean_latency_s"]


def build_planner(suite, name, args):
    if name in suite.PLANNERS:
        return suite.PLANNERS[name]()
    baselines = baselines_for(suite.DOMAIN)
    if name in baselines:
        return baselines[name]
    if name == "claude":
        from agent_reliability_lab.models.claude import ClaudeDecisionModel
        return ClaudeDecisionModel(args.model if args.model != "local-model" else "claude-opus-5-5", effort=args.effort,
                                   actions=tuple(suite.DOMAIN.actions), fail_safe=suite.DOMAIN.fail_safe)
    from agent_reliability_lab.models.lmstudio import LMStudioDecisionModel
    return LMStudioDecisionModel(args.model, args.base_url, actions=tuple(suite.DOMAIN.actions),
                                 fail_safe=suite.DOMAIN.fail_safe, max_tokens=args.max_tokens, timeout=args.timeout,
                                 reasoning_effort=args.reasoning_effort, native_tool_calls=args.native_tool_calls)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=sorted(SUITES), default="invoice")
    ap.add_argument("--planner", help="a planner from the domain suite, always-escalate, always-approve, lmstudio, or claude")
    ap.add_argument("--model", default="local-model")
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--max-tokens", type=int, default=512, help="lmstudio: reply token cap")
    ap.add_argument("--timeout", type=float, default=120.0, help="lmstudio: seconds per model call")
    ap.add_argument("--native-tool-calls", action="store_true",
                    help="lmstudio: also accept the model's own tool-call format (LFM <|tool_call_start|>) for lookups")
    ap.add_argument("--reasoning-effort", help="lmstudio: sent as reasoning_effort for thinking models (none, low, ...)")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"],
                    help="Claude effort level")
    ap.add_argument("--confirm-spend", action="store_true", help="required for --planner claude (real API calls)")
    args = ap.parse_args(argv)

    suite = load_suite(args.domain)
    if args.planner == "claude" and not args.confirm_spend:
        print("Refusing: --planner claude makes real API calls (one or more per case). Add --confirm-spend.")
        return 2
    names = [args.planner] if args.planner else [*suite.PLANNERS, "always-escalate", "always-approve"]
    cases = [case for path in suite.CASE_FILES for case in load_jsonl(path)]
    reports = {}
    for name in names:
        planner = build_planner(suite, name, args)
        reports[name] = run_cases(build_runtime(suite.DOMAIN, planner), cases)
        if name in ("lmstudio", "claude"):
            reports[name]["model_stats"] = planner.stats

    print(f"domain={args.domain}, {len(cases)} cases\n")
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
        rescued = [r for r in rep["cases"] if r["correct"] and not r["planner_correct"]]
        if rescued and name not in ("always-escalate", "always-approve"):
            print(f"\n{name}: {len(rescued)} planner mistakes caught by the gate")
            for r in rescued:
                print(f"  {r['id']}: planner proposed {r['planner_action']}; blocked by {', '.join(r['violations'])}")
        if "model_stats" in reports[name]:
            print(f"\nmodel stats: {reports[name]['model_stats']}")

    failed = [n for n in names if n in suite.REFERENCE and reports[n]["summary"]["accuracy"] < 1.0]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
