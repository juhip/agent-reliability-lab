from pprint import pprint
from agent_reliability_lab.domains.invoice.app import build_invoice_runtime
from agent_reliability_lab.evals.runner import load_jsonl, run_cases

if __name__ == "__main__":
    report = run_cases(build_invoice_runtime(), load_jsonl("data/invoice_cases.jsonl"))
    pprint(report["summary"])
