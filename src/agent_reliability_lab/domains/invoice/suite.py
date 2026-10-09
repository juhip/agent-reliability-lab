"""What run_eval.py and run_orchestrator.py need to run the invoice domain."""
from .domain import INVOICE as DOMAIN
from .observing_planner import ObservingInvoicePlanner
from .planner import InvoicePlanner
from .standin import careful_model

CASE_FILES = ["data/invoice_cases.jsonl", "data/untrusted_input_cases.jsonl"]
PLANNERS = {"oneshot": InvoicePlanner, "observing": ObservingInvoicePlanner}
REFERENCE = {"oneshot", "observing"}           # must reach 100% final accuracy, or CI fails
PIPELINE_PLANNER = "observing"                   # the scripted-pipeline row in run_orchestrator.py
STAND_INS = {"careful": careful_model}
