"""What run_eval.py and run_orchestrator.py need to run the refund domain."""
from .domain import REFUND as DOMAIN
from .planner import ObservingRefundPlanner, PlantedBugRefundPlanner
from .standin import careful_model

CASE_FILES = ["data/refund_cases.jsonl"]
PLANNERS = {"observing": ObservingRefundPlanner, "planted-bug": PlantedBugRefundPlanner}
REFERENCE = {"observing", "planted-bug"}       # the planted bugs must all be caught: final accuracy stays 100%
PIPELINE_PLANNER = "observing"
STAND_INS = {"careful": careful_model}
