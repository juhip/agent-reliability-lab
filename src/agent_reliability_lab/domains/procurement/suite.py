"""What run_eval.py and run_orchestrator.py need to run the procurement domain.

Unlike invoice and refund, the procurement cases are generated at run time from the agent's golden
set (into a temp dir), and the Domain is bound to that run's scenarios and rulebook. So this suite
exposes `main(argv)` instead of DOMAIN / CASE_FILES, and the runners hand over to it."""
from .evaluate import main  # noqa: F401
