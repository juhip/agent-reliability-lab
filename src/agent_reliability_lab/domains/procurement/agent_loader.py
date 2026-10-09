"""Locate and import the purchase-order agent at runtime.

The agent is vendored at examples/procurement_agent (a fictional manufacturer, synthetic data). The
PROCUREMENT_AGENT_PATH environment variable points the harness at another checkout with the same
layout. Nothing is copied into the harness; if the folder is missing, `available()` is False and the
procurement tests skip.
"""
from __future__ import annotations
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

# repo root / examples / procurement_agent (this file is src/agent_reliability_lab/domains/procurement/)
DEFAULT_PATH = Path(__file__).resolve().parents[4] / "examples" / "procurement_agent"


class AgentUnavailable(RuntimeError):
    pass


def agent_root() -> Path:
    return Path(os.environ.get("PROCUREMENT_AGENT_PATH") or DEFAULT_PATH).expanduser().resolve()


def available() -> bool:
    root = agent_root()
    return (root / "procurement" / "pipeline.py").is_file() and (root / "rules" / "policy_rules.json").is_file()


_cache: SimpleNamespace | None = None


def load() -> SimpleNamespace:
    """Import the agent's modules (read-only use). Returns a namespace of the modules we wrap."""
    global _cache
    if _cache is not None:
        return _cache
    root = agent_root()
    if not available():
        raise AgentUnavailable(f"procurement agent not found at {root}; set PROCUREMENT_AGENT_PATH")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Never let an import of the agent drop .pyc files into its folder.
    prev, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        names = ["db", "rules", "part_labels", "classifier", "planner", "decider", "validator", "release",
                 "alerts", "explain", "exceptions", "pipeline", "writer"]
        mods = {n: importlib.import_module(f"procurement.{n}") for n in names}
        mods["golden_eval"] = _load_file("procurement_agent_golden_eval", root / "evals" / "golden" / "golden_eval.py")
    finally:
        sys.dont_write_bytecode = prev
    _cache = SimpleNamespace(root=root, rules_path=root / "rules" / "policy_rules.json",
                             golden_dir=root / "evals" / "golden", scenarios_dir=root / "data" / "scenarios", **mods)
    return _cache


def _load_file(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_misreadings() -> list:
    """The agent's own planted-misreading list (evals/golden/misreading_check.py), imported, not copied."""
    root = load().root
    prev, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        if str(root / "evals" / "golden") not in sys.path:
            sys.path.insert(0, str(root / "evals" / "golden"))
        mod = _load_file("procurement_agent_misreading_check", root / "evals" / "golden" / "misreading_check.py")
    finally:
        sys.dont_write_bytecode = prev
    return list(mod.MISREADINGS)
