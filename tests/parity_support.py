"""Shared helpers for the runner parity tests (laptop vs notebook runner, the timing allow-list).

Golden runs, a local-process stand-in for the Databricks workspace, and probe steps.

Golden fixtures (tests/fixtures/parity/):
- gold_tables.json
  # hostile: `audit_log` has no keys -> propose must not draft it as a node
  # hostile: `claims` has a composite key -> key order must survive serialization
  # hostile: `patients` profiles an MRN-shaped column (care-subject data must not leak)
- schema_fk.json (propose-only golden)
  # hostile: a cycle a->b->c->a -> shape must classify as "web" in both runners
  # hostile: `a` declares the same FK twice -> one edge, identically in both runners
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from lgk.steps.preflight import preflight
from lgk.steps.profile import profile
from lgk.steps.propose import propose

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "parity"
NOTEBOOK_SOURCE = REPO_ROOT / "notebooks" / "lgk_runner.py"

StepList = List[Tuple[str, Callable[[Dict[str, Any]], Any]]]

FULL_GOLDEN_STEPS: StepList = [("preflight", preflight), ("profile", profile), ("propose", propose)]
PROPOSE_GOLDEN_STEPS: StepList = [("propose", propose)]

# Exact outputs the full golden run must produce (so a vacuous "both empty" can't pass).
FULL_GOLDEN_FILES = {
    "run.json",
    "gold_tables.json",
    "preflight_result.json",
    "schema.json",
    "graph_mapping.yaml",
    "flags.json",
}


def golden_config(run_dir: Path) -> Dict[str, Any]:
    """JSON-only config for a golden run (no live clients)."""
    return {
        "graph_name": "parity_golden",
        "run_folder": str(run_dir),
        "lakebase_project": "parity-project",
        "branch_id": "production",  # named explicitly: there is no default branch (owner "branch a")
        "lakebase_catalog": "parity_catalog",
        "source_catalog": "parity_catalog",
        "source_schema": "parity_gold",
        "sync_owner_user": "operator@example.com",
        # Preflight's checks run against scripted services that pass (JSON-safe, so the same
        # factory runs through both runners); the live parity test drops this and checks for real.
        "preflight_env": "tests.preflight_support:golden_preflight_env",
    }


def seed_full_golden(run_dir: Path) -> Path:
    """Run folder seeded with gold_tables.json for preflight -> profile -> propose."""
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "gold_tables.json", run_dir / "gold_tables.json")
    return run_dir


def seed_propose_golden(run_dir: Path) -> Path:
    """Run folder seeded with an FK-rich schema.json for propose only."""
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "schema_fk.json", run_dir / "schema.json")
    return run_dir


# --------------------------------------------------------------------------- probe steps
def dump_config_step(config: Dict[str, Any]) -> None:
    """Probe: write the config the step sees (minus the run folder location) to a file."""
    run_dir = Path(config["run_folder"])
    seen = {k: v for k, v in config.items() if k != "run_folder"}
    (run_dir / "config_seen.json").write_text(json.dumps(seen, indent=2, sort_keys=True))


def runner_sniffing_step(config: Dict[str, Any]) -> None:
    """Hostile probe: a step with a runner-specific branch (what the parity rule forbids)."""
    run_dir = Path(config["run_folder"])
    where = "notebook" if os.environ.get("DATABRICKS_RUNTIME_VERSION") else "laptop"
    (run_dir / "flags.json").write_text(json.dumps({"shape": "tree", "where": where}, indent=2))


def failing_step(config: Dict[str, Any]) -> None:
    """Probe: a step that always raises."""
    raise ValueError("golden failure: schema not found")


# --------------------------------------------------------------- local workspace stand-in
_DRIVER = r"""
import json, os, sys
class _Widgets:
    def __init__(self, values): self._v = dict(values)
    def text(self, name, default, label=None): self._v.setdefault(name, default)
    def get(self, name): return self._v[name]
class _Exit(Exception):
    pass
class _Notebook:
    def exit(self, value):
        print("NOTEBOOK_EXIT:" + str(value)); raise _Exit()
class _Library:
    def restartPython(self): pass  # local stand-in: the process is already fresh
class _DbUtils:
    def __init__(self, values): self.widgets = _Widgets(values); self.notebook = _Notebook(); self.library = _Library()
src = open(sys.argv[1]).read()
dbutils = _DbUtils(json.loads(sys.argv[2]))
lines = [l for l in src.splitlines() if not l.startswith("# MAGIC")]
try:
    exec(compile("\n".join(lines), sys.argv[1], "exec"), {"dbutils": dbutils, "__name__": "__main__"})
except _Exit:
    pass
"""


class LocalProcessBackend:
    """Stand-in for the Databricks workspace: runs the committed notebook in a fresh process.

    The "workspace" is a local directory under `<root>/Workspace/Users/<user>/...`. The notebook source
    in notebooks/lgk_runner.py is executed with a dbutils stub and the env var the Databricks
    runtime sets, so a step that sniffs its environment behaves as it would in the workspace.
    """

    def __init__(self, root: Path):
        self.workspace_dir = root / "Workspace" / "Users" / "parity@example.com" / "lgk_v1_000000"
        self.remote_run_dir = str(self.workspace_dir / "run")
        self.submissions: List[Dict[str, str]] = []

    def push_run_folder(self, local_dir: Path) -> None:
        remote = Path(self.remote_run_dir)
        if remote.exists():
            shutil.rmtree(remote)
        shutil.copytree(local_dir, remote)

    def run_notebook(self, parameters: Dict[str, str]) -> None:
        self.submissions.append(dict(parameters))
        env = dict(os.environ)
        env["DATABRICKS_RUNTIME_VERSION"] = "client.parity-fake"
        env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT)])
        params = dict(parameters, wheel="")  # toolkit is importable from src/; nothing to install
        proc = subprocess.run(
            [sys.executable, "-c", _DRIVER, str(NOTEBOOK_SOURCE), json.dumps(params)],
            cwd=str(self.workspace_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"notebook process failed: {proc.stderr[-2000:]}")

    def pull_run_folder(self, local_dir: Path) -> None:
        remote = Path(self.remote_run_dir)
        for src in remote.rglob("*"):
            if src.is_file():
                dst = local_dir / src.relative_to(remote)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)
