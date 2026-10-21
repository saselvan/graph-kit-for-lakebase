"""Seam-1 parity gate: the real workspace-notebook runner vs the laptop runner.

Uploads a freshly built wheel of this toolkit plus notebooks/lgk_runner.py to a per-run
workspace dir (/Workspace/Users/<me>/lgk_v1_<6hex>/), submits it as a serverless notebook task
(no cluster), downloads the run folder and compares it with a laptop run of the same golden.
Touches no Lakebase branch and no UC object. The workspace dir is deleted in `finally`.
"""

import json
import secrets
import subprocess
from pathlib import Path

import pytest

from lgk.notebook_runner import DatabricksNotebookBackend, NotebookRunner
from lgk.parity import diff_run_folders
from lgk.runner import LaptopRunner, RunStatus
from lgk.runtime import ENVIRONMENT_VERSION
from tests import parity_support as ps

pytestmark = pytest.mark.seam1


def _build_wheel(out_dir: Path) -> Path:
    subprocess.run(
        ["uv", "build", "--wheel", "-q", "-o", str(out_dir)],
        cwd=str(ps.REPO_ROOT),
        check=True,
        timeout=300,
    )
    (wheel,) = out_dir.glob("lgk-*.whl")
    return wheel


def _live_config(run_dir, project, catalog, me):
    """The golden config with preflight checking for real: no scripted environment, the
    real project and catalog, the operator as the sync owner. Both runners must pass it."""
    cfg = dict(ps.golden_config(run_dir), lakebase_project=project, lakebase_catalog=catalog,
               sync_owner_user=me)
    del cfg["preflight_env"]
    return cfg


def test_live_notebook_runner_matches_laptop_runner(workspace_client, lakebase_project, lakebase_catalog,
                                                    tmp_path):
    w = workspace_client
    me = w.current_user.me().user_name
    workdir = f"/Workspace/Users/{me}/lgk_v1_{secrets.token_hex(3)}"
    wheel = _build_wheel(tmp_path / "dist")

    lap = ps.seed_full_golden(tmp_path / "laptop")
    LaptopRunner(_live_config(lap, lakebase_project, lakebase_catalog, me)).run_steps(ps.FULL_GOLDEN_STEPS)

    nb = ps.seed_full_golden(tmp_path / "nb")
    backend = DatabricksNotebookBackend(w, workdir, wheel)  # the packaged runner, as a workspace host uses it
    try:
        backend.setup()
        results = NotebookRunner(_live_config(nb, lakebase_project, lakebase_catalog, me), backend).run_steps(ps.FULL_GOLDEN_STEPS)
        assert backend.last_run_id is not None  # a real job run happened
    finally:
        backend.cleanup()

    assert [r.status for r in results] == [RunStatus.SUCCESS] * 3, results
    assert {p.name for p in nb.iterdir()} == ps.FULL_GOLDEN_FILES | {"runtime.json"}
    # P2-runtime (B, D): the run used the pinned environment, so it ran Python 3.12.
    runtime = json.loads((nb / "runtime.json").read_text())
    print(f"runtime.json python_version: {runtime['python_version']!r} "
          f"environment_version: {runtime['environment_version']!r}")
    assert runtime["python_version"].startswith("3.12"), runtime
    assert runtime["environment_version_requested"] == ENVIRONMENT_VERSION
    assert (runtime["environment_version"] or "").startswith(f"client.{ENVIRONMENT_VERSION}."), runtime
    assert runtime["steps"] == [name for name, _ in ps.FULL_GOLDEN_STEPS]
    assert json.loads((nb / "flags.json").read_text()) == {"shape": "tree"}
    # Timing fields were written by the remote run (different clock) and differ raw.
    assert (lap / "run.json").read_bytes() != (nb / "run.json").read_bytes()
    assert diff_run_folders(lap, nb) == []

    # Cleanup really removed the workspace dir.
    from databricks.sdk.errors import NotFound

    with pytest.raises(NotFound):
        w.workspace.get_status(workdir)
