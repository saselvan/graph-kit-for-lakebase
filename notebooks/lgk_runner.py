# Databricks notebook source
# lgk workspace-notebook runner. Submitted by lgk.notebook_runner.NotebookRunner as a
# serverless one-time run. It installs the toolkit wheel, then runs the requested steps with the
# same executor the laptop runner uses (lgk.runner.execute_steps). No step logic lives here.
import sys

MIN_PYTHON = (3, 12)


def require_python(version_info):
    """Stop with the fix when this Python is older than the kit needs (before anything installs)."""
    if tuple(version_info[:2]) < MIN_PYTHON:
        have = ".".join(str(p) for p in version_info[:3])
        raise RuntimeError(
            f"lgk needs Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}; this notebook runs Python {have}. "
            "Run it on serverless environment version 3 or later (Python 3.12): in a job, an "
            "environments entry with spec.environment_version '3' named by the task's environment_key "
            "(lgk.notebook_runner does this); interactively, Environment side panel -> Environment version 3."
        )


require_python(sys.version_info)

dbutils.widgets.text("wheel", "")
dbutils.widgets.text("steps", "[]")
dbutils.widgets.text("config", "{}")
dbutils.widgets.text("environment_version", "")

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade "databricks-sdk>=0.81"

# COMMAND ----------

# The serverless image preloads an older databricks-sdk (no w.postgres). %pip installs into the
# notebook's own environment, which wins over the system site-packages, but only after a restart.
dbutils.library.restartPython()

# COMMAND ----------

import subprocess
import sys

# The job environment (lgk.runtime, version 3 = Python 3.12) already lists the wheel; installing
# it here again keeps a run on a stale environment cache on the uploaded wheel.
_wheel = dbutils.widgets.get("wheel")
if _wheel:
    _pip = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", _wheel],
        capture_output=True,
        text=True,
    )
    if _pip.returncode != 0:
        raise RuntimeError(f"pip install {_wheel} failed: {_pip.stderr[-3000:]}")
    # The serverless image preloads an older databricks-sdk (no w.postgres). Upgrade it explicitly,
    # then drop the already-imported copy so the next import loads the upgraded one.
    _sdk = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "databricks-sdk>=0.81"],
        capture_output=True,
        text=True,
    )
    if _sdk.returncode != 0:
        raise RuntimeError(f"pip upgrade databricks-sdk failed: {_sdk.stderr[-3000:]}")
    import importlib

    for _name in [m for m in sys.modules if m == "databricks.sdk" or m.startswith("databricks.sdk.")]:
        del sys.modules[_name]
    importlib.invalidate_caches()
    import databricks.sdk as _sdk_mod

    if not hasattr(_sdk_mod.WorkspaceClient, "postgres"):
        raise RuntimeError(
            f"databricks-sdk at {_sdk_mod.__file__} has no postgres client after the upgrade"
        )

# COMMAND ----------

from lgk.notebook_runner import run_in_workspace

dbutils.notebook.exit(run_in_workspace(dbutils.widgets.get("steps"), dbutils.widgets.get("config"),
                                      dbutils.widgets.get("environment_version")))
