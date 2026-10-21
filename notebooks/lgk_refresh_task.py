# Databricks notebook source
# lgk nightly refresh task. Each notebook task of the job made by
# lgk.refresh.job.create_refresh_job runs this with widgets task / config / wheel. It installs the
# toolkit wheel and runs one task (enrich, sync or reapply). No task logic lives here; an
# exception fails the task, and so the job.
import sys

MIN_PYTHON = (3, 12)


def require_python(version_info):
    """Stop with the fix when this Python is older than the kit needs (before anything installs)."""
    if tuple(version_info[:2]) < MIN_PYTHON:
        have = ".".join(str(p) for p in version_info[:3])
        raise RuntimeError(
            f"lgk needs Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}; this notebook runs Python {have}. "
            "Run it on serverless environment version 3 or later (Python 3.12): the job needs an "
            "environments entry with spec.environment_version '3' named by each task's environment_key "
            "(lgk.refresh.job.create_refresh_job does this)."
        )


require_python(sys.version_info)

dbutils.widgets.text("task", "")
dbutils.widgets.text("config", "{}")
dbutils.widgets.text("wheel", "")

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade "databricks-sdk>=0.81"

# COMMAND ----------

# The serverless image preloads an older databricks-sdk (no w.postgres: the sync and reapply tasks
# fail with "'WorkspaceClient' object has no attribute 'postgres'", live 2026-10-04). %pip installs
# into the notebook's own environment, which wins over the system site-packages after a restart.
dbutils.library.restartPython()

# COMMAND ----------

import subprocess
import sys

_wheel = dbutils.widgets.get("wheel")
if _wheel:
    _pip = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", _wheel],
        capture_output=True,
        text=True,
    )
    if _pip.returncode != 0:
        raise RuntimeError(f"pip install {_wheel} failed: {_pip.stderr[-3000:]}")

# COMMAND ----------

from lgk.refresh.tasks import run_task

dbutils.notebook.exit(run_task(dbutils.widgets.get("task"), dbutils.widgets.get("config")))
