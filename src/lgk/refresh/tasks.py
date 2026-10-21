"""Entry the refresh job's notebook calls: run one task by name, raise on failure.

Deliberately not `lgk.runner.execute_steps`: that skips a step already marked succeeded in
run.json, and a nightly refresh must run every task every night.
"""

import json
from typing import Any, Callable, Dict


def _task_functions() -> Dict[str, Callable[[Dict[str, Any]], Any]]:
    from lgk.refresh.operator_stamp import stamp_operator_tables
    from lgk.refresh.sync import sync
    from lgk.steps.build import build
    from lgk.steps.enrich import enrich
    from lgk.steps.reapply import reapply

    return {"build": build, "stamp": stamp_operator_tables, "enrich": enrich, "sync": sync, "reapply": reapply}


def run_task(task: str, config_json: str) -> str:
    """Run refresh task `task` with the JSON config; any exception fails the job task."""
    functions = _task_functions()
    if task not in functions:
        raise ValueError(f"unknown refresh task {task!r}; expected one of {sorted(functions)}")
    functions[task](json.loads(config_json))
    return json.dumps({"task": task, "status": "success"})
