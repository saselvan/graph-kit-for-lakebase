"""The nightly refresh job: one Lakeflow job, its tasks in a strict chain.

    [sdp_pipeline] -> [build | stamp] -> [enrich] -> sync -> reapply      (refresh_chain)

`build` is the kit's own SQL build (nodes, edges, node_id_map from the source tables): it runs
by default, so every graph is rebuilt nightly with no operator pipeline. An operator SDP
pipeline (`sdp_pipeline_id`) runs first when set; with it, the kit build runs only when the run
config says `build_mode: sql` (else the operator's pipeline may already rebuild the kit schema,
and a second build would race it). A chain with no kit build runs `stamp` instead
(lgk.refresh.operator_stamp, owner decision: the nightly stamp step): it checks the tables the
operator rebuilt (gate, floors, allowlist) and labels them, so the sync accepts them. `enrich` recomputes scores (or they go stale), `sync` runs
the synced tables' pipelines, and `reapply` (freshness guard, then indexes/views/grants) is the
next task in the SAME job, so a gap is seconds and any failure is a failed job. Tasks run
serverless (no cluster), as the job's owner, through notebooks/lgk_refresh_task.py, on the
pinned job environment (lgk.runtime: version 3 = Python 3.12; the default is Python 3.10).

The config travels to every task as JSON. Live clients (a WorkspaceClient, a connect function)
never do: job_config drops them, at any depth, before anything is written into the job. Its
run_folder is a workspace one: a laptop run folder is copied to the workspace first
(lgk.refresh.nightly_folder, which also states what each task reads and where state lives).
"""

import json
import logging
import os
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional, Tuple

from lgk.runtime import ENVIRONMENT_KEY, job_environment

logger = logging.getLogger(__name__)

TASK_ORDER = ("sdp_pipeline", "build", "stamp", "enrich", "sync", "reapply")
BUILD_MODES = ("sql",)  # the kit's SQL build is the only build mode


def _builds(config: Dict[str, Any]) -> bool:
    """Whether the kit's SQL build runs nightly; refuses an unknown build_mode."""
    mode = config.get("build_mode")
    if mode is not None and mode not in BUILD_MODES:
        raise ValueError(f"config['build_mode'] must be one of {list(BUILD_MODES)} or absent, got {mode!r}")
    return mode == "sql" or not config.get("sdp_pipeline_id")


def run_enriched(run_folder: Optional[str]) -> bool:
    """Whether the run enriched (an enrich.yaml in its run folder), so the nightly chain recomputes scores.

    The one test for both the workspace job (refresh_job_tasks) and the bundle's job (lgk.export):
    enrich stops with no enrich.yaml, so a chain that ran it there would fail every night.
    """
    return bool(run_folder) and (Path(run_folder) / "enrich.yaml").exists()


def refresh_chain(config: Dict[str, Any], enriched: bool) -> List[str]:
    """The nightly task keys, in order; each task depends on the one before it.

    The one source of the chain for the workspace job (refresh_job_tasks) and the exported
    bundle's job (lgk.export.refresh_tasks), so the two cannot drift.
    """
    builds = _builds(config)
    present = {"sdp_pipeline": bool(config.get("sdp_pipeline_id")), "build": builds, "stamp": not builds,
               "enrich": enriched, "sync": True, "reapply": True}
    return [key for key in TASK_ORDER if present[key]]


_JSON_SCALARS = (str, int, float, bool, type(None))


def _jsonable(value: Any, path: str, dropped: List[str]) -> Tuple[bool, Any]:
    """(keep, value) for one config value: JSON values kept, paths as strings, anything else dropped."""
    if isinstance(value, _JSON_SCALARS):
        return True, value
    if isinstance(value, PurePath):
        return True, os.fspath(value)
    if isinstance(value, dict):
        return True, _json_dict(value, path, dropped)
    if isinstance(value, (list, tuple)):
        kept = [_jsonable(v, f"{path}[{i}]", dropped) for i, v in enumerate(value)]
        return True, [v for keep, v in kept if keep]
    dropped.append(f"{path} ({type(value).__name__})")
    return False, None


def _json_dict(d: Dict[Any, Any], path: str, dropped: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in d.items():
        keep, v = _jsonable(value, f"{path}[{key!r}]", dropped)
        if keep and isinstance(key, str):
            out[key] = v
        elif keep:
            dropped.append(f"{path}[{key!r}] (non-string key)")
    return out


def job_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """The config as the job's tasks get it: JSON values only, so no live client is ever written.

    A WorkspaceClient (the laptop runner's `workspace_client`), a `pg_connect` function, or any
    other non-JSON object is dropped wherever it sits; each drop is logged by its key path.
    """
    dropped: List[str] = []
    out = _json_dict(config, "config", dropped)
    if dropped:
        logger.warning("refresh job config: dropped values that cannot travel as JSON: %s", ", ".join(dropped))
    return out


def job_name(config: Dict[str, Any]) -> str:
    return f"lgk_refresh_{config['kit_schema']}"


def _notebook_task(key: str, after: Optional[str], config: Dict[str, Any], notebook_path: str, wheel_path: str):
    from databricks.sdk.service import jobs

    return jobs.Task(
        task_key=key,
        depends_on=[jobs.TaskDependency(task_key=after)] if after else None,
        environment_key=ENVIRONMENT_KEY,
        notebook_task=jobs.NotebookTask(
            notebook_path=notebook_path,
            base_parameters={"task": key, "wheel": wheel_path, "config": json.dumps(config)},
            source=jobs.Source.WORKSPACE,
        ),
    )


def refresh_job_tasks(config: Dict[str, Any], notebook_path: str, wheel_path: str,
                      job_run_folder: Optional[str] = None) -> List[Any]:
    """The jobs.Task objects of refresh_chain, each depending on exactly the one before it.

    enrich is in the chain only when the run enriched (run_enriched on the config's own run
    folder, as the bundle decides). The tasks' config carries `job_run_folder` when given (the
    workspace copy), else the config's run_folder. The SDP pipeline is a pipeline task; the rest
    are notebook tasks."""
    from databricks.sdk.service import jobs

    keys = refresh_chain(config, enriched=run_enriched(config.get("run_folder")))
    config = job_config(config if job_run_folder is None else dict(config, run_folder=job_run_folder))
    tasks = []
    for i, key in enumerate(keys):
        if key == "sdp_pipeline":
            tasks.append(jobs.Task(task_key=key, pipeline_task=jobs.PipelineTask(
                pipeline_id=config["sdp_pipeline_id"])))
        else:
            tasks.append(_notebook_task(key, keys[i - 1] if i else None, config, notebook_path, wheel_path))
    return tasks


def record_job(config: Dict[str, Any], job_id: Any, kind: str, **metadata: Any) -> None:
    """Record a kit-made job in the run manifest, with the branch it syncs (for the reaper)."""
    from lgk import manifest as mf

    from lgk.branch import BranchRef

    branch = BranchRef.from_config(config).path
    mf.record(config, mf.ObjectType.JOB, str(job_id), "jobs", {"kind": kind, "branch": branch, **metadata})


def create_refresh_job(
    config: Dict[str, Any], notebook_path: str, wheel_path: str, w: Optional[Any] = None
) -> int:
    """Create the persistent refresh job, tagged with the graph, and record it for teardown.

    A laptop run folder is checked (every file the tasks read, else NightlyFolderError before
    any job exists), the job is pointed at its workspace copy (lgk.refresh.nightly_folder), the
    job is recorded in the laptop manifest (with the copy's folder), and then the files are
    copied, so the copy's manifest holds the job too. A /Workspace run_folder is used as is.
    """
    from lgk.branch import BranchRef
    from lgk.manifest import job_tags
    from lgk.refresh import nightly_folder as nf

    BranchRef.from_config(config)  # record_job needs it after jobs.create: refuse before, never orphan a job
    local = config.get("run_folder")
    seeded = not nf.is_workspace_path(local or "")
    names = nf.nightly_files(local) if seeded else []
    folder = nf.workspace_run_folder(config, notebook_path) if seeded else str(local)  # str: a Path is no JSON
    if w is None:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
    tasks = refresh_job_tasks(config, notebook_path, wheel_path, job_run_folder=folder)
    job_id = w.jobs.create(name=job_name(config), tasks=tasks, max_concurrent_runs=1,
                           environments=[job_environment([wheel_path])],
                           tags=job_tags(config["graph_name"])).job_id
    record_job(config, job_id, "refresh", run_folder=folder)
    if seeded:
        nf.upload_run_files(w, local, names, folder)
    return job_id
