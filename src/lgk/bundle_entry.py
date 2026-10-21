"""The exported bundle's job task: run `preflight`, `provision` or one refresh task, as the job's run_as.

The `preflight` task runs first in the provision job, as the SP, with `--sync-owner` = the SP
(the bundle variable): it records the SP as the sync-job owner that provision requires.

The `provision` task runs the whole serving flow: provision -> advise -> switch, so the
new version is proven and tuned before the served views move onto it.

Called by the bundle's `lgk_task.py` bootstrap with
    --src <bundled kit code> --run-src <bundled run files> --run-dir <writable run folder> --task <name>
    --job-id <the running job's id> [--sync-owner <the SP's application id> (preflight only)]
Every task first refuses a config whose kit schema is not the SP's own
(lgk.sp_grants.require_own_kit_schema): the SP builds only into its own kit schema, never the
workshop's, so a hand-edited bundle stops before anything runs or is recorded.
Each task then records its own job (ObjectType.JOB) in the run folder's manifest, so teardown
of the SP's graph deletes the bundle's jobs before anything else.
The bundled run files are copied into the writable run folder (the deployed files are replaced on
every deploy; the run folder keeps the SP's own manifest). Any failure raises, which fails the task.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

REFRESH_TASKS = ("build", "stamp", "enrich", "sync", "reapply")  # refresh_chain's python tasks
PROVISION_JOB_TASKS = ("preflight", "provision")


def _ws(path: str) -> Path:
    return Path(path if path.startswith("/Workspace") or not path.startswith("/Users") else "/Workspace" + path)


def _parse(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="lgk_task")
    for flag in ("--src", "--run-src", "--run-dir", "--task", "--job-id"):
        p.add_argument(flag, required=True)
    p.add_argument("--sync-owner", default=None)
    return p.parse_args(argv)


def _stage_run_folder(run_src: Path, run_dir: Path) -> Dict[str, Any]:
    """Copy the bundled inputs over the run folder; return the config bound to that folder."""
    run_dir.mkdir(parents=True, exist_ok=True)
    # Each bundle run starts with no step status: a run.json left by an earlier successful run
    # would make the runner skip provision silently (Isaac Review). The manifest and privacy
    # records stay, so teardown and the binding checks still see what earlier runs made.
    (run_dir / "run.json").unlink(missing_ok=True)
    shutil.copytree(run_src, run_dir, dirs_exist_ok=True)
    config = json.loads((run_dir / "config.json").read_text())
    config["run_folder"] = str(run_dir)
    return config


def _serving_steps() -> list:
    """provision -> advise -> switch: the views move last, onto a proven, tuned version."""
    from lgk.steps.advise import advise
    from lgk.steps.provision import provision
    from lgk.steps.switch import switch

    return [("provision", provision), ("advise", advise), ("switch", switch)]


def _run_provision(config: Dict[str, Any]) -> None:
    from lgk.runner import RunFolder, RunStatus, execute_steps

    results = execute_steps(RunFolder(Path(config["run_folder"])), config, _serving_steps())
    failed = [r for r in results if r.status != RunStatus.SUCCESS]
    if failed:
        raise RuntimeError(f"{failed[0].step} failed: {failed[0].error}")


def _run_preflight(config: Dict[str, Any], sync_owner: Any) -> None:
    """Preflight as this task's identity, into the run folder provision reads."""
    from lgk.steps.preflight import preflight

    if not sync_owner or "{" in str(sync_owner):
        raise ValueError(f"--sync-owner must be the SP's application id, got {sync_owner!r}")
    preflight({**config, "sync_owner_user": sync_owner})


def _record_bundle_job(config: Dict[str, Any], job_id: str, task: str) -> None:
    from lgk.refresh.job import record_job

    if not job_id.isdigit():
        raise ValueError(f"--job-id must be the running job's numeric id, got {job_id!r}")
    record_job(config, job_id, "provision" if task in PROVISION_JOB_TASKS else "refresh")


def main(argv: List[str]) -> None:
    args = _parse(argv)
    if args.task not in PROVISION_JOB_TASKS and args.task not in REFRESH_TASKS:
        raise ValueError(f"unknown bundle task {args.task!r}")
    config = _stage_run_folder(_ws(args.run_src), _ws(args.run_dir))
    from lgk.sp_grants import require_own_kit_schema

    require_own_kit_schema(config)
    _record_bundle_job(config, args.job_id, args.task)
    if args.task == "preflight":
        _run_preflight(config, args.sync_owner)
    elif args.task == "provision":
        _run_provision(config)
    else:
        from lgk.refresh.tasks import run_task

        run_task(args.task, json.dumps(config))
