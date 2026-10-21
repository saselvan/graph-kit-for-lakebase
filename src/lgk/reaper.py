"""Reaper: delete what a deleted branch left behind of recorded synced tables.

Deleting a branch does not cascade. Observed live: from the moment the branch is gone, GET
synced_tables/<name> raises NotFound ("Branch with ID ... not found"), while the UC table
(type FOREIGN) and its pipeline (IDLE) remain. So a NotFound GET does NOT mean "gone".
There is no `list_synced_tables` API either, so the reaper reads only the run manifest:
for each recorded synced table whose branch no longer exists, it reaps whatever remains
of the synced table, its UC table and its recorded pipeline, each one directly.
A recorded job (the refresh job) bound to a deleted branch is an orphan too: it would re-sync
into nothing every night. It is reaped FIRST, as in teardown, so nothing re-syncs meanwhile.
Deleting needs the graph name typed exactly.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lgk import manifest as mf
from lgk import versions
from lgk.manifest import ObjectType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Orphan:
    synced_table: str  # "<cat>.<schema>.<table>" (also the UC table's full name)
    branch: str  # the deleted branch it was bound to
    synced_get: bool  # the synced-table GET still answers
    uc_table: bool  # the UC table still exists
    pipeline_id: Optional[str]  # the recorded pipeline, if it still exists


@dataclass(frozen=True)
class OrphanJob:
    job_id: str
    branch: str  # the deleted branch its refresh syncs


def find_orphan_jobs(config: Dict[str, Any], w: Optional[Any] = None) -> List[OrphanJob]:
    """Recorded jobs that still exist and whose recorded branch is gone."""
    from lgk.steps.teardown import job_settings

    w = w or _client()
    orphans = []
    for entry in mf.load_run_manifest(config).entries:
        branch = entry.metadata.get("branch")
        if entry.object_type != ObjectType.JOB or branch is None or _branch_exists(w, branch):
            continue
        if job_settings(w, entry.object_id) is not None:
            orphans.append(OrphanJob(entry.object_id, branch))
    return orphans


def find_orphans(config: Dict[str, Any], w: Optional[Any] = None) -> List[Orphan]:
    """Recorded synced tables whose branch is gone and of which something remains."""
    w = w or _client()
    orphans = []
    for entry in mf.load_run_manifest(config).entries:
        if entry.object_type != ObjectType.SYNCED_TABLE:
            continue
        orphan = _orphan_of(w, entry)
        if orphan is not None:
            orphans.append(orphan)
    return orphans


def reap(config: Dict[str, Any], confirm_graph_name: str, w: Optional[Any] = None) -> List[Any]:
    """Delete every orphan of this graph's manifest, jobs first. Returns what was reaped."""
    from lgk.steps.teardown import Plan, _check_jobs, delete_jobs

    versions.confirm_graph_name(config, confirm_graph_name)
    w = w or _client()
    jobs = find_orphan_jobs(config, w=w)
    _check_jobs(w, Plan(jobs=[j.job_id for j in jobs]), config["graph_name"])
    delete_jobs(w, [j.job_id for j in jobs])
    orphans = find_orphans(config, w=w)
    for orphan in orphans:
        _reap_one(w, orphan)
    return [*jobs, *orphans]


def _client() -> Any:
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


def _orphan_of(w: Any, entry: Any) -> Optional[Orphan]:
    """The orphan for one recorded synced table, or None (branch alive, or nothing left)."""
    from databricks.sdk.errors import NotFound

    branch = entry.metadata.get("branch")
    try:
        table = w.postgres.get_synced_table(name=f"synced_tables/{entry.object_id}")
        synced_get = True
        # A sparse GET has no spec.branch; then the branch recorded at creation is used.
        branch = getattr(getattr(table, "spec", None), "branch", None) or branch
    except NotFound:
        synced_get = False  # deleted, OR its branch is gone (the UC table may remain)
    if branch is None or _branch_exists(w, branch):
        return None
    uc_table = _uc_table_exists(w, entry.object_id)
    pid = entry.metadata.get("pipeline_id")
    pipeline_id = pid if pid and _pipeline_exists(w, pid) else None
    if not (synced_get or uc_table or pipeline_id):
        return None
    return Orphan(entry.object_id, branch, synced_get, uc_table, pipeline_id)


def _reap_one(w: Any, orphan: Orphan) -> None:
    """Delete the synced table, then the UC table and the pipeline directly. Gone = done."""
    from databricks.sdk.errors import NotFound

    steps = (
        ("synced table", lambda: w.postgres.delete_synced_table(name=f"synced_tables/{orphan.synced_table}").wait()),
        ("UC table", lambda: w.tables.delete(orphan.synced_table)),
        ("pipeline", lambda: orphan.pipeline_id and w.pipelines.delete(orphan.pipeline_id)),
    )
    for what, delete in steps:
        try:
            delete()
            logger.info(f"Reaped {what} of {orphan.synced_table} (branch {orphan.branch} is gone)")
        except NotFound:
            logger.info(f"{what} of {orphan.synced_table} already gone")


def _branch_exists(w: Any, branch: str) -> bool:
    from databricks.sdk.errors import NotFound

    try:
        w.postgres.get_branch(name=branch)
        return True
    except NotFound:
        return False


def _uc_table_exists(w: Any, full_name: str) -> bool:
    from databricks.sdk.errors import NotFound

    try:
        w.tables.get(full_name)
        return True
    except NotFound:
        return False


def _pipeline_exists(w: Any, pipeline_id: str) -> bool:
    """A pipeline counts as gone when GET is NotFound or its state is DELETED."""
    from databricks.sdk.errors import NotFound

    try:
        state = w.pipelines.get(pipeline_id).state
    except NotFound:
        return False
    return str(getattr(state, "value", state)) != "DELETED"
