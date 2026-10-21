"""Freshness guard: no DATA-changing source commit after the synced Delta version.

The synced signal is `status.last_sync.delta_table_sync_info.delta_commit_version`.
Never `status.last_sync_time`: it is always None live, so a guard on it would never fire.

Fresh iff last_data_version <= synced <= current, where last_data_version is the newest source
history version whose commit changes data. Commits that change no rows (OPTIMIZE, VACUUM,
ANALYZE, SET TBLPROPERTIES, predictive-optimization compaction, ...) come right after SDP rewrites
the sources and must not fail the job. A synced version still missing after the stated wait
(SYNCED_VERSION_WAIT_S) counts as stale; an unknown operation without row metrics counts as a
data change (fail loud). Every stale table is named.
"""

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from lgk import versions

logger = logging.getLogger(__name__)

SYNCED_VERSION_WAIT_S = 120
SYNCED_VERSION_POLL_S = 5

Sql = Callable[[str], List[List[Any]]]


class StaleSyncError(RuntimeError):
    """A synced table does not hold its source's current Delta version."""


def enrichment_tables(sql: Sql, config: Dict[str, Any]) -> List[str]:
    """The synced enrichment tables this run's refresh covers.

    The candidates are provision's own selection (`provision.enrichment_sync_set`: written by a
    method this run's enrich report lists), so the sync set and the guard set cannot drift. A
    table counts only if it is ALSO present in the kit schema in UC AND recorded as a
    SYNCED_TABLE in the run manifest: provision synced it. A table enrich wrote after provision
    is in UC with no synced table yet; it is skipped with a WARNING (re-provision to serve it),
    never refreshed (its synced table does not exist) and never a crash.
    """
    from lgk.steps.provision import enrichment_sync_set

    fq = f"{config['lakebase_catalog']}.{config['kit_schema']}"
    present = {str(row[1]) for row in sql(f"SHOW TABLES IN {fq}")}
    synced = _recorded_synced_tables(config)
    dest = f"{config['lakebase_catalog']}.{versions.version_schema(config)}"
    kept = []
    for t in sorted(t for t in enrichment_sync_set(config) if t in present):
        if f"{dest}.{t}" in synced:
            kept.append(t)
        else:
            logger.warning(f"{fq}.{t} is in UC but has no recorded synced table {dest}.{t} (enriched after "
                           "provision?): not refreshed; provision a new version to serve it")
    return kept


def _recorded_synced_tables(config: Dict[str, Any]) -> set:
    """object_ids of every SYNCED_TABLE in the run manifest; empty (with a WARNING) if none was recorded."""
    from lgk import manifest as mf

    try:
        entries = mf.load_run_manifest(config).entries
    except FileNotFoundError as e:
        logger.warning(f"refresh: {e}; no enrichment table is refreshed")
        return set()
    return {e.object_id for e in entries if e.object_type == mf.ObjectType.SYNCED_TABLE}


def synced_pairs(config: Dict[str, Any], sql: Sql) -> List[Tuple[str, str]]:
    """(synced table id, source table) for every version table AND synced enrichment table of this run."""
    cat = config["lakebase_catalog"]
    version = versions.version_schema(config)
    tables = list(versions.VERSION_TABLES) + enrichment_tables(sql, config)
    return [(f"{cat}.{version}.{t}", f"{cat}.{config['kit_schema']}.{t}") for t in tables]


def _read_synced_version(w: Any, dest_id: str) -> Optional[int]:
    status = w.postgres.get_synced_table(name=f"synced_tables/{dest_id}").status
    info = getattr(getattr(status, "last_sync", None), "delta_table_sync_info", None)
    return getattr(info, "delta_commit_version", None)


def synced_version(w: Any, dest_id: str) -> Optional[int]:
    """The synced Delta commit version, read again every 5 s for up to 120 s; None after that.

    Right after a sync the platform status can lose last_sync for a moment, so one empty read is
    not final. A version is required within 120 s, else the guard fails closed. A read error raises
    at once and is never waited out.
    """
    deadline = time.monotonic() + SYNCED_VERSION_WAIT_S
    while True:
        version = _read_synced_version(w, dest_id)
        if version is not None:
            return version
        left = deadline - time.monotonic()
        if left <= 0:
            return None
        time.sleep(min(SYNCED_VERSION_POLL_S, left))


# Columns by name, never by position: the warehouse's history has extra columns
# (queryHistoryStatementId), so operationMetrics is not at its Delta-docs index.
_HISTORY_SQL = "SELECT version, operation, operationMetrics FROM (DESCRIBE HISTORY {table})"

NO_DATA_OPERATIONS = frozenset({
    "OPTIMIZE", "VACUUM START", "VACUUM END", "ANALYZE", "COMPUTE STATS",
    "SET TBLPROPERTIES", "UNSET TBLPROPERTIES", "UPGRADE PROTOCOL",
    "ADD CONSTRAINT", "DROP CONSTRAINT", "REORG",
})
# Row-count metrics: if a commit reports any of them and all are 0, it changed no data.
_ROW_METRICS = ("numOutputRows", "numDeletedRows", "numUpdatedRows", "numTargetRowsInserted",
                "numTargetRowsUpdated", "numTargetRowsDeleted")


def _metrics(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return {}


def changes_data(operation: str, metrics: Dict[str, Any]) -> bool:
    """False for a commit that changes no rows; True otherwise (unknown counts as data)."""
    if operation.upper() in NO_DATA_OPERATIONS:
        return False
    counts = [int(metrics[k]) for k in _ROW_METRICS if metrics.get(k) not in (None, "")]
    return any(counts) if counts else True


def source_versions(sql: Sql, source_table: str) -> Tuple[int, Optional[int]]:
    """(current version, newest version whose commit changes data) from DESCRIBE HISTORY."""
    rows = sorted(((int(v), str(op), _metrics(m)) for v, op, m in
                   sql(_HISTORY_SQL.format(table=source_table))), reverse=True)
    current = rows[0][0]
    last_data = next((v for v, op, m in rows if changes_data(op, m)), None)
    return current, last_data


def assert_fresh(w: Any, sql: Sql, pairs: Sequence[Tuple[str, str]]) -> None:
    """Raise StaleSyncError naming every synced table with a data commit after its version."""
    stale = []
    for dest_id, source in pairs:
        synced = synced_version(w, dest_id)
        current, last_data = source_versions(sql, source)
        if synced is None:
            stale.append(f"{dest_id}: no synced version after {SYNCED_VERSION_WAIT_S} s (source version {current})")
        elif not ((last_data is None or last_data <= int(synced)) and int(synced) <= current):
            stale.append(f"{dest_id}: synced version {synced}, but source {source} has a data "
                         f"change at version {last_data} (current {current})")
    if stale:
        raise StaleSyncError("Stale sync, failing the refresh: " + "; ".join(stale))


def live_clients(config: Dict[str, Any]) -> Tuple[Any, Sql]:
    """(WorkspaceClient, warehouse SQL) for the refresh tasks; the warehouse is required."""
    from databricks.sdk import WorkspaceClient

    from lgk.uc_sql import execute_sql, warehouse_id

    wh = warehouse_id(config)
    w = WorkspaceClient()
    return w, lambda s: execute_sql(w, wh, s)


def check_freshness(config: Dict[str, Any]) -> None:
    """The guard for this run's synced tables (version + enrichment), with a live client and warehouse."""
    w, sql = live_clients(config)
    assert_fresh(w, sql, synced_pairs(config, sql))
