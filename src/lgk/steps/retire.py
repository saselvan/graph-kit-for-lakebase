"""Retire version schemas: `retire` (one) and `retire_versions` (several).

Both need the graph name typed exactly, and both refuse the version the served views
point at (read from pg_depend, never from a stored flag), and refuse any schema that
holds anything but this graph's version and enrichment tables (versions.OURS_TABLES). EVERY check runs before ANY delete:
live, outside dependents, the Postgres contents, and the UC schema's marker and tables
(which must all be synced tables recorded in the run manifest, or the event log of one of
their pipelines: the pipeline id recorded, else read from Unity Catalog, else refused;
lgk.event_logs). Then, per version (verified live: a dependent view blocks
delete_synced_table): the recorded UC synced tables and their pipelines, then those
pipelines' event-log tables, then the Postgres version schema, then the UC version
schema. Both schema drops are RESTRICT. Whole-graph teardown is `lgk.steps.teardown`.
"""

import logging
from typing import Any, Dict, List, Optional

from lgk import event_logs as el
from lgk import manifest as mf
from lgk import served_functions
from lgk import versions
from lgk.manifest import ObjectType
from lgk.branch import connect_branch

logger = logging.getLogger(__name__)


def retire(config: Dict[str, Any], version: int, confirm_graph_name: str) -> None:
    """Drop an old version kept for rollback (normally N-1). Refuses the live version."""
    _drop_versions(config, [version], confirm_graph_name)


def retire_versions(config: Dict[str, Any], versions: List[int], confirm_graph_name: str) -> None:
    """Drop the named versions (not the whole graph: that is lgk.steps.teardown.teardown). Refuses all of them if any is live; drops nothing then."""
    _drop_versions(config, versions, confirm_graph_name)


def _drop_versions(config: Dict[str, Any], numbers: List[int], confirm_graph_name: str) -> None:
    """Dispatcher: confirm, run every check, then drop each version in the verified order."""
    versions.confirm_graph_name(config, confirm_graph_name)
    schemas = [versions.version_schema(config, n) for n in numbers]
    served = versions.served_schema(config)
    recorded = _recorded_synced_tables(config)
    conn = connect_branch(config)
    conn.autocommit = True  # each `with conn.transaction()` below is then a real, committed txn
    try:
        logs = _check_all(config, conn, served, schemas, recorded)
        for schema in schemas:
            _delete_synced_tables(list(_in_schema(config, schema, recorded)))
            _drop_event_logs(config, logs[schema])  # after the pipelines are gone (teardown's order)
            _drop_pg_schema(conn, served, schema)
            _drop_uc_schema(config, schema)
    finally:
        conn.close()


Recorded = Dict[str, Optional[str]]  # recorded synced table -> its recorded pipeline id (None: not recorded)


def _check_all(config: Dict[str, Any], conn: Any, served: str, schemas: List[str],
               recorded: Recorded) -> Dict[str, List[str]]:
    """Every refusal, before anything is deleted (a later failure would leave a half-deleted version).

    Returns each schema's own event-log tables (resolved here, so an unknown pipeline refuses early).
    """
    versions.refuse_if_live(conn, served, schemas)
    versions.refuse_outside_dependents(conn, schemas)
    logs: Dict[str, List[str]] = {}
    for schema in schemas:
        versions.refuse_unless_ours_pg(conn, schema)
        _refuse_unrecorded_synced_tables(conn, config, schema, recorded)
        logs[schema] = _own_event_logs(config, schema, recorded)
        _check_uc_version(config, schema, recorded, logs[schema])
    return logs


def _in_schema(config: Dict[str, Any], schema: str, recorded: Recorded) -> Recorded:
    """The recorded synced tables whose destination is this version schema."""
    prefix = f"{config['lakebase_catalog']}.{schema}."
    return {n: p for n, p in recorded.items() if n.startswith(prefix)}


def _own_event_logs(config: Dict[str, Any], schema: str, recorded: Recorded) -> List[str]:
    """The event-log tables of this version's recorded synced tables' pipelines (lgk.event_logs)."""
    mine = _in_schema(config, schema, recorded)
    return el.own_event_logs(_workspace_client(), mine) if mine else []


def _workspace_client() -> Any:
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


def _refuse_unrecorded_synced_tables(conn: Any, config: Dict[str, Any], schema: str, recorded: Recorded) -> None:
    """A Postgres version schema with tables but no recorded synced tables is refused.

    Without the manifest we cannot delete its UC synced tables and pipelines first, and dropping
    the Postgres side alone would leave them orphaned, still syncing.
    """
    prefix = f"{config['lakebase_catalog']}.{schema}."
    if any(n.startswith(prefix) for n in recorded):
        return
    # Synced tables are owned by the sync pipeline's databricks_writer_<n> role (verified fact);
    # plain tables have no UC synced table or pipeline behind them, so cannot be orphaned.
    row = conn.execute(
        "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "JOIN pg_roles r ON r.oid = c.relowner "
        "WHERE n.nspname = %s AND c.relkind IN ('r', 'p') AND r.rolname LIKE 'databricks\\_writer\\_%%'",
        (schema,),
    ).fetchone()
    if row and row[0]:
        raise versions.NotOursError(
            f"{schema} holds {row[0]} synced table(s) but the run manifest records none of them; "
            "refusing to drop it (its UC synced tables and pipelines would be orphaned). "
            "Restore lgk_manifest.json or run the reaper first."
        )


def _recorded_synced_tables(config: Dict[str, Any]) -> Recorded:
    """Synced tables the run manifest recorded, with their recorded pipeline ids; none if the run recorded nothing."""
    try:
        entries = mf.load_run_manifest(config).entries if config.get("run_folder") else []
    except FileNotFoundError:
        entries = []
    return {e.object_id: (e.metadata or {}).get("pipeline_id") or None
            for e in entries if e.object_type == ObjectType.SYNCED_TABLE}


def _check_uc_version(config: Dict[str, Any], schema: str, recorded: Recorded, event_logs: List[str]) -> None:
    """An existing UC version schema must carry this graph's marker and hold only recorded synced
    tables and their pipelines' event logs (`event_logs`, full names)."""
    full = f"{config['lakebase_catalog']}.{schema}"
    exists, comment = mf.uc_schema_comment(_workspace_client(), full)
    if not exists:
        return
    names = _uc_table_names(config, schema)
    own_logs = [t.rsplit(".", 1)[1] for t in event_logs]  # all in this schema (event_log_table)
    versions.refuse_unless_ours_uc(names, full, own_logs)
    mf.refuse_unless_marked(f"UC schema {full}", comment, config["graph_name"])
    unrecorded = sorted(n for n in names if f"{full}.{n}" not in recorded and n not in own_logs)
    if unrecorded:
        raise versions.NotOursError(f"UC schema {full} holds tables the run manifest never recorded: {unrecorded}")


def _drop_event_logs(config: Dict[str, Any], tables: List[str]) -> None:
    """The retired version's own pipelines' event-log tables (the pipelines are gone by now)."""
    if not tables:
        return
    from lgk.uc_sql import warehouse_id

    el.drop_event_logs(_workspace_client(), warehouse_id(config), tables)


def _delete_synced_tables(names: List[str]) -> None:
    """Delete recorded UC synced tables (this also deletes their pipelines). Gone = done."""
    from databricks.sdk.errors import NotFound

    w = _workspace_client()
    for full in names:
        name = f"synced_tables/{full}"
        try:
            w.postgres.delete_synced_table(name=name).wait()
            logger.info(f"Deleted synced table {name}")
        except NotFound:
            logger.info(f"No synced table {name}; nothing to delete")


def _uc_table_names(config: Dict[str, Any], schema: str) -> List[str]:
    """Table names in the UC version schema; empty if the schema does not exist."""
    from databricks.sdk.errors import NotFound

    try:
        return [t.name for t in _workspace_client().tables.list(config["lakebase_catalog"], schema)]
    except NotFound:
        return []


def _drop_pg_schema(conn: Any, served: str, schema: str) -> None:
    """DROP the version tables and schema RESTRICT, re-checking live + ours in one txn."""
    with conn.transaction():
        versions.refuse_if_live(conn, served, [schema])
        versions.refuse_unless_ours_pg(conn, schema)
        for fn in served_functions.staged_signatures(conn, schema):
            conn.execute(served_functions.drop_sql(fn))  # provision's staged copies
        for table in versions.OURS_TABLES:  # the version tables and the enrichment tables
            conn.execute(f"DROP TABLE IF EXISTS {schema}.{table} RESTRICT")
        conn.execute(f"DROP SCHEMA IF EXISTS {schema} RESTRICT")
    logger.info(f"Dropped Postgres version schema {schema}")


def _drop_uc_schema(config: Dict[str, Any], schema: str) -> None:
    from lgk.uc_sql import execute_sql, warehouse_id

    execute_sql(_workspace_client(), warehouse_id(config), f"DROP SCHEMA IF EXISTS {config['lakebase_catalog']}.{schema} RESTRICT")
    logger.info(f"Dropped UC version schema {schema}")
