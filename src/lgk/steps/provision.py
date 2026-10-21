"""Provision step: a new, NOT yet served version in Lakebase.

Creates, privately:
1. SNAPSHOT synced tables from UC Delta into the version schema `<kit>_v<N>`
2. The LOGIN-only reader role, with every right on the version's raw objects revoked
3. The reader-isolation proof on the new version, then the servable record

It never touches the served views: `advise` then tunes the not-yet-live version, and
`switch` (lgk.steps.switch) moves every served view onto it in one transaction.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from lgk import manifest as mf
from lgk import served_functions as sf
from lgk import versions
from lgk.branch import BranchRef, connect_branch
from lgk.attribution import read_attribution, serve_attribution
from lgk.manifest import ObjectType
from lgk.privacy import binding, floor
from lgk.privacy.gate import fold_policy, read_run_mapping, require_clear, required_run_folder
from lgk.privacy.probe import assert_function_roles_isolated, assert_reader_isolated
from lgk.steps.preflight import require_sync_owner
from lgk.uc_sql import execute_sql, warehouse_id

logger = logging.getLogger(__name__)


def provision(config: Dict[str, Any]) -> bool:
    """Run provision step: synced tables, views, and reader role.

    Reads:
    - UC Delta tables created by build step (nodes, edges, node_id_map)

    Creates:
    - Synced tables in Lakebase project (SNAPSHOT mode), in the version schema
    - The walk and lookup indexes on them (reapply.INDEXES and the walk-key indexes)
    - Reader LOGIN role with no right on the version's raw objects
    Does not create or move served views (that is `switch`). Returns True (a new version).

    Args:
        config: Configuration dict with:
            - graph_name: Name of the graph
            - run_folder: Path to run folder
            - lakebase_project: Lakebase project ID
            - lakebase_catalog: UC catalog
            - branch_id: Test branch ID (optional)
            - kit_schema: Schema in Postgres for kit tables
            - served_schema: Schema in Postgres for served views (default: kit_schema)
            - version: version number N; synced tables land in <kit_schema>_v<N>
    """
    # Kit v1 only supports snapshot mode; refuse everything else before any other step
    from lgk.refresh import modes
    modes.require_v1_sync_mode(config)

    # The privacy gate, before any SQL: nothing is served unless flags.json is clear and fresh
    run_folder = required_run_folder(config)
    cleared = require_clear(run_folder)
    stamp = binding.stamp_id(cleared)
    require_sync_owner(run_folder)  # only the recorded sync-job owner provisions

    # Step 0: refuse a run onto an existing version schema, before anything is made
    _refuse_existing_version(config)
    # ...and unless the Delta tables were built under the current decision
    _refuse_unbound_build(config, run_folder, stamp, cleared)

    # Step 1: Create synced tables, then mark the Postgres version schema they landed in
    _create_synced_tables(config)
    _mark_pg_version_schema(config)
    # ...and index the walk joins and every declared walk key: the same fixed names
    # reapply ensures, so a new version is never served on a scan of `edges` (scale spike)
    _create_version_indexes(config)

    # Step 2: the reader role exists and holds nothing on the new version's raw objects
    _lock_version_from_reader(config)
    # ...and the function surface is staged on the new version, executable by nobody yet
    _stage_version_functions(config)

    # Step 3: prove the reader reaches no raw table and is in no other role,
    # while the version is not live yet: a failed proof leaves the site on the old version
    _prove_reader_isolated(config)

    # Step 4: only a fully provisioned, isolation-proven version is recorded as servable
    # (switch's check_version trusts this record, so it must come last)
    binding.record_version(run_folder, versions.version_schema(config), stamp)

    logger.info(f"Provisioned {versions.version_schema(config)} (not served until switch)")
    return True


# A new version invalidates the later steps of the earlier one: advise and switch run again
# for it instead of being skipped as already succeeded (run.json).
provision.invalidates_on_change = ("advise", "switch")  # type: ignore[attr-defined]


def _kit_tables(config: Dict[str, Any]) -> list:
    return [f"{config['lakebase_catalog']}.{config['kit_schema']}.{t}" for t in binding.BUILT_TABLES]


def _refuse_unbound_build(config: Dict[str, Any], run_folder: str, stamp: str,
                          cleared: Dict[str, Any]) -> None:
    """Refuse unless the run folder AND every Delta table say: built under `stamp`.

    The enrichment tables this run will sync (`enrichment_sync_set`) are checked the same way:
    the run folder's enrich record and each table's own stamp. Then the
    folded edges: see `_refuse_unfloored_edges`. `cleared` is the gate data provision read
    (the floor and fold types). It is required and must be a dict: a missing or None value
    would skip the fold check, so it raises TypeError instead (final review S4).
    """
    from databricks.sdk import WorkspaceClient

    if not isinstance(cleared, dict):
        raise TypeError(f"_refuse_unbound_build needs the cleared gate data (a dict), got {type(cleared).__name__}")

    tables = _kit_tables(config)
    prefix = f"{config['lakebase_catalog']}.{config['kit_schema']}."
    enriched = [prefix + t for t in enrichment_sync_set(config)]
    binding.check_build_record(run_folder, stamp, tables)
    binding.check_enrich_record(run_folder, stamp, enriched)
    w = WorkspaceClient()
    wh = warehouse_id(config)
    sql = lambda s: execute_sql(w, wh, s)  # noqa: E731
    binding.check_table_stamps(sql, tables + enriched, stamp)
    _refuse_unfloored_edges(sql, config, run_folder, cleared)


def _refuse_unfloored_edges(sql: Any, config: Dict[str, Any], run_folder: str, cleared: Dict[str, Any]) -> None:
    """The recorded build folded every fold type under the current floor, and
    the Delta `edges` holds no folded edge below it (or NULL). Before any synced table is made."""
    fold_floor, fold_types = fold_policy(cleared)
    binding.check_build_folds(run_folder, fold_floor, fold_types)
    floor.assert_fold_floor(sql, f"{config['lakebase_catalog']}.{config['kit_schema']}.edges",
                            fold_types, fold_floor)
    _refuse_unfloored_joins(sql, config, run_folder, fold_floor)


def _refuse_unfloored_joins(sql: Any, config: Dict[str, Any], run_folder: str, fold_floor: int) -> None:
    """The recorded build made every join output under the current floor, and the
    Delta `nodes` holds no floored join output on a node below it. Before any synced table."""
    mapping = read_run_mapping(run_folder)
    binding.check_build_joins(run_folder, fold_floor, floor.join_report(mapping, fold_floor))
    floor.assert_join_floor(sql, f"{config['lakebase_catalog']}.{config['kit_schema']}.nodes",
                            mapping, fold_floor)


# The built tables every version syncs, with their synced-table primary keys.
BUILT_SYNCED_TABLES = {"nodes": ["node_id"], "edges": ["edge_id"], "node_id_map": ["node_id"]}


def _create_synced_tables(config: Dict[str, Any]) -> None:
    """Dispatcher: claim the UC schemas, then create, await and record each SNAPSHOT synced table."""
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    wh = warehouse_id(config)
    branch_full = BranchRef.from_config(config).path
    synced_schema = _claim_synced_uc_schemas(w, wh, config)
    catalog, kit_schema = config["lakebase_catalog"], config["kit_schema"]
    # Enrichment outputs sync too: those this run's enrich report says were written
    tables = {**BUILT_SYNCED_TABLES, **enrichment_sync_set(config)}
    _assert_per_source_room(w, wh, config, tables)
    for table_name, pk_cols in tables.items():
        dest_id = f"{catalog}.{synced_schema}.{table_name}"
        _create_one_synced_table(w, config, table_name, f"{catalog}.{kit_schema}.{table_name}",
                                 dest_id, pk_cols, branch_full)
        _wait_synced_table_online(w, dest_id)
        _record_synced_table(config, w, dest_id, branch_full)


def _assert_per_source_room(w: Any, wh: str, config: Dict[str, Any], tables: Dict[str, list]) -> None:
    """Before the first create: every table about to sync has room under its source's cap."""
    from lgk import sync_limits as sl

    headroom = sl.config_int(config, "sync_per_source_headroom", sl.DEFAULT_SYNC_PER_SOURCE_HEADROOM, 0)
    counts = sl.per_source_counts(lambda s: execute_sql(w, wh, s), config["lakebase_catalog"], config["kit_schema"])
    sl.assert_room(counts, tables, headroom=headroom)


def _claim_synced_uc_schemas(w: Any, wh: str, config: Dict[str, Any]) -> str:
    """Claim the UC kit schema and version schema (marker; recorded if created); return the latter.

    Synced tables land in the run's version schema, separate from the Delta sources.
    """
    catalog = config["lakebase_catalog"]
    synced_schema = versions.version_schema(config)
    for schema in (config["kit_schema"], synced_schema):
        if mf.claim_uc_schema(w, wh, catalog, schema, config["graph_name"]):
            mf.record(config, ObjectType.UC_SCHEMA, f"{catalog}.{schema}", "uc")
    return synced_schema


def _create_one_synced_table(w: Any, config: Dict[str, Any], table_name: str, source_table: str,
                             dest_id: str, pk_cols: list, branch_full: str) -> None:
    """Create one SNAPSHOT synced table; an existing one is reused only if bound to this branch."""
    from databricks.sdk.service import postgres as pg

    logger.info(f"Creating synced table from {source_table} to {dest_id} on branch {branch_full}")
    spec = pg.SyncedTableSyncedTableSpec(
        source_table_full_name=source_table,
        primary_key_columns=pk_cols,
        scheduling_policy=_scheduling_policy(w, config, table_name, source_table),
        branch=branch_full,
        postgres_database="databricks_postgres",
        create_database_objects_if_missing=True,
    )
    try:
        w.postgres.create_synced_table(synced_table=pg.SyncedTable(spec=spec), synced_table_id=dest_id)
        logger.info(f"Created synced table operation for {dest_id}")
    except Exception as e:
        from lgk import sync_limits as sl

        sl.raise_if_per_source_cap(e, source_table)
        if "already exists" not in str(e).lower():
            logger.error(f"Failed to create synced table {dest_id}: {e}")
            raise
        # Never silently reuse: a synced table outlives its branch (branch deletion
        # does not cascade), so an old one may be bound to a deleted branch.
        _assert_existing_synced_table_on_branch(w, dest_id, branch_full)
        logger.info(f"Synced table {dest_id} already exists on {branch_full}; reusing it")


def _record_synced_table(config: Dict[str, Any], w: Any, dest_id: str, branch_full: str) -> None:
    """Record the synced table with its branch (for the reaper) and its pipeline id."""
    status = w.postgres.get_synced_table(name=f"synced_tables/{dest_id}").status
    mf.record(
        config,
        ObjectType.SYNCED_TABLE,
        dest_id,
        "uc",
        {"branch": branch_full, "pipeline_id": getattr(status, "pipeline_id", None)},
    )


# Enrichment tables served behind the optional views, with their synced-table primary keys
# (unique by construction in lgk.enrich.methods). SNAPSHOT (modes.policy_name), version schema.
ENRICHMENT_SYNCED_TABLES = {
    "node_score": ["node_id", "method"],
    "node_group": ["node_id", "method"],
    "similar_to": ["src_id", "dst_id"],
    "closure": ["ancestor_id", "descendant_id"],
}


def serve_closure(config: Dict[str, Any]) -> bool:
    """`serve_closure` (default false): closure is opt-in by configuration."""
    value = config.get("serve_closure", False)
    if not isinstance(value, bool):
        raise ValueError(f"serve_closure must be true or false, got {value!r}")
    return value


def enrichment_sync_set(config: Dict[str, Any]) -> Dict[str, list]:
    """The enrichment tables this run syncs and serves, with their primary keys.

    A table counts only if this run's `enrich_report.json` lists a method that writes it (never
    because the name is present in UC: an earlier run's table may be stale); `closure` also
    needs `serve_closure: true`. provision checks each one's privacy stamp, and the refresh
    guard (lgk.refresh.freshness.enrichment_tables) uses this same set, so the two cannot drift.
    """
    from lgk.steps.enrich import REPORT_FILE, TABLE_OF

    run_folder = config.get("run_folder")
    report = Path(run_folder) / REPORT_FILE if run_folder else None
    if report is None or not report.exists():
        return {}
    written = {TABLE_OF[m] for m in json.loads(report.read_text()) if m in TABLE_OF}
    if not serve_closure(config):
        written.discard("closure")
    return {t: pk for t, pk in ENRICHMENT_SYNCED_TABLES.items() if t in written}


def _scheduling_policy(w: Any, config: Dict[str, Any], table: str, source_table: str) -> Any:
    """The synced table's sync mode; a triggered source must be a streaming table.

    Checked before create: a triggered table over a materialized view is accepted, then fails
    later as ONLINE_PIPELINE_FAILED.
    """
    from databricks.sdk.service import postgres as pg

    from lgk.refresh import modes

    # Kit v1 only supports snapshot mode; refuse everything else before reading the source
    modes.require_v1_sync_mode(config)

    name = modes.policy_name(config, table)
    if name == "TRIGGERED":
        modes.assert_streaming_source(w, source_table)
    return getattr(pg.SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy, name)


class StaleSyncedTableError(RuntimeError):
    """An existing synced table with the target name is not bound to the run's branch."""


def _assert_existing_synced_table_on_branch(w: Any, dest_id: str, branch_full: str) -> None:
    """Raise unless the existing synced table `dest_id` is bound to `branch_full`.

    A synced table left over from an earlier run can still point at a deleted branch;
    reusing it fails later with "Branch ... not found". If the GET is sparse and the
    branch cannot be proven, that also raises — reuse must be proven, not assumed.
    """
    name = f"synced_tables/{dest_id}"
    existing = w.postgres.get_synced_table(name=name)
    bound = getattr(getattr(existing, "spec", None), "branch", None)
    branch_id = branch_full.rsplit("/", 1)[-1]
    if bound and (bound == branch_full or bound.endswith(f"/branches/{branch_id}")):
        return
    raise StaleSyncedTableError(
        f"Synced table {dest_id} already exists but is bound to branch {bound!r}, not "
        f"{branch_full!r} (or its branch could not be read). It is stale: tear it down with "
        f"w.postgres.delete_synced_table(name={name!r}) (drop dependent served views first), "
        f"then re-run provision."
    )


def _wait_synced_table_online(w: Any, dest_id: str, max_wait_s: int = 900) -> None:
    """Block until the synced table's first snapshot lands in Postgres.

    Verified live: the GET name needs the `synced_tables/` prefix, and the
    ready signal is `status.detailed_state` == ONLINE_NO_PENDING_UPDATE. There is no
    "READY" state. A *_FAILED state raises — views must never be built over a missing table.
    """
    import time

    name = f"synced_tables/{dest_id}"
    deadline = time.monotonic() + max_wait_s
    state = ""
    while time.monotonic() < deadline:
        status = w.postgres.get_synced_table(name=name).status
        state = str(getattr(getattr(status, "detailed_state", None), "value", ""))
        if "ONLINE_NO_PENDING_UPDATE" in state:
            logger.info(f"Synced table {dest_id} is online")
            return
        if "FAILED" in state:
            raise RuntimeError(f"Synced table {dest_id} failed: {state}")
        logger.info(f"Waiting for synced table {dest_id}: {state}")
        time.sleep(15)
    raise TimeoutError(f"Synced table {dest_id} not online after {max_wait_s}s (last: {state})")


def _refuse_existing_version(config: Dict[str, Any]) -> None:
    """A run whose version schema already exists on the branch is refused."""
    conn = connect_branch(config)
    try:
        versions.assert_version_schema_absent(conn, versions.version_schema(config))
    finally:
        conn.close()


def _mark_pg_version_schema(config: Dict[str, Any]) -> None:
    """Mark and record the Postgres version schema the synced-table pipeline made for this run.

    The version schema check refused the run if it existed before, so an unmarked one here is this run's.
    """
    schema = versions.version_schema(config)
    conn = connect_branch(config)
    try:
        mf.claim_pg_schema(conn, schema, config["graph_name"], adopt_unmarked=True)
    finally:
        conn.close()
    mf.record(config, ObjectType.PG_SCHEMA, schema, mf.pg_location(config))


def _create_version_indexes(config: Dict[str, Any]) -> None:
    """reapply.ensure_indexes on the new version: `reapply.INDEXES` (edges src/dst and the
    lookup columns of the enrichment tables present) plus the walk-key index per declared key.

    Without the edges indexes every walk hop scans all of `edges` until the first nightly
    reapply (1.6 M edges: ~290 ms per hop), and advise adds only the outbound one, never on a
    mixed folded graph. Reapply re-ensures the same names (IF NOT EXISTS)."""
    from lgk.privacy.gate import read_run_mapping
    from lgk.steps.reapply import ensure_indexes

    mapping = read_run_mapping(required_run_folder(config))
    conn = connect_branch(config)
    try:
        conn.autocommit = True
        ensure_indexes(conn, versions.version_schema(config), mapping)
    finally:
        conn.close()


def serve(conn: Any, config: Dict[str, Any], target: str) -> None:
    """Point the served views AND functions at `target` in one transaction; record what was made.

    The function roles' isolation probe runs inside that transaction, so a surface that grants
    a role more than its functions never commits.
    """
    served = versions.served_schema(config)
    if mf.claim_pg_schema(conn, served, config["graph_name"]):
        mf.record(config, ObjectType.PG_SCHEMA, served, mf.pg_location(config))
    _ensure_function_roles(config, conn.cursor())
    expected = sf.expected_grants(served, config)
    required = [fn.signature(served) for fn in sf.served_functions(config, target)]
    kit = versions.kit_schema(config)
    # read on the caller's autocommit conn, so the switch still starts outside a transaction; an
    # overload made after this read is caught by the probe inside the switch and rolls it back
    present = sf.staged_signatures(conn, served)
    versions.switch_views(
        conn, served, target, sf.switch_statements(config, served, target, present=present),
        verify=lambda c: assert_function_roles_isolated(
            c, kit, served, expected, sf.FUNCTION_ROLES, sf.SEARCH_PATH, required),
        reader=READER_ROLE,
    )
    # Every view the switch may make, optional ones included: one absent now may be created by a
    # later switch (teardown drops each IF EXISTS, so an absent one costs nothing)
    for view, _ in versions.SERVED_VIEWS + versions.OPTIONAL_SERVED_VIEWS:
        mf.record(config, ObjectType.VIEW, f"{served}.{view}", mf.pg_location(config))
    for signature in expected:
        mf.record(config, ObjectType.FUNCTION, signature, mf.pg_location(config))


def _stage_version_functions(config: Dict[str, Any]) -> None:
    """Provision: every served function, created inside the new version schema.

    EXECUTE is revoked from PUBLIC and granted to no one, and each is called once as the
    operator, so a version whose tables do not fit the bodies fails here, before it is live.
    """
    version = versions.version_schema(config)
    conn = connect_branch(config)
    try:
        # The smoke calls run in the same transaction as the CREATEs: a failing call rolls every
        # staged function back, so nothing unrecorded is left to block teardown's DROP ... RESTRICT
        with conn.transaction():
            for stmt in sf.stage_statements(config, version):
                conn.execute(stmt)
            for call in sf.smoke_calls(config, version):
                conn.execute(call).fetchall()
        conn.commit()
    finally:
        conn.close()
    # recorded only after the transaction committed
    for fn in sf.served_functions(config, version):
        mf.record(config, ObjectType.FUNCTION, fn.signature(version), mf.pg_location(config))


def _create_served_views(config: Dict[str, Any]) -> None:
    """Re-serve the config's version (reapply after a sync): the one-transaction view switch."""
    conn = connect_branch(config)
    conn.autocommit = True
    try:
        serve(conn, config, versions.version_schema(config))
    finally:
        conn.close()


def _ensure_reader_role(config: Dict[str, Any], cursor: Any) -> None:
    """The LOGIN-only reader role, created if missing.

    Never marked or recorded: a Postgres role is cluster-wide, so every graph on the branch
    shares it, and a graph's teardown must never drop it (owner decision A6 (a)).
    """
    _create_reader_role_pg(cursor)


def _lock_version_from_reader(config: Dict[str, Any]) -> None:
    """Provision: the reader exists and holds no right on the version's raw objects.

    No view grant here: the served views are switch's, and the new version is not served yet.
    """
    conn = connect_branch(config)
    cursor = conn.cursor()
    try:
        _ensure_reader_role(config, cursor)
        _ensure_function_roles(config, cursor)
        _revoke_reader_raw_table_access(cursor, versions.version_schema(config))
    finally:
        conn.commit()
        cursor.close()
        conn.close()


def _ensure_function_roles(config: Dict[str, Any], cursor: Any) -> None:
    """The LOGIN-only public_reader and coordinator_reader, created if missing.

    Shared by every graph on the branch like `reader`: never marked or recorded (A6 (a)).
    """
    for role in sf.FUNCTION_ROLES:
        _create_login_role_pg(cursor, role)


def _setup_reader_role(config: Dict[str, Any]) -> None:
    """Serve time (switch, reapply): the reader role, SELECT on the served views, raw revoked."""
    conn = connect_branch(config)
    cursor = conn.cursor()

    try:
        _ensure_reader_role(config, cursor)
        _grant_reader_view_access(cursor, versions.served_schema(config))
        _revoke_reader_raw_table_access(cursor, versions.version_schema(config))
    finally:
        conn.commit()
        cursor.close()
        conn.close()


def _serve_attribution(config: Dict[str, Any]) -> None:
    """Serve attribution.json (lgk.attribution) as a recorded reader view."""
    record = read_attribution(str(config.get("run_folder", ".")))
    if record is None:
        return
    conn = connect_branch(config)
    try:
        serve_attribution(conn, config, record)
    finally:
        conn.close()


def _prove_reader_isolated(config: Dict[str, Any]) -> None:
    """The privacy proof (provision, and switch before it moves the views): raise unless
    `reader` is isolated from every version's raw data."""
    kit, served = versions.kit_schema(config), versions.served_schema(config)
    conn = connect_branch(config)
    try:
        assert_reader_isolated(conn, kit)
        assert_function_roles_isolated(conn, kit, served, sf.expected_grants(served, config),
                                       sf.FUNCTION_ROLES, sf.SEARCH_PATH)
    finally:
        conn.close()


# The one-time operator setup for an identity that cannot create roles (the bundle's SP).
ROLE_SETUP_SQL = {role: f"CREATE ROLE {role} WITH LOGIN;" for role in ("reader",) + sf.FUNCTION_ROLES}
READER_SETUP_SQL = " ".join(ROLE_SETUP_SQL.values())


class ReaderRoleMissingError(RuntimeError):
    """The `reader` role does not exist and the running identity may not create it."""


def _create_reader_role_pg(cursor: Any) -> bool:
    """Ensure the LOGIN-only reader role. True if it was created here.

    An existing `reader` is used as is: no CREATE is issued, so an identity without CREATEROLE
    (the exported bundle's SP) only GRANTs on the objects it owns. A missing role is created
    with plain CREATE ROLE (not databricks_create_role — that is for Databricks identities); if
    that is refused, the operator must create it first. Nothing tries to escalate.
    """
    return _create_login_role_pg(cursor, "reader")


def _create_login_role_pg(cursor: Any, role: str) -> bool:
    """Ensure the LOGIN-only `role` (plain CREATE ROLE, never escalated). True if created here."""
    import psycopg

    cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
    if cursor.fetchone() is not None:
        logger.info(f"Role {role} exists; not creating it")
        return False
    setup = ROLE_SETUP_SQL[role]
    try:
        cursor.execute(setup)
    except psycopg.errors.InsufficientPrivilege as e:
        raise ReaderRoleMissingError(
            f"role '{role}' does not exist on this branch and this identity cannot create it ({e}). "
            f"Operator, run once on the branch (or its parent; roles are copied to child branches): "
            f"{setup}"
        ) from e
    logger.info(f"Created role: {role}")
    return True


READER_ROLE = "reader"

def _present(cursor: Any, schema: str, names: tuple) -> list:
    """The `names` that exist as relations in `schema`, in `names` order (to_regclass is NULL
    for an absent relation)."""
    found = []
    for name in names:
        cursor.execute("SELECT to_regclass(%s)", (f"{schema}.{name}",))
        row = cursor.fetchone()
        if row is not None and row[0] is not None:
            found.append(name)
    return found


def _grant_reader_view_access(cursor: Any, served_schema: str) -> None:
    """Grant reader USAGE on the served schema and SELECT on the three views and each optional
    view present (the switch runs the same block inside its transaction; this re-run is a no-op)."""
    optional = _present(cursor, served_schema, tuple(v for v, _ in versions.OPTIONAL_SERVED_VIEWS))
    views = [v for v, _ in versions.SERVED_VIEWS] + optional
    for stmt in versions.reader_grant_sql(served_schema, served_schema, views, (), READER_ROLE):
        cursor.execute(stmt)
    logger.info(f"Granted SELECT on served views {views} to reader role")


def _revoke_reader_raw_table_access(cursor: Any, kit_schema: str) -> None:
    """Revoke reader role access to raw synced tables: the version tables and each enrichment
    table present.

    Args:
        cursor: psycopg cursor
        kit_schema: Postgres version schema the synced tables land in (<kit_schema>_v<N>)
    """
    cursor.execute(f"REVOKE ALL ON SCHEMA {kit_schema} FROM reader")
    cursor.execute(f"REVOKE ALL ON {kit_schema}.nodes FROM reader")
    cursor.execute(f"REVOKE ALL ON {kit_schema}.edges FROM reader")
    cursor.execute(f"REVOKE ALL ON {kit_schema}.node_id_map FROM reader")
    for table in _present(cursor, kit_schema, versions.OPTIONAL_TABLES):
        cursor.execute(f"REVOKE ALL ON {kit_schema}.{table} FROM reader")
    # Functions get EXECUTE for PUBLIC by default; no raw function or sequence is reachable.
    cursor.execute(f"REVOKE ALL ON ALL FUNCTIONS IN SCHEMA {kit_schema} FROM PUBLIC, reader")
    cursor.execute(f"REVOKE ALL ON ALL SEQUENCES IN SCHEMA {kit_schema} FROM PUBLIC, reader")

    logger.info(f"Revoked access to raw tables from reader role")
