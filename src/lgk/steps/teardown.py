"""Whole-graph teardown: drop exactly what the run manifest names.

Order (verified live: a served view blocks `delete_synced_table`, and deleting a branch
does not cascade to UC synced tables):
  0. recorded jobs (the refresh job, the bundle's jobs): runs cancelled, job deleted, so
     nothing re-syncs or re-serves while the rest goes
  1. served views and functions (served and the staged copies in version schemas)
     (before them, every grant on the graph's objects is revoked from the shared roles)
  2. UC synced tables (`synced_tables/` prefix; this also deletes their pipelines), then each
     recorded synced table's pipeline's managed `event_log_<pipeline_id>` table in the
     destination UC schema (the pipeline id: recorded, else read from Unity Catalog before any
     delete; with neither, teardown refuses; lgk.event_logs)
  3. Postgres tables and schemas
  4. (tests only) the branch, when the test recorded one
  5. UC tables, then UC schemas
The roles `reader`, `public_reader` and `coordinator_reader` are cluster-wide and shared by
every graph on the branch (owner decision A6 (a)): teardown never drops them, even when an
older manifest recorded them; it only revokes this graph's grants from them.
EVERY check runs before ANY delete: each existing job must carry this graph's tag, each
existing schema must carry this graph's marker
and hold only recorded objects, and nothing outside may depend on them. All drops are
RESTRICT; a missing object counts as already gone, so a run that failed part way can be
re-run. Errors are raised as they are.
It never calls a project or endpoint API.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lgk import event_logs as el
from lgk import manifest as mf
from lgk import served_functions
from lgk import versions
from lgk.branch import BranchRef, delete_branch_retrying_cap
from lgk.manifest import Manifest, ManifestEntry, ObjectType
from lgk.uc_sql import warehouse_id

logger = logging.getLogger(__name__)


@dataclass
class Plan:
    """Everything teardown will drop, grouped by phase, in manifest order."""

    jobs: List[str] = field(default_factory=list)  # job ids (strings)
    views: List[str] = field(default_factory=list)  # "<pg schema>.<view>"
    functions: List[str] = field(default_factory=list)  # "<pg schema>.<fn>(<arg types>)"
    synced_tables: List[str] = field(default_factory=list)  # "<cat>.<schema>.<table>"
    pg_tables: List[str] = field(default_factory=list)  # the PG tables behind the synced tables
    pg_schemas: List[str] = field(default_factory=list)  # served (views) first, then versions
    # synced table -> its recorded pipeline id (None: not recorded); the event logs are resolved from it
    synced_pipelines: Dict[str, Optional[str]] = field(default_factory=dict)
    event_logs: List[str] = field(default_factory=list)  # "<cat>.<dest schema>.event_log_<pipeline id>"
    branches: List[str] = field(default_factory=list)  # tests only
    uc_tables: List[str] = field(default_factory=list)  # "<cat>.<schema>.<table>"
    uc_schemas: List[str] = field(default_factory=list)  # "<cat>.<schema>"


def plan_from_manifest(manifest: Manifest) -> Plan:
    """Group the manifest's entries into phases. Nothing outside the manifest is added.

    ROLE entries are skipped: older manifests recorded the shared roles, which are never dropped.
    """
    by_type: Dict[ObjectType, List[str]] = {}
    for e in manifest.entries:
        if e.object_type == ObjectType.ROLE:
            logger.info(f"Role {e.object_id} is shared by every graph on the branch; teardown never drops it")
            continue
        by_type.setdefault(e.object_type, []).append(e.object_id)
    synced = by_type.get(ObjectType.SYNCED_TABLE, [])
    views = by_type.get(ObjectType.VIEW, [])
    view_schemas = {v.rsplit(".", 1)[0] for v in views}
    pg_schemas = by_type.get(ObjectType.PG_SCHEMA, [])
    return Plan(
        jobs=by_type.get(ObjectType.JOB, []),
        views=views,
        functions=list(dict.fromkeys(by_type.get(ObjectType.FUNCTION, []))),
        synced_tables=synced,
        pg_tables=[s.split(".", 1)[1] for s in synced],
        pg_schemas=sorted(pg_schemas, key=lambda s: s not in view_schemas),
        synced_pipelines=_synced_pipelines(manifest.entries),
        branches=by_type.get(ObjectType.BRANCH, []),
        uc_tables=by_type.get(ObjectType.UC_TABLE, []),
        uc_schemas=by_type.get(ObjectType.UC_SCHEMA, []),
    )


def _synced_pipelines(entries: List[ManifestEntry]) -> Dict[str, Optional[str]]:
    """Each recorded synced table and the pipeline id provision recorded for it (None if none), in order."""
    return {e.object_id: (e.metadata or {}).get("pipeline_id") or None
            for e in entries if e.object_type == ObjectType.SYNCED_TABLE}


def teardown(config: Dict[str, Any], confirm_graph_name: str, ops: Optional[Any] = None,
             **unexpected: Any) -> Plan:
    """Drop every object the run manifest recorded for this graph, in the order listed above.

    `confirm_graph_name` is the graph name typed exactly. A misspelt keyword (e.g. `confirm_name`)
    is refused with an error naming the right keyword, before anything runs.
    """
    _refuse_unknown_keywords(unexpected)
    versions.confirm_graph_name(config, confirm_graph_name)
    plan = plan_from_manifest(mf.load_run_manifest(config))
    ops = ops or LiveOps(config)
    try:
        plan.event_logs = ops.resolve_event_logs(plan.synced_pipelines)  # a read; refuses before any delete
        _log_plan(config["graph_name"], plan)
        ops.check(plan, config["graph_name"])
        ops.delete_jobs(plan.jobs)
        ops.revoke_shared_roles(plan.pg_schemas, plan.views, plan.functions)
        ops.drop_views(plan.views, plan.functions)
        ops.delete_synced_tables(plan.synced_tables)
        ops.drop_event_logs(plan.event_logs)
        ops.drop_pg(plan.pg_tables, plan.pg_schemas)
        ops.delete_branches(plan.branches)
        ops.drop_uc(plan.uc_tables, plan.uc_schemas)
    finally:
        ops.close()
    logger.info(f"Teardown of graph {config['graph_name']} complete")
    return plan


def _refuse_unknown_keywords(unexpected: Dict[str, Any]) -> None:
    if unexpected:
        raise TypeError(f"teardown needs confirm_graph_name=<the graph name, typed exactly>; "
                        f"got unexpected keyword(s) {sorted(unexpected)}")


def _log_plan(graph: str, plan: Plan) -> None:
    logger.info(f"Teardown of graph {graph} will drop, in order:")
    for phase, items in vars(plan).items():
        for item in items:
            logger.info(f"  {phase}: {item}")


# ============================================================================
# Live operations
# ============================================================================

_RELATIONS_SQL = """
SELECT n.nspname || '.' || c.relname
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = %s AND c.relkind IN ('r', 'p', 'f', 'v', 'm') AND NOT c.relispartition
"""

class LiveOps:
    """Teardown phases against the run's branch Postgres and Unity Catalog."""

    def __init__(self, config: Dict[str, Any]):
        from databricks.sdk import WorkspaceClient

        self.config = config
        self.w = WorkspaceClient()
        self.warehouse_id = warehouse_id(config)
        self._conn: Any = None
        self._branch_alive: Optional[bool] = None

    # -- connection -----------------------------------------------------------

    def _branch_name(self) -> str:
        return BranchRef.from_config(self.config).path

    def _pg(self) -> Any:
        """The branch connection, or None when the branch is gone (its Postgres objects went with it)."""
        from databricks.sdk.errors import NotFound

        if self._branch_alive is None:
            try:
                self.w.postgres.get_branch(name=self._branch_name())
                self._branch_alive = True
            except NotFound:
                logger.info(f"Branch {self._branch_name()} is gone; its Postgres objects went with it")
                self._branch_alive = False
        if self._branch_alive and self._conn is None:
            from lgk.branch import connect_branch

            self._conn = connect_branch(self.config)
            self._conn.autocommit = True
        return self._conn if self._branch_alive else None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- checks (no deletes) --------------------------------------------------

    def resolve_event_logs(self, synced_pipelines: Dict[str, Optional[str]]) -> List[str]:
        """The recorded synced tables' own event-log tables (pipeline id recorded, else from UC)."""
        return el.own_event_logs(self.w, synced_pipelines)

    def check(self, plan: Plan, graph: str) -> None:
        # A UC-only manifest (build/enrich, no provision) never connects to any branch.
        conn = self._pg() if (plan.views or plan.functions or plan.pg_schemas or plan.pg_tables) else None
        if conn is not None:
            _check_pg_schemas(conn, plan, graph)
            versions.refuse_outside_dependents(conn, plan.pg_schemas)
        _check_uc_schemas(self.w, plan, graph)
        _check_jobs(self.w, plan, graph)

    # -- deletes, in order ----------------------------------------------------

    def delete_jobs(self, job_ids: List[str]) -> None:
        delete_jobs(self.w, job_ids)

    def revoke_shared_roles(self, schemas: List[str], views: List[str], functions: List[str]) -> None:
        """Revoke every right the shared roles hold on this graph's objects; the roles stay."""
        if not (schemas or views or functions):
            return
        conn = self._pg()
        if conn is None:
            return
        with conn.transaction():
            for stmt in revoke_shared_sql(conn, schemas, views, functions):
                conn.execute(stmt)
        logger.info(f"Revoked the shared roles' rights on {schemas}, {views} and {functions}")

    def drop_views(self, views: List[str], functions: List[str]) -> None:
        if not (views or functions):
            return
        conn = self._pg()
        if conn is None:
            return
        with conn.transaction():
            for view in views:
                conn.execute(f"DROP VIEW IF EXISTS {view} RESTRICT")
            for signature in functions:
                conn.execute(served_functions.drop_sql(signature))
        logger.info(f"Dropped served views {views} and functions {functions}")

    def delete_synced_tables(self, names: List[str]) -> None:
        from databricks.sdk.errors import NotFound

        for full in names:
            name = f"synced_tables/{full}"
            try:
                self.w.postgres.delete_synced_table(name=name).wait()
                logger.info(f"Deleted synced table {name} (and its pipeline)")
            except NotFound:
                logger.info(f"Synced table {name} already gone")

    def drop_event_logs(self, tables: List[str]) -> None:
        """After the pipelines are gone: each recorded synced table's own event-log table, and no other."""
        el.drop_event_logs(self.w, self.warehouse_id, tables)

    def drop_pg(self, tables: List[str], schemas: List[str]) -> None:
        if not (tables or schemas):
            return
        conn = self._pg()
        if conn is None:
            return
        with conn.transaction():
            for table in tables:
                conn.execute(f"DROP TABLE IF EXISTS {table} RESTRICT")
            for schema in schemas:
                conn.execute(f"DROP SCHEMA IF EXISTS {schema} RESTRICT")
        logger.info(f"Dropped Postgres tables {tables} and schemas {schemas}")

    def delete_branches(self, branches: List[str]) -> None:
        from databricks.sdk.errors import NotFound

        if branches:
            self.close()  # the connection is to a branch that is about to go
        for branch in branches:
            try:
                # parallel suites' teardowns hit the project operation cap; only that is retried
                delete_branch_retrying_cap(self.w, self.config["lakebase_project"], branch)
                logger.info(f"Deleted recorded test branch {branch}")
            except NotFound:
                logger.info(f"Branch {branch} already gone")

    def drop_uc(self, tables: List[str], schemas: List[str]) -> None:
        from lgk.uc_sql import execute_sql

        for table in tables:
            execute_sql(self.w, self.warehouse_id, f"DROP TABLE IF EXISTS {table}")
        for schema in schemas:
            execute_sql(self.w, self.warehouse_id, f"DROP SCHEMA IF EXISTS {schema} RESTRICT")
        logger.info(f"Dropped UC tables {tables} and schemas {schemas}")


def _check_pg_schemas(conn: Any, plan: Plan, graph: str) -> None:
    """Each existing recorded schema carries the marker and holds only recorded relations."""
    allowed = set(plan.views) | set(plan.pg_tables)
    for schema in plan.pg_schemas:
        row = conn.execute(mf.PG_SCHEMA_COMMENT_SQL, (schema,)).fetchone()
        if row is None:
            continue
        mf.refuse_unless_marked(f"Postgres schema {schema}", row[0], graph)
        foreign = sorted(r[0] for r in conn.execute(_RELATIONS_SQL, (schema,)).fetchall() if r[0] not in allowed)
        if foreign:
            raise versions.NotOursError(f"Postgres schema {schema} holds objects the manifest never recorded: {foreign}")


# The cluster-wide roles every graph on a branch shares (A6 (a)): revoked from, never dropped.
SHARED_ROLES = ("reader",) + served_functions.FUNCTION_ROLES


def revoke_shared_sql(conn: Any, schemas: List[str], views: List[str], functions: List[str]) -> List[str]:
    """REVOKE statements for the shared roles that exist, on exactly this graph's objects that exist.

    Schemas: the schema itself and everything in it (a checked schema holds only recorded objects).
    Views and functions: one by one, so a recorded object in a schema not recorded is covered too.
    Nothing outside the arguments is named.
    """
    roles = [r for r in SHARED_ROLES if conn.execute(_ROLE_EXISTS_SQL, (r,)).fetchone() is not None]
    if not roles:
        return []
    who = ", ".join(roles)
    stmts: List[str] = []
    for schema in schemas:
        if conn.execute(_SCHEMA_EXISTS_SQL, (schema,)).fetchone() is None:
            continue
        stmts.append(f"REVOKE ALL ON SCHEMA {schema} FROM {who}")
        for kind in ("TABLES", "FUNCTIONS", "SEQUENCES"):
            stmts.append(f"REVOKE ALL ON ALL {kind} IN SCHEMA {schema} FROM {who}")
    for view in views:
        if _present(conn, "SELECT to_regclass(%s)", view):
            stmts.append(f"REVOKE ALL ON {view} FROM {who}")
    for signature in functions:
        if _present(conn, "SELECT to_regprocedure(%s)", signature):
            stmts.append(f"REVOKE ALL ON FUNCTION {signature} FROM {who}")
    return stmts


_ROLE_EXISTS_SQL = "SELECT 1 FROM pg_roles WHERE rolname = %s"
_SCHEMA_EXISTS_SQL = "SELECT 1 FROM pg_namespace WHERE nspname = %s"


def _present(conn: Any, sql: str, name: str) -> bool:
    row = conn.execute(sql, (name,)).fetchone()
    return row is not None and row[0] is not None


def _check_uc_schemas(w: Any, plan: Plan, graph: str) -> None:
    """Each existing recorded UC schema carries the marker and holds only recorded tables."""
    # a recorded pipeline's event log is ours; another pipeline's is not
    allowed = set(plan.uc_tables) | set(plan.synced_tables) | set(plan.event_logs)
    for full in plan.uc_schemas:
        exists, comment = mf.uc_schema_comment(w, full)
        if not exists:
            continue
        mf.refuse_unless_marked(f"UC schema {full}", comment, graph)
        catalog, schema = full.split(".", 1)
        names = sorted(f"{full}.{t.name}" for t in w.tables.list(catalog, schema))
        foreign = [n for n in names if n not in allowed]
        if foreign:
            raise versions.NotOursError(f"UC schema {full} holds tables the manifest never recorded: {foreign}")


def job_settings(w: Any, job_id: str) -> Optional[Any]:
    """The job's settings, or None when the job is gone (NotFound, or the jobs API's "does not exist")."""
    from databricks.sdk.errors import InvalidParameterValue, NotFound

    try:
        return w.jobs.get(job_id=int(job_id)).settings
    except NotFound:
        return None
    except InvalidParameterValue as e:
        if "does not exist" in str(e):
            return None
        raise


def _check_jobs(w: Any, plan: Plan, graph: str) -> None:
    """Each existing recorded job carries this graph's tag (a copied manifest cannot delete a stranger's job)."""
    for job_id in plan.jobs:
        settings = job_settings(w, job_id)
        if settings is None:
            continue
        tag = (getattr(settings, "tags", None) or {}).get(mf.JOB_TAG)
        if tag != graph:
            raise versions.NotOursError(f"job {job_id} is tagged {mf.JOB_TAG}={tag!r}, not {graph!r}; refusing to delete it")


def delete_jobs(w: Any, job_ids: List[str]) -> None:
    """Cancel every run of each existing job, then delete it. A job already gone counts as done."""
    for job_id in job_ids:
        if job_settings(w, job_id) is None:
            logger.info(f"Job {job_id} already gone")
            continue
        w.jobs.cancel_all_runs(job_id=int(job_id))
        w.jobs.delete(job_id=int(job_id))
        logger.info(f"Deleted job {job_id} (its runs cancelled)")
