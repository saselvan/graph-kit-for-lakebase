"""Versions behind stable views and the production rule.

- Each build goes into a version schema, default `lgk_<graph>_v<N>` (the Postgres schema
  its synced tables land in, named after their UC schema).
- The stable served schema holds only views over one version.
- The switch replaces every served view in ONE Postgres transaction, so the website
  sees either the old version or the new one, never a mix. Rollback is a switch back.
- "Live" is what the served views actually reference in the catalog (pg_depend), never
  a stored flag.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from lgk import ident

# Served view -> the version-schema table it reads. Switched in this order.
SERVED_VIEWS = (
    ("nodes_view", "nodes"),
    ("edges_view", "edges"),
    ("node_id_map_view", "node_id_map"),
)
VERSION_TABLES = tuple(table for _, table in SERVED_VIEWS)
# Served view -> the enrichment table it reads. Each is served only when its table
# exists in the target version; a switch to a version without it drops the view.
OPTIONAL_SERVED_VIEWS = (
    ("node_score_view", "node_score"),
    ("node_group_view", "node_group"),
    ("similar_to_view", "similar_to"),
    ("closure_view", "closure"),
)
OPTIONAL_TABLES = tuple(table for _, table in OPTIONAL_SERVED_VIEWS)
# Every table a version schema of this graph may hold (retire's and teardown's "ours" set).
OURS_TABLES = VERSION_TABLES + OPTIONAL_TABLES


class VersionExistsError(RuntimeError):
    """R2: the run's version schema already exists. Use a new version number."""


class LiveVersionError(RuntimeError):
    """The served views point at this version, so it cannot be dropped."""


class ServedVersionError(RuntimeError):
    """The served views read another version than this run's (a switch was made after the job)."""


class ConfirmationError(RuntimeError):
    """The graph name was not typed exactly to confirm a destructive step."""


class NotOursError(RuntimeError):
    """A schema to drop holds something other than this graph's version tables."""


_VERSION_SUFFIX = re.compile(r"_v\d+$")

# Postgres silently truncates identifiers to 63 bytes, which would let a long kit name and
# another graph's version schema collapse to the same schema. Cap names so that
# "<kit>_v<N>" always fits: N <= 9999 ("_v9999" = 6 bytes) and kit/served <= 57 bytes.
MAX_VERSION = 9999
MAX_NAME_BYTES = 63 - len(f"_v{MAX_VERSION}")


def _checked_name(value: str, what: str) -> str:
    """A plain lower-case identifier that does not end in _v<N>.

    The suffix rule keeps every kit/served name disjoint from every version schema name:
    graph "providers" v2 is `lgk_providers_v2`, which must never be another graph's
    kit or served schema (graph "providers_v2" would default to exactly that).
    """
    if (
        not isinstance(value, str)
        or not ident.is_valid(value)
        or _VERSION_SUFFIX.search(value)
        or len(value.encode()) > MAX_NAME_BYTES
    ):
        raise ValueError(
            f"{what} must match ^{ident.PATTERN}$, be at most {MAX_NAME_BYTES} bytes "
            f"and not end in _v<N>, got {value!r}"
        )
    return value


def kit_schema(config: Dict[str, Any]) -> str:
    graph = _checked_name(config["graph_name"], "graph_name")
    return _checked_name(config.get("kit_schema") or f"lgk_{graph}", "kit_schema")


def served_schema(config: Dict[str, Any]) -> str:
    """The stable schema the website reads. Defaults to the kit schema name."""
    return _checked_name(config.get("served_schema") or kit_schema(config), "served_schema")


def version_schema(config: Dict[str, Any], version: Optional[int] = None) -> str:
    """`<kit_schema>_v<N>`; N is `version`, else config["version"], else 1."""
    n = config.get("version", 1) if version is None else version
    if isinstance(n, bool) or not isinstance(n, int) or n < 1 or n > MAX_VERSION:
        raise ValueError(f"version must be an int in 1..{MAX_VERSION}, got {n!r}")
    return f"{kit_schema(config)}_v{n}"


def confirm_graph_name(config: Dict[str, Any], typed: str) -> None:
    if typed != config["graph_name"]:
        raise ConfirmationError(
            f"Type the graph name {config['graph_name']!r} exactly to confirm (got {typed!r})."
        )


def assert_version_schema_absent(conn: Any, schema: str) -> None:
    """R2: refuse a run whose version schema already exists on the branch."""
    row = conn.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (schema,)).fetchone()
    if row is not None:
        raise VersionExistsError(
            f"Version schema {schema} already exists. A run never rebuilds a version in place; "
            f"run with a new version number."
        )


_LIVE_SQL = """
SELECT DISTINCT tn.nspname
FROM pg_class v
JOIN pg_namespace vn ON vn.oid = v.relnamespace
JOIN pg_rewrite r ON r.ev_class = v.oid
JOIN pg_depend d ON d.objid = r.oid
  AND d.classid = 'pg_rewrite'::regclass AND d.refclassid = 'pg_class'::regclass
JOIN pg_class t ON t.oid = d.refobjid AND t.oid <> v.oid
JOIN pg_namespace tn ON tn.oid = t.relnamespace
WHERE vn.nspname = %s AND v.relkind = 'v'
"""


def live_version_schemas(conn: Any, served: str) -> Set[str]:
    """Schemas the served views actually read from, per the catalog's dependency rows."""
    return {row[0] for row in conn.execute(_LIVE_SQL, (served,)).fetchall()}


def refuse_if_live(conn: Any, served: str, schemas: list) -> None:
    """Raise LiveVersionError if any of `schemas` is referenced by a served view."""
    live = live_version_schemas(conn, served) & set(schemas)
    if live:
        raise LiveVersionError(
            f"Served views in {served} point at {sorted(live)}; switch to another version first."
        )


def refuse_unless_serving(conn: Any, served: str, target: str) -> None:
    """Raise ServedVersionError if the served views read anything but `target`.

    The nightly reapply re-serves the version its job was made for. After a switch made later
    (a rollback, or a switch forward to a newer version), that would move the views back
    silently, so the night fails instead. No served views yet (live is empty) passes: there is
    nothing to undo.
    """
    live = live_version_schemas(conn, served)
    if live and live != {target}:
        raise ServedVersionError(
            f"Served views in {served} read {sorted(live)}, not {target}: a switch was made after "
            f"this job was created. Reapply does not move the served views. Pause this job, or "
            f"create the refresh job again for the version you serve."
        )


_RELATIONS_SQL = """
SELECT c.relname, c.relkind
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = %s AND c.relkind IN ('r', 'p', 'f', 'v', 'm') AND NOT c.relispartition
"""


def refuse_unless_ours_pg(conn: Any, schema: str) -> None:
    """Refuse unless the Postgres schema holds only OURS_TABLES and no views.

    Partitions of a synced table, indexes and sequences are not checked. An absent
    schema passes: there is nothing to drop.
    """
    rows = conn.execute(_RELATIONS_SQL, (schema,)).fetchall()
    foreign = sorted(
        f"{name} ({kind})" for name, kind in rows if kind in ("v", "m") or name not in OURS_TABLES
    )
    if foreign:
        raise NotOursError(f"Postgres schema {schema} is not a version of this graph: {foreign}")


_OUTSIDE_DEPENDENTS_SQL = """
SELECT DISTINCT vn.nspname || '.' || v.relname
FROM pg_depend d
JOIN pg_rewrite r ON r.oid = d.objid AND d.classid = 'pg_rewrite'::regclass
JOIN pg_class v ON v.oid = r.ev_class
JOIN pg_namespace vn ON vn.oid = v.relnamespace
JOIN pg_class t ON t.oid = d.refobjid AND d.refclassid = 'pg_class'::regclass
JOIN pg_namespace tn ON tn.oid = t.relnamespace
WHERE tn.nspname = ANY(%s) AND t.oid <> v.oid AND NOT (vn.nspname = ANY(%s))
ORDER BY 1
"""


def outside_dependents(conn: Any, schemas: List[str]) -> List[str]:
    """Views outside `schemas` that read a relation inside them (a RESTRICT drop would fail)."""
    return [row[0] for row in conn.execute(_OUTSIDE_DEPENDENTS_SQL, (list(schemas), list(schemas))).fetchall()]


def refuse_outside_dependents(conn: Any, schemas: List[str]) -> None:
    """Checked before any delete, so a RESTRICT failure can never leave a half-deleted version."""
    found = outside_dependents(conn, schemas)
    if found:
        raise NotOursError(f"Objects outside {sorted(schemas)} depend on them: {found}")


def refuse_unless_ours_uc(table_names: List[str], full_schema: str, event_logs: Sequence[str] = ()) -> None:
    """Refuse unless the UC schema holds only OURS_TABLES (its synced tables) and `event_logs`.

    `event_logs` are the bare names of this version's own pipelines' event-log tables
    (lgk.event_logs); any other table, including another pipeline's event log, is foreign.
    """
    foreign = sorted(set(table_names) - set(OURS_TABLES) - set(event_logs))
    if foreign:
        raise NotOursError(f"UC schema {full_schema} is not a version of this graph: {foreign}")


_COLUMNS_SQL = """
SELECT a.attname, format_type(a.atttypid, a.atttypmod)
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = %s AND c.relname = %s AND c.relkind IN ('r', 'p', 'f', 'v', 'm')
  AND a.attnum > 0 AND NOT a.attisdropped
ORDER BY a.attnum
"""

Columns = List[Tuple[str, str]]


def relation_columns(conn: Any, schema: str, name: str) -> Columns:
    """(name, type) of each column of `schema.name`, in order; empty if there is no such relation."""
    return [(str(n), str(t)) for n, t in conn.execute(_COLUMNS_SQL, (schema, name)).fetchall()]


def view_statements(served: str, view: str, target: str, table: str, live: Columns,
                    target_cols: Columns, required: bool) -> List[str]:
    """The X3 switch rule for one served view (run inside the one switch transaction).

    1. an optional view whose table is absent from `target`: DROP VIEW IF EXISTS ... RESTRICT;
    2. the live view's columns are a prefix of the target's (name, type, order), or there is no
       live view: CREATE OR REPLACE VIEW (Postgres may append columns, never drop or retype);
    3. otherwise: DROP VIEW ... RESTRICT, then CREATE VIEW. RESTRICT makes an outside dependent
       fail the switch loudly instead of being dropped.
    A required view whose table is absent takes path 2 or 3, and its CREATE fails the switch.
    """
    body = f"VIEW {served}.{view} WITH (security_invoker=off) AS SELECT * FROM {target}.{table}"
    if not target_cols and not required:
        return [f"DROP VIEW IF EXISTS {served}.{view} RESTRICT"]
    if live == target_cols[: len(live)]:
        return [f"CREATE OR REPLACE {body}"]
    return [f"DROP VIEW {served}.{view} RESTRICT", f"CREATE {body}"]


def reader_grant_sql(served: str, target: str, views: Sequence[str], tables: Sequence[str],
                     role: str = "reader") -> List[str]:
    """The reader's grant block: the served schema, SELECT on each served view, nothing on a table.

    An absent optional view has no GRANT line; every `target` table behind a served view is revoked.
    """
    return ([f"GRANT USAGE ON SCHEMA {served} TO {role}"]
            + [f"GRANT SELECT ON {served}.{v} TO {role}" for v in views]
            + [f"REVOKE ALL ON {target}.{t} FROM {role}" for t in tables])


def _switch_each_view(conn: Any, served: str, target: str) -> List[Tuple[str, str]]:
    """Repoint every served view; return the (view, table) pairs served afterwards."""
    served_now = []
    for view, table in SERVED_VIEWS + OPTIONAL_SERVED_VIEWS:
        required = (view, table) in SERVED_VIEWS
        target_cols = relation_columns(conn, target, table)
        for stmt in view_statements(served, view, target, table, relation_columns(conn, served, view),
                                    target_cols, required):
            conn.execute(stmt)
        if required or target_cols:
            served_now.append((view, table))
    return served_now


def switch_views(conn: Any, served: str, target: str, functions_sql: Sequence[str] = (),
                 verify: Optional[Callable[[Any], None]] = None,
                 reader: Optional[str] = None) -> List[str]:
    """Repoint every served view at `target` in one transaction (all or nothing); return the
    served views.

    Each view follows a rule (`view_statements`): an optional view the target lacks is
    dropped, a view whose columns the target only extends is replaced, any other is dropped
    and re-created. Views keep `security_invoker` off. With `reader`, that role's grant
    block (`reader_grant_sql`) runs in the SAME transaction, so no committed state has an
    ungranted view (a re-created view has lost its grants).
    `functions_sql` (lgk.served_functions.switch_statements) repoints the served functions
    and re-runs their grant block in the same transaction, and `verify(conn)` (the function
    roles' isolation probe) runs last, before the commit. If any statement or the probe
    fails (for example a missing table in a broken build), the transaction rolls back and
    every view and function still points at the previous version.
    """
    import psycopg

    if conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
        # Inside an open transaction this block would only be a savepoint, and its
        # commit would depend on the caller.
        raise RuntimeError("switch_views must start outside a transaction")
    with conn.transaction():
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {served}")
        served_now = _switch_each_view(conn, served, target)
        if reader is not None:
            for stmt in reader_grant_sql(served, target, [v for v, _ in served_now],
                                         [t for _, t in served_now], reader):
                conn.execute(stmt)
        for stmt in functions_sql:
            conn.execute(stmt)
        if verify is not None:
            verify(conn)
    return [v for v, _ in served_now]
