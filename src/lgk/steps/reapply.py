"""Reapply: the refresh job's task after the sync. Safe to re-run; a no-op after a normal sync.

0. Privacy check: the one switch runs, `binding.require_servable`:
   the gate is clear for the current, newest privacy decision and this job's version was
   provisioned under it, else StaleBuildError / PrivacyGateError before it connects. This holds
   when the served schema is empty too (before any switch), where step 0b has nothing to compare.
0b. Served-version check: the served views still read this job's version, else the task fails
   before anything (a switch made after the job, e.g. a rollback, is never undone overnight).
1. Freshness guard: every synced table holds its source's current Delta version, else
   the task (and so the job) fails loudly before anything is served. Then the fold
   floor detector: no folded edge of the version table is NULL or below the floor.
2. Indexes on the version tables: fixed names, CREATE INDEX IF NOT EXISTS (an index on the
   partitioned parent persists through syncs and applies to new partitions), plus one
   partial expression index per declared walk key on `nodes`.
3. Served views: the same one-transaction CREATE OR REPLACE as provision (same OID and
   definition when nothing changed).
4. Grants: provision's reader grant block (GRANT of a held privilege, REVOKE of an absent one,
   change nothing).
"""

import logging
from typing import Any, Dict, Iterator, Optional, Tuple

from lgk import versions
from lgk.refresh.freshness import check_freshness
from lgk.branch import connect_branch
from lgk.privacy import binding, floor
from lgk.mapping import GraphMapping
from lgk.privacy.gate import fold_policy, read_run_mapping, require_clear, required_run_folder
from lgk.steps.provision import _create_served_views, _setup_reader_role
from lgk.walks.sql import walk_key_index_sql

logger = logging.getLogger(__name__)

# (index name, version table, column). Names are fixed so IF NOT EXISTS makes re-runs no-ops.
# The enrichment indexes serve the lookups.
# showed every synced table gets a unique `<table>_pkey` btree on its PK columns in PK order,
# so the PK's LEADING column is already indexed: closure (ancestor_id), node_group (node_id)
# and similar_to (src_id) need no index of their own. Only the non-leading columns remain.
INDEXES = (
    ("lgk_edges_src_node_id_idx", "edges", "src_node_id"),
    ("lgk_edges_dst_node_id_idx", "edges", "dst_node_id"),
    ("lgk_closure_descendant_id_idx", "closure", "descendant_id"),
    ("lgk_node_group_group_id_idx", "node_group", "group_id"),
)

_TABLES_SQL = """
SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = %s AND c.relkind IN ('r', 'p') AND NOT c.relispartition
"""


def walk_key_pairs(mapping: Optional[GraphMapping]) -> Iterator[Tuple[str, str]]:
    """(node type, walk key) for every declared walk key, in mapping order."""
    if mapping is not None:
        for node_type, spec in mapping.nodes.items():
            for key in spec.walk_keys:
                yield node_type, key


def ensure_indexes(conn: Any, version_schema: str, mapping: Optional[GraphMapping] = None) -> None:
    """Create the walk and lookup indexes on the version tables present, unless they exist.

    An enrichment table the version does not hold (not computed, or closure not served) gets
    no index: its CREATE INDEX would fail the reapply. With `mapping`, every declared walk key
    also gets its partial expression index on `nodes` (`ensure_walk_key_indexes`).
    """
    with conn.transaction():
        present = {str(r[0]) for r in conn.execute(_TABLES_SQL, (version_schema,)).fetchall()}
        for name, table, column in INDEXES:
            if table in present:
                conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {version_schema}.{table} ({column})")
    ensure_walk_key_indexes(conn, version_schema, mapping)


def ensure_walk_key_indexes(conn: Any, version_schema: str, mapping: Optional[GraphMapping]) -> None:
    """One partial expression index per declared walk key (lgk.walks.sql.walk_key_index_sql)."""
    pairs = list(walk_key_pairs(mapping))
    if not pairs:
        return
    with conn.transaction():
        for node_type, key in pairs:
            conn.execute(walk_key_index_sql(version_schema, node_type, key))


def _reapply_indexes(config: Dict[str, Any]) -> None:
    mapping = read_run_mapping(required_run_folder(config))
    conn = connect_branch(config)
    try:
        conn.autocommit = True
        ensure_indexes(conn, versions.version_schema(config), mapping)
    finally:
        conn.close()


def _detect_unfloored_rows(config: Dict[str, Any]) -> None:
    """No folded row of the version's `edges` is NULL or below the floor, else FoldFloorError.

    A detector: the sync task's Spark check is what stops unfloored rows; this catches a sync
    that bypassed the job. Floor and fold types come from the cleared gate."""
    fold_floor, fold_types = fold_policy(require_clear(required_run_folder(config)))
    conn = connect_branch(config)
    try:
        floor.assert_fold_floor_pg(conn, f"{versions.version_schema(config)}.edges", fold_types, fold_floor)
    finally:
        conn.close()


def _refuse_a_moved_site(config: Dict[str, Any]) -> None:
    """The served views still read this job's version, else ServedVersionError (a rollback stays)."""
    conn = connect_branch(config)
    try:
        versions.refuse_unless_serving(conn, versions.served_schema(config), versions.version_schema(config))
    finally:
        conn.close()


def _refuse_an_unservable_version(config: Dict[str, Any]) -> None:
    """Switch's privacy check for this job's version (binding.require_servable), before any connection."""
    binding.require_servable(required_run_folder(config), versions.version_schema(config))


def reapply(config: Dict[str, Any]) -> None:
    """The privacy check, the served-version check, the freshness guard, the fold floor
    detector, then indexes, views and grants for the run's version."""
    _refuse_an_unservable_version(config)
    _refuse_a_moved_site(config)
    check_freshness(config)
    _detect_unfloored_rows(config)
    _reapply_indexes(config)
    _create_served_views(config)
    _setup_reader_role(config)
    logger.info(f"reapply done for {versions.version_schema(config)}")
