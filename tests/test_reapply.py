"""Reapply's index list, given the PK index a synced table already has.

A synced table's primary key already gives a unique btree on its PK columns, in PK order, so an
index on the LEADING PK column is redundant; any other column (including a non-leading PK
column) still needs its own index. The recorded spike result below is the input; the expected
list is derived from it, never written out by hand.
"""

import contextlib

from lgk.steps.reapply import INDEXES, ensure_indexes

# Recorded live result (2026-10-04, a test branch, synced Delta tables): a PK index exists on the
# leading PK column of every synced table. The table below is what that run read, so this test
# needs no live run.
# `pg_indexes` on the version schema: table -> its `<table>_pkey` columns, in PK order.
STYPES_PK_INDEXES = {
    "node_group": ("node_id", "method"),
    "similar_to": ("src_id", "dst_id"),
    "closure": ("ancestor_id", "descendant_id"),
    "node_score": ("node_id", "method"),
    "nodes": ("node_id",),
    "edges": ("edge_id",),
    "node_id_map": ("node_id",),
}

# The (table, column) pairs the walks and lookups filter on (before the PK pruning): the input
# the pruning starts from.
CANDIDATES = (
    ("edges", "src_node_id"),
    ("edges", "dst_node_id"),
    ("closure", "ancestor_id"),
    ("closure", "descendant_id"),
    ("node_group", "node_id"),
    ("node_group", "group_id"),
    ("similar_to", "src_id"),
)


def _pk_covers(table: str, column: str) -> bool:
    return STYPES_PK_INDEXES[table][0] == column


class _Conn:
    def __init__(self, tables):
        self.tables, self.log = tables, []

    def transaction(self):
        return contextlib.nullcontext()

    def execute(self, sql, params=None):
        self.log.append(sql)
        rows = [(t,) for t in self.tables] if "FROM pg_class" in sql else []
        return type("R", (), {"fetchall": lambda self: rows})()


def test_reapply_keeps_every_index_the_pk_does_not_cover():
    kept = [(t, c) for t, c in CANDIDATES if not _pk_covers(t, c)]
    pruned = [(t, c) for t, c in CANDIDATES if _pk_covers(t, c)]
    # Non-vacuous: the spike result prunes some candidates and keeps a non-leading PK column.
    assert pruned and ("closure", "descendant_id") in kept
    assert [(t, c) for _, t, c in INDEXES] == kept
    assert [n for n, _, _ in INDEXES] == [f"lgk_{t}_{c}_idx" for t, c in kept]  # fixed names


def test_reapply_creates_no_index_the_pk_already_gives():
    # hostile: every enrichment table is present, so only the list decides
    conn = _Conn(["nodes", "edges", "node_id_map", "node_score", "node_group", "similar_to", "closure"])
    ensure_indexes(conn, "k_v1")
    made = [s for s in conn.log if s.startswith("CREATE INDEX")]
    on = [s.split(" ON k_v1.")[1] for s in made]
    for table, column in CANDIDATES:
        assert (f"{table} ({column})" in on) == (not _pk_covers(table, column)), (table, column)
    assert len(made) == len(set(made)) == sum(not _pk_covers(t, c) for t, c in CANDIDATES)
