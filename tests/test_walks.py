"""Offline checks of the walk engine.

String checks pin the SQL contract. The behaviour checks run the real SQL on a
throwaway local Postgres cluster (initdb in a temp dir, unix socket only); they skip when
no local Postgres binaries are installed. The answers against Spark are proven live in
tests/test_walks_live.py.
"""

import inspect
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from lgk.steps.walks import get_walk_query
from lgk.walks.sql import per_depth_sql, similar_sql, tree_sql, web_sql

KW = {"served_schema": "lgk_served", "enrich_schema": "lgk_g_v1", "sort_method": "degree"}


def _limits_follow_order_by(sql: str) -> bool:
    """R14: every LIMIT is directly preceded by an ORDER BY ending in the node_id tie-break."""
    import re

    parts = sql.split("LIMIT")[:-1]
    return bool(parts) and all(
        "ORDER BY" in p and re.search(r'node_id COLLATE "C"\s*$', p.rsplit("ORDER BY", 1)[1])
        for p in parts
    )


# ---------------------------------------------------------------------------- SQL contract


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_web_has_one_limited_ordered_cte_per_hop(depth):
    sql = web_sql(depth=depth, top_n=3, **KW)
    assert sql.count("LIMIT 3") == depth
    assert _limits_follow_order_by(sql)


def test_web_depth_cap_is_4():
    with pytest.raises(ValueError):
        web_sql(depth=5, top_n=3, **KW)
    with pytest.raises(ValueError):
        per_depth_sql("web", 5, top_n=3, **KW)


def test_tree_ordered_limit_and_cap_range():
    assert _limits_follow_order_by(tree_sql(depth=9, top_n=5, **KW))
    with pytest.raises(ValueError):
        tree_sql(depth=101, top_n=5, **KW)
    # the per-level cut sits inside the recursive arm, before the next level expands
    # (the live hub test proves it bounds each level; this pins where it is).
    recursive = tree_sql(depth=9, top_n=7, **KW).split("UNION ALL", 1)[1].split("reached AS", 1)[0]
    assert "LIMIT 7" in recursive and recursive.index("ORDER BY") < recursive.index("LIMIT 7")


def test_similar_orders_by_distance_then_node_id():
    sql = similar_sql("s.t", top_n=4)
    assert _limits_follow_order_by(sql)
    assert "<=>" in sql
    with pytest.raises(ValueError):
        similar_sql("s.t", top_n=4, depth=2)


def test_per_depth_sql_covers_every_depth():
    sqls = per_depth_sql("tree", 6, top_n=5, **KW)
    assert sorted(sqls) == [1, 2, 3, 4, 5, 6]
    assert sqls == {d: tree_sql(depth=d, top_n=5, **KW) for d in range(1, 7)}
    assert per_depth_sql("similar", 1, vector_table="s.t", top_n=2)[1] == similar_sql("s.t", top_n=2)


def test_identifiers_are_validated():
    with pytest.raises(ValueError):
        web_sql(depth=1, top_n=3, served_schema="x; DROP SCHEMA y")
    with pytest.raises(ValueError):
        web_sql(depth=1, top_n=3, **{**KW, "sort_method": "degree' OR 1=1"})


def test_folded_walk_takes_count_and_floor_from_privacy():
    """Folded: the served count is the weight and the checked floor hides low counts."""
    sql = web_sql(depth=2, top_n=3, served_schema="lgk_served", folded=True)
    assert sql.count("HAVING MIN(e.weight) >= 11") == 2
    assert "HAVING MIN(e.weight) >= 12" in web_sql(depth=1, top_n=3, served_schema="lgk_served",
                                                    folded=True, privacy_floor=12)
    with pytest.raises(ValueError):  # never lowerable
        web_sql(depth=1, top_n=3, served_schema="lgk_served", folded=True, privacy_floor=10)
    with pytest.raises(NotImplementedError):
        tree_sql(depth=2, top_n=3, served_schema="lgk_served", folded=True)


def test_unfolded_walk_has_no_floor():
    assert "HAVING" not in web_sql(depth=4, top_n=3, served_schema="lgk_served")


def test_get_walk_query_needs_the_gate_or_an_explicit_folded():
    from lgk.privacy.gate import PrivacyGateError

    with pytest.raises(PrivacyGateError):
        get_walk_query("x", 2, "lgk_served")


def test_get_walk_query_is_the_engine_web_sql():
    assert get_walk_query("ignored", 3, "lgk_served", top_n=10, folded=False) == web_sql(
        depth=3, served_schema="lgk_served", top_n=10)


# ---------------------------------------------------------------------------- local Postgres

PG_BIN = next((Path(p) for p in ("/opt/homebrew/opt/postgresql@17/bin", "/opt/homebrew/opt/postgresql@16/bin",
                                 "/usr/lib/postgresql/17/bin", "/usr/lib/postgresql/16/bin")
               if (Path(p) / "initdb").exists()), None) or (
    Path(shutil.which("initdb")).parent if shutil.which("initdb") else None)


@pytest.fixture(scope="module")
def local_pg():
    """A throwaway Postgres cluster on a unix socket, with lgk_served.edges_view as a table."""
    if PG_BIN is None:
        pytest.skip("no local Postgres binaries (initdb) installed")
    import psycopg

    tmp = Path(tempfile.mkdtemp(prefix="lgkpg"))
    data, sock = tmp / "data", tmp / "s"
    sock.mkdir()
    subprocess.run([PG_BIN / "initdb", "-D", data, "-U", "postgres", "--no-sync", "-E", "UTF8",
                    "--locale=C"], check=True, capture_output=True)
    subprocess.run([PG_BIN / "pg_ctl", "-D", data, "-w", "-l", tmp / "log", "-o",
                    f"-k {sock} -c listen_addresses='' -p 54329", "start"], check=True,
                   capture_output=True)
    conn = None
    try:
        conn = psycopg.connect(host=str(sock), port=54329, user="postgres", dbname="postgres",
                               autocommit=True)
        conn.execute("CREATE SCHEMA lgk_served")
        conn.execute("CREATE TABLE lgk_served.edges_view (src_node_id text, dst_node_id text, weight bigint, "
                     "relationship_type text)")
        conn.execute("CREATE SCHEMA lgk_g_v1")
        conn.execute("CREATE TABLE lgk_g_v1.node_score (node_id text, method text, score float8)")
        yield conn
    finally:
        if conn is not None:
            conn.close()
        subprocess.run([PG_BIN / "pg_ctl", "-D", data, "-m", "immediate", "stop"], capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)


def _load(conn, edges):
    """(src, dst[, weight[, relationship_type]]) rows; weight defaults to NULL, the type to 'e'."""
    conn.execute("TRUNCATE lgk_served.edges_view")
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO lgk_served.edges_view VALUES (%s, %s, %s, %s)",
                        [(*e, None, "e")[:4] if len(e) == 2 else (*e, "e")[:4] for e in edges])


def _tree(conn, depth, seed, top_n=1000, timeout="10s"):
    conn.execute(f"SET statement_timeout = '{timeout}'")
    rows = conn.execute(tree_sql(depth=depth, top_n=top_n, served_schema="lgk_served"),
                        {"start_node_id": seed}).fetchall()
    return [(n, d) for n, d, _score in rows]


def test_default_web_walk_returns_rows(local_pg):
    """B1 regression: with deduped edges every support is 1; the default walk is not blank."""
    # hostile: support 1 everywhere — any count floor > 1 blanks every hop
    _load(local_pg, [("s", "a"), ("s", "b"), ("a", "c"), ("b", "d"), ("c", "e")])
    rows = local_pg.execute(get_walk_query("s", 2, "lgk_served", top_n=25, folded=False),
                            {"start_node_id": "s"}).fetchall()
    assert [(h, n, sup) for h, n, sup, _ in rows] == [(1, "a", 1), (1, "b", 1), (2, "c", 1), (2, "d", 1)]


def test_folded_web_walk_hides_counts_below_the_floor(local_pg):
    # hostile: 10 = floor - 1, and a NULL weight that must never be served unweighted
    _load(local_pg, [("s", "a", 10), ("s", "b", 11), ("s", "c", 12), ("s", "d", None)])
    rows = local_pg.execute(web_sql(depth=1, top_n=25, served_schema="lgk_served", folded=True),
                            {"start_node_id": "s"}).fetchall()
    assert [(n, sup) for _h, n, sup, _ in rows] == [("c", 12), ("b", 11)]


def test_web_default_sort_is_support_then_node_id(local_pg):
    # hostile: z has the most support from hop 1; a and b tie on support
    _load(local_pg, [("s", "p"), ("s", "q"), ("p", "z"), ("q", "z"), ("p", "b"), ("q", "a")])
    rows = local_pg.execute(web_sql(depth=2, top_n=2, served_schema="lgk_served"),
                            {"start_node_id": "s"}).fetchall()
    assert [(h, n, sup) for h, n, sup, _ in rows] == [(1, "p", 1), (1, "q", 1), (2, "z", 2), (2, "a", 1)]


def test_tree_cycle_terminates_and_excludes_seed(local_pg):
    _load(local_pg, [("a", "b"), ("b", "c"), ("c", "a")])  # hostile: cycle
    for depth in (3, 4, 100):
        assert _tree(local_pg, depth, "a") == [("b", 1), ("c", 2)]


def test_tree_keeps_smallest_depth(local_pg):
    # hostile: shortcut s->d makes d depth 1, the long path makes it depth 3
    _load(local_pg, [("s", "a"), ("a", "b"), ("b", "d"), ("s", "d"), ("d", "e")])
    assert _tree(local_pg, 4, "s") == [("a", 1), ("d", 1), ("b", 2), ("e", 2)]


def test_tree_diamond_chain_does_not_explode(local_pg):
    """# hostile: 30 chained diamonds = 2^30 simple paths; a path walk would never finish."""
    k = 30
    edges = []
    for i in range(k):
        edges += [(f"d{i}", f"l{i}"), (f"d{i}", f"r{i}"), (f"l{i}", f"d{i + 1}"), (f"r{i}", f"d{i + 1}")]
    _load(local_pg, edges)
    t0 = time.monotonic()
    got = _tree(local_pg, 2 * k, "d0", timeout="5s")
    assert time.monotonic() - t0 < 2.0
    want = sorted([(f"l{i}", 2 * i + 1) for i in range(k)] + [(f"r{i}", 2 * i + 1) for i in range(k)]
                  + [(f"d{i}", 2 * i) for i in range(1, k + 1)], key=lambda r: (r[1], r[0]))
    assert got == want


def _recursion_levels(conn, sql, seed) -> int:
    plan = conn.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}", {"start_node_id": seed}).fetchone()[0]
    plan = plan if isinstance(plan, list) else json.loads(plan)

    def find(node):
        if node.get("Node Type") == "Recursive Union":
            return node["Actual Rows"]
        for child in node.get("Plans", []):
            found = find(child)
            if found is not None:
                return found
        return None

    return find(plan[0]["Plan"])


def test_tree_depth_cap_stops_the_recursion(local_pg):
    """The cap is inside the recursion: on a 500-node chain, depth 3 runs 4 levels, not 500."""
    # hostile: a long chain — a post-filter cap returns the same rows but recurses 500 times
    _load(local_pg, [(f"n{i}", f"n{i + 1}") for i in range(500)])
    sql = tree_sql(depth=3, top_n=1000, served_schema="lgk_served")
    assert _tree(local_pg, 3, "n0") == [("n1", 1), ("n2", 2), ("n3", 3)]
    assert _recursion_levels(local_pg, sql, "n0") <= 4


# ---------------------------------------------------------------- each node once -
#
# A property test on the Postgres output alone: a web walk returns each node_id at most once
# across all hops, never the seed, and a node cut by one hop's LIMIT may come back once, later.
# Parametrised over the walk direction now; `inbound` and `both` skip with a named reason until
# the walk-direction change adds `web_sql(direction=...)`, and then run with no edit here.

DIRECTIONS = ("outbound", "inbound", "both")
DIRECTION_SKIP = "direction not built yet: web_sql(direction=...) is missing"
UNIQUE_TOP_N = 3
UNIQUE_SEEDS = {"outbound": ("a",), "inbound": ("i_a",), "both": ("a", "i_a")}
_OUT_EDGES = [
    ("a", "b"),
    ("a", "b"),  # hostile: duplicate edge (support 2), one node in the result all the same
    ("a", "a"),  # hostile: self-loop on the seed
    *[("a", f"d{k}") for k in range(1, 6)],  # hostile: re-reachable after a cut (top_n=3 cuts d3 at hop 1)
    ("b", "c"),
    ("b", "d3"),  # ... and d3 is reachable again at hop 2, through b
    ("b", "b2"),
    ("d1", "b2"),  # hostile: two parents in one hop (b, d1 -> b2); b2 sorts first, so a row per
    #              parent would put b2 twice inside the cut
    ("c", "b"),  # hostile: two-back cycle b<->c (b at hop 1, a candidate again at hop 3)
    ("c", "a"),  # hostile: back to seed
    ("c", "e"),
    ("e", "b"),  # hostile: three-back (b is a candidate again at hop 4)
    ("e", "f"),
]
# The inbound twins: the same edges read in reverse, on their own i_ nodes, so an inbound walk
# from i_a meets every trap above. # hostile: two-back cycle, inbound (i_b<->i_c);
# # hostile: back to seed, inbound (i_a -> i_c); # hostile: re-reachable after a cut, inbound
# ({i_b, i_d1..i_d5} -> i_a, with i_d3 -> i_b).
_IN_EDGES = [(f"i_{dst}", f"i_{src}") for src, dst in _OUT_EDGES]


def direction_skip_reason(builder, direction: str):
    """The skip reason for a direction the walk builder cannot take yet, else None."""
    if direction != "outbound" and "direction" not in inspect.signature(builder).parameters:
        return DIRECTION_SKIP
    return None


def _web_for(builder, direction: str, **kw) -> str:
    """The web SQL for `direction`; today's builder (no `direction`) is called as it is."""
    if "direction" in inspect.signature(builder).parameters:
        kw["direction"] = direction
    return builder(**kw)


def assert_each_node_once(rows, seed_id: str, depth: int, top_n: int) -> None:
    """The three properties, on the walk's own rows: unique node_ids, no seed, hops in range."""
    ids = [str(r[1]) for r in rows]
    assert len(ids) == len(set(ids)), f"seed {seed_id} depth {depth}: a node repeats in {rows}"
    assert seed_id not in ids, f"seed {seed_id} depth {depth}: the seed came back in {rows}"
    hops = [int(r[0]) for r in rows]
    assert all(1 <= h <= depth for h in hops), f"seed {seed_id} depth {depth}: hops {hops}"
    assert all(hops.count(h) <= top_n for h in set(hops)), f"seed {seed_id} depth {depth}: {hops}"


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_web_returns_each_node_once_offline(local_pg, depth, direction):
    reason = direction_skip_reason(web_sql, direction)
    if reason:
        pytest.skip(reason)
    _load(local_pg, _OUT_EDGES + _IN_EDGES)
    local_pg.execute("SET statement_timeout = '10s'")
    for seed in UNIQUE_SEEDS[direction]:
        prefix = seed[:-1]  # "" or "i_"
        sql = _web_for(web_sql, direction, depth=depth, served_schema="lgk_served", top_n=UNIQUE_TOP_N)
        rows = local_pg.execute(sql, {"start_node_id": seed}).fetchall()
        assert rows, f"seed {seed} depth {depth}: empty walk"
        assert_each_node_once(rows, seed, depth, UNIQUE_TOP_N)
        cut = [int(h) for h, n, *_ in rows if n == f"{prefix}d3"]
        # d3 is cut at hop 1 (top_n=3), so it is not seen there, and comes back once at hop 2.
        assert cut == ([] if depth == 1 else [2]), f"seed {seed} depth {depth}: d3 at hops {cut}"


_REAL_WEB_SQL = web_sql


def _web_sql_with_direction(depth, served_schema, top_n, direction="outbound", **kw):
    """A stand-in for the direction builder: inbound swaps the edge columns, both reads a two-way view."""
    if direction == "both":
        return _REAL_WEB_SQL(depth=depth, served_schema=served_schema, top_n=top_n,
                       edges_relation="edges_both_way", **kw)
    sql = _REAL_WEB_SQL(depth=depth, served_schema=served_schema, top_n=top_n, **kw)
    if direction == "inbound":
        sql = (sql.replace("e.src_node_id", "e.__src").replace("e.dst_node_id", "e.src_node_id")
               .replace("e.__src", "e.dst_node_id"))
    return sql


@pytest.mark.parametrize("direction", ["inbound", "both"])
def test_inbound_uniqueness_runs_once_direction_exists(local_pg, monkeypatch, direction):
    """M3.8: once `web_sql` has `direction`, the inbound/both cases run (not skip) and pass."""
    import sys

    local_pg.execute("CREATE OR REPLACE VIEW lgk_served.edges_both_way AS "
                     "SELECT src_node_id, dst_node_id, weight FROM lgk_served.edges_view UNION ALL "
                     "SELECT dst_node_id, src_node_id, weight FROM lgk_served.edges_view")
    # a builder without `direction` (today's web_sql) skips with the named reason
    assert direction_skip_reason(lambda depth, served_schema, top_n: "", direction) == DIRECTION_SKIP
    assert direction_skip_reason(_web_sql_with_direction, direction) is None
    monkeypatch.setattr(sys.modules[__name__], "web_sql", _web_sql_with_direction)
    for depth in (1, 2, 3, 4):
        try:
            test_web_returns_each_node_once_offline(local_pg, depth, direction)
        except pytest.skip.Exception as skipped:  # a skip here is the defect, not a skip
            pytest.fail(f"{direction} depth {depth} skipped although web_sql has direction: {skipped}")


@pytest.mark.parametrize("rows, match", [
    ([(1, "x", 1, None), (2, "x", 1, None)], "repeats"),        # hostile: a node twice
    ([(1, "x", 1, None), (2, "s", 1, None)], "seed came back"),  # hostile: the seed returns
    ([(0, "x", 1, None)], "hops"),                               # hostile: hop 0
    ([(3, "x", 1, None)], "hops"),                               # hostile: a hop beyond depth
    ([(1, "x", 1, None), (1, "y", 1, None), (1, "z", 1, None)], "s depth 2"),  # top_n + 1 in a hop
])
def test_assert_each_node_once_refuses_each_bad_shape(rows, match):
    with pytest.raises(AssertionError, match=match):
        assert_each_node_once(rows, "s", 2, 2)
    assert_each_node_once([(1, "x", 1, None), (2, "y", 1, None)], "s", 2, 2)  # a good walk passes


# ---------------------------------------------------------------- walk direction -
#
# The goldens are the exact strings 7b80171 produces (contract B1 / B2), written out by hand: the
# advisor, the app and the coordinator function bodies depend on the outbound text.

GOLDEN_WEB_2 = '''WITH hop0 AS (SELECT CAST(%(start_node_id)s AS text) AS node_id),
hop1 AS (
  SELECT c.node_id, c.support, NULL::double precision AS score
  FROM (SELECT e.dst_node_id AS node_id, COUNT(*) AS support
        FROM lgk_providers.edges_view e JOIN hop0 f ON e.src_node_id = f.node_id
        WHERE e.dst_node_id NOT IN (SELECT node_id FROM hop0)
        GROUP BY e.dst_node_id ) c
  
  ORDER BY c.support DESC, c.node_id COLLATE "C"
  LIMIT 25),
hop2 AS (
  SELECT c.node_id, c.support, NULL::double precision AS score
  FROM (SELECT e.dst_node_id AS node_id, COUNT(*) AS support
        FROM lgk_providers.edges_view e JOIN hop1 f ON e.src_node_id = f.node_id
        WHERE e.dst_node_id NOT IN (SELECT node_id FROM hop0 UNION ALL SELECT node_id FROM hop1)
        GROUP BY e.dst_node_id ) c
  
  ORDER BY c.support DESC, c.node_id COLLATE "C"
  LIMIT 25)
SELECT * FROM (
SELECT 1 AS hop, node_id, support, score FROM hop1
UNION ALL
SELECT 2 AS hop, node_id, support, score FROM hop2
) w
ORDER BY hop, support DESC, node_id COLLATE "C"'''

GOLDEN_TREE_2 = '''WITH RECURSIVE walk(depth, frontier, seen) AS (
  SELECT 0, ARRAY[CAST(%(start_node_id)s AS text)], ARRAY[CAST(%(start_node_id)s AS text)]
  UNION ALL
  SELECT w.depth + 1, n.frontier, w.seen || n.frontier
  FROM walk w CROSS JOIN LATERAL (
    SELECT array_agg(k.node_id) AS frontier FROM (
      SELECT c.node_id FROM (
        SELECT DISTINCT e.dst_node_id AS node_id
        FROM lgk_providers.edges_view e
        WHERE e.src_node_id = ANY(w.frontier)
          AND NOT (e.dst_node_id = ANY(w.seen))
      ) c
      
      ORDER BY c.node_id COLLATE "C"
      LIMIT 25
    ) k
  ) n
  WHERE w.depth < 2 AND n.frontier IS NOT NULL
),
reached AS (SELECT u.node_id, w.depth FROM walk w CROSS JOIN LATERAL unnest(w.frontier) AS u(node_id) WHERE w.depth > 0)
SELECT t.node_id, t.depth, NULL::double precision AS score
FROM reached t

ORDER BY t.depth, t.node_id COLLATE "C"
LIMIT 25'''

# The inbound and both texts, by hand.
GOLDEN_WEB_1_INBOUND = '''WITH hop0 AS (SELECT CAST(%(start_node_id)s AS text) AS node_id),
hop1 AS (
  SELECT c.node_id, c.support, NULL::double precision AS score
  FROM (SELECT e.src_node_id AS node_id, COUNT(*) AS support
        FROM lgk_providers.edges_view e JOIN hop0 f ON e.dst_node_id = f.node_id
        WHERE e.src_node_id NOT IN (SELECT node_id FROM hop0)
        GROUP BY e.src_node_id ) c
  
  ORDER BY c.support DESC, c.node_id COLLATE "C"
  LIMIT 25)
SELECT * FROM (
SELECT 1 AS hop, node_id, support, score FROM hop1
) w
ORDER BY hop, support DESC, node_id COLLATE "C"'''

GOLDEN_TREE_BOTH_BLOCK = '''        SELECT DISTINCT x.node_id FROM (
          SELECT e.dst_node_id AS node_id FROM lgk_providers.edges_view e WHERE e.src_node_id = ANY(w.frontier)
          UNION ALL
          SELECT e.src_node_id AS node_id FROM lgk_providers.edges_view e WHERE e.dst_node_id = ANY(w.frontier)
        ) x
        WHERE NOT (x.node_id = ANY(w.seen))
'''

GOLDEN_TREE_INBOUND_BLOCK = '''        SELECT DISTINCT e.src_node_id AS node_id
        FROM lgk_providers.edges_view e
        WHERE e.dst_node_id = ANY(w.frontier)
          AND NOT (e.src_node_id = ANY(w.seen))
'''


def test_outbound_sql_is_byte_identical_to_7b80171():
    assert web_sql(2, "lgk_providers", 25) == GOLDEN_WEB_2
    assert web_sql(2, "lgk_providers", 25, direction="outbound") == GOLDEN_WEB_2
    assert tree_sql(2, "lgk_providers", 25) == GOLDEN_TREE_2
    assert tree_sql(2, "lgk_providers", 25, direction="outbound") == GOLDEN_TREE_2
    assert get_walk_query("x", 2, "lgk_providers", folded=False) == GOLDEN_WEB_2


def test_inbound_and_both_sql_are_the_ticket_texts():
    assert web_sql(1, "lgk_providers", 25, direction="inbound") == GOLDEN_WEB_1_INBOUND
    both = web_sql(2, "lgk_providers", 25, direction="both")
    for hop, prev in ((1, "hop0"), (2, "hop1")):
        assert (f"FROM (SELECT e.dst_node_id AS node_id, f.node_id AS via, e.relationship_type\n"
                f"              FROM lgk_providers.edges_view e JOIN {prev} f ON e.src_node_id = f.node_id\n"
                f"              UNION\n"
                f"              SELECT e.src_node_id AS node_id, f.node_id AS via, e.relationship_type\n"
                f"              FROM lgk_providers.edges_view e JOIN {prev} f ON e.dst_node_id = f.node_id) e\n"
                ) in both, hop
    assert "WHERE e.node_id NOT IN (SELECT node_id FROM hop0 UNION ALL SELECT node_id FROM hop1)" in both
    assert both.count("LIMIT 25") == 2
    tree_in, tree_both = (tree_sql(2, "lgk_providers", 25, direction=d) for d in ("inbound", "both"))
    assert tree_in == GOLDEN_TREE_2.replace(GOLDEN_TREE_2.split("      SELECT c.node_id FROM (\n")[1]
                                            .split("      ) c\n")[0], GOLDEN_TREE_INBOUND_BLOCK)
    assert tree_both == GOLDEN_TREE_2.replace(GOLDEN_TREE_2.split("      SELECT c.node_id FROM (\n")[1]
                                              .split("      ) c\n")[0], GOLDEN_TREE_BOTH_BLOCK)


@pytest.mark.parametrize("bad", ["Inbound", "out", None, "both; drop", "", " both", 1])
def test_direction_is_validated(bad):
    """A closed set, matched exactly: anything else raises naming the allowed set."""
    allowed = r"\('outbound', 'inbound', 'both'\)"
    with pytest.raises(ValueError, match=allowed):
        web_sql(depth=1, top_n=3, served_schema="lgk_served", direction=bad)
    with pytest.raises(ValueError, match=allowed):
        tree_sql(depth=1, top_n=3, served_schema="lgk_served", direction=bad)
    with pytest.raises(ValueError, match=allowed):
        per_depth_sql("tree", 2, top_n=3, served_schema="lgk_served", direction=bad)
    with pytest.raises(ValueError, match=allowed):
        get_walk_query("x", 1, "lgk_served", folded=False, direction=bad)


FOLDED_MSG = ("folded edges are symmetric (fold_edges_sql emits both orientations); "
              "use direction='outbound'")


@pytest.mark.parametrize("direction", ["inbound", "both"])
def test_folded_refuses_inbound_and_both(direction):
    import re

    with pytest.raises(ValueError, match=re.escape(FOLDED_MSG)):
        web_sql(depth=1, top_n=3, served_schema="lgk_served", folded=True, direction=direction)
    # the gate's folded answer reaches the guard through get_walk_query
    with pytest.raises(ValueError, match=re.escape(FOLDED_MSG)):
        get_walk_query("x", 2, "lgk_served", folded=True, direction=direction)
    # unfolded, the same direction builds; folded outbound still builds and floors
    assert "LIMIT 3" in web_sql(depth=1, top_n=3, served_schema="lgk_served", direction=direction)
    assert "HAVING MIN(e.weight) >= 11" in get_walk_query("x", 1, "lgk_served", top_n=3, folded=True)
    with pytest.raises(NotImplementedError):  # folded trees still raise
        tree_sql(depth=2, top_n=3, served_schema="lgk_served", folded=True, direction=direction)


def test_get_walk_query_passes_the_direction():
    for direction in ("inbound", "both"):
        assert get_walk_query("ignored", 3, "lgk_served", top_n=10, folded=False, direction=direction) == \
            web_sql(depth=3, served_schema="lgk_served", top_n=10, direction=direction)
        assert get_walk_query("ignored", 3, "lgk_served", top_n=10, folded=False, direction=direction) != \
            GOLDEN_WEB_2


def test_advisor_rewrites_every_edges_view_reference():
    """The both SQL names edges_view twice per hop (twice per tree level): every one is rewritten."""
    from lgk.advisor import version_queries

    for shape, builder in (("web", web_sql), ("tree", tree_sql)):
        for direction, per_hop in (("outbound", 1), ("inbound", 1), ("both", 2)):
            queries = {shape: {d: builder(depth=d, served_schema="lgk_served", top_n=5, direction=direction)
                               for d in (1, 2, 3)}}
            for depth, sql in version_queries(queries, "lgk_served", "lgk_g_v2")[shape].items():
                hops = depth if shape == "web" else 1  # a tree names edges once in its recursive arm
                assert sql.count("lgk_g_v2.edges e") == per_hop * hops, (shape, direction, depth)
                assert "lgk_served" not in sql


def _web_rows(conn, depth, seed, direction, top_n=25, **kw):
    sql = web_sql(depth=depth, top_n=top_n, served_schema="lgk_served", direction=direction, **kw)
    return [(h, n, sup) for h, n, sup, _ in conn.execute(sql, {"start_node_id": seed}).fetchall()]


def test_both_counts_a_mutual_pair_once(local_pg):
    # hostile: mutual pair of one edge type (double-count bait for both)
    _load(local_pg, [("a", "b", None, "x"), ("b", "a", None, "x")])
    assert _web_rows(local_pg, 1, "a", "both") == [(1, "b", 1)]


def test_both_counts_two_edge_types_as_two(local_pg):
    # hostile: second edge type — the pair counts once per (f, type): 2, not 1 and not 3
    _load(local_pg, [("a", "b", None, "x"), ("b", "a", None, "x"), ("a", "b", None, "y")])
    assert _web_rows(local_pg, 1, "a", "both") == [(1, "b", 2)]
    assert _web_rows(local_pg, 1, "a", "outbound") == [(1, "b", 2)]
    assert _web_rows(local_pg, 1, "a", "inbound") == [(1, "b", 1)]
    # hostile: two types on ONE arm only (out, then in): each arm must keep the type itself
    for edges in ([("a", "b", None, "x"), ("a", "b", None, "y")], [("b", "a", None, "x"), ("b", "a", None, "y")]):
        _load(local_pg, edges)
        assert _web_rows(local_pg, 1, "a", "both") == [(1, "b", 2)], edges


def test_each_direction_reaches_its_own_one_way_discriminator(local_pg):
    # hostile: one-way discriminators m_c (outbound only) and m_d (inbound only), plus the mutual pair
    _load(local_pg, [("m_a", "m_b"), ("m_b", "m_a"), ("m_a", "m_c"), ("m_d", "m_a")])
    assert _web_rows(local_pg, 1, "m_a", "outbound") == [(1, "m_b", 1), (1, "m_c", 1)]
    assert _web_rows(local_pg, 1, "m_a", "inbound") == [(1, "m_b", 1), (1, "m_d", 1)]
    assert _web_rows(local_pg, 1, "m_a", "both") == [(1, "m_b", 1), (1, "m_c", 1), (1, "m_d", 1)]
    # depth 2 from m_c inbound: m_a, then m_b and m_d (never m_c again)
    assert _web_rows(local_pg, 2, "m_c", "inbound") == [(1, "m_a", 1), (2, "m_b", 1), (2, "m_d", 1)]


@pytest.mark.parametrize("direction", ["inbound", "both"])
def test_inbound_sink_hop_is_cut_to_top_n(local_pg, direction):
    # hostile: fan-in sink with 40 in-edges (the inbound twin of the hub)
    _load(local_pg, [(f"l{i:02d}", "sink") for i in range(40)] + [("hubby", f"l{i:02d}") for i in range(40)])
    rows = _web_rows(local_pg, 2, "sink", direction, top_n=3)
    assert rows == [(1, "l00", 1), (1, "l01", 1), (1, "l02", 1), (2, "hubby", 3)]


def test_inbound_tree_cycle_excludes_seed(local_pg):
    _load(local_pg, [("a", "b"), ("b", "c"), ("c", "a")])  # hostile: cycle, walked against its edges
    for depth in (3, 4, 100):
        assert _tree_dir(local_pg, depth, "a", "inbound") == [("c", 1), ("b", 2)]
        # both: b (out) and c (in) at level 1; level 2 would re-reach a through either arm
        assert _tree_dir(local_pg, depth, "a", "both") == [("b", 1), ("c", 1)]


def _tree_dir(conn, depth, seed, direction, top_n=1000):
    conn.execute("SET statement_timeout = '10s'")
    rows = conn.execute(tree_sql(depth=depth, top_n=top_n, served_schema="lgk_served", direction=direction),
                        {"start_node_id": seed}).fetchall()
    return [(n, d) for n, d, _score in rows]


# A reference written from the spec text, in plain Python (no SQL): per hop, the far ends of the
# previous hop's edges in `direction`, support = distinct (node, via, type) rows, minus every node
# kept so far, ordered by (support desc, node_id bytes), cut at top_n. A tree level: the distinct
# unseen far ends, ordered by node_id bytes, cut at top_n; the result ordered by (depth, node_id).
def _far_ends(edges, frontier, direction):
    rows = set()
    for s, d, t in edges:
        if direction in ("outbound", "both") and s in frontier:
            rows.add((d, s, t))
        if direction in ("inbound", "both") and d in frontier:
            rows.add((s, d, t))
    return rows


def _py_web(edges, seed, depth, top_n, direction):
    seen, prev, out = {seed}, {seed}, []
    for hop in range(1, depth + 1):
        support = {}
        for node, _via, _t in _far_ends(edges, prev, direction):
            if node not in seen:
                support[node] = support.get(node, 0) + 1
        kept = sorted(support.items(), key=lambda kv: (-kv[1], kv[0].encode()))[:top_n]
        out += [(hop, n, c) for n, c in kept]
        prev = {n for n, _ in kept}
        seen |= prev
    return out


def _py_tree(edges, seed, depth, top_n, direction):
    seen, frontier, out = {seed}, {seed}, []
    for level in range(1, depth + 1):
        nxt = sorted({n for n, _v, _t in _far_ends(edges, frontier, direction) if n not in seen},
                     key=str.encode)[:top_n]
        if not nxt:
            break
        out += [(n, level) for n in nxt]
        frontier = set(nxt)
        seen |= frontier
    return sorted(out, key=lambda r: (r[1], r[0].encode()))[:top_n]


def _random_graph(seed):
    import random

    rnd = random.Random(seed)
    nodes = [f"n{i:02d}" for i in range(18)] + ["Z", "_x"]  # hostile: byte order != locale order
    edges = set()
    for _ in range(70):
        a, b = rnd.choice(nodes), rnd.choice(nodes)
        edges.add((a, b, rnd.choice(("x", "y"))))  # hostile: two edge types, self-loops, mutual pairs
    for a, b, t in list(edges)[:10]:
        edges.add((b, a, t))  # hostile: mutual pairs of one type
    return sorted(edges), nodes


@pytest.mark.parametrize("direction", ["outbound", "inbound", "both"])
def test_every_direction_equals_a_python_reference(local_pg, direction):
    for graph_seed in (1, 2, 3):
        edges, nodes = _random_graph(graph_seed)
        # hostile: reordered input — the SQL never sees the rows in the reference's order
        _load(local_pg, [(s, d, None, t) for s, d, t in reversed(edges)])
        for seed in nodes[:6] + ["Z"]:
            for depth in (1, 2, 3, 4):
                assert _web_rows(local_pg, depth, seed, direction, top_n=3) == \
                    _py_web(edges, seed, depth, 3, direction), (graph_seed, seed, depth)
                assert _tree_dir(local_pg, depth, seed, direction, top_n=4) == \
                    _py_tree(edges, seed, depth, 4, direction), (graph_seed, seed, depth)


# ---------------------------------------------------------------- function form (walk direction)


def _create_coordinator_functions(conn, folded=False):
    """The shipped coordinator functions, created in schema lgk_fn over a table lgk_fn.edges."""
    from lgk import served_functions as sf
    from lgk.privacy.floor import walk_count_terms

    conn.execute("DROP SCHEMA IF EXISTS lgk_fn CASCADE")
    conn.execute("CREATE SCHEMA lgk_fn")
    conn.execute("CREATE TABLE lgk_fn.edges AS SELECT * FROM lgk_served.edges_view")
    fns = [f for f in sf._kit_functions("lgk_fn", walk_count_terms(None, folded), folded, 3)
           if f.name.startswith("coordinator_")]
    for fn in fns:
        conn.execute(sf.create_sql(fn, "lgk_fn"))
    return {f.name for f in fns}


def test_coordinator_functions_walk_every_direction(local_pg):
    import psycopg

    edges, nodes = _random_graph(7)
    _load(local_pg, [(s, d, None, t) for s, d, t in edges])
    assert _create_coordinator_functions(local_pg) == {"coordinator_web", "coordinator_tree"}
    for seed in nodes[:5]:
        for depth in (1, 2, 3, 4):
            for direction in ("outbound", "inbound", "both"):
                web = local_pg.execute("SELECT hop, node_id, support FROM lgk_fn.coordinator_web(%s, %s, %s)",
                                       (seed, depth, direction)).fetchall()
                assert web == _py_web(edges, seed, depth, 3, direction), (seed, depth, direction)
                tree = local_pg.execute("SELECT node_id, depth FROM lgk_fn.coordinator_tree(%s, %s, %s)",
                                        (seed, depth, direction)).fetchall()
                assert tree == _py_tree(edges, seed, depth, 3, direction), (seed, depth, direction)
            # the two-argument call is the outbound walk (p_direction DEFAULT 'outbound')
            assert local_pg.execute("SELECT hop, node_id, support FROM lgk_fn.coordinator_web(%s, %s)",
                                    (seed, depth)).fetchall() == _py_web(edges, seed, depth, 3, "outbound")
    for bad in ("Inbound", "sideways", None):
        for fn in ("coordinator_web", "coordinator_tree"):
            with pytest.raises(psycopg.errors.RaiseException, match="direction must be one of"):
                local_pg.execute(f"SELECT * FROM lgk_fn.{fn}('n00', 1, %s)", (bad,)).fetchall()
    with pytest.raises(psycopg.errors.RaiseException, match="depth must be in 1..4"):
        local_pg.execute("SELECT * FROM lgk_fn.coordinator_web('n00', 5, 'inbound')").fetchall()


def test_folded_coordinator_web_refuses_inbound_and_both(local_pg):
    import psycopg

    # hostile: folded weights at the floor; outbound still serves, the other directions raise
    _load(local_pg, [("a", "b", 11), ("b", "a", 11), ("c", "a", 12)])
    assert _create_coordinator_functions(local_pg, folded=True) == {"coordinator_web"}
    assert local_pg.execute("SELECT node_id, support FROM lgk_fn.coordinator_web('a', 1, 'outbound')"
                            ).fetchall() == [("b", 11)]
    for direction in ("inbound", "both"):
        with pytest.raises(psycopg.errors.RaiseException, match=r"folded edges are symmetric .* "
                                                                 r"use direction='outbound'"):
            local_pg.execute("SELECT * FROM lgk_fn.coordinator_web('a', 1, %s)", (direction,)).fetchall()


# ---------------------------------------------------------------- per-call edge_types
#
# Hand-written goldens (not rendered by the code under test). The mixed run folder below is a
# real privacy pass: two fold types (co-membership `shares_refs`, pair `treats_at`) and two
# unfolded types (`affiliation`, `same_practice`) over the offline visits/providers schema.

GOLDEN_WEB_1_AFFILIATION_PRACTICE = '''WITH hop0 AS (SELECT CAST(%(start_node_id)s AS text) AS node_id),
hop1 AS (
  SELECT c.node_id, c.support, NULL::double precision AS score
  FROM (SELECT e.dst_node_id AS node_id, COUNT(*) AS support
        FROM lgk_care.edges_view e JOIN hop0 f ON e.src_node_id = f.node_id
        WHERE e.dst_node_id NOT IN (SELECT node_id FROM hop0)
          AND e.relationship_type IN ('affiliation', 'same_practice')
        GROUP BY e.dst_node_id ) c
  
  ORDER BY c.support DESC, c.node_id COLLATE "C"
  LIMIT 25)
SELECT * FROM (
SELECT 1 AS hop, node_id, support, score FROM hop1
) w
ORDER BY hop, support DESC, node_id COLLATE "C"'''

_UNFOLDED_EDGE = {"source_table": "visits", "src": "provider_id", "src_type": "provider",
                  "dst": "provider_id", "dst_type": "provider"}
MIXED_MAPPING = {
    "source_catalog": "c", "source_schema": "s",
    "nodes": {"provider": {"source_table": "providers", "natural_key": "provider_id",
                           "label": "name", "properties": ["npi"]},
              # not sourced from visits (it holds the care-subject ref_code); visit_slots is
              # the visit directory without it (added to schema.json by _cleared_run)
              "visit": {"source_table": "visit_slots", "natural_key": "visit_seq", "label": "visit_seq"}},
    "edges": {
        "shares_refs": {**_UNFOLDED_EDGE, "fold_subject": "ref_code"},  # co-membership fold
        # hostile: a second fold type (pair form), excluded from a shares_refs walk only by IN
        "treats_at": {**_UNFOLDED_EDGE, "dst": "visit_seq", "dst_type": "visit", "fold_subject": "ref_code"},
        "affiliation": dict(_UNFOLDED_EDGE),  # hostile: a mixed graph (unfolded types)
        "same_practice": dict(_UNFOLDED_EDGE),
    },
}


def _add_visit_slots(run_dir: Path) -> None:
    """schema.json + visit_slots(visit_seq, provider_id): a visit node source with no care column."""
    import json

    from tests.test_privacy import _col

    path = run_dir / "schema.json"
    schema = json.loads(path.read_text())
    schema["tables"].append({"name": "visit_slots", "key_columns": ["visit_seq"], "foreign_keys": [],
                             "columns": [_col("visit_seq", ["v1", "v2"]),
                                         _col("provider_id", ["p1", "p2"])]})
    path.write_text(json.dumps(schema))


def _cleared_run(tmp_path: Path, mapping: dict) -> str:
    """A run folder whose privacy step really passed for `mapping` (ref_code folded)."""
    from lgk.privacy.flags import OpenFlagError, resolve_flag
    from lgk.steps.privacy import privacy
    from tests.test_privacy import _visits_and_providers
    from tests.test_privacy_gate import _write_mapping

    _visits_and_providers(tmp_path)
    _add_visit_slots(tmp_path)
    _write_mapping(tmp_path, mapping)
    with pytest.raises(OpenFlagError):
        privacy({"run_folder": str(tmp_path)})
    folded = any(e.get("fold_subject") for e in mapping["edges"].values())
    resolve_flag(str(tmp_path), "visits", "ref_code", "fold" if folded else "drop", chooser="op")
    resolve_flag(str(tmp_path), "members", "patient_name", "drop", chooser="op")
    privacy({"run_folder": str(tmp_path)})
    return str(tmp_path)


@pytest.fixture
def mixed_run(tmp_path):
    return _cleared_run(tmp_path, MIXED_MAPPING)


@pytest.mark.parametrize("bad", ["nope", "shares_refs'); drop", "Shares_Refs", [], (), "shares_refs",
                                 ["shares_refs", "x y"], [None]])
def test_edge_types_are_validated(mixed_run, bad):
    """Closed list + identifier pattern: every bad selection raises ValueError naming edge_types."""
    selection = [bad] if bad in ("nope", "shares_refs'); drop", "Shares_Refs") else bad
    with pytest.raises(ValueError, match="edge_types"):
        get_walk_query("x", 1, "lgk_care", run_folder=mixed_run, edge_types=selection)
    if bad != "nope":  # the pattern check also runs in the builder, with no run folder
        with pytest.raises(ValueError, match="edge_types"):
            web_sql(1, "lgk_care", 3, edge_types=selection)
        with pytest.raises(ValueError, match="edge_types"):
            tree_sql(1, "lgk_care", 3, edge_types=selection)
    # hostile: a name that passes the pattern but is not an edge type of the mapping
    with pytest.raises(ValueError, match=r"not edge types of the mapping.*affiliation"):
        get_walk_query("x", 1, "lgk_care", run_folder=mixed_run, edge_types=["nope"])


def test_mixed_selection_is_refused(mixed_run):
    from lgk.privacy.gate import MixedFoldSelectionError

    for selection in (["shares_refs", "affiliation"], ["affiliation", "treats_at", "same_practice"]):
        with pytest.raises(MixedFoldSelectionError, match=r"mixes folded \[.*\] and unfolded"):
            get_walk_query("x", 1, "lgk_care", run_folder=mixed_run, edge_types=selection)


def test_folded_is_derived_from_the_selection(mixed_run):
    folded = get_walk_query("x", 2, "lgk_care", run_folder=mixed_run, edge_types=["shares_refs"])
    assert folded.count("MIN(e.weight) AS support") == 2 and folded.count("HAVING MIN(e.weight) >= 11") == 2
    assert "COUNT(*)" not in folded
    unfolded = get_walk_query("x", 2, "lgk_care", run_folder=mixed_run, edge_types=["affiliation"])
    assert unfolded.count("COUNT(*) AS support") == 2 and "HAVING" not in unfolded and "weight" not in unfolded
    # two fold types together are one kind: still folded
    both_folds = get_walk_query("x", 1, "lgk_care", run_folder=mixed_run, edge_types=["treats_at", "shares_refs"])
    assert "AND e.relationship_type IN ('shares_refs', 'treats_at')" in both_folds
    assert "HAVING MIN(e.weight) >= 11" in both_folds
    # walk direction x edge types: the outbound-only rule binds the folded selection, not the graph
    for direction in ("inbound", "both"):
        assert "COUNT(*)" in get_walk_query("x", 1, "lgk_care", run_folder=mixed_run,
                                            edge_types=["affiliation"], direction=direction)
        with pytest.raises(ValueError, match="use direction='outbound'"):
            get_walk_query("x", 1, "lgk_care", run_folder=mixed_run, edge_types=["shares_refs"],
                           direction=direction)


def test_edge_types_render_sorted_and_deduplicated(mixed_run):
    # hostile: reordered and duplicated input render one sorted, de-duplicated literal list
    for selection in (["affiliation", "same_practice"], ["same_practice", "affiliation", "same_practice"],
                      ("same_practice", "affiliation")):
        assert web_sql(1, "lgk_care", 25, edge_types=selection) == GOLDEN_WEB_1_AFFILIATION_PRACTICE
        assert get_walk_query("x", 1, "lgk_care", run_folder=mixed_run,
                              edge_types=selection) == GOLDEN_WEB_1_AFFILIATION_PRACTICE
    assert "AND e.relationship_type IN ('affiliation')\n" in web_sql(1, "lgk_care", 25, edge_types=["affiliation"])
    tree = tree_sql(2, "lgk_care", 25, edge_types=["same_practice", "affiliation", "affiliation"])
    assert tree.count("AND e.relationship_type IN ('affiliation', 'same_practice')") == 1


def test_unfiltered_sql_is_byte_identical_on_an_unfolded_graph(tmp_path):
    unfolded = {**MIXED_MAPPING, "edges": {"affiliation": dict(_UNFOLDED_EDGE)}}
    run = _cleared_run(tmp_path, unfolded)
    assert get_walk_query("x", 2, "lgk_providers", run_folder=run) == GOLDEN_WEB_2
    assert web_sql(2, "lgk_providers", 25, edge_types=None) == GOLDEN_WEB_2
    assert tree_sql(2, "lgk_providers", 25, edge_types=None) == GOLDEN_TREE_2


def test_omitted_edge_types_on_a_mixed_graph_is_refused(mixed_run, tmp_path_factory):
    from lgk.privacy.gate import served_walk_policy

    with pytest.raises(ValueError, match=r"edge_types is required on a graph with folded and unfolded"):
        get_walk_query("x", 1, "lgk_care", run_folder=mixed_run)
    with pytest.raises(ValueError, match="edge_types is required"):
        served_walk_policy(mixed_run, None, None)
    # every edge type folded: the omitted selection is all of them (folded, one IN line per hop)
    folds_only = {**MIXED_MAPPING, "edges": {k: v for k, v in MIXED_MAPPING["edges"].items()
                                             if v.get("fold_subject")}}
    sql = get_walk_query("x", 2, "lgk_care", run_folder=_cleared_run(tmp_path_factory.mktemp("f"), folds_only))
    assert sql.count("AND e.relationship_type IN ('shares_refs', 'treats_at')") == 2
    assert sql.count("HAVING MIN(e.weight) >= 11") == 2


def test_folded_selection_excludes_other_fold_types(local_pg):
    # hostile: a second fold type `treats_at` P1 -> F1 with weight 12 passes the floor; only the
    # IN clause keeps it out of a shares_patients walk. hostile: an unfolded same_practice edge
    # into a folded neighbour (NULL weight) must not change P2's floored support.
    _load(local_pg, [("P1", "P2", 11, "shares_patients"), ("P1", "P4", 12, "shares_patients"),
                     ("P1", "F1", 12, "treats_at"), ("P1", "H1", None, "affiliation"),
                     ("P1", "P2", None, "same_practice"), ("P1", "P3", 10, "shares_patients")])

    def rows(edge_types, folded):
        sql = web_sql(1, "lgk_served", 25, folded=folded, edge_types=edge_types)
        return [(n, s) for _h, n, s, _sc in local_pg.execute(sql, {"start_node_id": "P1"}).fetchall()]

    assert rows(["shares_patients"], True) == [("P4", 12), ("P2", 11)]
    assert rows(["treats_at"], True) == [("F1", 12)]
    assert rows(["affiliation"], False) == [("H1", 1)]
    assert rows(["affiliation", "same_practice"], False) == [("H1", 1), ("P2", 1)]
    tree = local_pg.execute(tree_sql(1, "lgk_served", 25, edge_types=["affiliation"]),
                            {"start_node_id": "P1"}).fetchall()
    assert [(n, d) for n, d, _s in tree] == [("H1", 1)]


def _create_typed_coordinator_web(conn, mapping_types, fold_types, mixed):
    """The shipped coordinator_web with one branch set per edge type, in lgk_fn over lgk_fn.edges."""
    from lgk import served_functions as sf
    from lgk.privacy.floor import walk_count_terms

    conn.execute("DROP SCHEMA IF EXISTS lgk_fn CASCADE")
    conn.execute("CREATE SCHEMA lgk_fn")
    conn.execute("CREATE TABLE lgk_fn.edges AS SELECT * FROM lgk_served.edges_view")
    typed = [(t, walk_count_terms(None, t in fold_types), t in fold_types) for t in mapping_types]
    mixed_error = "edge_types is required on a graph with folded and unfolded edge types" if mixed else ""
    [web] = [f for f in sf._kit_functions("lgk_fn", walk_count_terms(None, False), False, 25, typed, mixed_error)
             if f.name == "coordinator_web"]
    conn.execute(sf.create_sql(web, "lgk_fn"))
    return web


def test_coordinator_web_walks_one_edge_type_per_call(local_pg):
    import psycopg

    _load(local_pg, [("P1", "P2", 11, "shares_patients"), ("P1", "P4", 12, "shares_patients"),
                     ("P1", "F1", 12, "treats_at"), ("P1", "H1", None, "affiliation"),
                     ("P1", "P2", None, "same_practice")])
    web = _create_typed_coordinator_web(
        local_pg, ["affiliation", "same_practice", "shares_patients", "treats_at"],
        {"shares_patients", "treats_at"}, mixed=True)
    assert web.signature("s") == "s.coordinator_web(text, integer, text, text)"

    def call(*args):
        marks = ", ".join(["%s"] * len(args))
        return [(n, s) for _h, n, s in local_pg.execute(
            f"SELECT hop, node_id, support FROM lgk_fn.coordinator_web({marks})", args).fetchall()]

    assert call("P1", 1, "outbound", "shares_patients") == [("P4", 12), ("P2", 11)]
    assert call("P1", 1, "outbound", "treats_at") == [("F1", 12)]
    assert call("P1", 1, "outbound", "affiliation") == [("H1", 1)]
    assert call("P2", 1, "inbound", "same_practice") == [("P1", 1)]  # unfolded: every direction
    with pytest.raises(psycopg.errors.RaiseException, match="use direction='outbound'"):
        call("P1", 1, "both", "shares_patients")
    with pytest.raises(psycopg.errors.RaiseException, match="edge_types is required"):
        call("P1", 1)  # NULL p_edge_type on a mixed graph
    for bad in ("nope", "Shares_Patients", "shares_patients'); drop"):
        with pytest.raises(psycopg.errors.RaiseException, match="edge_type must be one of"):
            call("P1", 1, "outbound", bad)
    # one smoke call per served (edge type, direction, depth): 2 folds x 1 x 4 + 2 unfolded x 3 x 4
    assert len(web.smoke_args) == 2 * 4 + 2 * 3 * 4


def test_an_unfolded_tree_selection_filters_both_arms(local_pg):
    # hostile (prover O1): under direction='both' each arm of the tree's UNION ALL must carry the
    # IN line; P3 is reachable from P1 only through a `same_practice` edge pointing INTO P1
    _load(local_pg, [("P1", "H1", None, "affiliation"), ("P3", "P1", None, "same_practice"),
                     ("H2", "P1", None, "affiliation")])
    sql = tree_sql(2, "lgk_served", 25, direction="both", edge_types=["affiliation"])
    assert sql.count("AND e.relationship_type IN ('affiliation')") == 2
    rows = local_pg.execute(sql, {"start_node_id": "P1"}).fetchall()
    assert [(n, d) for n, d, _s in rows] == [("H1", 1), ("H2", 1)]


# ================================================================ the set walk
# Offline half of the set walk. The live half (Spark reference, kit data, index use) is
# tests/test_walks_live.py::TestWebSetWalk. The Python reference `_py_web_set` below is written
# from the semantics, not from the SQL: one frontier set, one cut per hop over it,
# a stop node kept but never expanded, seeds never tested against the stop.
import copy as _copy

from lgk.walks.sql import NodePredicate, web_set_sql

WS_KEYS = {"provider": ("specialty", "network_status"), "hospital": ("network_status", "specialty")}
SET_MAPPING = _copy.deepcopy(MIXED_MAPPING)
SET_MAPPING["nodes"]["provider"]["walk_keys"] = ["npi"]


def _ws_sql(depth=2, top_n=25, seed=NodePredicate("provider", "specialty"), stop=None, **kw):
    return web_set_sql(depth, "lgk_ws", top_n, seed, stop, walk_keys=WS_KEYS, **kw)


def test_web_set_sql_has_one_cut_per_hop_and_never_expands_a_match():
    for depth in (1, 2, 3, 4):
        sql = _ws_sql(depth, top_n=7, stop=NodePredicate("hospital", "network_status"))
        assert sql.count("LIMIT 7") == depth  # one cut per hop, over the whole set
        assert sql.count("NOT f.matched") == depth  # every hop after hop 0 joins unmatched nodes only
        assert sql.count("LIMIT 201") == 1  # the seed cap guard, bounded at max_seeds + 1
        assert _limits_follow_order_by(sql.replace("LIMIT 201", ""))


def test_web_set_binds_values_only_for_fields():
    assert "%(seed_value)s" in _ws_sql() and "%(stop_value)s" not in _ws_sql()
    bare = _ws_sql(seed=NodePredicate("provider"), stop=NodePredicate("hospital"))
    assert "%(seed_value)s" not in bare and "%(stop_value)s" not in bare
    assert "n.node_type = 'provider'\n" in bare and "m.node_type = 'hospital')" in bare
    both = _ws_sql(stop=NodePredicate("hospital", "network_status"))
    assert "m.node_type = 'hospital' AND m.properties ->> 'network_status' = %(stop_value)s" in both
    assert "n.node_type = 'provider' AND n.properties ->> 'specialty' = %(seed_value)s" in both
    assert "n.label = %(seed_value)s" in _ws_sql(seed=NodePredicate("hospital", "label"))
    assert "n.natural_key = %(seed_value)s" in _ws_sql(seed=NodePredicate("hospital", "natural_key"))
    assert "%(start_node_id)s" not in both
    # without a stop, matched is the constant false and there is no stop join
    assert "false AS matched" in _ws_sql() and " m\n" not in _ws_sql()


@pytest.mark.parametrize("pred, match", [
    (NodePredicate("provider", "location"), r"'location' is not a walk key of node type 'provider'"),
    # hostile: a key declared for ANOTHER node type
    (NodePredicate("provider", "zip"), r"'zip' is not a walk key of node type 'provider'"),
    (NodePredicate("clinic", "label"), r"node_type 'clinic' is not a node type of the mapping"),
    (NodePredicate("provider", "spe'cialty"), "field"),
    (NodePredicate("Provider", None), "node_type"),
])
def test_a_field_outside_walk_keys_is_refused(pred, match):
    keys = {**WS_KEYS, "hospital": ("zip",)}
    with pytest.raises(ValueError, match=match):
        web_set_sql(1, "lgk_ws", 5, pred, walk_keys=keys)
    with pytest.raises(ValueError, match=match.replace("seed", "stop") if "seed" in match else match):
        web_set_sql(1, "lgk_ws", 5, NodePredicate("provider", "specialty"), pred, walk_keys=keys)
    # with no mapping, only label and natural_key are fields
    with pytest.raises(ValueError, match="is not a walk key"):
        web_set_sql(1, "lgk_ws", 5, NodePredicate("provider", "specialty"))
    assert "n.label" in web_set_sql(1, "lgk_ws", 5, NodePredicate("provider", "label"))


@pytest.mark.parametrize("bad", [0, 1001, -1, True, 2.0, "200"])
def test_max_seeds_is_1_to_1000(bad):
    with pytest.raises(ValueError, match="max_seeds"):
        _ws_sql(max_seeds=bad)
    assert "count(*) > 1000" in _ws_sql(max_seeds=1000) and "LIMIT 1001" in _ws_sql(max_seeds=1000)
    assert "count(*) > 1 " in _ws_sql(max_seeds=1)


def test_web_set_depth_cap_and_per_depth():
    with pytest.raises(ValueError, match="depth"):
        _ws_sql(depth=5)
    from lgk.walks.sql import per_depth_sql as pds
    got = pds("web_set", 4, served_schema="lgk_ws", top_n=3, seed=NodePredicate("provider"), walk_keys=WS_KEYS)
    assert sorted(got) == [1, 2, 3, 4] and got[3] == _ws_sql(3, 3, NodePredicate("provider"))


def test_folded_web_set_takes_min_weight_and_the_floor():
    sql = _ws_sql(folded=True, privacy_floor=13)
    assert sql.count("MIN(e.weight) AS support") == 2 and sql.count("HAVING MIN(e.weight) >= 13") == 2
    assert "COUNT(*)" not in sql
    assert sql.count("AND e.relationship_type IN ('a', 'b')") == 0
    assert _ws_sql(edge_types=["b", "a", "b"]).count("AND e.relationship_type IN ('a', 'b')") == 2


# ------------------------------------------------------------ get_walk_query + the gate
@pytest.fixture
def set_run(tmp_path):
    return _cleared_run(tmp_path, SET_MAPPING)


def test_get_walk_query_builds_the_set_walk_from_the_cleared_mapping(set_run):
    sql = get_walk_query("ignored", 2, "lgk_care", run_folder=set_run, edge_types=["affiliation"],
                         seed=NodePredicate("provider", "npi"), stop=NodePredicate("visit", "label"))
    assert "n.properties ->> 'npi' = %(seed_value)s" in sql and "m.label = %(stop_value)s" in sql
    assert sql.count("AND e.relationship_type IN ('affiliation')") == 2
    # hostile: a column of the node table that is not a declared walk key
    with pytest.raises(ValueError, match="'name' is not a walk key of node type 'provider'"):
        get_walk_query("x", 1, "lgk_care", run_folder=set_run, edge_types=["affiliation"],
                       seed=NodePredicate("provider", "name"))
    # hostile: a walk key of provider asked of visit
    with pytest.raises(ValueError, match="'npi' is not a walk key of node type 'visit'"):
        get_walk_query("x", 1, "lgk_care", run_folder=set_run, edge_types=["affiliation"],
                       seed=NodePredicate("provider", "npi"), stop=NodePredicate("visit", "npi"))
    # the folded selection gives the floored count; the mixed graph still needs edge_types
    folded = get_walk_query("x", 1, "lgk_care", run_folder=set_run, edge_types=["shares_refs"],
                            seed=NodePredicate("provider", "npi"))
    assert "HAVING MIN(e.weight) >= 11" in folded
    with pytest.raises(ValueError, match="edge_types is required"):
        get_walk_query("x", 1, "lgk_care", run_folder=set_run, seed=NodePredicate("provider", "npi"))
    assert get_walk_query("x", 2, "lgk_care", run_folder=set_run, edge_types=["affiliation"],
                          seed=NodePredicate("provider", "npi"), max_seeds=7).count("count(*) > 7") == 1


def test_get_walk_query_refuses_a_stop_without_a_seed_and_other_directions():
    with pytest.raises(ValueError, match="stop needs a seed"):
        get_walk_query("x", 1, "s", folded=False, stop=NodePredicate("provider"))
    for direction in ("inbound", "both"):
        with pytest.raises(NotImplementedError, match="web_set with direction"):
            get_walk_query("x", 1, "s", folded=False, direction=direction, seed=NodePredicate("provider"))
    # without a run folder: no mapping, so only label / natural_key fields
    with pytest.raises(ValueError, match="is not a walk key"):
        get_walk_query("x", 1, "s", folded=False, seed=NodePredicate("provider", "npi"))


def test_walk_key_must_be_a_property(tmp_path):
    """check_walk_keys: a walk key that is not a property of ITS node type is refused (by the privacy
    step before it stamps, and by the gate), naming the key and the node type."""
    from lgk.mapping import parse_mapping
    from lgk.privacy.gate import MappingLeakError, check_walk_keys

    bad = _copy.deepcopy(SET_MAPPING)
    # hostile: `name` IS a column of providers (the label) and `npi` IS a property of provider,
    # but neither is a property of visit
    bad["nodes"]["visit"]["walk_keys"] = ["npi"]
    with pytest.raises(MappingLeakError, match=r"walk_keys npi \(node visit\)"):
        check_walk_keys(parse_mapping(bad))
    label = _copy.deepcopy(SET_MAPPING)
    label["nodes"]["provider"]["walk_keys"] = ["npi", "name"]
    with pytest.raises(MappingLeakError, match=r"name \(node provider\)"):
        check_walk_keys(parse_mapping(label))
    check_walk_keys(parse_mapping(SET_MAPPING))  # npi is provider's property
    with pytest.raises(MappingLeakError, match="walk_keys"):
        _cleared_run(tmp_path, bad)  # the privacy step itself refuses to stamp it


@pytest.mark.parametrize("keys, match", [
    (["npi", "npi"], "more than once"),
    (["a", "b", "c", "d", "e"], "at most 4"),
    (["Npi"], "walk_keys"),
    ("npi", "list"),
])
def test_walk_keys_shape_is_checked_by_the_mapping(keys, match):
    from lgk.mapping import MappingError, parse_mapping

    m = _copy.deepcopy(SET_MAPPING)
    m["nodes"]["provider"]["walk_keys"] = keys
    with pytest.raises(MappingError, match=match):
        parse_mapping(m)


def test_walk_keys_round_trip_only_when_declared():
    from lgk.mapping import parse_mapping

    out = parse_mapping(SET_MAPPING).to_dict()
    assert out["nodes"]["provider"]["walk_keys"] == ["npi"] and "walk_keys" not in out["nodes"]["visit"]
    assert parse_mapping(out).to_dict() == out
    assert "walk_keys" not in str(parse_mapping(MIXED_MAPPING).to_dict())


# ------------------------------------------------------------ walk-key indexes
def test_walk_key_index_names_and_sql():
    from lgk.walks.sql import walk_key_index_name, walk_key_index_sql

    assert walk_key_index_name("provider", "specialty") == "lgk_nodes_wk_provider_specialty_idx"
    long = walk_key_index_name("a" * 30, "b" * 30)  # hostile: 74 bytes before the cut
    full = f"lgk_nodes_wk_{'a' * 30}_{'b' * 30}_idx"
    import hashlib as _h
    assert long == full[:54] + "_" + _h.sha256(full.encode()).hexdigest()[:8] and len(long) == 63
    assert walk_key_index_name("a" * 30, "b" * 31) != long  # the hash keeps two long names apart
    assert walk_key_index_sql("k_v2", "hospital", "network_status") == (
        "CREATE INDEX IF NOT EXISTS lgk_nodes_wk_hospital_network_status_idx\n"
        "  ON k_v2.nodes ((properties ->> 'network_status')) WHERE node_type = 'hospital'")


class _IndexConn:
    def __init__(self, tables=("nodes", "edges")):
        self.sql, self.tables = [], tables

    def transaction(self):
        import contextlib
        return contextlib.nullcontext()

    def execute(self, sql, params=None):
        self.sql.append(sql)
        return type("R", (), {"fetchall": lambda s: [(t,) for t in self.tables]})()


def test_reapply_and_provision_create_every_walk_key_index(tmp_path, monkeypatch):
    from lgk.mapping import parse_mapping
    from lgk.steps import provision as prov
    from lgk.steps.reapply import ensure_indexes

    m = _copy.deepcopy(SET_MAPPING)
    m["nodes"]["visit"]["properties"] = ["provider_id"]
    m["nodes"]["visit"]["walk_keys"] = ["provider_id"]
    conn = _IndexConn()
    ensure_indexes(conn, "k_v3", parse_mapping(m))
    made = [s for s in conn.sql if "lgk_nodes_wk_" in s]
    assert made == ["CREATE INDEX IF NOT EXISTS lgk_nodes_wk_provider_npi_idx\n"
                    "  ON k_v3.nodes ((properties ->> 'npi')) WHERE node_type = 'provider'",
                    "CREATE INDEX IF NOT EXISTS lgk_nodes_wk_visit_provider_id_idx\n"
                    "  ON k_v3.nodes ((properties ->> 'provider_id')) WHERE node_type = 'visit'"]
    no_keys = _IndexConn()
    ensure_indexes(no_keys, "k_v3", parse_mapping(MIXED_MAPPING))
    assert not [s for s in no_keys.sql if "lgk_nodes_wk_" in s]
    # provision: same statements on the new version, plus both edges join indexes; with no walk
    # key it still connects and indexes edges (scale spike: a fresh version scanned 1.6 M edges)
    import yaml as _yaml
    edge_idx = ["CREATE INDEX IF NOT EXISTS lgk_edges_src_node_id_idx ON k_v3.edges (src_node_id)",
                "CREATE INDEX IF NOT EXISTS lgk_edges_dst_node_id_idx ON k_v3.edges (dst_node_id)"]
    cfg = {"run_folder": str(tmp_path), "graph_name": "g", "kit_schema": "k", "version": 3}
    (tmp_path / "graph_mapping.yaml").write_text(_yaml.safe_dump(m, sort_keys=False))
    pconn = _IndexConn()
    pconn.autocommit, pconn.close = False, lambda: None
    monkeypatch.setattr(prov, "connect_branch", lambda c: pconn)
    prov._create_version_indexes(cfg)
    assert [s for s in pconn.sql if "lgk_nodes_wk_" in s] == made
    assert [s for s in pconn.sql if "lgk_edges_" in s] == edge_idx
    (tmp_path / "graph_mapping.yaml").write_text(_yaml.safe_dump(MIXED_MAPPING, sort_keys=False))
    # hostile: no walk key, and closure/node_group absent from the version (no index on them)
    nconn = _IndexConn()
    nconn.autocommit, nconn.close = False, lambda: None
    monkeypatch.setattr(prov, "connect_branch", lambda c: nconn)
    prov._create_version_indexes(cfg)
    assert [s for s in nconn.sql if s.startswith("CREATE INDEX")] == edge_idx



def test_provision_runs_the_walk_key_indexes_on_the_new_version(tmp_path, monkeypatch):
    """provision() itself (its other parts recorded) creates the walk-key indexes on <kit>_v<N>,
    after the version schema is marked and before the version is recorded as servable."""
    import yaml as _yaml

    from lgk.steps import provision as prov

    (tmp_path / "graph_mapping.yaml").write_text(_yaml.safe_dump(SET_MAPPING, sort_keys=False))
    order = []
    monkeypatch.setattr(prov, "require_clear", lambda rf: {})
    monkeypatch.setattr(prov.binding, "stamp_id", lambda clear: "stamp")
    monkeypatch.setattr(prov.binding, "record_version", lambda *a: order.append("record_version"))
    for name in ("require_sync_owner", "_refuse_existing_version", "_refuse_unbound_build",
                 "_create_synced_tables", "_mark_pg_version_schema", "_lock_version_from_reader",
                 "_stage_version_functions", "_prove_reader_isolated"):
        monkeypatch.setattr(prov, name, lambda *a, _n=name, **k: order.append(_n))
    conn = _IndexConn()
    conn.autocommit, conn.close = False, lambda: None
    monkeypatch.setattr(prov, "connect_branch", lambda c: order.append("indexes") or conn)
    prov.provision({"graph_name": "g", "run_folder": str(tmp_path), "kit_schema": "kit", "version": 2})
    assert [s for s in conn.sql if "lgk_nodes_wk_" in s] == [
        "CREATE INDEX IF NOT EXISTS lgk_nodes_wk_provider_npi_idx\n"
        "  ON kit_v2.nodes ((properties ->> 'npi')) WHERE node_type = 'provider'"]
    assert [s for s in conn.sql if "lgk_edges_" in s] == [
        "CREATE INDEX IF NOT EXISTS lgk_edges_src_node_id_idx ON kit_v2.edges (src_node_id)",
        "CREATE INDEX IF NOT EXISTS lgk_edges_dst_node_id_idx ON kit_v2.edges (dst_node_id)"]
    assert order.index("_mark_pg_version_schema") < order.index("indexes") < order.index("_lock_version_from_reader")
    assert order.index("indexes") < order.index("record_version")


def test_the_reapply_step_reads_the_run_mapping_for_walk_key_indexes(tmp_path, monkeypatch):
    """reapply's index step itself (not only ensure_indexes) threads the run's mapping through."""
    import yaml as _yaml

    from lgk.steps import reapply as rp

    (tmp_path / "graph_mapping.yaml").write_text(_yaml.safe_dump(SET_MAPPING, sort_keys=False))
    conn = _IndexConn()
    conn.autocommit, conn.close = False, lambda: None
    monkeypatch.setattr(rp, "connect_branch", lambda c: conn)
    rp._reapply_indexes({"graph_name": "g", "run_folder": str(tmp_path), "kit_schema": "kit", "version": 4})
    assert [s for s in conn.sql if "lgk_nodes_wk_" in s] == [
        "CREATE INDEX IF NOT EXISTS lgk_nodes_wk_provider_npi_idx\n"
        "  ON kit_v4.nodes ((properties ->> 'npi')) WHERE node_type = 'provider'"]


# ------------------------------------------------------------ behaviour on a local Postgres
def _load_set_graph(conn, nodes, edges):
    """nodes: (node_id, node_type, natural_key, label, {props}); edges: (src, dst, type[, weight])."""
    conn.execute("DROP SCHEMA IF EXISTS lgk_ws CASCADE")
    conn.execute("CREATE SCHEMA lgk_ws")
    conn.execute("CREATE TABLE lgk_ws.nodes_view (node_id text PRIMARY KEY, node_type text, "
                 "natural_key text, label text, properties jsonb)")
    conn.execute("CREATE TABLE lgk_ws.edges_view (src_node_id text, dst_node_id text, "
                 "relationship_type text, weight bigint)")
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO lgk_ws.nodes_view VALUES (%s, %s, %s, %s, %s::jsonb)",
                        [(n, t, k, lb, json.dumps(p)) for n, t, k, lb, p in nodes])
        cur.executemany("INSERT INTO lgk_ws.edges_view VALUES (%s, %s, %s, %s)",
                        [(*e, None)[:4] for e in edges])


def _py_web_set(nodes, edges, seed, stop, depth, top_n):
    """The set walk from §3.1, in Python: [(hop, node_id, support, matched)]."""
    props = {n: (t, k, lb, p) for n, t, k, lb, p in nodes}

    def hit(pred, value, n):
        if pred is None or n not in props:
            return False
        t, k, lb, p = props[n]
        if t != pred.node_type:
            return False
        if pred.field is None:
            return True
        got = {"label": lb, "natural_key": k}.get(pred.field, p.get(pred.field))
        return got == value

    frontier = {n: False for n in props if hit(seed[0], seed[1], n)}  # seeds always expand
    seen, out = set(frontier), []
    for hop in range(1, depth + 1):
        counts: dict = {}
        for s_, d, *_ in edges:
            if s_ in frontier and not frontier[s_] and d not in seen:
                counts[d] = counts.get(d, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        frontier = {n: hit(stop[0] if stop else None, stop[1] if stop else None, n) for n, _ in ranked}
        seen |= set(frontier)
        out += [(hop, n, c, frontier[n]) for n, c in ranked]
    return out


def _set_rows(conn, sql, seed_value=None, stop_value=None):
    params = {"seed_value": seed_value, "stop_value": stop_value}
    return [(h, n, sup, m) for h, n, sup, _sc, m in conn.execute(sql, params).fetchall()]


# The hostile set graph (ticket §7 rows, offline twin of the live fixture).
SET_NODES = [
    # hostile: reordered input — the seeds are inserted in descending node_id order
    ("p_c4", "provider", "c4", "Dr C4", {"specialty": "cardiology", "network_status": "out"}),
    ("p_c3", "provider", "c3", "Dr C3", {"specialty": "cardiology"}),
    ("p_c2", "provider", "c2", "Dr C2", {"specialty": "cardiology"}),
    ("p_c1", "provider", "c1", "Dr C1", {"specialty": "cardiology"}),
    # hostile: case and a trailing space — equality is exact
    ("p_cx", "provider", "cx", "Dr CX", {"specialty": "Cardiology "}),
    ("p_out", "provider", "out", "Dr Out", {"specialty": "neuro", "network_status": "out"}),
    ("p_mid", "provider", "mid", "Dr Mid", {"specialty": "neuro"}),
    ("p_x", "provider", "x", "Dr X", {"specialty": "neuro"}),
    ("p_end", "provider", "end", "Dr End", {}),
    ("p_trap", "provider", "trap", "Dr Trap", {}),
    ("h_z", "hospital", "z", "H Z", {"network_status": "in"}),
    ("h_in", "hospital", "in", "H In", {"network_status": "in"}),
    ("h_out", "hospital", "hout", "H Out", {"network_status": "out"}),
    # hostile: wrong type, same value — a hospital with specialty cardiology
    ("h_card", "hospital", "card", "H Card", {"specialty": "cardiology", "network_status": "in"}),
    ("h_deep", "hospital", "deep", "H Deep", {"network_status": "in"}),
    ("h_far", "hospital", "far", "H Far", {"network_status": "in"}),
    ("h_only_x", "hospital", "onlyx", "H OnlyX", {"network_status": "in"}),
]
SET_EDGES = [
    ("p_c1", "h_z", "affiliation"), ("p_c2", "h_z", "affiliation"),  # hostile: Z from two seeds
    ("p_c1", "h_out", "affiliation"),
    ("p_c2", "h_in", "affiliation"), ("p_c3", "h_in", "affiliation"), ("p_c4", "h_in", "affiliation"),
    ("p_cx", "h_only_x", "affiliation"),
    ("h_out", "p_x", "refers"),  # hostile: X reachable only through the stop node
    ("p_x", "h_far", "affiliation"),
    ("h_card", "p_trap", "refers"),  # reached only if the seed scan ignores node_type
    ("h_z", "p_out", "refers"), ("h_in", "p_mid", "refers"), ("h_in", "p_c1", "refers"),
    ("p_out", "h_far", "affiliation"),  # hostile: a provider with network_status out expands
    ("p_mid", "h_deep", "affiliation"), ("h_deep", "p_end", "refers"), ("h_far", "p_c1", "refers"),
]
CARDIO = (NodePredicate("provider", "specialty"), "cardiology")
OUT = (NodePredicate("hospital", "network_status"), "out")


@pytest.fixture
def set_pg(local_pg):
    _load_set_graph(local_pg, SET_NODES, SET_EDGES)
    return local_pg


def test_set_walk_equals_the_python_reference(set_pg):
    preds = [(CARDIO, None), (CARDIO, OUT), ((NodePredicate("hospital", "network_status"), "out"), OUT),
             ((NodePredicate("provider"), None), OUT), ((NodePredicate("provider", "label"), "Dr C1"), None)]
    for seed, stop in preds:
        for depth in (1, 2, 3, 4):
            for top_n in (1, 2, 25):
                sql = _ws_sql(depth, top_n, seed[0], stop[0] if stop else None)
                got = _set_rows(set_pg, sql, seed[1], stop[1] if stop else None)
                assert got == _py_web_set(SET_NODES, SET_EDGES, seed, stop, depth, top_n), (seed, stop, depth, top_n)
    # the walk is not vacuous: X is behind the stop, Z has support 2, the stop node is matched
    rows = _set_rows(set_pg, _ws_sql(4, 25, CARDIO[0], OUT[0]), "cardiology", "out")
    assert (1, "h_z", 2, False) in rows and (1, "h_out", 1, True) in rows
    assert "p_x" not in {n for _h, n, _s, _m in rows} and "p_trap" not in {n for _h, n, _s, _m in rows}
    assert (3, "h_far", 1, False) in rows and (4, "p_end", 1, False) in rows


def test_seed_cap_raises_with_the_message_and_never_truncates(set_pg):
    import psycopg

    # hostile: varied configuration — the cap 3 is not the fixture's seed count (4)
    with pytest.raises(psycopg.errors.InvalidTextRepresentation,
                       match='invalid input syntax for type integer: "lgk: seed set exceeds max_seeds=3"'):
        set_pg.execute(_ws_sql(1, 25, CARDIO[0], max_seeds=3), {"seed_value": "cardiology"}).fetchall()
    # at the cap exactly it runs (the cast is not folded at plan time) and keeps every seed
    assert _set_rows(set_pg, _ws_sql(1, 25, CARDIO[0], max_seeds=4), "cardiology") == \
        [(1, "h_in", 3, False), (1, "h_z", 2, False), (1, "h_out", 1, False)]


def test_single_seed_set_equals_web_sql_offline(set_pg):
    """One natural_key seed: exactly web_sql's rows from that node (matched false)."""
    for node_id, _t, key, _lb, _p in SET_NODES:
        for depth in (1, 2, 3, 4):
            for top_n in (1, 2, 25):
                web = set_pg.execute(web_sql(depth, "lgk_ws", top_n), {"start_node_id": node_id}).fetchall()
                got = set_pg.execute(_ws_sql(depth, top_n, NodePredicate(_t, "natural_key")),
                                     {"seed_value": key}).fetchall()
                assert got == [(*r, False) for r in web], (node_id, depth, top_n)


def test_folded_set_walk_hides_below_floor_offline(local_pg):
    nodes = [(f"p{i}", "provider", f"k{i}", f"P{i}", {"g": "a" if i < 2 else "b"}) for i in range(5)]
    # p0 -> p2 weight 11 (kept), p1 -> p2 weight 12 (MIN 11), p0 -> p3 weight 10 (hidden), p1 -> p4 12
    edges = [("p0", "p2", "f", 11), ("p1", "p2", "f", 12), ("p0", "p3", "f", 10), ("p1", "p4", "f", 12)]
    _load_set_graph(local_pg, nodes, edges)
    sql = web_set_sql(1, "lgk_ws", 25, NodePredicate("provider", "g"), walk_keys={"provider": ("g",)},
                      folded=True, privacy_floor=11)
    assert _set_rows(local_pg, sql, "a") == [(1, "p4", 12, False), (1, "p2", 11, False)]


# ------------------------------------------------------------ the function form (X11)
def _set_function(conn, mapping_types=("affiliation", "refers"), fold_types=(), mixed=False):
    from lgk import served_functions as sf
    from lgk.privacy.floor import walk_count_terms

    conn.execute("DROP SCHEMA IF EXISTS lgk_fs CASCADE")
    conn.execute("CREATE SCHEMA lgk_fs")
    conn.execute("CREATE TABLE lgk_fs.nodes AS SELECT * FROM lgk_ws.nodes_view")
    conn.execute("CREATE TABLE lgk_fs.edges AS SELECT * FROM lgk_ws.edges_view")
    typed = [(t, walk_count_terms(11, t in fold_types), t in fold_types) for t in mapping_types]
    fn = sf._coordinator_web_set("lgk_fs", walk_count_terms(None, False), False, 3,
                                 {"provider": ("specialty",), "hospital": ("network_status",)}, typed,
                                 "edge_types is required" if mixed else "")
    conn.execute(sf.create_sql(fn, "lgk_fs"))
    for args in fn.smoke_args:
        conn.execute(f"SELECT count(*) FROM lgk_fs.{fn.name}({args})").fetchall()
    return fn


def test_coordinator_web_set_equals_the_view_form(set_pg):
    import psycopg

    fn = _set_function(set_pg)
    assert fn.signature("s") == "s.coordinator_web_set(text, text, text, text, integer, text)"
    assert len(fn.smoke_args) == 3 * 2 * 3 * 4  # selections x seed keys x (stop keys + NULL) x depths
    call = "SELECT hop, node_id, support, matched FROM lgk_fs.coordinator_web_set(%s, %s, %s, %s, %s, %s)"
    for depth in (1, 2, 3, 4):
        for stop in (None, OUT):
            for edge in (None, "affiliation"):
                got = set_pg.execute(call, ("provider.specialty", "cardiology",
                                            "hospital.network_status" if stop else None,
                                            stop[1] if stop else None, depth, edge)).fetchall()
                edges = [e for e in SET_EDGES if edge is None or e[2] == edge]
                assert got == _py_web_set(SET_NODES, edges, CARDIO, stop, depth, 3), (depth, stop, edge)
    # non-vacuous: at top_n 3 the stop decides whether X is reached (h_out is kept at hop 1)
    assert _py_web_set(SET_NODES, SET_EDGES, CARDIO, None, 2, 3) != _py_web_set(SET_NODES, SET_EDGES, CARDIO, OUT, 2, 3)
    for args, match in [(("provider.label", "x", None, None, 1, None), "seed_key must be one of"),
                        (("provider.specialty", "x", "provider.zip", "y", 1, None), "stop_key must be NULL or one of"),
                        (("provider.specialty", "x", None, None, 1, "nope"), "edge_type must be one of"),
                        (("provider.specialty", "x", None, None, 5, None), "depth must be in 1..4")]:
        with pytest.raises(psycopg.errors.RaiseException, match=match):
            set_pg.execute(call, args).fetchall()
    mixed = _set_function(set_pg, mixed=True)
    with pytest.raises(psycopg.errors.RaiseException, match="edge_types is required"):
        set_pg.execute(call, ("provider.specialty", "cardiology", None, None, 1, None)).fetchall()
    assert len(mixed.smoke_args) == 2 * 2 * 3 * 4  # no NULL selection on a mixed graph


def test_advisor_worst_case_seed_value_runs_on_postgres(set_pg):
    """advise's seed-value query is valid SQL and picks the value whose seed set has the most out-edges."""
    from lgk.steps.advise import _worst_case_seed_value

    set_pg.execute("DROP SCHEMA IF EXISTS lgk_adv CASCADE")
    set_pg.execute("CREATE SCHEMA lgk_adv")
    set_pg.execute("CREATE TABLE lgk_adv.nodes AS SELECT * FROM lgk_ws.nodes_view")
    set_pg.execute("CREATE TABLE lgk_adv.edges AS SELECT * FROM lgk_ws.edges_view")
    seed = NodePredicate("provider", "specialty")
    assert _worst_case_seed_value(set_pg, "lgk_adv", seed, 200) == "cardiology"  # 6 out-edges
    # hostile: fewer seeds, more out-edges — one ortho provider with 7 out-edges beats 4 cardiologists
    set_pg.execute("INSERT INTO lgk_adv.nodes VALUES ('p_or', 'provider', 'or', 'Dr Or', "
                   "'{\"specialty\": \"ortho\"}'::jsonb)")
    set_pg.execute("INSERT INTO lgk_adv.edges SELECT 'p_or', 'h_' || i, 'affiliation', NULL "
                   "FROM generate_series(1, 7) i")
    assert _worst_case_seed_value(set_pg, "lgk_adv", seed, 200) == "ortho"
    set_pg.execute("DELETE FROM lgk_adv.edges WHERE src_node_id = 'p_or'")
    set_pg.execute("DELETE FROM lgk_adv.nodes WHERE node_id = 'p_or'")
    # hostile: a cap below the cardiology set (4) excludes it; neuro (3 nodes, 3 out-edges) wins
    assert _worst_case_seed_value(set_pg, "lgk_adv", seed, 3) == "neuro"


# ------------------------------------- final review P1: edge_types needs the run folder's mapping
@pytest.mark.parametrize("folded", [True, False])
@pytest.mark.parametrize("selection", [["shares_refs", "affiliation"], ["affiliation"], ["shares_refs"]])
def test_edge_types_without_a_run_folder_are_refused(folded, selection):
    """With no mapping the gate cannot see which types are folded, so it cannot refuse a mix:
    naming edge_types without run_folder is refused, whatever `folded` the caller states."""
    from lgk.privacy.gate import PrivacyGateError, served_walk_selection

    # hostile: a mixed selection with a stated folded=True was accepted before the fix
    with pytest.raises(PrivacyGateError, match="needs run_folder"):
        served_walk_selection(None, None, folded, selection)
    with pytest.raises(PrivacyGateError, match="needs run_folder"):
        get_walk_query("x", 1, "lgk_care", folded=folded, edge_types=selection)
    with pytest.raises(PrivacyGateError, match="needs run_folder"):
        get_walk_query("x", 1, "lgk_care", folded=folded, edge_types=selection,
                       seed=NodePredicate("provider", "npi"))


def test_no_run_folder_and_no_edge_types_keeps_the_stated_folded():
    """The unchanged path: no run folder, no edge_types -> the caller's folded and floor, no IN."""
    from lgk.privacy.gate import served_walk_selection

    assert served_walk_selection(None, None, False, None) == (11, False, None)
    assert served_walk_selection(None, 13, True, None) == (13, True, None)
    folded = get_walk_query("x", 1, "lgk_care", folded=True)
    assert "HAVING MIN(e.weight) >= 11" in folded and "relationship_type IN" not in folded
    assert "COUNT(*) AS support" in get_walk_query("x", 1, "lgk_care", folded=False)


def test_edge_types_with_a_run_folder_still_derive_folded(mixed_run):
    """The fix is narrow: with a run folder the same selections work or are refused as before."""
    from lgk.privacy.gate import MixedFoldSelectionError, served_walk_selection

    assert served_walk_selection(mixed_run, None, None, ["shares_refs"]) == (11, True, ("shares_refs",))
    assert served_walk_selection(mixed_run, None, None, ["affiliation"]) == (11, False, ("affiliation",))
    with pytest.raises(MixedFoldSelectionError):
        served_walk_selection(mixed_run, None, True, ["shares_refs", "affiliation"])
