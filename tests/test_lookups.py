"""Offline checks of the four lookups (2f) and the served score join of a sorted walk (2e).

String checks pin the SQL contract. The behaviour checks run the real SQL on a throwaway
local Postgres cluster (initdb in a temp dir, unix socket only, its own port so it can run
in one session with tests/test_walks.py); they skip when no local Postgres is installed.
Every expected answer below is written by hand from the fixture, never computed by the code
under test.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from lgk.steps.walks import get_lookup_query, get_walk_query
from lgk.walks.lookups import (
    EnrichmentNotServedError,
    ancestors_sql,
    descendants_sql,
    same_group_sql,
    similar_jaccard_sql,
)

ANCESTORS_GOLDEN = (
    "SELECT c.ancestor_id AS node_id, c.depth\n"
    "FROM lgk_providers.closure_view c\n"
    "WHERE c.descendant_id = CAST(%(start_node_id)s AS text)\n"
    "  AND c.depth <= 3\n"
    'ORDER BY c.depth, c.ancestor_id COLLATE "C"\n'
    "LIMIT 25"
)
DESCENDANTS_GOLDEN = (
    "SELECT c.descendant_id AS node_id, c.depth\n"
    "FROM lgk_providers.closure_view c\n"
    "WHERE c.ancestor_id = CAST(%(start_node_id)s AS text)\n"
    "  AND c.depth <= 3\n"
    'ORDER BY c.depth, c.descendant_id COLLATE "C"\n'
    "LIMIT 25"
)
SAME_GROUP_GOLDEN = (
    "SELECT g.node_id, g.group_id\n"
    "FROM lgk_providers.node_group_view s\n"
    "JOIN lgk_providers.node_group_view g\n"
    "  ON g.group_id = s.group_id AND g.method = s.method\n"
    "WHERE s.node_id = CAST(%(start_node_id)s AS text)\n"
    "  AND s.method = 'components'\n"
    "  AND g.node_id <> s.node_id\n"
    'ORDER BY g.node_id COLLATE "C"\n'
    "LIMIT 25"
)
SIMILAR_GOLDEN = (
    "SELECT s.dst_id AS node_id, s.score\n"
    "FROM lgk_providers.similar_to_view s\n"
    "WHERE s.src_id = CAST(%(start_node_id)s AS text)\n"
    'ORDER BY s.score DESC, s.dst_id COLLATE "C"\n'
    "LIMIT 10"
)
CYCLE_REASON = "the graph has a cycle; closure is only for trees and DAGs"
NO_EDGES = "the graph has no edges"


# ---------------------------------------------------------------------------- SQL contract


def test_ancestors_sql_golden():
    assert ancestors_sql("lgk_providers", 3, 25) == ANCESTORS_GOLDEN


def test_descendants_sql_golden():
    assert descendants_sql("lgk_providers", 3, 25) == DESCENDANTS_GOLDEN


def test_top_n_defaults_to_25():
    assert ancestors_sql("lgk_providers", 3) == ANCESTORS_GOLDEN
    assert descendants_sql("lgk_providers", 3) == DESCENDANTS_GOLDEN


@pytest.mark.parametrize("builder", [ancestors_sql, descendants_sql])
@pytest.mark.parametrize("bad", [0, 101, True, -1, 2.0, "3"])
def test_bad_max_depth_refused(builder, bad):
    with pytest.raises(ValueError, match="max_depth"):
        builder("lgk_providers", bad, 25)


@pytest.mark.parametrize("builder", [ancestors_sql, descendants_sql])
def test_max_depth_cap_is_the_enrich_cap(builder):
    from lgk.enrich.shape import MAX_DEPTH

    assert "c.depth <= 100" in builder("lgk_providers", MAX_DEPTH, 25)
    with pytest.raises(ValueError):
        builder("lgk_providers", MAX_DEPTH + 1, 25)


@pytest.mark.parametrize("builder", [ancestors_sql, descendants_sql])
@pytest.mark.parametrize("bad", [0, -5, True])
def test_bad_top_n_refused(builder, bad):
    with pytest.raises(ValueError, match="top_n"):
        builder("lgk_providers", 3, bad)


@pytest.mark.parametrize("builder", [ancestors_sql, descendants_sql])
@pytest.mark.parametrize("bad", ["x; DROP", "Lgk", "", "a.b"])
def test_bad_schema_refused(builder, bad):
    with pytest.raises(ValueError):
        builder(bad, 3, 25)


def test_same_group_sql_golden():
    assert same_group_sql("lgk_providers", 25) == SAME_GROUP_GOLDEN
    assert same_group_sql("lgk_providers") == SAME_GROUP_GOLDEN  # defaults: top_n 25, components
    assert same_group_sql("lgk_providers", 25, method="components") == SAME_GROUP_GOLDEN


def test_similar_jaccard_sql_golden():
    assert similar_jaccard_sql("lgk_providers", 10) == SIMILAR_GOLDEN
    assert similar_jaccard_sql("lgk_providers") == SIMILAR_GOLDEN  # defaults: top_n 10, top_k 20


@pytest.mark.parametrize("bad", ["other", "Components", "pagerank", "similarity", "closure",
                                 "community_detection", "", None, "components'; DROP"])
def test_same_group_refuses_a_method_that_writes_no_node_group(bad):
    # hostile: pagerank/similarity/closure are real enrichment methods, community_detection
    # writes node_group but is "coming" (not computed by v1), "Components" differs in case only.
    with pytest.raises(ValueError, match="method"):
        same_group_sql("lgk_providers", 25, method=bad)


@pytest.mark.parametrize("bad", [0, -1, True, 2.0, "5"])
def test_same_group_refuses_a_bad_top_n(bad):
    with pytest.raises(ValueError, match="top_n"):
        same_group_sql("lgk_providers", bad)


@pytest.mark.parametrize("builder", [same_group_sql, similar_jaccard_sql])
@pytest.mark.parametrize("bad", ["x; DROP", "Lgk", "", "a.b"])
def test_same_group_and_similar_refuse_a_bad_schema(builder, bad):
    with pytest.raises(ValueError, match="served_schema"):
        builder(bad)


def test_similar_jaccard_refuses_top_n_above_top_k():
    with pytest.raises(ValueError, match="top_n"):
        similar_jaccard_sql("lgk_providers", 21)  # default top_k 20
    # hostile: a top_k other than the default 20, so a hard-coded 20 cannot pass
    with pytest.raises(ValueError, match="top_n"):
        similar_jaccard_sql("lgk_providers", 6, top_k=5)
    assert similar_jaccard_sql("lgk_providers", 5, top_k=5).endswith("LIMIT 5")  # boundary kept
    assert similar_jaccard_sql("lgk_providers", 20).endswith("LIMIT 20")
    assert similar_jaccard_sql("lgk_providers", 30, top_k=40).endswith("LIMIT 30")


@pytest.mark.parametrize("bad", [0, -1, True, 2.0])
def test_similar_jaccard_refuses_a_bad_top_n(bad):
    with pytest.raises(ValueError, match="top_n"):
        similar_jaccard_sql("lgk_providers", bad)


@pytest.mark.parametrize("bad", [0, -3, True, "20", 20.0])
def test_similar_jaccard_refuses_a_bad_top_k(bad):
    with pytest.raises(ValueError, match="top_k"):
        similar_jaccard_sql("lgk_providers", 1, top_k=bad)


# ---------------------------------------------------------------------------- get_lookup_query


def _report(folder: Path, closure_available: bool, reason: str = "", with_closure: bool = False,
            extra_menu=(), extra_methods=None):
    menu = [
        {"method": "degree", "question": "q", "example": "e", "table": "node_score",
         "available": True, "reason": ""},
        {"method": "closure", "question": "q", "example": "e", "table": "closure",
         "available": closure_available, "reason": reason},
    ]
    menu += [{"method": m, "question": "q", "example": "e", "table": t,
              "available": not why, "reason": why} for m, t, why in extra_menu]
    report = {"shape": {"nodes": 5}, "menu": menu, "degree": {"rows": 5}}
    if with_closure:
        report["closure"] = {"rows": 8}
    report.update(extra_methods or {})
    (folder / "enrich_report.json").write_text(json.dumps(report))


def test_lookup_refused_with_the_cycle_reason_when_closure_is_greyed(tmp_path):
    # hostile: the report exists and has OTHER methods (degree), so "a report exists" alone
    # must not decide the message; the greyed menu row's reason must be served.
    _report(tmp_path, closure_available=False, reason=CYCLE_REASON)
    with pytest.raises(EnrichmentNotServedError) as err:
        get_lookup_query("ancestors", "lgk_providers", str(tmp_path), max_depth=3)
    assert str(err.value) == f"closure was not computed: {CYCLE_REASON}"


def test_lookup_refused_when_closure_was_not_ticked(tmp_path):
    _report(tmp_path, closure_available=True)  # hostile: available, has other methods, not run
    with pytest.raises(EnrichmentNotServedError) as err:
        get_lookup_query("descendants", "lgk_providers", tmp_path, max_depth=3)
    assert str(err.value) == "closure was not computed: it was not ticked in enrich.yaml"


def test_lookup_refused_when_there_is_no_report(tmp_path):
    with pytest.raises(EnrichmentNotServedError) as err:
        get_lookup_query("ancestors", "lgk_providers", str(tmp_path), max_depth=3)
    assert str(err.value) == (f"closure was not computed: {tmp_path / 'enrich_report.json'} "
                              "does not exist (run enrich first)")


def test_enrichment_not_served_is_a_runtime_error():
    assert issubclass(EnrichmentNotServedError, RuntimeError)


def test_lookup_returns_the_golden_sql_when_closure_is_served(tmp_path):
    _report(tmp_path, closure_available=True, with_closure=True)
    assert get_lookup_query("ancestors", "lgk_providers", str(tmp_path),
                            max_depth=3, top_n=25) == ANCESTORS_GOLDEN
    assert get_lookup_query("descendants", "lgk_providers", str(tmp_path),
                            max_depth=3, top_n=25) == DESCENDANTS_GOLDEN


@pytest.mark.parametrize("kind", ["same_groups", "similar", "Ancestors", "", None])
def test_lookup_refuses_other_kinds(tmp_path, kind):
    _report(tmp_path, closure_available=True, with_closure=True)  # hostile: closure IS served
    with pytest.raises(ValueError, match="ancestors.*descendants.*same_group.*similar_jaccard"):
        get_lookup_query(kind, "lgk_providers", str(tmp_path), max_depth=3)


def _served_all(folder: Path, top_k: int = 20) -> None:
    _report(folder, closure_available=True, with_closure=True,
            extra_menu=[("components", "node_group", ""), ("similarity", "similar_to", "")],
            extra_methods={"components": {"rounds": 2, "groups": 3},
                           "similarity": {"rows": 9, "top_k": top_k}})


def test_lookup_returns_the_golden_sql_for_same_group_and_similar(tmp_path):
    _served_all(tmp_path)
    assert get_lookup_query("same_group", "lgk_providers", tmp_path, top_n=25) == SAME_GROUP_GOLDEN
    assert get_lookup_query("same_group", "lgk_providers", tmp_path) == SAME_GROUP_GOLDEN
    assert get_lookup_query("similar_jaccard", "lgk_providers", str(tmp_path), top_n=10) == SIMILAR_GOLDEN
    assert get_lookup_query("similar_jaccard", "lgk_providers", str(tmp_path)) == SIMILAR_GOLDEN


def test_same_group_refused_with_the_components_reason(tmp_path):
    # hostile: closure IS served, so the check must look at the method behind THIS kind
    _report(tmp_path, closure_available=True, with_closure=True,
            extra_menu=[("components", "node_group", NO_EDGES)])
    with pytest.raises(EnrichmentNotServedError) as err:
        get_lookup_query("same_group", "lgk_providers", tmp_path)
    assert str(err.value) == f"components was not computed: {NO_EDGES}"


def test_similar_refused_when_similarity_was_not_ticked(tmp_path):
    # hostile: components and closure are served; similarity is available but was not run
    _report(tmp_path, closure_available=True, with_closure=True,
            extra_menu=[("components", "node_group", ""), ("similarity", "similar_to", "")],
            extra_methods={"components": {"rounds": 2, "groups": 3}})
    with pytest.raises(EnrichmentNotServedError) as err:
        get_lookup_query("similar_jaccard", "lgk_providers", tmp_path)
    assert str(err.value) == "similarity was not computed: it was not ticked in enrich.yaml"
    assert get_lookup_query("same_group", "lgk_providers", tmp_path) == SAME_GROUP_GOLDEN


def test_closure_lookups_need_closure_even_when_the_others_are_served(tmp_path):
    _report(tmp_path, closure_available=False, reason=CYCLE_REASON,
            extra_menu=[("components", "node_group", ""), ("similarity", "similar_to", "")],
            extra_methods={"components": {"rounds": 2, "groups": 3},
                           "similarity": {"rows": 9, "top_k": 20}})
    with pytest.raises(EnrichmentNotServedError, match=f"closure was not computed: {CYCLE_REASON}"):
        get_lookup_query("descendants", "lgk_providers", tmp_path, max_depth=3)
    assert get_lookup_query("same_group", "lgk_providers", tmp_path) == SAME_GROUP_GOLDEN


def test_similar_lookup_reads_top_k_from_the_report(tmp_path):
    # hostile: the report's top_k is 5, not the default 20
    _served_all(tmp_path, top_k=5)
    with pytest.raises(ValueError, match="top_n"):
        get_lookup_query("similar_jaccard", "lgk_providers", tmp_path, top_n=6)
    assert get_lookup_query("similar_jaccard", "lgk_providers", tmp_path, top_n=5).endswith("LIMIT 5")
    # the default top_n 10 is above this run's top_k 5
    with pytest.raises(ValueError, match="top_n"):
        get_lookup_query("similar_jaccard", "lgk_providers", tmp_path)


def test_similar_lookup_refuses_a_caller_top_k(tmp_path):
    _served_all(tmp_path, top_k=5)
    with pytest.raises(ValueError, match="top_k"):
        get_lookup_query("similar_jaccard", "lgk_providers", tmp_path, top_n=10, top_k=40)


def test_similar_lookup_refuses_a_report_without_top_k(tmp_path):
    # hostile: similarity was computed, but its report block has no top_k (old or hand-edited)
    _served_all(tmp_path)
    report = json.loads((tmp_path / "enrich_report.json").read_text())
    del report["similarity"]["top_k"]
    (tmp_path / "enrich_report.json").write_text(json.dumps(report))
    with pytest.raises(EnrichmentNotServedError, match="similarity has no top_k"):
        get_lookup_query("similar_jaccard", "lgk_providers", tmp_path, top_n=1)


def test_same_group_lookup_passes_the_method_through(tmp_path):
    _served_all(tmp_path)
    with pytest.raises(ValueError, match="method"):
        get_lookup_query("same_group", "lgk_providers", tmp_path, method="other")


# ---------------------------------------------------------------------------- get_walk_query sort (2e)


def test_sorted_walk_joins_the_served_score_view():
    """2e: a score-sorted walk reads <served>.node_score_view, never a version schema."""
    sql = get_walk_query("x", 2, "lgk_providers", folded=False, sort_method="degree")
    assert sql.count("LEFT JOIN lgk_providers.node_score_view s "
                     "ON s.node_id = c.node_id AND s.method = 'degree'") == 2
    assert ".node_score s" not in sql


def test_unsorted_walk_has_no_score_join():
    assert "node_score" not in get_walk_query("x", 2, "lgk_providers", folded=False)


# ---------------------------------------------------------------------------- local Postgres

PORT = 54331  # tests/test_walks.py uses 54329; both module-scoped clusters may run together

PG_BIN = next((Path(p) for p in ("/opt/homebrew/opt/postgresql@17/bin", "/opt/homebrew/opt/postgresql@16/bin",
                                 "/usr/lib/postgresql/17/bin", "/usr/lib/postgresql/16/bin")
               if (Path(p) / "initdb").exists()), None) or (
    Path(shutil.which("initdb")).parent if shutil.which("initdb") else None)

FAN = ["C_99"] + [f"c_{i:02d}" for i in range(40)]  # fixture ids (input, not an expected answer)

CLOSURE_ROWS = [
    # hostile: a diamond with unequal arms r->a->d and r->b->x->d; closure holds the
    # SHORTEST depth per pair, so (r, d, 2) only, never (r, d, 3).
    ("r", "a", 1), ("r", "b", 1), ("a", "d", 1), ("b", "x", 1), ("x", "d", 1),
    ("r", "x", 2), ("b", "d", 2),
    ("r", "d", 2),  # hostile: depth == max_depth 2 boundary (catches <= -> <)
] + [
    # hostile: a fan-out root f with 41 children, so top_n=25 cuts; "C_99" sorts first in
    # byte order ("C" < "c") but last in a case-insensitive order.
    ("f", child, 1) for child in FAN
]


GROUP_ROWS = [
    # group g1 = {a, b, c, C_9}; inserted in descending byte order (the answer is ascending)
    ("c", "components", "g1"), ("b", "components", "g1"), ("a", "components", "g1"),
    ("C_9", "components", "g1"),  # hostile: "C" < "a" in byte order, last case-insensitively
    ("a", "other", "g1"),  # hostile: second method row; must not join (no duplicate a)
    ("y", "other", "g1"),  # hostile: in g1 only under another method; must not appear
    ("p", "components", "g2"), ("q", "components", "g2"),  # a second component
    ("z", "components", "g3"),  # hostile: isolated node, alone in its group
] + [
    # hostile: a group of 30, so top_n=25 cuts
    (f"m_{i:02d}", "components", "g4") for i in reversed(range(30))
]

SIMILAR_ROWS = [
    # hostile: Jaccard ties u3, u1, u2 at 1.0 (inserted out of order): dst_id byte order decides
    ("v", "u3", 1.0), ("v", "u1", 1.0), ("v", "w", 0.25), ("v", "u2", 1.0),
    ("v", "U9", 1.0),  # hostile: "U" < "u" in byte order
    ("v", "k", 0.5),
    ("u1", "v", 1.0),  # hostile: the reverse row belongs to u1, not to v
]


@pytest.fixture(scope="module")
def local_pg():
    """A throwaway Postgres cluster on a unix socket, with the served relations as tables."""
    if PG_BIN is None:
        pytest.skip("no local Postgres binaries (initdb) installed")
    import psycopg

    tmp = Path(tempfile.mkdtemp(prefix="lgkpgl"))
    data, sock = tmp / "data", tmp / "s"
    sock.mkdir()
    subprocess.run([PG_BIN / "initdb", "-D", data, "-U", "postgres", "--no-sync", "-E", "UTF8",
                    "--locale=C"], check=True, capture_output=True)
    subprocess.run([PG_BIN / "pg_ctl", "-D", data, "-w", "-l", tmp / "log", "-o",
                    f"-k {sock} -c listen_addresses='' -p {PORT}", "start"], check=True,
                   capture_output=True)
    conn = None
    try:
        conn = psycopg.connect(host=str(sock), port=PORT, user="postgres", dbname="postgres",
                               autocommit=True)
        conn.execute("CREATE SCHEMA lgk_providers")
        conn.execute("CREATE TABLE lgk_providers.closure_view "
                     "(ancestor_id text, descendant_id text, depth integer)")
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO lgk_providers.closure_view VALUES (%s, %s, %s)", CLOSURE_ROWS)
        # hostile: a non-byte collation on the text columns (the cluster itself is --locale=C), so
        # only the SQL's own COLLATE "C" gives byte order ("C_9" < "a", "U9" < "u1").
        conn.execute('CREATE TABLE lgk_providers.node_group_view (node_id text COLLATE "und-x-icu", '
                     'method text, group_id text)')
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO lgk_providers.node_group_view VALUES (%s, %s, %s)", GROUP_ROWS)
        conn.execute('CREATE TABLE lgk_providers.similar_to_view (src_id text, '
                     'dst_id text COLLATE "und-x-icu", score float8)')
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO lgk_providers.similar_to_view VALUES (%s, %s, %s)", SIMILAR_ROWS)
        # 2e: the score exists ONLY as the served view's relation (no node_score anywhere).
        conn.execute("CREATE SCHEMA lgk_scored")
        conn.execute("CREATE TABLE lgk_scored.edges_view (src_node_id text, dst_node_id text, weight bigint)")
        conn.execute("CREATE TABLE lgk_scored.node_score_view (node_id text, method text, score float8)")
        yield conn
    finally:
        if conn is not None:
            conn.close()
        subprocess.run([PG_BIN / "pg_ctl", "-D", data, "-m", "immediate", "stop"], capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)


def _run(conn, sql, seed):
    return [tuple(r) for r in conn.execute(sql, {"start_node_id": seed}).fetchall()]


def test_ancestors_of_the_diamond_bottom(local_pg):
    assert _run(local_pg, ancestors_sql("lgk_providers", 3), "d") == [
        ("a", 1), ("x", 1), ("b", 2), ("r", 2)]
    assert _run(local_pg, ancestors_sql("lgk_providers", 1), "d") == [("a", 1), ("x", 1)]


def test_ancestors_are_not_descendants(local_pg):
    # swapping ancestor_id/descendant_id turns the ancestors of d into its descendants (none)
    assert _run(local_pg, descendants_sql("lgk_providers", 3), "d") == []
    assert _run(local_pg, ancestors_sql("lgk_providers", 3), "a") == [("r", 1)]


@pytest.mark.parametrize("max_depth", [2, 3])
def test_descendants_reach_the_diamond_bottom_once_at_its_shortest_depth(local_pg, max_depth):
    assert _run(local_pg, descendants_sql("lgk_providers", max_depth), "r") == [
        ("a", 1), ("b", 1), ("d", 2), ("x", 2)]


def test_descendants_within_depth_1_exclude_the_diamond_bottom(local_pg):
    assert _run(local_pg, descendants_sql("lgk_providers", 1), "r") == [("a", 1), ("b", 1)]


def test_depth_equal_to_max_depth_is_kept(local_pg):
    # (r, d, 2) and (r, x, 2) sit exactly on max_depth 2
    assert ("d", 2) in _run(local_pg, descendants_sql("lgk_providers", 2), "r")
    assert ("r", 2) in _run(local_pg, ancestors_sql("lgk_providers", 2), "d")


def test_fan_out_cut_to_top_n_in_byte_order(local_pg):
    got = _run(local_pg, descendants_sql("lgk_providers", 3, top_n=25), "f")
    assert got == [(n, 1) for n in [
        "C_99", "c_00", "c_01", "c_02", "c_03", "c_04", "c_05", "c_06", "c_07", "c_08",
        "c_09", "c_10", "c_11", "c_12", "c_13", "c_14", "c_15", "c_16", "c_17", "c_18",
        "c_19", "c_20", "c_21", "c_22", "c_23"]]


def test_sorted_walk_follows_the_served_scores(local_pg):
    """2e as the served view only: the score relation exists only as lgk_scored.node_score_view."""
    local_pg.execute("TRUNCATE lgk_scored.edges_view, lgk_scored.node_score_view")
    with local_pg.cursor() as cur:
        cur.executemany("INSERT INTO lgk_scored.edges_view VALUES (%s, %s, NULL)",
                        [("s", "a"), ("s", "b"), ("s", "c"), ("s", "d")])
        cur.executemany("INSERT INTO lgk_scored.node_score_view VALUES (%s, %s, %s)", [
            ("a", "degree", 1.0), ("b", "degree", 3.0), ("c", "degree", 2.0),
            ("a", "pagerank", 99.0),  # hostile: another method's higher score must not join
        ])  # d has no score: NULLS LAST
    sql = get_walk_query("s", 1, "lgk_scored", folded=False, sort_method="degree")
    rows = _run(local_pg, sql, "s")
    assert [(h, n, score) for h, n, _sup, score in rows] == [
        (1, "b", 3.0), (1, "c", 2.0), (1, "a", 1.0), (1, "d", None)]


def test_sorted_inbound_walk_follows_the_served_scores(local_pg):
    """Sorted lookups x enrichment views: direction=inbound with a sort still reads <served>.node_score_view."""
    local_pg.execute("TRUNCATE lgk_scored.edges_view, lgk_scored.node_score_view")
    with local_pg.cursor() as cur:
        cur.executemany("INSERT INTO lgk_scored.edges_view VALUES (%s, %s, NULL)", [
            ("a", "s"), ("b", "s"), ("c", "s"),
            ("s", "z"),  # hostile: an out-neighbour with the top score must not appear inbound
        ])
        cur.executemany("INSERT INTO lgk_scored.node_score_view VALUES (%s, %s, %s)", [
            ("a", "degree", 1.0), ("b", "degree", 3.0), ("c", "degree", 2.0), ("z", "degree", 9.0),
            ("a", "pagerank", 99.0),  # hostile: another method's higher score must not join
        ])
    sql = get_walk_query("s", 1, "lgk_scored", folded=False, sort_method="degree", direction="inbound")
    rows = _run(local_pg, sql, "s")
    assert [(h, n, score) for h, n, _sup, score in rows] == [
        (1, "b", 3.0), (1, "c", 2.0), (1, "a", 1.0)]


def test_same_group_ignores_other_methods(local_pg):
    """M2.8: ('a', 'other', 'g1') and ('y', 'other', 'g1') share the group id but not the method."""
    assert _run(local_pg, same_group_sql("lgk_providers"), "b") == [
        ("C_9", "g1"), ("a", "g1"), ("c", "g1")]


def test_same_group_never_returns_the_seed(local_pg):
    assert _run(local_pg, same_group_sql("lgk_providers"), "a") == [
        ("C_9", "g1"), ("b", "g1"), ("c", "g1")]
    assert _run(local_pg, same_group_sql("lgk_providers"), "p") == [("q", "g2")]


def test_same_group_of_an_isolated_node_is_empty(local_pg):
    assert _run(local_pg, same_group_sql("lgk_providers"), "z") == []
    assert _run(local_pg, same_group_sql("lgk_providers"), "y") == []  # y has no components row


def test_same_group_cut_to_top_n_in_byte_order(local_pg):
    got = _run(local_pg, same_group_sql("lgk_providers"), "m_29")
    assert got == [(f"m_{i:02d}", "g4") for i in range(25)]
    assert _run(local_pg, same_group_sql("lgk_providers", 3), "m_00") == [
        ("m_01", "g4"), ("m_02", "g4"), ("m_03", "g4")]


def test_similar_jaccard_breaks_ties_by_dst_id_in_byte_order(local_pg):
    assert _run(local_pg, similar_jaccard_sql("lgk_providers"), "v") == [
        ("U9", 1.0), ("u1", 1.0), ("u2", 1.0), ("u3", 1.0), ("k", 0.5), ("w", 0.25)]
    assert _run(local_pg, similar_jaccard_sql("lgk_providers", 2), "v") == [("U9", 1.0), ("u1", 1.0)]


def test_similar_jaccard_reads_only_the_seeds_rows(local_pg):
    assert _run(local_pg, similar_jaccard_sql("lgk_providers"), "u1") == [("v", 1.0)]
    assert _run(local_pg, similar_jaccard_sql("lgk_providers"), "z") == []
