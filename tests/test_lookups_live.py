"""Live acceptance for the four lookups and the served enrichment views.

One provider graph (a DAG, ~60 nodes) is built, enriched with components, similarity,
closure and degree, provisioned with `serve_closure: true` and served as v1 through the
shared `provisioned_graph` (its teardown, manifest-driven, also covers v2). Hostile parts:

- # hostile: diamond with unequal arms r->a->d and r->b->x->d (d at depth 2 and 3 from r;
  only depth 2 may survive), with a tail d->e (e at depth 3 and 4; only 3 may survive).
- # hostile: duplicate source rows: the pair r->a appears twice (distinct affiliation_id).
- # hostile: self-loop s->s (dropped; s must not be its own ancestor).
- # hostile: a second component p->q, and an isolated node z (no edges).
- # hostile: Jaccard ties: u1, u2, u3 each share exactly n1, n2 with v (the two later references reuse these rows).
- # hostile: fan-out f -> f_00..f_39 (40 children), so top_n=25 cuts.
- # hostile: score order: f_39 -> g1, g2 and f_38 -> g3, so f's children differ in degree.

Two more references also read only the source and node_id_map:
- same_group: the undirected distinct pairs and every node_id, from Spark; a Python union-find
  groups them (not the kit's label propagation); group_id = the smallest node_id (byte order).
- similar_jaccard: Spark SQL over undirected neighbour sets built with collect_set; score =
  size(array_intersect) / size(array_union) (not the kit's k / (dx + dy - k)); rank (score DESC,
  y ASC), keep SIMILAR_TOP_K, serve top_n.

Then, inside the same fixture, a cyclic v2: # hostile: back-edge d->r. Enrich v2 cannot
compute closure (greyed), so v2 has no closure table, synced table or view. The fixture
switches back to v1 before the tests run.

The Spark reference reads ONLY the source fixture_affiliations and the Delta node_id_map
(natural key -> node_id): breadth-first rounds of joins, each round keeping only the pairs
not reached before (LEFT ANTI JOIN), then ROW_NUMBER per seed on (depth, node_id). It never
reads the kit's closure or edges tables. It is computed before the back-edge is appended.
"""

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import psycopg
import pytest

from lgk.manifest import ObjectType, load_run_manifest
from lgk.uc_sql import execute_sql
from lgk.versions import version_schema
from tests.test_build_provision_walks import WAREHOUSE_ID, _pg, provisioned_graph, served_graph_size
from tests.privacy_support import clear_gate_live

MAX_DEPTHS = (1, 2, 3, 100)
TOP_N = 25
KINDS = ("ancestors", "descendants")
SIMILAR_TOP_K = 20  # enrich's top_k; the test also checks the report says so
SIMILAR_TOP_NS = (1, 10, 20)
BACK_EDGE_ID = "aff_back_d_r"
CYCLE_REASON = "closure was not computed: the graph has a cycle; closure is only for trees and DAGs"
N_FAN = 40
FAN = [f"f_{i:02d}" for i in range(N_FAN)]

Ref = Dict[Tuple[str, int], Dict[str, List[Tuple[str, int]]]]  # (kind, max_depth) -> seed -> rows


def _edge_pairs() -> List[Tuple[str, str]]:
    pairs = [("r", "a"), ("a", "d"), ("r", "b"), ("b", "x"), ("x", "d")]  # hostile: unequal diamond
    pairs += [("d", "e")]  # tail below the diamond: r->e at depth 3 (and 4 by the long arm)
    pairs += [("r", "a")]  # hostile: duplicate pair r->a (its own affiliation_id)
    pairs += [("s", "s")]  # hostile: self-loop, dropped
    pairs += [("p", "q")]  # hostile: second component (z is isolated: no edge at all)
    pairs += [(u, n) for u in ("u1", "u2", "u3", "v") for n in ("n1", "n2")]  # hostile: Jaccard ties
    pairs += [("f", c) for c in FAN]  # hostile: fan-out of 40, cut by top_n=25
    pairs += [("f_39", "g1"), ("f_39", "g2"), ("f_38", "g3")]  # hostile: unequal degrees under f
    return pairs


def _node_keys() -> List[str]:
    return ["r", "a", "b", "x", "d", "e", "s", "p", "q", "z", "u1", "u2", "u3", "v", "n1", "n2", "f",
            *FAN, "g1", "g2", "g3"]


def _create_lookups_fixture(w: Any, catalog: str, fixture_schema: str) -> None:
    s = f"{catalog}.{fixture_schema}"
    execute_sql(w, WAREHOUSE_ID, f"CREATE SCHEMA IF NOT EXISTS {s}")
    execute_sql(w, WAREHOUSE_ID, f"CREATE OR REPLACE TABLE {s}.fixture_providers "
                "(provider_id STRING, name STRING, specialty STRING) USING DELTA")
    execute_sql(w, WAREHOUSE_ID, f"INSERT INTO {s}.fixture_providers VALUES "
                + ", ".join(f"('{k}', 'N {k}', 'x')" for k in _node_keys()))
    execute_sql(w, WAREHOUSE_ID, f"CREATE OR REPLACE TABLE {s}.fixture_affiliations "
                "(affiliation_id STRING, provider_id STRING, hospital_id STRING, start_date STRING) USING DELTA")
    execute_sql(w, WAREHOUSE_ID, f"INSERT INTO {s}.fixture_affiliations VALUES "
                + ", ".join(f"('a{n}', '{a}', '{b}', '2024-01-01')" for n, (a, b) in enumerate(_edge_pairs())))


# ---------------------------------------------------------------------------- Spark reference


def _ref_edges_sql(g: Dict[str, Any]) -> str:
    """Distinct directed node-id pairs from the SOURCE table (self-loops and the v2 back-edge out)."""
    cat, kit, fx = g["lakebase_catalog"], g["kit_schema"], g["fixture_schema"]
    return (f"SELECT DISTINCT ms.node_id AS src, md.node_id AS dst "
            f"FROM {cat}.{fx}.fixture_affiliations a "
            f"JOIN {cat}.{kit}.node_id_map ms ON ms.natural_key = a.provider_id "
            f"JOIN {cat}.{kit}.node_id_map md ON md.natural_key = a.hospital_id "
            f"WHERE ms.node_id <> md.node_id AND a.affiliation_id <> '{BACK_EDGE_ID}'")


def _rounds_ctes(g: Dict[str, Any], rounds: int) -> str:
    """WITH e, p1..p<rounds>: p_i = the pairs first reached in round i (each pair at its smallest depth)."""
    ctes = [f"e AS ({_ref_edges_sql(g)})", "p1 AS (SELECT src AS anc, dst AS dsc FROM e)"]
    for i in range(2, rounds + 1):
        seen = " UNION ALL ".join(f"SELECT anc, dsc FROM p{j}" for j in range(1, i))
        ctes.append(
            f"p{i} AS (SELECT n.anc, n.dsc FROM (SELECT DISTINCT p.anc, e.dst AS dsc "
            f"FROM p{i - 1} p JOIN e ON e.src = p.dsc) n "
            f"LEFT ANTI JOIN ({seen}) s ON s.anc = n.anc AND s.dsc = n.dsc)")
    return "WITH " + ", ".join(ctes)


def _productive_rounds(w: Any, g: Dict[str, Any]) -> int:
    """The number of rounds that add pairs (stop at the first round that adds nothing)."""
    last = 0
    for i in range(1, max(MAX_DEPTHS) + 1):
        n = int(execute_sql(w, WAREHOUSE_ID, f"{_rounds_ctes(g, i)} SELECT COUNT(*) FROM p{i}")[0][0])
        if n == 0:
            break
        last = i
    return last


def spark_lookup_reference(w: Any, g: Dict[str, Any]) -> Ref:
    """{(kind, max_depth): {seed: [(node_id, depth), ...]}} cut to TOP_N by (depth, node_id)."""
    last = _productive_rounds(w, g)
    assert last == 3, f"r->a->d->e is the longest shortest path (3 rounds), Spark found {last}"
    out: Ref = {}
    for max_depth in MAX_DEPTHS:
        r = min(max_depth, last)
        union = " UNION ALL ".join(f"SELECT anc, dsc, {i} AS depth FROM p{i}" for i in range(1, r + 1))
        sql = (f"{_rounds_ctes(g, r)}, u AS ({union}) "
               f"SELECT kind, seed, node, depth, rn FROM ("
               f"SELECT 'ancestors' AS kind, dsc AS seed, anc AS node, depth, ROW_NUMBER() OVER "
               f"(PARTITION BY dsc ORDER BY depth, anc) AS rn FROM u UNION ALL "
               f"SELECT 'descendants' AS kind, anc AS seed, dsc AS node, depth, ROW_NUMBER() OVER "
               f"(PARTITION BY anc ORDER BY depth, dsc) AS rn FROM u) WHERE rn <= {TOP_N}")
        for kind in KINDS:
            out[(kind, max_depth)] = {}
        for kind, seed, node, depth, rn in sorted(execute_sql(w, WAREHOUSE_ID, sql),
                                                  key=lambda row: (row[0], row[1], int(row[4]))):
            out[(kind, max_depth)].setdefault(str(seed), []).append((str(node), int(depth)))
    return out


def _out_neighbours(w: Any, g: Dict[str, Any], key: str) -> List[str]:
    rows = execute_sql(w, WAREHOUSE_ID,
                       f"SELECT DISTINCT e.dst FROM ({_ref_edges_sql(g)}) e WHERE e.src = '{g['ids'][key]}'")
    return [str(r[0]) for r in rows]


def _undirected_sql(g: Dict[str, Any]) -> str:
    """Both orientations of every distinct source pair (self-loops and the v2 back-edge out)."""
    e = _ref_edges_sql(g)
    return f"SELECT src AS x, dst AS y FROM ({e}) UNION SELECT dst AS x, src AS y FROM ({e})"


def _find(parent: Dict[str, str], n: str) -> str:
    while parent[n] != n:
        parent[n] = parent[parent[n]]
        n = parent[n]
    return n


def spark_same_group_reference(w: Any, g: Dict[str, Any]) -> Dict[str, List[Tuple[str, str]]]:
    """{seed: [(node_id, group_id), ...]}: the seed's other group members, byte order, cut to TOP_N."""
    nodes = [str(r[0]) for r in execute_sql(
        w, WAREHOUSE_ID, f"SELECT node_id FROM {g['lakebase_catalog']}.{g['kit_schema']}.node_id_map")]
    parent = {n: n for n in nodes}
    for x, y in execute_sql(w, WAREHOUSE_ID, _undirected_sql(g)):
        rx, ry = _find(parent, str(x)), _find(parent, str(y))
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)
    members: Dict[str, List[str]] = {}
    for n in nodes:
        members.setdefault(_find(parent, n), []).append(n)
    out: Dict[str, List[Tuple[str, str]]] = {}
    for group in members.values():
        gid = min(group, key=str.encode)
        for seed in group:
            others = sorted((n for n in group if n != seed), key=str.encode)[:TOP_N]
            out[seed] = [(n, gid) for n in others]
    return out


def spark_similar_reference(w: Any, g: Dict[str, Any]) -> Dict[str, List[Tuple[str, float]]]:
    """{seed: [(node_id, score), ...]} ranked (score DESC, node_id ASC), cut to SIMILAR_TOP_K."""
    sql = (f"WITH u AS ({_undirected_sql(g)}), "
           f"nb AS (SELECT x, collect_set(y) AS ns FROM u GROUP BY x), "
           f"j AS (SELECT a.x AS seed, b.x AS node, "
           f"CAST(size(array_intersect(a.ns, b.ns)) AS DOUBLE) / size(array_union(a.ns, b.ns)) AS score "
           f"FROM nb a JOIN nb b ON a.x <> b.x WHERE size(array_intersect(a.ns, b.ns)) > 0) "
           f"SELECT seed, node, score, rn FROM (SELECT seed, node, score, ROW_NUMBER() OVER "
           f"(PARTITION BY seed ORDER BY score DESC, node ASC) AS rn FROM j) WHERE rn <= {SIMILAR_TOP_K}")
    out: Dict[str, List[Tuple[str, float]]] = {}
    for seed, node, score, rn in sorted(execute_sql(w, WAREHOUSE_ID, sql), key=lambda r: (r[0], int(r[3]))):
        out.setdefault(str(seed), []).append((str(node), float(score)))
    return out


# ---------------------------------------------------------------------------- the module graph


def _read_v1_facts(w: Any, config: Dict[str, Any], run_folder: Path) -> None:
    """Everything the tests compare v1 against, read from Delta BEFORE v2 changes the source."""
    cat, kit = config["lakebase_catalog"], config["kit_schema"]
    config["ids"] = {k: str(v) for k, v in execute_sql(w, WAREHOUSE_ID,
                     f"SELECT natural_key, node_id FROM {cat}.{kit}.node_id_map")}
    assert sorted(config["ids"]) == sorted(_node_keys())
    config["v1_scores"] = {str(n): float(sc) for n, sc in execute_sql(w, WAREHOUSE_ID,
                           f"SELECT node_id, score FROM {cat}.{kit}.node_score WHERE method = 'degree'")}
    config["ref"] = spark_lookup_reference(w, config)
    config["group_ref"] = spark_same_group_reference(w, config)
    config["similar_ref"] = spark_similar_reference(w, config)
    config["f_children"] = _out_neighbours(w, config, "f")
    # v1's enrich report: the lookups of the served v1 are asked against it (the run folder's
    # own report becomes v2's, where closure is greyed).
    v1_folder = Path(tempfile.mkdtemp(prefix="lookups_v1_", dir=run_folder))
    shutil.copy(Path(config["run_folder"]) / "enrich_report.json", v1_folder / "enrich_report.json")
    config["v1_run_folder"] = str(v1_folder)


def _build_cyclic_v2(w: Any, config: Dict[str, Any]) -> None:
    """v2 = v1 plus the back-edge d->r (a cycle), built, enriched (no closure), provisioned."""
    from lgk.privacy.gate import PrivacyGateError
    from lgk.steps.build import build
    from lgk.steps.enrich import enrich
    from lgk.steps.provision import provision

    fx = f"{config['lakebase_catalog']}.{config['fixture_schema']}"
    # hostile: back-edge d->r closes the cycle r->a->d->r
    execute_sql(w, WAREHOUSE_ID,
                f"INSERT INTO {fx}.fixture_affiliations VALUES ('{BACK_EDGE_ID}', 'd', 'r', '2024-01-01')")
    config["version"] = 2
    # closure is greyed on a cyclic graph (select_methods would refuse it)
    (Path(config["run_folder"]) / "enrich.yaml").write_text("methods: [components, similarity, degree]\n")
    try:
        build(config)
    except PrivacyGateError:
        clear_gate_live(config, w, drops=[("fixture_affiliations", "start_date")])
        build(config)
    enrich(config)
    provision(config)


@pytest.fixture(scope="module")
def lookups(workspace_client, lakebase_project, lakebase_catalog, run_folder):
    from lgk.steps.switch import switch

    with provisioned_graph(
        workspace_client, lakebase_project, lakebase_catalog, run_folder,
        "lookups", _create_lookups_fixture,
        enrich_methods=["components", "similarity", "closure", "degree"],
        extra_config={"serve_closure": True},
    ) as config:
        _read_v1_facts(workspace_client, config, run_folder)
        _build_cyclic_v2(workspace_client, config)
        switch(config, 1)  # v1 is served when the tests start
        config["graph_size"] = served_graph_size(config)  # the served v1: what every metric line is measured at
        yield config


# ---------------------------------------------------------------------------- helpers


def _as_reader(conn: Any) -> None:
    conn.execute("GRANT reader TO CURRENT_USER")
    conn.execute("SET ROLE reader")


def _lookup(conn: Any, sql: str, seed_id: str) -> List[Tuple[str, int]]:
    return [(str(n), int(d)) for n, d in conn.execute(sql, {"start_node_id": seed_id}).fetchall()]


def _served_views(conn: Any, served: str) -> List[str]:
    return [r[0] for r in conn.execute(
        "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = %s AND c.relkind = 'v' ORDER BY c.relname", (served,)).fetchall()]


def _assert_lookup_equals_reference(g: Dict[str, Any], kind: str, max_depth: int) -> Dict[str, List[float]]:
    """Every seed's lookup equals the reference, as operator then reader; returns {role: query times (s)}."""
    from lgk.steps.walks import get_lookup_query

    ref = g["ref"][(kind, max_depth)]
    sql = get_lookup_query(kind, g["served_schema"], g["v1_run_folder"], max_depth=max_depth, top_n=TOP_N)
    samples: Dict[str, List[float]] = {"operator": [], "reader": []}
    conn = _pg(g)
    try:
        for role in ("operator", "reader"):
            if role == "reader":
                _as_reader(conn)
            for key, node_id in g["ids"].items():
                t0 = time.perf_counter()
                got = _lookup(conn, sql, node_id)
                samples[role].append(time.perf_counter() - t0)
                assert got == ref.get(node_id, []), f"{kind} as {role}, max_depth {max_depth}, seed {key}"
    finally:
        conn.rollback()
        conn.close()
    return samples


def _record_lookup(record_metric: Any, g: Dict[str, Any], lookup: str, samples: Dict[str, List[float]],
                   **context: Any) -> None:
    """p50/p95 per role of one lookup (opt-in numbers recorder); context may carry its own `kind`."""
    for role, times in samples.items():
        record_metric.latencies(f"lookup.{lookup}", times, branch=g["branch_id"], role=role, **g["graph_size"],
                                **context)


# ---------------------------------------------------------------------------- tests


@pytest.mark.seam1
@pytest.mark.provision
@pytest.mark.parametrize("max_depth", MAX_DEPTHS, ids=lambda d: f"max_depth={d}")
def test_ancestors_equals_spark_reference(lookups, max_depth, record_metric):
    ids, ref = lookups["ids"], lookups["ref"][("ancestors", max_depth)]
    # Non-vacuous: d's ancestors at depth 1 are a and x; r only at its shortest depth, 2.
    d_rows = ref[ids["d"]]
    assert sorted(d_rows[:2]) == sorted([(ids["a"], 1), (ids["x"], 1)])
    if max_depth >= 2:
        assert (ids["r"], 2) in d_rows and (ids["r"], 3) not in d_rows
    assert ids["s"] not in ref, "self-loop: s must not be its own ancestor"
    samples = _assert_lookup_equals_reference(lookups, "ancestors", max_depth)
    _record_lookup(record_metric, lookups, "ancestors", samples, depth=max_depth)


@pytest.mark.seam1
@pytest.mark.provision
@pytest.mark.parametrize("max_depth", MAX_DEPTHS, ids=lambda d: f"max_depth={d}")
def test_descendants_equals_spark_reference(lookups, max_depth, record_metric):
    ids, ref = lookups["ids"], lookups["ref"][("descendants", max_depth)]
    # Non-vacuous: f has 40 children (43 descendants) and is cut to 25; d appears once from r.
    assert len(ref[ids["f"]]) == TOP_N
    r_rows = ref[ids["r"]]
    if max_depth >= 2:
        assert [n for n, _ in r_rows].count(ids["d"]) == 1 and (ids["d"], 2) in r_rows
    assert ((ids["e"], 3) in r_rows) == (max_depth >= 3) and (ids["e"], 4) not in r_rows
    assert ids["z"] not in ref and ids["s"] not in ref
    samples = _assert_lookup_equals_reference(lookups, "descendants", max_depth)
    _record_lookup(record_metric, lookups, "descendants", samples, depth=max_depth)


@pytest.mark.seam1
@pytest.mark.provision
def test_reader_can_call_every_lookup(lookups):
    """As reader, all four lookups return their rows (a denied closure_view, node_group_view or
    similar_to_view would mean the switch's grant block skipped the optional views)."""
    from lgk.steps.walks import get_lookup_query

    ids, g = lookups["ids"], lookups
    cases = {  # kind -> (seed, builder kwargs, the reference rows)
        "ancestors": ("d", {"max_depth": 3}, g["ref"][("ancestors", 3)][ids["d"]]),
        "descendants": ("r", {"max_depth": 3}, g["ref"][("descendants", 3)][ids["r"]]),
        "same_group": ("r", {}, g["group_ref"][ids["r"]]),
        "similar_jaccard": ("v", {}, g["similar_ref"][ids["v"]][:10]),
    }
    conn = _pg(g)
    try:
        _as_reader(conn)
        for kind, (key, kw, want) in cases.items():
            sql = get_lookup_query(kind, g["served_schema"], g["v1_run_folder"], **kw)
            got = [tuple(r) for r in conn.execute(sql, {"start_node_id": ids[key]}).fetchall()]
            assert got and got == want, f"reader {kind} of {key}: {got} != {want}"
    finally:
        conn.rollback()
        conn.close()


def _assert_every_seed(g: Dict[str, Any], sql: str, want_of: Any, what: str,
                       no_index: bool = False) -> Dict[str, List[float]]:
    """As the operator, then as reader: every node as the seed returns exactly want_of(node_id).
    Returns {role: query times (s)} for the opt-in numbers recorder.

    no_index: also forbid index scans, so the rows reach the sort in heap order, not in the
    synced PK's (src_id, dst_id) order; only the SQL's own tie-break then orders the ties.
    """
    samples: Dict[str, List[float]] = {"operator": [], "reader": []}
    conn = _pg(g)
    try:
        if no_index:
            conn.execute("SET enable_indexscan = off")
            conn.execute("SET enable_bitmapscan = off")
            conn.execute("SET enable_indexonlyscan = off")
        for role in ("operator", "reader"):
            if role == "reader":
                _as_reader(conn)
            for key, node_id in g["ids"].items():
                t0 = time.perf_counter()
                rows = conn.execute(sql, {"start_node_id": node_id}).fetchall()
                samples[role].append(time.perf_counter() - t0)
                got = [(str(a), b) for a, b in rows]
                assert got == want_of(node_id), f"{what} as {role}, seed {key}"
    finally:
        conn.rollback()
        conn.close()
    return samples


@pytest.mark.seam1
@pytest.mark.provision
def test_same_group_equals_union_find(lookups, record_metric):
    from lgk.steps.walks import get_lookup_query

    g, ids, ref = lookups, lookups["ids"], lookups["group_ref"]
    # Non-vacuous: f's component (f, 40 children, g1..g3 = 44 nodes) is cut to 25; z and s
    # (self-loop only) are alone; the seed is never in its own list; p's only partner is q.
    assert len(ref[ids["f"]]) == TOP_N and ids["f"] not in [n for n, _ in ref[ids["f"]]]
    assert ref[ids["z"]] == [] and ref[ids["s"]] == []
    assert [n for n, _ in ref[ids["p"]]] == [ids["q"]]
    assert ref[ids["p"]][0][1] == min(ids["p"], ids["q"], key=str.encode)
    sql = get_lookup_query("same_group", g["served_schema"], g["v1_run_folder"])  # all defaults
    samples = _assert_every_seed(g, sql, lambda n: ref[n], "same_group")
    _record_lookup(record_metric, g, "same_group", samples)


@pytest.mark.seam1
@pytest.mark.provision
@pytest.mark.parametrize("top_n", SIMILAR_TOP_NS, ids=lambda n: f"top_n={n}")
def test_similar_jaccard_equals_spark_reference(lookups, top_n, record_metric):
    from lgk.steps.walks import get_lookup_query

    g, ids, ref = lookups, lookups["ids"], lookups["similar_ref"]
    report = json.loads((Path(g["v1_run_folder"]) / "enrich_report.json").read_text())
    assert report["similarity"]["top_k"] == SIMILAR_TOP_K
    # Non-vacuous: v ties with u1, u2, u3 at 1.0 (byte order decides); f_00 ties with 37 other
    # leaves at 1.0, cut to top_k 20; z (isolated) has no similar rows.
    ties = sorted([ids["u1"], ids["u2"], ids["u3"]], key=str.encode)
    assert ref[ids["v"]][:3] == [(n, 1.0) for n in ties]
    assert len(ref[ids["f_00"]]) == SIMILAR_TOP_K and ids["z"] not in ref
    sql = get_lookup_query("similar_jaccard", g["served_schema"], g["v1_run_folder"], top_n=top_n)
    want = lambda n: ref.get(n, [])[:top_n]  # noqa: E731
    samples = _assert_every_seed(g, sql, want, f"similar_jaccard top_n={top_n}")
    _record_lookup(record_metric, g, "similar_jaccard", samples, kind=f"top{top_n}")
    # hostile: the PK index (src_id, dst_id) hands rows over already in dst_id order; without it
    # the ties must still come out in byte order
    samples = _assert_every_seed(g, sql, want, f"similar_jaccard top_n={top_n}, no index", no_index=True)
    _record_lookup(record_metric, g, "similar_jaccard", samples, kind=f"top{top_n}-noindex")


def _lookup_latency_queries(g: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, Any], List[int]]]:
    """The lookups of the tests above (same builders, same seeds: every node), as the operator, and the
    row count each seed must return (from the Spark / union-find references)."""
    from lgk.steps.walks import get_lookup_query

    seeds = list(g["ids"].values())
    served, folder = g["served_schema"], g["v1_run_folder"]
    queries, want = [], {}
    for kind in KINDS:
        for d in MAX_DEPTHS:
            queries.append({"metric": f"lookup.{kind}", "depth": d, "role": "operator", "seeds": seeds,
                            "sql": get_lookup_query(kind, served, folder, max_depth=d, top_n=TOP_N)})
            want[(f"lookup.{kind}", d)] = [len(g["ref"][(kind, d)].get(n, [])) for n in seeds]
    queries.append({"metric": "lookup.same_group", "role": "operator", "seeds": seeds,
                    "sql": get_lookup_query("same_group", served, folder)})
    want[("lookup.same_group", None)] = [len(g["group_ref"][n]) for n in seeds]
    for top_n in SIMILAR_TOP_NS:
        queries.append({"metric": "lookup.similar_jaccard", "kind": f"top{top_n}", "role": "operator", "seeds": seeds,
                        "sql": get_lookup_query("similar_jaccard", served, folder, top_n=top_n)})
        want[("lookup.similar_jaccard", f"top{top_n}")] = [len(g["similar_ref"].get(n, [])[:top_n]) for n in seeds]
    return queries, want


@pytest.mark.seam1
@pytest.mark.provision
def test_lookups_in_region_latency(lookups, workspace_client, record_metric, tmp_path):
    """client=workspace: the same lookups over the same seeds, timed inside the workspace by a serverless
    notebook job (the shipped notebook runner, the wheel of this tree), on this suite's t- branch."""
    import secrets

    from lgk.workspace_latency import record_workspace_samples, run_in_region
    from tests.test_parity_live import _build_wheel

    g = lookups
    queries, want = _lookup_latency_queries(g)
    workdir = f"/Workspace/Users/{workspace_client.current_user.me().user_name}/lgk_v1_{secrets.token_hex(3)}"
    samples = run_in_region(workspace_client, g, queries, _build_wheel(tmp_path / "dist"), tmp_path / "run", workdir)
    assert len(samples) == len(queries)
    for s in samples:
        key = (s.metric, s.context.get("depth", s.context.get("kind")))
        # Non-vacuous: the workspace ran THESE queries over every seed (row counts = the references)
        assert s.rows == want[key], f"in-region {key}: {s.rows} != {want[key]}"
        assert len(s.seconds) == len(g["ids"])
    assert sum(sum(s.rows) for s in samples) > 0
    record_workspace_samples(record_metric, samples, branch=g["branch_id"], **g["graph_size"])


@pytest.mark.seam1
@pytest.mark.provision
def test_score_sorted_walk_runs_as_reader(lookups):
    """The degree-sorted web walk reads <served>.node_score_view as reader; order = Delta degree
    desc, then node_id (byte order)."""
    from lgk.steps.walks import get_walk_query

    g, ids, scores = lookups, lookups["ids"], lookups["v1_scores"]
    want = sorted(g["f_children"], key=lambda n: (-scores[n], n.encode()))[:TOP_N]
    # Non-vacuous: the two higher-degree children lead, and the cut drops 15 children.
    assert want[:2] == [ids["f_39"], ids["f_38"]] and len(g["f_children"]) == N_FAN
    sql = get_walk_query("", depth=1, served_schema=g["served_schema"], run_folder=g["run_folder"],
                         sort_method="degree", top_n=TOP_N)
    conn = _pg(g)
    try:
        _as_reader(conn)
        rows = conn.execute(sql, {"start_node_id": ids["f"]}).fetchall()
    finally:
        conn.rollback()
        conn.close()
    assert [str(r[1]) for r in rows] == want
    assert [float(r[3]) for r in rows] == [scores[n] for n in want]


@pytest.mark.seam1
@pytest.mark.provision
def test_greyed_closure_leaves_no_table_no_synced_table_no_view(lookups, workspace_client):
    from lgk.steps.switch import switch
    from lgk.steps.walks import get_lookup_query
    from lgk.walks.lookups import EnrichmentNotServedError

    g = lookups
    cat, kit, served = g["lakebase_catalog"], g["kit_schema"], g["served_schema"]
    uc_tables = {r[1] for r in execute_sql(workspace_client, WAREHOUSE_ID, f"SHOW TABLES IN {cat}.{kit}")}
    assert "node_score" in uc_tables and "closure" not in uc_tables, sorted(uc_tables)
    synced = {e.object_id for e in load_run_manifest(g).entries if e.object_type == ObjectType.SYNCED_TABLE}
    v1, v2 = f"{cat}.{version_schema(g, 1)}", f"{cat}.{version_schema(g, 2)}"
    assert f"{v1}.closure" in synced and f"{v2}.nodes" in synced and f"{v2}.node_score" in synced, sorted(synced)
    assert f"{v2}.closure" not in synced
    report = json.loads((Path(g["run_folder"]) / "enrich_report.json").read_text())
    assert "closure" not in report and "degree" in report
    try:
        switch(g, 2)
        with _pg(g) as conn:
            views = _served_views(conn, served)
        assert "closure_view" not in views and "node_score_view" in views, views
        with pytest.raises(EnrichmentNotServedError) as err:
            get_lookup_query("ancestors", served, g["run_folder"], max_depth=3)
        assert str(err.value) == CYCLE_REASON
        conn = _pg(g)
        conn.autocommit = True
        try:
            with pytest.raises(psycopg.errors.UndefinedTable) as missing:
                conn.execute(f"SELECT * FROM {served}.closure_view")
            assert "does not exist" in str(missing.value)
        finally:
            conn.close()
    finally:
        switch(g, 1)  # leave v1 served
    with _pg(g) as conn:
        assert "closure_view" in _served_views(conn, served)
        sql = get_lookup_query("ancestors", served, g["v1_run_folder"], max_depth=3)
        got = _lookup(conn, sql, g["ids"]["d"])
    assert got == g["ref"][("ancestors", 3)][g["ids"]["d"]]


@pytest.mark.seam1
@pytest.mark.provision
def test_retire_drops_a_version_with_enrichment_tables(lookups, workspace_client):
    """LAST: retire v2 (it holds node_score, node_group, similar_to); v1 stays served."""
    from databricks.sdk.errors import NotFound

    from lgk.steps.retire import retire

    g = lookups
    v2 = version_schema(g, 2)
    full_v2 = f"{g['lakebase_catalog']}.{v2}"
    with _pg(g) as conn:
        held = {r[0] for r in conn.execute(
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = %s AND c.relkind IN ('r', 'p')", (v2,)).fetchall()}
    assert {"node_score", "node_group", "similar_to"} <= held, held  # non-vacuous
    retire(g, 2, g["graph_name"])
    with _pg(g) as conn:
        assert conn.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (v2,)).fetchone() is None
        n_nodes = conn.execute(f"SELECT COUNT(*) FROM {g['served_schema']}.nodes_view").fetchone()[0]
        n_closure = conn.execute(f"SELECT COUNT(*) FROM {g['served_schema']}.closure_view").fetchone()[0]
    with pytest.raises(NotFound):
        workspace_client.schemas.get(full_name=full_v2)
    assert n_nodes == len(g["ids"]) and n_closure > 0
