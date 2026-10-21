"""Live acceptance for the walk engine: web, tree, similar vs. a Spark reference.

One hostile graph (all node type `provider`) is built, enriched (degree) and provisioned
through the shared `provisioned_graph` (same teardown as the spine):

- web part w_000..w_199, 4 out-edges each (many equal degrees: tie fodder).
- # hostile: hub — `hub` has 10,000 out-links to leaf_00000..leaf_09999 (all degree 1,
  so every leaf ties), and w_000..w_019 each link to the hub.
- tree part t_000..t_126 (binary tree) with a diamond t_1->t_6 and
  # hostile: back-edges t_7->t_0 and t_9->t_4 (cycles inside the tree).
- # hostile: cycle c_a->c_b->c_c->c_a.
- # hostile: a chain of 25 diamonds d_00 -> {dl_00, dr_00} -> d_01 -> ... -> d_25 (depth 50,
  2^25 simple paths): a tree walk that lists paths instead of nodes never finishes.
- walk direction: # hostile: mutual pair m_a <-> m_b (double-count bait for `both`),
  # hostile: one-way discriminators m_a -> m_c (outbound only) and m_d -> m_a (inbound only),
  # hostile: fan-in sink — every leaf links to `sink` (10,000 in-links, the hub's inbound twin).

The Spark reference reads only Delta: the SOURCE fixture tables, the Delta node_id_map
(natural key -> node_id) and the Delta node_score. It applies the same order and cut with
ROW_NUMBER windows (web: per hop; tree: per BFS level, then on depth), never the Postgres SQL.
"""

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from lgk.uc_sql import execute_sql
from lgk.versions import version_schema
from lgk.walks.sql import (NodePredicate, per_depth_sql, similar_sql, tree_sql, tree_walk_cte, vector_index_sql,
                           web_set_sql, web_sql)
from tests.test_build_provision_walks import WAREHOUSE_ID, _pg, provisioned_graph
from tests.test_walks import (DIRECTIONS, SET_EDGES, SET_NODES, _web_for, assert_each_node_once,
                              direction_skip_reason)

N_WEB, N_LEAVES, N_TREE = 200, 10_000, 127
WEB_SEEDS = [f"w_{i:03d}" for i in range(20)]
TREE_SEEDS = [f"t_{i:03d}" for i in range(20)]
# hostile: hub (10,000 links) and the 3-cycle c_a->c_b->c_c->c_a, on top of the 20 web seeds
UNIQUE_SEEDS = WEB_SEEDS + ["hub", "c_a"]
# hostile: tree seeds whose level 1 or 2 holds the 10,000-child hub's children
HUB_TREE_SEEDS = ["hub"] + [f"w_{i:03d}" for i in range(19)]
WEB_TOP_N, TREE_TOP_N, SIMILAR_TOP_N = 3, 5, 5
DEPTHS = (1, 2, 3, 4)
HUB_TIME_BOUND_S = 2.0
N_DIAMONDS = 25
DIAMOND_TIME_BOUND_S = 5.0
# walk direction: the mutual-pair seed joins every direction's reference comparison
DIRECTION_WEB_SEEDS = WEB_SEEDS + ["m_a"]
DIRECTION_TREE_SEEDS = TREE_SEEDS + ["m_a"]
MUTUAL_EDGES = [("m_a", "m_b"), ("m_b", "m_a"), ("m_a", "m_c"), ("m_d", "m_a")]


def _diamond_edges() -> List[Tuple[str, str]]:
    edges = []
    for i in range(N_DIAMONDS):
        d, nxt = f"d_{i:02d}", f"d_{i + 1:02d}"
        edges += [(d, f"dl_{i:02d}"), (d, f"dr_{i:02d}"), (f"dl_{i:02d}", nxt), (f"dr_{i:02d}", nxt)]
    return edges


def _web_targets(i: int) -> List[int]:
    return [(i + 1) % N_WEB, (i * 7 + 3) % N_WEB, (i * 13 + 11) % N_WEB, (i * 31 + 17) % N_WEB]


def _tree_edges() -> List[Tuple[str, str]]:
    edges = [(f"t_{i:03d}", f"t_{c:03d}") for i in range(63) for c in (2 * i + 1, 2 * i + 2)]
    edges += [("t_001", "t_006")]  # diamond: t_0->t_2->t_6 and t_0->t_1->t_6
    edges += [("t_007", "t_000"), ("t_009", "t_004")]  # hostile: cycles inside the tree
    return edges


def _create_walk_fixture(w: Any, catalog: str, fixture_schema: str) -> None:
    s = f"{catalog}.{fixture_schema}"
    execute_sql(w, WAREHOUSE_ID, f"CREATE SCHEMA IF NOT EXISTS {s}")
    execute_sql(w, WAREHOUSE_ID, f"CREATE OR REPLACE TABLE {s}.fixture_providers "
                "(provider_id STRING, name STRING, specialty STRING) USING DELTA")
    small = ([f"w_{i:03d}" for i in range(N_WEB)] + [f"t_{i:03d}" for i in range(N_TREE)]
             + ["hub", "c_a", "c_b", "c_c", "m_a", "m_b", "m_c", "m_d", "sink"]
             + sorted({n for e in _diamond_edges() for n in e}))
    execute_sql(w, WAREHOUSE_ID, f"INSERT INTO {s}.fixture_providers VALUES "
                + ", ".join(f"('{k}', 'N {k}', 'x')" for k in small))
    execute_sql(w, WAREHOUSE_ID, f"INSERT INTO {s}.fixture_providers SELECT "
                f"concat('leaf_', lpad(CAST(id AS STRING), 5, '0')), 'leaf', 'x' FROM range({N_LEAVES})")

    execute_sql(w, WAREHOUSE_ID, f"CREATE OR REPLACE TABLE {s}.fixture_affiliations "
                "(affiliation_id STRING, provider_id STRING, hospital_id STRING, start_date STRING) USING DELTA")
    pairs = [(f"w_{i:03d}", f"w_{j:03d}") for i in range(N_WEB) for j in _web_targets(i)]
    pairs += [(f"w_{i:03d}", "hub") for i in range(20)]
    pairs += _tree_edges()
    pairs += [("c_a", "c_b"), ("c_b", "c_c"), ("c_c", "c_a")]  # hostile: cycle
    pairs += _diamond_edges()  # hostile: diamond chain
    pairs += MUTUAL_EDGES  # hostile: mutual pair; hostile: one-way discriminators (m_c out, m_d in)
    execute_sql(w, WAREHOUSE_ID, f"INSERT INTO {s}.fixture_affiliations VALUES "
                + ", ".join(f"('a{n}', '{a}', '{b}', '2024-01-01')" for n, (a, b) in enumerate(pairs)))
    # hostile: the hub's 10,000 links
    execute_sql(w, WAREHOUSE_ID, f"INSERT INTO {s}.fixture_affiliations SELECT concat('h', id), 'hub', "
                f"concat('leaf_', lpad(CAST(id AS STRING), 5, '0')), '2024-01-01' FROM range({N_LEAVES})")
    # hostile: fan-in sink — 10,000 in-links, one from each leaf
    execute_sql(w, WAREHOUSE_ID, f"INSERT INTO {s}.fixture_affiliations SELECT concat('sk', id), "
                f"concat('leaf_', lpad(CAST(id AS STRING), 5, '0')), 'sink', '2024-01-01' FROM range({N_LEAVES})")
    # Embeddings for the similar walk: 200 distinct integer vectors; scalar multiples
    # ([1,1,1] and [2,2,2], ...) tie exactly on cosine distance. # hostile: distance ties
    execute_sql(w, WAREHOUSE_ID, f"CREATE OR REPLACE TABLE {s}.fixture_embeddings "
                "(provider_id STRING, x INT, y INT, z INT) USING DELTA")
    execute_sql(w, WAREHOUSE_ID, f"INSERT INTO {s}.fixture_embeddings VALUES " + ", ".join(
        f"('w_{i:03d}', {i % 5}, {(i // 5) % 5}, {(i // 25) % 8 + 1})" for i in range(N_WEB)))


@pytest.fixture(scope="module")
def walk_graph(workspace_client, lakebase_project, lakebase_catalog, run_folder):
    with provisioned_graph(
        workspace_client, lakebase_project, lakebase_catalog, run_folder,
        "walks", _create_walk_fixture, enrich_methods=["degree"],
    ) as config:
        config["ids"] = {k: str(v) for k, v in execute_sql(workspace_client, WAREHOUSE_ID,
            f"SELECT natural_key, node_id FROM {lakebase_catalog}.{config['kit_schema']}.node_id_map "
            "WHERE natural_key NOT LIKE 'leaf_%'")}  # leaves: 10,000 rows, read in Spark when needed
        yield config


# ---------------------------------------------------------------------------- Spark reference


def _ref_edges(g: Dict[str, Any], direction: str = "outbound") -> str:
    """(src, dst) pairs to walk src -> dst: the fixture edges as written (outbound), reversed
    (inbound), or both orientations, DISTINCT (one edge type, so this equals Postgres's
    (node_id, via, type) UNION)."""
    cat, kit, fx = g["lakebase_catalog"], g["kit_schema"], g["fixture_schema"]
    out = (f"SELECT DISTINCT ms.node_id AS src, md.node_id AS dst "
           f"FROM {cat}.{fx}.fixture_affiliations a "
           f"JOIN {cat}.{kit}.node_id_map ms ON ms.natural_key = a.provider_id "
           f"JOIN {cat}.{kit}.node_id_map md ON md.natural_key = a.hospital_id")
    inbound = f"SELECT dst AS src, src AS dst FROM ({out}) o"
    if direction == "outbound":
        return out
    if direction == "inbound":
        return inbound
    if direction == "both":
        return f"SELECT DISTINCT src, dst FROM ({out} UNION ALL {inbound}) b"
    raise ValueError(direction)


def _ref_prelude(g: Dict[str, Any], seeds: List[str], direction: str = "outbound") -> str:
    cat, kit = g["lakebase_catalog"], g["kit_schema"]
    keys = ", ".join(f"'{k}'" for k in seeds)
    return (f"e AS ({_ref_edges(g, direction)}), "
            f"sc AS (SELECT node_id, score FROM {cat}.{kit}.node_score WHERE method = 'degree'), "
            f"h0 AS (SELECT natural_key AS seed, node_id FROM {cat}.{kit}.node_id_map "
            f"WHERE natural_key IN ({keys}))")


def spark_web_reference(w: Any, g: Dict[str, Any], seeds: List[str], depth: int, top_n: int,
                        by_score: bool = True, direction: str = "outbound"
                        ) -> Dict[str, List[Tuple[int, str, int, Any]]]:
    """{seed: [(hop, node_id, support, score), ...]} in hop then rank order; the per-hop cut is
    a ROW_NUMBER window. Sort: degree score (by_score) or support, then node_id."""
    ctes = [_ref_prelude(g, seeds, direction)]
    order = "sc.score DESC NULLS LAST" if by_score else "c.support DESC"
    for i in range(1, depth + 1):
        seen = " UNION ALL ".join(f"SELECT seed, node_id FROM h{j}" for j in range(i))
        ctes.append(
            f"h{i} AS (SELECT seed, node_id, support, score, rn FROM (SELECT c.seed, c.node_id, "
            f"c.support, sc.score, ROW_NUMBER() OVER "
            f"(PARTITION BY c.seed ORDER BY {order}, c.node_id ASC) AS rn "
            f"FROM (SELECT f.seed, e.dst AS node_id, COUNT(*) AS support FROM h{i - 1} f "
            f"JOIN e ON e.src = f.node_id GROUP BY f.seed, e.dst) c "
            f"LEFT ANTI JOIN ({seen}) s ON s.seed = c.seed AND s.node_id = c.node_id "
            f"LEFT JOIN sc ON sc.node_id = c.node_id) WHERE rn <= {top_n})")
    union = " UNION ALL ".join(
        f"SELECT seed, {i} AS hop, rn, node_id, support, score FROM h{i}" for i in range(1, depth + 1))
    rows = execute_sql(w, WAREHOUSE_ID, f"WITH {', '.join(ctes)} {union}")
    out: Dict[str, List[Tuple[int, str, int, Any]]] = {k: [] for k in seeds}
    for seed, hop, rn, node, sup, score in sorted(rows, key=lambda r: (r[0], int(r[1]), int(r[2]))):
        out[seed].append((int(hop), str(node), int(sup), float(score) if by_score else None))
    return out


def spark_tree_reference(w: Any, g: Dict[str, Any], seeds: List[str], depth: int,
                         top_n: int, direction: str = "outbound") -> Dict[str, List[Tuple[str, int]]]:
    """{seed: [(node_id, depth), ...]}: BFS levels by repeated joins with the per-level cut
    level i = the distinct out-neighbours of level i-1 not kept at levels 0..i-1,
    ranked per seed by (score desc, node_id) with ROW_NUMBER and cut at top_n. The kept
    levels are then ranked by (depth, score desc, node_id) and cut at top_n."""
    ctes = [_ref_prelude(g, seeds, direction), "r0 AS (SELECT seed, node_id FROM h0)"]
    for i in range(1, depth + 1):
        kept = " UNION ALL ".join(f"SELECT seed, node_id FROM r{j}" for j in range(i))
        ctes.append(
            f"r{i} AS (SELECT seed, node_id FROM (SELECT c.seed, c.node_id, ROW_NUMBER() OVER "
            f"(PARTITION BY c.seed ORDER BY sc.score DESC NULLS LAST, c.node_id ASC) AS rn "
            f"FROM (SELECT DISTINCT r.seed, e.dst AS node_id FROM r{i - 1} r JOIN e ON e.src = r.node_id) c "
            f"LEFT ANTI JOIN ({kept}) k ON k.seed = c.seed AND k.node_id = c.node_id "
            f"LEFT JOIN sc ON sc.node_id = c.node_id) WHERE rn <= {top_n})")
    union = " UNION ALL ".join(f"SELECT seed, node_id, {i} AS d FROM r{i}" for i in range(1, depth + 1))
    sql = (f"WITH {', '.join(ctes)} "
           f"SELECT seed, node_id, d, rn FROM (SELECT m.seed, m.node_id, m.d, ROW_NUMBER() OVER "
           f"(PARTITION BY m.seed ORDER BY m.d, sc.score DESC NULLS LAST, m.node_id ASC) AS rn "
           f"FROM ({union}) m LEFT JOIN sc ON sc.node_id = m.node_id) WHERE rn <= {top_n}")
    out: Dict[str, List[Tuple[str, int]]] = {k: [] for k in seeds}
    for seed, node, d, rn in sorted(execute_sql(w, WAREHOUSE_ID, sql), key=lambda r: (r[0], int(r[3]))):
        out[seed].append((str(node), int(d)))
    return out


def spark_similar_reference(w: Any, g: Dict[str, Any], seeds: List[str],
                            top_n: int) -> Dict[str, List[str]]:
    """{seed: [node_id, ...]} by cosine distance 1 - dot/sqrt(|a|^2 |b|^2), ties by node_id."""
    cat, kit, fx = g["lakebase_catalog"], g["kit_schema"], g["fixture_schema"]
    keys = ", ".join(f"'{k}'" for k in seeds)
    sql = (f"WITH v AS (SELECT m.node_id, f.provider_id, CAST(f.x AS DOUBLE) x, CAST(f.y AS DOUBLE) y, "
           f"CAST(f.z AS DOUBLE) z FROM {cat}.{fx}.fixture_embeddings f "
           f"JOIN {cat}.{kit}.node_id_map m ON m.natural_key = f.provider_id) "
           f"SELECT seed, node_id, rn FROM (SELECT a.provider_id AS seed, b.node_id, ROW_NUMBER() OVER "
           f"(PARTITION BY a.provider_id ORDER BY 1 - (a.x*b.x + a.y*b.y + a.z*b.z) / "
           f"sqrt((a.x*a.x + a.y*a.y + a.z*a.z) * (b.x*b.x + b.y*b.y + b.z*b.z)), b.node_id) AS rn "
           f"FROM v a JOIN v b ON a.node_id <> b.node_id WHERE a.provider_id IN ({keys})) "
           f"WHERE rn <= {top_n}")
    out: Dict[str, List[str]] = {k: [] for k in seeds}
    for seed, node, rn in sorted(execute_sql(w, WAREHOUSE_ID, sql), key=lambda r: (r[0], int(r[2]))):
        out[seed].append(str(node))
    return out


# ---------------------------------------------------------------------------- helpers


def _enrich_kw(g: Dict[str, Any]) -> Dict[str, Any]:
    return {"served_schema": g["served_schema"],
            "enrich_schema": version_schema(g), "sort_method": "degree"}


def _run(conn: Any, sql: str, seed_id: str) -> List[tuple]:
    return conn.execute(sql, {"start_node_id": seed_id}).fetchall()


# ---------------------------------------------------------------------------- tests


class TestWebWalk:
    @pytest.mark.seam1
    @pytest.mark.provision
    def test_web_equals_spark_reference_20_seeds_every_depth(self, walk_graph, workspace_client, record_metric):
        g, ids = walk_graph, walk_graph["ids"]
        sqls = per_depth_sql("web", 4, top_n=WEB_TOP_N, **_enrich_kw(g))
        assert sorted(sqls) == list(DEPTHS)
        ref = spark_web_reference(workspace_client, g, WEB_SEEDS, 4, WEB_TOP_N)
        # Non-vacuous: every seed reaches the hub first, and hop 4 is full.
        assert all(r[0][:2] == (1, ids["hub"]) for r in ref.values())
        assert all(len([x for x in r if x[0] == 4]) == WEB_TOP_N for r in ref.values())
        samples: Dict[int, List[float]] = {d: [] for d in DEPTHS}
        with _pg(g) as conn:
            for depth in DEPTHS:
                for k in WEB_SEEDS:
                    t0 = time.perf_counter()
                    rows = _run(conn, sqls[depth], ids[k])
                    samples[depth].append(time.perf_counter() - t0)
                    got = [(int(h), str(n), int(s), float(sc)) for h, n, s, sc in rows]
                    want = [x for x in ref[k] if x[0] <= depth]
                    assert got == want, f"web depth {depth} seed {k}: {got} != {want}"
        for depth in DEPTHS:
            record_metric.latencies("walk.web", samples[depth], branch=g["branch_id"], depth=depth, kind="degree",
                                    **g["graph_size"])

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_web_walk_in_region_latency(self, walk_graph, workspace_client, record_metric, tmp_path):
        """client=workspace: the same web walk (degree sort, depths 1-4) over the same 20 seeds, timed
        inside the workspace by a serverless notebook job (the shipped notebook runner, the wheel of
        this tree), on this suite's t- branch."""
        import secrets

        from lgk.workspace_latency import record_workspace_samples, run_in_region
        from tests.test_parity_live import _build_wheel

        g, ids = walk_graph, walk_graph["ids"]
        sqls = per_depth_sql("web", 4, top_n=WEB_TOP_N, **_enrich_kw(g))
        seeds = [ids[k] for k in WEB_SEEDS]
        queries = [{"metric": "walk.web", "depth": d, "kind": "degree", "sql": sqls[d], "seeds": seeds}
                   for d in DEPTHS]
        with _pg(g) as conn:  # the row count each seed returns from the laptop, per depth
            want = {d: [len(_run(conn, sqls[d], n)) for n in seeds] for d in DEPTHS}
        assert all(want[4]), "non-vacuous: every seed has rows at depth 4"
        me = workspace_client.current_user.me().user_name
        samples = run_in_region(workspace_client, g, queries, _build_wheel(tmp_path / "dist"), tmp_path / "run",
                                f"/Workspace/Users/{me}/lgk_v1_{secrets.token_hex(3)}")
        assert sorted(s.context["depth"] for s in samples) == list(DEPTHS)
        for s in samples:
            assert s.rows == want[s.context["depth"]], f"in-region depth {s.context['depth']}: row counts differ"
        record_workspace_samples(record_metric, samples, branch=g["branch_id"], **g["graph_size"])

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_web_default_sort_by_support_equals_spark_reference(self, walk_graph, workspace_client):
        """sort_method=None: support DESC then node_id; support values must match too."""
        g, ids = walk_graph, walk_graph["ids"]
        ref = spark_web_reference(workspace_client, g, WEB_SEEDS, 4, WEB_TOP_N, by_score=False)
        # Non-vacuous: some candidate has support > 1, so the support sort is exercised.
        assert any(x[2] > 1 for r in ref.values() for x in r)
        sqls = per_depth_sql("web", 4, top_n=WEB_TOP_N, served_schema=g["served_schema"])
        with _pg(g) as conn:
            for depth in DEPTHS:
                for k in WEB_SEEDS:
                    got = [(int(h), str(n), int(s), sc) for h, n, s, sc in _run(conn, sqls[depth], ids[k])]
                    want = [x for x in ref[k] if x[0] <= depth]
                    assert got == want, f"web(support) depth {depth} seed {k}: {got} != {want}"

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_hub_next_hop_stays_bounded_and_fast(self, walk_graph, workspace_client, record_metric):
        """# hostile: hub with 10,000 links — each hop returns at most top_n rows, quickly."""
        g, ids = walk_graph, walk_graph["ids"]
        with _pg(g) as conn:
            n_links = conn.execute(
                f"SELECT COUNT(*) FROM {g['served_schema']}.edges_view WHERE src_node_id = %s",
                (ids["hub"],)).fetchone()[0]
            assert n_links == N_LEAVES
            for seed, depth in (("hub", 1), ("hub", 2), ("w_000", 2), ("w_000", 4)):
                sql = web_sql(depth=depth, top_n=WEB_TOP_N, **_enrich_kw(g))
                _run(conn, sql, ids[seed])  # warm-up
                t0 = time.monotonic()
                rows = _run(conn, sql, ids[seed])
                elapsed = time.monotonic() - t0
                per_hop = {h: sum(1 for r in rows if r[0] == h) for h in range(1, depth + 1)}
                assert max(per_hop.values()) <= WEB_TOP_N, f"{seed} d{depth}: {per_hop}"
                assert len(rows) <= WEB_TOP_N * depth
                assert elapsed < HUB_TIME_BOUND_S, f"{seed} d{depth} took {elapsed:.2f}s"
                record_metric("walk.web.hub_query", elapsed * 1000.0, "ms", branch=g["branch_id"],
                              depth=depth, kind="hub" if seed == "hub" else "web", **g["graph_size"])
            hub_rows = _run(conn, web_sql(depth=1, top_n=WEB_TOP_N, **_enrich_kw(g)), ids["hub"])
        assert len(hub_rows) == WEB_TOP_N

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_tied_scores_break_on_node_id(self, walk_graph, workspace_client):
        """# hostile: 10,000 leaves tie on degree 1; the byte-smallest node_ids win."""
        g, ids = walk_graph, walk_graph["ids"]
        scores = {r[0] for r in execute_sql(workspace_client, WAREHOUSE_ID,
            f"SELECT DISTINCT score FROM {g['lakebase_catalog']}.{g['kit_schema']}.node_score "
            f"WHERE method = 'degree' AND node_id IN (SELECT node_id FROM "
            f"{g['lakebase_catalog']}.{g['kit_schema']}.node_id_map WHERE natural_key LIKE 'leaf_%')")}
        assert len(scores) == 1, f"leaves must all tie, got {scores}"
        # Spark's STRING order is byte order, the order the walk must use for ties.
        want = [str(r[0]) for r in execute_sql(workspace_client, WAREHOUSE_ID,
            f"SELECT node_id FROM {g['lakebase_catalog']}.{g['kit_schema']}.node_id_map "
            f"WHERE natural_key LIKE 'leaf_%' ORDER BY node_id LIMIT {WEB_TOP_N}")]
        ref = spark_web_reference(workspace_client, g, ["hub"], 1, WEB_TOP_N)["hub"]
        assert [x[1] for x in ref] == want
        with _pg(g) as conn:
            got = [str(r[1]) for r in _run(conn, web_sql(depth=1, top_n=WEB_TOP_N, **_enrich_kw(g)), ids["hub"])]
        assert got == want


    @pytest.mark.seam1
    @pytest.mark.provision
    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_web_returns_each_node_once(self, walk_graph, direction):
        """Each node_id at most once across all hops, never the seed, on 22 seeds at
        depths 1-4, by support and by score. Reads only the Postgres result, never the reference."""
        reason = direction_skip_reason(web_sql, direction)
        if reason:
            pytest.skip(reason)
        g, ids = walk_graph, walk_graph["ids"]
        sorts = {"support": {"served_schema": g["served_schema"]}, "score": _enrich_kw(g)}
        with _pg(g) as conn:
            for sort, kw in sorts.items():
                for depth in DEPTHS:
                    sql = _web_for(web_sql, direction, depth=depth, top_n=WEB_TOP_N, **kw)
                    for k in UNIQUE_SEEDS:
                        rows = _run(conn, sql, ids[k])
                        assert rows, f"{sort} depth {depth} seed {k}: empty walk"
                        assert_each_node_once(rows, ids[k], depth, WEB_TOP_N)
                    cycle = {"outbound": ["c_b", "c_c"], "inbound": ["c_c", "c_b"]}.get(direction)
                    if cycle:
                        # Non-vacuous: the 3-cycle walks two nodes, then stops short of its seed.
                        got = [(int(r[0]), str(r[1])) for r in _run(conn, sql, ids["c_a"])]
                        want = [(1, ids[cycle[0]]), (2, ids[cycle[1]])][:depth]
                        assert got == want, f"{direction} c_a depth {depth}: {got}"


class TestTreeWalk:
    @pytest.mark.seam1
    @pytest.mark.provision
    def test_tree_equals_spark_reference_20_seeds_every_depth(self, walk_graph, workspace_client):
        g, ids = walk_graph, walk_graph["ids"]
        sqls = per_depth_sql("tree", 4, top_n=TREE_TOP_N, **_enrich_kw(g))
        with _pg(g) as conn:
            for depth in DEPTHS:
                ref = spark_tree_reference(workspace_client, g, TREE_SEEDS, depth, TREE_TOP_N)
                assert any(ref.values())
                for k in TREE_SEEDS:
                    got = [(str(n), int(d)) for n, d, _sc in _run(conn, sqls[depth], ids[k])]
                    assert got == ref[k], f"tree depth {depth} seed {k}: {got} != {ref[k]}"
                    assert ids[k] not in [n for n, _ in got], f"seed {k} is its own descendant"

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_hub_tree_levels_stay_bounded_and_equal_spark_reference(self, walk_graph, workspace_client):
        """# hostile: the hub's 10,000 children land in one tree level. Every level the walk
        keeps holds at most top_n nodes (read from the walk's own level rows), each query is
        fast, and the result equals the per-level-cut Spark reference for 20 seeds per depth."""
        g, ids = walk_graph, walk_graph["ids"]
        kw = _enrich_kw(g)
        with _pg(g) as conn:
            for depth in DEPTHS:
                levels_sql = (tree_walk_cte(depth=depth, top_n=TREE_TOP_N, **kw)
                              + "\nSELECT depth, cardinality(frontier) FROM walk WHERE depth > 0")
                for seed in ("hub", "w_000"):
                    sizes = dict(_run(conn, levels_sql, ids[seed]))
                    assert sizes and max(sizes.values()) <= TREE_TOP_N, f"{seed} d{depth}: {sizes}"
                    sql = tree_sql(depth=depth, top_n=TREE_TOP_N, **kw)
                    _run(conn, sql, ids[seed])  # warm-up
                    t0 = time.monotonic()
                    rows = _run(conn, sql, ids[seed])
                    elapsed = time.monotonic() - t0
                    assert len(rows) <= TREE_TOP_N
                    assert elapsed < HUB_TIME_BOUND_S, f"tree {seed} d{depth} took {elapsed:.2f}s"
            # Non-vacuous: the hub's first level really is a 10,000-wide candidate set.
            assert dict(_run(conn, tree_walk_cte(depth=1, top_n=N_LEAVES + 1, **kw)
                             + "\nSELECT depth, cardinality(frontier) FROM walk WHERE depth > 0",
                             ids["hub"])) == {1: N_LEAVES}
            sqls = per_depth_sql("tree", 4, top_n=TREE_TOP_N, **kw)
            for depth in DEPTHS:
                ref = spark_tree_reference(workspace_client, g, HUB_TREE_SEEDS, depth, TREE_TOP_N)
                assert all(ref.values())
                for k in HUB_TREE_SEEDS:
                    got = [(str(n), int(d)) for n, d, _sc in _run(conn, sqls[depth], ids[k])]
                    assert got == ref[k], f"hub tree depth {depth} seed {k}: {got} != {ref[k]}"

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_cycle_terminates_with_correct_set(self, walk_graph):
        """# hostile: cycle c_a->c_b->c_c->c_a terminates, even at the deepest cap."""
        g, ids = walk_graph, walk_graph["ids"]
        want = [(ids["c_b"], 1), (ids["c_c"], 2)]
        with _pg(g) as conn:
            conn.execute("SET statement_timeout = '20s'")
            for depth in (3, 4, 100):
                rows = _run(conn, tree_sql(depth=depth, top_n=50, **_enrich_kw(g)), ids["c_a"])
                assert [(str(n), int(d)) for n, d, _sc in rows] == want, f"depth {depth}: {rows}"

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_diamond_chain_is_fast_and_equals_spark_reference(self, walk_graph, workspace_client):
        """# hostile: 25 chained diamonds (2^25 paths) at depth 50 and 60: bounded time, the
        full node set at the right depths, equal to a reference built from the Delta sources.

        Reference: the diamond's source edges read from Delta (fixture affiliations mapped
        through the Delta node_id_map), then a breadth-first search in Python. It is
        computed BEFORE the Postgres connection opens, so no transaction sits idle.
        """
        g, ids = walk_graph, walk_graph["ids"]
        n_nodes = 3 * N_DIAMONDS
        cat, kit, fx = g["lakebase_catalog"], g["kit_schema"], g["fixture_schema"]
        adj: Dict[str, set] = {}
        for src, dst in execute_sql(workspace_client, WAREHOUSE_ID,
                f"SELECT DISTINCT e.src, e.dst FROM ({_ref_edges(g)}) e JOIN {cat}.{kit}.node_id_map m "
                f"ON m.node_id = e.src WHERE m.natural_key LIKE 'd%'"):
            adj.setdefault(str(src), set()).add(str(dst))
        dist, frontier = {ids["d_00"]: 0}, [ids["d_00"]]
        while frontier:
            nxt = sorted({d for n in frontier for d in adj.get(n, ()) if d not in dist})
            for d in nxt:
                dist[d] = dist[frontier[0]] + 1
            frontier = nxt
        scores = {str(n): float(sc) for n, sc in execute_sql(workspace_client, WAREHOUSE_ID,
            f"SELECT node_id, score FROM {cat}.{kit}.node_score WHERE method = 'degree'")
            if str(n) in dist}
        ref = sorted(((n, d) for n, d in dist.items() if d > 0),
                     key=lambda r: (r[1], -scores.get(r[0], float("-inf")), r[0].encode()))
        assert len(ref) == n_nodes and ref[-1] == (ids[f"d_{N_DIAMONDS:02d}"], 2 * N_DIAMONDS)
        conn = _pg(g)
        try:
            conn.autocommit = True
            conn.execute("SET statement_timeout = '30s'")
            for depth in (2 * N_DIAMONDS, 60):
                sql = tree_sql(depth=depth, top_n=n_nodes + 10, **_enrich_kw(g))
                _run(conn, sql, ids["d_00"])  # warm-up
                t0 = time.monotonic()
                rows = _run(conn, sql, ids["d_00"])
                elapsed = time.monotonic() - t0
                assert [(str(n), int(d)) for n, d, _sc in rows] == ref, f"depth {depth}"
                assert elapsed < DIAMOND_TIME_BOUND_S, f"depth {depth} took {elapsed:.2f}s"
        finally:
            conn.close()


class TestSimilarWalk:
    @pytest.mark.seam1
    @pytest.mark.provision
    def test_similar_pgvector_hnsw_equals_spark_reference(self, walk_graph, workspace_client):
        g, ids = walk_graph, walk_graph["ids"]
        table = f"lgk_vec_{uuid.uuid4().hex[:6]}.embeddings"  # branch-local; goes with the branch
        rows = execute_sql(workspace_client, WAREHOUSE_ID,
            f"SELECT provider_id, x, y, z FROM {g['lakebase_catalog']}.{g['fixture_schema']}.fixture_embeddings")
        ref = spark_similar_reference(workspace_client, g, WEB_SEEDS, SIMILAR_TOP_N)
        sql = similar_sql(table, top_n=SIMILAR_TOP_N)
        conn = _pg(g)
        try:
            conn.autocommit = True
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f"CREATE SCHEMA {table.split('.')[0]}")
            conn.execute(f"CREATE TABLE {table} (node_id text PRIMARY KEY, embedding vector(3))")
            with conn.cursor() as cur:
                cur.executemany(f"INSERT INTO {table} VALUES (%s, %s::vector)",
                                [(ids[k], f"[{x},{y},{z}]") for k, x, y, z in rows])
            conn.execute(vector_index_sql(table))
            conn.execute("SET enable_seqscan = off")  # prove the HNSW index serves the order
            plan = "\n".join(r[0] for r in conn.execute(f"EXPLAIN {sql}", {"start_node_id": ids["w_000"]}))
            assert "hnsw" in plan.lower() or "_hnsw" in plan, plan
            for k in WEB_SEEDS:
                got = [str(r[0]) for r in _run(conn, sql, ids[k])]
                assert got == ref[k], f"similar seed {k}: {got} != {ref[k]}"
        finally:
            conn.close()


class TestWalkDirection:
    """Inbound and both walks, against the same independent Spark reference."""

    @pytest.mark.seam1
    @pytest.mark.provision
    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_web_every_direction_equals_spark_reference(self, walk_graph, workspace_client, direction):
        g, ids = walk_graph, walk_graph["ids"]
        sorts = {"score": (_enrich_kw(g), True), "support": ({"served_schema": g["served_schema"]}, False)}
        for sort, (kw, by_score) in sorts.items():
            ref = spark_web_reference(workspace_client, g, DIRECTION_WEB_SEEDS, 4, WEB_TOP_N,
                                      by_score=by_score, direction=direction)
            # Non-vacuous: m_a's hop 1 holds exactly its direction's discriminator
            hop1 = {x[1] for x in ref["m_a"] if x[0] == 1}
            want1 = {"outbound": {"m_b", "m_c"}, "inbound": {"m_b", "m_d"}, "both": {"m_b", "m_c", "m_d"}}
            assert hop1 == {ids[k] for k in want1[direction]}, f"{sort} reference m_a hop 1: {hop1}"
            sqls = per_depth_sql("web", 4, top_n=WEB_TOP_N, direction=direction, **kw)
            with _pg(g) as conn:
                for depth in DEPTHS:
                    for k in DIRECTION_WEB_SEEDS:
                        got = [(int(h), str(n), int(s), float(sc) if by_score else sc)
                               for h, n, s, sc in _run(conn, sqls[depth], ids[k])]
                        want = [x for x in ref[k] if x[0] <= depth]
                        assert got == want, f"{direction} web({sort}) depth {depth} seed {k}: {got} != {want}"

    @pytest.mark.seam1
    @pytest.mark.provision
    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_tree_every_direction_equals_spark_reference(self, walk_graph, workspace_client, direction):
        g, ids = walk_graph, walk_graph["ids"]
        sqls = per_depth_sql("tree", 4, top_n=TREE_TOP_N, direction=direction, **_enrich_kw(g))
        for depth in DEPTHS:
            ref = spark_tree_reference(workspace_client, g, DIRECTION_TREE_SEEDS, depth, TREE_TOP_N,
                                       direction=direction)
            assert all(ref[k] for k in DIRECTION_TREE_SEEDS), f"{direction} depth {depth}: an empty reference"
            with _pg(g) as conn:
                for k in DIRECTION_TREE_SEEDS:
                    got = [(str(n), int(d)) for n, d, _sc in _run(conn, sqls[depth], ids[k])]
                    assert got == ref[k], f"{direction} tree depth {depth} seed {k}: {got} != {ref[k]}"
                    assert ids[k] not in [n for n, _ in got], f"{direction}: seed {k} is its own descendant"

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_sink_inbound_hop_stays_bounded_and_fast(self, walk_graph, workspace_client, record_metric):
        """# hostile: fan-in sink (10,000 in-links): inbound from the sink, and both from a leaf
        (hop 2 holds the hub's 10,000 leaves and the sink's), stay within top_n per hop, quickly,
        after reapply's indexes exist (src and dst)."""
        from lgk.steps.reapply import ensure_indexes

        g, ids = walk_graph, walk_graph["ids"]
        cat, kit = g["lakebase_catalog"], g["kit_schema"]
        [(leaf,)] = execute_sql(workspace_client, WAREHOUSE_ID,
                                f"SELECT node_id FROM {cat}.{kit}.node_id_map WHERE natural_key = 'leaf_00000'")
        conn = _pg(g)
        try:
            conn.autocommit = True
            ensure_indexes(conn, version_schema(g))  # what reapply makes before the site serves
            n_in = conn.execute(f"SELECT COUNT(*) FROM {g['served_schema']}.edges_view WHERE dst_node_id = %s",
                                (ids["sink"],)).fetchone()[0]
            assert n_in == N_LEAVES
            for seed, direction, depth in ((ids["sink"], "inbound", 1), (ids["sink"], "inbound", 2),
                                           (ids["sink"], "both", 2), (str(leaf), "both", 2),
                                           (str(leaf), "both", 4)):
                for kw in (_enrich_kw(g), {"served_schema": g["served_schema"]}):
                    sql = web_sql(depth=depth, top_n=WEB_TOP_N, direction=direction, **kw)
                    _run(conn, sql, seed)  # warm-up
                    t0 = time.monotonic()
                    rows = _run(conn, sql, seed)
                    elapsed = time.monotonic() - t0
                    per_hop = {h: sum(1 for r in rows if r[0] == h) for h in range(1, depth + 1)}
                    # non-vacuous: some hop is full (the sink's or the hub's 10,000 are cut to top_n)
                    assert max(per_hop.values()) == WEB_TOP_N, f"{direction} d{depth}: {per_hop}"
                    assert elapsed < HUB_TIME_BOUND_S, f"{direction} d{depth} took {elapsed:.2f}s"
                    sort = "degree" if "sort_method" in kw else "support"
                    record_metric(f"walk.web.sink_query.{sort}", elapsed * 1000.0, "ms", branch=g["branch_id"],
                                  depth=depth, direction=direction, kind="sink" if seed == ids["sink"] else "leaf",
                                  **g["graph_size"])
        finally:
            conn.close()

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_sink_inbound_tree_levels_stay_bounded_and_equal_spark_reference(self, walk_graph, workspace_client):
        """# hostile: the sink's 10,000 in-links land in one INBOUND tree level (final review P3,
        the tree twin of test_hub_tree_levels_stay_bounded_and_equal_spark_reference). Every
        level the walk keeps holds at most top_n nodes (read from the walk's own level rows), each
        query is fast once reapply's indexes exist, and the result equals the per-level-cut Spark
        reference (inbound) at every depth."""
        from lgk.steps.reapply import ensure_indexes

        g, ids = walk_graph, walk_graph["ids"]
        kw = _enrich_kw(g)
        levels = "\nSELECT depth, cardinality(frontier) FROM walk WHERE depth > 0"
        # the reference is read BEFORE the Postgres connection opens, so no transaction sits idle
        refs = {d: spark_tree_reference(workspace_client, g, ["sink"], d, TREE_TOP_N, direction="inbound")["sink"]
                for d in DEPTHS}
        assert all(refs.values()) and len(refs[1]) == TREE_TOP_N  # non-vacuous: level 1 is cut
        conn = _pg(g)
        try:
            conn.autocommit = True
            ensure_indexes(conn, version_schema(g))  # what reapply makes before the site serves
            # Non-vacuous: the sink's first inbound level really is a 10,000-wide candidate set.
            assert dict(_run(conn, tree_walk_cte(depth=1, top_n=N_LEAVES + 1, direction="inbound", **kw)
                             + levels, ids["sink"])) == {1: N_LEAVES}
            for depth in DEPTHS:
                sizes = dict(_run(conn, tree_walk_cte(depth=depth, top_n=TREE_TOP_N, direction="inbound",
                                                      **kw) + levels, ids["sink"]))
                assert sizes and max(sizes.values()) <= TREE_TOP_N, f"inbound sink d{depth}: {sizes}"
                assert sizes[1] == TREE_TOP_N, f"inbound sink d{depth}: level 1 {sizes}"
                sql = tree_sql(depth=depth, top_n=TREE_TOP_N, direction="inbound", **kw)
                _run(conn, sql, ids["sink"])  # warm-up
                t0 = time.monotonic()
                rows = _run(conn, sql, ids["sink"])
                elapsed = time.monotonic() - t0
                assert len(rows) <= TREE_TOP_N
                assert elapsed < HUB_TIME_BOUND_S, f"inbound tree sink d{depth} took {elapsed:.2f}s"
                got = [(str(n), int(d)) for n, d, _sc in rows]
                assert got == refs[depth], f"inbound sink tree depth {depth}: {got} != {refs[depth]}"
        finally:
            conn.close()

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_both_counts_the_mutual_pair_once_live(self, walk_graph):
        g, ids = walk_graph, walk_graph["ids"]
        with _pg(g) as conn:
            rows = _run(conn, web_sql(depth=1, top_n=WEB_TOP_N, served_schema=g["served_schema"],
                                      direction="both"), ids["m_a"])
        assert {str(n): int(s) for _h, n, s, _sc in rows} == {ids["m_b"]: 1, ids["m_c"]: 1, ids["m_d"]: 1}


# ============================================================================ the set walk
# A second, small graph (`web_set_graph`, its own t- branch) with two node types and two edge
# types: the walks fixture above is one node type, so it cannot hold "a hospital with the seed's
# specialty" or "a stop of another type". Its rows are tests.test_walks.SET_NODES / SET_EDGES (the
# offline twin), each hostile row named there:
# hostile: wrong type, same value (h_card, specialty cardiology, out-edge to p_trap);
# hostile: duplicate at the aggregation grain (h_z reached from p_c1 and p_c2: support 2);
# hostile: expansion past a stop (h_out -> p_x, p_x reachable only through h_out);
# hostile: stop-type check (p_out, a provider with network_status out, expands to h_far);
# a seed that matches the stop (seed hospital.network_status = out, stop the same);
# hostile: exact equality (p_cx, specialty 'Cardiology ', its hospital h_only_x never appears);
# hostile: reordered input (the cardiology seeds inserted in descending key order);
# hostile: varied configuration (max_seeds 3 with 4 cardiology seeds).
WS_TOP_NS = (2, 25)
CARDIO = (NodePredicate("provider", "specialty"), "cardiology")
OUT = (NodePredicate("hospital", "network_status"), "out")
WS_SEEDS = (CARDIO, OUT, (NodePredicate("provider"), None), (NodePredicate("provider", "label"), "Dr C1"))


def _sql_or_null(v: Any) -> str:
    return "NULL" if v is None else f"'{v}'"


def _create_web_set_fixture(w: Any, catalog: str, fixture_schema: str) -> None:
    s = f"{catalog}.{fixture_schema}"
    q = lambda sql: execute_sql(w, WAREHOUSE_ID, sql)  # noqa: E731
    q(f"CREATE SCHEMA IF NOT EXISTS {s}")
    for table, key, kind in (("fixture_providers", "provider_id", "provider"),
                             ("fixture_hospitals", "hospital_id", "hospital")):
        q(f"CREATE OR REPLACE TABLE {s}.{table} ({key} STRING, name STRING, specialty STRING, "
          f"network_status STRING) USING DELTA")
        q(f"INSERT INTO {s}.{table} VALUES " + ", ".join(  # SET_NODES order: seeds descending
            f"('{n}', '{lb}', {_sql_or_null(p.get('specialty'))}, {_sql_or_null(p.get('network_status'))})"
            for n, t, _k, lb, p in SET_NODES if t == kind))
    q(f"CREATE OR REPLACE TABLE {s}.fixture_affiliations (affiliation_id STRING, provider_id STRING, "
      f"hospital_id STRING, start_date STRING) USING DELTA")
    q(f"INSERT INTO {s}.fixture_affiliations VALUES " + ", ".join(
        f"('a{i}', '{a}', '{b}', '2024-01-01')" for i, (a, b, t) in enumerate(SET_EDGES) if t == "affiliation"))
    q(f"CREATE OR REPLACE TABLE {s}.fixture_referrals (hospital_id STRING, provider_id STRING) USING DELTA")
    q(f"INSERT INTO {s}.fixture_referrals VALUES " + ", ".join(
        f"('{a}', '{b}')" for a, b, t in SET_EDGES if t == "refers"))


def _write_web_set_mapping(path: Any, catalog: str, source_schema: str) -> None:
    path.write_text(f"""source_catalog: {catalog}
source_schema: {source_schema}

nodes:
  provider:
    source_table: fixture_providers
    natural_key: provider_id
    label: name
    properties: [specialty, network_status]
    walk_keys: [specialty]
  hospital:
    source_table: fixture_hospitals
    natural_key: hospital_id
    label: name
    properties: [specialty, network_status]
    walk_keys: [network_status]

edges:
  affiliation:
    source_table: fixture_affiliations
    src: provider_id
    src_type: provider
    dst: hospital_id
    dst_type: hospital
  refers:
    source_table: fixture_referrals
    src: hospital_id
    src_type: hospital
    dst: provider_id
    dst_type: provider
""")


@pytest.fixture(scope="module")
def web_set_graph(workspace_client, lakebase_project, lakebase_catalog, run_folder):
    with provisioned_graph(
        workspace_client, lakebase_project, lakebase_catalog, run_folder, "webset",
        _create_web_set_fixture, enrich_methods=["degree"], write_mapping=_write_web_set_mapping,
    ) as config:
        config["ids"] = {str(k): str(v) for k, v in execute_sql(workspace_client, WAREHOUSE_ID,
            f"SELECT natural_key, node_id FROM {lakebase_catalog}.{config['kit_schema']}.node_id_map")}
        yield config


def _ref_pred(alias: str, pred: Tuple[NodePredicate, Any]) -> str:
    """The predicate in Spark: Delta `nodes`, MAP access for a walk key (not the Postgres ->>)."""
    p, value = pred
    cond = f"{alias}.node_type = '{p.node_type}'"
    if p.field is None:
        return cond
    col = {"label": f"{alias}.label", "natural_key": f"{alias}.natural_key"}.get(
        p.field, f"{alias}.properties['{p.field}']")
    return f"{cond} AND {col} = '{value}'"


def spark_web_set_reference(w: Any, g: Dict[str, Any], seed: Tuple[NodePredicate, Any],
                            stop: Any, depth: int, top_n: int) -> List[Tuple[int, str, int, bool]]:
    """[(hop, node_id, support, matched)] from the §3.1 semantics in Spark: h0 = the seed set (Delta
    nodes, MAP access); h_i = out-neighbours of h_{i-1}'s unmatched nodes over the SOURCE edge tables,
    LEFT ANTI JOIN h0..h_{i-1}, ROW_NUMBER over (support DESC, node_id) with no PARTITION BY (one
    set), rn <= top_n; matched = a LEFT JOIN on Delta nodes with the stop predicate."""
    cat, kit, fx = g["lakebase_catalog"], g["kit_schema"], g["fixture_schema"]
    nodes = f"{cat}.{kit}.nodes"
    edges = (f"SELECT ms.node_id AS src, md.node_id AS dst FROM ("
             f"SELECT DISTINCT provider_id AS s, hospital_id AS d FROM {cat}.{fx}.fixture_affiliations "
             f"UNION ALL SELECT DISTINCT hospital_id AS s, provider_id AS d FROM {cat}.{fx}.fixture_referrals) a "
             f"JOIN {cat}.{kit}.node_id_map ms ON ms.natural_key = a.s "
             f"JOIN {cat}.{kit}.node_id_map md ON md.natural_key = a.d")
    ctes = [f"e AS ({edges})",
            f"h0 AS (SELECT CAST(n.node_id AS STRING) AS node_id, false AS matched, 0 AS rn, 0 AS support "
            f"FROM {nodes} n WHERE {_ref_pred('n', seed)})"]
    stop_rel = (f"(SELECT CAST(m.node_id AS STRING) AS node_id FROM {nodes} m WHERE {_ref_pred('m', stop)})"
                if stop else "(SELECT CAST(NULL AS STRING) AS node_id WHERE false)")
    for i in range(1, depth + 1):
        seen = " UNION ALL ".join(f"SELECT node_id FROM h{j}" for j in range(i))
        ctes.append(
            f"h{i} AS (SELECT r.node_id, (st.node_id IS NOT NULL) AS matched, r.rn, r.support FROM ("
            f"SELECT c.node_id, c.support, ROW_NUMBER() OVER (ORDER BY c.support DESC, c.node_id ASC) AS rn "
            f"FROM (SELECT e.dst AS node_id, COUNT(*) AS support FROM h{i - 1} f JOIN e ON e.src = f.node_id "
            f"WHERE NOT f.matched GROUP BY e.dst) c "
            f"LEFT ANTI JOIN ({seen}) s ON s.node_id = c.node_id) r "
            f"LEFT JOIN {stop_rel} st ON st.node_id = r.node_id WHERE r.rn <= {top_n})")
    union = " UNION ALL ".join(f"SELECT {i} AS hop, rn, node_id, support, matched FROM h{i}"
                               for i in range(1, depth + 1))
    rows = execute_sql(w, WAREHOUSE_ID, f"WITH {', '.join(ctes)} {union}")
    return [(int(h), str(n), int(s), str(m).lower() == "true")
            for h, _rn, n, s, m in sorted(rows, key=lambda r: (int(r[0]), int(r[1])))]


def _set_walk(g: Dict[str, Any], seed: Tuple[NodePredicate, Any], stop: Any, depth: int, top_n: int = 25,
              max_seeds: int = 200) -> List[Tuple[int, str, int, bool]]:
    from lgk.steps.walks import get_walk_query

    sql = get_walk_query("unused", depth, g["served_schema"], top_n=top_n, run_folder=g["run_folder"],
                         seed=seed[0], stop=stop[0] if stop else None, max_seeds=max_seeds)
    with _pg(g) as conn:
        rows = conn.execute(sql, {"seed_value": seed[1], "stop_value": stop[1] if stop else None}).fetchall()
    return [(int(h), str(n), int(s), bool(m)) for h, n, s, _sc, m in rows]


def _named_rows(g: Dict[str, Any], rows: List[Tuple[int, str, int, bool]]) -> List[Tuple[int, str, int, bool]]:
    names = {v: k for k, v in g["ids"].items()}
    return [(h, names[n], s, m) for h, n, s, m in rows]


class TestWebSetWalk:
    @pytest.mark.seam1
    @pytest.mark.provision
    def test_web_set_equals_spark_reference_every_depth(self, web_set_graph, workspace_client):
        g = web_set_graph
        for seed in WS_SEEDS:
            for stop in (None, OUT):
                for top_n in WS_TOP_NS:
                    ref = spark_web_set_reference(workspace_client, g, seed, stop, 4, top_n)
                    assert ref, f"empty reference for {seed} {stop} {top_n}"
                    for depth in DEPTHS:
                        got = _set_walk(g, seed, stop, depth, top_n)
                        want = [r for r in ref if r[0] <= depth]
                        assert got == want, f"{seed} stop={stop} top_n={top_n} depth {depth}: {got} != {want}"
        # Non-vacuous: the cardiology walk with the stop reaches hop 4 and marks the stop node
        full = _named_rows(g, _set_walk(g, CARDIO, OUT, 4))
        assert (1, "h_out", 1, True) in full and full[-1][0] == 4

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_single_seed_reference_is_todays_web_reference(self, web_set_graph, workspace_client):
        """The one-seed case of the set reference equals the existing spark_web_reference (by
        support), hop 1 from each provider (its affiliation edges: the old reference's edge table)."""
        g, ids = web_set_graph, web_set_graph["ids"]
        providers = [n for n, t, *_ in SET_NODES if t == "provider"]
        old = spark_web_reference(workspace_client, g, providers, 1, 25, by_score=False)
        assert any(old[k] for k in providers)
        for k in providers:
            new = spark_web_set_reference(workspace_client, g, (NodePredicate("provider", "natural_key"), k),
                                          None, 1, 25)
            assert [(h, n, s) for h, n, s, _m in new] == [(h, n, s) for h, n, s, _sc in old[k]], k
            assert ids[k] not in [n for _h, n, _s, _m in new]

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_single_seed_set_equals_web_sql(self, web_set_graph):
        """A natural_key seed gives exactly web_sql's rows from that node (matched false)."""
        g, ids = web_set_graph, web_set_graph["ids"]
        with _pg(g) as conn:
            for key, kind, *_ in SET_NODES:
                for depth in DEPTHS:
                    for top_n in (1, 25):  # top_n 1: ties at the cut (h_z / h_out from p_c1) break on node_id
                        web = conn.execute(web_sql(depth, g["served_schema"], top_n),
                                           {"start_node_id": ids[key]}).fetchall()
                        got = _set_walk(g, (NodePredicate(kind, "natural_key"), key), None, depth, top_n)
                        assert got == [(int(h), str(n), int(s), False) for h, n, s, _sc in web], (key, depth, top_n)

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_support_sums_across_seeds(self, web_set_graph):
        hop1 = {n: s for h, n, s, _m in _named_rows(web_set_graph, _set_walk(web_set_graph, CARDIO, None, 1))}
        assert hop1 == {"h_in": 3, "h_z": 2, "h_out": 1}

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_stop_node_is_not_expanded(self, web_set_graph):
        g = web_set_graph
        open_walk = {n for _h, n, _s, _m in _named_rows(g, _set_walk(g, CARDIO, None, 2))}
        assert "p_x" in open_walk  # non-vacuous: without the stop, X is reached through h_out
        for depth in DEPTHS:
            rows = _named_rows(g, _set_walk(g, CARDIO, OUT, depth))
            assert (1, "h_out", 1, True) in rows
            assert "p_x" not in {n for _h, n, _s, _m in rows}, depth

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_stop_requires_the_stop_type(self, web_set_graph):
        rows = _named_rows(web_set_graph, _set_walk(web_set_graph, CARDIO, OUT, 4))
        assert (2, "p_out", 1, False) in rows  # a provider with network_status out is not a stop
        assert (3, "h_far", 1, False) in rows  # ... so it expands
        # a seed that matches the stop predicate still expands
        seeded = _named_rows(web_set_graph, _set_walk(web_set_graph, OUT, OUT, 1))
        assert seeded == [(1, "p_x", 1, False)]

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_seed_type_is_enforced(self, web_set_graph):
        names = {n for _h, n, _s, _m in _named_rows(web_set_graph, _set_walk(web_set_graph, CARDIO, None, 4))}
        assert "p_trap" not in names and "h_card" not in names  # h_card is a hospital: never a seed
        assert "h_only_x" not in names  # 'Cardiology ' is not 'cardiology'
        assert {"h_z", "h_in", "p_mid"} <= names

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_seed_cap_raises_with_the_message(self, web_set_graph):
        import psycopg

        with pytest.raises(psycopg.errors.InvalidTextRepresentation,
                           match='invalid input syntax for type integer: "lgk: seed set exceeds max_seeds=3"'):
            _set_walk(web_set_graph, CARDIO, None, 1, max_seeds=3)
        assert len(_set_walk(web_set_graph, CARDIO, None, 1, max_seeds=4)) == 3  # 4 seeds fit a cap of 4

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_walk_key_index_is_used(self, web_set_graph):
        """Provision made the partial expression index; the seed scan uses it (enable_seqscan off).
        Reapply's ensure_indexes makes the same index on a fresh copy (in a rolled-back transaction)."""
        from lgk.mapping import load_mapping
        from lgk.steps.reapply import ensure_indexes

        g = web_set_graph
        version = version_schema(g)
        mapping = load_mapping(Path(g["run_folder"]) / "graph_mapping.yaml")
        conn = _pg(g)
        try:
            made = {r[0] for r in conn.execute(
                "SELECT indexname FROM pg_indexes WHERE schemaname = %s AND tablename = 'nodes'", (version,))}
            assert {"lgk_nodes_wk_provider_specialty_idx", "lgk_nodes_wk_hospital_network_status_idx"} <= made
            sql = web_set_sql(2, g["served_schema"], 25, CARDIO[0], walk_keys={"provider": ("specialty",)})
            assert "lgk_nodes_wk_provider_specialty_idx" in _plan_index_roots(conn, sql)
            conn.rollback()
            scratch = f"t13_reapply_{uuid.uuid4().hex[:6]}"
            conn.execute(f"CREATE SCHEMA {scratch}")
            conn.execute(f"CREATE TABLE {scratch}.nodes AS SELECT * FROM {version}.nodes")
            conn.execute(f"CREATE TABLE {scratch}.edges AS SELECT * FROM {version}.edges")
            ensure_indexes(conn, scratch, mapping)
            copy_sql = web_set_sql(2, scratch, 25, CARDIO[0], walk_keys={"provider": ("specialty",)},
                                   nodes_relation="nodes", edges_relation="edges")
            assert "lgk_nodes_wk_provider_specialty_idx" in _plan_index_roots(conn, copy_sql)
        finally:
            conn.rollback()  # the scratch schema never commits
            conn.close()


def _plan_index_roots(conn: Any, sql: str) -> set:
    """Every index the plan scans, mapped from a synced table's child partition index to its parent."""
    import json as _json

    conn.execute("SET LOCAL enable_seqscan = off")
    plan = conn.execute(f"EXPLAIN (VERBOSE, FORMAT JSON) {sql}", {"seed_value": "cardiology"}).fetchone()[0]
    plan = _json.loads(plan) if isinstance(plan, str) else plan
    names, stack = set(), [plan[0]["Plan"]]
    while stack:
        node = stack.pop()
        if "Index Name" in node:
            names.add(node["Index Name"])
        stack += node.get("Plans", [])
    roots = set()
    for name in names:
        row = conn.execute(
            "SELECT COALESCE(p.relname, c.relname) FROM pg_class c LEFT JOIN pg_inherits i ON i.inhrelid = c.oid "
            "LEFT JOIN pg_class p ON p.oid = i.inhparent WHERE c.relname = %s AND c.relkind IN ('i', 'I') "
            "LIMIT 1", (name,)).fetchone()
        roots.add(row[0] if row else name)
    return roots
