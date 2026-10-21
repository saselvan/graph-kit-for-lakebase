"""Live tests for the enrich step (Spark batch on a UC warehouse, no Lakebase branch).

Two graphs are built with the real `build` step, each in its own per-run `lgk_v1_<hex>`
schema, then enriched with the real `enrich` step:

- GATE graph (cyclic): a super-connector star (12 leaves), two exact-twin stars
  (11 leaves each: the near-tie runner-ups), a disconnected pair, an isolated node, and
  hostile reverse-direction duplicate edges on twin_a (same undirected pair twice).
- TREE graph (DAG): a diamond plus a shortcut edge, for transitive closure.

Every node is identified by natural key through node_id_map, never by a Python hash.
Teardown reuses the shared `_teardown_live` (no branch: it only drops UC schemas).
"""

import json
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from lgk.uc_sql import execute_sql
from tests.enrich_reference import (
    CONVERGING_EDGES,
    CONVERGING_NODES,
    GATE_COMPONENTS,
    GATE_EDGES,
    GATE_NODES,
    ISLAND,
    SUPER,
    TREE_CLOSURE,
    TREE_EDGES,
    TREE_NODES,
    TWIN_A,
    TWIN_B,
    reference_pagerank,
)
from tests.test_build_provision_walks import (
    WAREHOUSE_ID,
    _teardown_live,
    _write_provider_graph_mapping,
)

def _q(w: Any, sql: str) -> List[List[Any]]:
    return execute_sql(w, WAREHOUSE_ID, sql)


def _create_fixture(w: Any, catalog: str, schema: str, nodes: List[str], edges) -> None:
    fq = f"{catalog}.{schema}"
    _q(w, f"CREATE SCHEMA IF NOT EXISTS {fq}")
    _q(w, f"CREATE OR REPLACE TABLE {fq}.fixture_providers "
          "(provider_id STRING, name STRING, specialty STRING) USING DELTA")
    _q(w, f"INSERT INTO {fq}.fixture_providers VALUES "
          + ", ".join(f"('{k}', 'Node {k}', 'x')" for k in nodes))
    _q(w, f"CREATE OR REPLACE TABLE {fq}.fixture_affiliations "
          "(affiliation_id STRING, provider_id STRING, hospital_id STRING, start_date STRING) USING DELTA")
    _q(w, f"INSERT INTO {fq}.fixture_affiliations VALUES "
          + ", ".join(f"('e{i}', '{s}', '{d}', '2024-01-01')" for i, (s, d) in enumerate(edges)))


def _make_graph(w, catalog, tmp: Path, name, nodes, edges, methods, created: List) -> Dict[str, Any]:
    """Build + enrich one graph. Appends its config to `created` first, so teardown covers it."""
    from lgk.steps.build import build
    from lgk.steps.enrich import enrich

    suffix = uuid.uuid4().hex[:6]
    config: Dict[str, Any] = {
        "graph_name": name,
        "lakebase_catalog": catalog,
        "kit_schema": f"lgk_v1_{suffix}",
        "fixture_schema": f"lgk_v1_fx_{suffix}",
        "served_schema": f"lgk_served_{suffix}",
        "warehouse_id": WAREHOUSE_ID,
        "run_folder": str(tmp / name),
    }
    created.append(config)
    Path(config["run_folder"]).mkdir(parents=True)
    _create_fixture(w, catalog, config["fixture_schema"], nodes, edges)
    _write_provider_graph_mapping(
        Path(config["run_folder"]) / "graph_mapping.yaml", catalog, config["fixture_schema"]
    )
    from tests.privacy_support import clear_gate_live

    # start_date holds date-shaped strings: a care-subject value pattern. The mapping
    # does not carry it; the operator drops it (clear_gate_live asserts it really was flagged).
    clear_gate_live(config, w, drops=[("fixture_affiliations", "start_date")])
    build(config)
    (Path(config["run_folder"]) / "enrich.yaml").write_text(yaml.safe_dump({"methods": methods}))
    enrich(config)
    return config


@pytest.fixture(scope="module")
def live_graphs(workspace_client, lakebase_catalog):
    configs: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            gate = _make_graph(workspace_client, lakebase_catalog, Path(tmpdir), "gate",
                               GATE_NODES, GATE_EDGES,
                               ["similarity", "components", "degree", "pagerank"],  # reordered
                               configs)
            tree = _make_graph(workspace_client, lakebase_catalog, Path(tmpdir), "tree",
                               TREE_NODES, TREE_EDGES, ["closure"], configs)
            conv = _make_graph(workspace_client, lakebase_catalog, Path(tmpdir), "conv",
                               CONVERGING_NODES, CONVERGING_EDGES, ["pagerank"], configs)
            yield {"gate": gate, "tree": tree, "conv": conv}
        finally:
            for c in configs:
                _teardown_live(workspace_client, None, c)


def _t(config: Dict[str, Any]) -> str:
    return f"{config['lakebase_catalog']}.{config['kit_schema']}"


def _by_key(w, config, table: str, cols: str, where: str = "") -> List[List[Any]]:
    return _q(w, f"SELECT {cols} FROM {_t(config)}.{table} x {where}")


def _keys(w, config) -> Dict[str, str]:
    return {str(i): k for i, k in _q(w, f"SELECT node_id, natural_key FROM {_t(config)}.node_id_map")}


@pytest.mark.seam1
class TestPageRankGate:
    def test_super_connector_ranks_first_and_twins_tie_by_natural_key(self, live_graphs, workspace_client):
        """PageRank gate. Top 3 = super-connector, then the exact twins in natural-key order."""
        from lgk.enrich.ranking import top_nodes

        g = live_graphs["gate"]
        top = top_nodes(lambda s: _q(workspace_client, s), _t(g), "pagerank", 3)
        assert [k for _t_, k, _s in top] == [SUPER, TWIN_A, TWIN_B], top
        # The twins are a real tie (within float noise), so tie handling decides 2nd/3rd.
        assert abs(float(top[1][2]) - float(top[2][2])) < 1e-9
        assert float(top[0][2]) - float(top[1][2]) > 1e-4  # super-connector is NOT a tie

    def test_pagerank_scores_every_node_sum_one_and_respects_r15(self, live_graphs, workspace_client):
        g = live_graphs["gate"]
        n, total = _q(workspace_client,
            f"SELECT COUNT(*), SUM(score) FROM {_t(g)}.node_score WHERE method = 'pagerank'")[0]
        assert int(n) == len(GATE_NODES)  # the isolated node is scored too
        assert abs(float(total) - 1.0) < 1e-6
        rep = json.loads((Path(g["run_folder"]) / "enrich_report.json").read_text())["pagerank"]
        # The gate graph is all stars (bipartite), so power iteration oscillates and decays
        # only at rate d: the L1 change after 50 steps is about 4.5e-4. So R15's cap is hit.
        assert (rep["iterations"], rep["converged"]) == (50, False), rep
        assert rep["final_l1"] >= 1e-6

    def test_pagerank_matches_independent_reference(self, live_graphs, workspace_client, record_property):
        """Every node's score vs a pure-Python reference (test code, not src).

        Two checks, because a capped run on a bipartite graph cannot meet 1e-5 against the
        fixed point (measured: max error 3.6e-5 after 50 steps):
        (a) equal, within 1e-9, to the reference run for the same 50 steps from 1/N;
        (b) within 1e-4 of the reference fixed point (L1 < 1e-12). A 1-step stop is 0.10 off
            and damping 0.5 is 0.036 off, so both are far outside this bound.
        """
        g = live_graphs["gate"]
        keys = _keys(workspace_client, g)
        got = {keys[str(i)]: float(s) for i, s in _by_key(
            workspace_client, g, "node_score", "node_id, score", "WHERE method = 'pagerank'")}
        stepped = reference_pagerank(GATE_NODES, GATE_EDGES, iterations=50)
        fixed = reference_pagerank(GATE_NODES, GATE_EDGES)
        assert set(got) == set(GATE_NODES)
        # The live leg's measured differences, for the public "matches networkx" tolerance.
        diff_stepped = max(abs(got[n] - stepped[n]) for n in GATE_NODES)
        diff_fixed = max(abs(got[n] - fixed[n]) for n in GATE_NODES)
        record_property("max_abs_diff_gate_50_steps", diff_stepped)
        record_property("max_abs_diff_gate_fixed_point", diff_fixed)
        assert diff_stepped < 1e-9
        assert diff_fixed < 1e-4


    def test_pagerank_converges_to_the_reference_fixed_point(self, live_graphs, workspace_client, record_property):
        """The shipped SQL on the warehouse vs the pure-Python reference (test code, not src).

        The fixture (triangle, reversed duplicate pair, self-loop, isolated node) is not
        bipartite, so it converges inside the cap: this pins damping, the L1 stop rule (a
        1-step stop or an L1 of 0 ends at iteration 1), dangling mass, teleport and DISTINCT pairs.
        """
        g = live_graphs["conv"]
        rep = json.loads((Path(g["run_folder"]) / "enrich_report.json").read_text())["pagerank"]
        assert rep["converged"] and 1 < rep["iterations"] < 50, rep
        keys = _keys(workspace_client, g)
        got = {keys[str(i)]: float(s) for i, s in _by_key(
            workspace_client, g, "node_score", "node_id, score", "WHERE method = 'pagerank'")}
        ref = reference_pagerank(CONVERGING_NODES, CONVERGING_EDGES)
        assert set(got) == set(ref)
        diff = max(abs(got[n] - ref[n]) for n in ref)
        record_property("max_abs_diff_converging", diff)  # the live leg's number for the public tolerance
        assert diff < 1e-6, (got, ref)


@pytest.mark.seam1
class TestOtherMethods:
    def test_degree_counts_distinct_neighbours(self, live_graphs, workspace_client):
        g = live_graphs["gate"]
        keys = _keys(workspace_client, g)
        deg = {keys[str(i)]: float(s) for i, s in _by_key(
            workspace_client, g, "node_score", "node_id, score", "WHERE method = 'degree'")}
        assert len(deg) == len(GATE_NODES)
        assert (deg[SUPER], deg[TWIN_A], deg[TWIN_B], deg[ISLAND], deg["l_a01"]) == (12, 11, 11, 0, 1)

    def test_components_partition(self, live_graphs, workspace_client):
        g = live_graphs["gate"]
        keys = _keys(workspace_client, g)
        groups: Dict[str, set] = {}
        for node_id, gid in _by_key(workspace_client, g, "node_group", "node_id, group_id",
                                    "WHERE method = 'components'"):
            groups.setdefault(str(gid), set()).add(keys[str(node_id)])
        assert sorted(groups.values(), key=sorted) == sorted(GATE_COMPONENTS, key=sorted)

    def test_similarity_shared_neighbour_jaccard(self, live_graphs, workspace_client):
        g = live_graphs["gate"]
        keys = _keys(workspace_client, g)
        sim = {(keys[str(s)], keys[str(d)]): float(v)
               for s, d, v in _by_key(workspace_client, g, "similar_to", "src_id, dst_id, score")}
        assert sim[("l_a01", "l_a02")] == 1.0 and sim[("l_a02", "l_a01")] == 1.0
        assert not any(a == b for a, b in sim)  # no self-similarity
        assert not any(SUPER in pair or TWIN_A in pair for pair in sim)  # hubs share no neighbour
        # Every pair is within one component (no cross-star similarity).
        assert not any(a.startswith("l_s") and b.startswith("l_a") for a, b in sim)

    def test_closure_on_dag_min_depth(self, live_graphs, workspace_client):
        t = live_graphs["tree"]
        keys = _keys(workspace_client, t)
        rows = {(keys[str(a)], keys[str(d)], int(dep)) for a, d, dep in _by_key(
            workspace_client, t, "closure", "ancestor_id, descendant_id, depth")}
        assert rows == TREE_CLOSURE


@pytest.mark.seam1
class TestShapeAndMenuLive:
    def test_measured_shapes_grey_out_closure_on_cycle_only(self, live_graphs, workspace_client):
        from lgk.enrich.menu import build_menu
        from lgk.enrich.shape import measure_shape

        run = lambda s: _q(workspace_client, s)  # noqa: E731
        gate = measure_shape(run, _t(live_graphs["gate"]))
        tree = measure_shape(run, _t(live_graphs["tree"]))
        assert (gate.n_nodes, gate.n_edges, gate.max_degree, gate.is_dag, gate.example_key) == (
            len(GATE_NODES), len(GATE_EDGES), 12, False, SUPER)
        assert (tree.n_nodes, tree.n_edges, tree.is_dag) == (5, 6, True)
        gate_rows = {r.method: r for r in build_menu(gate)}
        assert not gate_rows["closure"].available and "cycle" in gate_rows["closure"].reason
        assert {r.method for r in build_menu(tree) if r.available} == {
            "pagerank", "degree", "components", "similarity", "closure"}

    def test_enrichment_tables_match_fixed_shapes(self, live_graphs, workspace_client):
        expected = {
            "node_score": [("node_id", "string"), ("method", "string"), ("score", "double"),
                           ("computed_at", "timestamp")],
            "node_group": [("node_id", "string"), ("method", "string"), ("group_id", "string")],
            "similar_to": [("src_id", "string"), ("dst_id", "string"), ("score", "double")],
        }
        for table, cols in expected.items():
            got = [(r[0], r[1]) for r in _q(workspace_client, f"DESCRIBE TABLE {_t(live_graphs['gate'])}.{table}")
                   if r[0] and not r[0].startswith("#")]
            assert got == cols, (table, got)
        got = [(r[0], r[1]) for r in _q(workspace_client, f"DESCRIBE TABLE {_t(live_graphs['tree'])}.closure")
               if r[0] and not r[0].startswith("#")]
        assert got == [("ancestor_id", "string"), ("descendant_id", "string"), ("depth", "int")]

    def test_no_work_tables_left_behind(self, live_graphs, workspace_client):
        for g in live_graphs.values():
            names = [r[1] for r in _q(workspace_client, f"SHOW TABLES IN {_t(g)}")]
            assert not [n for n in names if n.startswith("_enrich")], names


@pytest.mark.seam1
def test_whole_graph_teardown_of_an_enriched_graph_leaves_nothing(workspace_client, lakebase_catalog, tmp_path):
    """Enrich records its output tables, so teardown drops the whole kit schema.

    Every v1 method runs (the tree is a DAG, so closure too): if any output table or `_enrich_*`
    work table were unrecorded, teardown would refuse the kit schema and delete nothing.
    """
    from databricks.sdk.errors import NotFound

    from lgk.steps.teardown import teardown

    created: List[Dict[str, Any]] = []
    try:
        g = _make_graph(workspace_client, lakebase_catalog, tmp_path, "td_enriched", TREE_NODES, TREE_EDGES,
                        ["closure", "similarity", "components", "degree", "pagerank"], created)
        kit = _t(g)
        before = {r[1] for r in _q(workspace_client, f"SHOW TABLES IN {kit}")}
        assert {"node_score", "node_group", "similar_to", "closure"} <= before, before

        plan = teardown(g, "td_enriched")

        assert plan.uc_schemas == [kit]
        assert set(plan.uc_tables) == {f"{kit}.{t}" for t in before}
        with pytest.raises(NotFound):
            workspace_client.schemas.get(kit)
    finally:
        for c in created:
            _teardown_live(workspace_client, None, c)
