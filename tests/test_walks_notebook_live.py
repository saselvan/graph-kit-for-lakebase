"""Live acceptance for the walks notebook on a fresh branch of the test project.

The provider graph (a 30-link chain, two fan-outs and a hostile duplicate pair) is built and
provisioned through the shared `provisioned_graph` (privacy gate cleared the real way, same
teardown as the spine). The notebook is generated with ALL defaults and executed locally, cell
by cell, with a dbutils stub, against the branch (tests/notebook_support.py).

Each walk cell's answer must equal (a) the app's walk SQL (`get_walk_query`) run live on the
branch and (b) an independent reference: the Delta source edges (read in Spark) walked in plain
Python with the documented rules. The seed is changed through the widget AFTER generation, so a
notebook carrying a generation-time answer cannot pass.
"""

import contextlib
import io
from collections import Counter
from typing import Any, Dict, List

import pytest

from lgk.steps.walks import get_walk_query
from lgk.uc_sql import execute_sql
from lgk.walks.nb_runtime import jsonable
from lgk.walks.notebook import write_notebook
from tests.notebook_support import LocalNotebook
from tests.test_build_provision_walks import WAREHOUSE_ID, _create_provider_fixture, _pg, provisioned_graph
from tests.test_walks_live import _ref_edges

DEPTHS = (1, 2, 3, 4)
TOP_N = 25  # the notebook's default; never overridden here


@pytest.fixture(scope="module")
def nb_graph(workspace_client, lakebase_project, lakebase_catalog, run_folder):
    with provisioned_graph(workspace_client, lakebase_project, lakebase_catalog, run_folder,
                           "notebook", _create_provider_fixture) as config:
        cat, kit = config["lakebase_catalog"], config["kit_schema"]
        config["ids"] = {k: str(v) for k, v in execute_sql(workspace_client, WAREHOUSE_ID,
            f"SELECT natural_key, node_id FROM {cat}.{kit}.node_id_map")}
        adj: Dict[str, set] = {}
        for src, dst in execute_sql(workspace_client, WAREHOUSE_ID, _ref_edges(config)):
            adj.setdefault(str(src), set()).add(str(dst))
        config["adj"] = adj
        yield config


# ---------------------------------------------------------------------------- references
def ref_web(adj: Dict[str, set], seed: str, depth: int, top_n: int) -> List[List[Any]]:
    """Per hop: unseen out-neighbours of the previous hop, support = links in, ranked by
    (support desc, node_id bytes asc), cut at top_n. Rows [hop, node_id, support, None]."""
    seen, frontier, out = {seed}, [seed], []
    for hop in range(1, depth + 1):
        support = Counter(d for n in frontier for d in adj.get(n, ()) if d not in seen)
        ranked = sorted(support.items(), key=lambda kv: (-kv[1], kv[0].encode()))[:top_n]
        frontier = [n for n, _ in ranked]
        seen |= set(frontier)
        out += [[hop, n, s, None] for n, s in ranked]
    return out


def ref_tree(adj: Dict[str, set], seed: str, depth: int, top_n: int) -> List[List[Any]]:
    """Breadth-first minimum depth, seed excluded, ranked by (depth, node_id bytes), top_n."""
    dist, frontier = {seed: 0}, [seed]
    for d in range(1, depth + 1):
        frontier = sorted({x for n in frontier for x in adj.get(n, ()) if x not in dist})
        dist.update({x: d for x in frontier})
    ranked = sorted(((n, d) for n, d in dist.items() if d > 0), key=lambda r: (r[1], r[0].encode()))
    return [[n, d, None] for n, d in ranked[:top_n]]


def _worst_seed(adj: Dict[str, set]) -> str:
    return sorted(adj, key=lambda n: (-len(adj[n]), n.encode()))[0]


def _execute(path, widgets=None):
    book = LocalNotebook(path.read_text(), widgets)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = book.run_all()
    return result, out.getvalue()


def _app_answers(g: Dict[str, Any], seed: str) -> Dict[int, List[List[Any]]]:
    with _pg(g) as conn:
        return {d: jsonable(conn.execute(
            get_walk_query(seed, depth=d, served_schema=g["served_schema"], run_folder=g["run_folder"]),
            {"start_node_id": seed}).fetchall()) for d in DEPTHS}


@pytest.mark.seam1
@pytest.mark.provision
class TestWalksNotebookLive:
    def test_answers_equal_app_and_reference_and_rerun_identically(self, nb_graph):
        g, ids, adj = nb_graph, nb_graph["ids"], nb_graph["adj"]
        path = write_notebook(g)  # all defaults: web + tree, depths 1..4, top_n 25
        source = path.read_text()
        assert f"{g['kit_schema']}_v" not in source and "_synced" not in source

        # default seed (picked live, worst case), then widget overrides chosen AFTER generation:
        # a chain start (non-vacuous at every depth) and # hostile: a sink with no out-edges.
        cases = [({}, _worst_seed(adj)), ({"seed_node_id": ids["prov_000"]}, ids["prov_000"]),
                 ({"seed_node_id": ids["prov_030"]}, ids["prov_030"])]
        for widgets, seed in cases:
            result, out = _execute(path, widgets)
            assert result["seed"] == seed
            app = _app_answers(g, seed)
            for d in DEPTHS:
                web, tree = result["answers"][f"web:{d}"], result["answers"][f"tree:{d}"]
                assert web == app[d], f"web d{d} seed {seed}: notebook != app"
                assert web == ref_web(adj, seed, d, TOP_N), f"web d{d} seed {seed}: != reference"
                assert tree == ref_tree(adj, seed, d, TOP_N), f"tree d{d} seed {seed}: != reference"
            assert "eyJ" not in out, "a credential (JWT) was printed"
            assert out.count("plan: ") == 2 * len(DEPTHS) and out.count("this run: ") == 2 * len(DEPTHS)
            again, _ = _execute(path, widgets)  # a fresh second execution
            assert again == result

        chain = _execute(path, {"seed_node_id": ids["prov_000"]})[0]["answers"]
        assert [r[1] for r in chain["web:4"]] == [ids[f"prov_{i:03d}"] for i in (1, 2, 3, 4)]
        assert all(_execute(path, {"seed_node_id": ids["prov_030"]})[0]["answers"][k] == []
                   for k in ("web:1", "tree:4"))

    def test_rerunning_each_walk_cell_requeries_the_branch(self, nb_graph):
        g, ids = nb_graph, nb_graph["ids"]
        book = LocalNotebook(write_notebook(g).read_text(), {"seed_node_id": ids["prov_005"]})
        with contextlib.redirect_stdout(io.StringIO()):
            book.run_until_last()
            first = {k: list(v) for k, v in book.ns["ANSWERS"].items()}
            assert first["web:1"], "prov_005 has two out-neighbours"
            for i in book.walk_cells():
                book.ns["ANSWERS"].clear()
                book.run_cell(i)
                (key, rows), = book.ns["ANSWERS"].items()
                assert rows == first[key], key
            book.finish()
