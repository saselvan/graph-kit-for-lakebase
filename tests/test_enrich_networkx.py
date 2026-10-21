"""The networkx leg of the enrich proof (offline): the shared references vs networkx.

The live leg (tests/test_enrich_live.py) proves the shipped SQL equals the references on the
warehouse. This leg proves the references equal networkx on the same graphs. Both legs read the
graphs and references from tests/enrich_reference.py, and both use the undirected graph with
self-loops and duplicate pairs dropped, as the kit's enrich methods do.

The public tolerance for "matches networkx" is the sum of the two legs' measured maximum
differences; this leg records its part as the `max_abs_diff_<graph>` property.
"""

import ast
from pathlib import Path

import networkx as nx
import pytest

import tests.enrich_reference as ref

TESTS = Path(__file__).resolve().parent
NX_TOL = 1e-9  # reference (L1 < 1e-12) vs networkx (tol 1e-14 per node); measured diffs are far below
GRAPHS = {
    "gate": (ref.GATE_NODES, ref.GATE_EDGES),
    "converging": (ref.CONVERGING_NODES, ref.CONVERGING_EDGES),
    "tree": (ref.TREE_NODES, ref.TREE_EDGES),
}
# The files that read the shared references (the two legs and the former home of reference_pagerank).
LEG_FILES = ("test_enrich_live.py", "test_enrich_networkx.py", "test_enrich_pagerank_loop.py")
# The names that live only in enrich_reference.py; a second definition in a leg file is a copy.
SHARED = ("reference_pagerank", "GATE_NODES", "GATE_EDGES", "GATE_COMPONENTS", "TREE_NODES",
          "TREE_EDGES", "TREE_CLOSURE", "CONVERGING_NODES", "CONVERGING_EDGES")


def _nx_graph(nodes, edges) -> nx.Graph:
    """The undirected graph networkx sees: every node (isolated ones too), no self-loops."""
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from((a, b) for a, b in edges if a != b)
    return g


def _max_diff(nodes, edges, nx_edges=None) -> float:
    ours = ref.reference_pagerank(nodes, edges)
    theirs = nx.pagerank(_nx_graph(nodes, edges if nx_edges is None else nx_edges),
                         alpha=0.85, tol=1e-14, max_iter=100_000)
    assert set(ours) == set(theirs) == set(nodes)
    return max(abs(ours[n] - theirs[n]) for n in nodes)


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_reference_pagerank_matches_networkx_on_the_shared_fixture_graphs(name, record_property):
    nodes, edges = GRAPHS[name]
    diff = _max_diff(nodes, edges)
    record_property(f"max_abs_diff_{name}", diff)
    assert diff <= NX_TOL, (name, diff)


def test_reference_components_match_networkx_on_the_shared_fixture_graphs():
    key = [frozenset(c) for c in ref.GATE_COMPONENTS]
    # The answer key is a partition of the gate graph's nodes: no node twice, none missing.
    assert sum(len(c) for c in key) == len(set(ref.GATE_NODES)) == len(ref.GATE_NODES)
    assert frozenset().union(*key) == frozenset(ref.GATE_NODES)
    got = {frozenset(c) for c in nx.connected_components(_nx_graph(ref.GATE_NODES, ref.GATE_EDGES))}
    assert got == set(key)


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_one_edge_removed_fails_the_tolerance(name):
    """Negative control: networkx on the graph minus one undirected pair is outside NX_TOL."""
    nodes, edges = GRAPHS[name]
    drop = next((a, b) for a, b in edges if a != b)
    fewer = [(a, b) for a, b in edges if {a, b} != {drop[0], drop[1]}]
    assert len(fewer) < len(edges)
    assert _max_diff(nodes, edges, nx_edges=fewer) > NX_TOL


def test_the_references_and_graphs_live_in_enrich_reference_only():
    """test_enrich_live and this leg read one module: the same objects, and no second definition."""
    import tests.test_enrich_live as live

    for name in SHARED:
        if hasattr(live, name):
            assert getattr(live, name) is getattr(ref, name), name
    assert live.reference_pagerank is ref.reference_pagerank
    assert live.GATE_COMPONENTS is ref.GATE_COMPONENTS
    copies = []
    for path in sorted(TESTS.rglob("*.py")):
        if path.name == "enrich_reference.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "reference_pagerank":
                copies.append(f"{path.name}:{node.lineno} def {node.name}")
            if path.name not in LEG_FILES:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (
                [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name) and n.id in SHARED:
                        copies.append(f"{path.name}:{node.lineno} {n.id} =")
    assert copies == []


def test_the_live_components_test_compares_to_the_answer_key():
    """The live components check uses GATE_COMPONENTS, not an inline set literal of its own."""
    tree = ast.parse((TESTS / "test_enrich_live.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "test_components_partition")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert "GATE_COMPONENTS" in names
    assert not any(isinstance(n, (ast.Set, ast.SetComp)) for n in ast.walk(fn))
