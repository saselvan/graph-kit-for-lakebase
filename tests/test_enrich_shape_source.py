"""Offline (scripted fake executor over real edge rows, no SQL engine): closure eligibility comes
from the shape enrich MEASURES on the built edges, never from the `shape` propose recorded in
flags.json (persona 4: flags.json still said `shape: tree` after a co-membership fold made the
built graph cyclic).

`shape` in flags.json is propose's guess from the table-level draft, before build and before
any fold edge. The enrich menu shows the measured shape and labels the recorded one as that guess.
"""

import json

import pytest

from lgk.steps.enrich import run_enrich
from tests.enrich_fake import EdgeFakeSql

AFFIL = "affiliations"
FOLD = "provider_comembership"

# A real tree: two hospitals over three providers (every edge hospital -> provider).
TREE_EDGES = [("h1", "p1", AFFIL), ("h1", "p2", AFFIL), ("h2", "p3", AFFIL)]

CYCLIC_GRAPHS = {
    # hostile: the folded co-membership edge both ways (p1 <-> p2) is the ONLY cycle; every
    # unfolded edge alone is a tree, which is what propose saw when it wrote `shape: tree`
    "folded_both_ways": TREE_EDGES + [("p1", "p2", FOLD), ("p2", "p1", FOLD)],
    # hostile: a 3-cycle behind a tree tail (Kahn peels h1 -> a, then gets stuck on a -> b -> c -> a)
    "three_cycle": [("h1", "a", AFFIL), ("a", "b", AFFIL), ("b", "c", AFFIL), ("c", "a", AFFIL)],
}

# hostile: a diamond chain (fan-out then fan-in, twice) is a DAG but not a tree: a checker that
# confuses "two parents" with "a cycle" refuses it. Hand-counted closure: a reaches 6, b and c 4
# each, d 3, e and f 1 each = 19 (ancestor, descendant) pairs.
DIAMOND_CHAIN = [("a", "b", AFFIL), ("a", "c", AFFIL), ("b", "d", AFFIL), ("c", "d", AFFIL),
                 ("d", "e", AFFIL), ("d", "f", AFFIL), ("e", "g", AFFIL), ("f", "g", AFFIL)]
DIAMOND_CLOSURE_PAIRS = 19


def _run_folder(tmp_path, recorded_shape, methods=None):
    (tmp_path / "flags.json").write_text(json.dumps({"shape": recorded_shape, "flags": []}))
    if methods is not None:
        (tmp_path / "enrich.yaml").write_text(f"methods: [{', '.join(methods)}]\n")
    return tmp_path


def _closure_row(report):
    return next(r for r in report["menu"] if r["method"] == "closure")


@pytest.mark.parametrize("name", sorted(CYCLIC_GRAPHS))
def test_recorded_tree_with_cyclic_edges_greys_closure_in_the_menu(tmp_path, name):
    folder = _run_folder(tmp_path, "tree")  # hostile: the stale record says tree
    with pytest.raises(FileNotFoundError):
        run_enrich(EdgeFakeSql(CYCLIC_GRAPHS[name]), "c.s", folder)
    menu = (folder / "enrich_menu.md").read_text()
    closure = next(line for line in menu.splitlines() if line.startswith("| closure |"))
    assert closure.endswith("| greyed out: the graph has a cycle; closure is only for trees and DAGs |")
    assert "Measured on the built nodes and edges now: has a cycle" in menu


@pytest.mark.parametrize("name", sorted(CYCLIC_GRAPHS))
def test_recorded_tree_with_cyclic_edges_refuses_closure_twice(tmp_path, name):
    folder = _run_folder(tmp_path, "tree", ["closure"])
    sql = EdgeFakeSql(CYCLIC_GRAPHS[name])
    for _ in range(2):  # through the runner twice: the refusal holds on re-run
        with pytest.raises(ValueError, match="closure: the graph has a cycle"):
            run_enrich(sql, "c.s", folder)
    assert "c.s.closure" not in sql.tables
    assert not (folder / "enrich_report.json").exists()


def test_recorded_web_with_a_diamond_chain_offers_and_computes_closure(tmp_path):
    folder = _run_folder(tmp_path, "web", ["closure"])  # hostile: the stale record says cyclic
    sql = EdgeFakeSql(DIAMOND_CHAIN)
    report = run_enrich(sql, "c.s", folder)
    row = _closure_row(report)
    assert (row["available"], row["reason"]) == (True, "")
    assert report["shape"]["is_dag"] is True and report["shape"]["too_deep"] is False
    assert sql.pairs["c.s.closure"] == {
        ("a", "b"), ("a", "c"), ("a", "d"), ("a", "e"), ("a", "f"), ("a", "g"),
        ("b", "d"), ("b", "e"), ("b", "f"), ("b", "g"), ("c", "d"), ("c", "e"), ("c", "f"), ("c", "g"),
        ("d", "e"), ("d", "f"), ("d", "g"), ("e", "g"), ("f", "g")}
    assert report["closure"]["rows"] == DIAMOND_CLOSURE_PAIRS and report["closure"]["max_depth"] == 4


def test_recorded_web_with_a_diamond_chain_menu_says_closure_available(tmp_path):
    folder = _run_folder(tmp_path, "web")
    with pytest.raises(FileNotFoundError):
        run_enrich(EdgeFakeSql(DIAMOND_CHAIN), "c.s", folder)
    menu = (folder / "enrich_menu.md").read_text()
    closure = next(line for line in menu.splitlines() if line.startswith("| closure |"))
    assert closure.endswith("| closure | available |")
    assert "Measured on the built nodes and edges now: no cycle (a tree or DAG)" in menu


@pytest.mark.parametrize("recorded, edges", [("tree", CYCLIC_GRAPHS["folded_both_ways"]),
                                             ("web", DIAMOND_CHAIN)])
def test_menu_labels_the_recorded_shape_as_proposes_guess(tmp_path, recorded, edges):
    folder = _run_folder(tmp_path, recorded)
    with pytest.raises(FileNotFoundError):
        run_enrich(EdgeFakeSql(edges), "c.s", folder)
    menu = (folder / "enrich_menu.md").read_text()
    assert (f"flags.json `shape: {recorded}` is propose's guess from the table-level draft, made "
            "before build and before any fold edge; this menu does not use it.") in menu


def test_menu_without_a_recorded_shape_has_no_guess_line(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_enrich(EdgeFakeSql(DIAMOND_CHAIN), "c.s", tmp_path)  # no flags.json at all
    menu = (tmp_path / "enrich_menu.md").read_text()
    assert "Measured on the built nodes and edges now" in menu and "guess" not in menu


def test_a_query_that_skips_an_edge_type_is_not_answered_by_the_fake():
    """The fake refuses any edges read it does not know, so a shape check that filtered out the
    folded type would fail loudly instead of passing on a smaller graph."""
    sql = EdgeFakeSql(CYCLIC_GRAPHS["folded_both_ways"])
    with pytest.raises(AssertionError, match="unrecognised read of the edges"):
        sql(f"CREATE OR REPLACE TABLE c.s._x AS SELECT DISTINCT src_node_id AS src, dst_node_id AS dst "
            f"FROM c.s.edges WHERE src_node_id <> dst_node_id AND relationship_type <> '{FOLD}'")


def _menu_for(tmp_path, edges, flags):
    if flags is not None:
        (tmp_path / "flags.json").write_text(json.dumps(flags))
    with pytest.raises(FileNotFoundError):
        run_enrich(EdgeFakeSql(edges), "c.s", tmp_path)
    return (tmp_path / "enrich_menu.md").read_text()


def test_diamond_chain_header_states_the_measured_counts(tmp_path):
    menu = _menu_for(tmp_path, DIAMOND_CHAIN, {"shape": "web"})
    assert menu.splitlines()[0] == ("Measured on the built nodes and edges now: no cycle (a tree or DAG); "
                                    "7 nodes, 8 directed edges (every edge type, folded ones included).")


def test_noisy_diamond_chain_still_offers_closure_with_19_pairs(tmp_path):
    # hostile: a self-loop and a duplicate edge in the raw rows must not count as a cycle or an edge
    noisy = DIAMOND_CHAIN + [("d", "d", AFFIL), ("a", "b", AFFIL)]
    folder = _run_folder(tmp_path, "web", ["closure"])
    sql = EdgeFakeSql(noisy)
    report = run_enrich(sql, "c.s", folder)
    assert report["shape"]["n_edges"] == 8 and report["shape"]["is_dag"] is True
    assert report["closure"]["rows"] == DIAMOND_CLOSURE_PAIRS == len(sql.pairs["c.s.closure"])


def test_too_deep_chain_is_labelled_deep_not_cyclic(tmp_path):
    # hostile: a 101-edge chain (one past MAX_DEPTH) under a recorded `shape: tree`
    chain = [(f"v{i}", f"v{i + 1}", AFFIL) for i in range(101)]
    menu = _menu_for(tmp_path, chain, {"shape": "tree"})
    assert "Measured on the built nodes and edges now: no cycle found, but deeper than 100 levels" in menu
    assert "has a cycle" not in menu
    closure = next(line for line in menu.splitlines() if line.startswith("| closure |"))
    assert closure.endswith("| greyed out: the graph is deeper than 100 levels, the closure depth cap |")


def test_a_hub_with_fifty_children_is_a_dag(tmp_path):
    hub = [("h", f"c{i}", AFFIL) for i in range(50)]  # hostile: one node with 50 children
    menu = _menu_for(tmp_path, hub, {"shape": "web"})
    assert "no cycle (a tree or DAG); 51 nodes, 50 directed edges" in menu


@pytest.mark.parametrize("flags", [{"flags": []}, {"shape": {"kind": "tree"}}, {"shape": 3},
                                   {"shape": None}])
def test_no_or_non_string_recorded_shape_gives_no_guess_line(tmp_path, flags):
    # hostile: flags.json without `shape`, or with a shape that is not a string
    menu = _menu_for(tmp_path, DIAMOND_CHAIN, flags)
    assert "guess" not in menu and "`shape:" not in menu
