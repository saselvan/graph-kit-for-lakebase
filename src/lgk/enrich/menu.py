"""The fixed enrichment menu: one row per method, greyed out when it does not fit the graph."""

from dataclasses import dataclass
from typing import Callable, List, Optional

from lgk.enrich import shape as _shape
from lgk.enrich.shape import GraphShape

V1_METHODS = ("pagerank", "degree", "components", "similarity", "closure")

# The menu shows no build-time number: a fixed per-statement / per-edge model was about 6x too low
# for closure on a live 10^5-node run, so any number shown would mislead the operator.


@dataclass(frozen=True)
class MenuRow:
    method: str
    question: str
    example: str
    table: str
    available: bool
    reason: str  # why the row is greyed out; "" when available


@dataclass(frozen=True)
class _Spec:
    method: str
    question: str
    example: str  # formatted with {key}: a real node of the user's graph
    table: str
    unfit: Optional[Callable[[GraphShape], str]]  # returns a reason, or "" if it fits
    coming: bool = False


def _needs_edges(shape: GraphShape) -> str:
    return "" if shape.n_edges > 0 else "the graph has no edges"


def _unfit_similarity(shape: GraphShape) -> str:
    if shape.n_edges == 0:
        return "the graph has no edges"
    if shape.max_degree < 2:
        return "no node has two neighbours, so no two nodes share a neighbour"
    return ""


def _unfit_closure(shape: GraphShape) -> str:
    if shape.n_edges == 0:
        return "the graph has no edges"
    if shape.too_deep:
        return f"the graph is deeper than {_shape.MAX_DEPTH} levels, the closure depth cap"
    if not shape.is_dag:
        return "the graph has a cycle; closure is only for trees and DAGs"
    return ""


_MENU: List[_Spec] = [
    _Spec("pagerank", "Which nodes are the most central in the whole graph?",
          "where {key} ranks among all nodes", "node_score", _needs_edges),
    _Spec("degree", "How many direct neighbours does each node have?",
          "the number of neighbours of {key}", "node_score",
          lambda s: "" if s.n_nodes > 0 else "the graph has no nodes"),
    _Spec("components", "Which nodes are connected to each other at all?",
          "every node reachable from {key}", "node_group", _needs_edges),
    _Spec("similarity", "Which nodes share the most neighbours with this one?",
          "the nodes most like {key} by shared neighbours", "similar_to",
          _unfit_similarity),
    _Spec("closure", "What is everything below this node, at any depth?",
          "every descendant of {key} and its depth", "closure", _unfit_closure),
    _Spec("personalized_pagerank", "Which nodes matter most from this starting node?",
          "", "seed_score", None, coming=True),
    _Spec("community_detection", "Which tight clusters does the graph split into?",
          "", "node_group", None, coming=True),
    _Spec("embeddings", "Which nodes are alike by their features?",
          "", "similar_to", None, coming=True),
]


def _row(spec: _Spec, shape: GraphShape) -> MenuRow:
    if spec.coming:
        return MenuRow(spec.method, spec.question, "", spec.table, False, "coming in a later version")
    reason = spec.unfit(shape)
    return MenuRow(spec.method, spec.question, spec.example.format(key=shape.example_key),
                   spec.table, reason == "", reason)


def build_menu(shape: GraphShape) -> List[MenuRow]:
    """The full fixed menu for this graph, in menu order."""
    return [_row(spec, shape) for spec in _MENU]


def select_methods(menu: List[MenuRow], requested: List[str]) -> List[str]:
    """The requested methods in menu order; raise ValueError for any greyed or unknown one."""
    rows = {row.method: row for row in menu}
    wanted = set(requested)
    unknown = sorted(wanted - set(rows))
    if unknown:
        raise ValueError(f"not on the menu: {', '.join(unknown)}")
    refused = [f"{m}: {rows[m].reason}" for m in sorted(wanted) if not rows[m].available]
    if refused:
        raise ValueError("greyed out on this graph -> " + "; ".join(refused))
    return [row.method for row in menu if row.method in wanted]
