"""The C10 reference: walk answers computed from the SOURCE tables, never from the kit's tables.

It reads only `<source_catalog>.<source_schema>` (graph_mapping.yaml) on the SQL warehouse,
so it shares none of build's data: not `nodes`, not `edges`, not `node_id_map`. What it does
share with build is the documented contract, written here a second time on purpose:

- a node is one distinct non-NULL natural key of its node type; its id is
  `xxhash64('<type>::' || CAST(<key> AS STRING))` (Spark's own function, on the source row);
- an unfolded edge is one distinct (src, dst) pair of its edge type whose two ends are nodes
  (an orphan or NULL end is dropped);
- a folded edge is a (src, dst) pair weighted by its DISTINCT care subjects (co-membership:
  members sharing a subject, both directions; pair form: GROUP BY src, dst), kept only at or
  above the privacy floor. The weight is a count; no subject value leaves the warehouse.

The warehouse only extracts ids (and fold weights). Dedup, orphan and NULL dropping, the floor,
and every walk (web hops, tree levels, Jaccard similarity) are computed here in Python, so a
build bug in any of them shows as a difference instead of being shared.

Node types with a composite natural key are not loaded: no edge can reach them (the mapping
contract refuses it), so they never appear in a walk answer.

Every statement starts with a `/* lgk-measure:<kind>:<name> */` tag (kind nodes|edges).
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from lgk import ident
from lgk.mapping import EdgeSpec, GraphMapping

Sql = Callable[[str], List[List[Any]]]
EdgeKey = Tuple[str, str, str]  # (edge type, src node id, dst node id)

DEFAULT_MAX_ROWS = 5_000_000  # source rows pulled into this process (ids only); above it: refused
SCORE_DIGITS = 9  # similarity scores compare rounded (double division in Spark vs Postgres double)


class ReferenceTooLarge(RuntimeError):
    """The source has more rows than the reference may load (measure_reference_max_rows)."""


@dataclass(frozen=True)
class SourceGraph:
    """The graph the source tables define: node ids, and edges -> weight (None when unfolded)."""

    nodes: frozenset
    edges: Dict[EdgeKey, Optional[int]] = field(default_factory=dict)

    def out_edges(self) -> Dict[str, List[Tuple[str, Optional[int]]]]:
        """src -> [(dst, weight)], one entry per (edge type, src, dst)."""
        adj: Dict[str, List[Tuple[str, Optional[int]]]] = defaultdict(list)
        for (_t, src, dst), weight in self.edges.items():
            adj[src].append((dst, weight))
        return adj


# ---------------------------------------------------------------------------- SQL (ids only)


def _tag(kind: str, name: str) -> str:
    return f"/* lgk-measure:{kind}:{name} */ "


def _id(node_type: str, column: str) -> str:
    return f"CAST(xxhash64('{ident.check(node_type, 'node type')}::' || CAST({column} AS STRING)) AS STRING)"


def nodes_sql(src_fq: str, node_type: str, table: str, key: str) -> str:
    k = f"n.{ident.check(key, 'natural key')}"
    return (f"{_tag('nodes', node_type)}SELECT DISTINCT {_id(node_type, k)} "
            f"FROM {src_fq}.{ident.check(table, 'source table')} n WHERE {k} IS NOT NULL")


def edges_sql(src_fq: str, edge_type: str, spec: EdgeSpec) -> str:
    t = f"{src_fq}.{ident.check(spec.source_table, 'source table')}"
    src, dst = ident.check(spec.src, "src"), ident.check(spec.dst, "dst")
    tag = _tag("edges", edge_type)
    if spec.fold_subject is None:
        return (f"{tag}SELECT DISTINCT {_id(spec.src_type, 'e.' + src)}, {_id(spec.dst_type, 'e.' + dst)} "
                f"FROM {t} e WHERE e.{src} IS NOT NULL AND e.{dst} IS NOT NULL")
    s = ident.check(spec.fold_subject, "fold subject")
    if spec.fold_form == "co_membership":
        return (f"{tag}SELECT {_id(spec.src_type, 'a.' + src)}, {_id(spec.dst_type, 'b.' + src)}, "
                f"COUNT(DISTINCT a.{s}) FROM {t} a JOIN {t} b ON a.{s} = b.{s} AND a.{src} <> b.{src} "
                f"GROUP BY a.{src}, b.{src}")
    return (f"{tag}SELECT {_id(spec.src_type, src)}, {_id(spec.dst_type, dst)}, COUNT(DISTINCT {s}) "
            f"FROM {t} WHERE {src} IS NOT NULL AND {dst} IS NOT NULL GROUP BY {src}, {dst}")


# ---------------------------------------------------------------------------- loading


def _text(cell: Any) -> Optional[str]:
    """A key/id cell as text (the connector may hand back an int); None stays None."""
    return None if cell is None else str(cell)


def _fetch(sql: Sql, statement: str, budget: List[int]) -> List[List[Any]]:
    rows = sql(statement)
    budget[0] -= len(rows)
    if budget[0] < 0:
        raise ReferenceTooLarge("the source has more rows than measure_reference_max_rows")
    return rows


def load_source_graph(sql: Sql, mapping: GraphMapping, fold_floor: int,
                      max_rows: int = DEFAULT_MAX_ROWS) -> SourceGraph:
    """Read node ids and edge pairs from the source tables; drop NULL/orphan ends, dedupe, floor."""
    src_fq = f"{ident.check(mapping.source_catalog, 'source catalog')}.{ident.check(mapping.source_schema, 'source schema')}"
    budget = [max_rows]
    nodes: Set[str] = set()
    for node_type, spec in mapping.nodes.items():
        if isinstance(spec.natural_key, list):
            continue  # unreachable by any edge (mapping contract); never in a walk answer
        rows = _fetch(sql, nodes_sql(src_fq, node_type, spec.source_table, spec.natural_key), budget)
        nodes.update(t for t in (_text(r[0]) for r in rows) if t is not None)
    edges: Dict[EdgeKey, Optional[int]] = {}
    for edge_type, spec in mapping.edges.items():
        for row in _fetch(sql, edges_sql(src_fq, edge_type, spec), budget):
            src, dst = _text(row[0]), _text(row[1])
            if src not in nodes or dst not in nodes:
                continue  # NULL or orphan end: the graph never holds a made-up link
            weight = None if spec.fold_subject is None else int(row[2])
            if weight is not None and weight < fold_floor:
                continue  # below the floor: the pair is not produced at all
            edges[(edge_type, src, dst)] = weight
    return SourceGraph(nodes=frozenset(nodes), edges=edges)


# ---------------------------------------------------------------------------- walks


def _bytes(node_id: str) -> bytes:
    return node_id.encode("utf-8")  # COLLATE "C" (Postgres) and Spark STRING order = byte order


def web_reference(graph: SourceGraph, seed: str, depth: int, top_n: int, folded: bool = False,
                  floor: int = 11) -> List[Tuple[int, str, int]]:
    """[(hop, node_id, support)] for hops 1..depth, sorted. Each hop: the unseen out-neighbours
    of the previous hop, support = edges in (unfolded) or MIN(weight) at or above the floor
    (folded), ordered by (support desc, node_id bytes) and cut at top_n."""
    adj = graph.out_edges()
    seen, frontier, out = {seed}, [seed], []
    for hop in range(1, depth + 1):
        cand: Dict[str, List[Optional[int]]] = defaultdict(list)
        for f in frontier:
            for dst, weight in adj.get(f, ()):
                if dst not in seen:
                    cand[dst].append(weight)
        if folded:
            support = {n: min(ws) for n, ws in cand.items() if min(ws) >= floor}  # type: ignore[type-var]
        else:
            support = {n: len(ws) for n, ws in cand.items()}
        kept = sorted(support, key=lambda n: (-support[n], _bytes(n)))[:top_n]
        out += [(hop, n, support[n]) for n in kept]
        seen.update(kept)
        frontier = kept
    return sorted(out)


def tree_reference(graph: SourceGraph, seed: str, depth: int, top_n: int) -> List[Tuple[int, str]]:
    """[(depth, node_id)]: breadth-first levels, each the distinct unseen out-neighbours of the
    previous level ordered by node_id bytes and cut at top_n; stops at an empty level; the kept
    nodes are then ordered by (depth, node_id bytes) and cut at top_n."""
    adj = graph.out_edges()
    seen, frontier, levels = {seed}, [seed], []
    for d in range(1, depth + 1):
        cand = {dst for f in frontier for dst, _w in adj.get(f, ()) if dst not in seen}
        kept = sorted(cand, key=_bytes)[:top_n]
        if not kept:
            break
        levels += [(d, n) for n in kept]
        seen.update(kept)
        frontier = kept
    return sorted(levels, key=lambda r: (r[0], _bytes(r[1])))[:top_n]


def _undirected(graph: SourceGraph) -> Dict[str, Set[str]]:
    nbrs: Dict[str, Set[str]] = defaultdict(set)
    for (_t, a, b) in graph.edges:
        if a != b:
            nbrs[a].add(b)
            nbrs[b].add(a)
    return nbrs


def similar_reference(graph: SourceGraph, seed: str, top_n: int) -> List[Tuple[str, float]]:
    """[(node_id, score)]: shared-neighbour Jaccard over the undirected graph,
    k / (deg(seed) + deg(y) - k), ordered by (score desc, node_id bytes), cut at top_n."""
    nbrs = _undirected(graph)
    shared: Dict[str, int] = defaultdict(int)
    for d in nbrs.get(seed, ()):
        for y in nbrs[d]:
            if y != seed:
                shared[y] += 1
    dx = len(nbrs.get(seed, ()))
    scores = {y: k / (dx + len(nbrs[y]) - k) for y, k in shared.items()}
    kept = sorted(scores, key=lambda y: (-scores[y], _bytes(y)))[:top_n]
    return sorted((y, round(scores[y], SCORE_DIGITS)) for y in kept)


# ---------------------------------------------------------------------------- seeds


def out_degree(graph: SourceGraph) -> Dict[str, int]:
    deg: Dict[str, int] = defaultdict(int)
    for (_t, src, _dst) in graph.edges:
        deg[src] += 1
    return dict(deg)


def pick_seeds(graph: SourceGraph, n: int, degree: Optional[Dict[str, int]] = None) -> List[str]:
    """n seeds among nodes with at least one out-edge: the n//2 highest out-degree (ties: bytes),
    then the rest spread evenly over the byte-ordered remainder. All of them when there are <= n."""
    deg = out_degree(graph) if degree is None else degree
    cands = sorted((x for x, d in deg.items() if d > 0), key=_bytes)
    if len(cands) <= n:
        return cands
    top = sorted(cands, key=lambda x: (-deg[x], _bytes(x)))[: n // 2]
    rest = [c for c in cands if c not in set(top)]
    step = len(rest) / (n - len(top))
    return top + [rest[math.floor(i * step)] for i in range(n - len(top))]


def undirected_degree(graph: SourceGraph) -> Dict[str, int]:
    return {x: len(v) for x, v in _undirected(graph).items()}


def seeds_for(shape: str, graph: SourceGraph, n: int) -> Sequence[str]:
    """Seeds per shape: out-edges for web/tree, undirected neighbours for similar."""
    return pick_seeds(graph, n, undirected_degree(graph) if shape == "similar" else None)
