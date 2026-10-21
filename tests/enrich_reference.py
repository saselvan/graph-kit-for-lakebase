"""The enrich references and fixture graphs, shared by the live test and the networkx leg.

One module, no copies: tests/test_enrich_live.py runs the shipped SQL against these graphs and
compares it to `reference_pagerank` and `GATE_COMPONENTS`; tests/test_enrich_networkx.py compares
the same references to networkx on the same graphs. Test code only; it shares nothing with src.
"""

from typing import Dict, Iterable, Tuple


def reference_pagerank(nodes: Iterable[str], edges: Iterable[Tuple[str, str]], d: float = 0.85,
                       tol: float = 1e-12, iterations: int = 0) -> Dict[str, float]:
    """Undirected PageRank in plain Python: duplicate pairs and self-loops ignored,
    isolated-node mass spread over all N nodes, teleport (1-d)/N, start uniform 1/N.
    Runs to L1 < tol, or exactly `iterations` steps when that is > 0."""
    nodes = list(nodes)
    nb: Dict[str, set] = {n: set() for n in nodes}
    for a, b in edges:
        if a != b:
            nb[a].add(b)
            nb[b].add(a)
    n_all = len(nodes)
    rank = {n: 1.0 / n_all for n in nodes}
    step = 0
    while True:
        dangling = sum(rank[n] for n in nodes if not nb[n])
        new = {n: (1 - d) / n_all + d * (sum(rank[m] / len(nb[m]) for m in nb[n]) + dangling / n_all)
               for n in nodes}
        l1 = sum(abs(new[n] - rank[n]) for n in nodes)
        rank, step = new, step + 1
        if (iterations and step >= iterations) or (not iterations and l1 < tol):
            return rank


# hostile: a triangle (odd cycle, so not bipartite: it converges inside the 50 cap),
# a tail with a duplicate reversed pair, a self-loop, and an isolated node (dangling mass).
CONVERGING_NODES = ["a", "b", "c", "d", "e", "iso"]
CONVERGING_EDGES = [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("d", "c"), ("d", "e"), ("e", "e")]

# GATE graph (cyclic): a super-connector star (12 leaves), two exact-twin stars (11 leaves each:
# the near-tie runner-ups), a disconnected pair, an isolated node, and reverse-direction duplicates.
SUPER = "k_super"
TWIN_A, TWIN_B = "k_twin_a", "k_twin_b"
S_LEAVES = [f"l_s{i:02d}" for i in range(1, 13)]  # 12
A_LEAVES = [f"l_a{i:02d}" for i in range(1, 12)]  # 11
B_LEAVES = [f"l_b{i:02d}" for i in range(1, 12)]  # 11
PAIR = ("k_pair_x", "k_pair_y")
ISLAND = "k_island"

GATE_EDGES = (
    [(SUPER, leaf) for leaf in S_LEAVES]
    + [(TWIN_A, leaf) for leaf in A_LEAVES]
    + [(TWIN_B, leaf) for leaf in B_LEAVES]
    # hostile: the same undirected pairs again, reversed. Counted twice, twin_a would
    # have 14 edge-ends and outrank the super-connector.
    + [(leaf, TWIN_A) for leaf in A_LEAVES[:3]]
    + [PAIR]
)
GATE_NODES = [SUPER, TWIN_A, TWIN_B, *S_LEAVES, *A_LEAVES, *B_LEAVES, *PAIR, ISLAND]

# The GATE graph's connected components, written by hand and checked by a person: the answer key
# for both the live test (the shipped SQL) and the networkx leg. Never computed from GATE_EDGES.
GATE_COMPONENTS = [
    {SUPER, *S_LEAVES}, {TWIN_A, *A_LEAVES}, {TWIN_B, *B_LEAVES}, set(PAIR), {ISLAND},
]

# TREE graph (DAG): a diamond plus a shortcut edge, for transitive closure.
TREE_EDGES = [
    ("t_r", "t_a"), ("t_r", "t_b"),
    ("t_a", "t_c"), ("t_b", "t_c"),  # hostile: diamond, two paths r->c of depth 2
    ("t_c", "t_d"),
    ("t_r", "t_d"),  # hostile: shortcut, r->d is depth 1 (not 3)
    ("t_r", "t_a"),  # hostile: duplicate source row
]
TREE_NODES = ["t_r", "t_a", "t_b", "t_c", "t_d"]
TREE_CLOSURE = {
    ("t_r", "t_a", 1), ("t_r", "t_b", 1), ("t_r", "t_c", 2), ("t_r", "t_d", 1),
    ("t_a", "t_c", 1), ("t_a", "t_d", 2), ("t_b", "t_c", 1), ("t_b", "t_d", 2),
    ("t_c", "t_d", 1),
}
