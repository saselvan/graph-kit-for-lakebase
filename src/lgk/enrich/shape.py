"""Measure the graph's shape in Spark, so the menu can grey out rows that do not fit."""

from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

Sql = Callable[[str], List[List[Any]]]

# The one depth cap for trees: the DAG check accepts a longest path of at most MAX_DEPTH
# edges, and closure is computed to at most MAX_DEPTH. Deeper graphs are greyed out as
# "deeper than MAX_DEPTH", not as cyclic.
MAX_DEPTH = 100


@dataclass(frozen=True)
class GraphShape:
    n_nodes: int
    n_edges: int  # distinct directed (src, dst) pairs, self-loops dropped
    max_degree: int  # distinct undirected neighbours of the best-connected node
    is_dag: bool
    example_key: str  # natural key of the best-connected node (ties: type, key ascending)
    too_deep: bool = False  # acyclic so far, but a path is longer than MAX_DEPTH


def directed_pairs_sql(fq: str) -> str:
    """Distinct directed edges without self-loops."""
    return (f"SELECT DISTINCT src_node_id AS src, dst_node_id AS dst FROM {fq}.edges "
            "WHERE src_node_id <> dst_node_id")


def undirected_pairs_sql(fq: str) -> str:
    """Each undirected pair once per direction: a->b and b->a collapse to one pair."""
    return (f"SELECT DISTINCT a AS src, b AS dst FROM ("
            f"SELECT src_node_id AS a, dst_node_id AS b FROM {fq}.edges UNION ALL "
            f"SELECT dst_node_id AS a, src_node_id AS b FROM {fq}.edges) WHERE a <> b")


def _counts(sql: Sql, fq: str) -> tuple:
    n_nodes = int(sql(f"SELECT COUNT(DISTINCT node_id) FROM {fq}.nodes")[0][0])
    n_edges = int(sql(f"SELECT COUNT(*) FROM ({directed_pairs_sql(fq)})")[0][0])
    return n_nodes, n_edges


def _hub(sql: Sql, fq: str) -> tuple:
    rows = sql(
        f"WITH deg AS (SELECT src, COUNT(*) AS d FROM ({undirected_pairs_sql(fq)}) GROUP BY src) "
        f"SELECT m.natural_key, d.d FROM deg d JOIN {fq}.node_id_map m ON m.node_id = d.src "
        "ORDER BY d.d DESC, m.node_type ASC, m.natural_key ASC LIMIT 1"
    )
    if rows:
        return str(rows[0][0]), int(rows[0][1])
    any_key = sql(f"SELECT natural_key FROM {fq}.node_id_map ORDER BY node_type, natural_key LIMIT 1")
    return (str(any_key[0][0]) if any_key else ""), 0


def _dag_status(sql: Sql, fq: str) -> Tuple[bool, bool]:
    """(is_dag, too_deep) by Kahn's algorithm in Spark: drop edges whose source has no
    incoming edge, repeat. Each round peels one level of the longest path."""
    cur, nxt = f"{fq}._enrich_kahn_a", f"{fq}._enrich_kahn_b"
    try:
        sql(f"CREATE OR REPLACE TABLE {cur} AS {directed_pairs_sql(fq)}")
        left = _count(sql, cur)
        for peeled in range(MAX_DEPTH + 1):
            if left == 0:
                return True, False
            if peeled == MAX_DEPTH:
                return False, True
            sql(f"CREATE OR REPLACE TABLE {nxt} AS SELECT * FROM {cur} "
                f"WHERE src IN (SELECT dst FROM {cur})")
            now = _count(sql, nxt)
            if now == left:
                return False, False  # every remaining edge sits on or behind a cycle
            cur, nxt, left = nxt, cur, now
        raise AssertionError("unreachable")
    finally:
        for t in (cur, nxt):
            sql(f"DROP TABLE IF EXISTS {t}")


def _count(sql: Sql, table: str) -> int:
    return int(sql(f"SELECT COUNT(*) FROM {table}")[0][0])


def measure_shape(sql: Sql, fq: str) -> GraphShape:
    """Measure the shape of the built graph in schema `fq` (catalog.schema)."""
    n_nodes, n_edges = _counts(sql, fq)
    example_key, max_degree = _hub(sql, fq)
    is_dag, too_deep = _dag_status(sql, fq)
    return GraphShape(n_nodes, n_edges, max_degree, is_dag, example_key, too_deep)
