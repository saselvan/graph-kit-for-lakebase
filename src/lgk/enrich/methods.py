"""The v1 enrichment methods as Spark SQL batch jobs.

Each `run_*` takes `sql(statement) -> rows` and the kit schema `fq` (catalog.schema),
reads the work tables made by `prepare_work_tables`, writes its fixed enrichment table,
and returns a small report dict. Work tables are named `_enrich_*` and are dropped by
`drop_work_tables`.
"""

from typing import Any, Callable, Dict, List

from lgk.enrich import shape as _shape
from lgk.enrich.shape import directed_pairs_sql, undirected_pairs_sql

Sql = Callable[[str], List[List[Any]]]

# PageRank stops when the L1 change is below 1e-6, or after 50 iterations.
PAGERANK_TOLERANCE = 1e-6
PAGERANK_MAX_ITER = 50
PAGERANK_DAMPING = 0.85
MAX_COMPONENT_ROUNDS = 1000
SIMILARITY_TOP_K = 20

WORK_TABLES = ("und", "deg", "dir", "pr_a", "pr_b", "cc_a", "cc_b", "cl_a", "cl_b")


def _count(sql: Sql, table: str) -> int:
    return int(sql(f"SELECT COUNT(*) FROM {table}")[0][0])


def prepare_work_tables(sql: Sql, fq: str) -> None:
    """Undirected distinct pairs, per-node degree (isolated nodes = 0), directed pairs."""
    sql(f"CREATE OR REPLACE TABLE {fq}._enrich_und AS {undirected_pairs_sql(fq)}")
    sql(f"CREATE OR REPLACE TABLE {fq}._enrich_deg AS "
        f"SELECT n.node_id, COUNT(u.dst) AS deg FROM (SELECT DISTINCT node_id FROM {fq}.nodes) n "
        f"LEFT JOIN {fq}._enrich_und u ON u.src = n.node_id GROUP BY n.node_id")
    sql(f"CREATE OR REPLACE TABLE {fq}._enrich_dir AS {directed_pairs_sql(fq)}")


def drop_work_tables(sql: Sql, fq: str) -> None:
    for name in WORK_TABLES:
        sql(f"DROP TABLE IF EXISTS {fq}._enrich_{name}")


def run_degree(sql: Sql, fq: str) -> Dict[str, Any]:
    sql(f"INSERT INTO {fq}.node_score SELECT node_id, 'degree', CAST(deg AS DOUBLE), "
        f"current_timestamp FROM {fq}._enrich_deg")
    return {"rows": _count(sql, f"{fq}._enrich_deg")}


def _pagerank_step_sql(fq: str, cur: str, nxt: str, n: int, d: float) -> str:
    """One power-iteration step on the undirected graph; isolated-node rank is spread evenly.

    Literals are CAST to DOUBLE: Spark reads `0.025` as DECIMAL, which truncates ranks
    every iteration (measured live: scores summed to 1.00004).
    """
    return (
        f"CREATE OR REPLACE TABLE {nxt} AS "
        f"WITH c AS (SELECT u.dst AS node_id, SUM(r.rank / g.deg) AS s FROM {fq}._enrich_und u "
        f"JOIN {cur} r ON r.node_id = u.src JOIN {fq}._enrich_deg g ON g.node_id = u.src "
        f"GROUP BY u.dst), "
        f"dangling AS (SELECT COALESCE(SUM(r.rank), 0) AS m FROM {cur} r "
        f"JOIN {fq}._enrich_deg g ON g.node_id = r.node_id WHERE g.deg = 0) "
        f"SELECT g.node_id, CAST({(1 - d) / n!r} AS DOUBLE) + CAST({d!r} AS DOUBLE) * "
        f"(COALESCE(c.s, 0) + dangling.m / {n}) AS rank "
        f"FROM {fq}._enrich_deg g LEFT JOIN c ON c.node_id = g.node_id CROSS JOIN dangling"
    )


def run_pagerank(
    sql: Sql, fq: str, damping: float = PAGERANK_DAMPING,
    tolerance: float = PAGERANK_TOLERANCE, max_iter: int = PAGERANK_MAX_ITER,
) -> Dict[str, Any]:
    """Normalised PageRank (scores sum to 1) by power iteration."""
    n = _count(sql, f"{fq}._enrich_deg")
    cur, nxt = f"{fq}._enrich_pr_a", f"{fq}._enrich_pr_b"
    sql(f"CREATE OR REPLACE TABLE {cur} AS SELECT node_id, CAST({1.0 / n!r} AS DOUBLE) AS rank FROM {fq}._enrich_deg")
    iterations, l1, converged = 0, float("inf"), False
    while iterations < max_iter and not converged:
        sql(_pagerank_step_sql(fq, cur, nxt, n, damping))
        l1 = float(sql(f"SELECT SUM(ABS(x.rank - y.rank)) AS L1_CHANGE FROM {nxt} x "
                       f"JOIN {cur} y ON y.node_id = x.node_id")[0][0])
        cur, nxt = nxt, cur
        iterations += 1
        converged = l1 < tolerance
    sql(f"INSERT INTO {fq}.node_score SELECT node_id, 'pagerank', rank, current_timestamp "
        f"FROM {cur}")
    return {"iterations": iterations, "converged": converged, "final_l1": l1}


def run_components(sql: Sql, fq: str) -> Dict[str, Any]:
    """Connected components by min-label propagation; group_id = smallest node_id in it."""
    cur, nxt = f"{fq}._enrich_cc_a", f"{fq}._enrich_cc_b"
    sql(f"CREATE OR REPLACE TABLE {cur} AS SELECT node_id, node_id AS g FROM {fq}._enrich_deg")
    for rounds in range(1, MAX_COMPONENT_ROUNDS + 1):
        sql(f"CREATE OR REPLACE TABLE {nxt} AS SELECT c.node_id, "
            f"LEAST(c.g, COALESCE(MIN(nb.g), c.g)) AS g FROM {cur} c "
            f"LEFT JOIN {fq}._enrich_und u ON u.src = c.node_id "
            f"LEFT JOIN {cur} nb ON nb.node_id = u.dst GROUP BY c.node_id, c.g")
        changed = int(sql(f"SELECT COUNT(*) FROM {nxt} x JOIN {cur} y "
                          f"ON y.node_id = x.node_id WHERE x.g <> y.g")[0][0])
        cur, nxt = nxt, cur
        if changed == 0:
            sql(f"INSERT INTO {fq}.node_group SELECT node_id, 'components', g FROM {cur}")
            groups = int(sql(f"SELECT COUNT(DISTINCT g) FROM {cur}")[0][0])
            return {"rounds": rounds, "groups": groups}
    raise RuntimeError(f"components did not settle in {MAX_COMPONENT_ROUNDS} rounds")


def run_similarity(sql: Sql, fq: str, top_k: int = SIMILARITY_TOP_K) -> Dict[str, Any]:
    """Shared-neighbour (Jaccard) similarity, top_k per source node (ties: dst_id ascending)."""
    sql(
        f"INSERT INTO {fq}.similar_to "
        f"WITH shared AS (SELECT a.src AS x, b.src AS y, COUNT(*) AS k FROM {fq}._enrich_und a "
        f"JOIN {fq}._enrich_und b ON a.dst = b.dst AND a.src <> b.src GROUP BY a.src, b.src), "
        f"scored AS (SELECT s.x, s.y, CAST(s.k AS DOUBLE) / (dx.deg + dy.deg - s.k) AS score "
        f"FROM shared s JOIN {fq}._enrich_deg dx ON dx.node_id = s.x "
        f"JOIN {fq}._enrich_deg dy ON dy.node_id = s.y), "
        f"ranked AS (SELECT x, y, score, ROW_NUMBER() OVER (PARTITION BY x ORDER BY score DESC, y ASC) "
        f"AS rn FROM scored) "
        f"SELECT x, y, score FROM ranked WHERE rn <= {int(top_k)}"
    )
    return {"rows": _count(sql, f"{fq}.similar_to"), "top_k": top_k}


def run_closure(sql: Sql, fq: str) -> Dict[str, Any]:
    """Transitive closure by breadth-first rounds; each pair is kept at its smallest depth.

    The NOT EXISTS on pairs already in `closure` is the cycle guard: a pair is never re-added.
    The depth cap is the shared shape.MAX_DEPTH, the same cap the DAG check uses.
    """
    max_depth = _shape.MAX_DEPTH
    sql(f"INSERT INTO {fq}.closure SELECT src, dst, 1 FROM {fq}._enrich_dir")
    cur, nxt = f"{fq}._enrich_cl_a", f"{fq}._enrich_cl_b"
    sql(f"CREATE OR REPLACE TABLE {cur} AS SELECT src AS anc, dst AS dsc FROM {fq}._enrich_dir")
    for depth in range(2, max_depth + 2):
        sql(f"CREATE OR REPLACE TABLE {nxt} AS SELECT DISTINCT n.anc, n.dsc FROM "
            f"(SELECT f.anc, e.dst AS dsc FROM {cur} f JOIN {fq}._enrich_dir e ON e.src = f.dsc) n "
            f"WHERE NOT EXISTS (SELECT 1 FROM {fq}.closure c "
            f"WHERE c.ancestor_id = n.anc AND c.descendant_id = n.dsc)")
        if _count(sql, nxt) == 0:
            return {"rows": _count(sql, f"{fq}.closure"), "max_depth": depth - 1}
        if depth > max_depth:
            break
        sql(f"INSERT INTO {fq}.closure SELECT anc, dsc, {depth} FROM {nxt}")
        cur, nxt = nxt, cur
    raise RuntimeError(f"closure is deeper than the cap of {max_depth}")


RUNNERS: Dict[str, Callable[[Sql, str], Dict[str, Any]]] = {
    "pagerank": run_pagerank,
    "degree": run_degree,
    "components": run_components,
    "similarity": run_similarity,
    "closure": run_closure,
}
