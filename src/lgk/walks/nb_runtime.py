"""Runtime of the generated walks notebook.

This file is copied VERBATIM into the notebook's helper cell (see lgk.walks.notebook), so
the notebook runs with only psycopg and the Databricks SDK; it never needs the kit. For that
reason it imports nothing from lgk and only stdlib at module level. The kit's own branch
connection (`lgk.branch.connect_branch`) calls `connect` here, so the kit, its
tests and the notebook share ONE connection pattern.

Nothing here prints a credential: `connect` passes the token straight to psycopg.
"""

import json
import time
from decimal import Decimal
from typing import Any, Dict, Iterator, List, Optional, Tuple

INDEX_SCANS = ("Index Scan", "Index Only Scan", "Bitmap Index Scan")
STATEMENT_TIMEOUT = "30s"  # time bound on every walk, recursive ones included


def connect(project: str, branch: str) -> Any:
    """psycopg connection to the branch's Postgres, as the operator (proven pattern).

    host = the branch's primary endpoint host, user = the operator's Databricks identity,
    password = a fresh database credential. Not the workspace host, not user "token".
    """
    import psycopg
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    endpoint = f"projects/{project}/branches/{branch}/endpoints/primary"
    host = w.postgres.get_endpoint(name=endpoint).status.hosts.host
    cred = w.postgres.generate_database_credential(endpoint=endpoint)
    return psycopg.connect(
        host=host,
        dbname="databricks_postgres",
        user=w.current_user.me().user_name,
        password=cred.token,
        sslmode="require",
    )


# ---------------------------------------------------------------------------- answers
def _cell(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


def jsonable(rows: List[Tuple[Any, ...]]) -> List[List[Any]]:
    """Rows as JSON-ready lists (Decimal -> float, anything else exotic -> str)."""
    return [[_cell(v) for v in row] for row in rows]


def run_walk(conn: Any, sql: str, seed: str) -> Tuple[List[List[Any]], float]:
    """(rows, wall-clock ms) of one live run of `sql` for `seed` (bound, never formatted)."""
    t0 = time.perf_counter()
    rows = conn.execute(sql, {"start_node_id": seed}).fetchall()
    return jsonable(rows), (time.perf_counter() - t0) * 1000.0


def pick_seed(conn: Any, served_schema: str) -> str:
    """The node with the most out-edges in the SERVED edges view (the worst case)."""
    row = conn.execute(
        f"SELECT src_node_id FROM {served_schema}.edges_view GROUP BY 1 "
        'ORDER BY count(*) DESC, src_node_id COLLATE "C" LIMIT 1'
    ).fetchone()
    if row is None:
        raise ValueError(f"{served_schema}.edges_view is empty: no seed to walk from")
    return row[0]


# ---------------------------------------------------------------------------- plans
def _plan_nodes(node: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    yield node
    for child in node.get("Plans", []):
        yield from _plan_nodes(child)


def summarize_plan(doc: Any) -> Dict[str, Any]:
    """EXPLAIN (FORMAT JSON) reduced to node types, index scans vs seq scans, and rows."""
    if isinstance(doc, str):
        doc = json.loads(doc)
    top = doc[0] if isinstance(doc, list) else doc
    nodes = list(_plan_nodes(top["Plan"]))
    index_scans = [
        {"type": n["Node Type"], "index": n.get("Index Name"), "relation": n.get("Relation Name")}
        for n in nodes if n.get("Node Type") in INDEX_SCANS
    ]
    seq_scans = [n["Relation Name"] for n in nodes
                 if n.get("Node Type") == "Seq Scan" and "Relation Name" in n]
    return {
        "node_types": list(dict.fromkeys(n["Node Type"] for n in nodes)),
        "index_scans": index_scans,
        "seq_scans": seq_scans,
        "uses_index": bool(index_scans),
        "rows": top["Plan"].get("Actual Rows", top["Plan"].get("Plan Rows")),
        "execution_ms": top.get("Execution Time"),
    }


def plan_summary(conn: Any, sql: str, seed: str) -> Dict[str, Any]:
    """Live EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) of `sql` for `seed`, summarized."""
    row = conn.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}",
                       {"start_node_id": seed}).fetchone()
    return summarize_plan(row[0])


# ---------------------------------------------------------------------------- timings
def recorded_timing(report: Optional[Dict[str, Any]], shape: str,
                    depth: int) -> Optional[Dict[str, Any]]:
    """p50/p95 the index advisor measured for this shape and depth (after its indexes)."""
    row = (((report or {}).get("depths") or {}).get(shape) or {}).get(str(depth)) or {}
    after = row.get("after")
    if not after:
        return None
    return {"p50_ms": after["p50_ms"], "p95_ms": after["p95_ms"], "samples": after.get("samples")}


# ---------------------------------------------------------------------------- one cell
def walk_cell(conn: Any, shape: str, depth: int, sql: str, seed: str,
              report: Optional[Dict[str, Any]]) -> List[List[Any]]:
    """Show the SQL, run it live, show the answer, the times and the plan; return the rows."""
    print(f"--- {shape} walk, depth {depth}, seed {seed!r}\n{sql}\n")
    rows, ms = run_walk(conn, sql, seed)
    print(f"answer ({len(rows)} rows):")
    for row in rows:
        print("  ", row)
    print(f"this run: {ms:.1f} ms (wall clock, from this notebook)")
    recorded = recorded_timing(report, shape, depth)
    if recorded:
        print(f"index advisor: p50 {recorded['p50_ms']:.3f} ms, p95 {recorded['p95_ms']:.3f} ms "
              f"({recorded['samples']} server-side samples)")
    else:
        print("index advisor: no recorded timing for this depth (run the advise step)")
    print("plan:", json.dumps(plan_summary(conn, sql, seed)))
    return rows
