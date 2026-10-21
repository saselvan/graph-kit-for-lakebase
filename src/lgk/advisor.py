"""Index advisor: read the plan, propose indexes, apply them, re-time.

The advisor runs inside the NEW version schema, before the
served views switch to it. It refuses a version the served views read (pg_depend, via
`versions.refuse_if_live`), so the live site is never experimented on.

For every per-depth walk query (rewritten from the served views onto the version tables):
1. warm-up runs (fresh branches have cold caches), then one
   `EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON)`, then `samples` timed runs
   (`EXPLAIN (ANALYZE, TIMING OFF)` server-side Execution Time) -> p50 / p95;
2. a Seq Scan on a LARGE table (at least `min_rows` rows) in the version schema whose
   join condition or filter compares a column by equality proposes
   `CREATE INDEX IF NOT EXISTS lgk_<table>_<column>_idx` (deterministic, the same names
   reapply uses, so reapply and a re-run are no-ops);
3. the proposals are applied and step 1 runs again: the report holds before and after.
"""

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from lgk import ident, versions

DEFAULT_MIN_ROWS = 10_000  # "large": a Seq Scan below this many rows is cheap, never indexed
DEFAULT_WARMUP = 3
MIN_SAMPLES = 20  # pinned: p95 of fewer samples is the max of a handful of runs
DEFAULT_SAMPLES = MIN_SAMPLES

_JOIN_KEYS = ("Hash Cond", "Merge Cond", "Join Filter")


@dataclass(frozen=True)
class SeqScan:
    schema: str
    table: str
    alias: str
    columns: Tuple[str, ...]  # columns compared by equality (index candidates), sorted


@dataclass(frozen=True)
class Proposal:
    schema: str
    table: str
    column: str
    name: str
    statement: str
    applied: bool = False
    not_applied: Optional[str] = None  # why the index was not created (reported, never hidden)


# ---------------------------------------------------------------------------- names


def index_name(table: str, column: str) -> str:
    """lgk_<table>_<column>_idx: deterministic, so CREATE INDEX IF NOT EXISTS is idempotent."""
    name = f"lgk_{ident.check(table, 'table')}_{ident.check(column, 'column')}_idx"
    if len(name.encode()) > 63:
        raise ValueError(f"index name {name} exceeds Postgres' 63-byte identifier limit")
    return name


def create_index_sql(schema: str, table: str, column: str) -> str:
    return (f"CREATE INDEX IF NOT EXISTS {index_name(table, column)} "
            f"ON {ident.check(schema, 'schema')}.{table} ({column})")


# ---------------------------------------------------------------------------- plan reading


def _strip_negations(expr: str) -> str:
    """Remove every `NOT (...)` group: a negated comparison is never served by an index."""
    while (start := expr.find("NOT (")) != -1:
        depth, i = 0, start + 4
        while i < len(expr):
            depth += {"(": 1, ")": -1}.get(expr[i], 0)
            if depth == 0:
                break
            i += 1
        expr = expr[:start] + expr[i + 1:]
    return expr


def _equality_columns(expr: str, alias: str) -> set:
    """Columns of `alias` on either side of a positive `=` (incl. `= ANY`) in `expr`."""
    text = _strip_negations(expr)
    a = re.escape(alias)
    left = re.findall(rf"\b{a}\.([a-z_][a-z0-9_]*) = ", text)
    right = re.findall(rf"= {a}\.([a-z_][a-z0-9_]*)\b", text)
    return set(left) | set(right)


def _walk(node: Dict[str, Any], conds: Tuple[str, ...]) -> Iterator[SeqScan]:
    here = conds + tuple(node[k] for k in _JOIN_KEYS if k in node)
    if node.get("Node Type") == "Seq Scan" and "Relation Name" in node:
        alias = node.get("Alias", node["Relation Name"])
        cols = set()
        for expr in here + ((node["Filter"],) if "Filter" in node else ()):
            cols |= _equality_columns(expr, alias)
        yield SeqScan(node.get("Schema", ""), node["Relation Name"], alias, tuple(sorted(cols)))
    for child in node.get("Plans", []):
        yield from _walk(child, here)


def seq_scans(plan_doc: Any) -> List[SeqScan]:
    """Every Seq Scan in an EXPLAIN (VERBOSE, FORMAT JSON) document, with its candidates."""
    doc = plan_doc[0] if isinstance(plan_doc, list) else plan_doc
    return list(_walk(doc["Plan"], ()))


# ---------------------------------------------------------------------------- proposals


_LEADING_COLUMNS_SQL = (
    "SELECT a.attname FROM pg_index i JOIN pg_class t ON t.oid = i.indrelid "
    "JOIN pg_namespace n ON n.oid = t.relnamespace "
    "JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = i.indkey[0] "
    "WHERE n.nspname = %s AND t.relname = %s"
)


def _is_large(conn: Any, schema: str, table: str, min_rows: int) -> bool:
    count = conn.execute(f"SELECT count(*) FROM {schema}.{ident.check(table, 'table')}").fetchone()[0]
    return count >= min_rows


def _indexed_columns(conn: Any, schema: str, table: str) -> set:
    return {row[0] for row in conn.execute(_LEADING_COLUMNS_SQL, (schema, table)).fetchall()}


_PARTITION_ROOT_SQL = (
    "SELECT rn.nspname, r.relname FROM pg_class c "
    "JOIN pg_namespace n ON n.oid = c.relnamespace "
    "JOIN pg_class r ON r.oid = COALESCE(pg_partition_root(c.oid), c.oid) "
    "JOIN pg_namespace rn ON rn.oid = r.relnamespace "
    "WHERE n.nspname = %s AND c.relname = %s"
)


def _root(conn: Any, scan: SeqScan, version: str) -> Tuple[str, str]:
    """(schema, table) the scan's relation belongs to: its partition ROOT.

    A synced table is a partitioned parent; EXPLAIN names the child partition, which lives
    in another schema (`__db_system.partition_<oid>`). Scans already named in the
    version schema are taken as they are.
    """
    if scan.schema == version:
        return scan.schema, scan.table
    row = conn.execute(_PARTITION_ROOT_SQL, (scan.schema, scan.table)).fetchone()
    return (row[0], row[1]) if row else (scan.schema, scan.table)


def propose(conn: Any, version: str, plans: Sequence[Any], min_rows: int) -> List[Proposal]:
    """CREATE INDEX proposals for Seq Scans on large version tables, deduped, stable order.

    Only tables in `version` (after mapping a partition to its root) are considered; a column
    that already leads an index is skipped. The index goes on the root: on a partitioned
    parent CREATE INDEX covers every partition.
    """
    wanted: Dict[Tuple[str, str], set] = {}
    for plan in plans:
        for scan in seq_scans(plan):
            if not scan.columns:
                continue
            schema, table = _root(conn, scan, version)
            if schema == version:
                wanted.setdefault((schema, table), set()).update(scan.columns)
    out = []
    for (schema, table), cols in sorted(wanted.items()):
        if not _is_large(conn, schema, table, min_rows):
            continue
        for col in sorted(cols - _indexed_columns(conn, schema, table)):
            out.append(Proposal(schema, table, col, index_name(table, col),
                                create_index_sql(schema, table, col)))
    return out


def apply(conn: Any, proposal: Proposal) -> Proposal:
    """Create the index; if this identity may not (a synced table is owned by the sync's
    writer role), return the proposal marked not applied with the reason. Other errors raise."""
    import dataclasses

    import psycopg

    try:
        conn.execute(proposal.statement)
    except psycopg.errors.InsufficientPrivilege as e:
        return dataclasses.replace(proposal, applied=False, not_applied=str(e).strip())
    return dataclasses.replace(proposal, applied=True)


# ---------------------------------------------------------------------------- timing


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile: the ceil(q/100 * n)-th smallest value."""
    ordered = sorted(values)
    return ordered[max(1, math.ceil(q / 100 * len(ordered))) - 1]


def _plan_doc(row: Tuple[Any, ...]) -> Any:
    doc = row[0]
    return json.loads(doc) if isinstance(doc, str) else doc


def _scan_record(scan: SeqScan) -> Dict[str, Any]:
    return {"schema": scan.schema, "table": scan.table, "columns": list(scan.columns)}


def measure(conn: Any, sql: str, params: Dict[str, Any], warmup: int, samples: int
            ) -> Tuple[Dict[str, Any], Any]:
    """Warm up, read the plan with BUFFERS, then time `samples` runs. Returns (row, plan)."""
    for _ in range(warmup):
        conn.execute(sql, params).fetchall()
    plan = _plan_doc(conn.execute(
        f"EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON) {sql}", params).fetchone())
    times = [
        float(_plan_doc(conn.execute(
            f"EXPLAIN (ANALYZE, TIMING OFF, FORMAT JSON) {sql}", params).fetchone())[0]["Execution Time"])
        for _ in range(samples)
    ]
    top = plan[0]["Plan"]
    row = {
        "warmup_runs": warmup, "samples": len(times),
        "p50_ms": percentile(times, 50), "p95_ms": percentile(times, 95),
        "shared_hit_blocks": top.get("Shared Hit Blocks"),
        "shared_read_blocks": top.get("Shared Read Blocks"),
        "seq_scans": [_scan_record(s) for s in seq_scans(plan)],
    }
    return row, plan


def _measure_all(conn: Any, queries: Dict[str, Dict[int, str]], params: Dict[str, Any],
                 warmup: int, samples: int) -> Tuple[Dict[str, Dict[str, Any]], List[Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    plans: List[Any] = []
    for shape, per_depth in queries.items():
        for depth, sql in sorted(per_depth.items()):
            row, plan = measure(conn, sql, params, warmup, samples)
            rows.setdefault(shape, {})[str(depth)] = row
            plans.append(plan)
    return rows, plans


# ---------------------------------------------------------------------------- the advisor


def version_queries(queries: Dict[str, Dict[int, str]], served: str, version: str
                    ) -> Dict[str, Dict[int, str]]:
    """Point every per-depth walk SQL at the version tables instead of the served views.

    Rewrites the base views (SERVED_VIEWS) and the enrichment views (OPTIONAL_SERVED_VIEWS,
    e.g. a sorted walk's `node_score_view` -> `node_score`).

    Refuses SQL that still names the served schema afterwards (a relation with no version
    table): timing it would read the live site.
    """
    ident.check(served, "served schema")
    ident.check(version, "version schema")
    out: Dict[str, Dict[int, str]] = {}
    for shape, per_depth in queries.items():
        for depth, sql in per_depth.items():
            for view, table in versions.SERVED_VIEWS + versions.OPTIONAL_SERVED_VIEWS:
                sql = re.sub(rf"\b{served}\.{view}\b", f"{version}.{table}", sql)
            if re.search(rf"\b{served}\.", sql):
                raise ValueError(f"{shape} depth {depth} still reads the served schema {served}")
            out.setdefault(shape, {})[depth] = sql
    return out


def _check_settings(min_rows: Any, warmup: Any, samples: Any) -> None:
    if isinstance(min_rows, bool) or not isinstance(min_rows, int) or min_rows < 1:
        raise ValueError(f"min_rows must be a positive int, got {min_rows!r}")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1:
        raise ValueError(f"warmup must be at least 1 run (cold caches on fresh branches), got {warmup!r}")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < MIN_SAMPLES:
        raise ValueError(f"samples must be an int >= {MIN_SAMPLES}, got {samples!r}")


def _analyze_tables(conn: Any, queries: Dict[str, Dict[int, str]], version: str) -> None:
    """Fresh planner statistics for the version tables the queries read."""
    text = " ".join(sql for per_depth in queries.values() for sql in per_depth.values())
    for table in sorted(set(re.findall(rf"\b{version}\.([a-z_][a-z0-9_]*)", text))):
        conn.execute(f"ANALYZE {version}.{table}")


def advise(conn: Any, served: str, version: str, queries: Dict[str, Dict[int, str]],
           params: Dict[str, Any], min_rows: int = DEFAULT_MIN_ROWS,
           warmup: int = DEFAULT_WARMUP, samples: int = DEFAULT_SAMPLES) -> Dict[str, Any]:
    """Time, propose, apply, re-time. `conn` must be autocommit. Returns the report dict."""
    _check_settings(min_rows, warmup, samples)
    versions.refuse_if_live(conn, served, [version])
    vq = version_queries(queries, served, version)
    _analyze_tables(conn, vq, version)
    before, plans = _measure_all(conn, vq, params, warmup, samples)
    proposals = [apply(conn, p) for p in propose(conn, version, plans, min_rows)]
    if any(p.applied for p in proposals):
        _analyze_tables(conn, vq, version)
    after, _ = _measure_all(conn, vq, params, warmup, samples)
    return {
        "version_schema": version, "served_schema": served, "min_rows": min_rows,
        "params": params, "proposals": [asdict(p) for p in proposals],
        "depths": {shape: {d: {"before": before[shape][d], "after": after[shape][d]}
                           for d in before[shape]} for shape in before},
    }


def skipped_report(served: str, version: str, reason: str) -> Dict[str, Any]:
    """The report when this identity may not read the version's raw tables: nothing proposed."""
    return {"version_schema": version, "served_schema": served, "skipped": f"advisor skipped: {reason}",
            "proposals": [], "depths": {}}


def report_markdown(report: Dict[str, Any]) -> str:
    """The report for people: proposals, then p50/p95 before and after per shape and depth."""
    if report.get("skipped"):
        return f"# Index advisor: {report['version_schema']}\n\n{report['skipped']}\n"
    lines = [f"# Index advisor: {report['version_schema']}", "",
             f"Large = at least {report['min_rows']} rows. Seed: {report['params']}.", "",
             "## Proposed", ""]
    lines += [f"- `{p['statement']}`: " + ("applied" if p.get("applied") else f"not applied: {p.get('not_applied')}")
              for p in report["proposals"]] or ["- none"]
    lines += ["", "## Timings (ms, server-side, after warm-up)", "",
              "| shape | depth | p50 before | p95 before | p50 after | p95 after | samples |",
              "|---|---|---|---|---|---|---|"]
    for shape, per_depth in report["depths"].items():
        for d, row in per_depth.items():
            b, a = row["before"], row["after"]
            lines.append(f"| {shape} | {d} | {b['p50_ms']:.3f} | {b['p95_ms']:.3f} | "
                         f"{a['p50_ms']:.3f} | {a['p95_ms']:.3f} | {b['samples']} |")
    return "\n".join(lines) + "\n"


def write_report(run_folder: Optional[str], report: Dict[str, Any]) -> None:
    from pathlib import Path

    folder = Path(run_folder or ".")
    (folder / "advisor_report.json").write_text(json.dumps(report, indent=2, default=str))
    (folder / "advisor_report.md").write_text(report_markdown(report))
