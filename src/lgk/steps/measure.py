"""Measure step: the metrics of the served graph, after `switch`, in one record.

Run through the runner like any step:
`runner.run_steps([("measure", resolve_step("lgk.steps.measure:measure"))])`. Read-only: no
branch lease, no index, no grant, no CU change. Writes `<run_folder>/measure.jsonl` (one line
per metric) and `measure_report.md`; a rerun replaces both.

- C10 correctness: for N seeds per (shape, depth), the served walk (the SQL the website runs,
  over the served views) equals a reference computed from the SOURCE tables
  (`lgk.measure.reference`, the warehouse reads ids and the walks run in Python), never from the kit's tables.
  Counts only: seeds checked / equal / different.
- C13 speed: the client-side timer on a warm connection (in-region perf_counter, not Postgres
  execution time), per (shape, depth), `measure_samples` runs (default 200, never below the
  advisor's MIN_SAMPLES). `timing: client`. Execution time is measured separately via
  EXPLAIN (ANALYZE, TIMING OFF) and reported alongside the breakdown. The label is
  "Postgres execution time (EXPLAIN ANALYZE); excludes planning, result transfer and network".
- C23 time to first served graph: the sum of run.json `duration_s` over preflight..switch
  (step time, operator time excluded).
- C30b freshness: not measured; the line says `pending` with the fixed reason.

Shapes (`measure_shapes`, default web, tree, similar): web and tree outbound, depths
1..measure_max_depth (default 4); similar = the served Jaccard lookup (similar_to_view), only
when the run's enrich report holds similarity, else `not_served`. On a graph with folded and
unfolded edge types the web walk over every type is refused by the gate: `not_checked`.

Other config keys: measure_seeds (20), measure_warmup (advisor default), top_n (25),
measure_similar_top_n (10), measure_reference_max_rows (5,000,000).
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from lgk import advisor, versions
from lgk.mapping import load_mapping
from lgk.measure import compare, record, reference, timing
from lgk.privacy.gate import (
    MixedFoldSelectionError,
    required_run_folder,
    served_count_terms,
    served_walk_policy,
)
from lgk.walks.lookups import similar_jaccard_sql
from lgk.walks.sql import per_depth_sql

logger = logging.getLogger(__name__)

DEFAULT_SHAPES = ("web", "tree", "similar")
DEFAULT_MAX_DEPTH = 4
DEFAULT_SEEDS = 20
DEFAULT_TOP_N = 25
DEFAULT_SIMILAR_TOP_N = 10
REFERENCE = "python_over_source"


@dataclass(frozen=True)
class MeasureDeps:
    """The step's outside reads (Postgres, the warehouse, the endpoint) and clock, injectable for tests."""

    connect: Callable[[Dict[str, Any]], Any]
    warehouse_sql: Callable[[Dict[str, Any]], reference.Sql]
    endpoint_cu: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]
    clock: Callable[[], float] = time.perf_counter


def _connect(config: Dict[str, Any]) -> Any:
    from lgk.branch import connect_branch

    return connect_branch(config)


def _warehouse_sql(config: Dict[str, Any]) -> reference.Sql:
    from databricks.sdk import WorkspaceClient

    from lgk.measure.warehouse import fetch_all
    from lgk.uc_sql import warehouse_id

    w, wh = WorkspaceClient(), warehouse_id(config)
    return lambda statement: fetch_all(w, wh, statement)


def _endpoint_cu(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """{min, max} autoscaling CU of the branch's primary endpoint (a read-only GET), or None."""
    from databricks.sdk import WorkspaceClient

    from lgk.branch import BranchRef

    try:
        status = WorkspaceClient().postgres.get_endpoint(
            name=f"{BranchRef.from_config(config).path}/endpoints/primary").status
        return {"min": status.autoscaling_limit_min_cu, "max": status.autoscaling_limit_max_cu}
    except Exception as e:
        logger.warning(f"measure: endpoint CU read failed: {type(e).__name__}")
        return None


def default_deps() -> MeasureDeps:
    return MeasureDeps(connect=_connect, warehouse_sql=_warehouse_sql, endpoint_cu=_endpoint_cu)


def _positive(config: Dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive int, got {value!r}")
    return value


def settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """The step's settings, validated before anything connects."""
    shapes = tuple(config.get("measure_shapes", DEFAULT_SHAPES))
    unknown = [s for s in shapes if s not in DEFAULT_SHAPES]
    if unknown or not shapes:
        raise ValueError(f"measure_shapes must be a non-empty subset of {DEFAULT_SHAPES}, got {list(shapes)}")
    return {
        "shapes": shapes,
        "max_depth": _positive(config, "measure_max_depth", DEFAULT_MAX_DEPTH),
        "seeds": _positive(config, "measure_seeds", DEFAULT_SEEDS),
        "samples": compare.check_samples(config.get("measure_samples", compare.DEFAULT_SAMPLES)),
        "warmup": _positive(config, "measure_warmup", advisor.DEFAULT_WARMUP),
        "top_n": _positive(config, "top_n", DEFAULT_TOP_N),
        "similar_top_n": _positive(config, "measure_similar_top_n", DEFAULT_SIMILAR_TOP_N),
        "max_rows": _positive(config, "measure_reference_max_rows", reference.DEFAULT_MAX_ROWS),
    }


# ---------------------------------------------------------------------------- the served walks


def _similarity_top_k(run_folder: str) -> Optional[int]:
    path = Path(run_folder) / "enrich_report.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    sim = report.get("similarity")
    return sim.get("top_k") if isinstance(sim, dict) else None


def _shape_plan(shape: str, s: Dict[str, Any], run_folder: str, served: str, floor: int,
                folded: Optional[bool]) -> Tuple[Dict[int, str], Optional[Callable], str, str]:
    """({depth: served SQL}, reference(graph, seed, depth) or None, status, reason)."""
    top_n = s["top_n"]
    if shape == "web":
        if folded is None:
            return {}, None, "not_checked", "mixed_fold_selection"
        sql = per_depth_sql("web", s["max_depth"], served_schema=served, top_n=top_n,
                            count_terms=served_count_terms(run_folder, None, None))
        return sql, lambda g, seed, d: reference.web_reference(g, seed, d, top_n, folded, floor), "checked", ""
    if shape == "tree":
        sql = per_depth_sql("tree", s["max_depth"], served_schema=served, top_n=top_n)
        return sql, lambda g, seed, d: reference.tree_reference(g, seed, d, top_n), "checked", ""
    top_k = _similarity_top_k(run_folder)
    if top_k is None:
        return {}, None, "not_served", "similarity_not_enriched"
    n = min(s["similar_top_n"], top_k)
    return ({1: similar_jaccard_sql(served, top_n=n, top_k=top_k)},
            lambda g, seed, d: reference.similar_reference(g, seed, n), "checked", "")


def _walk_policy(run_folder: str) -> Tuple[int, Optional[bool]]:
    """(floor, folded) of a served walk over every edge type; folded None on a mixed graph."""
    try:
        return served_walk_policy(run_folder, None, None)
    except MixedFoldSelectionError:
        from lgk.privacy.floor import checked_floor

        return checked_floor(None), None


def _served_size(conn: Any, served: str) -> Tuple[int, int]:
    nodes = conn.execute(f"SELECT count(*) FROM {served}.nodes_view").fetchone()[0]
    edges = conn.execute(f"SELECT count(*) FROM {served}.edges_view").fetchone()[0]
    return int(nodes), int(edges)


def _timing_seed(conn: Any, served: str) -> Optional[str]:
    """The served node with the most out-edges (the advisor's worst case), or None on no edges."""
    row = conn.execute(f"SELECT src_node_id FROM {served}.edges_view GROUP BY 1 "
                       'ORDER BY count(*) DESC, src_node_id COLLATE "C" LIMIT 1').fetchone()
    return None if row is None else str(row[0])


def _correctness_line(shape: str, depth: Optional[int], counts: Dict[str, int], status: str,
                      reason: str) -> Dict[str, Any]:
    line = {"metric": "walk_correctness", "value": counts["seeds_equal"], "unit": "count", "shape": shape,
            "depth": depth, **counts, "status": status, "reference": REFERENCE}
    if reason:
        line["reason"] = reason
    return line


def _latency_line(shape: str, depth: int, t: Dict[str, Any]) -> Dict[str, Any]:
    """A single walk_latency line with client timing and breakdown."""
    return {
        "metric": "walk_latency",
        "value": t["client_p95_ms"],
        "unit": "ms",
        "shape": shape,
        "depth": depth,
        "n_samples": t["n_samples"],
        "timing": "client",
        "client_p95_ms": t["client_p95_ms"],
        "client_mean_ms": t["client_mean_ms"],
        "client_mean_se_ms": t["client_mean_se_ms"],
        "execution_p95_ms": t["execution_p95_ms"],
        "execution_mean_ms": t["execution_mean_ms"],
        "execution_mean_se_ms": t["execution_mean_se_ms"],
        "planning_mean_ms": t["planning_mean_ms"],
        "planning_mean_se_ms": t["planning_mean_se_ms"],
        "select1_rtt_mean_ms": t["select1_rtt_mean_ms"],
        "select1_rtt_mean_se_ms": t["select1_rtt_mean_se_ms"],
        "remainder_mean_ms": t["remainder_mean_ms"],
        "execution_label": timing.EXECUTION_LABEL,
        "remainder_label": timing.REMAINDER_LABEL,
        "prepared_statements": t["prepared_statements"],
    }


def _measure_shape(conn: Any, shape: str, plan: Tuple, graph: reference.SourceGraph,
                   s: Dict[str, Any], timing_seed: Optional[str], used: List[str],
                   clock: Callable[[], float] = time.perf_counter) -> List[Dict[str, Any]]:
    queries, ref, status, reason = plan
    none = {"seeds_checked": 0, "seeds_equal": 0, "seeds_different": 0}
    if ref is None:
        return [_correctness_line(shape, None, none, status, reason)]
    seeds = list(reference.seeds_for(shape, graph, s["seeds"]))
    used.extend(seeds)
    lines: List[Dict[str, Any]] = []
    for depth, sql in sorted(queries.items()):
        counts = compare.check_depth(conn, shape, sql, seeds, lambda seed, d=depth: ref(graph, seed, d))
        lines.append(_correctness_line(shape, depth, counts, status, reason))
        if timing_seed is not None:
            t = timing.time_walk(conn, sql, {"start_node_id": timing_seed}, s["warmup"],
                                s["samples"], clock=clock)
            lines.append(_latency_line(shape, depth, t))
    return lines


# ---------------------------------------------------------------------------- the step


def run_measure(config: Dict[str, Any], deps: MeasureDeps) -> Path:
    """Measure the served graph; write measure.jsonl and measure_report.md; return the jsonl path."""
    s = settings(config)  # refuses bad settings before any connection
    run_folder = required_run_folder(config)
    served = versions.served_schema(config)
    mapping = load_mapping(Path(run_folder) / "graph_mapping.yaml")
    floor, folded = _walk_policy(run_folder)
    plans = {shape: _shape_plan(shape, s, run_folder, served, floor, folded) for shape in s["shapes"]}
    graph = reference.load_source_graph(deps.warehouse_sql(config), mapping, floor, s["max_rows"])
    used: List[str] = []
    conn = deps.connect(config)
    try:
        conn.autocommit = True
        nodes, edges = _served_size(conn, served)
        timing_seed = _timing_seed(conn, served)
        if timing_seed is not None:
            used.append(timing_seed)
        lines = [line for shape in s["shapes"]
                 for line in _measure_shape(conn, shape, plans[shape], graph, s, timing_seed, used, deps.clock)]
    finally:
        conn.close()
    lines.append({"metric": "time_to_first_served_graph", "unit": "s", "timing": "step",
                  "excludes": "operator_time", **record.first_serve_time(Path(run_folder) / "run.json")})
    lines.append(record.freshness_line())
    common = {"build": record.build_identity(), "nodes": nodes, "edges": edges,
              "endpoint_cu": deps.endpoint_cu(config)}  # no client: steps never ask where they run
    path = record.write(run_folder, record.stamp(lines, common), forbidden=used)
    different = sum(x.get("seeds_different", 0) for x in lines if x["metric"] == "walk_correctness")
    logger.info(f"measure: {len(lines)} lines in {path}; {different} seed answer(s) differ from the reference")
    return path


def measure(config: Dict[str, Any]) -> str:
    """Runner entry: measure the served graph of this run (see the module docstring)."""
    return str(run_measure(config, default_deps()))
