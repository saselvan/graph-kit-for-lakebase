"""Advise step: the index advisor on the run's NEW version, before the switch.

Builds the per-depth walk SQL the website will run (the served count and floor from the
privacy gate), points it at the version tables, and runs `lgk.advisor.advise`. Refuses a
version the served views already read. If this identity may not read the version's raw
tables (InsufficientPrivilege), the report says "advisor skipped: <reason>" and the step
succeeds, so switch still runs; any other error fails it. Writes advisor_report.json and advisor_report.md to
the run folder.

Config keys (all optional): advisor_shapes (default web, tree), advisor_directions (default
[outbound]; each listed direction is measured, an inbound walk shows the dst_node_id scan the
advisor then proposes an index for), advisor_max_depth (4),
advisor_min_rows, advisor_warmup, advisor_samples, advisor_seed_node_id (default: the node
with the most out-edges in the version, the worst case), top_n (25).

Shape `web_set` is opt-in through advisor_shapes. It measures the set walk seeded on the
first declared walk key (node types in byte order) with the value whose seed set (at most
max_seeds = 200 nodes) has the most out-edges in the version (ties: the value in byte order), so
the seed scan reads the walk-key index provision made.
"""

import logging
from typing import Any, Dict, Optional

from lgk import advisor, versions
from lgk.privacy.gate import read_run_mapping, served_count_terms, served_walk_policy, walk_keys
from lgk.branch import connect_branch
from lgk.walks.sql import MAX_SEEDS_DEFAULT, NodePredicate, per_depth_sql

logger = logging.getLogger(__name__)

DEFAULT_SHAPES = ("web", "tree")
DEFAULT_DIRECTIONS = ("outbound",)
DEFAULT_MAX_DEPTH = 4
DEFAULT_TOP_N = 25


def _walk_queries(config: Dict[str, Any], served: str) -> Dict[str, Dict[int, str]]:
    """{shape key: {depth: SQL}} for every advisor shape and direction. An outbound walk keeps the
    bare shape key ("web"); another direction is keyed "<shape>:<direction>" ("web:inbound")."""
    top_n = config.get("top_n", DEFAULT_TOP_N)
    depth = config.get("advisor_max_depth", DEFAULT_MAX_DEPTH)
    queries = {}
    for direction in config.get("advisor_directions", DEFAULT_DIRECTIONS):
        for shape in config.get("advisor_shapes", DEFAULT_SHAPES):
            if shape == "web_set" and direction != "outbound":
                continue  # the set walk is outbound only
            kw: Dict[str, Any] = {"served_schema": served, "top_n": top_n}
            if direction != "outbound":
                kw["direction"] = direction
            if shape in ("web", "web_set"):
                kw["count_terms"] = served_count_terms(config.get("run_folder"), None, None)
                if direction != "outbound":  # the folded refusal sees the gate's answer
                    kw["folded"] = served_walk_policy(config.get("run_folder"), None, None)[1]
            if shape == "web_set":
                keys = walk_keys(read_run_mapping(config.get("run_folder"))) if config.get("run_folder") else None
                kw.update(seed=web_set_seed(keys), walk_keys=keys)
            key = shape if direction == "outbound" else f"{shape}:{direction}"
            queries[key] = per_depth_sql(shape, depth, **kw)
    return queries


def web_set_seed(keys: Optional[Dict[str, Any]]) -> NodePredicate:
    """The advisor's set-walk seed predicate: the first declared walk key (node types sorted)."""
    for node_type in sorted(keys or {}):
        if keys[node_type]:
            return NodePredicate(node_type, keys[node_type][0])
    raise ValueError("advisor shape web_set needs a node type with walk_keys in graph_mapping.yaml")


def _worst_case_seed_value(conn: Any, version: str, seed: NodePredicate, max_seeds: int) -> str:
    """The `seed.field` value whose seed set (<= max_seeds nodes) has the most out-edges."""
    row = conn.execute(
        f"SELECT v FROM (SELECT n.properties ->> '{seed.field}' AS v, count(DISTINCT n.node_id) AS seeds, "
        f"count(e.src_node_id) AS out_edges FROM {version}.nodes n "
        f"LEFT JOIN {version}.edges e ON e.src_node_id = n.node_id "
        f"WHERE n.node_type = '{seed.node_type}' AND n.properties ->> '{seed.field}' IS NOT NULL "
        f"GROUP BY 1) x WHERE seeds <= {max_seeds} "
        'ORDER BY out_edges DESC, v COLLATE "C" LIMIT 1').fetchone()
    if row is None:
        raise ValueError(f"no {seed.node_type}.{seed.field} value has a seed set of at most {max_seeds} nodes")
    return row[0]


def _worst_case_seed(conn: Any, version: str) -> str:
    row = conn.execute(f"SELECT src_node_id FROM {version}.edges GROUP BY 1 "
                       'ORDER BY count(*) DESC, src_node_id COLLATE "C" LIMIT 1').fetchone()
    if row is None:
        raise ValueError(f"{version}.edges is empty: nothing to advise on")
    return row[0]


def _advise_or_skip(conn: Any, config: Dict[str, Any], served: str, version: str,
                    queries: Dict[str, Dict[int, str]]) -> Dict[str, Any]:
    """The advisor's report; a skipped report if this identity may not READ the version's raw
    tables (e.g. a bundle SP without USAGE/SELECT on them). Every other error raises."""
    import psycopg

    try:
        seed = config.get("advisor_seed_node_id") or _worst_case_seed(conn, version)
        params = {"start_node_id": seed}
        if "web_set" in queries:  # the set walk binds a seed value, not a start node
            keys = walk_keys(read_run_mapping(config.get("run_folder")))
            params["seed_value"] = _worst_case_seed_value(conn, version, web_set_seed(keys), MAX_SEEDS_DEFAULT)
        return advisor.advise(
            conn, served, version, queries, params,
            min_rows=config.get("advisor_min_rows", advisor.DEFAULT_MIN_ROWS),
            warmup=config.get("advisor_warmup", advisor.DEFAULT_WARMUP),
            samples=config.get("advisor_samples", advisor.DEFAULT_SAMPLES),
        )
    except psycopg.errors.InsufficientPrivilege as e:
        return advisor.skipped_report(served, version, str(e).strip())


def advise(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the advisor on versions.version_schema(config); returns and writes the report."""
    served = versions.served_schema(config)
    version = versions.version_schema(config)
    queries = _walk_queries(config, served)
    conn = connect_branch(config)
    try:
        conn.autocommit = True
        report = _advise_or_skip(conn, config, served, version, queries)
    finally:
        conn.close()
    advisor.write_report(config.get("run_folder"), report)
    if report.get("skipped"):
        logger.warning(f"{version}: {report['skipped']}")
        return report
    applied = sum(1 for p in report["proposals"] if p["applied"])
    logger.info(f"advisor on {version}: {applied} of {len(report['proposals'])} index(es) applied")
    return report
