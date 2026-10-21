"""Walks: the public query helpers over the served views.

`get_walk_query` is a web walk; `get_lookup_query` is a lookup (ancestors, descendants,
same_group or similar_jaccard). The walk SQL comes from the walk engine (`lgk.walks.sql`), which also builds
tree and similar walks and the SQL for every depth (`per_depth_sql`); the lookup SQL comes
from `lgk.walks.lookups`. The served count and the privacy floor come from the privacy gate
(`lgk.privacy.gate.served_walk_selection`), and so does `folded`, derived from the edge types the
call selects, so a folded selection refuses any direction but outbound.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from lgk.privacy.floor import walk_count_terms
from lgk.privacy.gate import read_run_mapping, served_walk_selection, walk_keys
from lgk.walks.lookups import (
    EnrichmentNotServedError,
    ancestors_sql,
    descendants_sql,
    same_group_sql,
    similar_jaccard_sql,
)
from lgk.walks.sql import MAX_SEEDS_DEFAULT, NodePredicate, web_set_sql, web_sql

LOOKUP_KINDS = ("ancestors", "descendants", "same_group", "similar_jaccard")
_LOOKUPS = {"ancestors": ancestors_sql, "descendants": descendants_sql,
            "same_group": same_group_sql, "similar_jaccard": similar_jaccard_sql}
# The enrichment behind each lookup kind (same_group: its `method`, default components).
_LOOKUP_METHOD = {"ancestors": "closure", "descendants": "closure", "same_group": "components",
                  "similar_jaccard": "similarity"}


def get_walk_query(
    start_node_id: str,
    depth: int,
    served_schema: str,
    privacy_floor: Optional[int] = None,
    top_n: int = 25,
    folded: Optional[bool] = None,
    run_folder: Optional[str] = None,
    sort_method: Optional[str] = None,
    direction: str = "outbound",
    edge_types: Optional[Sequence[str]] = None,
    seed: Optional[NodePredicate] = None,
    stop: Optional[NodePredicate] = None,
    max_seeds: int = MAX_SEEDS_DEFAULT,
) -> str:
    """The web walk SQL for hops 1..depth (depth cap 4), rows (hop, node_id, support, score).

    With `seed` it is the set walk instead (lgk.walks.sql.web_set_sql): rows (hop,
    node_id, support, score, matched), parameters `%(seed_value)s` / `%(stop_value)s` (each only
    when its predicate names a field), `start_node_id` unused. Predicate fields are checked
    against the mapping's walk_keys (read through the cleared gate) before any SQL is built.

    Args:
        start_node_id: Node ID of the start node. Bound at execution time as the
            `%(start_node_id)s` parameter; never formatted into the SQL.
        depth: Hops 1..depth (at most 4).
        served_schema: Schema containing served views: `lgk.versions.served_schema(config)`
            (no default here, so the kit has ONE served-schema default).
        privacy_floor: Minimum distinct care subjects behind a served count (default 11,
            never lower; raises ValueError below 11). Applies to folded edges only.
        top_n: Maximum nodes per hop.
        folded: The hop runs over folded edges (care-subject data, weighted). Taken from
            flags.json when run_folder is given; otherwise it must be stated.
        run_folder: Run folder whose flags.json (a clear privacy gate) gives floor and folded.
        sort_method: Sort by the score of this enrichment method, read from the served
            view `<served_schema>.node_score_view` (so the reader role can run it); without
            it the sort is the support. Ties: node_id.
        direction: "outbound" (default), "inbound" or "both" (lgk.walks.sql.DIRECTIONS).
            Folded edges (the gate's answer) are walked outbound only: ValueError otherwise.
        edge_types: The edge types the walk follows (a closed list: the mapping's edge type
            keys when run_folder is given). `folded` is derived from it: all fold types ->
            folded (people counts), none -> unfolded (edge counts), both ->
            MixedFoldSelectionError. Omitted: every type, refused on a mixed graph.
        seed: The set walk's seed predicate (a node type and an optional field).
        stop: The set walk's stop predicate (a matching node is returned, never expanded).
        max_seeds: The seed cap (1..1000, default 200); a larger seed set raises in the query.
    """
    del start_node_id  # bound as a query parameter
    if seed is None and stop is not None:
        raise ValueError("stop needs a seed: a stop predicate applies to the set walk only")
    if seed is not None and direction != "outbound":
        raise NotImplementedError(f"web_set with direction={direction!r}: the set walk is outbound only")
    floor, is_folded, selection = served_walk_selection(run_folder, privacy_floor, folded, edge_types)
    if seed is not None:
        return web_set_sql(
            depth=depth, served_schema=served_schema, top_n=top_n, seed=seed, stop=stop,
            max_seeds=max_seeds, enrich_schema=served_schema if sort_method is not None else None,
            sort_method=sort_method, count_terms=walk_count_terms(floor, is_folded),
            folded=is_folded, walk_keys=walk_keys(read_run_mapping(run_folder)) if run_folder else None,
            edge_types=selection, score_relation="node_score_view",
        )
    return web_sql(
        depth=depth, served_schema=served_schema, top_n=top_n,
        enrich_schema=served_schema if sort_method is not None else None,
        sort_method=sort_method, count_terms=walk_count_terms(floor, is_folded), folded=is_folded,
        direction=direction, score_relation="node_score_view", edge_types=selection,
    )


def _read_report(report_path: Path) -> Optional[Dict[str, Any]]:
    return json.loads(report_path.read_text()) if report_path.exists() else None


def _not_served_reason(report_path: Path, report: Optional[Dict[str, Any]], method: str) -> str:
    """Why `method` is not in the run's enrich report ("" when it is)."""
    if report is None:
        return f"{report_path} does not exist (run enrich first)"
    if method in report:
        return ""
    for row in report.get("menu", []):
        if row.get("method") == method and not row.get("available") and row.get("reason"):
            return row["reason"]
    return "it was not ticked in enrich.yaml"


def _with_report_top_k(report: Dict[str, Any], kw: Dict[str, Any]) -> Dict[str, Any]:
    """similar_jaccard's top_k is the run's (enrich_report.json `similarity.top_k`), never the caller's."""
    if "top_k" in kw:
        raise ValueError("top_k is read from enrich_report.json (similarity.top_k); do not pass it")
    top_k = report["similarity"].get("top_k") if isinstance(report.get("similarity"), dict) else None
    if top_k is None:  # fail closed: never guess how many rows per node enrich kept
        raise EnrichmentNotServedError("similarity has no top_k in enrich_report.json (re-run enrich)")
    return {**kw, "top_k": top_k}


def get_lookup_query(kind: str, served_schema: str, run_folder: Union[str, Path], **kw: Any) -> str:
    """The lookup SQL for `kind` (one of LOOKUP_KINDS); one bound parameter `%(start_node_id)s`.

    `kw` is passed to the builder (lgk.walks.lookups):
    - ancestors / descendants: `max_depth` (1..100, required), `top_n` (default 25); rows
      (node_id text, depth integer).
    - same_group: `top_n` (default 25), `method` (default "components"); rows (node_id text,
      group_id text).
    - similar_jaccard: `top_n` (default 10, at most the run's `similarity.top_k`); rows
      (node_id text, score double precision).

    Raises:
        ValueError: `kind` is not one of LOOKUP_KINDS, or a builder argument is invalid
            (including a caller `top_k`, or a `top_n` above the run's top_k).
        EnrichmentNotServedError: `<run_folder>/enrich_report.json` lacks the method behind
            `kind` (the message says why: the menu row's greyed reason, not ticked, or no report).
    """
    builder = _LOOKUPS.get(kind) if isinstance(kind, str) else None
    if builder is None:
        raise ValueError(f"lookup kind must be one of {LOOKUP_KINDS}, got {kind!r}")
    method = _LOOKUP_METHOD[kind]
    report_path = Path(run_folder) / "enrich_report.json"
    report = _read_report(report_path)
    reason = _not_served_reason(report_path, report, method)
    if reason:
        raise EnrichmentNotServedError(f"{method} was not computed: {reason}")
    if kind == "similar_jaccard":
        kw = _with_report_top_k(report, kw)  # type: ignore[arg-type]  # report exists: reason was ""
    return builder(served_schema, **kw)
