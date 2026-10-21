"""Lookups (2f): one indexed read of a served enrichment view per call, no recursion.

- ancestors / descendants read `closure_view`;
- same_group reads `node_group_view` (the other members of the seed's group, one method);
- similar_jaccard reads `similar_to_view` (the seed's top shared-neighbour rows).

`closure_view` (served from the version's `closure` table, which enrich builds for trees and
DAGs only) holds one row (ancestor_id, descendant_id, depth) per reachable pair, at the
pair's SHORTEST depth. A lookup is one indexed read of it: no recursion at query time.

Rows (node_id text, depth integer), ordered by (depth, node_id) with node_id in byte order
(`COLLATE "C"`, the same order Spark uses for STRING), then LIMITed to `top_n`. One bound
parameter, `%(start_node_id)s`.

descendants is exact: it returns every node within `max_depth` of the start, cut ONCE to
`top_n` by (depth, node_id). The tree walk (`lgk.walks.sql.tree_sql`) cuts each level to
`top_n` before expanding the next, so below a hub the two disagree (the walk drops the
children of cut nodes; the lookup does not), by design.

same_group: rows (node_id text, group_id text), the seed's group under `method` (today only
`components` writes node_group), the seed excluded, ordered by node_id in byte order. A
group id is only compared within one method (`g.method = s.method`).

similar_jaccard: rows (node_id text, score double precision), the seed's similar_to rows by
(score DESC, dst_id byte order). Enrich keeps `top_k` rows per source (`enrich_report.json`
`similarity.top_k`), so a `top_n` above it would silently return fewer rows: refused.
"""

from lgk import ident
from lgk.enrich.methods import SIMILARITY_TOP_K
from lgk.enrich.shape import MAX_DEPTH
from lgk.steps.enrich import TABLE_OF
from lgk.walks.sql import _positive

# The enrichment methods that write node_group (today only components).
GROUP_METHODS = tuple(m for m, table in TABLE_OF.items() if table == "node_group")


class EnrichmentNotServedError(RuntimeError):
    """The enrichment a lookup reads was not computed for this graph (see the message)."""


def _lookup_sql(served_schema: str, max_depth: int, top_n: int, start_col: str, out_col: str) -> str:
    served = ident.check(served_schema, "served_schema")
    _positive(max_depth, "max_depth", MAX_DEPTH)
    _positive(top_n, "top_n")
    return (
        f"SELECT c.{out_col} AS node_id, c.depth\n"
        f"FROM {served}.closure_view c\n"
        f"WHERE c.{start_col} = CAST(%(start_node_id)s AS text)\n"
        f"  AND c.depth <= {max_depth}\n"
        f'ORDER BY c.depth, c.{out_col} COLLATE "C"\n'
        f"LIMIT {top_n}"
    )


def ancestors_sql(served_schema: str, max_depth: int, top_n: int = 25) -> str:
    """Rows (node_id, depth): the nodes the start descends from, within max_depth (1..100)."""
    return _lookup_sql(served_schema, max_depth, top_n, "descendant_id", "ancestor_id")


def descendants_sql(served_schema: str, max_depth: int, top_n: int = 25) -> str:
    """Rows (node_id, depth): every node below the start within max_depth (1..100), exact."""
    return _lookup_sql(served_schema, max_depth, top_n, "ancestor_id", "descendant_id")


def same_group_sql(served_schema: str, top_n: int = 25, method: str = "components") -> str:
    """Rows (node_id, group_id): the other members of the start's group under `method`."""
    served = ident.check(served_schema, "served_schema")
    _positive(top_n, "top_n")
    if not isinstance(method, str) or method not in GROUP_METHODS:
        raise ValueError(f"method must be one of {GROUP_METHODS} (the methods that write node_group), "
                         f"got {method!r}")
    return (
        "SELECT g.node_id, g.group_id\n"
        f"FROM {served}.node_group_view s\n"
        f"JOIN {served}.node_group_view g\n"
        "  ON g.group_id = s.group_id AND g.method = s.method\n"
        "WHERE s.node_id = CAST(%(start_node_id)s AS text)\n"
        f"  AND s.method = '{method}'\n"
        "  AND g.node_id <> s.node_id\n"
        'ORDER BY g.node_id COLLATE "C"\n'
        f"LIMIT {top_n}"
    )


def similar_jaccard_sql(served_schema: str, top_n: int = 10, top_k: int = SIMILARITY_TOP_K) -> str:
    """Rows (node_id, score): the start's most similar nodes, 1 <= top_n <= top_k."""
    served = ident.check(served_schema, "served_schema")
    _positive(top_k, "top_k")
    _positive(top_n, "top_n", top_k)
    return (
        "SELECT s.dst_id AS node_id, s.score\n"
        f"FROM {served}.similar_to_view s\n"
        "WHERE s.src_id = CAST(%(start_node_id)s AS text)\n"
        'ORDER BY s.score DESC, s.dst_id COLLATE "C"\n'
        f"LIMIT {top_n}"
    )
