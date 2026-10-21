"""Per-depth Postgres SQL for the three walk shapes.

- web: one CTE per hop. Each hop takes the distinct neighbours of the previous hop (out-,
  in- or both-way neighbours, `direction`) that were not seen in an earlier hop, orders
  them by (sort key, node_id) and LIMITs them, so a hub with 10,000 links never widens the
  next hop beyond `top_n`. Depth cap 4.
- tree: a breadth-first recursive CTE (one row per level, a `seen` array as the cycle
  guard, the depth cap inside the recursion). Each level is ordered by (sort key, node_id)
  and LIMITed to `top_n` before the next level expands, so a hub cannot blow up a level.
  Each node is reached once, so diamonds never multiply paths; the result is ordered by
  (depth, sort key, node_id) before its LIMIT.
- similar: a pgvector cosine order with a LIMIT, served by an HNSW index. One hop.

Direction: `outbound` follows each edge src -> dst (the default, byte-identical to the
SQL before directions existed), `inbound` follows it dst -> src, `both` follows either way.
Under `both` a neighbour counts once per (previous-hop node, relationship_type), whichever way
the edge points, so a mutual pair of one edge type is support 1. `direction` is a closed set,
validated before any SQL is built; it never reaches the SQL as text.

Edge types: `edge_types` names the edge types a walk follows. The names are checked
against the identifier pattern, de-duplicated and sorted in byte order, then spliced as
literals into one `AND e.relationship_type IN (...)` line per hop (a name that passed the
pattern holds no quote). Omitted: no line, the SQL is byte-identical to before.

Set walk (`web_set_sql`): a web walk whose hop 0 is every node matching a seed
predicate (`NodePredicate`: a node type, plus an optional equality on `label`, `natural_key` or
a declared walk key), whose frontier is the set (one cut per hop over all of it), and which
never expands a node matching the stop predicate (`matched = true`). Outbound only.

Every query takes one parameter, `%(start_node_id)s`, except the set walk, which takes
`%(seed_value)s` / `%(stop_value)s` (each only when its predicate names a field). node_id order
is byte order (`COLLATE "C"`), the same order Spark uses for STRING, so ties break the same way
in both.
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from lgk import ident
from lgk.privacy.floor import walk_count_terms

WEB_MAX_DEPTH = 4
TREE_MAX_DEPTH = 100  # trees go deeper than webs; same cap as lgk.enrich.shape.MAX_DEPTH
SHAPES = ("web", "tree", "similar", "web_set")
MAX_SEEDS_DEFAULT = 200  # the seed cap; a larger set raises, never truncates
MAX_SEEDS_CAP = 1000
NODE_FIELDS = ("label", "natural_key")  # always allowed as a predicate field; not listed in walk_keys
DIRECTIONS = ("outbound", "inbound", "both")
FOLDED_DIRECTION_ERROR = ("folded edges are symmetric (fold_edges_sql emits both orientations); "
                          "use direction='outbound'")

_TIE_BREAK = 'node_id COLLATE "C"'


def _positive(value: int, what: str, cap: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or (cap and value > cap):
        raise ValueError(f"{what} must be an int in 1..{cap or 'inf'}, got {value!r}")
    return value


def _validate_direction(direction: object) -> str:
    """`direction` if it is one of DIRECTIONS (exact, case-sensitive), else ValueError."""
    if not isinstance(direction, str) or direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    return direction


def checked_edge_types(edge_types: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
    """None, or the names sorted (byte order) and de-duplicated, each an identifier; else ValueError.

    The closed list (the mapping's edge type keys) is the gate's check
    (lgk.privacy.gate.served_walk_selection); this one runs in every case, before any SQL.
    """
    if edge_types is None:
        return None
    if isinstance(edge_types, (str, bytes)) or not isinstance(edge_types, (list, tuple)) or not edge_types:
        raise ValueError(f"edge_types must be a non-empty list of edge type names, got {edge_types!r}")
    for name in edge_types:
        try:
            ident.check(name, "edge_types")
        except ident.IdentifierError as e:
            raise ValueError(f"edge_types: {e}") from None
    return tuple(sorted(set(edge_types)))


def _type_filter(edge_types: Optional[Sequence[str]], indent: str, alias: str = "e") -> str:
    """The `AND <alias>.relationship_type IN (...)` line (with its newline), or "" when omitted."""
    names = checked_edge_types(edge_types)
    if names is None:
        return ""
    quoted = ", ".join(f"'{n}'" for n in names)  # one quote style per f-string: parses on Python 3.10
    return f"{indent}AND {alias}.relationship_type IN ({quoted})\n"


# ----------------------------------------------------------------------------
# PRIVACY FLOOR — the single place a walk takes its served count and floor. The rules
# live in lgk.privacy.floor: unfolded edges have no people behind them, so the count
# is structural and there is no floor; folded edges count distinct people (the weight)
# and hide a neighbour below the checked floor.
# ----------------------------------------------------------------------------
def _count_terms(folded: bool, privacy_floor: Optional[int] = None,
                 count_terms: Optional[Tuple[str, str]] = None) -> Tuple[str, str]:
    """(support expression, HAVING clause) for one hop's candidate aggregation."""
    return count_terms if count_terms is not None else walk_count_terms(privacy_floor, folded)


def _score_parts(enrich_schema: Optional[str], sort_method: Optional[str], alias: str,
                 default_sort: str, score_relation: str = "node_score") -> tuple:
    """(join, score expression, sort terms) for the enrichment sort, or `default_sort`.

    `score_relation` is the score relation in `enrich_schema`: the version's `node_score`
    table by default, the served `node_score_view` for a walk over the served schema.
    """
    if sort_method is None:
        return "", "NULL::double precision", default_sort
    schema = ident.check(enrich_schema, "enrich_schema")
    method = ident.check(sort_method, "sort_method")
    relation = ident.check(score_relation, "score_relation")
    join = (f"LEFT JOIN {schema}.{relation} s "
            f"ON s.node_id = {alias}.node_id AND s.method = '{method}'")
    return join, "s.score", "s.score DESC NULLS LAST, "


def _web_candidates(direction: str, hop: int, rel: str, seen: str, count_terms: Tuple[str, str],
                    edge_types: Optional[Sequence[str]] = None, only_unmatched: bool = False) -> str:
    """The derived table `c` (node_id, support) of hop<i>'s unseen candidates, for `direction`.

    outbound/inbound: one join (on src_node_id / dst_node_id) and a GROUP BY on the far end.
    both: the two joins' (node_id, via, relationship_type) rows deduplicated by UNION, so a
    neighbour counts once per (previous-hop node, edge type) whichever way the edge points.
    """
    count, having = count_terms
    prev = f"hop{hop - 1}"
    if direction == "both":
        return (
            f"(SELECT e.node_id, {count} AS support\n"
            f"        FROM (SELECT e.dst_node_id AS node_id, f.node_id AS via, e.relationship_type\n"
            f"              FROM {rel} e JOIN {prev} f ON e.src_node_id = f.node_id\n"
            f"              UNION\n"
            f"              SELECT e.src_node_id AS node_id, f.node_id AS via, e.relationship_type\n"
            f"              FROM {rel} e JOIN {prev} f ON e.dst_node_id = f.node_id) e\n"
            f"        WHERE e.node_id NOT IN ({seen})\n"
            f"{_type_filter(edge_types, '          ')}"
            f"        GROUP BY e.node_id {having}) c"
        )
    near, far = ("src_node_id", "dst_node_id") if direction == "outbound" else ("dst_node_id", "src_node_id")
    # a stop node is kept and seen, but never expanded
    stop = "NOT f.matched\n          AND " if only_unmatched else ""
    return (
        f"(SELECT e.{far} AS node_id, {count} AS support\n"
        f"        FROM {rel} e JOIN {prev} f ON e.{near} = f.node_id\n"
        f"        WHERE {stop}e.{far} NOT IN ({seen})\n"
        f"{_type_filter(edge_types, '          ')}"
        f"        GROUP BY e.{far} {having}) c"
    )


def _web_hop_cte(hop: int, served: str, enrich_schema: Optional[str], sort_method: Optional[str],
                 count_terms: Tuple[str, str], top_n: int, edges: str = "edges_view",
                 direction: str = "outbound", score_relation: str = "node_score",
                 edge_types: Optional[Sequence[str]] = None) -> str:
    """hop<i>: the top_n unseen `direction`-neighbours of hop<i-1>, ordered by (sort, node_id)."""
    seen = " UNION ALL ".join(f"SELECT node_id FROM hop{j}" for j in range(hop))
    join, score, sort = _score_parts(enrich_schema, sort_method, "c", "c.support DESC, ",
                                     score_relation)
    candidates = _web_candidates(direction, hop, f"{served}.{edges}", seen, count_terms, edge_types)
    return (
        f"hop{hop} AS (\n"
        f"  SELECT c.node_id, c.support, {score} AS score\n"
        f"  FROM {candidates}\n"
        f"  {join}\n"
        f"  ORDER BY {sort}c.{_TIE_BREAK}\n"
        f"  LIMIT {top_n})"
    )


def web_sql(depth: int, served_schema: str, top_n: int, enrich_schema: Optional[str] = None,
            sort_method: Optional[str] = None, folded: bool = False,
            privacy_floor: Optional[int] = None,
            count_terms: Optional[Tuple[str, str]] = None, edges_relation: str = "edges_view",
            direction: str = "outbound", score_relation: str = "node_score",
            edge_types: Optional[Sequence[str]] = None) -> str:
    """Rows (hop, node_id, support, score) for hops 1..depth, ordered by hop then the sort.

    `count_terms` (from lgk.privacy.gate.served_count_terms) overrides folded/privacy_floor.
    `edges_relation` is the edge relation in `served_schema`: the served view by default, the
    version's `edges` table for a served function body (lgk.served_functions).
    `score_relation` is the score relation in `enrich_schema` (`node_score` by default; the
    served `node_score_view` when `enrich_schema` is the served schema).

    Sort key: `<score_relation>.score` for `sort_method` (from `enrich_schema`), else the support
    (edges from the previous hop into the node). Ties: node_id ascending, byte order.
    `direction` (DIRECTIONS): outbound, inbound or both. Folded edges are walked outbound only.
    `edge_types`: the edge types followed (checked, sorted, de-duplicated); None = every type.
    """
    _validate_direction(direction)
    checked_edge_types(edge_types)
    if folded and direction != "outbound":
        raise ValueError(FOLDED_DIRECTION_ERROR)
    _positive(depth, "web depth", WEB_MAX_DEPTH)
    _positive(top_n, "top_n")
    served = ident.check(served_schema, "served_schema")
    edges = ident.check(edges_relation, "edges_relation")
    ctes = ["hop0 AS (SELECT CAST(%(start_node_id)s AS text) AS node_id)"]
    terms = _count_terms(folded, privacy_floor, count_terms)
    ctes += [_web_hop_cte(i, served, enrich_schema, sort_method, terms, top_n, edges, direction,
                          score_relation, edge_types)
             for i in range(1, depth + 1)]
    union = "\nUNION ALL\n".join(
        f"SELECT {i} AS hop, node_id, support, score FROM hop{i}" for i in range(1, depth + 1)
    )
    sort = "score DESC NULLS LAST" if sort_method else "support DESC"
    with_clause = ",\n".join(ctes)
    return (f"WITH {with_clause}\n"
            f"SELECT * FROM (\n{union}\n) w\nORDER BY hop, {sort}, {_TIE_BREAK}")


def _tree_candidates(direction: str, rel: str, edge_types: Optional[Sequence[str]] = None) -> str:
    """The inner block of a tree level: the distinct unseen `direction`-neighbours of the frontier.

    A tree serves no count, so `both` is a UNION ALL of the two arms under one DISTINCT, with the
    seen check (the cycle guard) applied to the merged rows.
    """
    if direction == "both":
        types = _type_filter(edge_types, "            ")
        return (
            "        SELECT DISTINCT x.node_id FROM (\n"
            f"          SELECT e.dst_node_id AS node_id FROM {rel} e WHERE e.src_node_id = ANY(w.frontier)\n"
            f"{types}"
            "          UNION ALL\n"
            f"          SELECT e.src_node_id AS node_id FROM {rel} e WHERE e.dst_node_id = ANY(w.frontier)\n"
            f"{types}"
            "        ) x\n"
            "        WHERE NOT (x.node_id = ANY(w.seen))\n"  # cycle guard
        )
    near, far = ("src_node_id", "dst_node_id") if direction == "outbound" else ("dst_node_id", "src_node_id")
    return (
        f"        SELECT DISTINCT e.{far} AS node_id\n"
        f"        FROM {rel} e\n"
        f"        WHERE e.{near} = ANY(w.frontier)\n"
        f"          AND NOT (e.{far} = ANY(w.seen))\n"  # cycle guard
        f"{_type_filter(edge_types, '          ')}"
    )


def tree_walk_cte(depth: int, served_schema: str, top_n: int, enrich_schema: Optional[str] = None,
                  sort_method: Optional[str] = None, edges_relation: str = "edges_view",
                  direction: str = "outbound", score_relation: str = "node_score",
                  edge_types: Optional[Sequence[str]] = None) -> str:
    """The recursive CTE `walk(depth, frontier, seen)` a tree walk reads: one row per level.

    Each level's frontier is the distinct unseen out-neighbours of the previous frontier,
    ordered by (sort, node_id) and LIMITed to `top_n`, so a hub with 10,000
    children never widens a level beyond `top_n`. `seen` holds every node kept so far (the
    seed included): the cycle guard. A node cut from a level is not seen, so it may be kept
    at a deeper level through another parent (the same rule as the web walk's hops).
    `direction` (DIRECTIONS) picks which neighbours a level expands to; nothing else changes.
    `edge_types`: the edge types followed (an unfolded selection); None = every type.
    """
    _validate_direction(direction)
    checked_edge_types(edge_types)
    _positive(depth, "tree depth", TREE_MAX_DEPTH)
    _positive(top_n, "top_n")
    served = ident.check(served_schema, "served_schema")
    edges = ident.check(edges_relation, "edges_relation")
    join, _score, sort = _score_parts(enrich_schema, sort_method, "c", "", score_relation)
    return (
        "WITH RECURSIVE walk(depth, frontier, seen) AS (\n"
        "  SELECT 0, ARRAY[CAST(%(start_node_id)s AS text)], ARRAY[CAST(%(start_node_id)s AS text)]\n"
        "  UNION ALL\n"
        "  SELECT w.depth + 1, n.frontier, w.seen || n.frontier\n"
        "  FROM walk w CROSS JOIN LATERAL (\n"
        "    SELECT array_agg(k.node_id) AS frontier FROM (\n"
        "      SELECT c.node_id FROM (\n"
        f"{_tree_candidates(direction, f'{served}.{edges}', edge_types)}"
        "      ) c\n"
        f"      {join}\n"
        f"      ORDER BY {sort}c.{_TIE_BREAK}\n"
        f"      LIMIT {top_n}\n"  # the per-level cut
        "    ) k\n"
        "  ) n\n"
        f"  WHERE w.depth < {depth} AND n.frontier IS NOT NULL\n"  # depth cap, inside the recursion
        ")"
    )


def tree_sql(depth: int, served_schema: str, top_n: int, enrich_schema: Optional[str] = None,
             sort_method: Optional[str] = None, folded: bool = False,
             privacy_floor: Optional[int] = None, edges_relation: str = "edges_view",
             direction: str = "outbound", score_relation: str = "node_score",
             edge_types: Optional[Sequence[str]] = None) -> str:
    """Rows (node_id, depth, score): nodes reachable in 1..depth hops, at their smallest
    depth, ordered by (depth, sort, node_id) and LIMITed to top_n.

    A breadth-first recursion with ONE row per level (`tree_walk_cte`): each level is cut to
    the top_n by (sort, node_id) before the next level is expanded, and the seen check is
    the cycle guard, so a cycle A->B->C->A stops at C, and each node is visited once: a
    chain of diamonds costs one row per level, not one per path. The depth cap stops the
    recursion itself.
    """
    if folded:
        # A tree visits each node once and serves no count, so there is nothing to floor;
        # walking care-subject (folded) edges as a hierarchy is not supported.
        raise NotImplementedError("tree walks over folded (care-subject) edges are not supported")
    walk = tree_walk_cte(depth, served_schema, top_n, enrich_schema, sort_method, edges_relation,
                         direction, score_relation, edge_types)
    join, score, sort = _score_parts(enrich_schema, sort_method, "t", "", score_relation)
    return (
        f"{walk},\n"
        "reached AS (SELECT u.node_id, w.depth FROM walk w CROSS JOIN LATERAL unnest(w.frontier) "
        "AS u(node_id) WHERE w.depth > 0)\n"
        f"SELECT t.node_id, t.depth, {score} AS score\n"
        "FROM reached t\n"
        f"{join}\n"
        f"ORDER BY t.depth, {sort}t.{_TIE_BREAK}\n"
        f"LIMIT {top_n}"
    )


def similar_sql(vector_table: str, top_n: int, depth: int = 1) -> str:
    """Rows (node_id, distance): the top_n nearest nodes to the seed by cosine distance.

    `vector_table` (schema.table) has columns node_id text and embedding vector(d). The
    seed vector is a scalar subquery, so the HNSW index (`vector_index_sql`) serves the
    order. No privacy floor: a similarity hop aggregates nothing and has no people behind it.
    """
    _positive(depth, "similar depth", 1)
    _positive(top_n, "top_n")
    table = ident.qualified(vector_table, "vector_table", 2, 2)
    return (
        "SELECT v.node_id, v.embedding <=> "
        f"(SELECT embedding FROM {table} WHERE node_id = %(start_node_id)s) AS distance\n"
        f"FROM {table} v\n"
        "WHERE v.node_id <> %(start_node_id)s\n"
        "ORDER BY v.embedding <=> "
        f"(SELECT embedding FROM {table} WHERE node_id = %(start_node_id)s), v.{_TIE_BREAK}\n"
        f"LIMIT {top_n}"
    )


def vector_index_sql(vector_table: str) -> str:
    """The HNSW index (cosine) the similar walk's ORDER BY uses."""
    table = ident.qualified(vector_table, "vector_table", 2, 2)
    name = table.replace(".", "_")[:50] + "_hnsw"
    return f"CREATE INDEX IF NOT EXISTS {name} ON {table} USING hnsw (embedding vector_cosine_ops)"


# ----------------------------------------------------------------------------
# SET WALK: seed sets and a stop predicate, outbound only
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class NodePredicate:
    """Nodes of `node_type`, and, when `field` is set, `<field> = <bound value>`.

    `field`: None (every node of the type), "label", "natural_key", or a walk key declared for
    `node_type` in the mapping (an equality on `properties ->> '<key>'`).
    """
    node_type: str
    field: Optional[str] = None


def _predicate_sql(pred: NodePredicate, alias: str, param: str,
                   walk_keys: Optional[Mapping[str, Sequence[str]]], what: str) -> str:
    """`<alias>.node_type = '<t>'[ AND <field expr> = %(<param>)s]`; ValueError on a type or field
    the mapping does not declare. Type and key are spliced (a partial expression index must see
    the literal key); both passed the identifier pattern, so neither holds a quote."""
    if not isinstance(pred, NodePredicate):
        raise ValueError(f"{what} must be a NodePredicate, got {pred!r}")
    node_type = ident.check(pred.node_type, f"{what} node_type")
    if walk_keys is not None and node_type not in walk_keys:
        raise ValueError(f"{what} node_type {node_type!r} is not a node type of the mapping "
                         f"({sorted(walk_keys)})")
    cond = f"{alias}.node_type = '{node_type}'"
    if pred.field is None:
        return cond
    field = ident.check(pred.field, f"{what} field")
    if field in NODE_FIELDS:
        expr = f"{alias}.{field}"
    elif walk_keys is not None and field in walk_keys.get(node_type, ()):
        expr = f"{alias}.properties ->> '{field}'"
    else:
        declared = sorted(walk_keys.get(node_type, ())) if walk_keys is not None else []
        raise ValueError(f"{what} field {field!r} is not a walk key of node type {node_type!r}: "
                         f"use label, natural_key or a declared walk key ({declared})")
    return f"{cond} AND {expr} = %({param})s"


def _seed_ctes(seed_where: str, nodes: str, max_seeds: int) -> str:
    """seed_all (every seed) and hop0 (the seeds, never matched), behind the seed cap.

    The cap guard is an uncorrelated scalar subquery (an InitPlan, evaluated once) over at most
    max_seeds + 1 rows. Above the cap the CAST of the message raises
    `invalid input syntax for type integer: "lgk: seed set exceeds max_seeds=<n>"`. The CASE
    sits INSIDE the cast, so the planner cannot fold the cast of a constant at plan time."""
    message = f"lgk: seed set exceeds max_seeds={max_seeds}"
    return (
        "seed_all AS (\n"
        "  SELECT CAST(n.node_id AS text) AS node_id\n"
        f"  FROM {nodes} n\n"
        f"  WHERE {seed_where}\n"
        "),\n"
        "hop0 AS (\n"
        "  SELECT s.node_id, false AS matched\n"
        "  FROM seed_all s\n"
        f"  WHERE (SELECT COALESCE(CAST(CASE WHEN count(*) > {max_seeds} THEN '{message}' END AS integer), 0)\n"
        f"         FROM (SELECT 1 FROM seed_all LIMIT {max_seeds + 1}) x) = 0\n"
        ")"
    )


def _web_set_hop_cte(hop: int, served: str, edges: str, nodes: str, enrich_schema: Optional[str],
                     sort_method: Optional[str], count_terms: Tuple[str, str], top_n: int,
                     stop_where: Optional[str], score_relation: str,
                     edge_types: Optional[Sequence[str]]) -> str:
    """hop<i>: the top_n unseen out-neighbours of hop<i-1>'s UNMATCHED nodes, one cut over the
    whole set, then `matched` (the stop predicate) by a primary-key LEFT JOIN after the LIMIT."""
    seen = " UNION ALL ".join(f"SELECT node_id FROM hop{j}" for j in range(hop))
    join, score, sort = _score_parts(enrich_schema, sort_method, "c", "c.support DESC, ",
                                     score_relation)
    candidates = _web_candidates("outbound", hop, f"{served}.{edges}", seen, count_terms,
                                 edge_types, only_unmatched=True)
    if stop_where is None:
        matched, stop_join = "false", ""
    else:
        matched = "(m.node_id IS NOT NULL)"
        stop_join = f"\n  LEFT JOIN {served}.{nodes} m\n    ON m.node_id = k.node_id AND {stop_where}"
    return (
        f"hop{hop} AS (\n"
        f"  SELECT k.node_id, k.support, k.score, {matched} AS matched\n"
        "  FROM (\n"
        f"    SELECT c.node_id, c.support, {score} AS score\n"
        f"    FROM {candidates}\n"
        f"    {join}\n"
        f"    ORDER BY {sort}c.{_TIE_BREAK}\n"
        f"    LIMIT {top_n}) k{stop_join})"
    )


def web_set_sql(depth: int, served_schema: str, top_n: int, seed: NodePredicate,
                stop: Optional[NodePredicate] = None, max_seeds: int = MAX_SEEDS_DEFAULT,
                enrich_schema: Optional[str] = None, sort_method: Optional[str] = None,
                folded: bool = False, privacy_floor: Optional[int] = None,
                count_terms: Optional[Tuple[str, str]] = None,
                walk_keys: Optional[Mapping[str, Sequence[str]]] = None,
                edge_types: Optional[Sequence[str]] = None, edges_relation: str = "edges_view",
                nodes_relation: str = "nodes_view", score_relation: str = "node_score") -> str:
    """Rows (hop, node_id, support, score, matched) for hops 1..depth of the set walk.

    Hop 0 is every node matching `seed` (at most `max_seeds`, else the query raises). Each hop is
    the out-neighbours of the previous hop's unmatched nodes taken together, not seen before,
    aggregated over the whole hop (support: edges in, or MIN(weight) folded, with the floor's
    HAVING), ordered by (sort, node_id) and cut to top_n once. A kept node matching `stop` is
    `matched` and is not expanded; seeds are never tested against `stop`. Without `stop`,
    `matched` is false. `walk_keys` ({node type: keys}, from the mapping) is the closed list of
    node types and property fields; None = only label / natural_key fields.
    `edge_types` / `folded`: as `web_sql` (the gate's selection). Outbound only.
    """
    checked_edge_types(edge_types)
    _positive(depth, "web_set depth", WEB_MAX_DEPTH)
    _positive(top_n, "top_n")
    _positive(max_seeds, "max_seeds", MAX_SEEDS_CAP)
    served = ident.check(served_schema, "served_schema")
    edges = ident.check(edges_relation, "edges_relation")
    nodes = f"{served}.{ident.check(nodes_relation, 'nodes_relation')}"
    seed_where = _predicate_sql(seed, "n", "seed_value", walk_keys, "seed")
    stop_where = None if stop is None else _predicate_sql(stop, "m", "stop_value", walk_keys, "stop")
    terms = _count_terms(folded, privacy_floor, count_terms)
    ctes = [_seed_ctes(seed_where, nodes, max_seeds)]
    ctes += [_web_set_hop_cte(i, served, edges, ident.check(nodes_relation, "nodes_relation"),
                              enrich_schema, sort_method, terms, top_n, stop_where, score_relation,
                              edge_types)
             for i in range(1, depth + 1)]
    union = "\nUNION ALL\n".join(
        f"SELECT {i} AS hop, node_id, support, score, matched FROM hop{i}" for i in range(1, depth + 1)
    )
    sort = "score DESC NULLS LAST" if sort_method else "support DESC"
    joined = ",\n".join(ctes)  # no backslash inside an f-string expression: parses on Python 3.10
    return (f"WITH {joined}\n"
            f"SELECT * FROM (\n{union}\n) w\nORDER BY hop, {sort}, {_TIE_BREAK}")


def walk_key_index_name(node_type: str, key: str) -> str:
    """lgk_nodes_wk_<type>_<key>_idx; above 63 bytes, the first 54 bytes + '_' + 8 hex of its sha256."""
    name = f"lgk_nodes_wk_{ident.check(node_type, 'node_type')}_{ident.check(key, 'walk key')}_idx"
    if len(name.encode()) > 63:
        name = f"{name[:54]}_{hashlib.sha256(name.encode()).hexdigest()[:8]}"
    return name


def walk_key_index_sql(version_schema: str, node_type: str, key: str) -> str:
    """The partial expression index a set walk's seed scan (and stop lookup) on `key` uses."""
    schema = ident.check(version_schema, "version_schema")
    return (f"CREATE INDEX IF NOT EXISTS {walk_key_index_name(node_type, key)}\n"
            f"  ON {schema}.nodes ((properties ->> '{key}')) WHERE node_type = '{node_type}'")


_BUILDERS = {"web": web_sql, "tree": tree_sql, "similar": similar_sql, "web_set": web_set_sql}
_MAX_DEPTH = {"web": WEB_MAX_DEPTH, "tree": TREE_MAX_DEPTH, "similar": 1, "web_set": WEB_MAX_DEPTH}


def walk_sql(shape: str, depth: int, **kwargs) -> str:
    """The SQL for one walk shape at one depth."""
    if shape not in _BUILDERS:
        raise ValueError(f"walk shape must be one of {SHAPES}, got {shape!r}")
    return _BUILDERS[shape](depth=depth, **kwargs)


def per_depth_sql(shape: str, max_depth: int, **kwargs) -> Dict[int, str]:
    """{depth: SQL} for depths 1..max_depth (spec: SQL for every depth of every walk)."""
    _positive(max_depth, f"{shape} max_depth", _MAX_DEPTH.get(shape))
    return {d: walk_sql(shape, d, **kwargs) for d in range(1, max_depth + 1)}
