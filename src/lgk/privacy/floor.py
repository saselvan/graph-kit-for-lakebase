"""The privacy floor: no served count may point to fewer than `floor` people.

A served count is the number of DISTINCT care subjects behind something the website shows.
Care-subject data reaches Lakebase only as folded edges whose weight is that number, and
only when the weight is at or above the floor. The floor defaults to 11. It can be raised
and never lowered.

Edges that carry no care-subject data (a provider directory, an ontology) have no people
behind them, so the floor does not apply to them: the old `HAVING COUNT(*) >= floor`
counted parallel edges (always 1 after dedup) and blanked every such walk.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from lgk import ident
from lgk.mapping import GraphMapping, JoinSpec, NodeSpec

DEFAULT_FLOOR = 11


class FoldFloorError(RuntimeError):
    """A folded edge in the written (Delta or version) `edges` has a weight below the floor, or NULL."""


class JoinFloorError(RuntimeError):
    """A floored join output is set on a node with fewer distinct care subjects than the floor."""


def checked_floor(value: Optional[int]) -> int:
    """The floor to use: DEFAULT_FLOOR when unset, else an int >= DEFAULT_FLOOR."""
    if value is None:
        return DEFAULT_FLOOR
    if isinstance(value, bool) or not isinstance(value, int) or value < DEFAULT_FLOOR:
        raise ValueError(
            f"privacy floor must be an int >= {DEFAULT_FLOOR} (it can be raised, never "
            f"lowered), got {value!r}"
        )
    return value


def fold_edges_sql(source_table: str, subject_col: str, member_col: str,
                   privacy_floor: Optional[int] = None) -> str:
    """SELECT (src_key, dst_key, weight): pairs of members linked by shared care subjects.

    The care-subject key (`subject_col`) of an edge/join table is folded away: two members
    are linked when they share subjects, weighted by the DISTINCT subjects behind the pair,
    and a pair below the floor is not produced at all. Plain ANSI SQL (Spark and Postgres).
    """
    t = ident.qualified(source_table, "source_table")
    s, m = ident.check(subject_col, "subject_col"), ident.check(member_col, "member_col")
    n = checked_floor(privacy_floor)
    return (
        f"SELECT a.{m} AS src_key, b.{m} AS dst_key, COUNT(DISTINCT a.{s}) AS weight "
        f"FROM {t} a JOIN {t} b ON a.{s} = b.{s} AND a.{m} <> b.{m} "
        f"GROUP BY a.{m}, b.{m} "
        f"HAVING COUNT(DISTINCT a.{s}) >= {n}"
    )


def fold_pair_sql(source_table: str, subject_col: str, src_col: str, dst_col: str,
                  privacy_floor: Optional[int] = None) -> str:
    """SELECT (src_key, dst_key, weight): the pair form of a fold (src != dst).

    One row per (src, dst) pair of the join table, weighted by the DISTINCT care subjects
    behind it; a pair with a NULL end, or below the floor, is not produced at all.
    """
    t = ident.qualified(source_table, "source_table")
    s = ident.check(subject_col, "subject_col")
    a, b = ident.check(src_col, "src_col"), ident.check(dst_col, "dst_col")
    n = checked_floor(privacy_floor)
    return (
        f"SELECT {a} AS src_key, {b} AS dst_key, COUNT(DISTINCT {s}) AS weight "
        f"FROM {t} "
        f"WHERE {a} IS NOT NULL AND {b} IS NOT NULL "
        f"GROUP BY {a}, {b} "
        f"HAVING COUNT(DISTINCT {s}) >= {n}"
    )


# ------------------------------------------------------------- the check after a write
FoldReport = Dict[str, Dict[str, Any]]


def _type_literals(fold_types: Sequence[str]) -> str:
    return ", ".join(f"'{ident.check(t, 'fold edge type')}'" for t in sorted(set(fold_types)))


def assert_fold_floor(sql: Callable[[str], List[List[Any]]], fq_edges: str,
                      fold_types: Sequence[str], privacy_floor: Optional[int]) -> FoldReport:
    """Read the Delta `edges` that was written; refuse any folded edge below the floor or NULL.

    A check, not a second filter (build's HAVING is the filter). Runs at build (before the
    stamp), provision (before any synced table) and the refresh sync task (before any update).
    Returns {edge_type: {"floor", "min_weight", "rows"}} (build records it). No fold type: no
    query, nothing to check (an unfolded graph has no people behind any edge).
    """
    n = checked_floor(privacy_floor)
    if not fold_types:
        return {}
    table = ident.qualified(fq_edges, "fq_edges", 3, 3)
    rows = sql(f"SELECT relationship_type, MIN(weight), COUNT_IF(weight IS NULL), COUNT(*) "
               f"FROM {table} WHERE relationship_type IN ({_type_literals(fold_types)}) "
               f"GROUP BY relationship_type")
    report: FoldReport = {t: {"floor": n, "min_weight": None, "rows": 0} for t in sorted(set(fold_types))}
    bad = []
    for edge_type, low, nulls, count in rows:
        low = None if low is None else int(low)
        report[str(edge_type)] = {"floor": n, "min_weight": low, "rows": int(count)}
        if int(nulls) or (low is not None and low < n):
            bad.append(f"{edge_type} (min weight {low}, {int(nulls)} NULL)")
    if bad:
        raise FoldFloorError(
            f"{table} holds folded edges below the privacy floor {n}: {', '.join(bad)}. Nothing "
            f"below the floor is stamped, synced or served; rebuild the folded edges."
        )
    return report


def assert_fold_floor_pg(conn: Any, version_edges: str, fold_types: Sequence[str],
                         privacy_floor: Optional[int]) -> None:
    """The Postgres detector (reapply): no folded row of the version table is NULL or below the floor."""
    n = checked_floor(privacy_floor)
    if not fold_types:
        return
    table = ident.qualified(version_edges, "version_edges", 2, 2)
    count = conn.execute(
        f"SELECT count(*) FROM {table} WHERE relationship_type = ANY(%s) "
        f"AND (weight IS NULL OR weight < %s)", (sorted(set(fold_types)), n)).fetchone()[0]
    if int(count):
        raise FoldFloorError(
            f"{table} holds {int(count)} folded edge(s) with a weight below the privacy floor {n} "
            f"or NULL: a sync bypassed the kit's checks. The job fails; nothing new is served."
        )


def walk_count_terms(privacy_floor: Optional[int], folded: bool) -> Tuple[str, str]:
    """(served count expression, HAVING clause) for a one-hop walk over `edges_view e`.

    Folded edges: the count is the edge weight (distinct people) and the HAVING hides any
    neighbour below the floor; a NULL weight fails the HAVING, so it is hidden, never served
    unweighted. Unfolded edges: a structural edge count, no people behind it, no floor.
    """
    n = checked_floor(privacy_floor)
    if folded:
        return "MIN(e.weight)", f"HAVING MIN(e.weight) >= {n}"
    return "COUNT(*)", ""


# ------------------------------------------------------------ node joins
SUBJECTS_COLUMN = "_lgk_subjects"  # the per-node distinct care-subject count of a floored join


def join_output_expr(alias: str, output: str, floored: bool, privacy_floor: Optional[int]) -> str:
    """One join output as a STRING map value: NULL below the floor on a floored join.

    "Below the floor" and "no joined rows" are both NULL, on purpose."""
    a, o = ident.check(alias, "join alias"), ident.check(output, "join output")
    value = f"CAST({a}.{o} AS STRING)"
    if not floored:
        return value
    return f"CASE WHEN {a}.{SUBJECTS_COLUMN} >= {checked_floor(privacy_floor)} THEN {value} END"


JoinReport = Dict[str, Dict[str, Any]]


def join_report(mapping: Optional[GraphMapping], privacy_floor: Optional[int]) -> JoinReport:
    """{"<node_type>.<output>": {"floor", "floored"}} for every join output of the mapping."""
    n = checked_floor(privacy_floor)
    if mapping is None:
        return {}
    return {f"{t}.{name}": {"floor": n, "floored": j.subject is not None}
            for t, spec in sorted(mapping.nodes.items()) for j in spec.joins for name in j.outputs}


def node_key_expr(natural_key: Union[str, List[str]], alias: Optional[str] = None) -> str:
    """SQL for a node's natural key, its columns qualified with `alias.` when given.

    The ONE key expression of the kit: build's node ids and joins and assert_join_floor all use
    it, so the floor check finds a node exactly the way build made and joined it.
    Single column: the column itself (unchanged formula, so existing node ids do not move).
    Composite: a JSON array of the parts as strings, e.g. ["a",null,"b"]. JSON quoting makes
    the encoding unambiguous (a part containing '::' or '"' cannot collide), and a NULL part
    is kept as JSON null at its position (the documented sentinel), so ("a",NULL,"b") and
    ("a","b",NULL) get different node ids.
    """
    q = (lambda c: f"{alias}.{c}") if alias else (lambda c: c)
    if isinstance(natural_key, str):
        return q(natural_key)
    return "to_json(array(" + ", ".join(f"CAST({q(c)} AS STRING)" for c in natural_key) + "))"


def node_id_expr(node_type: str, natural_key: Union[str, List[str]], alias: Optional[str] = None) -> str:
    """xxhash64('<type>::'||<key>): a node's id, from its source row (build and the floor check)."""
    return f"xxhash64('{ident.check(node_type, 'node type')}::'||{node_key_expr(natural_key, alias)})"


def join_on(join_alias: str, natural_key: Union[str, List[str]], node_alias: str = "n") -> str:
    """The ON condition of a node join: the joined `_lgk_key` equals the node's key value.

    Build's LEFT JOIN and assert_join_floor's JOIN both use this, so a value-equal key of
    another rendering cannot put an aggregate on a node that the check then fails to find."""
    return f"{ident.check(join_alias, 'join alias')}._lgk_key = {node_key_expr(natural_key, node_alias)}"


def _join_floor_sql(fq_nodes: str, node_source: str, node_type: str, spec: NodeSpec,
                    join_source: str, join: JoinSpec, privacy_floor: int) -> str:
    """COUNT of `node_type` nodes whose floored outputs are set though the node has fewer distinct
    subjects than the floor.

    The subject count is recomputed here, not read from the build. The node is found the way
    build found it: its source row joined to the subject counts on `join_on`, its id from
    `node_id_expr` over that same row, never a hash re-derived from the joined key.

    Fail closed: a node with a floored output set counts unless it matches a source row whose
    subject count is at or above the floor. So a node with no matching source row (a `nodes`
    written by an operator pipeline with another id encoding, or a source row deleted since)
    or with no joined rows is a leak, never a silent 0."""
    k, s = ident.check(join.key, "join key"), ident.check(join.subject, "join subject")
    t = ident.check(node_type, "node type")
    set_any = " OR ".join(f"d.properties['{ident.check(o, 'join output')}'] IS NOT NULL"
                          for o in join.outputs)
    return (f"SELECT COUNT(DISTINCT d.node_id) FROM {fq_nodes} d "
            f"LEFT JOIN (SELECT CAST({node_id_expr(t, spec.natural_key, 'n')} AS STRING) AS node_id, j.s "
            f"FROM {node_source} n "
            f"LEFT JOIN (SELECT {k} AS _lgk_key, COUNT(DISTINCT {s}) AS s FROM {join_source} "
            f"WHERE {k} IS NOT NULL GROUP BY {k}) j ON {join_on('j', spec.natural_key, 'n')}) b "
            f"ON b.node_id = d.node_id "
            f"WHERE d.node_type = '{t}' AND ({set_any}) AND (b.s IS NULL OR b.s < {privacy_floor})")


def assert_join_floor(sql: Callable[[str], List[List[Any]]], fq_nodes: str,
                      mapping: Optional[GraphMapping], privacy_floor: Optional[int]) -> JoinReport:
    """Read the Delta `nodes` that was written; refuse a floored join output on a node below the floor.

    A check, not a second filter (build's CASE is the filter). Runs at build (before the stamp),
    provision (before any synced table) and the refresh sync task (before any update). It cannot
    run in Postgres: the subject never reaches it. Returns `join_report` (build records it).
    """
    n = checked_floor(privacy_floor)
    report = join_report(mapping, n)
    if not report:
        return report
    nodes = ident.qualified(fq_nodes, "fq_nodes", 3, 3)
    schema = f"{ident.check(mapping.source_catalog, 'source_catalog')}.{ident.check(mapping.source_schema, 'source_schema')}"
    bad = []
    for node_type, spec in sorted(mapping.nodes.items()):
        for i, join in enumerate(spec.joins):
            if join.subject is None:
                continue
            node_source = f"{schema}.{ident.check(spec.source_table, 'source_table')}"
            source = f"{schema}.{ident.check(join.table, 'join table')}"
            count = int(sql(_join_floor_sql(nodes, node_source, node_type, spec, source, join, n))[0][0])
            if count:
                bad.append(f"{node_type}.joins[{i}] ({join.table}: {count} node(s))")
    if bad:
        raise JoinFloorError(
            f"{nodes} holds join outputs for nodes below the privacy floor {n}: {', '.join(bad)}. "
            f"Nothing below the floor is stamped, synced or served; rebuild the nodes.")
    return report
