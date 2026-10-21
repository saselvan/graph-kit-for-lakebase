"""The function read surface: two front doors, one graph.

Two LOGIN roles read the graph only through SECURITY DEFINER functions, never a table:
- `public_reader`: EXECUTE on the public directory functions (find-a-doctor lookups by
  provider, location, affiliation, specialty and accepting-new-patients). They return only
  the public directory columns below, read only provider nodes and affiliation edges, and
  never return the properties map, an edge weight or a node of another type.
- `coordinator_reader`: EXECUTE on the coordinator functions, the privacy-floored walks
  (lgk.walks.sql bodies, served count and floor from lgk.privacy.gate.served_count_terms).
  `coordinator_web` takes `p_edge_type` (one edge type per call, matched against the
  mapping's edge type literals in static IF branches, never spliced): a fold type walks people
  counts (outbound only), an unfolded type edge counts; NULL walks every type, refused on a graph
  with folded and unfolded edge types. Multi-type selections are the view form only.
  `coordinator_web_set` is the set walk: `p_seed_key` / `p_stop_key` are
  '<node_type>.<walk_key>' strings matched against the mapping's declared walk keys in static IF
  branches (never spliced), `p_stop_key` NULL = no stop; served only when the mapping declares
  walk keys: coordinator_reader may test equality on them; public_reader never.

Lifecycle (one builder, so the switch, the post-refresh re-apply and the rollback are the
same script):
- provision STAGES every function in the new version schema (`<kit>_v<N>.<fn>`), owned by
  the operator, EXECUTE revoked from PUBLIC and granted to nobody, and calls each one once:
  the version compiles before it can be served, and the isolation probe proves no role and
  not PUBLIC can execute a staged copy.
- switch runs `CREATE OR REPLACE FUNCTION <served>.<fn>` for every function, each body a
  static string naming the target version's tables, in the SAME transaction as the views,
  then the grant block (REVOKE ALL ... FROM PUBLIC before every GRANT).
Bodies are string bodies (plpgsql / sql), never BEGIN ATOMIC, so no pg_depend row binds a
served function to a version table: a full snapshot refresh or a switch never drops them.
Every function pins `search_path` to pg_catalog, pg_temp and names every relation by
schema, so a caller cannot hijack a name or an operator through the search path.
Caller input never builds SQL: keys, limits and depths are parameters of static SQL.

The `similar` walk is not served as a function because it needs a vector table.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lgk.walks.sql import (
    DIRECTIONS,
    FOLDED_DIRECTION_ERROR,
    TREE_MAX_DEPTH,
    WEB_MAX_DEPTH,
    NodePredicate,
    tree_sql,
    web_set_sql,
    web_sql,
)

PUBLIC_ROLE = "public_reader"
COORDINATOR_ROLE = "coordinator_reader"
FUNCTION_ROLES = (PUBLIC_ROLE, COORDINATOR_ROLE)

SEARCH_PATH = "pg_catalog, pg_temp"
DEFAULT_TOP_N = 25  # the same default as the walks step and advise
SERVED_TREE_DEPTH = 10  # tree depths 1..10 are served (lgk.walks.sql caps at TREE_MAX_DEPTH)
MAX_DIRECTORY_ROWS = 100  # a directory lookup never returns more than this many rows

# The directory is the provider node type of the mapping, and its affiliation edges.
DIRECTORY_NODE_TYPE = "provider"
DIRECTORY_EDGE_TYPE = "affiliation"
# Public directory properties (nodes.properties keys) and the real system each comes from.
# The kit's provider fixture carries them as illustrative values.
DIRECTORY_PROPERTIES = (
    "specialty",               # credentialing system (board certification / privileges)
    "location",                # website CMS (the practice location shown on find-a-doctor)
    "accepting_new_patients",  # Epic provider record (Provider Master File, scheduling flag)
)
# Every column a public directory function may return. Nothing else is public.
PUBLIC_COLUMNS = ("provider_key", "provider_name") + DIRECTORY_PROPERTIES + (
    "affiliated_key", "affiliated_name")


@dataclass(frozen=True)
class ServedFunction:
    name: str
    args: str  # "p_key text, p_limit integer"
    returns: str  # "TABLE (...)"
    language: str
    body: str  # static SQL naming one version schema
    role: str  # the one role granted EXECUTE
    smoke_args: Tuple[str, ...]  # provision's smoke calls, one SQL argument list each: "'', 1"

    def __post_init__(self) -> None:
        # a staged function nobody calls could fail only once it is live
        if not self.smoke_args:
            raise ValueError(f"{self.name} has no smoke call")

    @property
    def arg_types(self) -> str:
        """Identity argument types, as oidvectortypes prints them (no names, no DEFAULT)."""
        return ", ".join(a.split(" DEFAULT ", 1)[0].split()[-1] for a in self.args.split(", ")) if self.args else ""

    def signature(self, schema: str) -> str:
        return f"{schema}.{self.name}({self.arg_types})"


# ----------------------------------------------------------------------------
# Public directory functions
# ----------------------------------------------------------------------------
_DIRECTORY_RETURNS = ("TABLE (provider_key text, provider_name text, specialty text, "
                      "location text, accepting_new_patients text)")
_LOOKUP_SMOKE = ("'', 1",)  # an empty key and a limit of 1: the body's tables resolve


def _directory_select(version: str, where: str) -> str:
    """Provider rows with ONLY the public columns; the properties map itself is never returned."""
    props = ", ".join(f"n.properties ->> '{p}'" for p in DIRECTORY_PROPERTIES)
    return (f"SELECT n.natural_key, n.label, {props}\n"
            f"FROM {version}.nodes n\n"
            f"WHERE n.node_type = '{DIRECTORY_NODE_TYPE}' AND {where}\n"
            f"ORDER BY n.natural_key COLLATE \"C\"\n"
            f"LIMIT LEAST(GREATEST($2, 0), {MAX_DIRECTORY_ROWS})")


def _directory_functions(version: str) -> List[ServedFunction]:
    lookup = "p_value text, p_limit integer"
    by_property = [
        ServedFunction(f"directory_by_{p}", lookup, _DIRECTORY_RETURNS, "sql",
                       _directory_select(version, f"n.properties ->> '{p}' = $1"), PUBLIC_ROLE,
                       _LOOKUP_SMOKE)
        for p in ("specialty", "location")
    ]
    accepting = ServedFunction(
        "directory_accepting_new_patients", lookup, _DIRECTORY_RETURNS, "sql",
        _directory_select(version, "n.properties ->> 'specialty' = $1 "
                                   "AND n.properties ->> 'accepting_new_patients' = 'true'"),
        PUBLIC_ROLE, _LOOKUP_SMOKE)
    provider = ServedFunction(
        "directory_provider", lookup, _DIRECTORY_RETURNS, "sql",
        _directory_select(version, "n.natural_key = $1"), PUBLIC_ROLE, _LOOKUP_SMOKE)
    affiliation = ServedFunction(
        "directory_affiliations", lookup,
        "TABLE (provider_key text, affiliated_key text, affiliated_name text)", "sql",
        (f"SELECT s.natural_key, d.natural_key, d.label\n"
         f"FROM {version}.edges e\n"
         f"JOIN {version}.nodes s ON s.node_id = e.src_node_id AND s.node_type = '{DIRECTORY_NODE_TYPE}'\n"
         f"JOIN {version}.nodes d ON d.node_id = e.dst_node_id AND d.node_type = '{DIRECTORY_NODE_TYPE}'\n"
         f"WHERE e.relationship_type = '{DIRECTORY_EDGE_TYPE}' AND s.natural_key = $1\n"
         f"ORDER BY d.natural_key COLLATE \"C\"\n"
         f"LIMIT LEAST(GREATEST($2, 0), {MAX_DIRECTORY_ROWS})"),
        PUBLIC_ROLE, _LOOKUP_SMOKE)
    return [provider, *by_property, accepting, affiliation]


# ----------------------------------------------------------------------------
# Coordinator functions: the privacy-floored walks
# ----------------------------------------------------------------------------
def _walk_dispatch(directions: Sequence[str], depths: Sequence[int], query_for, columns: str,
                   order: str) -> str:
    """plpgsql (the tree): one static RETURN QUERY per (direction, depth); anything else raises
    (no dynamic SQL). The web has its own per-edge-type builder (`_web_body`)."""
    checks = (f"  IF p_direction IS NULL OR p_direction NOT IN ({_sql_list(DIRECTIONS)}) THEN\n"
              f"    RAISE EXCEPTION 'direction must be one of ({', '.join(DIRECTIONS)}), got %', p_direction;\n"
              "  END IF;\n")
    pairs = [(r, d) for r in directions for d in depths]
    branches = "\n".join(
        f"  {'IF' if i == 0 else 'ELSIF'} p_direction = '{r}' AND p_depth = {d} THEN\n"
        f"    RETURN QUERY SELECT {columns} FROM (\n{query_for(r, d)}\n    ) w ORDER BY {order};"
        for i, (r, d) in enumerate(pairs)
    )
    return ("#variable_conflict use_column\nBEGIN\n"
            f"{checks}{branches}\n"
            "  ELSE\n"
            f"    RAISE EXCEPTION 'depth must be in 1..{max(depths)}, got %', p_depth;\n"
            "  END IF;\nEND")


def _sql_list(values: Sequence[str]) -> str:
    return ", ".join(f"'{v}'" for v in values)


def _param(sql: str) -> str:
    """The walk SQL's one psycopg parameter becomes the function's first argument."""
    return sql.replace("%(start_node_id)s", "$1")


def _walk_smoke(directions: Sequence[str], depths: Sequence[int]) -> Tuple[str, ...]:
    """One call per served (direction, depth) (an empty start node): every branch compiles."""
    return tuple(f"'', {d}, '{r}'" for r in directions for d in depths)


@dataclass(frozen=True)
class WebSelection:
    """One p_edge_type value coordinator_web serves: NULL (`edge_type` None) or one edge type."""
    edge_type: Optional[str]
    edge_types: Optional[Tuple[str, ...]]  # the walk's IN list; None = every type (no IN line)
    folded: bool
    count_terms: Tuple[str, str]

    @property
    def condition(self) -> str:
        return "p_edge_type IS NULL" if self.edge_type is None else f"p_edge_type = '{self.edge_type}'"

    @property
    def directions(self) -> Tuple[str, ...]:
        # folded edges are symmetric: a folded selection serves outbound only (lgk.walks.sql.web_sql)
        return ("outbound",) if self.folded else DIRECTIONS

    def smoke_args(self, depths: Sequence[int]) -> Tuple[str, ...]:
        """One call per (direction, depth); the NULL selection uses the defaulted p_edge_type."""
        calls = _walk_smoke(self.directions, depths)
        return calls if self.edge_type is None else tuple(f"{c}, '{self.edge_type}'" for c in calls)


_WEB_ARGS = ("p_start_node_id text, p_depth integer, p_direction text DEFAULT 'outbound', "
             "p_edge_type text DEFAULT NULL")
_WEB_COLUMNS = "w.hop::integer, w.node_id::text, w.support::bigint, w.score::double precision"
_WEB_ORDER = 'w.hop, w.support DESC, w.node_id COLLATE "C"'


def _quoted(message: str) -> str:
    return message.replace(chr(39), chr(39) * 2)


def _web_checks(selections: Sequence[WebSelection], typed: Sequence[str], mixed_error: str) -> str:
    """The refusals before any branch: direction, edge type (closed list), NULL on a mixed graph,
    a folded selection walked other than outbound."""
    checks = (f"  IF p_direction IS NULL OR p_direction NOT IN ({_sql_list(DIRECTIONS)}) THEN\n"
              f"    RAISE EXCEPTION 'direction must be one of ({', '.join(DIRECTIONS)}), got %', p_direction;\n"
              "  END IF;\n")
    allowed = f"p_edge_type NOT IN ({_sql_list(typed)})" if typed else "TRUE"
    checks += (f"  IF p_edge_type IS NOT NULL AND {allowed} THEN\n"
               f"    RAISE EXCEPTION 'edge_type must be one of ({', '.join(typed)}), got %', p_edge_type;\n"
               "  END IF;\n")
    if mixed_error:
        checks += (f"  IF p_edge_type IS NULL THEN\n    RAISE EXCEPTION '{_quoted(mixed_error)}';\n"
                   "  END IF;\n")
    folded = [s.condition for s in selections if s.folded]
    if folded:
        checks += (f"  IF ({' OR '.join(folded)}) AND p_direction <> 'outbound' THEN\n"
                   f"    RAISE EXCEPTION '{_quoted(FOLDED_DIRECTION_ERROR)}';\n"
                   "  END IF;\n")
    return checks


def _web_body(version: str, top_n: int, selections: Sequence[WebSelection], typed: Sequence[str],
              mixed_error: str, depths: Sequence[int]) -> str:
    """plpgsql: one static RETURN QUERY per (selection, direction, depth); anything else raises."""
    def branch_sql(sel: WebSelection, r: str, d: int) -> str:
        return _param(web_sql(d, version, top_n, folded=sel.folded, count_terms=sel.count_terms,
                              edges_relation="edges", direction=r, edge_types=sel.edge_types))

    triples = [(sel, r, d) for sel in selections for r in sel.directions for d in depths]
    branches = "\n".join(
        f"  {'IF' if i == 0 else 'ELSIF'} {sel.condition + ' AND ' if typed else ''}"
        f"p_direction = '{r}' AND p_depth = {d} THEN\n"
        f"    RETURN QUERY SELECT {_WEB_COLUMNS} FROM (\n{branch_sql(sel, r, d)}\n    ) w ORDER BY {_WEB_ORDER};"
        for i, (sel, r, d) in enumerate(triples)
    )
    return ("#variable_conflict use_column\nBEGIN\n"
            f"{_web_checks(selections, typed, mixed_error)}{branches}\n"
            "  ELSE\n"
            f"    RAISE EXCEPTION 'depth must be in 1..{max(depths)}, got %', p_depth;\n"
            "  END IF;\nEND")


def _web_selections(count_terms: Tuple[str, str], folded: bool,
                    typed: Sequence[Tuple[str, Tuple[str, str], bool]],
                    mixed: bool) -> List[WebSelection]:
    """The NULL selection (unless the graph is mixed), then one per edge type of the mapping."""
    out = []
    if not mixed:
        every = tuple(t for t, _terms, _f in typed) if folded and typed else None
        out.append(WebSelection(None, every, folded, count_terms))
    out += [WebSelection(t, (t,), f, terms) for t, terms, f in typed]
    return out


def _coordinator_web(version: str, count_terms: Tuple[str, str], folded: bool, top_n: int,
                     typed: Sequence[Tuple[str, Tuple[str, str], bool]] = (),
                     mixed_error: str = "") -> ServedFunction:
    web_depths = range(1, WEB_MAX_DEPTH + 1)
    selections = _web_selections(count_terms, folded, typed, bool(mixed_error))
    names = [t for t, _terms, _f in typed]
    return ServedFunction(
        "coordinator_web", _WEB_ARGS,
        "TABLE (hop integer, node_id text, support bigint, score double precision)", "plpgsql",
        _web_body(version, top_n, selections, names, mixed_error, web_depths),
        COORDINATOR_ROLE, tuple(a for sel in selections for a in sel.smoke_args(web_depths)))


# ----------------------------------------------------------------------------
# coordinator_web_set: the set walk over declared walk keys
# ----------------------------------------------------------------------------
WEB_SET_NAME = "coordinator_web_set"
_WEB_SET_ARGS = ("p_seed_key text, p_seed_value text, p_stop_key text, p_stop_value text, "
                 "p_depth integer, p_edge_type text DEFAULT NULL")
_WEB_SET_RETURNS = ("TABLE (hop integer, node_id text, support bigint, score double precision, "
                    "matched boolean)")
_WEB_SET_COLUMNS = _WEB_COLUMNS + ", w.matched::boolean"


def web_set_keys(mapping: Any) -> List[str]:
    """'<node_type>.<walk_key>' for every declared walk key, sorted (byte order)."""
    if mapping is None:
        return []
    return sorted(f"{t}.{k}" for t, n in mapping.nodes.items() for k in n.walk_keys)


def _key_predicate(key: str) -> NodePredicate:
    node_type, field = key.split(".", 1)
    return NodePredicate(node_type, field)


def _web_set_param(sql: str) -> str:
    """The set walk's bound values become the function's arguments ($2 seed, $4 stop)."""
    return sql.replace("%(seed_value)s", "$2").replace("%(stop_value)s", "$4")


def _web_set_checks(keys: Sequence[str], typed: Sequence[str], mixed_error: str) -> str:
    """The refusals before any branch: seed and stop keys (closed lists), edge type, NULL type on a
    mixed graph."""
    listed = _sql_list(keys)
    checks = (f"  IF p_seed_key IS NULL OR p_seed_key NOT IN ({listed}) THEN\n"
              f"    RAISE EXCEPTION 'seed_key must be one of ({', '.join(keys)}), got %', p_seed_key;\n"
              "  END IF;\n"
              f"  IF p_stop_key IS NOT NULL AND p_stop_key NOT IN ({listed}) THEN\n"
              f"    RAISE EXCEPTION 'stop_key must be NULL or one of ({', '.join(keys)}), got %', p_stop_key;\n"
              "  END IF;\n")
    allowed = f"p_edge_type NOT IN ({_sql_list(typed)})" if typed else "TRUE"
    checks += (f"  IF p_edge_type IS NOT NULL AND {allowed} THEN\n"
               f"    RAISE EXCEPTION 'edge_type must be one of ({', '.join(typed)}), got %', p_edge_type;\n"
               "  END IF;\n")
    if mixed_error:
        checks += (f"  IF p_edge_type IS NULL THEN\n    RAISE EXCEPTION '{_quoted(mixed_error)}';\n"
                   "  END IF;\n")
    return checks


def _web_set_branches(version: str, top_n: int, selections: Sequence[WebSelection], keys: Sequence[str],
                      walk_keys: Dict[str, Tuple[str, ...]], depths: Sequence[int]
                      ) -> List[Tuple[WebSelection, str, Optional[str], int, str]]:
    """(selection, seed key, stop key or None, depth, static SQL) for every served combination."""
    out = []
    for sel in selections:
        for seed in keys:
            for stop in [None, *keys]:
                for d in depths:
                    sql = web_set_sql(d, version, top_n, _key_predicate(seed),
                                      None if stop is None else _key_predicate(stop),
                                      folded=sel.folded, count_terms=sel.count_terms, walk_keys=walk_keys,
                                      edge_types=sel.edge_types, edges_relation="edges",
                                      nodes_relation="nodes")
                    out.append((sel, seed, stop, d, _web_set_param(sql)))
    return out


def _web_set_body(branches: Sequence[Tuple[WebSelection, str, Optional[str], int, str]],
                  keys: Sequence[str], typed: Sequence[str], mixed_error: str, depths: Sequence[int]) -> str:
    """plpgsql: one static RETURN QUERY per (selection, seed key, stop key, depth); anything else raises."""
    def cond(sel: WebSelection, seed: str, stop: Optional[str], d: int) -> str:
        stop_cond = "p_stop_key IS NULL" if stop is None else f"p_stop_key = '{stop}'"
        return (f"{sel.condition + ' AND ' if typed else ''}p_seed_key = '{seed}' AND {stop_cond} "
                f"AND p_depth = {d}")

    lines = "\n".join(
        f"  {'IF' if i == 0 else 'ELSIF'} {cond(sel, seed, stop, d)} THEN\n"
        f"    RETURN QUERY SELECT {_WEB_SET_COLUMNS} FROM (\n{sql}\n    ) w ORDER BY {_WEB_ORDER};"
        for i, (sel, seed, stop, d, sql) in enumerate(branches)
    )
    return ("#variable_conflict use_column\nBEGIN\n"
            f"{_web_set_checks(keys, typed, mixed_error)}{lines}\n"
            "  ELSE\n"
            f"    RAISE EXCEPTION 'depth must be in 1..{max(depths)}, got %', p_depth;\n"
            "  END IF;\nEND")


def _web_set_smoke(branches: Sequence[Tuple[WebSelection, str, Optional[str], int, str]]) -> Tuple[str, ...]:
    """One call per branch (empty values): every branch's tables resolve on the version."""
    def call(sel: WebSelection, seed: str, stop: Optional[str], d: int) -> str:
        stop_args = "NULL, NULL" if stop is None else f"'{stop}', ''"
        edge = "" if sel.edge_type is None else f", '{sel.edge_type}'"
        return f"'{seed}', '', {stop_args}, {d}{edge}"

    return tuple(call(sel, seed, stop, d) for sel, seed, stop, d, _sql in branches)


def _coordinator_web_set(version: str, count_terms: Tuple[str, str], folded: bool, top_n: int,
                         walk_keys: Dict[str, Tuple[str, ...]],
                         typed: Sequence[Tuple[str, Tuple[str, str], bool]] = (),
                         mixed_error: str = "") -> ServedFunction:
    """The set walk function: (seed keys) x (stop keys + NULL) x 4 depths x edge-type selections."""
    depths = range(1, WEB_MAX_DEPTH + 1)
    keys = sorted(f"{t}.{k}" for t, ks in walk_keys.items() for k in ks)
    selections = _web_selections(count_terms, folded, typed, bool(mixed_error))
    names = [t for t, _terms, _f in typed]
    branches = _web_set_branches(version, top_n, selections, keys, walk_keys, depths)
    return ServedFunction(WEB_SET_NAME, _WEB_SET_ARGS, _WEB_SET_RETURNS, "plpgsql",
                          _web_set_body(branches, keys, names, mixed_error, depths),
                          COORDINATOR_ROLE, _web_set_smoke(branches))


def _coordinator_functions(version: str, count_terms: Tuple[str, str], folded: bool,
                           top_n: int, typed: Sequence[Tuple[str, Tuple[str, str], bool]] = (),
                           mixed_error: str = "") -> List[ServedFunction]:
    """coordinator_web (per edge type when the mapping is known), and the tree on an unfolded graph.

    `typed`: (edge type, count terms, folded) per mapping edge type. `mixed_error`: set on a graph
    with folded and unfolded types, where a NULL p_edge_type is refused and no tree is served."""
    args = "p_start_node_id text, p_depth integer, p_direction text DEFAULT 'outbound'"
    tree_depths = range(1, min(SERVED_TREE_DEPTH, TREE_MAX_DEPTH) + 1)
    web = _coordinator_web(version, count_terms, folded, top_n, typed, mixed_error)
    if folded or mixed_error:
        # A tree walk serves no count, so it cannot be floored (lgk.walks.sql.tree_sql).
        return [web]
    tree = ServedFunction(
        "coordinator_tree", args,
        "TABLE (node_id text, depth integer, score double precision)", "plpgsql",
        _walk_dispatch(
            DIRECTIONS, tree_depths,
            lambda r, d: _param(tree_sql(d, version, top_n, edges_relation="edges", direction=r)),
            "w.node_id::text, w.depth::integer, w.score::double precision",
            'w.depth, w.node_id COLLATE "C"'),
        COORDINATOR_ROLE, _walk_smoke(DIRECTIONS, tree_depths))
    return [web, tree]


def served_functions(config: Dict[str, Any], version: str) -> List[ServedFunction]:
    """Every served function, with bodies naming `version`; count terms from the privacy gate.

    The mapping's edge types give coordinator_web one branch set per type, each type's folded
    from the gate's fold edge types; on a mixed graph the NULL type is refused."""
    from lgk.privacy.floor import walk_count_terms
    from lgk.privacy.gate import fold_edge_types, read_run_mapping, served_walk_policy, served_walk_selection

    run_folder, asked = config.get("run_folder"), config.get("privacy_floor")
    mapping = read_run_mapping(run_folder) if run_folder else None
    types = sorted(mapping.edges) if mapping is not None else []
    folds = set(fold_edge_types(mapping))
    unfolded = [t for t in types if t not in folds]
    mixed_error = ""
    if folds and unfolded:
        mixed_error = (f"edge_types is required on a graph with folded and unfolded edge types: "
                       f"folded {sorted(folds)}, unfolded {unfolded}")
        floor, folded = served_walk_selection(run_folder, asked, None, [types[0]])[0], False
    else:
        floor, folded = served_walk_policy(run_folder, asked, None)
    typed = [(t, walk_count_terms(floor, t in folds), t in folds) for t in types]
    top_n = config.get("top_n", DEFAULT_TOP_N)
    fns = _kit_functions(version, walk_count_terms(floor, folded), folded, top_n, typed, mixed_error)
    if web_set_keys(mapping):  # served only when the mapping declares walk keys
        keys = {t: tuple(n.walk_keys) for t, n in mapping.nodes.items()}
        fns.append(_coordinator_web_set(version, walk_count_terms(floor, folded), folded, top_n, keys,
                                        typed, mixed_error))
    return fns


def servable_functions(version: str) -> List[ServedFunction]:
    """Every function this module can serve (an unfolded graph serves them all), without the gate:
    for their signatures and roles only (expected grants, the drop of a name the target lacks)."""
    from lgk.privacy.floor import walk_count_terms

    return _kit_functions(version, walk_count_terms(None, False), False, DEFAULT_TOP_N)


def _kit_functions(version: str, count_terms: Tuple[str, str], folded: bool, top_n: int,
                   typed: Sequence[Tuple[str, Tuple[str, str], bool]] = (),
                   mixed_error: str = "") -> List[ServedFunction]:
    """The one builder: served_functions and servable_functions differ only in the gate's terms."""
    return _directory_functions(version) + _coordinator_functions(version, count_terms, folded, top_n,
                                                                  typed, mixed_error)


# ----------------------------------------------------------------------------
# SQL
# ----------------------------------------------------------------------------
def create_sql(fn: ServedFunction, schema: str) -> str:
    """CREATE OR REPLACE ... SECURITY DEFINER with a pinned search_path and a string body."""
    return (f"CREATE OR REPLACE FUNCTION {schema}.{fn.name}({fn.args})\n"
            f"RETURNS {fn.returns}\nLANGUAGE {fn.language} STABLE SECURITY DEFINER\n"
            f"SET search_path = {SEARCH_PATH}\n"
            f"AS $lgkfn$\n{fn.body}\n$lgkfn$")


def grant_sql(fn: ServedFunction, schema: str) -> List[str]:
    """The grant block: everything revoked from PUBLIC (the default EXECUTE) before the one GRANT."""
    sig = fn.signature(schema)
    return [f"REVOKE ALL ON FUNCTION {sig} FROM PUBLIC",
            f"GRANT EXECUTE ON FUNCTION {sig} TO {fn.role}"]


def switch_statements(config: Dict[str, Any], served: str, target: str, *,
                      present: Sequence[str]) -> List[str]:
    """Served functions repointed at `target`, then the grant block. Runs inside the view switch.

    `present`: the kit-named functions now in the served schema (`staged_signatures(conn, served)`,
    read by provision.serve right before the switch).
    """
    fns = served_functions(config, target)
    stmts = [f"GRANT USAGE ON SCHEMA {served} TO {', '.join(FUNCTION_ROLES)}"]
    stmts += [drop_sql(sig) for sig in _stale_signatures(fns, served, present)]
    stmts += [create_sql(fn, served) for fn in fns]
    for fn in fns:
        stmts += grant_sql(fn, served)
    return stmts


def _stale_signatures(fns: Sequence[ServedFunction], served: str, present: Sequence[str]) -> List[str]:
    """Every kit function in `served` that the target does not serve with this exact signature.

    A function the target does not serve (a folded graph has no tree), and an overload of a
    served name with other identity arguments (CREATE OR REPLACE with a new argument list makes
    a second overload), must not stay behind, still granted and still naming the previous version.
    """
    serving = {fn.signature(served) for fn in fns}
    names = {fn.name for fn in fns}
    unserved = [fn.signature(served) for fn in servable_functions(served) if fn.name not in names]
    kit = {f"{served}.{name}" for name in FUNCTION_NAMES}
    overloads = [sig for sig in present if sig.split("(", 1)[0] in kit]  # never another schema's, never a foreign name
    return [sig for sig in dict.fromkeys(unserved + overloads) if sig not in serving]


def stage_statements(config: Dict[str, Any], version: str) -> List[str]:
    """Provision: the same functions inside the version schema, executable by nobody but the owner."""
    stmts = []
    for fn in served_functions(config, version):
        stmts += [create_sql(fn, version), f"REVOKE ALL ON FUNCTION {fn.signature(version)} FROM PUBLIC"]
    return stmts


def smoke_calls(config: Dict[str, Any], version: str) -> List[str]:
    """One call per staged function (as the operator): every branch's tables resolve on `version`."""
    return [f"SELECT count(*) FROM {version}.{fn.name}({args})"
            for fn in served_functions(config, version) for args in fn.smoke_args]


# Every function name this module can serve and its one role (retire drops only these names
# from a version schema; the switch drops only these names from the served schema). Each
# function's identity arguments are its own (ServedFunction.arg_types).
ROLE_OF = {
    "directory_provider": PUBLIC_ROLE, "directory_by_specialty": PUBLIC_ROLE,
    "directory_by_location": PUBLIC_ROLE, "directory_accepting_new_patients": PUBLIC_ROLE,
    "directory_affiliations": PUBLIC_ROLE,
    "coordinator_web": COORDINATOR_ROLE, "coordinator_tree": COORDINATOR_ROLE,
    WEB_SET_NAME: COORDINATOR_ROLE,
}
FUNCTION_NAMES = tuple(ROLE_OF)

_STAGED_SQL = """
SELECT n.nspname || '.' || p.proname || '(' || oidvectortypes(p.proargtypes) || ')'
FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = %s AND p.proname = ANY(%s)
ORDER BY 1
"""


def staged_signatures(conn: Any, schema: str) -> List[str]:
    """The kit-named functions in `schema`, with the identity arguments the catalog holds: a
    version schema's staged copies (retire), or the served schema's overloads (the switch)."""
    return [r[0] for r in conn.execute(_STAGED_SQL, (schema, list(FUNCTION_NAMES))).fetchall()]


def drop_sql(signature: str) -> str:
    return f"DROP FUNCTION IF EXISTS {signature} RESTRICT"


def web_set_signature(schema: str) -> str:
    """coordinator_web_set's identity signature in `schema` (fixed: it does not depend on the mapping)."""
    return ServedFunction(WEB_SET_NAME, _WEB_SET_ARGS, _WEB_SET_RETURNS, "plpgsql", "", COORDINATOR_ROLE,
                          ("",)).signature(schema)


def expected_grants(served: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """{served signature: the one role allowed to EXECUTE it}, for every function this module can
    serve (a folded graph serves no tree; a missing function is simply not executable). With
    `config`, coordinator_web_set too when the run's mapping declares walk keys."""
    expected = {fn.signature(served): fn.role for fn in servable_functions(served)}
    run_folder = (config or {}).get("run_folder")
    if run_folder:
        from lgk.privacy.gate import read_run_mapping

        if web_set_keys(read_run_mapping(run_folder)):
            expected[web_set_signature(served)] = COORDINATOR_ROLE
    return expected
