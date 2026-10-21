"""The stamp step's allowlist (owner decision, condition 1): nothing the cleared mapping did not allow.

Before the stamp step labels the tables an operator rebuilt, their shape must be exactly the
kit's, and their `properties` keys exactly what the CLEARED mapping allows:

- columns: nodes, edges and node_id_map have exactly the kit build's columns and types, in order
  (lgk.steps.build.KIT_COLUMNS, read from the build's own DDL: no second list);
- node `properties` keys: per node_type, the mapping's `properties` plus its join outputs (what
  build's _properties_expr writes); a node_type the mapping does not declare is unknown;
- edge `properties` keys: none (build writes `map()` on every edge); a relationship_type the
  mapping does not declare is unknown.

The mapping is the run folder's graph_mapping.yaml, the one the privacy gate just cleared
(require_clear refuses a mapping edited after it was cleared). Anything unknown fails with
AllowlistError naming the table and the column, type or key.
"""

from typing import Dict, List, Optional, Set, Tuple

from lgk.mapping import GraphMapping
from lgk.privacy.gate import PrivacyGateError
from lgk.refresh.freshness import Sql


class AllowlistError(PrivacyGateError):
    """A kit table holds a column, type, node/edge type or property key the cleared mapping does not allow."""


def _kit_columns() -> Dict[str, Tuple[Tuple[str, str], ...]]:
    from lgk.steps.build import KIT_COLUMNS

    return KIT_COLUMNS


def allowed_property_keys(mapping: GraphMapping) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """({node_type: keys}, {edge_type: keys}) the cleared mapping lets build write."""
    nodes = {name: set(spec.properties) | {o for j in spec.joins for o in j.outputs}
             for name, spec in mapping.nodes.items()}
    return nodes, {name: set() for name in mapping.edges}


def _described_columns(sql: Sql, fq_table: str) -> List[Tuple[str, str]]:
    """(name, type) of DESCRIBE TABLE, up to the first '#' section (partition info etc.)."""
    out = []
    for row in sql(f"DESCRIBE TABLE {fq_table}"):
        name = str(row[0] or "").strip()
        if not name or name.startswith("#"):
            break
        out.append((name, str(row[1] or "").replace(" ", "").lower()))
    return out


def _column_problem(fq_table: str, found: List[Tuple[str, str]], want: Tuple[Tuple[str, str], ...]) -> Optional[str]:
    if tuple(found) == want:
        return None
    unknown = [f"{n} {t}" for n, t in found if (n, t) not in want]
    missing = [f"{n} {t}" for n, t in want if (n, t) not in found]
    detail = "; ".join(filter(None, [f"unknown column(s) {', '.join(unknown)}" if unknown else "",
                                     f"missing column(s) {', '.join(missing)}" if missing else ""]))
    return f"{fq_table}: {detail or 'columns out of order'} (the kit's columns: {', '.join(n for n, _ in want)})"


def _key_problems(sql: Sql, fq_table: str, type_column: str, allowed: Dict[str, Set[str]]) -> List[str]:
    """Every (type, key) in the table's properties the mapping does not allow, and every unknown type."""
    rows = sql(f"SELECT DISTINCT {type_column}, k FROM {fq_table} "
               f"LATERAL VIEW OUTER explode(map_keys(properties)) x AS k")
    out = []
    for kind, key in sorted(((str(r[0]), r[1]) for r in rows), key=lambda r: (r[0], str(r[1]))):
        if kind not in allowed:
            out.append(f"{fq_table}: {type_column} {kind!r} is not in the cleared mapping")
        elif key is not None and str(key) not in allowed[kind]:
            out.append(f"{fq_table}: {type_column} {kind!r} has property key {str(key)!r} the cleared "
                       f"mapping does not allow")
    return sorted(set(out))


def assert_allowlisted(sql: Sql, fq_schema: str, mapping: Optional[GraphMapping]) -> None:
    """Refuse (AllowlistError) unless the kit tables in `fq_schema` hold only what `mapping` allows."""
    if mapping is None:
        raise AllowlistError("no graph_mapping.yaml in the run folder: nothing says which properties are allowed")
    problems = []
    for table, want in _kit_columns().items():
        problem = _column_problem(f"{fq_schema}.{table}", _described_columns(sql, f"{fq_schema}.{table}"), want)
        if problem:
            problems.append(problem)
    if problems:  # the key queries need the kit's columns (a map `properties`)
        raise AllowlistError("kit table shape is not the kit's: " + "; ".join(problems))
    node_keys, edge_keys = allowed_property_keys(mapping)
    problems = (_key_problems(sql, f"{fq_schema}.nodes", "node_type", node_keys)
                + _key_problems(sql, f"{fq_schema}.edges", "relationship_type", edge_keys))
    if problems:
        raise AllowlistError("kit table holds what the cleared mapping does not allow: " + "; ".join(problems))
