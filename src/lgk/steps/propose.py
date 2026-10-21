"""Propose step: draft graph structure from keyed tables and foreign keys.

Drafts a node type per keyed table, and edges from declared foreign keys:
- a keyless table with exactly two distinct FKs is a join table -> one edge between the
  two tables it references (the join table itself is not a node);
- any other FK on a keyed table -> an edge from that table's node to the referenced node.
REFUSES to draft an edge with no key behind it (raises ValueError).

Care subjects (A5a): a table whose key holds a care-subject column (lgk.privacy.classify, by
name or sampled values) is NEVER a node type: its node would publish the care subject's ids.
It is drafted only as an edge source by the join-table rule (exactly two FKs to node keys),
which never carries the care column; otherwise it is skipped with a warning. An FK on a
care-subject column, or to a care-keyed table, is never an edge end. Folding stays a privacy
step choice plus a hand-written `fold_subject` (propose drafts no folds).
Classifies the graph shape as "web" or "tree".

Writes (in the lgk.mapping contract that `build` reads):
- graph_mapping.yaml: proposed nodes and edges
- flags.json: configuration flags including graph shape
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from lgk.mapping import EdgeSpec, GraphMapping, MappingError, NodeSpec, dump_mapping, parse_mapping
from lgk.privacy import classify

logger = logging.getLogger(__name__)

_STRING_TYPES = ("STRING", "VARCHAR", "CHAR", "TEXT")


@dataclass
class NodeProposal:
    """Proposed node type (one per keyed table; the type is named after the table)."""

    table_name: str
    key_columns: List[str]


@dataclass
class EdgeProposal:
    """Proposed edge type: rows of `source_table` link a src node to a dst node."""

    name: str
    source_table: str
    src: str
    src_type: str
    dst: str
    dst_type: str


def propose(config: Dict[str, Any]) -> None:
    """Run propose step: draft graph structure.

    Reads:
    - run_folder/schema.json (table verdicts with keys and foreign keys, from profile)

    Writes:
    - run_folder/graph_mapping.yaml (lgk.mapping contract)
    - run_folder/flags.json (graph configuration including shape)

    Args:
        config: Configuration dict with:
            - run_folder: Path to run folder
            - source_catalog / source_schema: used when schema.json does not carry them
    """
    run_folder = Path(config.get("run_folder", "."))
    run_folder.mkdir(parents=True, exist_ok=True)

    schema_data = _read_schema(run_folder / "schema.json")
    source_catalog, source_schema = _source(schema_data, config)
    tables = schema_data.get("tables", [])
    fks = {t.get("name"): _distinct_fks(t) for t in tables}
    care_keyed = _care_keyed_tables(tables)
    edge_fks = _edge_fks(tables, fks, care_keyed)
    # tables with a care key or a care FK: skipped, not refused, when no edge can be drafted
    care_tables = care_keyed | {n for n in fks if edge_fks.get(n) != fks[n]}

    nodes = _extract_nodes(tables, care_keyed)
    edges = _validate_edges(_extract_edges(tables, edge_fks, nodes, care_tables), nodes)
    graph_shape = _classify_graph_shape(nodes, edges)

    mapping = _to_mapping(source_catalog, source_schema, tables, fks, nodes, edges)
    # Validate before writing, so a contract break is reported by propose, not later by build.
    parse_mapping(mapping.to_dict())
    dump_mapping(mapping, run_folder / "graph_mapping.yaml")
    _write_flags(run_folder, graph_shape)

    logger.info(f"Propose completed: {len(nodes)} nodes, {len(edges)} edges, shape={graph_shape}")


def _read_schema(schema_file: Path) -> Dict[str, Any]:
    if not schema_file.exists():
        logger.warning(f"schema.json not found at {schema_file}, creating empty schema")
        schema_file.write_text(json.dumps({"tables": []}))
    return json.loads(schema_file.read_text())


def _source(schema_data: Dict[str, Any], config: Dict[str, Any]) -> Tuple[str, str]:
    """source_catalog/source_schema: from schema.json (written by profile), else config."""
    values = []
    for key in ("source_catalog", "source_schema"):
        value = schema_data.get(key) or config.get(key)
        if not value:
            raise MappingError(
                f"field '{key}' is unknown: profile did not record it in schema.json and "
                f"config has no '{key}'"
            )
        values.append(value)
    return values[0], values[1]


def _distinct_fks(table: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Single-column FKs as (column, ref_table, ref_column), duplicates dropped, order kept."""
    out: List[Tuple[str, str, str]] = []
    for fk in table.get("foreign_keys", []) or []:
        cols, refs = fk.get("columns", []), fk.get("references", [])
        if len(cols) != 1 or not isinstance(refs, list) or len(refs) < 2:
            logger.warning(f"{table.get('name')}: skipping non single-column FK {fk}")
            continue
        triple = (cols[0], refs[0], refs[1])
        if triple not in out:
            out.append(triple)
    return out


def _care_columns(table: Dict[str, Any]) -> Set[str]:
    """Columns of `table` the privacy classifier calls care subject (name and sampled values)."""
    verdicts = {c["name"]: c.get("patterns") or {} for c in table.get("columns", [])}
    for key in table.get("key_columns", []) or []:
        verdicts.setdefault(key, {})
    return {c for c, rates in verdicts.items()
            if classify.classify_column(c, rates)[0] == classify.CARE_SUBJECT}


def _care_keyed_tables(tables: List[Dict[str, Any]]) -> Set[str]:
    """Tables whose key (single or composite) holds a care-subject column: never a node."""
    return {t["name"] for t in tables
            if t.get("name") and set(t.get("key_columns", []) or []) & _care_columns(t)}


def _edge_fks(tables: List[Dict[str, Any]], fks: Dict[str, List[Tuple[str, str, str]]],
              care_keyed: Set[str]) -> Dict[str, List[Tuple[str, str, str]]]:
    """FKs that may be edge ends: not on a care-subject column, not to a care-keyed table."""
    out = {}
    for table in tables:
        name, care = table.get("name"), _care_columns(table)
        out[name] = [fk for fk in fks.get(name, []) if fk[0] not in care and fk[1] not in care_keyed]
        for fk in set(fks.get(name, [])) - set(out[name]):
            logger.warning(f"{name}: FK {fk[0]} -> {fk[1]} is a care subject, not drafted as an edge")
    return out


def _extract_nodes(tables: List[Dict[str, Any]], care_keyed: Set[str]) -> Dict[str, NodeProposal]:
    """A node type for every keyed table whose key holds no care-subject column."""
    nodes = {}
    for table in tables:
        name, keys = table.get("name"), table.get("key_columns", []) or []
        if not name:
            continue
        if name in care_keyed:
            logger.warning(f"Table {name} is keyed on a care-subject column, never a node")
        elif keys:
            nodes[name] = NodeProposal(table_name=name, key_columns=list(keys))
        else:
            logger.warning(f"Table {name} has no key_columns, not a node")
    return nodes


def _is_join_table(table: Dict[str, Any], fks: List[Tuple[str, str, str]], care_keyed: bool) -> bool:
    """Keyless (or keyed on a care subject, so not a node) with exactly two FKs."""
    return (care_keyed or not table.get("key_columns")) and len(fks) == 2


def _extract_edges(
    tables: List[Dict[str, Any]],
    fks: Dict[str, List[Tuple[str, str, str]]],
    nodes: Dict[str, NodeProposal],
    care_tables: Set[str],
) -> List[EdgeProposal]:
    """Edge proposals from join tables and from FKs on keyed tables.

    `fks` are the edge-eligible FKs; a table in `care_tables` that is neither a node nor a join
    table is skipped (warned), never refused as keyless."""
    edges = []
    for table in tables:
        name = table.get("name")
        table_fks = [fk for fk in fks.get(name, []) if _hits_node_key(name, fk, nodes)]
        if _is_join_table(table, table_fks, name in care_tables and name not in nodes):
            (src, src_t, _), (dst, dst_t, _) = table_fks
            edges.append(EdgeProposal(name, name, src, src_t, dst, dst_t))
            continue
        if name in care_tables and name not in nodes:
            logger.warning(f"{name}: holds a care subject and is not a join table, no edge drafted")
            continue
        own_keys = nodes[name].key_columns if name in nodes else [None]
        if len(own_keys) > 1:
            logger.warning(f"{name}: composite key, its FKs are not drafted as edges")
            continue
        for col, ref_table, _ in table_fks:
            edges.append(EdgeProposal(f"{name}_{col}", name, own_keys[0], name, col, ref_table))
    return edges


def _hits_node_key(table: str, fk: Tuple[str, str, str], nodes: Dict[str, NodeProposal]) -> bool:
    """An FK resolves to node ids only if it references the target node's whole natural key.

    A target that is not a node at all is kept, so _validate_edges refuses it loudly.
    """
    _, ref_table, ref_col = fk
    if ref_table in nodes and nodes[ref_table].key_columns != [ref_col]:
        logger.warning(f"{table}: FK to {ref_table}.{ref_col} is not that node's key, skipped")
        return False
    return True


def _validate_edges(edges: List[EdgeProposal], nodes: Dict[str, NodeProposal]) -> List[EdgeProposal]:
    """Refuse an edge whose either end has no keyed node behind it.

    Raises:
        ValueError if an edge's source or target table has no key.
    """
    for edge in edges:
        for side, table in (("source", edge.src_type), ("target", edge.dst_type)):
            if table not in nodes:
                raise ValueError(f"Cannot create edge {edge.name}: {side} table {table} has no primary key")
    return edges


def _classify_graph_shape(nodes: Dict[str, NodeProposal], edges: List[EdgeProposal]) -> str:
    """Classify graph shape as "tree" (acyclic, few cross-links) or "web"."""
    if not nodes or not edges:
        return "tree"
    if len(edges) > len(nodes):
        return "web"
    if _has_cycles(nodes, edges):
        return "web"
    return "tree"


def _has_cycles(nodes: Dict[str, NodeProposal], edges: List[EdgeProposal]) -> bool:
    """Check if the node-type graph has a directed cycle (DFS)."""
    adj = defaultdict(list)
    for edge in edges:
        adj[edge.src_type].append(edge.dst_type)

    visited: Set[str] = set()
    rec_stack: Set[str] = set()

    def dfs(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node)
        return False

    return any(node not in visited and dfs(node) for node in nodes)


def _label_and_properties(
    table: Dict[str, Any], key_columns: List[str], fk_columns: Set[str]
) -> Tuple[Optional[str], List[str]]:
    """Label = first string column that is not a key or FK; the other such columns are properties."""
    cols = [c for c in table.get("columns", []) if c["name"] not in key_columns and c["name"] not in fk_columns]
    label = next((c["name"] for c in cols if str(c.get("type", "")).upper().startswith(_STRING_TYPES)), None)
    return label, [c["name"] for c in cols if c["name"] != label]


def _to_mapping(
    source_catalog: str,
    source_schema: str,
    tables: List[Dict[str, Any]],
    fks: Dict[str, List[Tuple[str, str, str]]],
    nodes: Dict[str, NodeProposal],
    edges: List[EdgeProposal],
) -> GraphMapping:
    """The proposal in the lgk.mapping contract."""
    by_name = {t.get("name"): t for t in tables}
    node_specs = {}
    for name, node in nodes.items():
        keys = node.key_columns
        label, props = _label_and_properties(by_name[name], keys, {c for c, _, _ in fks.get(name, [])})
        node_specs[name] = NodeSpec(
            source_table=name,
            natural_key=keys[0] if len(keys) == 1 else list(keys),
            label=label or keys[0],
            properties=props,
        )
    edge_specs = {
        e.name: EdgeSpec(e.source_table, e.src, e.src_type, e.dst, e.dst_type) for e in edges
    }
    return GraphMapping(source_catalog, source_schema, node_specs, edge_specs)


def _write_flags(run_folder: Path, graph_shape: str) -> None:
    """Write flags.json with graph configuration."""
    flags_file = run_folder / "flags.json"
    with open(flags_file, "w") as f:
        json.dump({"shape": graph_shape}, f, indent=2)
    logger.info(f"Flags written to {flags_file}")
