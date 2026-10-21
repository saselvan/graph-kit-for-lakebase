"""Build step: Creates the nodes and edges Delta tables from graph_mapping.yaml, by running SQL statements on the SQL warehouse.

Reads the mapping from the propose step and creates Delta tables in the kit schema:
- nodes: one row per graph node (properties included); a duplicate natural key FAILS the build
  (lgk.keys.DuplicateKeyError), it is never collapsed to a silent winner
- edges: one row per graph edge (deduplicated; a folded edge represents one weighted pair
  where the weight meets the privacy floor threshold)
- node_id_map: mapping from natural key to node_id (for lookup)

All node_ids are deterministic xxhash64 hashes, no joins needed.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from lgk import keys
from lgk import manifest as mf
from lgk.manifest import ObjectType
from lgk.mapping import EdgeSpec, GraphMapping, JoinSpec, MappingError, NodeSpec, NaturalKey, load_mapping
from lgk.privacy import binding, floor
from lgk.privacy.gate import fold_policy, read_schema, require_clear, required_run_folder
from lgk.refresh import label
from lgk.uc_sql import execute_sql, warehouse_id

logger = logging.getLogger(__name__)

# Floating-point sums in Spark depend on partition order; the same rows can give slightly different last digits.
_FLOATING_TYPES = ("FLOAT", "DOUBLE", "REAL")


class JoinNotUniqueError(RuntimeError):
    """A one-to-one (`columns`) join's key is not unique in its table: it would pick an arbitrary row."""


def _key_expr(natural_key: NaturalKey, alias: Optional[str] = None) -> str:
    """SQL for a node's natural key: the kit's one key expression (lgk.privacy.floor.node_key_expr)."""
    return floor.node_key_expr(natural_key, alias)


def build(config: Dict[str, Any]) -> None:
    """Run build step: SDP to nodes/edges/node_id_map Delta tables.

    Reads:
    - run_folder/graph_mapping.yaml (proposed nodes and edges)

    Writes:
    - UC Delta tables: nodes, edges, node_id_map

    Args:
        config: Configuration dict with:
            - graph_name: Name of the graph
            - run_folder: Path to run folder
            - lakebase_project: Lakebase project ID
            - lakebase_catalog: UC catalog for graph tables
            - branch_id: Test branch ID (optional, for testing)
            - kit_schema: Postgres/UC schema for kit tables
            - warehouse_id: SQL warehouse for the UC statements (required, no default)
    """
    if "workshop_kit_schema" in config:  # in an exported bundle: SP uses its own kit schema only
        from lgk.sp_grants import require_own_kit_schema

        require_own_kit_schema(config)
    run_folder = Path(required_run_folder(config))

    # Read graph_mapping.yaml through the contract (raises MappingError naming the field,
    # before any SQL runs)
    graph_mapping_file = run_folder / "graph_mapping.yaml"
    if not graph_mapping_file.exists():
        logger.error(f"graph_mapping.yaml not found at {graph_mapping_file}")
        raise FileNotFoundError(f"graph_mapping.yaml not found at {graph_mapping_file}")
    mapping = load_mapping(graph_mapping_file)
    # The privacy gate, before any SQL: flags.json clear and fresh for THIS mapping.
    cleared = require_clear(str(run_folder), mapping)
    if any(spec.joins for spec in mapping.nodes.values()):
        check_join_types(mapping, read_schema(str(run_folder)))
    stamp = binding.stamp_id(cleared)
    fold_floor, fold_types = fold_policy(cleared)
    source_catalog, source_schema = mapping.source_catalog, mapping.source_schema
    nodes_spec, edges_spec = mapping.nodes, mapping.edges

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    wh = warehouse_id(config)
    catalog, kit = config["lakebase_catalog"], config["kit_schema"]
    # The key checks, on the data as it is now (UC keys are informational), before any kit table
    # is replaced: a duplicate natural key or an undecided orphan leaves the last build in place.
    source_fq = f"{source_catalog}.{source_schema}"
    _check_node_keys(w, wh, source_fq, nodes_spec)
    keys.apply_edge_orphans(str(run_folder), _edge_orphan_counts(w, wh, source_fq, nodes_spec, edges_spec))
    _claim_kit_schema(w, wh, config)
    _build_nodes_table(w, wh, source_catalog, source_schema, catalog, kit, nodes_spec, fold_floor)
    _build_edges_table(w, wh, source_catalog, source_schema, catalog, kit, nodes_spec, edges_spec,
                       fold_floor)
    _build_node_id_map_table(w, wh, catalog, kit)
    for table in ("nodes", "edges", "node_id_map"):
        mf.record(config, ObjectType.UC_TABLE, f"{catalog}.{kit}.{table}", "uc")
    # The label's certified_version is the version checked below (owner decision, nightly stamp step):
    # identities are read before the floor checks, and the label is refused if a table moved since.
    before = label.read_identities(w, lambda s: execute_sql(w, wh, s),
                                   [f"{catalog}.{kit}.{t}" for t in binding.BUILT_TABLES])
    # verify the values that were written; a folded edge below the floor (or NULL) raises
    # FoldFloorError here, so nothing is stamped or recorded and provision refuses the tables.
    folds = floor.assert_fold_floor(lambda s: execute_sql(w, wh, s), f"{catalog}.{kit}.edges",
                                    fold_types, fold_floor)
    # a joined output on a node below the floor raises JoinFloorError here.
    joins = floor.assert_join_floor(lambda s: execute_sql(w, wh, s), f"{catalog}.{kit}.nodes",
                                    mapping, fold_floor)
    _bind_to_stamp(w, wh, config, before, stamp, fold_floor, folds, joins)
    logger.info("Build completed: nodes, edges, node_id_map created")


def _claim_kit_schema(w: Any, warehouse_id: str, config: Dict[str, Any]) -> None:
    """Create-and-mark the UC kit schema (Delta sources), or prove it is this graph's; record it if created."""
    catalog, kit = config["lakebase_catalog"], config["kit_schema"]
    if mf.claim_uc_schema(w, warehouse_id, catalog, kit, config["graph_name"]):
        mf.record(config, ObjectType.UC_SCHEMA, f"{catalog}.{kit}", "uc")


def _bind_to_stamp(w: Any, warehouse_id: str, config: Dict[str, Any], before: Dict[str, Any], stamp: str,
                   fold_floor: int, folds: floor.FoldReport,
                   joins: Optional[floor.JoinReport] = None) -> None:
    """Record the privacy stamp the tables were built from: on each table, then in the run folder.

    Only after every row is written and the fold floor check passed, so a build that failed
    half-way leaves tables that provision refuses (no stamp, or the previous build's). Each table
    gets the label {stamp, table_id, certified_version} and the run manifest the same, through the
    writer the operator's stamp step uses (lgk.refresh.label.write_labels): one label meaning on
    both chains. A graph with fold types also gets `lgk.fold_floor` on edges and its `folds` report
    in the record; a graph with node joins records its `joins` report."""
    sql = lambda s: execute_sql(w, warehouse_id, s)  # noqa: E731
    label.write_labels(w, sql, config, stamp, before)
    fq_schema = f"{config['lakebase_catalog']}.{config['kit_schema']}"
    if folds:
        sql(binding.set_fold_floor_sql(f"{fq_schema}.edges", fold_floor))
    binding.record_build(str(config["run_folder"]), stamp, list(before), folds, joins)


_NODES_DDL = ("node_id STRING, node_type STRING, natural_key STRING, label STRING, "
              "properties MAP<STRING, STRING>")
# `weight` is the last column: NULL on unfolded edges, the count of distinct care subjects
# behind a folded edge; last, so a served view over a new version can replace one over an old one.
_EDGES_DDL = ("edge_id STRING, src_node_id STRING, dst_node_id STRING, relationship_type STRING, "
              "properties MAP<STRING, STRING>, weight BIGINT")
_NODE_ID_MAP_DDL = "node_id STRING, node_type STRING, natural_key STRING"


def _ddl_columns(ddl: str) -> tuple:
    """((name, type), ...) of a DDL column list, types lower case with no spaces (DESCRIBE TABLE's spelling)."""
    return tuple((name, kind.replace(" ", "").lower()) for name, kind in re.findall(r"(\w+) (MAP<[^>]*>|\w+)", ddl))


# The one statement of what the kit tables hold: the stamp step's column allowlist reads it
# (lgk.refresh.allowlist), so a table rebuilt by an operator must have exactly these columns.
KIT_COLUMNS = {"nodes": _ddl_columns(_NODES_DDL), "edges": _ddl_columns(_EDGES_DDL),
               "node_id_map": _ddl_columns(_NODE_ID_MAP_DDL)}


def _recreate_table(w: Any, warehouse_id: str, table: str, columns: str) -> None:
    """Replace the table with an empty one of the same identity (the build always starts from nothing).

    CREATE OR REPLACE, never DROP + CREATE: a dropped and re-created
    table restarts its Delta history at version 0, so a nightly rebuild landed on the same
    version numbers as the synced copy and the freshness guard passed while Postgres served the
    old rows. CREATE OR REPLACE keeps the table and its history, so the rebuild is a newer
    version the guard sees. It also drops the table's custom properties, the privacy stamp
    included (live probe, 2026-10-04: a stamp set before CREATE OR REPLACE is gone after it); the
    stamp is bound again only after every row is written (_bind_to_stamp), so a night that fails
    half-way leaves a table the provision and sync stamp checks refuse.
    """
    execute_sql(w, warehouse_id, f"CREATE OR REPLACE TABLE {table} ({columns}) USING DELTA")


# ----------------------------------------------------------------- node joins
def check_join_types(mapping: GraphMapping, schema: Dict[str, Any]) -> None:
    """Before any SQL, validate node joins (MappingError naming the path):

    - a join's `key` must have the same type as the node's natural_key column: a value-equal
      key of another type ('007' vs 7) joins by value but renders differently, so the
      node-id-based checks downstream would not see the node;
    - `sum` over a floating column is refused. `min`, `max`, `count` and `count_distinct` are
      allowed on any type.
    """
    types = {(t["name"], c["name"]): str(c.get("type") or "").upper()
             for t in schema.get("tables", []) for c in t.get("columns", [])}
    for node_type, spec in mapping.nodes.items():
        for i, join in enumerate(spec.joins):
            _check_join_key_type(types, node_type, spec, i, join)
            for k, agg in enumerate(join.aggregates):
                kind = types.get((join.table, agg.column), "")
                if agg.func == "sum" and kind.startswith(_FLOATING_TYPES):
                    raise MappingError(
                        f"field 'nodes.{node_type}.joins[{i}].aggregates[{k}]' sums "
                        f"{join.table}.{agg.column} of type {kind}: Spark sums floating values in "
                        f"partition order, so two runs can differ; sum an integer or DECIMAL column")


def _check_join_key_type(types: Dict[Any, str], node_type: str, spec: NodeSpec, i: int,
                         join: JoinSpec) -> None:
    node_kind = types.get((spec.source_table, spec.natural_key), "")
    join_kind = types.get((join.table, join.key), "")
    if not node_kind or not join_kind or node_kind != join_kind:  # an unknown type is refused
        raise MappingError(
            f"field 'nodes.{node_type}.joins[{i}].key' is {join.table}.{join.key} of type "
            f"{join_kind or 'unknown'}, but the natural_key {spec.source_table}.{spec.natural_key} "
            f"is {node_kind or 'unknown'}: a join key must have the natural_key's type (equal "
            f"values of two types can render differently). Cast it in a gold view, then join that.")


def _unique_key_sql(source_table: str, key: str) -> str:
    """One row (the count of one duplicated key) when `key` is not unique; the value is never read."""
    return (f"SELECT COUNT(*) FROM {source_table} WHERE {key} IS NOT NULL "
            f"GROUP BY {key} HAVING COUNT(*) > 1 LIMIT 1")


def _check_columns_joins_unique(w: Any, warehouse_id: str, source_schema_fq: str,
                                nodes_spec: Dict[str, NodeSpec]) -> None:
    """Every `columns` join's key is unique in its table (one Spark query per such join)."""
    for node_type, spec in nodes_spec.items():
        for i, join in enumerate(spec.joins):
            if not join.columns:
                continue
            rows = execute_sql(w, warehouse_id, _unique_key_sql(f"{source_schema_fq}.{join.table}", join.key))
            if rows:
                raise JoinNotUniqueError(
                    f"nodes.{node_type}.joins[{i}]: {join.key} is not unique in {join.table} (one "
                    f"key value has {int(rows[0][0])} rows); a one-to-one 'columns' join would pick "
                    f"an arbitrary row. Join a table with one row per key, or use 'aggregates'.")


def _agg_sql(func: str, column: str) -> str:
    return f"COUNT(DISTINCT {column})" if func == "count_distinct" else f"{func.upper()}({column})"


def _join_subquery(source_table: str, join: JoinSpec) -> str:
    """One subquery per join, keyed `_lgk_key`: grouped by the key for `aggregates` (with the
    distinct subject count on a floored join), the plain rows for `columns`."""
    key = join.key
    if join.columns:
        cols = ", ".join(join.columns)
        return f"SELECT {key} AS _lgk_key, {cols} FROM {source_table} WHERE {key} IS NOT NULL"
    terms = [f"{_agg_sql(a.func, a.column)} AS {a.name}" for a in join.aggregates]
    if join.subject is not None:
        terms.append(f"COUNT(DISTINCT {join.subject}) AS {floor.SUBJECTS_COLUMN}")
    return (f"SELECT {key} AS _lgk_key, {', '.join(terms)} FROM {source_table} "
            f"WHERE {key} IS NOT NULL GROUP BY {key}")


def _join_clauses(source_schema_fq: str, spec: NodeSpec) -> str:
    """The LEFT JOINs of a node type: a node with no joined rows keeps its row (NULL outputs)."""
    return "".join(
        f"\n            LEFT JOIN ({_join_subquery(f'{source_schema_fq}.{j.table}', j)}) j{i} "
        f"ON {floor.join_on(f'j{i}', spec.natural_key, 'n')}"
        for i, j in enumerate(spec.joins))


def _properties_expr(spec: NodeSpec, privacy_floor: int) -> str:
    """The node's properties as a MAP<STRING, STRING>: native columns of `n`, then each join's
    outputs (floored ones wrapped in the CASE of lgk.privacy.floor)."""
    pairs = [f"'{col}', CAST(n.{col} AS STRING)" for col in spec.properties]
    for i, join in enumerate(spec.joins):
        floored = join.subject is not None
        pairs += [f"'{o}', {floor.join_output_expr(f'j{i}', o, floored, privacy_floor)}"
                  for o in join.outputs]
    return "map(" + ", ".join(pairs) + ")" if pairs else "map()"


def _insert_nodes_sql(nodes_table: str, source_table: str, node_type: str, spec: NodeSpec,
                      privacy_floor: Optional[int] = None) -> str:
    """INSERT one node type: one row per source row with a non-NULL natural key.

    No dedup: build refuses a duplicate natural key before this runs (_check_node_keys) and
    reads the table back after it (_check_node_ids), so a source row is a node (node_id is the
    synced table's primary key). Rows with a NULL natural key are dropped: they would all hash
    to the same node_id, and an edge can never point at them (a NULL is not an orphan, and a
    non-NULL edge end cannot match a NULL key). Join outputs come from one LEFT JOIN
    per join on the node key; each joined subquery has one row per key (GROUP BY, or a
    `columns` join checked unique by _check_columns_joins_unique), so the row count is unchanged.
    """
    key = _key_expr(spec.natural_key, "n")
    joins = _join_clauses(source_table.rsplit(".", 1)[0], spec) if spec.joins else ""
    return f"""
        INSERT INTO {nodes_table}
        SELECT
            {floor.node_id_expr(node_type, spec.natural_key, "n")} as node_id,
            '{node_type}' as node_type,
            CAST({key} AS STRING) as natural_key,
            CAST(n.{spec.label} AS STRING) as label,
            {_properties_expr(spec, floor.checked_floor(privacy_floor))} as properties
        FROM {source_table} n{joins}
        WHERE {key} IS NOT NULL
        """


# ----------------------------------------------------------------- key checks (lgk.keys)
def _check_node_keys(w: Any, warehouse_id: str, source_fq: str, nodes_spec: Dict[str, NodeSpec]) -> None:
    """Each node type's natural key is unique in its source (one query per node type, counts only).

    Raises DuplicateKeyError naming every node type with duplicates and the count."""
    failures = []
    for node_type, spec in nodes_spec.items():
        rows = execute_sql(w, warehouse_id, keys.duplicate_key_check_sql(f"{source_fq}.{spec.source_table}",
                                                                         spec.natural_key))
        if rows:
            failures.append(keys.duplicate_message(f"node type {node_type}", spec.source_table,
                                                   spec.natural_key, int(rows[0][0])))
    if failures:
        raise keys.DuplicateKeyError("build: " + "; ".join(failures))


def _check_node_ids(w: Any, warehouse_id: str, nodes_table: str) -> None:
    """Read-back after the nodes INSERT: no node id twice (a writer may add a duplicate after the
    pre-check). Raises DuplicateKeyError before anything is stamped, so provision refuses the table."""
    rows = execute_sql(w, warehouse_id, keys.node_readback_sql(nodes_table))
    if rows:
        n = int(rows[0][0])
        raise keys.DuplicateKeyError(
            f"build: {nodes_table} holds {n} duplicate node id{'' if n == 1 else 's'} after the insert "
            "(a source key was duplicated while the build ran); nothing was stamped. Run build again")


def _edge_ends(spec: EdgeSpec) -> list:
    """The (column, node type) ends of an edge type; a co-membership fold has one."""
    ends = [(spec.src, spec.src_type), (spec.dst, spec.dst_type)]
    return list(dict.fromkeys(ends))


def _edge_orphan_counts(w: Any, warehouse_id: str, source_fq: str, nodes_spec: Dict[str, NodeSpec],
                        edges_spec: Dict[str, EdgeSpec]) -> list:
    """The live orphan count of each edge end: rows whose non-NULL end has no node (counts only)."""
    out = []
    for edge_type, spec in edges_spec.items():
        for column, node_type in _edge_ends(spec):
            node = nodes_spec[node_type]
            rows = execute_sql(w, warehouse_id, keys.edge_orphan_sql(
                f"{source_fq}.{spec.source_table}", column, node_type, f"{source_fq}.{node.source_table}",
                node.natural_key))
            out.append({"edge_type": edge_type, "table": spec.source_table, "column": column,
                        "node_type": node_type, "orphan_rows": int(rows[0][0]) if rows else 0})
    return out


def _build_nodes_table(w: Any, warehouse_id: str, source_catalog: str, source_schema: str,
                       dest_catalog: str, dest_schema: str, nodes_spec: Dict[str, NodeSpec],
                       privacy_floor: int) -> None:
    """Check the one-to-one joins, then (re)create `nodes` and insert each node type.

    The uniqueness queries run before CREATE OR REPLACE, so a refused join leaves the last
    build's `nodes` in place."""
    nodes_table = f"{dest_catalog}.{dest_schema}.nodes"
    source_fq = f"{source_catalog}.{source_schema}"
    _check_columns_joins_unique(w, warehouse_id, source_fq, nodes_spec)
    _recreate_table(w, warehouse_id, nodes_table, _NODES_DDL)
    for node_type, spec in nodes_spec.items():
        source_table = f"{source_fq}.{spec.source_table}"
        execute_sql(w, warehouse_id, _insert_nodes_sql(nodes_table, source_table, node_type, spec, privacy_floor))
    _check_node_ids(w, warehouse_id, nodes_table)
    logger.info(f"Created nodes table: {nodes_table}")


def _insert_edges_sql(edges_table: str, nodes_table: str, source_table: str, edge_type: str,
                      spec: EdgeSpec) -> str:
    """INSERT one edge type, deduplicated per (src, dst).

    ROW_NUMBER keeps one row per src->dst pair. This dedup stays: it is not a key dedup. An edge
    source is usually a fact table (many rows between the same two nodes), and the
    graph holds one edge per pair; edge_id is a hash of the pair, so it is also what keeps
    edge_id unique. Semi-joins on the nodes table drop edges whose end has no node (an orphan or
    a NULL FK): build counted the orphans first (lgk.keys.apply_edge_orphans) and refused any the
    operator did not accept, so the drop is never silent.
    """
    src, dst = spec.src, spec.dst
    return f"""
        INSERT INTO {edges_table}
        SELECT e.* FROM (
            SELECT
                CAST(xxhash64('{edge_type}::'||{src}||'::'||{dst}) AS STRING) as edge_id,
                CAST(xxhash64('{spec.src_type}::'||{src}) AS STRING) as src_node_id,
                CAST(xxhash64('{spec.dst_type}::'||{dst}) AS STRING) as dst_node_id,
                '{edge_type}' as relationship_type,
                map() as properties,
                CAST(NULL AS BIGINT) as weight
            FROM (
                SELECT
                    {src},
                    {dst},
                    ROW_NUMBER() OVER (PARTITION BY {src}, {dst} ORDER BY 1) as rn
                FROM {source_table}
            )
            WHERE rn = 1
        ) e
        LEFT SEMI JOIN {nodes_table} s ON e.src_node_id = s.node_id
        LEFT SEMI JOIN {nodes_table} d ON e.dst_node_id = d.node_id
        """


def _fold_source_sql(source_table: str, spec: EdgeSpec, fold_floor: int) -> str:
    """The SELECT (src_key, dst_key, weight) of a fold edge: the floor stays in lgk.privacy.floor."""
    if spec.fold_form == "co_membership":
        return floor.fold_edges_sql(source_table, spec.fold_subject, spec.src, fold_floor)
    return floor.fold_pair_sql(source_table, spec.fold_subject, spec.src, spec.dst, fold_floor)


def _insert_fold_edges_sql(edges_table: str, nodes_table: str, source_table: str, edge_type: str,
                           spec: EdgeSpec, fold_floor: int) -> str:
    """INSERT one folded edge type: one row per (src, dst) pair at or above the privacy floor.

    The care-subject column is counted (`weight`), never carried. GROUP BY already gives one
    row per pair, so no ROW_NUMBER dedup; the same semi-joins drop pairs with no node.
    """
    return f"""
        INSERT INTO {edges_table}
        SELECT e.* FROM (
            SELECT
                CAST(xxhash64('{edge_type}::'||f.src_key||'::'||f.dst_key) AS STRING) AS edge_id,
                CAST(xxhash64('{spec.src_type}::'||f.src_key) AS STRING) AS src_node_id,
                CAST(xxhash64('{spec.dst_type}::'||f.dst_key) AS STRING) AS dst_node_id,
                '{edge_type}' AS relationship_type,
                map() AS properties,
                f.weight AS weight
            FROM (
                {_fold_source_sql(source_table, spec, fold_floor)}
            ) f
        ) e
        LEFT SEMI JOIN {nodes_table} s ON e.src_node_id = s.node_id
        LEFT SEMI JOIN {nodes_table} d ON e.dst_node_id = d.node_id
        """


def _edge_insert_sql(edges_table: str, nodes_table: str, source_table: str, edge_type: str,
                     spec: EdgeSpec, fold_floor: int) -> str:
    """The folded INSERT for an edge with `fold_subject`, else the unfolded one."""
    if spec.fold_subject is None:
        return _insert_edges_sql(edges_table, nodes_table, source_table, edge_type, spec)
    return _insert_fold_edges_sql(edges_table, nodes_table, source_table, edge_type, spec, fold_floor)


def _build_edges_table(w: Any, warehouse_id: str, source_catalog: str, source_schema: str,
                       dest_catalog: str, dest_schema: str, nodes_spec: Dict[str, NodeSpec],
                       edges_spec: Dict[str, EdgeSpec], fold_floor: int) -> None:
    """(Re)create `edges` and insert each edge type (deduplicated or folded, orphans dropped)."""
    edges_table = f"{dest_catalog}.{dest_schema}.edges"
    nodes_table = f"{dest_catalog}.{dest_schema}.nodes"
    _recreate_table(w, warehouse_id, edges_table, _EDGES_DDL)
    for edge_type, spec in edges_spec.items():
        source_table = f"{source_catalog}.{source_schema}.{spec.source_table}"
        execute_sql(w, warehouse_id,
                    _edge_insert_sql(edges_table, nodes_table, source_table, edge_type, spec, fold_floor))
    logger.info(f"Created edges table: {edges_table}")


def _build_node_id_map_table(w: Any, warehouse_id: str, dest_catalog: str, dest_schema: str) -> None:
    """(Re)create `node_id_map` from `nodes` (one row per node_id, since nodes has one)."""
    node_id_map_table = f"{dest_catalog}.{dest_schema}.node_id_map"
    _recreate_table(w, warehouse_id, node_id_map_table, _NODE_ID_MAP_DDL)
    execute_sql(w, warehouse_id, f"""
        INSERT INTO {node_id_map_table}
        SELECT node_id, node_type, natural_key
        FROM {dest_catalog}.{dest_schema}.nodes
        """)
    logger.info(f"Created node_id_map table: {node_id_map_table}")
