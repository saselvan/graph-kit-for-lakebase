"""Profile step: schema profiling without sampling sensitive data.

Profiles gold tables (keys, row counts, null rates) and writes pattern VERDICTS ONLY.
NEVER sampled values. Verdicts = match rate, entropy, distinct ratio.

No sampled value is written to the run folder: schema.json holds verdicts, counts and names.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from lgk import keys
from lgk.mapping import identifier
from lgk.privacy.classify import pattern_verdicts
from lgk.uc_sql import execute_sql

logger = logging.getLogger(__name__)

SqlFn = Callable[[str], List[List[Any]]]

# UC mode samples at most this many rows per table for the privacy pattern verdicts. The
# values stay in memory: only match rates (lgk.privacy.classify.pattern_verdicts) are written.
PATTERN_SAMPLE_ROWS = 1000


@dataclass
class ColumnVerdict:
    """Profile verdict for a single column (never sampled values)."""

    name: str
    type_name: str  # e.g., "INTEGER", "STRING"
    match_rate: float  # 0-1: proportion of non-null values
    entropy: Optional[float]  # 0-1 estimated entropy; None when not computed (UC mode)
    distinct_ratio: float  # 0-1: distinct_count / row_count
    null_rate: float  # 0-1: proportion of null values
    patterns: Optional[Dict[str, float]] = None  # privacy pattern match rates (UC mode)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize verdict to dict (no sampled data)."""
        out = {
            "name": self.name,
            "type": self.type_name,
            "match_rate": self.match_rate,
            "entropy": self.entropy,
            "distinct_ratio": self.distinct_ratio,
            "null_rate": self.null_rate,
        }
        if self.patterns is not None:
            out["patterns"] = self.patterns
        return out


@dataclass
class TableVerdict:
    """Profile verdict for a table."""

    name: str
    key_columns: List[str]
    row_count: int
    columns: List[ColumnVerdict]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize table verdict to dict."""
        return {
            "name": self.name,
            "key_columns": self.key_columns,
            "row_count": self.row_count,
            "columns": [col.to_dict() for col in self.columns],
        }


def profile(config: Dict[str, Any]) -> None:
    """Run profile step: compute verdicts for gold tables.

    Two sources, chosen by config:
    - **UC** (config has `warehouse_id`): reads the tables of `source_catalog.source_schema`
      (only those named in gold_tables.json, if it lists any), their declared PRIMARY KEY and
      FOREIGN KEY constraints from information_schema, and real row counts, null rates and
      distinct ratios. Entropy is not computed (null).
    - **declared** (no `warehouse_id`, offline): tables, keys and foreign keys exactly as
      declared in gold_tables.json, with synthetic verdicts.

    Reads:
    - run_folder/gold_tables.json (optional in UC mode)

    Writes:
    - run_folder/schema.json: source_catalog, source_schema, and per table name,
      key_columns, row_count, columns (verdicts only, no sampled data) and foreign_keys.
      UC mode also writes, from the data (UC keys are informational): per table
      `duplicate_key_rows` and `null_key_rows` of its key (None for a table with no key), and
      per foreign key `orphan_rows` (lgk.keys). Counts only, never a key value.
    - run_folder/key_decisions.json (UC mode): the orphan count of every FK, as an observation
      the operator decides on (lgk.keys.resolve_orphans).

    Raises lgk.keys.DuplicateKeyError (UC mode), AFTER writing schema.json, when a table's key
    has duplicate rows: naming each such table and its count. Orphans do not fail profile; build
    fails on any the operator has not accepted.
    """
    run_folder = Path(config.get("run_folder", "."))
    run_folder.mkdir(parents=True, exist_ok=True)
    gold_tables = _read_gold_tables(run_folder / "gold_tables.json")

    if config.get("warehouse_id"):
        schema = _profile_uc(config, gold_tables)
    else:
        schema = _profile_declared(config, gold_tables)

    schema_file = run_folder / "schema.json"
    with open(schema_file, "w") as f:
        json.dump(schema, f, indent=2)
    logger.info(f"Profile completed: {len(schema['tables'])} tables profiled -> {schema_file}")
    if config.get("warehouse_id"):
        _record_key_findings(str(run_folder), schema)


def _record_key_findings(run_folder: str, schema: Dict[str, Any]) -> None:
    """Record every FK's orphan count for the operator; fail on duplicate keys (counts only)."""
    observed = [(t["name"], fk["columns"], fk["orphan_rows"])
                for t in schema["tables"] for fk in t.get("foreign_keys", [])]
    if observed:
        keys.record_observations(run_folder, observed, "profile")
    for table, cols, n in observed:
        if n:
            logger.warning(f"{table}.{','.join(cols)}: {n} orphan rows; the operator decides "
                           "(lgk.keys.resolve_orphans) before build")
    failures = [keys.duplicate_message("profile", t["name"], t["key_columns"], t["duplicate_key_rows"])
                for t in schema["tables"] if t.get("duplicate_key_rows")]
    if failures:
        raise keys.DuplicateKeyError("; ".join(failures))


def _read_gold_tables(gold_tables_file: Path) -> List[Dict[str, Any]]:
    if not gold_tables_file.exists():
        logger.warning(f"gold_tables.json not found at {gold_tables_file}, creating empty one")
        gold_tables_file.write_text(json.dumps({"tables": []}))
    return json.loads(gold_tables_file.read_text()).get("tables", [])


def _with_source(config: Dict[str, Any], tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("source_catalog", "source_schema"):
        if config.get(key):
            out[key] = config[key]
    out["tables"] = tables
    return out


# ------------------------------------------------------------------------- declared (offline)
def _profile_declared(config: Dict[str, Any], gold_tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Offline profile: declared keys and FKs from gold_tables.json, synthetic verdicts."""
    tables = []
    for table_config in gold_tables:
        fks = list(table_config.get("foreign_keys", []))
        # Declared FK columns exist too: list them, so the privacy gate classifies them.
        fk_cols = [c for fk in fks for c in fk.get("columns", [])]
        keys = list(table_config.get("key_columns", []))
        verdict = _compute_table_verdict(table_config.get("name"), list(dict.fromkeys(keys + fk_cols)))
        verdict.key_columns = keys
        entry = verdict.to_dict()
        entry["foreign_keys"] = fks
        tables.append(entry)
    return _with_source(config, tables)


# ------------------------------------------------------------------------------- UC (live)
def _profile_uc(config: Dict[str, Any], gold_tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Live profile of source_catalog.source_schema through the SQL warehouse."""
    catalog = identifier(config.get("source_catalog"), "source_catalog")
    schema = identifier(config.get("source_schema"), "source_schema")
    w = config.get("workspace_client")
    if w is None:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()

    def sql(statement: str) -> List[List[Any]]:
        return execute_sql(w, config["warehouse_id"], statement)

    columns = _uc_columns(sql, catalog, schema)
    wanted = [t["name"] for t in gold_tables if t.get("name")] or list(columns)
    declared_keys = {t["name"]: t.get("key_columns", []) for t in gold_tables if t.get("name")}
    pks = _uc_primary_keys(sql, catalog, schema)
    fks = _uc_foreign_keys(sql, catalog, schema)

    tables = []
    for name in wanted:
        if name not in columns:
            raise ValueError(f"gold table {catalog}.{schema}.{name} not found in information_schema")
        verdict = _uc_table_verdict(sql, f"{catalog}.{schema}.{name}", name,
                                    pks.get(name) or declared_keys.get(name, []), columns[name])
        entry = verdict.to_dict()
        entry.update(_uc_key_counts(sql, f"{catalog}.{schema}.{name}", verdict.key_columns))
        entry["foreign_keys"] = [_with_orphans(sql, f"{catalog}.{schema}", name, fk) for fk in fks.get(name, [])]
        tables.append(entry)
    return {"source_catalog": catalog, "source_schema": schema, "tables": tables}


def _one_row(sql: SqlFn, statement: str) -> List[Any]:
    rows = sql(statement)
    if len(rows) != 1:  # an aggregate with no GROUP BY is one row; anything else is a broken read
        raise RuntimeError(f"a key-count query returned {len(rows)} rows, expected 1")
    return rows[0]


def _uc_key_counts(sql: SqlFn, fq_table: str, key_columns: List[str]) -> Dict[str, Optional[int]]:
    """duplicate_key_rows and null_key_rows of the table's key, in ONE query (None with no key)."""
    if not key_columns:
        return {"duplicate_key_rows": None, "null_key_rows": None}
    dup, nulls = _one_row(sql, keys.duplicate_key_sql(fq_table, key_columns))
    return {"duplicate_key_rows": int(dup), "null_key_rows": int(nulls)}


def _with_orphans(sql: SqlFn, fq_schema: str, table: str, fk: Dict[str, Any]) -> Dict[str, Any]:
    """The FK with `orphan_rows`: rows whose non-NULL FK matches no referenced row (ONE query)."""
    ref_table, *ref_cols = fk["references"]
    (n,) = _one_row(sql, keys.fk_orphan_sql(f"{fq_schema}.{table}", fk["columns"], f"{fq_schema}.{ref_table}",
                                            ref_cols))
    return {**fk, "orphan_rows": int(n)}


def _uc_columns(sql: SqlFn, catalog: str, schema: str) -> Dict[str, List[Tuple[str, str]]]:
    """{table: [(column, data_type), ...]} in ordinal order."""
    rows = sql(
        f"SELECT table_name, column_name, data_type FROM {catalog}.information_schema.columns "
        f"WHERE table_schema = '{schema}' ORDER BY table_name, ordinal_position"
    )
    out: Dict[str, List[Tuple[str, str]]] = {}
    for table, column, data_type in rows:
        out.setdefault(table, []).append((column, data_type))
    return out


def _uc_primary_keys(sql: SqlFn, catalog: str, schema: str) -> Dict[str, List[str]]:
    """{table: [pk column, ...]} from declared PRIMARY KEY constraints, in key order."""
    isc = f"{catalog}.information_schema"
    rows = sql(
        f"""SELECT k.table_name, k.column_name
        FROM {isc}.table_constraints t
        JOIN {isc}.key_column_usage k
          ON t.constraint_catalog = k.constraint_catalog
         AND t.constraint_schema = k.constraint_schema
         AND t.constraint_name = k.constraint_name
        WHERE t.table_schema = '{schema}' AND t.constraint_type = 'PRIMARY KEY'
        ORDER BY k.table_name, k.ordinal_position"""
    )
    out: Dict[str, List[str]] = {}
    for table, column in rows:
        out.setdefault(table, []).append(column)
    return out


def _uc_foreign_keys(sql: SqlFn, catalog: str, schema: str) -> Dict[str, List[Dict[str, Any]]]:
    """{table: [{"columns": [...], "references": [ref_table, ref_col, ...]}]} from declared FKs.

    Only FKs whose referenced table is in the same schema (the graph's source schema).
    Ordered by constraint name, so the proposal is deterministic.
    """
    isc = f"{catalog}.information_schema"
    rows = sql(
        f"""SELECT f.table_name, f.constraint_name, f.column_name, p.table_name, p.column_name
        FROM {isc}.referential_constraints r
        JOIN {isc}.key_column_usage f
          ON r.constraint_catalog = f.constraint_catalog
         AND r.constraint_schema = f.constraint_schema
         AND r.constraint_name = f.constraint_name
        JOIN {isc}.key_column_usage p
          ON r.unique_constraint_catalog = p.constraint_catalog
         AND r.unique_constraint_schema = p.constraint_schema
         AND r.unique_constraint_name = p.constraint_name
         AND p.ordinal_position = f.position_in_unique_constraint
        WHERE f.table_schema = '{schema}' AND p.table_schema = '{schema}'
        ORDER BY f.table_name, f.constraint_name, f.ordinal_position"""
    )
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for table, constraint, column, ref_table, ref_column in rows:
        fk = grouped.setdefault((table, constraint), {"table": table, "columns": [], "ref": ref_table, "ref_cols": []})
        fk["columns"].append(column)
        fk["ref_cols"].append(ref_column)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for fk in grouped.values():
        out.setdefault(fk["table"], []).append(
            {"columns": fk["columns"], "references": [fk["ref"], *fk["ref_cols"]]}
        )
    return out


def _uc_table_verdict(
    sql: SqlFn, fq_table: str, name: str, key_columns: List[str], columns: List[Tuple[str, str]]
) -> TableVerdict:
    """Row count, and per column null rate and distinct ratio, in ONE aggregate query."""
    aggs = ["COUNT(*)"]
    for col, _ in columns:
        aggs += [f"COUNT(`{col}`)", f"COUNT(DISTINCT `{col}`)"]
    row = sql(f"SELECT {', '.join(aggs)} FROM {fq_table}")[0]
    n = int(row[0])
    verdicts = []
    for i, (col, data_type) in enumerate(columns):
        non_null, distinct = int(row[1 + 2 * i]), int(row[2 + 2 * i])
        verdicts.append(
            ColumnVerdict(
                name=col,
                type_name=str(data_type),
                match_rate=non_null / n if n else 0.0,
                entropy=None,
                distinct_ratio=distinct / n if n else 0.0,
                null_rate=(n - non_null) / n if n else 0.0,
            )
        )
    patterns = _uc_pattern_verdicts(sql, fq_table, columns)
    for v in verdicts:
        v.patterns = patterns.get(v.name)
    return TableVerdict(name=name, key_columns=list(key_columns), row_count=n, columns=verdicts)


def _uc_pattern_verdicts(
    sql: SqlFn, fq_table: str, columns: List[Tuple[str, str]]
) -> Dict[str, Dict[str, float]]:
    """{column: pattern match rates} for the string columns, from one bounded sample query.

    The sampled values never leave this function; only the match rate percentages are returned.
    """
    text_cols = [col for col, data_type in columns if str(data_type).upper().startswith(("STRING", "VARCHAR", "CHAR"))]
    if not text_cols:
        return {}
    select = ", ".join(f"`{c}`" for c in text_cols)
    rows = sql(f"SELECT {select} FROM {fq_table} LIMIT {PATTERN_SAMPLE_ROWS}")
    return {c: pattern_verdicts(r[i] for r in rows) for i, c in enumerate(text_cols)}


def _compute_table_verdict(table_name: str, key_columns: List[str]) -> TableVerdict:
    """Compute verdict for a table (synthetic in Seam-2).

    Args:
        table_name: Name of the table
        key_columns: Key columns for this table

    Returns:
        TableVerdict with no sampled data
    """
    # Generate synthetic columns
    columns = []

    # Add key columns
    for key_col in key_columns:
        # Keys are typically high-entropy, low-null
        verdict = ColumnVerdict(
            name=key_col,
            type_name="VARCHAR",
            match_rate=1.0,  # Keys should be non-null
            entropy=0.95,  # High entropy
            distinct_ratio=1.0,  # All distinct (ideally)
            null_rate=0.0,
        )
        columns.append(verdict)

    return TableVerdict(
        name=table_name,
        key_columns=key_columns,
        row_count=5000,  # Synthetic count
        columns=columns,
    )
