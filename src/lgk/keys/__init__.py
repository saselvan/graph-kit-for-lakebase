"""Key checks: UC PRIMARY/FOREIGN KEY constraints are informational, so the kit checks the data.

Two rules, applied in profile (the declared keys) and again in build (the mapping's keys; the
nightly job rebuilds without profile):

1. **A duplicate key fails.** A key is rendered the way the node id renders it (single column:
   CAST(col AS STRING); composite: lgk.privacy.floor.node_key_expr, a JSON array of the parts), so
   int 7 and string '7' are one key, exactly as they would be one node id. `duplicate_key_rows`
   = rows with a non-NULL key minus the distinct keys (the rows a silent dedup would have
   dropped). A row with a NULL single-column key is NOT a duplicate: build drops it (it would
   hash to one shared node id); profile counts it as `null_key_rows`. A composite key with a
   NULL part is kept by build (JSON null at its position), so it counts toward duplicates and is
   also counted in `null_key_rows`.
2. **Orphans are the operator's decision.** An orphan is a row whose FK (profile) or edge end
   (build) holds a non-NULL value with no matching row (profile: the referenced table; build:
   the node of the end's type, i.e. the node type's source rows with a non-NULL key). A NULL FK
   is not an orphan: there is no link to make. Whatever the reason a node is missing (no row,
   a NULL key, a key of another rendering), the edge row counts as an orphan the same way,
   because build drops exactly those edges. (An edge end on a composite-key node type is refused
   earlier, by the mapping contract.)

The operator decides each FK with orphans: `accept` (build drops those edge rows, as before, and
records how many) or `fixed` (the data was fixed; build then requires zero). The record is
`key_decisions.json` in the run folder, NOT flags.json: the privacy gate hashes flags.json,
schema.json and graph_mapping.yaml, and a data-quality decision must not re-lock privacy.

    {"observations": [{"seq", "table", "columns", "orphan_rows", "source", "at"}, ...],
     "decisions":    [{"seq", "table", "columns", "choice", "chooser", "reason",
                       "orphan_rows", "observation", "at"}, ...],
     "dropped":      [{"edge_type", "table", "column", "node_type", "dropped_rows"}, ...]}

A decision carries the count it was made on and the seq of that observation. Build fails when
the live count is larger than the accepted count (an accept never silently covers growth), and
when a decision does not match its observation (a hand-edited count). The run folder is the only
record, so a person who edits BOTH entries consistently can still forge an accept: the same
residual risk as the privacy decision history (docs/privacy-decision-history.md).

Everything here carries counts and column/table names only, never a key value.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

KEY_DECISIONS_FILE = "key_decisions.json"
CHOICES = ("accept", "fixed")

Columns = Union[str, Sequence[str]]


class DuplicateKeyError(RuntimeError):
    """A key the graph relies on is not unique in the data (counts only, never a value)."""


class OrphanKeyError(RuntimeError):
    """Rows reference a key with no matching row, and no recorded decision covers them."""


class KeyDecisionsFormatError(ValueError):
    """key_decisions.json is not JSON, or not the documented shape."""


def _cols(columns: Columns) -> List[str]:
    return [columns] if isinstance(columns, str) else list(columns)


def _rows_word(n: int, noun: str = "row") -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


# ------------------------------------------------------------------------------- SQL
def key_expr(columns: Columns, alias: Optional[str] = None) -> str:
    """A key as the node id renders it: CAST(col AS STRING), or the composite JSON array."""
    from lgk.privacy import floor

    cols = _cols(columns)
    if len(cols) == 1:
        return f"CAST({alias + '.' if alias else ''}{cols[0]} AS STRING)"
    return floor.node_key_expr(cols, alias)


def duplicate_key_sql(fq_table: str, columns: Columns) -> str:
    """ONE row: (duplicate_key_rows, null_key_rows) of `columns` in `fq_table`. Counts only."""
    cols = _cols(columns)
    k = key_expr(cols)
    nulls = " OR ".join(f"{c} IS NULL" for c in cols)
    return (f"SELECT COUNT({k}) - COUNT(DISTINCT {k}) AS duplicate_key_rows, "
            f"COUNT_IF({nulls}) AS null_key_rows FROM {fq_table}")


def duplicate_key_check_sql(fq_table: str, columns: Columns) -> str:
    """Build's form: one row (the count) only when there are duplicates; no row when clean."""
    return (f"SELECT duplicate_key_rows FROM ({duplicate_key_sql(fq_table, columns)}) "
            f"WHERE duplicate_key_rows > 0")


def fk_orphan_sql(child_fq: str, columns: Columns, parent_fq: str, ref_columns: Columns) -> str:
    """ONE row: the child rows whose (non-NULL) FK matches no parent row, compared as rendered."""
    pairs = list(zip(_cols(columns), _cols(ref_columns)))
    on = " AND ".join(f"CAST(c.{a} AS STRING) = CAST(p.{b} AS STRING)" for a, b in pairs)
    where = " AND ".join(f"c.{a} IS NOT NULL" for a, _ in pairs)
    return f"SELECT COUNT(*) AS orphan_rows FROM {child_fq} c LEFT ANTI JOIN {parent_fq} p ON {on} WHERE {where}"


def edge_orphan_sql(edge_source_fq: str, column: str, node_type: str, node_source_fq: str,
                    natural_key: Columns) -> str:
    """Build's form: the edge rows whose non-NULL `column` has no `node_type` node; a row only when > 0.

    The node set is the node type's source rows with a non-NULL key, its id from the same
    formula build's nodes INSERT uses (lgk.privacy.floor.node_id_expr), so this counts exactly
    the edge rows build's semi-joins drop."""
    from lgk.privacy import floor

    nk = natural_key if isinstance(natural_key, str) else list(natural_key)
    return (f"SELECT orphan_rows FROM (SELECT COUNT(*) AS orphan_rows FROM {edge_source_fq} e "
            f"LEFT ANTI JOIN (SELECT {floor.node_id_expr(node_type, nk, 'n')} AS node_id "
            f"FROM {node_source_fq} n WHERE {floor.node_key_expr(nk, 'n')} IS NOT NULL) m "
            f"ON m.node_id = xxhash64('{node_type}::'||e.{column}) WHERE e.{column} IS NOT NULL) "
            f"WHERE orphan_rows > 0")


def node_readback_sql(fq_nodes: str) -> str:
    """One row (the count) only when the built nodes table holds a node id twice."""
    return (f"SELECT duplicate_node_ids FROM (SELECT COUNT(*) - COUNT(DISTINCT node_id) AS duplicate_node_ids "
            f"FROM {fq_nodes}) WHERE duplicate_node_ids > 0")


def duplicate_message(where: str, table: str, columns: Columns, count: int) -> str:
    return (f"{where} (table {table}, key {', '.join(_cols(columns))}): {_rows_word(count, 'duplicate key row')}. "
            f"UC key constraints are not enforced, so the data has more than one row per key; "
            f"fix the source (or map a view with one row per key), then run again")


# ------------------------------------------------------------------------- the record
def decisions_path(run_folder: str) -> Path:
    return Path(run_folder) / KEY_DECISIONS_FILE


def read_key_decisions(run_folder: str) -> Dict[str, List[Dict[str, Any]]]:
    """The record (empty lists when the file is absent). KeyDecisionsFormatError if malformed."""
    path = decisions_path(run_folder)
    if not path.exists():
        return {"observations": [], "decisions": [], "dropped": []}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise KeyDecisionsFormatError(f"{path} is not JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise KeyDecisionsFormatError(f"{path} must hold an object")
    out = {}
    for name in ("observations", "decisions", "dropped"):
        value = data.get(name, [])
        if not isinstance(value, list) or not all(isinstance(e, dict) for e in value):
            raise KeyDecisionsFormatError(f"{path}: '{name}' must be a list of objects")
        out[name] = value
    return out


def _write(run_folder: str, data: Dict[str, Any]) -> None:
    decisions_path(run_folder).write_text(json.dumps(data, indent=2))


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _same(entry: Dict[str, Any], table: str, cols: List[str]) -> bool:
    return entry.get("table") == table and entry.get("columns") == cols


def _newest(entries: List[Dict[str, Any]], table: str, cols: List[str]) -> Optional[Dict[str, Any]]:
    found = [e for e in entries if _same(e, table, cols)]
    return found[-1] if found else None


def _is_count(value: Any, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _append_observations(data: Dict[str, Any], observed: List[Tuple[str, List[str], int]], source: str) -> None:
    at = _now()
    for table, cols, count in observed:
        data["observations"].append({"seq": len(data["observations"]) + 1, "table": table, "columns": cols,
                                     "orphan_rows": int(count), "source": source, "at": at})


def record_observations(run_folder: str, observed: List[Tuple[str, Columns, int]], source: str) -> None:
    """Append the orphan counts seen by `source` (profile or build), one entry per FK."""
    data = read_key_decisions(run_folder)
    _append_observations(data, [(t, _cols(c), n) for t, c, n in observed], source)
    _write(run_folder, data)


def _uncovered(data: Dict[str, Any], table: str, cols: List[str], count: int) -> Optional[str]:
    """Why `count` orphans of (table, cols) are NOT covered by the newest decision; None if covered."""
    d = _newest(data["decisions"], table, cols)
    if d is None:
        return "no decision is recorded"
    if d.get("choice") not in CHOICES:
        return f"decision {d.get('seq')} has no valid choice"
    if d["choice"] == "fixed":
        return f"decision {d.get('seq')} recorded 'fixed', but the data still has them"
    accepted = d.get("orphan_rows")
    if not _is_count(accepted, 1):
        return f"decision {d.get('seq')} has no valid orphan_rows count"
    obs = [o for o in data["observations"] if o.get("seq") == d.get("observation") and not isinstance(o.get("seq"), bool)]
    if len(obs) != 1 or not _same(obs[0], table, cols) or obs[0].get("orphan_rows") != accepted:
        return f"decision {d.get('seq')} does not match the observation it was made on"
    if count > accepted:
        return f"the decision accepted {accepted}"
    return None


def resolve_orphans(run_folder: str, table: str, fk: Columns, choice: str, chooser: str,
                    reason: Optional[str] = None) -> Dict[str, Any]:
    """Record the operator's decision on the orphans of one FK; returns the decision.

    `fk` is the FK column (or the list of columns of a composite FK). `choice`: `accept` (keep
    building; those edge rows are dropped, up to the observed count) or `fixed` (the data was
    fixed; build then requires zero). The decision is made on the NEWEST observed count for the
    FK, and records it. KeyError when no orphans are observed for the FK."""
    cols = _cols(fk)
    if choice not in CHOICES:
        raise ValueError(f"choice must be one of {', '.join(CHOICES)}, got {choice!r}")
    if not isinstance(chooser, str) or not chooser.strip():
        raise ValueError("chooser is required: the name of the person deciding")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("reason must be text or None")
    data = read_key_decisions(run_folder)
    obs = _newest(data["observations"], table, cols)
    if obs is None or not _is_count(obs.get("orphan_rows"), 1):
        raise KeyError(f"no orphans observed for {table}.{','.join(cols)} in {KEY_DECISIONS_FILE}: "
                       "run profile or build first")
    decision = {"seq": len(data["decisions"]) + 1, "table": table, "columns": cols, "choice": choice,
                "chooser": chooser.strip(), "reason": reason, "orphan_rows": obs["orphan_rows"],
                "observation": obs["seq"], "at": _now()}
    data["decisions"].append(decision)
    _write(run_folder, data)
    return decision


def open_orphans(run_folder: str) -> List[Dict[str, Any]]:
    """Each FK whose newest observed orphan count is > 0 and not covered by its newest decision.

    Entries: table, columns, orphan_rows (newest observed), accepted_rows (the newest decision's
    count when it is an accept, else None), why (counts and names only)."""
    data = read_key_decisions(run_folder)
    keys: List[Tuple[str, Tuple[str, ...]]] = []
    for o in data["observations"]:
        k = (o.get("table"), tuple(o.get("columns") or []))
        if k not in keys:
            keys.append(k)
    out = []
    for table, cols_t in keys:
        cols = list(cols_t)
        obs = _newest(data["observations"], table, cols)
        count = obs.get("orphan_rows") if obs else 0
        if not _is_count(count, 1):
            continue
        why = _uncovered(data, table, cols, count)
        if why is None:
            continue
        d = _newest(data["decisions"], table, cols)
        accepted = d.get("orphan_rows") if d and d.get("choice") == "accept" else None
        out.append({"table": table, "columns": cols, "orphan_rows": count, "accepted_rows": accepted, "why": why})
    return out


def apply_edge_orphans(run_folder: str, ends: List[Dict[str, Any]]) -> None:
    """Build's gate on the live orphan counts of each edge end (before any kit table is replaced).

    `ends`: {edge_type, table, column, node_type, orphan_rows}. Records the counts (an end with
    orphans, or one observed before), then raises OrphanKeyError naming every end whose orphans no
    decision covers. Otherwise records, per accepted end, the edge rows build will drop."""
    data = read_key_decisions(run_folder)
    seen = {(o.get("table"), tuple(o.get("columns") or [])) for o in data["observations"]}
    observed, failures, dropped = [], [], []
    for end in ends:
        table, cols, count = end["table"], [end["column"]], int(end["orphan_rows"])
        if count > 0 or (table, tuple(cols)) in seen:
            observed.append((table, cols, count))
        if count == 0:
            continue
        why = _uncovered(data, table, cols, count)
        if why is not None:
            failures.append(f"{table}.{end['column']} (edge {end['edge_type']}, end {end['node_type']}): "
                            f"{_rows_word(count)} reference no {end['node_type']} node and {why}")
        else:
            dropped.append({"edge_type": end["edge_type"], "table": table, "column": end["column"],
                            "node_type": end["node_type"], "dropped_rows": count})
    if not observed and not failures and not decisions_path(run_folder).exists():
        return
    _append_observations(data, observed, "build")
    if not failures:
        data["dropped"] = dropped
    _write(run_folder, data)
    if failures:
        raise OrphanKeyError(
            "orphan keys (UC foreign keys are not enforced): " + "; ".join(failures) + ". Fix the data, or "
            "record a decision per FK: lgk.keys.resolve_orphans(run_folder, table, fk, 'accept' or 'fixed', "
            "chooser=..., reason=...); lgk.keys.open_orphans(run_folder) lists them")
