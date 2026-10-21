"""Bind built and served tables to the privacy decision they were made under.

The gate (lgk.privacy.gate) clears the CURRENT run-folder inputs. That alone does not prove
the tables about to be served were built under that decision: a build made under an older,
looser decision is skipped on rerun and would be served after the operator tightened it.
So every table carries the stamp it was built from, and nothing is served on a mismatch:

- build writes the stamp to each Delta table's TBLPROPERTIES (`lgk.privacy_stamp`) and to
  `privacy_bindings.json` in the run folder ("build"); on a graph with fold edge types it also
  sets `lgk.fold_floor` on `edges` and records `build.folds`, which provision compares
  with the current floor (`check_build_folds`); a graph with node joins records `build.joins`,
  compared the same way (`check_build_joins`);
- provision refuses unless BOTH the run-folder record and every Delta table's property equal
  the current stamp (a table with no property, from before this change, is refused too);
  after syncing it records the stamp of the version schema it made ("versions");
- switch refuses to point the served views at a version whose recorded stamp differs;
- enrich stamps each output table the same way and records "enrich"; provision checks
  both for every enrichment table it syncs.

The label on nodes, edges and node_id_map (owner decision, nightly stamp step) is not the bare
stamp but one JSON object, the same on every chain (the kit build and the operator's stamp step
write it with one writer, lgk.refresh.label.write_labels):

    {"certified_version": <int>, "stamp": "<stamp_id>", "table_id": "<UC table_id>"}

certified_version is the last Delta version whose data was checked (N-1); writing the label is
commit N, a SET TBLPROPERTIES commit that changes no data. The run manifest records the same three
per table. Enrichment tables keep the bare stamp: only the kit writes them.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from lgk.privacy.decisions import decision_id
from lgk.privacy.gate import STAMP_KEY, PrivacyGateError, require_clear

STAMP_PROPERTY = "lgk.privacy_stamp"
FOLD_FLOOR_PROPERTY = "lgk.fold_floor"
BINDINGS_FILE = "privacy_bindings.json"
BUILT_TABLES = ("nodes", "edges", "node_id_map")
LABEL_FIELDS = ("certified_version", "stamp", "table_id")

Sql = Callable[[str], List[List[Any]]]


class StaleBuildError(PrivacyGateError):
    """The tables to serve were built (or synced) under a different privacy decision."""


def stamp_id(flags_data: Dict[str, Any]) -> str:
    """One string for the stamp in a CLEARED flags.json (the dict require_clear returns)."""
    recorded = flags_data.get(STAMP_KEY)
    if not recorded:
        raise PrivacyGateError("flags.json has no privacy stamp to bind tables to")
    return decision_id(recorded)


def _path(run_folder: str) -> Path:
    return Path(run_folder) / BINDINGS_FILE


def _read(run_folder: str) -> Dict[str, Any]:
    path = _path(run_folder)
    return json.loads(path.read_text()) if path.exists() else {}


def _write(run_folder: str, data: Dict[str, Any]) -> None:
    _path(run_folder).write_text(json.dumps(data, indent=2, sort_keys=True))


# ----------------------------------------------------------------------------- build
def set_stamp_sql(fq_table: str, stamp: str) -> str:
    """The statement build runs on each Delta table once its rows are written."""
    return f"ALTER TABLE {fq_table} SET TBLPROPERTIES ('{STAMP_PROPERTY}' = '{stamp}')"


def label_json(stamp: str, table_id: str, certified_version: int) -> str:
    """The label value of a kit table: one compact JSON object, keys sorted (one spelling everywhere)."""
    value = {"certified_version": int(certified_version), "stamp": str(stamp), "table_id": str(table_id)}
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def parse_label(raw: Any) -> Optional[Dict[str, Any]]:
    """The label as a dict, or None unless it is exactly the three fields with their types."""
    try:
        value = json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        return None
    if not isinstance(value, dict) or tuple(sorted(value)) != LABEL_FIELDS:
        return None
    version = value["certified_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        return None
    if not (isinstance(value["stamp"], str) and isinstance(value["table_id"], str)):
        return None
    return value


def set_label_sql(fq_table: str, label: str) -> str:
    """The statement that writes a kit table's label (label_json) as its `lgk.privacy_stamp`."""
    if "'" in label:
        raise ValueError(f"a label never holds a quote: {label!r}")
    return f"ALTER TABLE {fq_table} SET TBLPROPERTIES ('{STAMP_PROPERTY}' = '{label}')"


def set_fold_floor_sql(fq_edges: str, floor: int) -> str:
    """The statement build runs on `edges` of a graph with fold types, next to the stamp."""
    return f"ALTER TABLE {fq_edges} SET TBLPROPERTIES ('{FOLD_FLOOR_PROPERTY}' = '{int(floor)}')"


def record_build(run_folder: str, stamp: str, fq_tables: Iterable[str],
                 folds: Optional[Dict[str, Dict[str, Any]]] = None,
                 joins: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    """Record the build. `folds` is assert_fold_floor's report ({edge_type: {floor, min_weight,
    rows}}); written only when the graph has fold types, so an unfolded build's record is unchanged.
    `joins` is assert_join_floor's report ({"<node_type>.<output>": {floor, floored}}),
    written only when the mapping has node joins."""
    data = _read(run_folder)
    data["build"] = {"stamp": stamp, "tables": sorted(fq_tables)}
    if folds:
        data["build"]["folds"] = {t: dict(r) for t, r in sorted(folds.items())}
    if joins:
        data["build"]["joins"] = {name: dict(r) for name, r in sorted(joins.items())}
    _write(run_folder, data)


def built_stamp(run_folder: str) -> Any:
    """The stamp the last recorded build was made under, or None."""
    return (_read(run_folder).get("build") or {}).get("stamp")


# ---------------------------------------------------------------------------- enrich
def record_enrich(run_folder: str, stamp: str, fq_tables: Iterable[str], methods: Iterable[str]) -> None:
    """Enrich's record: the stamp its output tables carry, the tables, the methods."""
    data = _read(run_folder)
    data["enrich"] = {"stamp": stamp, "tables": sorted(fq_tables), "methods": list(methods)}
    _write(run_folder, data)


# ------------------------------------------------------------------------- provision
def check_build_record(run_folder: str, stamp: str, fq_tables: Iterable[str]) -> None:
    """The run folder says these tables were built under `stamp`."""
    build = _read(run_folder).get("build")
    if not build:
        raise StaleBuildError(
            f"{_path(run_folder)} records no build: these tables were not built under the "
            f"current privacy decision (or were built before builds were bound to it). "
            f"Run build again, then provision."
        )
    if build.get("stamp") != stamp:
        raise StaleBuildError(
            "the tables were built under a different privacy decision than the current one. "
            "Run build again (the runner does this after a privacy change), then provision."
        )
    missing = sorted(set(fq_tables) - set(build.get("tables", [])))
    if missing:
        raise StaleBuildError(f"the recorded build did not make {', '.join(missing)}; run build again")


def check_build_folds(run_folder: str, floor: int, fold_types: Sequence[str]) -> None:
    """The recorded build folded every current fold type, each under the current floor.

    The floor is inside the stamp, so check_build_record already refuses a build from another
    decision; this refuses a build record whose folds say otherwise."""
    folds = (_read(run_folder).get("build") or {}).get("folds") or {}
    other = sorted(f"{t} (floor {r.get('floor')!r})" for t, r in folds.items() if r.get("floor") != floor)
    missing = sorted(set(fold_types) - set(folds))
    if other or missing:
        found = ", ".join(other + [f"{t} (not recorded)" for t in missing])
        raise StaleBuildError(
            f"the recorded build's folded edges were not made under the current privacy floor "
            f"{floor}: {found}. Run build again, then provision."
        )


def check_build_joins(run_folder: str, floor: int, outputs: Iterable[str]) -> None:
    """The recorded build made every current join output, each under the current floor."""
    joins = (_read(run_folder).get("build") or {}).get("joins") or {}
    other = sorted(f"{name} (floor {r.get('floor')!r})" for name, r in joins.items() if r.get("floor") != floor)
    missing = sorted(set(outputs) - set(joins))
    if other or missing:
        found = ", ".join(other + [f"{name} (not recorded)" for name in missing])
        raise StaleBuildError(
            f"the recorded build's node join outputs were not made under the current privacy "
            f"floor {floor}: {found}. Run build again, then provision."
        )


def table_label(sql: Sql, fq_table: str) -> Any:
    """The raw `lgk.privacy_stamp` value of a Delta table, or None."""
    return {str(row[0]): row[1] for row in sql(f"SHOW TBLPROPERTIES {fq_table}")}.get(STAMP_PROPERTY)


def is_built_table(fq_table: str) -> bool:
    """nodes, edges or node_id_map: the tables whose label is the JSON label, not the bare stamp."""
    return fq_table.rsplit(".", 1)[-1] in BUILT_TABLES


def _stamp_problem(fq: str, found: Any, stamp: str) -> Optional[str]:
    """Why `found` (the raw property) does not bind `fq` to `stamp`, or None."""
    if found is None:
        return f"no {STAMP_PROPERTY}"
    if is_built_table(fq):
        label = parse_label(found)
        if label is None:
            return "the label is not the {stamp, table_id, certified_version} JSON"
        found = label["stamp"]
    return None if found == stamp else "stale stamp"


def check_table_stamps(sql: Sql, fq_tables: Iterable[str], stamp: str) -> None:
    """Every Delta table carries `stamp` in TBLPROPERTIES; a missing property is refused.

    On nodes, edges and node_id_map the property must be the JSON label (label_json) and its
    `stamp` the one checked; the sync task checks the rest of the label (lgk.refresh.label)."""
    bad = []
    for fq in fq_tables:
        problem = _stamp_problem(fq, table_label(sql, fq), stamp)
        if problem:
            bad.append(f"{fq} ({problem})")
    if bad:
        raise StaleBuildError(
            f"Delta table(s) not built under the current privacy decision: {', '.join(bad)}. "
            f"Run build again, then provision."
        )


def check_enrich_record(run_folder: str, stamp: str, fq_tables: Iterable[str]) -> None:
    """The run folder says enrich wrote these tables under `stamp` (nothing to check if none)."""
    wanted = sorted(fq_tables)
    if not wanted:
        return
    enrich = _read(run_folder).get("enrich")
    if not enrich:
        raise StaleBuildError(
            f"{_path(run_folder)} records no enrich run: {', '.join(wanted)} were not computed "
            f"under the current privacy decision. Run enrich again, then provision."
        )
    if enrich.get("stamp") != stamp:
        raise StaleBuildError(
            "the enrichment tables were computed under a different privacy decision than the "
            "current one. Run enrich again, then provision."
        )
    missing = sorted(set(wanted) - set(enrich.get("tables", [])))
    if missing:
        raise StaleBuildError(f"the recorded enrich run did not write {', '.join(missing)}; run enrich again")


def record_version(run_folder: str, version_schema: str, stamp: str) -> None:
    data = _read(run_folder)
    data.setdefault("versions", {})[version_schema] = stamp
    _write(run_folder, data)


# ---------------------------------------------------------------------------- switch
def check_version(run_folder: str, version_schema: str, stamp: str) -> None:
    """The version about to be served was synced under the current privacy decision."""
    recorded = _read(run_folder).get("versions", {}).get(version_schema)
    if recorded is None:
        raise StaleBuildError(
            f"{version_schema} has no recorded privacy stamp in {_path(run_folder)}: it was "
            f"not provisioned by this run under the current decision, so it is not served."
        )
    if recorded != stamp:
        raise StaleBuildError(
            f"{version_schema} was provisioned under a different privacy decision than the "
            f"current one; it is not served. Build and provision a new version."
        )


def require_servable(run_folder: str, version_schema: str) -> str:
    """The one serving check (switch, rollback and reapply): gate clear, version made under it.

    The privacy gate must be clear for the CURRENT, newest decision (require_clear), and
    `version_schema` must have been provisioned under that decision (check_version). Reads
    files only, so a caller runs it before it connects or changes anything. Returns the stamp.
    """
    stamp = stamp_id(require_clear(run_folder))
    check_version(run_folder, version_schema, stamp)
    return stamp
