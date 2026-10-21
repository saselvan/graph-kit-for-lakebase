"""Privacy step: the gate between the interview and build.

Reads `schema.json` (pattern verdicts only) and, if present, the agreed
`graph_mapping.yaml`. Writes `flags.json` (keeping propose's keys such as `shape`):
- `floor`: the privacy floor (default 11, raisable, never lowerable)
- `columns`: the class of every column
- `flags`: one flag per care-subject column, with its resolution (lgk.privacy.flags)
- `foreign_keys`: schema.json's declared foreign keys, so the gate refuses a node keyed on a
  neutrally named FK that reaches a care-subject column; pinned to schema.json by the gate
- `key_columns`: schema.json's table keys, so the gate knows which tables are care-keyed (an FK
  to any column of one is refused too); pinned the same way
- `folds`: the mapping's fold edges (edge type, table, subject, src, dst, form), inside
  the stamp; the gate re-derives it from the mapping and refuses a difference
- `derived`: the mapping's node join outputs (node type, name, table, column, func, the
  input column's class after reclassification, floored), inside the stamp, re-derived the same way

After it stamps flags.json, the step appends the decision to `privacy_decisions.json`, newest
last (lgk.privacy.decisions; owner decision 2026-10-05). It is the only writer of that file. The
gate refuses any decision that is not the newest one there, so restoring an older flags.json and
graph_mapping.yaml is refused until this step runs again on them.

The step FAILS, which stops the run before build, when a care-subject flag is open or
when the mapping still uses a care-subject column that was dropped or folded.
Flags are resolved one by one with `resolve_flag`, then the step is run again.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from lgk.privacy import binding, classify, decisions, floor, gate
from lgk.privacy.flags import new_flag, read_flags_file, resolve_flag, write_flags_file
from lgk.privacy.gate import MappingLeakError

logger = logging.getLogger(__name__)

__all__ = ["privacy", "resolve_flag", "MappingLeakError"]


def privacy(config: Dict[str, Any]) -> bool:
    """Run the gate: classify, merge earlier decisions, refuse what is not clear, then stamp.

    flags.json is written before the checks (so open flags are visible to resolve), but the
    stamp that build and provision require (lgk.privacy.gate) is added only when every check
    passed. The step is re-run on every run (`always_rerun`): an earlier pass never stands in
    for a check of the current schema.json and graph_mapping.yaml. Returns whether the last
    build was made under a different decision (see `invalidates_on_change`).
    """
    run_folder = str(config.get("run_folder", "."))
    privacy_floor = floor.checked_floor(config.get("privacy_floor"))
    schema = gate.read_schema(run_folder)
    tables = schema.get("tables", [])
    columns = gate.classified_columns(tables)
    previous = read_flags_file(run_folder)
    flags = _merge_flags(previous.get("flags", []), columns, tables)
    mapping = gate.read_run_mapping(run_folder)
    data = {k: v for k, v in previous.items() if k != gate.STAMP_KEY}
    data.update(floor=privacy_floor, source=gate.profiled_source(schema), columns=columns,
                flags=flags, foreign_keys=gate.foreign_keys_record(tables),
                key_columns=gate.key_columns_record(tables),
                folds=gate.folds_record(mapping))
    data["derived"] = gate.derived_record(data, mapping)
    write_flags_file(run_folder, data)
    gate.check_flags(data)
    gate.check_mapping(data, mapping)
    stamped = gate.stamp(run_folder, data)
    write_flags_file(run_folder, stamped)
    # the newest decision (owner decision 2026-10-05): the gate refuses any other recorded one
    decisions.record_decision(run_folder, binding.stamp_id(stamped))
    logger.info(f"Privacy gate passed: {len(flags)} care-subject flags resolved, floor {privacy_floor}")
    # True when the tables were not built under THIS decision (it changed since the last
    # build, also by a privacy run outside the runner, or nothing was built yet): the runner
    # then reruns build/enrich/provision
    return binding.built_stamp(run_folder) != binding.stamp_id(stamped)


privacy.always_rerun = True  # type: ignore[attr-defined]
# Steps whose output rests on the privacy decision. When privacy returns True (the last
# build was made under another stamp, or none), the runner marks them not_run in run.json, so it rebuilds
# instead of skipping a build made under the old decision (provision would refuse it).
privacy.invalidates_on_change = ("build", "enrich", "provision")  # type: ignore[attr-defined]


def _fold_allowed(table: Dict[str, Any], column: str) -> bool:
    """A key in an edge or join table: a foreign-key column, or part of a composite key.

    A table's single-column key is never foldable: it IS the care subject's node, there is
    nothing on the other side of it to fold onto.
    """
    fk_cols = {c for fk in table.get("foreign_keys", []) for c in fk.get("columns", [])}
    keys = table.get("key_columns", [])
    if keys == [column]:
        return False
    return column in fk_cols or (len(keys) >= 2 and column in keys)


def _merge_flags(previous: List[Dict[str, Any]], columns: List[Dict[str, Any]],
                 tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One flag per care-subject column now in the schema; earlier decisions are kept."""
    by_name = {t["name"]: t for t in tables}
    kept = {(f["table"], f["column"]): f for f in previous}
    flags = []
    for col in columns:
        if col["class"] != classify.CARE_SUBJECT:
            continue
        key = (col["table"], col["column"])
        flags.append(kept.get(key) or new_flag(
            col["table"], col["column"], col["detected_by"],
            _fold_allowed(by_name[col["table"]], col["column"]),
        ))
    return flags
