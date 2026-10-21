"""The one shared privacy gate check: build and provision call it before any SQL.

The privacy step decides; this module enforces. `require_clear(run_folder, mapping)` refuses
unless ALL of these hold:
1. flags.json exists and carries a stamp written by a privacy step that PASSED;
2. the stamp is fresh: schema.json, graph_mapping.yaml and flags.json (without the stamp) hash
   to what they were when the step passed. Any later edit (a re-added dropped column, a new
   care-subject column in schema.json, a flag changed by hand) makes the stamp stale;
2b. (owner decision 2026-10-05) the stamp is the NEWEST decision the privacy step recorded in
   privacy_decisions.json (lgk.privacy.decisions): an exact older copy of flags.json and
   graph_mapping.yaml is refused (OlderDecisionError), and so is a stamp the step never
   recorded or a missing or malformed history (UnrecordedDecisionError). Running the privacy
   step again records the restored decision as the newest;
3. no care-subject flag is open, and every care-subject column has a flag;
4. every column the mapping carries (node key, label, properties; edge src and dst) has a
   classification. Doubt plays safe: an unclassified column is refused, never passed;
5. no mapping column is a dropped or folded care-subject column;
6. the mapping reads the source schema.json profiled (source_catalog / source_schema), and
   flags.json records that same source;
7. flags.json's classes are what classify gives for schema.json now, so a class edited by
   hand (with the stamp recomputed) is refused; likewise its `foreign_keys` are schema.json's;
7b. No node type is keyed on a care-subject column, or on a declared foreign key that
   reaches one through a chain of FKs (`check_care_key_nodes`);
8. Every edge `fold_subject` is a care-subject column whose flag chose `fold`, every
   `fold` flag is folded by some edge, and flags.json's `folds` is what the mapping gives now
   (`check_fold_subjects`);
9. Every node `walk_key` is one of that node type's `properties` (`check_walk_keys`),
   so a set walk tests only columns the checks above have already cleared. A join output is
   never a walk key (it is not a native property);
10. A node join that aggregates a table holding care-subject data names a folded
   `subject` (`check_join_subjects`); no join output takes a public directory property name
   (`check_join_outputs_not_public`); and flags.json's `derived` is what the mapping gives now.

A walk's `folded` is derived per call from the edge types it selects (`served_walk_selection`):
all fold types -> folded, none -> unfolded, both -> MixedFoldSelectionError.

Checks 3-6 are also what the privacy step runs before it stamps (`check_mapping`,
`check_flags`), so the step and the gate cannot disagree.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from lgk.mapping import GraphMapping, load_mapping
from lgk.served_functions import DIRECTORY_NODE_TYPE, DIRECTORY_PROPERTIES
from lgk.privacy import classify, decisions, floor
from lgk.privacy.flags import OpenFlagError, flags_path, is_open, read_flags_file
from lgk.walks.sql import checked_edge_types

STAMP_KEY = "gate"
STAMPED_FILES = ("schema.json", "graph_mapping.yaml")


class PrivacyGateError(RuntimeError):
    """The privacy gate is not clear for this run folder; nothing may be built or served."""


class UnrecordedDecisionError(PrivacyGateError):
    """flags.json's decision is not in privacy_decisions.json, or that history is missing or malformed."""


class OlderDecisionError(PrivacyGateError):
    """flags.json + graph_mapping.yaml are an older recorded decision, not the newest one."""


class MappingLeakError(PrivacyGateError):
    """graph_mapping.yaml carries a dropped/folded care-subject column, or an unclassified one."""


class CareKeyNodeError(MappingLeakError):
    """graph_mapping.yaml has a node type keyed on a care-subject column (that person's node)."""


class MixedFoldSelectionError(PrivacyGateError):
    """A walk selected folded and unfolded edge types at once: `support` would mean two things."""


# ---------------------------------------------------------------------------- the stamp
def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "absent"


def _flags_hash(data: Dict[str, Any]) -> str:
    body = {k: v for k, v in data.items() if k != STAMP_KEY}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def input_hashes(run_folder: str, data: Dict[str, Any]) -> Dict[str, str]:
    """Hashes of every input the gate decision rests on."""
    hashes = {name: _file_hash(Path(run_folder) / name) for name in STAMPED_FILES}
    hashes["flags.json"] = _flags_hash(data)
    return hashes


def stamp(run_folder: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """`data` with a fresh stamp; the privacy step writes this only after every check passed."""
    return {**data, STAMP_KEY: input_hashes(run_folder, data)}


def _check_stamp(run_folder: str, data: Dict[str, Any]) -> None:
    recorded = data.get(STAMP_KEY)
    if not recorded:
        raise PrivacyGateError(
            f"{flags_path(run_folder)} has no privacy stamp: the privacy step has not passed "
            f"for this run. Run the privacy step and resolve every flag first."
        )
    now = input_hashes(run_folder, data)
    changed = sorted(name for name in now if recorded.get(name) != now[name])
    if changed:
        raise PrivacyGateError(
            f"privacy decision is stale: {', '.join(changed)} changed after the privacy step "
            f"passed. Run the privacy step again before build or provision."
        )


def check_newest_decision(run_folder: str, data: Dict[str, Any]) -> None:
    """The stamp in `data` is the newest decision the privacy step recorded, else refuse.

    Read-only: only the privacy step writes the history (decisions.record_decision)."""
    current = decisions.decision_id(data[STAMP_KEY])
    path = decisions.history_path(run_folder)
    try:
        entries = decisions.read_history(run_folder)
    except FileNotFoundError:
        raise UnrecordedDecisionError(
            f"{path.name} not found in {run_folder}: the privacy step records every decision it "
            f"passes there, and the gate serves only the newest. Run the privacy step again."
        ) from None
    except decisions.HistoryFormatError as exc:
        raise UnrecordedDecisionError(
            f"{path} is malformed ({exc}); it is never read as 'allow'. Restore it, or delete it "
            f"and run the privacy step again."
        ) from None
    newest = entries[-1]
    if newest["decision"] == current:
        return
    older = [e for e in entries if e["decision"] == current]
    if not older:
        raise UnrecordedDecisionError(
            f"the privacy decision in flags.json ({current[:12]}) was not recorded by the privacy "
            f"step in {path.name}. Run the privacy step again."
        )
    last = older[-1]
    raise OlderDecisionError(
        f"flags.json and graph_mapping.yaml are an older privacy decision: decision "
        f"{last['seq']} of {newest['seq']} ({current[:12]}, recorded {last.get('at')}); the newest "
        f"is decision {newest['seq']} ({newest['decision'][:12]}). An older decision is not used "
        f"again by restoring its files. Run the privacy step again on these files to make this "
        f"decision the newest."
    )


# ------------------------------------------------------------------- shared rule checks
def check_flags(data: Dict[str, Any]) -> None:
    """No open flag, and every care-subject column has a flag (a deleted flag is not a pass)."""
    flags = data.get("flags", [])
    still_open = [f"{f['table']}.{f['column']}" for f in flags if is_open(f)]
    if still_open:
        raise OpenFlagError(
            f"{len(still_open)} open care-subject flag(s): {', '.join(still_open)}. Resolve each "
            f"with resolve_flag (fold / drop / reclassify), then run the privacy step again."
        )
    flagged = {(f["table"], f["column"]) for f in flags}
    unflagged = sorted(
        f"{c['table']}.{c['column']}" for c in data.get("columns", [])
        if c.get("class") == classify.CARE_SUBJECT and (c["table"], c["column"]) not in flagged
    )
    if unflagged:
        raise OpenFlagError(f"care-subject column(s) with no flag: {', '.join(unflagged)}")


def mapping_columns(mapping: GraphMapping) -> Iterator[Tuple[str, str, str]]:
    """(source table, column, where) for every column the mapping carries into the graph."""
    for node_type, n in mapping.nodes.items():
        keys = n.natural_key if isinstance(n.natural_key, list) else [n.natural_key]
        for col in keys:
            yield n.source_table, col, f"node {node_type} natural_key"
        yield n.source_table, n.label, f"node {node_type} label"
        for col in n.properties:
            yield n.source_table, col, f"node {node_type} properties"
        yield from _join_columns(node_type, n)
    for edge_type, e in mapping.edges.items():
        yield e.source_table, e.src, f"edge {edge_type} src"
        yield e.source_table, e.dst, f"edge {edge_type} dst"


# ------------------------------------------------------------------- walk keys
def check_walk_keys(mapping: GraphMapping) -> None:
    """Every `walk_keys` entry is a `properties` entry of its node type, else MappingLeakError.

    A walk key is tested by equality on `nodes.properties ->> '<key>'`; a key that is not a
    property is either never carried (every walk on it is empty) or names a column privacy did
    not clear for that node type, so it is refused, naming the key and the node type.

    A node join output is NOT a property here, so it is refused as a walk key: an
    aggregate output is a count / sum over care-subject rows, and testing it by equality from a
    seed or stop key would turn a floored figure into a selector. One-to-one `columns` outputs
    are refused too: the narrower rule needs no second classification path."""
    bad = sorted(f"{k} (node {t})" for t, n in mapping.nodes.items() for k in n.walk_keys
                 if k not in n.properties)
    if bad:
        raise MappingLeakError(
            f"graph_mapping.yaml's walk_keys {', '.join(bad)} are not properties of their node "
            f"type: a walk key must be listed under that node's properties"
        )


def walk_keys(mapping: Optional[GraphMapping]) -> Optional[Dict[str, Tuple[str, ...]]]:
    """{node type: its declared walk keys} for every node type, or None without a mapping."""
    if mapping is None:
        return None
    return {t: tuple(n.walk_keys) for t, n in mapping.nodes.items()}


def _join_columns(node_type: str, n: Any) -> Iterator[Tuple[str, str, str]]:
    """Each join's key, `columns` entries and aggregated columns, so the
    checks above refuse one that is unclassified, dropped or folded."""
    for j in n.joins:
        yield j.table, j.key, f"node {node_type} join key"
        for col in j.columns:
            yield j.table, col, f"node {node_type} join column"
        for agg in j.aggregates:
            yield j.table, agg.column, f"node {node_type} join aggregate"


# ------------------------------------------------------------------- folds
def fold_subjects(mapping: GraphMapping) -> Iterator[Tuple[str, str, str]]:
    """(source table, column, where) for every edge `fold_subject` and node `joins[].subject`
    Counted, never carried.

    Not yielded by `mapping_columns`, so the ban on carrying folded columns still holds for
    every src/dst/label/property."""
    for edge_type, e in mapping.edges.items():
        if e.fold_subject is not None:
            yield e.source_table, e.fold_subject, f"edge {edge_type} fold_subject"
    for node_type, n in mapping.nodes.items():
        for i, j in enumerate(n.joins):
            if j.subject is not None:
                yield j.table, j.subject, f"node {node_type} joins[{i}] subject"


def fold_edge_types(mapping: Optional[GraphMapping]) -> Tuple[str, ...]:
    """The mapping's folded (weighted) edge types, sorted; () without a mapping."""
    if mapping is None:
        return ()
    return tuple(sorted(t for t, e in mapping.edges.items() if e.fold_subject is not None))


def folds_record(mapping: Optional[GraphMapping]) -> List[Dict[str, str]]:
    """flags.json's `folds`: one entry per fold edge type, derived from the mapping."""
    if mapping is None:
        return []
    return [{"edge_type": t, "table": e.source_table, "subject": e.fold_subject, "src": e.src,
             "dst": e.dst, "form": e.fold_form}
            for t, e in sorted(mapping.edges.items()) if e.fold_subject is not None]


def fold_policy(data: Dict[str, Any]) -> Tuple[int, Tuple[str, ...]]:
    """(floor, fold edge types) from a CLEARED flags.json (the dict require_clear returns).

    The fold types are flags.json's `folds`, which the gate has just checked equals the
    mapping's fold edges; build, provision, the sync task, reapply and enrich check these."""
    types = tuple(sorted({f["edge_type"] for f in data.get("folds", [])}))
    return floor.checked_floor(data.get("floor")), types


def _check_fold_subjects_folded(data: Dict[str, Any], mapping: GraphMapping) -> None:
    """Rule 2: every fold subject has a flag that chose `fold`."""
    folded = {(f["table"], f["column"]) for f in data.get("flags", []) if f.get("choice") == "fold"}
    bad = sorted({f"{t}.{c} ({w})" for t, c, w in fold_subjects(mapping) if (t, c) not in folded})
    if bad:
        raise MappingLeakError(
            f"graph_mapping.yaml's fold_subject {', '.join(bad)} is not a folded care-subject "
            f"column: its flag must choose fold (resolve_flag ... 'fold'), then run privacy again."
        )


def _check_fold_flags_used(data: Dict[str, Any], mapping: GraphMapping) -> None:
    """Rule 3: every flag that chose `fold` is folded by some edge (else every walk is empty)."""
    used = {(t, c) for t, c, _w in fold_subjects(mapping)}
    unused = sorted(f"{f['table']}.{f['column']}" for f in data.get("flags", [])
                    if f.get("choice") == "fold" and (f["table"], f["column"]) not in used)
    if unused:
        raise MappingLeakError(
            f"flag {', '.join(unused)} chose fold but no edge or join folds it: add an edge with "
            f"fold_subject naming it, or resolve the flag as drop."
        )


def _check_folds_record(data: Dict[str, Any], mapping: GraphMapping) -> None:
    """flags.json's `folds` is what the mapping gives now (no hand edit, no stale record)."""
    if data.get("folds", []) != folds_record(mapping):
        raise PrivacyGateError(
            "flags.json's folds differ from the fold edges of graph_mapping.yaml now; run the "
            "privacy step again."
        )


def check_fold_subjects(data: Dict[str, Any], mapping: GraphMapping) -> None:
    """Enforce fold subject rules (care-subject src/dst are checked elsewhere)."""
    _check_fold_subjects_folded(data, mapping)
    _check_fold_flags_used(data, mapping)


# -------------------------------------------------------------- node joins
def _effective_classes(data: Dict[str, Any]) -> Dict[Tuple[str, str], Optional[str]]:
    """{(table, column): class after reclassification} from flags.json's columns and flags."""
    classes = {(c["table"], c["column"]): c.get("class") for c in data.get("columns", [])}
    for f in data.get("flags", []):
        if f.get("choice") == "reclassify":
            classes[(f["table"], f["column"])] = f.get("new_class")
    return classes


def check_join_subjects(data: Dict[str, Any], mapping: GraphMapping) -> None:
    """An `aggregates` join over a table holding any care-subject column
    (after reclassification) must name a folded `subject`. A `columns` join is like a native
    property: only its key and selected columns are checked (rule 1)."""
    classes = _effective_classes(data)
    for node_type, n in mapping.nodes.items():
        for i, j in enumerate(n.joins):
            if not j.aggregates or j.subject is not None:
                continue
            care = sorted(c for (t, c), cls in classes.items()
                          if t == j.table and cls == classify.CARE_SUBJECT)
            if care:
                raise MappingLeakError(
                    f"nodes.{node_type}.joins[{i}] aggregates {j.table}, which holds care-subject "
                    f"data ({', '.join(care)}), without a folded subject; name the care-subject "
                    f"column as 'subject' (its flag must choose fold), so every output is blank "
                    f"below the privacy floor.")


def check_join_outputs_not_public(mapping: GraphMapping) -> None:
    """A join output never takes a public directory property name of the
    directory node type: public functions return those keys, so a join output there would be
    served to the public role. Public directory fields stay native properties."""
    spec = mapping.nodes.get(DIRECTORY_NODE_TYPE)
    if spec is None:
        return
    public = sorted({o for j in spec.joins for o in j.outputs} & set(DIRECTORY_PROPERTIES))
    if public:
        raise MappingLeakError(
            f"nodes.{DIRECTORY_NODE_TYPE}.joins output(s) {', '.join(public)}, which are public "
            f"directory properties: a join output is never public. Rename the output, or map "
            f"the public field as a native property of {DIRECTORY_NODE_TYPE}.")


def derived_record(data: Dict[str, Any], mapping: Optional[GraphMapping]) -> List[Dict[str, Any]]:
    """flags.json's `derived`: one entry per join output, derived from the mapping.

    `class` is the input column's class after reclassification; `func` is None for a
    one-to-one `columns` output."""
    if mapping is None:
        return []
    classes = _effective_classes(data)
    out = []
    for node_type, n in sorted(mapping.nodes.items()):
        for j in n.joins:
            items = [(c, c, None) for c in j.columns] + [(a.name, a.column, a.func) for a in j.aggregates]
            out += [{"node_type": node_type, "name": name, "table": j.table, "column": col,
                     "func": func, "class": classes.get((j.table, col)),
                     "floored": j.subject is not None}
                    for name, col, func in items]
    return out


def _check_derived_record(data: Dict[str, Any], mapping: GraphMapping) -> None:
    """flags.json's `derived` is what the mapping and flags give now (no hand edit, no stale record)."""
    if data.get("derived", []) != derived_record(data, mapping):
        raise PrivacyGateError(
            "flags.json's derived join outputs differ from the node joins of graph_mapping.yaml "
            "now; run the privacy step again."
        )


def _check_source(data: Dict[str, Any], mapping: GraphMapping) -> None:
    """The mapping reads the source that was profiled; an unrecorded source is refused."""
    profiled = data.get("source") or {}
    if not profiled.get("source_catalog") or not profiled.get("source_schema"):
        raise MappingLeakError(
            "flags.json records no profiled source (schema.json has no source_catalog / "
            "source_schema): the mapping's source cannot be pinned to what privacy checked. "
            "Profile the source, then run the privacy step again."
        )
    used = {"source_catalog": mapping.source_catalog, "source_schema": mapping.source_schema}
    wrong = [f"{k}={used[k]!r} (profiled {profiled[k]!r})" for k in used if used[k] != profiled[k]]
    if wrong:
        raise MappingLeakError(
            f"graph_mapping.yaml reads a different source than privacy profiled: "
            f"{', '.join(wrong)}. Profile that source and run the privacy step again."
        )


Column = Tuple[str, str]


def foreign_keys_record(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """flags.json's `foreign_keys`: schema.json's declared FKs, one entry per FK, in order.

    {"table", "columns", "references": [ref_table, ref_col, ...]} (profile's shape); the gate
    pins it to schema.json now (`_check_against_schema`)."""
    return [{"table": t["name"], "columns": list(fk.get("columns") or []),
             "references": list(fk.get("references") or [])}
            for t in tables for fk in t.get("foreign_keys") or []]


def key_columns_record(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """flags.json's `key_columns`: schema.json's key (PK or composite key) of every table that
    declares one, {"table", "columns"}; pinned to schema.json like `foreign_keys`."""
    return [{"table": t["name"], "columns": list(t.get("key_columns") or [])}
            for t in tables if t.get("key_columns")]


def _norm(table: Any, column: Any) -> Column:
    """FK names are matched case-insensitively (Patients.PatientID == patients.patientid)."""
    return (str(table or "").casefold(), str(column or "").casefold())


def _fk_targets(data: Dict[str, Any]) -> Dict[Column, List[Column]]:
    """{(table, column): [(ref table, ref column), ...]} from flags.json's `foreign_keys`, all
    names case-folded; a composite FK pairs its i-th column with its i-th referenced column.
    Malformed entries (null, no table) are tolerated here; the pin to schema.json refuses them."""
    out: Dict[Column, List[Column]] = {}
    for fk in data.get("foreign_keys") or []:
        refs = fk.get("references") or []
        for col, ref_col in zip(fk.get("columns") or [], refs[1:]):
            out.setdefault(_norm(fk.get("table"), col), []).append(_norm(refs[0], ref_col))
    return out


def _care_sets(data: Dict[str, Any]) -> Tuple[set, Dict[str, Column]]:
    """(care-subject columns, {care-keyed table: one care column of its key}), case-folded.

    The classifier's class, except a typed reclassify to public_professional (owner decision A,
    2026-10-05: a doctor's name is not a care subject; the reason and chooser are in flags.json
    and the stamp). A drop or a reclassify to none does not clear a column (2a still holds for
    those). A table is care-keyed when its key holds a care-subject column; any column of it (a
    surrogate row_id too) names that person."""
    professional = {_norm(f["table"], f["column"]) for f in data.get("flags", [])
                    if f.get("choice") == "reclassify"
                    and f.get("new_class") == classify.PUBLIC_PROFESSIONAL}
    care = {_norm(c["table"], c["column"]) for c in data.get("columns", [])
            if c.get("class") == classify.CARE_SUBJECT} - professional
    keyed: Dict[str, Column] = {}
    for k in data.get("key_columns") or []:
        hits = sorted(_norm(k.get("table"), c) for c in k.get("columns") or []
                      if _norm(k.get("table"), c) in care)
        if hits:
            keyed[hits[0][0]] = hits[0]
    return care, keyed


def _care_reached(start: Column, care: set, keyed: Dict[str, Column],
                  targets: Dict[Column, List[Column]]) -> Optional[Tuple[Column, Column]]:
    """(column reached, care column behind it) when `start` is a care-subject column, or reaches
    through a chain of declared FKs a care-subject column or ANY column of a care-keyed table;
    else None. Depth-first, each column visited once (FK cycles terminate)."""
    start = _norm(*start)
    seen, stack = set(), [(start, 0)]
    while stack:
        col, depth = stack.pop()
        if col in care:
            return col, col
        if depth and col[0] in keyed:
            return col, keyed[col[0]]
        if col in seen:
            continue
        seen.add(col)
        stack.extend((t, depth + 1) for t in targets.get(col, []))
    return None


def _care_key_text(table: str, key: str, hit: Tuple[Column, Column]) -> str:
    reached, care_col = hit
    where = f"{table}.{key}"
    if reached == _norm(table, key):
        return where
    if reached == care_col:
        return f"{where} (foreign key to {reached[0]}.{reached[1]})"
    return (f"{where} (foreign key to {reached[0]}.{reached[1]}, a column of {reached[0]}, keyed "
            f"on care-subject {care_col[0]}.{care_col[1]})")


def _care_tables(care: set) -> Dict[str, List[str]]:
    """{source table: its care-subject columns}, case-folded, from the classifier's care set."""
    out: Dict[str, List[str]] = {}
    for table, column in sorted(care):
        out.setdefault(table, []).append(column)
    return out


def _care_source_text(node_type: str, table: str, care_tables: Dict[str, List[str]]) -> Optional[str]:
    """Why `node_type` is refused for its SOURCE table, or None: a table (or view) holding any
    care-subject column is person-level data, one person per node, whatever the key (A5)."""
    cols = care_tables.get(_norm(table, "")[0])
    if not cols:
        return None
    return (f"node type {node_type} is sourced from {table}, which holds care-subject "
            f"column(s) {', '.join(f'{table}.{c}' for c in cols)}")


def _care_key_reason(node_type: str, n: Any, care: set, keyed: Dict[str, Column],
                     targets: Dict[Column, List[Column]]) -> Optional[str]:
    """Why `node_type` is refused for its natural_key (a care column or an FK chain to one, or
    to a care-keyed table), or None."""
    keys = n.natural_key if isinstance(n.natural_key, list) else [n.natural_key]
    hits = [(k, _care_reached((n.source_table, k), care, keyed, targets)) for k in keys]
    cols = [_care_key_text(n.source_table, k, h) for k, h in hits if h is not None]
    if not cols:
        return None
    return f"node type {node_type} is keyed on care-subject column(s) {', '.join(cols)}"


def check_care_key_nodes(data: Dict[str, Any], mapping: GraphMapping) -> None:
    """No node type's natural_key holds a column the classifier calls care subject (A5b), or a
    declared foreign key that reaches, directly or through a chain of FKs, a care-subject column
    or any column of a care-keyed table. FK names match case-insensitively.

    Also (owner decision, A5): no node type is SOURCED from a table or view holding any
    care-subject column, whatever its key (patients keyed on row_id, a copy without the patient
    id but with patient_city). Fold is the only way patient data enters the graph. Known limit:
    a one-row-per-person table with no care-subject column at all is not detected.

    Reads flags.json `columns` (the classifier's class, pinned to schema.json by
    `_check_against_schema`). Only a typed reclassify to public_professional clears a column
    (owner decision A); a drop or a reclassify to none does not, so a hand-written node keyed on
    patient_id is refused when its flag chose either (decision 2a). FKs and keys are flags.json's
    `foreign_keys` and `key_columns`, pinned to schema.json the same way."""
    care, keyed = _care_sets(data)
    targets = _fk_targets(data)
    care_tables = _care_tables(care)
    bad = []
    for node_type, n in sorted(mapping.nodes.items()):
        reasons = (_care_key_reason(node_type, n, care, keyed, targets),
                   _care_source_text(node_type, n.source_table, care_tables))
        bad += [r for r in reasons if r is not None]
    if bad:
        raise CareKeyNodeError(
            f"graph_mapping.yaml: {'; '.join(bad)}. A node keyed on a care subject would publish "
            f"that person's ids in Lakebase. Remove the node type; use the table only as an edge "
            f"source (src/dst on its other keys), folding the care column with fold_subject.")


def check_mapping(data: Dict[str, Any], mapping: Optional[GraphMapping]) -> None:
    """The mapping reads the profiled source; every column is classified; none is dropped/folded."""
    if mapping is None:
        return
    _check_source(data, mapping)
    check_care_key_nodes(data, mapping)
    classified = {(c["table"], c["column"]) for c in data.get("columns", [])}
    banned = {(f["table"], f["column"]) for f in data.get("flags", [])
              if f.get("choice") in ("drop", "fold")}
    uses = list(mapping_columns(mapping))
    leaks = sorted({f"{t}.{c} ({w})" for t, c, w in uses if (t, c) in banned})
    if leaks:
        raise MappingLeakError(
            f"graph_mapping.yaml still carries dropped or folded care-subject columns: "
            f"{', '.join(leaks)}. A folded key reaches Lakebase only as folded, weighted edges."
        )
    unknown = sorted({f"{t}.{c} ({w})" for t, c, w in uses if (t, c) not in classified})
    if unknown:
        raise MappingLeakError(
            f"graph_mapping.yaml carries column(s) with no privacy classification: "
            f"{', '.join(unknown)}. Profile the table so schema.json lists them, then run the "
            f"privacy step again; an unclassified column is never served."
        )
    check_fold_subjects(data, mapping)
    _check_folds_record(data, mapping)
    check_walk_keys(mapping)
    check_join_subjects(data, mapping)
    check_join_outputs_not_public(mapping)
    _check_derived_record(data, mapping)


# ------------------------------------------------------- what schema.json says right now
def read_schema(run_folder: str) -> Dict[str, Any]:
    path = Path(run_folder) / "schema.json"
    if not path.exists():
        raise FileNotFoundError(f"schema.json not found at {path}; run profile first")
    return json.loads(path.read_text())


def profiled_source(schema: Dict[str, Any]) -> Dict[str, Any]:
    """The source schema.json profiled: {source_catalog, source_schema} (None when absent)."""
    return {k: schema.get(k) for k in ("source_catalog", "source_schema")}


def classified_columns(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The class of every column in schema.json's tables (name + pattern rates)."""
    out = []
    for table in tables:
        for col in table.get("columns", []):
            cls, detected_by = classify.classify_column(col["name"], col.get("patterns") or {})
            out.append({"table": table["name"], "column": col["name"], "class": cls,
                        "detected_by": detected_by})
    return out


def _folded(value: Any) -> Any:
    """`value` with every string case-folded (the pins compare names case-insensitively)."""
    if isinstance(value, str):
        return value.casefold()
    if isinstance(value, list):
        return [_folded(v) for v in value]
    if isinstance(value, dict):
        return {k: _folded(v) for k, v in value.items()}
    return value


def _check_structure_pin(data: Dict[str, Any], tables: List[Dict[str, Any]]) -> None:
    """flags.json's `foreign_keys` and `key_columns` are schema.json's now (case aside)."""
    for field, record, what in (("foreign_keys", foreign_keys_record, "foreign keys"),
                                ("key_columns", key_columns_record, "key columns")):
        if _folded(data.get(field) or []) != _folded(record(tables)):
            raise PrivacyGateError(
                f"flags.json's recorded {what} differ from schema.json's now; run the privacy "
                f"step again."
            )


def _check_against_schema(run_folder: str, data: Dict[str, Any]) -> None:
    """flags.json's source and classes are what schema.json gives now (no hand edits)."""
    schema = read_schema(run_folder)
    if data.get("source") != profiled_source(schema):
        raise PrivacyGateError(
            "flags.json's recorded source differs from the source schema.json profiled; "
            "run the privacy step again."
        )
    _check_structure_pin(data, schema.get("tables", []))
    now = {(c["table"], c["column"]): c["class"] for c in classified_columns(schema.get("tables", []))}
    recorded = {(c["table"], c["column"]): c.get("class") for c in data.get("columns", [])}
    wrong = sorted(f"{t}.{c}" for t, c in set(now) | set(recorded)
                   if now.get((t, c)) != recorded.get((t, c)))
    if wrong:
        raise PrivacyGateError(
            f"flags.json's column classes differ from classifying schema.json now: "
            f"{', '.join(wrong)}. Classes are never edited by hand; reclassify a flag with "
            f"resolve_flag, then run the privacy step again."
        )


# ------------------------------------------------------------------------- entry points
def required_run_folder(config: Dict[str, Any]) -> str:
    """The run folder holding flags.json; never a silent '.' fallback (build, provision, switch)."""
    run_folder = config.get("run_folder")
    if not run_folder:
        raise ValueError("config['run_folder'] is required: the privacy gate reads flags.json there")
    return str(run_folder)


def read_run_mapping(run_folder: str) -> Optional[GraphMapping]:
    """graph_mapping.yaml of the run folder through the contract, or None when absent."""
    path = Path(run_folder) / "graph_mapping.yaml"
    return load_mapping(path) if path.exists() else None


def require_clear(run_folder: str, mapping: Optional[GraphMapping] = None) -> Dict[str, Any]:
    """Raise unless the privacy gate is clear for `run_folder`; returns the flags.json data.

    `mapping` is the mapping the caller is about to use; when omitted it is read from the run
    folder. Called by build and provision before any SQL.
    """
    run_folder = str(run_folder)
    if not flags_path(run_folder).exists():
        raise PrivacyGateError(
            f"{flags_path(run_folder)} not found: the privacy step has not run. Nothing is "
            f"built or served before the privacy gate passes."
        )
    data = read_flags_file(run_folder)
    _check_stamp(run_folder, data)
    check_newest_decision(run_folder, data)
    check_flags(data)
    check_mapping(data, mapping if mapping is not None else read_run_mapping(run_folder))
    _check_against_schema(run_folder, data)
    return data


def _recorded_floor(data: Dict[str, Any], privacy_floor: Optional[int]) -> int:
    """The recorded floor, or a caller's higher one; never lower."""
    recorded = floor.checked_floor(data.get("floor"))
    chosen = recorded if privacy_floor is None else floor.checked_floor(privacy_floor)
    if chosen < recorded:
        raise ValueError(f"privacy floor {chosen} is below the recorded floor {recorded}")
    return chosen


def _derive_selection(mapping: GraphMapping, edge_types: Optional[Tuple[str, ...]]
                      ) -> Tuple[bool, Optional[Tuple[str, ...]]]:
    """(folded, selection) for a walk on `mapping`: `folded` from the selected types, never graph-wide.

    Omitted: no fold type -> unfolded, no IN clause (today's SQL); every type folded -> all of
    them; a mixed graph -> ValueError (a caller never gets one kind silently)."""
    folds = set(fold_edge_types(mapping))
    unfolded = sorted(set(mapping.edges) - folds)
    if edge_types is None:
        if not folds:
            return False, None
        if not unfolded:
            return True, tuple(sorted(folds))
        raise ValueError(f"edge_types is required on a graph with folded and unfolded edge types: "
                         f"folded {sorted(folds)}, unfolded {unfolded}")
    unknown = [t for t in edge_types if t not in mapping.edges]
    if unknown:
        raise ValueError(f"edge_types {unknown} are not edge types of the mapping; allowed: "
                         f"{sorted(mapping.edges)}")
    picked = [t for t in edge_types if t in folds]
    if picked and len(picked) != len(edge_types):
        raise MixedFoldSelectionError(
            f"edge_types mixes folded {picked} and unfolded "
            f"{[t for t in edge_types if t not in folds]} edge types: one walk serves one kind of "
            f"count (people for folded, edges for unfolded); walk them separately")
    return bool(picked), edge_types


def served_walk_selection(run_folder: Optional[str], privacy_floor: Optional[int],
                          folded: Optional[bool], edge_types: Optional[Sequence[str]] = None
                          ) -> Tuple[int, bool, Optional[Tuple[str, ...]]]:
    """(floor, folded, edge types) for one served walk, taken from the gate.

    `edge_types` is validated (identifier pattern, then the mapping's closed list), sorted and
    de-duplicated; `folded` is derived from that selection. Without a run folder `folded` must
    be stated, and naming `edge_types` is refused (final review P1): with no mapping the gate
    cannot tell a folded type from an unfolded one, so it could not refuse a mixed selection.
    An explicit `folded` must equal the derived value.
    """
    selection = checked_edge_types(edge_types)
    if run_folder is None:
        if selection is not None:
            raise PrivacyGateError(
                f"edge_types {list(selection)} needs run_folder: without graph_mapping.yaml the gate "
                f"cannot check which selected types are folded (a mixed selection is refused)"
            )
        if not isinstance(folded, bool):
            raise PrivacyGateError(
                "walk query needs run_folder (to read flags.json) or an explicit folded=True/False"
            )
        return floor.checked_floor(privacy_floor), folded, selection
    data = require_clear(run_folder)
    chosen = _recorded_floor(data, privacy_floor)
    mapping = read_run_mapping(run_folder)
    if mapping is None:
        raise PrivacyGateError(f"{run_folder} has no graph_mapping.yaml: a walk's edge types and "
                               f"folded come from it")
    is_folded, selection = _derive_selection(mapping, selection)
    if folded is not None and folded != is_folded:
        raise PrivacyGateError(f"folded={folded} contradicts the selected edge types "
                               f"(folded={is_folded})")
    return chosen, is_folded, selection


def served_walk_policy(run_folder: Optional[str], privacy_floor: Optional[int],
                       folded: Optional[bool]) -> Tuple[int, bool]:
    """(floor, folded) for a served walk that selects no edge types (every type of the graph).

    On a graph with folded and unfolded edge types this raises: such a walk must name its
    edge types (`served_walk_selection`)."""
    chosen, is_folded, _selection = served_walk_selection(run_folder, privacy_floor, folded, None)
    return chosen, is_folded


def served_count_terms(run_folder: Optional[str], privacy_floor: Optional[int],
                       folded: Optional[bool]) -> Tuple[str, str]:
    """(count expression, HAVING) for a walk, with floor and folded from `served_walk_policy`."""
    chosen, is_folded = served_walk_policy(run_folder, privacy_floor, folded)
    return floor.walk_count_terms(chosen, is_folded)
