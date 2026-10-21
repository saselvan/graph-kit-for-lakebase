"""The label on the kit tables: one writer for every chain, one check in the sync (owner decision).

The label of nodes, edges and node_id_map is the JSON {stamp, table_id, certified_version}
(binding.label_json). The run manifest records the same three per table.

- `read_identity` reads a table's UC table_id and its Delta history (current version, newest
  data-changing version, by the freshness guard's no-data rule).
- `write_labels` is the one writer: the kit build and the operator's stamp step call it after
  their checks passed on the identities they read BEFORE the checks. It refuses if a table was
  replaced or got new data while it was checked, then writes certified_version = the checked
  version (N-1; the label's own ALTER is commit N, a no-data commit) and records the manifest.
- `check_labels` is what the sync requires of each table: label.stamp == manifest.stamp == the
  current cleared stamp; label.table_id == manifest.table_id == the current UC table_id;
  label.certified_version == manifest.certified_version; and no data-changing commit after it
  (OPTIMIZE, SET TBLPROPERTIES and the like do not count). Any mismatch raises StaleBuildError
  naming the table and the field.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from lgk import manifest as mf
from lgk.privacy import binding
from lgk.privacy.binding import StaleBuildError
from lgk.refresh.freshness import Sql, source_versions


@dataclass(frozen=True)
class TableIdentity:
    table_id: str
    version: int  # the current Delta version
    last_data: Optional[int]  # the newest version whose commit changes data (None: none)


def uc_table_id(w: Any, fq_table: str) -> str:
    """The table's UC table_id (not the Delta `id` of DESCRIBE DETAIL, which differs)."""
    table_id = getattr(w.tables.get(full_name=fq_table), "table_id", None)
    if not table_id:
        raise StaleBuildError(f"{fq_table}: Unity Catalog returned no table_id")
    return str(table_id)


def read_identity(w: Any, sql: Sql, fq_table: str) -> TableIdentity:
    current, last_data = source_versions(sql, fq_table)
    return TableIdentity(uc_table_id(w, fq_table), current, last_data)


def read_identities(w: Any, sql: Sql, fq_tables: Sequence[str]) -> Dict[str, TableIdentity]:
    return {fq: read_identity(w, sql, fq) for fq in fq_tables}


def _changed_while_checked(fq: str, before: TableIdentity, now: TableIdentity) -> Optional[str]:
    if now.table_id != before.table_id:
        return f"{fq}: table_id changed from {before.table_id} to {now.table_id} while it was checked"
    if now.last_data is not None and now.last_data > before.version:
        return (f"{fq}: a data commit at version {now.last_data} came after the checked version "
                f"{before.version}")
    return None


def write_labels(w: Any, sql: Sql, config: Dict[str, Any], stamp: str,
                 before: Dict[str, TableIdentity]) -> Dict[str, Dict[str, Any]]:
    """Label each table with {stamp, table_id, certified_version} and record the same in the manifest.

    `before` is each table's identity read before the caller's checks; certified_version is that
    version. Nothing is written if any table changed since (all tables, or none)."""
    moved = [p for fq, b in before.items() if (p := _changed_while_checked(fq, b, read_identity(w, sql, fq)))]
    if moved:
        raise StaleBuildError("refusing to label tables that changed while they were checked: " + "; ".join(moved))
    labels = {fq: {"stamp": stamp, "table_id": b.table_id, "certified_version": b.version}
              for fq, b in before.items()}
    for fq, label in labels.items():
        sql(binding.set_label_sql(fq, binding.label_json(**label)))
    mf.record_table_labels(config, labels)
    return labels


def _label_problems(fq: str, raw: Any, record: Optional[Dict[str, Any]], stamp: str,
                    now: TableIdentity) -> List[str]:
    """Every field of `fq`'s label (and manifest record) the sync refuses, as '<table>: <field> ...'."""
    if raw is None:
        return [f"{fq} (no {binding.STAMP_PROPERTY})"]
    label = binding.parse_label(raw)
    if label is None:
        return [f"{fq}: label is not the {{stamp, table_id, certified_version}} JSON"]
    if record is None:
        return [f"{fq}: the run manifest records no label for it"]
    out = []
    if label["stamp"] != stamp:
        out.append(f"{fq}: label.stamp is a stale stamp (not the current cleared decision)")
    if record.get("stamp") != stamp:
        out.append(f"{fq}: manifest.stamp is a stale stamp (not the current cleared decision)")
    if label["table_id"] != now.table_id:
        out.append(f"{fq}: label.table_id {label['table_id']} is not the current table_id {now.table_id}")
    if record.get("table_id") != now.table_id:
        out.append(f"{fq}: manifest.table_id {record.get('table_id')} is not the current table_id {now.table_id}")
    certified = label["certified_version"]
    if record.get("certified_version") != certified:
        out.append(f"{fq}: label.certified_version {certified} differs from manifest.certified_version "
                   f"{record.get('certified_version')}")
    if now.last_data is not None and now.last_data > certified:
        out.append(f"{fq}: certified_version {certified}, but a data commit at version {now.last_data} came after it")
    if certified > now.version:
        out.append(f"{fq}: certified_version {certified} is ahead of the current version {now.version}")
    return out


def check_labels(w: Any, sql: Sql, config: Dict[str, Any], fq_tables: Sequence[str], stamp: str) -> None:
    """Raise StaleBuildError naming every table and field whose label the sync must refuse."""
    records = mf.read_table_labels(config)
    bad: List[str] = []
    for fq in fq_tables:
        bad += _label_problems(fq, binding.table_label(sql, fq), records.get(fq), stamp, read_identity(w, sql, fq))
    if bad:
        raise StaleBuildError(
            "Delta table(s) not certified under the current privacy decision: " + "; ".join(bad)
            + ". Run the stamp step (or the kit build) again, then sync."
        )
