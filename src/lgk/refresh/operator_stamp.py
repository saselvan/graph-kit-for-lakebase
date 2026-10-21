"""The nightly stamp step for tables an operator's pipeline rebuilt (owner decision, option 1).

Every refresh chain must carry the label on nodes, edges and node_id_map: the sync task refuses
any table whose label does not match the run manifest and the current cleared decision
(lgk.refresh.label.check_labels), whoever wrote it. The kit build labels its own tables. A chain
with NO kit build (an operator SDP pipeline rebuilds the kit tables as plain Delta, `CREATE OR
REPLACE`) gets the refresh task "stamp" between the pipeline and the sync (lgk.refresh.job), which
runs `stamp_operator_tables`:

1. the existing privacy gate (require_clear: flags.json clear and fresh for the run's mapping);
2. each table's UC table_id and Delta version, read BEFORE the checks;
3. the existing floors: assert_fold_floor on edges, assert_join_floor on nodes (no new floor code);
4. the allowlist (lgk.refresh.allowlist): the kit's columns, and only the property keys the
   cleared mapping allows;
5. the label {stamp, table_id, certified_version} on each table and the same in the run manifest
   (lgk.refresh.label.write_labels, the writer the kit build uses too), refused if a table was
   replaced or got new data while it was checked.

One meaning of the label on both chains: the stamp step overwrites whatever label the pipeline
set. The old SDP `table_properties={"lgk.privacy_stamp": <stamp>}` form is retired: a bare stamp
is not the JSON label, so the sync refuses it.

Streaming tables an SDP pipeline owns cannot take ALTER TBLPROPERTIES; their
label goes through the pipeline definition instead.
"""

from typing import Any, Dict, List, Optional, Tuple

from lgk.privacy import binding, floor
from lgk.privacy.gate import fold_policy, read_run_mapping, require_clear, required_run_folder
from lgk.refresh import allowlist, label
from lgk.refresh.freshness import Sql, live_clients


def operator_stamp(config: Dict[str, Any]) -> str:
    """The stamp the run's cleared privacy gate binds tables to; raises PrivacyGateError if not clear."""
    return binding.stamp_id(require_clear(required_run_folder(config)))


def kit_tables(config: Dict[str, Any]) -> List[str]:
    """The fully qualified kit tables every refresh chain must label (binding.BUILT_TABLES)."""
    fq = f"{config['lakebase_catalog']}.{config['kit_schema']}"
    return [f"{fq}.{t}" for t in binding.BUILT_TABLES]


def _assert_floors(sql: Sql, config: Dict[str, Any], cleared: Dict[str, Any]) -> None:
    """The checks the kit build runs before it labels: fold floor on edges, join floor on nodes."""
    nodes, edges, _ = kit_tables(config)
    fold_floor, fold_types = fold_policy(cleared)
    floor.assert_fold_floor(sql, edges, fold_types, fold_floor)
    floor.assert_join_floor(sql, nodes, read_run_mapping(required_run_folder(config)), fold_floor)


def _clients(config: Dict[str, Any], w: Optional[Any], sql: Optional[Sql]) -> Tuple[Any, Sql]:
    """The given clients; the run's warehouse (live_clients) for a missing `sql`, a WorkspaceClient for `w`."""
    if sql is None:
        live_w, sql = live_clients(config)
        w = w or live_w
    if w is None:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
    return w, sql


def stamp_operator_tables(config: Dict[str, Any], sql: Optional[Sql] = None, w: Optional[Any] = None) -> str:
    """The stamp task: gate, identities, floors, allowlist, then label + manifest; returns the stamp.

    `sql` / `w` default to the run's warehouse and workspace client (live_clients). Nothing is
    labelled unless every check passed."""
    cleared = require_clear(required_run_folder(config))
    stamp = binding.stamp_id(cleared)
    w, sql = _clients(config, w, sql)
    before = label.read_identities(w, sql, kit_tables(config))
    _assert_floors(sql, config, cleared)
    fq_schema = f"{config['lakebase_catalog']}.{config['kit_schema']}"
    allowlist.assert_allowlisted(sql, fq_schema, read_run_mapping(required_run_folder(config)))
    label.write_labels(w, sql, config, stamp, before)
    return stamp

