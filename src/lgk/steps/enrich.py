"""Enrich step: compute the enrichments the user ticked on the fixed menu (Spark batch).

Reads `enrich.yaml` (`methods: [...]`) from the run folder. If it is missing, writes the
menu for this graph to `enrich_menu.md` and stops, so the user can tick from it.
Writes the fixed enrichment tables into the kit schema in UC, and `enrich_report.json`.

Hygiene and binding: the step runs only when the privacy gate is clear (the same
gate build uses); every enrichment table this run does not write is dropped (an earlier run's
table, computed under an older decision or on an older graph, must never be served); each
output carries the run's `lgk.privacy_stamp`, and `privacy_bindings.json` records "enrich".
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from lgk import manifest as mf
from lgk.enrich.menu import MenuRow, build_menu, select_methods
from lgk.enrich.methods import RUNNERS, Sql, drop_work_tables, prepare_work_tables
from lgk.enrich.ranking import top_nodes
from lgk.enrich import shape as _shape
from lgk.enrich.shape import GraphShape, measure_shape
from lgk.enrich.tables import TABLE_SHAPES, create_table_sql
from lgk.manifest import ObjectType
from lgk.privacy import binding, floor
from lgk.privacy.gate import fold_policy, require_clear, required_run_folder
from lgk.uc_sql import execute_sql, warehouse_id

logger = logging.getLogger(__name__)

TOP_N_IN_REPORT = 5
REPORT_FILE = "enrich_report.json"
# The output table each method writes (provision serves a table only if a reported method wrote it).
TABLE_OF = {"pagerank": "node_score", "degree": "node_score", "components": "node_group",
            "similarity": "similar_to", "closure": "closure"}


def _measured_shape_line(shape: GraphShape) -> str:
    """The shape the menu used, as measured on the built graph (every edge type, folded ones too)."""
    if shape.too_deep:
        kind = f"no cycle found, but deeper than {_shape.MAX_DEPTH} levels"
    elif shape.is_dag:
        kind = "no cycle (a tree or DAG)"
    else:
        kind = "has a cycle"
    return (f"Measured on the built nodes and edges now: {kind}; {shape.n_nodes} nodes, "
            f"{shape.n_edges} directed edges (every edge type, folded ones included).")


def recorded_shape_guess(run_folder: Path) -> Optional[str]:
    """The `shape` propose wrote in flags.json, for LABELLING only: it is a guess from the
    table-level draft (before build and before any fold edge) and never decides eligibility."""
    flags = run_folder / "flags.json"
    if not flags.exists():
        return None
    value = json.loads(flags.read_text()).get("shape")
    return value if isinstance(value, str) else None


def render_menu(rows: List[MenuRow], shape: GraphShape, recorded: Optional[str] = None) -> str:
    """The menu as a Markdown table; greyed rows say why. Above it: the measured shape the rows
    were judged on, and (if flags.json has one) propose's recorded shape labelled as a guess."""
    lines = [_measured_shape_line(shape)]
    if recorded is not None:
        lines.append(f"flags.json `shape: {recorded}` is propose's guess from the table-level draft, "
                     "made before build and before any fold edge; this menu does not use it.")
    lines += ["", "| Method | Question | Example on your graph | Adds table | Status |",
              "|---|---|---|---|---|"]
    for r in rows:
        status = "available" if r.available else f"greyed out: {r.reason}"
        lines.append(f"| {r.method} | {r.question} | {r.example} | {r.table} | {status} |")
    return "\n".join(lines) + "\n"


Created = Callable[[str], None]


def _run_methods(sql: Sql, fq: str, methods: List[str], created: Created) -> Dict[str, Any]:
    """Create the needed output tables fresh (reporting each to `created`), then run each method.

    Work tables (`_enrich_*`) are always dropped and never reported: they must not outlive the run.
    """
    for table in sorted({TABLE_OF[m] for m in methods}):
        sql(create_table_sql(fq, table))
        created(f"{fq}.{table}")
    report: Dict[str, Any] = {}
    try:
        prepare_work_tables(sql, fq)
        for method in methods:
            logger.info(f"enrich: running {method} on {fq}")
            report[method] = RUNNERS[method](sql, fq)
            if TABLE_OF[method] == "node_score":
                report[method]["top"] = [r[1] for r in top_nodes(sql, fq, method, TOP_N_IN_REPORT)]
    finally:
        drop_work_tables(sql, fq)
    return report


def run_enrich(sql: Sql, fq: str, run_folder: Path, created: Created = lambda _table: None) -> Dict[str, Any]:
    """Measure the shape, show or apply the menu, and write the report.

    Eligibility (closure's cycle and depth checks) uses ONLY the shape measured here on the built
    nodes and edges; flags.json's `shape` is propose's pre-build guess and is only labelled as such.

    `created` is called with the full name of every output table made (the manifest record).
    """
    shape = measure_shape(sql, fq)
    menu = build_menu(shape)
    choice = run_folder / "enrich.yaml"
    if not choice.exists():
        (run_folder / "enrich_menu.md").write_text(
            render_menu(menu, shape, recorded_shape_guess(run_folder)))
        raise FileNotFoundError(
            f"{choice} not found. Tick methods from {run_folder / 'enrich_menu.md'} and save "
            "them as `methods: [...]` in enrich.yaml."
        )
    methods = select_methods(menu, (yaml.safe_load(choice.read_text()) or {}).get("methods", []))
    _drop_unwritten(sql, fq, methods)
    report = {"shape": asdict(shape), "menu": [asdict(r) for r in menu],
              **_run_methods(sql, fq, methods, created)}
    (run_folder / REPORT_FILE).write_text(json.dumps(report, indent=2))
    return report


def _drop_unwritten(sql: Sql, fq: str, methods: List[str]) -> None:
    """DROP every enrichment table (seed_score included) that this run's methods do not write.

    Before the methods run, so a run that fails part way leaves no earlier run's table behind
    either (closes P0.3: a stale table from an earlier run or decision is never synced).
    """
    written = {TABLE_OF[m] for m in methods}
    unwritten = sorted(set(TABLE_SHAPES) - written)
    _refuse_foreign_objects(sql, fq, unwritten)
    for table in unwritten:
        sql(f"DROP TABLE IF EXISTS {fq}.{table}")


# The only kinds of table the kit's `CREATE OR REPLACE TABLE ... USING DELTA` can have made.
DROPPABLE_TYPES = {"MANAGED", "EXTERNAL"}


def _refuse_foreign_objects(sql: Sql, fq: str, tables: List[str]) -> None:
    """Raise, before anything is dropped, if any of `tables` exists in `fq` as an object the kit
    could not have written: a STREAMING_TABLE or MATERIALIZED_VIEW (pipeline-owned), a view, a foreign table, or a non-Delta table.

    A table that does not exist is not listed by information_schema, so it passes (DROP IF EXISTS
    is then a no-op)."""
    if not tables:  # `IN ()` is a syntax error
        return
    catalog, schema = fq.split(".")
    names = ", ".join(f"'{t}'" for t in tables)
    # UC stores names in lower case but resolves `DROP TABLE c.Gold.t` case-insensitively: fold here too.
    rows = sql(f"SELECT table_name, table_type, data_source_format FROM {catalog}.information_schema.tables "
               f"WHERE table_schema = '{schema.lower()}' AND table_name IN ({names})") or []  # as execute_sql
    foreign = [(name, str(kind), fmt) for name, kind, fmt in rows
               if str(kind).upper() not in DROPPABLE_TYPES or str(fmt).upper() != "DELTA"]
    if foreign:
        found = "; ".join(f"{fq}.{name} is {kind} (format {fmt})" for name, kind, fmt in foreign)
        raise RuntimeError(
            f"enrich will not drop a pipeline-owned or non-kit object: {found}. Enrich drops only "
            "MANAGED or EXTERNAL Delta tables it could have written. Rename or move that object out "
            f"of the kit schema {fq}, then re-run."
        )


def _fold_floor_hook(sql: Sql, fq: str, cleared: Dict[str, Any]) -> None:
    """Plan X4: refuse to enrich a Delta `edges` holding a folded edge below the floor (or NULL).

    `lgk.privacy.floor.assert_fold_floor` filters to the graph's fold edge types (from the
    cleared gate data), so the NULL weight of an unfolded type on a mixed graph passes."""
    fold_floor, fold_types = fold_policy(cleared)
    floor.assert_fold_floor(sql, f"{fq}.edges", fold_types, fold_floor)


def _stamp_outputs(sql: Sql, run_folder: str, stamp: str, tables: List[str], methods: List[str]) -> None:
    """Stamp each output table with the run's privacy stamp, then record "enrich"."""
    for table in tables:
        sql(binding.set_stamp_sql(table, stamp))
    binding.record_enrich(run_folder, stamp, tables, methods)


def enrich(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the enrich step for `config` (lakebase_catalog, kit_schema, run_folder, warehouse_id)."""
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    wh = warehouse_id(config)
    run_folder = required_run_folder(config)
    cleared = require_clear(run_folder)  # the gate build uses, before any SQL
    stamp = binding.stamp_id(cleared)

    def sql(statement: str) -> List[List[Any]]:
        return execute_sql(w, wh, statement)

    fq = f"{config['lakebase_catalog']}.{config['kit_schema']}"
    _fold_floor_hook(sql, fq, cleared)
    made: List[str] = []

    def created(table: str) -> None:
        made.append(table)
        mf.record(config, ObjectType.UC_TABLE, table, "uc")

    report = run_enrich(sql, fq, Path(run_folder), created)
    _stamp_outputs(sql, run_folder, stamp, made, [m for m in report if m in TABLE_OF])
    return report
