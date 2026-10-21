"""The data-engineer walks notebook: every walk's SQL, a live answer, its time and plan.

`build_notebook(config)` returns a Databricks notebook in source format (`# COMMAND ----------`
cells); `write_notebook(config)` writes it to `<run_folder>/walks_notebook.py`.

- The SQL is the kit's: `per_depth_sql` with the served count and floor from the privacy gate
  (`served_walk_policy`), the same SQL the app and the advisor use. It reads the served views
  only; `assert_served_only` refuses any SQL that names another relation (a raw version table).
- The notebook holds NO answers. Each walk cell runs its SQL live against the branch, for the
  seed in the `seed_node_id` widget (default: the advisor's seed, else the live worst case),
  so re-running a cell re-queries the branch.
- Times: the wall clock of this run, plus the advisor's p50/p95 when advisor_report.json is in
  the run folder (recorded measurements, labelled as such).
- The connection is the operator's (endpoint host, Databricks identity, fresh credential), via
  the runtime cell copied verbatim from `lgk.walks.nb_runtime`. No credential is printed.

Config keys (optional): top_n (25), notebook_max_depth (4), notebook_seed_node_id.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from lgk import versions
from lgk.privacy.floor import walk_count_terms
from lgk.privacy.gate import served_walk_policy
from lgk.walks import nb_runtime
from lgk.walks.sql import per_depth_sql

NOTEBOOK_NAME = "walks_notebook.py"
DEFAULT_TOP_N = 25
DEFAULT_MAX_DEPTH = 4
CELL = "\n\n# COMMAND ----------\n\n"
SERVED_RELATIONS = tuple(view for view, _ in versions.SERVED_VIEWS)
_RELATION = re.compile(r"\b(?:FROM|JOIN)\s+([a-z_][a-z0-9_]*)\s*\.\s*([a-z_][a-z0-9_]*)",
                       re.IGNORECASE)


class RawRelationError(ValueError):
    """Notebook SQL names a relation other than the served views (for example a version table)."""


# ---------------------------------------------------------------------------- the SQL
def assert_served_only(sql: str, served: str) -> None:
    """Refuse SQL whose FROM/JOIN names anything but `<served>.<served view>`."""
    refs = _RELATION.findall(sql)
    bad = sorted({f"{s}.{t}" for s, t in refs if s != served or t not in SERVED_RELATIONS})
    if bad:
        raise RawRelationError(f"notebook SQL must read only {served}.{SERVED_RELATIONS}; "
                               f"it names {', '.join(bad)}")
    if not refs:
        raise RawRelationError(f"notebook SQL reads no served view of {served}")


def notebook_queries(config: Dict[str, Any]) -> Dict[str, Dict[int, str]]:
    """{shape: {depth: SQL}} over the served views, with the gate's count and floor.

    A web walk always; a tree walk too, unless the edges are folded care-subject data (a tree
    over folded edges is not supported by the walk engine).
    """
    served = versions.served_schema(config)
    privacy_floor, folded = served_walk_policy(config["run_folder"], None, None)  # the gate
    top_n = config.get("top_n", DEFAULT_TOP_N)
    depth = config.get("notebook_max_depth", DEFAULT_MAX_DEPTH)
    queries = {"web": per_depth_sql("web", depth, served_schema=served, top_n=top_n,
                                    count_terms=walk_count_terms(privacy_floor, folded))}
    if not folded:
        queries["tree"] = per_depth_sql("tree", depth, served_schema=served, top_n=top_n)
    for per_depth in queries.values():
        for sql in per_depth.values():
            assert_served_only(sql, served)
    return queries


def _advisor_report(run_folder: str) -> Optional[Dict[str, Any]]:
    path = Path(run_folder) / "advisor_report.json"
    return json.loads(path.read_text()) if path.exists() else None


def _default_seed(config: Dict[str, Any], report: Optional[Dict[str, Any]]) -> str:
    """The configured seed, else the advisor's, else "" (the notebook picks the worst case)."""
    seed = config.get("notebook_seed_node_id")
    if seed is None and report:
        seed = (report.get("params") or {}).get("start_node_id")
    return "" if seed is None else str(seed)


# ---------------------------------------------------------------------------- cells
def _md(lines: List[str]) -> str:
    return "\n".join(["# MAGIC %md", *(f"# MAGIC {line}".rstrip() for line in lines)])


def _intro_cell(config: Dict[str, Any], served: str, shapes: List[str]) -> str:
    return _md([
        f"# Walks notebook: {config['graph_name']}",
        "",
        f"Every walk the website can run ({', '.join(shapes)}), at every depth: the exact SQL, a",
        "live answer from the branch, the time of this run, the index advisor's p50/p95 when",
        "recorded, and a plan summary. Change the `seed_node_id` widget and re-run any cell.",
        "",
        f"All SQL reads the served views in `{served}` only, with the privacy floor applied.",
    ])


def _setup_cell(config: Dict[str, Any], served: str, report: Optional[Dict[str, Any]]) -> str:
    from lgk.branch import BranchRef

    branch = BranchRef.from_config(config).branch
    return "\n".join([
        f"PROJECT = {config['lakebase_project']!r}",
        f"BRANCH = {branch!r}",
        f"SERVED_SCHEMA = {served!r}",
        f"ADVISOR_REPORT = json.loads({json.dumps(report, default=str)!r})",
        "conn = connect(PROJECT, BRANCH)  # the operator; the credential is never printed",
        "conn.autocommit = True",
        "conn.execute(f\"SET statement_timeout = '{STATEMENT_TIMEOUT}'\")",
        'SEED = dbutils.widgets.get("seed_node_id") or pick_seed(conn, SERVED_SCHEMA)',
        'print(f"branch {BRANCH}, served schema {SERVED_SCHEMA}, seed {SEED!r}")',
        "ANSWERS = {}",
    ])


def _walk_cells(shape: str, depth: int, sql: str) -> List[str]:
    return [
        _md([f"## {shape} walk, depth {depth}"]),
        "\n".join([
            f"SQL_{shape.upper()}_{depth} = {sql!r}",
            f"ANSWERS[{f'{shape}:{depth}'!r}] = walk_cell(conn, {shape!r}, {depth}, "
            f"SQL_{shape.upper()}_{depth}, SEED, ADVISOR_REPORT)",
        ]),
    ]


def build_notebook(config: Dict[str, Any]) -> str:
    """The notebook source (Databricks source format). Refused unless the privacy gate is clear."""
    queries = notebook_queries(config)
    served = versions.served_schema(config)
    report = _advisor_report(config["run_folder"])
    cells = [
        _intro_cell(config, served, list(queries)),
        '# MAGIC %pip install "psycopg[binary]>=3.1"',
        f"dbutils.widgets.text(\"seed_node_id\", {_default_seed(config, report)!r})",
        Path(nb_runtime.__file__).read_text().rstrip(),
        _setup_cell(config, served, report),
    ]
    for shape, per_depth in queries.items():
        for depth, sql in sorted(per_depth.items()):
            cells += _walk_cells(shape, depth, sql)
    cells.append('conn.close()\ndbutils.notebook.exit(json.dumps({"seed": SEED, "answers": ANSWERS}))')
    return "# Databricks notebook source\n" + CELL.join(cells) + "\n"


def write_notebook(config: Dict[str, Any]) -> Path:
    """Write `<run_folder>/walks_notebook.py`; returns its path."""
    path = Path(config["run_folder"]) / NOTEBOOK_NAME
    path.write_text(build_notebook(config))
    return path
