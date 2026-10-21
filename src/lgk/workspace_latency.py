"""In-region query latency: the same walk / lookup SQL over the same seeds, timed INSIDE the workspace.

The laptop live tests time their queries from the pytest process (client `laptop`), so the numbers
include the laptop-to-region network. This module times the same queries from a serverless notebook
job in the workspace (client `workspace`), on the commit under test:

1. laptop: `run_in_region(w, config, queries, wheel)` builds a minimal config (project, branch,
   queries) and runs ONE step, `latency_step`, through the shipped notebook runner
   (`lgk.notebook_runner.NotebookRunner` + `DatabricksNotebookBackend`: the wheel of this tree and
   notebooks/lgk_runner.py, serverless, as the invoking user).
2. workspace: `latency_step` connects to the branch with the kit's one connector
   (`lgk.branch.connect_branch`), runs each query once per seed with `start_node_id` bound, times it
   with perf_counter, and writes SAMPLES_FILE into the run folder.
3. laptop: the runner pulls the run folder back (the run's result); `parse_samples` validates it and
   `record_workspace_samples` records p50/p95 per metric, depth and kind with client `workspace`.

Runs on the serverless image's Python 3.10: stdlib only at module level, nothing from lgk.metrics.
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

SAMPLES_FILE = "latency_samples.json"
STEP_NAME = "latency"
QUERY_KEYS = ("metric", "sql", "seeds")
OPTIONAL_INT = ("depth",)
OPTIONAL_NAME = ("kind", "role", "direction")


class LatencySamplesRefused(ValueError):
    """A samples document that is not a list of well-formed, non-empty timed queries."""


class Sample(NamedTuple):
    metric: str
    seconds: List[float]
    rows: List[int]
    context: Dict[str, Any]  # depth / kind / role / direction, as given in the query


# ------------------------------------------------------------------ workspace side
def time_queries(conn: Any, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Each query once per seed (bound as start_node_id): {..query context, seconds: [...], rows: [...]}."""
    out = []
    for q in queries:
        seconds, rows = [], []
        for seed in q["seeds"]:
            t0 = time.perf_counter()
            got = conn.execute(q["sql"], {"start_node_id": seed}).fetchall()
            seconds.append(time.perf_counter() - t0)
            rows.append(len(got))
        out.append({k: v for k, v in q.items() if k not in ("sql", "seeds")} | {"seconds": seconds, "rows": rows})
    return out


def latency_step(config: Dict[str, Any]) -> None:
    """Notebook-runner step: time config["latency_queries"] on the branch; write SAMPLES_FILE."""
    from lgk.branch import connect_branch

    conn = connect_branch(config)
    try:
        samples = time_queries(conn, config["latency_queries"])
    finally:
        conn.rollback()
        conn.close()
    (Path(config["run_folder"]) / SAMPLES_FILE).write_text(json.dumps(samples))


# ------------------------------------------------------------------ laptop side
def _check_sample(n: int, entry: Any) -> Sample:
    if not isinstance(entry, dict) or not isinstance(entry.get("metric"), str) or not entry["metric"]:
        raise LatencySamplesRefused(f"sample {n}: needs a metric name")
    seconds, rows = entry.get("seconds"), entry.get("rows")
    if not isinstance(seconds, list) or not seconds or not all(
            isinstance(s, (int, float)) and not isinstance(s, bool) and math.isfinite(s) and s >= 0
            for s in seconds):
        raise LatencySamplesRefused(f"sample {n} ({entry['metric']}): seconds must be non-empty finite numbers")
    if not isinstance(rows, list) or len(rows) != len(seconds) or not all(
            isinstance(r, int) and not isinstance(r, bool) and r >= 0 for r in rows):
        raise LatencySamplesRefused(f"sample {n} ({entry['metric']}): one row count per timed query")
    unknown = set(entry) - {"metric", "seconds", "rows", *OPTIONAL_INT, *OPTIONAL_NAME}
    if unknown:
        raise LatencySamplesRefused(f"sample {n}: unknown keys {sorted(unknown)}")
    context = {k: entry[k] for k in (*OPTIONAL_INT, *OPTIONAL_NAME) if entry.get(k) is not None}
    return Sample(entry["metric"], [float(s) for s in seconds], list(rows), context)


def parse_samples(text: str) -> List[Sample]:
    """The workspace samples from SAMPLES_FILE text; refuses anything malformed or empty."""
    try:
        doc = json.loads(text)
    except ValueError as e:
        raise LatencySamplesRefused(f"samples are not JSON: {e}") from e
    if not isinstance(doc, list) or not doc:
        raise LatencySamplesRefused("samples must be a non-empty list")
    return [_check_sample(n, entry) for n, entry in enumerate(doc, 1)]


def record_workspace_samples(record: Any, samples: List[Sample], **context: Any) -> None:
    """p50/p95 of every sample with client `workspace` (record = the conftest MetricsRecorder)."""
    for s in samples:
        record.latencies(s.metric, s.seconds, client="workspace", **s.context, **context)


def remote_config(config: Dict[str, Any], queries: List[Dict[str, Any]], run_dir: Path) -> Dict[str, Any]:
    """Only what the step needs, so the config crosses as JSON (a test graph's config may not)."""
    for q in queries:
        missing = [k for k in QUERY_KEYS if not q.get(k)]
        if missing:
            raise ValueError(f"latency query {q.get('metric')!r} lacks {missing}")
    return {"lakebase_project": config["lakebase_project"], "branch_id": config["branch_id"],
            "run_folder": str(run_dir), "latency_queries": queries}


def run_in_region(w: Any, config: Dict[str, Any], queries: List[Dict[str, Any]], wheel: Path,
                  run_dir: Path, workdir: str, backend: Optional[Any] = None) -> List[Sample]:
    """Run `latency_step` as a serverless notebook job; return the parsed samples it sent back.

    `workdir` = /Workspace/Users/<me>/lgk_v1_<6hex> (deleted afterwards); `backend` replaces the
    Databricks backend in offline tests.
    """
    from lgk.notebook_runner import DatabricksNotebookBackend, NotebookRunner
    from lgk.runner import RunStatus

    notebook = Path(__file__).resolve().parents[2] / "notebooks" / "lgk_runner.py"
    own = backend is None
    backend = backend or DatabricksNotebookBackend(w, workdir, wheel, notebook)
    try:
        if own:
            backend.setup()
        [result] = NotebookRunner(remote_config(config, queries, run_dir), backend).run_steps(
            [(STEP_NAME, latency_step)])
    finally:
        if own:
            backend.cleanup()
    if result.status != RunStatus.SUCCESS:
        raise RuntimeError(f"in-region latency step {result.status.value}: {result.error}")
    return parse_samples((Path(run_dir) / SAMPLES_FILE).read_text())
