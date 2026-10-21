"""Opt-in numbers recorder for the live tests: one JSON line per metric, so a deck can cite sourced numbers.

Off by default. Only `LGK_METRICS_FILE=<path>` turns it on; then each call appends one JSON line to that
file with the fixed fields (metric, value, unit, test, suite, commit, branch, project, endpoint_cu, run,
client, rows, edges, ts and the optional context keys). When it is off nothing is written, git is not
called and no endpoint is read.

Every line says what it was measured at: `rows` and `edges` (the size of the graph the test provisioned),
`run` (which kind of run made it: LGK_METRICS_RUN, one of RUNS, default `test`) and `client` (`laptop` =
the pytest process, `workspace` = a serverless notebook job inside the workspace's region). A call
without rows or edges is refused, also when the recorder is off.

The recorder refuses what is not a count or a name: a value must be a finite number, and context is limited to CONTEXT_KEYS,
whose values are non-negative ints (counts) or short identifier-shaped names (never free text).
"""

import json
import math
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

METRICS_FILE_ENV = "LGK_METRICS_FILE"
METRICS_RUN_ENV = "LGK_METRICS_RUN"
RUNS = frozenset({"dogfood", "load", "advisor", "refresh", "test"})
DEFAULT_RUN = "test"
CLIENTS = frozenset({"laptop", "workspace"})
UNITS = frozenset({"s", "ms", "min", "rows", "edges", "nodes", "count"})
INT_KEYS = frozenset({"rows", "edges", "n_samples", "depth"})
NAME_KEYS = frozenset({"branch", "kind", "direction", "table", "role"})
CONTEXT_KEYS = INT_KEYS | NAME_KEYS
GRAPH_SIZE_KEYS = ("rows", "edges")
# Every written line carries these (not None); the summariser refuses a line without one of them.
LINE_FIELDS = ("metric", "value", "unit", "commit", "run", "client", "rows", "edges")
# Identifier-shaped and short: a t- branch id is <= 63 chars; a token or a label never fits.
_NAME = re.compile(r"^[A-Za-z0-9_.:=/-]{1,64}$")


class MetricRefused(ValueError):
    """A metric call that could carry something other than a number plus fixed context."""


def metrics_path(environ: Optional[Dict[str, str]] = None) -> Optional[Path]:
    """The metrics file from LGK_METRICS_FILE, or None (off) when unset or empty."""
    env = os.environ if environ is None else environ
    value = env.get(METRICS_FILE_ENV, "")
    return Path(value) if value else None


def metrics_run(environ: Optional[Dict[str, str]] = None) -> str:
    """The run name from LGK_METRICS_RUN (unset or empty: `test`); a name outside RUNS is refused."""
    env = os.environ if environ is None else environ
    value = env.get(METRICS_RUN_ENV, "") or DEFAULT_RUN
    _check_run(value)
    return value


def freshness_lag_minutes(synced_at_s: float, source_commit_at_s: float) -> float:
    """Freshness lag (deck claim C30b) in minutes: how old the newest served data is when its sync lands.

    Definition: the sync completion time (when the synced table reached ONLINE_NO_PENDING_UPDATE after
    the update; live field `status.last_sync.sync_end_time`) MINUS the source Delta commit time of the
    data the synced table now holds (`status.last_sync.delta_table_sync_info.delta_commit_time`, cross-
    checked against DESCRIBE HISTORY of the source at `delta_commit_version`). Both in epoch seconds.
    A sync that completes before the commit it holds is impossible (swapped arguments or a stale
    reading), so it is refused, never recorded as a negative lag.
    """
    for what, v in (("synced_at_s", synced_at_s), ("source_commit_at_s", source_commit_at_s)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise MetricRefused(f"{what} must be a finite number of epoch seconds")
    if synced_at_s < source_commit_at_s:
        raise MetricRefused(f"sync completed at {synced_at_s} before the source commit it holds "
                            f"({source_commit_at_s}): not a freshness lag")
    return (synced_at_s - source_commit_at_s) / 60.0


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile: the ceil(q * n)-th smallest value (q in (0, 1])."""
    if not values:
        raise MetricRefused("percentile of no values")
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered) - 1e-9))
    return ordered[rank - 1]


def git_head(root: Path) -> str:
    """`git rev-parse HEAD` of the tree under test."""
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True,
                          text=True, check=True).stdout.strip()


def _utc_now() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _check_value(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MetricRefused(f"metric value must be a finite number, got {type(value).__name__}")


def _check_name(what: str, value: Any) -> None:
    if not isinstance(value, str) or not _NAME.match(value):
        raise MetricRefused(f"{what} must be a short identifier-shaped name")


def _check_run(value: Any) -> None:
    if value not in RUNS:
        raise MetricRefused(f"run must be one of {sorted(RUNS)}, got {value!r}")


def _check_client(value: Any) -> None:
    if value not in CLIENTS:
        raise MetricRefused(f"client must be one of {sorted(CLIENTS)}, got {value!r}")


def _check_context(context: Dict[str, Any]) -> None:
    missing = [k for k in GRAPH_SIZE_KEYS if k not in context]
    if missing:
        raise MetricRefused(f"a metric needs the graph size it was measured at: missing {missing}")
    unknown = sorted(set(context) - CONTEXT_KEYS)
    if unknown:
        raise MetricRefused(f"unknown metric context keys: {unknown}")
    for key, value in context.items():
        if key in INT_KEYS:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MetricRefused(f"context {key} must be a non-negative int")
        else:
            _check_name(f"context {key}", value)


def _check_line(record: Dict[str, Any]) -> None:
    """The last gate before a line is written: every LINE_FIELDS key present and not None."""
    missing = [k for k in LINE_FIELDS if record.get(k) is None]
    if missing:
        raise MetricRefused(f"metric line is missing {missing}")


class MetricsRecorder:
    """Callable `record(metric, value, unit, client="laptop", rows=.., edges=.., **context)`; a no-op (after
    validation) when path is None. `run` is fixed per recorder (the conftest reads LGK_METRICS_RUN).

    `commit()` and `endpoint_cu(branch)` are called lazily, only when recording, and cached.
    """

    def __init__(self, path: Optional[Path], *, suite: str, test: str, commit: Callable[[], str],
                 run: str, project: Optional[str] = None,
                 endpoint_cu: Optional[Callable[[str], Optional[Dict[str, float]]]] = None,
                 now: Callable[[], str] = _utc_now):
        _check_run(run)
        self.path = Path(path) if path is not None else None
        self.run = run
        self.suite, self.test, self.project = suite, test, project
        self._commit_fn, self._cu_fn, self._now = commit, endpoint_cu, now
        self._commit: Optional[str] = None
        self._cu: Dict[str, Optional[Dict[str, float]]] = {}

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def __call__(self, metric: str, value: Any, unit: str, client: str = "laptop", **context: Any) -> None:
        _check_name("metric", metric)
        if unit not in UNITS:
            raise MetricRefused(f"unit must be one of {sorted(UNITS)}")
        _check_value(value)
        _check_client(client)
        _check_context(context)
        if not self.enabled:
            return
        record = self._record(metric, value, unit, client, context)
        _check_line(record)
        self._append(record)

    def latencies(self, metric: str, samples_s: Iterable[float], client: str = "laptop", **context: Any) -> None:
        """Record `<metric>.p50` and `<metric>.p95` in ms over the samples (seconds), with n_samples."""
        ms = [s * 1000.0 for s in samples_s]
        if not ms:
            raise MetricRefused(f"{metric}: no latency samples")
        for q, suffix in ((0.5, "p50"), (0.95, "p95")):
            self(f"{metric}.{suffix}", percentile(ms, q), "ms", client=client, n_samples=len(ms), **context)

    def _record(self, metric: str, value: Any, unit: str, client: str,
                context: Dict[str, Any]) -> Dict[str, Any]:
        context = dict(context)
        branch = context.pop("branch", None)
        return {"metric": metric, "value": value, "unit": unit, "test": self.test, "suite": self.suite,
                "commit": self._head(), "branch": branch, "project": self.project,
                "endpoint_cu": self._endpoint_cu(branch), "run": self.run, "client": client,
                **context, "ts": self._now()}

    def _head(self) -> str:
        if self._commit is None:
            self._commit = self._commit_fn()
        return self._commit

    def _endpoint_cu(self, branch: Optional[str]) -> Optional[Dict[str, float]]:
        if branch is None or self._cu_fn is None:
            return None
        if branch not in self._cu:
            self._cu[branch] = self._cu_fn(branch)
        return self._cu[branch]

    def _append(self, record: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=False) + "\n")
