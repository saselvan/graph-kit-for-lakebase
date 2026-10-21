"""The measure record: `<run_folder>/measure.jsonl` (one line per metric) and `measure_report.md`.

Field names follow lgk.metrics where they exist (metric, value, unit, endpoint_cu, n_samples,
depth, edges, ts). Counts and timings only: every key is from a closed list, every
string is identifier-shaped (or a fixed constant), and the step passes the seeds it used as
`forbidden`, so no node key, label, property value or seed value can be written. A refused
record writes nothing. A rerun replaces both files.
"""

import hashlib
import json
import math
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from lgk.metrics import UNITS
from lgk.runner import RunFolder

JSONL, REPORT = "measure.jsonl", "measure_report.md"
FRESHNESS = {"status": "pending", "reason": "nightly refresh untested from inside a workspace at this build"}


# C23: the steps up to and including the first serve (switch), in run order.
# NOTE: This constant is kept for backward compatibility and testing, but is NOT used by first_serve_time.
# first_serve_time derives the steps from RunFolder.STEPS at call time, not at import time.
FIRST_SERVE_STEPS = tuple(RunFolder.STEPS[: RunFolder.STEPS.index("switch") + 1])

_NAME = re.compile(r"^[A-Za-z0-9_.:=/-]{1,64}$")
_TS = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]{8,15}Z$")
_NAME_KEYS = frozenset({"metric", "shape", "status", "reason", "reference", "timing", "excludes",
                        "execution_label", "remainder_label", "prepared_statements"})
_INT_KEYS = frozenset({"nodes", "edges", "depth", "seeds_checked", "seeds_equal", "seeds_different", "n_samples"})
_FLOAT_KEYS = frozenset({"client_p95_ms", "client_mean_ms", "execution_p95_ms", "execution_mean_ms",
                         "planning_mean_ms", "select1_rtt_mean_ms", "remainder_mean_ms",
                         "client_mean_se_ms", "select1_rtt_mean_se_ms", "planning_mean_se_ms", "execution_mean_se_ms"})
_OTHER_KEYS = frozenset({"value", "unit", "build", "endpoint_cu", "ts", "freshness", "steps_missing"})
KEYS = _NAME_KEYS | _INT_KEYS | _FLOAT_KEYS | _OTHER_KEYS
_MIN_FORBIDDEN_LEN = 6  # shorter tokens could sit inside a number by chance


class RecordRefused(ValueError):
    """A measure line that could carry something other than counts, timings and fixed labels."""


def _utc_now() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------- C23 / C30b


def first_serve_time(run_json: Path) -> Dict[str, Any]:
    """Sum of run.json `duration_s` over preflight..switch (step time; operator time excluded).

    A step not succeeded or without a duration is named in `steps_missing` and the sum is
    `partial` (the value is the sum of the steps that did run)."""
    state = json.loads(Path(run_json).read_text()) if Path(run_json).exists() else {}
    total, missing = 0.0, []
    # Derive the serve group from RunFolder.STEPS when called, not at import time
    first_serve_steps = tuple(RunFolder.STEPS[: RunFolder.STEPS.index("switch") + 1])
    for step in first_serve_steps:
        entry = state.get(step) or {}
        d = entry.get("duration_s")
        if entry.get("status") != "success" or isinstance(d, bool) or not isinstance(d, (int, float)):
            missing.append(step)
            continue
        total += float(d)
    return {"value": round(total, 6), "status": "partial" if missing else "complete", "steps_missing": missing}


def freshness_line() -> Dict[str, Any]:
    """C30b is not measured at this build: the line says so, with the fixed reason."""
    return {"metric": "freshness_lag", "value": None, "unit": "min", "freshness": dict(FRESHNESS)}


# ---------------------------------------------------------------------------- build identity


def _package_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _version() -> Optional[str]:
    from importlib import metadata

    try:
        return metadata.version("lgk")
    except metadata.PackageNotFoundError:
        return None


def _commit(root: Path) -> Optional[str]:
    """git HEAD of the source tree when the package runs from a checkout; None from a wheel."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    head = out.stdout.strip()
    return head if out.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", head) else None


def _packaged_commit() -> Optional[str]:
    """The commit a packaged wheel was built from: `lgk._build.COMMIT`, which tools/package_kit.py
    writes into the archived tree before `uv build`. None when the module is absent (not a
    packaged build) or holds no full sha."""
    try:
        from lgk import _build
    except ImportError:
        return None
    commit = getattr(_build, "COMMIT", None)
    return commit if isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) else None


def _source_sha256(pkg: Path) -> str:
    """16 hex of sha256 over the package's .py files (path + bytes, sorted): which code ran."""
    h = hashlib.sha256()
    for f in sorted(pkg.rglob("*.py")):
        h.update(str(f.relative_to(pkg)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def build_identity() -> Dict[str, Optional[str]]:
    """{version (wheel metadata), commit (the packaged wheel's lgk._build.COMMIT, else git HEAD of a
    checkout, else None), source_sha256}."""
    pkg = _package_dir()
    return {"version": _version(), "commit": _packaged_commit() or _commit(pkg),
            "source_sha256": _source_sha256(pkg)}


# ---------------------------------------------------------------------------- validation


def _name(key: str, value: Any) -> None:
    if not isinstance(value, str) or not _NAME.match(value):
        raise RecordRefused(f"{key} must be a short identifier-shaped name")


def _number(key: str, value: Any, allow_none: bool = True) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RecordRefused(f"{key} must be a finite number")


def _check_value(key: str, value: Any) -> None:
    if key in _NAME_KEYS:
        if key in ("execution_label", "remainder_label"):
            # These labels must be exact constants
            from lgk.measure import timing as timing_module
            if key == "execution_label" and value != timing_module.EXECUTION_LABEL:
                raise RecordRefused(f"execution_label must be the exact constant")
            if key == "remainder_label" and value != timing_module.REMAINDER_LABEL:
                raise RecordRefused(f"remainder_label must be the exact constant")
        elif key == "prepared_statements":
            # Must be exactly "on" or "off"
            from lgk.measure import timing as timing_module
            if value not in timing_module.PREPARED_MODES:
                raise RecordRefused(f"prepared_statements must be one of {sorted(timing_module.PREPARED_MODES)}")
        else:
            _name(key, value)
    elif key in _INT_KEYS:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise RecordRefused(f"{key} must be a non-negative int")
    elif key in _FLOAT_KEYS:
        _number(key, value, allow_none=False)
    elif key == "value":
        _number(key, value)
    elif key == "unit":
        if value not in UNITS:
            raise RecordRefused(f"unit must be one of {sorted(UNITS)}")
    elif key == "ts":
        if not isinstance(value, str) or not _TS.match(value):
            raise RecordRefused("ts must be a UTC timestamp")
    elif key == "freshness":
        if value != FRESHNESS:
            raise RecordRefused("freshness must be the fixed pending record")
    elif key == "steps_missing":
        if not isinstance(value, list) or any(s not in RunFolder.STEPS for s in value):
            raise RecordRefused("steps_missing must list toolkit step names")
    elif key == "build":
        if not isinstance(value, dict) or set(value) != {"version", "commit", "source_sha256"}:
            raise RecordRefused("build must be {version, commit, source_sha256}")
        for k, v in value.items():
            if v is not None:
                _name(f"build.{k}", v)
    elif key == "endpoint_cu":
        if value is not None:
            if not isinstance(value, dict) or set(value) != {"min", "max"}:
                raise RecordRefused("endpoint_cu must be {min, max}")
            for k, v in value.items():
                _number(f"endpoint_cu.{k}", v)


def check_line(line: Dict[str, Any], forbidden: Iterable[str] = ()) -> str:
    """The line as JSON, or RecordRefused: unknown key, free text, non-number, forbidden token."""
    unknown = sorted(set(line) - KEYS)
    if unknown:
        raise RecordRefused(f"unknown measure keys: {unknown}")
    for key in ("metric", "unit", "ts"):
        if key not in line:
            raise RecordRefused(f"measure line is missing {key}")

    # walk_latency lines must have prepared_statements and all standard error fields
    if line.get("metric") == "walk_latency":
        required_keys = {"prepared_statements", "client_mean_se_ms", "select1_rtt_mean_se_ms",
                         "planning_mean_se_ms", "execution_mean_se_ms"}
        for key in required_keys:
            if key not in line:
                raise RecordRefused(f"walk_latency line is missing {key}")

    for key, value in line.items():
        _check_value(key, value)
    text = json.dumps(line, sort_keys=False)
    for token in forbidden:
        if len(token) >= _MIN_FORBIDDEN_LEN and token in text:
            raise RecordRefused("a measure line holds a seed or node value")
    return text


# ---------------------------------------------------------------------------- writing


def _fmt(v: Any) -> str:
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def report_markdown(lines: Sequence[Dict[str, Any]]) -> str:
    """The short human report: counts and timings only."""
    first = lines[0] if lines else {}
    b = first.get("build") or {}
    cu = first.get("endpoint_cu") or {}
    out = ["# Measure report", "",
           f"Build: version {_fmt(b.get('version'))}, commit {_fmt(b.get('commit'))}, "
           f"source {_fmt(b.get('source_sha256'))}. Graph: {_fmt(first.get('nodes'))} nodes, "
           f"{_fmt(first.get('edges'))} edges. CU range {_fmt(cu.get('min'))}-{_fmt(cu.get('max'))}.", "",
           "## C10 walk correctness (served walks vs a reference computed in Python over node and edge ids read from the source tables)", "",
           "| shape | depth | seeds checked | seeds equal | seeds different | status |",
           "|---|---|---|---|---|---|"]
    for x in lines:
        if x["metric"] == "walk_correctness":
            status = x["status"] + (f" ({x['reason']})" if x.get("reason") else "")
            out.append(f"| {x['shape']} | {_fmt(x.get('depth'))} | {x['seeds_checked']} | "
                       f"{x['seeds_equal']} | {x['seeds_different']} | {status} |")

    # C13: walk speed with client timing breakdown
    execution_label = None
    remainder_label = None
    lat_by_shape_depth: Dict[tuple, Dict[str, Any]] = {}
    prepared_statements_modes: List[str] = []
    for x in lines:
        if x["metric"] == "walk_latency":
            lat_by_shape_depth[(x["shape"], x["depth"])] = x
            if execution_label is None and "execution_label" in x:
                execution_label = x["execution_label"]
            if remainder_label is None and "remainder_label" in x:
                remainder_label = x["remainder_label"]
            if "prepared_statements" in x:
                prepared_statements_modes.append(x["prepared_statements"])

    if lat_by_shape_depth:
        out += ["", f"## C13 walk speed (ms, client p95 = in-region time on warm connection)", "",
                "| shape | depth | client p95 | execution p95 | client mean | client mean SE | execution mean | execution mean SE | planning mean | planning mean SE | select 1 mean | select 1 mean SE | remainder mean | n | prepared statements |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for (shape, depth), x in sorted(lat_by_shape_depth.items()):
            mode = x.get("prepared_statements", "")
            out.append(f"| {shape} | {depth} | {_fmt(x.get('value'))} | "
                       f"{_fmt(x.get('execution_p95_ms'))} | {_fmt(x.get('client_mean_ms'))} | "
                       f"{_fmt(x.get('client_mean_se_ms'))} | {_fmt(x.get('execution_mean_ms'))} | "
                       f"{_fmt(x.get('execution_mean_se_ms'))} | {_fmt(x.get('planning_mean_ms'))} | "
                       f"{_fmt(x.get('planning_mean_se_ms'))} | {_fmt(x.get('select1_rtt_mean_ms'))} | "
                       f"{_fmt(x.get('select1_rtt_mean_se_ms'))} | {_fmt(x.get('remainder_mean_ms'))} | {x.get('n_samples')} | {mode} |")

        # Print summary line only if all walk_latency lines have the same mode
        if prepared_statements_modes and len(set(prepared_statements_modes)) == 1:
            out.append("")
            out.append(f"**prepared statements: {prepared_statements_modes[0]}**")
        if execution_label:
            out.append("")
            out.append(f"**Execution**: {execution_label}")
        if remainder_label:
            out.append(f"**Remainder**: {remainder_label}")

    for x in lines:
        if x["metric"] == "time_to_first_served_graph":
            missing = ", ".join(x.get("steps_missing") or []) or "none"
            out += ["", "## C23 time to first served graph", "",
                    f"{_fmt(x['value'])} s of step time (preflight..switch, run.json duration_s; excludes "
                    f"operator time). Status: {x['status']}; steps missing: {missing}."]
        if x["metric"] == "freshness_lag":
            out += ["", "## C30b freshness", "", f"{x['freshness']['status']}: {x['freshness']['reason']}."]
    return "\n".join(out) + "\n"


def write(run_folder: Any, lines: List[Dict[str, Any]], forbidden: Iterable[str]) -> Path:
    """Validate every line first, then (re)write measure.jsonl and measure_report.md."""
    forbidden = [str(f) for f in forbidden]
    texts = [check_line(x, forbidden) for x in lines]
    md = report_markdown(lines)  # renders only fields check_line passed
    folder = Path(run_folder)
    (folder / JSONL).write_text("".join(t + "\n" for t in texts))
    (folder / REPORT).write_text(md)
    return folder / JSONL


def stamp(lines: List[Dict[str, Any]], common: Dict[str, Any], now=_utc_now) -> List[Dict[str, Any]]:
    """Each line with the common fields (build, nodes, edges, endpoint_cu) and ts."""
    ts = now()
    return [{**x, **common, "ts": ts} for x in lines]
