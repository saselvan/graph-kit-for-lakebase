"""Summarise an LGK_METRICS_FILE: per metric, n / min / p50 / p95 / max with its run, client, graph size
(rows, edges) and branches, for ONE commit and ONE run.

A metric is grouped with its context dimensions (depth, kind, direction, table, role), its client and
its graph size, so depth 1 and depth 4 of the same walk, laptop and workspace, and two graph sizes are
separate rows. Columns: metric unit n min p50 p95 max branches run client rows edges.

    uv run --no-sync python tools/metrics_summary.py /tmp/lgk-metrics.jsonl [--run test]

Refuses (exit 2) a file that mixes commits; a file that mixes runs unless `--run <name>` keeps one
(and then refuses if no line has that run); a line without a commit, metric, value, unit, run, client,
rows or edges; a metric recorded in two units; a value that is not a finite number; and an empty file:
a deck number must come from one tree under test and one kind of run.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lgk.metrics import LINE_FIELDS, percentile  # noqa: E402

REQUIRED = LINE_FIELDS
DIMENSIONS = ("depth", "kind", "direction", "table", "role")


class SummaryRefused(ValueError):
    pass


def _check_lines(lines: List[Dict[str, Any]]) -> None:
    for n, line in enumerate(lines, 1):
        missing = [k for k in REQUIRED if k not in line or line[k] in (None, "")]
        if missing:
            raise SummaryRefused(f"line {n}: missing {missing}")
        v = line["value"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise SummaryRefused(f"line {n}: value is not a finite number")


def _one_run(lines: List[Dict[str, Any]], run: Optional[str]) -> List[Dict[str, Any]]:
    """The lines of one run: all of them if they share a run, else only `run` (required then)."""
    if run is not None:
        kept = [line for line in lines if line["run"] == run]
        if not kept:
            raise SummaryRefused(f"no metric line has run {run!r}")
        return kept
    runs = sorted({line["run"] for line in lines})
    if len(runs) > 1:
        raise SummaryRefused(f"file mixes runs: {runs}; pick one with --run <name>")
    return lines


def load(path: Path, run: Optional[str] = None) -> List[Dict[str, Any]]:
    lines = [json.loads(raw) for raw in path.read_text().splitlines() if raw.strip()]
    if not lines:
        raise SummaryRefused(f"{path}: no metric lines")
    _check_lines(lines)
    lines = _one_run(lines, run)
    commits = sorted({line["commit"] for line in lines})
    if len(commits) > 1:
        raise SummaryRefused(f"file mixes commits: {commits}")
    return lines


def _key(line: Dict[str, Any]) -> Tuple[str, str, int, int]:
    dims = ",".join(f"{d}={line[d]}" for d in DIMENSIONS if line.get(d) is not None)
    label = f"{line['metric']}[{dims}]" if dims else line["metric"]
    return label, line["client"], line["rows"], line["edges"]


def group(lines: List[Dict[str, Any]]) -> Dict[Tuple[str, str, int, int], Dict[str, Any]]:
    groups: Dict[Tuple[str, str, int, int], Dict[str, Any]] = {}
    units: Dict[str, str] = {}
    for line in lines:
        key = _key(line)
        if units.setdefault(key[0], line["unit"]) != line["unit"]:
            raise SummaryRefused(f"{key[0]}: unit {line['unit']!r} != {units[key[0]]!r}")
        g = groups.setdefault(key, {"unit": line["unit"], "run": line["run"], "values": [], "branches": set()})
        g["values"].append(line["value"])
        g["branches"].add(line.get("branch") or "-")
    return groups


def _num(x: float) -> str:
    return f"{x:.6g}"


def render(lines: List[Dict[str, Any]]) -> str:
    groups = group(lines)
    out = [f"commit {lines[0]['commit']}  run {lines[0]['run']}",
           f"{'metric':<56} {'unit':<5} {'n':>4} {'min':>10} {'p50':>10} {'p95':>10} {'max':>10} "
           f"{'branches':<24} {'run':<8} {'client':<9} {'rows':>7} {'edges':>7}"]
    for key in sorted(groups):
        label, client, rows, edges = key
        g = groups[key]
        v = g["values"]
        out.append(f"{label:<56} {g['unit']:<5} {len(v):>4} {_num(min(v)):>10} {_num(percentile(v, 0.5)):>10} "
                   f"{_num(percentile(v, 0.95)):>10} {_num(max(v)):>10} {','.join(sorted(g['branches'])):<24} "
                   f"{g['run']:<8} {client:<9} {rows:>7} {edges:>7}")
    return "\n".join(out)


def _parse_args(argv: List[str]) -> Tuple[Path, Optional[str]]:
    args, run = list(argv[1:]), None
    if "--run" in args:
        i = args.index("--run")
        if i + 1 >= len(args):
            raise SystemExit(2)
        run = args[i + 1]
        del args[i:i + 2]
    if len(args) != 1:
        raise SystemExit(2)
    return Path(args[0]), run


def main(argv: List[str]) -> int:
    try:
        path, run = _parse_args(argv)
    except SystemExit:
        print("usage: metrics_summary.py <metrics.jsonl> [--run <name>]", file=sys.stderr)
        return 2
    try:
        print(render(load(path, run)))
    except SummaryRefused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
