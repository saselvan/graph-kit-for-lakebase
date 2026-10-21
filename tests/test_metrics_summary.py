"""Offline: tools/metrics_summary.py groups a metrics file by metric (min/p50/p95/max, commit,
branch, run, client, graph size) and refuses a file that mixes commits, or runs without --run."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "metrics_summary.py"
A = "a" * 40
B = "b" * 40


def _line(metric, value, commit=A, branch="t-spine-1", unit="ms", run="test", client="laptop", rows=50, edges=32,
          **dims):
    return {"metric": metric, "value": value, "unit": unit, "test": "t::x", "suite": "s",
            "commit": commit, "branch": branch, "project": "p", "endpoint_cu": None, "run": run, "client": client,
            "rows": rows, "edges": edges, "ts": "2026-10-04T00:00:00Z", **dims}


def _run(tmp_path, lines, raw_extra="", args=()):
    f = tmp_path / "m.jsonl"
    f.write_text("".join(json.dumps(x) + "\n" for x in lines) + raw_extra)
    return subprocess.run([sys.executable, str(TOOL), str(f), *args], capture_output=True, text=True)


def _rows(stdout):
    """{metric: whitespace-split columns} for the table rows."""
    out = {}
    for line in stdout.splitlines():
        parts = line.split()
        if parts and parts[0] not in ("commit", "metric") and not set(line) <= {"-", " "}:
            out[parts[0]] = parts
    return out


def _refused(r):
    """A deliberate refusal (exit 2, "refused:"), not a crash (a traceback also exits non-zero)."""
    return r.returncode == 2 and r.stderr.startswith("refused:") and "Traceback" not in r.stderr


def test_groups_by_metric_with_min_p50_p95_max_and_branch(tmp_path):
    # hostile: interleaved metrics, unsorted values, a second branch for one metric, a blank line
    lines = [_line("walk.web.p50", v) for v in (40.0, 10.0, 30.0)]
    lines.insert(1, _line("build.wall", 12.5, unit="s"))
    lines.append(_line("walk.web.p50", 20.0, branch="t-walks-2"))
    lines.append(_line("walk.web.p50", 1.0))
    r = _run(tmp_path, lines, raw_extra="\n")
    assert r.returncode == 0, r.stderr
    assert f"commit {A}" in r.stdout
    rows = _rows(r.stdout)
    # columns: metric unit n min p50 p95 max branches
    assert rows["walk.web.p50"][:7] == ["walk.web.p50", "ms", "5", "1", "20", "40", "40"]
    assert rows["walk.web.p50"][7] == "t-spine-1,t-walks-2"
    assert rows["build.wall"][:7] == ["build.wall", "s", "1", "12.5", "12.5", "12.5", "12.5"]
    assert rows["build.wall"][7] == "t-spine-1"


def test_context_dimensions_split_a_metric_into_rows(tmp_path):
    # hostile: the same metric at two depths (a metric-only grouping would mix depth 1 and depth 4)
    lines = [_line("walk.web.p50", 5.0, depth=1, kind="degree", n_samples=20),
             _line("walk.web.p50", 50.0, depth=4, kind="degree", n_samples=20),
             _line("walk.web.p50", 7.0, depth=1, kind="degree", n_samples=20)]
    r = _run(tmp_path, lines)
    assert r.returncode == 0, r.stderr
    rows = _rows(r.stdout)
    assert rows["walk.web.p50[depth=1,kind=degree]"][1:7] == ["ms", "2", "5", "5", "7", "7"]
    assert rows["walk.web.p50[depth=4,kind=degree]"][1:7] == ["ms", "1", "50", "50", "50", "50"]
    assert "walk.web.p50" not in rows  # n_samples is not a dimension; no ungrouped row


def test_refuses_mixed_commits(tmp_path):
    r = _run(tmp_path, [_line("x", 1), _line("x", 2, commit=B)])
    assert _refused(r), r.stderr
    assert "mixes commits" in r.stderr and A in r.stderr and B in r.stderr
    assert "x" not in _rows(r.stdout)


def test_refuses_a_line_without_a_commit(tmp_path):
    bad = _line("x", 2)
    del bad["commit"]
    r = _run(tmp_path, [_line("x", 1), bad])
    assert _refused(r) and "commit" in r.stderr, r.stderr


def test_refuses_an_empty_file(tmp_path):
    r = _run(tmp_path, [])
    assert _refused(r), r.stderr


def test_refuses_mixed_units_for_one_metric(tmp_path):
    r = _run(tmp_path, [_line("x", 1, unit="ms"), _line("x", 2, unit="s")])
    assert _refused(r) and "unit" in r.stderr, r.stderr


def test_usage_without_a_file(tmp_path):
    r = subprocess.run([sys.executable, str(TOOL)], capture_output=True, text=True)
    assert r.returncode != 0 and "usage" in r.stderr.lower()


@pytest.mark.parametrize("missing", ["metric", "value", "unit"])
def test_refuses_a_line_missing_a_core_field(tmp_path, missing):
    bad = _line("x", 1)
    del bad[missing]
    r = _run(tmp_path, [bad])
    assert _refused(r) and missing in r.stderr, r.stderr


@pytest.mark.parametrize("commit", [None, ""])
def test_refuses_a_null_or_empty_commit(tmp_path, commit):
    # hostile: a one-line file whose commit is None would otherwise summarise "commit None"
    r = _run(tmp_path, [_line("x", 1, commit=commit)])
    assert _refused(r) and "commit" in r.stderr, r.stderr


@pytest.mark.parametrize("value", ["12", True, None, [1]])
def test_refuses_a_value_that_is_not_a_number(tmp_path, value):
    r = _run(tmp_path, [_line("x", 1), _line("x", value)])
    assert _refused(r), r.stderr


def test_every_dimension_splits_rows(tmp_path):
    # hostile: four lines of one metric that differ only in direction, table or role
    lines = [_line("q", 1.0, direction="inbound"), _line("q", 2.0, direction="both"),
             _line("q", 3.0, table="nodes"), _line("q", 4.0, role="reader")]
    rows = _rows(_run(tmp_path, lines).stdout)
    assert sorted(rows) == ["q[direction=both]", "q[direction=inbound]", "q[role=reader]", "q[table=nodes]"]


def test_p95_is_the_nearest_rank_of_twenty(tmp_path):
    # hostile: 20 values in descending order; p90 (18) and max (20) differ from p95 (19)
    r = _run(tmp_path, [_line("x", float(v)) for v in range(20, 0, -1)])
    assert _rows(r.stdout)["x"][3:7] == ["1", "10", "19", "20"]


# ------------------------------------------------------------------ run, client and graph size (NUM2)


def _table(stdout):
    """[(label, unit, n, min, p50, p95, max, branches, run, client, rows, edges)] for every table row."""
    return [tuple(line.split()) for line in stdout.splitlines()[2:] if line.strip()]


@pytest.mark.parametrize("missing", ["run", "client", "rows", "edges"])
def test_refuses_a_line_missing_its_context(tmp_path, missing):
    bad = _line("x", 1)
    del bad[missing]
    r = _run(tmp_path, [_line("x", 2), bad])
    assert _refused(r) and missing in r.stderr, r.stderr


@pytest.mark.parametrize("missing", ["run", "client", "rows", "edges"])
def test_refuses_a_null_context_field(tmp_path, missing):
    r = _run(tmp_path, [_line("x", 1, **{missing: None})])
    assert _refused(r) and missing in r.stderr, r.stderr


def test_every_row_shows_run_client_rows_and_edges(tmp_path):
    r = _run(tmp_path, [_line("walk.web.p50", 5.0, depth=1, rows=10209, edges=30123)])
    assert r.returncode == 0, r.stderr
    [row] = _table(r.stdout)
    assert row[0] == "walk.web.p50[depth=1]" and row[7:] == ("t-spine-1", "test", "laptop", "10209", "30123")
    assert r.stdout.splitlines()[1].split()[-4:] == ["run", "client", "rows", "edges"]


def test_laptop_and_workspace_are_separate_rows(tmp_path):
    # hostile: the same metric, depth and kind from both clients (a client-blind grouping mixes them)
    lines = [_line("walk.web.p50", 40.0, depth=2, client="laptop"),
             _line("walk.web.p50", 2.0, depth=2, client="workspace"),
             _line("walk.web.p50", 42.0, depth=2, client="laptop")]
    rows = _table(_run(tmp_path, lines).stdout)
    by_client = {row[9]: row for row in rows}
    assert sorted(by_client) == ["laptop", "workspace"]
    assert by_client["laptop"][2:5] == ("2", "40", "40") and by_client["workspace"][2:5] == ("1", "2", "2")


def test_two_graph_sizes_are_separate_rows(tmp_path):
    lines = [_line("q", 1.0, rows=50, edges=32), _line("q", 9.0, rows=5000, edges=3200)]
    rows = _table(_run(tmp_path, lines).stdout)
    assert sorted((row[10], row[11], row[4]) for row in rows) == [("50", "32", "1"), ("5000", "3200", "9")]


def test_refuses_mixed_runs_without_a_filter(tmp_path):
    r = _run(tmp_path, [_line("x", 1, run="test"), _line("x", 2, run="load")])
    assert _refused(r) and "mixes runs" in r.stderr and "load" in r.stderr and "test" in r.stderr, r.stderr
    assert r.stdout == ""


def test_run_filter_keeps_one_run_of_a_mixed_file(tmp_path):
    # hostile: the other run's lines are from ANOTHER commit too; the filter drops them before the commit check
    lines = [_line("x", 1.0, run="test"), _line("x", 100.0, run="load", commit=B), _line("x", 3.0, run="test")]
    r = _run(tmp_path, lines, args=("--run", "test"))
    assert r.returncode == 0, r.stderr
    [row] = _table(r.stdout)
    assert row[2:7] == ("2", "1", "1", "3", "3") and row[8] == "test"
    assert f"commit {A}  run test" in r.stdout


def test_run_filter_with_no_matching_line_is_refused(tmp_path):
    r = _run(tmp_path, [_line("x", 1, run="test")], args=("--run", "dogfood"))
    assert _refused(r) and "dogfood" in r.stderr, r.stderr
