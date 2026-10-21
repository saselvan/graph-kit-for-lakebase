"""Offline: the opt-in numbers recorder for live tests (`LGK_METRICS_FILE`).

Off by default (nothing written, no git call, no endpoint read). When on, one JSON line per metric
with the fixed fields; values are numbers only and context keys come from a fixed list, so no personal
data and no token can reach the file.
"""

import json
import math
import subprocess
from pathlib import Path

import pytest

import conftest
from lgk.metrics import CONTEXT_KEYS, MetricRefused, MetricsRecorder, git_head, metrics_path, percentile

SHA = "0123456789abcdef0123456789abcdef01234567"
G = {"rows": 50, "edges": 32}  # the graph size every line must carry


def _recorder(path, **kw):
    calls = {"commit": 0, "cu": []}

    def commit():
        calls["commit"] += 1
        return SHA

    def cu(branch):
        calls["cu"].append(branch)
        return {"min": 0.5, "max": 16.0}

    kw.setdefault("run", "test")
    rec = MetricsRecorder(path, suite="test_walks_live", test="tests/test_walks_live.py::T::test_x",
                          commit=commit, project="demo-project", endpoint_cu=cu, **kw)
    return rec, calls


def _lines(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# ------------------------------------------------------------------ off by default


def test_unset_variable_writes_nothing_and_calls_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # hostile: an empty string is "unset" too (a shell `export LGK_METRICS_FILE=`)
    for environ in ({}, {"LGK_METRICS_FILE": ""}):
        assert metrics_path(environ) is None
        rec, calls = _recorder(metrics_path(environ))
        assert rec.enabled is False
        rec("build.wall", 12.5, "s", branch="t-spine-1", rows=50, edges=32)
        rec.latencies("walk.web", [0.010, 0.020], depth=2, **G)
        assert calls == {"commit": 0, "cu": []}  # no git call, no endpoint read
    assert list(tmp_path.iterdir()) == []


def test_conftest_factory_is_off_without_the_variable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rec = conftest.metrics_recorder("tests/test_x.py::test_y", "test_x", environ={})
    assert rec.enabled is False
    rec("lookup.latency", 1.0, "ms", **G)
    assert list(tmp_path.iterdir()) == []


def test_conftest_factory_writes_with_the_real_commit_and_the_nodeid(tmp_path):
    out = tmp_path / "m.jsonl"
    rec = conftest.metrics_recorder("tests/test_x.py::test_y", "test_x", environ={"LGK_METRICS_FILE": str(out)})
    rec("lookup.latency", 3, "ms", **G)
    [line] = _lines(out)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(conftest.__file__).parent,
                          capture_output=True, text=True, check=True).stdout.strip()
    assert (line["test"], line["suite"], line["commit"]) == ("tests/test_x.py::test_y", "test_x", head)
    assert line["branch"] is None and line["endpoint_cu"] is None  # no branch given: no endpoint read


# ------------------------------------------------------------------ the JSON shape


def test_one_json_line_per_metric_with_every_field(tmp_path):
    out = tmp_path / "sub" / "m.jsonl"  # hostile: the folder does not exist yet
    rec, calls = _recorder(out, now=lambda: "2026-10-04T12:00:00Z")
    rec("build.wall", 12.5, "s", branch="t-spine-1", rows=50, edges=32)
    rec("provision.synced_rows", 50, "rows", branch="t-spine-1", table="nodes", rows=7, edges=9)
    first, second = _lines(out)
    assert first == {
        "metric": "build.wall", "value": 12.5, "unit": "s",
        "test": "tests/test_walks_live.py::T::test_x", "suite": "test_walks_live", "commit": SHA,
        "branch": "t-spine-1", "project": "demo-project", "endpoint_cu": {"min": 0.5, "max": 16.0},
        "run": "test", "client": "laptop", "rows": 50, "edges": 32, "ts": "2026-10-04T12:00:00Z",
    }
    assert second["table"] == "nodes" and second["value"] == 50 and (second["rows"], second["edges"]) == (7, 9)
    assert calls["commit"] == 1 and calls["cu"] == ["t-spine-1"]  # cached: once per session/branch


def test_appends_to_an_existing_file(tmp_path):
    out = tmp_path / "m.jsonl"
    out.write_text('{"metric": "earlier"}\n')
    rec, _ = _recorder(out)
    rec("x", 1, "count", **G)
    assert [d["metric"] for d in _lines(out)] == ["earlier", "x"]


def test_ts_is_utc_iso(tmp_path):
    from datetime import UTC, datetime

    out = tmp_path / "m.jsonl"
    rec, _ = _recorder(out)
    rec("x", 1, "count", **G)
    ts = _lines(out)[0]["ts"]
    assert ts.endswith("Z") and "T" in ts and len(ts) == len("2026-10-04T12:00:00.000000Z")
    # a local-time stamp with a Z suffix is off by the zone offset (hours), not seconds
    parsed = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    assert abs((datetime.now(UTC) - parsed).total_seconds()) < 60


def test_no_branch_means_no_endpoint_read(tmp_path):
    out = tmp_path / "m.jsonl"
    rec, calls = _recorder(out)
    rec("x", 1, "count", **G)
    assert calls["cu"] == [] and _lines(out)[0]["endpoint_cu"] is None


def test_n_samples_counts_repeated_samples(tmp_path):
    out = tmp_path / "m.jsonl"
    rec, _ = _recorder(out)
    rec.latencies("w", [0.01, 0.01, 0.02], **G)  # hostile: a repeated sample
    assert [d["n_samples"] for d in _lines(out)] == [3, 3]


def test_conftest_commit_is_the_conftest_tree_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # hostile: cwd is not a git tree
    conftest._tree_head.cache_clear()
    try:
        out = tmp_path / "m.jsonl"
        conftest.metrics_recorder("t::x", "t", environ={"LGK_METRICS_FILE": str(out)})("x", 1, "count", **G)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(conftest.__file__).parent,
                              capture_output=True, text=True, check=True).stdout.strip()
        assert _lines(out)[0]["commit"] == head
    finally:
        conftest._tree_head.cache_clear()


def test_latencies_write_p50_p95_in_ms_with_n_samples(tmp_path):
    out = tmp_path / "m.jsonl"
    rec, _ = _recorder(out)
    # hostile: unsorted samples (a lazy p50 that takes the middle of the input order is wrong)
    samples = [0.040, 0.001, 0.010, 0.030, 0.020]
    rec.latencies("walk.web", samples, branch="t-w-1", depth=3, **G)
    p50, p95 = _lines(out)
    assert (p50["metric"], p50["unit"], p50["n_samples"], p50["depth"]) == ("walk.web.p50", "ms", 5, 3)
    assert math.isclose(p50["value"], 20.0)  # nearest rank: 3rd of [1, 10, 20, 30, 40] ms
    assert p95["metric"] == "walk.web.p95" and math.isclose(p95["value"], 40.0)


def test_latencies_refuse_no_samples(tmp_path):
    rec, _ = _recorder(tmp_path / "m.jsonl")
    with pytest.raises(MetricRefused):
        rec.latencies("walk.web", [], **G)


@pytest.mark.parametrize("values,q,want", [
    ([5], 0.5, 5), ([5], 0.95, 5),
    ([4, 1, 3, 2], 0.5, 2), ([4, 1, 3, 2], 0.95, 4),  # hostile: even count, reordered
    (list(range(1, 21)), 0.95, 19), (list(range(1, 21)), 0.5, 10),
])
def test_percentile_is_nearest_rank(values, q, want):
    assert percentile(values, q) == want


def test_git_head_reads_the_tree_under_test(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "f").write_text("x")
    git = ["git", "-C", str(tmp_path), "-c", "user.email=t@example.com", "-c", "user.name=t"]
    subprocess.run(git + ["add", "f"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "one"], check=True)
    want = subprocess.run(git + ["log", "-1", "--format=%H"], capture_output=True, text=True).stdout.strip()
    assert git_head(tmp_path) == want and len(want) == 40


# ------------------------------------------------------------------ refusals (each guard on its own)


@pytest.mark.parametrize("bad", ["12.5", None, True, float("nan"), float("inf"), [1], {"v": 1}])
def test_value_that_is_not_a_finite_number_is_refused(tmp_path, bad):
    out = tmp_path / "m.jsonl"
    rec, _ = _recorder(out)
    with pytest.raises(MetricRefused):
        rec("x", bad, "s", **G)
    assert not out.exists()


def test_refusal_holds_when_the_recorder_is_off(tmp_path):
    rec, _ = _recorder(None)
    with pytest.raises(MetricRefused):
        rec("x", "Dr. Provider 0", "s", **G)  # a bad call fails offline too, not only when recording
    with pytest.raises(MetricRefused):
        rec("x", 1, "s", patient="p", **G)


@pytest.mark.parametrize("key", ["patient", "token", "natural_key", "label", "Branch"])
def test_unknown_context_key_is_refused(tmp_path, key):
    out = tmp_path / "m.jsonl"
    rec, _ = _recorder(out)
    with pytest.raises(MetricRefused, match=key):
        rec("x", 1, "s", **G, **{key: "v"})
    assert not out.exists()


@pytest.mark.parametrize("key", ["project", "commit", "test", "suite", "ts", "endpoint_cu", "value"])
def test_recorder_owned_fields_cannot_be_passed_as_context(tmp_path, key):
    rec, _ = _recorder(tmp_path / "m.jsonl")
    with pytest.raises((MetricRefused, TypeError)):
        rec("x", 1, "s", **G, **{key: "v"})


def test_context_keys_are_the_fixed_list():
    assert set(CONTEXT_KEYS) == {"branch", "rows", "edges", "n_samples", "depth", "kind", "direction",
                                 "table", "role"}


@pytest.mark.parametrize("ctx", [
    {"branch": "eyJhbGciOi" + "A" * 200},   # hostile: a token-length string
    {"kind": "Dr. Provider 0"},              # hostile: a label with a space (free text)
    {"table": "nodes; DROP"},
    {"role": 7},                              # a number where a name goes
    {"rows": "50"}, {"rows": 1.5}, {"rows": True}, {"depth": -1},
])
def test_context_values_are_refused_unless_ints_or_short_names(tmp_path, ctx):
    rec, _ = _recorder(tmp_path / "m.jsonl")
    with pytest.raises(MetricRefused):
        rec("x", 1, "s", **{**G, **ctx})


@pytest.mark.parametrize("metric,unit", [("x y", "s"), ("", "s"), ("x", "seconds"), ("x", "")])
def test_metric_name_and_unit_are_checked(tmp_path, metric, unit):
    rec, _ = _recorder(tmp_path / "m.jsonl")
    with pytest.raises(MetricRefused):
        rec(metric, 1, unit, **G)


def test_lookup_helper_records_per_role_with_its_own_kind(tmp_path):
    """The live lookups helper passes `kind` as context (similar_jaccard top_n) without a clash."""
    from tests.test_lookups_live import _record_lookup

    out = tmp_path / "m.jsonl"
    rec, _ = _recorder(out)
    _record_lookup(rec, {"branch_id": "t-lookups-1", "graph_size": G}, "similar_jaccard",
                   {"operator": [0.002, 0.001], "reader": [0.004]}, kind="top10-noindex")
    got = [(d["metric"], d["role"], d["kind"], d["n_samples"]) for d in _lines(out)]
    assert got == [("lookup.similar_jaccard.p50", "operator", "top10-noindex", 2),
                   ("lookup.similar_jaccard.p95", "operator", "top10-noindex", 2),
                   ("lookup.similar_jaccard.p50", "reader", "top10-noindex", 1),
                   ("lookup.similar_jaccard.p95", "reader", "top10-noindex", 1)]
