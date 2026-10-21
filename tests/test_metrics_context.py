"""Offline: every metric line says what it was measured at (graph size, run, client), the freshness lag
definition (C30b), and the in-region (workspace) latency samples coming back from a fake job run."""

import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import conftest
from lgk.metrics import MetricRefused, MetricsRecorder, freshness_lag_minutes, metrics_run
from lgk.refresh import lag
from lgk.runner import RunStatus
from lgk import workspace_latency as wl

SHA = "0123456789abcdef0123456789abcdef01234567"
G = {"rows": 50, "edges": 32}


def _rec(path, run="test"):
    return MetricsRecorder(path, suite="s", test="t::x", commit=lambda: SHA, run=run)


def _lines(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# ------------------------------------------------------------------ rows / edges


@pytest.mark.parametrize("missing", ["rows", "edges"])
@pytest.mark.parametrize("enabled", [True, False])
def test_a_line_without_the_graph_size_is_refused(tmp_path, missing, enabled):
    out = tmp_path / "m.jsonl"
    rec = _rec(out if enabled else None)
    size = {k: v for k, v in G.items() if k != missing}
    with pytest.raises(MetricRefused, match=missing):
        rec("walk.web.p50", 1.0, "ms", **size)
    with pytest.raises(MetricRefused, match=missing):
        rec.latencies("walk.web", [0.1], **size)
    assert not out.exists()


def test_the_graph_size_is_written_on_every_line(tmp_path):
    out = tmp_path / "m.jsonl"
    rec = _rec(out)
    # hostile: a zero-edge graph (a falsy size must still be written, not dropped as "missing")
    rec("x", 1, "count", rows=3, edges=0)
    rec.latencies("w", [0.2, 0.1], rows=10209, edges=30123, depth=4)
    assert [(d["rows"], d["edges"]) for d in _lines(out)] == [(3, 0), (10209, 30123), (10209, 30123)]


# ------------------------------------------------------------------ client


def test_client_defaults_to_laptop_and_takes_workspace(tmp_path):
    out = tmp_path / "m.jsonl"
    rec = _rec(out)
    rec("x", 1, "count", **G)
    rec("x", 2, "count", client="workspace", **G)
    rec.latencies("w", [0.1], client="workspace", **G)
    assert [d["client"] for d in _lines(out)] == ["laptop", "workspace", "workspace", "workspace"]


@pytest.mark.parametrize("client", ["phone", "", None, "Laptop", "workspace "])
@pytest.mark.parametrize("enabled", [True, False])
def test_any_other_client_is_refused(tmp_path, client, enabled):
    out = tmp_path / "m.jsonl"
    rec = _rec(out if enabled else None)
    with pytest.raises(MetricRefused, match="client"):
        rec("x", 1, "count", client=client, **G)
    assert not out.exists()


# ------------------------------------------------------------------ run (LGK_METRICS_RUN)


@pytest.mark.parametrize("environ,want", [({}, "test"), ({"LGK_METRICS_RUN": ""}, "test"),
                                          ({"LGK_METRICS_RUN": "load"}, "load"),
                                          ({"LGK_METRICS_RUN": "dogfood"}, "dogfood"),
                                          ({"LGK_METRICS_RUN": "advisor"}, "advisor"),
                                          ({"LGK_METRICS_RUN": "refresh"}, "refresh")])
def test_metrics_run_is_honoured_with_test_as_default(environ, want):
    assert metrics_run(environ) == want


@pytest.mark.parametrize("bad", ["prod", "TEST", "load ", "test,load"])
def test_metrics_run_outside_the_list_is_refused(bad):
    with pytest.raises(MetricRefused, match="run"):
        metrics_run({"LGK_METRICS_RUN": bad})


def test_conftest_factory_writes_the_run_from_the_environment(tmp_path):
    out = tmp_path / "m.jsonl"
    rec = conftest.metrics_recorder("t::x", "t", environ={"LGK_METRICS_FILE": str(out), "LGK_METRICS_RUN": "load"})
    rec("x", 1, "count", **G)
    rec2 = conftest.metrics_recorder("t::y", "t", environ={"LGK_METRICS_FILE": str(out)})
    rec2("x", 1, "count", **G)
    assert [d["run"] for d in _lines(out)] == ["load", "test"]


def test_conftest_factory_refuses_a_bad_run(tmp_path):
    with pytest.raises(MetricRefused, match="run"):
        conftest.metrics_recorder("t::x", "t", environ={"LGK_METRICS_FILE": str(tmp_path / "m"),
                                                          "LGK_METRICS_RUN": "nightly"})


@pytest.mark.parametrize("run", [None, "", "x"])
def test_a_recorder_without_a_valid_run_is_refused(run):
    with pytest.raises(MetricRefused, match="run"):
        _rec(None, run=run)


def test_a_line_that_lost_a_field_is_not_written(tmp_path, monkeypatch):
    """The last gate: a record builder that drops rows (or run, or client) never reaches the file."""
    out = tmp_path / "m.jsonl"
    rec = _rec(out)
    real = MetricsRecorder._record
    for field in ("rows", "edges", "run", "client"):
        monkeypatch.setattr(MetricsRecorder, "_record",
                            lambda self, *a, f=field: {k: v for k, v in real(self, *a).items() if k != f})
        with pytest.raises(MetricRefused, match=field):
            rec("x", 1, "count", **G)
    assert not out.exists()


# ------------------------------------------------------------------ freshness lag (C30b)


def test_freshness_lag_is_sync_end_minus_source_commit_in_minutes():
    # fixed timestamps: commit 2026-10-04T12:00:00Z, sync ended 2m30s later
    commit_s, synced_s = 1791115200.0, 1791115350.0
    assert freshness_lag_minutes(synced_s, commit_s) == 2.5
    assert freshness_lag_minutes(commit_s, commit_s) == 0.0


def test_a_sync_that_ends_before_its_commit_is_refused():
    # hostile: synced time BEFORE the commit time (swapped arguments, or a stale reading)
    with pytest.raises(MetricRefused, match="before the source commit"):
        freshness_lag_minutes(1791115200.0, 1791115350.0)


@pytest.mark.parametrize("bad", [None, "1791115350", float("nan"), True])
def test_lag_refuses_non_numeric_times(bad):
    with pytest.raises(MetricRefused):
        freshness_lag_minutes(bad, 1791115200.0)


def _ts(seconds: int, nanos: int = 0):
    return SimpleNamespace(seconds=seconds, nanos=nanos)  # the SDK's protobuf Timestamp shape


def _fake_w(state="SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE", version=7, commit=_ts(1791115200, 250_000_000),
            end=_ts(1791115380, 0)):
    info = SimpleNamespace(delta_commit_version=version, delta_commit_time=commit)
    status = SimpleNamespace(detailed_state=SimpleNamespace(value=state), last_sync_time=None,
                             last_sync=SimpleNamespace(delta_table_sync_info=info, sync_end_time=end,
                                                       sync_start_time=_ts(1791115300)))
    calls = []

    def get_synced_table(name):
        calls.append(name)
        return SimpleNamespace(status=status)

    return SimpleNamespace(postgres=SimpleNamespace(get_synced_table=get_synced_table)), calls


def _fake_sql(history_ms):
    seen = []

    def sql(stmt):
        seen.append(stmt)
        return [[history_ms]] if history_ms is not None else []

    return sql, seen


def test_table_lag_reads_sync_end_and_commit_and_cross_checks_history():
    w, calls = _fake_w()
    sql, seen = _fake_sql(1791115200250)
    got = lag.table_lag_minutes(w, sql, "cat.v.nodes", "cat.kit.nodes", not_before_s=1791115290.0)
    assert math.isclose(got, (1791115380 - 1791115200.25) / 60.0)
    assert calls == ["synced_tables/cat.v.nodes"]
    assert "DESCRIBE HISTORY cat.kit.nodes" in seen[0] and "version = 7" in seen[0]


def test_table_lag_refuses_a_history_mismatch():
    # hostile: the API names another commit than DESCRIBE HISTORY has at that version
    w, _ = _fake_w()
    sql, _ = _fake_sql(1791115100000)
    with pytest.raises(lag.LagUnavailable, match="DESCRIBE HISTORY"):
        lag.table_lag_minutes(w, sql, "cat.v.nodes", "cat.kit.nodes", not_before_s=1791115290.0)


def test_table_lag_refuses_a_reading_from_before_this_update():
    # hostile: the status still shows the PREVIOUS sync (ended long before tonight's update started)
    w, _ = _fake_w(end=_ts(1791100000))
    sql, _ = _fake_sql(1791115200250)
    with pytest.raises(lag.LagUnavailable, match="before this update"):
        lag.table_lag_minutes(w, sql, "cat.v.nodes", "cat.kit.nodes", not_before_s=1791115290.0)


@pytest.mark.parametrize("kw,match", [({"state": "SYNCED_TABLE_SYNCING"}, "ONLINE_NO_PENDING_UPDATE"),
                                      ({"end": None}, "end time"), ({"commit": None}, "commit time"),
                                      ({"version": None}, "version")])
def test_table_lag_refuses_an_unusable_status(kw, match):
    w, _ = _fake_w(**kw)
    sql, _ = _fake_sql(1791115200250)
    with pytest.raises(lag.LagUnavailable, match=match):
        lag.table_lag_minutes(w, sql, "cat.v.nodes", "cat.kit.nodes", not_before_s=0.0)


def test_table_lag_refuses_a_version_missing_from_history():
    w, _ = _fake_w()
    sql, _ = _fake_sql(None)
    with pytest.raises(lag.LagUnavailable, match="no history row"):
        lag.table_lag_minutes(w, sql, "cat.v.nodes", "cat.kit.nodes", not_before_s=0.0)


# ------------------------------------------------------------------ workspace (in-region) samples


class _FakeConn:
    """Rows per (sql, seed); the bound parameter must be start_node_id."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def execute(self, sql, params):
        self.calls.append((sql, params))
        rows = self.answers[(sql, params["start_node_id"])]
        return SimpleNamespace(fetchall=lambda: rows)


QUERIES = [
    {"metric": "walk.web", "depth": 1, "kind": "degree", "sql": "SQL1", "seeds": ["101", "102"]},
    {"metric": "walk.web", "depth": 4, "kind": "degree", "sql": "SQL4", "seeds": ["101", "102"]},
    {"metric": "lookup.ancestors", "depth": 3, "role": "operator", "sql": "LA", "seeds": ["9"]},
]


def test_time_queries_runs_every_seed_of_every_query_bound():
    conn = _FakeConn({("SQL1", "101"): [(1,)], ("SQL1", "102"): [(1,), (2,)], ("SQL4", "101"): [],
                      ("SQL4", "102"): [(1,)] * 12, ("LA", "9"): [(1,)] * 3})
    out = wl.time_queries(conn, QUERIES)
    assert [c[1] for c in conn.calls] == [{"start_node_id": s} for s in ("101", "102", "101", "102", "9")]
    assert [(o["metric"], o.get("depth"), o["rows"]) for o in out] == [
        ("walk.web", 1, [1, 2]), ("walk.web", 4, [0, 12]), ("lookup.ancestors", 3, [3])]
    assert all(len(o["seconds"]) == len(o["rows"]) and "sql" not in o and "seeds" not in o for o in out)


# The workspace folder shape run_in_region documents, assembled from parts so the source holds no user path.
WORKDIR = "/".join(("", "Workspace", "Users", "me", "lgk_v1_abc123"))


class _FakeBackend:
    """A stand-in workspace: 'runs' the notebook by writing what the real step would write remotely."""

    remote_run_dir = WORKDIR + "/run"

    def __init__(self, remote: Path, samples_text: str, status: str = "success"):
        self.remote, self.samples_text, self.status, self.parameters = remote, samples_text, status, None

    def push_run_folder(self, local_dir: Path) -> None:
        shutil.copytree(local_dir, self.remote, dirs_exist_ok=True)

    def run_notebook(self, parameters):
        self.parameters = parameters
        (self.remote / wl.SAMPLES_FILE).write_text(self.samples_text)
        state = json.loads((self.remote / "run.json").read_text())
        state[wl.STEP_NAME] = {"status": self.status, "error": None if self.status == "success" else "boom"}
        (self.remote / "run.json").write_text(json.dumps(state))

    def pull_run_folder(self, local_dir: Path) -> None:
        shutil.copytree(self.remote, local_dir, dirs_exist_ok=True)


# The job result a real run would send back: unsorted seconds (hostile: p50 must sort), a depth-4 row.
JOB_SAMPLES = [
    {"metric": "walk.web", "depth": 1, "kind": "degree", "seconds": [0.004, 0.001, 0.003, 0.002, 0.005],
     "rows": [3, 3, 3, 3, 3]},
    {"metric": "walk.web", "depth": 4, "kind": "degree", "seconds": [0.020, 0.010], "rows": [12, 12]},
    {"metric": "lookup.ancestors", "depth": 3, "role": "operator", "seconds": [0.0015], "rows": [2]},
]
CONFIG = {"lakebase_project": "proj-test", "branch_id": "t-walks-abc123", "ids": {"a": "1"}}


def _run_fake(tmp_path, samples_text, status="success"):
    backend = _FakeBackend(tmp_path / "remote", samples_text, status)
    run_dir = tmp_path / "local"
    samples = wl.run_in_region(None, CONFIG, [dict(q) for q in QUERIES], tmp_path / "x.whl", run_dir,
                               WORKDIR, backend=backend)
    return samples, backend


def test_workspace_samples_are_parsed_from_the_job_result_and_recorded_as_workspace(tmp_path):
    samples, backend = _run_fake(tmp_path, json.dumps(JOB_SAMPLES))
    # what crossed to the notebook: ONE latency step by module:qualname, and only the minimal config
    assert json.loads(backend.parameters["steps"]) == [["latency", "lgk.workspace_latency:latency_step"]]
    sent = json.loads(backend.parameters["config"])
    assert sent == {"lakebase_project": "proj-test", "branch_id": "t-walks-abc123",
                    "run_folder": backend.remote_run_dir, "latency_queries": QUERIES}
    assert [(s.metric, s.context, s.rows) for s in samples] == [
        ("walk.web", {"depth": 1, "kind": "degree"}, [3] * 5), ("walk.web", {"depth": 4, "kind": "degree"}, [12, 12]),
        ("lookup.ancestors", {"depth": 3, "role": "operator"}, [2])]

    out = tmp_path / "m.jsonl"
    wl.record_workspace_samples(_rec(out), samples, branch="t-walks-abc123", **G)
    got = [(d["metric"], d["client"], d.get("depth"), d.get("kind"), d.get("role"), d["n_samples"], d["value"])
           for d in _lines(out)]
    assert got == [("walk.web.p50", "workspace", 1, "degree", None, 5, pytest.approx(3.0)),
                   ("walk.web.p95", "workspace", 1, "degree", None, 5, pytest.approx(5.0)),
                   ("walk.web.p50", "workspace", 4, "degree", None, 2, pytest.approx(10.0)),
                   ("walk.web.p95", "workspace", 4, "degree", None, 2, pytest.approx(20.0)),
                   ("lookup.ancestors.p50", "workspace", 3, None, "operator", 1, pytest.approx(1.5)),
                   ("lookup.ancestors.p95", "workspace", 3, None, "operator", 1, pytest.approx(1.5))]
    assert all((d["rows"], d["edges"], d["branch"]) == (50, 32, "t-walks-abc123") for d in _lines(out))


def test_a_failed_job_step_is_raised_not_parsed(tmp_path):
    with pytest.raises(RuntimeError, match="failed: boom"):
        _run_fake(tmp_path, json.dumps(JOB_SAMPLES), status=RunStatus.FAILED.value)


@pytest.mark.parametrize("doc", [
    [], {"metric": "w"}, "nope",
    [{"metric": "w", "seconds": [], "rows": []}],                       # no samples
    [{"metric": "w", "seconds": [0.1, 0.2], "rows": [1]}],               # one row count short
    [{"metric": "w", "seconds": [True], "rows": [1]}],                   # hostile: a bool is not a time
    [{"metric": "w", "seconds": [-0.1], "rows": [1]}],
    [{"metric": "", "seconds": [0.1], "rows": [1]}],
    [{"metric": "w", "seconds": [0.1], "rows": [1], "patient": "p"}],    # an unknown key
])
def test_malformed_job_samples_are_refused(tmp_path, doc):
    with pytest.raises(wl.LatencySamplesRefused):
        _run_fake(tmp_path, json.dumps(doc))


def test_non_json_job_samples_are_refused():
    with pytest.raises(wl.LatencySamplesRefused, match="not JSON"):
        wl.parse_samples("{oops")


def test_a_query_without_sql_or_seeds_is_refused_before_any_job(tmp_path):
    with pytest.raises(ValueError, match="seeds"):
        wl.remote_config(CONFIG, [{"metric": "walk.web", "sql": "S", "seeds": []}], tmp_path)


# ------------------------------------------------------------------ prover round 1: boundaries the first tests missed


def test_latency_p95_is_the_nearest_rank_of_twenty_samples(tmp_path):
    # hostile: 20 samples, reversed; p90 (18 ms) and p95 (19 ms) differ only from n > 10
    out = tmp_path / "m.jsonl"
    _rec(out).latencies("w", [i / 1000 for i in range(20, 0, -1)], client="workspace", **G)
    p50, p95 = _lines(out)
    assert (p50["value"], p95["value"]) == (pytest.approx(10.0), pytest.approx(19.0))


@pytest.mark.parametrize("args", [(1791115380.0, True), (1791115380.0, None), (1791115380.0, float("inf"))])
def test_lag_refuses_a_non_numeric_commit_time(args):
    # hostile: a bool commit time (True == 1) would otherwise give a huge positive lag
    with pytest.raises(MetricRefused, match="source_commit_at_s"):
        freshness_lag_minutes(*args)


def test_clock_skew_allowance_accepts_a_few_seconds_but_not_more():
    sql, _ = _fake_sql(1791115200250)
    w, _ = _fake_w(end=_ts(1791115380 - 3))  # ended 3 s before the laptop's night start: clock skew
    assert lag.table_lag_minutes(w, sql, "c.v.n", "c.k.n", not_before_s=1791115380.0) > 0
    w, _ = _fake_w(end=_ts(1791115380 - 10))  # 10 s before: an earlier sync
    with pytest.raises(lag.LagUnavailable, match="before this update"):
        lag.table_lag_minutes(w, sql, "c.v.n", "c.k.n", not_before_s=1791115380.0)


def test_epoch_seconds_reads_every_time_shape_as_utc():
    from datetime import UTC, datetime, timedelta, timezone

    want = 1791072000.0  # 2026-10-04T00:00:00Z
    assert lag.epoch_seconds("2026-10-04T00:00:00Z") == want
    assert lag.epoch_seconds("2026-10-04T02:00:00+02:00") == want
    assert lag.epoch_seconds(datetime(2026, 10, 4, tzinfo=UTC)) == want
    assert lag.epoch_seconds(datetime(2026, 10, 3, 17, tzinfo=timezone(timedelta(hours=-7)))) == want
    assert lag.epoch_seconds(_ts(1791072000, 500_000_000)) == want + 0.5
    assert lag.epoch_seconds(None) is None


def test_lag_uses_the_api_commit_time_not_the_history_time():
    # hostile: history 0.5 s later than the API (inside the tolerance); the operand is delta_commit_time
    w, _ = _fake_w(commit=_ts(1791115200, 0), end=_ts(1791115380, 0))
    sql, _ = _fake_sql(1791115200500)
    assert lag.table_lag_minutes(w, sql, "c.v.n", "c.k.n", not_before_s=0.0) == 3.0


def test_a_negative_row_count_is_refused():
    with pytest.raises(wl.LatencySamplesRefused, match="row count"):
        wl.parse_samples(json.dumps([{"metric": "m", "seconds": [0.1], "rows": [-1]}]))


class _StepConn(_FakeConn):
    def __init__(self, answers, fail=False):
        super().__init__(answers)
        self.fail, self.rolled_back, self.closed = fail, False, False

    def execute(self, sql, params):
        if self.fail:
            raise RuntimeError("statement timeout")
        return super().execute(sql, params)

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def test_latency_step_writes_samples_the_laptop_parses_and_always_closes(tmp_path, monkeypatch):
    import lgk.branch

    conn = _StepConn({("SQL1", "101"): [(1,)] * 4})
    seen = []
    monkeypatch.setattr(lgk.branch, "connect_branch", lambda cfg: seen.append(cfg["branch_id"]) or conn)
    config = {"lakebase_project": "p", "branch_id": "t-walks-1", "run_folder": str(tmp_path),
              "latency_queries": [{"metric": "walk.web", "depth": 1, "kind": "degree", "sql": "SQL1", "seeds": ["101"]}]}
    wl.latency_step(config)
    [s] = wl.parse_samples((tmp_path / wl.SAMPLES_FILE).read_text())
    assert (s.metric, s.context, s.rows, len(s.seconds)) == ("walk.web", {"depth": 1, "kind": "degree"}, [4], 1)
    assert seen == ["t-walks-1"] and conn.rolled_back and conn.closed

    failing = _StepConn({}, fail=True)
    monkeypatch.setattr(lgk.branch, "connect_branch", lambda cfg: failing)
    (tmp_path / wl.SAMPLES_FILE).unlink()
    with pytest.raises(RuntimeError, match="statement timeout"):
        wl.latency_step(config)
    assert failing.rolled_back and failing.closed and not (tmp_path / wl.SAMPLES_FILE).exists()


# ------------------------------------------------------------------ prover round 2


@pytest.fixture
def non_utc_tz(monkeypatch):
    """# hostile: a laptop zone that is not UTC, so a 'Z read as local' bug shows on any runner."""
    import time as _time

    monkeypatch.setenv("TZ", "America/Los_Angeles")
    _time.tzset()
    yield
    monkeypatch.undo()
    _time.tzset()


def test_epoch_seconds_is_zone_safe_on_any_laptop(non_utc_tz):
    from datetime import datetime

    assert lag.epoch_seconds("2026-10-04T00:00:00Z") == 1791072000.0
    for naive in (datetime(2026, 10, 4), "2026-10-04T00:00:00"):
        with pytest.raises(lag.LagUnavailable, match="naive"):
            lag.epoch_seconds(naive)
    with pytest.raises(lag.LagUnavailable):
        lag.epoch_seconds(True)


@pytest.mark.parametrize("state", ["SYNCED_TABLE_ONLINE_PIPELINE_FAILED", "SYNCED_TABLE_ONLINE_UPDATING_PIPELINE_RESOURCES",
                                   "SYNCED_TABLE_ONLINE_TRIGGERED_UPDATE"])
def test_an_online_state_that_is_not_settled_is_refused(state):
    # hostile: ONLINE_* states other than ONLINE_NO_PENDING_UPDATE (an "ONLINE" substring check passes them)
    w, _ = _fake_w(state=state)
    sql, _ = _fake_sql(1791115200250)
    with pytest.raises(lag.LagUnavailable, match="ONLINE_NO_PENDING_UPDATE"):
        lag.table_lag_minutes(w, sql, "c.v.n", "c.k.n", not_before_s=0.0)


def test_the_skew_allowance_and_tolerance_edges_are_exact():
    sql, _ = _fake_sql(1791115200000)
    w, _ = _fake_w(commit=_ts(1791115200), end=_ts(1791115380 - 5))  # exactly the allowed clock skew early: accepted
    assert lag.table_lag_minutes(w, sql, "c.v.n", "c.k.n", not_before_s=1791115380.0) == pytest.approx(175 / 60)
    w, _ = _fake_w(commit=_ts(1791115200), end=_ts(1791115380 - 6))  # 6 s early: refused
    with pytest.raises(lag.LagUnavailable, match="before this update"):
        lag.table_lag_minutes(w, sql, "c.v.n", "c.k.n", not_before_s=1791115380.0)
    w, _ = _fake_w(commit=_ts(1791115200), end=_ts(1791115380))
    sql_late, _ = _fake_sql(1791115201000)  # history exactly 1 s later: accepted
    assert lag.table_lag_minutes(w, sql_late, "c.v.n", "c.k.n", not_before_s=0.0) == 3.0
    sql_off, _ = _fake_sql(1791115203000)  # hostile: 3 s apart, another commit
    with pytest.raises(lag.LagUnavailable, match="DESCRIBE HISTORY"):
        lag.table_lag_minutes(w, sql_off, "c.v.n", "c.k.n", not_before_s=0.0)


class _SlowConn:
    """A fake clock: execute costs 0.25 s and fetchall 0.5 s, so the timed span must hold both, in seconds."""

    def __init__(self, clock):
        self.clock = clock

    def execute(self, sql, params):
        self.clock[0] += 0.25

        def fetchall():
            self.clock[0] += 0.5
            return [(1,)]

        return SimpleNamespace(fetchall=fetchall)


def test_time_queries_times_execute_and_fetch_in_seconds(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(wl.time, "perf_counter", lambda: clock[0])
    [out] = wl.time_queries(_SlowConn(clock), [{"metric": "m", "sql": "S", "seeds": ["1", "2"]}])
    assert out["seconds"] == [pytest.approx(0.75), pytest.approx(0.75)] and out["rows"] == [1, 1]


@pytest.mark.parametrize("raw", ['[{"metric": "m", "seconds": [Infinity], "rows": [1]}]',
                                 '[{"metric": "m", "seconds": [NaN], "rows": [1]}]',
                                 '[{"metric": "m", "seconds": [0.1], "rows": [true]}]'])
def test_non_finite_seconds_and_bool_rows_are_refused(raw):
    with pytest.raises(wl.LatencySamplesRefused):
        wl.parse_samples(raw)
