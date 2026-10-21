"""Offline tests for the nightly refresh: job spec, freshness guard, sync modes.

No cloud calls. Fakes stand in for the Databricks SDK *API objects* only; their return values
are the SDK's own dataclasses, shaped like the live responses spikes 3 and 5 recorded
(`status.last_sync_time` is None; the version lives in last_sync.delta_table_sync_info).
"""

from pathlib import Path
from types import SimpleNamespace
import json
import re
from typing import Any, Dict, List, Sequence, Tuple

import pytest
from databricks.sdk.service import catalog as uc
from databricks.sdk.service import postgres as pg

from lgk.refresh import freshness, job, modes, sync
from lgk.steps import provision

CAT = "cat"
# a synthetic user, assembled at run time; the source holds only the template form
USER = "me"
WS_HOME = f"/Workspace/Users/{USER}"
USER_HOME = f"/Users/{USER}"


def _config(**extra: Any) -> Dict[str, Any]:
    cfg = {
        "graph_name": "provider_graph",
        "lakebase_project": "proj",
        "lakebase_catalog": CAT,
        "kit_schema": "lgk_v1_abc123",
        "served_schema": "lgk_served_abc123",
        "branch_id": "t-x",
        "run_folder": f"{WS_HOME}/lgk_v1_abc123/run",
        "warehouse_id": "wh",
        "version": 1,
    }
    cfg.update(extra)
    return cfg


def _enriched(tmp_path, **extra: Any) -> Dict[str, Any]:
    """A config whose run folder enriched (enrich.yaml present), so the workspace job enriches nightly."""
    (tmp_path / "enrich.yaml").write_text("methods: [degree]\n")
    return _config(run_folder=str(tmp_path), **extra)


# ----------------------------------------------------------------------- job spec -------
JOB_KW = dict(notebook_path=f"{WS_HOME}/lgk_v1_abc123/lgk_refresh_task",
              wheel_path=f"{WS_HOME}/lgk_v1_abc123/lgk-0.1.0-py3-none-any.whl")


class TestRefreshJobSpec:
    def test_tasks_chain_sdp_enrich_sync_reapply_in_one_job(self, tmp_path):
        tasks = [t.as_dict() for t in job.refresh_job_tasks(_enriched(tmp_path, sdp_pipeline_id="p-1"), **JOB_KW)]
        deps = {t["task_key"]: [d["task_key"] for d in t.get("depends_on", [])] for t in tasks}
        # hostile: a lazy "everything depends on the pipeline" fan-out has the same 4 keys but
        # lets reapply run before (or during) the sync. The chain must be exact.
        # owner decision (nightly stamp step): a chain with no kit build runs "stamp" after the pipeline
        assert deps == {"sdp_pipeline": [], "stamp": ["sdp_pipeline"], "enrich": ["stamp"],
                        "sync": ["enrich"], "reapply": ["sync"]}
        by_key = {t["task_key"]: t for t in tasks}
        assert by_key["sdp_pipeline"]["pipeline_task"]["pipeline_id"] == "p-1"
        for key in ("stamp", "enrich", "sync", "reapply"):
            params = by_key[key]["notebook_task"]["base_parameters"]
            assert params["task"] == key
            assert params["wheel"] == JOB_KW["wheel_path"]
            assert by_key[key]["notebook_task"]["notebook_path"] == JOB_KW["notebook_path"]

    def test_a_bad_graph_name_never_reaches_a_job_tag(self):
        from lgk import manifest as mf

        created: List[Any] = []
        w = SimpleNamespace(jobs=SimpleNamespace(create=lambda **kw: created.append(kw)))
        with pytest.raises(ValueError):
            mf.job_tags("bad name; drop")
        with pytest.raises(ValueError):
            job.create_refresh_job(_config(sdp_pipeline_id="p-1", graph_name="Bad Name"), w=w, **JOB_KW)
        assert created == []

    @pytest.mark.parametrize("no_pipeline", [{}, {"sdp_pipeline_id": None}, {"sdp_pipeline_id": ""}])
    def test_chain_without_sdp_pipeline_runs_kit_build_first(self, tmp_path, no_pipeline):
        """No operator pipeline -> the kit's own SQL build is the first task, every night."""
        # hostile: no sdp_pipeline_id AND no build_mode (the default most graphs have); today's
        # code refused this config, and a lazy fix that only drops the refusal yields enrich first.
        # hostile (prover): the key present but empty (None / "") is "no pipeline" too, never "no build"
        tasks = [t.as_dict() for t in job.refresh_job_tasks(_enriched(tmp_path, **no_pipeline), **JOB_KW)]
        assert [t["task_key"] for t in tasks] == ["build", "enrich", "sync", "reapply"]
        assert [[d["task_key"] for d in t.get("depends_on", [])] for t in tasks] == [
            [], ["build"], ["enrich"], ["sync"]]
        for t in tasks:  # every task, build included, is a notebook task running itself
            assert t["notebook_task"]["base_parameters"]["task"] == t["task_key"]
            assert "pipeline_task" not in t
        assert job.refresh_chain(_config(**no_pipeline), enriched=True) == ["build", "enrich", "sync", "reapply"]
        assert job.refresh_chain(_config(build_mode="sql"), enriched=False) == ["build", "sync", "reapply"]

    def test_chain_with_sdp_pipeline_and_no_build_mode_is_unchanged(self, tmp_path):
        """Golden: today's four tasks, byte for byte, when the operator's pipeline is set (an enriched run)."""
        config = _enriched(tmp_path, sdp_pipeline_id="p-1")
        nb = lambda key: {"notebook_path": JOB_KW["notebook_path"], "source": "WORKSPACE",  # noqa: E731
                          "base_parameters": {"task": key, "wheel": JOB_KW["wheel_path"],
                                              "config": json.dumps(config)}}
        # P2-runtime (B): every notebook task names the pinned job environment; the pipeline task has none
        golden = [  # owner decision (nightly stamp step): "stamp" between the pipeline and the rest
            {"task_key": "sdp_pipeline", "pipeline_task": {"pipeline_id": "p-1"}},
            {"task_key": "stamp", "depends_on": [{"task_key": "sdp_pipeline"}], "environment_key": "lgk",
             "notebook_task": nb("stamp")},
            {"task_key": "enrich", "depends_on": [{"task_key": "stamp"}], "environment_key": "lgk",
             "notebook_task": nb("enrich")},
            {"task_key": "sync", "depends_on": [{"task_key": "enrich"}], "environment_key": "lgk",
             "notebook_task": nb("sync")},
            {"task_key": "reapply", "depends_on": [{"task_key": "sync"}], "environment_key": "lgk",
             "notebook_task": nb("reapply")},
        ]
        assert [t.as_dict() for t in job.refresh_job_tasks(config, **JOB_KW)] == golden
        assert job.refresh_chain(config, enriched=False) == ["sdp_pipeline", "stamp", "sync", "reapply"]

    @pytest.mark.parametrize("extra", [{}, {"sdp_pipeline_id": "p-1"}, {"sdp_pipeline_id": "p-1", "build_mode": "sql"}])
    def test_a_run_without_enrich_yaml_makes_a_job_without_enrich(self, tmp_path, extra):
        """enrich stops with no enrich.yaml: a job that ran it would fail every night."""
        # hostile: the run folder exists and holds the other run files, only enrich.yaml is absent
        (tmp_path / "graph_mapping.yaml").write_text("x")
        (tmp_path / "enrich_report.json").write_text("{}")  # a name-alike that is not the choice
        (tmp_path / "enrich.yml").write_text("methods: [degree]\n")  # enrich reads only enrich.yaml
        tasks = [t.as_dict() for t in job.refresh_job_tasks(_config(run_folder=str(tmp_path), **extra), **JOB_KW)]
        keys = [t["task_key"] for t in tasks]
        assert "enrich" not in keys
        assert keys[-2:] == ["sync", "reapply"] and len(keys) == len(set(keys))
        assert [[d["task_key"] for d in t.get("depends_on", [])] for t in tasks] == [[], *[[k] for k in keys[:-1]]]

    @pytest.mark.parametrize("folder", ["absent", None, ""])
    def test_a_config_without_a_run_folder_makes_a_job_without_enrich(self, tmp_path, monkeypatch, folder):
        # hostile: the working directory HAS an enrich.yaml; only the run folder may decide
        (tmp_path / "enrich.yaml").write_text("methods: [degree]\n")
        monkeypatch.chdir(tmp_path)
        cfg = _config()
        if folder == "absent":
            del cfg["run_folder"]
        else:
            cfg["run_folder"] = folder
        assert [t.task_key for t in job.refresh_job_tasks(cfg, **JOB_KW)] == ["build", "sync", "reapply"]

    @pytest.mark.parametrize("extra", [{}, {"sdp_pipeline_id": "p-1"}])
    def test_a_run_with_enrich_yaml_makes_the_enrich_task(self, tmp_path, extra):
        keys = [t.task_key for t in job.refresh_job_tasks(_enriched(tmp_path, **extra), **JOB_KW)]
        # the task before enrich rebuilds or certifies the kit tables: build, or (no kit build) stamp
        before = "stamp" if extra else "build"
        assert keys[keys.index("enrich") - 1:] == [before, "enrich", "sync", "reapply"]

    def test_explicit_build_mode_sql_with_pipeline_runs_both_in_order(self, tmp_path):
        # hostile: both sources of a rebuild at once; the kit build must run AFTER the operator's
        # pipeline refreshed gold (before it, it would rebuild from yesterday's gold).
        tasks = [t.as_dict() for t in job.refresh_job_tasks(_enriched(tmp_path, sdp_pipeline_id="p-1", build_mode="sql"),
                                                            **JOB_KW)]
        assert [t["task_key"] for t in tasks] == ["sdp_pipeline", "build", "enrich", "sync", "reapply"]
        assert [[d["task_key"] for d in t.get("depends_on", [])] for t in tasks] == [
            [], ["sdp_pipeline"], ["build"], ["enrich"], ["sync"]]
        assert tasks[0]["pipeline_task"] == {"pipeline_id": "p-1"}
        assert tasks[1]["notebook_task"]["base_parameters"]["task"] == "build"

    @pytest.mark.parametrize("bad", ["Pipeline", "none; drop", "", "SQL", "pipeline", 1])
    @pytest.mark.parametrize("pipeline", [None, "p-1"])
    def test_unknown_build_mode_is_refused(self, bad, pipeline):
        # hostile: near-misses of the one allowed value ("SQL", a not-yet-built "pipeline"),
        # an injection-shaped string, the empty string and a non-string, with and without a pipeline
        extra = {"build_mode": bad, **({"sdp_pipeline_id": pipeline} if pipeline else {})}
        created: List[Any] = []
        w = SimpleNamespace(jobs=SimpleNamespace(create=lambda **kw: created.append(kw)))
        with pytest.raises(ValueError, match=r"build_mode.*'sql'"):
            job.refresh_chain(_config(**extra), enriched=True)
        with pytest.raises(ValueError, match="build_mode"):
            job.create_refresh_job(_config(**extra), w=w, **JOB_KW)
        assert created == []  # refused before any task (or job) is made

    def test_create_refresh_job_submits_the_chain_under_the_run_name(self, tmp_path):
        from lgk import manifest as mf
        from tests.refresh_support import fake_client, seed_run_folder

        w, calls = fake_client(job_id=42)
        seed_run_folder(tmp_path)
        cfg = _enriched(tmp_path, sdp_pipeline_id="p-1")
        job_id = job.create_refresh_job(cfg, w=w, **JOB_KW)
        assert job_id == 42
        assert calls[0]["name"] == "lgk_refresh_lgk_v1_abc123"
        # was list(job.TASK_ORDER); TASK_ORDER gained "build", the expected chain did not
        # owner decision (nightly stamp step): TASK_ORDER gained "stamp", and this chain (no kit build) runs it
        assert [t.task_key for t in calls[0]["tasks"]] == ["sdp_pipeline", "stamp", "enrich", "sync", "reapply"]
        assert calls[0]["max_concurrent_runs"] == 1  # two nightly runs must never overlap
        assert calls[0]["tags"] == {"lgk_graph": "provider_graph"}  # teardown's ownership check
        # recorded for teardown, with the branch it syncs (for the reaper)
        jobs = [(e.object_id, e.metadata) for e in mf.load_run_manifest(cfg).entries
                if e.object_type == mf.ObjectType.JOB]
        assert jobs == [("42", {"kind": "refresh", "branch": "projects/proj/branches/t-x",
                                "run_folder": f"{WS_HOME}/lgk_v1_abc123/runs/provider_graph"})]


# --------------------------------------------------------------------- freshness guard ------
def _synced(version, last_sync_time=None) -> pg.SyncedTable:
    info = pg.DeltaTableSyncInfo(delta_commit_version=version) if version is not None else None
    return pg.SyncedTable(status=pg.SyncedTableSyncedTableStatus(
        last_sync_time=last_sync_time,
        last_sync=pg.SyncedTablePosition(delta_table_sync_info=info) if info else None,
    ))


def _fake_w(versions: Dict[str, Any]):
    def get_synced_table(name):
        assert name.startswith("synced_tables/"), name
        v = versions[name.split(".")[-1]]
        return v if isinstance(v, pg.SyncedTable) else _synced(v)
    return SimpleNamespace(postgres=SimpleNamespace(get_synced_table=get_synced_table))


def _row(version: int, operation: str, metrics: Dict[str, str]) -> List[Any]:
    """One (version, operation, operationMetrics) row as the statement API returns it (strings)."""
    return [str(version), operation, json.dumps(metrics)]


WRITE = {"numOutputRows": "3", "numFiles": "1"}


def _history(current: int, *, top: Sequence[Tuple[str, Dict[str, str]]] = ()) -> List[List[Any]]:
    """Newest first: optional `top` commits above a WRITE history 0..current-len(top)."""
    rows = [_row(current - i, op, m) for i, (op, m) in enumerate(top)]
    base = current - len(top)
    rows += [_row(v, "WRITE" if v else "CREATE TABLE", WRITE if v else {}) for v in range(base, -1, -1)]
    return rows


def _fake_sql(histories: Dict[str, Any]):
    def sql(statement: str):
        m = re.fullmatch(r"SELECT version, operation, operationMetrics FROM \(DESCRIBE HISTORY (\S+)\)", statement)
        assert m, statement
        h = histories[m.group(1).split(".")[-1]]
        return h if isinstance(h, list) else _history(h)
    return sql


PAIRS = [("cat.lgk_v1_abc123_v1.nodes", "cat.lgk_v1_abc123.nodes"),
         ("cat.lgk_v1_abc123_v1.edges", "cat.lgk_v1_abc123.edges")]
OPTIMIZE = ("OPTIMIZE", {"numAddedFiles": "1", "numRemovedFiles": "2"})
PROPS = ("SET TBLPROPERTIES", {})


class TestFreshnessGuard:
    def test_equal_versions_pass(self):
        freshness.assert_fresh(_fake_w({"nodes": 7, "edges": 3}), _fake_sql({"nodes": 7, "edges": 3}), PAIRS)

    def test_no_data_commits_after_the_sync_are_fresh(self):
        # hostile: newest-first history whose top rows are OPTIMIZE + SET TBLPROPERTIES (+ a
        # zero-row WRITE) after synced version 7. Equality (current 10 != 7) would fail the job.
        nodes = _history(10, top=[OPTIMIZE, PROPS, ("WRITE", {"numOutputRows": "0"})])
        freshness.assert_fresh(_fake_w({"nodes": 7, "edges": 3}), _fake_sql({"nodes": nodes, "edges": 3}), PAIRS)

    def test_write_after_no_data_commits_is_stale(self):
        # hostile: the data WRITE (v9) sits under a newer OPTIMIZE (v10); only the newest commit
        # is no-data, so a "look at the top row only" guard would pass it.
        nodes = _history(10, top=[OPTIMIZE, ("WRITE", WRITE)])
        with pytest.raises(freshness.StaleSyncError, match=r"nodes.*synced version 7.*data change at version 9"):
            freshness.assert_fresh(_fake_w({"nodes": 7, "edges": 3}), _fake_sql({"nodes": nodes, "edges": 3}), PAIRS)

    def test_unknown_operation_without_row_metrics_counts_as_data(self):
        nodes = _history(8, top=[("RESTORE", {"numRestoredFiles": "2"})])
        with pytest.raises(freshness.StaleSyncError, match="nodes"):
            freshness.assert_fresh(_fake_w({"nodes": 7, "edges": 3}), _fake_sql({"nodes": nodes, "edges": 3}), PAIRS)

    def test_synced_version_ahead_of_source_is_stale(self):
        with pytest.raises(freshness.StaleSyncError, match="edges"):
            freshness.assert_fresh(_fake_w({"nodes": 7, "edges": 9}), _fake_sql({"nodes": 7, "edges": 3}), PAIRS)

    def test_new_source_row_without_sync_fails_loudly(self):
        # hostile: last_sync_time is None exactly as live. A guard that reads it never
        # fires; only the version comparison catches the unsynced write (source 8, synced 7).
        w = _fake_w({"nodes": _synced(7, last_sync_time=None), "edges": 3})
        with pytest.raises(freshness.StaleSyncError, match=r"nodes.*synced version 7.*data change at version 8"):
            freshness.assert_fresh(w, _fake_sql({"nodes": 8, "edges": 3}), PAIRS)

    def test_stale_version_with_a_recent_last_sync_time_still_fails(self):
        # hostile: a populated, recent last_sync_time must not mask a version mismatch.
        w = _fake_w({"nodes": 7, "edges": _synced(2, last_sync_time="2099-01-01T00:00:00Z")})
        with pytest.raises(freshness.StaleSyncError, match="edges"):
            freshness.assert_fresh(w, _fake_sql({"nodes": 7, "edges": 3}), PAIRS)

    def test_missing_sync_info_fails(self, clock):
        with pytest.raises(freshness.StaleSyncError, match="no synced version"):
            freshness.assert_fresh(_fake_w({"nodes": None, "edges": 3}), _fake_sql({"nodes": 0, "edges": 3}), PAIRS)

    def test_every_stale_table_is_named(self):
        with pytest.raises(freshness.StaleSyncError) as e:
            freshness.assert_fresh(_fake_w({"nodes": 1, "edges": 1}), _fake_sql({"nodes": 2, "edges": 2}), PAIRS)
        assert "nodes" in str(e.value) and "edges" in str(e.value)

    def test_check_freshness_requires_a_warehouse(self):
        cfg = _config()
        del cfg["warehouse_id"]
        with pytest.raises(ValueError, match="warehouse_id"):
            freshness.check_freshness(cfg)


class _Clock:
    """A fake monotonic clock: sleep advances it, nothing waits in real time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: List[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(freshness, "time", SimpleNamespace(monotonic=c.monotonic, sleep=c.sleep), raising=False)
    return c


def _seq_w(*reads: Any):
    """get_synced_table returns `reads` in order (a version, None, or a SyncedTable), then the last again."""
    seen: List[str] = []

    def get_synced_table(name):
        seen.append(name)
        v = reads[min(len(seen), len(reads)) - 1]
        if isinstance(v, BaseException):
            raise v
        return v if isinstance(v, pg.SyncedTable) else _synced(v)
    return SimpleNamespace(postgres=SimpleNamespace(get_synced_table=get_synced_table)), seen


DEST = "cat.lgk_v1_abc123_v1.nodes"
NO_INFO = pg.SyncedTable(status=pg.SyncedTableSyncedTableStatus(
    last_sync=pg.SyncedTablePosition(delta_table_sync_info=None)))


class TestSyncedVersionWait:
    """Live (rehearsal 10d): reapply's guard read a version, seconds later the same status had no
    last_sync. synced_version reads again within a stated limit, then fails closed."""

    def test_the_limit_and_the_interval_are_stated(self):
        assert freshness.SYNCED_VERSION_WAIT_S == 120
        assert freshness.SYNCED_VERSION_POLL_S == 5
        assert "120 s" in (freshness.synced_version.__doc__ or "")

    def test_a_version_on_the_first_read_returns_at_once(self, clock):
        w, seen = _seq_w(7)
        assert freshness.synced_version(w, DEST) == 7
        assert seen == [f"synced_tables/{DEST}"] and clock.slept == []

    def test_version_zero_is_a_version(self, clock):
        # hostile: 0 is falsy; a truthiness check would wait out the limit and fail a fresh table
        w, seen = _seq_w(0, 5)
        assert freshness.synced_version(w, DEST) == 0
        assert len(seen) == 1 and clock.slept == []

    def test_a_status_without_last_sync_is_read_again_until_it_reports(self, clock):
        w, seen = _seq_w(None, None, 11)
        assert freshness.synced_version(w, DEST) == 11
        assert len(seen) == 3 and clock.slept == [5, 5]

    def test_last_sync_without_sync_info_is_read_again(self, clock):
        w, seen = _seq_w(NO_INFO, 4)
        assert freshness.synced_version(w, DEST) == 4
        assert len(seen) == 2 and clock.slept == [5]

    def test_no_version_within_the_limit_is_none(self, clock):
        w, seen = _seq_w(None)
        assert freshness.synced_version(w, DEST) is None
        assert 120 <= sum(clock.slept) <= 125 and all(0 < s <= 5 for s in clock.slept)
        assert len(seen) == len(clock.slept) + 1  # a last read after the last sleep

    def test_a_version_on_the_last_read_inside_the_limit_counts(self, clock):
        # hostile: the version arrives on the read at exactly 120 s; stopping before it fails a fresh table
        w, seen = _seq_w(*([None] * 24), 9)
        assert freshness.synced_version(w, DEST) == 9
        assert sum(clock.slept) == 120 and len(seen) == 25

    def test_the_last_sleep_ends_at_the_deadline_and_a_read_there_counts(self, clock, monkeypatch):
        # hostile: a limit that is not a multiple of the interval; a full last sleep passes the
        # deadline, and stopping when less than one interval is left skips the read at it
        monkeypatch.setattr(freshness, "SYNCED_VERSION_WAIT_S", 7)
        w, seen = _seq_w(None, None, 3)
        assert freshness.synced_version(w, DEST) == 3
        assert clock.slept == [5, 2] and len(seen) == 3

    def test_the_guard_fails_closed_after_the_limit_and_says_so(self, clock):
        with pytest.raises(freshness.StaleSyncError,
                           match=r"nodes: no synced version after 120 s \(source version 0\)"):
            freshness.assert_fresh(_fake_w({"nodes": None, "edges": 3}), _fake_sql({"nodes": 0, "edges": 3}), PAIRS)
        assert clock.now >= 120

    def test_a_read_error_is_not_waited_out(self, clock):
        w, seen = _seq_w(RuntimeError("synced table not found"), 7)
        with pytest.raises(RuntimeError, match="not found"):
            freshness.synced_version(w, DEST)
        assert len(seen) == 1 and clock.slept == []


# ------------------------------------------------------------ sync modes ---------------
def _tables_w(table_type: uc.TableType):
    seen: List[str] = []
    def get(full_name):
        seen.append(full_name)
        return uc.TableInfo(full_name=full_name, table_type=table_type)
    return SimpleNamespace(tables=SimpleNamespace(get=get)), seen


POLICY = pg.SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy


class TestSyncModes:
    def test_first_run_is_snapshot_for_every_table(self):
        w, seen = _tables_w(uc.TableType.MANAGED)
        for t in ("nodes", "edges", "node_id_map"):
            assert provision._scheduling_policy(w, _config(), t, f"{CAT}.k.{t}") == POLICY.SNAPSHOT
        assert seen == []  # snapshot needs no source check

    def test_explicit_snapshot_is_snapshot_for_every_table(self):
        w, seen = _tables_w(uc.TableType.MANAGED)
        for t in ("nodes", "edges", "node_id_map"):
            assert provision._scheduling_policy(w, _config(sync_mode="snapshot"), t, f"{CAT}.k.{t}") == POLICY.SNAPSHOT
        assert seen == []

    def test_triggered_mode_is_refused_in_v1_even_over_a_streaming_table(self):
        # v1 is snapshot only: the pin that used to accept triggered here. hostile: a STREAMING_TABLE source,
        # the one shape the V1.1 path accepts today, so only the v1 rule can refuse it.
        w, seen = _tables_w(uc.TableType.STREAMING_TABLE)
        for t in ("nodes", "edges", "node_id_map"):
            with pytest.raises(modes.UnsupportedSyncModeError, match="Path B"):
                provision._scheduling_policy(w, _config(sync_mode="triggered"), t, f"c.k.{t}")
        assert seen == []  # refused before the source is even read

    @pytest.mark.parametrize("bad", [uc.TableType.MATERIALIZED_VIEW, uc.TableType.MANAGED])
    def test_triggered_from_a_non_streaming_source_is_refused_before_the_source_check(self, bad):
        # the v1 rule refuses before the streaming-source check runs (no UC read at all)
        w, seen = _tables_w(bad)
        with pytest.raises(modes.UnsupportedSyncModeError):
            provision._scheduling_policy(w, _config(sync_mode="triggered"), "nodes", "c.k.nodes_mv")
        assert seen == []

    @pytest.mark.parametrize("bad", [uc.TableType.MATERIALIZED_VIEW, uc.TableType.MANAGED])
    def test_the_v1_1_source_check_still_refuses_a_non_streaming_source(self, bad):
        # kept for the V1.1 path: hostile: the MV is a real Delta-backed source that a triggered sync
        # accepts at create time and only fails later as ONLINE_PIPELINE_FAILED (seen live).
        w, seen = _tables_w(bad)
        with pytest.raises(modes.TriggeredSourceError, match="(?s)streaming table.*ONLINE_PIPELINE_FAILED"):
            modes.assert_streaming_source(w, "c.k.nodes_mv")
        assert seen == ["c.k.nodes_mv"]

    def test_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="sync_mode"):
            modes.policy_name(_config(sync_mode="continuous"), "nodes")


# ------------------------------------------------------- v1 supports snapshot only ----------
# hostile rows: every value a lazy check lets through. A deny-list of "triggered" misses
# "continuous" and an unknown name; case and spaces catch a check that normalises first.
NOT_SNAPSHOT = ["triggered", "continuous", "incremental", "TRIGGERED", "Triggered", " triggered "]
# hostile rows: an empty value (YAML `sync_mode:` gives None). Owner ruling: refused, never read as unset; the
# message says remove the key for snapshot, or write "snapshot". "" catches `config.get("sync_mode") or "snapshot"`.
EMPTY = [None, ""]
# hostile rows: right word, wrong case. Refused, with the lower-case spelling (lgk.ident's rule).
WRONG_CASE = ["Snapshot", "SNAPSHOT"]


@pytest.mark.parametrize("mode", NOT_SNAPSHOT)
def test_a_non_snapshot_mode_is_refused_with_the_reason(mode):
    with pytest.raises(modes.UnsupportedSyncModeError) as e:
        modes.require_v1_sync_mode(_config(sync_mode=mode))
    msg = str(e.value)
    assert "sync_mode" in msg and repr(mode) in msg and "'snapshot'" in msg
    # the reason: a pipeline-owned streaming table cannot take the kit's label; Path B is not v1
    assert "streaming table" in msg and "label" in msg and "Path B" in msg
    # hostile ("Triggered", "TRIGGERED"): only a supported mode is ever offered as the fix
    assert "write 'triggered'" not in msg and "write 'continuous'" not in msg


@pytest.mark.parametrize("mode", WRONG_CASE)
def test_snapshot_in_the_wrong_case_is_refused_with_the_lower_case_spelling(mode):
    with pytest.raises(modes.UnsupportedSyncModeError) as e:
        modes.require_v1_sync_mode(_config(sync_mode=mode))
    msg = str(e.value)
    assert "sync_mode" in msg and repr(mode) in msg
    assert "lower case" in msg and "write 'snapshot'" in msg  # the lgk.ident form


# hostile rows: a YAML or JSON config can hold a number, a bool or a list. The check must refuse them
# with its own error, never crash on a str method (AttributeError would escape preflight's report).
NOT_A_STRING = [1, True, ["snapshot"], {"mode": "snapshot"}]


@pytest.mark.parametrize("mode", NOT_A_STRING, ids=["int", "bool", "list", "dict"])
def test_a_non_string_mode_is_refused_with_the_reason(mode):
    with pytest.raises(modes.UnsupportedSyncModeError) as e:
        modes.require_v1_sync_mode(_config(sync_mode=mode))
    msg = str(e.value)
    assert "sync_mode" in msg and repr(mode) in msg and "'snapshot'" in msg
    assert "write 'snapshot'" not in msg  # only the right word in the wrong case gets the spelling fix


@pytest.mark.parametrize("mode", EMPTY, ids=["none", "empty-string"])
def test_an_empty_mode_is_refused_remove_the_key_or_write_snapshot(mode):
    with pytest.raises(modes.UnsupportedSyncModeError) as e:
        modes.require_v1_sync_mode(_config(sync_mode=mode))
    msg = str(e.value)
    assert "sync_mode" in msg and repr(mode) in msg
    assert "remove the key" in msg and "write 'snapshot'" in msg
    # the owner's sentence, whole; hostile: "remove the key and write 'snapshot'" holds both phrases and
    # tells the operator two opposite things
    assert "remove the key for snapshot, or write 'snapshot'" in msg.lower()
    assert msg.lower().count("remove the key") == 1


def test_snapshot_and_unset_are_the_v1_mode():
    assert modes.require_v1_sync_mode(_config()) == "snapshot"  # unset means snapshot
    assert modes.require_v1_sync_mode(_config(sync_mode="snapshot")) == "snapshot"
    assert modes.V1_SYNC_MODES == ("snapshot",)


def _preflight_config(tmp_path, **kw) -> Dict[str, Any]:
    from tests.preflight_support import OPERATOR, FakePg, FakeWorkspace

    return {"graph_name": "g", "run_folder": str(tmp_path), "lakebase_project": "proj",
            "branch_id": "production", "lakebase_catalog": "cat", "kit_schema": "lgk_g",
            "sync_owner_user": OPERATOR, "workspace_client": FakeWorkspace(), "pg_connect": FakePg().connect, **kw}


@pytest.mark.parametrize("mode", NOT_SNAPSHOT + WRONG_CASE + EMPTY)
def test_preflight_allows_only_snapshot_in_v1_before_any_step_builds(tmp_path, mode):
    from lgk.steps.preflight import PreflightBlocked, preflight

    # every other check passes (full grants), so only the sync mode can block
    with pytest.raises(PreflightBlocked, match="sync mode"):
        preflight(_preflight_config(tmp_path, sync_mode=mode))
    record = json.loads((tmp_path / "preflight_result.json").read_text())
    (check,) = [c for c in record["checks"] if c["name"] == "sync mode"]
    assert check["status"] == "missing" and repr(mode) in check["detail"] and "snapshot" in check["fix"]
    if mode in EMPTY:
        assert "remove the key" in check["detail"] and "write 'snapshot'" in check["detail"]
    elif mode in WRONG_CASE:
        assert "write 'snapshot'" in check["detail"]
    else:
        assert "streaming table" in check["detail"] and "Path B" in check["detail"]
    assert record["status"] == "blocked"  # a blocked preflight raises, so no later step runs


@pytest.mark.parametrize("extra", [{}, {"sync_mode": "snapshot"}], ids=["unset", "snapshot"])
def test_preflight_passes_snapshot_and_unset(tmp_path, extra):
    from lgk.steps.preflight import preflight

    result = preflight(_preflight_config(tmp_path, **extra))
    assert result.status == "pass", result.reason
    (check,) = [c for c in result.checks if c["name"] == "sync mode"]
    assert check["status"] == "ok"


# Everything provision does, in order, as names a fake records (hostile: one record per call, so a
# refusal placed after any of them shows up as a non-empty list).
PROVISION_CALLS = ("required_run_folder", "require_clear", "require_sync_owner", "_refuse_existing_version",
                   "_refuse_unbound_build", "_create_synced_tables", "_mark_pg_version_schema",
                   "_create_version_indexes", "_lock_version_from_reader", "_stage_version_functions",
                   "_prove_reader_isolated")


def _recording_provision(monkeypatch, tmp_path) -> List[str]:
    calls: List[str] = []
    for name in PROVISION_CALLS:
        monkeypatch.setattr(provision, name, lambda *a, _n=name, **k: calls.append(_n) or str(tmp_path))
    monkeypatch.setattr(provision.binding, "stamp_id", lambda clear: calls.append("stamp_id") or "stamp")
    monkeypatch.setattr(provision.binding, "record_version", lambda *a, **k: calls.append("record_version"))
    return calls


@pytest.mark.parametrize("mode", NOT_SNAPSHOT + WRONG_CASE + EMPTY)
def test_provision_refuses_every_non_snapshot_mode_when_called_directly(monkeypatch, tmp_path, mode):
    # hostile: provision called without preflight, so its own check is the only guard
    calls = _recording_provision(monkeypatch, tmp_path)
    with pytest.raises(modes.UnsupportedSyncModeError):
        provision.provision(_config(run_folder=str(tmp_path), sync_mode=mode))
    assert calls == []  # refused before the privacy gate, the owner check and every build call


def test_provision_refuses_triggered_over_a_streaming_table_before_create(monkeypatch):
    # hostile: a STREAMING_TABLE source, which the V1.1 path accepts today, reached through the
    # function that builds the synced-table spec: no source read, no create
    w, seen = _tables_w(uc.TableType.STREAMING_TABLE)
    created: List[str] = []
    w.postgres = SimpleNamespace(create_synced_table=lambda **k: created.append(k["synced_table_id"]))
    with pytest.raises(modes.UnsupportedSyncModeError):
        provision._create_one_synced_table(w, _config(sync_mode="triggered"), "nodes", "c.k.nodes",
                                           "c.v.nodes", ["node_id"], "projects/p/branches/b")
    assert seen == [] and created == []


@pytest.mark.parametrize("extra", [{}, {"sync_mode": "snapshot"}], ids=["unset", "snapshot"])
def test_snapshot_still_provisions(monkeypatch, tmp_path, extra):
    calls = _recording_provision(monkeypatch, tmp_path)
    assert provision.provision(_config(run_folder=str(tmp_path), **extra)) is True
    # over-refusal check: the whole supported path still runs, in order, and records the version last
    assert [c for c in calls if c in ("_create_synced_tables", "record_version")] == \
        ["_create_synced_tables", "record_version"]
    assert calls[-1] == "record_version"


def _sync_w(update_states: Dict[str, List[str]]):
    """Fake w: each table's pipeline walks through `update_states` on get_update polls."""
    def get_synced_table(name):
        return pg.SyncedTable(status=pg.SyncedTableSyncedTableStatus(pipeline_id=f"pipe-{name.split('.')[-1]}"))

    polls = {k: list(v) for k, v in update_states.items()}
    seen: List[str] = []

    def get_update(pipeline_id, update_id):
        table = pipeline_id.split("-", 1)[1]
        seen.append(table)
        state = polls[table].pop(0) if len(polls[table]) > 1 else polls[table][0]
        return SimpleNamespace(update=SimpleNamespace(state=SimpleNamespace(value=state)))

    w = SimpleNamespace(
        postgres=SimpleNamespace(get_synced_table=get_synced_table),
        pipelines=SimpleNamespace(get_update=get_update,
                                  start_update=lambda pipeline_id, full_refresh: SimpleNamespace(update_id="u")),
    )
    return w, seen


def _show_tables(*names: str):
    """A warehouse that answers SHOW TABLES IN cat.lgk_v1_abc123 with `names` (database, tableName, isTemporary)."""
    seen: List[str] = []

    def sql(statement: str):
        seen.append(statement)
        assert statement == "SHOW TABLES IN cat.lgk_v1_abc123", statement
        return [["lgk_v1_abc123", n, "false"] for n in names]
    return sql


# hostile: the kit schema also holds enrich work tables and an enrichment table provision does
# NOT sync (node_group); only the synced enrichment tables (node_score) join the refresh set.
KIT_TABLES = ("node_id_map", "_enrich_deg", "node_group", "node_score", "edges", "nodes")
NO_ENRICH = ("nodes", "edges", "node_id_map")


def _synced_run(tmp_path, *tables: str, version_schema: str = "lgk_v1_abc123_v1") -> Dict[str, Any]:
    """A config whose run manifest records `tables` as SYNCED_TABLEs in `version_schema` (as provision does)."""
    from lgk import manifest as mf

    cfg = _config(run_folder=str(tmp_path))
    # an enrichment table is synced (and refreshed) only if this run's enrich report
    # lists a method writing it; this run's enrich ran degree (node_score), not components.
    (tmp_path / "enrich_report.json").write_text('{"degree": {"rows": 1}}')
    for t in tables:
        mf.record(cfg, mf.ObjectType.SYNCED_TABLE, f"{CAT}.{version_schema}.{t}", "uc")
    return cfg


ALL_SYNCED = ("nodes", "edges", "node_id_map", "node_score")


class _RoomyConn:
    """A branch connection with room for any sync (the free-slot read: 1000 max, nothing reserved or used)."""

    ANSWERS = {"SHOW max_connections": "1000", "SHOW superuser_reserved_connections": "0"}

    def execute(self, query):
        assert query in self.ANSWERS or query.startswith("SELECT count(*) FROM pg_stat_activity"), query
        return SimpleNamespace(fetchone=lambda: (self.ANSWERS.get(query, 0),))

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _roomy_branch(monkeypatch):
    """sync_all reads the branch's free slots before each start; offline it is roomy."""
    monkeypatch.setattr(sync, "connect_branch", lambda cfg: _RoomyConn(), raising=False)


class TestRefreshSet:
    def test_synced_enrichment_tables_join_the_version_tables(self, tmp_path):
        pairs = freshness.synced_pairs(_synced_run(tmp_path, *ALL_SYNCED), _show_tables(*KIT_TABLES))
        assert pairs == [(f"cat.lgk_v1_abc123_v1.{t}", f"cat.lgk_v1_abc123.{t}")
                         for t in ("nodes", "edges", "node_id_map", "node_score")]

    def test_a_run_without_enrich_refreshes_only_the_version_tables(self):
        pairs = freshness.synced_pairs(_config(), _show_tables(*NO_ENRICH))
        assert [d.rsplit(".", 1)[1] for d, _ in pairs] == ["nodes", "edges", "node_id_map"]

    def test_a_table_enriched_after_provision_is_skipped_with_a_warning(self, tmp_path, caplog):
        # hostile: node_score IS in UC (enrich ran after provision) but provision never synced it;
        # and an older version's node_score IS recorded, which must not count for this version
        cfg = _synced_run(tmp_path, "nodes", "edges", "node_id_map")
        _synced_run(tmp_path, "node_score", version_schema="lgk_v1_abc123_v0")
        with caplog.at_level("WARNING", logger="lgk.refresh.freshness"):
            pairs = freshness.synced_pairs(cfg, _show_tables(*KIT_TABLES))
        assert [d.rsplit(".", 1)[1] for d, _ in pairs] == ["nodes", "edges", "node_id_map"]
        assert any("lgk_v1_abc123.node_score" in r.getMessage() and r.levelname == "WARNING"
                   for r in caplog.records), caplog.text

    def test_a_non_synced_record_of_node_score_does_not_count(self, tmp_path):
        # hostile: the manifest names cat.<version>.node_score, but as a VIEW, not a SYNCED_TABLE
        from lgk import manifest as mf

        cfg = _synced_run(tmp_path, "nodes", "edges", "node_id_map")
        mf.record(cfg, mf.ObjectType.VIEW, f"{CAT}.lgk_v1_abc123_v1.node_score", "pg")
        pairs = freshness.synced_pairs(cfg, _show_tables(*KIT_TABLES))
        assert [d.rsplit(".", 1)[1] for d, _ in pairs] == ["nodes", "edges", "node_id_map"]

    def test_a_run_with_no_manifest_refreshes_only_the_version_tables(self, tmp_path, caplog):
        with caplog.at_level("WARNING", logger="lgk.refresh.freshness"):
            pairs = freshness.synced_pairs(_config(run_folder=str(tmp_path)), _show_tables(*KIT_TABLES))
        assert [d.rsplit(".", 1)[1] for d, _ in pairs] == ["nodes", "edges", "node_id_map"]
        assert any("No kit manifest" in r.getMessage() for r in caplog.records), caplog.text

    def test_sync_all_skips_an_unsynced_enrichment_table(self, tmp_path, monkeypatch):
        """The refresh's sync never asks for a synced table that does not exist (no crash)."""
        monkeypatch.setattr(sync.time, "sleep", lambda s: None)
        monkeypatch.setattr(sync, "_wait_synced_table_online", lambda w, dest: None)
        w, seen = _sync_w({"nodes": ["COMPLETED"], "edges": ["COMPLETED"], "node_id_map": ["COMPLETED"]})
        sync.sync_all(w, _show_tables(*KIT_TABLES), _synced_run(tmp_path, "nodes", "edges", "node_id_map"))
        assert sorted(set(seen)) == ["edges", "node_id_map", "nodes"]

    def test_the_guard_names_a_stale_node_score_when_the_graph_is_fresh(self, tmp_path, monkeypatch):
        # hostile: enrich re-ran (node_score v5, a data WRITE) after the sync (v4); nodes/edges fresh
        w = _fake_w({"nodes": 7, "edges": 3, "node_id_map": 2, "node_score": 4})
        hist = _fake_sql({"nodes": 7, "edges": 3, "node_id_map": 2, "node_score": 5})
        show = _show_tables(*KIT_TABLES)
        sql = lambda st: show(st) if st.startswith("SHOW TABLES") else hist(st)  # noqa: E731
        monkeypatch.setattr(freshness, "live_clients", lambda cfg: (w, sql))
        with pytest.raises(freshness.StaleSyncError, match=r"node_score: synced version 4.*version 5") as e:
            freshness.check_freshness(_synced_run(tmp_path, *ALL_SYNCED))
        assert "nodes:" not in str(e.value) and "edges:" not in str(e.value)


class TestSyncWaits:
    def test_a_failed_update_fails_the_sync(self, monkeypatch):
        monkeypatch.setattr(sync.time, "sleep", lambda s: None)
        monkeypatch.setattr(sync, "_wait_synced_table_online", lambda w, dest: None)
        w, _ = _sync_w({"nodes": ["RUNNING", "FAILED"], "edges": ["COMPLETED"], "node_id_map": ["COMPLETED"]})
        with pytest.raises(sync.SyncUpdateError, match="FAILED"):
            sync.sync_all(w, _show_tables(*NO_ENRICH), _config())

    def test_every_update_is_waited_for_not_just_the_first(self, monkeypatch):
        monkeypatch.setattr(sync.time, "sleep", lambda s: None)
        monkeypatch.setattr(sync, "_wait_synced_table_online", lambda w, dest: None)
        # hostile: the first table completes at once; only the LAST one fails, after polling
        w, _ = _sync_w({"nodes": ["COMPLETED"], "edges": ["COMPLETED"],
                        "node_id_map": ["RUNNING", "RUNNING", "CANCELED"]})
        with pytest.raises(sync.SyncUpdateError, match="node_id_map|CANCELED"):
            sync.sync_all(w, _show_tables(*NO_ENRICH), _config())

    def test_all_completed_waits_on_each_table(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sync.time, "sleep", lambda s: None)
        online: List[str] = []
        monkeypatch.setattr(sync, "_wait_synced_table_online", lambda w, dest: online.append(dest.rsplit(".", 1)[1]))
        w, seen = _sync_w({"nodes": ["RUNNING", "COMPLETED"], "edges": ["COMPLETED"], "node_id_map": ["COMPLETED"],
                           "node_score": ["RUNNING", "COMPLETED"]})
        sync.sync_all(w, _show_tables(*KIT_TABLES), _synced_run(tmp_path, *ALL_SYNCED))
        assert sorted(online) == ["edges", "node_id_map", "node_score", "nodes"]
        assert seen.count("node_score") == 2  # the enrichment table's update is waited for too
        assert seen.count("nodes") == 2


class TestSyncBatchedOrder:
    def test_the_next_update_starts_only_when_one_finishes(self, tmp_path, monkeypatch):
        """Updates are batched (at most sync_max_parallel running); here 1, so start/done alternate."""
        events: List[str] = []

        def get_synced_table(name):
            return pg.SyncedTable(status=pg.SyncedTableSyncedTableStatus(pipeline_id=f"pipe-{name.split('.')[-1]}"))

        w = SimpleNamespace(
            postgres=SimpleNamespace(get_synced_table=get_synced_table),
            pipelines=SimpleNamespace(start_update=lambda pipeline_id, full_refresh: events.append(
                f"start {pipeline_id}") or SimpleNamespace(update_id=pipeline_id)),
        )
        import time as _time

        def wait_done(w, pid, uid):
            _time.sleep(0.05)  # an update takes a while: an unbounded sync would start the next meanwhile
            events.append(f"done {pid}")
        monkeypatch.setattr(sync, "_wait_update_done", wait_done)
        monkeypatch.setattr(sync, "_wait_synced_table_online", lambda w, dest: None)
        sync.sync_all(w, _show_tables(*KIT_TABLES), dict(_synced_run(tmp_path, *ALL_SYNCED), sync_max_parallel=1))
        assert events == [e for t in ALL_SYNCED for e in (f"start pipe-{t}", f"done pipe-{t}")]


class TestSyncTask:
    """The nightly sync_all refuses a triggered config; snapshot-only runs refresh every table in full.

    The order of the checks inside sync_all (mode before live client, stamp, synced tables, policy_name) is pinned in
    tests/test_nightly_sync_mode.py; this test pins what reaches the pipelines: nothing, for any table.
    """

    @staticmethod
    def _recording_w():
        looked_up: List[str] = []
        started: List[Any] = []

        def get_synced_table(name):
            looked_up.append(name.split(".")[-1])
            return pg.SyncedTable(status=pg.SyncedTableSyncedTableStatus(pipeline_id=f"pipe-{name.split('.')[-1]}"))

        w = SimpleNamespace(
            postgres=SimpleNamespace(get_synced_table=get_synced_table),
            pipelines=SimpleNamespace(start_update=lambda pipeline_id, full_refresh: started.append(
                (pipeline_id, full_refresh)) or SimpleNamespace(update_id="u")),
        )
        return w, looked_up, started

    # hostile: reordered input, the snapshot-only tables come before the TRIGGERED_TABLES, so a sync that
    # checks the mode per table (or only for nodes/edges) would start node_id_map and node_score first
    REORDERED = ("node_id_map", "node_score", "nodes", "edges")

    def test_a_triggered_config_syncs_no_table_snapshot_or_triggered(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sync, "_wait_update_done", lambda w, pid, uid: None)
        monkeypatch.setattr(sync, "_wait_synced_table_online", lambda w, dest: None)
        w, looked_up, started = self._recording_w()
        cfg = dict(_synced_run(tmp_path, *self.REORDERED), sync_mode="triggered")
        with pytest.raises(modes.UnsupportedSyncModeError, match="v1 supports only 'snapshot'.*Path B"):
            sync.sync_all(w, _show_tables(*reversed(KIT_TABLES)), cfg)
        assert started == [] and looked_up == []
        # the full_refresh argument never skips the refusal either way
        for full in (True, False):
            with pytest.raises(modes.UnsupportedSyncModeError, match="Path B"):
                sync.sync_all(w, _show_tables(*reversed(KIT_TABLES)), cfg, full_refresh=full)
        assert started == [] and looked_up == []

    @pytest.mark.parametrize("mode", ["unset", "snapshot"])
    def test_a_snapshot_config_refreshes_every_table_in_full(self, tmp_path, monkeypatch, mode):
        monkeypatch.setattr(sync, "_wait_update_done", lambda w, pid, uid: None)
        monkeypatch.setattr(sync, "_wait_synced_table_online", lambda w, dest: None)
        w, looked_up, started = self._recording_w()
        cfg = _synced_run(tmp_path, *self.REORDERED)
        if mode == "snapshot":
            cfg = dict(cfg, sync_mode="snapshot")
        sync.sync_all(w, _show_tables(*reversed(KIT_TABLES)), cfg)
        # nodes and edges are snapshot too: no table is synced incrementally in v1
        assert sorted(started) == [("pipe-edges", True), ("pipe-node_id_map", True),
                                   ("pipe-node_score", True), ("pipe-nodes", True)]
        assert sorted(looked_up) == sorted(self.REORDERED)


class TestRunTask:
    def test_each_job_task_runs_its_function_every_time(self, monkeypatch):
        from lgk.refresh import tasks

        ran: List[Any] = []
        monkeypatch.setattr(tasks, "_task_functions", lambda: {
            k: (lambda cfg, k=k: ran.append((k, cfg["kit_schema"]))) for k in ("enrich", "sync", "reapply")})
        # hostile: the same task twice — a run.json-style "already succeeded" skip would drop it
        for key in ("sync", "sync", "reapply"):
            tasks.run_task(key, '{"kit_schema": "lgk_v1_abc123"}')
        assert ran == [("sync", "lgk_v1_abc123"), ("sync", "lgk_v1_abc123"), ("reapply", "lgk_v1_abc123")]

    def test_build_task_runs_build_every_night(self, tmp_path, monkeypatch):
        """run_task('build') runs the kit build each time, whatever run.json says."""
        import lgk.steps.build as build_mod
        from lgk.refresh import tasks

        real_build = build_mod.build
        assert tasks._task_functions()["build"] is real_build  # the shipped build, not a stand-in
        # hostile: run.json says build already succeeded; the runner would skip it, a night must not
        (tmp_path / "run.json").write_text(json.dumps({"build": {"status": "success", "error": None}}))
        ran: List[str] = []
        monkeypatch.setattr(build_mod, "build", lambda cfg: ran.append(cfg["run_folder"]))
        cfg = json.dumps(_config(run_folder=str(tmp_path)))
        tasks.run_task("build", cfg)
        tasks.run_task("build", cfg)
        assert ran == [str(tmp_path), str(tmp_path)]

    def test_unknown_task_fails(self):
        from lgk.refresh import tasks

        with pytest.raises(ValueError, match="unknown refresh task"):
            tasks.run_task("sdp_pipeline", "{}")

    def test_job_task_keys_resolve_to_task_functions(self):
        from lgk.refresh import tasks
        from lgk.refresh.sync import sync as sync_fn
        from lgk.steps.enrich import enrich
        from lgk.steps.reapply import reapply

        fns = tasks._task_functions()
        assert set(fns) == set(job.TASK_ORDER[1:])
        # hostile: every value is a real, callable task, so only identity catches a crossed wire
        assert fns["reapply"] is reapply and fns["sync"] is sync_fn and fns["enrich"] is enrich

    def test_reapply_task_fails_the_job_on_a_stale_source(self, monkeypatch):
        """run_task('reapply') -> the real guard, fed a stale history -> StaleSyncError; nothing served."""
        import databricks.sdk
        import lgk.uc_sql
        from lgk.refresh import tasks
        from lgk.steps import reapply as reapply_mod

        w = _fake_w({"nodes": 7, "edges": 3, "node_id_map": 2})
        hist = _fake_sql({"nodes": _history(8), "edges": 3, "node_id_map": 2})  # hostile: v8 WRITE unsynced
        show = _show_tables(*NO_ENRICH)
        sql = lambda st: show(st) if st.startswith("SHOW TABLES") else hist(st)  # noqa: E731
        monkeypatch.setattr(databricks.sdk, "WorkspaceClient", lambda *a, **k: w)
        monkeypatch.setattr(lgk.uc_sql, "execute_sql", lambda _w, _wh, stmt: sql(stmt))
        served: List[str] = []
        for name in ("_reapply_indexes", "_create_served_views", "_setup_reader_role"):
            monkeypatch.setattr(reapply_mod, name, lambda cfg, n=name: served.append(n))
        # the served-version check passes (the views read this job's version; tests/test_rollback.py)
        monkeypatch.setattr(reapply_mod, "_refuse_a_moved_site", lambda cfg: None)
        # the privacy check (switch's require_servable) is proven in tests/test_reapply_privacy.py
        monkeypatch.setattr(reapply_mod, "_refuse_an_unservable_version", lambda cfg: None)
        with pytest.raises(freshness.StaleSyncError, match="nodes"):
            tasks.run_task("reapply", json.dumps(_config()))
        assert served == []


class TestReapplyOrder:
    def test_guard_runs_before_indexes_views_and_grants(self, monkeypatch):
        from lgk.steps import reapply as reapply_mod

        order: List[str] = []
        # the fold floor detector runs right after the guard (tests/test_fold.py proves it)
        # rollback: the served-version check runs first (tests/test_rollback.py proves it)
        # P6-decision: switch's privacy check runs before it (tests/test_reapply_privacy.py proves it)
        for name in ("_refuse_an_unservable_version", "_refuse_a_moved_site", "check_freshness", "_detect_unfloored_rows", "_reapply_indexes",
                     "_create_served_views", "_setup_reader_role"):
            monkeypatch.setattr(reapply_mod, name, lambda cfg, n=name: order.append(n))
        reapply_mod.reapply(_config())
        assert order == ["_refuse_an_unservable_version", "_refuse_a_moved_site", "check_freshness",
                         "_detect_unfloored_rows", "_reapply_indexes", "_create_served_views", "_setup_reader_role"]


# ------------------------------------------------- the nightly kit build (offline)
class FakeDelta:
    """A scripted warehouse that keeps, per table IDENTITY, a Delta history and TBLPROPERTIES.

    No SQL engine: it reads only the statement shapes build and the guard issue. DROP forgets a
    table (a later CREATE TABLE starts a NEW history at version 0, as live spike S3b measured);
    CREATE OR REPLACE on an existing table appends a commit (same identity, history goes on)
    and resets its properties to the ones it names.
    """

    _ids = 0  # every new table identity gets its own UC table_id (DROP + CREATE: a new one)

    def __init__(self, fail_on: str = ""):
        self.tables: Dict[str, Dict[str, Any]] = {}
        self.fail_on = fail_on
        self.statements: List[str] = []
        # per table: the DISTINCT (node_type | relationship_type, property key) rows of its
        # `properties` (key None = an empty map), as the allowlist query reads them
        self.keys: Dict[str, List[List[Any]]] = {}

    def _commit(self, table: str, op: str, metrics: Dict[str, str]) -> None:
        h = self.tables[table]["history"]
        h.append((len(h), op, metrics))

    def version(self, table: str) -> int:
        return self.tables[table]["history"][-1][0]

    def table_id(self, table: str) -> str:
        return self.tables[table]["id"]

    def _create(self, table: str, replace: bool, columns: str, props: str) -> None:
        if table in self.tables and not replace:
            raise RuntimeError(f"TABLE_ALREADY_EXISTS: {table}")
        if table not in self.tables:
            FakeDelta._ids += 1
            self.tables[table] = {"history": [], "props": {}, "id": f"uc-tid-{FakeDelta._ids}"}
        self._commit(table, "CREATE OR REPLACE TABLE" if replace else "CREATE TABLE", {})
        self.tables[table]["props"] = dict(re.findall(r"'([^']+)' = '([^']*)'", props))
        self.tables[table]["columns"] = [[n, t.replace(" ", "").lower(), None] for n, t in
                                         re.findall(r"(\w+) (MAP<[^>]*>|\w+)", columns)]

    def __call__(self, statement: str) -> List[List[Any]]:
        s = " ".join(statement.split())
        self.statements.append(s)
        if self.fail_on and self.fail_on in s:
            raise RuntimeError("warehouse went away")
        if m := re.fullmatch(r"DROP TABLE IF EXISTS (\S+)", s):
            self.tables.pop(m[1], None)
        elif m := re.fullmatch(r"CREATE (OR REPLACE )?TABLE (\S+) \((.*?)\) USING DELTA(.*)", s):
            self._create(m[2], bool(m[1]), m[3], m[4])
        elif m := re.match(r"INSERT (?:INTO|OVERWRITE) (\S+) ", s):
            self._commit(m[1], "WRITE", {"numOutputRows": "2"})
        elif m := re.fullmatch(r"ALTER TABLE (\S+) SET TBLPROPERTIES \('([^']+)' = '([^']*)'\)", s):
            self._commit(m[1], "SET TBLPROPERTIES", {})
            self.tables[m[1]]["props"][m[2]] = m[3]
        elif m := re.fullmatch(r"SHOW TBLPROPERTIES (\S+)", s):
            return [[k, v] for k, v in self.tables[m[1]]["props"].items()]
        elif m := re.fullmatch(r"SELECT version, operation, operationMetrics FROM \(DESCRIBE HISTORY (\S+)\)", s):
            return [[str(v), op, json.dumps(mt)] for v, op, mt in reversed(self.tables[m[1]]["history"])]
        elif m := re.fullmatch(r"DESCRIBE TABLE (\S+)", s):
            return [list(c) for c in self.tables[m[1]]["columns"]]
        elif m := re.fullmatch(r"SELECT DISTINCT (\w+), k FROM (\S+) LATERAL VIEW OUTER "
                               r"explode\(map_keys\(properties\)\) x AS k", s):
            return [list(r) for r in self.keys.get(m[2], [])]
        elif m := re.fullmatch(r"SHOW TABLES IN (\S+)", s):
            return [[m[1].split(".")[1], t.rsplit(".", 1)[1], "false"] for t in self.tables
                    if t.rsplit(".", 1)[0] == m[1]]
        return []


KIT_FQ = [f"{CAT}.lgk_v1_abc123.{t}" for t in ("nodes", "edges", "node_id_map")]
# The kit tables' DDL, written out by hand (the spec the allowlist checks against).
KIT_DDL = {
    "nodes": "node_id STRING, node_type STRING, natural_key STRING, label STRING, properties MAP<STRING, STRING>",
    "edges": ("edge_id STRING, src_node_id STRING, dst_node_id STRING, relationship_type STRING, "
              "properties MAP<STRING, STRING>, weight BIGINT"),
    "node_id_map": "node_id STRING, node_type STRING, natural_key STRING",
}


def _uc_tables(delta: FakeDelta) -> SimpleNamespace:
    """UC's `w.tables`: get(full_name).table_id of the table identity the FakeDelta holds."""
    return SimpleNamespace(get=lambda full_name: SimpleNamespace(table_id=delta.table_id(full_name)))


def _label_tables(cfg: Dict[str, Any], delta: FakeDelta, stamps: Dict[str, Any], default: str) -> None:
    """Create the kit tables (kit DDL) and label each as a stamp step would: label + manifest record.

    stamps[t] None = no label and no record; else the label's stamp (default: `default`)."""
    from lgk import manifest as mf

    for fq in KIT_FQ:
        t = fq.rsplit(".", 1)[1]
        delta(f"CREATE TABLE {fq} ({KIT_DDL[t]}) USING DELTA")
        stamp = stamps.get(t, default)
        if stamp is None:
            continue
        label = {"stamp": stamp, "table_id": delta.table_id(fq), "certified_version": delta.version(fq)}
        delta(f"ALTER TABLE {fq} SET TBLPROPERTIES ('lgk.privacy_stamp' = '{json.dumps(label)}')")
        mf.record_table_labels(cfg, {fq: label})


def _cleared_run(tmp_path) -> Dict[str, Any]:
    """A run folder that passed the REAL privacy step (offline profile stand-in), and its config."""
    from tests.privacy_support import clear_gate_offline
    from tests.test_build_provision_walks import _write_provider_graph_mapping

    _write_provider_graph_mapping(tmp_path / "graph_mapping.yaml", "gold", "hls")
    clear_gate_offline(str(tmp_path))
    return _config(run_folder=str(tmp_path))


def _fake_build(monkeypatch, delta: FakeDelta) -> None:
    import lgk.steps.build as build_mod

    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: SimpleNamespace(tables=_uc_tables(delta)))
    monkeypatch.setattr(build_mod, "execute_sql", lambda w, wh, s: delta(s))
    # The kit-schema claim is tested in test_teardown.py; here only the rebuild matters.
    monkeypatch.setattr(build_mod, "_claim_kit_schema", lambda *a, **k: None)


def _synced_at(delta: FakeDelta):
    """A fake w whose synced tables each hold their source's version right now (a sync just ran)."""
    return _fake_w({fq.rsplit(".", 1)[1]: delta.version(fq) for fq in KIT_FQ})


class TestNightlyBuild:
    def test_build_records_its_tables_once_per_name(self, tmp_path, monkeypatch):
        from lgk import manifest as mf
        from lgk.refresh.tasks import run_task

        cfg = _cleared_run(tmp_path)
        _fake_build(monkeypatch, FakeDelta())
        for _ in range(2):  # hostile: two nights of the same build
            run_task("build", json.dumps(cfg))
        tables = [e.object_id for e in mf.load_run_manifest(cfg).entries if e.object_type == mf.ObjectType.UC_TABLE]
        assert sorted(tables) == sorted(KIT_FQ)

    def test_rebuild_keeps_delta_history_so_a_stale_sync_is_caught(self, tmp_path, monkeypatch):
        """S3b: a rebuild must CONTINUE each table's Delta history, else the guard is blind.

        Live spike S3b (2026-10-04): after DROP TABLE + CREATE TABLE the new table's versions
        restart at 0, so yesterday's synced version equals the new current version and the guard
        passed while Postgres still served the old rows. CREATE OR REPLACE keeps the history.
        """
        from lgk.refresh.tasks import run_task

        cfg = _cleared_run(tmp_path)
        delta = FakeDelta()
        _fake_build(monkeypatch, delta)
        run_task("build", json.dumps(cfg))  # night 1, then a sync
        w = _synced_at(delta)
        pairs = [(f"{CAT}.lgk_v1_abc123_v1.{fq.rsplit('.', 1)[1]}", fq) for fq in KIT_FQ]
        freshness.assert_fresh(w, delta, pairs)  # non-vacuous: fresh right after the sync
        night1 = {fq: delta.version(fq) for fq in KIT_FQ}
        # hostile: a rebuilt table whose version restarts. Night 2 rebuilds with NO sync after it;
        # a DROP/CREATE build lands on the same version numbers as night 1 and looks fresh.
        run_task("build", json.dumps(cfg))
        assert all(delta.version(fq) > night1[fq] for fq in KIT_FQ), "a rebuild restarted a Delta history"
        with pytest.raises(freshness.StaleSyncError) as e:
            freshness.assert_fresh(w, delta, pairs)
        assert all(f"{t}: synced version" in str(e.value) for t in ("nodes", "edges", "node_id_map"))
        assert not any(s.startswith("DROP TABLE") for s in delta.statements)

    def test_a_failed_rebuild_leaves_no_valid_stamp(self, tmp_path, monkeypatch):
        """The stamp is bound last: a night that fails mid-build leaves tables provision/sync refuse."""
        from lgk.privacy import binding
        from lgk.refresh.tasks import run_task

        cfg = _cleared_run(tmp_path)
        delta = FakeDelta()
        _fake_build(monkeypatch, delta)
        run_task("build", json.dumps(cfg))
        stamp = binding.built_stamp(str(tmp_path))
        binding.check_table_stamps(delta, KIT_FQ, stamp)  # non-vacuous: night 1 is bound
        # hostile: night 2 dies on the edges insert, after nodes was already replaced; the
        # replaced table must not keep night 1's stamp (CREATE OR REPLACE could carry it over)
        delta.fail_on = "INSERT INTO cat.lgk_v1_abc123.edges"
        with pytest.raises(RuntimeError, match="went away"):
            run_task("build", json.dumps(cfg))
        with pytest.raises(binding.StaleBuildError, match="nodes"):
            binding.check_table_stamps(delta, KIT_FQ, stamp)


def _stamp_sync(tmp_path, monkeypatch, stamps: Dict[str, Any], **extra: Any):
    """The sync task over a cleared run whose kit tables carry `stamps` (None = no property)."""
    from lgk.privacy import binding
    from lgk.privacy.gate import require_clear

    cfg = dict(_cleared_run(tmp_path), **extra)
    current = binding.stamp_id(require_clear(str(tmp_path)))
    delta = FakeDelta()
    _label_tables(cfg, delta, stamps, current)
    started: List[str] = []
    w, _ = _sync_w({t: ["COMPLETED"] for t in ("nodes", "edges", "node_id_map")})
    w.tables = _uc_tables(delta)
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: w)
    w.pipelines.start_update = lambda pipeline_id, full_refresh: started.append(pipeline_id) or SimpleNamespace(
        update_id="u")
    monkeypatch.setattr(sync.time, "sleep", lambda s: None)
    monkeypatch.setattr(sync, "_wait_synced_table_online", lambda w, dest: None)
    monkeypatch.setattr(sync, "live_clients", lambda c: (w, delta))
    return cfg, started


class TestSyncStampCheck:
    @pytest.mark.parametrize("chain", [{}, {"sdp_pipeline_id": "p-1", "build_mode": "sql"}])
    @pytest.mark.parametrize("stamps", [{"node_id_map": "0" * 64}, {"edges": None}])
    def test_sync_refuses_tables_without_the_run_stamp(self, tmp_path, monkeypatch, stamps, chain):
        from lgk.privacy.binding import StaleBuildError
        from lgk.refresh.tasks import run_task

        # hostile: only ONE table is off (the last one, or a middle one with no property at all),
        # on both chains whose kit build makes the tables (default, and pipeline + build_mode sql)
        cfg, started = _stamp_sync(tmp_path, monkeypatch, stamps, **chain)
        bad = next(iter(stamps))
        with pytest.raises(StaleBuildError, match=rf"lgk_v1_abc123\.{bad}"):
            run_task("sync", json.dumps(cfg))
        assert started == []  # refused before the first update

    def test_sync_refuses_tables_built_before_a_privacy_re_decision(self, tmp_path, monkeypatch):
        """The check is against the CURRENT cleared decision, not the stamp the last build recorded."""
        from lgk.privacy import binding
        from lgk.privacy.binding import StaleBuildError
        from lgk.refresh.tasks import run_task

        # hostile: the tables AND privacy_bindings.json agree on an older stamp (built, then the
        # operator re-decided privacy); comparing against the recorded build stamp would pass
        old = "a" * 64
        cfg, started = _stamp_sync(tmp_path, monkeypatch, {t: old for t in ("nodes", "edges", "node_id_map")})
        binding.record_build(str(tmp_path), old, KIT_FQ)
        with pytest.raises(StaleBuildError, match="stale stamp"):
            run_task("sync", json.dumps(cfg))
        assert started == []

    @pytest.mark.parametrize("chain", [{}, {"sdp_pipeline_id": "p-1"}])
    def test_sync_refuses_when_the_privacy_gate_is_no_longer_clear(self, tmp_path, monkeypatch, chain):
        from lgk.privacy.gate import PrivacyGateError
        from lgk.refresh.tasks import run_task

        # hostile: the operator-pipeline chain checks the gate too, with every stamp current
        cfg, started = _stamp_sync(tmp_path, monkeypatch, {}, **chain)  # every table on the current stamp
        # hostile: graph_mapping.yaml edited after privacy cleared it (the label is another column)
        mapping = tmp_path / "graph_mapping.yaml"
        mapping.write_text(mapping.read_text().replace("label: name", "label: specialty"))
        with pytest.raises(PrivacyGateError):
            run_task("sync", json.dumps(cfg))
        assert started == []

    # the stamp-check ruling replaces test_operator_pipeline_chain_syncs_unstamped_tables_with_a_warning
    # (which asserted the removed WARNING skip): the operator-pipeline chain now refuses too
    @pytest.mark.parametrize("stamps", [{t: None for t in ("nodes", "edges", "node_id_map")},
                                        {"node_id_map": None}, {"edges": "0" * 64}])
    def test_operator_pipeline_chain_refuses_unstamped_tables(self, tmp_path, monkeypatch, caplog, stamps):
        """sdp_pipeline_id, no build_mode: the operator's pipeline rebuilt the tables and did not stamp them."""
        from lgk.privacy.binding import StaleBuildError
        from lgk.refresh.tasks import run_task

        # hostile: every table unstamped (as a plain operator rebuild leaves them), only the LAST
        # table unstamped, and a middle table on a stale stamp
        cfg, started = _stamp_sync(tmp_path, monkeypatch, stamps, sdp_pipeline_id="p-1")
        assert job.refresh_chain(cfg, enriched=False) == ["sdp_pipeline", "stamp", "sync", "reapply"]  # non-vacuous
        with caplog.at_level("WARNING", logger="lgk.refresh.sync"):
            with pytest.raises(StaleBuildError, match=rf"lgk_v1_abc123\.{list(stamps)[-1]}"):
                run_task("sync", json.dumps(cfg))
        assert started == []  # refused before the first update
        assert not any("stamp check is skipped" in r.getMessage() for r in caplog.records)

    def test_operator_pipeline_chain_syncs_after_the_operator_stamp_helper(self, tmp_path, monkeypatch):
        from lgk.refresh.operator_stamp import stamp_operator_tables
        from lgk.refresh.tasks import run_task

        # hostile: all three unstamped, then the helper (as the operator runs it after the pipeline)
        cfg, started = _stamp_sync(tmp_path, monkeypatch, {t: None for t in ("nodes", "edges", "node_id_map")},
                                   sdp_pipeline_id="p-1")
        stamp_operator_tables(cfg, sql=sync.live_clients(cfg)[1])
        run_task("sync", json.dumps(cfg))
        assert sorted(started) == ["pipe-edges", "pipe-node_id_map", "pipe-nodes"]

    def test_sync_refuses_an_unknown_build_mode(self, tmp_path, monkeypatch):
        from lgk.refresh.tasks import run_task

        # hostile (prover): a pipeline and a bad build_mode; a re-derived skip would sync anyway
        cfg, started = _stamp_sync(tmp_path, monkeypatch, {}, sdp_pipeline_id="p-1", build_mode="bogus")
        with pytest.raises(ValueError, match="build_mode"):
            run_task("sync", json.dumps(cfg))
        assert started == []

    # the stamp-check ruling replaces test_the_skip_is_decided_by_refresh_chain (which asserted that a
    # chain without `build` synced an unstamped table): no chain shape skips the stamp check
    def test_no_chain_shape_skips_the_stamp_check(self, tmp_path, monkeypatch):
        import lgk.refresh.job as job_mod
        from lgk.privacy.binding import StaleBuildError
        from lgk.refresh.tasks import run_task

        cfg, started = _stamp_sync(tmp_path, monkeypatch, {"edges": None})
        # hostile: refresh_chain says the chain has no kit build at all
        monkeypatch.setattr(job_mod, "refresh_chain", lambda c, enriched: ["sync", "reapply"])
        with pytest.raises(StaleBuildError, match=r"lgk_v1_abc123\.edges"):
            run_task("sync", json.dumps(cfg))
        assert started == []

    def test_sync_with_every_table_on_the_run_stamp_runs_each_update(self, tmp_path, monkeypatch):
        from lgk.refresh.tasks import run_task

        cfg, started = _stamp_sync(tmp_path, monkeypatch, {})
        run_task("sync", json.dumps(cfg))
        assert sorted(started) == ["pipe-edges", "pipe-node_id_map", "pipe-nodes"]


# ------------------------- stamp check: the helper an operator pipeline stamps with
class TestOperatorStamp:
    def test_helper_sets_the_current_stamp_on_every_kit_table(self, tmp_path, monkeypatch):
        from lgk.privacy import binding
        from lgk.privacy.gate import require_clear
        from lgk.refresh.operator_stamp import stamp_operator_tables

        # hostile: nodes unstamped, edges on a STALE stamp, node_id_map unstamped (the last table);
        # and privacy_bindings.json records an older build stamp a lazy helper might copy
        old = "a" * 64
        cfg, _ = _stamp_sync(tmp_path, monkeypatch, {"nodes": None, "edges": old, "node_id_map": None},
                             sdp_pipeline_id="p-1")
        binding.record_build(str(tmp_path), old, KIT_FQ)
        delta = sync.live_clients(cfg)[1]
        current = binding.stamp_id(require_clear(str(tmp_path)))
        assert current != old  # non-vacuous
        assert stamp_operator_tables(cfg, sql=delta) == current
        for fq in KIT_FQ:  # the label is the JSON {stamp, table_id, certified_version} (owner decision)
            assert json.loads(delta.tables[fq]["props"]["lgk.privacy_stamp"])["stamp"] == current, fq
        binding.check_table_stamps(delta, KIT_FQ, current)

    def test_helper_uses_the_run_warehouse_by_default(self, tmp_path, monkeypatch):
        from lgk.privacy import binding
        from lgk.refresh import operator_stamp

        cfg, _ = _stamp_sync(tmp_path, monkeypatch, {t: None for t in ("nodes", "edges", "node_id_map")})
        delta = sync.live_clients(cfg)[1]
        monkeypatch.setattr(operator_stamp, "live_clients", lambda c: (None, delta))
        stamp = operator_stamp.stamp_operator_tables(cfg)
        binding.check_table_stamps(delta, KIT_FQ, stamp)

    @pytest.mark.parametrize("break_gate", ["edit_mapping", "no_flags"])
    def test_helper_refuses_when_the_privacy_gate_is_not_clear(self, tmp_path, monkeypatch, break_gate):
        from lgk.privacy.gate import PrivacyGateError
        from lgk.refresh.operator_stamp import stamp_operator_tables

        cfg, _ = _stamp_sync(tmp_path, monkeypatch, {t: None for t in ("nodes", "edges", "node_id_map")},
                             sdp_pipeline_id="p-1")
        delta = sync.live_clients(cfg)[1]
        if break_gate == "edit_mapping":
            # hostile: graph_mapping.yaml edited after privacy cleared it (the label is another column)
            mapping = tmp_path / "graph_mapping.yaml"
            mapping.write_text(mapping.read_text().replace("label: name", "label: specialty"))
        else:
            # hostile: the privacy step's flags.json is gone (never ran)
            (tmp_path / "flags.json").unlink()
        n = len(delta.statements)
        with pytest.raises(PrivacyGateError):
            stamp_operator_tables(cfg, sql=delta)
        assert delta.statements[n:] == []  # nothing stamped
        assert all("lgk.privacy_stamp" not in delta.tables[fq]["props"] for fq in KIT_FQ)


    def test_operator_stamp_is_the_current_cleared_decision(self, tmp_path, monkeypatch):
        """The value the docs tell an SDP pipeline to set as its table property."""
        from lgk.privacy import binding
        from lgk.privacy.gate import PrivacyGateError, require_clear
        from lgk.refresh.operator_stamp import operator_stamp

        cfg, _ = _stamp_sync(tmp_path, monkeypatch, {}, sdp_pipeline_id="p-1")
        # hostile: privacy_bindings.json records an older build stamp a lazy reader might return
        binding.record_build(str(tmp_path), "a" * 64, KIT_FQ)
        assert operator_stamp(cfg) == binding.stamp_id(require_clear(str(tmp_path))) != "a" * 64
        (tmp_path / "flags.json").unlink()  # hostile: the privacy step never ran
        with pytest.raises(PrivacyGateError):
            operator_stamp(cfg)

    def test_operator_chain_refuses_tables_stamped_before_a_privacy_re_decision(self, tmp_path, monkeypatch):
        from lgk.privacy import binding
        from lgk.privacy.binding import StaleBuildError
        from lgk.refresh.tasks import run_task

        # hostile: the operator's tables AND privacy_bindings.json agree on an older stamp
        old = "a" * 64
        cfg, started = _stamp_sync(tmp_path, monkeypatch, {t: old for t in ("nodes", "edges", "node_id_map")},
                                   sdp_pipeline_id="p-1")
        binding.record_build(str(tmp_path), old, KIT_FQ)
        with pytest.raises(StaleBuildError, match="stale stamp"):
            run_task("sync", json.dumps(cfg))
        assert started == []


# --------------------------------------- the sync task checks the fold floor
def _fold_sync(tmp_path, monkeypatch, check_rows, privacy_floor=None, **extra: Any):
    """The sync task over a cleared FOLDED run; the source Delta check answers `check_rows`."""
    from lgk.privacy import binding
    from lgk.privacy.gate import require_clear
    from tests.test_fold import _mixed_run

    _mixed_run(tmp_path, privacy_floor)
    cfg = dict(_config(run_folder=str(tmp_path)), **extra)
    current = binding.stamp_id(require_clear(str(tmp_path)))
    delta = FakeDelta()
    _label_tables(cfg, delta, {}, current)
    checks: List[str] = []

    def warehouse(statement: str) -> List[List[Any]]:
        if statement.startswith("SELECT relationship_type, MIN(weight)"):
            checks.append(statement)
            return check_rows
        return delta(statement)

    started: List[str] = []
    w, _ = _sync_w({t: ["COMPLETED"] for t in ("nodes", "edges", "node_id_map")})
    w.tables = _uc_tables(delta)
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: w)
    w.pipelines.start_update = lambda pipeline_id, full_refresh: started.append(pipeline_id) or SimpleNamespace(
        update_id="u")
    monkeypatch.setattr(sync.time, "sleep", lambda s: None)
    monkeypatch.setattr(sync, "_wait_synced_table_online", lambda w, dest: None)
    monkeypatch.setattr(sync, "live_clients", lambda c: (w, warehouse))
    return cfg, started, checks


class TestSyncFoldFloor:
    # hostile: the operator-pipeline-only chain (stamped by the operator); it is exactly the chain whose
    # writer (the operator's pipeline) the kit's HAVING never filtered, so the floor check must run
    @pytest.mark.parametrize("chain", [{}, {"sdp_pipeline_id": "p-1"}])
    def test_sync_refuses_an_unfloored_source_edge(self, tmp_path, monkeypatch, chain):
        from lgk.privacy.floor import FoldFloorError
        from lgk.refresh.tasks import run_task

        cfg, started, checks = _fold_sync(tmp_path, monkeypatch, [["shares_refs", 10, 0, 3]], **chain)
        with pytest.raises(FoldFloorError, match=r"shares_refs \(min weight 10"):
            run_task("sync", json.dumps(cfg))
        assert started == []  # no update starts on unfloored rows
        (check,) = checks
        assert f"FROM {CAT}.lgk_v1_abc123.edges WHERE relationship_type IN ('shares_refs', 'treats_at')" in check

    @pytest.mark.parametrize("chain", [{}, {"sdp_pipeline_id": "p-1"}])
    def test_sync_with_floored_source_edges_runs_each_update(self, tmp_path, monkeypatch, chain):
        from lgk.refresh.tasks import run_task

        cfg, started, checks = _fold_sync(tmp_path, monkeypatch,
                                          [["shares_refs", 11, 0, 3], ["treats_at", 12, 0, 1]], **chain)
        run_task("sync", json.dumps(cfg))
        assert len(checks) == 1 and sorted(started) == ["pipe-edges", "pipe-node_id_map", "pipe-nodes"]


class TestOperatorStampFoldFloor:
    def test_helper_refuses_to_stamp_an_unfloored_operator_edge(self, tmp_path, monkeypatch):
        from lgk.privacy.floor import FoldFloorError
        from lgk.refresh.operator_stamp import stamp_operator_tables

        # hostile: a RAISED floor (12) and an operator edge at 11, legal under the default floor 11
        cfg, _, checks = _fold_sync(tmp_path, monkeypatch, [["shares_refs", 11, 0, 3]], privacy_floor=12,
                                    sdp_pipeline_id="p-1")
        delta = sync.live_clients(cfg)[1]
        statements = []
        with pytest.raises(FoldFloorError, match=r"floor 12: shares_refs \(min weight 11"):
            stamp_operator_tables(cfg, sql=lambda q: statements.append(q) or delta(q))
        (check,) = checks
        assert f"FROM {CAT}.lgk_v1_abc123.edges WHERE" in check  # the kit's edges, not another table
        assert not any("SET TBLPROPERTIES" in q for q in statements)  # nothing stamped


# ------------------------------------------- owner "bugs a": the nightly job's workspace run folder
NIGHTLY_WS = f"{WS_HOME}/lgk_v1_abc123/runs/provider_graph"  # next to JOB_KW's notebook
NIGHTLY_API = NIGHTLY_WS[len("/Workspace"):]


def _job_configs(calls) -> List[Dict[str, Any]]:
    [kw] = calls
    return [json.loads(t.notebook_task.base_parameters["config"]) for t in kw["tasks"] if t.notebook_task]


class TestNightlyRunFolder:
    def _local(self, tmp_path, enriched=True, **extra):
        from tests.refresh_support import seed_run_folder

        seed_run_folder(tmp_path / "run", enriched=enriched)
        return _config(run_folder=str(tmp_path / "run"), **extra)

    def test_a_laptop_run_folder_is_copied_and_every_task_reads_the_copy(self, tmp_path):
        from lgk.manifest import MANIFEST_FILE
        from tests.refresh_support import fake_client

        cfg = self._local(tmp_path)
        run = Path(cfg["run_folder"])
        # hostile: unrelated files the nightly tasks never read sit in the laptop run folder
        (run / "walks_notebook.py").write_text("# Databricks notebook source\n")
        (run / "run.json").write_text("{}")
        (run / "enrich_menu.md").write_text("menu")
        (run / "app" / "x").mkdir(parents=True)
        (run / "app" / "x" / "app.py").write_text("print(1)")
        w, calls = fake_client()
        job.create_refresh_job(cfg, w=w, **JOB_KW)
        configs = _job_configs(calls)
        assert [c["run_folder"] for c in configs] == [NIGHTLY_WS] * len(configs) and configs
        uploaded = {p[len(NIGHTLY_API) + 1:]: b for p, b in w.workspace.files.items()}
        assert sorted(uploaded) == sorted(["flags.json", "schema.json", "graph_mapping.yaml",
                                           "privacy_decisions.json", MANIFEST_FILE,
                                           "enrich.yaml", "enrich_report.json", "privacy_bindings.json"])
        for name, content in uploaded.items():
            if name != MANIFEST_FILE:  # the manifest gains the job record (next test)
                assert content == (run / name).read_bytes(), name
        assert all(p.startswith(NIGHTLY_API + "/") for p in w.workspace.files)  # API form, not /Workspace/...

    def test_the_copy_passes_the_real_privacy_gate_the_tasks_run_first(self, tmp_path):
        """Behaviour, not names: materialise the uploaded files and run require_clear on them."""
        from lgk.manifest import MANIFEST_FILE, Manifest
        from lgk.privacy.gate import require_clear
        from tests.privacy_support import clear_gate_offline
        from tests.refresh_support import fake_client
        from tests.test_build_provision_walks import _write_provider_graph_mapping

        run = tmp_path / "run"
        run.mkdir()
        _write_provider_graph_mapping(run / "graph_mapping.yaml", "gold", "hls")
        clear_gate_offline(str(run))
        Manifest(graph_name="provider_graph").save(run / MANIFEST_FILE)
        w, calls = fake_client()
        job.create_refresh_job(_config(run_folder=str(run)), w=w, **JOB_KW)
        copy = tmp_path / "copy"
        copy.mkdir()
        for path, content in w.workspace.files.items():
            (copy / path.rsplit("/", 1)[-1]).write_bytes(content)
        require_clear(str(copy))  # raises if a gate input is missing or differs from the cleared one

    def test_the_job_is_recorded_in_both_manifests_with_the_copy_folder(self, tmp_path):
        from lgk import manifest as mf
        from tests.refresh_support import fake_client

        cfg = self._local(tmp_path)
        w, calls = fake_client(job_id=77)
        job.create_refresh_job(cfg, w=w, **JOB_KW)
        meta = {"kind": "refresh", "branch": "projects/proj/branches/t-x", "run_folder": NIGHTLY_WS}
        local = [(e.object_id, e.metadata) for e in mf.load_run_manifest(cfg).entries
                 if e.object_type == mf.ObjectType.JOB]
        assert local == [("77", meta)]
        copied = json.loads(w.workspace.files[f"{NIGHTLY_API}/{mf.MANIFEST_FILE}"])
        assert [(e["object_id"], e["metadata"]) for e in copied["entries"] if e["object_type"] == "job"] == [("77", meta)]

    def test_the_callers_config_is_never_changed(self, tmp_path):
        import copy

        from tests.refresh_support import fake_client

        cfg = self._local(tmp_path, preflight={"nested": [1, {"k": "v"}]})
        before = copy.deepcopy(cfg)
        w, _ = fake_client()
        job.create_refresh_job(cfg, w=w, **JOB_KW)
        assert cfg == before

    @pytest.mark.parametrize("name", ["flags.json", "schema.json", "graph_mapping.yaml", "privacy_decisions.json",
                                      "lgk_manifest.json"])
    def test_a_missing_required_file_is_refused_before_any_job_or_upload(self, tmp_path, name):
        from tests.refresh_support import fake_client

        cfg = self._local(tmp_path)
        (Path(cfg["run_folder"]) / name).unlink()
        w, calls = fake_client()
        with pytest.raises(job_nf().NightlyFolderError, match=re.escape(name)):
            job.create_refresh_job(cfg, w=w, **JOB_KW)
        assert calls == [] and w.workspace.log == []

    def test_a_required_name_that_is_a_directory_is_refused(self, tmp_path):
        from tests.refresh_support import fake_client

        cfg = self._local(tmp_path)
        run = Path(cfg["run_folder"])
        (run / "flags.json").unlink()
        (run / "flags.json").mkdir()  # hostile: the name exists, but it is no file to read
        w, calls = fake_client()
        with pytest.raises(job_nf().NightlyFolderError, match="flags.json"):
            job.create_refresh_job(cfg, w=w, **JOB_KW)
        assert calls == []

    @pytest.mark.parametrize("folder", ["absent", None, ""])
    def test_no_run_folder_is_refused_before_any_job(self, folder):
        from tests.refresh_support import fake_client

        cfg = _config()
        if folder == "absent":
            del cfg["run_folder"]
        else:
            cfg["run_folder"] = folder
        w, calls = fake_client()
        with pytest.raises(job_nf().NightlyFolderError, match="run_folder"):
            job.create_refresh_job(cfg, w=w, **JOB_KW)
        assert calls == [] and w.workspace.log == []

    def test_a_workspace_run_folder_is_used_as_is_and_nothing_is_copied(self, monkeypatch):
        from tests.refresh_support import fake_client

        recorded = []
        monkeypatch.setattr(job, "record_job", lambda *a, **k: recorded.append(k))
        w, calls = fake_client()
        job.create_refresh_job(_config(), w=w, **JOB_KW)
        assert {c["run_folder"] for c in _job_configs(calls)} == {f"{WS_HOME}/lgk_v1_abc123/run"}
        assert w.workspace.log == []
        assert recorded == [{"run_folder": f"{WS_HOME}/lgk_v1_abc123/run"}]

    def test_workspace_run_folder_key_chooses_the_copy(self, tmp_path):
        from tests.refresh_support import fake_client

        cfg = self._local(tmp_path, workspace_run_folder=f"{WS_HOME}/elsewhere/")
        w, calls = fake_client()
        job.create_refresh_job(cfg, w=w, **JOB_KW)
        assert {c["run_folder"] for c in _job_configs(calls)} == {f"{WS_HOME}/elsewhere"}
        assert f"{USER_HOME}/elsewhere/flags.json" in w.workspace.files

    @pytest.mark.parametrize("bad", ["", f"{USER_HOME}/x", "/tmp/x", "/Workspace", "/Workspace/", "Workspace/x"])
    def test_a_workspace_run_folder_outside_the_workspace_is_refused(self, tmp_path, bad):
        from tests.refresh_support import fake_client

        cfg = self._local(tmp_path, workspace_run_folder=bad)
        w, calls = fake_client()
        with pytest.raises(job_nf().NightlyFolderError, match="workspace_run_folder"):
            job.create_refresh_job(cfg, w=w, **JOB_KW)
        assert calls == [] and w.workspace.log == []

    def test_an_api_form_notebook_path_gives_a_workspace_copy_folder(self, tmp_path):
        from tests.refresh_support import fake_client

        cfg = self._local(tmp_path)
        w, calls = fake_client()
        job.create_refresh_job(cfg, notebook_path=f"{USER_HOME}/x/lgk_refresh_task", wheel_path=JOB_KW["wheel_path"], w=w)
        assert {c["run_folder"] for c in _job_configs(calls)} == {f"{WS_HOME}/x/runs/provider_graph"}
        assert f"{USER_HOME}/x/runs/provider_graph/flags.json" in w.workspace.files

    def test_an_optional_file_gone_from_the_laptop_is_removed_from_the_copy(self, tmp_path):
        from tests.refresh_support import fake_client

        cfg = self._local(tmp_path, enriched=False)
        w, calls = fake_client()
        # hostile: an earlier seeding left an enrich_report.json the sync would serve from
        stale = f"{NIGHTLY_API}/enrich_report.json"
        w.workspace.files[stale] = b'{"degree": {}}'
        job.create_refresh_job(cfg, w=w, **JOB_KW)
        assert stale not in w.workspace.files
        assert "enrich" not in [t.task_key for t in calls[0]["tasks"]]

    def test_creating_the_job_again_reseeds_the_copy(self, tmp_path):
        """The second create overwrites every file of the first seeding with the laptop's bytes."""
        from tests.refresh_support import fake_client

        cfg = self._local(tmp_path)
        w, calls = fake_client()
        job.create_refresh_job(cfg, w=w, **JOB_KW)
        (Path(cfg["run_folder"]) / "flags.json").write_text('{"flags": [], "kind": "flags v2"}')
        job.create_refresh_job(cfg, w=w, **JOB_KW)
        assert w.workspace.files[f"{NIGHTLY_API}/flags.json"] == b'{"flags": [], "kind": "flags v2"}'
        assert len(calls) == 2

    def test_a_workspace_run_folder_given_as_a_path_is_used_as_is(self, monkeypatch):
        from tests.refresh_support import fake_client

        monkeypatch.setattr(job, "record_job", lambda *a, **k: None)
        w, calls = fake_client()
        # hostile: a pathlib path, not a string (the laptop cannot see /Workspace, so seeding would refuse)
        job.create_refresh_job(_config(run_folder=Path(f"{WS_HOME}/lgk_v1_abc123/run")), w=w, **JOB_KW)
        assert {c["run_folder"] for c in _job_configs(calls)} == {f"{WS_HOME}/lgk_v1_abc123/run"}
        assert w.workspace.log == []

    def test_a_laptop_run_folder_given_as_a_path_is_seeded(self, tmp_path):
        from tests.refresh_support import fake_client

        cfg = self._local(tmp_path)
        cfg["run_folder"] = Path(cfg["run_folder"])
        w, calls = fake_client()
        job.create_refresh_job(cfg, w=w, **JOB_KW)
        assert {c["run_folder"] for c in _job_configs(calls)} == {NIGHTLY_WS}

    def test_an_optional_name_that_is_a_directory_is_not_copied(self, tmp_path):
        from tests.refresh_support import fake_client

        cfg = self._local(tmp_path, enriched=False)
        (Path(cfg["run_folder"]) / "enrich_report.json").mkdir()  # hostile: the name is a folder
        w, calls = fake_client()
        job.create_refresh_job(cfg, w=w, **JOB_KW)
        assert f"{NIGHTLY_API}/enrich_report.json" not in w.workspace.files

    def test_the_enrich_decision_reads_the_laptop_folder_not_the_copy_path(self, tmp_path):
        """The job's folder is a /Workspace path the laptop cannot see: enrich must still be chained."""
        from tests.refresh_support import fake_client

        w, calls = fake_client()
        job.create_refresh_job(self._local(tmp_path, enriched=True), w=w, **JOB_KW)
        assert [t.task_key for t in calls[0]["tasks"]] == ["build", "enrich", "sync", "reapply"]


def job_nf():
    from lgk.refresh import nightly_folder

    return nightly_folder


def test_the_refresh_notebook_upgrades_the_sdk_and_restarts_before_any_task_runs():
    """Live 2026-10-04: the job's sync task failed on the image's old SDK (no w.postgres)."""
    src = (Path(__file__).resolve().parents[1] / "notebooks" / "lgk_refresh_task.py").read_text()
    pip = src.index('# MAGIC %pip install --quiet --upgrade "databricks-sdk>=0.81"')
    restart = src.index("dbutils.library.restartPython()")
    run = src.index("from lgk.refresh.tasks import run_task")
    widgets = src.index('dbutils.widgets.text("task"')
    assert widgets < pip < restart < run
    # each in its own cell, as %pip and a restart must be
    cells = src.split("# COMMAND ----------")
    assert any(c.strip() == '# MAGIC %pip install --quiet --upgrade "databricks-sdk>=0.81"' for c in cells)


# ------------------------------------------- review round 2: no orphan job, a Path folder is recorded as str
@pytest.mark.parametrize("branch", ["absent", "", None])
def test_a_missing_branch_is_refused_before_any_job_or_upload(tmp_path, branch):
    """record_job reads the branch AFTER jobs.create: refusing there would leak an unrecorded job."""
    from lgk.branch import MissingBranchError
    from tests.refresh_support import fake_client, seed_run_folder

    seed_run_folder(tmp_path / "run")
    cfg = _config(run_folder=str(tmp_path / "run"))
    if branch == "absent":
        del cfg["branch_id"]
    else:
        cfg["branch_id"] = branch
    w, calls = fake_client()
    with pytest.raises(MissingBranchError, match="branch_id"):
        job.create_refresh_job(cfg, w=w, **JOB_KW)
    assert calls == [] and w.workspace.log == []


def test_a_workspace_run_folder_path_object_is_recorded_as_a_string(tmp_path, monkeypatch):
    """No record_job patch: the real manifest is written. In a notebook /Workspace is the local disk,
    so the workspace prefix is pointed at tmp_path to make this folder a workspace one, used as is."""
    from lgk import manifest as mf
    from lgk.refresh import nightly_folder as nf
    from tests.refresh_support import fake_client, seed_run_folder

    run = seed_run_folder(tmp_path / "ws" / "run")
    monkeypatch.setattr(nf, "WORKSPACE_PREFIX", f"{tmp_path}/ws/")
    cfg = _config(run_folder=run)  # hostile: a pathlib.Path, not a str
    w, calls = fake_client(job_id=5)
    job.create_refresh_job(cfg, w=w, **JOB_KW)
    assert w.workspace.log == []  # used as is, nothing copied
    saved = json.loads((run / mf.MANIFEST_FILE).read_text())
    [entry] = [e for e in saved["entries"] if e["object_type"] == "job"]
    assert entry["metadata"]["run_folder"] == str(run)
    assert {c["run_folder"] for c in _job_configs(calls)} == {str(run)}
