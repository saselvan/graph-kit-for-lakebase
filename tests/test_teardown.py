"""Offline tests for whole-graph teardown, the run manifest and the reaper.

Live behaviour (real drop order, markers, hostile objects on a branch) is in
tests/test_teardown_live.py. Here the planner, the executor's phase order and the
reaper's orphan rule are pinned with fakes at the cloud boundary only.
"""

import json
from typing import Any, List

import pytest

from lgk import manifest as mf
from lgk import versions
from lgk.manifest import ObjectType

GRAPH = "td_graph"
CAT = "cat"
BRANCH = "projects/p/branches/t-live"
GONE = "projects/p/branches/t-gone"


def _config(tmp_path, **kw):
    return {"graph_name": GRAPH, "run_folder": str(tmp_path), "lakebase_project": "p",
            "lakebase_catalog": CAT, "branch_id": "t-live", "kit_schema": "lgk_k", "warehouse_id": "wh", **kw}


def _record_full_graph(config):
    """What build + provision record for one graph, in creation order."""
    pg = f"postgres:{BRANCH}"
    mf.record(config, ObjectType.UC_SCHEMA, f"{CAT}.lgk_k", "uc")
    for t in ("nodes", "edges", "node_id_map"):
        mf.record(config, ObjectType.UC_TABLE, f"{CAT}.lgk_k.{t}", "uc")
    mf.record(config, ObjectType.UC_SCHEMA, f"{CAT}.lgk_k_v1", "uc")
    for t in ("nodes", "edges", "node_id_map"):
        mf.record(config, ObjectType.SYNCED_TABLE, f"{CAT}.lgk_k_v1.{t}", "uc", {"branch": BRANCH})
    mf.record(config, ObjectType.PG_SCHEMA, "lgk_k_v1", pg)
    mf.record(config, ObjectType.PG_SCHEMA, "lgk_k", pg)
    for v in ("nodes_view", "edges_view", "node_id_map_view"):
        mf.record(config, ObjectType.VIEW, f"lgk_k.{v}", pg)
    # hostile: a manifest from the pin (faf7e73) recorded the shared roles as graph-owned
    for role in ("reader", "public_reader", "coordinator_reader"):
        mf.record(config, ObjectType.ROLE, role, pg)
    # recorded LAST (the refresh job is made after provision), yet it must be deleted FIRST
    mf.record(config, ObjectType.JOB, "777", "jobs", {"kind": "refresh", "branch": BRANCH})


# ============================================================================
# Manifest record / load
# ============================================================================


def test_record_appends_once_per_object_and_survives_the_harness_manifest(tmp_path):
    from lgk.harness import BranchLeasor

    config = _config(tmp_path)
    mf.record(config, ObjectType.UC_SCHEMA, f"{CAT}.lgk_k", "uc")
    # hostile: the same object recorded twice (a re-run of build) must appear once.
    mf.record(config, ObjectType.UC_SCHEMA, f"{CAT}.lgk_k", "uc")
    # hostile: the harness keeps ITS manifest.json in the same run folder and rewrites it
    # from memory; the kit's manifest must be a different file, so neither clobbers the other.
    leasor = BranchLeasor(workspace_client=None, project_id="p", test_base_branch="b", run_folder=tmp_path)
    leasor._save_manifest()
    mf.record(config, ObjectType.ROLE, "reader", f"postgres:{BRANCH}")

    loaded = mf.load_run_manifest(config)
    assert [(e.object_type, e.object_id) for e in loaded.entries] == [
        (ObjectType.UC_SCHEMA, f"{CAT}.lgk_k"),
        (ObjectType.ROLE, "reader"),
    ]
    assert json.loads((tmp_path / "manifest.json").read_text())["graph_name"] == "test_harness"


def test_record_refuses_a_run_folder_of_another_graph(tmp_path):
    mf.record(_config(tmp_path), ObjectType.ROLE, "reader", "pg")
    with pytest.raises(versions.NotOursError):
        mf.record(_config(tmp_path, graph_name="other_graph"), ObjectType.ROLE, "reader", "pg")
    with pytest.raises(versions.NotOursError):
        mf.load_run_manifest(_config(tmp_path, graph_name="other_graph"))


def test_marker_holds_the_exact_graph_name():
    assert mf.marker("td_graph") == "lgk:graph=td_graph"
    assert mf.marker("td_graph") != mf.marker("td_graph_2")


# ============================================================================
# Plan: manifest -> ordered phases
# ============================================================================


def test_plan_takes_exactly_the_manifest_entries_grouped_by_phase(tmp_path):
    from lgk.steps.teardown import plan_from_manifest

    config = _config(tmp_path)
    _record_full_graph(config)
    mf.record(config, ObjectType.BRANCH, "t-live", "lakebase")
    plan = plan_from_manifest(mf.load_run_manifest(config))
    assert plan.views == ["lgk_k.nodes_view", "lgk_k.edges_view", "lgk_k.node_id_map_view"]
    assert plan.synced_tables == [f"{CAT}.lgk_k_v1.{t}" for t in ("nodes", "edges", "node_id_map")]
    # The Postgres tables behind the synced tables live in the same-named PG schema.
    assert plan.pg_tables == [f"lgk_k_v1.{t}" for t in ("nodes", "edges", "node_id_map")]
    # Served schema (views) drops before the version schema.
    assert plan.pg_schemas == ["lgk_k", "lgk_k_v1"]
    assert not hasattr(plan, "roles")  # the shared roles are never a drop target (A6 (a))
    assert plan.branches == ["t-live"]
    assert plan.uc_tables == [f"{CAT}.lgk_k.{t}" for t in ("nodes", "edges", "node_id_map")]
    assert sorted(plan.uc_schemas) == [f"{CAT}.lgk_k", f"{CAT}.lgk_k_v1"]


def test_plan_ignores_the_project_and_unrecorded_objects(tmp_path):
    from lgk.steps.teardown import plan_from_manifest

    config = _config(tmp_path)
    # hostile: only one view recorded; a "prefix" planner would add every lgk_k* object.
    mf.record(config, ObjectType.VIEW, "lgk_k.nodes_view", f"postgres:{BRANCH}")
    plan = plan_from_manifest(mf.load_run_manifest(config))
    assert plan.views == ["lgk_k.nodes_view"]
    assert plan.synced_tables == plan.pg_schemas == plan.uc_schemas == plan.branches == []


# ============================================================================
# Executor: confirm, every check before any delete, verified phase order
# ============================================================================


class FakeOps:
    def __init__(self, fail_at: str = "", error: Exception = None):
        self.calls: List[str] = []
        self.fail_at = fail_at
        self.error = error or RuntimeError(f"boom at {fail_at}")

    def _do(self, name: str, *_: Any) -> None:
        self.calls.append(name)
        if name == self.fail_at:
            raise self.error

    def resolve_event_logs(self, synced_pipelines):
        self._do("resolve")
        return []

    def check(self, plan, graph):
        self._do("check")

    def delete_jobs(self, job_ids):
        self._do("jobs")

    def revoke_shared_roles(self, schemas, views, functions):
        self._do("revoke")

    def drop_views(self, views, functions):
        self._do("views")

    def delete_synced_tables(self, names):
        self._do("synced_tables")

    def drop_event_logs(self, tables):
        self._do("event_logs")

    def drop_pg(self, tables, schemas):
        self._do("pg_schemas")

    def delete_branches(self, branches):
        self._do("branches")

    def drop_uc(self, tables, schemas):
        self._do("uc")

    def close(self):
        self.calls.append("close")


def test_teardown_runs_the_r11_order(tmp_path):
    from lgk.steps.teardown import teardown

    config = _config(tmp_path)
    _record_full_graph(config)
    ops = FakeOps()
    plan = teardown(config, GRAPH, ops=ops)
    assert ops.calls == ["resolve", "check", "jobs", "revoke", "views", "synced_tables", "event_logs",
                         "pg_schemas", "branches", "uc", "close"]
    assert plan.jobs == ["777"]


@pytest.mark.parametrize("typed", ["TD_GRAPH", " td_graph", "td_graph ", "td_grap", ""])
def test_teardown_needs_the_exact_graph_name(tmp_path, typed):
    from lgk.steps.teardown import teardown

    config = _config(tmp_path)
    _record_full_graph(config)
    ops = FakeOps()
    with pytest.raises(versions.ConfirmationError):
        teardown(config, typed, ops=ops)
    assert ops.calls == []


def test_a_failed_check_deletes_nothing(tmp_path):
    from lgk.steps.teardown import teardown

    config = _config(tmp_path)
    _record_full_graph(config)
    ops = FakeOps(fail_at="check", error=versions.NotOursError("stranger"))
    with pytest.raises(versions.NotOursError):
        teardown(config, GRAPH, ops=ops)
    assert ops.calls == ["resolve", "check", "close"]


def test_the_original_error_is_raised_and_later_phases_do_not_run(tmp_path):
    from lgk.steps.teardown import teardown

    config = _config(tmp_path)
    _record_full_graph(config)
    original = RuntimeError("dependent objects exist")
    ops = FakeOps(fail_at="synced_tables", error=original)
    with pytest.raises(RuntimeError) as info:
        teardown(config, GRAPH, ops=ops)
    assert info.value is original
    assert ops.calls == ["resolve", "check", "jobs", "revoke", "views", "synced_tables", "close"]


def test_teardown_without_a_manifest_refuses(tmp_path):
    from lgk.steps.teardown import teardown

    with pytest.raises(FileNotFoundError):
        teardown(_config(tmp_path), GRAPH, ops=FakeOps())


# ============================================================================
# Jobs: tag-checked before any delete, cancelled + deleted first
# ============================================================================


class FakeJobs:
    """The jobs API boundary. `jobs` = existing job id -> its tags."""

    def __init__(self, jobs: dict, gone_error: str = "not_found"):
        self.jobs, self.gone_error, self.calls = dict(jobs), gone_error, []

    def _missing(self, job_id):
        from databricks.sdk.errors import InvalidParameterValue, NotFound

        if self.gone_error == "not_found":
            return NotFound(f"Job {job_id} not found")
        return InvalidParameterValue(f"Job {job_id} does not exist.")

    def get(self, job_id):
        if job_id not in self.jobs:
            raise self._missing(job_id)
        return type("J", (), {"settings": type("S", (), {"tags": self.jobs[job_id]})()})()

    def cancel_all_runs(self, job_id):
        self.calls.append(("cancel", job_id))

    def delete(self, job_id):
        self.calls.append(("delete", job_id))
        self.jobs.pop(job_id)


def _jobs_w(jobs, **kw):
    return type("W", (), {"jobs": FakeJobs(jobs, **kw)})()


def test_a_job_tagged_for_another_graph_is_refused():
    from lgk.steps.teardown import Plan, _check_jobs

    # hostile: a copied manifest names a job of another graph; one without tags too
    w = _jobs_w({1: {"lgk_graph": GRAPH}, 2: {"lgk_graph": "td_other"}, 3: None})
    with pytest.raises(versions.NotOursError, match="job 2"):
        _check_jobs(w, Plan(jobs=["1", "2"]), GRAPH)
    with pytest.raises(versions.NotOursError, match="job 3"):
        _check_jobs(w, Plan(jobs=["3"]), GRAPH)
    _check_jobs(w, Plan(jobs=["1", "404"]), GRAPH)  # a job already gone is fine
    assert w.jobs.calls == []


@pytest.mark.parametrize("gone_error", ["not_found", "does_not_exist"])
def test_delete_jobs_cancels_runs_then_deletes_and_is_rerunnable(gone_error):
    from lgk.steps.teardown import delete_jobs

    w = _jobs_w({5: {"lgk_graph": GRAPH}, 6: {"lgk_graph": GRAPH}}, gone_error=gone_error)
    delete_jobs(w, ["5", "404", "6"])
    assert w.jobs.calls == [("cancel", 5), ("delete", 5), ("cancel", 6), ("delete", 6)]
    delete_jobs(w, ["5", "6"])  # both already gone: a no-op, not an error
    assert len(w.jobs.calls) == 4


def test_live_check_runs_the_job_tag_check(tmp_path, monkeypatch):
    from lgk.steps import teardown as td

    config = _config(tmp_path)
    mf.record(config, ObjectType.JOB, "9", "jobs", {"kind": "refresh", "branch": BRANCH})
    ops = td.LiveOps.__new__(td.LiveOps)
    ops.config, ops.w, ops._conn, ops._branch_alive = config, _jobs_w({9: {"lgk_graph": "td_other"}}), None, None
    monkeypatch.setattr(td, "_check_uc_schemas", lambda w, plan, graph: None)
    with pytest.raises(versions.NotOursError, match="job 9"):
        td.teardown(config, GRAPH, ops=ops)
    assert ops.w.jobs.calls == []  # nothing cancelled or deleted


def test_reaper_deletes_a_recorded_job_whose_branch_is_gone_before_the_tables(tmp_path):
    from lgk.reaper import OrphanJob, reap

    config = _config(tmp_path)
    mf.record(config, ObjectType.JOB, "31", "jobs", {"kind": "refresh", "branch": GONE})
    # hostile: a job syncing a LIVE branch, and a recorded job already deleted
    mf.record(config, ObjectType.JOB, "32", "jobs", {"kind": "refresh", "branch": BRANCH})
    mf.record(config, ObjectType.JOB, "33", "jobs", {"kind": "refresh", "branch": GONE})
    w = FakeW(synced={}, uc_tables=set(), pipelines=set(), branches={BRANCH})
    w.jobs = FakeJobs({31: {"lgk_graph": GRAPH}, 32: {"lgk_graph": GRAPH}})
    assert reap(config, GRAPH, w=w) == [OrphanJob("31", GONE)]
    assert w.jobs.calls == [("cancel", 31), ("delete", 31)]
    assert set(w.jobs.jobs) == {32}
    assert reap(config, GRAPH, w=w) == []


def test_reaper_refuses_an_orphan_job_of_another_graph_and_deletes_nothing(tmp_path):
    from lgk.reaper import reap

    config = _config(tmp_path)
    mf.record(config, ObjectType.JOB, "31", "jobs", {"kind": "refresh", "branch": GONE})
    mf.record(config, ObjectType.SYNCED_TABLE, f"{CAT}.s.t", "uc", {"branch": GONE, "pipeline_id": "p-t"})
    w = FakeW(synced={f"synced_tables/{CAT}.s.t": GONE}, uc_tables={f"{CAT}.s.t"}, pipelines={"p-t"},
              branches={BRANCH})
    w.jobs = FakeJobs({31: {"lgk_graph": "someone_else"}})  # hostile: a copied manifest
    with pytest.raises(versions.NotOursError, match="job 31"):
        reap(config, GRAPH, w=w)
    assert w.jobs.calls == [] and w.deleted == []


def test_reaper_deletes_the_orphan_job_before_any_synced_table(tmp_path):
    from lgk.reaper import reap

    config = _config(tmp_path)
    mf.record(config, ObjectType.SYNCED_TABLE, f"{CAT}.s.t", "uc", {"branch": GONE, "pipeline_id": "p-t"})
    mf.record(config, ObjectType.JOB, "31", "jobs", {"kind": "refresh", "branch": GONE})  # recorded last
    w = FakeW(synced={f"synced_tables/{CAT}.s.t": GONE}, uc_tables={f"{CAT}.s.t"}, pipelines={"p-t"},
              branches={BRANCH})
    w.jobs = FakeJobs({31: {"lgk_graph": GRAPH}})
    log = w.deleted  # one shared log: the job delete lands in it too
    real_delete = w.jobs.delete
    w.jobs.delete = lambda job_id: (log.append(f"job:{job_id}"), real_delete(job_id))
    reap(config, GRAPH, w=w)
    assert log[0] == "job:31" and len(log) == 4, log


# ============================================================================
# Reaper (offline): only synced tables whose branch is gone
# ============================================================================


def _nf(msg):
    from databricks.sdk.errors import NotFound

    return NotFound(msg)


class FakeW:
    """Cloud boundary for the reaper. `synced` = GET-able synced tables -> bound branch."""

    def __init__(self, synced: dict, uc_tables: set, pipelines: set, branches: set):
        self.synced, self.uc_tables, self.pipeline_ids, self.branches = synced, uc_tables, pipelines, branches
        self.deleted: List[str] = []
        w = self

        class _Pg:
            def get_synced_table(self, name):
                if name not in w.synced:
                    raise _nf(f"Branch with ID br-x not found ({name})")
                return type("T", (), {"spec": type("S", (), {"branch": w.synced[name]})()})()

            def get_branch(self, name):
                if name not in w.branches:
                    raise _nf(f"branch {name} not found")
                return object()

            def delete_synced_table(self, name):
                if name not in w.synced:
                    raise _nf(name)
                w.synced.pop(name)
                w.deleted.append(name)
                return type("Op", (), {"wait": lambda self: None})()

        class _Tables:
            def get(self, full):
                if full not in w.uc_tables:
                    raise _nf(full)

            def delete(self, full):
                if full not in w.uc_tables:
                    raise _nf(full)
                w.uc_tables.remove(full)
                w.deleted.append(f"uc:{full}")

        class _Pipelines:
            def get(self, pid):
                if pid not in w.pipeline_ids:
                    raise _nf(pid)
                return type("P", (), {"state": "IDLE"})()

            def delete(self, pid):
                if pid not in w.pipeline_ids:
                    raise _nf(pid)
                w.pipeline_ids.remove(pid)
                w.deleted.append(f"pipeline:{pid}")

        self.postgres, self.tables, self.pipelines = _Pg(), _Tables(), _Pipelines()


def test_reaper_finds_only_synced_tables_whose_branch_is_gone(tmp_path):
    from lgk.reaper import find_orphans, reap

    config = _config(tmp_path)
    tables = ("orphan_404", "orphan_get", "live", "already_gone", "sparse_orphan")
    for t in tables:
        branch = BRANCH if t == "live" else GONE
        mf.record(config, ObjectType.SYNCED_TABLE, f"{CAT}.s.{t}", "uc", {"branch": branch, "pipeline_id": f"p-{t}"})
    st = lambda t: f"synced_tables/{CAT}.s.{t}"  # noqa: E731
    pids = {"p-orphan_404", "p-orphan_get", "p-live", "p-sparse_orphan"}
    w = FakeW(
        synced={
            st("orphan_get"): GONE,
            # hostile: bound to a live branch — reaping it would break a running graph.
            st("live"): BRANCH,
            # hostile: the GET is sparse (no spec.branch); fall back to the recorded branch.
            st("sparse_orphan"): None,
            # hostile: a synced table on a deleted branch that the manifest never recorded.
            st("stranger"): GONE,
        },
        # hostile (observed live): orphan_404's GET is NotFound once its branch is
        # gone, but its UC table and pipeline remain. "NotFound = gone" would miss it.
        uc_tables={f"{CAT}.s.{t}" for t in ("orphan_404", "orphan_get", "live", "sparse_orphan")},
        pipelines=set(pids),
        branches={BRANCH},
    )
    orphans = find_orphans(config, w=w)
    expected = [f"{CAT}.s.{t}" for t in ("orphan_404", "orphan_get", "sparse_orphan")]
    assert sorted(o.synced_table for o in orphans) == expected
    o404 = next(o for o in orphans if o.synced_table.endswith("orphan_404"))
    assert (o404.synced_get, o404.uc_table, o404.pipeline_id) == (False, True, "p-orphan_404")

    with pytest.raises(versions.ConfirmationError):
        reap(config, "TD_GRAPH", w=w)
    assert w.deleted == []
    reap(config, GRAPH, w=w)
    assert w.uc_tables == {f"{CAT}.s.live"}
    assert w.pipeline_ids == {"p-live"}
    assert set(w.synced) == {st("live"), st("stranger")}
    # Re-runnable: nothing left to reap.
    assert reap(config, GRAPH, w=w) == []


# ============================================================================
# retire: every check before any delete (review ordering bug)
# ============================================================================


class _FakeConn:
    autocommit = False

    def close(self):
        pass


def test_retire_checks_outside_dependents_before_deleting_any_synced_table(tmp_path, monkeypatch):
    from lgk.steps import retire as rt

    config = _config(tmp_path, version=2)
    mf.record(config, ObjectType.SYNCED_TABLE, f"{CAT}.lgk_k_v1.nodes", "uc", {"branch": BRANCH})
    deleted: List[str] = []
    monkeypatch.setattr(rt, "connect_branch", lambda c: _FakeConn())
    monkeypatch.setattr(versions, "live_version_schemas", lambda conn, served: set())
    monkeypatch.setattr(versions, "refuse_unless_ours_pg", lambda conn, schema: None)
    monkeypatch.setattr(rt, "_check_uc_version", lambda c, schema, recorded, logs: None)
    monkeypatch.setattr(rt, "_own_event_logs", lambda c, schema, recorded: [])
    # hostile: a view in another schema reads v1 — dropping v1 RESTRICT would fail, but only
    # AFTER the synced tables were deleted, leaving v1 half-deleted.
    monkeypatch.setattr(versions, "outside_dependents", lambda conn, schemas: ["other.reads_v1"])
    monkeypatch.setattr(rt, "_delete_synced_tables", lambda names: deleted.extend(names))
    with pytest.raises(versions.NotOursError):
        rt.retire(config, version=1, confirm_graph_name=GRAPH)
    assert deleted == []


class _CountConn(_FakeConn):
    """Answers the synced-table owner count query with a fixed number."""

    def __init__(self, n):
        self.n = n

    def execute(self, sql, params=None):
        from types import SimpleNamespace

        assert "databricks\\_writer" in sql  # the count is of sync-owned tables only
        return SimpleNamespace(fetchone=lambda: (self.n,))


@pytest.mark.parametrize("synced_in_pg, recorded, refused", [
    (3, False, True),   # hostile (Isaac Review): no manifest record, real synced tables -> refuse
    (3, True, False),   # recorded -> the synced tables are deleted first, so allowed
    (0, False, False),  # plain hand-built tables only -> nothing can be orphaned
])
def test_retire_without_a_manifest_record_refuses_to_orphan_synced_tables(
    tmp_path, monkeypatch, synced_in_pg, recorded, refused
):
    from lgk.steps import retire as rt

    config = _config(tmp_path, version=2)
    if recorded:
        mf.record(config, ObjectType.SYNCED_TABLE, f"{CAT}.lgk_k_v1.nodes", "uc", {"branch": BRANCH})
    deleted: List[str] = []
    dropped: List[str] = []
    monkeypatch.setattr(rt, "connect_branch", lambda c: _CountConn(synced_in_pg))
    monkeypatch.setattr(versions, "live_version_schemas", lambda conn, served: set())
    monkeypatch.setattr(versions, "refuse_unless_ours_pg", lambda conn, schema: None)
    monkeypatch.setattr(versions, "outside_dependents", lambda conn, schemas: [])
    monkeypatch.setattr(rt, "_check_uc_version", lambda c, schema, recorded, logs: None)
    monkeypatch.setattr(rt, "_own_event_logs", lambda c, schema, recorded: [])
    monkeypatch.setattr(rt, "_delete_synced_tables", lambda names: deleted.extend(names))
    monkeypatch.setattr(rt, "_drop_pg_schema", lambda conn, served, schema: dropped.append(schema))
    monkeypatch.setattr(rt, "_drop_uc_schema", lambda c, schema: None)
    monkeypatch.setattr(rt, "_drop_event_logs", lambda c, tables: None)
    if refused:
        with pytest.raises(versions.NotOursError, match="orphaned"):
            rt.retire(config, version=1, confirm_graph_name=GRAPH)
        assert deleted == [] and dropped == []
    else:
        rt.retire(config, version=1, confirm_graph_name=GRAPH)
        assert dropped == ["lgk_k_v1"]


def test_uc_only_teardown_never_connects_to_any_branch(tmp_path, monkeypatch):
    """A build/enrich-only manifest (no branch_id) must not open Postgres on the default branch."""
    from types import SimpleNamespace

    from lgk.steps import teardown as td

    config = _config(tmp_path)
    del config["branch_id"]  # hostile: no branch_id, so _pg() would fall back to `production`
    mf.record(config, ObjectType.UC_SCHEMA, "c.lgk_v1_abc123", "uc")
    mf.record(config, ObjectType.UC_TABLE, "c.lgk_v1_abc123.node_score", "uc")
    marker = mf.marker(config["graph_name"])
    w = SimpleNamespace(
        schemas=SimpleNamespace(get=lambda full: SimpleNamespace(comment=marker)),
        tables=SimpleNamespace(list=lambda c, s: [SimpleNamespace(name="node_score")]),
    )
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: w)
    dropped: List[str] = []
    monkeypatch.setattr("lgk.uc_sql.execute_sql", lambda w_, wh, s, **k: dropped.append(s) or [])

    def no_pg(self):
        raise AssertionError("connected to Postgres for a UC-only manifest")

    monkeypatch.setattr(td.LiveOps, "_pg", no_pg)
    td.teardown(config, config["graph_name"])
    assert dropped == [
        "DROP TABLE IF EXISTS c.lgk_v1_abc123.node_score",
        "DROP SCHEMA IF EXISTS c.lgk_v1_abc123 RESTRICT",
    ]


# ---------------------------------------------------------------- teardown's own branch delete: operation cap
class _BranchApi:
    """Records each REST call; answers from a script (an exception is raised, anything else returned)."""

    def __init__(self, script: List[Any]):
        self.script, self.calls = list(script), []

    def do(self, method, path, **kw):
        self.calls.append((method, path))
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _branch_ops(tmp_path, script: List[Any]):
    from types import SimpleNamespace

    from lgk.steps import teardown as td

    ops = td.LiveOps.__new__(td.LiveOps)
    ops.config, ops._conn, ops._branch_alive = _config(tmp_path), None, None
    ops.w = SimpleNamespace(api_client=_BranchApi(script))
    return ops


def _cap():
    from databricks.sdk.errors.platform import Aborted

    return Aborted("Your project has too many operations in progress. Please wait until they are complete, "
                   "and then try again. [TraceId: 0f]")


def test_teardown_retries_its_branch_delete_on_the_operation_cap(tmp_path, monkeypatch):
    slept: List[float] = []
    monkeypatch.setattr("lgk.branch.time.sleep", slept.append)
    ops = _branch_ops(tmp_path, [_cap(), _cap(), {}, {}])
    # hostile: two branches; the first hits the cap twice, the second goes at once
    ops.delete_branches(["t-a", "t-b"])
    assert ops.w.api_client.calls == [("DELETE", "/api/2.0/postgres/projects/p/branches/t-a")] * 3 + [
        ("DELETE", "/api/2.0/postgres/projects/p/branches/t-b")]
    assert slept == [10, 20]  # backoff grows with each try


@pytest.mark.parametrize("error", [
    "aborted-other",  # hostile: an Aborted that is not the cap
    "cap-text-oserror",  # hostile: the cap text on another error class
    "permission",
])
def test_teardown_does_not_retry_other_branch_delete_errors(tmp_path, monkeypatch, error):
    from databricks.sdk.errors.platform import Aborted, PermissionDenied

    err = {"aborted-other": Aborted("Operation was aborted by the server"),
           "cap-text-oserror": OSError("too many operations in progress"),
           "permission": PermissionDenied("no CAN MANAGE on the project")}[error]
    monkeypatch.setattr("lgk.branch.time.sleep", lambda s: None)
    ops = _branch_ops(tmp_path, [err, {}])
    with pytest.raises(type(err)):
        ops.delete_branches(["t-a"])
    assert len(ops.w.api_client.calls) == 1


def test_teardown_gives_up_after_the_cap_tries_and_a_gone_branch_is_not_an_error(tmp_path, monkeypatch):
    from databricks.sdk.errors import NotFound
    from databricks.sdk.errors.platform import Aborted

    monkeypatch.setattr("lgk.branch.time.sleep", lambda s: None)
    ops = _branch_ops(tmp_path, [_cap()] * 6)
    with pytest.raises(Aborted, match="too many operations"):
        ops.delete_branches(["t-a"])
    assert len(ops.w.api_client.calls) == 6
    ops = _branch_ops(tmp_path, [NotFound("branch t-a not found"), {}])
    ops.delete_branches(["t-a", "t-b"])  # already gone: logged, and the next branch still goes
    assert len(ops.w.api_client.calls) == 2


# ============================================================================
# P4: shared roles (A6 (a)), grant revokes, event logs, marker guards (SQL-level fakes)
# ============================================================================

A_PIPELINE = "-".join(["a1b2c3d4", "0000", "4000", "8000", "0000000000aa"])
B_PIPELINE = "-".join(["0d1e2f30", "0000", "4000", "8000", "00000000beef"])


class FakePg:
    """Scripted branch Postgres: which roles, schemas (with markers), relations, views and functions exist.

    Logs every statement. `relations[schema]` = the relations pg_class lists in it.
    """

    def __init__(self, roles, schemas, relations=None, regclass=(), regproc=()):
        self.roles, self.schemas = set(roles), dict(schemas)
        self.relations = relations or {}
        self.regclass, self.regproc = set(regclass), set(regproc)
        self.sql: List[str] = []
        self.autocommit = True

    def execute(self, sql, params=None):
        flat = " ".join(sql.split())
        self.sql.append(flat)
        p = params[0] if params else None
        if "FROM pg_roles WHERE rolname" in flat:
            return _Rows([(1,)] if p in self.roles else [])
        if "obj_description(oid, 'pg_namespace')" in flat:
            return _Rows([(self.schemas[p],)] if p in self.schemas else [])
        if "FROM pg_namespace WHERE nspname" in flat:
            return _Rows([(1,)] if p in self.schemas else [])
        if "to_regclass" in flat:
            return _Rows([(p if p in self.regclass else None,)])
        if "to_regprocedure" in flat:
            return _Rows([(p if p in self.regproc else None,)])
        if "c.relkind IN" in flat and "relispartition" in flat:
            return _Rows([(r,) for r in self.relations.get(p, [])])
        if sql == versions._OUTSIDE_DEPENDENTS_SQL:
            return _Rows([(d,) for d in getattr(self, "outside", [])])
        return _Rows([])  # outside_dependents and the like: nothing

    def transaction(self):
        import contextlib

        return contextlib.nullcontext()

    def close(self):
        pass

    def of(self, verb):
        return [s for s in self.sql if s.upper().startswith(verb)]


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


def _two_graph_branch(config):
    """Graph A (td_graph, schemas lgk_k / lgk_k_v1) and graph B (lgk_b / lgk_b_v1) on one branch."""
    a, b = mf.marker(GRAPH), mf.marker("graph_b")
    return FakePg(
        roles={"reader", "public_reader", "coordinator_reader"},
        schemas={"lgk_k": a, "lgk_k_v1": a, "lgk_b": b, "lgk_b_v1": b},
        relations={"lgk_k": [f"lgk_k.{v}" for v in ("nodes_view", "edges_view", "node_id_map_view")],
                   "lgk_k_v1": [f"lgk_k_v1.{t}" for t in TABLES_]},
        regclass={f"lgk_k.{v}" for v in ("nodes_view", "edges_view", "node_id_map_view")} | {"lgk_b.nodes_view"},
        regproc={"lgk_k.directory_provider(text)", "lgk_b.directory_provider(text)"},
    )


TABLES_ = ("nodes", "edges", "node_id_map")


def _record_graph_with_pipelines(config, blank=()):
    """_record_full_graph, but the synced tables carry their pipeline ids (as provision records them).

    Tables in `blank` are recorded with pipeline_id None (provision's status had no id).
    """
    pg = f"postgres:{BRANCH}"
    mf.record(config, ObjectType.UC_SCHEMA, f"{CAT}.lgk_k", "uc")
    mf.record(config, ObjectType.UC_SCHEMA, f"{CAT}.lgk_k_v1", "uc")
    for i, t in enumerate(TABLES_):
        mf.record(config, ObjectType.SYNCED_TABLE, f"{CAT}.lgk_k_v1.{t}", "uc",
                  {"branch": BRANCH, "pipeline_id": None if t in blank else A_PIPELINE[:-1] + str(i)})
    mf.record(config, ObjectType.PG_SCHEMA, "lgk_k_v1", pg)
    mf.record(config, ObjectType.PG_SCHEMA, "lgk_k", pg)
    for v in ("nodes_view", "edges_view", "node_id_map_view"):
        mf.record(config, ObjectType.VIEW, f"lgk_k.{v}", pg)
    mf.record(config, ObjectType.FUNCTION, "lgk_k.directory_provider(text)", pg)
    # hostile: the pin's manifests recorded all three shared roles as this graph's
    for role in ("reader", "public_reader", "coordinator_reader"):
        mf.record(config, ObjectType.ROLE, role, pg)


class _FakeSynced:
    def __init__(self):
        self.deleted: List[str] = []

    def delete_synced_table(self, name):
        self.deleted.append(name)
        return type("Op", (), {"wait": lambda self: None})()


def _live_ops(tmp_path, monkeypatch, pg, uc_tables=None):
    """LiveOps with the branch connection and the UC/Postgres APIs faked; returns (ops, uc_sql log)."""
    from types import SimpleNamespace

    from lgk.steps import teardown as td

    ops = td.LiveOps.__new__(td.LiveOps)
    ops.config, ops._conn, ops._branch_alive, ops.warehouse_id = _config(tmp_path), pg, True, "wh"
    marker = mf.marker(GRAPH)
    tables = uc_tables if uc_tables is not None else {}
    ops.w = SimpleNamespace(
        schemas=SimpleNamespace(get=lambda full: SimpleNamespace(comment=marker)),
        tables=SimpleNamespace(list=lambda c, s: [SimpleNamespace(name=n) for n in tables.get(f"{c}.{s}", [])]),
        postgres=_FakeSynced(),
    )
    uc_sql: List[str] = []
    monkeypatch.setattr("lgk.uc_sql.execute_sql", lambda w_, wh, s, **k: uc_sql.append(s) or [])
    return ops, uc_sql


def test_teardown_of_an_old_manifest_never_drops_a_shared_role(tmp_path, monkeypatch):
    from lgk.steps.teardown import teardown

    config = _config(tmp_path)
    _record_graph_with_pipelines(config)
    pg = _two_graph_branch(config)
    ops, _ = _live_ops(tmp_path, monkeypatch, pg)
    teardown(config, GRAPH, ops=ops)
    assert not [s for s in pg.sql if "DROP ROLE" in s.upper()]
    assert not [s for s in pg.sql if "ALTER ROLE" in s.upper() or "COMMENT ON ROLE" in s.upper()]
    # it did drop the graph's own Postgres objects (so the run reached the drop phase)
    assert "DROP SCHEMA IF EXISTS lgk_k RESTRICT" in pg.sql and "DROP SCHEMA IF EXISTS lgk_k_v1 RESTRICT" in pg.sql


def test_teardown_revokes_the_shared_roles_on_exactly_this_graphs_objects(tmp_path, monkeypatch):
    from lgk.steps.teardown import teardown

    config = _config(tmp_path)
    _record_graph_with_pipelines(config)
    pg = _two_graph_branch(config)
    ops, _ = _live_ops(tmp_path, monkeypatch, pg)
    teardown(config, GRAPH, ops=ops)
    who = "reader, public_reader, coordinator_reader"
    # written out by hand: the served schema (views) first, then the version schema, then each object
    assert pg.of("REVOKE") == [
        f"REVOKE ALL ON SCHEMA lgk_k FROM {who}",
        f"REVOKE ALL ON ALL TABLES IN SCHEMA lgk_k FROM {who}",
        f"REVOKE ALL ON ALL FUNCTIONS IN SCHEMA lgk_k FROM {who}",
        f"REVOKE ALL ON ALL SEQUENCES IN SCHEMA lgk_k FROM {who}",
        f"REVOKE ALL ON SCHEMA lgk_k_v1 FROM {who}",
        f"REVOKE ALL ON ALL TABLES IN SCHEMA lgk_k_v1 FROM {who}",
        f"REVOKE ALL ON ALL FUNCTIONS IN SCHEMA lgk_k_v1 FROM {who}",
        f"REVOKE ALL ON ALL SEQUENCES IN SCHEMA lgk_k_v1 FROM {who}",
        f"REVOKE ALL ON lgk_k.nodes_view FROM {who}",
        f"REVOKE ALL ON lgk_k.edges_view FROM {who}",
        f"REVOKE ALL ON lgk_k.node_id_map_view FROM {who}",
        f"REVOKE ALL ON FUNCTION lgk_k.directory_provider(text) FROM {who}",
    ]
    # hostile: graph B's schemas, view and function exist with the same names on the same branch
    assert not [s for s in pg.sql if "lgk_b" in s]
    # the revokes run before any drop
    first_drop = next(i for i, s in enumerate(pg.sql) if s.startswith("DROP"))
    assert max(i for i, s in enumerate(pg.sql) if s.startswith("REVOKE")) < first_drop


def test_revoke_skips_a_missing_role_and_a_missing_object():
    from lgk.steps.teardown import revoke_shared_sql

    # hostile: public_reader was never created on this branch; a view and a function are gone already
    pg = FakePg(roles={"reader", "coordinator_reader"}, schemas={"lgk_k": "m"}, regclass={"lgk_k.nodes_view"})
    stmts = revoke_shared_sql(pg, ["lgk_k", "lgk_gone"], ["lgk_k.nodes_view", "lgk_k.edges_view"],
                              ["lgk_k.f(text)"])
    assert stmts == [
        "REVOKE ALL ON SCHEMA lgk_k FROM reader, coordinator_reader",
        "REVOKE ALL ON ALL TABLES IN SCHEMA lgk_k FROM reader, coordinator_reader",
        "REVOKE ALL ON ALL FUNCTIONS IN SCHEMA lgk_k FROM reader, coordinator_reader",
        "REVOKE ALL ON ALL SEQUENCES IN SCHEMA lgk_k FROM reader, coordinator_reader",
        "REVOKE ALL ON lgk_k.nodes_view FROM reader, coordinator_reader",
    ]
    assert revoke_shared_sql(FakePg(roles=set(), schemas={"lgk_k": "m"}), ["lgk_k"], [], []) == []


def test_shared_roles_are_the_three_provision_creates():
    from lgk.steps.provision import ROLE_SETUP_SQL
    from lgk.steps.teardown import SHARED_ROLES

    assert SHARED_ROLES == ("reader", "public_reader", "coordinator_reader")
    assert tuple(ROLE_SETUP_SQL) == SHARED_ROLES


def test_plan_keeps_each_synced_tables_recorded_pipeline_id_or_none(tmp_path):
    from lgk.steps.teardown import plan_from_manifest

    config = _config(tmp_path)
    mf.record(config, ObjectType.SYNCED_TABLE, f"{CAT}.lgk_k_v1.nodes", "uc", {"branch": BRANCH, "pipeline_id": A_PIPELINE})
    # hostile: a synced table recorded without a pipeline id (provision's status had none)
    mf.record(config, ObjectType.SYNCED_TABLE, f"{CAT}.lgk_k_v1.edges", "uc", {"branch": BRANCH, "pipeline_id": None})
    mf.record(config, ObjectType.SYNCED_TABLE, f"{CAT}.lgk_k_v2.nodes", "uc", {"branch": BRANCH, "pipeline_id": B_PIPELINE})
    plan = plan_from_manifest(mf.load_run_manifest(config))
    assert plan.synced_pipelines == {f"{CAT}.lgk_k_v1.nodes": A_PIPELINE, f"{CAT}.lgk_k_v1.edges": None,
                                     f"{CAT}.lgk_k_v2.nodes": B_PIPELINE}
    assert plan.event_logs == []  # resolved by the ops (a UC read), before the checks


def test_teardown_drops_only_its_own_pipelines_event_logs_after_the_synced_tables(tmp_path, monkeypatch):
    from lgk.steps.teardown import teardown

    config = _config(tmp_path)
    _record_graph_with_pipelines(config)
    own = [f"event_log_{(A_PIPELINE[:-1] + str(i)).replace('-', '_')}" for i in range(3)]
    ops, uc_sql = _live_ops(tmp_path, monkeypatch, _two_graph_branch(config), uc_tables={
        f"{CAT}.lgk_k_v1": list(TABLES_) + own,
    })
    teardown(config, GRAPH, ops=ops)
    drops = [s for s in uc_sql if "event_log_" in s]
    assert drops == [f"DROP TABLE IF EXISTS {CAT}.lgk_k_v1.{t}" for t in own]
    assert ops.w.postgres.deleted  # the synced tables (and so their pipelines) went first
    assert uc_sql.index(drops[0]) < uc_sql.index(f"DROP SCHEMA IF EXISTS {CAT}.lgk_k_v1 RESTRICT")


def test_another_pipelines_event_log_in_a_kit_schema_is_refused_and_nothing_is_deleted(tmp_path, monkeypatch):
    from lgk.steps.teardown import teardown

    config = _config(tmp_path)
    _record_graph_with_pipelines(config)
    stranger = f"event_log_{B_PIPELINE.replace('-', '_')}"
    pg = _two_graph_branch(config)
    # hostile: an event log of a pipeline this graph never recorded, inside the graph's UC schema
    ops, uc_sql = _live_ops(tmp_path, monkeypatch, pg, uc_tables={f"{CAT}.lgk_k_v1": list(TABLES_) + [stranger]})
    with pytest.raises(versions.NotOursError, match=stranger):
        teardown(config, GRAPH, ops=ops)
    assert uc_sql == [] and ops.w.postgres.deleted == []
    assert not [s for s in pg.sql if s.startswith(("DROP", "REVOKE"))]


@pytest.mark.parametrize("case", ["unmarked", "other_graph", "stranger_relation"])
def test_each_pg_schema_guard_refuses_before_any_delete(tmp_path, monkeypatch, case):
    from lgk.steps.teardown import teardown

    config = _config(tmp_path)
    _record_graph_with_pipelines(config)
    pg = _two_graph_branch(config)
    if case == "unmarked":
        pg.schemas["lgk_k_v1"] = None  # hostile: the version schema lost its marker
    elif case == "other_graph":
        pg.schemas["lgk_k"] = mf.marker("graph_b")  # hostile: the served schema is graph B's
    else:
        pg.relations["lgk_k"].append("lgk_k.sneaky")  # hostile: an unrecorded view in the served schema
    ops, uc_sql = _live_ops(tmp_path, monkeypatch, pg)
    with pytest.raises(versions.NotOursError):
        teardown(config, GRAPH, ops=ops)
    assert not [s for s in pg.sql if s.startswith(("DROP", "REVOKE"))]
    assert uc_sql == [] and ops.w.postgres.deleted == []


@pytest.mark.parametrize("case", ["unmarked", "stranger_table"])
def test_each_uc_schema_guard_refuses_before_any_delete(tmp_path, monkeypatch, case):
    from types import SimpleNamespace

    from lgk.steps.teardown import teardown

    config = _config(tmp_path)
    _record_graph_with_pipelines(config)
    pg = _two_graph_branch(config)
    tables = {f"{CAT}.lgk_k_v1": list(TABLES_)}
    if case == "stranger_table":
        tables[f"{CAT}.lgk_k"] = ["patients"]  # hostile: a table the kit never made in the kit schema
    ops, uc_sql = _live_ops(tmp_path, monkeypatch, pg, uc_tables=tables)
    if case == "unmarked":
        ops.w.schemas = SimpleNamespace(get=lambda full: SimpleNamespace(comment=None))
    with pytest.raises(versions.NotOursError):
        teardown(config, GRAPH, ops=ops)
    assert not [s for s in pg.sql if s.startswith(("DROP", "REVOKE"))]
    assert uc_sql == [] and ops.w.postgres.deleted == []


# hostile: the last case passes the right keyword too, so only teardown's own guard can refuse the stray one
@pytest.mark.parametrize("kwargs", [{"confirm_name": GRAPH}, {"confirm": GRAPH}, {},
                                    {"confirm_graph_name": GRAPH, "confirm_name": GRAPH}])
def test_a_misspelt_or_missing_confirm_keyword_names_confirm_graph_name(tmp_path, kwargs):
    from lgk.steps.teardown import teardown

    config = _config(tmp_path)
    _record_full_graph(config)
    ops = FakeOps()
    # hostile: Genie (persona 1) first called teardown(confirm_name=...) with the right graph name
    with pytest.raises(TypeError, match="confirm_graph_name"):
        teardown(config, ops=ops, **kwargs)
    assert ops.calls == []
    teardown(config, confirm_graph_name=GRAPH, ops=FakeOps())  # the right keyword works


def test_persona_leftover_event_log_is_dropped_and_a_sibling_graphs_survives(tmp_path, monkeypatch):
    """The exact persona-1 leftover: the edges pipeline's event log in the VERSION schema."""
    from lgk.steps.teardown import teardown

    config = _config(tmp_path)
    pg = f"postgres:{BRANCH}"
    pid = "-".join(["c3d4e5f6", "0000", "4000", "8000", "0000000000c3"])  # hostile: hyphenated id, underscores in the table name
    mf.record(config, ObjectType.UC_SCHEMA, f"{CAT}.lgk_k_v1", "uc")
    mf.record(config, ObjectType.SYNCED_TABLE, f"{CAT}.lgk_k_v1.edges", "uc", {"branch": BRANCH, "pipeline_id": pid})
    mf.record(config, ObjectType.PG_SCHEMA, "lgk_k_v1", pg)
    # hostile: graph B's event log, in a sibling version schema, must survive
    sibling = f"{CAT}.lgk_b_v1.event_log_0d1e2f30_0000_4000_8000_00000000beef"
    pgfake = FakePg(roles={"reader"}, schemas={"lgk_k_v1": mf.marker(GRAPH)},
                    relations={"lgk_k_v1": ["lgk_k_v1.edges"]})
    ops, uc_sql = _live_ops(tmp_path, monkeypatch, pgfake, uc_tables={
        f"{CAT}.lgk_k_v1": ["edges", "event_log_c3d4e5f6_0000_4000_8000_0000000000c3"],
        f"{CAT}.lgk_b_v1": [sibling.rsplit(".", 1)[1]],
    })
    teardown(config, GRAPH, ops=ops)
    assert uc_sql == [
        f"DROP TABLE IF EXISTS {CAT}.lgk_k_v1.event_log_c3d4e5f6_0000_4000_8000_0000000000c3",
        f"DROP SCHEMA IF EXISTS {CAT}.lgk_k_v1 RESTRICT",
    ]
    assert not [s for s in uc_sql if "lgk_b_v1" in s]


def test_an_outside_dependent_view_refuses_before_any_revoke_or_drop(tmp_path, monkeypatch):
    from lgk.steps.teardown import teardown

    config = _config(tmp_path)
    _record_graph_with_pipelines(config)
    pg = _two_graph_branch(config)
    pg.outside = ["lgk_b.reads_a_view"]  # hostile: graph B's view reads one of A's relations
    ops, uc_sql = _live_ops(tmp_path, monkeypatch, pg)
    with pytest.raises(versions.NotOursError, match="reads_a_view"):
        teardown(config, GRAPH, ops=ops)
    assert not [s for s in pg.sql if s.startswith(("DROP", "REVOKE"))]
    assert uc_sql == [] and ops.w.postgres.deleted == []


def test_a_schema_only_manifest_still_runs_the_marker_check(tmp_path, monkeypatch):
    """Provision failed before serve: PG schemas recorded, no views or functions."""
    from lgk.steps.teardown import teardown

    config = _config(tmp_path)
    mf.record(config, ObjectType.PG_SCHEMA, "lgk_b_v1", f"postgres:{BRANCH}")  # hostile: graph B's schema
    pg = _two_graph_branch(config)
    ops, uc_sql = _live_ops(tmp_path, monkeypatch, pg)
    with pytest.raises(versions.NotOursError):
        teardown(config, GRAPH, ops=ops)
    assert not [s for s in pg.sql if s.startswith(("DROP", "REVOKE"))]


def test_objects_recorded_outside_a_recorded_schema_are_still_revoked(tmp_path, monkeypatch):
    """The served schema pre-existed (not recorded), so only its views and functions are named."""
    pg = FakePg(roles={"reader"}, schemas={"lgk_k": mf.marker(GRAPH)}, regclass={"lgk_k.nodes_view"},
                regproc={"lgk_k.f(text)"})
    ops, _ = _live_ops(tmp_path, monkeypatch, pg)
    ops.revoke_shared_roles([], ["lgk_k.nodes_view"], ["lgk_k.f(text)"])
    assert pg.of("REVOKE") == ["REVOKE ALL ON lgk_k.nodes_view FROM reader",
                               "REVOKE ALL ON FUNCTION lgk_k.f(text) FROM reader"]


# ============================================================================
# Event logs of synced tables recorded without a pipeline id (B2) and retire's event logs (B1)
# ============================================================================

A_IDS = [A_PIPELINE[:-1] + str(i) for i in range(3)]  # the pipeline of nodes, edges, node_id_map
A_LOGS = [f"event_log_{pid.replace('-', '_')}" for pid in A_IDS]
WRONG_PIPELINE = "-".join(["ffffffff", "ffff", "4fff", "8fff", "ffffffffffff"])


def _uc_get(pids):
    """tables.get: `pids` maps a full name to its TableInfo.pipeline_id (as live); any other name is NotFound."""
    from types import SimpleNamespace

    from databricks.sdk.errors import NotFound

    def get(full):
        if full not in pids:
            raise NotFound(f"Table '{full}' does not exist.")
        # hostile: a wrong id under properties (live has none there); only the TableInfo field counts
        return SimpleNamespace(table_type="FOREIGN", pipeline_id=pids[full],
                               properties={"source_table": "x", "pipeline_id": WRONG_PIPELINE})
    return get


# hostile: UC answers a WRONG id for the tables whose id IS recorded, so an implementation that
# ignores the recorded value names the wrong event log and refuses the real one as foreign
UC_PIDS = {f"{CAT}.lgk_k_v1.nodes": WRONG_PIPELINE, f"{CAT}.lgk_k_v1.edges": A_IDS[1],
           f"{CAT}.lgk_k_v1.node_id_map": WRONG_PIPELINE}


def _teardown_ops(tmp_path, monkeypatch, config, uc_tables, uc_pids):
    """LiveOps over the two-graph branch; synced deletes and UC statements go to ONE ordered log."""
    pg = _two_graph_branch(config)
    ops, log = _live_ops(tmp_path, monkeypatch, pg, uc_tables=uc_tables)
    ops.w.tables.get = _uc_get(uc_pids)
    real = ops.w.postgres.delete_synced_table
    ops.w.postgres.delete_synced_table = lambda name: log.append(f"delete {name}") or real(name)
    return ops, log, pg


def test_teardown_reads_a_missing_pipeline_id_from_uc_and_cleans_up_fully(tmp_path, monkeypatch):
    from lgk.steps.teardown import teardown

    config = _config(tmp_path)
    _record_graph_with_pipelines(config, blank=("edges",))  # hostile: pipeline_id None in the manifest
    ops, log, _ = _teardown_ops(tmp_path, monkeypatch, config, {f"{CAT}.lgk_k_v1": list(TABLES_) + A_LOGS}, UC_PIDS)
    teardown(config, GRAPH, ops=ops)
    drops = [s for s in log if "event_log_" in s]
    assert drops == [f"DROP TABLE IF EXISTS {CAT}.lgk_k_v1.{t}" for t in A_LOGS]
    deletes = [s for s in log if s.startswith("delete ")]
    assert deletes == [f"delete synced_tables/{CAT}.lgk_k_v1.{t}" for t in TABLES_]
    assert log.index(deletes[-1]) < log.index(drops[0])  # the pipelines go first (teardown's order)
    assert log.index(drops[-1]) < log.index(f"DROP SCHEMA IF EXISTS {CAT}.lgk_k_v1 RESTRICT")


def test_teardown_with_no_recorded_id_and_no_uc_table_refuses_before_any_delete(tmp_path, monkeypatch):
    from lgk.event_logs import UnknownPipelineError
    from lgk.steps.teardown import teardown

    config = _config(tmp_path)
    _record_graph_with_pipelines(config, blank=("edges",))
    # hostile: UC no longer has edges (its schema does), so its event log cannot be told from a stranger's
    no_edges = {k: v for k, v in UC_PIDS.items() if not k.endswith(".edges")}
    ops, log, pg = _teardown_ops(tmp_path, monkeypatch, config, {f"{CAT}.lgk_k_v1": list(TABLES_) + A_LOGS}, no_edges)
    with pytest.raises(UnknownPipelineError, match=rf"{CAT}\.lgk_k_v1\.edges"):
        teardown(config, GRAPH, ops=ops)
    assert log == [] and ops.w.postgres.deleted == []
    assert not [s for s in pg.sql if s.startswith(("DROP", "REVOKE"))]


def test_teardown_still_refuses_a_sibling_pipelines_event_log_after_a_lookup(tmp_path, monkeypatch):
    from lgk.steps.teardown import teardown

    config = _config(tmp_path)
    _record_graph_with_pipelines(config, blank=("edges",))
    sibling = f"event_log_{B_PIPELINE.replace('-', '_')}"  # hostile: another pipeline's log in the version schema
    ops, log, pg = _teardown_ops(tmp_path, monkeypatch, config,
                                 {f"{CAT}.lgk_k_v1": list(TABLES_) + A_LOGS + [sibling]}, UC_PIDS)
    with pytest.raises(versions.NotOursError, match=sibling):
        teardown(config, GRAPH, ops=ops)
    assert log == [] and not [s for s in pg.sql if s.startswith(("DROP", "REVOKE"))]


def test_teardown_names_one_event_log_for_two_synced_tables_of_one_pipeline(tmp_path, monkeypatch):
    from lgk.steps.teardown import teardown

    config = _config(tmp_path)
    mf.record(config, ObjectType.UC_SCHEMA, f"{CAT}.lgk_k_v1", "uc")
    mf.record(config, ObjectType.SYNCED_TABLE, f"{CAT}.lgk_k_v1.nodes", "uc", {"pipeline_id": A_PIPELINE})
    # hostile: the same pipeline again, this time not recorded (read from UC)
    mf.record(config, ObjectType.SYNCED_TABLE, f"{CAT}.lgk_k_v1.edges", "uc", {"pipeline_id": None})
    log_name = f"event_log_{A_PIPELINE.replace('-', '_')}"
    ops, log, _ = _teardown_ops(tmp_path, monkeypatch, config, {f"{CAT}.lgk_k_v1": ["nodes", "edges", log_name]},
                                {f"{CAT}.lgk_k_v1.edges": A_PIPELINE})
    plan = teardown(config, GRAPH, ops=ops)
    assert plan.event_logs == [f"{CAT}.lgk_k_v1.{log_name}"]
    assert [s for s in log if "event_log_" in s] == [f"DROP TABLE IF EXISTS {CAT}.lgk_k_v1.{log_name}"]


class _RetireW:
    """The workspace client retire sees: marked UC schemas with tables, tables.get, synced-table deletes."""

    def __init__(self, uc_tables, uc_pids, log):
        from types import SimpleNamespace

        from databricks.sdk.errors import NotFound

        def get_schema(full):
            if full not in uc_tables:
                raise NotFound(f"Schema '{full}' does not exist.")
            return SimpleNamespace(comment=mf.marker(GRAPH))

        def list_tables(catalog, schema):
            if f"{catalog}.{schema}" not in uc_tables:
                raise NotFound("no schema")
            return [SimpleNamespace(name=n) for n in uc_tables[f"{catalog}.{schema}"]]

        def delete_synced(name):
            log.append(f"delete {name}")
            return SimpleNamespace(wait=lambda: None)

        self.schemas = SimpleNamespace(get=get_schema)
        self.tables = SimpleNamespace(list=list_tables, get=_uc_get(uc_pids))
        self.postgres = SimpleNamespace(delete_synced_table=delete_synced)


def _retire_env(tmp_path, monkeypatch, uc_tables, uc_pids, blank=("edges",)):
    """v2 is live (recorded, its pipeline B); v1 is the old version to retire. Returns (config, ordered log)."""
    from lgk.steps import retire as rt

    config = _config(tmp_path, version=2)
    for i, t in enumerate(TABLES_):
        mf.record(config, ObjectType.SYNCED_TABLE, f"{CAT}.lgk_k_v1.{t}", "uc",
                  {"branch": BRANCH, "pipeline_id": None if t in blank else A_IDS[i]})
    # hostile: the live version's pipeline is recorded too; its event log must never be dropped
    mf.record(config, ObjectType.SYNCED_TABLE, f"{CAT}.lgk_k_v2.nodes", "uc", {"branch": BRANCH, "pipeline_id": B_PIPELINE})
    log: List[str] = []
    monkeypatch.setattr(rt, "connect_branch", lambda c: _FakeConn())
    monkeypatch.setattr(versions, "live_version_schemas", lambda conn, served: {"lgk_k_v2"})
    monkeypatch.setattr(versions, "refuse_unless_ours_pg", lambda conn, schema: None)
    monkeypatch.setattr(versions, "outside_dependents", lambda conn, schemas: [])
    monkeypatch.setattr(rt, "_workspace_client", lambda: _RetireW(uc_tables, uc_pids, log))
    monkeypatch.setattr(rt, "_drop_pg_schema", lambda conn, served, schema: log.append(f"pg drop {schema}"))
    monkeypatch.setattr("lgk.uc_sql.execute_sql", lambda w_, wh, s, **k: log.append(s) or [])
    return config, log


B_LOG = f"event_log_{B_PIPELINE.replace('-', '_')}"
UC_AFTER_SYNCS = {f"{CAT}.lgk_k_v1": list(TABLES_) + A_LOGS, f"{CAT}.lgk_k_v2": ["nodes", B_LOG]}


def test_retire_drops_exactly_the_old_versions_own_event_logs_after_its_synced_tables(tmp_path, monkeypatch):
    from lgk.steps.retire import retire

    config, log = _retire_env(tmp_path, monkeypatch, UC_AFTER_SYNCS, UC_PIDS)
    retire(config, version=1, confirm_graph_name=GRAPH)
    assert log == (
        [f"delete synced_tables/{CAT}.lgk_k_v1.{t}" for t in TABLES_]
        + [f"DROP TABLE IF EXISTS {CAT}.lgk_k_v1.{t}" for t in A_LOGS]
        + ["pg drop lgk_k_v1", f"DROP SCHEMA IF EXISTS {CAT}.lgk_k_v1 RESTRICT"]
    )
    assert not [s for s in log if "lgk_k_v2" in s]  # the live version's event log is untouched


@pytest.mark.parametrize("stranger", [
    B_LOG,                                                     # hostile: another pipeline's event log
    f"event_log_{WRONG_PIPELINE.replace('-', '_')}",           # hostile: an event log of no recorded pipeline
    "patients",                                                # hostile: a plain stranger's table
])
def test_retire_refuses_any_other_table_in_the_version_schema_and_deletes_nothing(tmp_path, monkeypatch, stranger):
    from lgk.steps.retire import retire

    uc = {**UC_AFTER_SYNCS, f"{CAT}.lgk_k_v1": UC_AFTER_SYNCS[f"{CAT}.lgk_k_v1"] + [stranger]}
    config, log = _retire_env(tmp_path, monkeypatch, uc, UC_PIDS)
    with pytest.raises(versions.NotOursError, match=stranger):
        retire(config, version=1, confirm_graph_name=GRAPH)
    assert log == []


def test_retire_with_no_recorded_id_and_no_uc_table_refuses_naming_it(tmp_path, monkeypatch):
    from lgk.event_logs import UnknownPipelineError
    from lgk.steps.retire import retire

    no_edges = {k: v for k, v in UC_PIDS.items() if not k.endswith(".edges")}
    config, log = _retire_env(tmp_path, monkeypatch, UC_AFTER_SYNCS, no_edges)
    with pytest.raises(UnknownPipelineError, match=rf"{CAT}\.lgk_k_v1\.edges"):
        retire(config, version=1, confirm_graph_name=GRAPH)
    assert log == []


def test_retiring_two_versions_drops_each_versions_logs_only_after_its_own_synced_tables(tmp_path, monkeypatch):
    """Prover gap 5c: retire_versions([1, 3]); v3's event log must not go in v1's iteration."""
    from lgk.steps.retire import retire_versions

    v3_pid = "-".join(["3c3c3c3c", "0000", "4000", "8000", "000000000003"])
    v3_log = "event_log_3c3c3c3c_0000_4000_8000_000000000003"
    uc = {**UC_AFTER_SYNCS, f"{CAT}.lgk_k_v3": ["nodes", v3_log]}
    config, log = _retire_env(tmp_path, monkeypatch, uc, UC_PIDS)
    mf.record(config, ObjectType.SYNCED_TABLE, f"{CAT}.lgk_k_v3.nodes", "uc", {"branch": BRANCH, "pipeline_id": v3_pid})
    retire_versions(config, versions=[1, 3], confirm_graph_name=GRAPH)
    assert log == (
        [f"delete synced_tables/{CAT}.lgk_k_v1.{t}" for t in TABLES_]
        + [f"DROP TABLE IF EXISTS {CAT}.lgk_k_v1.{t}" for t in A_LOGS]
        + ["pg drop lgk_k_v1", f"DROP SCHEMA IF EXISTS {CAT}.lgk_k_v1 RESTRICT"]
        + [f"delete synced_tables/{CAT}.lgk_k_v3.nodes", f"DROP TABLE IF EXISTS {CAT}.lgk_k_v3.{v3_log}",
           "pg drop lgk_k_v3", f"DROP SCHEMA IF EXISTS {CAT}.lgk_k_v3 RESTRICT"]
    )
