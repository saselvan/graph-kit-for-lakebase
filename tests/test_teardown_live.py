"""Seam-1 tests for whole-graph teardown and the reaper.

One graph is built and provisioned for real (synced tables + served views), so the drop
order is proven against the platform: a served view blocks delete_synced_table.
A second run of the same graph (version 2) goes onto its own t- branch, which the test
deletes BEFORE its synced tables, leaving orphans for the reaper.

Every object carries this run's `lgk_v1_<6hex>` suffix; the reaper reads only this
run's manifests, so it can never touch another stream's objects.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict

import pytest

from lgk import manifest as mf
from lgk import versions
from lgk.manifest import ObjectType
from lgk.uc_sql import execute_sql
from tests.test_build_provision_walks import (
    WAREHOUSE_ID,
    _create_provider_fixture,
    _teardown_live,
    _write_provider_graph_mapping,
)
from tests.test_versions_live import _connect_with_retry

logger = logging.getLogger(__name__)

GRAPH = "td_graph"
TABLES = ("nodes", "edges", "node_id_map")


@pytest.fixture(scope="module")
def tgraph(workspace_client, lakebase_project, lakebase_catalog, tmp_path_factory):
    from lgk.harness import BranchLeasor
    from lgk.steps.build import build
    from lgk.steps.provision import provision
    from lgk.steps.switch import switch

    suffix = uuid.uuid4().hex[:6]
    kit = f"lgk_v1_{suffix}"
    run1 = tmp_path_factory.mktemp("td_run1")
    run2 = tmp_path_factory.mktemp("td_run2")
    leasor = BranchLeasor(workspace_client, lakebase_project, "test-base", run1, ttl_seconds=14400)
    base: Dict[str, Any] = {
        "graph_name": GRAPH,
        "lakebase_project": lakebase_project,
        "lakebase_catalog": lakebase_catalog,
        "kit_schema": kit,
        "served_schema": f"{kit}_served",
        "fixture_schema": f"{kit}_fx",
        "warehouse_id": WAREHOUSE_ID,
    }
    g1 = dict(base, version=1, run_folder=str(run1))
    g2 = dict(base, version=2, run_folder=str(run2))
    state = {"g1": g1, "g2": g2, "leasor": leasor, "bystander_uc": f"{lakebase_catalog}.{kit}_bystander"}
    try:
        g1["branch_id"] = leasor.create_test_branch(f"td1-{suffix}")["branch_id"]
        g2["branch_id"] = leasor.create_test_branch(f"td2-{suffix}")["branch_id"]
        _create_provider_fixture(workspace_client, lakebase_catalog, g1["fixture_schema"])
        from tests.preflight_support import preflight_as_operator
        from tests.privacy_support import clear_gate_live

        from lgk.steps.privacy import privacy

        _write_provider_graph_mapping(run1 / "graph_mapping.yaml", lakebase_catalog, g1["fixture_schema"])
        # start_date holds date-shaped strings (a care-subject value pattern); the
        # mapping does not carry it, so the operator drops it. The helper asserts it was flagged.
        clear_gate_live(g1, workspace_client, drops=[("fixture_affiliations", "start_date")])
        # Run 2 serves the same Delta tables, so it takes run 1's decision (same profile, mapping
        # and resolved flags) and passes the real privacy step on it in its own folder.
        for name in ("schema.json", "graph_mapping.yaml", "flags.json"):
            (run2 / name).write_bytes((run1 / name).read_bytes())
        privacy(g2)
        _connect_with_retry(g1).close()
        build(g1)
        preflight_as_operator(workspace_client, g1)  # each run folder's own preflight
        provision(g1)
        switch(g1)  # provision never serves; the switch does
        _share_build_record(run1, run2)
        _connect_with_retry(g2).close()
        preflight_as_operator(workspace_client, g2)
        provision(g2)  # reads g1's Delta sources; synced tables bound to branch 2
        switch(g2)
        _create_bystanders(workspace_client, g1, state["bystander_uc"])
        yield state
    finally:
        try:
            execute_sql(workspace_client, WAREHOUSE_ID, f"DROP SCHEMA IF EXISTS {state['bystander_uc']} CASCADE")
        except Exception as e:  # noqa: BLE001
            logger.error(f"teardown: drop bystander UC schema failed: {e}")
        _teardown_live(workspace_client, leasor, g2)
        _teardown_live(workspace_client, leasor, g1)


def _share_build_record(run1: Path, run2: Path) -> None:
    """Run 2 serves the Delta tables run 1 built (shared kit schema), so it carries run 1's build record.

    Provision refuses a run folder that records no build. Run 2 does not rebuild: that would
    make its manifest own the shared kit tables, which is what these tests forbid. The record is
    only copied when run 2's gate (re-checked here through require_clear) is clear under the SAME
    stamp run 1 built with; provision still checks the stamp on every live Delta table, so nothing
    is bypassed.
    """
    from lgk.privacy import binding
    from lgk.privacy.gate import require_clear

    built = binding.built_stamp(str(run1))
    assert built and built == binding.stamp_id(require_clear(str(run2))), "run folders cleared different decisions"
    tables = json.loads((run1 / "privacy_bindings.json").read_text())["build"]["tables"]
    binding.record_build(str(run2), built, tables)


def _create_bystanders(w: Any, config: Dict[str, Any], bystander_uc: str) -> None:
    """Objects that share the graph's name prefix but that the kit never recorded.

    hostile: `<kit>_v9` looks exactly like a version schema of this graph, and
    `<kit>_bystander` exists in both Postgres and UC. A prefix-based teardown drops them.
    """
    from lgk.branch import connect_branch

    kit = config["kit_schema"]
    conn = connect_branch(config)
    conn.autocommit = True
    with conn:
        for schema in (f"{kit}_v9", f"{kit}_bystander"):
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            conn.execute(f"CREATE TABLE IF NOT EXISTS {schema}.nodes (node_id bigint)")
    execute_sql(w, WAREHOUSE_ID, f"CREATE SCHEMA IF NOT EXISTS {bystander_uc}")
    execute_sql(w, WAREHOUSE_ID, f"CREATE TABLE IF NOT EXISTS {bystander_uc}.nodes (node_id BIGINT)")


def _pg(config):
    from lgk.branch import connect_branch

    return connect_branch(config)


def _pg_schema_exists(config: Dict[str, Any], schema: str) -> bool:
    with _pg(config) as conn:
        return conn.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (schema,)).fetchone() is not None


def _pg_views(config: Dict[str, Any]) -> list:
    with _pg(config) as conn:
        q = "SELECT viewname FROM pg_views WHERE schemaname = %s ORDER BY 1"
        return [r[0] for r in conn.execute(q, (config["served_schema"],)).fetchall()]


def _synced_exists(w: Any, full: str) -> bool:
    from databricks.sdk.errors import NotFound

    try:
        w.postgres.get_synced_table(name=f"synced_tables/{full}")
        return True
    except NotFound:
        return False


def _uc_schema_exists(w: Any, full: str) -> bool:
    from databricks.sdk.errors import NotFound

    try:
        w.schemas.get(full)
        return True
    except NotFound:
        return False


def _uc_table_exists(w: Any, full: str) -> bool:
    from databricks.sdk.errors import NotFound

    try:
        w.tables.get(full)
        return True
    except NotFound:
        return False


def _synced(config: Dict[str, Any], table: str) -> str:
    return f"{config['lakebase_catalog']}.{versions.version_schema(config)}.{table}"


def _assert_g1_intact(w: Any, g1: Dict[str, Any], extra_views: tuple = ()) -> None:
    assert _pg_views(g1) == sorted(["edges_view", "node_id_map_view", "nodes_view", *extra_views])
    for t in TABLES:
        assert _synced_exists(w, _synced(g1, t)), t
        assert _uc_table_exists(w, f"{g1['lakebase_catalog']}.{g1['kit_schema']}.{t}"), t
    with _pg(g1) as conn:
        assert conn.execute(f"SELECT count(*) FROM {g1['served_schema']}.nodes_view").fetchone()[0] > 0


# ============================================================================
# Reaper
# ============================================================================


@pytest.mark.seam1
@pytest.mark.provision
def test_reaper_finds_and_deletes_synced_tables_left_by_a_deleted_branch(tgraph, workspace_client):
    from lgk.reaper import find_orphans, reap

    g1, g2, leasor = tgraph["g1"], tgraph["g2"], tgraph["leasor"]
    # hostile: g2's manifest also records one of g1's synced tables, which is bound to a
    # LIVE branch. The reaper must leave it; reaping every recorded table would break g1.
    live = _synced(g1, "nodes")
    mf.record(g2, ObjectType.SYNCED_TABLE, live, "uc", {"branch": f"projects/{g1['lakebase_project']}/branches/{g1['branch_id']}"})

    recorded = {e.object_id: e.metadata.get("pipeline_id") for e in mf.load_run_manifest(g2).entries
                if e.object_type == ObjectType.SYNCED_TABLE and e.object_id != live}
    expected = sorted(_synced(g2, t) for t in TABLES)
    assert sorted(recorded) == expected and all(recorded.values())

    # The wrong order on purpose: the branch goes first.
    assert leasor.delete_branch(g2["branch_id"])
    # Observed live: the synced-table GET is NotFound at once, but the UC table and
    # the pipeline stay behind. Pinned here, so a platform change shows up as a red test.
    for full, pid in recorded.items():
        assert not _synced_exists(workspace_client, full), f"{full}: GET should 404 once its branch is gone"
        assert _uc_table_exists(workspace_client, full), f"{full}: UC table should outlive its branch"
        assert not _pipeline_gone(workspace_client, pid), f"{full}: pipeline should outlive its branch"

    orphans = find_orphans(g2)
    assert sorted(o.synced_table for o in orphans) == expected
    assert all(o.uc_table and o.pipeline_id for o in orphans)
    with pytest.raises(versions.ConfirmationError):
        reap(g2, GRAPH.upper())
    assert all(_uc_table_exists(workspace_client, full) for full in expected)

    reaped = reap(g2, GRAPH)
    assert sorted(o.synced_table for o in reaped) == expected
    for full, pid in recorded.items():
        assert not _uc_table_exists(workspace_client, full), full
        assert _pipeline_gone(workspace_client, pid), pid
    assert _synced_exists(workspace_client, live)
    assert reap(g2, GRAPH) == []  # re-runnable: nothing left


# ============================================================================
# Teardown refusals: nothing is deleted
# ============================================================================


@pytest.mark.seam1
@pytest.mark.provision
def test_teardown_refuses_a_stranger_in_a_kit_schema_and_deletes_nothing(tgraph, workspace_client):
    from lgk.steps.teardown import teardown

    g1 = tgraph["g1"]
    with pytest.raises(versions.ConfirmationError):
        teardown(g1, " " + GRAPH)

    # hostile: a UC table the kit did not create, inside the kit's own UC schema.
    stranger = f"{g1['lakebase_catalog']}.{g1['kit_schema']}.patients"
    execute_sql(workspace_client, WAREHOUSE_ID, f"CREATE TABLE {stranger} (id BIGINT)")
    try:
        with pytest.raises(versions.NotOursError, match="patients"):
            teardown(g1, GRAPH)
        assert _uc_table_exists(workspace_client, stranger)
        _assert_g1_intact(workspace_client, g1)
    finally:
        execute_sql(workspace_client, WAREHOUSE_ID, f"DROP TABLE IF EXISTS {stranger}")

    # hostile: a Postgres view the kit did not create, inside the served schema.
    with _pg(g1) as conn:
        conn.execute(f"CREATE VIEW {g1['served_schema']}.sneaky AS SELECT 1 AS x")
        conn.commit()
    try:
        with pytest.raises(versions.NotOursError, match="sneaky"):
            teardown(g1, GRAPH)
        _assert_g1_intact(workspace_client, g1, extra_views=("sneaky",))
    finally:
        with _pg(g1) as conn:
            conn.execute(f"DROP VIEW IF EXISTS {g1['served_schema']}.sneaky")
            conn.commit()


@pytest.mark.seam1
@pytest.mark.provision
def test_two_graphs_sharing_a_kit_schema_cannot_touch_each_other(tgraph, workspace_client, tmp_path):
    from lgk.branch import connect_branch
    from lgk.steps.teardown import teardown

    g1 = tgraph["g1"]
    other = dict(g1, graph_name="td_other", run_folder=str(tmp_path))
    # Claiming g1's schemas for another graph is refused (the marker holds g1's name).
    with pytest.raises(versions.NotOursError):
        mf.claim_uc_schema(workspace_client, WAREHOUSE_ID, g1["lakebase_catalog"], g1["kit_schema"], "td_other")
    conn = connect_branch(other)
    try:
        with pytest.raises(versions.NotOursError):
            mf.claim_pg_schema(conn, g1["served_schema"], "td_other")
    finally:
        conn.close()

    # hostile: the other graph's manifest names every one of g1's objects (copied run folder).
    data = json.loads((Path(g1["run_folder"]) / mf.MANIFEST_FILE).read_text())
    data["graph_name"] = "td_other"
    (tmp_path / mf.MANIFEST_FILE).write_text(json.dumps(data))
    with pytest.raises(versions.NotOursError, match="lgk:graph=td_graph"):
        teardown(other, "td_other")
    _assert_g1_intact(workspace_client, g1)


# ============================================================================
# Teardown: exactly the manifest, in the verified order, re-runnable
# ============================================================================


@pytest.mark.seam1
@pytest.mark.provision
def test_teardown_drops_exactly_the_manifest_in_the_verified_order(tgraph, workspace_client, lakebase_project):
    from lgk.steps.teardown import teardown

    g1 = tgraph["g1"]
    cat, kit = g1["lakebase_catalog"], g1["kit_schema"]
    pipelines = [e.metadata.get("pipeline_id") for e in mf.load_run_manifest(g1).entries
                 if e.object_type == ObjectType.SYNCED_TABLE]
    assert len(pipelines) == 3 and all(pipelines)
    # the refresh job, created for real (never run) and recorded; teardown deletes it FIRST
    from lgk.refresh.job import create_refresh_job
    from lgk.steps.teardown import job_settings

    home = f"/Workspace/Users/{workspace_client.current_user.me().user_name}/{kit}"
    job_id = create_refresh_job(dict(g1, sdp_pipeline_id=pipelines[0]), f"{home}/lgk_refresh_task",
                                f"{home}/lgk.whl", w=workspace_client)
    survived = True
    try:
        assert job_settings(workspace_client, str(job_id)).tags == {"lgk_graph": GRAPH}
        plan = teardown(g1, GRAPH)
        survived = job_settings(workspace_client, str(job_id)) is not None
    finally:
        if survived and job_settings(workspace_client, str(job_id)) is not None:  # never leak it
            workspace_client.jobs.delete(job_id=job_id)
        # the job's workspace copy of g1's laptop run folder (lgk.refresh.nightly_folder)
        from databricks.sdk.errors import NotFound

        try:
            workspace_client.workspace.delete(home[len("/Workspace"):], recursive=True)
        except NotFound:
            pass
    assert not survived, "teardown left the refresh job"
    assert plan.jobs == [str(job_id)]
    assert sorted(plan.synced_tables) == sorted(_synced(g1, t) for t in TABLES)

    # Everything recorded is gone ...
    assert _pg_views(g1) == []
    for schema in (g1["served_schema"], versions.version_schema(g1)):
        assert not _pg_schema_exists(g1, schema), schema
    for t in TABLES:
        assert not _synced_exists(workspace_client, _synced(g1, t)), t
    for schema in (kit, versions.version_schema(g1)):
        assert not _uc_schema_exists(workspace_client, f"{cat}.{schema}"), schema
    with _pg(g1) as conn:  # A6 (a): the shared roles stay (old-manifest ROLE entries are skipped)
        for role in ("reader", "public_reader", "coordinator_reader"):
            assert conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone() is not None, role
    for pid in pipelines:
        assert _pipeline_gone(workspace_client, pid), pid

    # ... and nothing else: the unrecorded look-alikes and the fixture sources stay.
    assert _pg_schema_exists(g1, f"{kit}_v9")
    assert _pg_schema_exists(g1, f"{kit}_bystander")
    assert _uc_table_exists(workspace_client, f"{tgraph['bystander_uc']}.nodes")
    assert _uc_table_exists(workspace_client, f"{cat}.{g1['fixture_schema']}.fixture_providers")
    branches = {b.name.rsplit("/", 1)[-1] for b in workspace_client.postgres.list_branches(parent=f"projects/{lakebase_project}")}
    assert {"production", "test-base", g1["branch_id"]} <= branches

    # Re-runnable: everything is already gone, so a second run is a clean no-op.
    teardown(g1, GRAPH)


def _pipeline_gone(w: Any, pipeline_id: str) -> bool:
    from databricks.sdk.errors import NotFound

    try:
        p = w.pipelines.get(pipeline_id)
    except NotFound:
        return True
    return str(getattr(p.state, "value", p.state)) == "DELETED"


# ============================================================================
# P4 (A6 (a)): two graphs on ONE branch share the three roles; tearing down A leaves B serving
# ============================================================================

SHARED = ("reader", "public_reader", "coordinator_reader")


@pytest.fixture(scope="module")
def two_graphs(workspace_client, lakebase_project, lakebase_catalog, run_folder, tmp_path_factory):
    """Graph A by the shared provisioned_graph helper; graph B (its own name, kit, served and
    version schemas) built, provisioned and switched by the shipped steps on A's branch."""
    from lgk.steps.build import build
    from lgk.steps.provision import provision
    from lgk.steps.switch import switch
    from tests.preflight_support import preflight_as_operator
    from tests.privacy_support import clear_gate_live
    from tests.test_build_provision_walks import provisioned_graph

    with provisioned_graph(workspace_client, lakebase_project, lakebase_catalog, run_folder,
                           "p4a", _create_provider_fixture) as a:
        run_b = tmp_path_factory.mktemp("p4b")
        b = dict(a, graph_name="p4_graph_b", kit_schema=f"{a['kit_schema']}_b",
                 served_schema=f"{a['served_schema']}_b", run_folder=str(run_b))
        b.pop("graph_size", None)
        try:
            (run_b / "graph_mapping.yaml").write_bytes((Path(a["run_folder"]) / "graph_mapping.yaml").read_bytes())
            clear_gate_live(b, workspace_client, drops=[("fixture_affiliations", "start_date")])
            build(b)
            preflight_as_operator(workspace_client, b)
            provision(b)
            switch(b)
            yield a, b
        finally:
            _cleanup_graph_on_shared_branch(workspace_client, b)


def _cleanup_graph_on_shared_branch(w: Any, config: Dict[str, Any]) -> None:
    """B's leftovers, before A's helper deletes the shared branch: its served schema (views block
    synced-table deletes), then the shared `_teardown_live` without the branch (A's helper owns it)."""
    from lgk.branch import connect_branch

    try:
        conn = connect_branch(config)
        conn.autocommit = True
        with conn:
            conn.execute(f"DROP SCHEMA IF EXISTS {config['served_schema']} CASCADE")
    except Exception as e:  # noqa: BLE001
        logger.error(f"teardown: drop B served schema failed: {e}")
    _teardown_live(w, None, {k: v for k, v in config.items() if k != "branch_id"})


def _role_grants_in(config: Dict[str, Any], role: str, schemas: tuple) -> list:
    """Every object in `schemas` that `role` holds a privilege on (pg_shdepend ACL rows), by name."""
    sql = """
    SELECT coalesce(ns.nspname, n2.nspname, n3.nspname) AS schema,
           coalesce(ns.nspname, c.relname, p.proname) AS obj
    FROM pg_shdepend sd
    JOIN pg_roles ro ON ro.oid = sd.refobjid
    LEFT JOIN pg_namespace ns ON sd.classid = 'pg_namespace'::regclass AND ns.oid = sd.objid
    LEFT JOIN pg_class c ON sd.classid = 'pg_class'::regclass AND c.oid = sd.objid
    LEFT JOIN pg_namespace n2 ON n2.oid = c.relnamespace
    LEFT JOIN pg_proc p ON sd.classid = 'pg_proc'::regclass AND p.oid = sd.objid
    LEFT JOIN pg_namespace n3 ON n3.oid = p.pronamespace
    WHERE ro.rolname = %s AND sd.deptype = 'a'
      AND sd.dbid = (SELECT oid FROM pg_database WHERE datname = current_database())
    """
    with _pg(config) as conn:
        rows = conn.execute(sql, (role,)).fetchall()
    return sorted((r[0], r[1]) for r in rows if r[0] in schemas)


def _executable_functions(config: Dict[str, Any], role: str) -> list:
    sql = """SELECT p.oid::regprocedure::text FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
             WHERE n.nspname = %s AND has_function_privilege(%s, p.oid, 'EXECUTE') ORDER BY 1"""
    with _pg(config) as conn:
        return [r[0] for r in conn.execute(sql, (config["served_schema"], role)).fetchall()]


def _count_as(config: Dict[str, Any], role: str, sql: str) -> int:
    """Run `sql` as `role` (granted with SET, per the facts) in a transaction that is rolled back."""
    conn = _pg(config)
    try:
        conn.execute(f"GRANT {role} TO CURRENT_USER WITH SET TRUE")
        conn.execute(f"SET ROLE {role}")
        return conn.execute(sql).fetchone()[0]
    finally:
        conn.rollback()
        conn.close()


def _event_logs(w: Any, catalog: str, pipeline_ids: list) -> list:
    """The `event_log_*` tables in the catalog that belong to `pipeline_ids` (name = id with '_' for '-')."""
    rows = execute_sql(w, WAREHOUSE_ID, f"SELECT table_schema, table_name FROM {catalog}.information_schema.tables "
                                        f"WHERE table_name LIKE 'event_log_%'")
    wanted = {"event_log_" + pid.replace("-", "_") for pid in pipeline_ids}
    return sorted(f"{r[0]}.{r[1]}" for r in rows if r[1] in wanted)


def _pipelines(config: Dict[str, Any]) -> list:
    return [e.metadata["pipeline_id"] for e in mf.load_run_manifest(config).entries
            if e.object_type == ObjectType.SYNCED_TABLE and e.metadata.get("pipeline_id")]


def _blank_pipeline_id(config: Dict[str, Any], synced_table: str) -> str:
    """Rewrite the run manifest as if provision had recorded pipeline_id None for `synced_table`."""
    manifest = mf.load_run_manifest(config)
    entry = next(e for e in manifest.entries if e.object_type == ObjectType.SYNCED_TABLE and e.object_id == synced_table)
    pid = entry.metadata["pipeline_id"]
    entry.metadata["pipeline_id"] = None
    manifest.save(Path(config["run_folder"]) / mf.MANIFEST_FILE)
    return pid


@pytest.mark.seam1
@pytest.mark.provision
def test_retire_of_an_old_version_after_its_syncs_ran_drops_its_event_logs(two_graphs, workspace_client):
    """B1: B's v1 synced (so its pipelines wrote their event logs into v1's UC schema); B moves to v2
    and v1 is retired. Before the fix retire refused the event logs as foreign tables."""
    from lgk.steps.provision import provision
    from lgk.steps.retire import retire
    from lgk.steps.switch import switch
    from tests.preflight_support import preflight_as_operator

    a, b = two_graphs
    cat = b["lakebase_catalog"]
    v1 = versions.version_schema(b, 1)
    v1_pids = [e.metadata["pipeline_id"] for e in mf.load_run_manifest(b).entries
               if e.object_type == ObjectType.SYNCED_TABLE and e.object_id.startswith(f"{cat}.{v1}.")]
    assert len(v1_pids) >= 3 and all(v1_pids)
    b["version"] = 2  # B now serves v2; the teardown test below tears down both of B's versions
    preflight_as_operator(workspace_client, b)
    provision(b)
    switch(b)
    v1_logs = _event_logs(workspace_client, cat, v1_pids)
    logger.info(f"B's v1 sync event logs before retire: {v1_logs}")
    assert v1_logs and all(t.startswith(f"{v1}.") for t in v1_logs), v1_logs  # non-vacuous
    # hostile: a pipeline_id blanked in the manifest; retire reads it from Unity Catalog
    _blank_pipeline_id(b, f"{cat}.{v1}.edges")
    a_logs_before = _event_logs(workspace_client, cat, _pipelines(a))
    assert a_logs_before

    retire(b, 1, b["graph_name"])

    assert _event_logs(workspace_client, cat, v1_pids) == []
    assert not _uc_schema_exists(workspace_client, f"{cat}.{v1}")
    assert not _pg_schema_exists(b, v1)
    for pid in v1_pids:
        assert _pipeline_gone(workspace_client, pid), pid
    assert _event_logs(workspace_client, cat, _pipelines(a)) == a_logs_before  # graph A's are untouched
    with _pg(b) as conn:
        assert conn.execute(f"SELECT count(*) FROM {b['served_schema']}.nodes_view").fetchone()[0] > 0


@pytest.mark.seam1
@pytest.mark.provision
def test_teardown_of_one_graph_leaves_the_shared_roles_serving_the_other(two_graphs, workspace_client):
    from lgk.steps.teardown import teardown

    a, b = two_graphs
    cat = a["lakebase_catalog"]
    a_schemas = (a["served_schema"], versions.version_schema(a))
    b_view = f"SELECT count(*) FROM {b['served_schema']}.nodes_view"
    # Before: both graphs serve through the SAME three roles
    assert _count_as(b, "reader", b_view) > 0
    for role in SHARED:
        assert _role_grants_in(a, role, a_schemas), f"{role} held nothing on A before teardown"
    b_funcs = {role: _executable_functions(b, role) for role in ("public_reader", "coordinator_reader")}
    assert all(b_funcs.values()), b_funcs
    a_pids, b_pids = _pipelines(a), _pipelines(b)
    assert len(a_pids) >= 3 and len(b_pids) >= 3
    before_logs = _event_logs(workspace_client, cat, a_pids)
    logger.info(f"A's sync event logs before teardown: {before_logs}")
    assert before_logs, "no event_log_<pipeline_id> of A was found; the platform behaviour changed"

    teardown(a, a["graph_name"])

    # (b) the roles still exist
    with _pg(b) as conn:
        for role in SHARED:
            assert conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone(), role
    # (a) B still reads through them
    assert _count_as(b, "reader", b_view) > 0
    assert {role: _executable_functions(b, role) for role in b_funcs} == b_funcs
    for role in SHARED:
        # (c) nothing of A's is left on any role, and B's grants are untouched
        assert _role_grants_in(b, role, a_schemas) == [], role
    assert _role_grants_in(b, "reader", (b["served_schema"],))
    # (d) no event log of A's pipelines remains; B's are still there
    assert _event_logs(workspace_client, cat, a_pids) == []
    assert _event_logs(workspace_client, cat, b_pids)

    # B2 hostile: one of B's live synced tables recorded with pipeline_id None; teardown reads
    # it from Unity Catalog and still cleans up fully (before the fix it refused everything)
    blanked = _blank_pipeline_id(b, f"{cat}.{versions.version_schema(b)}.nodes")
    assert blanked in b_pids and _event_logs(workspace_client, cat, [blanked])
    teardown(b, b["graph_name"])
    assert not _uc_schema_exists(workspace_client, f"{cat}.{versions.version_schema(b)}")
    with _pg(b) as conn:
        for role in SHARED:
            assert conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone(), role
    assert _event_logs(workspace_client, cat, b_pids) == []
