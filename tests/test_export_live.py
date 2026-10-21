"""Seam-1: the exported bundle, deployed and run, builds and provisions as the SP; the SP owns it all.

The operator builds the graph (workshop). `export` writes the bundle; `databricks bundle deploy`
puts it in a per-run root (`lgk_<kit>` bundle name), and `bundle run` runs preflight, build and
provision as the SP (run_as). The SP builds into a kit schema of
its own (`<kit>_sp`, pre-created by the operator), so the nightly `lgk_refresh` (build -> sync ->
reapply) CREATE OR REPLACEs only tables the SP owns; the workshop's kit tables are never touched.
Ownership of the synced tables is read from each one's managed pipeline (creator and run_as).

The SP id comes ONLY from LGK_SP_APP_ID. The test grants the SP exactly the schema-level UC
privileges of lgk.sp_grants (the list the README and SKILL show), and only on this run's own
throwaway lgk_v1_<6hex>* schemas (its source fixture, its SP kit schema, its version schema; the
catalog grant is the owner's), plus a Postgres role on its own t- branch; all go with teardown.
"""

import json
import logging
import os
import subprocess
import uuid
from typing import Any, Dict

import pytest

from lgk import manifest as mf
from lgk import sp_grants, versions
from lgk.mapping import load_mapping
from lgk.steps.provision import ROLE_SETUP_SQL
from lgk.uc_sql import execute_sql
from tests.test_build_provision_walks import (
    SYNCED_TABLES,
    WAREHOUSE_ID,
    _create_provider_fixture,
    _teardown_live,
    _write_provider_graph_mapping,
)
from tests.test_versions_live import _connect_with_retry

logger = logging.getLogger(__name__)
GRAPH = "export_graph"
SP = os.environ.get("LGK_SP_APP_ID", "")

pytestmark = [pytest.mark.seam1, pytest.mark.provision]


def _cli(args, cwd, timeout=1500) -> subprocess.CompletedProcess:
    res = subprocess.run(["databricks", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    logger.info(f"databricks {' '.join(args[:3])} -> {res.returncode}\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
    return res


def _grant_sp_uc(w: Any, sp_cfg: Dict[str, Any], mapping: Any, throwaway_prefix: str) -> list:
    """The README's grants (lgk.sp_grants, one list): schema-level only, and only on this run's
    own throwaway schemas. The operator pre-creates (and marks) the SP's schemas first. Returns
    the grants made, as (principal, privilege, object)."""
    made = []
    for securable, name, privileges, operator_creates in sp_grants.uc_grants(sp_grants.targets(sp_cfg, mapping)):
        if securable == "CATALOG":
            continue  # the catalog grant is the workspace owner's, never the test's
        assert name.split(".", 1)[1].startswith(throwaway_prefix), f"refusing a grant on non-throwaway {name}"
        if operator_creates:
            mf.claim_uc_schema(w, WAREHOUSE_ID, *name.split("."), sp_cfg["graph_name"])
        execute_sql(w, WAREHOUSE_ID, sp_grants.grant_sql(securable, name, privileges, SP))
        made += [(SP, p, f"{securable} {name}") for p in privileges]
    return made


def _sp_pg_role(config: Dict[str, Any]) -> None:
    """The SP's Postgres role on this run's t- branch (dropped with the branch)."""
    conn = _connect_with_retry(config)
    conn.autocommit = True
    with conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS databricks_auth")
        exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (SP,)).fetchone()
        if not exists:
            conn.execute("SELECT databricks_create_role(%s, 'service_principal')", (SP,))
        # least privilege: create the served schema. (GRANT databricks_superuser is refused.)
        conn.execute(f'GRANT CONNECT, CREATE ON DATABASE databricks_postgres TO "{SP}"')
        # The operator owns `reader` and the function roles; the SP never creates roles
        # itself.
        for role, setup in ROLE_SETUP_SQL.items():
            if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone():
                conn.execute(setup)


def _sweep_run_pipelines(w: Any, catalog: str, kit: str) -> list:
    """Delete leftover synced-table pipelines of THIS run only (name holds `<catalog>.<kit>_`)."""
    left = []
    for p in w.pipelines.list_pipelines(filter=f"name LIKE '%{catalog}.{kit}_%'"):
        if f"{catalog}.{kit}_" not in (p.name or ""):
            continue
        left.append(p.pipeline_id)
        try:
            w.pipelines.delete(p.pipeline_id)
        except Exception as e:  # noqa: BLE001
            logger.error(f"teardown: delete pipeline {p.pipeline_id} failed: {e}")
    return left


def _run_tasks(w: Any, run_id: int) -> Dict[str, str]:
    """{task_key: result_state} of one job run."""
    run = w.jobs.get_run(run_id)
    return {t.task_key: (t.state.result_state.value if t.state and t.state.result_state else None)
            for t in run.tasks or []}


def _table_state(w: Any, full: str) -> Dict[str, Any]:
    """Owner and latest Delta version of a UC table (the workshop tables must never move)."""
    version = execute_sql(w, WAREHOUSE_ID, f"SELECT max(version) FROM (DESCRIBE HISTORY {full})")[0][0]
    return {"owner": w.tables.get(full).owner, "version": int(version)}


def _pipeline_identities(w: Any, full: str) -> Dict[str, str]:
    st = w.postgres.get_synced_table(name=f"synced_tables/{full}")
    p = w.pipelines.get(st.status.pipeline_id)
    return {"creator": p.creator_user_name, "run_as": p.run_as_user_name}


@pytest.fixture(scope="module")
def deployed(workspace_client, lakebase_project, lakebase_catalog, tmp_path_factory):
    from lgk.export import export
    from lgk.harness import BranchLeasor
    from lgk.steps.build import build
    from lgk.steps.provision import provision
    from lgk.steps.switch import switch
    from tests.preflight_support import preflight_as_operator
    from tests.privacy_support import clear_gate_live

    if not SP:
        pytest.skip("LGK_SP_APP_ID is not set")
    w = workspace_client
    suffix = uuid.uuid4().hex[:6]
    kit = f"lgk_v1_{suffix}"
    run = tmp_path_factory.mktemp("export_run")
    leasor = BranchLeasor(w, lakebase_project, "test-base", run, ttl_seconds=14400)
    g: Dict[str, Any] = {
        "graph_name": GRAPH, "lakebase_project": lakebase_project, "lakebase_catalog": lakebase_catalog,
        "kit_schema": kit, "served_schema": f"{kit}_served", "fixture_schema": f"{kit}_fx",
        "warehouse_id": WAREHOUSE_ID, "version": 1, "run_folder": str(run),
    }
    ws_g = dict(g, served_schema=f"{kit}_ws_served")  # the workshop's own served views (operator-owned)
    state: Dict[str, Any] = {"g": g, "sp_g": None, "bundle": None, "grants": []}
    try:
        g["branch_id"] = ws_g["branch_id"] = leasor.create_test_branch(f"ex-{suffix}")["branch_id"]
        _create_provider_fixture(w, lakebase_catalog, g["fixture_schema"])
        _write_provider_graph_mapping(run / "graph_mapping.yaml", lakebase_catalog, g["fixture_schema"])
        # start_date holds date-shaped strings (a care-subject value pattern); the
        # mapping does not carry it, so the operator drops it (the helper asserts it was flagged).
        # flags.json, schema.json and privacy_bindings.json then travel in the bundle, so the SP's
        # provision re-checks the same stamp against the same Delta tables (no gate is skipped).
        clear_gate_live(g, w, drops=[("fixture_affiliations", "start_date")])
        build(g)  # the workshop, as the operator: build + provision v1 (operator-owned synced tables)
        _connect_with_retry(g).close()
        preflight_as_operator(w, ws_g)  # the workshop's real preflight, as the operator
        provision(ws_g)
        switch(ws_g)  # the workshop's v1 is served by the switch, not by provision
        bundle = export(g)
        state["bundle"] = bundle
        bundled = state["bundled"] = json.loads((bundle / "run" / "config.json").read_text())
        # what the bundle builds and provisions: the SP's own kit schema, version N+1
        sp_g = state["sp_g"] = dict(g, kit_schema=bundled["kit_schema"], version=bundled["version"])
        assert sp_g["kit_schema"] != kit and bundled["workshop_kit_schema"] == kit
        workshop_tables = [f"{lakebase_catalog}.{kit}.{t}" for t in ("nodes", "edges", "node_id_map")]
        state["workshop_before"] = {t: _table_state(w, t) for t in workshop_tables}
        # The SP cannot CREATE SCHEMA in the catalog; the operator pre-creates (and marks) its schemas
        mapping = load_mapping(run / "graph_mapping.yaml")
        state["grants"] = _grant_sp_uc(w, sp_g, mapping, throwaway_prefix=kit)
        _sp_pg_role(g)
        var = f"--var=sp_app_id={SP}"
        dep = _cli(["bundle", "deploy", var], bundle)
        assert dep.returncode == 0, dep.stderr[-3000:]
        ran = _cli(["bundle", "run", "lgk_provision", var], bundle)
        state["run"] = ran
        # the nightly refresh is its own job; it runs repeatedly and never re-runs provision
        state["refresh_runs"] = [_cli(["bundle", "run", "lgk_refresh", var], bundle) for _ in range(2)]
        summary = _cli(["bundle", "summary", "-o", "json", var], bundle)
        state["job_ids"] = {k: int(v["id"]) for k, v in json.loads(summary.stdout)["resources"]["jobs"].items()}
        state["workshop_after"] = {t: _table_state(w, t) for t in workshop_tables}
        logger.warning(f"SP grants made: {state['grants']}")
        yield state
    finally:
        if state["bundle"] is not None:
            d = _cli(["bundle", "destroy", "--auto-approve", f"--var=sp_app_id={SP}"], state["bundle"], timeout=600)
            if d.returncode != 0:
                logger.error(f"teardown: bundle destroy failed: {d.stderr[-2000:]}")
        # The served schema is the SP's, so the operator cannot DROP it ("must be owner", live
        # 2026-10-03). Delete the branch first (it drops every PG object, unblocking
        # delete_synced_table), then the synced tables + pipelines and the UC schemas.
        if g.get("branch_id"):
            try:
                leasor.delete_branch(g["branch_id"])
            except Exception as e:  # noqa: BLE001
                logger.error(f"teardown: delete branch failed: {e}")
        for cfg in (state["sp_g"], ws_g):
            if cfg is not None:
                _teardown_live(w, leasor, {k: v for k, v in cfg.items() if k != "branch_id"})
        left = _sweep_run_pipelines(w, lakebase_catalog, kit)
        logger.info(f"teardown: orphan pipelines of this run deleted: {left}")


def test_bundle_job_succeeds(deployed, workspace_client):
    from lgk.export import provision_chain
    from lgk.refresh.job import run_enriched

    ran = deployed["run"]
    assert ran.returncode == 0, (ran.stdout[-3000:], ran.stderr[-3000:])
    runs = list(workspace_client.jobs.list_runs(job_id=deployed["job_ids"]["lgk_provision"]))
    tasks = _run_tasks(workspace_client, runs[0].run_id)
    logger.warning(f"SP lgk_provision run {runs[0].run_id}: {tasks}")
    # the SP builds its own kit tables before provision; every task of the chain succeeded
    want = provision_chain(deployed["bundled"], run_enriched(deployed["g"]["run_folder"]))
    assert tasks == {k: "SUCCESS" for k in want} and "build" in tasks


def test_refresh_job_runs_twice_and_never_reruns_provision(deployed, workspace_client):
    runs = deployed["refresh_runs"]
    assert [r.returncode for r in runs] == [0, 0], [(r.stdout[-2000:], r.stderr[-2000:]) for r in runs]
    ids = deployed["job_ids"]
    assert set(ids) == {"lgk_provision", "lgk_refresh"}
    provision_runs = list(workspace_client.jobs.list_runs(job_id=ids["lgk_provision"]))
    refresh_runs = list(workspace_client.jobs.list_runs(job_id=ids["lgk_refresh"]))
    assert len(provision_runs) == 1  # the two refresh runs did not provision again
    assert len(refresh_runs) == 2
    refresh_tasks = {t.task_key for t in workspace_client.jobs.get(ids["lgk_refresh"]).settings.tasks}
    from lgk.refresh.job import refresh_chain, run_enriched

    # the expected tasks come from the chain the export wrote (refresh_chain), never retyped
    want = refresh_chain(deployed["bundled"], run_enriched(deployed["g"]["run_folder"]))
    assert refresh_tasks == set(want) and "provision" not in refresh_tasks and "build" in refresh_tasks


def test_nightly_refresh_as_the_sp_succeeds_every_task_twice(deployed, workspace_client):
    """C16 / owner condition 4: two nightly runs AS THE SP, every task (build, sync, reapply) SUCCESS."""
    from lgk.refresh.job import refresh_chain, run_enriched

    job = workspace_client.jobs.get(deployed["job_ids"]["lgk_refresh"])
    assert job.settings.run_as.service_principal_name == SP
    want = refresh_chain(deployed["bundled"], run_enriched(deployed["g"]["run_folder"]))
    runs = sorted(workspace_client.jobs.list_runs(job_id=job.job_id), key=lambda r: r.run_id)
    states = {r.run_id: _run_tasks(workspace_client, r.run_id) for r in runs}
    logger.warning(f"SP lgk_refresh job {job.job_id} runs: {states}")
    assert len(states) == 2
    for run_id, tasks in states.items():
        assert tasks == {k: "SUCCESS" for k in want}, (run_id, tasks)


def test_the_sp_owns_its_kit_tables_and_the_workshop_tables_never_move(deployed, workspace_client):
    """The SP's build made (and owns) its own kit tables; the workshop's stay the operator's,
    at the same Delta version, after the provision job and two nightly runs."""
    operator = workspace_client.current_user.me().user_name
    sp_g = deployed["sp_g"]
    for t in ("nodes", "edges", "node_id_map"):
        assert workspace_client.tables.get(f"{sp_g['lakebase_catalog']}.{sp_g['kit_schema']}.{t}").owner == SP, t
    assert deployed["workshop_after"] == deployed["workshop_before"]
    assert {s["owner"] for s in deployed["workshop_after"].values()} == {operator}


def test_sp_owns_every_synced_table(deployed, workspace_client):
    operator = workspace_client.current_user.me().user_name
    sp_g = deployed["sp_g"]
    vs = versions.version_schema(sp_g)
    for table in SYNCED_TABLES:
        who = _pipeline_identities(workspace_client, f"{sp_g['lakebase_catalog']}.{vs}.{table}")
        assert who == {"creator": SP, "run_as": SP}, (table, who)
        assert operator not in who.values()


_VIEW_TABLES_SQL = """
SELECT DISTINCT v.relname, tn.nspname, t.relname
FROM pg_class v
JOIN pg_namespace vn ON vn.oid = v.relnamespace
JOIN pg_rewrite r ON r.ev_class = v.oid
JOIN pg_depend d ON d.objid = r.oid AND d.classid = 'pg_rewrite'::regclass AND d.refclassid = 'pg_class'::regclass
JOIN pg_class t ON t.oid = d.refobjid
JOIN pg_namespace tn ON tn.oid = t.relnamespace
WHERE vn.nspname = %s AND v.relkind = 'v' AND t.relkind IN ('r', 'p')
"""


def test_only_the_sps_tables_back_the_served_views(deployed, workspace_client):
    """Every table a served view reads is a synced table of the SP's
    version, whose UC source is in the SP's kit schema, never the workshop's."""
    from lgk.branch import connect_branch

    sp_g = deployed["sp_g"]
    cat, served, vs = sp_g["lakebase_catalog"], sp_g["served_schema"], versions.version_schema(sp_g)
    with connect_branch(sp_g) as conn:
        rows = conn.execute(_VIEW_TABLES_SQL, (served,)).fetchall()
    assert {"nodes_view", "edges_view", "node_id_map_view"} <= {r[0] for r in rows}, rows
    sources = {}
    for view, schema, table in rows:
        assert schema == vs, (view, schema, table)
        props = workspace_client.tables.get(f"{cat}.{schema}.{table}").properties or {}
        sources[f"{view} -> {schema}.{table}"] = props.get("source_table")
    logger.warning(f"SP served views -> UC sources: {sources}")
    for edge, source in sources.items():
        assert source and source.startswith(f"{cat}.{sp_g['kit_schema']}."), (edge, source)
        assert not source.startswith(f"{cat}.{deployed['g']['kit_schema']}."), (edge, source)


def test_sp_owns_the_served_views(deployed):
    from lgk.branch import connect_branch

    sp_g = deployed["sp_g"]
    with connect_branch(sp_g) as conn:
        owners = conn.execute(
            "SELECT DISTINCT viewowner FROM pg_views WHERE schemaname = %s", (sp_g["served_schema"],)
        ).fetchall()
    assert owners == [(SP,)]


def test_reader_reads_served_views_and_is_denied_raw_tables(deployed):
    from lgk.branch import connect_branch

    sp_g = deployed["sp_g"]
    served, raw = sp_g["served_schema"], versions.version_schema(sp_g)
    with connect_branch(sp_g) as conn:
        q = "SELECT has_table_privilege('reader', %s, 'SELECT')"
        for view in ("nodes_view", "edges_view", "node_id_map_view"):
            assert conn.execute(q, (f"{served}.{view}",)).fetchone() == (True,), view
        for table in SYNCED_TABLES:
            assert conn.execute(q, (f"{raw}.{table}",)).fetchone() == (False,), table
        assert conn.execute("SELECT has_schema_privilege('reader', %s, 'USAGE')", (raw,)).fetchone() == (False,)
        assert conn.execute(f"SELECT count(*) FROM {served}.nodes_view").fetchone()[0] > 0
