"""Seam-1 gates for the nightly refresh: reapply no-op, refresh survival, the freshness guard.

One module-scoped graph: fresh t- branch, build + provision with per-run lgk_v1_<6hex> names,
torn down by the shared `_teardown_live` in a finally. The tests run in file order on it:
1. reapply is a no-op after a normal sync (catalog state before == after, not "no error").
2. hostile: a new source row with NO sync -> the freshness guard (and reapply) fail loudly.
3. After a full refresh, the served view's OID is unchanged and its count has the new row.
4. Enrich re-run with changed scores -> the guard names node_score; the refresh's sync
   then serves the new scores in the synced node_score.
5. v1 refuses sync_mode "triggered" before any source read, even with a real
   materialized view as the source (the V1.1 spike-3 check never runs).
"""

import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import pytest

from lgk.uc_sql import execute_sql
from lgk.versions import version_schema
from tests.test_build_provision_walks import (
    WAREHOUSE_ID,
    _create_provider_fixture,
    _teardown_live,
    _write_provider_graph_mapping,
    served_graph_size,
)

pytestmark = [pytest.mark.seam1, pytest.mark.provision]


@pytest.fixture(scope="module")
def refresh_graph(workspace_client, lakebase_project, lakebase_catalog, run_folder):
    from lgk.harness import BranchLeasor
    from lgk.steps.build import build
    from lgk.steps.enrich import enrich
    from lgk.steps.provision import provision
    from lgk.steps.switch import switch
    from tests.preflight_support import preflight_as_operator
    from tests.privacy_support import clear_gate_live

    suffix = uuid.uuid4().hex[:6]
    leasor = BranchLeasor(workspace_client=workspace_client, project_id=lakebase_project,
                          test_base_branch="test-base", run_folder=run_folder, ttl_seconds=14400)
    config: Dict[str, Any] = {
        "graph_name": "provider_graph",
        "lakebase_project": lakebase_project,
        "lakebase_catalog": lakebase_catalog,
        "kit_schema": f"lgk_v1_{suffix}",
        "served_schema": f"lgk_served_{suffix}",
        "fixture_schema": f"lgk_v1_fx_{suffix}",
        "warehouse_id": WAREHOUSE_ID,
        "enriched": True,  # node_score is synced too; the shared teardown deletes it
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            config["branch_id"] = leasor.create_test_branch(f"refresh-{suffix}")["branch_id"]
            run_dir = Path(tmpdir) / "runs" / f"refresh_{suffix}"
            run_dir.mkdir(parents=True)
            config["run_folder"] = str(run_dir)
            _create_provider_fixture(workspace_client, lakebase_catalog, config["fixture_schema"])
            _write_provider_graph_mapping(run_dir / "graph_mapping.yaml", lakebase_catalog,
                                          config["fixture_schema"])
            # The real profile -> privacy step on the fixture source. start_date holds
            # date-shaped strings (a care-subject value pattern); the mapping does not
            # carry it, so the operator drops it (the helper asserts it really was flagged).
            clear_gate_live(config, workspace_client, drops=[("fixture_affiliations", "start_date")])
            build(config)
            (run_dir / "enrich.yaml").write_text("methods: [degree]\n")
            enrich(config)  # node_score in UC, so provision syncs it (SNAPSHOT)
            preflight_as_operator(workspace_client, config)  # the real preflight first
            provision(config)
            switch(config)  # provision never serves; the switch does
            yield config
        finally:
            _teardown_live(workspace_client, leasor, config)


def _pg(config):
    from lgk.branch import connect_branch

    return connect_branch(config)


_PRIVS = ("SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER")


def _catalog_state(config: Dict[str, Any]) -> Dict[str, Any]:
    """Everything reapply may touch, read from the Postgres catalog."""
    served, version = config["served_schema"], version_schema(config)
    with _pg(config) as conn:
        q = lambda sql, *a: [tuple(r) for r in conn.execute(sql, a).fetchall()]  # noqa: E731
        return {
            "indexes": q("SELECT tablename, indexname, indexdef FROM pg_indexes "
                         "WHERE schemaname = %s ORDER BY 1, 2", version),
            "views": q("SELECT c.relname, c.oid, pg_get_viewdef(c.oid), c.reloptions::text, "
                       "c.relacl::text FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                       "WHERE n.nspname = %s AND c.relkind = 'v' ORDER BY 1", served),
            "tables": q("SELECT c.relname, c.oid, c.relacl::text FROM pg_class c JOIN pg_namespace n "
                        "ON n.oid = c.relnamespace WHERE n.nspname = %s AND c.relkind IN ('r','p') "
                        "AND NOT c.relispartition ORDER BY 1", version),
            "schemas": q("SELECT nspname, nspacl::text FROM pg_namespace WHERE nspname IN (%s, %s) "
                         "ORDER BY 1", served, version),
            "reader": q(
                "SELECT n.nspname, c.relname, p.priv, has_table_privilege('reader', c.oid, p.priv) "
                "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace, "
                "unnest(%s::text[]) AS p(priv) WHERE n.nspname IN (%s, %s) "
                "AND c.relkind IN ('v','r','p') AND NOT c.relispartition ORDER BY 1, 2, 3",
                list(_PRIVS), served, version),
        }


def _reader_can(state, schema, rel, priv="SELECT"):
    return {(s, r, p): ok for s, r, p, ok in state["reader"]}[(schema, rel, priv)]


def _q(w, sql):
    return execute_sql(w, WAREHOUSE_ID, sql)


class TestRefreshLive:
    def test_reapply_is_a_noop_after_a_normal_sync(self, refresh_graph):
        from lgk.refresh.sync import sync
        from lgk.steps.reapply import reapply

        reapply(refresh_graph)  # the job's first reapply after provision
        before = _catalog_state(refresh_graph)
        # Non-vacuous: the state really holds what reapply owns.
        assert len(before["indexes"]) >= 2 and any("src_node_id" in i[2] for i in before["indexes"])
        served, version = refresh_graph["served_schema"], version_schema(refresh_graph)
        assert _reader_can(before, served, "nodes_view") is True
        assert _reader_can(before, version, "nodes") is False

        sync(refresh_graph)  # a normal nightly sync (snapshot tables: full refresh)
        after_sync = _catalog_state(refresh_graph)
        reapply(refresh_graph)
        after_reapply = _catalog_state(refresh_graph)

        assert after_sync == before, "the sync itself changed indexes/views/grants"
        assert after_reapply == after_sync, "reapply is not a no-op after a normal sync"
        # A wrong grant made identically on every run is idempotent, so equality alone cannot see
        # it: pin the reader's exact privileges (SELECT on the served views, nothing else).
        granted = {(s, r, p) for s, r, p, ok in after_reapply["reader"] if ok}
        # the conditional list - the three views plus each optional view present; this
        # graph is enriched with degree, so node_score_view is served (and nothing else optional)
        assert granted == {(served, v, "SELECT")
                           for v in ("nodes_view", "edges_view", "node_id_map_view", "node_score_view")}

    def test_new_source_row_without_sync_fails_the_freshness_guard(self, refresh_graph, workspace_client):
        import json

        from lgk.refresh.freshness import StaleSyncError, check_freshness
        from lgk.refresh.tasks import run_task

        t = f"{refresh_graph['lakebase_catalog']}.{refresh_graph['kit_schema']}.nodes"
        # hostile (B1): commits that change no data, right after the sync, must stay fresh
        _q(workspace_client, f"OPTIMIZE {t}")
        _q(workspace_client, f"ALTER TABLE {t} SET TBLPROPERTIES ('lgk.refresh_test' = '1')")
        check_freshness(refresh_graph)
        # hostile: a real committed write to the source, and NO sync afterwards
        _q(workspace_client, f"INSERT INTO {t} VALUES ('lgk-r10-new', 'provider', 'r10_new', "
                             f"'R10 new', map())")
        with pytest.raises(StaleSyncError, match="nodes"):
            check_freshness(refresh_graph)
        with pytest.raises(StaleSyncError):
            run_task("reapply", json.dumps(refresh_graph))  # the job's reapply task, by its key

    def test_full_refresh_keeps_view_oid_and_serves_the_new_row(self, refresh_graph):
        from lgk.refresh.freshness import check_freshness
        from lgk.refresh.sync import sync

        served, version = refresh_graph["served_schema"], version_schema(refresh_graph)
        oid_sql = ("SELECT c.oid FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                   "WHERE n.nspname = %s AND c.relname = %s")
        with _pg(refresh_graph) as conn:
            view_oid = conn.execute(oid_sql, (served, "nodes_view")).fetchone()[0]
            table_oid = conn.execute(oid_sql, (version, "nodes")).fetchone()[0]
            n_before = conn.execute(f"SELECT count(*) FROM {served}.nodes_view").fetchone()[0]
            assert conn.execute(f"SELECT count(*) FROM {served}.nodes_view WHERE natural_key = "
                                f"'r10_new'").fetchone()[0] == 0

        # owner decision (nightly stamp step): test 2 wrote straight into the kit Delta table (a
        # writer outside the kit build); the sync takes it only once the stamp step certified it
        from lgk.refresh.operator_stamp import stamp_operator_tables

        stamp_operator_tables(refresh_graph)
        sync(refresh_graph, full_refresh=True)  # full refresh of every synced table

        with _pg(refresh_graph) as conn:
            assert conn.execute(oid_sql, (served, "nodes_view")).fetchone()[0] == view_oid
            assert conn.execute(oid_sql, (version, "nodes")).fetchone()[0] == table_oid
            assert conn.execute(f"SELECT count(*) FROM {served}.nodes_view").fetchone()[0] == n_before + 1
            assert conn.execute(f"SELECT count(*) FROM {served}.nodes_view WHERE natural_key = "
                                f"'r10_new'").fetchone()[0] == 1
        check_freshness(refresh_graph)  # and the guard agrees it is fresh again

    def test_rerun_enrich_scores_reach_the_synced_node_score(self, refresh_graph, workspace_client):
        """#22: node_score is in the refresh set: stale is caught, and the sync serves the new scores."""
        from lgk.refresh.freshness import StaleSyncError, check_freshness
        from lgk.refresh.sync import sync
        from lgk.steps.enrich import enrich

        cat, kit = refresh_graph["lakebase_catalog"], refresh_graph["kit_schema"]
        version = version_schema(refresh_graph)
        new = "lgk-r10-new"  # the isolated node test 3 synced: degree 0
        hub, rel = _q(workspace_client, f"SELECT src_node_id, relationship_type FROM {cat}.{kit}.edges "
                                        f"ORDER BY edge_id LIMIT 1")[0]

        def pg_scores():
            with _pg(refresh_graph) as conn:
                return {(str(n), m): float(v) for n, m, v in conn.execute(
                    f"SELECT node_id, method, score FROM {version}.node_score").fetchall()}

        before = pg_scores()
        # node_score predates test 2's isolated node, so it has no row yet; the hub has one
        assert (new, "degree") not in before and (str(hub), "degree") in before  # non-vacuous
        # hostile: the new edge changes an EXISTING score (the hub's), not only adds a row
        _q(workspace_client, f"INSERT INTO {cat}.{kit}.edges VALUES ('lgk-r22-edge', '{new}', '{hub}', "
                             f"'{rel}', map(), CAST(NULL AS BIGINT))")  # edges.weight (unfolded: NULL)
        enrich(refresh_graph)  # the nightly enrich task: node_score recomputed in Delta
        with pytest.raises(StaleSyncError, match=r"node_score: synced version"):
            check_freshness(refresh_graph)

        # owner decision (nightly stamp step): the edge above is a write outside the kit build
        from lgk.refresh.operator_stamp import stamp_operator_tables

        stamp_operator_tables(refresh_graph)
        sync(refresh_graph)  # the refresh's sync task (snapshot tables: full refresh)
        after = pg_scores()
        delta = {(str(n), m): float(v) for n, m, v in _q(workspace_client,
                 f"SELECT node_id, method, score FROM {cat}.{kit}.node_score")}
        assert after == delta  # the synced table holds exactly the new enrich output
        assert after[(new, "degree")] == 1.0
        assert after[(str(hub), "degree")] == before[(str(hub), "degree")] + 1.0
        check_freshness(refresh_graph)  # and every synced table, node_score included, is fresh

    def test_triggered_mode_is_refused_in_v1_over_a_real_materialized_view(self, refresh_graph, workspace_client):
        from lgk.refresh.modes import UnsupportedSyncModeError
        from lgk.steps.provision import _scheduling_policy

        cat = refresh_graph["lakebase_catalog"]
        mv = f"{cat}.{refresh_graph['fixture_schema']}.nodes_mv"
        # hostile: a real MV over the real nodes table (dropped with the fixture schema)
        _q(workspace_client, f"CREATE MATERIALIZED VIEW {mv} AS SELECT * FROM "
                             f"{cat}.{refresh_graph['kit_schema']}.nodes")
        cfg = dict(refresh_graph, sync_mode="triggered")
        with pytest.raises(UnsupportedSyncModeError, match="Path B"):
            _scheduling_policy(workspace_client, cfg, "nodes", mv)


# ------------------------------------------------ the nightly kit build (live)
def _nightly_source(w: Any, cat: str, schema: str) -> None:
    """S3b's source shape: 3 providers and 2 affiliations (provider -> provider, the mapping's edge)."""
    _q(w, f"CREATE SCHEMA IF NOT EXISTS {cat}.{schema}")
    _q(w, f"CREATE OR REPLACE TABLE {cat}.{schema}.fixture_providers "
          f"(provider_id STRING, name STRING, specialty STRING) USING DELTA")
    _q(w, f"INSERT INTO {cat}.{schema}.fixture_providers VALUES ('P1', 'Dr. One', 'Cardiology'), "
          f"('P2', 'Dr. Two', 'Cardiology'), ('P3', 'Dr. Three', 'Oncology')")
    _q(w, f"CREATE OR REPLACE TABLE {cat}.{schema}.fixture_affiliations "
          f"(affiliation_id STRING, provider_id STRING, hospital_id STRING, start_date STRING) USING DELTA")
    _q(w, f"INSERT INTO {cat}.{schema}.fixture_affiliations VALUES "
          f"('A1', 'P1', 'P2', '2024-01-01'), ('A2', 'P2', 'P3', '2024-01-01')")


@pytest.fixture(scope="module")
def nightly_graph(workspace_client, lakebase_project, lakebase_catalog, run_folder):
    """A provisioned, switched graph over the 3-provider source, on its own t- branch (no enrich)."""
    from lgk.harness import BranchLeasor
    from lgk.steps.build import build
    from lgk.steps.provision import provision
    from lgk.steps.switch import switch
    from tests.preflight_support import preflight_as_operator
    from tests.privacy_support import clear_gate_live

    suffix = uuid.uuid4().hex[:6]
    leasor = BranchLeasor(workspace_client=workspace_client, project_id=lakebase_project,
                          test_base_branch="test-base", run_folder=run_folder, ttl_seconds=14400)
    config: Dict[str, Any] = {
        "graph_name": "provider_graph",
        "lakebase_project": lakebase_project,
        "lakebase_catalog": lakebase_catalog,
        "kit_schema": f"lgk_v1_{suffix}",
        "served_schema": f"lgk_served_{suffix}",
        "fixture_schema": f"lgk_v1_fx_{suffix}",
        "warehouse_id": WAREHOUSE_ID,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            config["branch_id"] = leasor.create_test_branch(f"nightly-{suffix}")["branch_id"]
            run_dir = Path(tmpdir) / "runs" / f"nightly_{suffix}"
            run_dir.mkdir(parents=True)
            config["run_folder"] = str(run_dir)
            _nightly_source(workspace_client, lakebase_catalog, config["fixture_schema"])
            _write_provider_graph_mapping(run_dir / "graph_mapping.yaml", lakebase_catalog,
                                          config["fixture_schema"])
            clear_gate_live(config, workspace_client, drops=[("fixture_affiliations", "start_date")])
            build(config)  # v1, the manual build
            preflight_as_operator(workspace_client, config)
            provision(config)
            switch(config)
            yield config
        finally:
            _teardown_live(workspace_client, leasor, config)


def _served_graph(config: Dict[str, Any]):
    """(nodes {natural_key: label}, node count, edges {(src key, dst key)}) as the website's reader."""
    from tests.test_privacy_live import _as_reader

    s = config["served_schema"]
    with _pg(config) as conn:
        _as_reader(conn)
        nodes = conn.execute(f"SELECT natural_key, label FROM {s}.nodes_view").fetchall()
        edges = conn.execute(
            f"SELECT a.natural_key, b.natural_key FROM {s}.edges_view e "
            f"JOIN {s}.nodes_view a ON a.node_id = e.src_node_id "
            f"JOIN {s}.nodes_view b ON b.node_id = e.dst_node_id").fetchall()
    return {k: l for k, l in nodes}, len(nodes), {(a, b) for a, b in edges}


def _kit_versions(w: Any, config: Dict[str, Any]) -> Dict[str, int]:
    fq = f"{config['lakebase_catalog']}.{config['kit_schema']}"
    return {t: max(int(r[0]) for r in _q(w, f"SELECT version, operation, operationMetrics FROM "
                                            f"(DESCRIBE HISTORY {fq}.{t})"))
            for t in ("nodes", "edges", "node_id_map")}


def _night(config: Dict[str, Any], enriched: bool = False) -> None:
    """One night of the default chain (no pipeline; enriched only when asked), task by task, as the
    job runs it. `enriched=True` takes the run folder's own decision (enrich.yaml, `run_enriched`)."""
    import json

    from lgk.refresh.job import refresh_chain, run_enriched
    from lgk.refresh.tasks import run_task

    if enriched:
        assert run_enriched(config["run_folder"]) is True  # the run folder holds enrich.yaml
        chain = refresh_chain(config, enriched=run_enriched(config["run_folder"]))
        assert chain == ["build", "enrich", "sync", "reapply"]  # non-vacuous: the chain enriches
    else:
        chain = refresh_chain(config, enriched=False)
        assert chain == ["build", "sync", "reapply"]  # non-vacuous: the default chain builds
    for key in chain:
        run_task(key, json.dumps(config))


def _record_freshness_lag(record_metric: Any, config: Dict[str, Any], not_before_s: float, kind: str,
                          size: Dict[str, int]) -> Dict[str, float]:
    """Freshness lag (C30b) of EVERY synced table of the run after tonight's sync, in minutes:
    sync_end_time - the source Delta commit time it holds (lgk.refresh.lag; cross-checked against
    DESCRIBE HISTORY, and refused if the reading is not from this night's update)."""
    from lgk.refresh.freshness import live_clients, synced_pairs
    from lgk.refresh.lag import table_lag_minutes

    w, sql = live_clients(config)
    lags = {}
    for dest_id, source in synced_pairs(config, sql):
        table = dest_id.rsplit(".", 1)[1]
        lags[table] = table_lag_minutes(w, sql, dest_id, source, not_before_s)
        record_metric("refresh.freshness_lag", lags[table], "min", branch=config["branch_id"], table=table,
                      kind=kind, **size)
    return lags


class TestNightlyKitBuildLive:
    def test_nightly_build_task_rebuilds_nodes_and_edges(self, nightly_graph, workspace_client, record_metric):
        cat, fx = nightly_graph["lakebase_catalog"], nightly_graph["fixture_schema"]
        nodes, _, edges = _served_graph(nightly_graph)
        assert set(nodes) == {"P1", "P2", "P3"} and edges == {("P1", "P2"), ("P2", "P3")}  # night 0

        # hostile: a deleted source row (P3; a MERGE without a delete rule or an append keeps it),
        # deleted with its affiliation P2->P3: an edge row left pointing at it would be an orphan,
        # which build now refuses until the operator accepts it (lgk.keys, tests/test_keys.py)
        _q(workspace_client, f"DELETE FROM {cat}.{fx}.fixture_providers WHERE provider_id = 'P3'")
        _q(workspace_client, f"DELETE FROM {cat}.{fx}.fixture_affiliations WHERE affiliation_id = 'A2'")
        # A duplicate natural key is no longer collapsed to a winner (build refuses it, lgk.keys):
        # P4 is added once.
        _q(workspace_client, f"INSERT INTO {cat}.{fx}.fixture_providers VALUES ('P5', 'Dr. Five', "
                             f"'Cardiology'), ('P4', 'Dr. Four', 'Cardiology')")
        _q(workspace_client, f"INSERT INTO {cat}.{fx}.fixture_affiliations VALUES "
                             f"('A3', 'P1', 'P5', '2025-01-01')")  # a provider joined yesterday
        night_start_s, t0 = time.time(), time.perf_counter()
        _night(nightly_graph)
        wall = time.perf_counter() - t0

        nodes, n_rows, edges = _served_graph(nightly_graph)
        assert nodes == {"P1": "Dr. One", "P2": "Dr. Two", "P4": "Dr. Four", "P5": "Dr. Five"}
        assert n_rows == 4  # P4 once, P3 gone
        assert edges == {("P1", "P2"), ("P1", "P5")}
        record_metric("refresh.nightly_chain.wall", wall, "s", branch=nightly_graph["branch_id"],
                      kind="build-sync-reapply", rows=n_rows, edges=len(edges))
        lags = _record_freshness_lag(record_metric, nightly_graph, night_start_s, "build-sync-reapply",
                                     {"rows": n_rows, "edges": len(edges)})
        assert sorted(lags) == ["edges", "node_id_map", "nodes"], lags  # every synced table has a lag

    def test_no_change_night_passes_freshness(self, nightly_graph, workspace_client):
        import json

        from lgk.refresh.freshness import StaleSyncError, check_freshness
        from lgk.refresh.tasks import run_task

        before = _served_graph(nightly_graph)
        versions = _kit_versions(workspace_client, nightly_graph)
        # hostile: a no-change night. The rebuild is still a NEW Delta version of the SAME table
        # (CREATE OR REPLACE keeps the history, S3b); a DROP/CREATE build restarts at version 0,
        # lands on the synced version again, and the guard passes with nothing synced.
        run_task("build", json.dumps(nightly_graph))
        after_build = _kit_versions(workspace_client, nightly_graph)
        assert all(after_build[t] > versions[t] for t in versions), (versions, after_build)
        with pytest.raises(StaleSyncError, match="nodes"):
            check_freshness(nightly_graph)  # rebuilt, not yet synced: the guard sees it
        run_task("sync", json.dumps(nightly_graph))
        run_task("reapply", json.dumps(nightly_graph))  # its freshness guard passes
        check_freshness(nightly_graph)
        assert _served_graph(nightly_graph) == before  # the same graph is served

    def test_build_task_refuses_stale_flags(self, nightly_graph, workspace_client):
        import json

        from lgk.privacy.gate import PrivacyGateError
        from lgk.refresh.tasks import run_task

        mapping = Path(nightly_graph["run_folder"]) / "graph_mapping.yaml"
        original = mapping.read_text()
        versions = _kit_versions(workspace_client, nightly_graph)
        # hostile: stale flags. graph_mapping.yaml edited (the label is now another column)
        # without re-running privacy; the nightly build must refuse before any SQL.
        mapping.write_text(original.replace("label: name", "label: specialty"))
        assert mapping.read_text() != original
        try:
            with pytest.raises(PrivacyGateError):
                run_task("build", json.dumps(nightly_graph))
        finally:
            mapping.write_text(original)
        assert _kit_versions(workspace_client, nightly_graph) == versions  # no table moved


# ------------------------------------- final review P2: the nightly chain recomputes enrichment
def _expected_degree(pairs) -> Dict[str, float]:
    """Hand reference: distinct undirected neighbours per natural key (isolated = 0), from the
    SOURCE rows as written, never the kit's SQL or its Delta tables."""
    neighbours: Dict[str, set] = {}
    for a, b in pairs:
        neighbours.setdefault(a, set())
        neighbours.setdefault(b, set())
        if a != b:
            neighbours[a].add(b)
            neighbours[b].add(a)
    return {k: float(len(v)) for k, v in neighbours.items()}


def _served_degree(config: Dict[str, Any]) -> Dict[str, float]:
    """{natural_key: degree score} as the website's reader sees it (served views only)."""
    from tests.test_privacy_live import _as_reader

    s = config["served_schema"]
    with _pg(config) as conn:
        _as_reader(conn)
        rows = conn.execute(
            f"SELECT n.natural_key, sc.score FROM {s}.node_score_view sc "
            f"JOIN {s}.nodes_view n ON n.node_id = sc.node_id WHERE sc.method = 'degree'").fetchall()
    return {str(k): float(v) for k, v in rows}


class TestNightlyEnrichLive:
    def test_nightly_chain_with_enrich_serves_scores_of_the_rebuilt_graph(self, refresh_graph, workspace_client,
                                                                          record_metric):
        """Recomputes enrichment, live: a run folder with enrich.yaml makes the nightly chain
        build -> enrich -> sync -> reapply; after a source change the SERVED node_score is the
        degree of the rebuilt graph (hand reference from the source rows)."""
        from tests.test_build_provision_walks import FAN_OUT, N_CHAIN, N_PROVIDERS

        g = refresh_graph
        cat, fx = g["lakebase_catalog"], g["fixture_schema"]
        # earlier tests' MV over the kit nodes table (test 5) is not part of the graph
        _q(workspace_client, f"DROP MATERIALIZED VIEW IF EXISTS {cat}.{fx}.nodes_mv")
        before = _served_degree(g)
        # Non-vacuous: prov_045 is isolated (0) and prov_900 does not exist yet (other scores may
        # carry the earlier tests' direct Delta writes, so they are not pinned here)
        assert before["prov_045"] == 0.0 and "prov_900" not in before

        # hostile: a NEW node with edges to an existing node and to an isolated one (scores rise)
        _q(workspace_client, f"INSERT INTO {cat}.{fx}.fixture_providers VALUES ('prov_900', 'Dr. New', 'Cardiology')")
        _q(workspace_client, f"INSERT INTO {cat}.{fx}.fixture_affiliations VALUES "
                             f"('aff_new_1', 'prov_900', 'prov_000', '2026-01-01'), "
                             f"('aff_new_2', 'prov_900', 'prov_045', '2026-01-01')")
        # hostile: a DELETED source edge (scores fall: an appending enrich keeps the old value)
        _q(workspace_client, f"DELETE FROM {cat}.{fx}.fixture_affiliations WHERE affiliation_id = 'aff_005'")
        night_start_s, t0 = time.time(), time.perf_counter()
        _night(g, enriched=True)
        wall = time.perf_counter() - t0

        # the source as it now stands: chain minus aff_005, the fan-out, the duplicate, the new rows
        pairs = [(f"prov_{i % N_PROVIDERS:03d}", f"prov_{(i + 1) % N_PROVIDERS:03d}") for i in range(N_CHAIN)
                 if i != 5]
        pairs += list(FAN_OUT) + [("prov_000", "prov_001"), ("prov_900", "prov_000"), ("prov_900", "prov_045")]
        want = {f"prov_{i:03d}": 0.0 for i in range(N_PROVIDERS)}
        want.update(_expected_degree(pairs))
        want["prov_900"] = 2.0  # spelled out: the new node
        assert want["prov_005"] == 2.0 and want["prov_006"] == 1.0 and want["prov_045"] == 1.0
        # hostile: earlier tests wrote 'r10_new' straight into the kit Delta tables; a rebuild
        # from the source drops it, so a stale score for it must not be served either
        assert _served_degree(g) == want
        size = served_graph_size(g)
        record_metric("refresh.nightly_chain.wall", wall, "s", branch=g["branch_id"],
                      kind="build-enrich-sync-reapply", rows=len(want), edges=size["edges"])
        lags = _record_freshness_lag(record_metric, g, night_start_s, "build-enrich-sync-reapply",
                                     {"rows": len(want), "edges": size["edges"]})
        assert "node_score" in lags and {"nodes", "edges", "node_id_map"} <= set(lags), lags
# ------------------------- the nightly stamp step: an operator pipeline chain (live)
def _operator_rebuild(w: Any, config: Dict[str, Any]) -> None:
    """The sdp_pipeline task's stand-in: an operator writer rebuilds the kit tables as plain Delta.

    A plain CREATE OR REPLACE ... AS SELECT run as the operator (copied out first so the replace
    reads another table); it writes no kit label."""
    cat, kit, fx = (config[k] for k in ("lakebase_catalog", "kit_schema", "fixture_schema"))
    for t in ("nodes", "edges", "node_id_map"):
        _q(w, f"CREATE OR REPLACE TABLE {cat}.{fx}.op_{t} AS SELECT * FROM {cat}.{kit}.{t}")
        _q(w, f"CREATE OR REPLACE TABLE {cat}.{kit}.{t} AS SELECT * FROM {cat}.{fx}.op_{t}")


def _labels(w: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    fq = f"{config['lakebase_catalog']}.{config['kit_schema']}"
    return {t: {str(r[0]): r[1] for r in _q(w, f"SHOW TBLPROPERTIES {fq}.{t}")}.get("lgk.privacy_stamp")
            for t in ("nodes", "edges", "node_id_map")}


class TestOperatorPipelineStampLive:
    def test_operator_chain_runs_the_job_task_order_and_the_stamp_step_certifies(self, nightly_graph,
                                                                               workspace_client):
        """sdp_pipeline_id, no build_mode: the job's own order sdp-like rebuild -> stamp -> sync -> reapply."""
        import json

        from lgk import manifest as mf
        from lgk.privacy.binding import StaleBuildError
        from lgk.refresh.freshness import check_freshness
        from lgk.refresh.job import refresh_chain
        from lgk.refresh.tasks import run_task

        w, cat, kit = workspace_client, nightly_graph["lakebase_catalog"], nightly_graph["kit_schema"]
        cfg = dict(nightly_graph, sdp_pipeline_id="operator-pipeline-not-run-here")
        chain = refresh_chain(cfg, enriched=False)
        assert chain == ["sdp_pipeline", "stamp", "sync", "reapply"]  # no kit build: the stamp step
        before = _served_graph(cfg)
        tables = ("nodes", "edges", "node_id_map")

        _operator_rebuild(w, cfg)
        assert _labels(w, cfg) == {t: None for t in tables}  # non-vacuous: the rebuild left no label
        with pytest.raises(StaleBuildError) as e:  # hostile: the sync without the stamp step
            run_task("sync", json.dumps(cfg))
        assert all(f"{kit}.{t} (no lgk.privacy_stamp)" in str(e.value) for t in tables), str(e.value)

        checked = _kit_versions(w, cfg)
        for key in chain:  # the job's own task order
            if key == "sdp_pipeline":
                _operator_rebuild(w, cfg)
                checked = _kit_versions(w, cfg)
            else:
                run_task(key, json.dumps(cfg))
        labels = {t: json.loads(v) for t, v in _labels(w, cfg).items()}
        recorded = mf.read_table_labels(cfg)
        for t in tables:
            fq = f"{cat}.{kit}.{t}"
            want = {"stamp": labels["nodes"]["stamp"], "table_id": w.tables.get(full_name=fq).table_id,
                    "certified_version": checked[t]}
            assert labels[t] == want and recorded[fq] == want, (t, labels[t], recorded.get(fq))
        assert _kit_versions(w, cfg) == {t: checked[t] + 1 for t in tables}  # the label is commit N
        check_freshness(cfg)
        assert _served_graph(cfg) == before  # the operator's rebuild is what is served

    def test_a_column_added_after_stamping_is_refused(self, nightly_graph, workspace_client):
        """Hostile: after the stamp step, a column lands on nodes; the sync refuses, then the stamp step too."""
        import json

        from lgk.privacy.binding import StaleBuildError
        from lgk.refresh.allowlist import AllowlistError
        from lgk.refresh.tasks import run_task

        w, cat, kit = workspace_client, nightly_graph["lakebase_catalog"], nightly_graph["kit_schema"]
        cfg = dict(nightly_graph, sdp_pipeline_id="operator-pipeline-not-run-here")
        _operator_rebuild(w, cfg)
        run_task("stamp", json.dumps(cfg))
        try:
            _q(w, f"ALTER TABLE {cat}.{kit}.nodes ADD COLUMNS (patient_ssn STRING)")
            with pytest.raises(StaleBuildError, match=r"nodes: certified_version \d+, but a data commit"):
                run_task("sync", json.dumps(cfg))
            with pytest.raises(AllowlistError, match=r"nodes: unknown column\(s\) patient_ssn string"):
                run_task("stamp", json.dumps(cfg))
        finally:  # leave the module's graph as the operator chain made it
            _q(w, f"CREATE OR REPLACE TABLE {cat}.{kit}.nodes AS SELECT * FROM {cat}.{nightly_graph['fixture_schema']}"
                  f".op_nodes")
            run_task("stamp", json.dumps(cfg))
            run_task("sync", json.dumps(cfg))


# ------------------------------- owner "bugs a": the real job, from a laptop run folder, in the workspace
def _ws_manifest(w: Any, api_folder: str) -> Dict[str, Any]:
    import json

    from lgk.manifest import MANIFEST_FILE

    with w.workspace.download(f"{api_folder}/{MANIFEST_FILE}") as f:
        return json.loads(f.read())


class TestRefreshJobFromALaptopRunFolderLive:
    def test_the_job_runs_every_task_in_the_workspace_on_the_copied_run_folder(self, refresh_graph, workspace_client,
                                                                               tmp_path):
        """create_refresh_job with the LOCAL run folder the LaptopRunner gives, then run_now: every task
        of the job run is SUCCESS (not in-process), and its writes land in the workspace copy."""
        import copy
        import json
        from datetime import timedelta

        from databricks.sdk.errors import NotFound, OperationFailed
        from databricks.sdk.service.workspace import ImportFormat, Language

        from lgk import manifest as mf
        from lgk.refresh import nightly_folder as nf
        from lgk.refresh.job import create_refresh_job
        from tests.test_parity_live import _build_wheel

        w, cfg = workspace_client, refresh_graph
        assert not nf.is_workspace_path(cfg["run_folder"])  # a laptop path, as the LaptopRunner config has
        home_api = f"/Users/{w.current_user.me().user_name}/{cfg['kit_schema']}"  # kit_schema = lgk_v1_<6hex>
        wheel = _build_wheel(tmp_path / "dist")
        notebook = Path(__file__).resolve().parents[1] / "notebooks" / "lgk_refresh_task.py"
        before = copy.deepcopy(cfg)
        job_id = None
        try:
            w.workspace.mkdirs(home_api)
            w.workspace.upload(f"{home_api}/{wheel.name}", wheel.read_bytes(), format=ImportFormat.AUTO,
                               overwrite=True)
            w.workspace.upload(f"{home_api}/lgk_refresh_task", notebook.read_bytes(), format=ImportFormat.SOURCE,
                               language=Language.PYTHON, overwrite=True)
            job_id = create_refresh_job(cfg, f"/Workspace{home_api}/lgk_refresh_task",
                                        f"/Workspace{home_api}/{wheel.name}", w=w)
            assert cfg == before  # the caller's config is never changed
            copy_api = f"{home_api}/runs/{cfg['graph_name']}"
            seeded = _ws_manifest(w, copy_api)
            waiter = w.jobs.run_now(job_id=job_id)
            try:
                waiter.result(timeout=timedelta(minutes=40))
            except OperationFailed:
                pass  # the per-task states below say which task failed
            run = w.jobs.get_run(waiter.run_id)
            states = {t.task_key: t.state.result_state.value if t.state.result_state else None for t in run.tasks}
            print(f"LIVE refresh job {job_id} run {run.run_id}: {json.dumps(states)} ({run.run_page_url})")
            for t in run.tasks:
                if states[t.task_key] != "SUCCESS":
                    print(f"LIVE task {t.task_key} error: {w.jobs.get_run_output(t.run_id).error}")
            assert states == {k: "SUCCESS" for k in ("build", "enrich", "sync", "reapply")}
            # The tasks wrote into the copy: the nightly build relabelled the kit tables there.
            nightly = _ws_manifest(w, copy_api)
            assert nightly["table_labels"] != seeded["table_labels"]
            # State rule evidence: the nightly chain recorded no object the laptop manifest lacks,
            # so teardown from the laptop still sees everything.
            local_ids = {(e.object_type.value, e.object_id) for e in mf.load_run_manifest(cfg).entries}
            ws_ids = {(e["object_type"], e["object_id"]) for e in nightly["entries"]}
            print(f"LIVE workspace-only manifest records: {sorted(ws_ids - local_ids)}")
            assert ws_ids <= local_ids
        finally:
            if job_id is not None:
                w.jobs.delete(job_id=job_id)
            try:
                w.workspace.delete(home_api, recursive=True)
            except NotFound:
                pass
