"""Live: provision proves N+1 before anything serves it; advise runs between; switch last.

One provider graph on a fresh branch, provisioned and served as v1 by the shared fixture. Then:
1. v2 is provisioned with a raw grant leaked to `reader` (hostile: SELECT on the raw v2.nodes
   synced table, injected right after provision locks the reader out). Provision must raise, and
   the served views must still read v1 (pg_depend), with the site answering from v1.
2. v3 runs the shipped flow through the runner: provision -> advise -> switch. The advisor's
   indexes exist on v3, which is now live, and walks equal the source reference.
Each test removes the synced tables and UC schema of the extra version it made, after moving the
served views back onto v1 (a served view blocks delete_synced_table); the fixture tears down the rest.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pytest

from lgk import versions
from lgk.uc_sql import execute_sql
from tests.test_build_provision_walks import (
    N_PROVIDERS, SYNCED_TABLES, WAREHOUSE_ID, _create_provider_fixture, _pg,
    assert_walks_match_reference, provisioned_graph,
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def ograph(workspace_client, lakebase_project, lakebase_catalog, run_folder):
    with provisioned_graph(workspace_client, lakebase_project, lakebase_catalog, run_folder,
                           "order", _create_provider_fixture) as config:
        yield config


def _cfg(g: Dict[str, Any], version: int, **kw) -> Dict[str, Any]:
    return {**g, "version": version, **kw}


def _live(g: Dict[str, Any]) -> set:
    with _pg(g) as conn:
        return versions.live_version_schemas(conn, versions.served_schema(g))


def _remove_version(w: Any, g: Dict[str, Any], version: int) -> None:
    """Serve v1 again, then delete the extra version's synced tables and UC schema. Logs, never raises."""
    v1, extra = versions.version_schema(g), versions.version_schema(g, version)
    try:
        conn = _pg(g)
        conn.autocommit = True
        with conn:
            versions.switch_views(conn, versions.served_schema(g), v1)
    except Exception as e:  # noqa: BLE001
        logger.error(f"cleanup: switch back to {v1} failed: {e}")
    for table in SYNCED_TABLES:
        name = f"synced_tables/{g['lakebase_catalog']}.{extra}.{table}"
        try:
            w.postgres.delete_synced_table(name=name).wait()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"cleanup: delete {name}: {e}")
    try:
        execute_sql(w, WAREHOUSE_ID, f"DROP SCHEMA IF EXISTS {g['lakebase_catalog']}.{extra} CASCADE")
    except Exception as e:  # noqa: BLE001
        logger.error(f"cleanup: drop UC schema {extra}: {e}")


def _served_nodes(g: Dict[str, Any]) -> int:
    with _pg(g) as conn:
        return conn.execute(f"SELECT count(*) FROM {versions.served_schema(g)}.nodes_view").fetchone()[0]


@pytest.mark.seam1
@pytest.mark.provision
def test_failed_isolation_proof_on_n_plus_1_leaves_the_site_on_n(ograph, workspace_client, monkeypatch):
    from lgk.privacy import binding
    from lgk.privacy.probe import ReaderNotIsolatedError
    from lgk.steps import provision as prov
    from lgk.steps.switch import switch

    g, v1, v2 = ograph, versions.version_schema(ograph), versions.version_schema(ograph, 2)
    assert _live(g) == {v1}
    real_lock = prov._lock_version_from_reader

    def lock_then_leak(config):
        real_lock(config)
        with _pg(config) as conn:  # hostile: a raw grant the kit did not make
            conn.execute(f"GRANT USAGE ON SCHEMA {v2} TO reader")
            conn.execute(f"GRANT SELECT ON {v2}.nodes TO reader")

    monkeypatch.setattr(prov, "_lock_version_from_reader", lock_then_leak)
    try:
        with pytest.raises(ReaderNotIsolatedError, match=rf"reader has SELECT on raw {v2}\.nodes"):
            prov.provision(_cfg(g, 2))
        # the served views still read v1, per the catalog's dependency rows, and serve v1's rows
        assert _live(g) == {v1}
        assert _served_nodes(g) == N_PROVIDERS
        with _pg(g) as conn:
            assert conn.execute(f"SELECT count(*) FROM {v2}.nodes").fetchone()[0] == N_PROVIDERS, \
                "non-vacuous: v2 really was synced, only the proof stopped it"
        bindings = json.loads((Path(g["run_folder"]) / binding.BINDINGS_FILE).read_text())
        assert v2 not in bindings.get("versions", {})
        with pytest.raises(binding.StaleBuildError, match=f"{v2} has no recorded privacy stamp"):
            switch(g, 2)
        assert _live(g) == {v1}
    finally:
        with _pg(g) as conn:
            conn.execute(f"REVOKE ALL ON SCHEMA {v2} FROM reader")
            conn.execute(f"REVOKE ALL ON {v2}.nodes FROM reader")
        _remove_version(workspace_client, g, 2)


def _index_names(g: Dict[str, Any], schema: str) -> set:
    with _pg(g) as conn:
        return {r[0] for r in conn.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = %s", (schema,)).fetchall()}


@pytest.mark.seam1
@pytest.mark.provision
def test_shipped_flow_provision_advise_switch_serves_a_tuned_version(ograph, workspace_client):
    from lgk.runner import LaptopRunner, RunStatus
    from lgk.steps.advise import advise
    from lgk.steps.provision import provision
    from lgk.steps.switch import switch

    g, v1, v3 = ograph, versions.version_schema(ograph), versions.version_schema(ograph, 3)
    # the fixture graph is ~32 edges; below the default 10k rows nothing is "large"
    # (the default itself is proven in test_advisor_live)
    cfg = _cfg(g, 3, advisor_min_rows=1)
    try:
        runner = LaptopRunner(cfg)
        results = runner.run_steps([("provision", provision)])
        v1_indexes = _index_names(g, v1)
        # provision indexes edges src_node_id on every new version, which leaves the advisor
        # nothing to propose; without that one index it must find the scan and propose it
        if results[-1].status == RunStatus.SUCCESS:
            with _pg(g) as conn:
                conn.execute(f"DROP INDEX {v3}.lgk_edges_src_node_id_idx")
            results += runner.run_steps([("advise", advise), ("switch", switch)])
        assert [(r.step, r.status) for r in results] == [
            ("provision", RunStatus.SUCCESS), ("advise", RunStatus.SUCCESS), ("switch", RunStatus.SUCCESS),
        ], [r.to_dict() for r in results]
        report = json.loads((Path(g["run_folder"]) / "advisor_report.json").read_text())
        # the advisor ran on v3 while v1 was live (it refuses a live version)
        assert report["version_schema"] == v3
        # Real synced tables are partitioned parents; EXPLAIN names a child partition in
        # __db_system. The proposal must still be made, on the v3 parent.
        assert report["proposals"], "non-vacuous: the advisor proposed at least one index"
        assert {p["schema"] for p in report["proposals"]} == {v3}
        assert ("edges", "src_node_id") in {(p["table"], p["column"]) for p in report["proposals"]}
        # each proposal is either applied (the index exists) or reported with its reason
        applied = {p["name"] for p in report["proposals"] if p["applied"]}
        for p in report["proposals"]:
            assert p["applied"] or p["not_applied"], p
        assert applied <= _index_names(g, v3), (applied, _index_names(g, v3))
        assert _index_names(g, v1) == v1_indexes, "the live v1 was not experimented on"
        md = (Path(g["run_folder"]) / "advisor_report.md").read_text()
        assert all(p["statement"] in md for p in report["proposals"])
        if applied:  # on a partitioned parent the index cascades to the partitions
            with _pg(g) as conn:
                n = conn.execute(
                    "SELECT count(*) FROM pg_inherits i JOIN pg_index x ON x.indrelid = i.inhrelid "
                    "JOIN pg_class ix ON ix.oid = x.indexrelid "
                    "WHERE i.inhparent = %s::regclass AND x.indkey[0] = "
                    "(SELECT attnum FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'src_node_id')",
                    (f"{v3}.edges", f"{v3}.edges")).fetchone()[0]
            assert n >= 1, "the parent index did not reach the partition"
        # the switch moved the site onto the tuned v3, and walks equal the source reference
        assert _live(g) == {v3}
        assert_walks_match_reference(g, workspace_client)
    finally:
        _remove_version(workspace_client, g, 3)
