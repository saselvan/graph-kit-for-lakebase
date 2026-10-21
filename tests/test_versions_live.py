"""Seam-1 tests for versions, the one-transaction switch, and retire/teardown.

One module-scoped branch holds three version schemas built directly in Postgres
(no synced tables are needed to prove the switch, which only replaces views):
- v1 and v2: full versions whose rows differ only in content
- v3: a broken build that is missing `node_id_map`, so a switch to it fails part way

Every schema carries the per-run `lgk_v1_<6hex>` prefix, so teardown can find it.
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import pytest

from lgk import versions
from tests.test_build_provision_walks import WAREHOUSE_ID, _teardown_live

logger = logging.getLogger(__name__)

# hostile: v1 and v2 have the SAME ids and row counts, only the content differs, so a
# test that checks counts or view names cannot tell which version is being served.
V_ROWS = {
    1: {"nodes": [(1, "a_v1"), (2, "b_v1")], "edges": [(10, 1, 2)], "node_id_map": [(1, "k1_v1")]},
    2: {"nodes": [(1, "a_v2"), (2, "b_v2")], "edges": [(10, 2, 1)], "node_id_map": [(1, "k1_v2")]},
    # hostile: lgk_..._v1 is a string prefix of lgk_..._v10, so a prefix-based live check
    # would think v1 is live while v10 is.
    10: {"nodes": [(1, "a_v10"), (2, "b_v10")], "edges": [(10, 1, 1)], "node_id_map": [(1, "k1_v10")]},
}
# Hostile non-versions, each named after a version number of this graph:
#   v4: our table plus a VIEW;  v5: our table plus a stranger's table `patients`;
#   v6: only our table, but a view in another schema (`<kit>_other`) depends on it;
#   v7: UC only, a schema holding a stranger's Delta table `patients`.
HOSTILE_PG = (4, 5, 6)
# The first columns are the ones the rows fill; the rest complete the real synced shape
# (build's nodes/edges), which the served functions compile against at switch time.
DDL = {
    "nodes": "node_id bigint PRIMARY KEY, label text, node_type text, natural_key text, properties jsonb",
    "edges": "edge_id bigint PRIMARY KEY, src_node_id bigint, dst_node_id bigint, relationship_type text",
    "node_id_map": "node_id bigint PRIMARY KEY, natural_key text",
}
GRAPH = "versions_test"
# v11's edges gain a column (as the folded edges' weight does), so a switch v1 -> v11 replaces
# edges_view in place and the rollback v11 -> v1 must drop and re-create it.
COLUMN_CHANGE = 11
COLUMN_CHANGE_DDL = {**DDL, "edges": DDL["edges"] + ", weight bigint"}


def _connect_with_retry(config: Dict[str, Any], tries: int = 12) -> Any:
    """A just-created branch endpoint can take a few seconds to accept connections."""
    from lgk.branch import connect_branch

    for i in range(tries):
        try:
            return connect_branch(config)
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            logger.info(f"branch not ready yet ({e}); retrying")
            time.sleep(5)


def _create_version(conn: Any, schema: str, rows: Dict[str, list], tables=tuple(DDL), ddl=DDL) -> None:
    with conn.transaction():
        conn.execute(f"CREATE SCHEMA {schema}")
        for table in tables:
            conn.execute(f"CREATE TABLE {schema}.{table} ({ddl[table]})")
            for row in rows.get(table, []):
                marks = ", ".join(["%s"] * len(row))
                conn.execute(f"INSERT INTO {schema}.{table} VALUES ({marks})", row)


@pytest.fixture(scope="module")
def vgraph(workspace_client, lakebase_project, lakebase_catalog, tmp_path_factory):
    from lgk.harness import BranchLeasor

    suffix = uuid.uuid4().hex[:6]
    run_folder = tmp_path_factory.mktemp("versions_run")
    leasor = BranchLeasor(
        workspace_client=workspace_client,
        project_id=lakebase_project,
        test_base_branch="test-base",
        run_folder=run_folder,
        ttl_seconds=14400,
    )
    kit = f"lgk_v1_{suffix}"
    config: Dict[str, Any] = {
        "graph_name": GRAPH,
        "lakebase_project": lakebase_project,
        "lakebase_catalog": lakebase_catalog,
        "kit_schema": kit,
        "served_schema": f"{kit}_served",
        "fixture_schema": f"{kit}_fx",
        "version": 2,
        "warehouse_id": WAREHOUSE_ID,
        "run_folder": str(run_folder),
    }
    try:
        config["branch_id"] = leasor.create_test_branch(f"versions-{suffix}")["branch_id"]
        conn = _connect_with_retry(config)
        conn.autocommit = True
        with conn:
            for n, rows in V_ROWS.items():
                _create_version(conn, versions.version_schema(config, n), rows)
            # hostile: v3 is a broken build — node_id_map (the LAST view switched) is missing,
            # so a switch to v3 fails only after nodes_view and edges_view were replaced.
            _create_version(conn, versions.version_schema(config, 3), V_ROWS[2], ("nodes", "edges"))
            _create_version(conn, versions.version_schema(config, COLUMN_CHANGE),
                            {**V_ROWS[2], "edges": [(10, 2, 1, 42)]}, ddl=COLUMN_CHANGE_DDL)
            _create_hostile_pg(conn, config)
        _create_hostile_uc(workspace_client, config)
        _bind_versions_to_gate(config, (1, 2, 3, 10, COLUMN_CHANGE))
        # what provision does before recording a version: the reader exists and is locked
        # out of the version's raw tables (switch proves that isolation before it moves a view)
        from lgk.steps.provision import _lock_version_from_reader

        for n in (*V_ROWS, COLUMN_CHANGE):
            _lock_version_from_reader({**config, "version": n})
        yield config
    finally:
        _drop_pg_schemas(config)
        _teardown_live(workspace_client, leasor, config)
        _drop_hostile_uc(workspace_client, config)


def _bind_versions_to_gate(config: Dict[str, Any], numbers) -> None:
    """Pass the privacy gate the real way, then record each hand-made version's stamp.

    switch refuses a version not provisioned under the current privacy decision. These
    versions are built directly in Postgres (this module proves views, not syncs), so the
    fixture records for each what provision records after its sync - no gate is bypassed:
    flags.json is cleared by the shipped privacy step and the stamp is read through the gate.
    """
    from lgk.privacy import binding
    from lgk.privacy.gate import require_clear
    from tests.privacy_support import clear_gate_offline
    from tests.test_build_provision_walks import _write_provider_graph_mapping

    _write_provider_graph_mapping(Path(config["run_folder"]) / "graph_mapping.yaml", "c", "s")
    clear_gate_offline(config["run_folder"])
    stamp = binding.stamp_id(require_clear(config["run_folder"]))
    for n in numbers:
        binding.record_version(config["run_folder"], versions.version_schema(config, n), stamp)


def _create_hostile_pg(conn: Any, config: Dict[str, Any]) -> None:
    v4, v5, v6 = (versions.version_schema(config, n) for n in HOSTILE_PG)
    other = f"{config['kit_schema']}_other"
    for schema in (v4, v5, v6):
        _create_version(conn, schema, V_ROWS[1], ("nodes",))
    conn.execute(f"CREATE VIEW {v4}.sneaky AS SELECT * FROM {v4}.nodes")
    conn.execute(f"CREATE TABLE {v5}.patients (id bigint)")
    conn.execute(f"CREATE SCHEMA {other}")
    conn.execute(f"CREATE VIEW {other}.reads_v6 AS SELECT * FROM {v6}.nodes")


def _create_hostile_uc(w: Any, config: Dict[str, Any]) -> None:
    from lgk.uc_sql import execute_sql

    full = f"{config['lakebase_catalog']}.{versions.version_schema(config, 7)}"
    execute_sql(w, WAREHOUSE_ID, f"CREATE SCHEMA IF NOT EXISTS {full}")
    execute_sql(w, WAREHOUSE_ID, f"CREATE TABLE IF NOT EXISTS {full}.patients (id BIGINT)")


def _drop_hostile_uc(w: Any, config: Dict[str, Any]) -> None:
    from lgk.uc_sql import execute_sql

    try:
        full = f"{config['lakebase_catalog']}.{versions.version_schema(config, 7)}"
        execute_sql(w, WAREHOUSE_ID, f"DROP SCHEMA IF EXISTS {full} CASCADE")
    except Exception as e:  # noqa: BLE001
        logger.error(f"teardown: drop hostile UC schema failed: {e}")


def _drop_pg_schemas(config: Dict[str, Any]) -> None:
    """Teardown step 1 for this module: served views, then every version schema."""
    if not config.get("branch_id"):
        return
    try:
        from lgk.branch import connect_branch

        conn = connect_branch(config)
        conn.autocommit = True
        with conn:
            conn.execute(f"DROP SCHEMA IF EXISTS {config['served_schema']} CASCADE")
            conn.execute(f"DROP SCHEMA IF EXISTS {config['kit_schema']}_other CASCADE")
            for n in (1, 2, 3, 10, COLUMN_CHANGE, *HOSTILE_PG):
                conn.execute(f"DROP SCHEMA IF EXISTS {versions.version_schema(config, n)} CASCADE")
    except Exception as e:  # noqa: BLE001
        logger.error(f"teardown: drop Postgres schemas failed: {e}")


def _pg(config):
    from lgk.branch import connect_branch

    return connect_branch(config)


def _answers(config: Dict[str, Any]) -> Dict[str, list]:
    """What the website would read now: every served view's rows, in key order."""
    s = config["served_schema"]
    with _pg(config) as conn:
        return {
            "nodes": [tuple(r) for r in conn.execute(f"SELECT node_id, label FROM {s}.nodes_view ORDER BY 1")],
            "edges": [
                tuple(r)
                for r in conn.execute(f"SELECT edge_id, src_node_id, dst_node_id FROM {s}.edges_view ORDER BY 1")
            ],
            "node_id_map": [
                tuple(r) for r in conn.execute(f"SELECT node_id, natural_key FROM {s}.node_id_map_view ORDER BY 1")
            ],
        }


def _view_definitions(config: Dict[str, Any]) -> Dict[str, str]:
    """pg_views text per served view (independent of the kit's pg_depend reader)."""
    with _pg(config) as conn:
        rows = conn.execute(
            "SELECT viewname, definition FROM pg_views WHERE schemaname = %s", (config["served_schema"],)
        ).fetchall()
    return {name: d for name, d in rows}


def _schema_exists(config: Dict[str, Any], schema: str) -> bool:
    with _pg(config) as conn:
        return conn.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (schema,)).fetchone() is not None


def _assert_serving(config: Dict[str, Any], n: int) -> None:
    target = versions.version_schema(config, n)
    defs = _view_definitions(config)
    assert sorted(defs) == ["edges_view", "node_id_map_view", "nodes_view"]
    for name, d in defs.items():
        assert f"{target}." in d, f"{name} does not point at {target}: {d}"
    assert _answers(config) == V_ROWS[n]


# ============================================================================
# Switch
# ============================================================================


@pytest.mark.seam1
@pytest.mark.provision
def test_switch_repoints_every_view_and_switch_back_restores_old_answers(vgraph):
    from lgk.steps.switch import switch

    switch(vgraph, 1)
    _assert_serving(vgraph, 1)
    switch(vgraph, 2)
    _assert_serving(vgraph, 2)
    switch(vgraph, 1)  # rollback = a switch back
    _assert_serving(vgraph, 1)
    flag = json.loads((Path(vgraph["run_folder"]) / "versions.json").read_text())
    assert flag["live_version_schema"] == versions.version_schema(vgraph, 1)


@pytest.mark.seam1
@pytest.mark.provision
def test_switch_is_one_transaction_a_failure_part_way_moves_no_view(vgraph):
    import psycopg

    from lgk.steps.switch import switch

    switch(vgraph, 2)
    with pytest.raises(psycopg.errors.UndefinedTable):
        switch(vgraph, 3)  # nodes_view + edges_view would move, then node_id_map is missing
    _assert_serving(vgraph, 2)  # NO view moved to v3, and v2's rows are still served


# ============================================================================
# the switch's version check
# ============================================================================


def _edges_view_columns(config: Dict[str, Any]) -> list:
    with _pg(config) as conn:
        return [c for c, _t in versions.relation_columns(conn, config["served_schema"], "edges_view")]


def _reader_grants(conn: Any, served: str) -> Dict[str, bool]:
    return {v: conn.execute("SELECT has_table_privilege('reader', %s, 'SELECT')", (f"{served}.{v}",)).fetchone()[0]
            for v, _ in versions.SERVED_VIEWS}


class _ProbeAtCommit:
    """The switch's connection; when its transaction ends, a SECOND session reads reader's grants.

    That second session sees exactly the committed state, so a grant made after the commit (in a
    later transaction) is absent from what it records.
    """

    def __init__(self, conn: Any, config: Dict[str, Any], seen: list):
        self._c, self._config, self._seen = conn, config, seen

    def __getattr__(self, name):
        return getattr(self._c, name)

    def transaction(self):
        inner, config, seen = self._c.transaction(), self._config, self._seen

        class _T:
            def __enter__(self):
                return inner.__enter__()

            def __exit__(self, *exc):
                out = inner.__exit__(*exc)
                with _pg(config) as probe:
                    seen.append(_reader_grants(probe, config["served_schema"]))
                return out

        return _T()


@pytest.mark.seam1
@pytest.mark.provision
def test_rollback_across_a_column_change_switches_and_keeps_grants(vgraph, monkeypatch):
    from lgk.steps.switch import switch

    switch(vgraph, 1)
    switch(vgraph, COLUMN_CHANGE)  # appends weight: CREATE OR REPLACE
    assert _edges_view_columns(vgraph) == ["edge_id", "src_node_id", "dst_node_id", "relationship_type", "weight"]
    seen: list = []
    real = versions.switch_views
    # only the switch transaction is probed (the shipped switch step calls it through the module)
    monkeypatch.setattr(versions, "switch_views",
                        lambda conn, *a, **k: real(_ProbeAtCommit(conn, vgraph, seen), *a, **k))
    switch(vgraph, 1)  # hostile: the rollback drops a column, so edges_view is dropped and re-created
    monkeypatch.undo()
    assert _edges_view_columns(vgraph) == ["edge_id", "src_node_id", "dst_node_id", "relationship_type"]
    _assert_serving(vgraph, 1)
    # the switch transaction's commit already held every grant (no window with an ungranted view)
    assert seen == [{v: True for v, _ in versions.SERVED_VIEWS}], seen
    with _pg(vgraph) as conn:
        conn.execute("GRANT reader TO CURRENT_USER")  # the operator may become reader (as the app does)
        conn.execute("SET ROLE reader")
        assert conn.execute(f"SELECT count(*) FROM {vgraph['served_schema']}.edges_view").fetchone()[0] == 1


@pytest.mark.seam1
@pytest.mark.provision
def test_r2_a_run_onto_an_existing_version_schema_is_refused(vgraph, workspace_client):
    from databricks.sdk.errors import NotFound

    from lgk.steps.provision import provision

    from tests.privacy_support import clear_gate_offline
    from tests.test_build_provision_walks import _write_provider_graph_mapping

    # The privacy gate runs first in provision; pass it the real way so the switch's version check is what refuses.
    _write_provider_graph_mapping(Path(vgraph["run_folder"]) / "graph_mapping.yaml", "c", "s")
    clear_gate_offline(vgraph["run_folder"])
    # the real preflight as the operator, so the switch's version check is what refuses
    from tests.preflight_support import preflight_as_operator

    preflight_as_operator(workspace_client, dict(vgraph))
    with pytest.raises(versions.VersionExistsError):
        provision(dict(vgraph, version=2))  # v2 already exists on this branch
    # Refused before anything was made: no UC destination schema for the run.
    with pytest.raises(NotFound):
        workspace_client.schemas.get(f"{vgraph['lakebase_catalog']}.{versions.version_schema(vgraph, 2)}")
    # A new version number is allowed.
    with _pg(vgraph) as conn:
        versions.assert_version_schema_absent(conn, versions.version_schema(vgraph, 9))


# ============================================================================
# retire / teardown refuse the live version
# ============================================================================


@pytest.mark.seam1
@pytest.mark.provision
def test_retire_and_teardown_refuse_the_version_the_views_point_at(vgraph):
    from lgk.steps.retire import retire, retire_versions
    from lgk.steps.switch import switch

    switch(vgraph, 2)
    # hostile: the stored flag says v1 is live, but the views point at v2. Only the
    # views count; an implementation that trusts the flag would drop the live v2.
    (Path(vgraph["run_folder"]) / "versions.json").write_text(
        json.dumps({"live_version_schema": versions.version_schema(vgraph, 1)})
    )
    with pytest.raises(versions.LiveVersionError):
        retire(vgraph, version=2, confirm_graph_name=GRAPH)
    with pytest.raises(versions.LiveVersionError):
        retire_versions(vgraph, versions=[2], confirm_graph_name=GRAPH)
    # hostile: the live version listed after a droppable one — refuse all, drop nothing.
    with pytest.raises(versions.LiveVersionError):
        retire_versions(vgraph, versions=[3, 2], confirm_graph_name=GRAPH)
    assert _schema_exists(vgraph, versions.version_schema(vgraph, 3))
    assert _schema_exists(vgraph, versions.version_schema(vgraph, 2))
    _assert_serving(vgraph, 2)


@pytest.mark.seam1
@pytest.mark.provision
def test_retire_old_version_and_teardown_unswitched_version_keep_the_site_up(vgraph):
    from lgk.steps.retire import retire, retire_versions
    from lgk.steps.switch import switch

    switch(vgraph, 2)
    with pytest.raises(versions.ConfirmationError):
        retire(vgraph, version=1, confirm_graph_name="wrong")
    assert _schema_exists(vgraph, versions.version_schema(vgraph, 1))

    switch(vgraph, 10)
    retire(vgraph, version=1, confirm_graph_name=GRAPH)  # v1 is not live; v10 is
    assert not _schema_exists(vgraph, versions.version_schema(vgraph, 1))
    _assert_serving(vgraph, 10)

    switch(vgraph, 2)
    retire_versions(vgraph, versions=[3], confirm_graph_name=GRAPH)
    assert not _schema_exists(vgraph, versions.version_schema(vgraph, 3))
    _assert_serving(vgraph, 2)


# ============================================================================
# retire / teardown drop only what is provably ours
# ============================================================================


def _view_exists(config: Dict[str, Any], schema: str, view: str) -> bool:
    with _pg(config) as conn:
        q = "SELECT 1 FROM pg_views WHERE schemaname = %s AND viewname = %s"
        return conn.execute(q, (schema, view)).fetchone() is not None


@pytest.mark.seam1
@pytest.mark.provision
def test_teardown_refuses_schemas_that_are_not_purely_this_graphs_version(vgraph, workspace_client):
    from lgk.steps.retire import retire_versions
    from lgk.steps.switch import switch

    switch(vgraph, 2)
    for n in (4, 5):  # a view inside / a stranger's table inside
        with pytest.raises(versions.NotOursError):
            retire_versions(vgraph, versions=[n], confirm_graph_name=GRAPH)
        assert _schema_exists(vgraph, versions.version_schema(vgraph, n))
    assert _view_exists(vgraph, versions.version_schema(vgraph, 4), "sneaky")

    # UC schema with a stranger's table: refused, the table is untouched.
    with pytest.raises(versions.NotOursError):
        retire_versions(vgraph, versions=[7], confirm_graph_name=GRAPH)
    uc = f"{vgraph['lakebase_catalog']}.{versions.version_schema(vgraph, 7)}"
    assert workspace_client.tables.get(f"{uc}.patients").name == "patients"

    # Only our table, but something outside depends on it: refused by the up-front check,
    # before any delete, so RESTRICT can never leave a half-deleted version.
    with pytest.raises(versions.NotOursError, match="reads_v6"):
        retire_versions(vgraph, versions=[6], confirm_graph_name=GRAPH)
    assert _view_exists(vgraph, f"{vgraph['kit_schema']}_other", "reads_v6")
    assert _schema_exists(vgraph, versions.version_schema(vgraph, 6))
    _assert_serving(vgraph, 2)
