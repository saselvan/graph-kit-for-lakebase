"""Tests for build, provision, and walks steps: happy-path spine v1.

Tests the end-to-end flow:
1. Build: nodes/edges Delta tables from graph_mapping.yaml (with dedup)
2. Provision: synced tables, served views, reader role (no privacy flags)
3. Walks: SQL generation for per-depth hops (one walk test)

Seam-1 tests share ONE module-scoped provisioned graph (`live_graph`): one branch, one
build, one provision, all with a unique per-run schema suffix so runs never collide.
Its teardown runs even when setup or a test fails (see `_teardown_live`).
"""

import contextlib
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

import pytest

from lgk.uc_sql import execute_sql
from lgk.versions import version_schema

logger = logging.getLogger(__name__)

# The live tests' SQL warehouse: LGK_WAREHOUSE_ID only. No env var, no live run.
# Shipped code has no default warehouse (lgk.uc_sql.warehouse_id raises MissingWarehouseError).
WAREHOUSE_ID = os.environ.get("LGK_WAREHOUSE_ID")
SYNCED_TABLES = ("nodes", "edges", "node_id_map")
# Synced by provision only when enrich wrote them (closure also needs serve_closure: true).
ENRICH_SYNCED_TABLES = ("node_score", "node_group", "similar_to", "closure")
N_PROVIDERS = 50
N_CHAIN = 30  # chain affiliations prov_i -> prov_{i+1}
FAN_OUT = [("prov_005", "prov_040"), ("prov_010", "prov_041")]  # seeds with 2 neighbours
N_AFFILIATIONS = N_CHAIN + len(FAN_OUT)  # distinct pairs; the fixture adds 1 duplicate
# Wall times (s) of each provisioned_graph's build and provision, keyed by kit_schema, for the
# opt-in numbers recorder (`record_metric`). Kept out of the config the steps read.
GRAPH_TIMINGS: Dict[str, Dict[str, float]] = {}


# ============================================================================
# Shared live setup / teardown
# ============================================================================


def _teardown_live(w: Any, leasor: Any, config: Dict[str, Any]) -> None:
    """Tear down everything one live run created, in the verified order.

    Each step logs and continues on error, so one failure never strands the rest.
    1. DROP the served schema in branch Postgres (views block synced-table deletion).
    2. Delete each synced table (this also deletes its managed pipeline): the fixed list for
       the config's version, plus every SYNCED_TABLE the run manifest recorded (other
       versions, enrichment tables).
    3. Delete the branch (does NOT cascade to synced tables — hence step 2 first).
    4. DROP the UC schemas (synced, kit, fixture, plus every UC_SCHEMA the manifest recorded).
    """
    from lgk.branch import connect_branch

    catalog, kit = config["lakebase_catalog"], config["kit_schema"]
    synced = version_schema(config)

    if config.get("branch_id"):
        try:
            conn = connect_branch(config)
            conn.autocommit = True
            with conn:
                conn.execute(f"DROP SCHEMA IF EXISTS {config['served_schema']} CASCADE")
        except Exception as e:  # noqa: BLE001
            logger.error(f"teardown: drop served schema failed: {e}")

    tables = SYNCED_TABLES + (ENRICH_SYNCED_TABLES if config.get("enriched") else ())
    recorded_tables, recorded_schemas = _manifest_teardown_names(config)
    fixed = [f"{catalog}.{synced}.{table}" for table in tables]
    for full in dict.fromkeys(fixed + recorded_tables):  # dedupe, keep order
        name = f"synced_tables/{full}"
        try:
            op = w.postgres.delete_synced_table(name=name)
            try:
                op.wait()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"teardown: wait on delete {name}: {e}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"teardown: delete synced table {name} failed: {e}")

    if config.get("branch_id"):
        try:
            leasor.delete_branch(config["branch_id"])
        except Exception as e:  # noqa: BLE001
            logger.error(f"teardown: delete branch failed: {e}")

    fixed_schemas = [f"{catalog}.{schema}" for schema in (synced, kit, config["fixture_schema"])]
    for full in dict.fromkeys(fixed_schemas + recorded_schemas):
        try:
            execute_sql(w, WAREHOUSE_ID, f"DROP SCHEMA IF EXISTS {full} CASCADE")
        except Exception as e:  # noqa: BLE001
            logger.error(f"teardown: drop UC schema {full} failed: {e}")


def _manifest_teardown_names(config: Dict[str, Any]) -> tuple:
    """(synced tables, UC schemas) the run manifest recorded, as full names; ([], []) without one."""
    from lgk.manifest import ObjectType, load_run_manifest

    if not config.get("run_folder"):
        return [], []
    try:
        entries = load_run_manifest(config).entries
    except FileNotFoundError:
        return [], []
    except Exception as e:  # noqa: BLE001 — teardown must go on with the fixed lists
        logger.error(f"teardown: reading the run manifest failed: {e}")
        return [], []
    tables = [e.object_id for e in entries if e.object_type == ObjectType.SYNCED_TABLE]
    schemas = [e.object_id for e in entries if e.object_type == ObjectType.UC_SCHEMA]
    return tables, schemas


@contextlib.contextmanager
def provisioned_graph(
    workspace_client: Any,
    lakebase_project: str,
    lakebase_catalog: str,
    run_folder: Path,
    label: str,
    create_fixture: Callable[[Any, str, str], None],
    enrich_methods: Optional[list] = None,
    write_mapping: Optional[Callable[[Path, str, str], None]] = None,
    reclassify: tuple = (),
    extra_config: Optional[Dict[str, Any]] = None,
    accept_orphans: tuple = (),
) -> Iterator[Dict[str, Any]]:
    """Fresh branch + fixture + build (+ enrich) + provision + switch; full teardown afterwards.

    `create_fixture(w, catalog, fixture_schema)` writes fixture_providers and
    fixture_affiliations (the provider mapping shape). With `enrich_methods`, the real
    enrich step runs between build and provision, so provision also syncs node_score.
    `extra_config` is merged into the config before anything runs (e.g. serve_closure).
    """
    from lgk.harness import BranchLeasor
    from lgk.steps.build import build
    from lgk.steps.enrich import enrich
    from lgk.steps.provision import provision
    from lgk.steps.switch import switch

    suffix = uuid.uuid4().hex[:6]
    leasor = BranchLeasor(
        workspace_client=workspace_client,
        project_id=lakebase_project,
        test_base_branch="test-base",
        run_folder=run_folder,
        ttl_seconds=14400,
    )
    config: Dict[str, Any] = {
        "graph_name": "provider_graph",
        "lakebase_project": lakebase_project,
        "lakebase_catalog": lakebase_catalog,
        "kit_schema": f"lgk_v1_{suffix}",
        "served_schema": f"lgk_served_{suffix}",
        "fixture_schema": f"lgk_v1_fx_{suffix}",
        "warehouse_id": WAREHOUSE_ID,
        "enriched": bool(enrich_methods),
        **(extra_config or {}),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            config["branch_id"] = leasor.create_test_branch(f"{label}-{suffix}")["branch_id"]
            run_dir = Path(tmpdir) / "runs" / f"{label}_{suffix}"
            run_dir.mkdir(parents=True)
            config["run_folder"] = str(run_dir)

            create_fixture(workspace_client, lakebase_catalog, config["fixture_schema"])
            (write_mapping or _write_provider_graph_mapping)(
                run_dir / "graph_mapping.yaml", lakebase_catalog, config["fixture_schema"]
            )
            # Pass the privacy gate the real way: UC profile (pattern rates), then privacy.
            from tests.privacy_support import clear_gate_live

            # start_date holds date-shaped strings: a care-subject value pattern.
            # The mapping does not carry it; the operator drops it.
            clear_gate_live(config, workspace_client,
                            drops=[("fixture_affiliations", "start_date")], reclassify=reclassify)
            t0 = time.perf_counter()
            if accept_orphans:
                # The operator flow: build stops on the planted orphan FK and records what it saw;
                # the operator accepts it; build runs again.
                from lgk.keys import OrphanKeyError, resolve_orphans

                with pytest.raises(OrphanKeyError):
                    build(config)
                for table, fk in accept_orphans:
                    resolve_orphans(config["run_folder"], table, fk, "accept", chooser="op",
                                    reason="planted orphan in the test fixture")
            build(config)
            GRAPH_TIMINGS.setdefault(config["kit_schema"], {})["build_s"] = time.perf_counter() - t0
            _assert_build_deduped(workspace_client, config)
            # preflight records the operator as the sync-job owner; provision checks it.
            from tests.preflight_support import preflight_as_operator

            preflight_as_operator(workspace_client, config)
            if enrich_methods:
                (run_dir / "enrich.yaml").write_text(f"methods: {list(enrich_methods)}\n")
                enrich(config)
            t0 = time.perf_counter()
            provision(config)
            GRAPH_TIMINGS.setdefault(config["kit_schema"], {})["provision_s"] = time.perf_counter() - t0
            switch(config)  # provision never serves; the switch is the last step
            config["graph_size"] = served_graph_size(config)  # what every metric line is measured at
            yield config
        finally:
            _teardown_live(workspace_client, leasor, config)


@pytest.fixture(scope="module")
def live_graph(workspace_client, lakebase_project, lakebase_catalog, run_folder):
    """Build + provision one provider graph on a fresh branch; tear it all down after."""
    with provisioned_graph(
        workspace_client, lakebase_project, lakebase_catalog, run_folder,
        "spine", _create_provider_fixture,
    ) as config:
        yield config


def _assert_build_deduped(w: Any, config: Dict[str, Any]) -> None:
    """Gate before provision: build's edge dedup removed the fixture's duplicate pair.

    Checked on Delta, so a missing dedup fails here with a clear message instead of
    later as a synced-table failure (duplicate primary key).
    """
    t = f"{config['lakebase_catalog']}.{config['kit_schema']}.edges"
    total, ids, pairs = map(
        int,
        execute_sql(w, WAREHOUSE_ID,
            f"SELECT COUNT(*), COUNT(DISTINCT edge_id), COUNT(DISTINCT src_node_id, dst_node_id) FROM {t}",
        )[0],
    )
    assert total == ids, f"edge dedup missing: {total} edges but {ids} distinct edge_id in {t}"
    assert total == pairs, f"edge dedup missing: {total} edges but {pairs} distinct (src,dst) in {t}"


def _pg(config: Dict[str, Any]):
    from lgk.branch import connect_branch

    return connect_branch(config)


def served_graph_size(config: Dict[str, Any]) -> Dict[str, int]:
    """{"rows": nodes, "edges": edges} of the SERVED graph (the views the walks read), for the metric lines."""
    s = config["served_schema"]
    with _pg(config) as conn:
        rows = conn.execute(f"SELECT COUNT(*) FROM {s}.nodes_view").fetchone()[0]
        edges = conn.execute(f"SELECT COUNT(*) FROM {s}.edges_view").fetchone()[0]
    return {"rows": int(rows), "edges": int(edges)}


# ============================================================================
# Build
# ============================================================================


class TestBuildStep:
    """Test the build step: SDP to nodes/edges Delta tables."""

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_build_creates_nodes_edges_tables_from_graph_mapping(self, live_graph, workspace_client, record_metric):
        """Build creates nodes/edges in UC, and the hostile duplicate edge is removed."""
        t = f"{live_graph['lakebase_catalog']}.{live_graph['kit_schema']}"
        nodes = int(execute_sql(workspace_client, WAREHOUSE_ID, f"SELECT COUNT(*) FROM {t}.nodes")[0][0])
        edges, distinct_ids = map(
            int,
            execute_sql(workspace_client, WAREHOUSE_ID, f"SELECT COUNT(*), COUNT(DISTINCT edge_id) FROM {t}.edges")[0],
        )
        assert nodes == N_PROVIDERS
        # One row per node_id in nodes and node_id_map (the synced tables' primary key), and
        # the NULL key made no node.
        for table in ("nodes", "node_id_map"):
            n, d = map(int, execute_sql(workspace_client, WAREHOUSE_ID,
                                        f"SELECT COUNT(*), COUNT(DISTINCT node_id) FROM {t}.{table}")[0])
            assert n == d == N_PROVIDERS, (table, n, d)
        labels = [r[0] for r in execute_sql(workspace_client, WAREHOUSE_ID,
                                            f"SELECT label FROM {t}.nodes WHERE natural_key = 'prov_000'")]
        assert labels == ["Dr. Provider 0"]
        assert execute_sql(workspace_client, WAREHOUSE_ID,
                           f"SELECT COUNT(*) FROM {t}.nodes WHERE natural_key IS NULL")[0][0] in (0, "0")
        # The fixture has N_AFFILIATIONS + 1 rows; the extra is a duplicate pair.
        assert edges == N_AFFILIATIONS
        assert distinct_ids == edges
        branch = live_graph["branch_id"]
        record_metric("build.wall", GRAPH_TIMINGS[live_graph["kit_schema"]]["build_s"], "s",
                      branch=branch, rows=nodes, edges=edges)
        record_metric("build.nodes", nodes, "nodes", branch=branch, rows=nodes, edges=edges)
        record_metric("build.edges", edges, "edges", branch=branch, rows=nodes, edges=edges)


# ============================================================================
# Provision
# ============================================================================


def _record_provision(record_metric: Any, graph: Dict[str, Any]) -> None:
    """Provision wall time, and the synced row count per table as read in the branch's Postgres."""
    branch = graph["branch_id"]
    size = graph["graph_size"]
    record_metric("provision.wall", GRAPH_TIMINGS[graph["kit_schema"]]["provision_s"], "s", branch=branch, **size)
    with _pg(graph) as conn:
        for table in SYNCED_TABLES:
            rows = conn.execute(f"SELECT COUNT(*) FROM {version_schema(graph)}.{table}").fetchone()[0]
            record_metric("provision.synced_rows", int(rows), "rows", branch=branch, table=table, **size)


class TestProvisionStep:
    """Test the provision step: synced tables, views, roles."""

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_provision_creates_synced_table_snapshot(self, live_graph, workspace_client, record_metric):
        """Provision creates an ONLINE synced table on the run's branch, with every row."""
        dest = f"{live_graph['lakebase_catalog']}.{version_schema(live_graph)}.nodes"
        st = workspace_client.postgres.get_synced_table(name=f"synced_tables/{dest}")
        assert "ONLINE_NO_PENDING_UPDATE" in str(st.status.detailed_state.value)
        # Binding to the run's branch is proven by the rows being in THIS branch's
        # Postgres (_pg connects to the run's branch endpoint).
        with _pg(live_graph) as conn:
            n = conn.execute(f"SELECT COUNT(*) FROM {version_schema(live_graph)}.nodes").fetchone()[0]
        assert n == N_PROVIDERS
        if record_metric.enabled:
            _record_provision(record_metric, live_graph)


    @pytest.mark.seam1
    @pytest.mark.provision
    def test_provision_indexes_both_edges_join_columns(self, live_graph):
        """Provision itself (advise never makes the dst one) indexes edges src and dst on the
        version: without them a fresh version scans every edge per hop (scale spike, 10^6 nodes)."""
        with _pg(live_graph) as conn:
            rows = conn.execute(
                "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = %s AND tablename = 'edges'",
                (version_schema(live_graph),)).fetchall()
        names = {r[0]: r[1] for r in rows}
        assert "(src_node_id)" in names.get("lgk_edges_src_node_id_idx", ""), rows
        assert "(dst_node_id)" in names.get("lgk_edges_dst_node_id_idx", ""), rows

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_provision_creates_served_view_security_invoker_off(self, live_graph):
        """All three served views exist with security_invoker OFF (checked in pg_class)."""
        with _pg(live_graph) as conn:
            rows = conn.execute(
                """
                SELECT c.relname, c.reloptions
                FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = %s AND c.relkind = 'v'
                ORDER BY c.relname
                """,
                (live_graph["served_schema"],),
            ).fetchall()
        assert [r[0] for r in rows] == ["edges_view", "node_id_map_view", "nodes_view"]
        for name, opts in rows:
            opts = [o.lower() for o in (opts or [])]
            assert any(o in ("security_invoker=off", "security_invoker=false", "security_invoker=0") for o in opts), (
                f"{name} must have security_invoker OFF, got {opts}"
            )

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_provision_creates_reader_role(self, live_graph):
        """Provision should create a LOGIN-only reader role."""
        with _pg(live_graph) as conn:
            result = conn.execute(
                "SELECT rolname, rolcanlogin, rolsuper FROM pg_roles WHERE rolname = 'reader'"
            ).fetchone()
        assert result is not None, "reader role should exist"
        role_name, can_login, is_super = result
        assert role_name == "reader"
        assert can_login is True, "reader role should have LOGIN privilege"
        assert is_super is False


class TestPrivacy:
    """Test privacy controls: reader denied on raw tables, allowed on view."""

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_reader_denied_on_raw_nodes_table(self, live_graph):
        """Reader gets 'permission denied' on every raw synced table (nodes, edges, node_id_map)."""
        import psycopg

        synced = version_schema(live_graph)
        conn = _pg(live_graph)
        try:
            # Prove each raw table exists with rows as the operator first, so "denied"
            # can't be a missing-table error.
            for table in SYNCED_TABLES:
                assert conn.execute(f"SELECT COUNT(*) FROM {synced}.{table}").fetchone()[0] > 0

            # Operator becomes a member of reader (rolled back below), then acts as reader.
            conn.execute("GRANT reader TO CURRENT_USER")
            conn.execute("SET ROLE reader")
            denied = []
            for table in SYNCED_TABLES:
                try:
                    with conn.transaction():  # savepoint: one denial must not mask the next probe
                        conn.execute(f"SELECT COUNT(*) FROM {synced}.{table}")
                except psycopg.errors.InsufficientPrivilege:
                    denied.append(table)
            assert denied == list(SYNCED_TABLES), f"reader was NOT denied on {set(SYNCED_TABLES) - set(denied)}"
        finally:
            conn.rollback()
            conn.close()

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_reader_can_select_served_view(self, live_graph):
        """Reader role should be able to SELECT from served view."""
        conn = _pg(live_graph)
        cursor = conn.cursor()
        try:
            raw = f"{version_schema(live_graph)}.nodes"
            cursor.execute(f"SELECT COUNT(*) FROM {raw}")
            raw_count = cursor.fetchone()[0]
            assert raw_count > 0

            cursor.execute("GRANT reader TO CURRENT_USER")
            cursor.execute("SET ROLE reader")
            cursor.execute(f"SELECT COUNT(*) FROM {live_graph['served_schema']}.nodes_view")
            # The reader sees every row through the view, not a partial or empty set.
            assert cursor.fetchone()[0] == raw_count
        finally:
            conn.rollback()
            cursor.close()
            conn.close()


# ============================================================================
# Walks
# ============================================================================


class TestWalks:
    """Test walks step: SQL generation for graph traversal."""

    @pytest.mark.seam1
    @pytest.mark.provision
    def test_walk_same_answer_reference_for_20_seeds(self, live_graph, workspace_client, record_metric):
        """The live walk over the served views equals a Spark reference for 20 seeds.

        The reference is computed in Spark from the SOURCE fixture affiliations (one edge
        per distinct provider->hospital pair), mapped to node_ids through the Delta
        node_id_map. It does not read the build's edges table or any Postgres table.
        Both the neighbour set and edge_count per neighbour must match.
        These edges are unfolded (no care-subject data), so the privacy floor does not apply.
        """
        samples = assert_walks_match_reference(live_graph, workspace_client)
        record_metric.latencies("walk.spine.depth1", samples, branch=live_graph["branch_id"], depth=1,
                                **live_graph["graph_size"])


# ============================================================================
# Fixtures and helper functions
# ============================================================================


def assert_walks_match_reference(graph: Dict[str, Any], workspace_client: Any) -> list:
    """The live depth-1 walk over the served views equals the Spark source reference, 20 seeds.

    Returns the per-seed walk query times (s, perf_counter) for the opt-in numbers recorder."""
    from lgk.steps.walks import get_walk_query

    cat, kit, fx = graph["lakebase_catalog"], graph["kit_schema"], graph["fixture_schema"]
    keys = [f"prov_{i:03d}" for i in range(20)]
    key_list = ", ".join(f"'{k}'" for k in keys)
    reference: Dict[str, Dict[str, int]] = {k: {} for k in keys}
    for src_key, dst_id, cnt in execute_sql(workspace_client, WAREHOUSE_ID,
        f"""
        SELECT a.provider_id, m_dst.node_id, COUNT(*)
        FROM (SELECT DISTINCT provider_id, hospital_id FROM {cat}.{fx}.fixture_affiliations) a
        JOIN {cat}.{kit}.node_id_map m_dst
          ON m_dst.node_type = 'provider' AND m_dst.natural_key = a.hospital_id
        WHERE a.provider_id IN ({key_list})
        GROUP BY a.provider_id, m_dst.node_id
        """,
    ):
        reference[src_key][str(dst_id)] = int(cnt)
    seed_ids = {
        k: str(v)
        for k, v in execute_sql(workspace_client, WAREHOUSE_ID,
            f"SELECT natural_key, node_id FROM {cat}.{kit}.node_id_map "
            f"WHERE node_type = 'provider' AND natural_key IN ({key_list})",
        )
    }
    assert sorted(seed_ids) == keys
    # Non-vacuous: every seed has a neighbour, and the fan-out seeds have two.
    assert all(reference[k] for k in keys)
    assert sum(len(v) for v in reference.values()) == 20 + len(FAN_OUT)

    walk_query = get_walk_query(
        start_node_id="unused", depth=1, served_schema=graph["served_schema"],
        top_n=1000, run_folder=graph["run_folder"],  # floor/folded from flags.json
    )
    samples = []
    with _pg(graph) as conn:
        for k in keys:
            t0 = time.perf_counter()
            rows = conn.execute(walk_query, {"start_node_id": seed_ids[k]}).fetchall()
            samples.append(time.perf_counter() - t0)
            walked = {str(node_id): int(support) for _hop, node_id, support, _score in rows}
            assert walked == reference[k], f"walk mismatch for seed {k}: {walked} != {reference[k]}"
    return samples


def _create_provider_fixture(w: Any, catalog: str, fixture_schema: str) -> None:
    """Create the provider-directory source tables, deterministically (CREATE OR REPLACE).

    - fixture_providers: N_PROVIDERS provider rows
    - fixture_affiliations: N_AFFILIATIONS distinct provider->provider pairs (a chain
      plus FAN_OUT extras, so two seeds have two neighbours), plus ONE
      hostile duplicate row (aff_dup_000 repeats aff_000's pair). Build's dedup must
      remove it, so edges must have exactly N_AFFILIATIONS rows.
    """
    schema = f"{catalog}.{fixture_schema}"
    execute_sql(w, WAREHOUSE_ID, f"CREATE SCHEMA IF NOT EXISTS {schema}")

    execute_sql(w, WAREHOUSE_ID,
        f"""
        CREATE OR REPLACE TABLE {schema}.fixture_providers (
            provider_id STRING, name STRING, specialty STRING
        ) USING DELTA
        """,
    )
    values = [f"('prov_{i:03d}', 'Dr. Provider {i}', 'Cardiology')" for i in range(N_PROVIDERS)]
    # hostile: a NULL natural key; build drops it (it would hash to one shared node_id).
    # A DUPLICATE natural key is no longer collapsed to a winner: build refuses it
    # (lgk.keys.DuplicateKeyError, offline in tests/test_keys.py), so this shared fixture has none.
    values += ["(NULL, 'Dr. No Key', 'Cardiology')"]
    execute_sql(w, WAREHOUSE_ID, f"INSERT INTO {schema}.fixture_providers VALUES {', '.join(values)}")

    execute_sql(w, WAREHOUSE_ID,
        f"""
        CREATE OR REPLACE TABLE {schema}.fixture_affiliations (
            affiliation_id STRING, provider_id STRING, hospital_id STRING, start_date STRING
        ) USING DELTA
        """,
    )
    aff = [
        f"('aff_{i:03d}', 'prov_{i % N_PROVIDERS:03d}', 'prov_{(i + 1) % N_PROVIDERS:03d}', '2024-01-01')"
        for i in range(N_CHAIN)
    ]
    aff += [f"('aff_fan_{i}', '{src}', '{dst}', '2024-01-01')" for i, (src, dst) in enumerate(FAN_OUT)]
    # HOSTILE ROW: same (provider_id, hospital_id) pair as aff_000, different id/date.
    aff.append("('aff_dup_000', 'prov_000', 'prov_001', '2025-06-01')")
    execute_sql(w, WAREHOUSE_ID, f"INSERT INTO {schema}.fixture_affiliations VALUES {', '.join(aff)}")


def _write_provider_graph_mapping(
    graph_mapping: Path, catalog: str, source_schema: str = "fixture_provider_dir"
) -> None:
    """Write a provider-directory graph_mapping.yaml."""
    content = f"""# Provider directory graph mapping
source_catalog: {catalog}
source_schema: {source_schema}

nodes:
  provider:
    source_table: fixture_providers
    natural_key: provider_id
    label: name
    properties: [specialty]

edges:
  affiliation:
    source_table: fixture_affiliations
    src: provider_id
    src_type: provider
    dst: hospital_id
    dst_type: provider
"""
    with open(graph_mapping, "w") as f:
        f.write(content)
