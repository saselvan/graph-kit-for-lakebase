"""Seam-1 gates for the function read surface: two front doors, one graph.

One module-scoped provider graph (fresh t- branch, per-run lgk_v1_<6hex> names, the shared
`_teardown_live` in a finally), built, provisioned and switched by the shipped steps with ALL
defaults. Tests act as `public_reader` / `coordinator_reader` by GRANTing the role to the
operator inside a transaction that is rolled back (the same pattern as the reader tests).

Order matters only for the last two: the switch test leaves v1 live, then the full refresh.
"""

import re
from pathlib import Path
from typing import Any, Dict, List

import psycopg
import pytest

from lgk.uc_sql import execute_sql
from lgk.versions import version_schema
from tests.test_build_provision_walks import WAREHOUSE_ID, provisioned_graph

pytestmark = [pytest.mark.seam1, pytest.mark.provision]

N = 12
# Written out by hand (not imported from lgk.served_functions): the public directory contract.
PUBLIC_FUNCTIONS = {"directory_provider", "directory_by_specialty", "directory_by_location",
                    "directory_accepting_new_patients", "directory_affiliations"}
COORDINATOR_FUNCTIONS = {"coordinator_web", "coordinator_tree"}
PUBLIC_COLUMNS = {"provider_key", "provider_name", "specialty", "location", "accepting_new_patients",
                  "affiliated_key", "affiliated_name"}
PUBLIC_PROPERTIES = {"specialty", "location", "accepting_new_patients"}


def _specialty(i: int) -> str:
    return "Cardiology" if i % 2 == 0 else "Dermatology"


def _location(i: int) -> str:
    return "Chicago, IL" if i < 6 else "Evanston, IL"


def _accepting(i: int) -> str:
    return "true" if i % 3 == 0 else "false"


def _create_directory_fixture(w: Any, catalog: str, fixture_schema: str) -> None:
    """Providers with public directory fields, and a chain of affiliations.

    Illustrative values. In a real hospital: specialty from credentialing, location from the
    website CMS, accepting_new_patients from the Epic provider record (scheduling flag).
    """
    schema = f"{catalog}.{fixture_schema}"
    execute_sql(w, WAREHOUSE_ID, f"CREATE SCHEMA IF NOT EXISTS {schema}")
    execute_sql(w, WAREHOUSE_ID, f"""
        CREATE OR REPLACE TABLE {schema}.fixture_providers (
            provider_id STRING, name STRING, specialty STRING, location STRING,
            accepting_new_patients STRING, panel_size STRING
        ) USING DELTA""")
    # hostile: panel_size is a property in the SAME properties map as the public fields;
    # a directory function that returns the map (or reads any key) would leak it.
    rows = [f"('prov_{i:03d}', 'Dr. Provider {i}', '{_specialty(i)}', '{_location(i)}', "
            f"'{_accepting(i)}', '{1000 + i}')" for i in range(N)]
    execute_sql(w, WAREHOUSE_ID, f"INSERT INTO {schema}.fixture_providers VALUES {', '.join(rows)}")
    execute_sql(w, WAREHOUSE_ID, f"""
        CREATE OR REPLACE TABLE {schema}.fixture_affiliations (
            affiliation_id STRING, provider_id STRING, hospital_id STRING, start_date STRING
        ) USING DELTA""")
    aff = [f"('aff_{i:03d}', 'prov_{i:03d}', 'prov_{i + 1:03d}', '2024-01-01')" for i in range(N - 1)]
    aff.append("('aff_dup_000', 'prov_000', 'prov_001', '2025-06-01')")  # hostile: dedup'd by build
    execute_sql(w, WAREHOUSE_ID, f"INSERT INTO {schema}.fixture_affiliations VALUES {', '.join(aff)}")


def _write_directory_mapping(path: Path, catalog: str, source_schema: str) -> None:
    path.write_text(f"""source_catalog: {catalog}
source_schema: {source_schema}
nodes:
  provider:
    source_table: fixture_providers
    natural_key: provider_id
    label: name
    properties: [specialty, location, accepting_new_patients, panel_size]
edges:
  affiliation:
    source_table: fixture_affiliations
    src: provider_id
    src_type: provider
    dst: hospital_id
    dst_type: provider
""")


@pytest.fixture(scope="module")
def graph(workspace_client, lakebase_project, lakebase_catalog, run_folder):
    with provisioned_graph(workspace_client, lakebase_project, lakebase_catalog, run_folder,
                           "roles", _create_directory_fixture,
                           # accepting_new_patients is a provider attribute (A3): no flag
                           write_mapping=_write_directory_mapping) as config:
        yield config


def _pg(config: Dict[str, Any]) -> Any:
    from lgk.branch import connect_branch

    return connect_branch(config)


def _as(conn: Any, role: str) -> None:
    """Act as `role` inside the current transaction (the caller rolls it back)."""
    conn.execute(f"GRANT {role} TO CURRENT_USER")
    conn.execute(f"SET ROLE {role}")


def _denied(conn: Any, sql: str, params: Any = None) -> bool:
    try:
        with conn.transaction():  # savepoint: one denial must not mask the next probe
            conn.execute(sql, params).fetchall()
    except psycopg.errors.InsufficientPrivilege:
        return True
    return False


def _call(config, role: str, sql: str, params: Any = None) -> List[tuple]:
    conn = _pg(config)
    try:
        _as(conn, role)
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.rollback()
        conn.close()


def _node_id(config, key: str) -> str:
    with _pg(config) as conn:
        return str(conn.execute(f"SELECT node_id FROM {version_schema(config)}.node_id_map "
                                f"WHERE natural_key = %s", (key,)).fetchone()[0])


def _row(i: int) -> tuple:
    return (f"prov_{i:03d}", f"Dr. Provider {i}", _specialty(i), _location(i), _accepting(i))


class TestFunctionRolesLive:
    def test_public_reader_gets_the_directory_with_all_defaults(self, graph):
        s = graph["served_schema"]
        call = lambda sql, *p: _call(graph, "public_reader", sql, p)  # noqa: E731
        assert call(f"SELECT * FROM {s}.directory_provider(%s, %s)", "prov_003", 5) == [_row(3)]
        assert call(f"SELECT * FROM {s}.directory_by_specialty(%s, %s)", "Cardiology", 100) == \
            [_row(i) for i in (0, 2, 4, 6, 8, 10)]
        assert call(f"SELECT * FROM {s}.directory_by_specialty(%s, %s)", "Cardiology", 2) == \
            [_row(0), _row(2)]  # the limit is a parameter
        assert call(f"SELECT * FROM {s}.directory_by_location(%s, %s)", "Evanston, IL", 100) == \
            [_row(i) for i in range(6, N)]
        assert call(f"SELECT * FROM {s}.directory_accepting_new_patients(%s, %s)", "Cardiology", 100) == \
            [_row(0), _row(6)]
        assert call(f"SELECT * FROM {s}.directory_affiliations(%s, %s)", "prov_004", 10) == \
            [("prov_004", "prov_005", "Dr. Provider 5")]

    def test_coordinator_reader_walks_and_public_reader_is_denied_them(self, graph):
        s = graph["served_schema"]
        seed = _node_id(graph, "prov_000")
        web = _call(graph, "coordinator_reader",
                    f"SELECT hop, node_id, support FROM {s}.coordinator_web(%s, %s)", (seed, 2))
        assert web == [(1, _node_id(graph, "prov_001"), 1), (2, _node_id(graph, "prov_002"), 1)]
        tree = _call(graph, "coordinator_reader",
                     f"SELECT node_id, depth FROM {s}.coordinator_tree(%s, %s)", (seed, 3))
        assert tree == [(_node_id(graph, f"prov_{i:03d}"), i) for i in (1, 2, 3)]

        conn = _pg(graph)
        try:
            _as(conn, "public_reader")
            assert _denied(conn, f"SELECT * FROM {s}.coordinator_web(%s, 1)", (seed,))
            assert _denied(conn, f"SELECT * FROM {s}.coordinator_tree(%s, 1)", (seed,))
            conn.execute("RESET ROLE")
            _as(conn, "coordinator_reader")
            assert _denied(conn, f"SELECT * FROM {s}.directory_provider('prov_000', 1)")
        finally:
            conn.rollback()
            conn.close()

    def test_coordinator_reader_walks_every_direction(self, graph):
        """Walk direction: p_direction on both functions, called as coordinator_reader (chain prov_000 -> ... -> prov_011)."""
        s = graph["served_schema"]
        n = {i: _node_id(graph, f"prov_{i:03d}") for i in range(6)}
        call = lambda sql, *p: _call(graph, "coordinator_reader", sql, p)  # noqa: E731
        web = f"SELECT hop, node_id, support FROM {s}.coordinator_web(%s, %s, %s)"
        tree = f"SELECT node_id, depth FROM {s}.coordinator_tree(%s, %s, %s)"
        by_id = lambda *ks: sorted((n[k] for k in ks), key=str.encode)  # noqa: E731  (node_id byte order)
        assert call(web, n[2], 2, "outbound") == [(1, n[3], 1), (2, n[4], 1)]
        assert call(web, n[2], 2, "inbound") == [(1, n[1], 1), (2, n[0], 1)]
        assert call(web, n[2], 2, "both") == [(1, x, 1) for x in by_id(1, 3)] + [(2, x, 1) for x in by_id(0, 4)]
        # hostile: the build-deduplicated prov_000 -> prov_001 pair still counts once inbound
        assert call(web, n[1], 1, "inbound") == [(1, n[0], 1)]
        # the two-argument call is the outbound walk (DEFAULT 'outbound')
        assert call(f"SELECT hop, node_id, support FROM {s}.coordinator_web(%s, %s)", n[2], 2) == \
            call(web, n[2], 2, "outbound")
        assert call(tree, n[2], 3, "outbound") == [(n[3], 1), (n[4], 2), (n[5], 3)]
        assert call(tree, n[2], 3, "inbound") == [(n[1], 1), (n[0], 2)]
        assert call(tree, n[2], 3, "both") == ([(x, 1) for x in by_id(1, 3)] + [(x, 2) for x in by_id(0, 4)]
                                               + [(n[5], 3)])
        for fn in ("coordinator_web", "coordinator_tree"):
            with pytest.raises(psycopg.errors.RaiseException, match="direction must be one of"):
                call(f"SELECT * FROM {s}.{fn}(%s, 1, 'sideways')", n[2])

    def test_neither_role_can_select_any_table_or_view(self, graph):
        s, v = graph["served_schema"], version_schema(graph)
        relations = [f"{v}.nodes", f"{v}.edges", f"{v}.node_id_map",
                     f"{s}.nodes_view", f"{s}.edges_view", f"{s}.node_id_map_view"]
        conn = _pg(graph)
        try:
            for rel in relations:  # they exist with rows, so "denied" is not "missing"
                assert conn.execute(f"SELECT count(*) FROM {rel}").fetchone()[0] > 0
            for role in ("public_reader", "coordinator_reader"):
                _as(conn, role)
                allowed = [rel for rel in relations if not _denied(conn, f"SELECT 1 FROM {rel} LIMIT 1")]
                assert allowed == [], f"{role} can SELECT {allowed}"
                conn.execute("RESET ROLE")
            # catalog: no table right at all, even one hidden by a missing schema USAGE
            rows = conn.execute(
                "SELECT r, n.nspname || '.' || c.relname FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace, "
                "unnest(ARRAY['public_reader', 'coordinator_reader']) AS r "
                "WHERE n.nspname IN (%s, %s) AND c.relkind IN ('r','p','v','m','f') "
                "AND has_table_privilege(r, c.oid, 'SELECT')", (s, v)).fetchall()
            assert rows == []
        finally:
            conn.rollback()
            conn.close()

    def test_public_functions_return_only_public_columns_catalog_proof(self, graph):
        s, v = graph["served_schema"], version_schema(graph)
        with _pg(graph) as conn:
            fns = conn.execute(
                "SELECT p.proname, p.prosrc, p.prosecdef, p.proconfig, "
                "array(SELECT a FROM unnest(p.proargnames, p.proargmodes) AS x(a, m) WHERE m = 't'), "
                "has_function_privilege('public_reader', p.oid, 'EXECUTE'), "
                "has_function_privilege('coordinator_reader', p.oid, 'EXECUTE'), "
                "has_function_privilege('public', p.oid, 'EXECUTE') "
                "FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace WHERE n.nspname = %s",
                (s,)).fetchall()
        by_name = {f[0]: f for f in fns}
        assert {f[0] for f in fns if f[5]} == PUBLIC_FUNCTIONS
        assert {f[0] for f in fns if f[6]} == COORDINATOR_FUNCTIONS
        assert [f[0] for f in fns if f[7]] == []  # PUBLIC executes nothing
        for name in PUBLIC_FUNCTIONS:
            _, src, secdef, config, out_cols, *_ = by_name[name]
            assert secdef and config == ["search_path=pg_catalog, pg_temp"], (name, config)
            assert set(out_cols) <= PUBLIC_COLUMNS, (name, out_cols)
            keys = re.findall(r"properties\s*->>\s*'(\w+)'", src)
            assert set(keys) <= PUBLIC_PROPERTIES, (name, keys)
            assert keys or name == "directory_affiliations", name
            assert src.count("properties") == len(keys), f"{name} reads the properties map itself"
            # every schema-qualified relation is the live version's nodes or edges: no other route
            assert set(re.findall(r"\b(lgk_\w+\.\w+)", src)) <= {f"{v}.nodes", f"{v}.edges"}, name
        for name in COORDINATOR_FUNCTIONS:
            assert by_name[name][2] and by_name[name][3] == ["search_path=pg_catalog, pg_temp"]
        # hostile: panel_size is in the very same properties map, and is never returned
        row = _call(graph, "public_reader", f"SELECT * FROM {s}.directory_provider('prov_001', 1)")
        assert "1001" not in {str(c) for c in row[0]}

    def test_functions_follow_the_switch_and_rollback(self, graph):
        from lgk.steps.provision import serve

        s, v1 = graph["served_schema"], version_schema(graph)
        v2 = f"{graph['kit_schema']}_v2"
        conn = _pg(graph)
        conn.autocommit = True
        try:
            # v2: same ids and counts, only the labels differ (hostile: counts cannot tell them apart)
            conn.execute(f"CREATE SCHEMA {v2}")
            for t in ("nodes", "edges", "node_id_map"):
                conn.execute(f"CREATE TABLE {v2}.{t} AS SELECT * FROM {v1}.{t}")
            conn.execute(f"UPDATE {v2}.nodes SET label = label || ' (v2)'")
            serve(conn, graph, v2)
            assert _call(graph, "public_reader", f"SELECT provider_name FROM {s}.directory_provider('prov_003', 1)") \
                == [("Dr. Provider 3 (v2)",)]
            serve(conn, graph, v1)  # rollback = the same script with N-1's name
            assert _call(graph, "public_reader", f"SELECT provider_name FROM {s}.directory_provider('prov_003', 1)") \
                == [("Dr. Provider 3",)]
            # no pg_depend row binds a served function to a version table (string bodies)
            deps = conn.execute(
                "SELECT p.proname FROM pg_depend d JOIN pg_proc p ON p.oid = d.objid "
                "AND d.classid = 'pg_proc'::regclass JOIN pg_namespace n ON n.oid = p.pronamespace "
                "JOIN pg_class c ON c.oid = d.refobjid AND d.refclassid = 'pg_class'::regclass "
                "WHERE n.nspname = %s", (s,)).fetchall()
            assert deps == []
        finally:
            conn.execute(f"DROP SCHEMA IF EXISTS {v2} CASCADE")
            conn.close()

    def test_switch_drops_a_stale_overload(self, graph):
        """An overload of a kit function name the target does not serve is dropped. Walk direction
        made the real change: the older coordinator_web(text, integer) is the stale overload."""
        from lgk.steps.provision import serve

        s, v1 = graph["served_schema"], version_schema(graph)
        stale = f"{s}.coordinator_web(text, integer)"
        count_sql = ("SELECT p.proname, count(*) FROM pg_proc p JOIN pg_namespace n "
                     "ON n.oid = p.pronamespace WHERE n.nspname = %s GROUP BY 1")
        conn = _pg(graph)
        conn.autocommit = True
        try:
            # hostile: same name as a served function, other identity arguments, still granted
            # to its role like a previous version's copy would be
            conn.execute(
                f"CREATE FUNCTION {stale} RETURNS TABLE (hop integer, node_id text, support bigint, "
                f"score double precision) LANGUAGE sql STABLE SECURITY DEFINER "
                f"SET search_path = pg_catalog, pg_temp AS $f$ SELECT 1, ''::text, 1::bigint, 1.0::float8 $f$")
            conn.execute(f"REVOKE ALL ON FUNCTION {stale} FROM PUBLIC")
            conn.execute(f"GRANT EXECUTE ON FUNCTION {stale} TO coordinator_reader")
            counts = dict(conn.execute(count_sql, (s,)).fetchall())
            assert counts["coordinator_web"] == 2  # the stale overload really is there
            serve(conn, graph, v1)
            counts = dict(conn.execute(count_sql, (s,)).fetchall())
            assert counts == {name: 1 for name in PUBLIC_FUNCTIONS | COORDINATOR_FUNCTIONS}
        finally:
            conn.execute(f"DROP FUNCTION IF EXISTS {stale} RESTRICT")
            conn.close()

    def test_full_refresh_keeps_the_functions_and_serves_fresh_rows(self, graph, workspace_client):
        """After a full snapshot refresh the same function (same OID, same body) serves the new row."""
        from lgk.refresh.sync import sync

        s = graph["served_schema"]
        oid_sql = ("SELECT p.oid, p.prosrc, p.proacl::text FROM pg_proc p JOIN pg_namespace n "
                   "ON n.oid = p.pronamespace WHERE n.nspname = %s AND p.proname = 'directory_by_specialty'")
        with _pg(graph) as conn:
            before = tuple(conn.execute(oid_sql, (s,)).fetchone())
        cat, kit = graph["lakebase_catalog"], graph["kit_schema"]
        execute_sql(workspace_client, WAREHOUSE_ID,
                    f"INSERT INTO {cat}.{kit}.nodes VALUES ('lgk-c38-new', 'provider', 'prov_new', "
                    f"'Dr. New', map('specialty', 'Cardiology', 'location', 'Chicago, IL', "
                    f"'accepting_new_patients', 'true', 'panel_size', '9999'))")
        # owner decision (nightly stamp step): a write outside the kit build syncs only once the
        # stamp step has certified it (gate, floors, allowlist, label + manifest)
        from lgk.refresh.operator_stamp import stamp_operator_tables

        stamp_operator_tables(graph)
        sync(graph, full_refresh=True)
        with _pg(graph) as conn:
            assert tuple(conn.execute(oid_sql, (s,)).fetchone()) == before
        got = _call(graph, "public_reader",
                    f"SELECT provider_key FROM {s}.directory_by_specialty('Cardiology', 100)")
        assert got == [(k,) for k in ("prov_000", "prov_002", "prov_004", "prov_006", "prov_008",
                                      "prov_010", "prov_new")]
