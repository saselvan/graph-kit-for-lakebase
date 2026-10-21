"""Offline tests for version naming and the retire/teardown confirmation.

The live switch / refusal behaviour is in test_versions_live.py (Seam 1).
"""

import pytest

from lgk import versions
from lgk.steps import retire as retire_step


def test_version_schema_defaults_to_lgk_graph_vN():
    assert versions.version_schema({"graph_name": "providers", "version": 3}) == "lgk_providers_v3"


def test_version_schema_follows_explicit_kit_schema():
    cfg = {"graph_name": "providers", "kit_schema": "lgk_v1_ab12cd", "version": 2}
    assert versions.version_schema(cfg) == "lgk_v1_ab12cd_v2"
    # An explicit version argument wins over the config's version.
    assert versions.version_schema(cfg, 7) == "lgk_v1_ab12cd_v7"


def test_served_schema_defaults_to_kit_schema_and_explicit_wins():
    assert versions.served_schema({"graph_name": "providers"}) == "lgk_providers"
    assert versions.served_schema({"graph_name": "p", "served_schema": "lgk_v1_x"}) == "lgk_v1_x"


@pytest.mark.parametrize("bad", [0, -1, "2", True])  # hostile: "2" is a str, True is an int subclass
def test_version_number_must_be_a_positive_int(bad):
    with pytest.raises(ValueError):
        versions.version_schema({"graph_name": "providers", "version": bad})


def _no_db(_config):
    raise AssertionError("a refused retire/teardown must not even connect")


@pytest.mark.parametrize("typed", ["", "provider", "PROVIDERS", " providers"])  # hostile: near-misses
def test_retire_and_teardown_refuse_without_exact_graph_name(monkeypatch, typed):
    monkeypatch.setattr(retire_step, "connect_branch", _no_db)
    cfg = {"graph_name": "providers", "lakebase_catalog": "c"}
    with pytest.raises(versions.ConfirmationError):
        retire_step.retire(cfg, version=1, confirm_graph_name=typed)
    with pytest.raises(versions.ConfirmationError):
        retire_step.retire_versions(cfg, versions=[1], confirm_graph_name=typed)


# ---------------------------------------------------------------------------
# Names cannot collide with another graph's version schemas (review blocker)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "graph",
    # hostile: 'providers_v2' would default to lgk_providers_v2 — graph "providers"' v2
    ["providers_v2", "Bad-Name", "x; drop", "", "9lives", "a v1"],
)
def test_hostile_graph_names_are_refused(graph):
    with pytest.raises(ValueError):
        versions.served_schema({"graph_name": graph})
    with pytest.raises(ValueError):
        versions.version_schema({"graph_name": graph, "version": 1})


@pytest.mark.parametrize("key", ["kit_schema", "served_schema"])
@pytest.mark.parametrize("name", ["lgk_providers_v2", "lgk_x_v10", "LGK_X", "lgk-x", "lgk_x; drop schema y"])
def test_hostile_kit_and_served_schema_names_are_refused(key, name):
    cfg = {"graph_name": "providers", key: name}
    with pytest.raises(ValueError):
        versions.served_schema(cfg)


def test_names_with_v_inside_but_not_a_version_suffix_are_fine():
    # hostile the other way: a lazy "contains _v" rule would refuse these.
    assert versions.served_schema({"graph_name": "lgk_v1_ab12cd"}) == "lgk_lgk_v1_ab12cd"
    assert versions.version_schema({"graph_name": "p", "kit_schema": "lgk_v1_ab12cd_served"}, 3) == (
        "lgk_v1_ab12cd_served_v3"
    )


class _RecordingConn:
    """Records the one catalog query live_version_schemas sends (no database)."""

    def __init__(self):
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        return self

    def fetchall(self):
        return []


def test_live_check_matches_the_served_schema_by_exact_name():
    # A prefix/LIKE match would treat views in lgk_x_served_old (or lgk_x_v1) as live views of lgk_x.
    conn = _RecordingConn()
    versions.live_version_schemas(conn, "lgk_x")
    (sql, params), = conn.calls
    assert params == ("lgk_x",)
    where = sql.split("WHERE", 1)[1]
    assert "vn.nspname = %s" in where
    assert "LIKE" not in sql.upper()


def test_uc_ownership_refuses_any_table_that_is_not_a_version_table():
    versions.refuse_unless_ours_uc(["nodes", "edges", "node_id_map"], "c.s")
    versions.refuse_unless_ours_uc([], "c.s")  # absent schema: nothing to drop
    with pytest.raises(versions.NotOursError):
        versions.refuse_unless_ours_uc(["nodes", "patients"], "c.s")  # hostile: a stranger's table


def test_uc_ownership_allows_exactly_the_named_event_logs():
    own, sibling = "event_log_a1b2c3d4_0000_4000_8000_0000000000aa", "event_log_0d1e2f30_0000_4000_8000_00000000beef"
    versions.refuse_unless_ours_uc(["nodes", own], "c.s", [own])
    with pytest.raises(versions.NotOursError, match=own):  # not named: an event log is not ours by its name
        versions.refuse_unless_ours_uc(["nodes", own], "c.s")
    with pytest.raises(versions.NotOursError, match=sibling):  # hostile: another pipeline's log, same schema
        versions.refuse_unless_ours_uc(["nodes", own, sibling], "c.s", [own])


class TestIdentifierLength:
    """Postgres truncates identifiers at 63 bytes; names must never rely on that."""

    def test_long_graph_cannot_collide_with_another_graphs_version_after_truncation(self):
        from lgk import versions

        # hostile: A's v1 is "lgk_" + a*56 + "_v1" (63 bytes); B's kit name would truncate onto it.
        with pytest.raises(ValueError):
            versions.version_schema({"graph_name": "a" * 56}, 1)
        with pytest.raises(ValueError):
            versions.served_schema({"graph_name": "a" * 56 + "_v1x"})

    def test_explicit_kit_schema_is_validated_at_the_version_seam(self):
        from lgk import versions

        with pytest.raises(ValueError):
            versions.version_schema({"graph_name": "p", "kit_schema": "lgk_x_v2"}, 1)
        with pytest.raises(ValueError):
            versions.version_schema({"graph_name": "p", "kit_schema": "k" * 58}, 1)

    def test_every_valid_name_and_version_fits_in_63_bytes(self):
        from lgk import versions

        longest = "k" * versions.MAX_NAME_BYTES
        name = versions.version_schema({"graph_name": "p", "kit_schema": longest}, versions.MAX_VERSION)
        assert len(name.encode()) == 63
        with pytest.raises(ValueError):
            versions.version_schema({"graph_name": "p", "kit_schema": longest}, versions.MAX_VERSION + 1)


# ============================================================================
# enrichment views: optional views, the switch rule, the grant block, the "ours" set
# ============================================================================
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

COLS_NODES = [("node_id", "text"), ("label", "text")]


class TestViewStatements:
    """The view rule for one view, as a pure function of the live and target column lists."""

    def test_no_live_view_is_created_or_replaced(self):
        assert versions.view_statements("s", "nodes_view", "k_v2", "nodes", [], COLS_NODES, True) == [
            "CREATE OR REPLACE VIEW s.nodes_view WITH (security_invoker=off) AS SELECT * FROM k_v2.nodes"]

    def test_target_that_appends_a_column_is_replaced_in_place(self):
        target = COLS_NODES + [("weight", "bigint")]
        assert versions.view_statements("s", "nodes_view", "k_v2", "nodes", COLS_NODES, target, True) == [
            "CREATE OR REPLACE VIEW s.nodes_view WITH (security_invoker=off) AS SELECT * FROM k_v2.nodes"]

    @pytest.mark.parametrize("target", [
        COLS_NODES[:1],                                    # a column dropped (rollback across weight)
        [("node_id", "bigint"), ("label", "text")],        # hostile: same names, a type changed
        [("label", "text"), ("node_id", "text")],          # hostile: same columns, reordered
        [("node_id", "text"), ("name", "text")],           # a column renamed
    ])
    def test_any_other_change_drops_and_recreates_restrict(self, target):
        assert versions.view_statements("s", "nodes_view", "k_v1", "nodes", COLS_NODES, target, True) == [
            "DROP VIEW s.nodes_view RESTRICT",
            "CREATE VIEW s.nodes_view WITH (security_invoker=off) AS SELECT * FROM k_v1.nodes"]

    def test_an_optional_view_whose_table_is_absent_is_dropped(self):
        assert versions.view_statements("s", "closure_view", "k_v2", "closure", COLS_NODES, [], False) == [
            "DROP VIEW IF EXISTS s.closure_view RESTRICT"]

    def test_a_required_view_whose_table_is_absent_is_never_silently_dropped(self):
        # the CREATE names the missing table, so the switch fails and rolls back
        stmts = versions.view_statements("s", "node_id_map_view", "k_v3", "node_id_map", COLS_NODES, [], True)
        assert stmts[-1].startswith("CREATE VIEW s.node_id_map_view") and "IF EXISTS" not in " ".join(stmts)


class _CatalogConn:
    """A fake connection: answers the column query from `columns`, records every statement and
    whether it ran inside the one transaction."""

    def __init__(self, columns):
        import psycopg

        self.columns = columns
        self.log = []
        self.in_txn = False
        self.committed = False
        self.info = type("I", (), {"transaction_status": psycopg.pq.TransactionStatus.IDLE})()

    def execute(self, sql, params=None):
        if "pg_attribute" in sql:
            return _Rows(self.columns.get(params, []))
        self.log.append(("txn:" if self.in_txn else "auto:") + sql)
        return _Rows([])

    def transaction(self):
        conn = self

        class _T:
            def __enter__(self):
                conn.in_txn = True

            def __exit__(self, exc_type, *a):
                conn.in_txn = False
                conn.committed = exc_type is None
                return False

        return _T()


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return list(self.rows)


def _version(schema, *tables):
    return {(schema, t): [("c", "text")] for t in tables}


class TestSwitchOptionalViews:
    def test_switch_drops_an_optional_view_the_target_lacks(self):
        # hostile: every optional view is live now (v1 had them all); v2 has only node_score
        served_views = {("s", v): [("c", "text")] for v, _ in versions.SERVED_VIEWS + versions.OPTIONAL_SERVED_VIEWS}
        conn = _CatalogConn({**served_views, **_version("k_v2", "nodes", "edges", "node_id_map", "node_score")})
        served = versions.switch_views(conn, "s", "k_v2", reader="reader")
        assert conn.committed
        for view in ("node_group_view", "similar_to_view", "closure_view"):
            assert f"txn:DROP VIEW IF EXISTS s.{view} RESTRICT" in conn.log
            assert not any(view in s and "CREATE" in s for s in conn.log)
        assert ("txn:CREATE OR REPLACE VIEW s.node_score_view WITH (security_invoker=off) AS "
                "SELECT * FROM k_v2.node_score") in conn.log
        assert served == ["nodes_view", "edges_view", "node_id_map_view", "node_score_view"]

    def test_a_rollback_to_a_version_with_the_table_brings_the_view_back(self):
        conn = _CatalogConn(_version("k_v1", "nodes", "edges", "node_id_map", "closure"))
        assert "closure_view" in versions.switch_views(conn, "s", "k_v1")
        assert ("txn:CREATE OR REPLACE VIEW s.closure_view WITH (security_invoker=off) AS "
                "SELECT * FROM k_v1.closure") in conn.log

    def test_grant_block_covers_present_optional_views(self):
        conn = _CatalogConn(_version("k_v2", "nodes", "edges", "node_id_map", "node_score", "closure"))
        versions.switch_views(conn, "s", "k_v2", ["GRANT EXECUTE fn"], reader="reader")
        grants = [s for s in conn.log if "GRANT" in s or "REVOKE" in s]
        assert grants == [
            "txn:GRANT USAGE ON SCHEMA s TO reader",
            *(f"txn:GRANT SELECT ON s.{v} TO reader"
              for v in ("nodes_view", "edges_view", "node_id_map_view", "node_score_view", "closure_view")),
            *(f"txn:REVOKE ALL ON k_v2.{t} FROM reader"
              for t in ("nodes", "edges", "node_id_map", "node_score", "closure")),
            "txn:GRANT EXECUTE fn",  # the function block follows the view grants, same transaction
        ]

    def test_without_a_reader_role_the_switch_grants_nothing(self):
        conn = _CatalogConn(_version("k_v2", "nodes", "edges", "node_id_map"))
        versions.switch_views(conn, "s", "k_v2")
        assert not [s for s in conn.log if "GRANT" in s or "REVOKE" in s]

    def test_the_view_grants_are_inside_the_switch_transaction(self):
        conn = _CatalogConn(_version("k_v2", "nodes", "edges", "node_id_map"))
        versions.switch_views(conn, "s", "k_v2", reader="reader")
        assert all(s.startswith("txn:") for s in conn.log), conn.log


class TestOursSet:
    def test_retire_accepts_a_version_holding_enrichment_tables(self):
        versions.refuse_unless_ours_uc(["nodes", "edges", "node_id_map", "node_score", "node_group",
                                        "similar_to", "closure"], "c.s")
        with pytest.raises(versions.NotOursError, match="seed_score"):
            versions.refuse_unless_ours_uc(["nodes", "seed_score"], "c.s")  # hostile: no version syncs it

    def test_pg_ours_set_includes_the_enrichment_tables(self):
        conn = _RowsConn([("nodes", "p"), ("closure", "p"), ("similar_to", "r")])
        versions.refuse_unless_ours_pg(conn, "k_v2")
        with pytest.raises(versions.NotOursError, match="closure_view"):
            versions.refuse_unless_ours_pg(_RowsConn([("closure", "p"), ("closure_view", "v")]), "k_v2")

    def test_retire_drops_every_ours_table_restrict(self, monkeypatch):
        conn = _CatalogConn({})
        conn.execute = lambda sql, params=None: (conn.log.append(sql), _Rows([]))[1]
        monkeypatch.setattr(retire_step.versions, "refuse_if_live", lambda *a: None)
        monkeypatch.setattr(retire_step.versions, "refuse_unless_ours_pg", lambda *a: None)
        retire_step._drop_pg_schema(conn, "s", "k_v2")
        drops = [s for s in conn.log if s.startswith("DROP TABLE")]
        assert drops == [f"DROP TABLE IF EXISTS k_v2.{t} RESTRICT" for t in
                         ("nodes", "edges", "node_id_map", "node_score", "node_group", "similar_to", "closure")]


class _RowsConn:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, sql, params=None):
        return _Rows(self.rows)


# ------------------------------------------------------------------- local Postgres (X3, real)
PG_BIN = next((Path(p) for p in ("/opt/homebrew/opt/postgresql@17/bin", "/opt/homebrew/opt/postgresql@16/bin",
                                 "/usr/lib/postgresql/17/bin", "/usr/lib/postgresql/16/bin")
               if (Path(p) / "initdb").exists()), None) or (
    Path(shutil.which("initdb")).parent if shutil.which("initdb") else None)
PG_PORT = 54332  # tests/test_walks.py uses 54329, tests/test_lookups.py 54331


@pytest.fixture(scope="module")
def pg():
    """A throwaway Postgres cluster: versions v1 (base columns, closure) and v2 (edges gains a
    column, no closure), a `reader` role the operator can SET ROLE to."""
    if PG_BIN is None:
        pytest.skip("no local Postgres binaries (initdb) installed")
    import psycopg

    tmp = Path(tempfile.mkdtemp(prefix="lgkpgv"))
    data, sock = tmp / "data", tmp / "s"
    sock.mkdir()
    subprocess.run([PG_BIN / "initdb", "-D", data, "-U", "postgres", "--no-sync", "-E", "UTF8",
                    "--locale=C"], check=True, capture_output=True)
    subprocess.run([PG_BIN / "pg_ctl", "-D", data, "-w", "-l", tmp / "log", "-o",
                    f"-k {sock} -c listen_addresses='' -p {PG_PORT}", "start"], check=True,
                   capture_output=True)
    conn = None
    try:
        conn = psycopg.connect(host=str(sock), port=PG_PORT, user="postgres", dbname="postgres",
                               autocommit=True)
        conn.execute("CREATE ROLE reader LOGIN")
        for v, extra in (("k_v1", ""), ("k_v2", ", weight bigint")):
            conn.execute(f"CREATE SCHEMA {v}")
            conn.execute(f"CREATE TABLE {v}.nodes (node_id text, label text)")
            conn.execute(f"CREATE TABLE {v}.edges (src_node_id text, dst_node_id text{extra})")
            conn.execute(f"CREATE TABLE {v}.node_id_map (node_id text, natural_key text)")
            conn.execute(f"INSERT INTO {v}.nodes VALUES ('a', '{v}')")
        conn.execute("CREATE TABLE k_v1.closure (ancestor_id text, descendant_id text, depth integer)")
        conn.execute("INSERT INTO k_v1.closure VALUES ('r', 'd', 2)")
        conn.connect_args = dict(host=str(sock), port=PG_PORT, user="postgres", dbname="postgres")
        yield conn
    finally:
        if conn is not None:
            conn.close()
        subprocess.run([PG_BIN / "pg_ctl", "-D", data, "-m", "immediate", "stop"], capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)


def _reader_can(conn, relation):
    return conn.execute("SELECT has_table_privilege('reader', %s, 'SELECT')", (relation,)).fetchone()[0]


def _view_cols(conn, view):
    return [c for c, _t in versions.relation_columns(conn, "srv", view)]


class TestSwitchOnPostgres:
    def test_rollback_across_a_column_change_and_back_keeps_reader_grants(self, pg):
        versions.switch_views(pg, "srv", "k_v1", reader="reader")
        assert _view_cols(pg, "edges_view") == ["src_node_id", "dst_node_id"]
        assert _reader_can(pg, "srv.closure_view")
        versions.switch_views(pg, "srv", "k_v2", reader="reader")  # appends weight: replaced in place
        assert _view_cols(pg, "edges_view") == ["src_node_id", "dst_node_id", "weight"]
        assert pg.execute("SELECT to_regclass('srv.closure_view')").fetchone()[0] is None
        # hostile: the rollback drops a column, which CREATE OR REPLACE VIEW cannot do
        versions.switch_views(pg, "srv", "k_v1", reader="reader")
        assert _view_cols(pg, "edges_view") == ["src_node_id", "dst_node_id"]
        assert pg.execute("SELECT * FROM srv.closure_view").fetchall() == [("r", "d", 2)]
        for view in ("nodes_view", "edges_view", "node_id_map_view", "closure_view"):
            assert _reader_can(pg, f"srv.{view}"), view
        for table in ("nodes", "edges", "closure"):
            assert not _reader_can(pg, f"k_v1.{table}"), table

    def test_no_committed_state_has_an_ungranted_view(self, pg):
        """A second session probes reader's grant the moment the switch commits (M2.18)."""
        import psycopg

        versions.switch_views(pg, "srv", "k_v2", reader="reader")
        probe = psycopg.connect(**pg.connect_args, autocommit=True)
        seen = []

        class _Probing:
            """The switch connection; on commit of its transaction, the probe reads the grant."""

            def __init__(self, conn):
                self._c = conn
                self.info = conn.info

            def execute(self, *a, **k):
                return self._c.execute(*a, **k)

            def transaction(self):
                inner = self._c.transaction()

                class _T:
                    def __enter__(self_t):
                        return inner.__enter__()

                    def __exit__(self_t, *exc):
                        out = inner.__exit__(*exc)
                        seen.append(_reader_can(probe, "srv.edges_view"))
                        return out

                return _T()

        try:
            versions.switch_views(_Probing(pg), "srv", "k_v1", reader="reader")  # drop/create path
        finally:
            probe.close()
        assert seen == [True]
        assert _reader_can(pg, "srv.edges_view")

    def test_a_dependent_outside_the_kit_fails_the_drop_loudly(self, pg):
        import psycopg

        versions.switch_views(pg, "srv", "k_v2", reader="reader")
        pg.execute("CREATE SCHEMA IF NOT EXISTS app")
        pg.execute("CREATE VIEW app.mine AS SELECT * FROM srv.edges_view")
        try:
            with pytest.raises(psycopg.errors.DependentObjectsStillExist):
                versions.switch_views(pg, "srv", "k_v1", reader="reader")
            assert _view_cols(pg, "edges_view") == ["src_node_id", "dst_node_id", "weight"]  # rolled back
        finally:
            pg.execute("DROP SCHEMA app CASCADE")


class TestCatalogReadsOnPostgres:
    """Prover survivors P09/P13: only tables (and views for the served side) count, never an
    index or a view that carries an enrichment table's name."""

    def test_an_index_named_like_an_absent_table_has_no_columns_and_the_view_is_dropped(self, pg):
        # hostile: k_v3 has no closure TABLE, but an INDEX named closure (it has pg_attribute rows)
        pg.execute("CREATE SCHEMA k_v3")
        for t, cols in (("nodes", "node_id text, label text"), ("edges", "src_node_id text, dst_node_id text"),
                        ("node_id_map", "node_id text, natural_key text")):
            pg.execute(f"CREATE TABLE k_v3.{t} ({cols})")
        pg.execute("CREATE INDEX closure ON k_v3.nodes (node_id)")
        try:
            assert versions.relation_columns(pg, "k_v3", "closure") == []
            versions.switch_views(pg, "srv", "k_v1", reader="reader")  # closure_view is live
            assert "closure_view" not in versions.switch_views(pg, "srv", "k_v3", reader="reader")
            assert pg.execute("SELECT to_regclass('srv.closure_view')").fetchone()[0] is None
        finally:
            versions.switch_views(pg, "srv", "k_v1", reader="reader")
            pg.execute("DROP SCHEMA k_v3 CASCADE")

    def test_reapply_indexes_no_view_that_carries_an_enrichment_name(self, pg):
        from lgk.steps.reapply import ensure_indexes

        # hostile: node_group in k_v4 is a VIEW; CREATE INDEX on it would fail the reapply
        pg.execute("CREATE SCHEMA k_v4")
        pg.execute("CREATE TABLE k_v4.edges (src_node_id text, dst_node_id text)")
        pg.execute("CREATE VIEW k_v4.node_group AS SELECT 'a'::text AS node_id, 'g'::text AS group_id")
        try:
            ensure_indexes(pg, "k_v4")
            idx = {r[0] for r in pg.execute("SELECT indexname FROM pg_indexes WHERE schemaname = 'k_v4'")}
            assert idx == {"lgk_edges_src_node_id_idx", "lgk_edges_dst_node_id_idx"}
        finally:
            pg.execute("DROP SCHEMA k_v4 CASCADE")
