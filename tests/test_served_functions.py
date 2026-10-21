"""Offline: the function read surface — SQL text, grant block, probe logic, switch transaction.

The live proof (roles really denied, refresh survival) is tests/test_function_roles_live.py.
Here every guard gets its own red test with scripted catalog rows.
"""

import re
from typing import Any, List

import pytest

from lgk import served_functions as sf
from lgk.privacy import probe
from lgk.privacy.probe import ReaderNotIsolatedError

# Hand-written contract (not imported from the code under test).
PUBLIC = {"directory_provider", "directory_by_specialty", "directory_by_location",
          "directory_accepting_new_patients", "directory_affiliations"}
COORDINATOR = {"coordinator_web", "coordinator_tree"}
# web_set: served only when the mapping declares walk keys, so not in the default surface
WEB_SET = {"coordinator_web_set"}
PUBLIC_KEYS = {"specialty", "location", "accepting_new_patients"}


@pytest.fixture
def policy(monkeypatch):
    """The privacy gate's (floor, folded), scripted; default an unfolded graph at floor 11."""
    state = {"value": (11, False)}
    monkeypatch.setattr("lgk.privacy.gate.served_walk_policy", lambda rf, fl, fo: state["value"])
    return state


def _fns(config=None, version="kit_v3"):
    return {f.name: f for f in sf.served_functions(config or {"run_folder": "rf"}, version)}


# ------------------------------------------------------------------ roles and grants
def test_each_function_is_granted_to_exactly_its_role_after_revoking_public(policy):
    fns = _fns()
    assert {n for n, f in fns.items() if f.role == "public_reader"} == PUBLIC
    assert {n for n, f in fns.items() if f.role == "coordinator_reader"} == COORDINATOR
    stmts = sf.switch_statements({"run_folder": "rf"}, "served", "kit_v3", present=())
    for name, fn in fns.items():
        sig = f"served.{name}({fn.arg_types})"
        revoke = stmts.index(f"REVOKE ALL ON FUNCTION {sig} FROM PUBLIC")
        grants = [i for i, s in enumerate(stmts) if s.startswith(f"GRANT EXECUTE ON FUNCTION {sig} ")]
        assert grants == [revoke + 1], name  # one GRANT, right after the REVOKE
        assert stmts[revoke + 1].endswith(f" TO {fn.role}")
    creates = [i for i, s in enumerate(stmts) if s.startswith("CREATE OR REPLACE FUNCTION")]
    first_grant = min(i for i, s in enumerate(stmts) if s.startswith(("REVOKE", "GRANT EXECUTE")))
    assert len(creates) == len(fns) and max(creates) < first_grant
    # nothing grants a table: the roles reach data only through the functions
    assert not [s for s in stmts if re.search(r"GRANT (SELECT|ALL)", s)]


def test_staged_copies_revoke_public_and_grant_nobody(policy):
    stmts = sf.stage_statements({"run_folder": "rf"}, "kit_v3")
    assert not [s for s in stmts if s.startswith("GRANT")]
    for name, fn in _fns().items():
        assert f"REVOKE ALL ON FUNCTION kit_v3.{name}({fn.arg_types}) FROM PUBLIC" in stmts
        assert any(s.startswith(f"CREATE OR REPLACE FUNCTION kit_v3.{name}(") for s in stmts)


def test_signature_matches_the_catalog_identity_format(policy):
    # oidvectortypes() prints "text, integer"; the probe keys on that exact text
    assert _fns()["directory_provider"].signature("s") == "s.directory_provider(text, integer)"
    # walk direction: p_direction text DEFAULT 'outbound' is the third identity argument (no DEFAULT in the type)
    # edge types: coordinator_web's p_edge_type text DEFAULT NULL is the fourth
    assert _fns()["coordinator_web"].signature("s") == "s.coordinator_web(text, integer, text, text)"
    assert _fns()["coordinator_tree"].signature("s") == "s.coordinator_tree(text, integer, text)"


# ------------------------------------------------------------------ bodies
def test_every_function_is_security_definer_with_a_pinned_search_path_and_a_string_body(policy):
    for fn in _fns().values():
        sql = sf.create_sql(fn, "served")
        assert "SECURITY DEFINER" in sql and "SET search_path = pg_catalog, pg_temp" in sql
        assert "BEGIN ATOMIC" not in sql and "EXECUTE " not in fn.body  # no dependency, no dynamic SQL


def test_public_bodies_read_only_public_keys_of_provider_nodes(policy):
    for name in PUBLIC:
        body = _fns()[name].body
        keys = re.findall(r"properties ->> '(\w+)'", body)
        assert set(keys) <= PUBLIC_KEYS, (name, keys)
        assert body.count("properties") == len(keys), f"{name} reads the map itself"
        assert keys or name == "directory_affiliations"  # the lookups really read the public fields
        assert set(re.findall(r"\b(kit_v\d+\.\w+)", body)) <= {"kit_v3.nodes", "kit_v3.edges"}
        assert "node_type = 'provider'" in body
        assert re.search(r"LIMIT LEAST\(GREATEST\(\$2, 0\), 100\)", body), name  # capped limit
        assert "weight" not in body  # never an edge weight (care-subject count)


def test_public_output_columns_are_only_public(policy):
    allowed = {"provider_key", "provider_name", "specialty", "location", "accepting_new_patients",
               "affiliated_key", "affiliated_name"}
    for name in PUBLIC:
        cols = re.findall(r"(\w+) text", _fns()[name].returns)
        assert cols and set(cols) <= allowed, (name, cols)


def test_coordinator_bodies_read_the_version_edges_with_the_gate_count_terms(policy):
    web = _fns()["coordinator_web"].body
    assert "kit_v3.edges e" in web and "edges_view" not in web
    assert "%(start_node_id)s" not in web and "CAST($1 AS text)" in web
    assert "COUNT(*) AS support" in web and "HAVING" not in web  # unfolded: structural count
    for d in range(1, 5):
        assert f"p_depth = {d} THEN" in web
    assert "p_depth = 5" not in web and "RAISE EXCEPTION" in web
    tree = _fns()["coordinator_tree"].body
    assert "kit_v3.edges e" in tree and "p_depth = 10 THEN" in tree and "p_depth = 11" not in tree


def test_a_folded_graph_floors_the_web_and_serves_no_tree(policy):
    # hostile: a raised floor of 25 on folded (care-subject) edges
    policy["value"] = (25, True)
    fns = _fns()
    assert "HAVING MIN(e.weight) >= 25" in fns["coordinator_web"].body
    assert "coordinator_tree" not in fns


def test_the_gate_supplies_the_floor(monkeypatch, tmp_path):
    # no gate, no surface: served_functions goes through the real served_walk_policy
    from lgk.privacy.gate import PrivacyGateError

    with pytest.raises(PrivacyGateError):
        sf.served_functions({"run_folder": str(tmp_path)}, "kit_v1")


def test_smoke_calls_cover_every_depth_of_every_staged_function(policy):
    calls = sf.smoke_calls({"run_folder": "rf"}, "kit_v3")
    for r in ("outbound", "inbound", "both"):  # every (direction, depth) branch is called
        for d in range(1, 5):
            assert f"SELECT count(*) FROM kit_v3.coordinator_web('', {d}, '{r}')" in calls
        for d in range(1, 11):
            assert f"SELECT count(*) FROM kit_v3.coordinator_tree('', {d}, '{r}')" in calls
    assert len([c for c in calls if ".coordinator_web(" in c]) == 12
    assert len([c for c in calls if ".coordinator_tree(" in c]) == 30
    for name in PUBLIC:
        assert f"SELECT count(*) FROM kit_v3.{name}('', 1)" in calls


# ------------------------------------------------------------------ probe
GOOD_CONFIG = "search_path=pg_catalog, pg_temp"
SIG_PUB = "served.directory_provider(text, integer)"
SIG_COORD = "served.coordinator_web(text, integer)"
EXPECTED = {SIG_PUB: "public_reader", SIG_COORD: "coordinator_reader"}


class FakeCatalog:
    """Scripted catalog rows for the probe's queries."""

    def __init__(self, surface: List[tuple], tables: List[tuple] = (), role=(False, False),
                 members: List[tuple] = (), missing: tuple = ()):
        self.surface, self.tables, self.role, self.members = surface, list(tables), role, list(members)
        self.missing = missing
        self.patterns: List[str] = []
        self.surface_roles: List[str] = []

    def execute(self, sql, params=None):
        rows: Any = []
        if sql == probe._SURFACE_SQL:
            self.surface_roles = params["roles"]
            rows = self.surface
        elif sql == probe._TABLE_PRIV_SQL:
            self.patterns.append(params["pattern"])
            rows = [r for r in self.tables if r[0] == params["role"]]
            rows = [r[1:] for r in rows]
        elif sql == probe._ROLE_SQL:
            rows = [] if params[0] in self.missing else [self.role]
        elif sql == probe._MEMBER_SQL:
            rows = [m[1:] for m in self.members if m[0] == params[0]]
        return _Rows(rows)


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


def _surface(**over):
    rows = {
        SIG_PUB: ["served", SIG_PUB, True, GOOD_CONFIG, False, ["public_reader"]],
        SIG_COORD: ["served", SIG_COORD, True, GOOD_CONFIG, False, ["coordinator_reader"]],
    }
    for sig, changes in over.items():
        for idx, value in changes.items():
            rows[{"pub": SIG_PUB, "coord": SIG_COORD}[sig]][idx] = value
    return [tuple(r) for r in rows.values()]


def _violations(cat):
    return probe.function_role_violations(cat, "kit", "served", EXPECTED,
                                          ("public_reader", "coordinator_reader"), "pg_catalog, pg_temp")


def test_probe_passes_the_intended_surface():
    assert _violations(FakeCatalog(_surface())) == []


def test_probe_flags_public_reader_executing_a_coordinator_function():
    found = _violations(FakeCatalog(_surface(coord={5: ["coordinator_reader", "public_reader"]})))
    assert found == [f"public_reader has EXECUTE on {SIG_COORD}"]


def test_probe_flags_public_execute():
    found = _violations(FakeCatalog(_surface(pub={4: True})))
    assert found == [f"PUBLIC has EXECUTE on {SIG_PUB}"]


def test_probe_flags_the_reader_role_executing_a_served_function():
    found = _violations(FakeCatalog(_surface(pub={5: ["public_reader", "reader"]})))
    assert found == [f"reader has EXECUTE on {SIG_PUB}"]


def test_probe_flags_an_unknown_served_function_any_role_can_execute():
    extra = ("served", "served.dump_all()", True, GOOD_CONFIG, False, ["coordinator_reader"])
    found = _violations(FakeCatalog(_surface() + [extra]))
    assert found == ["coordinator_reader has EXECUTE on served.dump_all()"]


def test_probe_flags_any_role_executing_a_staged_version_function():
    # hostile: the staged copy has the SAME name as a served one, in a version schema
    staged = ("kit_v2", "kit_v2.directory_provider(text, integer)", True, GOOD_CONFIG, False, ["public_reader"])
    found = _violations(FakeCatalog(_surface() + [staged]))
    assert found == ["public_reader has EXECUTE on kit_v2.directory_provider(text, integer)"]


@pytest.mark.parametrize("change", [{2: False}, {3: ""}, {3: "search_path=public"},
                                    {3: "search_path=pg_catalog, pg_temp, public"}])
def test_probe_flags_a_function_without_security_definer_and_a_pinned_search_path(change):
    found = _violations(FakeCatalog(_surface(pub=change)))
    assert found == [f"{SIG_PUB} is not SECURITY DEFINER with search_path=pg_catalog, pg_temp pinned"]


def test_probe_flags_a_table_right_on_the_served_or_a_version_schema():
    tables = [("coordinator_reader", "kit_v1", "nodes", "SELECT", True, False, False),
              ("public_reader", "served", "nodes_view", "SELECT", True, False, False)]
    cat = FakeCatalog(_surface(), tables)
    found = _violations(cat)
    assert "coordinator_reader has SELECT on raw kit_v1.nodes" in found
    assert "public_reader has SELECT on raw served.nodes_view" in found
    # the served schema is in the pattern, not only the version schemas
    assert all(p == "^served$|^kit_v[0-9]+$" for p in cat.patterns), cat.patterns


def test_probe_flags_a_function_role_in_another_role_or_a_superuser():
    found = _violations(FakeCatalog(_surface(), role=(True, False),
                                    members=[("public_reader", "databricks_superuser")]))
    assert "public_reader is a superuser" in found
    assert "public_reader is a member of role databricks_superuser" in found


def test_assert_raises_with_every_violation():
    with pytest.raises(ReaderNotIsolatedError, match="PUBLIC has EXECUTE"):
        probe.assert_function_roles_isolated(FakeCatalog(_surface(pub={4: True})), "kit", "served",
                                             EXPECTED, ("public_reader",), "pg_catalog, pg_temp")


# ------------------------------------------------------------------ switch transaction
class FakeConn:
    """Records statements and whether they ran inside the one transaction."""

    def __init__(self):
        import psycopg

        self.log: List[str] = []
        self.in_txn = False
        self.committed = False
        self.info = type("I", (), {"transaction_status": psycopg.pq.TransactionStatus.IDLE})()

    def execute(self, sql, params=None):
        self.log.append(("txn" if self.in_txn else "auto") + ":" + sql.split("\n")[0])
        return _Rows([])  # the switch reads view/table columns (none here)

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


def test_switch_runs_functions_and_the_probe_inside_the_view_transaction():
    from lgk.versions import switch_views

    conn, seen = FakeConn(), []
    switch_views(conn, "served", "kit_v2", ["CREATE OR REPLACE FUNCTION served.f()", "GRANT x"],
                 verify=lambda c: seen.append(c.in_txn))
    assert conn.log[-2:] == ["txn:CREATE OR REPLACE FUNCTION served.f()", "txn:GRANT x"]
    assert seen == [True] and conn.committed


def test_a_failing_probe_rolls_the_switch_back():
    from lgk.versions import switch_views

    conn = FakeConn()

    def leak(c):
        raise ReaderNotIsolatedError("public_reader has EXECUTE on served.coordinator_web")

    with pytest.raises(ReaderNotIsolatedError):
        switch_views(conn, "served", "kit_v2", ["GRANT x"], verify=leak)
    assert not conn.committed


def test_serve_passes_the_function_statements_and_the_probe(monkeypatch, policy, tmp_path):
    from lgk.steps import provision as prov

    got = {}
    monkeypatch.setattr("lgk.manifest.claim_pg_schema", lambda *a, **k: False)
    monkeypatch.setattr(prov, "_ensure_function_roles", lambda config, cursor: None)
    monkeypatch.setattr("lgk.versions.switch_views",
                        lambda conn, s, t, fsql, verify, **k: got.update(fsql=fsql, verify=verify))
    leaks = []
    monkeypatch.setattr(prov, "assert_function_roles_isolated", lambda *a: leaks.append(a))
    # F0: serve reads the served schema's kit functions first (none here)
    conn = type("C", (), {"cursor": lambda self: None, "execute": lambda self, sql, p=None: _Rows([])})()
    prov.serve(conn, {"graph_name": "g", "kit_schema": "kit", "run_folder": str(tmp_path),
                      "served_schema": "served", "lakebase_project": "p", "branch_id": "t-x"}, "kit_v2")
    assert any(s.startswith("CREATE OR REPLACE FUNCTION served.coordinator_web") for s in got["fsql"])
    got["verify"]("conn")
    assert leaks and leaks[0][1:3] == ("kit", "served") and leaks[0][3][SIG_PUB] == "public_reader"
    assert SIG_PUB in leaks[0][6] and len(leaks[0][6]) == 7  # every served function must exist


# ------------------------------------------------------------------ teardown / retire
def test_teardown_drops_recorded_functions_in_the_views_phase(tmp_path):
    from lgk.manifest import Manifest, ObjectType
    from lgk.steps.teardown import plan_from_manifest
    from datetime import UTC, datetime
    from lgk.manifest import ManifestEntry

    m = Manifest(graph_name="g")
    for sig in (SIG_PUB, SIG_PUB, "kit_v1.directory_provider(text, integer)"):  # hostile: a duplicate record
        m.add_entry(ManifestEntry(ObjectType.FUNCTION, sig, sig, datetime.now(tz=UTC), "pg"))
    plan = plan_from_manifest(m)
    assert plan.functions == [SIG_PUB, "kit_v1.directory_provider(text, integer)"]
    assert sf.drop_sql(SIG_PUB) == f"DROP FUNCTION IF EXISTS {SIG_PUB} RESTRICT"


def test_live_ops_drop_views_drops_functions_in_the_same_transaction(monkeypatch):
    from lgk.steps.teardown import LiveOps

    ops = LiveOps.__new__(LiveOps)
    conn = FakeConn()
    monkeypatch.setattr(LiveOps, "_pg", lambda self: conn)
    ops.drop_views(["served.nodes_view"], [SIG_PUB])
    assert conn.log == ["txn:DROP VIEW IF EXISTS served.nodes_view RESTRICT",
                        f"txn:DROP FUNCTION IF EXISTS {SIG_PUB} RESTRICT"]


def test_retire_drops_the_staged_functions_before_the_version_schema(monkeypatch):
    from lgk.steps import retire

    conn = FakeConn()
    monkeypatch.setattr(retire.versions, "refuse_if_live", lambda *a: None)
    monkeypatch.setattr(retire.versions, "refuse_unless_ours_pg", lambda *a: None)
    monkeypatch.setattr(retire.served_functions, "staged_signatures",
                        lambda c, schema: [f"{schema}.coordinator_web(text, integer)"])
    retire._drop_pg_schema(conn, "served", "kit_v1")
    drops = [s for s in conn.log if "DROP" in s]
    assert drops[0] == "txn:DROP FUNCTION IF EXISTS kit_v1.coordinator_web(text, integer) RESTRICT"
    assert drops[-1] == "txn:DROP SCHEMA IF EXISTS kit_v1 RESTRICT"


def test_staged_signatures_ask_only_for_the_kit_function_names():
    seen = {}

    class C:
        def execute(self, sql, params):
            seen["params"] = params
            return _Rows([("kit_v1.coordinator_web(text, integer)",)])

    assert sf.staged_signatures(C(), "kit_v1") == ["kit_v1.coordinator_web(text, integer)"]
    assert seen["params"][0] == "kit_v1" and set(seen["params"][1]) == PUBLIC | COORDINATOR | WEB_SET


def test_a_function_role_that_does_not_exist_reaches_nothing():
    # before the first provision (or on a fixture without the roles) the probe still runs
    cat = FakeCatalog(_surface(pub={5: []}, coord={5: []}), missing=("public_reader", "coordinator_reader"))
    assert _violations(cat) == []
    assert cat.surface_roles == ["reader"]  # has_function_privilege is never asked about a missing role
    assert cat.patterns == []


def test_expected_grants_need_no_gate_and_agree_with_the_builders(policy):
    expected = sf.expected_grants("served")
    assert {s.split(".")[1].split("(")[0] for s, r in expected.items() if r == "public_reader"} == PUBLIC
    assert {s.split(".")[1].split("(")[0] for s, r in expected.items() if r == "coordinator_reader"} == COORDINATOR
    for fn in sf.served_functions({"run_folder": "rf"}, "served"):
        assert expected[fn.signature("served")] == fn.role


def test_probe_flags_a_required_served_function_that_is_missing():
    found = probe.function_role_violations(
        FakeCatalog(_surface()), "kit", "served", EXPECTED, ("public_reader", "coordinator_reader"),
        "pg_catalog, pg_temp", required=[SIG_PUB, "served.coordinator_tree(text, integer)"])
    assert found == ["served function served.coordinator_tree(text, integer) is missing"]


def test_a_folded_switch_drops_the_tree_left_by_an_unfolded_version(policy):
    policy["value"] = (11, True)
    stmts = sf.switch_statements({"run_folder": "rf"}, "served", "kit_v2", present=())
    drop = "DROP FUNCTION IF EXISTS served.coordinator_tree(text, integer, text) RESTRICT"
    assert drop in stmts and stmts.index(drop) < min(
        i for i, s in enumerate(stmts) if s.startswith("CREATE OR REPLACE"))
    policy["value"] = (11, False)
    assert not [s for s in sf.switch_statements({"run_folder": "rf"}, "served", "kit_v2", present=()) if "DROP" in s]


def test_affiliations_follow_only_affiliation_edges_between_providers(policy):
    body = _fns()["directory_affiliations"].body
    assert "e.relationship_type = 'affiliation'" in body
    assert "s.node_type = 'provider'" in body and "d.node_type = 'provider'" in body


# ------------------------------------------------------------------ provision wiring
def test_the_provision_proof_runs_the_function_role_probe(monkeypatch):
    from lgk.steps import provision as prov

    seen = []
    conn = type("C", (), {"close": lambda self: None})()
    monkeypatch.setattr(prov, "connect_branch", lambda c: conn)
    monkeypatch.setattr(prov, "assert_reader_isolated", lambda c, kit: seen.append(("reader", kit)))
    monkeypatch.setattr(prov, "assert_function_roles_isolated", lambda *a: seen.append(a[1:3] + (a[4],)))
    prov._prove_reader_isolated({"graph_name": "g", "kit_schema": "kit", "served_schema": "served"})
    assert seen == [("reader", "kit"), ("kit", "served", ("public_reader", "coordinator_reader"))]


def test_provision_creates_the_shared_roles_but_never_marks_or_records_them(monkeypatch, tmp_path):
    from lgk.steps import provision as prov
    from tests.test_reader_role import FakeCursor

    cur = type("Cur", (FakeCursor,), {"close": lambda self: None})(exists=False, may_create=True)
    conn = type("C", (), {"cursor": lambda self: cur, "commit": lambda self: None,
                          "close": lambda self: None})()
    monkeypatch.setattr(prov, "connect_branch", lambda c: conn)
    config = {"graph_name": "g", "kit_schema": "kit", "run_folder": str(tmp_path), "lakebase_project": "p",
              "branch_id": "t-x"}
    prov._lock_version_from_reader(config)
    for role in ("reader", "public_reader", "coordinator_reader"):
        assert f"CREATE ROLE {role} WITH LOGIN;" in cur.sql
    # A6 (a): a role is cluster-wide, shared by every graph on the branch; no graph marks or owns it
    assert not [s for s in cur.sql if s.upper().startswith("COMMENT ON ROLE")]
    from lgk.manifest import MANIFEST_FILE

    # nothing else is recorded by this step, so no manifest at all is the strongest form
    assert not (tmp_path / MANIFEST_FILE).exists()


# ------------------------------------------------------------------ mutation-prover additions (merge 698dd02)
class _StageConn(FakeConn):
    """FakeConn whose execute returns a cursor-like result, with commit/close logged."""

    def execute(self, sql, params=None):
        super().execute(sql, params)
        return type("R", (), {"fetchall": lambda self: []})()

    def commit(self):
        self.log.append("commit")

    def close(self):
        self.log.append("close")


_CFG = {"graph_name": "g", "kit_schema": "kit", "served_schema": "served", "version": 3,
        "lakebase_project": "p", "branch_id": "t-x"}
_LOC = "postgres:projects/p/branches/t-x"


def test_staging_creates_and_smokes_every_function_in_one_transaction_and_records_it(monkeypatch, policy, tmp_path):
    from lgk.manifest import ObjectType, load_run_manifest
    from lgk.steps import provision as prov

    conn, config = _StageConn(), {**_CFG, "run_folder": str(tmp_path)}
    monkeypatch.setattr(prov, "connect_branch", lambda c: conn)
    prov._stage_version_functions(config)
    stage = [s.split("\n")[0] for s in sf.stage_statements(config, "kit_v3")]
    smokes = [s.split("\n")[0] for s in sf.smoke_calls(config, "kit_v3")]
    assert stage and smokes
    assert conn.log == (["txn:" + s for s in stage] + ["txn:" + s for s in smokes]
                        + ["commit", "close"])
    recorded = {e.object_id: e.location for e in load_run_manifest(config).entries
                if e.object_type == ObjectType.FUNCTION}
    assert recorded == {fn.signature("kit_v3"): _LOC for fn in sf.served_functions(config, "kit_v3")}


def test_serve_ensures_function_roles_and_records_schema_views_and_functions(monkeypatch, policy, tmp_path):
    from lgk import versions
    from lgk.manifest import ObjectType, load_run_manifest
    from lgk.steps import provision as prov

    cur, ensured = object(), []
    monkeypatch.setattr("lgk.manifest.claim_pg_schema", lambda *a, **k: True)
    monkeypatch.setattr(prov, "_ensure_function_roles", lambda config, cursor: ensured.append(cursor))
    monkeypatch.setattr("lgk.versions.switch_views", lambda *a, **k: None)
    conn = type("C", (), {"cursor": lambda self: cur, "execute": lambda self, sql, p=None: _Rows([])})()
    config = {**_CFG, "run_folder": str(tmp_path)}
    prov.serve(conn, config, "kit_v2")
    assert ensured == [cur]
    entries = {(e.object_type, e.object_id): e.location for e in load_run_manifest(config).entries}
    assert entries[(ObjectType.PG_SCHEMA, "served")] == _LOC
    for view, _ in versions.SERVED_VIEWS:
        assert entries[(ObjectType.VIEW, f"served.{view}")] == _LOC
    functions = {oid: loc for (t, oid), loc in entries.items() if t == ObjectType.FUNCTION}
    assert functions == {sig: _LOC for sig in sf.expected_grants("served")} and len(functions) == 7


def test_provision_connects_and_records_on_the_configured_branch(monkeypatch, tmp_path):
    from lgk.manifest import ObjectType, load_run_manifest
    from lgk.steps import provision as prov

    opened = []
    conn = type("C", (), {"close": lambda self: None})()
    monkeypatch.setattr("lgk.walks.nb_runtime.connect", lambda p, b: opened.append((p, b)) or conn)
    monkeypatch.setattr("lgk.versions.assert_version_schema_absent", lambda c, s: None)
    monkeypatch.setattr("lgk.manifest.claim_pg_schema", lambda *a, **k: True)
    config = {**_CFG, "run_folder": str(tmp_path)}
    prov._refuse_existing_version(config)
    prov._mark_pg_version_schema(config)
    assert opened == [("p", "t-x"), ("p", "t-x")]
    [loc] = [e.location for e in load_run_manifest(config).entries
             if (e.object_type, e.object_id) == (ObjectType.PG_SCHEMA, "kit_v3")]
    assert loc == _LOC


@pytest.mark.parametrize("branch_id", ["", None])  # hostile: present but empty (owner "branch a")
def test_provision_refuses_an_empty_branch_id_before_connecting(monkeypatch, tmp_path, branch_id):
    from lgk.branch import MissingBranchError
    from lgk.steps import provision as prov

    opened = []
    monkeypatch.setattr("lgk.walks.nb_runtime.connect", lambda p, b: opened.append((p, b)))
    config = {**_CFG, "branch_id": branch_id, "run_folder": str(tmp_path)}
    with pytest.raises(MissingBranchError, match="branch_id"):
        prov._refuse_existing_version(config)
    assert opened == []  # never connected, to production or anywhere


# ------------------------------------------------------------------ provision staging is atomic
class _TxConn:
    """A psycopg-shaped conn: statements are pending until their transaction (or commit) ends.

    `transaction()` commits its pending statements on a clean exit and drops them on an error,
    like psycopg; `commit()` commits whatever an implicit transaction holds.
    """

    def __init__(self, fail_on: str = ""):
        self.fail_on, self.pending, self.committed, self.closed = fail_on, [], [], False

    def execute(self, sql):
        if self.fail_on and sql.startswith(self.fail_on):
            raise RuntimeError(f"smoke call failed: {sql}")
        self.pending.append(sql)
        return type("R", (), {"fetchall": lambda self: [(0,)]})()

    def transaction(self):
        conn = self

        class _Tx:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, *_):
                if exc_type is None:
                    conn.committed += conn.pending
                conn.pending = []
                return False

        return _Tx()

    def commit(self):
        self.committed += self.pending
        self.pending = []

    def close(self):
        self.closed = True


def _stage(monkeypatch, tmp_path, conn):
    from lgk.steps import provision as prov

    monkeypatch.setattr(prov, "connect_branch", lambda c: conn)
    config = {"graph_name": "g", "kit_schema": "kit", "version": 3, "run_folder": str(tmp_path),
              "lakebase_project": "p", "branch_id": "t-x"}
    return prov, config


def test_a_failing_smoke_call_rolls_back_every_staged_function_and_records_none(policy, monkeypatch, tmp_path):
    from lgk.manifest import load_run_manifest

    # hostile: the LAST smoke call fails, after every CREATE and every earlier call succeeded
    version = "kit_v3"
    last = sf.smoke_calls({"run_folder": str(tmp_path)}, version)[-1]
    conn = _TxConn(fail_on=last)
    prov, config = _stage(monkeypatch, tmp_path, conn)
    with pytest.raises(RuntimeError, match="smoke call failed"):
        prov._stage_version_functions(config)
    assert not [s for s in conn.committed if s.startswith("CREATE")]  # nothing stranded on the branch
    assert conn.committed == []
    with pytest.raises(FileNotFoundError, match="nothing was recorded"):  # nothing recorded for a failed version
        load_run_manifest(config)
    assert conn.closed


def test_staging_records_every_function_only_after_its_creates_commit(policy, monkeypatch, tmp_path):
    from lgk.manifest import ObjectType, load_run_manifest

    conn = _TxConn()
    prov, config = _stage(monkeypatch, tmp_path, conn)
    prov._stage_version_functions(config)
    created = [s for s in conn.committed if s.startswith("CREATE")]
    expected = {fn.signature("kit_v3") for fn in sf.served_functions(config, "kit_v3")}
    assert len(created) == len(expected) > 0
    recorded = {e.object_id for e in load_run_manifest(config).entries if e.object_type == ObjectType.FUNCTION}
    assert recorded == expected
    smoke = sf.smoke_calls(config, "kit_v3")
    # every smoke call ran in the same committed unit as the CREATEs, after them
    assert conn.committed.index(smoke[0]) > max(conn.committed.index(s) for s in created)


# ------------------------------------------------------------------ per-function signatures
# Hand-written: coordinator_web's identity arguments with the walk direction.
DIRECTION_ARGS = "p_start_node_id text, p_depth integer, p_direction text"
STALE_WEB = "served.coordinator_web(text, integer)"
NEW_WEB = "served.coordinator_web(text, integer, text)"


def _with_direction(fns, smoke_args=("'', 1, 'out'", "'', 2, 'in'")):
    """The served list with coordinator_web replaced by a stand-in taking (text, integer, text)."""
    import dataclasses

    return [dataclasses.replace(f, args=DIRECTION_ARGS, smoke_args=smoke_args)
            if f.name == "coordinator_web" else f for f in fns]


def _serve_direction(monkeypatch):
    real = sf.served_functions
    monkeypatch.setattr(sf, "served_functions", lambda config, version: _with_direction(real(config, version)))


def test_switch_drops_a_stale_overload(monkeypatch, policy):
    _serve_direction(monkeypatch)
    present = [
        STALE_WEB,  # hostile: an overload whose name is served but whose identity arguments differ
        NEW_WEB,  # the signature being served: never dropped
        "served.directory_provider(text, integer)",  # served with this exact signature: kept
        "served.dump_all()",  # hostile: not a kit function name, never the kit's to drop
        "kit_v1.coordinator_web(text, integer)",  # hostile: a kit name in ANOTHER schema
        "served_old.coordinator_web(text, integer)",  # hostile: a schema whose name starts with served's
    ]
    stmts = sf.switch_statements({"run_folder": "rf"}, "served", "kit_v2", present=present)
    drop = f"DROP FUNCTION IF EXISTS {STALE_WEB} RESTRICT"
    first_create = min(i for i, s in enumerate(stmts) if s.startswith("CREATE OR REPLACE"))
    assert drop in stmts and stmts.index(drop) < first_create
    assert f"CREATE OR REPLACE FUNCTION served.coordinator_web({DIRECTION_ARGS})" in [
        s.split("\n")[0] for s in stmts]
    assert [s for s in stmts if s.startswith("DROP")] == [drop]  # nothing else dropped
    assert f"GRANT EXECUTE ON FUNCTION {NEW_WEB} TO coordinator_reader" in stmts


def test_switch_drops_each_stale_overload_once_with_the_unserved_names(monkeypatch, policy):
    # hostile: a folded target serves no tree, AND the catalog lists the tree's overload
    policy["value"] = (11, True)
    tree = "served.coordinator_tree(text, integer, text)"
    stmts = sf.switch_statements({"run_folder": "rf"}, "served", "kit_v2", present=[tree, tree])
    assert [s for s in stmts if s.startswith("DROP")] == [f"DROP FUNCTION IF EXISTS {tree} RESTRICT"]


def test_serve_reads_the_served_schemas_overloads_and_drops_the_stale_one(monkeypatch, policy, tmp_path):
    from lgk.steps import provision as prov

    _serve_direction(monkeypatch)
    got, asked = {}, []

    class Conn:
        def cursor(self):
            return None

        def execute(self, sql, params=None):
            asked.append(params)
            # hostile: the target version schema holds kit functions too; only served's count
            rows = {"served": [(STALE_WEB,), (NEW_WEB,)],
                    "kit_v2": [("kit_v2.coordinator_web(text, integer)",)]}.get(params[0], [])
            return _Rows(rows if sql == sf._STAGED_SQL else [])

    monkeypatch.setattr("lgk.manifest.claim_pg_schema", lambda *a, **k: False)
    monkeypatch.setattr(prov, "_ensure_function_roles", lambda config, cursor: None)
    monkeypatch.setattr("lgk.versions.switch_views",
                        lambda conn, s, t, fsql, verify, **k: got.update(fsql=list(fsql)))
    prov.serve(Conn(), {"graph_name": "g", "kit_schema": "kit", "run_folder": str(tmp_path),
                        "served_schema": "served", "lakebase_project": "p", "branch_id": "t-x"}, "kit_v2")
    assert [s for s in got["fsql"] if s.startswith("DROP")] == [f"DROP FUNCTION IF EXISTS {STALE_WEB} RESTRICT"]
    assert asked and asked[0][0] == "served"


def test_expected_grants_and_staged_signatures_follow_each_functions_arg_types(monkeypatch, policy):
    real = sf.servable_functions
    monkeypatch.setattr(sf, "servable_functions", lambda version: _with_direction(real(version)))
    expected = sf.expected_grants("served")
    assert expected[NEW_WEB] == "coordinator_reader"
    assert STALE_WEB not in expected
    assert expected["served.directory_provider(text, integer)"] == "public_reader"  # others unchanged
    assert len(expected) == 7

    class C:
        def execute(self, sql, params):
            return _Rows([("kit_v1.coordinator_web(text, integer, text)",)])

    # staged copies come from the catalog's identity arguments, never a module constant
    [staged] = sf.staged_signatures(C(), "kit_v1")
    assert sf.drop_sql(staged) == "DROP FUNCTION IF EXISTS kit_v1.coordinator_web(text, integer, text) RESTRICT"


def test_smoke_calls_use_each_functions_own_arguments(monkeypatch, policy):
    _serve_direction(monkeypatch)
    calls = sf.smoke_calls({"run_folder": "rf"}, "kit_v3")
    web = [c for c in calls if ".coordinator_web(" in c]
    assert web == ["SELECT count(*) FROM kit_v3.coordinator_web('', 1, 'out')",
                   "SELECT count(*) FROM kit_v3.coordinator_web('', 2, 'in')"]
    # the other functions keep their own calls
    assert "SELECT count(*) FROM kit_v3.coordinator_tree('', 10, 'both')" in calls
    assert "SELECT count(*) FROM kit_v3.directory_provider('', 1)" in calls


def test_a_function_without_smoke_calls_is_refused(policy):
    import dataclasses

    web = _fns()["coordinator_web"]
    with pytest.raises(ValueError, match="coordinator_web has no smoke call"):
        dataclasses.replace(web, smoke_args=())


# ------------------------------------------------------------------ walk direction: p_direction (function form)
DIRS = ("outbound", "inbound", "both")  # hand-written, not lgk.walks.sql.DIRECTIONS


def test_coordinator_functions_take_a_defaulted_direction_and_branch_per_pair(policy):
    fns = _fns()
    for name, depths in (("coordinator_web", range(1, 5)), ("coordinator_tree", range(1, 11))):
        fn = fns[name]
        assert fn.args == "p_start_node_id text, p_depth integer, p_direction text DEFAULT 'outbound'" + (
            ", p_edge_type text DEFAULT NULL" if name == "coordinator_web" else "")  # edge types
        branches = re.findall(r"p_direction = '(\w+)' AND p_depth = (\d+) THEN", fn.body)
        assert sorted(branches) == sorted((r, str(d)) for r in DIRS for d in depths), name
        assert "EXECUTE" not in fn.body and "format(" not in fn.body  # static SQL only
        assert "RAISE EXCEPTION 'direction must be one of (outbound, inbound, both), got %'" in fn.body
    web = fns["coordinator_web"].body
    # each direction's branch holds that direction's join: inbound joins on dst, both on both ends
    inbound = web.split("p_direction = 'inbound' AND p_depth = 1 THEN")[1].split("ELSIF")[0]
    assert "JOIN hop0 f ON e.dst_node_id = f.node_id" in inbound and "e.src_node_id = f.node_id" not in inbound
    both = web.split("p_direction = 'both' AND p_depth = 1 THEN")[1].split("ELSIF")[0]
    assert "UNION\n" in both and both.count("kit_v3.edges e") == 2
    outbound = web.split("p_direction = 'outbound' AND p_depth = 1 THEN")[1].split("ELSIF")[0]
    assert "JOIN hop0 f ON e.src_node_id = f.node_id" in outbound and "e.dst_node_id = f.node_id" not in outbound


def test_a_folded_web_serves_outbound_only_and_refuses_the_rest(policy):
    # hostile: folded (care-subject) edges, where inbound/both would double the people count
    policy["value"] = (11, True)
    web = _fns()["coordinator_web"]
    assert re.findall(r"p_direction = '(\w+)' AND p_depth", web.body) == ["outbound"] * 4
    assert ("RAISE EXCEPTION 'folded edges are symmetric (fold_edges_sql emits both orientations); "
            "use direction=''outbound'''") in web.body
    assert web.smoke_args == tuple(f"'', {d}, 'outbound'" for d in range(1, 5))
    assert web.signature("s") == "s.coordinator_web(text, integer, text, text)"  # same signature either way


def test_switch_drops_both_pre_item_1_overloads(policy):
    present = ["served.coordinator_web(text, integer)", "served.coordinator_tree(text, integer)",
               "served.coordinator_web(text, integer, text, text)"]  # the new one (edge types), already there: kept
    stmts = sf.switch_statements({"run_folder": "rf"}, "served", "kit_v2", present=present)
    drops = [s for s in stmts if s.startswith("DROP")]
    assert drops == ["DROP FUNCTION IF EXISTS served.coordinator_web(text, integer) RESTRICT",
                     "DROP FUNCTION IF EXISTS served.coordinator_tree(text, integer) RESTRICT"]
    first_create = min(i for i, s in enumerate(stmts) if s.startswith("CREATE OR REPLACE"))
    assert all(stmts.index(d) < first_create for d in drops)
    for name, args in (("coordinator_web", "text, integer, text, text"), ("coordinator_tree", "text, integer, text")):
        assert f"REVOKE ALL ON FUNCTION served.{name}({args}) FROM PUBLIC" in stmts
        assert f"GRANT EXECUTE ON FUNCTION served.{name}({args}) TO coordinator_reader" in stmts


# ------------------------------------------------------------------ edge types: p_edge_type (function form)
def test_a_mixed_graph_serves_one_static_branch_set_per_edge_type(tmp_path):
    # hostile: a real mixed run folder (two fold types, two unfolded types), no scripted policy
    from tests.test_walks import MIXED_MAPPING, _cleared_run

    run = _cleared_run(tmp_path, MIXED_MAPPING)
    fns = {f.name: f for f in sf.served_functions({"run_folder": run}, "kit_v3")}
    assert "coordinator_tree" not in fns  # a mixed graph serves no tree (it cannot be floored per type)
    web = fns["coordinator_web"]
    assert web.signature("s") == "s.coordinator_web(text, integer, text, text)"
    branches = re.findall(r"p_edge_type = '(\w+)' AND p_direction = '(\w+)' AND p_depth = (\d) THEN", web.body)
    expected = {(t, r, str(d)) for t in ("shares_refs", "treats_at") for r in ("outbound",) for d in range(1, 5)}
    expected |= {(t, r, str(d)) for t in ("affiliation", "same_practice") for r in DIRS for d in range(1, 5)}
    assert set(branches) == expected and len(branches) == len(expected)
    assert "p_edge_type IS NULL AND p_direction" not in web.body  # NULL is refused, never walked
    assert "RAISE EXCEPTION 'edge_types is required on a graph with folded and unfolded edge types" in web.body
    assert ("IF p_edge_type IS NOT NULL AND p_edge_type NOT IN "
            "('affiliation', 'same_practice', 'shares_refs', 'treats_at') THEN") in web.body
    fold_branch = web.body.split("p_edge_type = 'shares_refs' AND p_direction = 'outbound' AND p_depth = 1 THEN")[1]
    fold_branch = fold_branch.split("ELSIF")[0]
    assert "AND e.relationship_type IN ('shares_refs')" in fold_branch and "HAVING MIN(e.weight) >= 11" in fold_branch
    plain = web.body.split("p_edge_type = 'affiliation' AND p_direction = 'both' AND p_depth = 1 THEN")[1]
    plain = plain.split("ELSIF")[0]
    assert "AND e.relationship_type IN ('affiliation')" in plain and "COUNT(*) AS support" in plain
    assert "HAVING" not in plain
    assert "EXECUTE " not in web.body and "format(" not in web.body  # static SQL only
    assert "'', 1, 'outbound', 'shares_refs'" in web.smoke_args and "'', 1, 'outbound'" not in web.smoke_args


def test_an_unfolded_graph_keeps_the_null_walk_and_adds_per_type_branches(tmp_path):
    from tests.test_walks import MIXED_MAPPING, _UNFOLDED_EDGE, _cleared_run

    unfolded = {**MIXED_MAPPING, "edges": {"affiliation": dict(_UNFOLDED_EDGE)}}
    fns = {f.name: f for f in sf.served_functions({"run_folder": _cleared_run(tmp_path, unfolded)}, "kit_v3")}
    assert set(fns) >= {"coordinator_web", "coordinator_tree"}
    web = fns["coordinator_web"].body
    null_branch = web.split("p_edge_type IS NULL AND p_direction = 'outbound' AND p_depth = 2 THEN")[1].split("ELSIF")[0]
    assert "relationship_type IN" not in null_branch  # every type: today's SQL
    assert "p_edge_type = 'affiliation' AND p_direction = 'inbound' AND p_depth = 4 THEN" in web


# ------------------------------------------------------------------ coordinator_web_set
def _walk_key_mapping(walk_keys):
    from lgk.mapping import parse_mapping

    return parse_mapping({
        "source_catalog": "c", "source_schema": "s",
        "nodes": {"provider": {"source_table": "p", "natural_key": "id", "label": "name",
                               "properties": ["specialty"], "walk_keys": walk_keys},
                  "hospital": {"source_table": "h", "natural_key": "id", "label": "name",
                               "properties": ["network_status"], "walk_keys": walk_keys and ["network_status"]}},
        "edges": {"affiliation": {"source_table": "a", "src": "pid", "src_type": "provider",
                                  "dst": "hid", "dst_type": "hospital"}}})


@pytest.fixture
def walk_keyed(monkeypatch, policy):
    """served_functions / expected_grants read this mapping; `state["keys"]` toggles walk keys."""
    state = {"keys": ["specialty"]}
    monkeypatch.setattr("lgk.privacy.gate.read_run_mapping", lambda rf: _walk_key_mapping(state["keys"]))
    return state


def test_web_set_is_served_only_when_the_mapping_declares_walk_keys(walk_keyed):
    fns = _fns()
    ws = fns["coordinator_web_set"]
    assert ws.role == "coordinator_reader" and ws.signature("s") == (
        "s.coordinator_web_set(text, text, text, text, integer, text)")
    # (seed keys 2) x (stop keys 2 + NULL) x 4 depths x (NULL + 1 edge type)
    assert len(ws.smoke_args) == 2 * 3 * 4 * 2
    assert "p_seed_key = 'hospital.network_status' AND p_stop_key = 'provider.specialty'" in ws.body
    assert "kit_v3.nodes" in ws.body and "nodes_view" not in ws.body and "$2" in ws.body and "$4" in ws.body
    assert "EXECUTE " not in ws.body  # static branches only: keys are matched, never spliced from input
    walk_keyed["keys"] = []
    assert "coordinator_web_set" not in _fns()


def test_expected_grants_list_web_set_only_for_a_walk_keyed_run(walk_keyed):
    sig = "served.coordinator_web_set(text, text, text, text, integer, text)"
    assert sf.expected_grants("served", {"run_folder": "rf"})[sig] == "coordinator_reader"
    assert sig not in sf.expected_grants("served")
    walk_keyed["keys"] = []
    assert sig not in sf.expected_grants("served", {"run_folder": "rf"})


def test_a_switch_to_a_version_without_walk_keys_drops_web_set(walk_keyed):
    sig = "served.coordinator_web_set(text, text, text, text, integer, text)"
    walk_keyed["keys"] = []
    stmts = sf.switch_statements({"run_folder": "rf"}, "served", "kit_v2", present=[sig])
    assert f"DROP FUNCTION IF EXISTS {sig} RESTRICT" in stmts
    walk_keyed["keys"] = ["specialty"]
    stmts = sf.switch_statements({"run_folder": "rf"}, "served", "kit_v2", present=[sig])
    assert not [s for s in stmts if s.startswith("DROP")]
    assert f"GRANT EXECUTE ON FUNCTION {sig} TO coordinator_reader" in stmts


# ------------------------------------------- function guard over node-join outputs
@pytest.mark.parametrize("public_name", sf.DIRECTORY_PROPERTIES)
def test_a_join_output_never_takes_a_public_directory_name(tmp_path, public_name):
    """Public functions return DIRECTORY_PROPERTIES keys of the directory node type; a join output
    with one of those names would be served to the public role, so the gate refuses it."""
    import copy

    from lgk.privacy.gate import MappingLeakError
    from tests.test_build_sql import JOIN_MAPPING, pass_privacy, write_run

    m = copy.deepcopy(JOIN_MAPPING)
    assert sf.DIRECTORY_NODE_TYPE in m["nodes"]
    m["nodes"][sf.DIRECTORY_NODE_TYPE]["properties"] = []  # no native collision to hide behind
    # hostile: a floored aggregate named like a public field
    m["nodes"][sf.DIRECTORY_NODE_TYPE]["joins"][0]["aggregates"][1] = f"count(claim_id) as {public_name}"
    write_run(tmp_path, m)
    with pytest.raises(MappingLeakError, match=rf"{public_name}.*never public"):
        pass_privacy(tmp_path)


def test_join_outputs_on_another_node_type_may_use_any_name(tmp_path):
    import copy

    from lgk.privacy.gate import require_clear
    from tests.test_build_sql import JOIN_MAPPING, pass_privacy, write_run

    m = copy.deepcopy(JOIN_MAPPING)
    m["nodes"]["clinic"] = m["nodes"].pop(sf.DIRECTORY_NODE_TYPE)
    m["nodes"]["clinic"]["properties"] = []
    m["nodes"]["clinic"]["joins"][1]["columns"] = ["taxonomy_code"]
    m["nodes"]["clinic"]["joins"][0]["aggregates"][1] = f"count(claim_id) as {sf.DIRECTORY_PROPERTIES[0]}"
    write_run(tmp_path, m)
    pass_privacy(tmp_path)
    assert require_clear(str(tmp_path))


def test_a_public_name_in_a_later_join_is_refused(tmp_path):
    """The guard reads every join: the public-named output is in joins[1]."""
    import copy

    from lgk.privacy.gate import MappingLeakError
    from tests.test_build_sql import JOIN_MAPPING, pass_privacy, write_run

    m = copy.deepcopy(JOIN_MAPPING)
    m["nodes"][sf.DIRECTORY_NODE_TYPE]["properties"] = []
    m["nodes"][sf.DIRECTORY_NODE_TYPE]["joins"][1] = {
        "table": "claims", "key": "provider_id", "subject": "patient_id",
        "aggregates": [f"max(billed_amount) as {sf.DIRECTORY_PROPERTIES[1]}"]}
    write_run(tmp_path, m)
    with pytest.raises(MappingLeakError, match=rf"{sf.DIRECTORY_PROPERTIES[1]}.*never public"):
        pass_privacy(tmp_path)
