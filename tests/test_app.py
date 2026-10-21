"""Seam-2 tests for the app generator: the folder, app.yaml, the printed
deploy command, the never-deploy rule, the privacy floor, and the generated query layer.

Live proof (the reader role over real served views vs a Spark reference) is in
tests/test_app_live.py.
"""

import ast
import importlib.util
import os
import py_compile
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from lgk.app import generate
from lgk.privacy.gate import PrivacyGateError
from lgk.steps.app import app as app_step
from tests.privacy_support import clear_gate_offline
from tests.test_privacy_gate import FOLDED_MAPPING, PROVIDER_MAPPING, _passed, _write_mapping

EXPECTED_FILES = {
    ".lgk-app",
    ".cursor/rules/data-contract.mdc",
    "DATA_CONTRACT.md",
    "README.md",
    "app.yaml",
    "requirements.txt",
    "serve.py",
    "backend/app/__init__.py",
    "backend/app/db.py",
    "backend/app/main.py",
    "backend/app/reader.py",
    "backend/app/walk.py",
    "static/index.html",
}
SPEC_ENV = ["PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGSSLMODE", "ENDPOINT_NAME"]


def _config(run_dir: Path, **app: Any) -> Dict[str, Any]:
    return {
        "graph_name": "care_graph", "kit_schema": "lgk_care", "served_schema": "lgk_care_srv",
        "lakebase_project": "proj-x",
        "branch_id": "t-abc123",  # hostile: not production, so a hard-coded branch is caught
        "run_folder": str(run_dir), "operator_email": "op@example.com",
        "app": {"pghost": "ep-host.database.example.com", **app},
    }


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A run folder whose privacy gate passed for a directory graph (nothing folded)."""
    _write_mapping(tmp_path, PROVIDER_MAPPING)
    clear_gate_offline(str(tmp_path))
    return tmp_path


@pytest.fixture
def folded_run_dir(tmp_path: Path) -> Path:
    """A run folder whose gate passed with a care-subject column FOLDED (weights = people)."""
    _passed(tmp_path)
    return tmp_path


def _files(folder: str) -> set:
    root = Path(folder)
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def _load(folder: str, rel: str):
    """Import one generated module by path, under a unique name (no lgk import in it)."""
    name = f"gen_{abs(hash((folder, rel)))}"
    spec = importlib.util.spec_from_file_location(name, Path(folder) / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------- folder
class TestFolder:
    def test_generated_folder_has_exactly_the_app_files(self, run_dir):
        out = generate.generate_app(_config(run_dir))
        assert out["app_folder"] == str(run_dir / "app" / "lgk-care-graph")  # default place + name
        assert _files(out["app_folder"]) == EXPECTED_FILES
        assert not (Path(out["app_folder"]) / "app.py").exists()  # would shadow backend/app/

    def test_generated_python_compiles_and_has_no_kit_import(self, run_dir):
        folder = generate.generate_app(_config(run_dir))["app_folder"]
        for rel in sorted(f for f in _files(folder) if f.endswith(".py")):
            py_compile.compile(str(Path(folder) / rel), doraise=True)
            text = (Path(folder) / rel).read_text()
            assert not re.search(r"^\s*(from|import) lgk\b", text, re.M), rel  # standalone app

    def test_app_yaml_env_names_match_the_spec_with_explicit_values(self, run_dir):
        folder = generate.generate_app(_config(run_dir, pguser="sp-client-id"))["app_folder"]
        doc = yaml.safe_load((Path(folder) / "app.yaml").read_text())
        assert [e["name"] for e in doc["env"]] == SPEC_ENV
        assert all(set(e) == {"name", "value"} for e in doc["env"])  # no valueFrom: database
        env = {e["name"]: e["value"] for e in doc["env"]}
        assert env == {
            "PGHOST": "ep-host.database.example.com", "PGPORT": "5432",
            "PGDATABASE": "databricks_postgres", "PGUSER": "sp-client-id", "PGSSLMODE": "require",
            "ENDPOINT_NAME": "projects/proj-x/branches/t-abc123/endpoints/primary",
        }
        assert doc["command"] == ["python", "serve.py"]

    def test_pguser_defaults_to_empty_so_the_app_uses_its_own_client_id(self, run_dir):
        folder = generate.generate_app(_config(run_dir))["app_folder"]
        env = {e["name"]: e["value"] for e in yaml.safe_load((Path(folder) / "app.yaml").read_text())["env"]}
        assert env["PGUSER"] == ""
        assert 'os.environ.get("PGUSER") or os.environ["DATABRICKS_CLIENT_ID"]' in (
            Path(folder) / "backend/app/db.py").read_text()

    def test_db_pool_follows_the_lakebase_apps_pattern(self, run_dir):
        folder = Path(generate.generate_app(_config(run_dir))["app_folder"])
        db = (folder / "backend/app/db.py").read_text()
        calls = [n for n in ast.walk(ast.parse(db)) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", None) == "ConnectionPool"]
        assert len(calls) == 1
        kw = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
        assert kw == {"conninfo": "conninfo()", "connection_class": "OAuthConnection",
                      "min_size": "1", "max_size": "5", "max_lifetime": "2700",
                      "configure": "configure_reader", "open": "False"}
        for needle in ("class OAuthConnection(psycopg.Connection)",
                       'endpoint=os.environ["ENDPOINT_NAME"]', 'kwargs["password"] = credential.token'):
            assert needle in db, needle
        assert "get_endpoint" not in db  # never at app startup
        main = (folder / "backend/app/main.py").read_text()
        assert "pool.open(wait=True)" in main and "lifespan=lifespan" in main
        assert "walk.run_walk(conn" in main

    def test_serve_detects_the_apps_runtime_and_reqs_pin_psycopg_pool(self, run_dir):
        folder = Path(generate.generate_app(_config(run_dir))["app_folder"])
        assert 'os.environ.get("DATABRICKS_APP_NAME")' in (folder / "serve.py").read_text()
        reqs = (folder / "requirements.txt").read_text()
        assert "psycopg[binary,pool]" in reqs and "asyncpg" not in reqs

    def test_refuses_to_overwrite_a_folder_the_kit_did_not_generate(self, run_dir, tmp_path):
        out = tmp_path / "mine"
        out.mkdir()
        (out / "notes.txt").write_text("someone else's work")  # hostile: a foreign non-empty folder
        with pytest.raises(generate.AppFolderError):
            generate.generate_app(_config(run_dir, output_dir=str(out)))
        assert _files(str(out)) == {"notes.txt"}

    def test_regenerates_over_its_own_folder(self, run_dir, tmp_path):
        out = tmp_path / "again"
        generate.generate_app(_config(run_dir, output_dir=str(out), depth=1))
        generate.generate_app(_config(run_dir, output_dir=str(out), depth=3))
        assert _load(str(out), "backend/app/walk.py").DEPTH == 3

    @pytest.mark.parametrize("name", ["Bad_Name", "x", "-lead", "trail-", "a" * 31, "has space"])
    def test_invalid_app_name_is_refused(self, run_dir, name):
        with pytest.raises(ValueError, match="app name"):
            generate.generate_app(_config(run_dir, name=name))
        assert not (run_dir / "app").exists()

    def test_camel_and_underscore_graph_name_gives_a_valid_default_name(self, run_dir):
        cfg = {**_config(run_dir), "graph_name": "my_graph_2"}
        assert generate.app_name(cfg) == "lgk-my-graph-2"


# ----------------------------------------------------------------------- data contract
class TestDataContract:
    def test_data_contract_has_three_sections_and_every_endpoint(self, run_dir):
        folder = Path(generate.generate_app(_config(run_dir))["app_folder"])
        text = (folder / "DATA_CONTRACT.md").read_text()
        sections = re.findall(r"^## (.+)$", text, re.M)
        assert sections == ["1. What exists", "2. What does NOT exist", "3. Interpretation rules"]
        for endpoint in ("GET /api/walk", "GET /api/health", "GET /"):
            assert f"### {endpoint}" in text
        for field in ("rows[].hop", "rows[].node_id", "rows[].label", "rows[].support", "start.node_id"):
            assert f"`{field}`" in text
        assert "lgk_care_srv.nodes_view" in text and "$" not in text  # every placeholder filled

    def test_cursor_rule_has_frontmatter_globs_and_the_same_contract(self, run_dir):
        folder = Path(generate.generate_app(_config(run_dir))["app_folder"])
        mdc = (folder / ".cursor/rules/data-contract.mdc").read_text()
        front, body = mdc.split("---\n", 2)[1:]
        meta = yaml.safe_load(front)
        assert meta["globs"] == ["static/**", "backend/**"]
        assert body.strip() == (folder / "DATA_CONTRACT.md").read_text().strip()

    def test_contract_states_the_folded_privacy_rule(self, folded_run_dir):
        folder = Path(generate.generate_app(_config(folded_run_dir))["app_folder"])
        text = (folder / "DATA_CONTRACT.md").read_text()
        assert "distinct people" in text and "fewer than 11" in text


# --------------------------------------------------------------------- deploy command
class TestDeployCommand:
    def test_step_prints_exactly_one_deploy_command_with_the_source_path(self, run_dir, capsys):
        result = app_step(_config(run_dir))
        out = capsys.readouterr().out
        want = ("databricks apps deploy lgk-care-graph "
                "--source-code-path /Workspace/Users/op@example.com/lgk-apps/lgk-care-graph")
        assert out.count("databricks apps deploy") == 1
        assert out.strip().splitlines() == [want]
        assert result["deploy_command"] == want
        assert want in (Path(result["app_folder"]) / "README.md").read_text()

    def test_workspace_path_from_config_is_the_source_path(self, run_dir, capsys):
        app_step(_config(run_dir, name="web-search", workspace_path="/Workspace/Shared/apps/ws"))
        assert capsys.readouterr().out.strip() == (
            "databricks apps deploy web-search --source-code-path /Workspace/Shared/apps/ws")


class TestNeverDeploys:
    @pytest.fixture
    def no_deploy(self, monkeypatch):
        """Any Apps client, any shell-out, any WorkspaceClient at all raises."""
        import databricks.sdk
        from databricks.sdk.service import apps as apps_service

        calls: List[str] = []

        def boom(what):
            def _raise(*a, **k):
                calls.append(what)
                raise AssertionError(f"the kit must never {what}")
            return _raise

        monkeypatch.setattr(databricks.sdk.WorkspaceClient, "apps",
                            property(boom("touch WorkspaceClient.apps")))
        monkeypatch.setattr(databricks.sdk.WorkspaceClient, "__init__", boom("build a WorkspaceClient"))
        monkeypatch.setattr(apps_service.AppsAPI, "__init__", boom("build an AppsAPI"))
        for fn in ("run", "Popen", "call", "check_call", "check_output"):
            monkeypatch.setattr(subprocess, fn, boom(f"run subprocess.{fn}"))
        monkeypatch.setattr(os, "system", boom("run os.system"))
        return calls

    def test_step_never_deploys_and_never_calls_apps(self, run_dir, no_deploy, capsys):
        result = app_step(_config(run_dir))
        assert no_deploy == []
        assert _files(result["app_folder"]) == EXPECTED_FILES  # it did its job
        assert capsys.readouterr().out.count("databricks apps deploy") == 1

    def test_kit_source_has_no_apps_api_or_shell_call(self):
        root = Path(generate.__file__).parent
        kit_code = (root / "generate.py").read_text() + Path(sys.modules[app_step.__module__].__file__).read_text()
        assert not re.search(r"\.apps\b|AppsAPI|subprocess|os\.system", kit_code)


# ------------------------------------------------------------------ privacy and the walk
def _sql_relations(sql: str) -> set:
    """(schema, relation) of every schema-qualified relation after FROM or JOIN."""
    return set(re.findall(r"\b(?:FROM|JOIN)\s+([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)", sql))


class TestWalkQueries:
    def test_every_app_query_reads_only_the_served_views(self, run_dir):
        walk = _load(generate.generate_app(_config(run_dir))["app_folder"], "backend/app/walk.py")
        rels = set()
        for sql in (walk.WALK_SQL, walk.RESOLVE_SQL, walk.LABELS_SQL):
            rels |= _sql_relations(sql)
        assert rels == {("lgk_care_srv", "edges_view"), ("lgk_care_srv", "node_id_map_view"),
                        ("lgk_care_srv", "nodes_view")}, rels
        assert "lgk_care_v" not in walk.WALK_SQL + walk.RESOLVE_SQL + walk.LABELS_SQL  # no raw table

    def test_defaults_give_a_depth_2_top_25_unfolded_walk(self, run_dir):
        walk = _load(generate.generate_app(_config(run_dir))["app_folder"], "backend/app/walk.py")
        assert (walk.DEPTH, walk.TOP_N, walk.PRIVACY_FLOOR, walk.FOLDED) == (2, 25, 11, False)
        assert "hop2 AS (" in walk.WALK_SQL and "hop3" not in walk.WALK_SQL
        assert walk.WALK_SQL.count("LIMIT 25") == 2
        assert "COUNT(*) AS support" in walk.WALK_SQL and "HAVING" not in walk.WALK_SQL

    def test_folded_gate_bakes_the_floor_into_the_walk(self, folded_run_dir):
        walk = _load(generate.generate_app(_config(folded_run_dir))["app_folder"], "backend/app/walk.py")
        assert walk.FOLDED is True and walk.PRIVACY_FLOOR == 11
        assert walk.WALK_SQL.count("HAVING MIN(e.weight) >= 11") == walk.DEPTH

    def test_floor_can_be_raised(self, folded_run_dir):
        walk = _load(generate.generate_app(_config(folded_run_dir, privacy_floor=15))["app_folder"],
                     "backend/app/walk.py")
        assert walk.PRIVACY_FLOOR == 15 and "HAVING MIN(e.weight) >= 15" in walk.WALK_SQL

    @pytest.mark.parametrize("floor", [10, 1, 0, True])
    def test_floor_can_never_be_lowered(self, folded_run_dir, floor):
        with pytest.raises(ValueError, match="floor"):
            generate.generate_app(_config(folded_run_dir, privacy_floor=floor))
        assert not (folded_run_dir / "app").exists()

    def test_no_clear_privacy_gate_no_app(self, tmp_path):
        _write_mapping(tmp_path, PROVIDER_MAPPING)  # hostile: mapping present, privacy never ran
        with pytest.raises(PrivacyGateError):
            generate.generate_app(_config(tmp_path))
        assert not (tmp_path / "app").exists()

    def test_stale_gate_no_app(self, run_dir):
        mapping = dict(PROVIDER_MAPPING, source_schema="edited_after_privacy")
        _write_mapping(run_dir, mapping)
        with pytest.raises(PrivacyGateError, match="stale"):
            generate.generate_app(_config(run_dir))

    @pytest.mark.parametrize("depth", [0, 5])
    def test_web_depth_cap(self, run_dir, depth):
        with pytest.raises(ValueError, match="depth"):
            generate.generate_app(_config(run_dir, depth=depth))


# ---------------------------------------------------------- the generated query layer
class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0]


class _FakeConn:
    """Scripted connection: answers each SQL by its first matching key."""

    def __init__(self, answers):
        self.answers, self.executed, self.commits = answers, [], 0

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        for key, rows in self.answers.items():
            if key in sql:
                return _FakeResult(rows)
        return _FakeResult([])

    def commit(self):
        self.commits += 1


class TestGeneratedQueryLayer:
    @pytest.fixture
    def walk(self, run_dir):
        return _load(generate.generate_app(_config(run_dir))["app_folder"], "backend/app/walk.py")

    def test_ambiguous_key_needs_a_type(self, walk):
        # hostile: the same natural key under two node types
        conn = _FakeConn({"node_id_map_view": [("id1", "hospital"), ("id2", "provider")]})
        with pytest.raises(walk.AmbiguousStart, match="hospital"):
            walk.run_walk(conn, "shared")

    def test_unknown_key_is_not_found(self, walk):
        with pytest.raises(walk.StartNotFound):
            walk.run_walk(_FakeConn({}), "nope", "provider")

    def test_run_walk_binds_parameters_and_joins_labels_in_walk_order(self, walk):
        conn = _FakeConn({
            "node_id_map_view": [("s1", "provider")],
            "WITH hop0": [(1, "b", 3, None), (1, "a", 3, None), (2, "c", 1, None)],
            "nodes_view": [("a", "provider", "A::1", "Alpha"), ("c", "hospital", "C", "Gamma"),
                           ("b", "provider", "B", "Beta")],  # hostile: reordered
        })
        got = walk.run_walk(conn, "S", "provider")
        assert got["start"] == {"node_id": "s1", "node_type": "provider", "natural_key": "S"}
        assert [(r["hop"], r["node_id"], r["label"], r["support"]) for r in got["rows"]] == [
            (1, "b", "Beta", 3), (1, "a", "Alpha", 3), (2, "c", "Gamma", 1)]
        assert conn.executed[0][1] == {"natural_key": "S", "node_type": "provider"}
        assert conn.executed[1][1] == {"start_node_id": "s1"}
        assert conn.executed[2][1] == {"node_ids": ["b", "a", "c"]}

    def test_configure_reader_sets_role_and_refuses_a_non_reader_session(self, run_dir):
        reader = _load(generate.generate_app(_config(run_dir))["app_folder"], "backend/app/reader.py")
        ok = _FakeConn({"current_user": [("reader",)]})
        reader.configure_reader(ok)
        assert [s for s, _ in ok.executed][:2] == ["SET ROLE reader", "SET default_transaction_read_only = on"]
        assert ok.commits == 1
        with pytest.raises(reader.NotReaderError):
            reader.configure_reader(_FakeConn({"current_user": [("operator@example.com",)]}))


# ------------------------------------------------------- prover round: closing survivors
def _load_pkg(folder: str):
    """Import the generated backend/app package (relative imports work); return the package name."""
    pkg = f"genpkg_{abs(hash(folder))}"
    root = Path(folder) / "backend" / "app"
    spec = importlib.util.spec_from_file_location(pkg, root / "__init__.py",
                                                  submodule_search_locations=[str(root)])
    sys.modules[pkg] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules[pkg])
    return pkg


def _calls(tree, attr: str) -> list:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == attr]


class TestProverRound:
    def test_top_n_config_reaches_every_hop(self, run_dir):
        walk = _load(generate.generate_app(_config(run_dir, top_n=7, depth=3))["app_folder"],
                     "backend/app/walk.py")
        assert walk.TOP_N == 7 and walk.WALK_SQL.count("LIMIT 7") == 3 and "LIMIT 25" not in walk.WALK_SQL

    def test_typed_lookup_of_a_shared_key_resolves_by_type(self, run_dir):
        walk = _load(generate.generate_app(_config(run_dir))["app_folder"], "backend/app/walk.py")
        assert "node_type = %(node_type)s" in walk.RESOLVE_SQL

        class TypedConn(_FakeConn):  # applies the type predicate like Postgres would
            def execute(self, sql, params=None):
                if "node_id_map_view" in sql:
                    rows = [("id1", "hospital"), ("id2", "provider")]  # hostile: shared key
                    t = params["node_type"]
                    return _FakeResult([r for r in rows if t is None or r[1] == t])
                return super().execute(sql, params)

        assert walk.resolve_start(TypedConn({}), "shared", "provider")["node_id"] == "id2"

    def test_policy_without_run_folder_checks_the_floor(self):
        from lgk.privacy.gate import served_walk_policy

        with pytest.raises(ValueError, match="floor"):
            served_walk_policy(None, 5, True)
        assert served_walk_policy(None, None, False) == (11, False)

    def test_app_cannot_go_below_a_raised_recorded_floor(self, tmp_path):
        from lgk.privacy.flags import OpenFlagError, resolve_flag
        from lgk.steps.privacy import privacy
        from tests.test_privacy import _visits_and_providers

        _visits_and_providers(tmp_path)
        _write_mapping(tmp_path, FOLDED_MAPPING)  # the fold flag is folded by an edge
        with pytest.raises(OpenFlagError):
            privacy({"run_folder": str(tmp_path), "privacy_floor": 15})
        resolve_flag(str(tmp_path), "visits", "ref_code", "fold", chooser="op")
        resolve_flag(str(tmp_path), "members", "patient_name", "drop", chooser="op")
        privacy({"run_folder": str(tmp_path), "privacy_floor": 15})
        with pytest.raises(ValueError, match="recorded floor"):
            generate.generate_app(_config(tmp_path, privacy_floor=12))  # 12 >= 11, < 15
        walk = _load(generate.generate_app(_config(tmp_path))["app_folder"], "backend/app/walk.py")
        assert walk.PRIVACY_FLOOR == 15 and "HAVING MIN(e.weight) >= 15" in walk.WALK_SQL

    def test_support_is_an_int_even_from_a_text_or_decimal_cell(self, run_dir):
        from decimal import Decimal

        walk = _load(generate.generate_app(_config(run_dir))["app_folder"], "backend/app/walk.py")
        conn = _FakeConn({"node_id_map_view": [("s1", "provider")],
                          "WITH hop0": [(1, "a", "3", None), (2, "b", Decimal("2"), None)]})
        rows = walk.run_walk(conn, "S")["rows"]  # hostile: differently-typed cells
        assert [r["support"] for r in rows] == [3, 2] and all(type(r["support"]) is int for r in rows)

    def test_db_conninfo_defaults_and_credential_endpoint(self, run_dir, monkeypatch):
        import importlib

        import psycopg

        pkg = _load_pkg(generate.generate_app(_config(run_dir))["app_folder"])
        db = importlib.import_module(f"{pkg}.db")
        for name in ("PGUSER", "PGSSLMODE", "PGPORT", "PGDATABASE"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("PGHOST", "h.example")
        monkeypatch.setenv("DATABRICKS_CLIENT_ID", "sp-123")
        monkeypatch.setenv("ENDPOINT_NAME", "projects/p/branches/t-x/endpoints/primary")
        info = psycopg.conninfo.conninfo_to_dict(db.conninfo())
        assert info == {"host": "h.example", "port": "5432", "dbname": "databricks_postgres",
                        "user": "sp-123", "sslmode": "require"}
        seen = {}

        class _Postgres:
            def generate_database_credential(self, endpoint):
                seen["endpoint"] = endpoint
                return type("C", (), {"token": "tok"})()

        monkeypatch.setattr(db, "_workspace_client", lambda: type("W", (), {"postgres": _Postgres()})())
        monkeypatch.setattr(psycopg.Connection, "connect",
                            classmethod(lambda cls, conninfo="", **kw: seen.update(kw) or "conn"))
        assert db.OAuthConnection.connect("x") == "conn"
        assert seen == {"endpoint": "projects/p/branches/t-x/endpoints/primary", "password": "tok"}

    def test_main_wiring_by_ast(self, run_dir):
        tree = ast.parse((Path(generate.generate_app(_config(run_dir))["app_folder"])
                          / "backend/app/main.py").read_text())
        opens = _calls(tree, "open")
        assert len(opens) == 1 and ast.unparse(opens[0]) == "pool.open(wait=True)"
        assert [ast.unparse(c) for c in _calls(tree, "close")] == ["pool.close()"]
        handlers = {ast.unparse(h.type): ast.unparse(h.body[0]) for h in ast.walk(tree)
                    if isinstance(h, ast.ExceptHandler)}
        assert "status_code=404" in handlers["walk.StartNotFound"]
        assert "status_code=409" in handlers["walk.AmbiguousStart"]
        returns = [ast.unparse(r.value) for r in ast.walk(tree) if isinstance(r, ast.Return)]
        assert "{'status': 'ok', 'role': role}" in returns

    @pytest.mark.parametrize("in_apps, host, port", [(False, "127.0.0.1", 8000), (True, "0.0.0.0", 8123)])
    def test_serve_binds_by_runtime_and_reads_the_apps_port(self, run_dir, monkeypatch, in_apps, host, port):
        import runpy
        import types

        folder = generate.generate_app(_config(run_dir))["app_folder"]
        got = {}
        fake_uvicorn = types.SimpleNamespace(run=lambda app, host, port: got.update(host=host, port=port))
        fake_main = types.ModuleType("app.main")
        fake_main.app = object()
        monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
        monkeypatch.setitem(sys.modules, "app", types.ModuleType("app"))
        monkeypatch.setitem(sys.modules, "app.main", fake_main)
        monkeypatch.delenv("DATABRICKS_APP_NAME", raising=False)
        monkeypatch.delenv("DATABRICKS_APP_PORT", raising=False)
        if in_apps:
            monkeypatch.setenv("DATABRICKS_APP_NAME", "x")
            monkeypatch.setenv("DATABRICKS_APP_PORT", "8123")
        monkeypatch.setattr(sys, "path", list(sys.path))
        runpy.run_path(str(Path(folder) / "serve.py"), run_name="__main__")
        assert got == {"host": host, "port": port}


# ------------------------------------------------------------------ app.direction
class TestWalkDirection:
    def test_default_walk_is_outbound_and_the_contract_says_so(self, run_dir):
        folder = generate.generate_app(_config(run_dir))["app_folder"]
        walk = _load(folder, "backend/app/walk.py")
        assert "JOIN hop0 f ON e.src_node_id = f.node_id" in walk.WALK_SQL
        assert "e.dst_node_id = f.node_id" not in walk.WALK_SQL
        text = (Path(folder) / "DATA_CONTRACT.md").read_text()
        assert "It follows edges `outbound`: along each edge, from src_node_id to dst_node_id." in text
        assert "| `rows[].node_id` | string | `lgk_care_srv.edges_view.dst_node_id` |" in text

    def test_inbound_direction_is_baked_into_the_walk_and_the_contract(self, run_dir):
        folder = generate.generate_app(_config(run_dir, direction="inbound"))["app_folder"]
        walk = _load(folder, "backend/app/walk.py")
        assert walk.WALK_SQL.count("f ON e.dst_node_id = f.node_id") == walk.DEPTH == 2
        assert "e.src_node_id = f.node_id" not in walk.WALK_SQL
        text = (Path(folder) / "DATA_CONTRACT.md").read_text()
        assert "It follows edges `inbound`: against each edge" in text and "$" not in text
        assert "| `rows[].node_id` | string | `lgk_care_srv.edges_view.src_node_id` |" in text

    def test_both_direction_contract_states_the_support_rule(self, run_dir):
        folder = generate.generate_app(_config(run_dir, direction="both"))["app_folder"]
        walk = _load(folder, "backend/app/walk.py")
        assert walk.WALK_SQL.count("UNION\n") == 2  # one deduplicating UNION per hop
        text = (Path(folder) / "DATA_CONTRACT.md").read_text()
        assert "counts once per edge type" in text
        assert ("| `rows[].node_id` | string | `lgk_care_srv.edges_view.dst_node_id or src_node_id "
                "(the far end of the edge)` |") in text

    @pytest.mark.parametrize("direction", ["inbound", "both"])
    def test_folded_gate_refuses_any_direction_but_outbound(self, folded_run_dir, direction):
        with pytest.raises(ValueError, match="folded edges are symmetric"):
            generate.generate_app(_config(folded_run_dir, direction=direction))
        assert not (folded_run_dir / "app").exists()

    def test_unknown_direction_is_refused(self, run_dir):
        with pytest.raises(ValueError, match="direction must be one of"):
            generate.generate_app(_config(run_dir, direction="Inbound"))


# ---------------------------------------------------------------- app.edge_types
class TestAppEdgeTypes:
    @pytest.fixture
    def mixed_dir(self, tmp_path: Path) -> Path:
        from tests.test_walks import MIXED_MAPPING, _cleared_run

        _cleared_run(tmp_path, MIXED_MAPPING)
        return tmp_path

    def test_a_mixed_graph_needs_app_edge_types(self, mixed_dir):
        # hostile: folded and unfolded edge types; the app may never get one kind silently
        with pytest.raises(ValueError, match="edge_types is required"):
            generate.generate_app(_config(mixed_dir))

    def test_the_app_walks_its_selection_and_says_so(self, mixed_dir):
        folder = generate.generate_app(_config(mixed_dir, edge_types=["shares_refs"]))["app_folder"]
        walk = _load(folder, "backend/app/walk.py")
        assert "AND e.relationship_type IN ('shares_refs')" in walk.WALK_SQL
        assert "HAVING MIN(e.weight) >= 11" in walk.WALK_SQL and walk.FOLDED is True
        text = (Path(folder) / "DATA_CONTRACT.md").read_text()
        assert "It follows only the edge types `shares_refs`." in text
        assert "Each served count is floored independently; differences between served counts are not protected." in text
        assert "a folded selection's support counts\n  people, an unfolded one's counts edges; mixing is refused" in text
        unfolded = generate.generate_app(_config(mixed_dir, edge_types=["affiliation"],
                                                 output_dir=str(mixed_dir / "app2")))["app_folder"]
        walk2 = _load(unfolded, "backend/app/walk.py")
        assert walk2.FOLDED is False and "HAVING" not in walk2.WALK_SQL

    def test_a_mixed_app_selection_is_refused(self, mixed_dir):
        from lgk.privacy.gate import MixedFoldSelectionError

        with pytest.raises(MixedFoldSelectionError):
            generate.generate_app(_config(mixed_dir, edge_types=["shares_refs", "affiliation"]))

    def test_an_unfolded_graph_app_says_every_edge_type(self, run_dir):
        text = (Path(generate.generate_app(_config(run_dir))["app_folder"]) / "DATA_CONTRACT.md").read_text()
        assert "It follows every edge type of the graph." in text


# ------------------------------------- the walk timeout and the set-walk endpoint
TIMEOUT_STATEMENT = "SET LOCAL statement_timeout = '5s'"  # 5 s, per walk transaction


def _set_mapping() -> dict:
    """Two node types with walk keys.
    # hostile: provider declares TWO walk keys (npi, specialty), so a wrong seed key on the same
    # type still builds valid SQL and only the predicate text tells them apart.
    # hostile: the stop type (hospital) differs from the seed type."""
    import copy

    m = copy.deepcopy(PROVIDER_MAPPING)
    m["nodes"]["provider"].update(properties=["npi", "specialty"], walk_keys=["npi", "specialty"])
    m["nodes"]["hospital"] = {"source_table": "hospitals", "natural_key": "hospital_id",
                              "label": "hospital_name", "properties": ["network_status"],
                              "walk_keys": ["network_status"]}
    return m


@pytest.fixture
def set_run_dir(tmp_path: Path) -> Path:
    _write_mapping(tmp_path, _set_mapping())
    clear_gate_offline(str(tmp_path))
    return tmp_path


class _FakePool:
    """psycopg_pool stand-in: connection() hands out one scripted connection and records whether a
    statement ran inside the `with` (a walk transaction) or outside it."""

    def __init__(self, conn=None, **kwargs):
        self.kwargs, self.conn, self.inside = kwargs, conn, False

    def connection(self):
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            self.inside = True
            try:
                yield self.conn
            finally:
                self.inside = False
        return _ctx()


def _load_main(folder: str, monkeypatch):
    """Import the generated main.py with fastapi and psycopg_pool stubbed (not kit dev deps): the
    route functions stay plain functions, so a test calls an endpoint as the server would."""
    import importlib
    import types

    class _App:
        def __init__(self, **kw):
            self.routes = {}

        def get(self, path):
            def deco(fn):
                self.routes[path] = fn
                return fn
            return deco

    class _HTTPException(Exception):
        def __init__(self, status_code, detail=None):
            super().__init__(detail)
            self.status_code, self.detail = status_code, detail

    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI, fastapi.HTTPException = _App, _HTTPException
    class _Query:  # records its kwargs, so a test can read a parameter's validation
        def __init__(self, default=None, **kw):
            self.default, self.kw = default, kw

    fastapi.Query = _Query
    responses = types.ModuleType("fastapi.responses")
    responses.FileResponse = lambda path: path
    pool_mod = types.ModuleType("psycopg_pool")
    pool_mod.ConnectionPool = _FakePool
    for name, mod in (("fastapi", fastapi), ("fastapi.responses", responses), ("psycopg_pool", pool_mod)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("PGHOST", "h.example")
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "sp-1")
    pkg = _load_pkg(folder)
    return importlib.import_module(f"{pkg}.main"), _HTTPException


class _SessionConn(_FakeConn):
    """Records, per statement, whether it ran inside the pool's connection context."""

    def __init__(self, answers, pool_ref):
        super().__init__(answers)
        self.pool_ref, self.inside = pool_ref, []

    def execute(self, sql, params=None):
        self.inside.append(self.pool_ref[0].inside)
        return super().execute(sql, params)


def _wire(main, answers):
    ref = []
    conn = _SessionConn(answers, ref)
    main.pool.conn = conn
    ref.append(main.pool)
    return conn


class TestWalkTimeoutAndSetWalk:
    def test_every_walk_runs_in_a_transaction_that_starts_with_the_5s_timeout(self, run_dir, monkeypatch):
        main, _exc = _load_main(generate.generate_app(_config(run_dir))["app_folder"], monkeypatch)
        conn = _wire(main, {"node_id_map_view": [("s1", "provider")],
                            "WITH hop0": [(1, "a", 2, None)],
                            "nodes_view": [("a", "provider", "A", "Alpha")]})
        got = main.get_walk("S", "provider")
        assert [r["node_id"] for r in got["rows"]] == ["a"]
        sqls = [s for s, _ in conn.executed]
        assert sqls[0] == TIMEOUT_STATEMENT  # first statement of the walk's transaction
        assert sqls.count(TIMEOUT_STATEMENT) == 1 and len(sqls) == 4
        assert all(conn.inside)  # timeout and walk share one pooled connection context
        # the timeout is per walk (SET LOCAL), not baked into the pool's configure callback
        assert main.pool.kwargs["configure"].__name__ == "configure_reader"

    def test_health_check_is_not_a_walk_and_gets_no_timeout(self, run_dir, monkeypatch):
        main, _exc = _load_main(generate.generate_app(_config(run_dir))["app_folder"], monkeypatch)
        conn = _wire(main, {"current_user": [("reader",)]})
        assert main.health() == {"status": "ok", "role": "reader"}
        assert TIMEOUT_STATEMENT not in [s for s, _ in conn.executed]

    def test_set_walk_endpoint_runs_the_kits_set_walk_with_seed_and_stop(self, set_run_dir, monkeypatch):
        from lgk.steps.walks import get_walk_query
        from lgk.walks.sql import NodePredicate

        folder = generate.generate_app(_config(set_run_dir))["app_folder"]
        main, _exc = _load_main(folder, monkeypatch)
        conn = _wire(main, {
            "WITH seed_all": [(1, "h1", 3, None, True), (1, "p2", 3, None, False), (2, "p9", "1", None, False)],
            "nodes_view": [("p9", "provider", "P9", "Nine"), ("h1", "hospital", "H1", "Mercy"),
                           ("p2", "provider", "P2", "Two")],  # hostile: reordered
        })
        got = main.get_set_walk(seed_key="provider.specialty", seed_value="cardiology",
                                   stop_key="hospital.network_status", stop_value="out")
        sql, params = conn.executed[1]
        assert conn.executed[0][0] == TIMEOUT_STATEMENT and all(conn.inside)
        assert params == {"seed_value": "cardiology", "stop_value": "out"}
        assert "n.node_type = 'provider' AND n.properties ->> 'specialty' = %(seed_value)s" in sql
        assert "m.node_type = 'hospital' AND m.properties ->> 'network_status' = %(stop_value)s" in sql
        assert sql == get_walk_query("", depth=2, served_schema="lgk_care_srv", run_folder=str(set_run_dir),
                                     seed=NodePredicate("provider", "specialty"),
                                     stop=NodePredicate("hospital", "network_status"))
        assert got["seed"] == {"key": "provider.specialty", "value": "cardiology"}
        assert got["stop"] == {"key": "hospital.network_status", "value": "out"}
        assert [(r["hop"], r["node_id"], r["label"], r["support"], r["matched"]) for r in got["rows"]] == [
            (1, "h1", "Mercy", 3, True), (1, "p2", "Two", 3, False), (2, "p9", "Nine", 1, False)]
        assert all(type(r["support"]) is int and type(r["matched"]) is bool for r in got["rows"])

    def test_set_walk_without_stop_binds_only_the_seed(self, set_run_dir, monkeypatch):
        from lgk.steps.walks import get_walk_query
        from lgk.walks.sql import NodePredicate

        main, _exc = _load_main(generate.generate_app(_config(set_run_dir))["app_folder"], monkeypatch)
        conn = _wire(main, {"WITH seed_all": []})
        got = main.get_set_walk(seed_key="provider.npi", seed_value="123")
        sql, params = conn.executed[1]
        assert params == {"seed_value": "123"} and got["stop"] is None and got["rows"] == []
        assert "%(stop_value)s" not in sql
        assert sql == get_walk_query("", depth=2, served_schema="lgk_care_srv", run_folder=str(set_run_dir),
                                     seed=NodePredicate("provider", "npi"))

    @pytest.mark.parametrize("kw", [
        {"seed_key": "provider.name", "seed_value": "x"},  # hostile: a property-like key never declared
        {"seed_key": "provider.specialty", "seed_value": "x", "stop_key": "hospital.npi", "stop_value": "y"},
        {"seed_key": "provider.specialty", "seed_value": "x", "stop_key": "hospital.network_status"},  # no value
    ])
    def test_set_walk_refuses_an_undeclared_key_or_a_stop_without_value(self, set_run_dir, monkeypatch, kw):
        main, exc = _load_main(generate.generate_app(_config(set_run_dir))["app_folder"], monkeypatch)
        conn = _wire(main, {})
        with pytest.raises(exc) as e:
            main.get_set_walk(**kw)
        assert e.value.status_code == 400
        assert not any("seed_all" in s for s, _ in conn.executed)  # no walk SQL ran

    def test_no_walk_keys_no_set_walk_endpoint(self, run_dir, monkeypatch):
        folder = generate.generate_app(_config(run_dir))["app_folder"]
        main, _exc = _load_main(folder, monkeypatch)
        assert "/api/set-walk" not in main.app.routes and "/api/walk" in main.app.routes
        assert _load(folder, "backend/app/walk.py").SET_WALK_SQL == {}
        assert "/api/set-walk" not in (Path(folder) / "DATA_CONTRACT.md").read_text()

    def test_set_walk_endpoint_is_registered_and_in_the_contract(self, set_run_dir, monkeypatch):
        folder = generate.generate_app(_config(set_run_dir))["app_folder"]
        main, _exc = _load_main(folder, monkeypatch)
        assert main.app.routes["/api/set-walk"] is main.get_set_walk
        text = (Path(folder) / "DATA_CONTRACT.md").read_text()
        assert "### GET /api/set-walk" in text and "`rows[].matched`" in text and "$" not in text
        walk = _load(folder, "backend/app/walk.py")
        # every (seed key, stop key or none) pair, and nothing else
        keys = ["hospital.network_status", "provider.npi", "provider.specialty"]
        assert set(walk.SET_WALK_SQL) == {(s, t) for s in keys for t in [None, *keys]}


class _MultiConnPool(_FakePool):
    """Hands out a NEW scripted connection per connection() call, so a timeout set on one
    connection cannot pass for a timeout on the walk's connection (prover M4)."""

    def __init__(self, answers, **kwargs):
        super().__init__(**kwargs)
        self.answers, self.handed = answers, []

    def connection(self):
        self.conn = _FakeConn(self.answers)
        self.handed.append(self.conn)
        return super().connection()


class TestSetWalkProverRound:
    def test_timeout_and_walk_share_one_connection_per_walk(self, set_run_dir, monkeypatch):
        main, _exc = _load_main(generate.generate_app(_config(set_run_dir))["app_folder"], monkeypatch)
        main.pool = _MultiConnPool({"node_id_map_view": [("s1", "provider")], "WITH hop0": [(1, "a", 1, None)]})
        main.get_walk("S", "provider")
        main.get_set_walk(seed_key="provider.npi", seed_value="1")
        assert len(main.pool.handed) == 2  # one connection per walk
        for conn, marker in zip(main.pool.handed, ("WITH hop0", "WITH seed_all")):
            sqls = [s for s, _ in conn.executed]
            assert sqls[0] == TIMEOUT_STATEMENT and any(marker in s for s in sqls[1:])

    @pytest.mark.parametrize("kind", ["folded", "mixed"])
    def test_set_walk_takes_the_apps_depth_top_n_floor_and_edge_types(self, tmp_path, kind):
        # hostile: non-default depth / top_n / raised floor, and a mixed graph with an app selection
        import copy

        from lgk.steps.walks import get_walk_query
        from lgk.walks.sql import NodePredicate
        from tests.test_privacy_gate import FOLDED_MAPPING as FM

        if kind == "folded":
            m = copy.deepcopy(FM)
            m["nodes"]["provider"]["walk_keys"] = ["npi"]
            _passed(tmp_path, m)
            app, edge_types = {"privacy_floor": 15}, None
        else:
            from tests.test_walks import SET_MAPPING, _cleared_run

            _cleared_run(tmp_path, SET_MAPPING)
            app, edge_types = {"edge_types": ["affiliation"]}, ["affiliation"]
        walk = _load(generate.generate_app(_config(tmp_path, depth=3, top_n=7, **app))["app_folder"],
                     "backend/app/walk.py")
        sql = walk.SET_WALK_SQL[("provider.npi", "provider.npi")]
        assert sql == get_walk_query("", depth=3, served_schema="lgk_care_srv", run_folder=str(tmp_path),
                                     top_n=7, privacy_floor=app.get("privacy_floor"), edge_types=edge_types,
                                     seed=NodePredicate("provider", "npi"), stop=NodePredicate("provider", "npi"))
        assert "hop3 AS" in sql and "hop4 AS" not in sql and sql.count("LIMIT 7)") == 3
        if kind == "folded":
            assert sql.count("HAVING MIN(e.weight) >= 15") == 3
        else:
            assert "IN ('affiliation')" in sql and "shares_refs" not in sql and "HAVING" not in sql

    def test_an_error_inside_the_walk_session_reaches_the_caller(self, run_dir, monkeypatch):
        main, exc = _load_main(generate.generate_app(_config(run_dir))["app_folder"], monkeypatch)

        class _Timeout(_FakeConn):
            def execute(self, sql, params=None):
                if "WITH hop0" in sql:
                    raise RuntimeError("canceling statement due to statement timeout")
                return super().execute(sql, params)

        main.pool.conn = _Timeout({"node_id_map_view": [("s1", "provider")]})
        with pytest.raises(RuntimeError, match="statement timeout"):
            main.get_walk("S", "provider")
        main.pool.conn = _FakeConn({})
        with pytest.raises(exc) as e:
            main.get_walk("nope", "provider")
        assert e.value.status_code == 404

    @pytest.mark.parametrize("kw", [
        {"seed_key": "provider.npi", "seed_value": "x", "stop_key": "", "stop_value": "y"},  # hostile: empty key
        {"seed_key": "provider.npi", "seed_value": "x", "stop_value": "y"},  # a stop value with no stop key
    ])
    def test_a_stop_value_never_goes_unused(self, set_run_dir, monkeypatch, kw):
        main, exc = _load_main(generate.generate_app(_config(set_run_dir))["app_folder"], monkeypatch)
        conn = _wire(main, {})
        with pytest.raises(exc) as e:
            main.get_set_walk(**kw)
        assert e.value.status_code == 400 and not any("seed_all" in s for s, _ in conn.executed)

    def test_empty_values_are_exact_values(self, set_run_dir, monkeypatch):
        main, _exc = _load_main(generate.generate_app(_config(set_run_dir))["app_folder"], monkeypatch)
        conn = _wire(main, {"WITH seed_all": [(1, "h1", 1, None, 1), (1, "p2", 1, None, 0)]})
        got = main.get_set_walk(seed_key="provider.npi", seed_value="",
                                stop_key="hospital.network_status", stop_value="")
        assert conn.executed[1][1] == {"seed_value": "", "stop_value": ""}
        # hostile: integer cells for matched; the app serves real booleans
        assert [(r["matched"], type(r["matched"])) for r in got["rows"]] == [(True, bool), (False, bool)]

    def test_contract_states_the_set_walks_real_settings(self, set_run_dir, tmp_path_factory, monkeypatch):
        monkeypatch.setattr(generate, "MAX_SEEDS_DEFAULT", 37)  # hostile: reading, not hard-coding
        text = (Path(generate.generate_app(_config(set_run_dir, depth=3, top_n=7))["app_folder"])
                / "DATA_CONTRACT.md").read_text()
        for needle in ("(at most 37, more is an error)", "then depth 3, at most 7 nodes per hop over all seeds together",  # the set-walk paragraph only
                       "edges followed outbound", "`hospital.network_status`, `provider.npi`, "
                       "`provider.specialty`", "no other walk than these two."):
            assert needle in text, needle
        run_dir = tmp_path_factory.mktemp("plain")  # its own folder: no walk keys
        _write_mapping(run_dir, PROVIDER_MAPPING)
        clear_gate_offline(str(run_dir))
        plain = (Path(generate.generate_app(_config(run_dir))["app_folder"]) / "DATA_CONTRACT.md").read_text()
        assert "no other walk." in plain and "than these two" not in plain

    def test_set_walk_query_validation_requires_a_key_and_allows_empty_values(self, set_run_dir, monkeypatch):
        import inspect

        main, _exc = _load_main(generate.generate_app(_config(set_run_dir))["app_folder"], monkeypatch)
        params = inspect.signature(main.get_set_walk).parameters
        assert params["seed_key"].default.kw == {"min_length": 1}
        assert params["seed_value"].default.kw == {}  # "" is a value: equality is exact
        assert params["stop_key"].default is None and params["stop_value"].default is None
