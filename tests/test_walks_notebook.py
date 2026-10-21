"""Offline acceptance for the walks notebook.

The generated notebook is executed cell by cell (tests/notebook_support.py) against a scripted,
parameter-aware fake Postgres: the answer depends on the SQL text AND the bound seed, so a
notebook that hardcodes an answer, formats the seed into SQL, or shows other SQL than the
app's is caught. The live counterpart (tests/test_walks_notebook_live.py) runs it on a branch.
"""

import contextlib
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from lgk.privacy.gate import PrivacyGateError
from lgk.steps.notebook import notebook as notebook_step
from lgk.steps.walks import get_walk_query
from lgk.walks import nb_runtime
from lgk.walks import notebook as nb
from lgk.walks.sql import tree_sql
from tests.notebook_support import LocalNotebook
from tests.privacy_support import clear_gate_offline
from tests.test_build_provision_walks import _write_provider_graph_mapping

SERVED = "lgk_served_abc123"
KIT = "lgk_v1_abc123"
WORST = "n_worst"
TOKEN = "tok-SECRET-" + "0123456789"
HOST = "ep-fake.database.example.com"
EMAIL = "operator@example.com"
# hostile: a seed with quote, comment, semicolon and a psycopg placeholder in it; it must be
# bound as a parameter, never formatted into the SQL.
HOSTILE_SEED = "x'); DROP TABLE t;--\"%s"

PLAN_DOC = [{"Plan": {"Node Type": "Sort", "Actual Rows": 3, "Plans": [
    {"Node Type": "Nested Loop", "Plans": [
        {"Node Type": "Index Scan", "Index Name": "lgk_edges_src_node_id_idx",
         "Relation Name": "edges"},
        {"Node Type": "Seq Scan", "Relation Name": "nodes"},
        {"Node Type": "Bitmap Index Scan", "Index Name": "nodes_pkey"},
    ]}]}, "Execution Time": 0.42}]


def fake_answer(sql: str, seed: str) -> List[List[Any]]:
    """The fake's answer: a function of the exact SQL text and the bound seed."""
    return [[1, f"{seed}|{len(sql)}", len(sql) % 7, None], [2, f"{seed}|tail", 1, 0.5]]


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConn:
    def __init__(self):
        self.calls: List[tuple] = []
        self.autocommit = False
        self.closed = False

    def execute(self, sql: str, params: Optional[Dict[str, Any]] = None) -> _Result:
        self.calls.append((sql, params))
        if sql.startswith("SET "):
            return _Result([])
        if sql.startswith("EXPLAIN"):
            assert "%(start_node_id)s" in sql and set(params or {}) == {"start_node_id"}, sql
            # hostile: the plan arrives as JSON text, not parsed (driver-dependent)
            return _Result([(json.dumps(PLAN_DOC),)])
        if params is None:  # pick_seed: the served view, worst case first, byte-order ties
            assert f"FROM {SERVED}.edges_view" in sql, sql
            assert 'ORDER BY count(*) DESC, src_node_id COLLATE "C" LIMIT 1' in sql, sql
            return _Result([(WORST,)])
        # every seeded statement (EXPLAIN too) binds the seed; it is never formatted in
        assert "%(start_node_id)s" in sql and set(params) == {"start_node_id"}, sql
        return _Result([tuple(r) for r in fake_answer(sql, params["start_node_id"])])

    def close(self):
        self.closed = True


class _FakeWorkspaceClient:
    calls: List[tuple] = []

    def __init__(self, *a, **k):
        outer = self

        class _PG:
            def get_endpoint(self, name):
                outer.calls.append(("get_endpoint", name))
                return type("E", (), {"status": type("S", (), {"hosts": type("H", (), {"host": HOST})})})

            def generate_database_credential(self, endpoint):
                outer.calls.append(("credential", endpoint))
                return type("C", (), {"token": TOKEN})

        class _Me:
            def me(self):
                return type("U", (), {"user_name": EMAIL})

        self.postgres, self.current_user = _PG(), _Me()


@pytest.fixture
def fake_pg(monkeypatch):
    """Fake SDK + psycopg.connect: records the connect kwargs, hands out one FakeConn."""
    import databricks.sdk
    import psycopg

    state: Dict[str, Any] = {"conns": [], "connect_kwargs": []}
    _FakeWorkspaceClient.calls = []

    def _connect(**kwargs):
        state["connect_kwargs"].append(kwargs)
        conn = FakeConn()
        state["conns"].append(conn)
        return conn

    monkeypatch.setattr(databricks.sdk, "WorkspaceClient", _FakeWorkspaceClient)
    monkeypatch.setattr(psycopg, "connect", _connect)
    return state


def _config(tmp_path: Path, **kw) -> Dict[str, Any]:
    run = tmp_path / "run"
    run.mkdir()
    _write_provider_graph_mapping(run / "graph_mapping.yaml", "cat", "fx")
    clear_gate_offline(str(run))
    # hostile: version 3 and a kit schema different from the served schema, so a notebook that
    # read version tables (lgk_v1_abc123_v3.*) is visible in the source.
    return {"graph_name": "provider_graph", "lakebase_project": "proj-x", "branch_id": "t-nb-1",
            "kit_schema": KIT, "served_schema": SERVED, "version": 3, "run_folder": str(run), **kw}


def _app_sql(config: Dict[str, Any], shape: str, depth: int) -> str:
    """The SQL the app serves (web) / the walk engine's tree SQL — computed independently."""
    if shape == "web":
        return get_walk_query("", depth=depth, served_schema=SERVED, run_folder=config["run_folder"])
    return tree_sql(depth=depth, served_schema=SERVED, top_n=25)


def _run(config, widgets=None):
    book = LocalNotebook(nb.build_notebook(config), widgets)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = book.run_all()
    return book, result, out.getvalue()


EXPECTED_KEYS = [f"{s}:{d}" for s in ("web", "tree") for d in (1, 2, 3, 4)]


class TestNotebookAnswersAreLive:
    def test_every_walk_cell_runs_the_app_sql_live_and_shows_it(self, tmp_path, fake_pg):
        """All defaults: web + tree, depths 1..4, worst-case seed picked live."""
        config = _config(tmp_path)
        _, result, out = _run(config)
        assert result["seed"] == WORST
        assert sorted(result["answers"]) == sorted(EXPECTED_KEYS)
        for key in EXPECTED_KEYS:
            shape, depth = key.split(":")
            sql = _app_sql(config, shape, int(depth))
            assert result["answers"][key] == fake_answer(sql, WORST), key
            assert sql in out, f"{key}: the exact SQL is not shown"
            assert f"--- {shape} walk, depth {depth}" in out
        assert out.count("this run: ") == len(EXPECTED_KEYS)
        assert out.count("plan: ") == len(EXPECTED_KEYS)
        conn = fake_pg["conns"][0]
        walk_runs = [c for c in conn.calls if c[1] and not c[0].startswith("EXPLAIN")]
        assert len(walk_runs) == len(EXPECTED_KEYS)

    def test_seed_widget_changes_every_answer(self, tmp_path, fake_pg):
        """# hostile: a different seed at run time than at generation — a hardcoded answer fails."""
        config = _config(tmp_path)
        _, result, _ = _run(config, widgets={"seed_node_id": HOSTILE_SEED})
        assert result["seed"] == HOSTILE_SEED
        for key in EXPECTED_KEYS:
            shape, depth = key.split(":")
            assert result["answers"][key] == fake_answer(_app_sql(config, shape, int(depth)), HOSTILE_SEED)
        for sql, params in fake_pg["conns"][0].calls:
            assert HOSTILE_SEED not in sql, "seed formatted into SQL"
        assert all(p == {"start_node_id": HOSTILE_SEED}
                   for _s, p in fake_pg["conns"][0].calls if p is not None)

    def test_rerunning_a_cell_requeries_and_reproduces(self, tmp_path, fake_pg):
        config = _config(tmp_path)
        book = LocalNotebook(nb.build_notebook(config))
        with contextlib.redirect_stdout(io.StringIO()):
            book.run_until_last()
            first = json.loads(json.dumps(book.ns["ANSWERS"]))
            n_calls = len(fake_pg["conns"][0].calls)
            assert len(book.walk_cells()) == len(EXPECTED_KEYS)
            for i in book.walk_cells():
                book.run_cell(i)
            assert len(fake_pg["conns"][0].calls) == n_calls + 2 * len(EXPECTED_KEYS)
            assert book.ns["ANSWERS"] == first
            result = book.finish()
        assert result["answers"] == first
        assert fake_pg["conns"][0].closed

    def test_default_seed_from_advisor_then_config(self, tmp_path, fake_pg):
        config = _config(tmp_path)
        report = {"params": {"start_node_id": "n_adv"},
                  "depths": {"web": {"2": {"after": {"p50_ms": 1.234, "p95_ms": 5.678, "samples": 20}}}}}
        (Path(config["run_folder"]) / "advisor_report.json").write_text(json.dumps(report))
        _, result, out = _run(config)
        assert result["seed"] == "n_adv"
        assert "index advisor: p50 1.234 ms, p95 5.678 ms (20 server-side samples)" in out
        assert out.count("index advisor: p50") == 1  # only web depth 2 was recorded
        assert out.count("index advisor: no recorded timing") == len(EXPECTED_KEYS) - 1
        _, result, _ = _run({**config, "notebook_seed_node_id": "n_cfg"})
        assert result["seed"] == "n_cfg"


class TestPlanSummary:
    def test_summary_reduces_explain_json(self):
        s = nb_runtime.summarize_plan(json.dumps(PLAN_DOC))
        assert s["node_types"] == ["Sort", "Nested Loop", "Index Scan", "Seq Scan", "Bitmap Index Scan"]
        assert s["index_scans"] == [
            {"type": "Index Scan", "index": "lgk_edges_src_node_id_idx", "relation": "edges"},
            {"type": "Bitmap Index Scan", "index": "nodes_pkey", "relation": None}]
        assert s["seq_scans"] == ["nodes"]
        assert s["uses_index"] is True
        assert s["rows"] == 3 and s["execution_ms"] == 0.42

    def test_seq_scan_only_plan_reports_no_index(self):
        doc = [{"Plan": {"Node Type": "Seq Scan", "Relation Name": "edges", "Actual Rows": 7}}]
        s = nb_runtime.summarize_plan(doc)
        assert s == {"node_types": ["Seq Scan"], "index_scans": [], "seq_scans": ["edges"],
                     "uses_index": False, "rows": 7, "execution_ms": None}

    def test_plan_cell_output_names_the_index(self, tmp_path, fake_pg):
        _, _, out = _run(_config(tmp_path))
        plans = [json.loads(line[len("plan: "):]) for line in out.splitlines() if line.startswith("plan: ")]
        assert plans and all(p["index_scans"][0]["index"] == "lgk_edges_src_node_id_idx" for p in plans)

    def test_recorded_timing_reads_after_rows_only(self):
        report = {"depths": {"web": {"1": {"before": {"p50_ms": 9, "p95_ms": 9},
                                           "after": {"p50_ms": 1, "p95_ms": 2, "samples": 20}}}}}
        assert nb_runtime.recorded_timing(report, "web", 1) == {"p50_ms": 1, "p95_ms": 2, "samples": 20}
        assert nb_runtime.recorded_timing(report, "web", 2) is None
        assert nb_runtime.recorded_timing(report, "tree", 1) is None
        assert nb_runtime.recorded_timing(None, "web", 1) is None
        assert nb_runtime.recorded_timing({"depths": {"web": {"1": {"before": {}}}}}, "web", 1) is None
        # hostile: a before-only row with real numbers must not be shown as the served timing
        before_only = {"depths": {"web": {"1": {"before": {"p50_ms": 9, "p95_ms": 9, "samples": 5}}}}}
        assert nb_runtime.recorded_timing(before_only, "web", 1) is None

    def test_jsonable_converts_decimal(self):
        from decimal import Decimal

        assert nb_runtime.jsonable([(1, Decimal("2.5"), None, True, b"x")]) == [[1, 2.5, None, True, "b'x'"]]


class TestPrivacy:
    def test_notebook_sql_never_names_a_raw_version_table(self, tmp_path):
        config = _config(tmp_path)
        source = nb.build_notebook(config)
        assert f"{KIT}_v" not in source and "_synced" not in source
        refs = re.findall(r"\b(?:FROM|JOIN)\s+([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)", source)  # SQL keywords are upper case; python "from x.y import" is not SQL
        assert refs and {s for s, _ in refs} == {SERVED}
        assert {t for _, t in refs} <= {"edges_view", "nodes_view", "node_id_map_view"}

    @pytest.mark.parametrize("sql", [
        "SELECT * FROM lgk_v1_abc123_v3.edges",                      # raw version table
        "SELECT * FROM lgk_served_abc123.edges_view e\n JOIN\n  lgk_v1_abc123_synced . nodes n ON 1=1",
        "select * from LGK_SERVED_ABC123.edges_view",                # wrong case = other schema
        "SELECT * FROM lgk_served_abc123.secret_table",              # served schema, not a view
        "SELECT 1",                                                  # reads no served view
    ])
    def test_assert_served_only_refuses(self, sql):
        with pytest.raises(nb.RawRelationError):
            nb.assert_served_only(sql, SERVED)

    def test_assert_served_only_passes_shipped_sql(self, tmp_path):
        config = _config(tmp_path)
        for per_depth in nb.notebook_queries(config).values():
            for sql in per_depth.values():
                nb.assert_served_only(sql, SERVED)

    def test_generation_refuses_without_privacy_gate(self, tmp_path):
        run = tmp_path / "run"
        run.mkdir()
        _write_provider_graph_mapping(run / "graph_mapping.yaml", "cat", "fx")
        config = {"graph_name": "g", "lakebase_project": "p", "kit_schema": KIT,
                  "served_schema": SERVED, "run_folder": str(run)}
        with pytest.raises(PrivacyGateError):
            nb.build_notebook(config)
        with pytest.raises(PrivacyGateError):
            notebook_step(config)
        assert not (run / nb.NOTEBOOK_NAME).exists()

    def test_generation_refuses_a_stale_gate(self, tmp_path):
        config = _config(tmp_path)
        mapping = Path(config["run_folder"]) / "graph_mapping.yaml"
        mapping.write_text(mapping.read_text() + "\n# edited after the privacy step\n")
        with pytest.raises(PrivacyGateError):
            nb.build_notebook(config)

    def test_folded_edges_apply_the_floor_and_drop_tree(self, tmp_path, monkeypatch):
        config = _config(tmp_path)
        monkeypatch.setattr(nb, "served_walk_policy", lambda rf, f, fo: (13, True))
        queries = nb.notebook_queries(config)
        assert list(queries) == ["web"]
        assert all("HAVING MIN(e.weight) >= 13" in sql for sql in queries["web"].values())

    def test_credential_is_used_never_printed(self, tmp_path, fake_pg):
        config = _config(tmp_path)
        book, _, out = _run(config)
        assert TOKEN not in out and TOKEN not in nb.build_notebook(config)
        assert TOKEN not in json.dumps(book.dbutils.notebook.exit_value)
        assert fake_pg["connect_kwargs"] == [{"host": HOST, "dbname": "databricks_postgres",
                                              "user": EMAIL, "password": TOKEN, "sslmode": "require"}]
        assert _FakeWorkspaceClient.calls == [
            ("get_endpoint", "projects/proj-x/branches/t-nb-1/endpoints/primary"),
            ("credential", "projects/proj-x/branches/t-nb-1/endpoints/primary")]
        conn = fake_pg["conns"][0]
        assert conn.autocommit is True
        assert ("SET statement_timeout = '30s'", None) in conn.calls

    def test_kit_connection_uses_the_same_runtime(self, fake_pg):
        from lgk.branch import connect_branch

        connect_branch({"lakebase_project": "proj-y", "branch_id": "t-b"})
        assert fake_pg["connect_kwargs"][0]["password"] == TOKEN
        assert _FakeWorkspaceClient.calls[0] == ("get_endpoint", "projects/proj-y/branches/t-b/endpoints/primary")


class TestNoDefaultBranch:
    def test_notebook_and_kit_refuse_a_config_without_a_branch(self, tmp_path, fake_pg):
        """Owner "branch a": no production fallback in the notebook nor in the kit connector."""
        from lgk.branch import MissingBranchError, connect_branch

        config = _config(tmp_path)
        del config["branch_id"]
        with pytest.raises(MissingBranchError, match="branch_id"):
            _run(config)
        with pytest.raises(MissingBranchError, match="branch_id"):
            connect_branch({"lakebase_project": "proj-x"})
        assert [c for c in _FakeWorkspaceClient.calls if c[0] == "get_endpoint"] == []


class TestNotebookFile:
    def test_step_writes_the_notebook_into_the_run_folder(self, tmp_path):
        config = _config(tmp_path)
        path = Path(notebook_step(config))
        assert path == Path(config["run_folder"]) / "walks_notebook.py"
        text = path.read_text()
        assert text.startswith("# Databricks notebook source\n")
        assert text == nb.build_notebook(config)
        # the runtime cell is the shipped module, verbatim
        assert Path(nb_runtime.__file__).read_text().rstrip() in text

    def test_depth_and_top_n_settings_are_honoured(self, tmp_path, fake_pg):
        config = _config(tmp_path, notebook_max_depth=2, top_n=7)
        _, result, _ = _run(config)
        assert sorted(result["answers"]) == ["tree:1", "tree:2", "web:1", "web:2"]
        web2 = get_walk_query("", depth=2, served_schema=SERVED, top_n=7, run_folder=config["run_folder"])
        assert result["answers"]["web:2"] == fake_answer(web2, WORST)
        assert result["answers"]["tree:2"] == fake_answer(tree_sql(depth=2, served_schema=SERVED, top_n=7), WORST)


class TestGuardsWiredIn:
    def test_notebook_queries_refuses_sql_that_reads_a_version_table(self, tmp_path, monkeypatch):
        """# hostile: a walk builder that points at the version tables must be refused."""
        config = _config(tmp_path)
        monkeypatch.setattr(nb, "per_depth_sql", lambda shape, depth, **kw: {
            1: f"SELECT node_id FROM {KIT}_v3.edges WHERE src_node_id = %(start_node_id)s"})
        with pytest.raises(nb.RawRelationError, match=f"{KIT}_v3.edges"):
            nb.notebook_queries(config)
        with pytest.raises(nb.RawRelationError):
            nb.build_notebook(config)

    def test_pick_seed_refuses_an_empty_served_graph(self):
        class _Empty(FakeConn):
            def execute(self, sql, params=None):
                return _Result([])

        with pytest.raises(ValueError, match="edges_view is empty"):
            nb_runtime.pick_seed(_Empty(), SERVED)
