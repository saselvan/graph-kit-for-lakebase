"""Offline (scripted fake executor, no SQL engine): enrich hygiene and binding.

- enrich drops every enrichment table this run does not write (seed_score included), so an
  earlier run's table can never be synced (M2.1);
- enrich runs only behind a clear privacy gate, stamps each output with the run's stamp and
  records "enrich" in privacy_bindings.json.

The live proof is tests/test_lookups_live.py::test_greyed_closure_leaves_no_table_no_synced_table_no_view.
"""

import json
import re

import pytest

from lgk.enrich.shape import GraphShape
from tests.enrich_fake import FakeSql

DAG = GraphShape(n_nodes=4, n_edges=4, max_degree=2, is_dag=True, example_key="a")
STAMP_SQL = "ALTER TABLE {t} SET TBLPROPERTIES ('lgk.privacy_stamp' = '{s}')"


@pytest.fixture
def step(monkeypatch):
    import lgk.steps.enrich as step

    monkeypatch.setattr(step, "measure_shape", lambda sql, fq: DAG)
    return step


def _stale(sql: FakeSql) -> FakeSql:
    # hostile: an earlier run left every enrichment table behind (closure from an acyclic build,
    # similar_to and node_group from a looser decision, a seed_score no method writes)
    for t in ("closure", "similar_to", "node_group", "seed_score", "node_score"):
        sql.tables[f"c.s.{t}"] = 3
    return sql


def test_enrich_drops_tables_it_did_not_write(step, tmp_path):
    sql = _stale(FakeSql())
    (tmp_path / "enrich.yaml").write_text("methods: [degree]\n")
    step.run_enrich(sql, "c.s", tmp_path, lambda t: None)

    drops = {s.split()[-1] for s in sql.statements if s.startswith("DROP TABLE IF EXISTS c.s.")
             and "_enrich_" not in s}
    assert drops == {"c.s.closure", "c.s.similar_to", "c.s.node_group", "c.s.seed_score"}
    assert sql.names() == {"nodes", "edges", "node_id_map", "node_score"}  # only this run's output


def test_enrich_drops_nothing_it_writes(step, tmp_path):
    sql = _stale(FakeSql())
    # hostile: reordered list; pagerank and degree share node_score
    (tmp_path / "enrich.yaml").write_text("methods: [closure, pagerank, similarity, degree, components]\n")
    step.run_enrich(sql, "c.s", tmp_path, lambda t: None)
    drops = {s.split()[-1] for s in sql.statements if s.startswith("DROP TABLE IF EXISTS c.s.")
             and "_enrich_" not in s}
    assert drops == {"c.s.seed_score"}


def test_enrich_drops_before_a_failing_method_runs(step, tmp_path):
    # hostile: the run crashes inside PageRank; the stale closure must already be gone
    sql = _stale(FakeSql(l1_values=[1.0], fail_on="L1_CHANGE", fail_at=1))
    (tmp_path / "enrich.yaml").write_text("methods: [pagerank]\n")
    with pytest.raises(RuntimeError, match="warehouse went away"):
        step.run_enrich(sql, "c.s", tmp_path, lambda t: None)
    assert "closure" not in sql.names() and "similar_to" not in sql.names()


def _run_step(step, tmp_path, monkeypatch, methods: str, stamp: str = "S1"):
    sql = FakeSql()
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: object())
    monkeypatch.setattr(step, "execute_sql", lambda w, wh, s: sql(s))
    gate = []
    monkeypatch.setattr(step, "require_clear", lambda rf: gate.append(rf) or {"stamp": stamp})
    monkeypatch.setattr(step.binding, "stamp_id", lambda clear: clear["stamp"])
    (tmp_path / "enrich.yaml").write_text(f"methods: {methods}\n")
    config = {"graph_name": "g", "lakebase_catalog": "c", "kit_schema": "s", "warehouse_id": "wh",
              "run_folder": str(tmp_path)}
    step.enrich(config)
    return sql, gate


def test_enrich_stamps_every_output(step, tmp_path, monkeypatch):
    # hostile: degree and pagerank both write node_score (one stamp, not two); reordered list
    sql, gate = _run_step(step, tmp_path, monkeypatch, "[components, pagerank, degree]")
    assert gate == [str(tmp_path)]
    stamps = [s for s in sql.statements if "lgk.privacy_stamp" in s]
    assert sorted(stamps) == sorted(STAMP_SQL.format(t=t, s="S1") for t in ("c.s.node_group", "c.s.node_score"))
    # each stamp is set after its table's rows are written (the last write precedes it)
    last_insert = max(i for i, s in enumerate(sql.statements) if s.startswith("INSERT INTO c.s."))
    assert all(sql.statements.index(s) > last_insert for s in stamps)
    record = json.loads((tmp_path / "privacy_bindings.json").read_text())["enrich"]
    assert record == {"stamp": "S1", "tables": ["c.s.node_group", "c.s.node_score"],
                      "methods": ["pagerank", "degree", "components"]}


def test_enrich_record_replaces_an_earlier_one(step, tmp_path, monkeypatch):
    # hostile: an earlier enrich under another stamp recorded closure; the new record must not keep it
    (tmp_path / "privacy_bindings.json").write_text(json.dumps(
        {"build": {"stamp": "S1", "tables": []},
         "enrich": {"stamp": "S0", "tables": ["c.s.closure"], "methods": ["closure"]}}))
    _run_step(step, tmp_path, monkeypatch, "[degree]")
    data = json.loads((tmp_path / "privacy_bindings.json").read_text())
    assert data["enrich"] == {"stamp": "S1", "tables": ["c.s.node_score"], "methods": ["degree"]}
    assert data["build"] == {"stamp": "S1", "tables": []}  # the build record is untouched


def test_enrich_refuses_without_a_clear_gate(step, tmp_path, monkeypatch):
    from lgk.privacy.gate import PrivacyGateError

    ran = []
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: object())
    monkeypatch.setattr(step, "execute_sql", lambda w, wh, s: ran.append(s) or [])
    (tmp_path / "enrich.yaml").write_text("methods: [degree]\n")
    # hostile: enrich.yaml is ready, but there is no flags.json: the privacy step never ran
    with pytest.raises(PrivacyGateError, match="flags.json"):
        step.enrich({"graph_name": "g", "lakebase_catalog": "c", "kit_schema": "s",
                     "warehouse_id": "wh", "run_folder": str(tmp_path)})
    assert ran == []


# ------------------------------------------- the enrich floor hook for folded edges
FOLDS = [{"edge_type": "shares_patients", "table": "encounters", "subject": "patient_id",
          "src": "provider_id", "dst": "provider_id", "form": "co_membership"}]


class _MixedEdges:
    """Wraps FakeSql; answers the floor check from per-type edge weights, honouring its IN list."""

    def __init__(self, weights):
        self.sql, self.weights, self.checks = FakeSql(), weights, []

    def __call__(self, statement):
        if not statement.startswith("SELECT relationship_type, MIN(weight)"):
            return self.sql(statement)
        self.checks.append(statement)
        listed = re.search(r"IN \(([^)]*)\)", statement)[1]
        rows = []
        for t in sorted(re.findall(r"'([^']+)'", listed)):
            ws = self.weights.get(t, [])
            known = [w for w in ws if w is not None]
            rows.append([t, min(known) if known else None, ws.count(None), len(ws)])
        return rows


def _run_folded(step, tmp_path, monkeypatch, weights):
    wh = _MixedEdges(weights)
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: object())
    monkeypatch.setattr(step, "execute_sql", lambda w, wid, s: wh(s))
    monkeypatch.setattr(step, "require_clear", lambda rf: {"stamp": "S1", "floor": 11, "folds": FOLDS})
    monkeypatch.setattr(step.binding, "stamp_id", lambda clear: clear["stamp"])
    (tmp_path / "enrich.yaml").write_text("methods: [degree]\n")
    step.enrich({"graph_name": "g", "lakebase_catalog": "c", "kit_schema": "s", "warehouse_id": "wh",
                 "run_folder": str(tmp_path)})
    return wh


def test_enrich_refuses_folded_edges_below_floor(step, tmp_path, monkeypatch):
    from lgk.privacy.floor import FoldFloorError

    with pytest.raises(FoldFloorError, match=r"shares_patients \(min weight 10"):
        _run_folded(step, tmp_path, monkeypatch, {"shares_patients": [12, 10, 11]})
    assert not (tmp_path / "privacy_bindings.json").exists()  # nothing computed, stamped or recorded


def test_enrich_accepts_null_weight_on_unfolded_types(step, tmp_path, monkeypatch):
    # hostile: a mixed graph; the unfolded affiliation edges have NULL weight by design (and one
    # would be below the floor if read as a count), so a check that did not filter would refuse
    wh = _run_folded(step, tmp_path, monkeypatch,
                     {"shares_patients": [11, 12], "affiliation": [None, None, None]})
    (check,) = wh.checks
    assert "FROM c.s.edges WHERE relationship_type IN ('shares_patients')" in check
    assert json.loads((tmp_path / "privacy_bindings.json").read_text())["enrich"]["methods"] == ["degree"]
