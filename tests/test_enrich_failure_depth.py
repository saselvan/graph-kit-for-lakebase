"""Offline tests (scripted fake executor, no SQL engine): work tables are dropped on
failure; one shared depth cap for the DAG check and closure."""

import pytest

from lgk.enrich import shape as shape_mod
from lgk.enrich.menu import build_menu
from lgk.enrich.methods import prepare_work_tables, run_closure
from lgk.enrich.shape import GraphShape
from lgk.enrich.tables import create_table_sql
from lgk.steps.enrich import run_enrich
from tests.enrich_fake import BASE_TABLES, FakeSql

DAG = GraphShape(n_nodes=4, n_edges=3, max_degree=2, is_dag=True, example_key="v0")


@pytest.mark.seam2
def test_work_tables_dropped_when_a_method_fails_mid_run(tmp_path, monkeypatch):
    import lgk.steps.enrich as step

    monkeypatch.setattr(step, "measure_shape", lambda sql, fq: DAG)
    # hostile: fail inside PageRank on its 2nd L1 query, after both ping-pong tables exist
    sql = FakeSql(l1_values=[1.0], fail_on="L1_CHANGE", fail_at=2)
    (tmp_path / "enrich.yaml").write_text("methods: [degree, pagerank]\n")
    with pytest.raises(RuntimeError, match="warehouse went away"):
        run_enrich(sql, "c.s", tmp_path)
    assert sql.fail_seen == 2
    assert any("_enrich_pr_b" in s for s in sql.statements)  # the work tables really existed
    assert sql.names() == set(BASE_TABLES) | {"node_score"}  # the run really started writing


@pytest.mark.seam2
def test_dag_check_and_closure_share_one_depth_cap(monkeypatch):
    # hostile: a varied cap (3, not the real 100), so a hardcoded cap cannot pass.
    monkeypatch.setattr(shape_mod, "MAX_DEPTH", 3)

    sql = FakeSql(chain=3)  # longest path = 3 = cap -> fits
    assert shape_mod._dag_status(sql, "c.s") == (True, False)
    sql(create_table_sql("c.s", "closure"))
    prepare_work_tables(sql, "c.s")
    assert run_closure(sql, "c.s") == {"rows": 6, "max_depth": 3}

    sql = FakeSql(chain=4)  # longest path = 4 > cap
    assert shape_mod._dag_status(sql, "c.s") == (False, True)
    sql(create_table_sql("c.s", "closure"))
    prepare_work_tables(sql, "c.s")
    with pytest.raises(RuntimeError, match="cap of 3"):
        run_closure(sql, "c.s")


@pytest.mark.seam2
def test_cycle_is_not_reported_as_too_deep(monkeypatch):
    monkeypatch.setattr(shape_mod, "MAX_DEPTH", 3)
    sql = FakeSql(chain=3, cyclic=True)  # no edge ever peels off
    assert shape_mod._dag_status(sql, "c.s") == (False, False)
    assert not [t for t in sql.names() if t.startswith("_enrich")]  # Kahn tables dropped


@pytest.mark.seam2
def test_too_deep_greys_closure_with_depth_reason_not_cycle(monkeypatch):
    monkeypatch.setattr(shape_mod, "MAX_DEPTH", 7)
    deep = GraphShape(n_nodes=9, n_edges=8, max_degree=2, is_dag=False, example_key="v0", too_deep=True)
    row = {r.method: r for r in build_menu(deep)}["closure"]
    assert not row.available
    assert "deeper than 7" in row.reason and "cycle" not in row.reason
