"""Offline (scripted fake executor, no SQL engine): enrich reports exactly its output tables
for the teardown manifest.

Teardown refuses a UC kit schema that holds a table the manifest never recorded, so:
- every output table enrich creates must be reported (else teardown of an enriched graph refuses);
- no `_enrich_*` work table may be reported or survive (else teardown refuses, or drops scratch
  it should never have seen).
"""

import pytest

from lgk.enrich.shape import GraphShape
from tests.enrich_fake import BASE_TABLES, FakeSql

# A DAG with a hub of degree 2, so every v1 method (closure included) is available.
DAG = GraphShape(n_nodes=4, n_edges=4, max_degree=2, is_dag=True, example_key="a")


@pytest.fixture
def step(monkeypatch):
    import lgk.steps.enrich as step

    monkeypatch.setattr(step, "measure_shape", lambda sql, fq: DAG)
    return step


@pytest.mark.seam2
def test_enrich_reports_every_output_table_once_and_no_work_table(step, tmp_path):
    sql = FakeSql()
    # hostile: degree and pagerank both write node_score (one table, two methods); reordered list
    (tmp_path / "enrich.yaml").write_text("methods: [closure, pagerank, similarity, degree, components]\n")
    created = []
    step.run_enrich(sql, "c.s", tmp_path, created.append)

    assert sorted(created) == ["c.s.closure", "c.s.node_group", "c.s.node_score", "c.s.similar_to"]
    assert any("_enrich_cl_b" in s for s in sql.statements)  # work tables really were made
    # Every table left in the schema is a build table or a reported one: teardown can account for all.
    assert sql.names() == set(BASE_TABLES) | {t.rsplit(".", 1)[1] for t in created}


@pytest.mark.seam2
def test_output_table_is_reported_even_when_the_run_fails(step, tmp_path):
    # hostile: crash inside PageRank, after node_score and the work tables exist
    sql = FakeSql(l1_values=[1.0], fail_on="L1_CHANGE", fail_at=1)
    (tmp_path / "enrich.yaml").write_text("methods: [pagerank]\n")
    created = []
    with pytest.raises(RuntimeError, match="warehouse went away"):
        step.run_enrich(sql, "c.s", tmp_path, created.append)
    assert created == ["c.s.node_score"]
    assert sql.names() == set(BASE_TABLES) | {"node_score"}  # no work table survives the crash


@pytest.mark.seam2
def test_enrich_step_records_output_tables_in_the_run_manifest(step, tmp_path, monkeypatch):
    """The step wires the callback to the manifest (not just run_enrich's default no-op)."""
    from lgk import manifest as mf
    from lgk.manifest import ObjectType

    sql = FakeSql()
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: object())
    monkeypatch.setattr(step, "execute_sql", lambda w, wh, s: sql(s))
    # enrich passes the privacy gate first (proven in tests/test_enrich_serving.py)
    monkeypatch.setattr(step, "require_clear", lambda rf: {})
    monkeypatch.setattr(step.binding, "stamp_id", lambda clear: "stamp")
    (tmp_path / "enrich.yaml").write_text("methods: [degree]\n")
    config = {"graph_name": "g", "lakebase_catalog": "c", "kit_schema": "s", "warehouse_id": "wh", "run_folder": str(tmp_path)}
    step.enrich(config)

    entries = [(e.object_type, e.object_id) for e in mf.load_run_manifest(config).entries]
    assert entries == [(ObjectType.UC_TABLE, "c.s.node_score")]
