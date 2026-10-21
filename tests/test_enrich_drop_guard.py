"""Offline (scripted fake executor, no SQL engine): enrich never drops a pipeline-owned object.

`_drop_unwritten` drops every fixed enrichment table name this run does not write. An operator's
pipeline may own a table with one of those names in the kit schema (seen live: an owner CAN drop a
pipeline-managed STREAMING_TABLE from a warehouse). Enrich must check the table type first and
refuse anything that is not a MANAGED or EXTERNAL Delta table, before it drops anything.
"""

import pytest

from lgk.enrich.shape import GraphShape
from lgk.steps.enrich import _drop_unwritten, run_enrich
from tests.enrich_fake import FakeSql

DAG = GraphShape(n_nodes=4, n_edges=3, max_degree=2, is_dag=True, example_key="v0")


def _drops(sql: FakeSql) -> set:
    return {s.split()[-1] for s in sql.statements
            if s.startswith("DROP TABLE IF EXISTS c.s.") and "_enrich_" not in s}


@pytest.fixture
def step(monkeypatch):
    import lgk.steps.enrich as step

    monkeypatch.setattr(step, "measure_shape", lambda sql, fq: DAG)
    return step


@pytest.mark.parametrize("kind", [
    ("STREAMING_TABLE", "DELTA"),
    ("MATERIALIZED_VIEW", "DELTA"),
    ("VIEW", None),
    ("FOREIGN", "POSTGRESQL"),
    ("EXTERNAL", "PARQUET"),  # hostile: an allowed table type, but not a Delta table the kit writes
    ("Streaming_Table", "delta"),  # hostile: odd case must not slip past the refusal
])
def test_pipeline_owned_table_with_an_enrichment_name_is_refused_and_not_dropped(kind):
    # hostile: the operator's object is `similar_to`, which sorts AFTER `closure`; a guard that
    # checks inside the drop loop would already have dropped the kit's closure table
    sql = FakeSql(kinds={"similar_to": kind})
    sql.tables["c.s.similar_to"] = 7
    sql.tables["c.s.closure"] = 3
    with pytest.raises(RuntimeError) as err:
        _drop_unwritten(sql, "c.s", ["degree"])
    msg = str(err.value)
    assert "c.s.similar_to" in msg and "will not drop" in msg and kind[0] in msg
    assert _drops(sql) == set()  # nothing at all was dropped
    assert {"similar_to", "closure"} <= sql.names()


def test_refusal_holds_through_the_runner_twice(step, tmp_path):
    sql = FakeSql(kinds={"seed_score": ("STREAMING_TABLE", "DELTA")})
    sql.tables["c.s.seed_score"] = 5
    (tmp_path / "enrich.yaml").write_text("methods: [degree]\n")
    for _ in range(2):
        with pytest.raises(RuntimeError, match=r"c\.s\.seed_score.*STREAMING_TABLE"):
            run_enrich(sql, "c.s", tmp_path)
    assert "seed_score" in sql.names()
    assert _drops(sql) == set()
    assert not any(s.startswith("CREATE") for s in sql.statements)  # refused before any method ran


@pytest.mark.parametrize("kind", [("MANAGED", "DELTA"), ("EXTERNAL", "DELTA"), ("MANAGED", "delta"),
                                  ("managed", "DELTA")])  # hostile: lower-case type is still the kit table
def test_kit_delta_table_is_dropped_as_before(kind):
    sql = FakeSql(kinds={"closure": kind, "similar_to": kind})
    sql.tables["c.s.closure"] = 3
    sql.tables["c.s.similar_to"] = 4
    _drop_unwritten(sql, "c.s", ["degree"])
    assert {"closure", "similar_to"}.isdisjoint(sql.names())
    assert "nodes" in sql.names() and "edges" in sql.names()


def test_missing_table_is_a_no_op():
    # hostile: a pipeline object named `closure` in ANOTHER schema must not be mistaken for ours
    sql = FakeSql(kinds={"closure": ("STREAMING_TABLE", "DELTA")})
    sql.tables["c.other.closure"] = 9
    _drop_unwritten(sql, "c.s", ["degree"])  # no enrichment table exists in c.s: no error
    assert sql.names() == {"nodes", "edges", "node_id_map", "closure"}
    assert "c.other.closure" in sql.tables


def test_mixed_case_kit_schema_still_finds_the_pipeline_table():
    # hostile: operator config says `kit_schema: S`; UC stores `s`, and DROP resolves either case
    sql = FakeSql(kinds={"similar_to": ("STREAMING_TABLE", "DELTA")})
    sql.tables["c.s.similar_to"] = 7
    with pytest.raises(RuntimeError, match=r"c\.S\.similar_to is STREAMING_TABLE"):
        _drop_unwritten(sql, "c.S", ["degree"])
    assert not any(s.startswith("DROP") for s in sql.statements)


def test_a_seam_returning_none_is_treated_as_no_rows():
    # hostile: a statement seam that returns None (list.append) rather than []
    statements = []
    _drop_unwritten(statements.append, "c.s", ["degree"])
    assert any("information_schema.tables" in s for s in statements)
    assert sum(s.startswith("DROP TABLE IF EXISTS c.s.") for s in statements) == 4


def test_nothing_unwritten_issues_no_type_query(monkeypatch):
    import lgk.steps.enrich as step

    # hostile: every enrichment table is written (as once personalized PageRank ships)
    monkeypatch.setitem(step.TABLE_OF, "ppr", "seed_score")
    statements = []
    _drop_unwritten(statements.append, "c.s", list(step.TABLE_OF))
    assert statements == []
