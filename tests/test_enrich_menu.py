"""Offline tests for the enrichment menu, the fixed table shapes, and method selection.

No cloud calls: the menu is a pure function of a measured GraphShape.
"""

import re
from dataclasses import fields

import pytest

from lgk.enrich.menu import V1_METHODS, MenuRow, build_menu, select_methods
from lgk.enrich.shape import GraphShape
from lgk.enrich.tables import TABLE_SHAPES, create_table_sql
from lgk.steps.enrich import render_menu

# A star-with-back-edges graph: has edges, a hub, and a 2-cycle (not a DAG).
CYCLIC = GraphShape(n_nodes=40, n_edges=60, max_degree=12, is_dag=False, example_key="k_super")
# A two-node path: a DAG, but no node has two neighbours.
PAIR_DAG = GraphShape(n_nodes=2, n_edges=1, max_degree=1, is_dag=True, example_key="k_a")
# Nodes but no edges at all.
EDGELESS = GraphShape(n_nodes=5, n_edges=0, max_degree=0, is_dag=True, example_key="k_x")
# A tree: DAG, a node with two children.
TREE = GraphShape(n_nodes=5, n_edges=6, max_degree=3, is_dag=True, example_key="k_root")


def _rows(shape):
    return {row.method: row for row in build_menu(shape)}


def _available(shape):
    return {m for m, row in _rows(shape).items() if row.available}


class TestMenuGreying:
    """Rows that do not fit the graph's shape are greyed out with a reason."""

    @pytest.mark.seam2
    def test_cyclic_graph_greys_out_closure_only(self):
        # hostile: the graph HAS edges and a hub, so only the cycle can grey closure out.
        rows = _rows(CYCLIC)
        assert _available(CYCLIC) == {"pagerank", "degree", "components", "similarity"}
        assert "cycle" in rows["closure"].reason

    @pytest.mark.seam2
    def test_pair_dag_greys_out_similarity_but_offers_closure(self):
        # hostile: a DAG with edges — closure must be available, similarity must not.
        rows = _rows(PAIR_DAG)
        assert _available(PAIR_DAG) == {"pagerank", "degree", "components", "closure"}
        assert "neighbour" in rows["similarity"].reason

    @pytest.mark.seam2
    def test_edgeless_graph_offers_only_degree(self):
        # hostile: is_dag=True on an edgeless graph must not make closure available.
        rows = _rows(EDGELESS)
        assert _available(EDGELESS) == {"degree"}
        for m in ("pagerank", "components", "similarity", "closure"):
            assert "no edges" in rows[m].reason, (m, rows[m].reason)

    @pytest.mark.seam2
    def test_tree_offers_every_v1_method(self):
        assert _available(TREE) == set(V1_METHODS)
        assert all(_rows(TREE)[m].reason == "" for m in V1_METHODS)

    @pytest.mark.seam2
    def test_coming_rows_are_shown_and_always_greyed(self):
        for shape in (CYCLIC, PAIR_DAG, EDGELESS, TREE):
            coming = [r for r in build_menu(shape) if r.method not in V1_METHODS]
            assert {r.method for r in coming} == {
                "personalized_pagerank", "community_detection", "embeddings"
            }
            assert all(not r.available and "coming" in r.reason for r in coming)

    @pytest.mark.seam2
    def test_every_row_shows_question_example_and_table(self):
        for row in build_menu(CYCLIC):
            assert row.question.endswith("?"), row
            assert row.table in TABLE_SHAPES, row
            if row.method in V1_METHODS:
                # The example is on the user's own graph: it names a real node key.
                assert "k_super" in row.example, row

    @pytest.mark.seam2
    def test_a_row_carries_no_build_time(self):
        # a live run showed closure ~6x slower than the old model, so no row holds a time.
        # The field list is written here, not read from MenuRow, so a re-added field goes red.
        assert [f.name for f in fields(MenuRow)] == [
            "method", "question", "example", "table", "available", "reason"]

    @pytest.mark.seam2
    def test_the_rendered_menu_shows_no_time_number_even_on_a_huge_graph(self):
        # hostile: 9x10^7 edges, where the old model printed "about N min" on every row.
        huge = GraphShape(5_000_000, 90_000_000, 900, True, "k_super")
        text = render_menu(build_menu(huge), huge)
        table = text[text.index("| Method"):]
        assert "Build time" not in table and "about" not in table, table
        assert not re.search(r"\d+\s*(s|sec|seconds|min|minutes)\b", table), table
        assert table.splitlines()[0] == (
            "| Method | Question | Example on your graph | Adds table | Status |")


class TestSelectMethods:
    """enrich refuses greyed, coming and unknown methods with the menu's reason."""

    @pytest.mark.seam2
    def test_greyed_method_is_refused_with_reason(self):
        with pytest.raises(ValueError, match="closure.*cycle"):
            select_methods(build_menu(CYCLIC), ["pagerank", "closure"])

    @pytest.mark.seam2
    def test_coming_and_unknown_methods_refused(self):
        with pytest.raises(ValueError, match="coming"):
            select_methods(build_menu(TREE), ["personalized_pagerank"])
        with pytest.raises(ValueError, match="not on the menu"):
            select_methods(build_menu(TREE), ["betweenness"])

    @pytest.mark.seam2
    def test_selection_keeps_menu_order_and_drops_duplicates(self):
        # hostile: reordered and duplicated input.
        picked = select_methods(build_menu(TREE), ["closure", "pagerank", "closure", "degree"])
        assert picked == ["pagerank", "degree", "closure"]


class TestTableShapes:
    """The five enrichment tables have the fixed shapes from the spec."""

    @pytest.mark.seam2
    def test_fixed_shapes(self):
        assert {name: [c for c, _ in cols] for name, cols in TABLE_SHAPES.items()} == {
            "node_score": ["node_id", "method", "score", "computed_at"],
            "seed_score": ["seed_id", "node_id", "score"],
            "node_group": ["node_id", "method", "group_id"],
            "similar_to": ["src_id", "dst_id", "score"],
            "closure": ["ancestor_id", "descendant_id", "depth"],
        }

    @pytest.mark.seam2
    def test_create_sql_uses_fixed_columns_in_order(self):
        sql = create_table_sql("c.s", "closure")
        assert sql.startswith("CREATE OR REPLACE TABLE c.s.closure (")
        assert re.search(r"ancestor_id STRING,\s*descendant_id STRING,\s*depth INT", sql)


class TestEnrichStepMenuFile:
    """The enrich step shows the menu when nothing is ticked, and refuses greyed rows."""

    @pytest.fixture
    def cyclic(self, monkeypatch):
        import lgk.steps.enrich as step

        monkeypatch.setattr(step, "measure_shape", lambda sql, fq: CYCLIC)
        return step

    @pytest.mark.seam2
    def test_missing_choice_writes_menu_with_greyed_reason(self, cyclic, tmp_path):
        statements = []
        with pytest.raises(FileNotFoundError, match="enrich.yaml"):
            cyclic.run_enrich(statements.append, "c.s", tmp_path)
        menu = (tmp_path / "enrich_menu.md").read_text()
        assert "| closure |" in menu and "greyed out: the graph has a cycle" in menu
        assert "| pagerank |" in menu and "where k_super ranks" in menu
        assert statements == []  # nothing is written to UC before the user ticks

    @pytest.mark.seam2
    def test_greyed_choice_refused_before_any_table_is_written(self, cyclic, tmp_path):
        (tmp_path / "enrich.yaml").write_text("methods: [pagerank, closure]\n")
        statements = []
        with pytest.raises(ValueError, match="closure: the graph has a cycle"):
            cyclic.run_enrich(statements.append, "c.s", tmp_path)
        assert statements == []

    @pytest.mark.seam2
    def test_empty_methods_is_a_clean_no_op(self, cyclic, tmp_path):
        """`methods: []` (no enrichment): a report is written, no output table, no work table left."""
        import json

        from lgk.enrich.methods import WORK_TABLES
        from lgk.enrich.tables import TABLE_SHAPES

        (tmp_path / "enrich.yaml").write_text("methods: []\n")
        # hostile: a stale report from an earlier run that computed pagerank must not survive
        (tmp_path / "enrich_report.json").write_text(json.dumps({"pagerank": {"iterations": 7}}))
        statements, created = [], []
        report = cyclic.run_enrich(statements.append, "c.s", tmp_path, created.append)
        assert created == []
        assert not [s for s in statements for t in TABLE_SHAPES if f"TABLE c.s.{t} " in s
                    or f"INSERT INTO c.s.{t} " in s], statements
        made = {m for s in statements for m in re.findall(r"CREATE OR REPLACE TABLE c\.s\.(_enrich_\w+)", s)}
        dropped = {m for s in statements for m in re.findall(r"DROP TABLE IF EXISTS c\.s\.(_enrich_\w+)", s)}
        assert made <= dropped == {f"_enrich_{n}" for n in WORK_TABLES}
        on_disk = json.loads((tmp_path / "enrich_report.json").read_text())
        assert on_disk == report and set(report) == {"shape", "menu"}
        assert report["shape"]["is_dag"] is False and len(report["menu"]) == len(build_menu(CYCLIC))
