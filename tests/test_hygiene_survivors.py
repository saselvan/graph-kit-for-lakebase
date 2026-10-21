"""#24 prover survivors: guards and decomposed behaviour that no offline test pinned before."""

import re
from types import SimpleNamespace

import pytest

from tests.test_hygiene import HOSTILE_NAMES


# ---------------------------------------------------------------------------- ident, more callers
def _tree(v):
    from lgk.walks.sql import tree_walk_cte
    return tree_walk_cte(depth=1, served_schema=v, top_n=3)


def _index_column(v):
    from lgk.advisor import index_name
    return index_name("col", v)


def _floor_member(v):
    from lgk.privacy.floor import fold_edges_sql
    return fold_edges_sql("c.s.t", "subject", v)


def _is_large(v):
    from lgk.advisor import _is_large

    class Conn:
        def execute(self, *a):
            raise AssertionError("SQL ran with an unchecked table name")
    return _is_large(Conn(), "s", v, 10)


def _web_edges(v):
    from lgk.walks.sql import web_sql
    return web_sql(depth=1, served_schema="served", top_n=3, edges_relation=v)


def _tree_edges(v):
    from lgk.walks.sql import tree_walk_cte
    return tree_walk_cte(depth=1, served_schema="served", top_n=3, edges_relation=v)


def _tree_sql_edges(v):
    from lgk.walks.sql import tree_sql
    return tree_sql(depth=1, served_schema="served", top_n=3, edges_relation=v)


@pytest.mark.parametrize("caller", [_tree, _index_column, _floor_member, _is_large,
                                    _web_edges, _tree_edges, _tree_sql_edges],
                         ids=lambda c: c.__name__)
@pytest.mark.parametrize("name", HOSTILE_NAMES, ids=repr)
def test_more_callers_reject_the_same_hostile_names(caller, name):
    with pytest.raises(ValueError):
        caller(name)


# hostile: one part, three parts, mixed case, trailing newline in the second part
@pytest.mark.parametrize("table", ["a", "a.b.c", "Foo.bar", "a.b\n", "a.B"])
@pytest.mark.parametrize("fn", ["similar_sql", "vector_index_sql"])
def test_vector_table_must_be_exactly_schema_dot_table(fn, table):
    import lgk.walks.sql as ws

    with pytest.raises(ValueError):
        getattr(ws, fn)(table, 3) if fn == "similar_sql" else getattr(ws, fn)(table)


# ---------------------------------------------------------------------------- branch
@pytest.mark.parametrize("branch_id", ["", None])  # hostile: present but empty
def test_empty_branch_id_is_refused_not_production(branch_id, monkeypatch):
    import lgk.walks.nb_runtime as rt
    from lgk.branch import BranchRef, MissingBranchError, connect_branch

    cfg = {"lakebase_project": "p", "branch_id": branch_id}
    with pytest.raises(MissingBranchError, match="branch_id"):
        BranchRef.from_config(cfg)
    calls = []
    monkeypatch.setattr(rt, "connect", lambda p, b: calls.append((p, b)))
    with pytest.raises(MissingBranchError):
        connect_branch(cfg)
    assert calls == []


def test_teardown_ops_use_the_configured_branch(monkeypatch):
    from lgk.steps import teardown as td

    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: object())
    ops = td.LiveOps({"lakebase_project": "p", "branch_id": "t-x", "warehouse_id": "wh",
                      "lakebase_catalog": "c", "graph_name": "g"})
    assert ops._branch_name() == "projects/p/branches/t-x"


# ---------------------------------------------------------------------------- harness
def _leasor(tmp_path, base="tb-base"):
    from unittest.mock import MagicMock

    from lgk.harness import BranchLeasor

    w = SimpleNamespace(api_client=SimpleNamespace(do=MagicMock(return_value={})))
    return BranchLeasor(w, "proj", base, run_folder=tmp_path), w


def test_harness_posts_from_the_test_base_with_a_string_ttl(tmp_path, monkeypatch):
    monkeypatch.setattr("lgk.harness._seam2_guard_active", lambda: False)
    leasor, w = _leasor(tmp_path)
    out = leasor.create_test_branch("x")
    (method, path), kw = w.api_client.do.call_args
    assert (method, path) == ("POST", "/api/2.0/postgres/projects/proj/branches")
    assert kw["body"]["spec"]["source_branch"] == "projects/proj/branches/tb-base"
    assert kw["body"]["spec"]["ttl"] == f"{leasor.ttl_seconds}s"
    assert kw["query"]["branch_id"] == out["branch_id"]  # response had none: the requested id


# hostile: upper case + underscore + space, over-long, ends in a separator after the cut
@pytest.mark.parametrize("name", ["My_Graph Test", "a" * 80, "x" * 60 + "_"])
def test_safe_branch_id_is_a_legal_lakebase_branch_id(name):
    from lgk.harness import _safe_branch_id

    assert re.fullmatch(r"[a-z]([a-z0-9-]{0,61}[a-z0-9])?", _safe_branch_id(name))


def test_safe_branch_id_keeps_the_name_lower_cased():
    from lgk.harness import _safe_branch_id

    # hostile: upper case would otherwise be replaced by '-', not lower-cased
    assert _safe_branch_id("My_Graph Test").startswith("t-my-graph-test-")


# ---------------------------------------------------------------------------- build SQL
def test_edges_sql_dedups_per_pair_and_hashes_each_end_with_its_type():
    from lgk.mapping import EdgeSpec
    from lgk.steps.build import _insert_edges_sql

    sql = _insert_edges_sql("c.k.edges", "c.k.nodes", "c.s.t", "works_at",
                            EdgeSpec("t", "provider_id", "provider", "site_id", "site"))
    assert "PARTITION BY provider_id, site_id" in sql
    assert "xxhash64('provider::'||provider_id)" in sql
    assert "xxhash64('site::'||site_id)" in sql


def test_a_node_without_properties_gets_an_empty_map():
    from lgk.mapping import NodeSpec
    from lgk.steps.build import _insert_nodes_sql

    assert "map() as properties" in _insert_nodes_sql("n", "s", "t", NodeSpec("t", "k", "label", []))


def test_build_inserts_nodes_before_edges_before_node_id_map(tmp_path, monkeypatch):
    from lgk.mapping import EdgeSpec, NodeSpec
    from lgk.steps import build as b

    (tmp_path / "graph_mapping.yaml").write_text("x")
    mapping = SimpleNamespace(source_catalog="c", source_schema="s",
                              nodes={"p": NodeSpec("pt", "id", "name", [])},
                              edges={"e": EdgeSpec("et", "a", "p", "b", "p")})
    monkeypatch.setattr(b, "load_mapping", lambda p: mapping)
    monkeypatch.setattr(b, "require_clear", lambda *a: {})
    monkeypatch.setattr(b.binding, "stamp_id", lambda flags: "stamp")
    monkeypatch.setattr(b.binding, "record_build", lambda *a: None)
    monkeypatch.setattr(b, "_claim_kit_schema", lambda *a: None)
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: object())
    from tests.label_support import stub_table_identities

    stub_table_identities(monkeypatch)  # the recording warehouse has no Delta history
    sql = []
    monkeypatch.setattr(b, "execute_sql", lambda w, wh, s: sql.append(" ".join(s.split())) or [])
    b.build({"graph_name": "g", "run_folder": str(tmp_path), "lakebase_catalog": "c",
             "kit_schema": "k", "warehouse_id": "wh"})
    inserts = [s.split()[2] for s in sql if s.startswith("INSERT INTO")]
    assert inserts == ["c.k.nodes", "c.k.edges", "c.k.node_id_map"]


# ---------------------------------------------------------------------------- provision decomposition
CFG = {"graph_name": "g", "lakebase_project": "p", "lakebase_catalog": "c", "kit_schema": "k",
       "branch_id": "t-x", "warehouse_id": "wh", "version": 2}


@pytest.fixture
def prov_log(monkeypatch):
    from lgk.steps import provision as prov

    log = []
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: object())
    monkeypatch.setattr(prov.mf, "claim_uc_schema",
                        lambda w, wh, cat, schema, graph: log.append(("claim", schema)) or False)
    # the enrichment sync set comes from the run's enrich report (tests/test_provision_enrichment.py)
    monkeypatch.setattr(prov, "enrichment_sync_set", lambda config: {"node_score": ["node_id", "method"]})
    monkeypatch.setattr(prov, "_create_one_synced_table",
                        lambda w, c, t, src, dest, pk, br: log.append(("create", dest, tuple(pk), br)))
    monkeypatch.setattr(prov, "_wait_synced_table_online", lambda w, dest: log.append(("wait", dest)))
    # the per-source room check reads UC; tested in tests/test_sync_limits.py
    monkeypatch.setattr(prov, "_assert_per_source_room", lambda w, wh, c, tables: None)
    monkeypatch.setattr(prov, "_record_synced_table",
                        lambda c, w, dest, br: log.append(("record", dest)))
    return prov, log


def test_synced_tables_claim_both_schemas_then_create_wait_record_each(prov_log):
    prov, log = prov_log
    prov._create_synced_tables(CFG)
    assert log[:2] == [("claim", "k"), ("claim", "k_v2")]
    br = "projects/p/branches/t-x"
    expected = []
    for t, pk in [("nodes", ("node_id",)), ("edges", ("edge_id",)), ("node_id_map", ("node_id",)),
                  ("node_score", ("node_id", "method"))]:
        d = f"c.k_v2.{t}"
        expected += [("create", d, pk, br), ("wait", d), ("record", d)]
    assert log[2:] == expected


def test_a_table_that_never_comes_online_is_not_recorded(prov_log, monkeypatch):
    prov, log = prov_log

    def times_out(w, dest):
        raise TimeoutError(dest)
    monkeypatch.setattr(prov, "_wait_synced_table_online", times_out)
    with pytest.raises(TimeoutError):
        prov._create_synced_tables(CFG)
    assert not [e for e in log if e[0] == "record"]


def test_a_create_error_other_than_already_exists_propagates(monkeypatch):
    from lgk.steps import provision as prov

    checked = []
    monkeypatch.setattr(prov, "_scheduling_policy", lambda *a: None)
    monkeypatch.setattr(prov, "_assert_existing_synced_table_on_branch", lambda *a: checked.append(a))

    def refuse(**kw):
        raise RuntimeError("permission denied")
    w = SimpleNamespace(postgres=SimpleNamespace(create_synced_table=refuse))
    with pytest.raises(RuntimeError, match="permission denied"):
        prov._create_one_synced_table(w, CFG, "nodes", "c.k.nodes", "c.k_v2.nodes", ["node_id"],
                                      "projects/p/branches/t-x")
    assert checked == []
