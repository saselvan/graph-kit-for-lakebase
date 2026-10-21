"""#24 hygiene: one warehouse_id, one identifier rule, one branch connector, no dead modules.

Each test calls the shipped code and goes red when its guard is loosened (see the PR's
mutation table).
"""

import ast
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from lgk import ident
from lgk.uc_sql import MissingWarehouseError, warehouse_id

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "lgk"


# ---------------------------------------------------------------------------- warehouse_id
@pytest.mark.parametrize("config", [{}, {"warehouse_id": ""}, {"warehouse_id": None},
                                    {"warehouse_id": "  "}])  # hostile: blank, not just absent
def test_warehouse_id_missing_raises_naming_the_key(config):
    with pytest.raises(MissingWarehouseError, match="warehouse_id"):
        warehouse_id(config)


def test_warehouse_id_returns_the_configured_one():
    assert warehouse_id({"warehouse_id": "abc123"}) == "abc123"


def test_no_warehouse_id_literal_in_src():
    # hostile: any 16-hex literal (a warehouse id), not only the one that used to be the default
    hits = [f"{p.relative_to(ROOT)}:{i}" for p in SRC.rglob("*.py")
            for i, line in enumerate(p.read_text().splitlines(), 1)
            if re.search(r"\b[0-9a-f]{16}\b", line)]
    assert hits == []


class _NoSql:
    def __getattr__(self, name):
        raise AssertionError(f"workspace call {name} ran without a warehouse")


@pytest.fixture
def no_workspace(monkeypatch):
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: _NoSql())


CFG = {"graph_name": "g", "lakebase_project": "p", "lakebase_catalog": "c", "kit_schema": "k",
       "branch_id": "t-x"}


def _build(tmp_path, monkeypatch):
    from lgk.steps import build as b

    (tmp_path / "graph_mapping.yaml").write_text("x")
    mapping = SimpleNamespace(source_catalog="c", source_schema="s", nodes={}, edges={})
    monkeypatch.setattr(b, "load_mapping", lambda p: mapping)
    monkeypatch.setattr(b, "require_clear", lambda *a: {})
    monkeypatch.setattr(b.binding, "stamp_id", lambda flags: "stamp")
    monkeypatch.setattr(b, "execute_sql", lambda *a, **k: pytest.fail("SQL ran"))
    b.build({**CFG, "run_folder": str(tmp_path)})


def _enrich(tmp_path, monkeypatch):
    from lgk.steps import enrich as e

    e.enrich({**CFG, "run_folder": str(tmp_path)})


def _provision_stamps(tmp_path, monkeypatch):
    from lgk.steps import provision as prov

    monkeypatch.setattr(prov.binding, "check_build_record", lambda *a: None)
    prov._refuse_unbound_build(CFG, str(tmp_path), "stamp", {})


def _provision_sync(tmp_path, monkeypatch):
    from lgk.steps import provision as prov

    prov._create_synced_tables(CFG)


def _retire_uc(tmp_path, monkeypatch):
    from lgk.steps import retire as rt

    rt._drop_uc_schema(CFG, "k_v1")


def _teardown_ops(tmp_path, monkeypatch):
    from lgk.steps import teardown as td

    ops_cls = td.LiveOps
    ops_cls({**CFG, "run_folder": str(tmp_path)})


def _freshness(tmp_path, monkeypatch):
    from lgk.refresh import freshness

    freshness.live_clients(CFG)


@pytest.mark.parametrize("site", [_build, _enrich, _provision_stamps, _provision_sync, _retire_uc,
                                  _teardown_ops, _freshness])
def test_every_step_refuses_a_config_without_warehouse_id(site, tmp_path, monkeypatch, no_workspace):
    """Each former default site raises before any SQL (a reintroduced default goes red here)."""
    with pytest.raises(MissingWarehouseError):
        site(tmp_path, monkeypatch)


# ---------------------------------------------------------------------------- one identifier rule
HOSTILE_NAMES = [
    "Foo",            # mixed case: Postgres would fold it to foo
    "FOO",
    "fooBar",         # camelCase
    "foo\n",          # trailing newline: `$` (not fullmatch) lets it through
    "1foo",
    "foo-bar",
    "foo bar",
    "foo;drop",
    "",
    None,
    "a" * 64,         # over Postgres' 63-byte limit
    "föo",       # non-ASCII
    "foo.bar",        # dotted, where one part is expected
]


def _mapping(v):
    from lgk.mapping import identifier
    return identifier(v, "nodes.x.label")


def _versions(v):
    from lgk.versions import kit_schema
    return kit_schema({"graph_name": v})


def _walks(v):
    from lgk.walks.sql import web_sql
    return web_sql(depth=1, served_schema=v, top_n=3)


def _walks_enrich(v):
    from lgk.walks.sql import web_sql
    return web_sql(depth=1, served_schema="s", top_n=3, enrich_schema=v, sort_method="degree")


def _advisor(v):
    from lgk.advisor import index_name
    return index_name(v, "col")


def _floor(v):
    from lgk.privacy.floor import fold_edges_sql
    return fold_edges_sql("c.s.t", v, "member")


def _mondo(v):
    from lgk.loaders.mondo import load_mondo
    return load_mondo(None, "wh", v, "s", "/nonexistent", doc={})


def _app_queries(v):
    from lgk.app.queries import attribution_query
    return attribution_query(v)


CALLERS = [_mapping, _versions, _walks, _walks_enrich, _advisor, _floor, _mondo, _app_queries]


@pytest.mark.parametrize("caller", CALLERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("name", HOSTILE_NAMES, ids=repr)
def test_every_caller_rejects_the_same_hostile_names(caller, name):
    with pytest.raises(ValueError):
        caller(name)


@pytest.mark.parametrize("caller", [c for c in CALLERS if c is not _mondo], ids=lambda c: c.__name__)
def test_every_caller_accepts_the_same_good_name(caller):
    caller("foo_1")  # no raise


def test_mapping_mixed_case_message_says_lower_case_and_names_the_field():
    from lgk.mapping import MappingError, identifier

    with pytest.raises(MappingError, match=r"nodes\.x\.label.*lower case.*'provider_id'"):
        identifier("Provider_Id", "nodes.x.label")


def test_qualified_names_need_every_part_valid():
    assert ident.qualified("a.b_1.c", "t") == "a.b_1.c"
    for bad in ("a.B", "a..b", "a.b.c.d", ".a", "a.", 3):
        with pytest.raises(ident.IdentifierError):
            ident.qualified(bad, "t")


def test_no_module_keeps_its_own_identifier_pattern():
    """The rule lives in lgk.ident only; a copied pattern elsewhere is a second rule."""
    hits = [str(p.relative_to(ROOT)) for p in sorted(SRC.rglob("*.py"))
            if p.name != "ident.py" and re.search(r"\[a-z_\]\[a-z0-9_\]\*|\[A-Za-z_\]\[A-Za-z0-9_\]\*",
                                                  p.read_text())]
    # advisor.py and walks/notebook.py parse names OUT of SQL/EXPLAIN text (readers, not validators)
    assert hits == ["src/lgk/advisor.py", "src/lgk/walks/notebook.py"]


# ---------------------------------------------------------------------------- branch
def test_branch_ref_builds_the_path():
    from lgk.branch import BranchRef

    # no default branch (owner "branch a"): tests/test_branch_default.py holds the refusals
    ref = BranchRef.from_config({"lakebase_project": "p", "branch_id": "t-b"})
    assert (ref.path, ref.pg_location) == ("projects/p/branches/t-b", "postgres:projects/p/branches/t-b")


def test_connect_branch_uses_the_one_runtime_connector(monkeypatch):
    import lgk.walks.nb_runtime as rt
    from lgk.branch import connect_branch

    calls = []
    monkeypatch.setattr(rt, "connect", lambda project, branch: calls.append((project, branch)) or "conn")
    assert connect_branch({"lakebase_project": "p", "branch_id": "t-b"}) == "conn"
    assert connect_branch({"lakebase_project": "q", "branch_id": "production"}) == "conn"  # named, not defaulted
    assert calls == [("p", "t-b"), ("q", "production")]


def test_branch_id_is_read_only_by_branch_ref():
    hits = [f"{p.relative_to(ROOT)}:{i}" for p in SRC.rglob("*.py") if p.name != "branch.py"
            for i, line in enumerate(p.read_text().splitlines(), 1)
            if re.search(r"""config\.get\(\s*["']branch_id["']""", line)]
    assert hits == []


def test_no_module_imports_a_private_connector():
    assert not any("_connect_branch_pg" in p.read_text() for p in SRC.rglob("*.py"))


def test_retire_has_no_function_named_teardown():
    from lgk.steps import retire, teardown

    assert not hasattr(retire, "teardown") and callable(retire.retire_versions)
    assert callable(teardown.teardown)


# ---------------------------------------------------------------------------- served schema default
def test_walk_helpers_have_no_served_schema_default_of_their_own():
    from lgk.app.queries import attribution_query
    from lgk.steps.walks import get_walk_query

    for fn in (get_walk_query, attribution_query):
        assert inspect.signature(fn).parameters["served_schema"].default is inspect.Parameter.empty
    assert not any('"lgk_served"' in p.read_text() for p in SRC.rglob("*.py"))


def test_the_one_served_default_is_the_kit_schema():
    from lgk.versions import served_schema

    assert served_schema({"graph_name": "g"}) == "lgk_g"
    assert served_schema({"graph_name": "g", "kit_schema": "kk"}) == "kk"


# ---------------------------------------------------------------------------- nodes dedup (SQL shape)
def test_nodes_insert_has_no_silent_dedup_and_drops_null_keys():
    """Build refuses a duplicate natural key (tests/test_keys.py), so the INSERT keeps every row
    with a non-NULL key: no ROW_NUMBER winner. The live proof of the NULL key is
    tests/test_build_provision_walks.py."""
    from lgk.mapping import NodeSpec
    from lgk.steps.build import _insert_nodes_sql

    sql = _insert_nodes_sql("c.k.nodes", "c.s.t", "provider",
                            NodeSpec("t", "provider_id", "name", ["specialty"]))
    assert "ROW_NUMBER" not in sql and "rn = 1" not in sql and "ORDER BY" not in sql
    assert "WHERE n.provider_id IS NOT NULL" in sql


# ---------------------------------------------------------------------------- no dead modules
# Modules reached without a static import: loaded by name (runner/skill step strings), copied
# into the notebook verbatim, or entry points.
ENTRY_POINTS = {
    "lgk", "lgk.bundle_entry", "lgk.notebook_runner", "lgk.harness", "lgk.interview", "lgk.parity",
    "lgk.walks.nb_runtime",                       # copied verbatim into the walks notebook
    "lgk.templates",                              # package data (graph_mapping.yaml)
    "lgk.loaders", "lgk.loaders.mondo",           # operator-called loader (load_mondo)
    "lgk.loaders.scale_graph",                    # operator-called loader (build_scale_graph)
    "lgk.reaper", "lgk.steps.retire",             # called directly (they need the typed graph name)
    "lgk.steps.notebook",                         # runner step, resolved by name
    "lgk.metrics",                                # live-test numbers recorder: conftest + tools/metrics_summary.py
    "lgk.refresh.lag",                            # freshness-lag reader of the recorder (test_refresh_live)
    "lgk.workspace_latency",                      # in-region latency step, resolved by name in the notebook
}


def _module_name(p: Path) -> str:
    parts = list(p.relative_to(SRC.parent).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _referenced_modules() -> set:
    refs = set()
    for p in SRC.rglob("*.py"):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                refs.add(node.module)
                refs.update(f"{node.module}.{a.name}" for a in node.names)
            elif isinstance(node, ast.Import):
                refs.update(a.name for a in node.names)
    text = "\n".join(p.read_text() for p in [*SRC.rglob("*.py"), *ROOT.glob("skill/**/*.md"),
                                               ROOT / "pyproject.toml"] if p.is_file())
    refs.update(re.findall(r"\blgk(?:\.[a-z_0-9]+)+", text))
    return refs


def test_every_src_module_is_reached():
    refs = _referenced_modules()
    dead = sorted(m for m in map(_module_name, SRC.rglob("*.py"))
                  if m not in ENTRY_POINTS and m not in refs
                  and not any(r.startswith(m + ".") for r in refs))
    assert dead == []
