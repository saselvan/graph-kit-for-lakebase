"""Seam-2: the graph_mapping.yaml contract.

One contract that `propose` writes, the interview edits and `build` reads:
- the loader rejects a mapping that breaks the contract, with a message naming the field;
- `build` rejects such a mapping before it runs any SQL;
- `propose` output round-trips through the loader (front feeds back).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from lgk.mapping import EdgeSpec, GraphMapping, MappingError, NodeSpec, dump_mapping, load_mapping
from lgk.steps.propose import propose

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "src" / "lgk" / "templates" / "graph_mapping.yaml"


def _good() -> dict:
    return {
        "source_catalog": "cat",
        "source_schema": "gold",
        "nodes": {
            "provider": {
                "source_table": "providers",
                "natural_key": "provider_id",
                "label": "name",
                "properties": ["specialty"],
            },
            "org": {"source_table": "organizations", "natural_key": "org_id", "label": "org_name"},
        },
        "edges": {
            "affiliation": {
                "source_table": "affiliations",
                "src": "provider_id",
                "src_type": "provider",
                "dst": "org_id",
                "dst_type": "org",
            }
        },
    }


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "graph_mapping.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


# ----------------------------------------------------------------- loader accepts the contract
def test_good_mapping_loads_typed(tmp_path):
    m = load_mapping(_write(tmp_path, _good()))
    assert m.source_catalog == "cat" and m.source_schema == "gold"
    assert m.nodes["provider"] == NodeSpec("providers", "provider_id", "name", ["specialty"])
    assert m.nodes["org"].properties == []  # optional field defaults to empty
    assert m.edges["affiliation"] == EdgeSpec("affiliations", "provider_id", "provider", "org_id", "org")


def test_committed_template_satisfies_the_contract():
    m = load_mapping(TEMPLATE)
    assert set(m.nodes) == {"entity_a", "entity_b"}
    assert m.edges["edge_type_1"].dst_type == "entity_b"


def test_dump_then_load_round_trips(tmp_path):
    m = load_mapping(_write(tmp_path, _good()))
    out = tmp_path / "again.yaml"
    dump_mapping(m, out)
    assert load_mapping(out) == m


# ------------------------------------------------------- loader rejects, naming the field
def _mutated(path: str, value=None, delete=False) -> dict:
    data = _good()
    *parents, leaf = path.split(".")
    node = data
    for key in parents:
        node = node[key]
    if delete:
        del node[leaf]
    else:
        node[leaf] = value
    return data


def _renamed(section: str, old: str, new: str) -> dict:
    data = _good()
    data[section][new] = data[section].pop(old)
    if section == "nodes":  # keep edges pointing at the renamed type, so only the name is bad
        for e in data["edges"].values():
            for side in ("src_type", "dst_type"):
                if e[side] == old:
                    e[side] = new
    return data


INJECT = "x' OR '1'='1"  # hostile: a quote breaks out of the SQL literal build splices it into

# Every field that reaches build's SQL, each with a hostile value and the dotted path the
# message must name. Removing any single check must turn its row red.
BAD = [
    # top level
    (_mutated("source_catalog", ""), "source_catalog"),
    (_mutated("source_catalog", INJECT), "source_catalog"),
    (_mutated("source_schema", delete=True), "source_schema"),
    (_mutated("source_schema", "gold; DROP SCHEMA x"), "source_schema"),
    (_mutated("extra_top_level", 1), "extra_top_level"),
    # node type name (spliced as a SQL string literal)
    (_renamed("nodes", "provider", INJECT), f"nodes.{INJECT}"),
    # node fields
    (_mutated("nodes.provider.source_table", "providers p JOIN secrets s"), "nodes.provider.source_table"),
    (_mutated("nodes.provider.natural_key", delete=True), "nodes.provider.natural_key"),
    (_mutated("nodes.provider.natural_key", "provider_id || x"), "nodes.provider.natural_key"),
    (_mutated("nodes.provider.natural_key", []), "nodes.provider.natural_key"),
    (_mutated("nodes.provider.natural_key", ["provider_id", "1); --"]), "nodes.provider.natural_key[1]"),
    (_mutated("nodes.provider.label", "name FROM x --"), "nodes.provider.label"),
    (_mutated("nodes.provider.properties", "specialty"), "nodes.provider.properties"),
    (_mutated("nodes.provider.properties", ["specialty", INJECT]), "nodes.provider.properties[1]"),
    # hostile: a renamed contract field (what "rename a field in propose only" produces)
    (_mutated("nodes.provider.natural_keys", "provider_id"), "nodes.provider.natural_keys"),
    # edge type name (spliced as a SQL string literal)
    (_renamed("edges", "affiliation", INJECT), f"edges.{INJECT}"),
    # edge fields
    (_mutated("edges.affiliation.weight", "w"), "edges.affiliation.weight"),
    (_mutated("edges.affiliation.dst", delete=True), "edges.affiliation.dst"),
    (_mutated("edges.affiliation.source_table", "affiliations a, secrets"), "edges.affiliation.source_table"),
    (_mutated("edges.affiliation.src", "provider_id; DROP TABLE x"), "edges.affiliation.src"),
    (_mutated("edges.affiliation.dst", "org_id)--"), "edges.affiliation.dst"),
    (_mutated("edges.affiliation.src_type", INJECT), "edges.affiliation.src_type"),
    # hostile: a non-string (unhashable) type ref must be a MappingError, not a TypeError
    (_mutated("edges.affiliation.src_type", ["provider"]), "edges.affiliation.src_type"),
    (_mutated("edges.affiliation.dst_type", ["org"]), "edges.affiliation.dst_type"),
    # hostile: case-only typo of a declared node type
    (_mutated("edges.affiliation.dst_type", "Org"), "edges.affiliation.dst_type"),
    # hostile: edge into a composite-keyed node with a single dst column (would drop every edge)
    (_mutated("nodes.org.natural_key", ["org_id", "region"]), "edges.affiliation.dst_type"),
]


def test_edgeless_mapping_is_valid(tmp_path):
    data = _good()
    del data["edges"]
    m = load_mapping(_write(tmp_path, data))
    assert m.edges == {} and set(m.nodes) == {"provider", "org"}


@pytest.mark.parametrize("data,field", BAD, ids=[f for _, f in BAD])
def test_bad_mapping_rejected_naming_the_field(tmp_path, data, field):
    with pytest.raises(MappingError) as exc:
        load_mapping(_write(tmp_path, data))
    assert f"'{field}'" in str(exc.value)


def test_old_propose_form_rejected(tmp_path):
    # hostile: the exact older propose output (nodes keyed by table, edges as a list)
    old = {
        "nodes": {"providers": {"key_columns": ["provider_id"], "type": "entity"}},
        "edges": [{"from": "a", "to": "b", "from_key": "x", "to_key": "y", "type": "references"}],
    }
    with pytest.raises(MappingError, match="'source_catalog'"):
        load_mapping(_write(tmp_path, old))


def test_build_rejects_bad_mapping_before_any_sql(tmp_path, monkeypatch):
    w = MagicMock()
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: w)
    _write(tmp_path, _mutated("edges.affiliation.src_type", "providers"))
    from lgk.steps.build import build

    with pytest.raises(MappingError, match="'edges.affiliation.src_type'"):
        build({"run_folder": str(tmp_path), "lakebase_catalog": "c", "kit_schema": "k", "warehouse_id": "wh"})
    w.statement_execution.execute_statement.assert_not_called()


# --------------------------------------------------- propose output round-trips the loader
def _schema_json(tmp_path: Path) -> None:
    def col(name, type_name="STRING"):
        return {"name": name, "type": type_name, "match_rate": 1.0, "entropy": None,
                "distinct_ratio": 1.0, "null_rate": 0.0}

    tables = [
        # hostile: the join table comes FIRST, before the tables it references (reordered input)
        {
            "name": "affiliations",
            "key_columns": [],
            "row_count": 10,
            "columns": [col("provider_id"), col("org_id"), col("since")],
            "foreign_keys": [
                {"columns": ["provider_id"], "references": ["providers", "provider_id"]},
                {"columns": ["org_id"], "references": ["organizations", "org_id"]},
            ],
        },
        {
            "name": "providers",
            "key_columns": ["provider_id"],
            "row_count": 6,
            # hostile: an INT column before the string label, so "first non-key column" is wrong
            "columns": [col("provider_id"), col("years", "INT"), col("name"), col("specialty")],
            "foreign_keys": [],
        },
        {
            "name": "organizations",
            "key_columns": ["org_id"],
            "row_count": 3,
            "columns": [col("org_id"), col("org_name")],
            "foreign_keys": [],
        },
        {
            "name": "referrals",
            "key_columns": ["referral_id"],
            "row_count": 4,
            "columns": [col("referral_id"), col("provider_id"), col("note")],
            # hostile: the same FK declared twice must give ONE edge, not a type collision
            "foreign_keys": [
                {"columns": ["provider_id"], "references": ["providers", "provider_id"]},
                {"columns": ["provider_id"], "references": ["providers", "provider_id"]},
            ],
        },
    ]
    (tmp_path / "schema.json").write_text(
        json.dumps({"source_catalog": "cat", "source_schema": "gold", "tables": tables})
    )


def test_propose_output_round_trips_through_the_loader(tmp_path):
    _schema_json(tmp_path)
    propose({"graph_name": "g", "run_folder": str(tmp_path)})

    m = load_mapping(tmp_path / "graph_mapping.yaml")
    assert m == GraphMapping(
        source_catalog="cat",
        source_schema="gold",
        nodes={
            "providers": NodeSpec("providers", "provider_id", "name", ["years", "specialty"]),
            "organizations": NodeSpec("organizations", "org_id", "org_name", []),
            "referrals": NodeSpec("referrals", "referral_id", "note", []),
        },
        edges={
            "affiliations": EdgeSpec("affiliations", "provider_id", "providers", "org_id", "organizations"),
            "referrals_provider_id": EdgeSpec(
                "referrals", "referral_id", "referrals", "provider_id", "providers"
            ),
        },
    )
    # The written file is exactly what the loader reads back (nothing propose-only in it).
    dumped = tmp_path / "dumped.yaml"
    dump_mapping(m, dumped)
    assert yaml.safe_load(dumped.read_text()) == yaml.safe_load(
        (tmp_path / "graph_mapping.yaml").read_text()
    )


def test_propose_takes_source_from_config_when_schema_json_has_none(tmp_path):
    (tmp_path / "schema.json").write_text(
        json.dumps({"tables": [{"name": "t", "key_columns": ["id"]}]})
    )
    propose({"run_folder": str(tmp_path), "source_catalog": "c2", "source_schema": "s2"})
    m = load_mapping(tmp_path / "graph_mapping.yaml")
    assert (m.source_catalog, m.source_schema) == ("c2", "s2")
    assert m.nodes["t"] == NodeSpec("t", "id", "id", [])  # no string column -> label is the key


def test_propose_composite_key_round_trips(tmp_path):
    # hostile: composite key order must survive (line_no before claim_id would hash differently)
    (tmp_path / "schema.json").write_text(
        json.dumps({"source_catalog": "c", "source_schema": "s",
                    "tables": [{"name": "claims", "key_columns": ["claim_id", "line_no"]}]})
    )
    propose({"run_folder": str(tmp_path)})
    m = load_mapping(tmp_path / "graph_mapping.yaml")
    assert m.nodes["claims"].natural_key == ["claim_id", "line_no"]


def test_propose_without_any_source_names_the_field(tmp_path):
    (tmp_path / "schema.json").write_text(json.dumps({"tables": []}))
    with pytest.raises(MappingError, match="'source_catalog'"):
        propose({"run_folder": str(tmp_path)})


def test_keyless_table_with_one_fk_declared_twice_is_refused_not_a_join(tmp_path):
    # hostile: the duplicate makes it LOOK like a two-FK join table (a providers self-join)
    dup = {"columns": ["provider_id"], "references": ["providers", "provider_id"]}
    (tmp_path / "schema.json").write_text(json.dumps({
        "source_catalog": "c", "source_schema": "s",
        "tables": [{"name": "providers", "key_columns": ["provider_id"]},
                   {"name": "notes", "key_columns": [], "foreign_keys": [dup, dup]}],
    }))
    with pytest.raises(ValueError, match="source table notes has no primary key"):
        propose({"run_folder": str(tmp_path)})


def test_declared_profile_carries_foreign_keys_to_a_proposed_edge(tmp_path):
    """Offline front: declared gold_tables.json -> profile -> propose -> loader has the edge."""
    from lgk.steps.profile import profile

    (tmp_path / "gold_tables.json").write_text(json.dumps({"tables": [
        {"name": "orgs", "key_columns": ["org_id"]},
        {"name": "staff", "key_columns": ["staff_id"],
         "foreign_keys": [{"columns": ["org_id"], "references": ["orgs", "org_id"]}]},
    ]}))
    cfg = {"run_folder": str(tmp_path), "source_catalog": "c", "source_schema": "s"}
    profile(cfg)
    propose(cfg)
    m = load_mapping(tmp_path / "graph_mapping.yaml")
    assert m.edges == {"staff_org_id": EdgeSpec("staff", "staff_id", "staff", "org_id", "orgs")}


def test_propose_validates_before_writing(tmp_path):
    # hostile: a table name that is not a SQL identifier becomes a node type name
    (tmp_path / "schema.json").write_text(json.dumps({
        "source_catalog": "c", "source_schema": "s",
        "tables": [{"name": "bad-name", "key_columns": ["id"]}],
    }))
    with pytest.raises(MappingError, match="'nodes.bad-name'"):
        propose({"run_folder": str(tmp_path)})
    assert not (tmp_path / "graph_mapping.yaml").exists()


def test_build_sql_semi_joins_both_ends_and_keeps_composite_key_order(tmp_path, monkeypatch):
    statements = []
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: MagicMock())
    monkeypatch.setattr("lgk.steps.build.execute_sql", lambda w, wh, sql, **k: statements.append(sql) or [])
    from tests.label_support import stub_table_identities

    stub_table_identities(monkeypatch)  # the recording warehouse has no Delta history
    data = _good()
    # hostile: key order line_no-then-claim_id is NOT alphabetical, so a sorted() would show
    data["nodes"]["claim"] = {"source_table": "claims", "natural_key": ["line_no", "claim_id"], "label": "claim_id"}
    _write(tmp_path, data)
    # The kit-schema claim is tested in test_teardown.py; here only the build SQL matters.
    monkeypatch.setattr("lgk.steps.build._claim_kit_schema", lambda *a, **k: None)
    from lgk.steps.build import build
    from tests.privacy_support import clear_gate_offline

    clear_gate_offline(str(tmp_path))
    build({"graph_name": "g", "run_folder": str(tmp_path), "lakebase_catalog": "c", "kit_schema": "k", "warehouse_id": "wh"})
    sql = "\n".join(statements)
    assert "to_json(array(CAST(n.line_no AS STRING), CAST(n.claim_id AS STRING)))" in sql
    assert "xxhash64('provider::'||n.provider_id)" in sql  # single-key formula unchanged (alias n., node joins)
    (edge_insert,) = [s for s in statements if "INSERT INTO c.k.edges" in s]
    assert "LEFT SEMI JOIN c.k.nodes s ON e.src_node_id = s.node_id" in edge_insert
    assert "LEFT SEMI JOIN c.k.nodes d ON e.dst_node_id = d.node_id" in edge_insert


# ------------------------------------------------------------- node joins -
def _with_joins(*joins, **node) -> dict:
    data = _good()
    data["nodes"]["provider"].update(node)
    data["nodes"]["provider"]["joins"] = list(joins)
    return data


_AGG_JOIN = {"table": "claims", "key": "provider_id", "subject": "patient_id",
             "aggregates": ["sum(billed_amount) as total_billed", "count(claim_id) as claim_count"]}
_COL_JOIN = {"table": "provider_taxonomy", "key": "provider_id", "columns": ["taxonomy_code"]}


def test_joins_load_typed_and_round_trip(tmp_path):
    from lgk.mapping import Aggregate, JoinSpec

    m = load_mapping(_write(tmp_path, _with_joins(_AGG_JOIN, _COL_JOIN)))
    assert m.nodes["provider"].joins == [
        JoinSpec("claims", "provider_id", [], [Aggregate("sum", "billed_amount", "total_billed"),
                                               Aggregate("count", "claim_id", "claim_count")], "patient_id"),
        JoinSpec("provider_taxonomy", "provider_id", ["taxonomy_code"], [], None),
    ]
    out = tmp_path / "again.yaml"
    dump_mapping(m, out)
    assert load_mapping(out) == m
    assert yaml.safe_load(out.read_text())["nodes"]["provider"]["joins"][0] == _AGG_JOIN


def test_a_mapping_without_joins_dumps_no_joins_key(tmp_path):
    out = tmp_path / "again.yaml"
    dump_mapping(load_mapping(_write(tmp_path, _good())), out)
    assert "joins" not in out.read_text()


@pytest.mark.parametrize("agg", [
    "avg(x) as y",             # outside the closed list
    "first(x) as y", "last(x) as y", "any_value(x) as y", "collect_list(x) as y",
    "collect_set(x) as y", "percentile_approx(x) as y", "approx_count_distinct(x) as y",
    "rand(x) as y",
    "sum(x+y) as z",           # an expression
    "sum(x)",                  # no `as`
    "sum(x) AS y",             # the grammar is exact
    "sum( x ) as y",
    "count(*) as n",
    "sum(x) as y; DROP TABLE t",
    7,
])
def test_non_deterministic_aggregate_is_refused(tmp_path, agg):
    join = {**_AGG_JOIN, "aggregates": ["count(claim_id) as claim_count", agg]}
    with pytest.raises(MappingError, match=r"'nodes\.provider\.joins\[0\]\.aggregates\[1\]'"):
        load_mapping(_write(tmp_path, _with_joins(join)))


@pytest.mark.parametrize("func", ["sum", "min", "max", "count", "count_distinct"])
def test_each_listed_aggregate_parses(tmp_path, func):
    join = {**_AGG_JOIN, "aggregates": [f"{func}(billed_amount) as out_{func}"]}
    (j,) = load_mapping(_write(tmp_path, _with_joins(join))).nodes["provider"].joins
    assert (j.aggregates[0].func, j.aggregates[0].column) == (func, "billed_amount")


def test_an_upper_case_aggregate_name_is_refused_by_the_identifier_rule(tmp_path):
    join = {**_AGG_JOIN, "aggregates": ["sum(Billed) as total"]}
    with pytest.raises(MappingError, match=r"aggregates\[0\].*lower case"):
        load_mapping(_write(tmp_path, _with_joins(join)))


@pytest.mark.parametrize("join,path,why", [
    ({**_COL_JOIN, "columns": ["specialty"]}, r"joins\[0\]\.columns\[0\]", "native property"),
    ({**_COL_JOIN, "columns": ["name"]}, r"joins\[0\]\.columns\[0\]", "the label"),
    ({**_AGG_JOIN, "aggregates": ["count(claim_id) as node_id"]}, r"joins\[0\]\.aggregates\[0\]", "nodes column"),
    ({**_AGG_JOIN, "aggregates": ["count(claim_id) as properties"]}, r"joins\[0\]\.aggregates\[0\]", "nodes column"),
    ({**_AGG_JOIN, "aggregates": ["count(claim_id) as natural_key"]}, r"joins\[0\]\.aggregates\[0\]", "nodes column"),
    ({**_AGG_JOIN, "aggregates": ["count(claim_id) as node_type"]}, r"joins\[0\]\.aggregates\[0\]", "nodes column"),
    ({**_AGG_JOIN, "aggregates": ["count(claim_id) as label"]}, r"joins\[0\]\.aggregates\[0\]", "nodes column"),
    ({**_AGG_JOIN, "aggregates": ["count(claim_id) as n", "max(claim_id) as n"]},
     r"joins\[0\]\.aggregates\[1\]", "output of"),
])
def test_output_name_collisions_are_refused(tmp_path, join, path, why):
    with pytest.raises(MappingError, match=rf"'nodes\.provider\.{path}'.*{why}"):
        load_mapping(_write(tmp_path, _with_joins(join)))


def test_an_output_colliding_with_another_joins_output_is_refused(tmp_path):
    # hostile: the collision is across joins (the second join's column = the first's `as` name)
    second = {**_COL_JOIN, "columns": ["claim_count"]}
    with pytest.raises(MappingError, match=r"'nodes\.provider\.joins\[1\]\.columns\[0\]'.*joins\[0\]\.aggregates\[1\]"):
        load_mapping(_write(tmp_path, _with_joins(_AGG_JOIN, second)))


@pytest.mark.parametrize("where", ["table", "key", "subject", "columns", "aggregates", "label", "properties"])
def test_the_lgk_prefix_is_reserved(tmp_path, where):
    data = {"table": _with_joins({**_AGG_JOIN, "table": "_lgk_x"}),
            "key": _with_joins({**_AGG_JOIN, "key": "_lgk_x"}),
            "subject": _with_joins({**_AGG_JOIN, "subject": "_lgk_x"}),
            "columns": _with_joins({**_COL_JOIN, "columns": ["_lgk_x"]}),
            "aggregates": _with_joins({**_AGG_JOIN, "aggregates": ["count(claim_id) as _lgk_subjects"]}),
            "label": _with_joins(label="_lgk_x"),
            "properties": _with_joins(properties=["_lgk_x"])}[where]
    with pytest.raises(MappingError, match=r"_lgk_.*reserved"):
        load_mapping(_write(tmp_path, data))


@pytest.mark.parametrize("join,path", [
    ({"table": "claims", "key": "provider_id"}, r"joins\[0\]"),                       # neither
    ({**_COL_JOIN, "aggregates": ["count(x) as y"]}, r"joins\[0\]"),                  # both
    ({**_COL_JOIN, "columns": []}, r"joins\[0\]\.columns"),                           # empty
    ({**_COL_JOIN, "subject": "patient_id"}, r"joins\[0\]\.subject"),                 # subject on columns
    ({**_COL_JOIN, "where": "x"}, r"joins\[0\]\.where"),                              # unknown key
    ({"table": "claims", "columns": ["x"]}, r"joins\[0\]\.key"),                      # key required
    ({**_COL_JOIN, "table": "a b"}, r"joins\[0\]\.table"),
    ({**_COL_JOIN, "columns": [f"c{i}" for i in range(17)]}, r"joins\[0\]\.columns"),  # > 16 outputs
])
def test_bad_join_is_refused_naming_the_path(tmp_path, join, path):
    with pytest.raises(MappingError, match=rf"'nodes\.provider\.{path}'"):
        load_mapping(_write(tmp_path, _with_joins(join)))


def test_sixteen_outputs_and_eight_joins_are_the_bounds(tmp_path):
    joins = [{"table": f"t{i}", "key": "provider_id", "columns": [f"c{i}_{k}" for k in range(16)]}
             for i in range(8)]
    assert len(load_mapping(_write(tmp_path, _with_joins(*joins))).nodes["provider"].joins) == 8
    with pytest.raises(MappingError, match=r"'nodes\.provider\.joins' has 9 joins"):
        load_mapping(_write(tmp_path, _with_joins(*joins, {**_COL_JOIN, "columns": ["z"]})))


def test_joins_on_a_composite_natural_key_are_refused(tmp_path):
    data = _with_joins(_COL_JOIN, natural_key=["provider_id", "site"])
    with pytest.raises(MappingError, match=r"'nodes\.provider\.joins'.*single-column natural_key"):
        load_mapping(_write(tmp_path, data))
