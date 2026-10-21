"""The scale graph generator: a synthetic graph with invented names, built with SQL only.

Offline: `execute_sql` is replaced by a recorder, so no statement reaches a warehouse.
The tables declare their keys, so the kit's own profile and
propose steps, run over a fake warehouse built from the generator's SQL, draft the generator's mapping.
"""

import re

import pytest
import yaml

import lgk.steps.profile as profile_step
from lgk import ident, keys, mapping
from lgk.loaders import scale_graph as sg
from lgk.steps.propose import propose

CAT, SCH = "main", "lgk_scale_t1"
FQ = f"{CAT}.{SCH}"

# The round values the generator must use: counts per node type, referral edges, and the
# percent of providers with a second practice.
ROUND_COUNTS = {"provider": 800_000, "practice": 120_000, "location": 50_000, "system": 20_000,
                "market": 9_000, "network": 1_000}
ROUND_REFERRALS = 250_000
ROUND_SECOND_PRACTICE_PCT = 25
# Each node type is named after its table, as propose names it.
NODE_TABLE = {"provider": "providers", "practice": "practices", "location": "locations", "system": "systems",
              "market": "markets", "network": "networks"}


class Recorder:
    """A fake warehouse: records each statement; answers the schema lookup with `schemata` rows."""

    def __init__(self, schemata=()):
        self.statements = []
        self.schemata = [list(r) for r in schemata]

    def __call__(self, w, warehouse_id, statement, *a, **k):
        self.statements.append(statement)
        if "information_schema.schemata" in statement:
            return self.schemata
        return []

    def creates(self):
        return [s for s in self.statements if s.lstrip().upper().startswith("CREATE")]

    def drops(self):
        return [s for s in self.statements if s.lstrip().upper().startswith("DROP")]


@pytest.fixture
def rec(monkeypatch):
    r = Recorder()
    monkeypatch.setattr(sg, "execute_sql", r)
    return r


def _tables(statements):
    """{table: its CREATE statement and its INSERT statement} for each table the SQL creates in FQ."""
    out = {}
    for s in statements:
        m = re.match(rf"\s*CREATE TABLE {re.escape(FQ)}\.(\w+)\b", s)
        if m:
            assert m.group(1) not in out, f"table {m.group(1)} created twice"
            out[m.group(1)] = s
    for s in statements:
        m = re.match(rf"\s*INSERT INTO {re.escape(FQ)}\.(\w+)\b", s)
        if m:
            assert m.group(1) in out, f"insert into {m.group(1)}, which no statement creates"
            out[m.group(1)] += "\n" + s
    return out


def _mapping():
    return yaml.safe_load(sg.mapping_yaml(CAT, SCH))


def _edge_table(data, src_type, dst_type):
    tables = [e["source_table"] for e in data["edges"].values()
              if e["src_type"] == src_type and e["dst_type"] == dst_type]
    assert len(tables) == 1, f"{src_type}->{dst_type} edges: {tables}"
    return tables[0]


# --- the values

def test_the_counts_and_ratios_are_the_round_values():
    assert sg.COUNTS == ROUND_COUNTS
    assert sg.N_REFERRALS == ROUND_REFERRALS
    assert sg.SECOND_PRACTICE_PCT == ROUND_SECOND_PRACTICE_PCT


@pytest.mark.parametrize("scale, total", [(1.0, 1_000_000), (0.5, 500_000), (2.0, 2_000_000),
                                          (0.001, 1_000)])
def test_the_node_count_is_a_million_at_the_default_scale_and_grows_linearly(scale, total):
    counts = sg.node_counts(scale)
    assert sum(counts.values()) == total
    assert counts == {k: round(v * scale) for k, v in ROUND_COUNTS.items()}
    assert sg.node_counts() == ROUND_COUNTS


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_the_sql_makes_each_node_table_from_its_count(scale):
    data = _mapping()
    tables = _tables(sg.build_sql(CAT, SCH, scale))
    counts = sg.node_counts(scale)
    assert set(data["nodes"]) == set(NODE_TABLE.values())
    for kind, table in NODE_TABLE.items():
        assert data["nodes"][table]["source_table"] == table
        assert f"range({counts[kind]})" in tables[table], kind
    referrals = tables[_edge_table(data, "providers", "providers")]
    assert f"range({round(ROUND_REFERRALS * scale)})" in referrals


def test_the_second_practice_share_is_the_round_percent():
    data = _mapping()
    members = _tables(sg.build_sql(CAT, SCH))[_edge_table(data, "providers", "practices")]
    assert re.search(rf"\b100\)\s*<\s*{ROUND_SECOND_PRACTICE_PCT}\b", members), members


# --- the SQL

def test_two_calls_give_the_same_sql_and_nothing_random():
    first = sg.build_sql(CAT, SCH, 1.0)
    assert first == sg.build_sql(CAT, SCH, 1.0)
    text = "\n".join(first).lower()
    assert "hash(" in text and "range(" in text
    for word in ("rand", "uuid", "shuffle", "current_timestamp", "current_date", "now("):
        assert word not in text, word


def test_the_sql_makes_a_new_schema_and_never_replaces_anything():
    statements = sg.build_sql(CAT, SCH)
    assert re.fullmatch(rf"CREATE SCHEMA {re.escape(FQ)}\b.*", statements[0], re.S), statements[0]
    for s in statements:
        assert "OR REPLACE" not in s.upper(), s
        assert "IF NOT EXISTS" not in s.upper(), s
        assert not s.lstrip().upper().startswith("DROP"), s


@pytest.mark.parametrize("catalog, schema", [("main", "x; DROP SCHEMA main.y"), ("main", "Lgk"),
                                             ("ma`in", "lgk_t"), ("main", "")])
def test_a_bad_name_is_refused_before_any_sql(rec, tmp_path, catalog, schema):
    with pytest.raises(ident.IdentifierError):
        sg.build_scale_graph(None, "wh", catalog, schema, tmp_path / "graph_mapping.yaml")
    assert rec.statements == []


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), True, "1"])
def test_a_bad_scale_is_refused_before_any_sql(rec, tmp_path, scale):
    with pytest.raises(sg.ScaleGraphError):
        sg.build_sql(CAT, SCH, scale)
    with pytest.raises(sg.ScaleGraphError):
        sg.build_scale_graph(None, "wh", CAT, SCH, tmp_path / "graph_mapping.yaml", scale)
    assert rec.statements == []


# --- the mapping

def test_the_mapping_is_valid_and_names_only_what_the_sql_creates():
    data = _mapping()
    parsed = mapping.parse_mapping(data)
    assert (parsed.source_catalog, parsed.source_schema) == (CAT, SCH)
    tables = _tables(sg.build_sql(CAT, SCH))
    used = set()
    for name, spec in data["nodes"].items():
        sql = tables[spec["source_table"]]
        for col in [spec["natural_key"], spec["label"], *spec.get("properties", [])]:
            assert re.search(rf"\bAS {col}\b", sql), f"nodes.{name}: {col}"
        used.add(spec["source_table"])
    for name, spec in data["edges"].items():
        sql = tables[spec["source_table"]]
        for col in (spec["src"], spec["dst"]):
            assert re.search(rf"\bAS {col}\b", sql), f"edges.{name}: {col}"
        used.add(spec["source_table"])
    assert used == set(tables)


def test_the_mapping_links_every_node_type_into_one_hierarchy():
    pairs = sorted((e["src_type"], e["dst_type"]) for e in _mapping()["edges"].values())
    assert pairs == sorted([("providers", "practices"), ("practices", "systems"), ("practices", "locations"),
                            ("systems", "markets"), ("markets", "networks"), ("providers", "providers")])


def test_every_link_column_the_sql_makes_is_an_edge_end():
    data = _mapping()
    for table, sql in _tables(sg.build_sql(CAT, SCH)).items():
        used = {spec["natural_key"] for spec in data["nodes"].values() if spec["source_table"] == table}
        for spec in data["edges"].values():
            if spec["source_table"] == table:
                used |= {spec["src"], spec["dst"]}
        made = set(re.findall(r"\bAS (\w+_id)\b", sql))
        assert made <= used, f"{table}: link columns with no edge: {sorted(made - used)}"


# --- the declared keys

def _flat(text):
    """`text` with every parenthesised part replaced by <>, so only the top level is left."""
    while re.search(r"\([^()]*\)", text):
        text = re.sub(r"\([^()]*\)", "<>", text)
    return text


def _split(body):
    """Split a column list at its top-level commas."""
    parts, depth, cur = [], 0, ""
    for ch in body:
        depth += (ch == "(") - (ch == ")")
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    return parts + [cur.strip()]


def _declared(statements):
    """{table: {"columns": [(name, type, not_null)], "pk": [cols], "fks": [(name, col, ref_table, ref_col)],
    "insert": text}} from the CREATE TABLE column lists (the lgk.loaders.mondo form) and the INSERTs.

    A table with no column list (a CREATE TABLE ... AS SELECT) fails here: its keys are not declared."""
    out = {}
    for s in statements:
        m = re.fullmatch(rf"\s*CREATE TABLE {re.escape(FQ)}\.(\w+)\s*\((.*)\)\s*USING DELTA\s*", s, re.S)
        if not m:
            assert not re.match(r"\s*CREATE TABLE", s), f"no declared column list: {s[:80]}"
            continue
        table, cols, pk, fks = m.group(1), [], [], []
        for part in _split(m.group(2)):
            p = " ".join(part.split())
            if (k := re.fullmatch(r"CONSTRAINT (\w+) PRIMARY KEY \((\w+)\)", p)):
                assert not pk, f"{table}: two primary keys"
                pk = [k.group(2)]
            elif (k := re.fullmatch(rf"CONSTRAINT (\w+) FOREIGN KEY \((\w+)\) REFERENCES {re.escape(FQ)}\.(\w+)"
                                    r"(?: \((\w+)\))?", p)):
                fks.append((k.group(1), k.group(2), k.group(3), k.group(4)))
            elif (k := re.fullmatch(r"(\w+) (BIGINT|STRING)( NOT NULL)?", p)):
                cols.append((k.group(1), k.group(2), bool(k.group(3))))
            else:
                raise AssertionError(f"{table}: a column list part this test does not read: {p}")
        out[table] = {"columns": cols, "pk": pk, "fks": fks, "insert": None}
    for s in statements:
        m = re.match(rf"\s*INSERT INTO {re.escape(FQ)}\.(\w+)\s+(SELECT\b.*)", s, re.S)
        if m:
            assert out[m.group(1)]["insert"] is None, f"{m.group(1)}: two inserts"
            out[m.group(1)]["insert"] = m.group(2)
    for table, d in out.items():  # a REFERENCES with no column names the parent's primary key
        d["fks"] = [(n, c, rt, rc or (out[rt]["pk"] or [None])[0]) for n, c, rt, rc in d["fks"]]
    return out


def _select_names(select):
    """The output column names of each top-level SELECT of an INSERT, in order."""
    lists = []
    for branch in re.split(r"\bUNION ALL\b", _flat(select)):
        m = re.search(r"\bSELECT\b(.*?)\bFROM\b", branch, re.S)
        assert m, branch
        names = []
        for item in m.group(1).split(","):
            a = re.search(r"\bAS (\w+)\s*$", item.strip()) or re.fullmatch(r"(\w+)", item.strip())
            assert a, f"a select item with no name: {item.strip()}"
            names.append(a.group(1))
        lists.append(names)
    return lists


def test_each_table_declares_its_columns_and_keys_and_its_insert_fills_them_in_order():
    tables = _declared(sg.build_sql(CAT, SCH))
    assert set(tables) == set(_tables(sg.build_sql(CAT, SCH)))
    for table, d in tables.items():
        names = [c for c, _, _ in d["columns"]]
        assert d["insert"], f"{table}: no INSERT fills it"
        for listed in _select_names(d["insert"]):  # INSERT ... SELECT fills columns by position
            assert listed == names, f"{table}: insert order {listed} != declared {names}"
        not_null = {c for c, _, nn in d["columns"] if nn}
        assert set(d["pk"]) <= not_null, f"{table}: a primary key column is not declared NOT NULL"
        for name, col, ref_table, ref_col in d["fks"]:
            assert col in names, f"{table}.{name}: no column {col}"
            assert ref_table in tables and tables[ref_table]["pk"] == [ref_col], f"{table}.{name}: not a key"
    for kind, table in NODE_TABLE.items():
        assert tables[table]["pk"] == [f"{kind}_id"], table
    order = list(tables)  # creation order: a foreign key needs its parent table to exist
    for table, d in tables.items():
        for name, _, ref_table, _ in d["fks"]:
            assert order.index(ref_table) < order.index(table), f"{table}.{name}: parent made later"


def test_each_constraint_name_is_used_once_in_the_schema():
    # profile joins information_schema on (schema, constraint name), so a repeated name mixes two keys
    tables = _declared(sg.build_sql(CAT, SCH))
    names = [n for d in tables.values() for n, *_ in d["fks"]]
    for statement in sg.build_sql(CAT, SCH):
        names += re.findall(r"CONSTRAINT (\w+) PRIMARY KEY", statement)
    assert len(names) == len(set(names)) and len(names) == len(NODE_TABLE) + 8, sorted(names)  # 6 keys, 8 links


class Warehouse:
    """A fake warehouse holding the tables the generator's SQL declares: it answers profile's catalog,
    key, count and sample reads from that SQL, and refuses any other statement."""

    ROWS = 100

    def __init__(self, statements):
        self.tables = _declared(statements)

    def _sample(self, table, cols):
        """String values as the INSERT makes them (CONCAT('<word> ', i)); a column it does not make
        that way gets '<column> <i>'."""
        made = dict((c, w) for w, c in re.findall(r"CONCAT\('([^']*)', i\) AS (\w+)", self.tables[table]["insert"]))
        return [[f"{made[c]}{i}" if c in made else f"{c} {i}" for c in cols] for i in range(self.ROWS)]

    def __call__(self, w, warehouse_id, statement, *a, **k):
        s = " ".join(statement.split())
        t = self.tables
        if "information_schema.columns" in s:
            return [[n, c, ty] for n in sorted(t) for c, ty, _ in t[n]["columns"]]
        if "constraint_type = 'PRIMARY KEY'" in s:
            return [[n, c] for n in sorted(t) for c in t[n]["pk"]]
        if "referential_constraints" in s:
            return [[n, fk, c, rt, rc] for n in sorted(t) for fk, c, rt, rc in sorted(t[n]["fks"])]
        for n, d in t.items():
            fq = f"{FQ}.{n}"
            if d["pk"] and statement == keys.duplicate_key_sql(fq, d["pk"]):
                return [[0, 0]]
            for _, c, rt, rc in d["fks"]:
                if statement == keys.fk_orphan_sql(fq, [c], f"{FQ}.{rt}", [rc]):
                    return [[0]]
            if re.fullmatch(rf"SELECT COUNT\(\*\)(, COUNT\((DISTINCT )?`\w+`\))* FROM {re.escape(fq)}", s):
                return [[self.ROWS] + [self.ROWS] * (2 * len(d["columns"]))]
            m = re.fullmatch(rf"SELECT (`\w+`(?:, `\w+`)*) FROM {re.escape(fq)} LIMIT \d+", s)
            if m:
                return self._sample(n, re.findall(r"`(\w+)`", m.group(1)))
        raise AssertionError(f"a statement the fake warehouse does not answer: {s[:100]}")


def _shape(m):
    """Node types with table and key; edge types with table and both ends."""
    return ({n: (s.source_table, s.natural_key) for n, s in m.nodes.items()},
            {n: (e.source_table, e.src, e.src_type, e.dst, e.dst_type) for n, e in m.edges.items()})


def test_propose_over_the_declared_keys_drafts_the_generator_mapping(monkeypatch, tmp_path):
    monkeypatch.setattr(profile_step, "execute_sql", Warehouse(sg.build_sql(CAT, SCH)))
    config = {"run_folder": str(tmp_path), "warehouse_id": "wh", "workspace_client": object(),
              "source_catalog": CAT, "source_schema": SCH}
    profile_step.profile(config)
    propose(config)
    drafted = mapping.load_mapping(tmp_path / "graph_mapping.yaml")
    generator = mapping.parse_mapping(_mapping())
    assert (drafted.source_catalog, drafted.source_schema) == (CAT, SCH)
    assert _shape(drafted) == _shape(generator)


# --- the build and the drop

def test_build_runs_the_sql_in_order_and_writes_the_mapping(rec, tmp_path):
    path = tmp_path / "graph_mapping.yaml"
    sg.build_scale_graph(None, "wh", CAT, SCH, path)
    lookup, *rest = rec.statements
    assert f"{CAT}.information_schema.schemata" in lookup and f"'{SCH}'" in lookup, lookup
    assert rest == sg.build_sql(CAT, SCH)
    assert path.read_text() == sg.mapping_yaml(CAT, SCH)
    assert rec.drops() == []


@pytest.mark.parametrize("row", [[None], ["a schema made by someone else"], [""], "made-by-build"])
def test_an_existing_schema_stops_the_build_before_any_create(monkeypatch, tmp_path, row):
    if row == "made-by-build":  # hostile: a leftover from an earlier build is not reused either
        row = [_made_by_build_comment(monkeypatch, tmp_path)]
    r = Recorder(schemata=[row])
    monkeypatch.setattr(sg, "execute_sql", r)
    path = tmp_path / "graph_mapping.yaml"
    with pytest.raises(sg.ScaleGraphError):
        sg.build_scale_graph(None, "wh", CAT, SCH, path)
    assert r.creates() == [] and r.drops() == []
    assert not path.exists()


def test_an_existing_schema_leaves_an_old_mapping_file_alone(monkeypatch, tmp_path):
    r = Recorder(schemata=[["a schema made by someone else"]])
    monkeypatch.setattr(sg, "execute_sql", r)
    path = tmp_path / "graph_mapping.yaml"
    path.write_text("an earlier mapping\n")
    with pytest.raises(sg.ScaleGraphError):
        sg.build_scale_graph(None, "wh", CAT, SCH, path)
    assert path.read_text() == "an earlier mapping\n"


def _made_by_build_comment(monkeypatch, tmp_path):
    """The schema comment the build writes, read from its CREATE SCHEMA statement."""
    r = Recorder()
    monkeypatch.setattr(sg, "execute_sql", r)
    earlier = tmp_path / "earlier-build"
    earlier.mkdir(exist_ok=True)
    sg.build_scale_graph(None, "wh", CAT, SCH, earlier / "graph_mapping.yaml")
    m = re.search(r"COMMENT\s+'([^']+)'", r.creates()[0])
    assert m, r.creates()[0]
    return m.group(1)


def test_drop_removes_a_schema_the_build_made(monkeypatch, tmp_path):
    comment = _made_by_build_comment(monkeypatch, tmp_path)
    r = Recorder(schemata=[[comment]])
    monkeypatch.setattr(sg, "execute_sql", r)
    sg.drop_scale_graph(None, "wh", CAT, SCH)
    assert r.drops() == [f"DROP SCHEMA {FQ} CASCADE"]


@pytest.mark.parametrize("schemata", [[], [[None]], [["a schema made by someone else"]],
                                      "made-by-build-plus-text"])
def test_drop_refuses_a_schema_the_build_did_not_make(monkeypatch, tmp_path, schemata):
    if schemata == "made-by-build-plus-text":  # hostile: the build's comment inside a longer one
        schemata = [[_made_by_build_comment(monkeypatch, tmp_path) + " (edited)"]]
    r = Recorder(schemata=schemata)
    monkeypatch.setattr(sg, "execute_sql", r)
    with pytest.raises(sg.ScaleGraphError):
        sg.drop_scale_graph(None, "wh", CAT, SCH)
    assert r.drops() == []


# --- the module says what it builds and names no source

def test_the_module_says_what_it_builds_and_names_no_source():
    doc = (sg.__doc__ or "").lower()
    for word in ("providers", "practices", "locations", "systems", "markets", "networks", "referral",
                 "invented"):
        assert word in doc, word
    text = open(sg.__file__, encoding="utf-8").read().lower()
    for word in ("hospital", "directory", "based on", "taken from", "derived from",
                 "modelled on", "modeled on", "copied from"):
        assert word not in text, word
    assert not re.search(r"\b20\d\d-\d\d-\d\d\b", text), "a date ties the values to a source"


# Spark SQL resolves a WHERE clause against the FROM clause's columns, never against an alias its
# own SELECT list defines: `SELECT id AS x FROM range(n) WHERE f(x)` fails on a warehouse with
# UNRESOLVED_COLUMN. Offline SQL engines (SQLite) accept it, so this check reads the SQL text.
_SELECT_WHERE = re.compile(r"SELECT\s+(?P<cols>(?:(?!SELECT).)*?)\s+FROM\s+range\([^()]*\)\s+WHERE\s+(?P<cond>[^\n]*)",
                           re.IGNORECASE | re.DOTALL)


@pytest.mark.parametrize("scale", [1.0, 0.001])
def test_no_where_clause_names_an_alias_its_own_select_defines(scale):
    checked = 0
    for statement in sg.build_sql(CAT, SCH, scale):
        for m in _SELECT_WHERE.finditer(statement):
            aliases = set(re.findall(r"\bAS\s+(\w+)", m.group("cols"), re.IGNORECASE))
            named = set(re.findall(r"\b[A-Za-z_]\w*\b", m.group("cond")))
            assert not aliases & named, f"WHERE names its own SELECT alias {sorted(aliases & named)}: {m.group(0)}"
            checked += 1
    assert checked, "the generator has a filtered range subquery (the second practice); the check found none"
