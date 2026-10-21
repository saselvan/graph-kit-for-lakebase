"""A scripted fake `sql` executor for the offline enrich tests. No SQL engine.

It tracks which tables exist (and a row count for each), so the loop, ordering and cleanup
logic of the shipped code can be tested offline. The math of the shipped SQL is proven only
live, on the Databricks SQL warehouse (tests/test_enrich_live.py).

Row counts model one simple graph shape, a directed chain of `chain` edges (or, with
`cyclic=True`, edges that never peel off). Kahn rounds peel one edge per round; closure rounds
find one fewer new pair per depth. That is enough to drive every loop to its real exit.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

BASE_TABLES = ("nodes", "edges", "node_id_map")
_CREATE = re.compile(r"CREATE (?:OR REPLACE )?TABLE (\S+)")
_INSERT = re.compile(r"INSERT INTO (\S+) .*?FROM (\S+)", re.S)
_DROP = re.compile(r"DROP TABLE IF EXISTS (\S+)")
_COUNT_OF = re.compile(r"^SELECT COUNT\(\*\) FROM (\S+)$")
_TABLE_TYPES = re.compile(r"FROM (\w+)\.information_schema\.tables\s+WHERE table_schema = '(\w+)'"
                          r"\s+AND table_name IN \(([^)]*)\)", re.S)


class FakeSql:
    def __init__(self, fq: str = "c.s", chain: int = 3, cyclic: bool = False,
                 l1_values: Optional[List[float]] = None, fail_on: Optional[str] = None, fail_at: int = 1,
                 kinds: Optional[Dict[str, Tuple[str, str]]] = None):
        self.fq, self.chain, self.cyclic = fq, chain, cyclic
        # short table name -> (information_schema table_type, data_source_format); default MANAGED DELTA
        self.kinds: Dict[str, Tuple[str, str]] = dict(kinds or {})
        self.tables: Dict[str, int] = {f"{fq}.{t}": chain + 1 for t in BASE_TABLES}
        self.l1_values = list(l1_values or [0.0])
        self.l1_calls = 0
        self.fail_on, self.fail_at, self.fail_seen = fail_on, fail_at, 0
        self.statements: List[str] = []

    def names(self) -> set:
        """Short names of the tables that exist now."""
        return {t.rsplit(".", 1)[1] for t in self.tables}

    def _maybe_fail(self, statement: str) -> None:
        if self.fail_on and self.fail_on in statement:
            self.fail_seen += 1
            if self.fail_seen == self.fail_at:
                raise RuntimeError("warehouse went away")

    def _must_exist(self, table: str) -> int:
        if table not in self.tables:
            raise RuntimeError(f"TABLE_OR_VIEW_NOT_FOUND: {table}")
        return self.tables[table]

    def _created_rows(self, statement: str) -> int:
        peel = re.search(r"SELECT \* FROM (\S+) WHERE", statement)
        if peel:  # one Kahn round
            rows = self._must_exist(peel.group(1))
            return rows if self.cyclic else rows - 1
        step = re.search(r"FROM \(SELECT f\.anc, e\.dst AS dsc FROM (\S+) f", statement)
        if step:  # one closure round
            return max(self._must_exist(step.group(1)) - 1, 0)
        return self.chain if " AS " in statement else 0  # a fixed-shape output table starts empty

    def _table_types(self, catalog: str, schema: str, in_list: str) -> List[List[Any]]:
        """information_schema.tables rows (name, type, format) for the listed tables that exist."""
        wanted = re.findall(r"'(\w+)'", in_list)
        return [[name, *self.kinds.get(name, ("MANAGED", "DELTA"))] for name in wanted
                if f"{catalog}.{schema}.{name}" in self.tables]  # case-sensitive, as UC's stored names

    def __call__(self, statement: str) -> List[List[Any]]:
        self.statements.append(statement)
        self._maybe_fail(statement)
        s = statement.strip()
        types = _TABLE_TYPES.search(s)
        if types:
            return self._table_types(*types.groups())
        if "L1_CHANGE" in s:
            value = self.l1_values[min(self.l1_calls, len(self.l1_values) - 1)]
            self.l1_calls += 1
            return [[str(value)]]
        if s.startswith("CREATE"):
            self.tables[_CREATE.match(s).group(1)] = self._created_rows(s)
            return []
        if s.startswith("INSERT"):
            dst, src = _INSERT.match(s).groups()
            self.tables[dst] = self._must_exist(dst) + self.tables.get(src, 0)
            return []
        if s.startswith("DROP"):
            self.tables.pop(_DROP.match(s).group(1), None)
            return []
        count = _COUNT_OF.match(s)
        if count:
            return [[str(self._must_exist(count.group(1)))]]
        if s.startswith("SELECT COUNT(*) FROM (") or "COUNT(DISTINCT" in s:
            return [[str(self.chain)]]
        if s.startswith("SELECT COUNT(*)"):  # components: nodes whose label changed
            return [["0"]]
        if s.startswith("WITH deg"):
            return [["k0", 1]]
        return []


class EdgeFakeSql(FakeSql):
    """FakeSql over REAL edge rows (src, dst, relationship_type): the DAG check and closure work
    on the actual pairs, so a cycle is found (or not) from the built edges, not from a flag.

    It answers only the literal statement forms below (written out here, NOT imported from the
    shipped code, so a changed edges query is caught): any other statement that reads
    `<fq>.edges` raises, e.g. a query that leaves out one edge type. Every statement it does not
    know falls through to FakeSql's row counting (degree, the hub, the output tables).
    """

    def __init__(self, edges: List[Tuple[str, str, str]], fq: str = "c.s", **kw: Any):
        super().__init__(fq=fq, chain=len(edges), **kw)
        self.edges = list(edges)
        self.pairs: Dict[str, set] = {}  # table -> set of 2-tuples (pair tables only)
        e = f"{fq}.edges"
        self._directed = (f"SELECT DISTINCT src_node_id AS src, dst_node_id AS dst FROM {e} "
                          "WHERE src_node_id <> dst_node_id")
        self._undirected = (f"SELECT DISTINCT a AS src, b AS dst FROM (SELECT src_node_id AS a, "
                            f"dst_node_id AS b FROM {e} UNION ALL SELECT dst_node_id AS a, "
                            f"src_node_id AS b FROM {e}) WHERE a <> b")

    def directed(self) -> set:
        return {(s, d) for s, d, _rel in self.edges if s != d}

    def _pair_create(self, table: str, body: str) -> bool:
        """Materialise `CREATE OR REPLACE TABLE table AS body` as a pair set, if it is one."""
        if body == self._directed:
            self.pairs[table] = self.directed()
        elif body == self._undirected:
            self.pairs[table] = self.directed() | {(d, s) for s, d in self.directed()}
        elif re.fullmatch(r"SELECT src AS anc, dst AS dsc FROM (\S+)", body):
            self.pairs[table] = set(self.pairs[body.rsplit(" ", 1)[1]])
        elif (m := re.fullmatch(r"SELECT \* FROM (\S+) WHERE src IN \(SELECT dst FROM (\S+)\)", body)):
            cur = self.pairs[m.group(1)]
            heads = {d for _s, d in self.pairs[m.group(2)]}
            self.pairs[table] = {(s, d) for s, d in cur if s in heads}
        elif (m := re.fullmatch(r"SELECT DISTINCT n\.anc, n\.dsc FROM \(SELECT f\.anc, e\.dst AS dsc "
                                r"FROM (\S+) f JOIN (\S+) e ON e\.src = f\.dsc\) n WHERE NOT EXISTS "
                                r"\(SELECT 1 FROM (\S+) c WHERE c\.ancestor_id = n\.anc AND "
                                r"c\.descendant_id = n\.dsc\)", body)):
            cur, step, done = (self.pairs[g] for g in m.groups())
            self.pairs[table] = {(a, t) for a, d in cur for s, t in step if s == d} - done
        else:
            return False
        self.tables[table] = len(self.pairs[table])
        return True

    def __call__(self, statement: str) -> List[List[Any]]:
        s = statement.strip()
        create = re.fullmatch(r"CREATE OR REPLACE TABLE (\S+) AS (.*)", s, re.S)
        if create and self._pair_create(*create.groups()):
            self.statements.append(statement)
            return []
        if s == f"SELECT COUNT(*) FROM ({self._directed})":
            self.statements.append(statement)
            return [[str(len(self.directed()))]]
        if s == f"SELECT COUNT(DISTINCT node_id) FROM {self.fq}.nodes":
            self.statements.append(statement)
            return [[str(len({n for s_, d, _r in self.edges for n in (s_, d)}))]]
        insert = re.fullmatch(r"INSERT INTO (\S+) SELECT (?:src, dst|anc, dsc), \d+ FROM (\S+)", s)
        if insert and insert.group(2) in self.pairs:
            dst, src = insert.groups()
            self.pairs[dst] = self.pairs.get(dst, set()) | self.pairs[src]
            self.tables[dst] = len(self.pairs[dst])
            self.statements.append(statement)
            return []
        if s.startswith(f"CREATE OR REPLACE TABLE {self.fq}.closure ("):
            self.pairs[f"{self.fq}.closure"] = set()
        if f"{self.fq}.edges" in s and not s.startswith("WITH deg"):
            raise AssertionError(f"EdgeFakeSql: unrecognised read of the edges: {s}")
        if s.startswith("DROP TABLE IF EXISTS "):
            self.pairs.pop(s.split()[-1], None)
        return super().__call__(statement)
