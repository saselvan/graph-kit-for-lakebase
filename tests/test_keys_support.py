"""Support for the key-check tests (tests/test_keys*.py): a row-holding fake SQL warehouse.

The fake holds plain Python rows per source table and answers the statements profile and build
send, computing every count in Python from those rows. It is the REFERENCE: its counts come from
its own reading of the rows (keys rendered the way Spark's CAST(... AS STRING) and the kit's
node_id formula render them), never from lgk code. A key-check statement whose shape it does not
recognise raises AssertionError, so a mutant that changes the SQL (another column, COUNT for
COUNT(DISTINCT), a dropped NULL filter) fails the test instead of passing on an empty answer.

No SQL engine is used (no DuckDB or any non-Databricks engine): only regexes over the kit's own
statement shapes, and Python counting.
"""

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

Row = Dict[str, Any]


def render(value: Any) -> Optional[str]:
    """Spark's CAST(value AS STRING) for the value types the fixtures use (str, int, None)."""
    if value is None:
        return None
    assert not isinstance(value, (bool, float)), "fixtures use str/int keys only"
    return str(value)


def _parse_key(expr: str, alias: Optional[str] = None) -> List[str]:
    """The key columns of a key expression: CAST(c AS STRING), or to_json(array(CAST(a AS STRING), ...))."""
    p = re.escape(alias + ".") if alias else ""
    single = re.fullmatch(rf"CAST\({p}(\w+) AS STRING\)", expr)
    if single:
        return [single.group(1)]
    comp = re.fullmatch(r"to_json\(array\((.+)\)\)", expr)
    assert comp, f"unrecognised key expression: {expr}"
    cols = re.findall(rf"CAST\({p}(\w+) AS STRING\)", comp.group(1))
    assert ", ".join(f"CAST({alias + '.' if alias else ''}{c} AS STRING)" for c in cols) == comp.group(1), expr
    return cols


def key_of(row: Row, cols: Sequence[str]) -> Optional[str]:
    """A node's rendered natural key: the value (single), the JSON array of parts (composite)."""
    if len(cols) == 1:
        return render(row[cols[0]])
    return json.dumps([render(row[c]) for c in cols], separators=(",", ":"))


class RowWarehouse:
    """tables: {name: {"columns": [(col, type)], "rows": [dict], "pk": [cols], "fks": [(name, cols, ref, ref_cols)]}}.

    `after_check`: {table: [rows]} appended right after build's duplicate check of that table
    (a writer racing the build), so the read-back of the built nodes is what must catch it.
    """

    def __init__(self, tables: Dict[str, Dict[str, Any]], catalog: str = "main", schema: str = "gold",
                 after_check: Optional[Dict[str, List[Row]]] = None):
        self.tables, self.catalog, self.schema = tables, catalog, schema
        self.after_check = dict(after_check or {})
        self.statements: List[str] = []
        self.nodes: List[str] = []  # node ids the INSERTs wrote (type::key)

    # ---------------------------------------------------------------- helpers
    def _rows(self, fq: str) -> List[Row]:
        prefix = f"{self.catalog}.{self.schema}."
        assert fq.startswith(prefix), f"unexpected table {fq}"
        return self.tables[fq[len(prefix):]]["rows"]

    def __call__(self, *args: Any, **_k: Any) -> List[List[Any]]:
        sql = args[-1] if isinstance(args[-1], str) else args[2]
        self.statements.append(sql)
        flat = " ".join(sql.split())
        for handler in (self._info_schema, self._dup_build, self._dup_profile, self._orphan_build,
                        self._orphan_profile, self._node_readback, self._insert_nodes, self._verdicts,
                        self._sample):
            out = handler(flat)
            if out is not None:
                return out
        assert not re.search(r"duplicate_key_rows|orphan_rows|duplicate_node_ids", flat), \
            f"unrecognised key-check statement: {flat}"
        return []

    # ---------------------------------------------------------------- information_schema
    def _info_schema(self, flat: str) -> Optional[List[List[Any]]]:
        if "information_schema.columns" in flat:
            return [[t, c, ty] for t in sorted(self.tables) for c, ty in self.tables[t]["columns"]]
        if "constraint_type = 'PRIMARY KEY'" in flat:
            return [[t, c] for t in sorted(self.tables) for c in self.tables[t].get("pk", [])]
        if "referential_constraints" in flat:
            out = []
            for t in sorted(self.tables):
                for name, cols, ref, ref_cols in sorted(self.tables[t].get("fks", [])):
                    out += [[t, name, c, ref, rc] for c, rc in zip(cols, ref_cols)]
            return out
        return None

    # ---------------------------------------------------------------- duplicate keys
    _DUP = (r"SELECT COUNT\((?P<k>.+?)\) - COUNT\(DISTINCT (?P=k)\) AS duplicate_key_rows, "
            r"COUNT_IF\((?P<n>.+?)\) AS null_key_rows FROM (?P<t>\S+)")

    def _dup_counts(self, m: "re.Match") -> Tuple[int, int]:
        cols = _parse_key(m.group("k"))
        assert m.group("n") == " OR ".join(f"{c} IS NULL" for c in cols), m.group("n")
        fq = m.group("t")
        rows = self._rows(fq)
        keys = [key_of(r, cols) for r in rows]
        non_null = [k for k in keys if k is not None]
        nulls = sum(1 for r in rows if any(r[c] is None for c in cols))
        dup = len(non_null) - len(set(non_null))
        table = fq.rsplit(".", 1)[1]
        rows.extend(self.after_check.pop(table, []))  # the race lands after the check
        return dup, nulls

    def _dup_profile(self, flat: str) -> Optional[List[List[Any]]]:
        m = re.fullmatch(self._DUP, flat)
        if not m:
            return None
        return [list(self._dup_counts(m))]

    def _dup_build(self, flat: str) -> Optional[List[List[Any]]]:
        m = re.fullmatch(rf"SELECT duplicate_key_rows FROM \({self._DUP}\) WHERE duplicate_key_rows > 0", flat)
        if not m:
            return None
        dup, _ = self._dup_counts(m)
        return [[dup]] if dup > 0 else []

    # ---------------------------------------------------------------- orphans
    def _orphan_profile(self, flat: str) -> Optional[List[List[Any]]]:
        m = re.fullmatch(r"SELECT COUNT\(\*\) AS orphan_rows FROM (?P<c>\S+) c LEFT ANTI JOIN (?P<p>\S+) p "
                         r"ON (?P<on>.+?) WHERE (?P<w>.+)", flat)
        if not m:
            return None
        pairs = re.findall(r"CAST\(c\.(\w+) AS STRING\) = CAST\(p\.(\w+) AS STRING\)", m.group("on"))
        assert " AND ".join(f"CAST(c.{a} AS STRING) = CAST(p.{b} AS STRING)" for a, b in pairs) == m.group("on")
        assert m.group("w") == " AND ".join(f"c.{a} IS NOT NULL" for a, _ in pairs), m.group("w")
        parents = {tuple(render(r[b]) for _, b in pairs) for r in self._rows(m.group("p"))}
        n = sum(1 for r in self._rows(m.group("c"))
                if all(r[a] is not None for a, _ in pairs)
                and tuple(render(r[a]) for a, _ in pairs) not in parents)
        return [[n]]

    def _orphan_build(self, flat: str) -> Optional[List[List[Any]]]:
        m = re.fullmatch(
            r"SELECT orphan_rows FROM \(SELECT COUNT\(\*\) AS orphan_rows FROM (?P<e>\S+) e LEFT ANTI JOIN "
            r"\(SELECT xxhash64\('(?P<t1>\w+)::'\|\|(?P<k>.+?)\) AS node_id FROM (?P<n>\S+) n "
            r"WHERE (?P<k2>.+?) IS NOT NULL\) m ON m\.node_id = xxhash64\('(?P<t2>\w+)::'\|\|e\.(?P<col>\w+)\) "
            r"WHERE e\.(?P<col2>\w+) IS NOT NULL\) WHERE orphan_rows > 0", flat)
        if not m:
            return None
        assert m.group("t1") == m.group("t2") and m.group("col") == m.group("col2"), flat
        k = m.group("k")
        cols = [k[2:]] if re.fullmatch(r"n\.\w+", k) else _parse_key(k, "n")
        assert m.group("k2") == k, flat
        node_type = m.group("t1")
        nodes = {f"{node_type}::{key_of(r, cols)}" for r in self._rows(m.group("n"))
                 if key_of(r, cols) is not None and (len(cols) > 1 or r[cols[0]] is not None)}
        col = m.group("col")
        n = sum(1 for r in self._rows(m.group("e"))
                if r[col] is not None and f"{node_type}::{render(r[col])}" not in nodes)
        return [[n]] if n > 0 else []

    # ---------------------------------------------------------------- nodes
    def _insert_nodes(self, flat: str) -> Optional[List[List[Any]]]:
        if not re.match(r"INSERT INTO \S+\.nodes ", flat):
            return None
        m = re.search(r"'(?P<t>\w+)' as node_type, CAST\((?P<k>.+?) AS STRING\) as natural_key, .* "
                      r"FROM (?P<src>\S+) n\b", flat)
        assert m, flat
        k = m.group("k")
        cols = [k[2:]] if re.fullmatch(r"n\.\w+", k) else _parse_key(k, "n")
        for r in self._rows(m.group("src")):
            key = key_of(r, cols)
            if len(cols) == 1 and r[cols[0]] is None:
                continue
            self.nodes.append(f"{m.group('t')}::{key}")
        return []

    def _node_readback(self, flat: str) -> Optional[List[List[Any]]]:
        if not re.fullmatch(r"SELECT duplicate_node_ids FROM \(SELECT COUNT\(\*\) - COUNT\(DISTINCT node_id\) "
                            r"AS duplicate_node_ids FROM \S+\.nodes\) WHERE duplicate_node_ids > 0", flat):
            return None
        d = len(self.nodes) - len(set(self.nodes))
        return [[d]] if d > 0 else []

    # ---------------------------------------------------------------- profile verdicts
    def _verdicts(self, flat: str) -> Optional[List[List[Any]]]:
        m = re.fullmatch(r"SELECT COUNT\(\*\), (?P<a>.+) FROM (?P<t>\S+)", flat)
        if not m or "COUNT(`" not in m.group("a"):
            return None
        rows = self._rows(m.group("t"))
        out: List[Any] = [len(rows)]
        for col in re.findall(r"COUNT\(`(\w+)`\), COUNT\(DISTINCT `\w+`\)", m.group("a")):
            vals = [r[col] for r in rows if r[col] is not None]
            out += [len(vals), len(set(vals))]
        return [out]

    def _sample(self, flat: str) -> Optional[List[List[Any]]]:
        m = re.fullmatch(r"SELECT (?P<c>.+) FROM (?P<t>\S+) LIMIT \d+", flat)
        if not m:
            return None
        cols = re.findall(r"`(\w+)`", m.group("c"))
        return [[r[c] for c in cols] for r in self._rows(m.group("t"))]
