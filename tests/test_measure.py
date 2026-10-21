"""Offline tests for the measure step (the kit's metrics C10, C13, C23, C30b).

Fakes only: the warehouse is a tag-dispatching fake (each reference statement starts with a
`/* lgk-measure:<kind>:<name> */` tag) and Postgres is a scripted fake whose walk answers are
written BY HAND in this file, never computed by the code under test. No DuckDB, no engine.

The reference is code that can be wrong, so it gets its own hostile fixture: a SOURCE with a
duplicate edge row, an orphan edge (its end has no node row), a NULL foreign key, ids whose
byte order differs from their numeric order, and a hub that forces a tie cut. A served answer
that a kit-table-based reference would share (orphan kept, duplicate counted) must DISAGREE.
"""

import json
from pathlib import Path

import pytest

from lgk import advisor
from lgk.mapping import parse_mapping
from lgk.measure import compare, record, reference
from lgk.measure.reference import SourceGraph
from lgk.runner import RunFolder

CAT, SRC = "cat_src", "gold"
KIT = "lgk_v1_kit"
SERVED = "lgk_served_x"

MAPPING = parse_mapping({
    "source_catalog": CAT, "source_schema": SRC,
    "nodes": {"prov": {"source_table": "providers", "natural_key": "provider_id", "label": "name"}},
    "edges": {"aff": {"source_table": "affiliations", "src": "provider_id", "src_type": "prov",
                      "dst": "hospital_id", "dst_type": "prov"}},
})

# ids chosen so byte order != numeric order ("n10" < "n9"): # hostile: tie-break order
NODE_ROWS = [["n1"], ["n2"], ["n3"], ["n4"], ["n9"], ["n10"], ["n11"]]
EDGE_ROWS = [
    ["n1", "n2"], ["n1", "n3"], ["n2", "n3"], ["n3", "n4"],
    ["n1", "n2"],              # hostile: duplicate edge row in the SOURCE (support must stay 1)
    ["n1", "nX"],              # hostile: orphan edge, nX has no node row (must be dropped)
    [None, "n2"],              # hostile: NULL foreign key
    ["n4", "n9"], ["n4", "n10"], ["n4", "n11"],  # hostile: hub tie at support 1 (cut by byte order)
    ["n9", "n1"],              # hostile: a cycle back to the seed
]


class FakeWarehouse:
    """Returns rows by the statement's lgk-measure tag; records every statement."""

    def __init__(self, rows_by_tag):
        self.rows_by_tag, self.log = rows_by_tag, []

    def __call__(self, sql):
        self.log.append(sql)
        tag = sql.split("*/", 1)[0].split("lgk-measure:", 1)[1].strip()
        return [list(r) for r in self.rows_by_tag[tag]]


def _wh(nodes=NODE_ROWS, edges=EDGE_ROWS):
    return FakeWarehouse({"nodes:prov": nodes, "edges:aff": edges})


def _graph(**kw):
    return reference.load_source_graph(_wh(**kw), MAPPING, fold_floor=11)


# ---------------------------------------------------------------------------- reference: source only


class TestReferenceReadsSourceOnly:
    def test_every_statement_reads_the_source_schema_and_never_kit_tables(self):
        wh = _wh()
        reference.load_source_graph(wh, MAPPING, fold_floor=11)
        assert len(wh.log) == 2
        for sql in wh.log:
            assert f"{CAT}.{SRC}." in sql
            for kit_table in ("node_id_map", ".nodes ", ".edges ", KIT):
                assert kit_table not in sql

    def test_node_ids_are_derived_from_the_source_key_by_the_documented_formula(self):
        wh = _wh()
        reference.load_source_graph(wh, MAPPING, fold_floor=11)
        nodes_sql = next(s for s in wh.log if "lgk-measure:nodes:prov" in s)
        assert "xxhash64('prov::' || CAST(n.provider_id AS STRING))" in nodes_sql
        assert f"FROM {CAT}.{SRC}.providers n" in nodes_sql
        assert "n.provider_id IS NOT NULL" in nodes_sql

    def test_orphan_and_null_edges_are_dropped_and_duplicates_collapse(self):
        g = _graph()
        assert ("aff", "n1", "nX") not in g.edges
        assert all(None not in k for k in g.edges)
        assert sum(1 for k in g.edges if k[1:] == ("n1", "n2")) == 1
        assert len(g.edges) == 8

    def test_differently_typed_cells_are_compared_as_text(self):
        # hostile: the connector hands back an int key cell for one side, a str for the other
        g = reference.load_source_graph(
            _wh(nodes=[[7], ["8"]], edges=[["7", 8], [7, "8"]]), MAPPING, fold_floor=11)
        assert set(g.nodes) == {"7", "8"}
        assert set(g.edges) == {("aff", "7", "8")}

    def test_refuses_a_source_larger_than_the_cap(self):
        with pytest.raises(reference.ReferenceTooLarge):
            reference.load_source_graph(_wh(), MAPPING, fold_floor=11, max_rows=5)


class TestReferenceFolds:
    FOLD = parse_mapping({
        "source_catalog": CAT, "source_schema": SRC,
        "nodes": {"prov": {"source_table": "providers", "natural_key": "provider_id", "label": "name"}},
        "edges": {"shared": {"source_table": "visits", "src": "provider_id", "src_type": "prov",
                             "dst": "provider_id", "dst_type": "prov", "fold_subject": "patient_id"}},
    })

    def test_fold_pairs_below_the_floor_are_dropped(self):
        wh = FakeWarehouse({"nodes:prov": [["a"], ["b"], ["c"]],
                            "edges:shared": [["a", "b", 12], ["b", "a", 12], ["a", "c", 10], ["c", "a", 10]]})
        g = reference.load_source_graph(wh, self.FOLD, fold_floor=11)
        assert g.edges == {("shared", "a", "b"): 12, ("shared", "b", "a"): 12}
        fold_sql = next(s for s in wh.log if "edges:shared" in s)
        assert "COUNT(DISTINCT a.patient_id)" in fold_sql and f"{CAT}.{SRC}.visits a" in fold_sql

    def test_folded_web_support_is_the_min_weight(self):
        g = SourceGraph(nodes=frozenset({"s", "x", "y", "t"}),
                        edges={("f1", "s", "t"): 12, ("f2", "s", "t"): 30, ("f1", "s", "x"): 11})
        assert reference.web_reference(g, "s", 1, 25, folded=True, floor=11) == [
            (1, "t", 12), (1, "x", 11)]


# ---------------------------------------------------------------------------- reference: walk semantics


class TestReferenceWalks:
    def test_web_depth1_dedupes_drops_orphan(self):
        assert reference.web_reference(_graph(), "n1", 1, 25) == [(1, "n2", 1), (1, "n3", 1)]

    def test_web_depth3_excludes_seen_and_cuts_ties_by_byte_order(self):
        # hop2 from {n2,n3}: n4 ; hop3 from {n4}: n9, n10, n11 tie -> byte order n10 < n11 (top_n=2)
        assert reference.web_reference(_graph(), "n1", 3, 2) == [
            (1, "n2", 1), (1, "n3", 1), (2, "n4", 1), (3, "n10", 1), (3, "n11", 1)]

    def test_web_support_counts_edges_from_the_frontier(self):
        g = SourceGraph(nodes=frozenset({"s", "a", "b", "t"}),
                        edges={("e", "s", "a"): None, ("e", "s", "b"): None,
                               ("e", "a", "t"): None, ("e", "b", "t"): None})
        assert reference.web_reference(g, "s", 2, 25) == [(1, "a", 1), (1, "b", 1), (2, "t", 2)]

    def test_tree_cycle_guard_and_per_level_cut(self):
        # n4 -> n9 -> n1 (the seed): the cycle guard stops it; level 4 cut to 2 by byte order
        assert reference.tree_reference(_graph(), "n2", 4, 2) == [(1, "n3"), (2, "n4")]
        assert reference.tree_reference(_graph(), "n3", 3, 3) == [(1, "n4"), (2, "n10"), (2, "n11")]
        # from the seed n1: level 4 would be n9 -> n1, already seen, so the walk stops at level 3
        assert reference.tree_reference(_graph(), "n1", 4, 25) == [
            (1, "n2"), (1, "n3"), (2, "n4"), (3, "n10"), (3, "n11"), (3, "n9")]

    def test_similar_is_shared_neighbour_jaccard(self):
        # undirected: a-c, b-c, b-d. a: N={c}, deg 1; b: N={c,d}, deg 2; shared(a,b)=1 -> 1/(1+2-1)
        g = SourceGraph(nodes=frozenset("abcd"),
                        edges={("e", "a", "c"): None, ("e", "b", "c"): None, ("e", "d", "b"): None})
        assert reference.similar_reference(g, "a", 10) == [("b", 0.5)]
        assert reference.similar_reference(g, "c", 10) == [("d", 0.5)]
        # k = 2: with a -> d added, a and b share {c, d}: 2 / (2 + 2 - 2) = 1.0
        g2 = SourceGraph(nodes=frozenset("abcd"), edges={**g.edges, ("e", "a", "d"): None})
        assert reference.similar_reference(g2, "a", 10) == [("b", 1.0)]
        assert reference.similar_reference(g2, "c", 1) == [("d", 1.0)]

    def test_seeds_are_nodes_with_out_edges_and_include_the_hub(self):
        g = _graph()
        # the hub (3 out-edges) first, then the rest spread over byte order: n1 n2 n3 n9 -> n1, n3
        assert reference.pick_seeds(g, 3) == ["n4", "n1", "n3"]
        assert reference.pick_seeds(g, 100) == sorted(["n1", "n2", "n3", "n4", "n9"], key=str.encode)


# ---------------------------------------------------------------------------- comparison


class ServedConn:
    """Scripted Postgres: walk answers by (sql, seed), written by hand; EXPLAIN times from a list.

    A fake clock (`now`, seconds) moves only when a statement runs: a walk by `walk_seconds()`
    (4 ms), SELECT 1 by `rtt_s` (1 ms); EXPLAIN (ANALYZE, TIMING OFF) answers `plan_times()`
    (planning 0 ms, execution from `times`, else 1 ms). So client timings are fixed numbers."""

    def __init__(self, answers=None, times=None, counts=(7, 8), worst_seed="n4", rtt_s=0.001):
        self.answers, self.log = answers or {}, []
        self.times = list(times or [])
        self.counts, self.worst_seed = counts, worst_seed
        self.autocommit = False
        self.t, self.rtt_s = 1000.0, rtt_s

    def now(self):
        return self.t

    def walk_seconds(self):
        return 0.004

    def plan_times(self):
        return 0.0, (self.times.pop(0) if self.times else 1.0)

    def execute(self, sql, params=None):
        self.log.append((sql, params))
        if sql.strip() == "SELECT 1":
            self.t += self.rtt_s
            return _R([(1,)])
        if sql.startswith("EXPLAIN (ANALYZE, TIMING OFF"):
            planning, execution = self.plan_times()
            return _R([([{"Plan": {"Node Type": "Result"}, "Planning Time": planning,
                          "Execution Time": execution}],)])
        if sql.startswith("EXPLAIN"):
            return _R([([{"Plan": {"Node Type": "Result"}}],)])
        if "GROUP BY 1" in sql and "edges_view" in sql:  # before the counts: it also says count(*)
            return _R([(self.worst_seed,)])
        if "count(*)" in sql and "nodes_view" in sql:
            return _R([(self.counts[0],)])
        if "count(*)" in sql and "edges_view" in sql:
            return _R([(self.counts[1],)])
        self.t += self.walk_seconds()
        return _R(self.answers.get((sql, (params or {}).get("start_node_id")), []))

    def close(self):
        pass


class _R:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


WEB1 = "web-depth-1-sql"


class TestCompare:
    def _check(self, served_rows, seeds=("n1",)):
        conn = ServedConn({(WEB1, s): rows for s, rows in served_rows.items()})
        ref = lambda seed: reference.web_reference(_graph(), seed, 1, 25)  # noqa: E731
        return compare.check_depth(conn, "web", WEB1, list(seeds), ref)

    def test_correct_served_answer_is_equal(self):
        # hand-written: what Postgres serves for n1 at depth 1 over the correctly built graph
        got = self._check({"n1": [(1, "n2", 1, None), (1, "n3", 1, None)]})
        assert got == {"seeds_checked": 1, "seeds_equal": 1, "seeds_different": 0}

    def test_kit_table_bug_shared_by_a_kit_reference_disagrees(self):
        # hostile: a build that kept the orphan and counted the duplicate row. A reference that
        # read the kit's edges would share both and agree; the source reference must not.
        got = self._check({"n1": [(1, "n2", 2, None), (1, "n3", 1, None), (1, "nX", 1, None)]})
        assert got["seeds_different"] == 1 and got["seeds_equal"] == 0

    def test_a_kept_orphan_alone_differs(self):
        # hostile: only the orphan bug (no duplicate): a reference that kept orphans would agree
        got = self._check({"n1": [(1, "n2", 1, None), (1, "n3", 1, None), (1, "nX", 1, None)]})
        assert got["seeds_different"] == 1

    def test_one_extra_node_differs(self):
        got = self._check({"n1": [(1, "n2", 1, None), (1, "n3", 1, None), (1, "n4", 1, None)]})
        assert got["seeds_different"] == 1

    def test_one_missing_node_differs(self):
        assert self._check({"n1": [(1, "n2", 1, None)]})["seeds_different"] == 1

    def test_reordered_rows_and_typed_cells_are_still_equal(self):
        # hostile: rows reordered; support as a Decimal-ish string and hop as a str
        from decimal import Decimal
        got = self._check({"n1": [("1", "n3", Decimal(1), None), (1, "n2", "1", None)]})
        assert got["seeds_equal"] == 1

    def test_seeds_are_bound_as_parameters_never_formatted(self):
        conn = ServedConn()
        compare.check_depth(conn, "web", WEB1, ["n1", "n2"], lambda s: [])
        assert [p for _, p in conn.log] == [{"start_node_id": "n1"}, {"start_node_id": "n2"}]

    def test_tree_and_similar_normalisation(self):
        assert compare.normalize("tree", [("b", 2, None), ("a", "1", None)]) == [(1, "a"), (2, "b")]
        assert compare.normalize("similar", [("b", 0.5000000000001), ("a", 1)]) == [("a", 1.0), ("b", 0.5)]


# ---------------------------------------------------------------------------- timing (C13)


class TestTiming:
    # The walk timer (client p95, the means breakdown) is tested in tests/test_measure_timing.py.

    @pytest.mark.parametrize("bad", [advisor.MIN_SAMPLES - 1, 0, True, 20.0, "200"])
    def test_samples_below_the_advisor_minimum_are_refused(self, bad):
        with pytest.raises(ValueError, match="samples"):
            compare.check_samples(bad)

    def test_default_samples_is_200_and_not_below_the_advisor_minimum(self):
        assert compare.DEFAULT_SAMPLES == 200 >= advisor.MIN_SAMPLES
        compare.check_samples(advisor.MIN_SAMPLES)


# ---------------------------------------------------------------------------- record (C23, C30b, no personal data)


def _run_json(folder: Path, durations, status="success"):
    rf = RunFolder(folder)
    state = json.loads(rf.run_json.read_text())
    for step, d in durations.items():
        state[step] = {"status": status, "error": None, "started_at": "x", "finished_at": "y",
                       "duration_s": d}
    rf.run_json.write_text(json.dumps(state))
    return rf.run_json


class TestFirstServeTime:
    # the serve group is derived from the runner's step list, never hardcoded.
    STEPS = tuple(RunFolder.STEPS[: RunFolder.STEPS.index("switch") + 1])

    def test_the_serve_group_ends_at_switch_and_excludes_later_steps(self):
        assert self.STEPS[0] == "preflight" and self.STEPS[-1] == "switch"
        assert "advise" in self.STEPS and "walks" not in self.STEPS

    def test_serve_group_is_derived_from_runner(self, tmp_path, monkeypatch):
        # hostile: a runner with one extra step before switch; a hardcoded or import-time list misses it
        steps = list(RunFolder.STEPS)
        steps.insert(steps.index("switch"), "stamp")
        monkeypatch.setattr(RunFolder, "STEPS", steps)
        durations = {s: 1.0 for s in steps[: steps.index("switch") + 1]}
        durations["stamp"] = 100.0
        got = record.first_serve_time(_run_json(tmp_path, durations))
        assert got == {"value": 100.0 + len(self.STEPS), "status": "complete", "steps_missing": []}
        del durations["stamp"]
        got = record.first_serve_time(_run_json(tmp_path / "b", durations))
        assert got["status"] == "partial" and got["steps_missing"] == ["stamp"]

    def test_sums_preflight_through_switch_only(self, tmp_path):
        durations = {s: float(i + 1) for i, s in enumerate(self.STEPS)}
        durations["walks"] = 1000.0  # hostile: a step after switch must not count
        got = record.first_serve_time(_run_json(tmp_path, durations))
        assert got == {"value": 45.0, "status": "complete", "steps_missing": []}

    def test_a_step_not_run_is_named_and_the_sum_is_partial(self, tmp_path):
        durations = {s: 2.0 for s in self.STEPS if s != "enrich"}
        got = record.first_serve_time(_run_json(tmp_path, durations))
        assert got == {"value": 16.0, "status": "partial", "steps_missing": ["enrich"]}

    def test_a_failed_step_is_missing(self, tmp_path):
        rj = _run_json(tmp_path, {s: 1.0 for s in self.STEPS})
        state = json.loads(rj.read_text())
        state["build"]["status"] = "failed"
        rj.write_text(json.dumps(state))
        assert record.first_serve_time(rj)["steps_missing"] == ["build"]


class TestFreshness:
    def test_freshness_is_always_pending_with_the_exact_reason(self):
        line = record.freshness_line()
        assert line["freshness"] == {
            "status": "pending",
            "reason": "nightly refresh untested from inside a workspace at this build"}
        assert line["value"] is None and line["metric"] == "freshness_lag"


class TestRecordPhi:
    COMMON = {"build": {"version": "0.1.0", "commit": None, "source_sha256": "ab12"}, "nodes": 7,
              "edges": 8, "endpoint_cu": {"min": 0.5, "max": 4.0}, "ts": "2026-10-05T00:00:00Z"}

    def _lines(self):
        return [
            {"metric": "walk_correctness", "value": 1, "unit": "count", "shape": "web", "depth": 1,
             "seeds_checked": 1, "seeds_equal": 1, "seeds_different": 0, "status": "checked",
             "reference": "python_over_source", **self.COMMON},
            {"metric": "walk_latency", "value": 4.0, "unit": "ms", "shape": "web", "depth": 1,
             "n_samples": 200, "timing": "client", "client_p95_ms": 4.0, "client_mean_ms": 3.5,
             "execution_p95_ms": 2.0, "execution_mean_ms": 1.8, "planning_mean_ms": 0.4,
             "select1_rtt_mean_ms": 1.0, "remainder_mean_ms": 0.3,
             "execution_label": "Postgres execution time (EXPLAIN ANALYZE); excludes planning, result transfer "
                                "and network",
             "remainder_label": "result transfer and client overhead", "client_mean_se_ms": 0.11,
             "select1_rtt_mean_se_ms": 0.02, "planning_mean_se_ms": 0.03, "execution_mean_se_ms": 0.07,
             "prepared_statements": "off", **self.COMMON},
            {**record.freshness_line(), **self.COMMON},
        ]

    def test_writes_jsonl_and_report_with_counts_only(self, tmp_path):
        record.write(tmp_path, self._lines(), forbidden=["-7771234567890123456"])
        lines = [json.loads(x) for x in (tmp_path / "measure.jsonl").read_text().splitlines()]
        assert [x["metric"] for x in lines] == ["walk_correctness", "walk_latency", "freshness_lag"]
        md = (tmp_path / "measure_report.md").read_text()
        assert "| web | 1 | 1 | 1 | 0 |" in md and "pending" in md and "client p95" in md

    def test_the_public_report_names_the_python_reference_and_no_customer_word(self, tmp_path):
        # owner review 2026-10-14: the report ships with the dated run, so it is public text and must match
        # v1's reference walks run in Python over ids read from the source tables (Spark SQL is V1.1)
        record.write(tmp_path, self._lines(), forbidden=[])
        md = (tmp_path / "measure_report.md").read_text()
        assert md.splitlines()[0] == "# Measure report"
        [heading] = [l for l in md.splitlines() if l.startswith("## C10")]
        assert heading == ("## C10 walk correctness (served walks vs a reference computed in Python over node "
                           "and edge ids read from the source tables)")
        assert "spark" not in md.lower()
        # the register's ban, built from parts (the repo-wide check, tests/test_shipping_words.py, reads this file)
        for word in ("P" + "OC", "cust" + "omer", "real" + " data", "P" + "HI"):
            assert word.lower() not in md.lower(), word

    def test_a_forbidden_seed_value_is_refused_before_anything_is_written(self, tmp_path):
        # hostile: a recognisable seed id smuggled into a field
        lines = self._lines()
        lines[0]["status"] = "-7771234567890123456"
        with pytest.raises(record.RecordRefused):
            record.write(tmp_path, lines, forbidden=["-7771234567890123456"])
        assert not (tmp_path / "measure.jsonl").exists()

    def test_check_line_refuses_a_forbidden_token_in_any_field(self):
        line = {**self._lines()[1], "reason": "-7771234567890123456"}
        with pytest.raises(record.RecordRefused):
            record.check_line(line, forbidden=["-7771234567890123456"])
        record.check_line(line, forbidden=["-1111111111111111111"])

    @pytest.mark.parametrize("key,value", [
        ("label", "Dr Jane Smith"),           # unknown key
        ("status", "Dr Jane Smith"),          # free text in a known key
        ("value", "12"),                       # a value must be a number
        ("value", float("nan")),
    ])
    def test_free_text_or_unknown_keys_are_refused(self, tmp_path, key, value):
        lines = self._lines()
        lines[0][key] = value
        with pytest.raises(record.RecordRefused):
            record.write(tmp_path, lines, forbidden=[])

    def test_build_identity_has_version_and_source_hash(self):
        b = record.build_identity()
        assert set(b) == {"version", "commit", "source_sha256"}
        assert isinstance(b["source_sha256"], str) and len(b["source_sha256"]) == 16


# ---------------------------------------------------------------------------- warehouse: every chunk


class _Conn:
    """A fake SQL-connector connection: one statement, its rows in fetchmany batches."""

    def __init__(self, batches, fail=None):
        self.batches, self.fail, self.sql, self.closed = list(batches), fail, [], False

    def cursor(self):
        return self

    def execute(self, statement):
        self.sql.append(statement)
        if self.fail:
            raise RuntimeError(self.fail)

    def fetchmany(self, n):
        return self.batches.pop(0) if self.batches else []

    def close(self):
        self.closed = True


class TestFetchAll:
    """Live 2026-10-06: an INLINE Statement Execution pull of the 10^6-node reference failed
    ("Inline byte limit exceeded", 25 MiB). fetch_all reads through the SQL connector instead."""

    # The workspace host shape fetch_all strips (https://<name>.<cloud domain>/), assembled from parts at
    # run time so the source holds no host.
    HOSTNAME = ".".join(("dbc-x", "cloud", "databricks", "com"))

    def _w(self):
        from types import SimpleNamespace as NS
        return NS(config=NS(host=f"https://{self.HOSTNAME}/", authenticate=lambda: {"Authorization": "x"}))

    def test_reads_every_batch_as_json_array_text(self):
        from lgk.measure.warehouse import fetch_all
        # hostile: typed cells (int, bool, None, float) where JSON_ARRAY gave text, over 3 batches
        conn = _Conn([[("a", 1, True)], [("b", None, False)], [("c", 2.5, True)]])
        seen = {}
        rows = fetch_all(self._w(), "wh1", "SELECT 1", connect=lambda **kw: seen.update(kw) or conn)
        assert rows == [["a", "1", "true"], ["b", None, "false"], ["c", "2.5", "true"]]
        assert conn.sql == ["SELECT 1"] and conn.closed

    def test_connects_to_the_warehouse_with_telemetry_off_and_no_token_string(self):
        from lgk.measure.warehouse import fetch_all
        seen = {}
        fetch_all(self._w(), "wh1", "SELECT 1", connect=lambda **kw: seen.update(kw) or _Conn([]))
        assert seen["server_hostname"] == self.HOSTNAME
        assert seen["http_path"] == "/sql/1.0/warehouses/wh1"
        assert seen["enable_telemetry"] is False  # no telemetry from the connector either
        assert "access_token" not in seen and seen["credentials_provider"]()() == {"Authorization": "x"}

    def test_a_failed_statement_raises_and_closes(self):
        from lgk.measure.warehouse import fetch_all
        from lgk.uc_sql import StatementFailedError
        conn = _Conn([], fail="boom")
        with pytest.raises(StatementFailedError, match="FAILED: RuntimeError: boom"):
            fetch_all(self._w(), "wh1", "SELECT 1", connect=lambda **kw: conn)
        assert conn.closed


class TestMixedFoldPolicy:
    def test_a_mixed_graph_marks_web_not_checked(self, monkeypatch):
        from lgk.privacy.gate import MixedFoldSelectionError
        from lgk.steps import measure as step

        def refuse(*a):
            raise MixedFoldSelectionError("mixed")

        monkeypatch.setattr(step, "served_walk_policy", refuse)
        floor, folded = step._walk_policy("run")
        assert folded is None and floor >= 11
        s = step.settings({})
        queries, ref, status, reason = step._shape_plan("web", s, "run", "served", floor, folded)
        assert (queries, ref, status, reason) == ({}, None, "not_checked", "mixed_fold_selection")
