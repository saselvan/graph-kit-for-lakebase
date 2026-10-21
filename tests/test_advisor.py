"""Offline tests for the index advisor: plan reading, proposals, percentiles, guards.

Plans are hand-written EXPLAIN (VERBOSE, FORMAT JSON) documents in the shape Postgres emits,
not produced by the code under test. The connection is a scripted fake that records every
statement, so a guard that fires "too late" (after a write) is visible.
"""

import random

import pytest

from lgk import advisor
from lgk.versions import LiveVersionError

V2 = "lgk_ab12cd_v2"
V1 = "lgk_ab12cd_v1"
SERVED = "lgk_served_ab12cd"


def _seq(schema, table, alias, flt=None):
    node = {"Node Type": "Seq Scan", "Schema": schema, "Relation Name": table, "Alias": alias}
    if flt:
        node["Filter"] = flt
    return node


def _hash_join(cond, outer, inner_scan):
    return {"Node Type": "Hash Join", "Hash Cond": cond,
            "Plans": [outer, {"Node Type": "Hash", "Plans": [inner_scan]}]}


WEB_PLAN = {"Plan": {"Node Type": "Sort", "Plans": [
    _hash_join("(e.src_node_id = f.node_id)",
               {"Node Type": "CTE Scan", "Alias": "f"},
               _seq(V2, "edges", "e", "(NOT (hashed SubPlan 1))")),
    # hostile: a small table in the version schema, joined with a positive equality
    _hash_join("(c.node_id = s.node_id)",
               {"Node Type": "Subquery Scan", "Alias": "c"},
               _seq(V2, "node_score", "s", "(s.method = 'degree'::text)")),
    # hostile: a large table in ANOTHER schema (the live version) must never be indexed
    _hash_join("(x.src_node_id = f.node_id)",
               {"Node Type": "CTE Scan", "Alias": "f"},
               _seq(V1, "edges", "x")),
]}}

# hostile: the negated dst_node_id comparison is the cycle guard, not an index candidate
TREE_PLAN = {"Plan": {"Node Type": "Nested Loop", "Plans": [
    {"Node Type": "WorkTable Scan", "Alias": "w"},
    _seq(V2, "edges", "e",
         "((e.src_node_id = ANY (w.frontier)) AND (NOT (e.dst_node_id = ANY (w.seen))))"),
]}}

# hostile: the PG17 rendering of NOT IN names the column inside the negation
PG17_PLAN = {"Plan": _seq(V2, "edges", "e",
                          "(NOT (ANY (e.dst_node_id = (hashed SubPlan 1).col1)))")}


class TestSeqScanReading:
    def test_join_condition_column_is_a_candidate(self):
        scans = advisor.seq_scans(WEB_PLAN)
        edges = [s for s in scans if (s.schema, s.table) == (V2, "edges")]
        assert [s.columns for s in edges] == [("src_node_id",)]

    def test_filter_equality_and_join_both_collected_for_small_table(self):
        scans = {(s.schema, s.table): s.columns for s in advisor.seq_scans(WEB_PLAN)}
        assert scans[(V2, "node_score")] == ("method", "node_id")

    def test_negated_comparison_is_not_a_candidate(self):
        [scan] = advisor.seq_scans(TREE_PLAN)
        assert scan.columns == ("src_node_id",)

    def test_pg17_not_in_rendering_is_not_a_candidate(self):
        [scan] = advisor.seq_scans(PG17_PLAN)
        assert scan.columns == ()

    def test_explain_list_document_is_accepted(self):
        # EXPLAIN (FORMAT JSON) returns a one-element list
        assert advisor.seq_scans([TREE_PLAN])[0].columns == ("src_node_id",)


class _FakeConn:
    """Scripted Postgres: row counts per table, existing leading index columns, live schemas."""

    def __init__(self, rows=None, indexed=(), live=()):
        self.rows, self.indexed, self.live, self.log = rows or {}, set(indexed), set(live), []

    def execute(self, sql, params=None):
        self.log.append(sql)
        result = []
        if "pg_rewrite" in sql:
            result = [(s,) for s in self.live]
        elif sql.startswith("SELECT count(*) FROM"):
            result = [(self.rows[sql.split("FROM ")[1].strip()],)]
        elif "indkey[0]" in sql:
            result = [(c,) for (s, t, c) in self.indexed if (s, t) == tuple(params)]
        return _Result(result)


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class TestProposals:
    def test_large_table_seq_scan_gets_deterministic_if_not_exists_index(self):
        conn = _FakeConn(rows={f"{V2}.edges": 50_000, f"{V2}.node_score": 40})
        props = advisor.propose(conn, V2, [WEB_PLAN, TREE_PLAN], min_rows=10_000)
        assert [p.statement for p in props] == [
            f"CREATE INDEX IF NOT EXISTS lgk_edges_src_node_id_idx ON {V2}.edges (src_node_id)"
        ]

    def test_name_matches_reapply_fixed_name(self):
        # reapply.INDEXES names this index exactly so; same name => IF NOT EXISTS no-op
        assert advisor.index_name("edges", "src_node_id") == "lgk_edges_src_node_id_idx"

    def test_small_table_is_not_large(self):
        # hostile: node_score has a candidate column but only 40 rows
        conn = _FakeConn(rows={f"{V2}.edges": 50_000, f"{V2}.node_score": 40})
        props = advisor.propose(conn, V2, [WEB_PLAN], min_rows=10_000)
        assert {p.table for p in props} == {"edges"}

    def test_min_rows_boundary_is_inclusive(self):
        at = _FakeConn(rows={f"{V2}.edges": 5_000})
        below = _FakeConn(rows={f"{V2}.edges": 4_999})
        assert len(advisor.propose(at, V2, [TREE_PLAN], min_rows=5_000)) == 1
        assert advisor.propose(below, V2, [TREE_PLAN], min_rows=5_000) == []

    def test_min_rows_parameter_changes_the_verdict(self):
        conn = _FakeConn(rows={f"{V2}.edges": 50_000, f"{V2}.node_score": 40})
        props = advisor.propose(conn, V2, [WEB_PLAN], min_rows=10)
        assert {(p.table, p.column) for p in props} == {
            ("edges", "src_node_id"), ("node_score", "method"), ("node_score", "node_id")}

    def test_existing_index_on_column_proposes_nothing(self):
        conn = _FakeConn(rows={f"{V2}.edges": 50_000}, indexed={(V2, "edges", "src_node_id")})
        assert advisor.propose(conn, V2, [TREE_PLAN], min_rows=10_000) == []

    def test_other_schema_never_proposed_even_when_large(self):
        conn = _FakeConn(rows={f"{V2}.edges": 1, f"{V2}.node_score": 1, f"{V1}.edges": 10**6})
        assert advisor.propose(conn, V2, [WEB_PLAN], min_rows=10_000) == []
        assert not any(f"{V1}.edges" in s for s in conn.log)

    def test_duplicates_across_plans_are_proposed_once(self):
        conn = _FakeConn(rows={f"{V2}.edges": 50_000})
        assert len(advisor.propose(conn, V2, [TREE_PLAN, TREE_PLAN, TREE_PLAN], min_rows=1)) == 1


class TestPercentile:
    def test_nearest_rank_on_reordered_input(self):
        values = [5.0, 1.0, 4.0, 2.0, 3.0, 10.0, 9.0, 8.0, 7.0, 6.0]  # hostile: unsorted
        assert advisor.percentile(values, 50) == 5.0
        assert advisor.percentile(values, 95) == 10.0

    def test_twenty_samples(self):
        values = [float(v) for v in range(1, 21)]
        random.Random(7).shuffle(values)
        assert advisor.percentile(values, 50) == 10.0
        assert advisor.percentile(values, 95) == 19.0


class TestVersionQueries:
    def test_served_views_are_rewritten_to_version_tables(self):
        q = {"web": {1: f"SELECT * FROM {SERVED}.edges_view e JOIN {SERVED}.nodes_view n ON true"}}
        out = advisor.version_queries(q, SERVED, V2)
        assert out["web"][1] == f"SELECT * FROM {V2}.edges e JOIN {V2}.nodes n ON true"

    def test_sorted_web_walk_over_the_served_score_view_is_rewritten(self):
        """2e: a sorted walk joins <served>.node_score_view; the advisor times <version>.node_score."""
        from lgk.walks.sql import web_sql

        sql = web_sql(depth=2, served_schema=SERVED, top_n=3, enrich_schema=SERVED,
                      sort_method="degree", score_relation="node_score_view")
        out = advisor.version_queries({"web": {2: sql}}, SERVED, V2)["web"][2]
        assert out.count(f"LEFT JOIN {V2}.node_score s ") == 2
        assert out.count(f"FROM {V2}.edges e") == 2
        assert SERVED not in out

    def test_unknown_served_relation_refused(self):
        # hostile: a served relation with no version table would be timed on the live site
        q = {"web": {1: f"SELECT * FROM {SERVED}.edges_view e, {SERVED}.extra_view x"}}
        with pytest.raises(ValueError, match="served"):
            advisor.version_queries(q, SERVED, V2)


class TestGuards:
    def _advise(self, conn, **kw):
        args = dict(min_rows=10_000, warmup=3, samples=advisor.MIN_SAMPLES)
        args.update(kw)
        return advisor.advise(conn, SERVED, V2, {"web": {1: "SELECT 1"}}, {"start_node_id": "x"},
                              **args)

    def test_refuses_live_version_before_any_write(self):
        conn = _FakeConn(live={V2})
        with pytest.raises(LiveVersionError):
            self._advise(conn)
        assert len(conn.log) == 1 and "pg_rewrite" in conn.log[0]

    def test_too_few_samples_refused(self):
        conn = _FakeConn()
        with pytest.raises(ValueError, match="samples"):
            self._advise(conn, samples=advisor.MIN_SAMPLES - 1)
        assert conn.log == []

    def test_min_samples_is_pinned(self):
        assert advisor.MIN_SAMPLES == 20

    def test_no_warmup_refused(self):
        conn = _FakeConn()
        with pytest.raises(ValueError, match="warmup"):
            self._advise(conn, warmup=0)
        assert conn.log == []

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, "10"])
    def test_bad_min_rows_refused(self, bad):
        with pytest.raises(ValueError, match="min_rows"):
            self._advise(_FakeConn(), min_rows=bad)

    def test_defaults(self):
        assert advisor.DEFAULT_MIN_ROWS == 10_000
        assert advisor.DEFAULT_WARMUP >= 1
        assert advisor.DEFAULT_SAMPLES >= advisor.MIN_SAMPLES


class TestNamesAndInputs:
    def test_index_name_over_63_bytes_refused(self):
        with pytest.raises(ValueError, match="63"):
            advisor.index_name("t" * 40, "c" * 20)

    @pytest.mark.parametrize("bad", ["Edges", "edges;drop", "1x", ""])
    def test_index_name_bad_identifier_refused(self, bad):
        with pytest.raises(ValueError):
            advisor.index_name(bad, "src_node_id")

    @pytest.mark.parametrize("served,version", [("bad-name", V2), (SERVED, "v2; drop")])
    def test_version_queries_bad_identifier_refused(self, served, version):
        with pytest.raises(ValueError):
            advisor.version_queries({"web": {1: "SELECT 1"}}, served, version)

    def test_worst_case_seed_on_empty_edges_refused(self):
        from lgk.steps.advise import _worst_case_seed

        with pytest.raises(ValueError, match="empty"):
            _worst_case_seed(_FakeConn(), V2)


class _PlanConn(_FakeConn):
    """Full advisor run, scripted: EXPLAIN docs, timings, and an index that appears once created."""

    WALK = "SELECT walk FROM " + V2 + ".edges e"

    def execute(self, sql, params=None):
        if sql.startswith("CREATE INDEX"):
            self.indexed.add((V2, "edges", "src_node_id"))
        if "BUFFERS" in sql:
            self.log.append(sql)
            plan = TREE_PLAN if not self.indexed else {"Plan": {"Node Type": "Index Scan"}}
            return _Result([([plan],)])
        if "TIMING OFF" in sql:
            self.log.append(sql)
            return _Result([([{"Plan": {}, "Execution Time": 2.5}],)])
        return super().execute(sql, params)


class TestAdviseRun:
    def test_warmup_precedes_samples_in_both_phases_and_index_applied_once(self):
        conn = _PlanConn(rows={f"{V2}.edges": 50_000})
        q = {"tree": {1: f"SELECT walk FROM {SERVED}.edges_view e"}}
        report = advisor.advise(conn, SERVED, V2, q, {"start_node_id": "x"}, warmup=3)
        walk = _PlanConn.WALK
        kinds = ["warm" if s == walk else "explain" if "BUFFERS" in s and s.endswith(walk)
                 else "sample" if "TIMING OFF" in s and s.endswith(walk)
                 else "create" if s.startswith("CREATE INDEX") else None for s in conn.log]
        kinds = [k for k in kinds if k]
        phase = ["warm"] * 3 + ["explain"] + ["sample"] * advisor.DEFAULT_SAMPLES
        assert kinds == phase + ["create"] + phase
        row = report["depths"]["tree"]["1"]
        assert row["before"]["warmup_runs"] == 3 and row["before"]["samples"] == advisor.DEFAULT_SAMPLES
        assert row["before"]["p50_ms"] == 2.5 and row["after"]["p95_ms"] == 2.5
        assert [s["table"] for s in row["before"]["seq_scans"]] == ["edges"]
        assert row["after"]["seq_scans"] == []
        assert [p["name"] for p in report["proposals"]] == ["lgk_edges_src_node_id_idx"]
        assert sum(1 for s in conn.log if s.startswith(f"ANALYZE {V2}.edges")) == 2

    def test_report_files_written(self, tmp_path):
        conn = _PlanConn(rows={f"{V2}.edges": 50_000})
        q = {"tree": {1: f"SELECT walk FROM {SERVED}.edges_view e"}}
        report = advisor.advise(conn, SERVED, V2, q, {"start_node_id": "x"})
        advisor.write_report(str(tmp_path), report)
        import json as _json

        assert _json.loads((tmp_path / "advisor_report.json").read_text())["version_schema"] == V2
        md = (tmp_path / "advisor_report.md").read_text()
        assert "| tree | 1 | 2.500 | 2.500 | 2.500 | 2.500 | 20 |" in md
        assert f"CREATE INDEX IF NOT EXISTS lgk_edges_src_node_id_idx ON {V2}.edges (src_node_id)" in md


# ---------------------------------------------------------------------------- prover gaps


class TestProverGaps:
    def test_refusal_runs_before_analyze_on_a_served_query(self):
        conn = _FakeConn(live={V2})
        q = {"web": {1: f"SELECT * FROM {SERVED}.edges_view e"}}
        with pytest.raises(LiveVersionError):
            advisor.advise(conn, SERVED, V2, q, {"start_node_id": "x"})
        assert len(conn.log) == 1 and "pg_rewrite" in conn.log[0]

    def test_non_equality_comparisons_are_not_candidates(self):
        # hostile: range and inequality filters cannot use a btree equality lookup here
        plan = {"Plan": _seq(V2, "edges", "e",
                             "((e.weight > 5) AND (e.src_node_id <> 'x'::text) AND (e.dst_node_id >= 'a'::text))")}
        assert advisor.seq_scans(plan)[0].columns == ()

    def test_merge_cond_and_join_filter_are_read(self):
        plan = {"Plan": {"Node Type": "Merge Join", "Merge Cond": "(e.src_node_id = f.node_id)",
                         "Plans": [{"Node Type": "Nested Loop", "Join Filter": "(e.dst_node_id = g.node_id)",
                                    "Plans": [_seq(V2, "edges", "e")]}]}}
        assert advisor.seq_scans(plan)[0].columns == ("dst_node_id", "src_node_id")

    def test_view_rewrite_respects_word_boundary(self):
        # hostile: a relation whose name starts with a served view's name
        q = {"web": {1: f"SELECT * FROM {SERVED}.edges_view_old e"}}
        with pytest.raises(ValueError, match="served"):
            advisor.version_queries(q, SERVED, V2)

    def test_proposals_in_stable_table_then_column_order(self):
        # hostile: plans fed in reverse order
        p_b = {"Plan": _seq(V2, "zeta", "z", "((z.b_col = 1) AND (z.a_col = 2))")}
        p_a = {"Plan": _seq(V2, "alpha", "a", "(a.y_col = 1)")}
        conn = _FakeConn(rows={f"{V2}.zeta": 100, f"{V2}.alpha": 100})
        props = advisor.propose(conn, V2, [p_b, p_a], min_rows=1)
        assert [(p.table, p.column) for p in props] == [
            ("alpha", "y_col"), ("zeta", "a_col"), ("zeta", "b_col")]

    @pytest.mark.parametrize("bad", [True, 1.5, "3"])
    def test_bad_warmup_types_refused(self, bad):
        with pytest.raises(ValueError, match="warmup"):
            advisor.advise(_FakeConn(), SERVED, V2, {}, {}, warmup=bad)

    @pytest.mark.parametrize("bad", [True, 20.0, "20"])
    def test_bad_samples_types_refused(self, bad):
        with pytest.raises(ValueError, match="samples"):
            advisor.advise(_FakeConn(), SERVED, V2, {}, {}, samples=bad)

    def test_index_name_63_byte_edge(self):
        # "lgk_" + table + "_" + column + "_idx" = 9 + len(table) + len(column) bytes
        assert len(advisor.index_name("t" * 29, "c" * 25)) == 63
        with pytest.raises(ValueError, match="63"):
            advisor.index_name("t" * 30, "c" * 25)

    def test_create_index_sql_validates_schema(self):
        with pytest.raises(ValueError):
            advisor.create_index_sql("v2; drop table x", "edges", "src_node_id")


class _TimedConn(_PlanConn):
    """Varied server timings 1..20 shuffled, and buffer counts on the plan."""

    def __init__(self, **kw):
        super().__init__(**kw)
        order = list(range(1, 21))
        random.Random(3).shuffle(order)
        self.times = order * 4

    def execute(self, sql, params=None):
        if "TIMING OFF" in sql:
            self.log.append(sql)
            return _Result([([{"Plan": {}, "Execution Time": float(self.times.pop(0))}],)])
        if "BUFFERS" in sql:
            self.log.append(sql)
            return _Result([([{"Plan": {"Node Type": "Result", "Shared Hit Blocks": 7,
                                        "Shared Read Blocks": 2}}],)])
        return super().execute(sql, params)


class TestReportContent:
    def test_percentiles_buffers_params_and_md_values(self, tmp_path):
        conn = _TimedConn(rows={f"{V2}.edges": 50_000})
        q = {"tree": {2: f"SELECT walk2 FROM {SERVED}.edges_view e",
                      1: f"SELECT walk FROM {SERVED}.edges_view e"}}  # hostile: depths reordered
        report = advisor.advise(conn, SERVED, V2, q, {"start_node_id": "seed9"}, min_rows=123)
        assert list(report["depths"]["tree"]) == ["1", "2"]
        row = report["depths"]["tree"]["1"]["before"]
        assert (row["p50_ms"], row["p95_ms"]) == (10.0, 19.0)  # nearest rank over 1..20
        assert (row["shared_hit_blocks"], row["shared_read_blocks"]) == (7, 2)
        assert report["params"] == {"start_node_id": "seed9"} and report["min_rows"] == 123
        advisor.write_report(str(tmp_path), report)
        md = (tmp_path / "advisor_report.md").read_text()
        assert "| tree | 1 | 10.000 | 19.000 | 10.000 | 19.000 | 20 |" in md
        assert "- none" in md  # no Seq Scan in these plans -> nothing proposed


class TestAdviseStep:
    def test_step_wires_gate_terms_seed_settings_and_report(self, tmp_path, monkeypatch):
        from lgk.steps import advise as step

        calls = {}

        class _StepConn(_FakeConn):
            autocommit = False
            closed = False

            def execute(self, sql, params=None):
                self.log.append(sql)
                # hostile: the worst case is not the alphabetically first node
                return _Result([("node_zz",)])

            def close(self):
                self.closed = True

        conn = _StepConn()
        monkeypatch.setattr(step, "connect_branch", lambda cfg: conn)
        monkeypatch.setattr(step, "served_count_terms",
                            lambda rf, floor, folded: ("count(DISTINCT e.weight) /*GATE*/", "HAVING true /*GATE*/"))

        def fake_advise(c, served, version, queries, params, **kw):
            calls.update(conn=c, served=served, version=version, queries=queries, params=params, kw=kw,
                         autocommit=c.autocommit)
            return {"version_schema": version, "min_rows": kw["min_rows"], "params": params,
                    "proposals": [], "depths": {}}

        monkeypatch.setattr(step.advisor, "advise", fake_advise)
        cfg = {"graph_name": "g", "version": 2, "run_folder": str(tmp_path),
               "advisor_min_rows": 77, "advisor_warmup": 5, "advisor_samples": 30}
        step.advise(cfg)
        assert calls["autocommit"] is True and conn.closed
        assert calls["version"] == "lgk_g_v2" and calls["served"] == "lgk_g"
        assert calls["params"] == {"start_node_id": "node_zz"}
        assert calls["kw"] == {"min_rows": 77, "warmup": 5, "samples": 30}
        assert sorted(calls["queries"]) == ["tree", "web"]
        assert sorted(calls["queries"]["web"]) == [1, 2, 3, 4]
        assert all("/*GATE*/" in s for s in calls["queries"]["web"].values())
        assert (tmp_path / "advisor_report.json").exists()

        step.advise({**cfg, "advisor_seed_node_id": "given"})
        assert calls["params"] == {"start_node_id": "given"}


# ------------------------------------------------- real synced tables (partitioned parents)

# What EXPLAIN shows over a synced version table (seen live): the scan names the CHILD
# partition, in the system schema, never the version schema.
PART = ("__db_system", "partition_294977")
PARTITION_PLAN = {"Plan": {"Node Type": "Hash Join", "Hash Cond": "(e.src_node_id = f.node_id)",
                           "Plans": [{"Node Type": "CTE Scan", "Alias": "f"},
                                     {"Node Type": "Hash", "Plans": [_seq(*PART, "e")]}]}}


class _PartitionConn(_FakeConn):
    """Scripted catalog: partition -> root, plus an optional refusal of CREATE INDEX."""

    def __init__(self, roots, refuse=None, **kw):
        super().__init__(**kw)
        self.roots, self.refuse = roots, refuse

    def execute(self, sql, params=None):
        if "pg_partition_root" in sql:
            self.log.append(sql)
            root = self.roots.get(tuple(params))
            return _Result([root] if root else [])
        if sql.startswith("CREATE INDEX") and self.refuse:
            self.log.append(sql)
            raise self.refuse
        return super().execute(sql, params)


class TestPartitionedSyncedTables:
    def test_partition_scan_is_proposed_on_its_root_in_the_version(self):
        conn = _PartitionConn({PART: (V2, "edges")}, rows={f"{V2}.edges": 50_000})
        props = advisor.propose(conn, V2, [PARTITION_PLAN], min_rows=10_000)
        assert [p.statement for p in props] == [
            f"CREATE INDEX IF NOT EXISTS lgk_edges_src_node_id_idx ON {V2}.edges (src_node_id)"]

    def test_partition_of_the_live_version_is_never_proposed(self):
        # hostile: same partition shape, but its root is the live v1
        conn = _PartitionConn({PART: (V1, "edges")}, rows={f"{V1}.edges": 10**6})
        assert advisor.propose(conn, V2, [PARTITION_PLAN], min_rows=1) == []

    def test_index_refused_for_this_identity_is_reported_not_applied(self):
        import psycopg

        refusal = psycopg.errors.InsufficientPrivilege("must be owner of table edges")
        conn = _PartitionConn({PART: (V2, "edges")}, refuse=refusal, rows={f"{V2}.edges": 50_000})
        (p,) = advisor.propose(conn, V2, [PARTITION_PLAN], min_rows=10_000)
        got = advisor.apply(conn, p)
        assert got.applied is False and "must be owner of table edges" in got.not_applied
        md = advisor.report_markdown({"version_schema": V2, "min_rows": 1, "params": {},
                                      "proposals": [vars(got)], "depths": {}})
        assert "not applied: must be owner of table edges" in md

    def test_other_errors_are_not_swallowed(self):
        conn = _PartitionConn({PART: (V2, "edges")}, refuse=RuntimeError("disk full"),
                              rows={f"{V2}.edges": 50_000})
        (p,) = advisor.propose(conn, V2, [PARTITION_PLAN], min_rows=10_000)
        with pytest.raises(RuntimeError, match="disk full"):
            advisor.apply(conn, p)

    def test_applied_index_is_marked_applied(self):
        conn = _PartitionConn({PART: (V2, "edges")}, rows={f"{V2}.edges": 50_000})
        (p,) = advisor.propose(conn, V2, [PARTITION_PLAN], min_rows=10_000)
        assert advisor.apply(conn, p).applied is True
        assert p.statement in conn.log

    def test_scan_already_in_the_version_is_taken_as_is(self):
        # hostile (prover M2): a catalog that would map V2.edges elsewhere; never consulted
        conn = _PartitionConn({(V2, "edges"): (V1, "edges")}, rows={f"{V2}.edges": 50_000})
        props = advisor.propose(conn, V2, [TREE_PLAN], min_rows=10_000)
        assert [(p.schema, p.table) for p in props] == [(V2, "edges")]
        assert not [s for s in conn.log if "pg_partition_root" in s]

    def test_unrootable_foreign_scan_never_lands_in_the_version(self):
        # hostile (prover M12): a scan elsewhere with no root row, on a column no V2 scan has
        plan = {"Plan": _seq("other", "edges", "o", "(o.dst_node_id = 'x'::text)")}
        conn = _PartitionConn({}, rows={f"{V2}.edges": 50_000, "other.edges": 50_000})
        assert advisor.propose(conn, V2, [plan], min_rows=1) == []

    def test_refused_index_end_to_end_is_reported_and_not_reanalyzed(self):
        # prover M7a: no index made -> no second ANALYZE; the report says why
        import psycopg

        class Refusing(_PlanConn):
            def execute(self, sql, params=None):
                if sql.startswith("CREATE INDEX"):
                    self.log.append(sql)
                    raise psycopg.errors.InsufficientPrivilege("must be owner of table edges")
                return super().execute(sql, params)

        conn = Refusing(rows={f"{V2}.edges": 50_000})
        q = {"tree": {1: f"SELECT walk FROM {SERVED}.edges_view e"}}
        report = advisor.advise(conn, SERVED, V2, q, {"start_node_id": "x"})
        (p,) = report["proposals"]
        assert p["applied"] is False and "must be owner" in p["not_applied"]
        assert sum(1 for s in conn.log if s.startswith(f"ANALYZE {V2}.edges")) == 1


# ------------------------------------------------- walk direction: advisor_directions

# An inbound web hop over a version with no dst_node_id index yet (before the first reapply):
# the planner hash-joins the previous hop to a Seq Scan of edges ON dst_node_id.
INBOUND_PLAN = {"Plan": {"Node Type": "Sort", "Plans": [
    _hash_join("(e.dst_node_id = f.node_id)", {"Node Type": "CTE Scan", "Alias": "f"},
               _seq(V2, "edges", "e", "(NOT (hashed SubPlan 1))"))]}}


class TestAdvisorDirections:
    def test_inbound_plan_proposes_the_dst_index_reapply_creates(self):
        from lgk.steps.reapply import INDEXES

        conn = _FakeConn(rows={f"{V2}.edges": 50_000})
        props = advisor.propose(conn, V2, [INBOUND_PLAN], min_rows=10_000)
        assert [p.statement for p in props] == [
            f"CREATE INDEX IF NOT EXISTS lgk_edges_dst_node_id_idx ON {V2}.edges (dst_node_id)"]
        assert ("lgk_edges_dst_node_id_idx", "edges", "dst_node_id") in INDEXES  # same fixed name

    def _queries(self, monkeypatch, cfg, folded=False):
        from lgk.steps import advise as step

        monkeypatch.setattr(step, "served_count_terms", lambda rf, fl, fo: ("COUNT(*)", ""))
        monkeypatch.setattr(step, "served_walk_policy", lambda rf, fl, fo: (11, folded))
        return step._walk_queries(cfg, "lgk_g")

    def test_default_measures_outbound_only(self, monkeypatch):
        q = self._queries(monkeypatch, {"run_folder": "rf"})
        assert sorted(q) == ["tree", "web"]
        assert "JOIN hop0 f ON e.src_node_id = f.node_id" in q["web"][1]

    def test_each_listed_direction_is_measured(self, monkeypatch):
        # hostile: reordered list, and a direction listed once per shape
        q = self._queries(monkeypatch, {"run_folder": "rf", "advisor_directions": ["both", "inbound", "outbound"]})
        assert sorted(q) == ["tree", "tree:both", "tree:inbound", "web", "web:both", "web:inbound"]
        assert sorted(q["web:inbound"]) == [1, 2, 3, 4]
        assert "JOIN hop0 f ON e.dst_node_id = f.node_id" in q["web:inbound"][1]
        assert "e.src_node_id = f.node_id" not in q["web:inbound"][1]
        assert "WHERE e.dst_node_id = ANY(w.frontier)" in q["tree:inbound"][2]
        assert q["web:both"][2].count("UNION\n") == 2
        assert "JOIN hop0 f ON e.src_node_id = f.node_id" in q["web"][1]  # outbound stays the bare key

    def test_a_folded_gate_refuses_an_inbound_advisor_walk(self, monkeypatch):
        with pytest.raises(ValueError, match="folded edges are symmetric"):
            self._queries(monkeypatch, {"run_folder": "rf", "advisor_directions": ["inbound"]}, folded=True)
        assert sorted(self._queries(monkeypatch, {"run_folder": "rf"}, folded=True)) == ["tree", "web"]


# ------------------------------------------------- the opt-in web_set shape
def test_advise_measures_web_set_with_the_worst_case_seed_value(monkeypatch, tmp_path):
    import yaml

    from lgk.steps import advise as step

    mapping = {"source_catalog": "c", "source_schema": "s",
               "nodes": {"provider": {"source_table": "p", "natural_key": "id", "label": "name",
                                      "properties": ["zone", "specialty"], "walk_keys": ["specialty", "zone"]},
                         # hostile: a type sorting first with no walk keys is skipped
                         "clinic": {"source_table": "c", "natural_key": "id", "label": "name"}},
               "edges": {}}
    (tmp_path / "graph_mapping.yaml").write_text(yaml.safe_dump(mapping, sort_keys=False))
    calls, log = {}, []

    class _C:
        autocommit = False

        def execute(self, sql, params=None):
            log.append(sql)
            return type("R", (), {"fetchone": lambda s: ("cardiology",) if "properties ->>" in sql else ("n1",)})()

        def close(self):
            pass

    monkeypatch.setattr(step, "connect_branch", lambda cfg: _C())
    monkeypatch.setattr(step, "served_count_terms", lambda rf, fl, fo: ("COUNT(*)", ""))
    monkeypatch.setattr(step.advisor, "advise",
                        lambda c, served, version, queries, params, **kw: calls.update(q=queries, p=params)
                        or {"proposals": [], "depths": {}, "version_schema": version})
    monkeypatch.setattr(step.advisor, "write_report", lambda rf, r: None)
    step.advise({"graph_name": "g", "version": 2, "run_folder": str(tmp_path), "advisor_shapes": ["web_set"]})
    assert sorted(calls["q"]) == ["web_set"] and sorted(calls["q"]["web_set"]) == [1, 2, 3, 4]
    assert all("n.node_type = 'provider' AND n.properties ->> 'specialty' = %(seed_value)s" in q
               for q in calls["q"]["web_set"].values())
    assert calls["p"] == {"start_node_id": "n1", "seed_value": "cardiology"}
    worst = [s for s in log if "properties ->>" in s]
    assert worst == ["SELECT v FROM (SELECT n.properties ->> 'specialty' AS v, count(DISTINCT n.node_id) AS "
                     "seeds, count(e.src_node_id) AS out_edges FROM lgk_g_v2.nodes n LEFT JOIN lgk_g_v2.edges e "
                     "ON e.src_node_id = n.node_id WHERE n.node_type = 'provider' AND n.properties ->> "
                     "'specialty' IS NOT NULL GROUP BY 1) x WHERE seeds <= 200 "
                     'ORDER BY out_edges DESC, v COLLATE "C" LIMIT 1']
    # the set walk is outbound only: another advisor direction measures no web_set (and does not crash)
    step.advise({"graph_name": "g", "version": 2, "run_folder": str(tmp_path), "advisor_shapes": ["web_set"],
                 "advisor_directions": ["outbound", "inbound"]})
    assert sorted(calls["q"]) == ["web_set"]
    # the default shapes do not change: web_set is opt-in
    assert step.DEFAULT_SHAPES == ("web", "tree")
    import pytest as _pt
    with _pt.raises(ValueError, match="needs a node type with walk_keys"):
        step.web_set_seed({"clinic": ()})
