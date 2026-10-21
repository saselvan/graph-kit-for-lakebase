"""Live tests for the index advisor on a fresh branch of the test project.

Setup: the provider graph is built and provisioned as version 1 (live: the served views read
it). On the same branch, the test then creates NEW, not-yet-served version schemas as plain
Postgres tables, the way a next build's version looks before the switch:
- v2: a LARGE edges table (v1's edges + SYNTH_EDGES synthetic rows), no index at all.
  # hostile: a large edges table with no index on src_node_id
- v3: a SMALL edges table (v1's edges only, ~32 rows). # hostile: small table, must get nothing
- v4, v5: copies of v2 for the threshold-parameter and warm-up-count tests.
The branch (and these schemas) are deleted by provisioned_graph's teardown; the test drops
its own schemas first.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from lgk import advisor, versions
from lgk.steps.advise import advise
from lgk.steps.reapply import ensure_indexes
from tests.test_build_provision_walks import _create_provider_fixture, _pg, provisioned_graph

SYNTH_EDGES = 30_000
SYNTH_NODES = 3_000
EXTRA_VERSIONS = (2, 3, 4, 5)


def _make_version(conn: Any, v1: str, target: str, large: bool) -> None:
    conn.execute(f"CREATE SCHEMA {target}")
    conn.execute(f"CREATE TABLE {target}.edges AS SELECT * FROM {v1}.edges")
    if large:
        conn.execute(
            f"INSERT INTO {target}.edges (src_node_id, dst_node_id) "
            f"SELECT 'syn_' || (g % {SYNTH_NODES}), 'syn_' || ((g * 7 + 1) % {SYNTH_NODES}) "
            f"FROM generate_series(0, {SYNTH_EDGES - 1}) g")


@pytest.fixture(scope="module")
def adv_graph(workspace_client, lakebase_project, lakebase_catalog, run_folder):
    with provisioned_graph(workspace_client, lakebase_project, lakebase_catalog, run_folder,
                           "advisor", _create_provider_fixture) as config:
        v1 = versions.version_schema(config)
        schemas = {n: versions.version_schema(config, n) for n in EXTRA_VERSIONS}
        conn = _pg(config)
        conn.autocommit = True
        try:
            for n, schema in schemas.items():
                _make_version(conn, v1, schema, large=(n != 3))
            yield {**config, "versions": schemas}
        finally:
            for schema in schemas.values():
                conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            conn.close()


def _indexes(conn: Any, schema: str) -> List[tuple]:
    return conn.execute(
        "SELECT tablename, indexname, indexdef FROM pg_indexes WHERE schemaname = %s ORDER BY 2",
        (schema,)).fetchall()


def _src_index_count(conn: Any, schema: str) -> int:
    return conn.execute(
        "SELECT count(*) FROM pg_index i JOIN pg_class t ON t.oid = i.indrelid "
        "JOIN pg_namespace n ON n.oid = t.relnamespace "
        "JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = i.indkey[0] "
        "WHERE n.nspname = %s AND t.relname = 'edges' AND a.attname = 'src_node_id'",
        (schema,)).fetchone()[0]


def _cfg(g: Dict[str, Any], version: int, **kw) -> Dict[str, Any]:
    return {**{k: v for k, v in g.items() if k != "versions"}, "version": version, **kw}


@pytest.mark.seam1
@pytest.mark.provision
class TestAdvisorLive:
    def test_large_unindexed_edges_gets_src_index_with_before_after(self, adv_graph):
        """All defaults: the seq scan on the large v2 edges gets lgk_edges_src_node_id_idx."""
        g = adv_graph
        v1, v2 = versions.version_schema(g), g["versions"][2]
        with _pg(g) as conn:
            v1_before = _indexes(conn, v1)
            assert _src_index_count(conn, v2) == 0
        report = advise(_cfg(g, 2))
        assert [p["statement"] for p in report["proposals"]] == [
            f"CREATE INDEX IF NOT EXISTS lgk_edges_src_node_id_idx ON {v2}.edges (src_node_id)"]
        assert report["version_schema"] == v2
        for shape in ("web", "tree"):
            assert sorted(int(d) for d in report["depths"][shape]) == [1, 2, 3, 4]
            for d, row in report["depths"][shape].items():
                assert any(s["table"] == "edges" and s["schema"] == v2
                           for s in row["before"]["seq_scans"]), f"{shape} d{d}: no seq scan"
                for phase in ("before", "after"):
                    assert row[phase]["samples"] == advisor.DEFAULT_SAMPLES
                    assert 0 < row[phase]["p50_ms"] <= row[phase]["p95_ms"]
        # web depth 1 from one seed: after the index there is no seq scan on edges
        assert not [s for s in report["depths"]["web"]["1"]["after"]["seq_scans"]
                    if s["table"] == "edges"]
        with _pg(g) as conn:
            assert _src_index_count(conn, v2) == 1
            assert _indexes(conn, v1) == v1_before  # the live version is untouched
            assert versions.live_version_schemas(conn, g["served_schema"]) == {v1}
        rf = Path(g["run_folder"])
        assert json.loads((rf / "advisor_report.json").read_text())["proposals"] == report["proposals"]
        md = (rf / "advisor_report.md").read_text()
        assert "lgk_edges_src_node_id_idx" in md and "p95" in md and v2 in md

    def test_rerun_proposes_nothing_and_reapply_adds_nothing(self, adv_graph):
        g = adv_graph
        v2 = g["versions"][2]
        first = advise(_cfg(g, 2))  # idempotent whether or not the test above ran
        again = advise(_cfg(g, 2))
        assert again["proposals"] == []
        with _pg(g) as conn:
            conn.autocommit = True
            before = _indexes(conn, v2)
            ensure_indexes(conn, v2)  # reapply's fixed names: the src index is already there
            assert _src_index_count(conn, v2) == 1
            for p in first["proposals"]:
                conn.execute(p["statement"])  # IF NOT EXISTS: a re-run is a no-op
            assert _src_index_count(conn, v2) == 1
            assert [r for r in _indexes(conn, v2) if r[1] == "lgk_edges_src_node_id_idx"] == \
                [r for r in before if r[1] == "lgk_edges_src_node_id_idx"]

    def test_small_table_gets_no_proposal(self, adv_graph):
        g = adv_graph
        v3 = g["versions"][3]
        report = advise(_cfg(g, 3))
        assert any(s["schema"] == v3 for row in report["depths"]["web"].values()
                   for s in row["before"]["seq_scans"]), "non-vacuous: v3 is seq scanned"
        assert report["proposals"] == []
        with _pg(g) as conn:
            assert _indexes(conn, v3) == []

    def test_min_rows_parameter_decides_large(self, adv_graph):
        g = adv_graph
        v4 = g["versions"][4]
        with _pg(g) as conn:
            n = conn.execute(f"SELECT count(*) FROM {v4}.edges").fetchone()[0]
        assert advise(_cfg(g, 4, advisor_min_rows=n + 1))["proposals"] == []
        with _pg(g) as conn:
            assert _src_index_count(conn, v4) == 0
        props = advise(_cfg(g, 4, advisor_min_rows=n))["proposals"]
        assert [(p["table"], p["column"]) for p in props] == [("edges", "src_node_id")]

    def test_refuses_the_live_version(self, adv_graph):
        """Isolation: pointed at the live schema (v1), it refuses and changes nothing."""
        g = adv_graph
        v1 = versions.version_schema(g)
        rf = Path(g["run_folder"])
        json_before = (rf / "advisor_report.json").read_text() if (rf / "advisor_report.json").exists() else None
        with _pg(g) as conn:
            before = _indexes(conn, v1)
        with pytest.raises(versions.LiveVersionError):
            advise(_cfg(g, 1, advisor_min_rows=1))
        with _pg(g) as conn:
            assert _indexes(conn, v1) == before
        after = (rf / "advisor_report.json").read_text() if (rf / "advisor_report.json").exists() else None
        assert after == json_before

    def test_warmup_runs_before_every_timed_sample(self, adv_graph):
        """Each depth, each phase: `warmup` real executions precede the first timed sample."""
        g = adv_graph
        v5 = g["versions"][5]
        from lgk.walks.sql import per_depth_sql

        queries = {"web": per_depth_sql("web", 4, served_schema=g["served_schema"], top_n=25,
                                        folded=False)}
        with _pg(g) as raw:
            raw.autocommit = True
            seed = raw.execute(f"SELECT src_node_id FROM {v5}.edges GROUP BY 1 "
                               "ORDER BY count(*) DESC, 1 LIMIT 1").fetchone()[0]
            conn = _CountingConn(raw)
            report = advisor.advise(conn, g["served_schema"], v5, queries,
                                    {"start_node_id": seed}, min_rows=advisor.DEFAULT_MIN_ROWS,
                                    warmup=4, samples=advisor.MIN_SAMPLES)
        assert report["proposals"], "non-vacuous: an index is applied between the two phases"
        version_sqls = advisor.version_queries(queries, g["served_schema"], v5)["web"]
        for depth, sql in version_sqls.items():
            runs = conn.runs_of(sql)
            assert [k for k in runs if k == "sample"].__len__() == 2 * advisor.MIN_SAMPLES
            for phase in (0, 1):
                block = _phase(runs, phase)
                first_sample = block.index("sample")
                assert block[:first_sample].count("warm") == 4, f"depth {depth} phase {phase}: {block}"
            assert report["depths"]["web"][str(depth)]["before"]["warmup_runs"] == 4


def _phase(runs: List[str], phase: int) -> List[str]:
    """Split a query's run kinds into phases: a phase ends after its last consecutive sample."""
    blocks, cur = [], []
    for i, k in enumerate(runs):
        cur.append(k)
        if k == "sample" and (i + 1 == len(runs) or runs[i + 1] != "sample"):
            blocks.append(cur)
            cur = []
    return blocks[phase]


class _CountingConn:
    """Wraps the real connection; records how each walk SQL was executed (plain or timed)."""

    def __init__(self, conn: Any):
        self._conn, self.log = conn, []

    def execute(self, sql: str, params: Any = None):
        self.log.append(sql)
        return self._conn.execute(sql, params)

    def runs_of(self, walk_sql: str) -> List[str]:
        kinds = []
        for s in self.log:
            if s == walk_sql:
                kinds.append("warm")
            elif s.endswith(walk_sql) and "TIMING OFF" in s:
                kinds.append("sample")
            elif s.endswith(walk_sql) and "BUFFERS" in s:
                kinds.append("explain")
        return kinds

    def __getattr__(self, name):
        return getattr(self._conn, name)
