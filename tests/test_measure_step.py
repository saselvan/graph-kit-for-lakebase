"""Offline tests for the measure step end to end (lgk.steps.measure), fakes only.

The privacy gate is cleared the legitimate offline way (tests.privacy_support). Served answers
are written by hand per (SQL, seed); the warehouse returns the hostile SOURCE rows of
tests.test_measure. A recognisable natural key, label and node id are planted in the fakes and
must never reach measure.jsonl or measure_report.md.
"""

import json
from pathlib import Path

import pytest

from lgk.mapping import dump_mapping
from lgk.measure import record
from lgk.privacy.floor import walk_count_terms
from lgk.steps import measure as step
from lgk.walks.sql import tree_sql, web_sql
from tests.privacy_support import clear_gate_offline
from tests.test_measure import EDGE_ROWS, MAPPING, NODE_ROWS, SERVED, FakeWarehouse, ServedConn

# hostile: a recognisable id that would be a personal-data leak if it reached the output
PLANTED = "n9"
SEEDS = ("n4", "n1")  # pick_seeds(graph, 2): the hub, then the first of the rest by byte order

WEB = {
    "n4": [(1, "n10", 1, None), (1, "n11", 1, None), (1, "n9", 1, None)],
    "n1": [(1, "n2", 1, None), (1, "n3", 1, None)],
}
TREE = {
    "n4": [("n10", 1, None), ("n11", 1, None), ("n9", 1, None)],
    "n1": [("n2", 1, None), ("n3", 1, None)],
}


def _answers(web=WEB, tree=TREE):
    w1 = web_sql(depth=1, served_schema=SERVED, top_n=25, count_terms=walk_count_terms(11, False))
    t1 = tree_sql(depth=1, served_schema=SERVED, top_n=25)
    out = {(w1, s): rows for s, rows in web.items()}
    out.update({(t1, s): rows for s, rows in tree.items()})
    return out


def _config(tmp_path: Path, **kw):
    run = tmp_path / "run"
    run.mkdir()
    dump_mapping(MAPPING, run / "graph_mapping.yaml")
    clear_gate_offline(str(run))
    return {"graph_name": "g", "run_folder": str(run), "served_schema": SERVED,
            "lakebase_project": "p", "branch_id": "t-measure", "warehouse_id": "wh",
            "measure_shapes": ["web", "tree"], "measure_max_depth": 1, "measure_seeds": 2,
            "measure_samples": 20, "measure_warmup": 1, **kw}


def _deps(conn, wh=None, cu=None):
    wh = wh or FakeWarehouse({"nodes:prov": NODE_ROWS, "edges:aff": EDGE_ROWS})
    return step.MeasureDeps(connect=lambda cfg: conn, warehouse_sql=lambda cfg: wh,
                            endpoint_cu=lambda cfg: cu if cu is not None else {"min": 0.5, "max": 4.0},
                            clock=conn.now)


def _lines(run_folder):
    return [json.loads(x) for x in (Path(run_folder) / "measure.jsonl").read_text().splitlines()]


def _by(lines, metric, shape=None, depth=None):
    return [x for x in lines if x["metric"] == metric and (shape is None or x.get("shape") == shape)
            and (depth is None or x.get("depth") == depth)]


class TestMeasureStep:
    def test_correct_served_graph_is_all_equal_per_shape_and_depth(self, tmp_path):
        cfg = _config(tmp_path)
        step.run_measure(cfg, _deps(ServedConn(_answers())))
        lines = _lines(cfg["run_folder"])
        for shape in ("web", "tree"):
            [c] = _by(lines, "walk_correctness", shape, 1)
            assert (c["seeds_checked"], c["seeds_equal"], c["seeds_different"]) == (2, 2, 0)
            # the record names the Python reference over the source tables, never "spark"
            assert c["status"] == "checked" and c["reference"] == "python_over_source"

    def test_a_served_answer_with_an_orphan_node_is_counted_different(self, tmp_path):
        cfg = _config(tmp_path)
        bad = {**WEB, "n1": WEB["n1"] + [(1, "nX", 1, None)]}  # hostile: kept orphan
        step.run_measure(cfg, _deps(ServedConn(_answers(web=bad))))
        [c] = _by(_lines(cfg["run_folder"]), "walk_correctness", "web", 1)
        assert (c["seeds_equal"], c["seeds_different"]) == (1, 1)

    def test_one_client_timed_latency_line_per_walk_with_n_samples(self, tmp_path):
        # the timer's numbers are pinned in tests/test_measure_timing.py
        cfg = _config(tmp_path)
        step.run_measure(cfg, _deps(ServedConn(_answers(), times=[0.0] * 1000)))
        lines = _lines(cfg["run_folder"])
        for shape in ("web", "tree"):
            [x] = _by(lines, "walk_latency", shape, 1)
            assert x["unit"] == "ms" and x["n_samples"] == 20 and x["timing"] == "client"

    def test_every_line_carries_build_size_cu_and_ts(self, tmp_path):
        cfg = _config(tmp_path)
        step.run_measure(cfg, _deps(ServedConn(_answers(), counts=(7, 8)), cu={"min": 0.5, "max": 2.0}))
        for x in _lines(cfg["run_folder"]):
            assert x["nodes"] == 7 and x["edges"] == 8
            assert x["endpoint_cu"] == {"min": 0.5, "max": 2.0}
            assert set(x["build"]) == {"version", "commit", "source_sha256"}
            assert x["ts"]

    def test_first_serve_and_freshness_lines_are_written(self, tmp_path):
        cfg = _config(tmp_path)
        step.run_measure(cfg, _deps(ServedConn(_answers())))
        lines = _lines(cfg["run_folder"])
        [t] = _by(lines, "time_to_first_served_graph")
        assert t["unit"] == "s" and t["timing"] == "step" and t["excludes"] == "operator_time"
        [f] = _by(lines, "freshness_lag")
        assert f["freshness"]["status"] == "pending"

    def test_similar_without_the_similarity_enrichment_is_not_served(self, tmp_path):
        cfg = _config(tmp_path, measure_shapes=["web", "tree", "similar"])
        step.run_measure(cfg, _deps(ServedConn(_answers())))
        [s] = _by(_lines(cfg["run_folder"]), "walk_correctness", "similar")
        assert s["status"] == "not_served" and s["seeds_checked"] == 0

    def test_no_planted_key_label_or_id_reaches_the_output(self, tmp_path):
        cfg = _config(tmp_path)
        step.run_measure(cfg, _deps(ServedConn(_answers(), worst_seed="-7771234567890123456")))
        run = Path(cfg["run_folder"])
        for name in ("measure.jsonl", "measure_report.md"):
            text = (run / name).read_text()
            for planted in ("-7771234567890123456", '"n4"', '"n1"', '"n10"', "nX"):
                assert planted not in text, (name, planted)

    def test_samples_below_the_minimum_are_refused_before_any_connection(self, tmp_path):
        cfg = _config(tmp_path, measure_samples=5)
        opened = []
        deps = step.MeasureDeps(connect=lambda c: opened.append(c), warehouse_sql=lambda c: None,
                                endpoint_cu=lambda c: None)
        with pytest.raises(ValueError, match="samples"):
            step.run_measure(cfg, deps)
        assert opened == []

    def test_defaults_measure_web_and_tree_to_depth_4_with_200_samples(self):
        d = step.settings({"run_folder": "x"})
        assert d["shapes"] == ("web", "tree", "similar")
        assert d["max_depth"] == 4 and d["samples"] == 200 and d["seeds"] >= 10

    def test_rerun_replaces_the_record(self, tmp_path):
        cfg = _config(tmp_path)
        step.run_measure(cfg, _deps(ServedConn(_answers())))
        n = len(_lines(cfg["run_folder"]))
        step.run_measure(cfg, _deps(ServedConn(_answers())))
        assert len(_lines(cfg["run_folder"])) == n

    def test_runs_through_the_runner_as_a_step(self, tmp_path, monkeypatch):
        from lgk.notebook_runner import resolve_step
        from lgk.runner import LaptopRunner

        cfg = _config(tmp_path)
        monkeypatch.setattr(step, "default_deps", lambda: _deps(ServedConn(_answers())))
        [res] = LaptopRunner(cfg).run_steps([("measure", resolve_step("lgk.steps.measure:measure"))])
        assert res.status.value == "success", res.error
        assert (Path(cfg["run_folder"]) / "measure.jsonl").exists()

    def test_the_freshness_reason_is_the_record_constant(self):
        assert record.FRESHNESS["reason"] == "nightly refresh untested from inside a workspace at this build"
