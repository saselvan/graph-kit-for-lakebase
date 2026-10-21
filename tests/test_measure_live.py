"""Live acceptance for the measure step: a provisioned graph on a throwaway t- branch.

One provider graph (shared `provisioned_graph`: fresh t- branch, ttl 14400 s, no CU change,
full teardown) with enrich degree + similarity, so web, tree and similar are all served. The
SOURCE fixture is hostile for the reference:
- # hostile: duplicate affiliation row (support must stay 1)
- # hostile: orphan edge to a provider that has no provider row, and a NULL hospital_id
- # hostile: self-loop, a 3-cycle, and a hub with 150 leaves (ties cut by byte order)
- # hostile: a recognisable provider key and label that must never reach the record

Then measure runs through the runner. Every seed must match (C10), latency lines carry the
default 200 samples (C13), freshness is pending (C30b). Then the falsifiability check: the
hub's out-edges are deleted from the version's edges table on this branch only (the served
graph is now wrong) and measure, rerun, must count a difference.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from lgk.uc_sql import execute_sql
from lgk.versions import version_schema
from tests.test_build_provision_walks import WAREHOUSE_ID, _pg, provisioned_graph

PLANTED_KEY, PLANTED_LABEL = "p_zebra_planted_key", "Zebra Planted Label 7731"
N_WEB, N_LEAVES = 120, 150


def _create_measure_fixture(w: Any, catalog: str, fixture_schema: str) -> None:
    s = f"{catalog}.{fixture_schema}"
    execute_sql(w, WAREHOUSE_ID, f"CREATE SCHEMA IF NOT EXISTS {s}")
    execute_sql(w, WAREHOUSE_ID, f"CREATE OR REPLACE TABLE {s}.fixture_providers "
                "(provider_id STRING, name STRING, specialty STRING) USING DELTA")
    keys = [f"w_{i:03d}" for i in range(N_WEB)] + ["hub", "c_a", "c_b", "c_c", PLANTED_KEY]
    execute_sql(w, WAREHOUSE_ID, f"INSERT INTO {s}.fixture_providers VALUES "
                + ", ".join(f"('{k}', '{PLANTED_LABEL if k == PLANTED_KEY else 'N ' + k}', 'x')" for k in keys))
    execute_sql(w, WAREHOUSE_ID, f"INSERT INTO {s}.fixture_providers SELECT "
                f"concat('leaf_', lpad(CAST(id AS STRING), 3, '0')), 'leaf', 'x' FROM range({N_LEAVES})")
    execute_sql(w, WAREHOUSE_ID, f"CREATE OR REPLACE TABLE {s}.fixture_affiliations "
                "(affiliation_id STRING, provider_id STRING, hospital_id STRING, start_date STRING) USING DELTA")
    pairs = [(f"w_{i:03d}", f"w_{j % N_WEB:03d}") for i in range(N_WEB) for j in (i + 1, i * 7 + 3, i * 13 + 11)]
    pairs += [(f"w_{i:03d}", "hub") for i in range(10)]
    pairs += [("c_a", "c_b"), ("c_b", "c_c"), ("c_c", "c_a")]  # hostile: cycle
    pairs += [("w_000", "w_001")]  # hostile: duplicate row
    pairs += [("w_002", "ghost_provider")]  # hostile: orphan (no provider row)
    pairs += [("w_003", "w_003")]  # hostile: self-loop
    pairs += [("w_004", PLANTED_KEY), (PLANTED_KEY, "w_005")]
    values = [f"('a{n}', '{a}', '{b}', '2024-01-01')" for n, (a, b) in enumerate(pairs)]
    values.append("('anull', 'w_006', NULL, '2024-01-01')")  # hostile: NULL FK
    execute_sql(w, WAREHOUSE_ID, f"INSERT INTO {s}.fixture_affiliations VALUES " + ", ".join(values))
    execute_sql(w, WAREHOUSE_ID, f"INSERT INTO {s}.fixture_affiliations SELECT concat('h', id), 'hub', "
                f"concat('leaf_', lpad(CAST(id AS STRING), 3, '0')), '2024-01-01' FROM range({N_LEAVES})")


@pytest.fixture(scope="module")
def measure_graph(workspace_client, lakebase_project, lakebase_catalog, run_folder):
    with provisioned_graph(workspace_client, lakebase_project, lakebase_catalog, run_folder,
                           "measure", _create_measure_fixture, enrich_methods=["degree", "similarity"],
                           accept_orphans=[("fixture_affiliations", "hospital_id")]) as config:
        yield config


def _lines(run_folder: str):
    return [json.loads(x) for x in (Path(run_folder) / "measure.jsonl").read_text().splitlines()]


def _run(config):
    from lgk.notebook_runner import resolve_step
    from lgk.runner import LaptopRunner

    runner = LaptopRunner(config)
    runner.invalidate_steps(["measure"], "measure again")
    [res] = runner.run_steps([("measure", resolve_step("lgk.steps.measure:measure"))])
    assert res.status.value == "success", res.error
    return _lines(config["run_folder"])


@pytest.mark.seam1
@pytest.mark.provision
def test_measure_live_records_c10_c13_c23_c30b_and_catches_a_broken_served_graph(measure_graph):
    cfg = {**measure_graph, "measure_seeds": 20}
    lines = _run(cfg)
    print("\nMEASURE_LINE " + json.dumps(next(x for x in lines if x["metric"] == "walk_latency")))
    print("MEASURE_LINE " + json.dumps(next(x for x in lines if x["metric"] == "walk_correctness")))

    correctness = [x for x in lines if x["metric"] == "walk_correctness"]
    assert {(x["shape"], x["depth"]) for x in correctness} == (
        {("web", d) for d in range(1, 5)} | {("tree", d) for d in range(1, 5)} | {("similar", 1)})
    for x in correctness:
        assert x["status"] == "checked", x
        assert x["seeds_checked"] == 20 and x["seeds_different"] == 0, x

    latency = [x for x in lines if x["metric"] == "walk_latency"]
    assert len(latency) == 9
    assert all(x["n_samples"] == 200 and x["timing"] == "client" and x["value"] > 0 for x in latency)
    assert all(x["execution_p95_ms"] <= x["client_p95_ms"] == x["value"] for x in latency)
    [f] = [x for x in lines if x["metric"] == "freshness_lag"]
    assert f["freshness"]["status"] == "pending"
    [t] = [x for x in lines if x["metric"] == "time_to_first_served_graph"]
    assert t["timing"] == "step" and t["status"] in ("complete", "partial")
    assert all(x["nodes"] == N_WEB + 4 + 1 + N_LEAVES for x in lines)
    run = Path(cfg["run_folder"])
    for name in ("measure.jsonl", "measure_report.md"):
        text = (run / name).read_text()
        assert PLANTED_KEY not in text and PLANTED_LABEL not in text and "ghost_provider" not in text

    # falsifiability: break the served graph on THIS branch only, then the record must disagree
    version = version_schema(cfg)
    with _pg(cfg) as conn:
        conn.execute(f"DELETE FROM {version}.edges WHERE src_node_id = (SELECT src_node_id FROM "
                     f"{version}.edges GROUP BY 1 ORDER BY count(*) DESC LIMIT 1)")
    broken = _run({**cfg, "measure_samples": 20, "measure_shapes": ["web"]})
    web1 = next(x for x in broken if x["metric"] == "walk_correctness" and x["depth"] == 1)
    print("MEASURE_BROKEN " + json.dumps({k: web1[k] for k in ("seeds_checked", "seeds_equal", "seeds_different")}))
    assert web1["seeds_different"] >= 1
