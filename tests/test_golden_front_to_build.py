"""Seam-1 golden run: profile -> propose -> build, no hand edits.

A per-run UC fixture schema `lgk_v1_<6hex>` holds two node tables with declared
(informational) primary keys and a link table whose foreign-key columns carry declared
FOREIGN KEY constraints. The run goes through the laptop runner with nothing but config:
profile reads keys/FKs/counts from UC, propose drafts the mapping, build reads it.

Hostile rows in the link table (each would inflate the edge count if mishandled):
- a duplicate link row (same provider/org pair as the first row)
- an orphan FK (provider 'prov_999' does not exist)
- a NULL org FK
Hostile column: providers.home_org_id looks like an FK but has no constraint, so it must
NOT become an edge (edges come from declared FKs, not column names).
"""

import json
import secrets
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from lgk.privacy.flags import resolve_flag
from lgk.runner import LaptopRunner, RunStatus
from lgk.steps.build import build
from lgk.steps.privacy import privacy
from lgk.steps.profile import profile
from lgk.steps.propose import propose
from lgk.uc_sql import execute_sql
from tests.test_build_provision_walks import WAREHOUSE_ID, _teardown_live

pytestmark = pytest.mark.seam1

PROVIDERS = [f"prov_{i}" for i in range(6)]
ORGS = ["org_a", "org_b", "org_c"]
GOOD_PAIRS = [
    ("prov_0", "org_a"), ("prov_0", "org_b"), ("prov_1", "org_a"), ("prov_2", "org_c"),
    ("prov_3", "org_b"), ("prov_4", "org_c"), ("prov_5", "org_a"),
]
N_NODES = len(PROVIDERS) + len(ORGS)  # 9
N_EDGES = len(GOOD_PAIRS)  # 7
N_LINK_ROWS = len(GOOD_PAIRS) + 3  # + duplicate, orphan, null FK


def _create_fixture(w: Any, schema: str) -> None:
    q = lambda s: execute_sql(w, WAREHOUSE_ID, s)  # noqa: E731
    q(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    q(f"""CREATE TABLE {schema}.providers (
            provider_id STRING NOT NULL, name STRING, specialty STRING, home_org_id STRING,
            CONSTRAINT providers_pk PRIMARY KEY (provider_id)) USING DELTA""")
    q(f"""CREATE TABLE {schema}.organizations (
            org_id STRING NOT NULL, org_name STRING,
            CONSTRAINT organizations_pk PRIMARY KEY (org_id)) USING DELTA""")
    q(f"""CREATE TABLE {schema}.affiliations (
            provider_id STRING, org_id STRING, since STRING,
            CONSTRAINT fk_a_provider FOREIGN KEY (provider_id) REFERENCES {schema}.providers,
            CONSTRAINT fk_b_org FOREIGN KEY (org_id) REFERENCES {schema}.organizations)
          USING DELTA""")
    prov = ", ".join(f"('{p}', 'Dr {p}', 'Cardiology', 'org_a')" for p in PROVIDERS)
    q(f"INSERT INTO {schema}.providers VALUES {prov}")
    orgs = ", ".join(f"('{o}', 'Org {o}')" for o in ORGS)
    q(f"INSERT INTO {schema}.organizations VALUES {orgs}")
    links = [f"('{p}', '{o}', '2024-01-01')" for p, o in GOOD_PAIRS]
    links.append("('prov_0', 'org_a', '2025-06-01')")  # hostile: duplicate link row
    links.append("('prov_999', 'org_b', '2024-01-01')")  # hostile: orphan FK
    links.append("('prov_1', NULL, '2024-01-01')")  # hostile: NULL FK
    q(f"INSERT INTO {schema}.affiliations VALUES {', '.join(links)}")


@pytest.fixture(scope="module")
def golden(workspace_client, lakebase_catalog):
    suffix = secrets.token_hex(3)
    config: Dict[str, Any] = {
        "graph_name": f"golden_{suffix}",
        "lakebase_catalog": lakebase_catalog,
        "kit_schema": f"lgk_v1_{suffix}_kit",
        "fixture_schema": f"lgk_v1_{suffix}",
        "source_catalog": lakebase_catalog,
        "source_schema": f"lgk_v1_{suffix}",
        "warehouse_id": WAREHOUSE_ID,
        "workspace_client": workspace_client,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            config["run_folder"] = str(Path(tmpdir) / f"golden_{suffix}")
            _create_fixture(workspace_client, f"{lakebase_catalog}.{config['fixture_schema']}")
            steps = [("profile", profile), ("propose", propose), ("privacy", privacy), ("build", build)]
            runner = LaptopRunner(config)
            first = runner.run_steps(steps)
            # affiliations.since holds date strings: a care-subject value pattern,
            # so the gate stops the run until the operator decides (here: drop it).
            assert [(r.step, r.status) for r in first][-1] == ("privacy", RunStatus.FAILED), \
                [r.to_dict() for r in first]
            assert "affiliations.since" in first[-1].error
            resolve_flag(config["run_folder"], "affiliations", "since", "drop", chooser="op")
            # The planted orphan (prov_999) needs the operator's decision before build.
            from lgk.keys import resolve_orphans

            resolve_orphans(config["run_folder"], "affiliations", "provider_id", "accept", chooser="op",
                            reason="planted orphan in the golden fixture")
            results = runner.run_steps(steps)
            yield config, results
        finally:
            # No branch and no synced tables in this run: the shared helper skips the
            # branch steps and drops the UC schemas.
            _teardown_live(workspace_client, None, config)


def _count(w: Any, sql: str) -> int:
    return int(execute_sql(w, WAREHOUSE_ID, sql)[0][0])


def test_golden_steps_all_succeed(golden):
    _, results = golden
    assert [(r.step, r.status) for r in results] == [
        ("profile", RunStatus.SUCCESS), ("propose", RunStatus.SUCCESS),
        ("privacy", RunStatus.SUCCESS), ("build", RunStatus.SUCCESS),
    ], [r.to_dict() for r in results]


def test_profile_reads_real_keys_fks_and_counts(golden):
    config, _ = golden
    schema = json.loads((Path(config["run_folder"]) / "schema.json").read_text())
    tables = {t["name"]: t for t in schema["tables"]}
    assert tables["providers"]["key_columns"] == ["provider_id"]
    assert tables["affiliations"]["key_columns"] == []
    assert tables["affiliations"]["row_count"] == N_LINK_ROWS
    assert tables["affiliations"]["foreign_keys"] == [
        {"columns": ["provider_id"], "references": ["providers", "provider_id"], "orphan_rows": 1},  # prov_999
        {"columns": ["org_id"], "references": ["organizations", "org_id"], "orphan_rows": 0},
    ]
    assert tables["providers"]["foreign_keys"] == []  # hostile: home_org_id has no constraint
    org_col = {c["name"]: c for c in tables["affiliations"]["columns"]}["org_id"]
    assert org_col["null_rate"] == pytest.approx(1 / N_LINK_ROWS)


def test_golden_builds_nodes_and_edges(golden, workspace_client):
    config, _ = golden
    w = workspace_client
    t = f"{config['lakebase_catalog']}.{config['kit_schema']}"
    assert _count(w, f"SELECT COUNT(*) FROM {t}.nodes") == N_NODES
    assert _count(w, f"SELECT COUNT(*) FROM {t}.nodes WHERE node_type = 'providers'") == len(PROVIDERS)
    assert _count(w, f"SELECT COUNT(*) FROM {t}.edges") == N_EDGES
    # Every edge resolves to a provider node on one side and an org node on the other.
    resolved = _count(w, f"""
        SELECT COUNT(*) FROM {t}.edges e
        JOIN {t}.nodes s ON e.src_node_id = s.node_id AND s.node_type = 'providers'
        JOIN {t}.nodes d ON e.dst_node_id = d.node_id AND d.node_type = 'organizations'""")
    assert resolved == N_EDGES


def test_composite_key_encoding_has_no_collisions(workspace_client):
    """Live Spark: the shipped composite-key SQL gives distinct ids to look-alike keys."""
    from lgk.steps.build import _key_expr

    def ids(cols, rows):
        values = ", ".join("(" + ", ".join("NULL" if v is None else f"'{v}'" for v in r) + ")" for r in rows)
        return [r[0] for r in execute_sql(workspace_client, WAREHOUSE_ID,
            f"SELECT CAST(xxhash64('t::'||{_key_expr(cols)}) AS STRING) "
            f"FROM VALUES {values} AS v({', '.join(cols)})")]

    # hostile: a NULL part in different positions (concat_ws skips NULLs -> same id)
    a, b = ids(["k1", "k2", "k3"], [("a", None, "b"), ("a", "b", None)])
    assert a != b
    # hostile: the separator inside a part ('a::b','c') vs ('a','b::c')
    c, d = ids(["k1", "k2"], [("a::b", "c"), ("a", "b::c")])
    assert c != d
