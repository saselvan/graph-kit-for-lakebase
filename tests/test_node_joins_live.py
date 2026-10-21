"""Live acceptance for node properties from a join.

One module-scoped graph on a fresh t- branch: fixture -> mapping (a floored `claims` aggregate
join and a one-to-one `provider_taxonomy` join on `provider`) -> live UC profile -> privacy
(claims.patient_id resolved `fold`, patients.patient_id `drop`) -> build -> preflight ->
provision -> switch, torn down by the shared `_teardown_live` in a finally.

The fixture. Every hostile row is named:

- providers P1, P2, P3; hospital H1; affiliations P1 -> H1 (an ordinary edge, so the graph has one).
- claims(claim_id STRING, provider_id STRING, patient_id BIGINT, billed_amount DECIMAL(18,2)):
  P1: 11 distinct patients over 31 rows, one with a NULL claim_id (claim_count 30, not 31 or 11);
  # hostile: P2 has 10 distinct patients over 40 claims — below the floor by subjects, above it by
  #   rows (a floor on COUNT(*) would pass it);
  # hostile: P3 has no claims (the LEFT JOIN keeps the node; NULL values);
  # hostile: one claim with provider_id NULL (dropped by WHERE ... IS NOT NULL);
  # hostile: a differently typed cell — billed_amount DECIMAL(18,2), values 0.10 and 0.20
  #   (an exact decimal sum, rendered by Spark);
  # hostile: reordered input — claims inserted in descending provider order.
- provider_taxonomy(provider_id, taxonomy_code): P1, P2 once each.
  # hostile: duplicate at the join grain — test 4 adds a second P1 row (JoinNotUniqueError).
- patients(patient_id PRIMARY KEY), claims.patient_id a FOREIGN KEY to it, so the profile lets
  the claims.patient_id flag fold.
# hostile: varied configuration — test 5 raises the floor to 12, so P1 (11) becomes NULL.

The reference is computed in Python from the rows above (Decimal sums, non-NULL claim_id counts,
distinct patient sets), never with the build SQL, then rendered by ONE Spark
`SELECT CAST(... AS STRING)` over a VALUES list, so DECIMAL formatting is Spark's.

Test order matters: read-only tests first, then the refused rebuild (4, the duplicate row is
deleted in a finally), then the raised-floor rebuild (5), which leaves Delta at floor 12.
"""

import contextlib
import tempfile
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest

from lgk.uc_sql import execute_sql
from tests.test_build_provision_walks import WAREHOUSE_ID, _pg, _teardown_live

pytestmark = [pytest.mark.seam1, pytest.mark.provision]

FLOOR = 11
RAISED_FLOOR = 12
PROVIDERS = {"P1": ("Dr. P1", "cardiology"), "P2": ("Dr. P2", "dermatology"), "P3": ("Dr. P3", None)}
TAXONOMY = [("P1", "207RC0000X"), ("P2", "208D00000X")]
OUTPUTS = ("specialty", "total_billed", "claim_count", "taxonomy_code")


# ============================================================================ fixture rows
def _claim_rows() -> List[Tuple[Optional[str], Optional[str], int, str]]:
    """(claim_id, provider_id, patient_id, billed_amount) in DESCENDING provider order."""
    rows: List[Tuple[Optional[str], Optional[str], int, str]] = []
    for i in range(40):  # P2: 10 distinct patients over 40 claims (below the floor by subjects)
        rows.append((f"C2-{i:02d}", "P2", 200 + i % 10, "0.20" if i % 2 else "0.10"))
    for i in range(31):  # P1: 11 distinct patients over 31 rows; one NULL claim_id
        rows.append((None if i == 30 else f"C1-{i:02d}", "P1", 100 + i % 11, "0.10" if i % 3 else "0.20"))
    rows.append(("C0-00", None, 999, "5.00"))  # hostile: a NULL provider_id
    return rows


CLAIMS = _claim_rows()


def _lit(v: Optional[str]) -> str:
    return "NULL" if v is None else f"'{v}'"


def _create_fixture(w: Any, catalog: str, schema: str) -> None:
    s = f"{catalog}.{schema}"
    q = lambda sql: execute_sql(w, WAREHOUSE_ID, sql)  # noqa: E731
    q(f"CREATE SCHEMA IF NOT EXISTS {s}")
    q(f"CREATE OR REPLACE TABLE {s}.providers (provider_id STRING, provider_name STRING, "
      f"specialty STRING) USING DELTA")
    q(f"INSERT INTO {s}.providers VALUES "
      + ", ".join(f"('{k}', '{n}', {_lit(sp)})" for k, (n, sp) in PROVIDERS.items()))
    q(f"CREATE OR REPLACE TABLE {s}.hospitals (hospital_id STRING, name STRING) USING DELTA")
    q(f"INSERT INTO {s}.hospitals VALUES ('H1', 'Hospital H1')")
    q(f"CREATE OR REPLACE TABLE {s}.affiliations (provider_id STRING, hospital_id STRING) USING DELTA")
    q(f"INSERT INTO {s}.affiliations VALUES ('P1', 'H1')")
    q(f"CREATE OR REPLACE TABLE {s}.patients (patient_id BIGINT NOT NULL, "
      f"CONSTRAINT lgk_t14_patients_pk PRIMARY KEY (patient_id)) USING DELTA")
    q(f"INSERT INTO {s}.patients VALUES " + ", ".join(f"({p})" for p in sorted({r[2] for r in CLAIMS})))
    q(f"CREATE OR REPLACE TABLE {s}.claims (claim_id STRING, provider_id STRING, patient_id BIGINT, "
      f"billed_amount DECIMAL(18,2), CONSTRAINT lgk_t14_claims_patient_fk FOREIGN KEY (patient_id) "
      f"REFERENCES {s}.patients) USING DELTA")
    q(f"INSERT INTO {s}.claims VALUES " + ", ".join(
        f"({_lit(c)}, {_lit(p)}, CAST({pt} AS BIGINT), CAST('{amt}' AS DECIMAL(18,2)))"
        for c, p, pt, amt in CLAIMS))
    q(f"CREATE OR REPLACE TABLE {s}.provider_taxonomy (provider_id STRING, taxonomy_code STRING) USING DELTA")
    q(f"INSERT INTO {s}.provider_taxonomy VALUES " + ", ".join(f"('{p}', '{t}')" for p, t in TAXONOMY))


def _write_mapping(path: Path, catalog: str, source_schema: str) -> None:
    path.write_text(f"""source_catalog: {catalog}
source_schema: {source_schema}

nodes:
  provider:
    source_table: providers
    natural_key: provider_id
    label: provider_name
    properties: [specialty]
    joins:
      - table: claims
        key: provider_id
        subject: patient_id
        aggregates:
          - sum(billed_amount) as total_billed
          - count(claim_id) as claim_count
      - table: provider_taxonomy
        key: provider_id
        columns: [taxonomy_code]
  hospital:
    source_table: hospitals
    natural_key: hospital_id
    label: name

edges:
  affiliation:
    source_table: affiliations
    src: provider_id
    src_type: provider
    dst: hospital_id
    dst_type: hospital
""")


def _clear_gate(config: Dict[str, Any], w: Any) -> None:
    """Live UC profile, then privacy: fold claims.patient_id, drop patients.patient_id."""
    from lgk.mapping import load_mapping
    from lgk.privacy.flags import OpenFlagError, open_flags, resolve_flag
    from lgk.steps.privacy import privacy
    from lgk.steps.profile import profile

    run_folder = config["run_folder"]
    mapping = load_mapping(Path(run_folder) / "graph_mapping.yaml")
    profile({**config, "source_catalog": mapping.source_catalog,
             "source_schema": mapping.source_schema, "workspace_client": w})
    with pytest.raises(OpenFlagError):
        privacy(config)
    flagged = {(f["table"], f["column"]): f for f in open_flags(run_folder)}
    expected = {("claims", "patient_id"), ("patients", "patient_id")}
    assert expected <= set(flagged), f"expected flags {sorted(expected)}, gate flagged {sorted(flagged)}"
    resolve_flag(run_folder, "claims", "patient_id", "fold", chooser="lgk-live-test")
    resolve_flag(run_folder, "patients", "patient_id", "drop", chooser="lgk-live-test")
    privacy(config)


@contextlib.contextmanager
def joins_graph_session(w: Any, lakebase_project: str, lakebase_catalog: str,
                        run_folder: Path) -> Iterator[Dict[str, Any]]:
    from lgk.harness import BranchLeasor
    from lgk.steps.build import build
    from lgk.steps.provision import provision
    from lgk.steps.switch import switch
    from tests.preflight_support import preflight_as_operator

    suffix = uuid.uuid4().hex[:6]
    leasor = BranchLeasor(workspace_client=w, project_id=lakebase_project,
                          test_base_branch="test-base", run_folder=run_folder, ttl_seconds=14400)
    config: Dict[str, Any] = {
        "graph_name": "joins_graph",
        "lakebase_project": lakebase_project,
        "lakebase_catalog": lakebase_catalog,
        "kit_schema": f"lgk_v1_{suffix}",
        "served_schema": f"lgk_served_{suffix}",
        "fixture_schema": f"lgk_v1_fx_{suffix}",
        "warehouse_id": WAREHOUSE_ID,
        "enriched": False,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            config["branch_id"] = leasor.create_test_branch(f"joins-{suffix}")["branch_id"]
            run_dir = Path(tmpdir) / "runs" / f"joins_{suffix}"
            run_dir.mkdir(parents=True)
            config["run_folder"] = str(run_dir)
            _create_fixture(w, lakebase_catalog, config["fixture_schema"])
            _write_mapping(run_dir / "graph_mapping.yaml", lakebase_catalog, config["fixture_schema"])
            _clear_gate(config, w)
            build(config)
            preflight_as_operator(w, config)
            provision(config)
            switch(config)
            yield config
        finally:
            _teardown_live(w, leasor, config)


@pytest.fixture(scope="module")
def joins_graph(workspace_client, lakebase_project, lakebase_catalog, run_folder):
    with joins_graph_session(workspace_client, lakebase_project, lakebase_catalog, run_folder) as config:
        yield config


# ============================================================================ reference
def _q(w: Any, sql: str) -> List[List[Any]]:
    return execute_sql(w, WAREHOUSE_ID, sql)


def python_reference(floor: int) -> Dict[str, Dict[str, Optional[str]]]:
    """{provider: {output: python value as text or None}}, before Spark's rendering."""
    out: Dict[str, Dict[str, Optional[str]]] = {}
    taxonomy = dict(TAXONOMY)
    for p, (_name, specialty) in PROVIDERS.items():
        rows = [r for r in CLAIMS if r[1] == p]
        subjects = len({r[2] for r in rows})
        above = bool(rows) and subjects >= floor
        out[p] = {
            "specialty": specialty,
            "total_billed": str(sum((Decimal(r[3]) for r in rows), Decimal("0"))) if above else None,
            "claim_count": str(sum(1 for r in rows if r[0] is not None)) if above else None,
            "taxonomy_code": taxonomy.get(p),
        }
    return out


def spark_rendered_reference(w: Any, floor: int) -> Dict[str, Dict[str, Optional[str]]]:
    """The Python reference with each DECIMAL/BIGINT rendered by ONE Spark CAST(... AS STRING)."""
    ref = python_reference(floor)
    values = ", ".join(
        f"('{p}', CAST({_lit(v['total_billed'])} AS DECIMAL(28,2)), CAST({_lit(v['claim_count'])} AS BIGINT))"
        for p, v in ref.items())
    for p, billed, count in _q(w, f"SELECT p, CAST(b AS STRING), CAST(c AS STRING) "
                                  f"FROM VALUES {values} AS v(p, b, c)"):
        ref[str(p)]["total_billed"], ref[str(p)]["claim_count"] = billed, count
    return ref


def _delta_properties(w: Any, g: Dict[str, Any]) -> Dict[str, Dict[str, Optional[str]]]:
    t = f"{g['lakebase_catalog']}.{g['kit_schema']}.nodes"
    picks = ", ".join(f"properties['{o}']" for o in OUTPUTS)
    rows = _q(w, f"SELECT natural_key, array_join(array_sort(map_keys(properties)), ','), {picks} "
                 f"FROM {t} WHERE node_type = 'provider'")
    out = {}
    for row in rows:
        assert row[1] == ",".join(sorted(OUTPUTS)), f"{row[0]}: map keys {row[1]} (keys must never drop)"
        out[str(row[0])] = {o: (None if v is None else str(v)) for o, v in zip(OUTPUTS, row[2:])}
    return out


# ============================================================================ tests
def test_join_outputs_equal_the_python_reference(joins_graph, workspace_client):
    got = _delta_properties(workspace_client, joins_graph)
    want = spark_rendered_reference(workspace_client, FLOOR)
    assert got == want
    # non-vacuous: P1 is above the floor with an exact decimal sum and 30 non-NULL claim ids
    assert got["P1"]["claim_count"] == "30" and got["P1"]["total_billed"] is not None
    assert Decimal(got["P1"]["total_billed"]) == sum(Decimal(r[3]) for r in CLAIMS if r[1] == "P1")


def test_below_floor_aggregates_are_null_in_delta_and_postgres(joins_graph, workspace_client):
    """P2 (10 patients, 40 claims): NULL in Delta and in nodes_view as reader; P1 is set in both,
    with the same string as Spark's."""
    g = joins_graph
    delta = _delta_properties(workspace_client, g)
    assert delta["P2"]["total_billed"] is None and delta["P2"]["claim_count"] is None
    assert delta["P2"]["taxonomy_code"] == "208D00000X"  # the unfloored join is still served
    conn = _pg(g)
    try:
        conn.execute("GRANT reader TO CURRENT_USER")
        conn.execute("SET ROLE reader")
        rows = conn.execute(
            f"SELECT natural_key, properties ->> 'total_billed', properties ->> 'claim_count' "
            f"FROM {g['served_schema']}.nodes_view WHERE node_type = 'provider'").fetchall()
    finally:
        conn.rollback()
        conn.close()
    pg = {str(k): (b, c) for k, b, c in rows}
    assert pg["P2"] == (None, None), f"P2 below the floor is served: {pg['P2']}"
    assert pg["P3"] == (None, None)
    assert pg["P1"] == (delta["P1"]["total_billed"], delta["P1"]["claim_count"]), pg["P1"]
    assert pg["P1"][1] == "30"


def test_node_count_unchanged_by_joins(joins_graph, workspace_client):
    g = joins_graph
    t = f"{g['lakebase_catalog']}.{g['kit_schema']}.nodes"
    (n_nodes,), = _q(workspace_client, f"SELECT COUNT(*) FROM {t} WHERE node_type = 'provider'")
    (n_src,), = _q(workspace_client, f"SELECT COUNT(*) FROM {g['lakebase_catalog']}.{g['fixture_schema']}.providers")
    assert int(n_nodes) == int(n_src) == len(PROVIDERS)
    assert set(_delta_properties(workspace_client, g)) == set(PROVIDERS)  # P3 (no claims) kept


def test_node_ids_unchanged_by_joins(joins_graph, workspace_client):
    """node_id equals the pre-Item-6 formula xxhash64('provider::'||key), computed by Spark alone."""
    g = joins_graph
    values = ", ".join(f"('{k}')" for k in PROVIDERS)
    want = {str(k): str(h) for k, h in _q(workspace_client, f"SELECT k, CAST(xxhash64('provider::'||k) AS STRING) "
                                                         f"FROM VALUES {values} AS v(k)")}
    t = f"{g['lakebase_catalog']}.{g['kit_schema']}.nodes"
    got = {str(k): str(i) for k, i in _q(workspace_client, f"SELECT natural_key, node_id FROM {t} "
                                                        f"WHERE node_type = 'provider'")}
    assert got == want


def _nodes_version(w: Any, g: Dict[str, Any]) -> int:
    t = f"{g['lakebase_catalog']}.{g['kit_schema']}.nodes"
    return int(_q(w, f"SELECT MAX(version) FROM (DESCRIBE HISTORY {t})")[0][0])


def test_non_unique_columns_join_is_refused_before_insert(joins_graph, workspace_client):
    """A second P1 row in provider_taxonomy: build raises JoinNotUniqueError and `nodes` is not
    replaced (same Delta version, same rows)."""
    from lgk.steps.build import JoinNotUniqueError, build

    g = joins_graph
    tax = f"{g['lakebase_catalog']}.{g['fixture_schema']}.provider_taxonomy"
    before_version, before = _nodes_version(workspace_client, g), _delta_properties(workspace_client, g)
    try:
        _q(workspace_client, f"INSERT INTO {tax} VALUES ('P1', '207Q00000X')")  # hostile: duplicate at the grain
        with pytest.raises(JoinNotUniqueError, match="provider_taxonomy"):
            build(g)
        assert _nodes_version(workspace_client, g) == before_version
        assert _delta_properties(workspace_client, g) == before
    finally:
        _q(workspace_client, f"DELETE FROM {tax} WHERE taxonomy_code = '207Q00000X'")


def test_join_floor_check_fails_closed_on_a_node_it_cannot_match(joins_graph, workspace_client):
    """A provider row in Delta nodes with a floored output set but a node_id no source row hashes
    to (an operator pipeline's other id encoding): assert_join_floor raises, it never reports 0."""
    from lgk.mapping import load_mapping
    from lgk.privacy.floor import JoinFloorError, assert_join_floor

    g = joins_graph
    t = f"{g['lakebase_catalog']}.{g['kit_schema']}.nodes"
    mapping = load_mapping(Path(g["run_folder"]) / "graph_mapping.yaml")
    sql = lambda q: _q(workspace_client, q)  # noqa: E731
    assert_join_floor(sql, t, mapping, FLOOR)  # non-vacuous: the kit's own nodes pass
    try:
        # hostile: an id from another encoding (the hash of the bare key, no 'provider::' prefix)
        _q(workspace_client, f"INSERT INTO {t} VALUES (CAST(xxhash64('P2') AS STRING), 'provider', 'P2', "
                             f"'Dr. P2', map('total_billed', '6.00', 'claim_count', '40'))")
        with pytest.raises(JoinFloorError, match=r"provider\.joins\[0\]"):
            assert_join_floor(sql, t, mapping, FLOOR)
    finally:
        _q(workspace_client, f"DELETE FROM {t} WHERE node_id = CAST(xxhash64('P2') AS STRING)")


def test_raised_floor_nulls_p1(joins_graph, workspace_client):
    """privacy_floor=12 through the runner: build reruns and P1 (11 patients) becomes NULL."""
    from lgk.runner import RunFolder, RunResult, RunStatus, execute_step
    from lgk.steps.build import build
    from lgk.steps.privacy import privacy

    g = joins_graph
    config = {**g, "privacy_floor": RAISED_FLOOR}
    rf = RunFolder(Path(g["run_folder"]))
    rf.update_step(RunResult(step="build", status=RunStatus.SUCCESS))
    result = execute_step(rf, config, "privacy", privacy)
    assert result.status == RunStatus.SUCCESS, result.error
    result = execute_step(rf, config, "build", build)
    assert result.status == RunStatus.SUCCESS, result.error
    got = _delta_properties(workspace_client, g)
    assert got == spark_rendered_reference(workspace_client, RAISED_FLOOR)
    assert got["P1"]["total_billed"] is None and got["P1"]["claim_count"] is None
    assert got["P1"]["taxonomy_code"] == "207RC0000X"
