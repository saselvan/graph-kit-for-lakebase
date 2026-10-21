"""Live acceptance for folded edges with a `weight` column, per-call `edge_types`.

One module-scoped graph on a fresh t- branch (`fold_graph`, mirroring `provisioned_graph`):
fixture -> mapping -> live UC profile -> privacy (encounters.patient_id resolved `fold`) ->
build -> preflight -> provision -> switch, torn down by the shared `_teardown_live` in a finally.

The fixture (ticket §7). Every hostile row is named:

- providers P1..P6, facilities F1 F2, hospital H1 (node tables).
- encounters(patient_id BIGINT, provider_id STRING, facility_id STRING):
  # hostile: duplicate at the aggregation grain — P1 and P2 share exactly 11 patients, each with
  #   3 encounter rows per side (a non-distinct count gives 99, COUNT(DISTINCT) gives 11).
  # hostile: one below the floor — P1 and P3 share exactly 10 patients, 5 rows per side.
  #   P1 and P4 share 12 patients (above the floor; P1's rows are at F1).
  # hostile: a NULL member — P5 shares 11 patients with P1, but provider_id is NULL in 1 row.
  # hostile: a differently typed cell — patient_id is BIGINT; the member keys are STRING.
  # hostile: MIN, not SUM/MAX, at hop 2 — P2-P6 share 11 and P4-P6 share 12 patients, so P6 is
  #   reached at hop 2 over two folded edges and its support is MIN(11, 12) = 11.
  # hostile: a second fold type — `treats_at` (pair form, provider -> facility): P1 -> F1 = 12
  #   patients. A shares_patients selection must not return F1 (only the IN clause excludes it).
- affiliations: P1 -> H1 (unfolded `affiliation`), P2 -> H1, P3 -> H1;
  # hostile: a mixed graph — the folded walk must not return H1.
  # hostile: a duplicate affiliation row P1 -> H1 (build dedups; support stays 1).
- practices: same_practice P1 -> P2, P2 -> P3, P5 -> P1;
  # hostile: an unfolded edge into a folded neighbour (P1 -> P2 is also shares_patients 11).
- patients(patient_id PRIMARY KEY), with encounters.patient_id a FOREIGN KEY to it: that
  informational constraint is what the UC profile reads (information_schema
  referential_constraints -> schema.json `foreign_keys`), so `_fold_allowed` lets the
  encounters.patient_id flag choose fold. patients.patient_id itself (a single-column key) can
  only be dropped; the mapping never reads that table.

The reference is independent of the kit's fold SQL: `{(pa, pb): set(patients)}` in Python from
the rows above, kept at >= floor, hashed by ONE Spark `SELECT xxhash64(...)` over a VALUES list;
the expected walk is a Spark walk (the `spark_web_reference` style) over those reference edges,
filtered by the test's OWN selection list.

Test order matters (pytest runs a module in file order): read-only tests first, then the
rolled-back Postgres write (7), then the Delta injection (6, row deleted in a finally), then the
raised-floor rebuild (5), which leaves the Delta tables at floor 12.
"""

import contextlib
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pytest

from lgk.uc_sql import execute_sql
from lgk.versions import version_schema
from lgk.walks.sql import DIRECTIONS
from tests.test_build_provision_walks import SYNCED_TABLES, WAREHOUSE_ID, _pg, _teardown_live

pytestmark = [pytest.mark.seam1, pytest.mark.provision]

FLOOR = 11
RAISED_FLOOR = 12
TOP_N = 25
FOLD_TYPES = ("shares_patients", "treats_at")
UNFOLDED_TYPES = ("affiliation", "same_practice")
INJECTED_EDGE_ID = "lgk-t12-injected-below-floor"

PROVIDERS = [f"P{i}" for i in range(1, 7)]
FACILITIES = ["F1", "F2"]
HOSPITALS = ["H1"]
NODES = ([("provider", k) for k in PROVIDERS] + [("facility", k) for k in FACILITIES]
         + [("hospital", k) for k in HOSPITALS])


# ============================================================================ fixture rows
def _encounter_rows() -> List[Tuple[int, Optional[str], Optional[str]]]:
    """(patient_id, provider_id, facility_id) rows; patient sets are disjoint across groups."""
    rows: List[Tuple[int, Optional[str], Optional[str]]] = []
    for p in range(101, 112):  # hostile: duplicate at the aggregation grain (11 patients, 3 x 3 rows)
        rows += [(p, "P1", "F2")] * 3 + [(p, "P2", "F2")] * 3
    for p in range(201, 211):  # hostile: one below the floor (10 patients, 5 rows per side)
        rows += [(p, "P1", "F2")] * 5 + [(p, "P3", "F2")] * 5
    for p in range(401, 413):  # P1-P4: 12 patients; P1's rows at F1 (treats_at P1 -> F1 = 12)
        rows += [(p, "P1", "F1")] * 2 + [(p, "P4", "F2")] * 2
    for p in range(301, 312):  # hostile: a NULL member (11 patients, provider_id NULL in 1 row)
        rows += [(p, "P1", "F2"), (p, None if p == 311 else "P5", "F2")]
    for p in range(501, 512):  # hostile: MIN at hop 2 (P2-P6 = 11 ...
        rows += [(p, "P2", "F2"), (p, "P6", "F2")]
    for p in range(601, 613):  # ... and P4-P6 = 12)
        rows += [(p, "P4", "F2"), (p, "P6", "F2")]
    return rows


ENCOUNTERS = _encounter_rows()
AFFILIATIONS = [("P1", "H1"), ("P1", "H1"),  # hostile: duplicate affiliation row
                ("P2", "H1"), ("P3", "H1")]
PRACTICES = [("P1", "P2"),  # hostile: an unfolded edge into a folded neighbour
             ("P2", "P3"), ("P5", "P1")]


def _sql_str(v: Optional[str]) -> str:
    return "NULL" if v is None else f"'{v}'"


def _create_fold_fixture(w: Any, catalog: str, fixture_schema: str) -> None:
    s = f"{catalog}.{fixture_schema}"
    q = lambda sql: execute_sql(w, WAREHOUSE_ID, sql)  # noqa: E731
    q(f"CREATE SCHEMA IF NOT EXISTS {s}")
    for table, key, keys, label in (("providers", "provider_id", PROVIDERS, "Dr."),
                                    ("facilities", "facility_id", FACILITIES, "Facility"),
                                    ("hospitals", "hospital_id", HOSPITALS, "Hospital")):
        q(f"CREATE OR REPLACE TABLE {s}.{table} ({key} STRING, name STRING) USING DELTA")
        q(f"INSERT INTO {s}.{table} VALUES " + ", ".join(f"('{k}', '{label} {k}')" for k in keys))
    # The informational PK/FK pair the UC profile reads, so encounters.patient_id may fold.
    q(f"CREATE OR REPLACE TABLE {s}.patients (patient_id BIGINT NOT NULL, "
      f"CONSTRAINT lgk_fx_patients_pk PRIMARY KEY (patient_id)) USING DELTA")
    q(f"INSERT INTO {s}.patients VALUES "
      + ", ".join(f"({p})" for p in sorted({r[0] for r in ENCOUNTERS})))
    # hostile: a differently typed cell — patient_id BIGINT (members are STRING)
    q(f"CREATE OR REPLACE TABLE {s}.encounters (patient_id BIGINT, provider_id STRING, "
      f"facility_id STRING, CONSTRAINT lgk_fx_enc_patient_fk FOREIGN KEY (patient_id) "
      f"REFERENCES {s}.patients) USING DELTA")
    q(f"INSERT INTO {s}.encounters VALUES "
      + ", ".join(f"(CAST({p} AS BIGINT), {_sql_str(a)}, {_sql_str(b)})" for p, a, b in ENCOUNTERS))
    q(f"CREATE OR REPLACE TABLE {s}.affiliations (provider_id STRING, hospital_id STRING) USING DELTA")
    q(f"INSERT INTO {s}.affiliations VALUES " + ", ".join(f"('{a}', '{b}')" for a, b in AFFILIATIONS))
    q(f"CREATE OR REPLACE TABLE {s}.practices (provider_id STRING, colleague_id STRING) USING DELTA")
    q(f"INSERT INTO {s}.practices VALUES " + ", ".join(f"('{a}', '{b}')" for a, b in PRACTICES))


def _write_fold_mapping(path: Path, catalog: str, source_schema: str) -> None:
    path.write_text(f"""source_catalog: {catalog}
source_schema: {source_schema}

nodes:
  provider:
    source_table: providers
    natural_key: provider_id
    label: name
  facility:
    source_table: facilities
    natural_key: facility_id
    label: name
  hospital:
    source_table: hospitals
    natural_key: hospital_id
    label: name

edges:
  shares_patients:
    source_table: encounters
    src: provider_id
    src_type: provider
    dst: provider_id
    dst_type: provider
    fold_subject: patient_id
  treats_at:
    source_table: encounters
    src: provider_id
    src_type: provider
    dst: facility_id
    dst_type: facility
    fold_subject: patient_id
  affiliation:
    source_table: affiliations
    src: provider_id
    src_type: provider
    dst: hospital_id
    dst_type: hospital
  same_practice:
    source_table: practices
    src: provider_id
    src_type: provider
    dst: colleague_id
    dst_type: provider
""")


# ============================================================================ the gate, live
def _clear_gate_with_fold(config: Dict[str, Any], w: Any) -> None:
    """Live UC profile, then privacy: fold encounters.patient_id, drop patients.patient_id.

    Each expected flag must really be raised, and the fold flag must be fold-allowed (the
    profile read the FOREIGN KEY); any other open flag fails the final privacy run.
    """
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
    expected = {("encounters", "patient_id"), ("patients", "patient_id")}
    assert expected <= set(flagged), f"expected flags {sorted(expected)}, gate flagged {sorted(flagged)}"
    assert flagged[("encounters", "patient_id")]["fold_allowed"] is True, (
        "encounters.patient_id is not fold-allowed: the UC profile did not record its FOREIGN KEY")
    resolve_flag(run_folder, "encounters", "patient_id", "fold", chooser="lgk-live-test")
    resolve_flag(run_folder, "patients", "patient_id", "drop", chooser="lgk-live-test")
    privacy(config)


@contextlib.contextmanager
def fold_graph_session(w: Any, lakebase_project: str, lakebase_catalog: str,
                       run_folder: Path) -> Iterator[Dict[str, Any]]:
    """Fresh t- branch + fold fixture + gate + build + preflight + provision + switch; teardown."""
    from lgk.harness import BranchLeasor
    from lgk.steps.build import build
    from lgk.steps.provision import provision
    from lgk.steps.switch import switch
    from tests.preflight_support import preflight_as_operator

    suffix = uuid.uuid4().hex[:6]
    leasor = BranchLeasor(workspace_client=w, project_id=lakebase_project,
                          test_base_branch="test-base", run_folder=run_folder, ttl_seconds=14400)
    config: Dict[str, Any] = {
        "graph_name": "fold_graph",
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
            config["branch_id"] = leasor.create_test_branch(f"fold-{suffix}")["branch_id"]
            run_dir = Path(tmpdir) / "runs" / f"fold_{suffix}"
            run_dir.mkdir(parents=True)
            config["run_folder"] = str(run_dir)
            _create_fold_fixture(w, lakebase_catalog, config["fixture_schema"])
            _write_fold_mapping(run_dir / "graph_mapping.yaml", lakebase_catalog, config["fixture_schema"])
            _clear_gate_with_fold(config, w)
            build(config)
            # Not _assert_build_deduped: shares_patients and same_practice share the pair P1 -> P2.
            _assert_unique_per_type(w, config)
            preflight_as_operator(w, config)
            provision(config)
            switch(config)
            yield config
        finally:
            _teardown_live(w, leasor, config)


@pytest.fixture(scope="module")
def fold_graph(workspace_client, lakebase_project, lakebase_catalog, run_folder):
    with fold_graph_session(workspace_client, lakebase_project, lakebase_catalog, run_folder) as config:
        config["ids"] = _spark_ids(workspace_client, [f"{t}::{k}" for t, k in NODES])
        yield config


def _assert_unique_per_type(w: Any, config: Dict[str, Any]) -> None:
    t = f"{config['lakebase_catalog']}.{config['kit_schema']}.edges"
    total, ids, triples = map(int, _q(w, f"SELECT COUNT(*), COUNT(DISTINCT edge_id), "
                                         f"COUNT(DISTINCT relationship_type, src_node_id, dst_node_id) "
                                         f"FROM {t}")[0])
    assert total == ids == triples, f"{t}: {total} rows, {ids} edge_ids, {triples} (type, src, dst)"


# ============================================================================ reference
def _q(w: Any, sql: str) -> List[List[Any]]:
    return execute_sql(w, WAREHOUSE_ID, sql)


def _spark_ids(w: Any, strings: Sequence[str]) -> Dict[str, str]:
    """{s: CAST(xxhash64(s) AS STRING)}, by ONE Spark SELECT over a VALUES list."""
    values = ", ".join(f"('{s}')" for s in dict.fromkeys(strings))
    return {str(s): str(h) for s, h in _q(w, f"SELECT s, CAST(xxhash64(s) AS STRING) "
                                             f"FROM VALUES {values} AS v(s)")}


def reference_fold_weights(floor: int) -> Dict[Tuple[str, str, str], int]:
    """{(edge_type, src_key, dst_key): distinct patients} for both fold types, kept at >= floor."""
    by_patient: Dict[int, set] = defaultdict(set)
    pairs: Dict[Tuple[str, str, str], set] = defaultdict(set)
    for patient, provider, facility in ENCOUNTERS:
        if provider is not None:
            by_patient[patient].add(provider)
            if facility is not None:
                pairs[("treats_at", provider, facility)].add(patient)
    for patient, members in by_patient.items():
        for a in members:
            for b in members:
                if a != b:
                    pairs[("shares_patients", a, b)].add(patient)
    return {k: len(v) for k, v in pairs.items() if len(v) >= floor}


def reference_unfolded() -> set:
    """{(edge_type, src_key, dst_key)}: the distinct unfolded pairs."""
    return ({("affiliation", a, b) for a, b in AFFILIATIONS}
            | {("same_practice", a, b) for a, b in PRACTICES})


_DST_TYPE = {"shares_patients": "provider", "treats_at": "facility",
             "affiliation": "hospital", "same_practice": "provider"}


def reference_edges(g: Dict[str, Any], floor: int = FLOOR) -> List[Tuple[str, str, str, Optional[int], str]]:
    """(type, src_node_id, dst_node_id, weight, edge_id-string) for every reference edge."""
    out = []
    for (t, a, b), weight in reference_fold_weights(floor).items():
        out.append((t, a, b, weight))
    out += [(t, a, b, None) for t, a, b in reference_unfolded()]
    ids = g["ids"]
    return [(t, ids[f"provider::{a}"], ids[f"{_DST_TYPE[t]}::{b}"], wt, f"{t}::{a}::{b}")
            for t, a, b, wt in out]


def spark_selection_walk(w: Any, g: Dict[str, Any], seed: str, selection: Sequence[str], depth: int,
                         folded: bool, direction: str = "outbound", floor: int = FLOOR,
                         top_n: int = TOP_N) -> List[Tuple[int, str, int]]:
    """[(hop, node_id, support), ...] in hop then rank order, over the reference edges filtered
    to `selection` in Spark. Folded: MIN(weight) and HAVING >= floor; unfolded: COUNT(*)."""
    rows = ", ".join(
        f"('{s}', '{d}', '{t}', CAST({'NULL' if wt is None else wt} AS BIGINT))"
        for t, s, d, wt, _e in reference_edges(g, floor))
    sel = ", ".join(f"'{t}'" for t in sorted(set(selection)))
    picked = f"SELECT src, dst, t, wt FROM VALUES {rows} AS v(src, dst, t, wt) WHERE t IN ({sel})"
    inbound = f"SELECT dst AS src, src AS dst, t, wt FROM ({picked}) p"
    edges = {"outbound": picked, "inbound": inbound,
             "both": f"SELECT DISTINCT src, dst, t, wt FROM ({picked} UNION ALL {inbound}) b"}[direction]
    agg, having = ("MIN(e.wt)", f"HAVING MIN(e.wt) >= {floor}") if folded else ("COUNT(*)", "")
    ctes = [f"e AS ({edges})", f"h0 AS (SELECT '{seed}' AS node_id)"]
    for i in range(1, depth + 1):
        seen = " UNION ALL ".join(f"SELECT node_id FROM h{j}" for j in range(i))
        ctes.append(
            f"h{i} AS (SELECT node_id, support, rn FROM (SELECT c.node_id, c.support, ROW_NUMBER() "
            f"OVER (ORDER BY c.support DESC, c.node_id ASC) AS rn FROM (SELECT e.dst AS node_id, "
            f"{agg} AS support FROM h{i - 1} f JOIN e ON e.src = f.node_id GROUP BY e.dst {having}) c "
            f"LEFT ANTI JOIN ({seen}) s ON s.node_id = c.node_id) WHERE rn <= {top_n})")
    union = " UNION ALL ".join(f"SELECT {i} AS hop, rn, node_id, support FROM h{i}" for i in range(1, depth + 1))
    got = _q(w, f"WITH {', '.join(ctes)} {union}")
    return [(int(h), str(n), int(s)) for h, _rn, n, s in sorted(got, key=lambda r: (int(r[0]), int(r[1])))]


def _live_walk(g: Dict[str, Any], seed_id: str, depth: int, edge_types: Sequence[str],
               direction: str = "outbound") -> List[Tuple[int, str, int]]:
    from lgk.steps.walks import get_walk_query

    sql = get_walk_query(start_node_id="unused", depth=depth, served_schema=g["served_schema"],
                         top_n=TOP_N, run_folder=g["run_folder"], edge_types=list(edge_types),
                         direction=direction)
    with _pg(g) as conn:
        rows = conn.execute(sql, {"start_node_id": seed_id}).fetchall()
    return [(int(h), str(n), int(s)) for h, n, s, _score in rows]


def _delta_edges(w: Any, g: Dict[str, Any]) -> List[Tuple[str, str, str, Optional[int], str]]:
    t = f"{g['lakebase_catalog']}.{g['kit_schema']}.edges"
    return [(str(rt), str(s), str(d), None if wt is None else int(wt), str(e))
            for rt, s, d, wt, e in _q(w, f"SELECT relationship_type, src_node_id, dst_node_id, "
                                         f"weight, edge_id FROM {t}")]


def _named(g: Dict[str, Any]) -> Dict[str, str]:
    """node_id -> natural key, for readable assertions."""
    return {v: k.split("::", 1)[1] for k, v in g["ids"].items()}


# ============================================================================ tests
def test_reference_ids_equal_the_kit_node_ids(fold_graph, workspace_client):
    """Guard for every comparison below: the Spark-hashed reference ids are the kit's node ids."""
    g = fold_graph
    kit = {(str(t), str(k)): str(i) for t, k, i in _q(
        workspace_client, f"SELECT node_type, natural_key, node_id FROM "
                          f"{g['lakebase_catalog']}.{g['kit_schema']}.node_id_map")}
    assert kit == {(t, k): g["ids"][f"{t}::{k}"] for t, k in NODES}


def test_edges_columns_end_with_weight_bigint(fold_graph):
    """The synced version `edges` has exactly 6 columns, the last `weight bigint`."""
    vs = version_schema(fold_graph)
    with _pg(fold_graph) as conn:
        rows = [tuple(r) for r in conn.execute(
            f"SELECT attnum, attname, format_type(atttypid, atttypmod) FROM pg_attribute "
            f"WHERE attrelid = '{vs}.edges'::regclass AND attnum > 0 AND NOT attisdropped "
            f"ORDER BY attnum").fetchall()]
    print(f"{vs}.edges pg_attribute:")
    for row in rows:
        print(f"  {row[0]} | {row[1]} | {row[2]}")
    assert len(rows) == 6, rows
    assert (rows[-1][1], rows[-1][2]) == ("weight", "bigint"), rows


def test_build_writes_weight_as_distinct_subjects(fold_graph, workspace_client):
    """Delta edges for both fold types equal the Python reference exactly (both directions of
    shares_patients); unfolded rows have weight IS NULL; nothing else is written."""
    g = fold_graph
    weights = reference_fold_weights(FLOOR)
    # Non-vacuous reference: the hostile pairs land where the ticket says.
    assert weights[("shares_patients", "P1", "P2")] == weights[("shares_patients", "P2", "P1")] == 11
    assert weights[("shares_patients", "P1", "P4")] == 12 and weights[("treats_at", "P1", "F1")] == 12
    ref = reference_edges(g)
    hashed = _spark_ids(workspace_client, [e for *_x, e in ref if _x[0] in FOLD_TYPES])
    want_folded = {(t, s, d, wt, hashed[e]) for t, s, d, wt, e in ref if t in FOLD_TYPES}
    got = _delta_edges(workspace_client, g)
    got_folded = {r for r in got if r[0] in FOLD_TYPES}
    assert got_folded == want_folded
    got_unfolded = [r for r in got if r[0] not in FOLD_TYPES]
    assert [r for r in got_unfolded if r[3] is not None] == [], "an unfolded edge carries a weight"
    assert {r[:3] for r in got_unfolded} == {(t, s, d) for t, s, d, _w, _e in ref if t not in FOLD_TYPES}
    assert len(got) == len(ref), f"{len(got)} Delta edges, {len(ref)} reference edges"


def test_pair_below_floor_never_exists_in_delta(fold_graph, workspace_client):
    """P1-P3 (10 patients), P1-P5 (10 after the NULL member) and P3 -> F2 (10) are not in Delta."""
    g = fold_graph
    ids = g["ids"]
    p1, p3, p5 = ids["provider::P1"], ids["provider::P3"], ids["provider::P5"]
    pairs = {(r[1], r[2]) for r in _delta_edges(workspace_client, g) if r[0] in FOLD_TYPES}
    # Non-vacuous: the above-floor pair of the same seed is there.
    assert (p1, ids["provider::P2"]) in pairs
    for a, b in ((p1, p3), (p3, p1), (p1, p5), (p5, p1), (p3, ids["facility::F2"])):
        assert (a, b) not in pairs, f"below-floor pair {_named(g)[a]} -> {_named(g)[b]} is in Delta"


def test_no_count_below_floor_leaves_postgres(fold_graph):
    """As reader on the served edges_view, and as the operator on <kit>_v<N>.edges: no weight
    below 11, and the smallest weight is exactly 11."""
    g = fold_graph
    n_folded = len(reference_fold_weights(FLOOR))
    checks = (
        f"SELECT count(*) FROM {{rel}} WHERE weight < {FLOOR}",
        "SELECT min(weight) FROM {rel} WHERE weight IS NOT NULL",
        "SELECT count(*) FROM {rel} WHERE weight IS NOT NULL",
    )
    conn = _pg(g)
    try:
        conn.execute("GRANT reader TO CURRENT_USER")
        for role, rel in (("reader", f"{g['served_schema']}.edges_view"),
                          (None, f"{version_schema(g)}.edges")):
            conn.execute(f"SET ROLE {role}" if role else "RESET ROLE")
            below, low, count = (conn.execute(c.format(rel=rel)).fetchone()[0] for c in checks)
            assert int(below) == 0, f"{role or 'operator'}: {below} rows of {rel} below the floor"
            assert int(low) == FLOOR, f"{role or 'operator'}: min weight on {rel} is {low}"
            assert int(count) == n_folded, f"{role or 'operator'}: {count} weighted rows on {rel}"
    finally:
        conn.rollback()
        conn.close()


def test_folded_walk_on_kit_data_equals_reference(fold_graph, workspace_client):
    """['shares_patients'] from P1, depths 1 and 2: equals the Spark reference; hop 1 is
    {P2: 11, P4: 12}; H1 and F1 never appear; hop 2 holds P6 at MIN(11, 12) = 11."""
    from lgk.steps.walks import get_walk_query

    g, ids, named = fold_graph, fold_graph["ids"], _named(fold_graph)
    seed = ids["provider::P1"]
    for depth in (1, 2):
        ref = spark_selection_walk(workspace_client, g, seed, ["shares_patients"], depth, folded=True)
        got = _live_walk(g, seed, depth, ["shares_patients"])
        assert got == ref, f"depth {depth}: {got} != {ref}"
        assert {named[n]: s for h, n, s in got if h == 1} == {"P2": 11, "P4": 12}
        assert not {ids["hospital::H1"], ids["facility::F1"], ids["facility::F2"]} & {n for _h, n, _s in got}
    assert [(h, named[n], s) for h, n, s in got if h == 2] == [(2, "P6", 11)]
    with pytest.raises(ValueError):  # a folded selection walks outbound only
        get_walk_query(start_node_id="unused", depth=1, served_schema=g["served_schema"],
                       run_folder=g["run_folder"], edge_types=["shares_patients"], direction="inbound")


def test_folded_web_set_hides_below_floor(fold_graph, workspace_client):
    """On the mixed folded graph: the set walk takes the folded selection's floored
    MIN(weight). One natural_key seed (P1) equals the folded single-start reference at depths 1-2
    (P3, 10 patients, and P5, 10 non-NULL, never appear); the set of ALL providers over treats_at
    reaches each facility at the MIN over its kept edges, never an edge below the floor."""
    from lgk.steps.walks import get_walk_query
    from lgk.walks.sql import NodePredicate

    g, ids, named = fold_graph, fold_graph["ids"], _named(fold_graph)

    def set_walk(seed, value, depth, edge_types):
        sql = get_walk_query("unused", depth, g["served_schema"], top_n=TOP_N, run_folder=g["run_folder"],
                             edge_types=edge_types, seed=seed)
        with _pg(g) as conn:
            return [(int(h), str(n), int(s), bool(m))
                    for h, n, s, _sc, m in conn.execute(sql, {"seed_value": value}).fetchall()]

    for depth in (1, 2):
        ref = spark_selection_walk(workspace_client, g, ids["provider::P1"], ["shares_patients"], depth, folded=True)
        got = set_walk(NodePredicate("provider", "natural_key"), "P1", depth, ["shares_patients"])
        assert got == [(h, n, s, False) for h, n, s in ref], f"depth {depth}: {got} != {ref}"
        assert not {"P3", "P5"} & {named[n] for _h, n, _s, _m in got}
    weights = reference_fold_weights(FLOOR)
    want = {f: min(w for (t, _p, d), w in weights.items() if t == "treats_at" and d == f)
            for (t, _p, f) in weights if t == "treats_at"}
    got = set_walk(NodePredicate("provider"), None, 1, ["treats_at"])
    assert {named[n]: s for _h, n, s, _m in got} == want and all(s >= FLOOR for _h, _n, s, _m in got)
    assert want["F1"] == 12  # non-vacuous: P1 -> F1 is the only kept edge into F1


UNFOLDED_SELECTIONS = (
    ["affiliation"],
    ["affiliation", "same_practice"],
    # hostile: reordered and duplicated input — the same walk as the sorted, de-duplicated list
    ["same_practice", "affiliation", "same_practice"],
)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_unfolded_selection_on_a_mixed_graph_equals_reference(fold_graph, workspace_client, direction):
    """Unfolded selections from P1, depths 1-2, every direction: equal the reference with
    COUNT(*) support; hop 1 of ['affiliation'] outbound is {H1: 1}."""
    g, ids, named = fold_graph, fold_graph["ids"], _named(fold_graph)
    seed = ids["provider::P1"]
    for selection in UNFOLDED_SELECTIONS:
        for depth in (1, 2):
            ref = spark_selection_walk(workspace_client, g, seed, selection, depth, folded=False,
                                       direction=direction)
            got = _live_walk(g, seed, depth, selection, direction)
            assert got == ref, f"{selection} {direction} depth {depth}: {got} != {ref}"
            assert all(named[n] not in ("P4", "P6", "F1", "F2") for _h, n, _s in got), got
    if direction == "outbound":
        hop1 = {named[n]: s for h, n, s in _live_walk(g, seed, 1, ["affiliation"]) if h == 1}
        assert hop1 == {"H1": 1}
    # Non-vacuous for every direction: the two-type selection reaches someone at hop 1.
    assert spark_selection_walk(workspace_client, g, seed, UNFOLDED_SELECTIONS[1], 1,
                                folded=False, direction=direction)


class _BorrowedConn:
    """The test's open connection handed to reapply: close() is a no-op (the test rolls back),
    every other attribute (also a set, e.g. `autocommit`) goes to the real connection."""

    def __init__(self, conn: Any):
        object.__setattr__(self, "_conn", conn)

    def close(self) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._conn, name, value)

    def __enter__(self):
        return self

    def __exit__(self, *exc: Any) -> None:
        pass


def test_reapply_detects_unfloored_version_rows(fold_graph, monkeypatch):
    """As the operator, in a rolled-back transaction: one version edges row set to weight 10;
    reapply (its freshness guard, then the fold floor detector, on this transaction's
    connection) raises FoldFloorError. The detector passes on the same rows before the UPDATE.

    If reapply ran past the detector, its index step would set autocommit on this connection
    inside the open transaction and fail with a different error, so nothing is committed."""
    import lgk.steps.reapply as reapply_mod
    from lgk.privacy.floor import FoldFloorError, assert_fold_floor_pg

    g, ids = fold_graph, fold_graph["ids"]
    edges = f"{version_schema(g)}.edges"
    conn = _pg(g)
    try:
        assert_fold_floor_pg(conn, edges, FOLD_TYPES, FLOOR)  # control: the served rows are floored
        cur = conn.execute(
            f"UPDATE {edges} SET weight = 10 WHERE relationship_type = 'shares_patients' "
            f"AND src_node_id = %s AND dst_node_id = %s", (ids["provider::P1"], ids["provider::P2"]))
        assert cur.rowcount == 1
        monkeypatch.setattr(reapply_mod, "connect_branch", lambda _config: _BorrowedConn(conn))
        with pytest.raises(FoldFloorError):
            reapply_mod.reapply(g)
    finally:
        monkeypatch.undo()
        conn.rollback()
        conn.close()
    with _pg(g) as conn:  # the rollback held
        assert conn.execute(f"SELECT min(weight) FROM {edges} WHERE weight IS NOT NULL").fetchone()[0] == FLOOR


def test_refresh_sync_refuses_unfloored_source(fold_graph, workspace_client):
    """A folded-type row with weight 10 in the source Delta edges: sync raises FoldFloorError
    and starts no pipeline update (every synced delta_commit_version is unchanged)."""
    from lgk.privacy.floor import FoldFloorError
    from lgk.refresh.freshness import synced_version
    from lgk.refresh.sync import sync

    g, ids = fold_graph, fold_graph["ids"]
    t = f"{g['lakebase_catalog']}.{g['kit_schema']}.edges"
    dests = [f"{g['lakebase_catalog']}.{version_schema(g)}.{table}" for table in SYNCED_TABLES]
    before = {d: synced_version(workspace_client, d) for d in dests}
    assert all(v is not None for v in before.values()), before
    try:
        # hostile: the operator-pipeline case — the below-floor pair P1 -> P3 written with weight 10
        _q(workspace_client, f"INSERT INTO {t} VALUES ('{INJECTED_EDGE_ID}', '{ids['provider::P1']}', "
                             f"'{ids['provider::P3']}', 'shares_patients', map(), CAST(10 AS BIGINT))")
        with pytest.raises(FoldFloorError):
            sync(g)
        assert {d: synced_version(workspace_client, d) for d in dests} == before
    finally:
        _q(workspace_client, f"DELETE FROM {t} WHERE edge_id = '{INJECTED_EDGE_ID}'")


def test_raised_floor_rebuilds(fold_graph, workspace_client):
    """privacy_floor=12 through the runner: privacy marks build not_run, build reruns, and the
    rebuilt Delta edges equal the floor-12 reference (P1-P2 at 11 is gone, P1-P4 at 12 stays)."""
    from lgk.runner import RunFolder, RunResult, RunStatus, execute_step
    from lgk.steps.build import build
    from lgk.steps.privacy import privacy

    g, ids = fold_graph, fold_graph["ids"]
    config = {**g, "privacy_floor": RAISED_FLOOR}
    rf = RunFolder(Path(g["run_folder"]))
    rf.update_step(RunResult(step="build", status=RunStatus.SUCCESS))  # the build already made
    result = execute_step(rf, config, "privacy", privacy)
    assert result.status == RunStatus.SUCCESS, result.error
    assert rf.get_step_status("build") == RunStatus.NOT_RUN, "privacy did not invalidate build"
    result = execute_step(rf, config, "build", build)
    assert result.status == RunStatus.SUCCESS, result.error

    got = {r[:4] for r in _delta_edges(workspace_client, g) if r[0] in FOLD_TYPES}
    want = {r[:4] for r in reference_edges(g, RAISED_FLOOR) if r[0] in FOLD_TYPES}
    assert got == want
    p1, p2, p4 = ids["provider::P1"], ids["provider::P2"], ids["provider::P4"]
    assert ("shares_patients", p1, p2, 11) not in got and ("shares_patients", p2, p1, 11) not in got
    assert ("shares_patients", p1, p4, 12) in got
