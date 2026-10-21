"""Offline: node joins in the build SQL and the build's checks.

The shape of the generated Spark SQL (one LEFT JOIN subquery per join, the floor CASE, the
helper columns), the checks build runs before any SQL (`sum` over a floating type) and before
the nodes table is replaced (a non-unique one-to-one join), and the floor check after the
INSERT (assert_join_floor) before the stamp. Values are proven live (tests/test_node_joins_live.py).

`joins_run` is the shared offline run folder: schema.json with providers, claims (care-subject
`patient_id`, a FOREIGN KEY so its flag may fold) and provider_taxonomy, the mapping with
two node joins, and the REAL privacy step with claims.patient_id resolved `fold`.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
import yaml

from lgk.mapping import MappingError
from lgk.privacy import classify
from lgk.privacy.flags import OpenFlagError, resolve_flag
from lgk.steps.privacy import privacy

JOIN_MAPPING: Dict[str, Any] = {
    "source_catalog": "main", "source_schema": "gold",
    "nodes": {
        "provider": {
            "source_table": "providers", "natural_key": "provider_id", "label": "provider_name",
            "properties": ["specialty"],
            "joins": [
                {"table": "claims", "key": "provider_id", "subject": "patient_id",
                 "aggregates": ["sum(billed_amount) as total_billed", "count(claim_id) as claim_count"]},
                {"table": "provider_taxonomy", "key": "provider_id", "columns": ["taxonomy_code"]},
            ],
        },
    },
    "edges": {},
}


def _col(name: str, type_name: str = "STRING", values: Optional[List[str]] = None) -> Dict[str, Any]:
    col = {"name": name, "type": type_name, "match_rate": 1.0, "entropy": 0.9,
           "distinct_ratio": 1.0, "null_rate": 0.0}
    if values is not None:
        col["patterns"] = classify.pattern_verdicts(values)
    return col


def join_schema(billed_type: str = "DECIMAL(18,2)") -> Dict[str, Any]:
    return {"source_catalog": "main", "source_schema": "gold", "tables": [
        {"name": "providers", "key_columns": ["provider_id"], "foreign_keys": [],
         "columns": [_col("provider_id"), _col("provider_name"), _col("specialty")]},
        {"name": "claims", "key_columns": ["claim_id"],
         "foreign_keys": [{"columns": ["patient_id"], "references": ["patients", "patient_id"]},
                          {"columns": ["provider_id"], "references": ["providers", "provider_id"]}],
         "columns": [_col("claim_id"), _col("provider_id"), _col("patient_id", "BIGINT"),
                     _col("billed_amount", billed_type), _col("allowed_amount", "DOUBLE")]},
        {"name": "provider_taxonomy", "key_columns": [], "foreign_keys": [],
         "columns": [_col("provider_id"), _col("taxonomy_code")]},
    ]}


def write_run(run_dir: Path, mapping: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "schema.json").write_text(json.dumps(schema or join_schema()))
    (run_dir / "graph_mapping.yaml").write_text(yaml.safe_dump(mapping, sort_keys=False))


def pass_privacy(run_dir: Path, folds=(("claims", "patient_id"),), drops=(), reclassify=()) -> None:
    """The real privacy step; resolve exactly the named flags, then run it again (must pass)."""
    try:
        privacy({"run_folder": str(run_dir)})
        return
    except OpenFlagError:
        pass
    for t, c in folds:
        resolve_flag(str(run_dir), t, c, "fold", chooser="op")
    for t, c in drops:
        resolve_flag(str(run_dir), t, c, "drop", chooser="op")
    for t, c in reclassify:
        resolve_flag(str(run_dir), t, c, "reclassify", chooser="op",
                     reason="an operational code, not about a patient", new_class="none")
    privacy({"run_folder": str(run_dir)})


def joins_run(run_dir: Path, mapping: Optional[Dict[str, Any]] = None,
              schema: Optional[Dict[str, Any]] = None) -> Path:
    write_run(run_dir, mapping or JOIN_MAPPING, schema)
    pass_privacy(run_dir)
    return run_dir


def mapping_with(**join0: Any) -> Dict[str, Any]:
    m = copy.deepcopy(JOIN_MAPPING)
    m["nodes"]["provider"]["joins"][0].update(join0)
    return m


class _Warehouse:
    """A scripted warehouse: records every statement; answers the two node-join queries."""

    def __init__(self, duplicate_count: Optional[int] = None, below_floor: int = 0):
        self.statements: List[str] = []
        self.duplicate_count, self.below_floor = duplicate_count, below_floor

    def __call__(self, w: Any, wh: str, sql: str, **_k: Any) -> List[List[Any]]:
        self.statements.append(sql)
        if "HAVING COUNT(*) > 1 LIMIT 1" in sql:
            return [] if self.duplicate_count is None else [[self.duplicate_count]]
        if "AS s FROM" in sql and "b.s <" in sql:
            return [[self.below_floor]]
        return []


def run_build(run_dir: Path, monkeypatch, warehouse: Optional[_Warehouse] = None) -> _Warehouse:
    wh = warehouse or _Warehouse()
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: MagicMock())
    monkeypatch.setattr("lgk.steps.build.execute_sql", wh)
    monkeypatch.setattr("lgk.steps.build._claim_kit_schema", lambda *a, **k: None)
    from tests.label_support import stub_table_identities

    stub_table_identities(monkeypatch)  # the scripted warehouse has no Delta history
    from lgk.steps.build import build

    build({"graph_name": "g", "run_folder": str(run_dir), "lakebase_catalog": "main",
           "kit_schema": "lgk_care", "warehouse_id": "wh"})
    return wh


def _nodes_insert(wh: _Warehouse) -> str:
    (sql,) = [s for s in wh.statements if "INSERT INTO main.lgk_care.nodes" in s]
    return sql


# ------------------------------------------------------------------ the build SQL shape
def test_build_sql_has_one_left_join_subquery_per_join_and_the_floor_case(tmp_path, monkeypatch):
    sql = _nodes_insert(run_build(joins_run(tmp_path), monkeypatch))
    flat = " ".join(sql.split())
    assert flat.count("LEFT JOIN (") == 2
    assert ("LEFT JOIN (SELECT provider_id AS _lgk_key, SUM(billed_amount) AS total_billed, "
            "COUNT(claim_id) AS claim_count, COUNT(DISTINCT patient_id) AS _lgk_subjects "
            "FROM main.gold.claims WHERE provider_id IS NOT NULL GROUP BY provider_id) j0 "
            "ON j0._lgk_key = n.provider_id") in flat
    assert ("LEFT JOIN (SELECT provider_id AS _lgk_key, taxonomy_code FROM main.gold.provider_taxonomy "
            "WHERE provider_id IS NOT NULL) j1 ON j1._lgk_key = n.provider_id") in flat
    assert ("map('specialty', CAST(n.specialty AS STRING), "
            "'total_billed', CASE WHEN j0._lgk_subjects >= 11 THEN CAST(j0.total_billed AS STRING) END, "
            "'claim_count', CASE WHEN j0._lgk_subjects >= 11 THEN CAST(j0.claim_count AS STRING) END, "
            "'taxonomy_code', CAST(j1.taxonomy_code AS STRING)) as properties") in flat
    # the unfloored one-to-one output carries no CASE; only the floored join has a GROUP BY
    assert "CASE WHEN j1." not in flat and flat.count("GROUP BY") == 1
    # node ids do not move: the same formula, only qualified with n.
    assert "xxhash64('provider::'||n.provider_id) as node_id" in flat
    assert "FROM main.gold.providers n" in flat and "WHERE n.provider_id IS NOT NULL" in flat


def test_count_distinct_renders_count_distinct_and_min_max_render_plainly(tmp_path, monkeypatch):
    m = mapping_with(aggregates=["count_distinct(claim_id) as claims_seen", "min(billed_amount) as low",
                                 "max(billed_amount) as high"])
    flat = " ".join(_nodes_insert(run_build(joins_run(tmp_path, m), monkeypatch)).split())
    assert ("COUNT(DISTINCT claim_id) AS claims_seen, MIN(billed_amount) AS low, "
            "MAX(billed_amount) AS high, COUNT(DISTINCT patient_id) AS _lgk_subjects") in flat


def test_a_raised_floor_reaches_the_case(tmp_path, monkeypatch):
    # hostile: varied configuration — floor 12, so a hard-coded 11 shows
    write_run(tmp_path, JOIN_MAPPING)
    resolve = {"run_folder": str(tmp_path), "privacy_floor": 12}
    with pytest.raises(OpenFlagError):
        privacy(resolve)
    resolve_flag(str(tmp_path), "claims", "patient_id", "fold", chooser="op")
    privacy(resolve)
    wh = run_build(tmp_path, monkeypatch)
    flat = " ".join(_nodes_insert(wh).split())
    assert "CASE WHEN j0._lgk_subjects >= 12 THEN" in flat and ">= 11" not in flat
    (check,) = [s for s in wh.statements if "b.s <" in s]
    assert "b.s < 12" in check


def test_a_mapping_without_joins_builds_the_same_nodes_as_before(tmp_path, monkeypatch):
    m = copy.deepcopy(JOIN_MAPPING)
    del m["nodes"]["provider"]["joins"]
    write_run(tmp_path, m, {**join_schema(), "tables": join_schema()["tables"][:1]})
    pass_privacy(tmp_path, folds=())
    wh = run_build(tmp_path, monkeypatch)
    flat = " ".join(_nodes_insert(wh).split())
    assert "JOIN" not in flat and "map('specialty', CAST(n.specialty AS STRING)) as properties" in flat
    assert not [s for s in wh.statements if "b.s <" in s or "HAVING COUNT(*) > 1" in s]


# ------------------------------------------------------------ checks before any SQL
def test_sum_over_double_is_refused(tmp_path, monkeypatch):
    # hostile: a differently typed cell — allowed_amount is DOUBLE
    m = mapping_with(aggregates=["count(claim_id) as claim_count", "sum(allowed_amount) as total_allowed"])
    joins_run(tmp_path, m)
    wh = _Warehouse()
    with pytest.raises(MappingError, match=r"nodes\.provider\.joins\[0\]\.aggregates\[1\].*DOUBLE"):
        run_build(tmp_path, monkeypatch, wh)
    assert wh.statements == []


@pytest.mark.parametrize("kind", ["FLOAT", "REAL", "double"])
def test_sum_over_any_floating_type_is_refused(tmp_path, monkeypatch, kind):
    joins_run(tmp_path, schema=join_schema(billed_type=kind))
    with pytest.raises(MappingError, match=r"aggregates\[0\]"):
        run_build(tmp_path, monkeypatch)


def test_min_max_count_over_double_are_allowed(tmp_path, monkeypatch):
    m = mapping_with(aggregates=["min(allowed_amount) as low", "max(allowed_amount) as high",
                                 "count(allowed_amount) as n_allowed"])
    wh = run_build(joins_run(tmp_path, m), monkeypatch)
    assert "MIN(allowed_amount) AS low" in _nodes_insert(wh)


def test_sum_over_decimal_builds_and_summing_the_folded_subject_is_refused(tmp_path, monkeypatch):
    m = mapping_with(aggregates=["sum(patient_id) as never_served"])  # BIGINT, but the folded subject
    write_run(tmp_path, m)
    # summing the folded subject is refused by the gate (rule 1), not by the type check
    from lgk.privacy.gate import MappingLeakError
    with pytest.raises(MappingLeakError):
        pass_privacy(tmp_path)
    wh = run_build(joins_run(tmp_path / "ok"), monkeypatch)  # DECIMAL(18,2) sum builds
    assert "SUM(billed_amount) AS total_billed" in _nodes_insert(wh)


# ------------------------------------------------ the uniqueness query, before the replace
def test_non_unique_columns_join_is_refused_before_nodes_is_replaced(tmp_path, monkeypatch):
    from lgk.steps.build import JoinNotUniqueError

    # hostile: duplicate at the join grain (one provider listed twice in provider_taxonomy)
    wh = _Warehouse(duplicate_count=2)
    with pytest.raises(JoinNotUniqueError, match=r"provider_taxonomy.*2 rows"):
        run_build(joins_run(tmp_path), monkeypatch, wh)
    (check,) = [s for s in wh.statements if "HAVING COUNT(*) > 1" in s]
    assert check == ("SELECT COUNT(*) FROM main.gold.provider_taxonomy WHERE provider_id IS NOT NULL "
                     "GROUP BY provider_id HAVING COUNT(*) > 1 LIMIT 1")
    assert not [s for s in wh.statements if "main.lgk_care.nodes" in s], "nodes was touched"


def test_aggregate_joins_get_no_uniqueness_query(tmp_path, monkeypatch):
    m = copy.deepcopy(JOIN_MAPPING)
    m["nodes"]["provider"]["joins"] = m["nodes"]["provider"]["joins"][:1]
    wh = run_build(joins_run(tmp_path, m), monkeypatch, _Warehouse(duplicate_count=5))
    assert not [s for s in wh.statements if "HAVING COUNT(*) > 1" in s]


# ------------------------------------------------- the floor check, after the INSERT
def test_join_floor_check_runs_after_the_insert_and_before_the_stamp(tmp_path, monkeypatch):
    wh = run_build(joins_run(tmp_path), monkeypatch)
    order = wh.statements
    insert = next(i for i, s in enumerate(order) if "INSERT INTO main.lgk_care.nodes" in s)
    check = next(i for i, s in enumerate(order) if "b.s <" in s)
    stamp = next(i for i, s in enumerate(order) if "lgk.privacy_stamp" in s)
    assert insert < check < stamp
    flat = " ".join(order[check].split())
    assert flat == ("SELECT COUNT(DISTINCT d.node_id) FROM main.lgk_care.nodes d "
                    "LEFT JOIN (SELECT CAST(xxhash64('provider::'||n.provider_id) AS STRING) AS node_id, j.s "
                    "FROM main.gold.providers n "
                    "LEFT JOIN (SELECT provider_id AS _lgk_key, COUNT(DISTINCT patient_id) AS s FROM main.gold.claims "
                    "WHERE provider_id IS NOT NULL GROUP BY provider_id) j ON j._lgk_key = n.provider_id) b "
                    "ON b.node_id = d.node_id "
                    "WHERE d.node_type = 'provider' AND "
                    "(d.properties['total_billed'] IS NOT NULL OR d.properties['claim_count'] IS NOT NULL) "
                    "AND (b.s IS NULL OR b.s < 11)")
    bindings = json.loads((tmp_path / "privacy_bindings.json").read_text())
    assert bindings["build"]["joins"] == {
        "provider.claim_count": {"floor": 11, "floored": True},
        "provider.taxonomy_code": {"floor": 11, "floored": False},
        "provider.total_billed": {"floor": 11, "floored": True},
    }


def test_a_node_below_the_floor_with_outputs_fails_the_build_before_the_stamp(tmp_path, monkeypatch):
    from lgk.privacy.floor import JoinFloorError

    wh = _Warehouse(below_floor=1)
    with pytest.raises(JoinFloorError, match=r"provider\.joins\[0\].*claims"):
        run_build(joins_run(tmp_path), monkeypatch, wh)
    assert not [s for s in wh.statements if "lgk.privacy_stamp" in s]
    assert not (tmp_path / "privacy_bindings.json").exists()


# ------------------------------------------------------- provision and sync re-check
def _provision_refusal(run_dir: Path, monkeypatch, wh: _Warehouse) -> None:
    from lgk.steps import provision as p

    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: MagicMock())
    monkeypatch.setattr(p, "execute_sql", wh)
    monkeypatch.setattr(p.binding, "check_table_stamps", lambda *a, **k: None)
    from lgk.privacy.gate import require_clear
    from lgk.privacy import binding

    cleared = require_clear(str(run_dir))
    config = {"run_folder": str(run_dir), "lakebase_catalog": "main", "kit_schema": "lgk_care",
              "warehouse_id": "wh"}
    p._refuse_unbound_build(config, str(run_dir), binding.stamp_id(cleared), cleared)


def test_provision_refuses_a_recorded_join_floor_that_differs(tmp_path, monkeypatch):
    from lgk.privacy.binding import StaleBuildError

    run_build(joins_run(tmp_path), monkeypatch)
    path = tmp_path / "privacy_bindings.json"
    _provision_refusal(tmp_path, monkeypatch, _Warehouse())  # non-vacuous: the real record passes
    data = json.loads(path.read_text())
    data["build"]["joins"]["provider.total_billed"]["floor"] = 12
    path.write_text(json.dumps(data))
    with pytest.raises(StaleBuildError, match=r"provider\.total_billed \(floor 12\)"):
        _provision_refusal(tmp_path, monkeypatch, _Warehouse())
    del data["build"]["joins"]
    path.write_text(json.dumps(data))
    with pytest.raises(StaleBuildError, match=r"not recorded"):
        _provision_refusal(tmp_path, monkeypatch, _Warehouse())


def test_provision_runs_the_join_floor_check_on_delta_nodes(tmp_path, monkeypatch):
    from lgk.privacy.floor import JoinFloorError

    run_build(joins_run(tmp_path), monkeypatch)
    with pytest.raises(JoinFloorError):
        _provision_refusal(tmp_path, monkeypatch, _Warehouse(below_floor=3))


def test_refresh_sync_runs_the_join_floor_check_before_any_update(tmp_path, monkeypatch):
    from lgk.privacy.floor import JoinFloorError
    from lgk.refresh import sync as s

    run_build(joins_run(tmp_path), monkeypatch)
    wh = _Warehouse(below_floor=2)
    sql = lambda q: wh(None, "wh", q)  # noqa: E731
    config = {"run_folder": str(tmp_path), "lakebase_catalog": "main", "kit_schema": "lgk_care"}
    monkeypatch.setattr("lgk.refresh.job.refresh_chain", lambda *a, **k: ["build"])
    with pytest.raises(JoinFloorError):
        s._require_stamped_tables(sql, config, None)
    assert [q for q in wh.statements if "b.s < 11" in q]


def test_refresh_sync_runs_the_join_floor_check_on_the_operator_chain(tmp_path, monkeypatch):
    from lgk.privacy.floor import JoinFloorError
    from lgk.refresh import sync as s

    run_build(joins_run(tmp_path), monkeypatch)
    wh = _Warehouse(below_floor=2)
    # hostile: an operator-pipeline-only chain (no kit build), stamps all current
    config = {"run_folder": str(tmp_path), "lakebase_catalog": "main", "kit_schema": "lgk_care",
              "sdp_pipeline_id": "p-1"}
    monkeypatch.setattr(s.label, "check_labels", lambda *a, **k: None)
    with pytest.raises(JoinFloorError):
        s._require_stamped_tables(lambda q: wh(None, "wh", q), config, None)


def test_operator_stamp_helper_refuses_below_the_join_floor(tmp_path, monkeypatch):
    from lgk.privacy.floor import JoinFloorError
    from lgk.refresh.operator_stamp import stamp_operator_tables

    # hostile: a RAISED floor (12), so a helper that checks the default 11 is caught
    run_build(_raised_floor_run(tmp_path), monkeypatch)
    wh = _Warehouse(below_floor=2)  # hostile: the operator's nodes carry a below-floor join output
    config = {"run_folder": str(tmp_path), "lakebase_catalog": "main", "kit_schema": "lgk_care"}
    with pytest.raises(JoinFloorError):
        stamp_operator_tables(config, sql=lambda q: wh(None, "wh", q))
    checks = [q for q in wh.statements if "b.s <" in q]
    assert checks and all("b.s < 12" in q and "FROM main.lgk_care.nodes d" in q for q in checks)
    assert not any("SET TBLPROPERTIES" in q for q in wh.statements)  # nothing stamped


# ------------------------------------- every join, not only the first (prover survivors)
SECOND_FLOORED = {"table": "claims", "key": "provider_id", "subject": "patient_id",
                  "aggregates": ["max(billed_amount) as top_billed"]}


def three_joins(third: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Floored join at 0, the one-to-one join at 1, a SECOND floored join at 2."""
    m = copy.deepcopy(JOIN_MAPPING)
    m["nodes"]["provider"]["joins"].append(copy.deepcopy(third or SECOND_FLOORED))
    return m


class _SecondJoinBelow(_Warehouse):
    """Answers 'below the floor' only for the floor check of the join outputting top_billed."""

    def __call__(self, w: Any, wh: str, sql: str, **_k: Any) -> List[List[Any]]:
        if "b.s <" in sql:
            self.statements.append(sql)
            return [[1 if "properties['top_billed']" in sql else 0]]
        return super().__call__(w, wh, sql)


def test_every_floored_join_gets_its_case_and_its_own_floor_check(tmp_path, monkeypatch):
    # hostile: a second floored join, after an unfloored one (indices 0 and 2)
    wh = run_build(joins_run(tmp_path, three_joins()), monkeypatch)
    flat = " ".join(_nodes_insert(wh).split())
    assert "'top_billed', CASE WHEN j2._lgk_subjects >= 11 THEN CAST(j2.top_billed AS STRING) END" in flat
    assert "CASE WHEN j0._lgk_subjects >= 11" in flat and "CASE WHEN j1." not in flat
    checks = [s for s in wh.statements if "b.s <" in s]
    assert len(checks) == 2 and "properties['top_billed']" in checks[1]


def test_a_second_floored_join_below_the_floor_fails_the_build(tmp_path, monkeypatch):
    from lgk.privacy.floor import JoinFloorError

    with pytest.raises(JoinFloorError, match=r"provider\.joins\[2\]") as e:
        run_build(joins_run(tmp_path, three_joins()), monkeypatch, _SecondJoinBelow())
    assert "joins[0]" not in str(e.value)


def test_sum_over_double_in_a_second_join_is_refused(tmp_path, monkeypatch):
    third = {**SECOND_FLOORED, "aggregates": ["sum(allowed_amount) as total_allowed"]}
    joins_run(tmp_path, three_joins(third))
    with pytest.raises(MappingError, match=r"nodes\.provider\.joins\[2\]\.aggregates\[0\]"):
        run_build(tmp_path, monkeypatch)


def _raised_floor_run(run_dir: Path) -> Path:
    write_run(run_dir, JOIN_MAPPING)
    config = {"run_folder": str(run_dir), "privacy_floor": 12}
    with pytest.raises(OpenFlagError):
        privacy(config)
    resolve_flag(str(run_dir), "claims", "patient_id", "fold", chooser="op")
    privacy(config)
    return run_dir


def test_provision_checks_the_join_floor_at_the_raised_floor(tmp_path, monkeypatch):
    # hostile: varied configuration — a provision that forgets the raised floor checks `< 11`
    run_build(_raised_floor_run(tmp_path), monkeypatch)
    wh = _Warehouse()
    _provision_refusal(tmp_path, monkeypatch, wh)
    checks = [s for s in wh.statements if "b.s <" in s]
    assert checks and all("b.s < 12" in s for s in checks)


def test_refresh_sync_checks_the_join_floor_at_the_raised_floor(tmp_path, monkeypatch):
    from lgk.refresh import sync as s

    run_build(_raised_floor_run(tmp_path), monkeypatch)
    wh = _Warehouse()
    config = {"run_folder": str(tmp_path), "lakebase_catalog": "main", "kit_schema": "lgk_care"}
    monkeypatch.setattr("lgk.refresh.job.refresh_chain", lambda *a, **k: ["build"])
    monkeypatch.setattr(s.label, "check_labels", lambda *a, **k: None)
    s._require_stamped_tables(lambda q: wh(None, "wh", q), config, None)
    checks = [q for q in wh.statements if "b.s <" in q]
    assert checks and all("b.s < 12" in q for q in checks)


# ------------------------- the floor check finds nodes the way build joins them (fix round 1)
def _ref_key_schema(join_key_type: str = "STRING") -> Dict[str, Any]:
    """claims carries the provider as `provider_ref` (a name other than the node's natural_key)."""
    schema = join_schema()
    schema["tables"][1]["columns"].append(_col("provider_ref", join_key_type))
    return schema


def _ref_key_mapping() -> Dict[str, Any]:
    return mapping_with(key="provider_ref")


@pytest.mark.parametrize("join_key_type", ["BIGINT", "INT", "DECIMAL(10,0)", "VARCHAR(10)"])
def test_a_join_key_of_another_type_than_the_natural_key_is_refused(tmp_path, monkeypatch, join_key_type):
    # hostile: equal in value, other type ('007' STRING vs 7 BIGINT): the join matches by value,
    # but a node id hashed from the joined key would not be the node's
    joins_run(tmp_path, _ref_key_mapping(), _ref_key_schema(join_key_type))
    wh = _Warehouse()
    with pytest.raises(MappingError, match=r"'nodes\.provider\.joins\[0\]\.key'.*claims\.provider_ref.*"
                                           r"natural_key providers\.provider_id is STRING"):
        run_build(tmp_path, monkeypatch, wh)
    assert wh.statements == []


def test_a_join_key_type_differing_only_in_case_is_accepted(tmp_path, monkeypatch):
    wh = run_build(joins_run(tmp_path, _ref_key_mapping(), _ref_key_schema("string")), monkeypatch)
    assert _nodes_insert(wh)


def test_floor_check_identifies_nodes_with_builds_join_and_node_id(tmp_path, monkeypatch):
    """The floor check joins the node's source row on build's ON condition and takes build's
    node id expression over that row; a check re-deriving the id from the joined key differs."""
    import re

    # hostile: the join key has another NAME than the natural key, so `j.provider_ref` and
    # `n.provider_id` are told apart
    wh = run_build(joins_run(tmp_path, _ref_key_mapping(), _ref_key_schema()), monkeypatch)
    insert = " ".join(_nodes_insert(wh).split())
    (check,) = [" ".join(s.split()) for s in wh.statements if "b.s <" in s]
    build_on = re.search(r"\) j0 ON j0\._lgk_key = (\S+)", insert).group(1)
    check_on = re.search(r"\) j ON j\._lgk_key = ([^)\s]+)", check).group(1)
    assert build_on == check_on == "n.provider_id"
    build_id = re.search(r"SELECT (xxhash64\('provider::'\|\|\S+?\)) as node_id", insert).group(1)
    assert f"CAST({build_id} AS STRING) AS node_id, j.s FROM main.gold.providers n" in check
    assert "SELECT provider_ref AS _lgk_key" in check and "GROUP BY provider_ref" in check


# ------------------------------------------- the key type check, per case (fix round 1 prover)
def _schema_with(table: int, column: str, type_name: Optional[str]) -> Dict[str, Any]:
    schema = join_schema()
    for c in schema["tables"][table]["columns"]:
        if c["name"] == column:
            if type_name is None:
                del c["type"]
            else:
                c["type"] = type_name
    return schema


def _refused_key(tmp_path: Path, monkeypatch, mapping: Dict[str, Any], schema: Dict[str, Any],
                 path: str = r"joins\[0\]\.key") -> None:
    joins_run(tmp_path, mapping, schema)
    with pytest.raises(MappingError, match=rf"'nodes\.provider\.{path}'"):
        run_build(tmp_path, monkeypatch)


def test_a_same_named_join_key_of_another_type_is_refused(tmp_path, monkeypatch):
    # hostile: claims.provider_id BIGINT, providers.provider_id STRING, same column name
    _refused_key(tmp_path, monkeypatch, JOIN_MAPPING, _schema_with(1, "provider_id", "BIGINT"))


def test_the_natural_key_type_is_read_from_the_node_table(tmp_path, monkeypatch):
    # hostile: claims has a BIGINT column NAMED like the natural key, but the join key used is
    # provider_ref (STRING, = providers.provider_id STRING): accepted. Reading the natural_key's
    # type from the join table would see BIGINT and refuse.
    schema = _schema_with(1, "provider_id", "BIGINT")
    schema["tables"][1]["columns"].append(_col("provider_ref", "STRING"))
    m = mapping_with(key="provider_ref")
    m["nodes"]["provider"]["joins"] = m["nodes"]["provider"]["joins"][:1]
    wh = run_build(joins_run(tmp_path, m, schema), monkeypatch)
    assert "SELECT provider_ref AS _lgk_key" in _nodes_insert(wh)


@pytest.mark.parametrize("side", ["join key", "natural_key", "both"])
def test_an_unknown_key_type_is_refused(tmp_path, monkeypatch, side):
    schema = join_schema()
    if side in ("join key", "both"):
        schema = _schema_with(1, "provider_id", None)
    if side in ("natural_key", "both"):
        schema["tables"][0] = _schema_with(0, "provider_id", None)["tables"][0]
    joins_run(tmp_path, JOIN_MAPPING, schema)
    with pytest.raises(MappingError, match=r"'nodes\.provider\.joins\[0\]\.key'.*unknown"):
        run_build(tmp_path, monkeypatch)


def test_decimal_keys_of_another_precision_are_refused(tmp_path, monkeypatch):
    schema = _schema_with(0, "provider_id", "DECIMAL(10,0)")
    schema["tables"][1] = _schema_with(1, "provider_id", "DECIMAL(12,0)")["tables"][1]
    schema["tables"][2] = _schema_with(2, "provider_id", "DECIMAL(10,0)")["tables"][2]
    _refused_key(tmp_path, monkeypatch, JOIN_MAPPING, schema)


def test_a_columns_join_key_of_another_type_is_refused(tmp_path, monkeypatch):
    # hostile: the aggregate join (index 0) is fine; the one-to-one join at index 1 is not
    _refused_key(tmp_path, monkeypatch, JOIN_MAPPING, _schema_with(2, "provider_id", "INT"),
                 path=r"joins\[1\]\.key")


# ------------------------------------- walk keys x node joins: a join output is never a walk key
@pytest.mark.parametrize("join_output", ["claim_count", "total_billed", "taxonomy_code"])
def test_a_join_output_is_refused_as_a_walk_key(tmp_path, join_output):
    """Walk keys and node joins together: a walk key must be a native property. An aggregate
    output (a floored figure over care-subject rows) must never become a seed / stop selector,
    and the narrower rule refuses a one-to-one `columns` output too."""
    import copy

    from lgk.privacy.gate import MappingLeakError, require_clear

    ok = copy.deepcopy(JOIN_MAPPING)
    ok["nodes"]["provider"]["walk_keys"] = ["specialty"]  # control: a native property passes
    write_run(tmp_path / "ok", ok)
    pass_privacy(tmp_path / "ok")
    assert require_clear(str(tmp_path / "ok"))

    m = copy.deepcopy(JOIN_MAPPING)
    # hostile: a walk key naming a join output that the gate otherwise clears
    m["nodes"]["provider"]["walk_keys"] = ["specialty", join_output]
    write_run(tmp_path / "bad", m)
    with pytest.raises(MappingLeakError, match=rf"walk_keys {join_output} \(node provider\)"):
        pass_privacy(tmp_path / "bad")
