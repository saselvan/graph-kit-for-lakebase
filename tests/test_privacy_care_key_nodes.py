"""A5b (persona round): the privacy gate refuses a node type keyed on a care-subject column,
even in a hand-written graph_mapping.yaml that never went through propose.

The check reads the CLASSIFIER's class (flags.json `columns`, pinned to schema.json by the
gate): a reclassify does not clear a node key, because a node keyed on a care subject IS that
person's node in Lakebase. A drop/fold choice would also be caught by the generic leak check;
the named error says why. Refused by the privacy step, by require_clear, by build and
provision called directly (no SQL runs), after an edit, and through the runner twice.
"""

import json
from unittest.mock import MagicMock

import pytest
import yaml

from lgk.privacy import gate
from lgk.privacy.flags import OpenFlagError, read_flags_file, resolve_flag
from lgk.privacy.gate import CareKeyNodeError, PrivacyGateError
from lgk.runner import LaptopRunner, RunStatus
from lgk.steps.privacy import privacy
from tests.forged_decision import forge


def _col(name):
    return {"name": name, "type": "STRING"}


SCHEMA = {"source_catalog": "c", "source_schema": "s", "tables": [
    {"name": "providers", "key_columns": ["provider_id"], "foreign_keys": [],
     "columns": [_col("provider_id"), _col("name"), _col("specialty"),
                 _col("accepting_new_patients")]},
    {"name": "hospitals", "key_columns": ["hospital_id"], "foreign_keys": [],
     "columns": [_col("hospital_id"), _col("name"), _col("city")]},
    {"name": "encounters", "key_columns": ["patient_id", "provider_id", "hospital_id"],
     "foreign_keys": [{"columns": ["provider_id"], "references": ["providers", "provider_id"]},
                      {"columns": ["hospital_id"], "references": ["hospitals", "hospital_id"]}],
     "columns": [_col("patient_id"), _col("provider_id"), _col("hospital_id")]},
    {"name": "patient_roster", "key_columns": ["patientid"], "foreign_keys": [],
     "columns": [_col("patientid"), _col("specialty")]},
]}

CLEAN = {"source_catalog": "c", "source_schema": "s",
         "nodes": {
             "providers": {"source_table": "providers", "natural_key": "provider_id",
                           "label": "name", "properties": ["specialty", "accepting_new_patients"]},
             "hospitals": {"source_table": "hospitals", "natural_key": "hospital_id",
                           "label": "name", "properties": ["city"]}},
         "edges": {"encounters": {"source_table": "encounters", "src": "provider_id",
                                  "src_type": "providers", "dst": "hospital_id",
                                  "dst_type": "hospitals"}}}


def hand_written(node_type="patient", table="encounters", key="patient_id", label="hospital_id"):
    """CLEAN plus a hand-written node type keyed on a care-subject column."""
    m = json.loads(json.dumps(CLEAN))
    m["nodes"][node_type] = {"source_table": table, "natural_key": key, "label": label,
                             "properties": []}
    return m


def _write(run_dir, mapping):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "schema.json").write_text(json.dumps(SCHEMA))
    (run_dir / "graph_mapping.yaml").write_text(yaml.safe_dump(mapping, sort_keys=False))


def _resolve(run_dir, choice):
    try:
        privacy({"run_folder": str(run_dir)})
    except OpenFlagError:
        pass
    for table, col in (("encounters", "patient_id"), ("patient_roster", "patientid")):
        kw = {"reason": "hand-written to test the gate", "new_class": "none"} \
            if choice == "reclassify" else {}
        resolve_flag(str(run_dir), table, col, choice, chooser="op", **kw)


def _cleared_then_forged(run_dir, mapping, choice):
    """Pass privacy on CLEAN, then swap in `mapping` and re-stamp it as if the step had
    passed (a bypass: hand edit + forged stamp). Only the gate's own checks can refuse now."""
    _write(run_dir, CLEAN)
    _resolve(run_dir, choice)
    privacy({"run_folder": str(run_dir)})
    (run_dir / "graph_mapping.yaml").write_text(yaml.safe_dump(mapping, sort_keys=False))
    data = {k: v for k, v in read_flags_file(str(run_dir)).items() if k != gate.STAMP_KEY}
    forge(str(run_dir), data)


CASES = [
    # hostile: the persona case, a node keyed on the care-subject column of encounters
    hand_written(),
    # hostile: a composite key that only CONTAINS the care-subject column
    hand_written("encounter", key=["patient_id", "provider_id", "hospital_id"]),
    # hostile (prover P6/G5): the care column LAST in a composite key
    hand_written("encounter_last", key=["provider_id", "hospital_id", "patient_id"]),
    # hostile: one glued token (the mapping contract is lower case), only the id-affix rule catches it
    hand_written("roster", table="patient_roster", key="patientid", label="specialty"),
]


def test_the_clean_mapping_passes(tmp_path):
    """Non-vacuous: the same run with no care-keyed node clears the gate."""
    _write(tmp_path, CLEAN)
    _resolve(tmp_path, "drop")
    privacy({"run_folder": str(tmp_path)})
    assert gate.require_clear(str(tmp_path))


@pytest.mark.parametrize("choice", ["drop", "reclassify"])
@pytest.mark.parametrize("mapping", CASES)
def test_the_privacy_step_refuses_and_writes_no_stamp(tmp_path, mapping, choice):
    _write(tmp_path, mapping)
    _resolve(tmp_path, choice)
    node_type = next(t for t in mapping["nodes"] if t not in CLEAN["nodes"])
    with pytest.raises(CareKeyNodeError, match=rf"node type {node_type} is keyed on care-subject"):
        privacy({"run_folder": str(tmp_path)})
    assert gate.STAMP_KEY not in read_flags_file(str(tmp_path))


@pytest.mark.parametrize("choice", ["drop", "reclassify"])
@pytest.mark.parametrize("mapping", CASES)
def test_require_clear_refuses_a_forged_stamp(tmp_path, mapping, choice):
    _cleared_then_forged(tmp_path, mapping, choice)
    with pytest.raises(CareKeyNodeError):
        gate.require_clear(str(tmp_path))


@pytest.mark.parametrize("choice", ["drop", "reclassify"])
def test_build_called_directly_refuses_before_any_sql(tmp_path, monkeypatch, choice):
    from tests.test_build_sql import _Warehouse, run_build

    _cleared_then_forged(tmp_path, hand_written(), choice)
    wh = _Warehouse()
    with pytest.raises(CareKeyNodeError):
        run_build(tmp_path, monkeypatch, wh)
    assert wh.statements == []


@pytest.mark.parametrize("choice", ["drop", "reclassify"])
def test_provision_called_directly_refuses_before_any_sql(tmp_path, monkeypatch, choice):
    from lgk.steps import provision as p

    _cleared_then_forged(tmp_path, hand_written(), choice)
    statements = []
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: MagicMock())
    monkeypatch.setattr(p, "execute_sql", lambda *a, **k: statements.append(a) or [])
    with pytest.raises(CareKeyNodeError):
        p.provision({"run_folder": str(tmp_path), "lakebase_catalog": "main",
                     "kit_schema": "lgk_care", "warehouse_id": "wh", "version": 1})
    assert statements == []


def test_an_edit_after_clearing_is_refused_and_stays_refused(tmp_path, monkeypatch):
    """Re-run after editing the input; each path still refuses."""
    from tests.test_build_sql import _Warehouse, run_build

    _write(tmp_path, CLEAN)
    _resolve(tmp_path, "reclassify")
    privacy({"run_folder": str(tmp_path)})
    (tmp_path / "graph_mapping.yaml").write_text(yaml.safe_dump(hand_written(), sort_keys=False))
    wh = _Warehouse()
    with pytest.raises(PrivacyGateError, match="stale"):
        run_build(tmp_path, monkeypatch, wh)
    with pytest.raises(CareKeyNodeError):
        privacy({"run_folder": str(tmp_path)})
    with pytest.raises(PrivacyGateError):
        run_build(tmp_path, monkeypatch, wh)
    assert wh.statements == []


def test_the_runner_refuses_twice(tmp_path):
    _write(tmp_path, hand_written())
    _resolve(tmp_path, "reclassify")
    built = []
    for _ in range(2):
        runner = LaptopRunner({"graph_name": "g", "run_folder": str(tmp_path)})
        results = runner.run_steps([("privacy", privacy), ("build", lambda c: built.append(1))])
        assert [r.step for r in results] == ["privacy"]
        assert results[0].status == RunStatus.FAILED
        assert "keyed on care-subject" in results[0].error
    assert built == []


def test_care_key_node_error_is_a_mapping_leak():
    assert issubclass(CareKeyNodeError, gate.MappingLeakError)


def test_a_clean_key_sharing_a_care_column_name_in_another_table_passes(tmp_path):
    """(prover G2) the check pairs table and column: hospitals.patient_id is not care here."""
    schema = json.loads(json.dumps(SCHEMA))
    schema["tables"].append({"name": "wards", "key_columns": ["ward_seq"], "foreign_keys": [],
                             "columns": [_col("ward_seq"), _col("title")]})
    m = json.loads(json.dumps(CLEAN))
    m["nodes"]["wards"] = {"source_table": "wards", "natural_key": "ward_seq", "label": "title",
                           "properties": []}
    _write(tmp_path, m)
    (tmp_path / "schema.json").write_text(json.dumps(schema))
    data = {"columns": [{"table": "encounters", "column": "ward_seq", "class": "care_subject"}]}
    gate.check_care_key_nodes(data, gate.load_mapping(tmp_path / "graph_mapping.yaml"))
