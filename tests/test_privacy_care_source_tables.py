"""A node type is refused when its SOURCE table holds ANY column the
classifier calls care subject, whatever column the mapping keys it on (A5: one person per node).

The rule reads flags.json `columns` (the classifier's class), not the flag's resolution, so a
drop or reclassify never clears it (2a). Fold stays the only way patient data enters the graph.

Views: profile does not tell views from tables. `_uc_columns` (src/lgk/steps/profile.py:172-181)
reads information_schema.columns with no table_type filter, so a view's columns are profiled and
it lands in schema.json as a table entry with no key_columns and no foreign_keys (views carry no
constraints). The view case below is that shape.

Known limit (not detected, documented by a passing test): a one-row-per-person table with no
care-subject column at all.
"""

import json
from unittest.mock import MagicMock

import pytest
import yaml

from lgk.privacy import gate
from lgk.privacy.flags import OpenFlagError, open_flags, read_flags_file, resolve_flag
from lgk.privacy.gate import CareKeyNodeError
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
    # case 1: the patient table, keyed on patient_id, with a neutral surrogate row_id
    {"name": "patients", "key_columns": ["patient_id"], "foreign_keys": [],
     "columns": [_col("patient_id"), _col("row_id"), _col("city")]},
    # case 2 (table form): a copy that drops the patient key but keeps patient_city (flagged, 3a)
    {"name": "patients_copy", "key_columns": ["row_id"], "foreign_keys": [],
     "columns": [_col("row_id"), _col("patient_city"), _col("title")]},
    # case 2 (view form): what profile writes for a VIEW: no key_columns, no foreign_keys
    {"name": "patient_view", "key_columns": [], "foreign_keys": [],
     "columns": [_col("row_id"), _col("patient_city"), _col("title")]},
    # case 3: a care-ish column the classifier calls none
    {"name": "panels", "key_columns": ["panel_id"], "foreign_keys": [],
     "columns": [_col("panel_id"), _col("title"), _col("patient_count")]},
    # known limit: one row per person, no care-subject column
    {"name": "people", "key_columns": ["row_id"], "foreign_keys": [],
     "columns": [_col("row_id"), _col("title"), _col("home_city")]},
]}

CARE = {("patients", "patient_id"), ("patients_copy", "patient_city"),
        ("patient_view", "patient_city")}

CLEAN = {"source_catalog": "c", "source_schema": "s",
         "nodes": {
             "providers": {"source_table": "providers", "natural_key": "provider_id",
                           "label": "name", "properties": ["specialty", "accepting_new_patients"]},
             "hospitals": {"source_table": "hospitals", "natural_key": "hospital_id",
                           "label": "name", "properties": ["city"]}},
         "edges": {}}


def with_node(node_type, table, key, label, properties=()):
    m = json.loads(json.dumps(CLEAN))
    m["nodes"][node_type] = {"source_table": table, "natural_key": key, "label": label,
                             "properties": list(properties)}
    return m


# hostile: keyed on row_id (class none) of the patient table; the care column is NOT carried
PATIENT_ROW = with_node("patient", "patients", "row_id", "city")
# hostile: the copy drops patient_id; only a non-key column (patient_city) is care
COPY = with_node("patient_copy", "patients_copy", "row_id", "title")
# hostile: the same shape as a view
VIEW = with_node("patient_view", "patient_view", "row_id", "title")
# hostile (prover M13/M14): a composite key and carried properties on the copy
COPY_COMPOSITE = with_node("patient_copy", "patients_copy", ["row_id", "title"], "title", ["title"])
REFUSED = [(PATIENT_ROW, "patient", "patients", "patients.patient_id"),
           (COPY, "patient_copy", "patients_copy", "patients_copy.patient_city"),
           (VIEW, "patient_view", "patient_view", "patient_view.patient_city"),
           (COPY_COMPOSITE, "patient_copy", "patients_copy", "patients_copy.patient_city")]


def _write(run_dir, mapping):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "schema.json").write_text(json.dumps(SCHEMA))
    (run_dir / "graph_mapping.yaml").write_text(yaml.safe_dump(mapping, sort_keys=False))


def _resolve(run_dir, choice):
    try:
        privacy({"run_folder": str(run_dir)})
    except OpenFlagError:
        pass
    assert {(f["table"], f["column"]) for f in open_flags(str(run_dir))} == CARE
    kw = {"reason": "hand-written to test the gate", "new_class": "none"} \
        if choice == "reclassify" else {}
    for table, column in sorted(CARE):
        resolve_flag(str(run_dir), table, column, choice, chooser="op", **kw)


def _cleared_then_forged(run_dir, mapping, choice):
    _write(run_dir, CLEAN)
    _resolve(run_dir, choice)
    privacy({"run_folder": str(run_dir)})
    (run_dir / "graph_mapping.yaml").write_text(yaml.safe_dump(mapping, sort_keys=False))
    data = {k: v for k, v in read_flags_file(str(run_dir)).items() if k != gate.STAMP_KEY}
    forge(str(run_dir), data)


def _message(node_type, table, care):
    return (rf"node type {node_type} is sourced from {table}, which holds care-subject "
            rf"column\(s\) {care.replace('.', r'\.')}")


@pytest.mark.parametrize("choice", ["drop", "reclassify"])
@pytest.mark.parametrize("mapping,node_type,table,care", REFUSED)
def test_the_privacy_step_refuses_a_node_on_a_care_table(tmp_path, mapping, node_type, table,
                                                         care, choice):
    _write(tmp_path, mapping)
    _resolve(tmp_path, choice)
    with pytest.raises(CareKeyNodeError, match=_message(node_type, table, care)):
        privacy({"run_folder": str(tmp_path)})
    assert gate.STAMP_KEY not in read_flags_file(str(tmp_path))


@pytest.mark.parametrize("choice", ["drop", "reclassify"])
@pytest.mark.parametrize("mapping,node_type,table,care", REFUSED)
def test_require_clear_refuses_a_forged_stamp(tmp_path, mapping, node_type, table, care, choice):
    _cleared_then_forged(tmp_path, mapping, choice)
    with pytest.raises(CareKeyNodeError, match=_message(node_type, table, care)):
        gate.require_clear(str(tmp_path))


@pytest.mark.parametrize("mapping", [PATIENT_ROW, COPY, VIEW])
def test_build_called_directly_refuses_before_any_sql(tmp_path, monkeypatch, mapping):
    from tests.test_build_sql import _Warehouse, run_build

    _cleared_then_forged(tmp_path, mapping, "reclassify")
    wh = _Warehouse()
    with pytest.raises(CareKeyNodeError):
        run_build(tmp_path, monkeypatch, wh)
    assert wh.statements == []


def test_provision_called_directly_refuses_before_any_sql(tmp_path, monkeypatch):
    from lgk.steps import provision as p

    _cleared_then_forged(tmp_path, COPY, "drop")
    statements = []
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: MagicMock())
    monkeypatch.setattr(p, "execute_sql", lambda *a, **k: statements.append(a) or [])
    with pytest.raises(CareKeyNodeError):
        p.provision({"run_folder": str(tmp_path), "lakebase_catalog": "main",
                     "kit_schema": "lgk_care", "warehouse_id": "wh", "version": 1})
    assert statements == []


def test_an_edit_after_clearing_is_refused_and_the_runner_refuses_twice(tmp_path):
    _write(tmp_path, CLEAN)
    _resolve(tmp_path, "reclassify")
    privacy({"run_folder": str(tmp_path)})
    (tmp_path / "graph_mapping.yaml").write_text(yaml.safe_dump(PATIENT_ROW, sort_keys=False))
    built = []
    for _ in range(2):
        runner = LaptopRunner({"graph_name": "g", "run_folder": str(tmp_path)})
        results = runner.run_steps([("privacy", privacy), ("build", lambda c: built.append(1))])
        assert [(r.step, r.status) for r in results] == [("privacy", RunStatus.FAILED)]
        assert "is sourced from patients" in results[0].error
    assert built == []


@pytest.mark.parametrize("choice", ["drop", "reclassify"])
def test_providers_hospitals_and_a_none_class_care_ish_column_pass(tmp_path, choice):
    """Case 3, non-vacuous: accepting_new_patients and patient_count are class none."""
    _write(tmp_path, with_node("panel", "panels", "panel_id", "title", ["patient_count"]))
    _resolve(tmp_path, choice)
    privacy({"run_folder": str(tmp_path)})
    assert gate.require_clear(str(tmp_path))


def test_person_grain_table_without_a_flag_is_a_known_limit(tmp_path):
    """KNOWN LIMIT, not a detection: `people` is one row per person, but no column of it is
    care subject, so the gate cannot tell and the node PASSES. Operators must not map such a
    table as a node; this test makes the limit visible (it will need changing if it is closed)."""
    _write(tmp_path, with_node("person", "people", "row_id", "title", ["home_city"]))
    _resolve(tmp_path, "drop")
    privacy({"run_folder": str(tmp_path)})
    assert gate.require_clear(str(tmp_path))


def _gate_on(tmp_path, mapping, schema=SCHEMA):
    """check_care_key_nodes with classes from classifying `schema` (the gate's own classifier)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "graph_mapping.yaml").write_text(yaml.safe_dump(mapping, sort_keys=False))
    m = gate.load_mapping(tmp_path / "graph_mapping.yaml")
    tables = schema["tables"]
    data = {"columns": gate.classified_columns(tables),
            "foreign_keys": gate.foreign_keys_record(tables),
            "key_columns": gate.key_columns_record(tables)}
    gate.check_care_key_nodes(data, m)


def test_every_refused_node_is_named(tmp_path):
    """(prover M12) two care-sourced nodes: both are named, not just the first."""
    m = json.loads(json.dumps(PATIENT_ROW))
    m["nodes"]["patient_copy"] = COPY["nodes"]["patient_copy"]
    with pytest.raises(CareKeyNodeError) as err:
        _gate_on(tmp_path, m)
    assert "is sourced from patients," in str(err.value)
    assert "is sourced from patients_copy," in str(err.value)


def test_a_care_key_and_a_care_source_are_both_named(tmp_path):
    """(prover M3/M8) keyed on patient_id itself: the key reason AND the source reason."""
    with pytest.raises(CareKeyNodeError) as err:
        _gate_on(tmp_path, with_node("patient", "patients", "patient_id", "city"))
    msg = str(err.value)
    assert "node type patient is keyed on care-subject column(s) patients.patient_id" in msg
    assert "node type patient is sourced from patients, which holds" in msg


def test_every_care_column_of_the_source_is_named_sorted(tmp_path):
    """(prover M4/M10) a table with two care columns, listed out of order in schema.json."""
    schema = json.loads(json.dumps(SCHEMA))
    schema["tables"].append({"name": "intake", "key_columns": ["row_id"], "foreign_keys": [],
                             "columns": [_col("row_id"), _col("patient_name"), _col("patient_city")]})
    with pytest.raises(CareKeyNodeError,
                       match=r"sourced from intake, which holds care-subject column\(s\) "
                             r"intake\.patient_city, intake\.patient_name\."):
        _gate_on(tmp_path, with_node("intake", "intake", "row_id", "row_id"), schema)
