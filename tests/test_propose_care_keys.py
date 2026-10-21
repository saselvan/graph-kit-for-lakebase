"""A5a (persona round): propose never drafts a node type keyed on a care-subject column.

Persona 4 (graph expert) found propose drafting `encounters` (composite key with patient_id)
as a node type with `label: patient_id`: raw patient ids would have been published as node
labels. Now a table whose key holds a care-subject column (by name or by sampled values) is
never a node. It is drafted only as an edge source, by the join-table rule (exactly two FKs to
node keys, here provider -> hospital), never carrying the care column. Fold stays the privacy
step's choice plus a hand-written `fold_subject` (propose drafts no folds today).
"""

import json

import yaml

from lgk.privacy import classify
from lgk.steps.propose import propose

MRN_VALUES = [f"MRN{n:07d}" for n in range(1000, 1030)]


def _col(name, values=None):
    col = {"name": name, "type": "STRING"}
    if values is not None:
        col["patterns"] = classify.pattern_verdicts(values)
    return col


def _fk(col, table, ref):
    return {"columns": [col], "references": [table, ref]}


def persona_schema():
    """The persona fixture plus hostile care-keyed tables."""
    return {"source_catalog": "c", "source_schema": "s", "tables": [
        {"name": "providers", "key_columns": ["provider_id"], "foreign_keys": [],
         "columns": [_col("provider_id"), _col("name"), _col("specialty"), _col("location"),
                     _col("accepting_new_patients", ["Y", "N"])]},
        {"name": "hospitals", "key_columns": ["hospital_id"], "foreign_keys": [],
         "columns": [_col("hospital_id"), _col("name"), _col("city")]},
        {"name": "affiliations", "key_columns": [],
         "foreign_keys": [_fk("provider_id", "providers", "provider_id"),
                          _fk("hospital_id", "hospitals", "hospital_id")],
         "columns": [_col("provider_id"), _col("hospital_id")]},
        # hostile: composite key holding the care subject; FK to a care-keyed patients table
        {"name": "encounters", "key_columns": ["patient_id", "provider_id", "hospital_id"],
         "foreign_keys": [_fk("patient_id", "patients", "patient_id"),
                          _fk("provider_id", "providers", "provider_id"),
                          _fk("hospital_id", "hospitals", "hospital_id")],
         "columns": [_col("patient_id"), _col("provider_id"), _col("hospital_id")]},
        # hostile: a single-column care key, camelCase
        {"name": "patients", "key_columns": ["patientId"], "foreign_keys": [],
         "columns": [_col("patientId"), _col("full_name"), _col("dob")]},
        # hostile: a harmless key NAME whose VALUES are MRNs (care subject by values)
        {"name": "charts", "key_columns": ["ref"], "foreign_keys": [],
         "columns": [_col("ref", MRN_VALUES), _col("title")]},
    ]}


def _run(tmp_path, schema):
    (tmp_path / "schema.json").write_text(json.dumps(schema))
    propose({"run_folder": str(tmp_path)})
    return yaml.safe_load((tmp_path / "graph_mapping.yaml").read_text())


def test_no_node_type_is_keyed_on_a_care_subject_column(tmp_path):
    mapping = _run(tmp_path, persona_schema())
    assert sorted(mapping["nodes"]) == ["hospitals", "providers"]
    # the provider attribute stays a property (A3: not a care-subject identifier)
    assert "accepting_new_patients" in mapping["nodes"]["providers"]["properties"]


def test_a_care_keyed_table_is_drafted_as_an_edge_without_the_care_column(tmp_path):
    mapping = _run(tmp_path, persona_schema())
    assert mapping["edges"]["encounters"] == {
        "source_table": "encounters", "src": "provider_id", "src_type": "providers",
        "dst": "hospital_id", "dst_type": "hospitals"}
    assert mapping["edges"]["affiliations"]["src"] == "provider_id"
    assert sorted(mapping["edges"]) == ["affiliations", "encounters"]
    text = (tmp_path / "graph_mapping.yaml").read_text()
    assert "patient_id" not in text and "patientId" not in text


def test_a_care_keyed_table_with_other_fks_is_skipped_not_refused(tmp_path):
    # hostile: one FK beside the care key: not a join table, and it is not a node either.
    # It must be skipped (no edge), not refused as "has no primary key".
    schema = persona_schema()
    schema["tables"].append(
        {"name": "visits", "key_columns": ["visit_id"],
         "foreign_keys": [_fk("provider_id", "providers", "provider_id")],
         "columns": [_col("visit_id"), _col("provider_id")]})
    mapping = _run(tmp_path, schema)
    assert "visits" not in mapping["nodes"]
    assert not [e for e in mapping["edges"].values() if e["source_table"] == "visits"]


def test_a_keyed_table_with_an_fk_on_a_care_column_gets_no_edge_to_it(tmp_path):
    # hostile: claims is a fine node (claim_seq key), but its patient_id FK must not become
    # an edge end (it would carry patient_id as src/dst)
    schema = persona_schema()
    schema["tables"].append(
        {"name": "claims", "key_columns": ["claim_seq"],
         "foreign_keys": [_fk("patient_id", "patients", "patient_id"),
                          _fk("provider_id", "providers", "provider_id")],
         "columns": [_col("claim_seq"), _col("patient_id"), _col("provider_id")]})
    mapping = _run(tmp_path, schema)
    assert "claims" in mapping["nodes"]
    claims_edges = {k: e for k, e in mapping["edges"].items() if e["source_table"] == "claims"}
    assert claims_edges == {"claims_provider_id": {
        "source_table": "claims", "src": "claim_seq", "src_type": "claims",
        "dst": "provider_id", "dst_type": "providers"}}
    assert "patient_id" not in mapping["nodes"]["claims"]["properties"]


def test_the_care_column_last_in_a_composite_key_still_blocks_the_node(tmp_path):
    # hostile (prover P6): patient_id is the LAST key column, not the first
    schema = persona_schema()
    enc = next(t for t in schema["tables"] if t["name"] == "encounters")
    enc["key_columns"] = ["provider_id", "hospital_id", "patient_id"]
    mapping = _run(tmp_path, schema)
    assert "encounters" not in mapping["nodes"]
    assert mapping["edges"]["encounters"]["src"] == "provider_id"


def test_an_fk_to_a_care_keyed_table_is_skipped_whatever_its_column_name(tmp_path):
    # hostile (prover P2): owner_ref is not a care word, but it points at the patients table
    schema = persona_schema()
    schema["tables"].append(
        {"name": "claims", "key_columns": ["claim_seq"],
         "foreign_keys": [_fk("owner_ref", "patients", "patientId"),
                          _fk("provider_id", "providers", "provider_id")],
         "columns": [_col("claim_seq"), _col("owner_ref"), _col("provider_id")]})
    mapping = _run(tmp_path, schema)
    assert sorted(k for k, e in mapping["edges"].items() if e["source_table"] == "claims") == \
        ["claims_provider_id"]


def test_a_keyless_table_left_with_one_fk_after_the_care_fk_is_skipped(tmp_path):
    # hostile (prover P5): keyless, a care FK plus one other FK: skipped, not refused
    schema = persona_schema()
    schema["tables"].append(
        {"name": "referrals", "key_columns": [],
         "foreign_keys": [_fk("patient_id", "patients", "patientId"),
                          _fk("provider_id", "providers", "provider_id")],
         "columns": [_col("patient_id"), _col("provider_id")]})
    mapping = _run(tmp_path, schema)
    assert not [e for e in mapping["edges"].values() if e["source_table"] == "referrals"]


def test_a_key_column_missing_from_columns_is_still_classified(tmp_path):
    # hostile (prover P1): the key is not listed under columns
    schema = persona_schema()
    schema["tables"].append({"name": "roster", "key_columns": ["patient_id"], "foreign_keys": [],
                             "columns": [_col("title")]})
    assert "roster" not in _run(tmp_path, schema)["nodes"]
