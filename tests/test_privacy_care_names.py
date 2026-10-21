"""A3 (persona round): a provider ATTRIBUTE whose name merely contains a care word is not a
care-subject identifier; a real care-subject column still is.

Every persona hit `providers.accepting_new_patients` flagged as care subject (detected by
name: the bare word "patients"), and had to reclassify it by hand. The rule now: a care
subject word ("patient", "member", "pt", ...) makes a column care subject when the column is
about ONE such person: the word alone, glued to an id affix ("PatientID", "ptkey"), or
qualified by another attribute ("patient_city"). It does not when the column is an attribute
SHAPE: a plural at the end ("accepting_new_patients", a population the provider serves) or a
figure / flag suffix at the end ("patient_count", "new_patient_ok"). Direct identifiers (mrn,
ssn, dob, email, ...) stay care subject in any shape.
"""

import json

import pytest

from lgk.privacy import classify
from lgk.privacy.flags import OpenFlagError
from lgk.steps.privacy import privacy

CARE, NONE = classify.CARE_SUBJECT, classify.NONE

# column -> expected class, and why. "before" is the class at pin faf7e73 (for the report).
HOSTILE_TABLE = {
    # hostile: the persona false positive; a Y/N scheduling flag of the provider
    "accepting_new_patients": NONE,     # before: care (bare plural "patients")
    # hostile: a figure about many patients (aggregate, identifies no one)
    "patients_seen_count": NONE,        # before: care (bare "patients")
    "patient_count": NONE,              # before: care (bare "patient")
    "new_patient_ok": NONE,             # before: care (bare "patient")
    "accepts_medicare_patients": NONE,  # before: care (bare plural "patients")
    # real care-subject identifiers: must stay flagged
    "patient_id": CARE,                 # subject word + id suffix
    "patientId": CARE,                  # camelCase split -> [patient, id]
    "PatientID": CARE,                  # camelCase with an upper-case suffix
    "patientid": CARE,                  # hostile: one glued token, only the id-affix rule splits it
    "pt_id": CARE,
    "patient_mrn": CARE,
    "member_id": CARE,
    "mrn": CARE,
    "subject_id": CARE,
    "person_id": CARE,
    # a subject word qualified by another attribute is about one person (quasi-identifier)
    "patient_city": CARE,
    "patients": CARE,                   # the bare word alone: doubt plays safe
    "referring_patient": CARE,          # a singular at the end names one person
    # hostile: a direct identifier in an attribute shape stays care subject
    "mrn_count": CARE,
    # hostile (prover C5): a glued plural id is an identifier, not a population
    "referring_patientids": CARE,
    "other_ptids": CARE,
}


@pytest.mark.parametrize("column, expected", sorted(HOSTILE_TABLE.items()))
def test_hostile_column_class_by_name(column, expected):
    assert classify.classify_column(column, {})[0] == expected


def test_values_still_beat_an_attribute_shaped_name():
    # hostile: an attribute-shaped name whose VALUES are MRNs is care subject by values
    rates = classify.pattern_verdicts([f"MRN{n:07d}" for n in range(1000, 1020)])
    assert classify.classify_column("patient_count", rates) == (CARE, "values")


def test_the_privacy_step_flags_only_the_real_identifiers(tmp_path):
    """Through the shipped step: flags.json holds a flag for exactly the care columns."""
    cols = [{"name": c, "type": "STRING"} for c in HOSTILE_TABLE]
    (tmp_path / "schema.json").write_text(json.dumps({
        "source_catalog": "c", "source_schema": "s",
        "tables": [{"name": "providers_wide", "key_columns": [], "foreign_keys": [],
                    "columns": cols}]}))
    with pytest.raises(OpenFlagError):
        privacy({"run_folder": str(tmp_path)})
    data = json.loads((tmp_path / "flags.json").read_text())
    flagged = {f["column"] for f in data["flags"]}
    assert flagged == {c for c, cls in HOSTILE_TABLE.items() if cls == CARE}
    assert "accepting_new_patients" not in flagged


def test_the_persona_providers_table_has_no_flag(tmp_path):
    """The persona fixture's providers table alone passes the privacy step with no flag."""
    (tmp_path / "schema.json").write_text(json.dumps({
        "source_catalog": "c", "source_schema": "s",
        "tables": [{"name": "providers", "key_columns": ["provider_id"], "foreign_keys": [],
                    "columns": [{"name": c, "type": "STRING", "patterns":
                                 classify.pattern_verdicts(v)} for c, v in (
                        ("provider_id", ["prov_001", "prov_002"]), ("name", ["Dr A", "Dr B"]),
                        ("specialty", ["Cardiology"]), ("location", ["Springfield"]),
                        ("accepting_new_patients", ["Y", "N"]))]}]}))
    privacy({"run_folder": str(tmp_path)})
    assert json.loads((tmp_path / "flags.json").read_text())["flags"] == []


ATTRIBUTE_SUFFIXES = ("count", "cnt", "total", "sum", "avg", "mean", "pct", "percent", "rate",
                      "ratio", "size", "ok", "flag", "flg", "ind", "yn", "allowed")


def test_the_attribute_suffix_set_is_exactly_the_proven_one():
    assert classify._ATTRIBUTE_SUFFIXES == set(ATTRIBUTE_SUFFIXES)


@pytest.mark.parametrize("suffix", ATTRIBUTE_SUFFIXES)
def test_every_attribute_suffix_is_load_bearing(monkeypatch, suffix):
    # hostile: the subject word sits two tokens before the suffix, so only "last token" counts
    name = f"patient_seen_{suffix}"
    assert classify.classify_column(name, {})[0] == NONE
    monkeypatch.setattr(classify, "_ATTRIBUTE_SUFFIXES", classify._ATTRIBUTE_SUFFIXES - {suffix})
    assert classify.classify_column(name, {})[0] == CARE


def test_a_suffix_must_be_the_last_token():
    # hostile: "count" in the middle does not make the subject an aggregate
    assert classify.classify_column("patient_count_id", {})[0] == CARE
