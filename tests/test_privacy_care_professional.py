"""Owner decision A (2026-10-05): a typed reclassify to public_professional clears a column for the
care-subject rule, so a find-a-doctor graph can serve doctors' names (`providers.full_name`). A drop or a
reclassify to none still does not clear it (2a), and the reclassify clears only that one column
of that one table."""

import json

import pytest
import yaml

from lgk.privacy import gate
from lgk.privacy.flags import OpenFlagError, open_flags, resolve_flag
from lgk.privacy.gate import CareKeyNodeError
from lgk.steps.privacy import privacy


def _col(name):
    return {"name": name, "type": "STRING"}


SCHEMA = {"source_catalog": "c", "source_schema": "s", "tables": [
    {"name": "providers", "key_columns": ["provider_id"], "foreign_keys": [],
     "columns": [_col("provider_id"), _col("full_name"), _col("specialty")]},
    # hostile: the same column name in a second table; reclassifying providers must not clear it
    {"name": "members", "key_columns": ["member_ref"], "foreign_keys": [],
     "columns": [_col("member_ref"), _col("full_name"), _col("plan")]},
    # hostile: a doctor-name column AND a patient column in one table
    {"name": "panel_rows", "key_columns": ["row_ref"], "foreign_keys": [],
     "columns": [_col("row_ref"), _col("full_name"), _col("patient_id")]},
]}


def _node(table, key, label):
    return {"source_table": table, "natural_key": key, "label": label, "properties": []}


def _mapping(*node_types):
    nodes = {"providers": {"source_table": "providers", "natural_key": "provider_id",
                           "label": "full_name", "properties": ["specialty"]}}
    nodes.update({t: _node(t, k, l) for t, k, l in node_types})
    return {"source_catalog": "c", "source_schema": "s", "nodes": nodes, "edges": {}}


def _write(run_dir, mapping):
    (run_dir / "schema.json").write_text(json.dumps(SCHEMA))
    (run_dir / "graph_mapping.yaml").write_text(yaml.safe_dump(mapping, sort_keys=False))


def _open_flags(run_dir):
    try:
        privacy({"run_folder": str(run_dir)})
    except OpenFlagError:
        pass
    return {(f["table"], f["column"]) for f in open_flags(str(run_dir))}


def _resolve_rest(run_dir, skip):
    """Drop every open flag except `skip` (so only the decision under test matters)."""
    for table, column in sorted(_open_flags(run_dir) - {skip}):
        resolve_flag(str(run_dir), table, column, "drop", chooser="op")


def _reclassify(run_dir, table, column, new_class):
    resolve_flag(str(run_dir), table, column, "reclassify", chooser="op",
                 reason="a provider's public directory name", new_class=new_class)


def test_full_name_is_flagged_care_subject_by_the_classifier(tmp_path):
    _write(tmp_path, _mapping())
    assert ("providers", "full_name") in _open_flags(tmp_path)


def test_a_reclassify_to_public_professional_serves_the_doctor_name(tmp_path):
    _write(tmp_path, _mapping())
    _resolve_rest(tmp_path, ("providers", "full_name"))
    _reclassify(tmp_path, "providers", "full_name", "public_professional")
    privacy({"run_folder": str(tmp_path)})
    assert gate.require_clear(str(tmp_path))


@pytest.mark.parametrize("choice", ["drop", "none"])
def test_a_drop_or_a_reclassify_to_none_still_refuses(tmp_path, choice):
    _write(tmp_path, _mapping())
    _resolve_rest(tmp_path, ("providers", "full_name"))
    if choice == "drop":
        resolve_flag(str(tmp_path), "providers", "full_name", "drop", chooser="op")
    else:
        _reclassify(tmp_path, "providers", "full_name", "none")
    with pytest.raises(CareKeyNodeError, match=r"providers is sourced from providers"):
        privacy({"run_folder": str(tmp_path)})


def test_the_reclassify_clears_only_its_own_table(tmp_path):
    _write(tmp_path, _mapping(("members", "member_ref", "plan")))
    _resolve_rest(tmp_path, ("providers", "full_name"))
    _reclassify(tmp_path, "providers", "full_name", "public_professional")
    with pytest.raises(CareKeyNodeError, match=r"members, which holds care-subject column\(s\) "
                                               r"members\.full_name") as e:
        privacy({"run_folder": str(tmp_path)})
    assert "sourced from providers" not in str(e.value)


def test_the_reclassify_clears_only_its_own_column(tmp_path):
    _write(tmp_path, _mapping(("panel_rows", "row_ref", "row_ref")))
    _resolve_rest(tmp_path, ("panel_rows", "full_name"))
    _reclassify(tmp_path, "panel_rows", "full_name", "public_professional")
    with pytest.raises(CareKeyNodeError) as e:
        privacy({"run_folder": str(tmp_path)})
    msg = str(e.value)
    assert "panel_rows.patient_id" in msg and "panel_rows.full_name" not in msg
