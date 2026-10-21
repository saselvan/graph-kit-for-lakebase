"""Seam-2 (no cloud) acceptance tests for the privacy gate.

1. Classification by name AND sampled values; doubt counts as care subject.
2. flags.json records column, class, detected_by, choice, chooser, reason, timestamp.
3. Resolution rules: fold only for an edge/join-table key, drop, reclassify with a typed reason.
4. An open care-subject flag stops the run before build (through the runner).
5. The floor defaults to 11, can be raised, never lowered.

The live half (served counts vs the floor, the reader probe) is tests/test_privacy_live.py.
"""

import json
from pathlib import Path

import pytest
import yaml

from lgk.privacy import classify, floor
from lgk.privacy.flags import FlagChoiceError, OpenFlagError, open_flags, resolve_flag
from lgk.runner import LaptopRunner, RunStatus
from lgk.steps.privacy import MappingLeakError, privacy

MRN_VALUES = [f"MRN{n:07d}" for n in range(1000, 1050)]
def _npi(base: str) -> str:
    """A checksum-valid NPI built at run time: the 9 base digits plus the CMS check digit (Luhn, 80840 prefix).

    Fixtures never hold a 10-digit literal, so none can be a real provider's number (checked against the public NPI
    registry, 2026-10-14: none of the values below is registered).
    """
    total = 0
    for i, d in enumerate(reversed([int(c) for c in "80840" + base + "0"])):
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return base + str((10 - total % 10) % 10)


NPI_VALUES = [_npi("1234" + "56789"), _npi("1000" + "00000"), _npi("1999" + "99999")]  # valid NPI check digits


def _col(name: str, values=None) -> dict:
    """A schema.json column verdict: name plus pattern match rates (never values)."""
    col = {"name": name, "type": "VARCHAR", "match_rate": 1.0, "entropy": 0.9,
           "distinct_ratio": 1.0, "null_rate": 0.0}
    if values is not None:
        col["patterns"] = classify.pattern_verdicts(values)
    return col


def _schema(run_dir: Path, tables: list) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "schema.json").write_text(json.dumps(
        {"source_catalog": "c", "source_schema": "s", "tables": tables}))
    # propose already wrote the shape; the gate must keep it
    (run_dir / "flags.json").write_text(json.dumps({"shape": "web"}))


def _visits_and_providers(run_dir: Path) -> None:
    _schema(run_dir, [
        {"name": "providers", "key_columns": ["provider_id"], "foreign_keys": [],
         "columns": [_col("provider_id", ["p1", "p2"]), _col("npi", NPI_VALUES),
                     _col("name", ["Dr A", "Dr B"])]},
        # hostile: `ref_code` is a harmless name whose values are MRN-shaped
        # (the surrogate key is `visit_seq`: `visit_id` is itself a care identifier)
        {"name": "visits", "key_columns": ["visit_seq"],
         "foreign_keys": [{"columns": ["ref_code"], "references": ["patients", "mrn"]},
                          {"columns": ["provider_id"], "references": ["providers", "provider_id"]}],
         "columns": [_col("visit_seq", ["v1", "v2"]), _col("ref_code", MRN_VALUES),
                     _col("provider_id", ["p1", "p2"])]},
        # hostile: a care-subject PROPERTY on a node table (not an edge key) -> no fold
        {"name": "members", "key_columns": ["row_key"], "foreign_keys": [],
         "columns": [_col("row_key", ["k1", "k2"]), _col("patient_name", ["x", "y"])]},
    ])


def _flags(run_dir: Path) -> dict:
    return json.loads((run_dir / "flags.json").read_text())


def _flag(run_dir: Path, table: str, column: str) -> dict:
    hits = [f for f in _flags(run_dir)["flags"] if (f["table"], f["column"]) == (table, column)]
    assert len(hits) == 1, f"expected one flag for {table}.{column}, got {hits}"
    return hits[0]


# --------------------------------------------------------------------- 1. classification
class TestClassify:
    def test_harmless_name_with_mrn_values_is_flagged_by_values(self, tmp_path):
        # hostile: the name `ref_code` has no care-subject word; only the values can catch it
        _visits_and_providers(tmp_path)
        with pytest.raises(OpenFlagError):
            privacy({"run_folder": str(tmp_path)})
        flag = _flag(tmp_path, "visits", "ref_code")
        assert flag["class"] == "care_subject"
        assert flag["detected_by"] == "values"

    def test_name_alone_and_name_plus_values(self):
        assert classify.classify_column("patient_name", {}) == ("care_subject", "name")
        mrn = classify.pattern_verdicts(MRN_VALUES)
        assert classify.classify_column("patient_mrn", mrn) == ("care_subject", "both")

    def test_public_professional_by_name_and_by_npi_values(self):
        assert classify.classify_column("npi", classify.pattern_verdicts(NPI_VALUES))[0] == (
            "public_professional"
        )
        # hostile: harmless name, NPI-valid values (Luhn with the 80840 prefix)
        assert classify.classify_column("code", classify.pattern_verdicts(NPI_VALUES)) == (
            "public_professional", "values",
        )
        # hostile: 10 digits that FAIL the NPI check digit are not professional
        bad_npi = ["1234" + "567890", "1" * 10, "2" + "0" * 9]
        assert classify.classify_column("code", classify.pattern_verdicts(bad_npi))[0] == "none"

    def test_doubt_counts_as_care_subject(self):
        # hostile: only 30% of values are MRN-shaped - below "sure", above "no signal"
        mixed = MRN_VALUES[:3] + ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta"]
        verdict = classify.pattern_verdicts(mixed)
        assert 0.2 <= verdict["mrn"] < 0.8
        assert classify.classify_column("code", verdict) == ("care_subject", "values")
        # hostile: a professional name does not outweigh care-subject values
        assert classify.classify_column("provider_ref", classify.pattern_verdicts(MRN_VALUES)) == (
            "care_subject", "values",
        )

    def test_plain_column_is_none(self):
        assert classify.classify_column("specialty", classify.pattern_verdicts(["Cardio"])) == (
            "none", None,
        )
        # hostile: `patient` as a substring of an unrelated word is not a name match
        assert classify.classify_column("outpatientless_flag", {})[0] == "none"

    def test_verdicts_hold_rates_only_never_values(self):
        verdict = classify.pattern_verdicts(MRN_VALUES)
        assert set(verdict) == set(classify.PATTERNS)
        assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in verdict.values())
        assert not any(v in json.dumps(verdict) for v in MRN_VALUES)


# ------------------------------------------------------------------------ 2. flags.json
class TestFlagsFile:
    def test_flags_record_every_field_and_keep_the_shape(self, tmp_path):
        _visits_and_providers(tmp_path)
        with pytest.raises(OpenFlagError):
            privacy({"run_folder": str(tmp_path)})
        data = _flags(tmp_path)
        assert data["shape"] == "web"
        assert data["floor"] == floor.DEFAULT_FLOOR == 11
        flagged = {(f["table"], f["column"]) for f in data["flags"]}
        assert flagged == {("visits", "ref_code"), ("members", "patient_name")}
        for f in data["flags"]:
            assert set(f) >= {"table", "column", "class", "detected_by", "choice", "chooser",
                              "reason", "timestamp", "fold_allowed"}
            assert f["choice"] is None
        # every column is classified, not only the flagged ones
        classes = {(c["table"], c["column"]): c["class"] for c in data["columns"]}
        assert classes[("providers", "npi")] == "public_professional"
        assert classes[("providers", "name")] == "none"
        assert len(classes) == 8

    def test_resolution_is_recorded_and_survives_a_rerun(self, tmp_path):
        _visits_and_providers(tmp_path)
        with pytest.raises(OpenFlagError):
            privacy({"run_folder": str(tmp_path)})
        resolve_flag(str(tmp_path), "visits", "ref_code", "fold", chooser="op@example.com")
        with pytest.raises(OpenFlagError):  # members.patient_name is still open
            privacy({"run_folder": str(tmp_path)})
        flag = _flag(tmp_path, "visits", "ref_code")
        assert (flag["choice"], flag["chooser"]) == ("fold", "op@example.com")
        assert flag["timestamp"]
        assert [(f["table"], f["column"]) for f in open_flags(str(tmp_path))] == [
            ("members", "patient_name")
        ]


# ---------------------------------------------------------------- 3. resolution rules
class TestResolution:
    @pytest.fixture
    def run_dir(self, tmp_path):
        _visits_and_providers(tmp_path)
        with pytest.raises(OpenFlagError):
            privacy({"run_folder": str(tmp_path)})
        return str(tmp_path)

    def test_fold_refused_for_a_node_property_and_says_why(self, run_dir):
        with pytest.raises(FlagChoiceError, match="only be dropped or reclassified"):
            resolve_flag(run_dir, "members", "patient_name", "fold", chooser="op")
        assert _flag(Path(run_dir), "members", "patient_name")["choice"] is None

    def test_reclassify_needs_a_typed_reason(self, run_dir):
        for reason in (None, "", "   "):
            with pytest.raises(FlagChoiceError, match="reason"):
                resolve_flag(run_dir, "members", "patient_name", "reclassify", chooser="op",
                             reason=reason, new_class="none")
        resolve_flag(run_dir, "members", "patient_name", "reclassify", chooser="op",
                     reason="synthetic test names", new_class="none")
        f = _flag(Path(run_dir), "members", "patient_name")
        assert (f["choice"], f["reason"], f["new_class"]) == (
            "reclassify", "synthetic test names", "none")

    def test_reclassify_cannot_target_care_subject_or_unknown_class(self, run_dir):
        for bad in ("care_subject", "secret", None):
            with pytest.raises(FlagChoiceError):
                resolve_flag(run_dir, "members", "patient_name", "reclassify", chooser="op",
                             reason="r", new_class=bad)

    def test_unknown_choice_and_missing_chooser_are_refused(self, run_dir):
        with pytest.raises(FlagChoiceError):
            resolve_flag(run_dir, "members", "patient_name", "ignore", chooser="op")
        with pytest.raises(FlagChoiceError, match="chooser"):
            resolve_flag(run_dir, "members", "patient_name", "drop", chooser="")
        with pytest.raises(KeyError):
            resolve_flag(run_dir, "members", "nope", "drop", chooser="op")


# ------------------------------------------------- 4. the gate stops the run before build
class TestGateStopsTheRun:
    def test_open_flag_stops_the_run_before_build(self, tmp_path):
        _visits_and_providers(tmp_path)
        built = []
        runner = LaptopRunner({"graph_name": "g", "run_folder": str(tmp_path)})
        results = runner.run_steps([
            ("privacy", privacy),
            ("build", lambda config: built.append(True)),
        ])
        assert [r.step for r in results] == ["privacy"]
        assert results[0].status == RunStatus.FAILED
        assert "open care-subject flag" in results[0].error
        assert built == []

    def test_all_resolved_lets_build_run(self, tmp_path):
        _visits_and_providers(tmp_path)
        with pytest.raises(OpenFlagError):
            privacy({"run_folder": str(tmp_path)})
        resolve_flag(str(tmp_path), "visits", "ref_code", "fold", chooser="op")
        resolve_flag(str(tmp_path), "members", "patient_name", "drop", chooser="op")
        built = []
        runner = LaptopRunner({"graph_name": "g", "run_folder": str(tmp_path)})
        results = runner.run_steps([("privacy", privacy), ("build", lambda c: built.append(1))])
        assert [r.status for r in results] == [RunStatus.SUCCESS, RunStatus.SUCCESS]
        assert built == [1]

    def test_a_new_care_subject_column_reopens_the_gate(self, tmp_path):
        # hostile: everything was resolved, then the source grew a new care-subject column
        _visits_and_providers(tmp_path)
        with pytest.raises(OpenFlagError):
            privacy({"run_folder": str(tmp_path)})
        resolve_flag(str(tmp_path), "visits", "ref_code", "fold", chooser="op")
        resolve_flag(str(tmp_path), "members", "patient_name", "drop", chooser="op")
        privacy({"run_folder": str(tmp_path)})
        schema = json.loads((tmp_path / "schema.json").read_text())
        schema["tables"][1]["columns"].append(_col("dob"))
        (tmp_path / "schema.json").write_text(json.dumps(schema))
        with pytest.raises(OpenFlagError, match="visits.dob"):
            privacy({"run_folder": str(tmp_path)})

    def test_dropped_or_folded_column_left_in_the_mapping_is_refused(self, tmp_path):
        _visits_and_providers(tmp_path)
        with pytest.raises(OpenFlagError):
            privacy({"run_folder": str(tmp_path)})
        resolve_flag(str(tmp_path), "visits", "ref_code", "fold", chooser="op")
        resolve_flag(str(tmp_path), "members", "patient_name", "drop", chooser="op")
        # hostile: the agreed mapping still serves the dropped property on a node
        mapping = {
            "source_catalog": "c", "source_schema": "s",
            "nodes": {"member": {"source_table": "members", "natural_key": "row_key",
                                 "label": "row_key", "properties": ["patient_name"]}},
            "edges": {},
        }
        (tmp_path / "graph_mapping.yaml").write_text(yaml.safe_dump(mapping))
        # members holds a care-subject column, so the node itself is refused first
        # (CareKeyNodeError is a MappingLeakError) and names the column
        with pytest.raises(MappingLeakError, match="members.patient_name"):
            privacy({"run_folder": str(tmp_path)})
        # hostile: the folded key used as a plain (unfolded) edge endpoint (no node on a care table)
        mapping["nodes"] = {"provider": {"source_table": "providers", "natural_key": "provider_id",
                                         "label": "name", "properties": []}}
        mapping["edges"] = {"saw": {"source_table": "visits", "src": "ref_code",
                                    "src_type": "provider", "dst": "provider_id",
                                    "dst_type": "provider"}}
        (tmp_path / "graph_mapping.yaml").write_text(yaml.safe_dump(mapping))
        with pytest.raises(MappingLeakError, match="visits.ref_code"):
            privacy({"run_folder": str(tmp_path)})
        # a fold flag must be folded by an edge, so the clean mapping folds it
        mapping["edges"] = {"shares_refs": {"source_table": "visits", "src": "provider_id",
                                            "src_type": "provider", "dst": "provider_id",
                                            "dst_type": "provider", "fold_subject": "ref_code"}}
        (tmp_path / "graph_mapping.yaml").write_text(yaml.safe_dump(mapping))
        privacy({"run_folder": str(tmp_path)})


# ---------------------------------------------------------------------------- 5. floor
class TestFloor:
    def test_default_is_11_and_raisable(self):
        assert floor.checked_floor(None) == 11
        assert floor.checked_floor(11) == 11
        assert floor.checked_floor(25) == 25

    @pytest.mark.parametrize("bad", [10, 1, 0, -11, True, 11.5, "11"])
    def test_cannot_be_lowered_or_faked(self, bad):
        # hostile: True == 1 and "11" looks like 11; neither is an int floor
        with pytest.raises(ValueError):
            floor.checked_floor(bad)

    def test_gate_refuses_a_lowered_floor_in_config(self, tmp_path):
        _schema(tmp_path, [])
        with pytest.raises(ValueError, match="floor"):
            privacy({"run_folder": str(tmp_path), "privacy_floor": 10})
        privacy({"run_folder": str(tmp_path), "privacy_floor": 12})
        assert _flags(tmp_path)["floor"] == 12

    def test_walk_query_refuses_a_lowered_floor(self):
        from lgk.steps.walks import get_walk_query

        with pytest.raises(ValueError):
            get_walk_query("x", depth=1, served_schema="s", privacy_floor=10, folded=False)
        with pytest.raises(ValueError):
            get_walk_query("x", depth=1, served_schema="s", privacy_floor=10, folded=True)

    def test_fold_sql_refuses_unsafe_identifiers(self):
        with pytest.raises(ValueError):
            floor.fold_edges_sql("visits; DROP TABLE x", "ref_code", "provider_id")
