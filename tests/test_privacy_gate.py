"""Seam-2 tests for the privacy gate: it cannot be bypassed.

B1 build and provision call ONE gate (lgk.privacy.gate.require_clear) before any SQL; it needs
   flags.json, a passed-and-fresh stamp, no open flag, no dropped/folded/unclassified column.
B2 a mapping column with no classification is refused (doubt plays safe).
B3 names are tokenised (camelCase, run-together) and direct / contact identifiers are care
   subject.
Also: label leak, single care-subject PK not foldable, NULL-diluted MRN, K_ANON floor, walks
reading folded/floor from flags.json, UC profile writing pattern rates only.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from lgk.privacy import classify, gate
from lgk.privacy.flags import OpenFlagError, read_flags_file, resolve_flag, write_flags_file
from lgk.privacy.gate import MappingLeakError, PrivacyGateError
from lgk.runner import LaptopRunner, RunStatus
from lgk.steps.privacy import privacy
from tests.test_privacy import MRN_VALUES, _col, _visits_and_providers
from tests.forged_decision import forge

PROVIDER_MAPPING = {
    "source_catalog": "c", "source_schema": "s",
    "nodes": {"provider": {"source_table": "providers", "natural_key": "provider_id",
                           "label": "name", "properties": ["npi"]}},
    "edges": {},
}


# check_fold_subjects rule 3: a flag that chose fold must be folded by an edge, so a
# run that folds visits.ref_code maps a co-membership fold edge over it.
FOLDED_MAPPING = {
    **PROVIDER_MAPPING,
    "edges": {"shares_refs": {"source_table": "visits", "src": "provider_id", "src_type": "provider",
                              "dst": "provider_id", "dst_type": "provider", "fold_subject": "ref_code"}},
}


def _write_mapping(run_dir: Path, mapping: dict) -> None:
    (run_dir / "graph_mapping.yaml").write_text(yaml.safe_dump(mapping, sort_keys=False))


def _passed(run_dir: Path, mapping: dict = FOLDED_MAPPING) -> None:
    """A run folder where the privacy step really passed: flags resolved, mapping clean."""
    _visits_and_providers(run_dir)
    _write_mapping(run_dir, mapping)
    with pytest.raises(OpenFlagError):
        privacy({"run_folder": str(run_dir)})
    resolve_flag(str(run_dir), "visits", "ref_code", "fold", chooser="op")
    resolve_flag(str(run_dir), "members", "patient_name", "drop", chooser="op")
    privacy({"run_folder": str(run_dir)})


def _build(run_dir: Path, monkeypatch) -> MagicMock:
    """Call build directly with a mocked warehouse; returns the mock to assert no SQL ran."""
    w = MagicMock()
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: w)
    # Records each statement on the mock and answers no rows (a MagicMock result would read as
    # a duplicate-key count to build's key checks, lgk.keys).
    monkeypatch.setattr("lgk.steps.build.execute_sql", lambda *a, **k: (w.sql(*a), [])[1])
    # The kit-schema claim is tested in test_teardown.py; here only the gate matters.
    monkeypatch.setattr("lgk.steps.build._claim_kit_schema", lambda *a, **k: None)
    from tests.label_support import stub_table_identities

    stub_table_identities(monkeypatch)  # the mocked warehouse has no Delta history
    from lgk.steps.build import build

    build({"graph_name": "g", "run_folder": str(run_dir), "lakebase_catalog": "c", "kit_schema": "k", "warehouse_id": "wh"})
    return w


# ============================================================== B1: one gate, before any SQL
class TestBuildCallsTheGate:
    def test_passed_gate_lets_build_run_sql(self, tmp_path, monkeypatch):
        _passed(tmp_path)
        w = _build(tmp_path, monkeypatch)
        assert w.sql.called  # non-vacuous: the clear path really reaches SQL

    def test_build_with_no_flags_json_raises_before_sql(self, tmp_path, monkeypatch):
        # hostile: a valid mapping and schema.json, but privacy never ran
        _visits_and_providers(tmp_path)
        (tmp_path / "flags.json").unlink()
        _write_mapping(tmp_path, PROVIDER_MAPPING)
        w = MagicMock()
        monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: w)
        monkeypatch.setattr("lgk.steps.build.execute_sql", lambda *a, **k: w.sql(*a) or [])
        from lgk.steps.build import build

        with pytest.raises(PrivacyGateError, match="flags.json not found"):
            build({"run_folder": str(tmp_path), "lakebase_catalog": "c", "kit_schema": "k", "warehouse_id": "wh"})
        w.sql.assert_not_called()

    def test_flags_json_from_propose_only_has_no_stamp(self, tmp_path, monkeypatch):
        # hostile: flags.json exists (propose wrote the shape) but privacy never passed
        _visits_and_providers(tmp_path)
        _write_mapping(tmp_path, PROVIDER_MAPPING)
        with pytest.raises(PrivacyGateError, match="no privacy stamp"):
            _build(tmp_path, monkeypatch)

    def test_failed_privacy_run_leaves_no_stamp(self, tmp_path):
        _visits_and_providers(tmp_path)
        _write_mapping(tmp_path, PROVIDER_MAPPING)
        with pytest.raises(OpenFlagError):
            privacy({"run_folder": str(tmp_path)})
        assert gate.STAMP_KEY not in read_flags_file(str(tmp_path))
        with pytest.raises(PrivacyGateError, match="no privacy stamp"):
            gate.require_clear(str(tmp_path))

    def test_mapping_edit_after_privacy_makes_build_raise(self, tmp_path, monkeypatch):
        _passed(tmp_path)
        # hostile: re-add the dropped care-subject column after the gate passed
        edited = json.loads(json.dumps(PROVIDER_MAPPING))
        edited["nodes"]["member"] = {"source_table": "members", "natural_key": "row_key",
                                     "label": "row_key", "properties": ["patient_name"]}
        _write_mapping(tmp_path, edited)
        with pytest.raises(PrivacyGateError, match="graph_mapping.yaml changed"):
            _build(tmp_path, monkeypatch)

    def test_new_care_column_in_schema_after_privacy_makes_build_raise(self, tmp_path, monkeypatch):
        _passed(tmp_path)
        # hostile: the source grew a care-subject column after the gate passed
        schema = json.loads((tmp_path / "schema.json").read_text())
        schema["tables"][0]["columns"].append(_col("PatientID"))
        (tmp_path / "schema.json").write_text(json.dumps(schema))
        with pytest.raises(PrivacyGateError, match="schema.json changed"):
            _build(tmp_path, monkeypatch)

    def test_flag_edited_after_privacy_makes_build_raise(self, tmp_path, monkeypatch):
        _passed(tmp_path)
        # hostile: a flag choice changed by hand (drop -> reclassify) without re-running privacy
        data = read_flags_file(str(tmp_path))
        for f in data["flags"]:
            if f["column"] == "patient_name":
                f.update(choice="reclassify", new_class="none", reason="x")
        write_flags_file(str(tmp_path), data)
        with pytest.raises(PrivacyGateError, match="flags.json changed"):
            _build(tmp_path, monkeypatch)

    def test_provision_calls_the_gate_before_any_sql(self, tmp_path, monkeypatch):
        from lgk.steps import provision as prov

        connected = []
        monkeypatch.setattr(prov, "connect_branch", lambda c: connected.append(1))
        _visits_and_providers(tmp_path)  # hostile: flags.json without a stamp
        _write_mapping(tmp_path, PROVIDER_MAPPING)
        with pytest.raises(PrivacyGateError):
            prov.provision({"run_folder": str(tmp_path), "kit_schema": "k", "version": 1})
        assert connected == []


class TestGateChecksEachRule:
    """Defence in depth: a FRESH stamp around a bad state (a forged flags.json) still fails."""

    def _forge(self, run_dir: Path, mutate) -> None:
        data = read_flags_file(str(run_dir))
        mutate(data)
        forge(str(run_dir), data)

    def test_forged_open_flag_is_refused(self, tmp_path):
        _passed(tmp_path)
        self._forge(tmp_path, lambda d: d["flags"][0].update(choice=None))
        with pytest.raises(OpenFlagError, match="open care-subject flag"):
            gate.require_clear(str(tmp_path))

    def test_forged_deleted_flag_is_refused(self, tmp_path):
        _passed(tmp_path)
        self._forge(tmp_path, lambda d: d["flags"].pop())
        with pytest.raises(OpenFlagError, match="with no flag"):
            gate.require_clear(str(tmp_path))

    def test_forged_dropped_column_in_mapping_is_refused(self, tmp_path):
        _passed(tmp_path)
        leaky = json.loads(json.dumps(PROVIDER_MAPPING))
        # hostile: the dropped column used as a node LABEL (the label leak path)
        leaky["nodes"]["member"] = {"source_table": "members", "natural_key": "row_key",
                                    "label": "patient_name"}
        _write_mapping(tmp_path, leaky)
        self._forge(tmp_path, lambda d: None)
        # members holds a care-subject column, so the node is refused for its source table
        # before the label leak check (CareKeyNodeError is a MappingLeakError)
        with pytest.raises(gate.CareKeyNodeError,
                           match=r"node type member is sourced from members, which holds "
                                 r"care-subject column\(s\) members\.patient_name"):
            gate.require_clear(str(tmp_path))

    @pytest.mark.parametrize("column,where", [("name", "label"), ("npi", "properties")])
    def test_forged_drop_on_a_node_column_reaches_the_leak_check(self, tmp_path, column, where):
        """A node on a care table is now refused for its source table first, so
        the label/property leak check is proved here on a non-care table: a forged drop flag
        on providers.<column> (class none), re-stamped, is still a leak."""
        _passed(tmp_path)
        self._forge(tmp_path, lambda d: d["flags"].append(
            {"table": "providers", "column": column, "detected_by": "name", "fold_allowed": False,
             "choice": "drop", "chooser": "op", "reason": None, "timestamp": "t"}))
        with pytest.raises(MappingLeakError,
                           match=rf"dropped or folded.*providers\.{column} \(node provider {where}\)"):
            gate.require_clear(str(tmp_path))

    def test_forged_unclassified_mapping_column_is_refused(self, tmp_path):
        _passed(tmp_path)
        self._forge(tmp_path, lambda d: d.update(columns=[
            c for c in d["columns"] if (c["table"], c["column"]) != ("providers", "npi")]))
        with pytest.raises(MappingLeakError, match=r"providers.npi \(node provider properties\)"):
            gate.require_clear(str(tmp_path))


class TestRunnerRerunsPrivacy:
    def test_succeeded_privacy_is_rerun_and_catches_a_later_mapping_edit(self, tmp_path):
        _passed(tmp_path)
        runner = LaptopRunner({"graph_name": "g", "run_folder": str(tmp_path)})
        built = []
        steps = [("privacy", privacy), ("build", lambda c: built.append(1))]
        assert [r.status for r in runner.run_steps(steps)] == [RunStatus.SUCCESS] * 2
        leaky = json.loads(json.dumps(PROVIDER_MAPPING))
        leaky["nodes"]["provider"]["properties"] = ["npi", "home_zip"]
        _write_mapping(tmp_path, leaky)
        results = runner.run_steps(steps)
        assert [r.step for r in results] == ["privacy"]  # build was not reached
        assert results[0].status == RunStatus.FAILED
        assert "providers.home_zip" in results[0].error
        assert built == [1]


# ========================================================== B2: unclassified columns refused
class TestUnclassifiedRefused:
    @pytest.mark.parametrize("where, edit", [
        ("node provider label", lambda m: m["nodes"]["provider"].update(label="display_nm")),
        ("node provider properties", lambda m: m["nodes"]["provider"].update(properties=["x9"])),
        ("node provider natural_key", lambda m: m["nodes"]["provider"].update(natural_key="pk2")),
        ("edge e src", lambda m: m["edges"].update(e={
            "source_table": "visits", "src": "visitor", "src_type": "provider",
            "dst": "provider_id", "dst_type": "provider"})),
        ("edge e dst", lambda m: m["edges"].update(e={
            "source_table": "visits", "src": "provider_id", "src_type": "provider",
            "dst": "other_end", "dst_type": "provider"})),
    ])
    def test_mapping_column_missing_from_schema_json_is_refused(self, tmp_path, where, edit):
        # hostile: a harmless-looking name that schema.json never classified
        mapping = json.loads(json.dumps(PROVIDER_MAPPING))
        edit(mapping)
        with pytest.raises(MappingLeakError, match=f"no privacy classification.*{where}"):
            _passed(tmp_path, mapping)

    def test_table_missing_from_schema_json_is_refused(self, tmp_path):
        mapping = json.loads(json.dumps(PROVIDER_MAPPING))
        mapping["nodes"]["provider"]["source_table"] = "providers_copy"
        with pytest.raises(MappingLeakError, match="providers_copy.provider_id"):
            _passed(tmp_path, mapping)


# ===================================================================== B3: name classification
# Every care word set has its own per-member hostile name test: tests/test_privacy_identifiers.py
class TestNameClassification:
    @pytest.mark.parametrize("name, tokens", [
        ("PatientID", ["patient", "id"]), ("mbrId", ["mbr", "id"]), ("zip5", ["zip", "5"]),
        ("patientMRN", ["patient", "mrn"]), ("home_org_id", ["home", "org", "id"]),
    ])
    def test_tokeniser_splits_camel_case_and_digits(self, name, tokens):
        assert classify.name_tokens(name) == tokens

    @pytest.mark.parametrize("name", [
        "outpatientless_flag", "hospital_id", "specialty", "org_name", "start_date", "code",
        "patho_code", "pathway_id",
    ])
    def test_unrelated_names_stay_none(self, name):
        # hostile: `pat`/`patient` inside longer unrelated words is not a match
        assert classify.classify_column(name, {})[0] == "none"

    def test_mostly_null_mrn_column_is_not_diluted(self):
        # hostile: 90% NULL and blank cells; the non-NULL values are all MRNs
        values = [None] * 80 + [""] * 10 + MRN_VALUES[:10]
        verdict = classify.pattern_verdicts(values)
        assert verdict["mrn"] == 1.0
        assert classify.classify_column("code", verdict) == ("care_subject", "values")


# ============================================================ fold rules for a care-subject PK
class TestSingleKeyNotFoldable:
    def test_single_care_subject_pk_is_not_foldable_even_when_also_an_fk(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        # hostile: the patients table's only key is also declared as an FK column
        (tmp_path / "schema.json").write_text(json.dumps({"tables": [
            {"name": "patients", "key_columns": ["mrn"],
             "foreign_keys": [{"columns": ["mrn"], "references": ["people", "mrn"]}],
             "columns": [_col("mrn"), _col("specialty")]},
        ]}))
        with pytest.raises(OpenFlagError):
            privacy({"run_folder": str(tmp_path)})
        from lgk.privacy.flags import FlagChoiceError

        with pytest.raises(FlagChoiceError, match="only be dropped or reclassified"):
            resolve_flag(str(tmp_path), "patients", "mrn", "fold", chooser="op")


# ============================================================ floor from env / flags.json
class TestFloorSources:
    def test_walk_without_run_folder_or_folded_is_refused(self):
        from lgk.steps.walks import get_walk_query

        with pytest.raises(PrivacyGateError, match="folded"):
            get_walk_query("x", depth=1, served_schema="s")

    def test_walk_reads_folded_and_floor_from_flags_json(self, tmp_path):
        from lgk.steps.walks import get_walk_query

        _visits_and_providers(tmp_path)
        _write_mapping(tmp_path, FOLDED_MAPPING)  # the fold flag is folded by an edge
        with pytest.raises(OpenFlagError):
            privacy({"run_folder": str(tmp_path), "privacy_floor": 15})
        resolve_flag(str(tmp_path), "visits", "ref_code", "fold", chooser="op")
        resolve_flag(str(tmp_path), "members", "patient_name", "drop", chooser="op")
        privacy({"run_folder": str(tmp_path), "privacy_floor": 15})
        sql = get_walk_query("x", depth=1, served_schema="s", run_folder=str(tmp_path),
                             edge_types=["shares_refs"])
        assert "MIN(e.weight) >= 15" in sql  # folded (a fold flag) and the recorded floor
        assert "MIN(e.weight) >= 20" in get_walk_query(
            "x", depth=1, served_schema="s", run_folder=str(tmp_path), privacy_floor=20,
            edge_types=["shares_refs"])
        with pytest.raises(ValueError, match="below the recorded floor"):
            get_walk_query("x", depth=1, served_schema="s", run_folder=str(tmp_path),
                           privacy_floor=12, edge_types=["shares_refs"])
        with pytest.raises(PrivacyGateError, match="contradicts"):
            get_walk_query("x", depth=1, served_schema="s", run_folder=str(tmp_path),
                           folded=False, edge_types=["shares_refs"])

    def test_walk_with_no_fold_flag_is_unfolded(self, tmp_path):
        from lgk.steps.walks import get_walk_query

        _visits_and_providers(tmp_path)
        _write_mapping(tmp_path, PROVIDER_MAPPING)
        with pytest.raises(OpenFlagError):
            privacy({"run_folder": str(tmp_path)})
        resolve_flag(str(tmp_path), "visits", "ref_code", "drop", chooser="op")
        resolve_flag(str(tmp_path), "members", "patient_name", "drop", chooser="op")
        privacy({"run_folder": str(tmp_path)})
        sql = get_walk_query("x", depth=1, served_schema="s", run_folder=str(tmp_path))
        assert "COUNT(*) AS support" in sql and "HAVING" not in sql  # walk engine column

    def test_walk_with_a_stale_gate_is_refused(self, tmp_path):
        from lgk.steps.walks import get_walk_query

        _passed(tmp_path)
        (tmp_path / "schema.json").write_text(json.dumps({"tables": []}))
        with pytest.raises(PrivacyGateError, match="stale"):
            get_walk_query("x", depth=1, served_schema="s", run_folder=str(tmp_path))


# ============================================================ UC profile: pattern rates only
class TestUcProfilePatterns:
    def test_uc_verdict_writes_rates_for_string_columns_only(self):
        from lgk.steps.profile import _uc_table_verdict

        sample = [(m if i % 3 == 0 else None, f"Dr {i}", i) for i, m in enumerate(MRN_VALUES)]
        statements = []

        def sql(statement):
            statements.append(statement)
            if "LIMIT" in statement:
                # hostile: the int column must not be sampled; a mostly-NULL MRN column
                assert "`visits`" not in statement
                return [[a, b] for a, b, _ in sample]
            return [[50, 17, 17, 50, 50, 50, 50]]

        verdict = _uc_table_verdict(sql, "c.s.t", "t", [], [
            ("ref", "STRING"), ("label", "STRING"), ("visits", "INT")])
        out = verdict.to_dict()
        cols = {c["name"]: c for c in out["columns"]}
        assert cols["ref"]["patterns"]["mrn"] == 1.0
        assert cols["label"]["patterns"]["mrn"] == 0.0
        assert "patterns" not in cols["visits"]
        text = json.dumps(out)
        assert not any(v in text for v in MRN_VALUES) and "Dr 1" not in text
        assert classify.classify_column("ref", cols["ref"]["patterns"])[0] == "care_subject"


# ================================================== node joins through the gate
class TestNodeJoinsGate:
    """§3.5 rules 1-3 and §5 (`derived`): a join is classified like a native column."""

    @staticmethod
    def _run(tmp_path, mapping=None, schema=None, folds=(("claims", "patient_id"),), drops=(),
             reclassify=()):
        from tests.test_build_sql import JOIN_MAPPING, pass_privacy, write_run

        write_run(tmp_path, mapping or JOIN_MAPPING, schema)
        pass_privacy(tmp_path, folds=folds, drops=drops, reclassify=reclassify)

    def test_the_item_mapping_passes_and_records_derived(self, tmp_path):
        self._run(tmp_path)
        derived = read_flags_file(str(tmp_path))["derived"]
        assert derived == [
            {"node_type": "provider", "name": "total_billed", "table": "claims", "column": "billed_amount",
             "func": "sum", "class": "none", "floored": True},
            {"node_type": "provider", "name": "claim_count", "table": "claims", "column": "claim_id",
             "func": "count", "class": "none", "floored": True},
            {"node_type": "provider", "name": "taxonomy_code", "table": "provider_taxonomy",
             "column": "taxonomy_code", "func": None, "class": "none", "floored": False},
        ]

    def test_care_subject_table_aggregate_without_subject_is_refused(self, tmp_path):
        from tests.test_build_sql import mapping_with

        m = mapping_with()
        del m["nodes"]["provider"]["joins"][0]["subject"]
        # hostile: patient_id is not carried by the mapping at all; only the table holds it.
        # The fold flag is kept used by an edge, so only check_join_subjects can refuse.
        m["edges"] = {"shared": {"source_table": "claims", "src": "provider_id", "src_type": "provider",
                                 "dst": "provider_id", "dst_type": "provider", "fold_subject": "patient_id"}}
        with pytest.raises(MappingLeakError, match=r"nodes\.provider\.joins\[0\] aggregates claims, which "
                                                   r"holds care-subject data \(patient_id\), without a folded subject"):
            self._run(tmp_path, m)

    def test_a_reclassified_care_column_does_not_need_a_subject(self, tmp_path):
        """After reclassification the table holds no care-subject column (rule 3 reads the flag)."""
        from tests.test_build_sql import mapping_with

        m = mapping_with()
        del m["nodes"]["provider"]["joins"][0]["subject"]
        self._run(tmp_path, m, folds=(), reclassify=(("claims", "patient_id"),))
        assert gate.require_clear(str(tmp_path))  # passes

    def test_a_subject_whose_flag_chose_drop_is_refused(self, tmp_path):
        with pytest.raises(MappingLeakError, match=r"fold_subject claims\.patient_id \(node provider joins\[0\] subject\)"):
            self._run(tmp_path, folds=(), drops=(("claims", "patient_id"),))

    def test_a_join_subject_counts_as_a_use_of_the_fold_flag(self, tmp_path):
        """Rule 2 reverse: the only fold of claims.patient_id is the join subject; it passes."""
        self._run(tmp_path)
        data = read_flags_file(str(tmp_path))
        assert [f["choice"] for f in data["flags"]] == ["fold"]
        assert gate.require_clear(str(tmp_path))

    def test_aggregated_care_subject_column_is_refused(self, tmp_path):
        from tests.test_build_sql import mapping_with

        # hostile: count_distinct of the folded subject itself (the patient count, unfloored)
        m = mapping_with(aggregates=["count_distinct(patient_id) as patients"])
        with pytest.raises(MappingLeakError, match=r"claims\.patient_id \(node provider join aggregate\)"):
            self._run(tmp_path, m)

    def test_a_dropped_join_key_is_refused(self, tmp_path):
        from tests.test_build_sql import mapping_with

        # hostile: a join THROUGH a care-subject key (claims keyed by patient_id)
        m = mapping_with(key="patient_id", aggregates=["count(claim_id) as claim_count"])
        del m["nodes"]["provider"]["joins"][0]["subject"]
        with pytest.raises(MappingLeakError, match=r"claims\.patient_id \(node provider join key\)"):
            self._run(tmp_path, m, folds=(), drops=(("claims", "patient_id"),))

    def test_an_unclassified_join_column_is_refused(self, tmp_path):
        from tests.test_build_sql import JOIN_MAPPING

        m = json.loads(json.dumps(JOIN_MAPPING))
        m["nodes"]["provider"]["joins"][1]["columns"] = ["taxonomy_code", "not_profiled"]
        with pytest.raises(MappingLeakError, match=r"provider_taxonomy\.not_profiled \(node provider join column\)"):
            self._run(tmp_path, m)

    def test_derived_edited_by_hand_is_refused(self, tmp_path, monkeypatch):
        self._run(tmp_path)
        data = read_flags_file(str(tmp_path))
        # hostile: un-floor an output in the record and recompute the stamp, as a hand edit would
        data["derived"][0]["floored"] = False
        forge(str(tmp_path), data)
        with pytest.raises(PrivacyGateError, match="derived join outputs differ"):
            gate.require_clear(str(tmp_path))

    def test_a_join_edit_after_privacy_makes_build_raise(self, tmp_path, monkeypatch):
        """The stamp mechanism (test_mapping_edit_after_privacy_makes_build_raise) covers joins."""
        from tests.test_build_sql import JOIN_MAPPING, run_build

        self._run(tmp_path)
        edited = json.loads(json.dumps(JOIN_MAPPING))
        del edited["nodes"]["provider"]["joins"][0]["subject"]  # hostile: drop the floor
        _write_mapping(tmp_path, edited)
        with pytest.raises(PrivacyGateError, match="graph_mapping.yaml changed"):
            run_build(tmp_path, monkeypatch)

    def test_a_subjectless_care_aggregate_in_a_later_join_is_refused(self, tmp_path):
        """check_join_subjects reads every join: joins[0] is floored, joins[1] is not."""
        from tests.test_build_sql import JOIN_MAPPING

        m = json.loads(json.dumps(JOIN_MAPPING))
        m["nodes"]["provider"]["joins"][1] = {"table": "claims", "key": "provider_id",
                                              "aggregates": ["max(billed_amount) as top_billed"]}
        with pytest.raises(MappingLeakError, match=r"nodes\.provider\.joins\[1\] aggregates claims"):
            self._run(tmp_path, m)

    def test_derived_class_is_the_class_after_reclassification(self, tmp_path):
        """An output whose input column was care subject and reclassified records the new class."""
        from tests.test_build_sql import JOIN_MAPPING, join_schema, write_run, _col

        schema = join_schema()
        schema["tables"][1]["columns"].append(_col("visit_id"))  # care subject by name
        m = json.loads(json.dumps(JOIN_MAPPING))
        m["nodes"]["provider"]["joins"][0]["aggregates"].append("count_distinct(visit_id) as visits")
        write_run(tmp_path, m, schema)
        with pytest.raises(OpenFlagError):
            privacy({"run_folder": str(tmp_path)})
        resolve_flag(str(tmp_path), "claims", "patient_id", "fold", chooser="op")
        resolve_flag(str(tmp_path), "claims", "visit_id", "reclassify", chooser="op",
                     reason="a scheduling slot id, not a person", new_class="public_professional")
        privacy({"run_folder": str(tmp_path)})
        (entry,) = [d for d in read_flags_file(str(tmp_path))["derived"] if d["name"] == "visits"]
        assert entry["class"] == "public_professional"
