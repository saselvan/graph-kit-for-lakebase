"""Privacy binding (offline), plus the gate's re-classify and run_folder checks.

1. Served tables are bound to the privacy decision: build stamps each Delta table
   (TBLPROPERTIES lgk.privacy_stamp) and the run folder; provision refuses unless BOTH match
   the current stamp; switch refuses a version provisioned under another stamp; a privacy
   run that changes the stamp makes the runner rebuild.
2. The mapping's source is pinned to the source schema.json profiled.

The warehouse is a scripted fake that keeps TBLPROPERTIES per table (no SQL engine).
The live half (a real Delta table with a stale / missing stamp) is
tests/test_privacy_binding_live.py.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

from lgk.privacy import binding, gate
from lgk.privacy.binding import STAMP_PROPERTY, StaleBuildError
from lgk.privacy.flags import OpenFlagError, read_flags_file, resolve_flag, write_flags_file
from lgk.privacy.gate import MappingLeakError, PrivacyGateError
from lgk.runner import LaptopRunner, RunStatus
from lgk.steps.privacy import privacy
from tests.label_support import kit_label, stub_table_identities
from tests.test_privacy import _col

# The real schema-claim steps, captured before the fixtures stub them (the merge test runs them).
from lgk.steps.build import _claim_kit_schema as _REAL_CLAIM_KIT_SCHEMA
from lgk.steps.provision import _mark_pg_version_schema as _REAL_MARK_PG_VERSION_SCHEMA
from tests.forged_decision import forge

KIT = ["cat.kit.nodes", "cat.kit.edges", "cat.kit.node_id_map"]


class FakeWarehouse:
    """Keeps TBLPROPERTIES per table; DROP TABLE and CREATE OR REPLACE forget them. Every statement is logged."""

    def __init__(self):
        self.props = {}
        self.statements = []

    def __call__(self, statement):
        self.statements.append(statement)
        s = " ".join(statement.split())
        m = re.match(r"ALTER TABLE (\S+) SET TBLPROPERTIES \('([^']+)' = '([^']*)'\)", s)
        if m:
            self.props.setdefault(m[1], {})[m[2]] = m[3]
            return []
        m = re.match(r"(?:DROP TABLE IF EXISTS|CREATE OR REPLACE TABLE) (\S+)", s)
        if m:  # both forget the table's properties (CREATE OR REPLACE: live probe)
            self.props.pop(m[1], None)
            return []
        m = re.match(r"SHOW TBLPROPERTIES (\S+)", s)
        if m:
            return [[k, v] for k, v in self.props.get(m[1], {}).items()] + [["delta.minReaderVersion", "1"]]
        return []

    def drops(self, table):
        """How many times build replaced `table` (CREATE OR REPLACE, no longer DROP)."""
        return sum(1 for s in self.statements if s.strip().startswith(f"CREATE OR REPLACE TABLE {table} "))


@pytest.fixture
def wh(monkeypatch):
    fake = FakeWarehouse()
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: object())
    monkeypatch.setattr("lgk.steps.build.execute_sql", lambda w, wid, s: fake(s))
    monkeypatch.setattr("lgk.steps.provision.execute_sql", lambda w, wid, s: fake(s))
    # The kit-schema claim is tested in test_teardown.py; here only the binding matters.
    monkeypatch.setattr("lgk.steps.build._claim_kit_schema", lambda *a, **k: None)
    stub_table_identities(monkeypatch)  # the fake keeps properties only, no Delta history
    return fake


@pytest.fixture
def served(monkeypatch):
    """Provision's Postgres / sync parts, recorded instead of run (the gate parts run as shipped)."""
    from lgk.steps import provision as prov

    calls = []
    for name in ("_refuse_existing_version", "_create_synced_tables", "_lock_version_from_reader",
                 "_setup_reader_role", "_serve_attribution", "_prove_reader_isolated",
                 "_mark_pg_version_schema", "_create_version_indexes", "_stage_version_functions"):
        monkeypatch.setattr(prov, name, lambda c, _n=name: calls.append(_n))
    monkeypatch.setattr(prov, "_ensure_function_roles", lambda config, cursor: None)
    monkeypatch.setattr(prov, "require_sync_owner", lambda r: None)  # the sync-owner check is tested in test_preflight.py
    # switch connects through provision's helper
    monkeypatch.setattr(prov, "connect_branch", lambda c: _Conn())
    # The served-schema claim is tested in test_teardown.py; switch's binding check is what matters here.
    monkeypatch.setattr("lgk.manifest.claim_pg_schema", lambda *a, **k: False)
    monkeypatch.setattr("lgk.versions.switch_views",
                        lambda conn, s, target, *a, **k: calls.append(("switch_views", target)))
    return calls


class _Conn:
    autocommit = False

    def cursor(self):
        return None

    def execute(self, sql, params=None):
        # serve reads the served schema's kit functions before the switch (F0); there are none
        return type("R", (), {"fetchall": lambda self: []})()

    def close(self):
        pass


def _config(run_dir: Path, **kw) -> dict:
    return {"graph_name": "g", "run_folder": str(run_dir), "lakebase_catalog": "cat", "warehouse_id": "wh",
            "lakebase_project": "p", "branch_id": "t-x", "kit_schema": "kit", "version": 1, **kw}


MEMBER_MAPPING = {
    "source_catalog": "gold", "source_schema": "hls",
    # a node is never sourced from a table holding a care-subject column,
    # so the care column patient_name lives in member_notes, which the mapping does not read;
    # the loose decision reclassifies it, the tight one drops it (the stamp changes either way)
    "nodes": {"member": {"source_table": "members", "natural_key": "row_key",
                         "label": "display_name"}},
    "edges": {},
}


def _write_mapping(run_dir: Path, mapping: dict) -> None:
    (run_dir / "graph_mapping.yaml").write_text(yaml.safe_dump(mapping, sort_keys=False))


def _member_schema(run_dir: Path, catalog="gold", schema="hls") -> None:
    (run_dir / "schema.json").write_text(json.dumps({
        "source_catalog": catalog, "source_schema": schema,
        "tables": [{"name": "members", "key_columns": ["row_key"], "foreign_keys": [],
                    "columns": [_col("row_key"), _col("display_name")]},
                   {"name": "member_notes", "key_columns": ["note_key"], "foreign_keys": [],
                    "columns": [_col("note_key"), _col("patient_name")]}]}))


def _loose_decision(run_dir: Path) -> None:
    """patient_name reclassified to none (the looser decision), and privacy passes."""
    _member_schema(run_dir)
    _write_mapping(run_dir, MEMBER_MAPPING)
    with pytest.raises(OpenFlagError):
        privacy({"run_folder": str(run_dir)})
    resolve_flag(str(run_dir), "member_notes", "patient_name", "reclassify", chooser="op",
                 reason="display name of a care team, not a patient", new_class="none")
    privacy({"run_folder": str(run_dir)})


def _tighten(run_dir: Path) -> None:
    """The operator tightens: patient_name dropped, the label moved off it, privacy passes."""
    resolve_flag(str(run_dir), "member_notes", "patient_name", "drop", chooser="op")
    tight = json.loads(json.dumps(MEMBER_MAPPING))
    tight["nodes"]["member"]["label"] = "row_key"
    _write_mapping(run_dir, tight)
    privacy({"run_folder": str(run_dir)})


def _current(run_dir: Path) -> str:
    return binding.stamp_id(gate.require_clear(str(run_dir)))


def _steps():
    from lgk.steps.build import build
    from lgk.steps.provision import provision

    return [("privacy", privacy), ("build", build), ("provision", provision)]


# =========================================== criterion 1: the "tighten after build" repro
class TestTightenAfterBuild:
    def test_build_stamps_every_table_and_the_run_folder(self, tmp_path, wh):
        from lgk.steps.build import build

        _loose_decision(tmp_path)
        build(_config(tmp_path))
        stamp = _current(tmp_path)
        # the label is the JSON {stamp, table_id, certified_version} (owner decision: nightly stamp step)
        assert {t: json.loads(wh.props[t][STAMP_PROPERTY]) for t in KIT} == {
            t: {"stamp": stamp, "table_id": f"tid:{t}", "certified_version": 1} for t in KIT}
        record = json.loads((tmp_path / "privacy_bindings.json").read_text())["build"]
        assert record == {"stamp": stamp, "tables": sorted(KIT)}
        # the stamp is set only after the rows are written
        last_insert = max(i for i, s in enumerate(wh.statements) if "INSERT INTO" in s)
        first_stamp = min(i for i, s in enumerate(wh.statements) if STAMP_PROPERTY in s)
        assert first_stamp > last_insert

    def test_provision_refuses_the_table_built_under_the_looser_decision(self, tmp_path, wh, served):
        from lgk.steps.build import build
        from lgk.steps.provision import provision

        _loose_decision(tmp_path)
        build(_config(tmp_path))
        _tighten(tmp_path)  # the gate is clear again, but the tables are the old ones
        with pytest.raises(StaleBuildError, match="different privacy decision"):
            provision(_config(tmp_path))
        assert served == ["_refuse_existing_version"]  # refused before any sync or view

    def test_runner_rebuilds_after_tightening_and_serves_the_new_table(self, tmp_path, wh, served):
        _loose_decision(tmp_path)
        runner = LaptopRunner(_config(tmp_path))
        assert [r.status for r in runner.run_steps(_steps())] == [RunStatus.SUCCESS] * 3
        old = _current(tmp_path)
        _tighten(tmp_path)
        new = _current(tmp_path)
        assert new != old
        results = runner.run_steps(_steps())
        assert [(r.step, r.status) for r in results] == [
            ("privacy", RunStatus.SUCCESS), ("build", RunStatus.SUCCESS),
            ("provision", RunStatus.SUCCESS)], [r.to_dict() for r in results]
        assert wh.drops("cat.kit.nodes") == 2  # build really ran again, not skipped
        assert json.loads(wh.props["cat.kit.nodes"][STAMP_PROPERTY])["stamp"] == new
        versions = json.loads((tmp_path / "privacy_bindings.json").read_text())["versions"]
        assert versions == {"kit_v1": new}

    def test_unchanged_privacy_rerun_does_not_rebuild(self, tmp_path, wh, served):
        _loose_decision(tmp_path)
        runner = LaptopRunner(_config(tmp_path))
        runner.run_steps(_steps())
        results = runner.run_steps(_steps()[:2])
        assert [r.status for r in results] == [RunStatus.SUCCESS] * 2
        assert wh.drops("cat.kit.nodes") == 1  # same decision: build skipped as before

    def test_invalidation_marks_build_enrich_provision_not_run(self, tmp_path, wh, served):
        _loose_decision(tmp_path)
        runner = LaptopRunner(_config(tmp_path))
        runner.run_steps(_steps() + [("enrich", lambda c: None)])
        _tighten(tmp_path)  # direct call: no runner, run.json untouched
        runner.run_steps(_steps()[:1])
        state = json.loads((tmp_path / "run.json").read_text())
        assert {s: state[s]["status"] for s in ("privacy", "build", "enrich", "provision")} == {
            "privacy": "success", "build": "not_run", "enrich": "not_run", "provision": "not_run"}


class TestEachBindingCheck:
    def _built_then_tightened(self, tmp_path):
        from lgk.steps.build import build

        _loose_decision(tmp_path)
        build(_config(tmp_path))
        _tighten(tmp_path)
        return _current(tmp_path)

    def test_stale_table_property_is_refused_even_with_a_fresh_run_folder(self, tmp_path, wh, served):
        from lgk.steps.provision import provision

        new = self._built_then_tightened(tmp_path)
        # hostile: the run folder was copied from a fresh run; the Delta tables are the old ones
        binding.record_build(str(tmp_path), new, KIT)
        with pytest.raises(StaleBuildError, match="cat.kit.nodes \\(stale stamp\\)"):
            provision(_config(tmp_path))
        assert served == ["_refuse_existing_version"]

    def test_missing_table_property_is_refused(self, tmp_path, wh, served):
        from lgk.steps.provision import provision

        new = self._built_then_tightened(tmp_path)
        binding.record_build(str(tmp_path), new, KIT)
        for t in KIT:
            wh.props[t][STAMP_PROPERTY] = new
        del wh.props["cat.kit.edges"][STAMP_PROPERTY]  # hostile: one table only
        with pytest.raises(StaleBuildError, match="cat.kit.edges \\(no lgk.privacy_stamp\\)"):
            provision(_config(tmp_path))

    def test_stale_run_folder_is_refused_even_with_fresh_table_properties(self, tmp_path, wh, served):
        from lgk.steps.provision import provision

        new = self._built_then_tightened(tmp_path)
        for t in KIT:
            wh.props[t][STAMP_PROPERTY] = new
        with pytest.raises(StaleBuildError, match="different privacy decision"):
            provision(_config(tmp_path))
        assert served == ["_refuse_existing_version"]

    def test_run_folder_built_before_this_change_is_refused_not_trusted(self, tmp_path, wh, served):
        from lgk.steps.provision import provision

        # hostile: a clear gate and tables that exist, but no binding record and no property
        _loose_decision(tmp_path)
        assert not (tmp_path / "privacy_bindings.json").exists()
        with pytest.raises(StaleBuildError, match="records no build"):
            provision(_config(tmp_path))
        binding.record_build(str(tmp_path), _current(tmp_path), KIT)  # record only
        with pytest.raises(StaleBuildError, match="no lgk.privacy_stamp"):
            provision(_config(tmp_path))
        assert "_create_synced_tables" not in served

    def test_build_record_for_other_tables_is_refused(self, tmp_path, wh, served):
        from lgk.steps.provision import provision

        _loose_decision(tmp_path)
        stamp = _current(tmp_path)
        # hostile: the record is for another kit schema
        binding.record_build(str(tmp_path), stamp, [t.replace("kit", "other") for t in KIT])
        for t in KIT:
            wh.props[t] = {STAMP_PROPERTY: stamp}
        with pytest.raises(StaleBuildError, match="did not make"):
            provision(_config(tmp_path))

    def test_fresh_build_is_provisioned_and_its_version_recorded(self, tmp_path, wh, served):
        from lgk.steps.build import build
        from lgk.steps.provision import provision

        _loose_decision(tmp_path)
        build(_config(tmp_path))
        provision(_config(tmp_path))  # non-vacuous: the clear path really provisions
        assert served[:2] == ["_refuse_existing_version", "_create_synced_tables"]
        versions = json.loads((tmp_path / "privacy_bindings.json").read_text())["versions"]
        assert versions == {"kit_v1": _current(tmp_path)}


    @pytest.mark.parametrize("forged", ["", "prefix"])
    def test_empty_or_truncated_build_stamp_is_refused(self, tmp_path, wh, served, forged):
        from lgk.steps.build import build
        from lgk.steps.provision import provision

        _loose_decision(tmp_path)
        build(_config(tmp_path))
        stamp = _current(tmp_path)
        # hostile: a substring of the real stamp (an `in` comparison would let it through)
        binding.record_build(str(tmp_path), stamp[:12] if forged else "", KIT)
        with pytest.raises(StaleBuildError, match="different privacy decision"):
            provision(_config(tmp_path))

    def test_failed_sync_records_no_version(self, tmp_path, wh, served, monkeypatch):
        from lgk.steps import provision as prov
        from lgk.steps.build import build

        _loose_decision(tmp_path)
        build(_config(tmp_path))

        def sync_fails(config):
            raise RuntimeError("Synced table failed: SYNCED_TABLE_OFFLINE_FAILED")

        monkeypatch.setattr(prov, "_create_synced_tables", sync_fails)
        with pytest.raises(RuntimeError, match="OFFLINE_FAILED"):
            prov.provision(_config(tmp_path))
        # a half-synced version is never recorded as servable
        bindings = json.loads((tmp_path / "privacy_bindings.json").read_text())
        assert "versions" not in bindings

    def test_failed_isolation_probe_records_no_version(self, tmp_path, wh, served, monkeypatch):
        # hostile (Isaac Review): sync succeeds, then the reader-isolation probe fails; the
        # version must NOT be recorded, or a later switch would serve an unproven version
        from lgk.steps import provision as prov
        from lgk.steps.build import build

        _loose_decision(tmp_path)
        build(_config(tmp_path))

        def probe_fails(config):
            raise RuntimeError("reader can SELECT a raw table")

        monkeypatch.setattr(prov, "_create_synced_tables", lambda config: None)
        monkeypatch.setattr(prov, "_prove_reader_isolated", probe_fails)
        with pytest.raises(RuntimeError, match="raw table"):
            prov.provision(_config(tmp_path))
        bindings = json.loads((tmp_path / "privacy_bindings.json").read_text())
        assert "versions" not in bindings
        # provision never moves the served views: the site stays on the old version
        assert not [c for c in served if isinstance(c, tuple) and c[0] == "switch_views"], served

    def test_runner_rebuilds_a_build_from_before_this_change(self, tmp_path, wh, served):
        # hostile: run.json says build succeeded, but there is no binding record (built
        # before binding existed); privacy must make the runner rebuild, not skip and get stuck
        _loose_decision(tmp_path)
        runner = LaptopRunner(_config(tmp_path))
        runner.run_steps(_steps())
        (tmp_path / "privacy_bindings.json").unlink()
        results = runner.run_steps(_steps())
        assert [r.status for r in results] == [RunStatus.SUCCESS] * 3, [r.to_dict() for r in results]
        assert wh.drops("cat.kit.nodes") == 2


class TestSwitchRefusesUnboundVersions:
    def _provisioned_v1(self, tmp_path):
        from lgk.steps.build import build
        from lgk.steps.provision import provision

        _loose_decision(tmp_path)
        build(_config(tmp_path))
        provision(_config(tmp_path))

    def test_switch_to_a_version_provisioned_under_the_current_stamp(self, tmp_path, wh, served):
        from lgk.steps.switch import switch

        self._provisioned_v1(tmp_path)
        assert switch(_config(tmp_path), 1) == "kit_v1"
        assert ("switch_views", "kit_v1") in served

    def test_rollback_to_a_version_from_the_looser_decision_is_refused(self, tmp_path, wh, served):
        from lgk.steps.switch import switch

        self._provisioned_v1(tmp_path)
        _tighten(tmp_path)
        served.clear()
        with pytest.raises(StaleBuildError, match="kit_v1 was provisioned under a different"):
            switch(_config(tmp_path), 1)
        assert served == []

    def test_unrecorded_version_is_refused(self, tmp_path, wh, served):
        from lgk.steps.switch import switch

        self._provisioned_v1(tmp_path)
        served.clear()
        # hostile: kit_v2 exists in Postgres (made by hand), but no run provisioned it
        with pytest.raises(StaleBuildError, match="kit_v2 has no recorded privacy stamp"):
            switch(_config(tmp_path), 2)
        assert served == []

    @pytest.mark.parametrize("forged", ["", "prefix"])
    def test_empty_or_truncated_version_stamp_is_refused(self, tmp_path, wh, served, forged):
        from lgk.steps.switch import switch

        self._provisioned_v1(tmp_path)
        stamp = _current(tmp_path)
        binding.record_version(str(tmp_path), "kit_v1", stamp[:12] if forged else "")
        served.clear()
        with pytest.raises(StaleBuildError, match="different privacy decision"):
            switch(_config(tmp_path), 1)
        assert served == []

    def test_gated_build_provision_switch_still_record_every_object_for_teardown(
            self, tmp_path, wh, served, monkeypatch):
        """Binding and schema claims together: the gated steps still claim and record what they make (teardown reads it)."""
        from lgk.manifest import ObjectType, load_run_manifest
        from lgk.steps import build as b
        from lgk.steps import provision as prov
        from lgk.steps.switch import switch

        monkeypatch.setattr(b, "_claim_kit_schema", _REAL_CLAIM_KIT_SCHEMA)
        monkeypatch.setattr(prov, "_mark_pg_version_schema", _REAL_MARK_PG_VERSION_SCHEMA)
        monkeypatch.setattr("lgk.manifest.claim_uc_schema", lambda *a, **k: True)  # created by this run
        monkeypatch.setattr("lgk.manifest.claim_pg_schema", lambda *a, **k: True)
        monkeypatch.setattr(prov, "connect_branch", lambda c: _Conn())
        self._provisioned_v1(tmp_path)
        switch(_config(tmp_path), 1)

        got = {(e.object_type, e.object_id) for e in load_run_manifest(_config(tmp_path)).entries}
        assert {(ObjectType.UC_SCHEMA, "cat.kit")} | {(ObjectType.UC_TABLE, t) for t in KIT} <= got, got
        assert {(ObjectType.PG_SCHEMA, "kit_v1"), (ObjectType.PG_SCHEMA, "kit")} <= got, got
        assert any(t == ObjectType.VIEW and i.startswith("kit.") for t, i in got), got

    def test_switch_with_a_stale_gate_is_refused(self, tmp_path, wh, served):
        from lgk.steps.switch import switch

        self._provisioned_v1(tmp_path)
        _write_mapping(tmp_path, {**MEMBER_MAPPING, "edges": {}, "source_schema": "hls2"})
        served.clear()
        with pytest.raises(PrivacyGateError, match="stale"):
            switch(_config(tmp_path), 1)
        assert served == []


class TestRunFolderRequired:
    def test_provision_without_run_folder_is_refused(self, tmp_path, monkeypatch, wh, served):
        from lgk.steps.provision import provision

        _loose_decision(tmp_path)
        monkeypatch.chdir(tmp_path)  # hostile: a clear gate in '.' must not be used
        cfg = _config(tmp_path)
        del cfg["run_folder"]
        with pytest.raises(ValueError, match="run_folder"):
            provision(cfg)
        assert served == []

    def test_build_without_run_folder_is_refused(self, tmp_path, monkeypatch, wh):
        from lgk.steps.build import build

        _loose_decision(tmp_path)
        monkeypatch.chdir(tmp_path)
        cfg = _config(tmp_path)
        del cfg["run_folder"]
        with pytest.raises(ValueError, match="run_folder"):
            build(cfg)
        assert wh.statements == []

    def test_switch_without_run_folder_is_refused(self, tmp_path, monkeypatch, wh, served):
        from lgk.steps.switch import switch

        _loose_decision(tmp_path)
        monkeypatch.chdir(tmp_path)
        cfg = _config(tmp_path)
        del cfg["run_folder"]
        with pytest.raises(ValueError, match="run_folder"):
            switch(cfg, 1)
        assert served == []


# ====================================================== criterion 2: the source is pinned
class TestSourcePinned:
    @pytest.mark.parametrize("catalog, schema", [
        ("raw_phi", "ehr"),  # the issue's repro
        ("raw_phi", "hls"),  # hostile: only the catalog differs
        ("gold", "ehr"),  # hostile: only the schema differs
    ])
    def test_mapping_on_another_source_is_refused_by_privacy(self, tmp_path, catalog, schema):
        _loose_decision(tmp_path)
        _write_mapping(tmp_path, {**MEMBER_MAPPING, "source_catalog": catalog,
                                  "source_schema": schema})
        with pytest.raises(MappingLeakError, match="different source than privacy profiled"):
            privacy({"run_folder": str(tmp_path)})
        assert gate.STAMP_KEY not in read_flags_file(str(tmp_path))

    @pytest.mark.parametrize("key, profiled, used", [
        ("source_catalog", "gold_raw", "gold"),  # hostile: the mapping's is a substring
        ("source_schema", "hls_phi", "hls"),
    ])
    def test_substring_source_is_refused(self, tmp_path, key, profiled, used):
        _loose_decision(tmp_path)
        schema = json.loads((tmp_path / "schema.json").read_text())
        schema[key] = profiled
        (tmp_path / "schema.json").write_text(json.dumps(schema))
        _write_mapping(tmp_path, {**MEMBER_MAPPING, key: used})
        with pytest.raises(MappingLeakError, match="different source"):
            privacy({"run_folder": str(tmp_path)})

    def test_forged_stamp_over_another_source_is_refused_by_the_gate(self, tmp_path, wh):
        from lgk.steps.build import build

        _loose_decision(tmp_path)
        _write_mapping(tmp_path, {**MEMBER_MAPPING, "source_catalog": "raw_phi",
                                  "source_schema": "ehr"})
        data = read_flags_file(str(tmp_path))
        forge(str(tmp_path), data)  # hostile: re-stamped
        with pytest.raises(MappingLeakError, match="raw_phi"):
            build(_config(tmp_path))
        assert wh.statements == []

    def test_forged_recorded_source_is_refused(self, tmp_path):
        _loose_decision(tmp_path)
        _write_mapping(tmp_path, {**MEMBER_MAPPING, "source_catalog": "raw_phi",
                                  "source_schema": "ehr"})
        data = read_flags_file(str(tmp_path))
        # hostile: flags.json's source edited to match the mapping, stamp recomputed
        data["source"] = {"source_catalog": "raw_phi", "source_schema": "ehr"}
        forge(str(tmp_path), data)
        with pytest.raises(PrivacyGateError, match="recorded source differs"):
            gate.require_clear(str(tmp_path))

    def test_privacy_records_the_profiled_source(self, tmp_path):
        _loose_decision(tmp_path)
        assert read_flags_file(str(tmp_path))["source"] == {
            "source_catalog": "gold", "source_schema": "hls"}

    def test_schema_json_without_a_source_is_refused(self, tmp_path):
        # hostile: nothing says what was profiled, so nothing can be pinned
        _member_schema(tmp_path)
        schema = json.loads((tmp_path / "schema.json").read_text())
        del schema["source_catalog"], schema["source_schema"]
        (tmp_path / "schema.json").write_text(json.dumps(schema))
        _write_mapping(tmp_path, MEMBER_MAPPING)
        with pytest.raises(OpenFlagError):
            privacy({"run_folder": str(tmp_path)})
        resolve_flag(str(tmp_path), "member_notes", "patient_name", "reclassify", chooser="op",
                     reason="x", new_class="none")
        with pytest.raises(MappingLeakError, match="records no profiled source"):
            privacy({"run_folder": str(tmp_path)})

    def test_runner_refuses_the_other_source_twice(self, tmp_path):
        _loose_decision(tmp_path)
        _write_mapping(tmp_path, {**MEMBER_MAPPING, "source_catalog": "raw_phi",
                                  "source_schema": "ehr"})
        built = []
        runner = LaptopRunner(_config(tmp_path))
        for _ in range(2):
            results = runner.run_steps([("privacy", privacy), ("build", lambda c: built.append(1))])
            assert [(r.step, r.status) for r in results] == [("privacy", RunStatus.FAILED)]
        assert built == []


# ================================================= the gate re-classifies schema.json
class TestClassesRecomputed:
    def test_hand_edited_class_with_a_recomputed_stamp_is_refused(self, tmp_path):
        _loose_decision(tmp_path)
        data = read_flags_file(str(tmp_path))
        # hostile: patient_name's class edited to none and its flag removed, then re-stamped
        for c in data["columns"]:
            if c["column"] == "patient_name":
                c["class"] = "none"
        data["flags"] = []
        forge(str(tmp_path), data)
        with pytest.raises(PrivacyGateError, match="classes differ.*member_notes.patient_name"):
            gate.require_clear(str(tmp_path))

    def test_column_removed_from_flags_json_is_refused(self, tmp_path):
        _loose_decision(tmp_path)
        tight = json.loads(json.dumps(MEMBER_MAPPING))
        tight["nodes"]["member"]["label"] = "row_key"
        _write_mapping(tmp_path, tight)
        privacy({"run_folder": str(tmp_path)})
        data = read_flags_file(str(tmp_path))
        # hostile: a care column the mapping does not use, deleted from columns and flags
        data["columns"] = [c for c in data["columns"] if c["column"] != "patient_name"]
        data["flags"] = []
        forge(str(tmp_path), data)
        with pytest.raises(PrivacyGateError, match="classes differ.*member_notes.patient_name"):
            gate.require_clear(str(tmp_path))

    def test_column_added_to_flags_json_is_refused(self, tmp_path):
        _loose_decision(tmp_path)
        ghost = json.loads(json.dumps(MEMBER_MAPPING))
        ghost["nodes"]["member"]["label"] = "ghost_col"
        _write_mapping(tmp_path, ghost)
        data = read_flags_file(str(tmp_path))
        # hostile: a column schema.json never profiled, declared "none" by hand, re-stamped
        data["columns"].append({"table": "members", "column": "ghost_col", "class": "none",
                                "detected_by": None})
        forge(str(tmp_path), data)
        with pytest.raises(PrivacyGateError, match="classes differ.*members.ghost_col"):
            gate.require_clear(str(tmp_path))


# ================================== provision checks the folded edges' floor
class TestProvisionFoldFloor:
    def _built_folded(self, tmp_path, floor=12):
        from lgk.steps.build import build
        from tests.test_fold import _mixed_run

        _mixed_run(tmp_path, privacy_floor=floor)
        build(_config(tmp_path))

    def _edit_build_folds(self, tmp_path, edit):
        path = tmp_path / "privacy_bindings.json"
        data = json.loads(path.read_text())
        edit(data["build"]["folds"])
        path.write_text(json.dumps(data))

    @pytest.mark.parametrize("edit, named", [
        (lambda f: f["shares_refs"].update(floor=11), r"shares_refs \(floor 11\)"),
        (lambda f: f.pop("treats_at"), r"treats_at \(not recorded\)"),
        # hostile (prover O7): a build under a HIGHER floor is not this decision either (equal, not >=)
        (lambda f: f["shares_refs"].update(floor=13), r"shares_refs \(floor 13\)"),
    ])
    def test_provision_refuses_build_made_under_lower_floor(self, tmp_path, wh, served, edit, named):
        from lgk.steps.provision import provision

        self._built_folded(tmp_path)  # flags say floor 12
        # hostile: the build record keeps the CURRENT stamp (so check_build_record passes) and every
        # table property is current; only build.folds says 11 (or lacks a fold type)
        self._edit_build_folds(tmp_path, edit)
        with pytest.raises(StaleBuildError, match=rf"current privacy floor 12: .*{named}"):
            provision(_config(tmp_path))
        assert served == ["_refuse_existing_version"]  # before any synced table is created

    def test_provision_refuses_unfloored_delta_edges(self, tmp_path, wh, served, monkeypatch):
        from lgk.privacy.floor import FoldFloorError
        from lgk.steps.provision import provision

        self._built_folded(tmp_path)

        def warehouse(w, wid, s):
            # hostile: after the build, a writer put a treats_at row of weight 10 into Delta edges
            if s.startswith("SELECT relationship_type, MIN(weight)"):
                assert "FROM cat.kit.edges WHERE relationship_type IN ('shares_refs', 'treats_at')" in s
                return [["treats_at", 10, 0, 1]]
            return wh(s)

        monkeypatch.setattr("lgk.steps.provision.execute_sql", warehouse)
        with pytest.raises(FoldFloorError, match="treats_at"):
            provision(_config(tmp_path))
        assert served == ["_refuse_existing_version"]

    def test_a_floored_fold_build_is_provisioned(self, tmp_path, wh, served):
        from lgk.steps.provision import provision

        self._built_folded(tmp_path)
        provision(_config(tmp_path))  # non-vacuous: the clear path really provisions
        assert served[:2] == ["_refuse_existing_version", "_create_synced_tables"]
