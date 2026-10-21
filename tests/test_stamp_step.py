"""The nightly stamp step, Path A: an operator pipeline rebuilds the kit tables as plain Delta.

Owner decision (2026-10-04, "nightly stamp step, option 1, 4 conditions" + "both table kinds"):
- a chain with no kit build runs the refresh task "stamp" after sdp_pipeline and before sync;
- it runs the existing gate, assert_fold_floor and assert_join_floor, then a column / property-key
  allowlist built from the cleared mapping, then labels each table with
  {stamp, table_id, certified_version} and records the same in the run manifest;
- the sync refuses unless label.stamp == manifest.stamp == the current cleared stamp,
  label.table_id == manifest.table_id == the current UC table_id, and no data-changing commit came
  after certified_version (OPTIMIZE and the like do not count);
- the kit build writes the same label with the same writer.

Offline: FakeDelta (tests/test_refresh.py) keeps per table identity a Delta history, a UC table_id,
columns and the (type, property key) rows the allowlist reads. No SQL engine.
"""

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from lgk import manifest as mf
from lgk.privacy import binding
from lgk.privacy.binding import StaleBuildError
from lgk.privacy.gate import PrivacyGateError, require_clear
from lgk.refresh import job, sync
from lgk.refresh.allowlist import AllowlistError
from lgk.refresh.tasks import run_task
from tests.test_refresh import (CAT, KIT_DDL, KIT_FQ, FakeDelta, _RoomyConn, _cleared_run, _config,
                                _fake_build, _sync_w, _uc_tables)

NODES, EDGES, MAP = KIT_FQ


@pytest.fixture(autouse=True)
def _roomy_branch(monkeypatch):
    monkeypatch.setattr(sync, "connect_branch", lambda cfg: _RoomyConn(), raising=False)


class Night:
    """A cleared run whose kit tables an operator pipeline just rebuilt (CREATE OR REPLACE + INSERT)."""

    def __init__(self, tmp_path, monkeypatch, **extra: Any):
        self.cfg = dict(_cleared_run(tmp_path), sdp_pipeline_id="p-1", **extra)
        self.tmp = tmp_path
        self.delta = FakeDelta()
        self.started: List[str] = []
        w, _ = _sync_w({t: ["COMPLETED"] for t in ("nodes", "edges", "node_id_map")})
        w.tables = _uc_tables(self.delta)
        w.pipelines.start_update = lambda pipeline_id, full_refresh: self.started.append(pipeline_id) or \
            SimpleNamespace(update_id="u")
        self.w = w
        monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: w)
        monkeypatch.setattr(sync.time, "sleep", lambda s: None)
        monkeypatch.setattr(sync, "_wait_synced_table_online", lambda w, dest: None)
        monkeypatch.setattr(sync, "live_clients", lambda c: (w, self.sql))
        from lgk.refresh import operator_stamp

        monkeypatch.setattr(operator_stamp, "live_clients", lambda c: (w, self.sql))
        self.hook = None  # called with each statement before the FakeDelta answers it
        self.rebuild()
        # the operator's rows: provider nodes with the mapping's one property, affiliation edges
        self.delta.keys[NODES] = [["provider", "specialty"]]
        self.delta.keys[EDGES] = [["affiliation", None]]

    def sql(self, statement: str) -> List[List[Any]]:
        if self.hook:
            self.hook(statement)
        return self.delta(statement)

    def rebuild(self, ddl: Optional[Dict[str, str]] = None) -> None:
        for fq in KIT_FQ:
            t = fq.rsplit(".", 1)[1]
            self.delta(f"CREATE OR REPLACE TABLE {fq} ({(ddl or {}).get(t, KIT_DDL[t])}) USING DELTA")
            self.delta(f"INSERT INTO {fq} SELECT 1")

    def label(self, fq: str) -> Any:
        raw = self.delta.tables[fq]["props"].get("lgk.privacy_stamp")
        return None if raw is None else json.loads(raw)

    def history(self, fq: str) -> List[Any]:
        return self.delta.tables[fq]["history"]

    def stamp(self) -> str:
        return binding.stamp_id(require_clear(str(self.tmp)))

    def labelled(self) -> bool:
        return any("lgk.privacy_stamp" in s and "SET TBLPROPERTIES" in s for s in self.delta.statements)


def _write_label(night: Night, fq: str, **override: Any) -> None:
    """Hand-write a label (and, unless `manifest` is given, the same manifest record) on `fq`."""
    label = {"stamp": night.stamp(), "table_id": night.delta.table_id(fq),
             "certified_version": night.delta.version(fq)}
    record = dict(label, **override.pop("manifest", {}))
    label.update(override)
    night.delta(f"ALTER TABLE {fq} SET TBLPROPERTIES ('lgk.privacy_stamp' = '{json.dumps(label)}')")
    mf.record_table_labels(night.cfg, {fq: record})


# ------------------------------------------------------------------------------ the chain
class TestChain:
    def test_stamp_sits_after_the_pipeline_and_before_sync_when_there_is_no_kit_build(self):
        assert job.refresh_chain(_config(sdp_pipeline_id="p-1"), enriched=False) == [
            "sdp_pipeline", "stamp", "sync", "reapply"]
        assert job.refresh_chain(_config(sdp_pipeline_id="p-1"), enriched=True) == [
            "sdp_pipeline", "stamp", "enrich", "sync", "reapply"]

    @pytest.mark.parametrize("extra", [{}, {"build_mode": "sql"}, {"sdp_pipeline_id": "p-1", "build_mode": "sql"}])
    def test_a_chain_with_the_kit_build_has_no_stamp_task(self, extra):
        # hostile: pipeline + build_mode sql has BOTH a pipeline and a build: the build labels
        chain = job.refresh_chain(_config(**extra), enriched=False)
        assert "build" in chain and "stamp" not in chain

    def test_the_stamp_task_is_the_stamp_step_on_both_jobs(self):
        from lgk import bundle_entry
        from lgk.refresh import tasks
        from lgk.refresh.operator_stamp import stamp_operator_tables

        assert tasks._task_functions()["stamp"] is stamp_operator_tables
        assert "stamp" in bundle_entry.REFRESH_TASKS


# ------------------------------------------------------------------------- the stamp step
class TestStampStep:
    def test_stamp_labels_each_table_and_records_the_manifest_then_the_sync_runs(self, tmp_path, monkeypatch):
        night = Night(tmp_path, monkeypatch)
        # hostile: an OPTIMIZE after the operator's INSERT (no data): the checked version is the
        # OPTIMIZE's, and certified_version is that version (N-1), never the label's own commit
        night.delta._commit(EDGES, "OPTIMIZE", {"numAddedFiles": "1", "numRemovedFiles": "2"})
        checked = {fq: night.delta.version(fq) for fq in KIT_FQ}
        run_task("stamp", json.dumps(night.cfg))
        recorded = mf.read_table_labels(night.cfg)
        for fq in KIT_FQ:
            want = {"stamp": night.stamp(), "table_id": night.delta.table_id(fq), "certified_version": checked[fq]}
            assert night.label(fq) == want, fq
            assert recorded[fq] == want, fq
            # the label is commit N = certified_version + 1, a SET TBLPROPERTIES (no data)
            assert night.history(fq)[-1][:2] == (checked[fq] + 1, "SET TBLPROPERTIES"), fq
        run_task("sync", json.dumps(night.cfg))
        assert sorted(night.started) == ["pipe-edges", "pipe-node_id_map", "pipe-nodes"]

    def test_the_stamp_overwrites_a_retired_bare_stamp_the_pipeline_set(self, tmp_path, monkeypatch):
        night = Night(tmp_path, monkeypatch)
        for fq in KIT_FQ:  # hostile: the retired SDP form, table_properties={"lgk.privacy_stamp": <stamp>}
            night.delta(f"ALTER TABLE {fq} SET TBLPROPERTIES ('lgk.privacy_stamp' = '{night.stamp()}')")
        with pytest.raises(StaleBuildError, match=r"lgk_v1_abc123\.nodes: label is not the"):
            run_task("sync", json.dumps(night.cfg))
        run_task("stamp", json.dumps(night.cfg))
        run_task("sync", json.dumps(night.cfg))
        assert len(night.started) == 3

    @pytest.mark.parametrize("break_gate", ["edit_mapping", "no_flags"])
    def test_the_gate_fails_closed(self, tmp_path, monkeypatch, break_gate):
        night = Night(tmp_path, monkeypatch)
        if break_gate == "edit_mapping":  # hostile: mapping edited after privacy cleared it
            mapping = tmp_path / "graph_mapping.yaml"
            mapping.write_text(mapping.read_text().replace("label: name", "label: specialty"))
        else:
            (tmp_path / "flags.json").unlink()
        with pytest.raises(PrivacyGateError):
            run_task("stamp", json.dumps(night.cfg))
        assert not night.labelled() and mf.read_table_labels(night.cfg) == {}

    def test_the_fold_floor_fails_closed(self, tmp_path, monkeypatch):
        from lgk.privacy.floor import FoldFloorError
        from tests.test_fold import _mixed_run

        night = Night(tmp_path, monkeypatch)
        _mixed_run(tmp_path, 12)  # hostile: a RAISED floor 12; the operator's edge weighs 11
        night.delta.keys[EDGES] = [["shares_refs", None], ["treats_at", None]]
        real = night.sql

        def sql(statement):
            if statement.startswith("SELECT relationship_type, MIN(weight)"):
                return [["shares_refs", 11, 0, 3]]
            return real(statement)

        night.sql = sql
        with pytest.raises(FoldFloorError, match=r"floor 12: shares_refs \(min weight 11"):
            run_task("stamp", json.dumps(night.cfg))
        assert not night.labelled() and mf.read_table_labels(night.cfg) == {}


# ------------------------------------------------------------------------------ allowlist
def _allowlist_refusal(night: Night, match: str) -> None:
    with pytest.raises(AllowlistError, match=match):
        run_task("stamp", json.dumps(night.cfg))
    assert not night.labelled() and mf.read_table_labels(night.cfg) == {}


class TestAllowlist:
    def test_a_hostile_unknown_column_is_refused(self, tmp_path, monkeypatch):
        night = Night(tmp_path, monkeypatch)
        # hostile: the operator's nodes carry one extra column (a care subject the mapping dropped)
        night.rebuild({"nodes": KIT_DDL["nodes"] + ", patient_ssn STRING"})
        _allowlist_refusal(night, r"lgk_v1_abc123\.nodes: unknown column\(s\) patient_ssn string")

    @pytest.mark.parametrize("table,ddl,match", [
        ("edges", KIT_DDL["edges"].replace("weight BIGINT", "weight STRING"),
         r"edges: unknown column\(s\) weight string; missing column\(s\) weight bigint"),
        ("node_id_map", "node_id STRING, node_type STRING", r"node_id_map: missing column\(s\) natural_key"),
        ("edges", "edge_id STRING, src_node_id STRING, dst_node_id STRING, relationship_type STRING, "
                  "weight BIGINT, properties MAP<STRING, STRING>", r"edges: columns out of order"),
    ])
    def test_a_changed_type_a_missing_column_or_another_order_is_refused(self, tmp_path, monkeypatch, table, ddl,
                                                                          match):
        night = Night(tmp_path, monkeypatch)
        night.rebuild({table: ddl})
        _allowlist_refusal(night, match)

    def test_a_hostile_unknown_property_key_is_refused(self, tmp_path, monkeypatch):
        night = Night(tmp_path, monkeypatch)
        # hostile: one provider node's map carries a key the mapping never allowed
        night.delta.keys[NODES] = [["provider", "specialty"], ["provider", "patient_name"]]
        _allowlist_refusal(night, r"nodes: node_type 'provider' has property key 'patient_name'")

    def test_an_unknown_edge_property_key_or_type_is_refused(self, tmp_path, monkeypatch):
        night = Night(tmp_path, monkeypatch)
        # hostile: build writes map() on every edge, so ANY edge key is unknown; plus a type not mapped
        night.delta.keys[EDGES] = [["affiliation", "member_id"], ["shares_patients", None]]
        _allowlist_refusal(night, r"relationship_type 'affiliation' has property key 'member_id'.*"
                                  r"relationship_type 'shares_patients' is not in the cleared mapping")

    def test_an_unknown_node_type_with_an_empty_map_is_refused(self, tmp_path, monkeypatch):
        night = Night(tmp_path, monkeypatch)
        night.delta.keys[NODES] = [["provider", "specialty"], ["member", None]]  # hostile: no keys at all
        _allowlist_refusal(night, r"node_type 'member' is not in the cleared mapping")

    def test_the_allowed_keys_come_from_the_cleared_mapping_with_join_outputs(self, tmp_path):
        """Varied vocabulary: a mapping with joins allows its native properties AND every join output."""
        from lgk.mapping import parse_mapping
        from lgk.refresh.allowlist import assert_allowlisted
        from tests.test_build_sql import JOIN_MAPPING

        delta = FakeDelta()
        for fq in KIT_FQ:
            delta(f"CREATE TABLE {fq} ({KIT_DDL[fq.rsplit('.', 1)[1]]}) USING DELTA")
        delta.keys[NODES] = [["provider", k] for k in ("total_billed", "claim_count", "taxonomy_code", "specialty")]
        mapping = parse_mapping(JOIN_MAPPING)
        assert_allowlisted(delta, f"{CAT}.lgk_v1_abc123", mapping)  # non-vacuous: every allowed key passes
        delta.keys[NODES].append(["provider", "patient_id"])  # hostile: the join's folded subject column
        with pytest.raises(AllowlistError, match="'patient_id'"):
            assert_allowlisted(delta, f"{CAT}.lgk_v1_abc123", mapping)

    def test_no_mapping_is_refused(self):
        from lgk.refresh.allowlist import assert_allowlisted

        with pytest.raises(AllowlistError, match="no graph_mapping.yaml"):
            assert_allowlisted(FakeDelta(), "cat.kit", None)


# ------------------------------------------------- the label is checked between read and write
class TestChangedWhileChecked:
    def test_a_data_commit_during_the_checks_is_refused(self, tmp_path, monkeypatch):
        night = Night(tmp_path, monkeypatch)

        def write_once(statement):  # hostile: the operator's pipeline writes again mid-check
            if statement.startswith("DESCRIBE TABLE") and night.hook:
                night.hook = None
                night.delta(f"INSERT INTO {NODES} SELECT 2")

        night.hook = write_once
        with pytest.raises(StaleBuildError, match=r"nodes: a data commit at version \d+ came after the checked"):
            run_task("stamp", json.dumps(night.cfg))
        assert not night.labelled()

    def test_a_table_replaced_during_the_checks_is_refused(self, tmp_path, monkeypatch):
        night = Night(tmp_path, monkeypatch)

        def replace_once(statement):  # hostile: DROP + CREATE: a new UC table_id
            if statement.startswith("DESCRIBE TABLE") and night.hook:
                night.hook = None
                night.delta(f"DROP TABLE IF EXISTS {EDGES}")
                night.delta(f"CREATE TABLE {EDGES} ({KIT_DDL['edges']}) USING DELTA")

        night.hook = replace_once
        with pytest.raises(StaleBuildError, match=r"edges: table_id changed"):
            run_task("stamp", json.dumps(night.cfg))
        assert not night.labelled()


# ----------------------------------------------------------- the sync's label requirements
class TestSyncRequiresTheLabel:
    @pytest.mark.parametrize("override,match", [
        ({"stamp": "0" * 64}, r"edges: label\.stamp is a stale stamp"),
        ({"manifest": {"stamp": "0" * 64}}, r"edges: manifest\.stamp is a stale stamp"),
        ({"table_id": "uc-tid-other"}, r"edges: label\.table_id uc-tid-other is not the current table_id"),
        ({"manifest": {"table_id": "uc-tid-other"}}, r"edges: manifest\.table_id uc-tid-other is not the current"),
        # hostile: label and manifest disagree on the version, while each alone is consistent
        ({"certified_version": 1}, r"edges: label\.certified_version 1 differs from manifest\.certified_version"),
        ({"manifest": {"certified_version": 1}},
         r"edges: label\.certified_version \d+ differs from manifest\.certified_version 1"),
        ({"certified_version": 99, "manifest": {"certified_version": 99}},
         r"edges: certified_version 99 is ahead of the current version"),
    ])
    def test_each_mismatched_field_is_refused_naming_table_and_field(self, tmp_path, monkeypatch, override, match):
        night = Night(tmp_path, monkeypatch)
        run_task("stamp", json.dumps(night.cfg))
        _write_label(night, EDGES, **override)
        with pytest.raises(StaleBuildError, match=match):
            run_task("sync", json.dumps(night.cfg))
        assert night.started == []

    def test_a_label_with_no_manifest_record_is_refused(self, tmp_path, monkeypatch):
        night = Night(tmp_path, monkeypatch)
        run_task("stamp", json.dumps(night.cfg))
        path = tmp_path / mf.MANIFEST_FILE
        data = json.loads(path.read_text())
        del data["table_labels"][MAP]  # hostile: only the last table's record is missing
        path.write_text(json.dumps(data))
        with pytest.raises(StaleBuildError, match=r"node_id_map: the run manifest records no label"):
            run_task("sync", json.dumps(night.cfg))

    @pytest.mark.parametrize("raw", [
        '{"stamp": "x"}', "not json", '{"stamp": "s", "table_id": "t", "certified_version": true}',
        # prover survivors: a negative version, non-string stamp / table_id, a fourth key
        '{"stamp": "s", "table_id": "t", "certified_version": -1}',
        '{"stamp": 1, "table_id": "t", "certified_version": 0}',
        '{"stamp": "s", "table_id": 2, "certified_version": 0}',
        '{"stamp": "s", "table_id": "t", "certified_version": 0, "x": 1}',
    ])
    def test_a_label_that_is_not_the_three_fields_is_refused(self, tmp_path, monkeypatch, raw):
        night = Night(tmp_path, monkeypatch)
        run_task("stamp", json.dumps(night.cfg))
        night.delta(f"ALTER TABLE {NODES} SET TBLPROPERTIES ('lgk.privacy_stamp' = '{raw}')")
        with pytest.raises(StaleBuildError, match=r"nodes: label is not the \{stamp, table_id, certified_version\}"):
            run_task("sync", json.dumps(night.cfg))

    def test_a_data_commit_after_certified_version_is_refused(self, tmp_path, monkeypatch):
        night = Night(tmp_path, monkeypatch)
        run_task("stamp", json.dumps(night.cfg))
        night.delta(f"INSERT INTO {NODES} SELECT 3")  # hostile: a manual write after stamping
        with pytest.raises(StaleBuildError, match=r"nodes: certified_version \d+, but a data commit at version"):
            run_task("sync", json.dumps(night.cfg))
        assert night.started == []

    def test_optimize_and_properties_after_certified_version_are_allowed(self, tmp_path, monkeypatch):
        night = Night(tmp_path, monkeypatch)
        run_task("stamp", json.dumps(night.cfg))
        for fq in KIT_FQ:  # hostile: maintenance commits after the label (predictive optimization)
            night.delta._commit(fq, "OPTIMIZE", {"numAddedFiles": "1", "numRemovedFiles": "3"})
        night.delta(f"ALTER TABLE {EDGES} SET TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')")
        run_task("sync", json.dumps(night.cfg))
        assert len(night.started) == 3

    def test_a_rebuild_after_stamping_is_refused_even_with_the_same_version_numbers(self, tmp_path, monkeypatch):
        night = Night(tmp_path, monkeypatch)
        run_task("stamp", json.dumps(night.cfg))
        old = {fq: (night.delta.table_id(fq), night.delta.version(fq)) for fq in KIT_FQ}
        yesterday = night.delta.tables[EDGES]["props"]["lgk.privacy_stamp"]
        # hostile (S3b): DROP + CREATE lands on the same version numbers, and the pipeline sets
        # yesterday's label again (a table property it carries over): only the table_id differs
        night.delta(f"DROP TABLE IF EXISTS {EDGES}")
        night.delta(f"CREATE TABLE {EDGES} ({KIT_DDL['edges']}) USING DELTA")
        while night.delta.version(EDGES) < old[EDGES][1] - 1:
            night.delta._commit(EDGES, "OPTIMIZE", {"numAddedFiles": "0", "numRemovedFiles": "0"})
        night.delta(f"ALTER TABLE {EDGES} SET TBLPROPERTIES ('lgk.privacy_stamp' = '{yesterday}')")
        assert night.delta.version(EDGES) == old[EDGES][1] and night.delta.table_id(EDGES) != old[EDGES][0]
        with pytest.raises(StaleBuildError, match=r"edges: label\.table_id uc-tid-\d+ is not the current table_id"):
            run_task("sync", json.dumps(night.cfg))

    def test_a_privacy_re_decision_after_stamping_is_refused_until_stamped_again(self, tmp_path, monkeypatch):
        from tests.privacy_support import clear_gate_offline

        night = Night(tmp_path, monkeypatch)
        run_task("stamp", json.dumps(night.cfg))
        old = night.stamp()
        # hostile: the operator re-clears privacy with a different decision (another stamp)
        flags = json.loads((tmp_path / "flags.json").read_text())
        mapping = tmp_path / "graph_mapping.yaml"
        mapping.write_text(mapping.read_text().replace("properties: [specialty]", "properties: []"))
        clear_gate_offline(str(tmp_path))
        assert night.stamp() != old, flags  # non-vacuous
        night.delta.keys[NODES] = [["provider", None]]
        with pytest.raises(StaleBuildError, match=r"label\.stamp is a stale stamp"):
            run_task("sync", json.dumps(night.cfg))
        run_task("stamp", json.dumps(night.cfg))
        run_task("sync", json.dumps(night.cfg))
        assert len(night.started) == 3


# --------------------------------------------------------- the kit build writes the same label
class TestBuildWritesTheSameLabel:
    def test_the_kit_build_labels_with_the_same_shape_and_the_sync_checks_it_the_same_way(self, tmp_path,
                                                                                           monkeypatch):
        cfg = _cleared_run(tmp_path)
        delta = FakeDelta()
        _fake_build(monkeypatch, delta)
        run_task("build", json.dumps(cfg))
        recorded = mf.read_table_labels(cfg)
        stamp = binding.stamp_id(require_clear(str(tmp_path)))
        for fq in KIT_FQ:
            hist = delta.tables[fq]["history"]
            label_commit = next(v for v, op, _ in hist if op == "SET TBLPROPERTIES")
            # certified_version: the last data commit the build wrote and the floors checked (N-1)
            assert hist[label_commit - 1][1] == "WRITE", fq
            want = {"stamp": stamp, "table_id": delta.table_id(fq), "certified_version": label_commit - 1}
            assert json.loads(delta.tables[fq]["props"]["lgk.privacy_stamp"]) == want, fq
            assert recorded[fq] == want, fq
        # the same check the operator chain's sync runs accepts the kit build's labels
        w = SimpleNamespace(tables=_uc_tables(delta))
        from lgk.refresh.label import check_labels

        check_labels(w, delta, cfg, KIT_FQ, stamp)
        delta(f"INSERT INTO {MAP} SELECT 9")  # hostile: a write after the build's label
        with pytest.raises(StaleBuildError, match=r"node_id_map: certified_version"):
            check_labels(w, delta, cfg, KIT_FQ, stamp)


# ------------------------------------------------------------- prover survivors (Path A)
class TestStrictLabelEdges:
    def test_provision_check_refuses_the_retired_bare_stamp_on_a_kit_table(self):
        delta = FakeDelta()
        stamp = "a" * 64
        for fq in KIT_FQ:
            delta(f"CREATE TABLE {fq} ({KIT_DDL[fq.rsplit('.', 1)[1]]}) USING DELTA")
            delta(f"ALTER TABLE {fq} SET TBLPROPERTIES ('lgk.privacy_stamp' = '{stamp}')")  # hostile: bare
        with pytest.raises(StaleBuildError, match=r"nodes \(the label is not the \{stamp, table_id, certified_version\}"):
            binding.check_table_stamps(delta, KIT_FQ, stamp)
        # an enrichment table keeps the bare stamp (only the kit writes it)
        delta(f"CREATE TABLE {CAT}.lgk_v1_abc123.node_score (node_id STRING) USING DELTA")
        delta(f"ALTER TABLE {CAT}.lgk_v1_abc123.node_score SET TBLPROPERTIES ('lgk.privacy_stamp' = '{stamp}')")
        binding.check_table_stamps(delta, [f"{CAT}.lgk_v1_abc123.node_score"], stamp)

    def test_the_build_refuses_to_label_a_table_written_during_its_floor_check(self, tmp_path, monkeypatch):
        from tests.test_fold import CHECK_PREFIX, _mixed_run

        _mixed_run(tmp_path)  # a folded graph: the build runs a fold-floor query before it labels
        cfg = _config(run_folder=str(tmp_path))
        delta = FakeDelta()
        _fake_build(monkeypatch, delta)
        import lgk.steps.build as build_mod

        def sql(w, wh, statement):
            if " ".join(statement.split()).startswith(CHECK_PREFIX):  # hostile: a writer lands mid-check
                delta(f"INSERT INTO {NODES} SELECT 7")
            return delta(statement)

        monkeypatch.setattr(build_mod, "execute_sql", sql)
        with pytest.raises(StaleBuildError, match=r"nodes: a data commit at version \d+ came after the checked"):
            run_task("build", json.dumps(cfg))
        assert not any("lgk.privacy_stamp" in s for s in delta.statements)

    @pytest.mark.parametrize("table_id", [None, ""])
    def test_no_uc_table_id_is_refused(self, table_id):
        from lgk.refresh.label import uc_table_id

        w = SimpleNamespace(tables=SimpleNamespace(get=lambda full_name: SimpleNamespace(table_id=table_id)))
        with pytest.raises(StaleBuildError, match="returned no table_id"):
            uc_table_id(w, NODES)

    def test_a_label_with_a_quote_is_never_spliced(self):
        with pytest.raises(ValueError, match="never holds a quote"):
            binding.set_label_sql(NODES, binding.label_json("s'; DROP TABLE x; --", "t", 1))

    def test_a_partitioned_table_passes_the_column_check(self):
        """DESCRIBE TABLE of a partitioned table repeats a column under '# Partition Information'."""
        from lgk.mapping import parse_mapping
        from lgk.refresh.allowlist import assert_allowlisted
        from tests.test_build_sql import JOIN_MAPPING

        delta = FakeDelta()
        for fq in KIT_FQ:
            delta(f"CREATE TABLE {fq} ({KIT_DDL[fq.rsplit('.', 1)[1]]}) USING DELTA")
        delta.tables[NODES]["columns"] += [["", "", None], ["# Partition Information", "", None],
                                           ["# col_name", "data_type", "comment"], ["node_type", "string", None]]
        assert_allowlisted(delta, f"{CAT}.lgk_v1_abc123", parse_mapping(JOIN_MAPPING))
