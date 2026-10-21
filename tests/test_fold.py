"""Offline tests for fold edges with a `weight` column.

- mapping: `fold_subject` round-trips and its two shape rules;
- gate: rules 2 and 3 of check_fold_subjects, the care-subject src ban, the folds re-derive check;
- build: the folded INSERT text (co-membership and pair), the unfolded `weight` NULL, the DDL,
  and the floor check between the INSERTs and the stamp;
- assert_fold_floor (Spark) and assert_fold_floor_pg (Postgres detector);
- reapply runs the Postgres detector after the freshness guard.

The warehouse is a scripted fake (no SQL engine): it logs every statement and answers only the
floor check query, with rows the test states. The live half is tests/test_fold_live.py.
"""

import json
import re
from pathlib import Path
from typing import Any, List

import pytest
import yaml

from lgk.mapping import MappingError, dump_mapping, load_mapping, parse_mapping
from lgk.privacy import floor, gate
from lgk.privacy.flags import OpenFlagError, read_flags_file, resolve_flag, write_flags_file
from lgk.privacy.floor import FoldFloorError
from lgk.privacy.gate import MappingLeakError, PrivacyGateError
from lgk.steps.privacy import privacy
from tests.test_privacy import MRN_VALUES, NPI_VALUES, _col, _schema, _visits_and_providers
from tests.test_privacy_gate import FOLDED_MAPPING, PROVIDER_MAPPING, _passed, _write_mapping
from tests.forged_decision import forge

# ----------------------------------------------------------------------------- fixtures
# hostile: a mixed graph. Two fold edge types (one co-membership, one pair) AND an unfolded type
# over the same join table, so a check that does not filter on the fold types sees NULL weights.
MIXED_MAPPING = {
    "source_catalog": "c", "source_schema": "s",
    "nodes": {"provider": {"source_table": "providers", "natural_key": "provider_id",
                           "label": "name", "properties": ["npi"]},
              "facility": {"source_table": "facilities", "natural_key": "facility_id",
                           "label": "site"}},
    "edges": {
        "shares_refs": {"source_table": "visits", "src": "provider_id", "src_type": "provider",
                        "dst": "provider_id", "dst_type": "provider", "fold_subject": "ref_code"},
        "treats_at": {"source_table": "visits", "src": "provider_id", "src_type": "provider",
                      "dst": "facility_id", "dst_type": "facility", "fold_subject": "ref_code"},
        "affiliated": {"source_table": "visits", "src": "provider_id", "src_type": "provider",
                       "dst": "facility_id", "dst_type": "facility"},
    },
}


def _mixed_schema(run_dir: Path) -> None:
    _schema(run_dir, [
        {"name": "providers", "key_columns": ["provider_id"], "foreign_keys": [],
         "columns": [_col("provider_id", ["p1", "p2"]), _col("npi", NPI_VALUES),
                     _col("name", ["Dr A", "Dr B"])]},
        {"name": "facilities", "key_columns": ["facility_id"], "foreign_keys": [],
         "columns": [_col("facility_id", ["f1", "f2"]), _col("site", ["North", "South"])]},
        {"name": "visits", "key_columns": ["visit_seq"],
         "foreign_keys": [{"columns": ["ref_code"], "references": ["patients", "mrn"]},
                          {"columns": ["provider_id"], "references": ["providers", "provider_id"]},
                          {"columns": ["facility_id"], "references": ["facilities", "facility_id"]}],
         "columns": [_col("visit_seq", ["v1", "v2"]), _col("ref_code", MRN_VALUES),
                     _col("provider_id", ["p1", "p2"]), _col("facility_id", ["f1", "f2"])]},
    ])


def _mixed_run(run_dir: Path, privacy_floor=None) -> None:
    """A mixed graph whose privacy step really passed (ref_code folded), at `privacy_floor`."""
    _mixed_schema(run_dir)
    _write_mapping(run_dir, MIXED_MAPPING)
    cfg = {"run_folder": str(run_dir)}
    if privacy_floor is not None:
        cfg["privacy_floor"] = privacy_floor
    with pytest.raises(OpenFlagError):
        privacy(cfg)
    resolve_flag(str(run_dir), "visits", "ref_code", "fold", chooser="op")
    privacy(cfg)


def _unfolded_run(run_dir: Path) -> None:
    _visits_and_providers(run_dir)
    _write_mapping(run_dir, PROVIDER_MAPPING)
    with pytest.raises(OpenFlagError):
        privacy({"run_folder": str(run_dir)})
    resolve_flag(str(run_dir), "visits", "ref_code", "drop", chooser="op")
    resolve_flag(str(run_dir), "members", "patient_name", "drop", chooser="op")
    privacy({"run_folder": str(run_dir)})


CHECK_PREFIX = "SELECT relationship_type, MIN(weight)"


class ScriptedWarehouse:
    """Logs each statement; answers the floor check query with `fold_rows`, everything else []."""

    def __init__(self, fold_rows: List[List[Any]] = ()):
        self.fold_rows = [list(r) for r in fold_rows]
        self.statements: List[str] = []

    def __call__(self, statement: str) -> List[List[Any]]:
        s = " ".join(statement.split())
        self.statements.append(s)
        return self.fold_rows if s.startswith(CHECK_PREFIX) else []

    def index(self, needle: str) -> int:
        return next(i for i, s in enumerate(self.statements) if needle in s)


def _build(run_dir: Path, monkeypatch, wh: ScriptedWarehouse) -> None:
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: object())
    monkeypatch.setattr("lgk.steps.build.execute_sql", lambda w, wid, s: wh(s))
    monkeypatch.setattr("lgk.steps.build._claim_kit_schema", lambda *a, **k: None)
    from tests.label_support import stub_table_identities

    stub_table_identities(monkeypatch)  # the scripted warehouse has no Delta history
    from lgk.steps.build import build

    build({"graph_name": "g", "run_folder": str(run_dir), "lakebase_catalog": "c", "kit_schema": "k",
           "warehouse_id": "wh"})


def _insert_for(wh: ScriptedWarehouse, edge_type: str) -> str:
    """The one edges INSERT for `edge_type` (the unfolded INSERT keeps its lower-case `as`)."""
    tag = re.compile(rf"'{edge_type}' (?:AS|as) relationship_type")
    (stmt,) = [s for s in wh.statements if s.startswith("INSERT INTO c.k.edges") and tag.search(s)]
    return stmt


def _bindings(run_dir: Path) -> dict:
    path = run_dir / "privacy_bindings.json"
    return json.loads(path.read_text()) if path.exists() else {}


# ------------------------------------------------------------------------------ mapping
class TestMappingFoldSubject:
    def test_fold_subject_round_trips_and_an_unfolded_dump_has_no_key(self, tmp_path):
        mapping = parse_mapping(MIXED_MAPPING)
        dump_mapping(mapping, tmp_path / "m.yaml")
        again = load_mapping(tmp_path / "m.yaml")
        assert again.edges["shares_refs"].fold_subject == "ref_code"
        assert again.edges["treats_at"].fold_form == "pair"
        assert again.edges["shares_refs"].fold_form == "co_membership"
        raw = yaml.safe_load((tmp_path / "m.yaml").read_text())
        # hostile: the unfolded edge sits in the same mapping; a dump writing fold_subject: null
        # would change every unfolded mapping's bytes
        assert "fold_subject" not in raw["edges"]["affiliated"]
        assert raw["edges"]["treats_at"]["fold_subject"] == "ref_code"

    def test_fold_subject_equal_to_src_is_refused(self):
        data = json.loads(json.dumps(MIXED_MAPPING))
        data["edges"]["treats_at"]["fold_subject"] = "provider_id"  # hostile: the subject carried as an end
        with pytest.raises(MappingError, match=r"edges\.treats_at\.fold_subject"):
            parse_mapping(data)

    def test_co_membership_across_two_node_types_is_refused(self):
        data = json.loads(json.dumps(MIXED_MAPPING))
        data["edges"]["shares_refs"]["dst_type"] = "facility"  # hostile: src == dst column, two types
        with pytest.raises(MappingError, match=r"edges\.shares_refs\.dst_type"):
            parse_mapping(data)


# --------------------------------------------------------------------------------- gate
class TestGateFoldRules:
    def test_fold_subject_with_drop_choice_is_refused(self, tmp_path):
        _visits_and_providers(tmp_path)
        _write_mapping(tmp_path, FOLDED_MAPPING)
        with pytest.raises(OpenFlagError):
            privacy({"run_folder": str(tmp_path)})
        # hostile: ref_code is dropped, yet an edge still counts it as its fold subject
        resolve_flag(str(tmp_path), "visits", "ref_code", "drop", chooser="op")
        resolve_flag(str(tmp_path), "members", "patient_name", "drop", chooser="op")
        with pytest.raises(MappingLeakError, match=r"fold_subject visits\.ref_code .*not a folded"):
            privacy({"run_folder": str(tmp_path)})

    def test_fold_flag_without_fold_edge_is_refused(self, tmp_path):
        _visits_and_providers(tmp_path)
        _write_mapping(tmp_path, PROVIDER_MAPPING)  # hostile: no edge at all folds ref_code
        with pytest.raises(OpenFlagError):
            privacy({"run_folder": str(tmp_path)})
        resolve_flag(str(tmp_path), "visits", "ref_code", "fold", chooser="op")
        resolve_flag(str(tmp_path), "members", "patient_name", "drop", chooser="op")
        with pytest.raises(MappingLeakError, match=r"flag visits\.ref_code chose fold but no edge"):
            privacy({"run_folder": str(tmp_path)})

    def test_fold_edge_with_a_care_subject_src_is_refused(self, tmp_path):
        _visits_and_providers(tmp_path)
        leaky = json.loads(json.dumps(FOLDED_MAPPING))
        # hostile: a second fold edge whose END is the folded care-subject column itself
        leaky["edges"]["by_ref"] = {"source_table": "visits", "src": "ref_code", "src_type": "provider",
                                    "dst": "provider_id", "dst_type": "provider", "fold_subject": "visit_seq"}
        _write_mapping(tmp_path, leaky)
        with pytest.raises(OpenFlagError):
            privacy({"run_folder": str(tmp_path)})
        resolve_flag(str(tmp_path), "visits", "ref_code", "fold", chooser="op")
        resolve_flag(str(tmp_path), "members", "patient_name", "drop", chooser="op")
        with pytest.raises(MappingLeakError, match=r"still carries .*visits\.ref_code \(edge by_ref src\)"):
            privacy({"run_folder": str(tmp_path)})

    def test_hand_edited_folds_with_a_recomputed_stamp_is_refused(self, tmp_path):
        _passed(tmp_path)
        gate.require_clear(str(tmp_path))  # non-vacuous: the untouched run is clear
        data = read_flags_file(str(tmp_path))
        # hostile: the folds record says pair form, then the stamp is recomputed over the edit
        data["folds"][0]["form"] = "pair"
        forge(str(tmp_path), data)
        with pytest.raises(PrivacyGateError, match="folds differ"):
            gate.require_clear(str(tmp_path))


# ------------------------------------------------------------------------------ build SQL
class TestBuildFoldSql:
    def test_co_membership_insert_counts_distinct_subjects_above_the_floor(self, tmp_path, monkeypatch):
        _mixed_run(tmp_path)
        wh = ScriptedWarehouse()
        _build(tmp_path, monkeypatch, wh)
        ins = _insert_for(wh, "shares_refs")
        assert "COUNT(DISTINCT a.ref_code) AS weight" in ins
        assert "HAVING COUNT(DISTINCT a.ref_code) >= 11" in ins
        assert "FROM c.s.visits a JOIN c.s.visits b ON a.ref_code = b.ref_code AND a.provider_id <> b.provider_id" in ins
        assert "CAST(xxhash64('shares_refs::'||f.src_key||'::'||f.dst_key) AS STRING) AS edge_id" in ins
        assert "CAST(xxhash64('provider::'||f.dst_key) AS STRING) AS dst_node_id" in ins
        assert "f.weight AS weight" in ins
        assert ins.endswith("LEFT SEMI JOIN c.k.nodes s ON e.src_node_id = s.node_id "
                            "LEFT SEMI JOIN c.k.nodes d ON e.dst_node_id = d.node_id")

    def test_raised_floor_is_the_build_having(self, tmp_path, monkeypatch):
        _mixed_run(tmp_path, privacy_floor=12)
        wh = ScriptedWarehouse()
        _build(tmp_path, monkeypatch, wh)
        for edge_type, having in (("shares_refs", "HAVING COUNT(DISTINCT a.ref_code) >= 12"),
                                  ("treats_at", "HAVING COUNT(DISTINCT ref_code) >= 12")):
            ins = _insert_for(wh, edge_type)
            assert having in ins and ">= 11" not in ins

    def test_pair_insert_groups_by_both_ends_and_drops_null_ends(self, tmp_path, monkeypatch):
        _mixed_run(tmp_path)
        wh = ScriptedWarehouse()
        _build(tmp_path, monkeypatch, wh)
        ins = _insert_for(wh, "treats_at")
        assert ("SELECT provider_id AS src_key, facility_id AS dst_key, COUNT(DISTINCT ref_code) AS weight "
                "FROM c.s.visits WHERE provider_id IS NOT NULL AND facility_id IS NOT NULL "
                "GROUP BY provider_id, facility_id HAVING COUNT(DISTINCT ref_code) >= 11") in ins
        assert "CAST(xxhash64('facility::'||f.dst_key) AS STRING) AS dst_node_id" in ins

    def test_unfolded_insert_writes_a_null_weight(self, tmp_path, monkeypatch):
        _mixed_run(tmp_path)
        wh = ScriptedWarehouse()
        _build(tmp_path, monkeypatch, wh)
        ins = _insert_for(wh, "affiliated")
        assert "map() as properties, CAST(NULL AS BIGINT) as weight FROM" in ins
        assert "COUNT(DISTINCT" not in ins and "ROW_NUMBER() OVER (PARTITION BY provider_id, facility_id" in ins

    def test_weight_is_the_last_edges_column(self, tmp_path, monkeypatch):
        _mixed_run(tmp_path)
        wh = ScriptedWarehouse()
        _build(tmp_path, monkeypatch, wh)
        (ddl,) = [s for s in wh.statements if s.startswith("CREATE OR REPLACE TABLE c.k.edges ")]
        cols = re.fullmatch(r"CREATE OR REPLACE TABLE c\.k\.edges \((.*)\) USING DELTA", ddl)[1]
        assert cols.endswith(", properties MAP<STRING, STRING>, weight BIGINT")

    def test_a_fold_row_below_the_floor_stops_build_before_the_stamp(self, tmp_path, monkeypatch):
        _mixed_run(tmp_path)
        # hostile: shares_refs is fine; only treats_at, the second fold type, holds a weight of 10
        wh = ScriptedWarehouse([["shares_refs", 11, 0, 4], ["treats_at", 10, 0, 2]])
        with pytest.raises(FoldFloorError, match=r"treats_at \(min weight 10"):
            _build(tmp_path, monkeypatch, wh)
        assert not [s for s in wh.statements if "lgk.privacy_stamp" in s or "lgk.fold_floor" in s]
        assert "build" not in _bindings(tmp_path)

    def test_a_floored_build_records_its_folds_and_sets_the_fold_floor(self, tmp_path, monkeypatch):
        _mixed_run(tmp_path, privacy_floor=12)
        wh = ScriptedWarehouse([["treats_at", 13, 0, 1], ["shares_refs", 12, 0, 6]])  # hostile: reordered
        _build(tmp_path, monkeypatch, wh)
        check = wh.index(CHECK_PREFIX)
        assert "FROM c.k.edges WHERE relationship_type IN ('shares_refs', 'treats_at')" in wh.statements[check]
        last_insert = max(i for i, s in enumerate(wh.statements) if s.startswith("INSERT INTO"))
        assert last_insert < check < wh.index("lgk.privacy_stamp")
        assert "ALTER TABLE c.k.edges SET TBLPROPERTIES ('lgk.fold_floor' = '12')" in wh.statements
        assert _bindings(tmp_path)["build"]["folds"] == {
            "shares_refs": {"floor": 12, "min_weight": 12, "rows": 6},
            "treats_at": {"floor": 12, "min_weight": 13, "rows": 1}}

    def test_an_unfolded_graph_has_no_fold_check_no_fold_floor_no_folds_record(self, tmp_path, monkeypatch):
        _unfolded_run(tmp_path)
        wh = ScriptedWarehouse([["x", 1, 5, 5]])  # hostile: would fail if the check ever ran
        _build(tmp_path, monkeypatch, wh)
        assert not [s for s in wh.statements if s.startswith(CHECK_PREFIX) or "lgk.fold_floor" in s]
        assert "folds" not in _bindings(tmp_path)["build"]
        assert [s for s in wh.statements if "lgk.privacy_stamp" in s]  # non-vacuous: the build bound


# --------------------------------------------------------------------- assert_fold_floor
class _Sql:
    def __init__(self, rows):
        self.rows, self.statements = rows, []

    def __call__(self, statement):
        self.statements.append(statement)
        return self.rows


class TestAssertFoldFloor:
    def test_min_below_the_floor_raises(self):
        with pytest.raises(FoldFloorError, match=r"a \(min weight 10, 0 NULL\)"):
            floor.assert_fold_floor(_Sql([["a", 10, 0, 5]]), "c.k.edges", ["a"], 11)

    def test_a_null_weight_raises_even_when_the_min_passes(self):
        # hostile: MIN ignores NULL, so the min alone is 11; only the NULL count catches it
        with pytest.raises(FoldFloorError, match=r"a \(min weight 11, 1 NULL\)"):
            floor.assert_fold_floor(_Sql([["a", 11, 1, 5]]), "c.k.edges", ["a"], 11)

    def test_exactly_the_floor_passes_and_is_reported(self):
        report = floor.assert_fold_floor(_Sql([["b", 12, 0, 3], ["a", 12, 0, 5]]), "c.k.edges", ["a", "b"], 12)
        assert report == {"a": {"floor": 12, "min_weight": 12, "rows": 5},
                          "b": {"floor": 12, "min_weight": 12, "rows": 3}}

    def test_no_fold_types_runs_no_query(self):
        sql = _Sql([["a", 1, 1, 1]])
        assert floor.assert_fold_floor(sql, "c.k.edges", [], 11) == {}
        assert sql.statements == []

    def test_the_where_lists_the_fold_types_sorted_once(self):
        sql = _Sql([])
        floor.assert_fold_floor(sql, "c.k.edges", ["treats_at", "shares_refs", "treats_at"], 11)
        (stmt,) = sql.statements
        assert "WHERE relationship_type IN ('shares_refs', 'treats_at')" in stmt


class _PgConn:
    def __init__(self, count):
        self.count, self.calls, self.closed = count, [], False

    def execute(self, sql, params=None):
        self.calls.append((" ".join(sql.split()), params))
        return type("Cur", (), {"fetchone": lambda _s: (self.count,)})()

    def close(self):
        self.closed = True


class TestAssertFoldFloorPg:
    def test_a_counted_row_raises(self):
        conn = _PgConn(2)
        with pytest.raises(FoldFloorError, match="2 folded edge"):
            floor.assert_fold_floor_pg(conn, "k_v1.edges", ["treats_at", "shares_refs"], 11)
        ((sql, params),) = conn.calls
        assert "FROM k_v1.edges WHERE relationship_type = ANY(%s) AND (weight IS NULL OR weight < %s)" in sql
        assert params == (["shares_refs", "treats_at"], 11)

    def test_zero_rows_passes(self):
        floor.assert_fold_floor_pg(_PgConn(0), "k_v1.edges", ["shares_refs"], 11)


# ------------------------------------------------------------------------------- reapply
class TestReapplyDetector:
    def _reapply(self, tmp_path, monkeypatch, count):
        from lgk.steps import reapply as reapply_mod

        _mixed_run(tmp_path)
        order: List[str] = []
        conn = _PgConn(count)
        for name in ("_refuse_an_unservable_version", "_refuse_a_moved_site", "check_freshness", "_reapply_indexes", "_create_served_views",
                     "_setup_reader_role"):
            monkeypatch.setattr(reapply_mod, name, lambda cfg, n=name: order.append(n))

        def connect(cfg):
            order.append("detector")
            return conn

        monkeypatch.setattr(reapply_mod, "connect_branch", connect)
        cfg = {"graph_name": "g", "run_folder": str(tmp_path), "kit_schema": "k", "version": 2}
        return reapply_mod, cfg, order, conn

    def test_reapply_refuses_unfloored_version_rows_after_the_guard(self, tmp_path, monkeypatch):
        reapply_mod, cfg, order, conn = self._reapply(tmp_path, monkeypatch, count=1)
        with pytest.raises(FoldFloorError, match="k_v2.edges"):
            reapply_mod.reapply(cfg)
        assert order == ["_refuse_an_unservable_version", "_refuse_a_moved_site", "check_freshness", "detector"]  # nothing re-served after the detector fails
        ((sql, params),) = conn.calls
        assert "FROM k_v2.edges" in sql and params == (["shares_refs", "treats_at"], 11)
        assert conn.closed

    def test_reapply_with_floored_rows_goes_on(self, tmp_path, monkeypatch):
        reapply_mod, cfg, order, conn = self._reapply(tmp_path, monkeypatch, count=0)
        reapply_mod.reapply(cfg)
        assert order == ["_refuse_an_unservable_version", "_refuse_a_moved_site", "check_freshness", "detector",
                         "_reapply_indexes",
                         "_create_served_views", "_setup_reader_role"]
        assert conn.closed
