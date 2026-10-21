"""A node keyed on a NEUTRALLY named foreign key that points, directly or
through a chain of declared foreign keys, at a care-subject column is that person's node too.

claims.owner_ref -> patients.patientid passes the name classifier ('owner_ref' is class none),
so before this check a hand-written node keyed on owner_ref would publish patient ids. The gate
reads schema.json's declared foreign_keys (recorded in flags.json by the privacy step and pinned
to schema.json by require_clear) and follows them transitively, with a cycle guard.

Decision 2a: a reclassify never clears it (the classifier's class is what counts).
"""

import json
import multiprocessing
from unittest.mock import MagicMock

import pytest
import yaml

from lgk.privacy import gate
from lgk.privacy.flags import OpenFlagError, open_flags, read_flags_file, resolve_flag
from lgk.privacy.gate import CareKeyNodeError, PrivacyGateError
from lgk.runner import LaptopRunner, RunStatus
from lgk.steps.privacy import privacy
from tests.forged_decision import forge


def _col(name):
    return {"name": name, "type": "STRING"}


def _fk(cols, ref_table, *ref_cols):
    return {"columns": list(cols), "references": [ref_table, *ref_cols]}


SCHEMA = {"source_catalog": "c", "source_schema": "s", "tables": [
    {"name": "providers", "key_columns": ["provider_id"], "foreign_keys": [],
     "columns": [_col("provider_id"), _col("name"), _col("city")]},
    {"name": "patients", "key_columns": ["patientid"], "foreign_keys": [],
     "columns": [_col("patientid"), _col("city")]},
    # hostile: owner_ref is a neutral name (class none) that holds patient ids
    {"name": "claims", "key_columns": ["claim_id"],
     "foreign_keys": [_fk(["owner_ref"], "patients", "patientid"),
                      _fk(["provider_ref"], "providers", "provider_id")],
     "columns": [_col("claim_id"), _col("owner_ref"), _col("provider_ref"), _col("amount")]},
    # hostile: two hops, referrals.ref_owner -> claims.owner_ref -> patients.patientid
    {"name": "referrals", "key_columns": ["referral_id"],
     "foreign_keys": [_fk(["ref_owner"], "claims", "owner_ref")],
     "columns": [_col("referral_id"), _col("ref_owner"), _col("title")]},
    # hostile: an FK cycle between two tables with no care column on it (must terminate, pass)
    {"name": "loop_a", "key_columns": ["a_id"],
     "foreign_keys": [_fk(["b_ref"], "loop_b", "a_ref")],
     "columns": [_col("a_id"), _col("b_ref"), _col("title")]},
    {"name": "loop_b", "key_columns": ["b_id"],
     "foreign_keys": [_fk(["a_ref"], "loop_a", "b_ref")],
     "columns": [_col("b_id"), _col("a_ref"), _col("title")]},
]}

CLEAN = {"source_catalog": "c", "source_schema": "s",
         "nodes": {"providers": {"source_table": "providers", "natural_key": "provider_id",
                                 "label": "name", "properties": ["city"]}},
         "edges": {}}


def with_node(node_type, table, key, label):
    m = json.loads(json.dumps(CLEAN))
    m["nodes"][node_type] = {"source_table": table, "natural_key": key, "label": label,
                             "properties": []}
    return m


OWNER = with_node("owner", "claims", "owner_ref", "amount")
TWO_HOP = with_node("referral_owner", "referrals", "ref_owner", "title")
# hostile: composite key where only the SECOND column is the care FK
COMPOSITE = with_node("claim_owner", "claims", ["claim_id", "owner_ref"], "amount")
# an FK to a non-care table: must still pass
PROVIDER_REF = with_node("claim_provider", "claims", "provider_ref", "amount")
# an FK cycle with no care column: must terminate and pass
LOOP = with_node("loop", "loop_a", "b_ref", "title")

REFUSED = [(OWNER, "claims.owner_ref", "patients.patientid"),
           (TWO_HOP, "referrals.ref_owner", "patients.patientid"),
           (COMPOSITE, "claims.owner_ref", "patients.patientid")]
PASSING = [PROVIDER_REF, LOOP]


def _write(run_dir, mapping, schema=SCHEMA):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "schema.json").write_text(json.dumps(schema))
    (run_dir / "graph_mapping.yaml").write_text(yaml.safe_dump(mapping, sort_keys=False))


def _resolve(run_dir, choice):
    """Resolve every open flag (here only patients.patientid) with `choice`."""
    try:
        privacy({"run_folder": str(run_dir)})
    except OpenFlagError:
        pass
    flagged = {(f["table"], f["column"]) for f in open_flags(str(run_dir))}
    assert flagged == {("patients", "patientid")}, flagged
    kw = {"reason": "hand-written to test the gate", "new_class": "none"} \
        if choice == "reclassify" else {}
    resolve_flag(str(run_dir), "patients", "patientid", choice, chooser="op", **kw)


def _cleared_then_forged(run_dir, mapping, choice):
    """Pass privacy on CLEAN, then swap in `mapping` and forge a fresh stamp."""
    _write(run_dir, CLEAN)
    _resolve(run_dir, choice)
    privacy({"run_folder": str(run_dir)})
    (run_dir / "graph_mapping.yaml").write_text(yaml.safe_dump(mapping, sort_keys=False))
    data = {k: v for k, v in read_flags_file(str(run_dir)).items() if k != gate.STAMP_KEY}
    forge(str(run_dir), data)


def _bounded(fn, seconds=5.0):
    """Run fn in a forked child; fail (not hang) if it does not finish in `seconds`, and kill
    the child, so a missing cycle guard reports cleanly instead of spinning a thread."""
    ctx = multiprocessing.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)

    def target():
        try:
            child.send(("value", fn()))
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            child.send(("error", exc))

    proc = ctx.Process(target=target, daemon=True)
    proc.start()
    ready = parent.poll(seconds)
    if not ready:
        proc.kill()
        proc.join()
    assert ready, f"did not finish in {seconds}s (FK cycle not guarded?)"
    kind, value = parent.recv()
    proc.join()
    if kind == "error":
        raise value
    return value


# ------------------------------------------------------------------------ the privacy step
@pytest.mark.parametrize("choice", ["drop", "reclassify"])
@pytest.mark.parametrize("mapping,column,care", REFUSED)
def test_the_privacy_step_refuses_a_node_keyed_on_a_care_fk(tmp_path, mapping, column, care, choice):
    _write(tmp_path, mapping)
    _resolve(tmp_path, choice)
    node_type = next(t for t in mapping["nodes"] if t not in CLEAN["nodes"])
    with pytest.raises(CareKeyNodeError) as err:
        _bounded(lambda: privacy({"run_folder": str(tmp_path)}))
    msg = str(err.value)
    assert f"node type {node_type} is keyed on care-subject" in msg
    assert f"{column} (foreign key to {care})" in msg
    assert gate.STAMP_KEY not in read_flags_file(str(tmp_path))


@pytest.mark.parametrize("mapping", PASSING)
def test_an_fk_to_a_non_care_table_and_an_fk_cycle_pass(tmp_path, mapping):
    """Non-vacuous: FK keys that never reach a care column clear the gate (and terminate)."""
    _write(tmp_path, mapping)
    _resolve(tmp_path, "drop")
    _bounded(lambda: privacy({"run_folder": str(tmp_path)}))
    assert _bounded(lambda: gate.require_clear(str(tmp_path)))


def test_a_cycle_that_also_leads_to_care_is_refused_and_terminates(tmp_path):
    """hostile: a cycle a<->b where b also points at the care column; a DFS that stops at the
    first revisit must still find the exit, and must not loop."""
    data = {"columns": [{"table": "patients", "column": "patientid", "class": "care_subject"}],
            "foreign_keys": [
                {"table": "loop_a", "columns": ["b_ref"], "references": ["loop_b", "a_ref"]},
                {"table": "loop_b", "columns": ["a_ref"], "references": ["loop_a", "b_ref"]},
                {"table": "loop_b", "columns": ["a_ref"], "references": ["patients", "patientid"]}]}
    _write(tmp_path, LOOP)
    m = gate.load_mapping(tmp_path / "graph_mapping.yaml")
    with pytest.raises(CareKeyNodeError, match=r"loop_a\.b_ref \(foreign key to patients\.patientid\)"):
        _bounded(lambda: gate.check_care_key_nodes(data, m))


def test_the_privacy_step_records_schema_foreign_keys(tmp_path):
    _write(tmp_path, CLEAN)
    _resolve(tmp_path, "drop")
    privacy({"run_folder": str(tmp_path)})
    recorded = read_flags_file(str(tmp_path))["foreign_keys"]
    assert {"table": "claims", "columns": ["owner_ref"],
            "references": ["patients", "patientid"]} in recorded
    assert len(recorded) == 5


# ------------------------------------------------------------------- require_clear (forged)
@pytest.mark.parametrize("choice", ["drop", "reclassify"])
@pytest.mark.parametrize("mapping,column,care", REFUSED)
def test_require_clear_refuses_a_forged_stamp(tmp_path, mapping, column, care, choice):
    _cleared_then_forged(tmp_path, mapping, choice)
    with pytest.raises(CareKeyNodeError, match=rf"{column} \(foreign key to {care}\)"):
        _bounded(lambda: gate.require_clear(str(tmp_path)))


def test_require_clear_refuses_flags_with_the_fk_record_removed(tmp_path):
    """hostile: hand-delete flags.json's foreign_keys (and re-stamp) to hide the FK: refused."""
    _cleared_then_forged(tmp_path, OWNER, "drop")
    data = {k: v for k, v in read_flags_file(str(tmp_path)).items() if k != gate.STAMP_KEY}
    data["foreign_keys"] = []
    forge(str(tmp_path), data)
    with pytest.raises(PrivacyGateError, match="foreign keys differ"):
        gate.require_clear(str(tmp_path))


def test_require_clear_refuses_an_fk_added_to_schema_after_clearing(tmp_path):
    """hostile: schema.json gains the FK after the step passed; even with a re-forged stamp
    the recorded FKs are pinned to schema.json now."""
    schema = json.loads(json.dumps(SCHEMA))
    claims = next(t for t in schema["tables"] if t["name"] == "claims")
    claims["foreign_keys"] = [fk for fk in claims["foreign_keys"] if fk["columns"] != ["owner_ref"]]
    _write(tmp_path, OWNER, schema)
    _resolve(tmp_path, "drop")
    privacy({"run_folder": str(tmp_path)})  # no FK declared: owner_ref looks neutral
    (tmp_path / "schema.json").write_text(json.dumps(SCHEMA))
    data = {k: v for k, v in read_flags_file(str(tmp_path)).items() if k != gate.STAMP_KEY}
    forge(str(tmp_path), data)
    with pytest.raises(PrivacyGateError, match="foreign keys differ"):
        gate.require_clear(str(tmp_path))
    with pytest.raises(CareKeyNodeError):
        privacy({"run_folder": str(tmp_path)})


# ------------------------------------------------------------ build / provision directly
@pytest.mark.parametrize("choice", ["drop", "reclassify"])
def test_build_called_directly_refuses_before_any_sql(tmp_path, monkeypatch, choice):
    from tests.test_build_sql import _Warehouse, run_build

    _cleared_then_forged(tmp_path, OWNER, choice)
    wh = _Warehouse()
    with pytest.raises(CareKeyNodeError):
        run_build(tmp_path, monkeypatch, wh)
    assert wh.statements == []


@pytest.mark.parametrize("choice", ["drop", "reclassify"])
def test_provision_called_directly_refuses_before_any_sql(tmp_path, monkeypatch, choice):
    from lgk.steps import provision as p

    _cleared_then_forged(tmp_path, TWO_HOP, choice)
    statements = []
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: MagicMock())
    monkeypatch.setattr(p, "execute_sql", lambda *a, **k: statements.append(a) or [])
    with pytest.raises(CareKeyNodeError):
        p.provision({"run_folder": str(tmp_path), "lakebase_catalog": "main",
                     "kit_schema": "lgk_care", "warehouse_id": "wh", "version": 1})
    assert statements == []


def test_an_edit_after_clearing_is_refused_and_stays_refused(tmp_path, monkeypatch):
    from tests.test_build_sql import _Warehouse, run_build

    _write(tmp_path, CLEAN)
    _resolve(tmp_path, "reclassify")
    privacy({"run_folder": str(tmp_path)})
    (tmp_path / "graph_mapping.yaml").write_text(yaml.safe_dump(OWNER, sort_keys=False))
    wh = _Warehouse()
    with pytest.raises(PrivacyGateError, match="stale"):
        run_build(tmp_path, monkeypatch, wh)
    with pytest.raises(CareKeyNodeError):
        privacy({"run_folder": str(tmp_path)})
    with pytest.raises(PrivacyGateError):
        run_build(tmp_path, monkeypatch, wh)
    assert wh.statements == []


def test_a_reclassify_of_owner_ref_does_not_clear_it(tmp_path):
    """Decision 2a: owner_ref has no flag to reclassify (KeyError), and reclassifying the care
    column it points at (patients.patientid -> none) does not clear the node either."""
    _write(tmp_path, OWNER)
    _resolve(tmp_path, "reclassify")
    with pytest.raises(KeyError):
        resolve_flag(str(tmp_path), "claims", "owner_ref", "reclassify", chooser="op",
                     reason="it is only an owner reference", new_class="none")
    with pytest.raises(CareKeyNodeError, match=r"claims\.owner_ref"):
        privacy({"run_folder": str(tmp_path)})


def test_the_runner_refuses_twice(tmp_path):
    _write(tmp_path, COMPOSITE)
    _resolve(tmp_path, "reclassify")
    built = []
    for _ in range(2):
        runner = LaptopRunner({"graph_name": "g", "run_folder": str(tmp_path)})
        results = runner.run_steps([("privacy", privacy), ("build", lambda c: built.append(1))])
        assert [r.step for r in results] == ["privacy"]
        assert results[0].status == RunStatus.FAILED
        assert "foreign key to patients.patientid" in results[0].error
    assert built == []


# ------------------------------------------------- prover round (independent mutation pass)
CARE_COLS = [{"table": "patients", "column": "patientid", "class": "care_subject"},
             {"table": "encounters", "column": "patientid", "class": "care_subject"}]


def _fk_rec(table, cols, ref_table, *ref_cols):
    return {"table": table, "columns": list(cols), "references": [ref_table, *ref_cols]}


def _check(tmp_path, foreign_keys, table, key, label="title"):
    """check_care_key_nodes on CLEAN + one node, with the given flags.json foreign_keys."""
    _write(tmp_path, with_node("probe", table, key, label))
    m = gate.load_mapping(tmp_path / "graph_mapping.yaml")
    data = {"columns": CARE_COLS, "foreign_keys": foreign_keys}
    return _bounded(lambda: gate.check_care_key_nodes(data, m), 2.0)


CYCLE_WITH_EXIT = [_fk_rec("loop_a", ["b_ref"], "loop_b", "a_ref"),
                   _fk_rec("loop_b", ["a_ref"], "loop_a", "b_ref"),
                   _fk_rec("loop_b", ["a_ref"], "patients", "patientid")]


@pytest.mark.parametrize("order", [CYCLE_WITH_EXIT, CYCLE_WITH_EXIT[::-1],
                                   [CYCLE_WITH_EXIT[0], CYCLE_WITH_EXIT[2], CYCLE_WITH_EXIT[1]]])
def test_a_cycle_with_a_care_exit_is_refused_in_any_fk_order(tmp_path, order):
    """hostile: reordered input. A DFS that stops (or gives up) at the first revisit, or follows
    only one FK of a column, misses the exit for some order."""
    with pytest.raises(CareKeyNodeError, match=r"loop_a\.b_ref \(foreign key to patients\.patientid\)"):
        _check(tmp_path, order, "loop_a", "b_ref")


def test_fk_targets_pair_table_and_column(tmp_path):
    """hostile: the same column name in two tables; only claims.owner_ref points at care."""
    fks = [_fk_rec("claims", ["owner_ref"], "patients", "patientid"),
           _fk_rec("providers", ["owner_ref"], "providers", "provider_id")]
    _check(tmp_path, fks, "providers", "owner_ref", "name")
    _check(tmp_path, fks[:1], "providers", "owner_ref", "name")
    with pytest.raises(CareKeyNodeError):
        _check(tmp_path, fks, "claims", "owner_ref", "amount")


def test_a_200_hop_chain_is_refused_in_bounded_time(tmp_path):
    """hostile scale: t0.r -> t1.r -> ... -> t199.r -> patients.patientid, listed in reverse."""
    fks = [_fk_rec(f"t{i}", ["r"], f"t{i + 1}", "r") for i in range(199)]
    fks.append(_fk_rec("t199", ["r"], "patients", "patientid"))
    with pytest.raises(CareKeyNodeError, match=r"t0\.r \(foreign key to patients\.patientid\)"):
        _check(tmp_path, fks[::-1], "t0", "r")


def test_a_composite_fk_pairs_each_column_with_its_referenced_column(tmp_path):
    """hostile: claims(enc_ref, owner_ref) -> encounters(enc_id, patientid); only the SECOND
    column of the FK reaches care."""
    fks = [_fk_rec("claims", ["enc_ref", "owner_ref"], "encounters", "enc_id", "patientid")]
    with pytest.raises(CareKeyNodeError, match=r"claims\.owner_ref \(foreign key to encounters\.patientid\)"):
        _check(tmp_path, fks, "claims", "owner_ref", "amount")
    _check(tmp_path, fks, "claims", "enc_ref", "amount")


def test_a_same_name_fk_still_names_the_referenced_care_column(tmp_path):
    """hostile: claims.patientid (class none here) -> patients.patientid: same column name."""
    fks = [_fk_rec("claims", ["patientid"], "patients", "patientid")]
    with pytest.raises(CareKeyNodeError, match=r"claims\.patientid \(foreign key to patients\.patientid\)"):
        _check(tmp_path, fks, "claims", "patientid", "amount")


def test_a_self_referencing_fk_chain_is_followed(tmp_path):
    fks = [_fk_rec("staff", ["mentor_ref"], "staff", "patient_ref"),
           _fk_rec("staff", ["patient_ref"], "patients", "patientid")]
    with pytest.raises(CareKeyNodeError, match=r"staff\.mentor_ref"):
        _check(tmp_path, fks, "staff", "mentor_ref")


def test_the_record_tolerates_null_fk_fields_and_tables_without_keys():
    tables = [{"name": "x", "columns": [], "foreign_keys": [{"columns": None, "references": None}]},
              {"name": "claims", "columns": [],  # no key_columns at all
               "foreign_keys": [{"columns": ["owner_ref"], "references": ["patients", "patientid"]}]},
              {"name": "y", "columns": [], "foreign_keys": None}]
    assert gate.foreign_keys_record(tables) == [
        {"table": "x", "columns": [], "references": []},
        {"table": "claims", "columns": ["owner_ref"], "references": ["patients", "patientid"]}]


def test_malformed_fk_records_are_refused_by_the_gate_not_crashed_on(tmp_path):
    """noise: foreign_keys null, or an entry with no table, in a forged flags.json: the
    care check does not crash, and require_clear refuses with PrivacyGateError (the pin)."""
    for i, bad in enumerate((None, [{"columns": ["owner_ref"], "references": ["patients", "patientid"]}])):
        run = tmp_path / str(i)
        _check(run, bad, "claims", "claim_id", "amount")
        _cleared_then_forged(run, CLEAN, "drop")
        data = {k: v for k, v in read_flags_file(str(run)).items() if k != gate.STAMP_KEY}
        data["foreign_keys"] = bad
        forge(str(run), data)
        with pytest.raises(PrivacyGateError, match="foreign keys differ"):
            gate.require_clear(str(run))


def _forge_fks(run_dir, edit):
    data = {k: v for k, v in read_flags_file(str(run_dir)).items() if k != gate.STAMP_KEY}
    edit(data)
    forge(str(run_dir), data)


def test_the_pin_refuses_a_redirected_reference(tmp_path):
    """hostile: same FK count and columns, owner_ref redirected to providers.provider_id."""
    _cleared_then_forged(tmp_path, OWNER, "drop")

    def redirect(data):
        for fk in data["foreign_keys"]:
            if fk["columns"] == ["owner_ref"]:
                fk["references"] = ["providers", "provider_id"]
    _forge_fks(tmp_path, redirect)
    with pytest.raises(PrivacyGateError, match="foreign keys differ"):
        gate.require_clear(str(tmp_path))


def test_the_pin_refuses_a_deleted_fk_record(tmp_path):
    """hostile: an older or hand-edited flags.json with no foreign_keys key at all."""
    _cleared_then_forged(tmp_path, OWNER, "drop")
    _forge_fks(tmp_path, lambda data: data.pop("foreign_keys"))
    with pytest.raises(PrivacyGateError, match="foreign keys differ"):
        gate.require_clear(str(tmp_path))


# --------------------------- owner decision (a): FK to ANY column of a care-keyed table; case
CARE_KEYS = [{"table": "patients", "columns": ["patientid"]},
             # hostile: the care column is NOT first in the composite key
             {"table": "encounters", "columns": ["visit_no", "patientid"]},
             {"table": "vendors", "columns": ["vendor_id"]}]


def _check_keyed(tmp_path, foreign_keys, table, key, columns=CARE_COLS, keys=CARE_KEYS):
    _write(tmp_path, with_node("probe", table, key, "title"))
    m = gate.load_mapping(tmp_path / "graph_mapping.yaml")
    data = {"columns": columns, "foreign_keys": foreign_keys, "key_columns": keys}
    return _bounded(lambda: gate.check_care_key_nodes(data, m), 2.0)


def test_an_fk_to_a_surrogate_of_a_care_keyed_table_is_refused(tmp_path):
    """hostile: claims.owner_row -> patients.row_id; row_id is class none, but patients is keyed
    on the care subject, so row_id names one patient."""
    fks = [_fk_rec("claims", ["owner_row"], "patients", "row_id")]
    with pytest.raises(CareKeyNodeError,
                       match=r"claims\.owner_row \(foreign key to patients\.row_id, a column of "
                             r"patients, keyed on care-subject patients\.patientid\)"):
        _check_keyed(tmp_path, fks, "claims", "owner_row")


def test_an_fk_to_a_surrogate_of_a_non_care_table_passes(tmp_path):
    fks = [_fk_rec("claims", ["vendor_row"], "vendors", "row_id")]
    _check_keyed(tmp_path, fks, "claims", "vendor_row")


def test_a_chain_ending_on_a_non_key_column_of_a_care_keyed_table_is_refused(tmp_path):
    """hostile: two hops, the last onto patients.city (neither key nor care)."""
    fks = [_fk_rec("visits", ["note_ref"], "notes", "row_ref"),
           _fk_rec("notes", ["row_ref"], "patients", "city")]
    with pytest.raises(CareKeyNodeError, match=r"visits\.note_ref \(foreign key to patients\.city"):
        _check_keyed(tmp_path, fks, "visits", "note_ref")


def test_an_fk_to_the_non_care_part_of_a_composite_care_key_is_refused(tmp_path):
    """hostile: encounters is keyed on (patientid, visit_no); an FK to visit_no alone."""
    fks = [_fk_rec("claims", ["visit_ref"], "encounters", "visit_no")]
    with pytest.raises(CareKeyNodeError, match=r"encounters\.patientid"):
        _check_keyed(tmp_path, fks, "claims", "visit_ref")


MIXED = [_fk_rec("Claims", ["Owner_Ref"], "Patients", "PatientID")]


def test_fk_map_matches_mixed_case_names(tmp_path):
    """hostile: a hand-written FK in mixed case against lower-case schema/mapping names."""
    with pytest.raises(CareKeyNodeError, match=r"claims\.owner_ref \(foreign key to patients\.patientid\)"):
        _check_keyed(tmp_path, MIXED, "claims", "owner_ref")


def test_the_walk_matches_mixed_case_care_columns_and_keys(tmp_path):
    """hostile: schema.json names the care table in mixed case; the FK is lower case."""
    cols = [{"table": "Patients", "column": "PatientID", "class": "care_subject"}]
    keys = [{"table": "Patients", "columns": ["PatientID"]}]
    with pytest.raises(CareKeyNodeError, match=r"claims\.owner_ref \(foreign key to patients\.patientid\)"):
        _check_keyed(tmp_path, [_fk_rec("claims", ["owner_ref"], "patients", "patientid")],
                     "claims", "owner_ref", cols, keys)
    with pytest.raises(CareKeyNodeError, match=r"keyed on care-subject patients\.patientid"):
        _check_keyed(tmp_path, [_fk_rec("claims", ["owner_row"], "patients", "row_id")],
                     "claims", "owner_row", cols, keys)


SURROGATE_SCHEMA = json.loads(json.dumps(SCHEMA))
next(t for t in SURROGATE_SCHEMA["tables"] if t["name"] == "patients")["columns"].append(_col("row_id"))
_claims = next(t for t in SURROGATE_SCHEMA["tables"] if t["name"] == "claims")
_claims["columns"].append(_col("owner_row"))
_claims["foreign_keys"].append(_fk(["owner_row"], "patients", "row_id"))
OWNER_ROW = with_node("owner", "claims", "owner_row", "amount")


@pytest.mark.parametrize("choice", ["drop", "reclassify"])
def test_the_step_and_require_clear_refuse_a_surrogate_fk_node(tmp_path, monkeypatch, choice):
    from tests.test_build_sql import _Warehouse, run_build

    _write(tmp_path, OWNER_ROW, SURROGATE_SCHEMA)
    _resolve(tmp_path, choice)
    with pytest.raises(CareKeyNodeError, match=r"claims\.owner_row \(foreign key to patients\.row_id"):
        privacy({"run_folder": str(tmp_path)})
    _write(tmp_path, CLEAN, SURROGATE_SCHEMA)
    privacy({"run_folder": str(tmp_path)})
    (tmp_path / "graph_mapping.yaml").write_text(yaml.safe_dump(OWNER_ROW, sort_keys=False))
    _forge_fks(tmp_path, lambda data: None)
    with pytest.raises(CareKeyNodeError, match=r"claims\.owner_row"):
        gate.require_clear(str(tmp_path))
    wh = _Warehouse()
    with pytest.raises(CareKeyNodeError):
        run_build(tmp_path, monkeypatch, wh)
    assert wh.statements == []


def test_the_privacy_step_records_key_columns_and_the_pin_refuses_a_removed_key(tmp_path):
    _write(tmp_path, CLEAN, SURROGATE_SCHEMA)
    _resolve(tmp_path, "drop")
    privacy({"run_folder": str(tmp_path)})
    assert {"table": "patients", "columns": ["patientid"]} in read_flags_file(str(tmp_path))["key_columns"]
    _forge_fks(tmp_path, lambda data: data.update(
        key_columns=[k for k in data["key_columns"] if k["table"] != "patients"]))
    with pytest.raises(PrivacyGateError, match="key columns differ"):
        gate.require_clear(str(tmp_path))


def test_the_pins_ignore_case_only_differences(tmp_path):
    """Case-insensitive pin: re-casing the recorded FKs and keys is not a different decision."""
    _write(tmp_path, CLEAN, SURROGATE_SCHEMA)
    _resolve(tmp_path, "drop")
    privacy({"run_folder": str(tmp_path)})

    def recase(data):
        for fk in data["foreign_keys"]:
            fk.update(table=fk["table"].upper(), columns=[c.upper() for c in fk["columns"]],
                      references=[r.upper() for r in fk["references"]])
        for k in data["key_columns"]:
            k.update(table=k["table"].title(), columns=[c.upper() for c in k["columns"]])
    _forge_fks(tmp_path, recase)
    assert gate.require_clear(str(tmp_path))


def test_the_walk_case_folds_its_start_column():
    """The mapping contract is lower case, so a mixed-case start reaches the walk only from a
    caller that bypasses load_mapping; the walk still folds it (defence in depth)."""
    care, keyed = gate._care_sets({"columns": CARE_COLS, "key_columns": CARE_KEYS})
    targets = gate._fk_targets({"foreign_keys": [_fk_rec("claims", ["owner_row"], "patients", "row_id")]})
    assert gate._care_reached(("Claims", "Owner_Row"), care, keyed, targets) == (
        ("patients", "row_id"), ("patients", "patientid"))
    assert gate._care_reached(("Patients", "PatientID"), care, keyed, targets) == (
        ("patients", "patientid"), ("patients", "patientid"))


# ------------------------------------------------------------------ prover round 2 gaps
def test_a_chain_through_a_care_keyed_table_that_continues_is_refused(tmp_path):
    """hostile: the walk reaches patients.row_id, which itself has an FK onward to vendors."""
    fks = [_fk_rec("claims", ["owner_row"], "patients", "row_id"),
           _fk_rec("patients", ["row_id"], "vendors", "vendor_id")]
    with pytest.raises(CareKeyNodeError, match=r"keyed on care-subject patients\.patientid"):
        _check_keyed(tmp_path, fks, "claims", "owner_row")


def _repoint_patients_key(data):
    for k in data["key_columns"]:
        if k["table"] == "patients":
            k["columns"] = ["city"]


def test_the_pin_refuses_repointed_key_columns(tmp_path):
    """hostile: same tables, but patients' key moved off the care column (forged, re-stamped)."""
    _write(tmp_path, CLEAN, SURROGATE_SCHEMA)
    _resolve(tmp_path, "drop")
    privacy({"run_folder": str(tmp_path)})
    (tmp_path / "graph_mapping.yaml").write_text(yaml.safe_dump(OWNER_ROW, sort_keys=False))
    _forge_fks(tmp_path, _repoint_patients_key)
    with pytest.raises(PrivacyGateError, match="key columns differ"):
        gate.require_clear(str(tmp_path))


def test_the_step_rederives_key_columns_from_schema(tmp_path):
    """A forged key_columns in the previous flags.json never stands in for schema.json's."""
    _write(tmp_path, CLEAN, SURROGATE_SCHEMA)
    _resolve(tmp_path, "drop")
    privacy({"run_folder": str(tmp_path)})
    _forge_fks(tmp_path, _repoint_patients_key)
    (tmp_path / "graph_mapping.yaml").write_text(yaml.safe_dump(OWNER_ROW, sort_keys=False))
    with pytest.raises(CareKeyNodeError, match=r"claims\.owner_row"):
        privacy({"run_folder": str(tmp_path)})


def test_names_are_case_folded_not_just_lowered(tmp_path):
    """noise: non-ASCII names; casefold('Straße') == casefold('STRASSE'), lower() differs."""
    data = {"columns": [{"table": "Straße", "column": "pid", "class": "care_subject"}],
            "key_columns": [{"table": "Straße", "columns": ["pid"]}]}
    care, keyed = gate._care_sets(data)
    targets = gate._fk_targets({"foreign_keys": [_fk_rec("claims", ["x"], "STRASSE", "row_id")]})
    assert gate._care_reached(("claims", "x"), care, keyed, targets) is not None
    schema = {"tables": [{"name": "straße", "key_columns": ["pid"], "columns": [],
                          "foreign_keys": [{"columns": ["x"], "references": ["straße", "pid"]}]}]}
    flags = {"foreign_keys": [{"table": "STRASSE", "columns": ["x"], "references": ["STRASSE", "pid"]}],
             "key_columns": [{"table": "STRASSE", "columns": ["pid"]}]}
    gate._check_structure_pin(flags, schema["tables"])
