"""Owner decision 2 (2026-10-05, "a"): no going back to an older privacy decision without the privacy step.

The decision is the run folder's flags.json + graph_mapping.yaml (the stamp is derived from their
bytes, with schema.json). Restoring an exact older copy made an older, looser decision "current"
again. Now the privacy step records every decision it passes in privacy_decisions.json, newest
last, and the gate (require_clear, which every enforcing step calls) refuses a decision that is
not the newest recorded one (OlderDecisionError) or that was never recorded
(UnrecordedDecisionError). The way back to an older decision is to run the privacy step again,
which records it as the newest. docs/privacy-decision-history.md states the residual risk.

Hostile fixture: three decisions T(ight) / L(oose) / T, the loose one in the middle, all passed by
the privacy step. Restoring L's exact bytes after the second T is refused, although L once passed.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from lgk.privacy import binding, decisions, gate
from lgk.privacy.flags import OpenFlagError, resolve_flag
from lgk.privacy.gate import OlderDecisionError, UnrecordedDecisionError
from lgk.steps.privacy import privacy
from tests.test_privacy_binding import (  # noqa: F401 - fixtures used by name
    MEMBER_MAPPING, _Conn, _config, _member_schema, _write_mapping, served, wh)
from tests.test_rollback import _build_provision, _nightly_reapply_parts, site  # noqa: F401

DECISION_FILES = ("flags.json", "graph_mapping.yaml")


def _tight_mapping() -> dict:
    tight = json.loads(json.dumps(MEMBER_MAPPING))
    tight["nodes"]["member"]["label"] = "row_key"
    return tight


def _save(run: Path, name: str, files=DECISION_FILES) -> Path:
    keep = run.parent / f"{run.name}_keep_{name}"
    keep.mkdir()
    for f in files:
        shutil.copy2(run / f, keep / f)
    return keep


def _restore(run: Path, keep: Path) -> None:
    for f in keep.iterdir():
        shutil.copy2(f, run / f.name)


def _stamp_of_files(run: Path) -> str:
    """The decision id computed from flags.json alone (independent of the gate's code path)."""
    recorded = json.loads((run / "flags.json").read_text())["gate"]
    return hashlib.sha256(json.dumps(recorded, sort_keys=True).encode()).hexdigest()


def _history_ids(run: Path) -> list:
    return [e["decision"] for e in json.loads((run / "privacy_decisions.json").read_text())["decisions"]]


def _snapshot(run: Path) -> dict:
    return {p.name: p.read_bytes() for p in sorted(run.iterdir()) if p.is_file()}


@pytest.fixture
def tlt(tmp_path, wh, site):
    """T passed (v1 provisioned), L passed (v2), T passed again (v3). Returns (run, saved dirs, ids)."""
    run = tmp_path / "run"
    run.mkdir()
    _member_schema(run)
    _write_mapping(run, _tight_mapping())
    with pytest.raises(OpenFlagError):
        privacy({"run_folder": str(run)})
    resolve_flag(str(run), "member_notes", "patient_name", "drop", chooser="op")
    privacy({"run_folder": str(run)})
    t_id = _stamp_of_files(run)
    tight = _save(run, "tight")
    _build_provision(run, 1)

    resolve_flag(str(run), "member_notes", "patient_name", "reclassify", chooser="op",
                 reason="display name of a care team, not a patient", new_class="none")
    _write_mapping(run, MEMBER_MAPPING)
    privacy({"run_folder": str(run)})
    l_id = _stamp_of_files(run)
    loose = _save(run, "loose")
    loose_history = _save(run, "loose_history", ("privacy_decisions.json",))
    _build_provision(run, 2)

    _restore(run, tight)
    privacy({"run_folder": str(run)})  # T again, through the step: the newest decision
    assert _stamp_of_files(run) == t_id != l_id  # the fixture really has two decisions
    _build_provision(run, 3)
    assert _history_ids(run) == [t_id, l_id, t_id]
    return run, {"tight": tight, "loose": loose, "loose_history": loose_history}, {"T": t_id, "L": l_id}


# ------------------------------------------------------------------------ the record
def test_the_privacy_step_records_each_decision_newest_last(tlt):
    run, _saved, ids = tlt
    entries = json.loads((run / "privacy_decisions.json").read_text())["decisions"]
    assert [e["seq"] for e in entries] == [1, 2, 3]
    assert [e["decision"] for e in entries] == [ids["T"], ids["L"], ids["T"]]


def test_a_privacy_rerun_with_the_same_inputs_adds_no_entry(tlt):
    run, _saved, ids = tlt
    privacy({"run_folder": str(run)})
    assert _history_ids(run) == [ids["T"], ids["L"], ids["T"]]


def test_the_gate_never_writes_the_history(tlt):
    run, saved, _ids = tlt
    before = (run / "privacy_decisions.json").read_bytes()
    gate.require_clear(str(run))
    _restore(run, saved["loose"])
    with pytest.raises(OlderDecisionError):
        gate.require_clear(str(run))
    assert (run / "privacy_decisions.json").read_bytes() == before


def test_only_the_privacy_step_records_a_decision():
    # the history is written by the privacy step alone: no other module calls the writer
    src = Path(__file__).resolve().parents[1] / "src" / "lgk"
    writers = sorted(str(p.relative_to(src)) for p in src.rglob("*.py")
                     if "record_decision(" in p.read_text() and p.name != "decisions.py")
    assert writers == ["steps/privacy.py"]


# ------------------------------------------------------------- restoring older bytes
def test_restoring_the_loose_decision_after_the_second_tight_one_is_refused(tlt):
    run, saved, ids = tlt
    _restore(run, saved["loose"])
    assert _stamp_of_files(run) == ids["L"]  # the exact older bytes: the stamp itself is fresh
    with pytest.raises(OlderDecisionError, match=rf"older privacy decision: decision 2 of 3 \({ids['L'][:12]}, recorded .*the newest is decision 3 \({ids['T'][:12]}\)"):
        gate.require_clear(str(run))


def _enforcing_steps():
    """Every step that calls require_clear / stamp_id, as one call each with the run's config."""
    from lgk.app.generate import walk_queries
    from lgk.privacy.gate import served_walk_policy, served_walk_selection
    from lgk.refresh import sync as sync_mod
    from lgk.refresh.operator_stamp import operator_stamp, stamp_operator_tables
    from lgk.refresh.tasks import run_task
    from lgk.served_functions import served_functions
    from lgk.steps.build import build
    from lgk.steps.enrich import enrich
    from lgk.steps.provision import provision
    from lgk.steps.reapply import reapply
    from lgk.steps.switch import rollback, switch
    from lgk.steps.walks import get_walk_query

    def sync(cfg, wh):
        sync_mod._require_stamped_tables(wh, cfg, object())

    return {
        "build": lambda c, wh: build(c),
        "enrich": lambda c, wh: enrich(c),
        "provision": lambda c, wh: provision(c),
        "reapply": lambda c, wh: reapply(c),
        "reapply_nightly_task": lambda c, wh: run_task("reapply", json.dumps(c)),
        "switch": lambda c, wh: switch(c, 1),
        "rollback": lambda c, wh: rollback(c, 1),
        "refresh_sync": sync,
        "stamp_step": lambda c, wh: stamp_operator_tables(c, sql=wh, w=object()),
        "operator_stamp": lambda c, wh: operator_stamp(c),
        "app_generate": lambda c, wh: walk_queries(c),
        "served_functions": lambda c, wh: served_functions(c, "kit_v1"),
        "walks": lambda c, wh: get_walk_query("", depth=2, served_schema="kit", run_folder=c["run_folder"]),
        "served_walk_selection": lambda c, wh: served_walk_selection(c["run_folder"], None, None),
        "served_walk_policy": lambda c, wh: served_walk_policy(c["run_folder"], None, None),
    }


STEPS = sorted(_enforcing_steps())


@pytest.mark.parametrize("step", STEPS)
def test_every_enforcing_step_refuses_the_restored_loose_decision(tlt, wh, served, site, monkeypatch, step):
    run, saved, ids = tlt
    monkeypatch.setattr("lgk.steps.enrich.execute_sql", lambda w, wid, s: wh(s))
    _nightly_reapply_parts(monkeypatch)
    _restore(run, saved["loose"])
    before_files, before_sql, before_calls, before_moves = _snapshot(run), list(wh.statements), list(served), list(site.moves)
    with pytest.raises(OlderDecisionError, match=ids["L"][:12]):
        _enforcing_steps()[step](_config(run, version=1), wh)
    assert _snapshot(run) == before_files  # nothing written to the run folder
    assert wh.statements == before_sql  # no warehouse statement
    assert served == before_calls and site.moves == before_moves  # no sync, view or grant


@pytest.mark.parametrize("step", STEPS)
def test_every_enforcing_step_passes_the_gate_on_the_newest_decision(tlt, wh, served, site, monkeypatch, step):
    # control: on the newest decision the history check is not what stops any step (any later
    # refusal is the step's own business, never the history's)
    run, _saved, _ids = tlt
    monkeypatch.setattr("lgk.steps.enrich.execute_sql", lambda w, wid, s: wh(s))
    _nightly_reapply_parts(monkeypatch)
    try:
        _enforcing_steps()[step](_config(run, version=1), wh)
    except (OlderDecisionError, UnrecordedDecisionError):
        raise
    except Exception:
        pass


def test_rollback_still_serves_a_version_of_the_current_decision_and_refuses_the_loose_one(tlt, site):
    from lgk.steps.switch import rollback, switch

    run, _saved, _ids = tlt
    switch(_config(run, version=3))
    with pytest.raises(binding.StaleBuildError, match="kit_v2 was provisioned under a different"):
        rollback(_config(run, version=3), 2)
    assert rollback(_config(run, version=3), 1) == "kit_v1"  # v1: the same (current) decision T


# ------------------------------------------------------------- the way back: re-run privacy
def test_rerunning_privacy_with_the_older_files_makes_them_newest_and_servable(tlt, site):
    from lgk.steps.switch import rollback

    run, saved, ids = tlt
    _restore(run, saved["loose"])
    with pytest.raises(OlderDecisionError):
        rollback(_config(run, version=3), 2)
    privacy({"run_folder": str(run)})  # the operator re-runs the privacy step on the L files
    assert _history_ids(run) == [ids["T"], ids["L"], ids["T"], ids["L"]]
    assert binding.stamp_id(gate.require_clear(str(run))) == ids["L"]
    assert rollback(_config(run, version=3), 2) == "kit_v2"  # check_version: v2 was made under L
    with pytest.raises(binding.StaleBuildError, match="kit_v3 was provisioned under a different"):
        rollback(_config(run, version=3), 3)  # and the T versions are now the other decision


# --------------------------------------------------------------- missing or edited history
def test_a_missing_history_refuses_until_the_privacy_step_runs(tlt):
    run, _saved, ids = tlt
    (run / "privacy_decisions.json").unlink()
    with pytest.raises(UnrecordedDecisionError, match="privacy_decisions.json not found"):
        gate.require_clear(str(run))
    privacy({"run_folder": str(run)})  # the step creates it, with the current decision
    assert _history_ids(run) == [ids["T"]]
    gate.require_clear(str(run))


def test_a_hand_stamped_decision_the_step_never_recorded_is_refused(tlt):
    # hostile: flags.json re-stamped by hand with gate.stamp (fresh hashes, every check passes)
    run, _saved, _ids = tlt
    data = json.loads((run / "flags.json").read_text())
    data["note"] = "edited by hand"
    (run / "flags.json").write_text(json.dumps(gate.stamp(str(run), {k: v for k, v in data.items() if k != "gate"})))
    with pytest.raises(UnrecordedDecisionError, match="not recorded by the privacy step"):
        gate.require_clear(str(run))


@pytest.mark.parametrize("body", ["", "{", "[]", '{"decisions": []}', '{"decisions": {}}',
                                  '{"decisions": [{"seq": 1}]}', '{"decisions": [1]}', '{"decisions": [["x"]]}',
                                  '{"decisions": [{"seq": 1, "decision": 7}]}',
                                  '{"decisions": [{"seq": 1, "decision": ""}]}',
                                  # hostile: the CURRENT id, so only the format check can refuse these
                                  '{"decisions": [{"seq": true, "decision": "<T>"}]}',
                                  '{"decisions": [{"seq": 2, "decision": "<T>"}]}',
                                  '{"decisions": [{"seq": "1", "decision": "<T>"}]}'])
def test_a_malformed_history_is_refused_not_read_as_allow(tlt, body):
    run, _saved, ids = tlt
    (run / "privacy_decisions.json").write_text(body.replace("<T>", ids["T"]))
    with pytest.raises(UnrecordedDecisionError, match="is malformed"):
        gate.require_clear(str(run))


def test_the_privacy_step_never_replaces_a_malformed_history(tlt):
    # hostile: a damaged record is never silently restarted (that would forget the newest decision)
    run, _saved, _ids = tlt
    path = run / "privacy_decisions.json"
    path.write_text('{"decisions": [{"seq": 1}]}')
    with pytest.raises(decisions.HistoryFormatError):
        privacy({"run_folder": str(run)})
    assert path.read_bytes() == b'{"decisions": [{"seq": 1}]}'


def test_a_failed_stamp_write_records_no_decision(tlt, monkeypatch):
    # the decision is recorded only after flags.json holds its stamp (fail closed)
    import lgk.steps.privacy as step

    run, _saved, ids = tlt
    real = step.write_flags_file

    def write(run_folder, data):
        if "gate" in data:
            raise OSError("disk full")
        real(run_folder, data)

    resolve_flag(str(run), "member_notes", "patient_name", "reclassify", chooser="op",
                 reason="another decision", new_class="none")
    monkeypatch.setattr(step, "write_flags_file", write)
    with pytest.raises(OSError, match="disk full"):
        privacy({"run_folder": str(run)})
    assert _history_ids(run) == [ids["T"], ids["L"], ids["T"]]


def test_the_refusal_names_the_latest_entry_of_a_repeated_older_decision(tlt):
    # hostile: L recorded twice (seq 1 and 3), then a newer one; restoring L names seq 3, not 1
    run, saved, ids = tlt
    entries = [{"seq": i + 1, "decision": d, "at": f"t{i + 1}"} for i, d in
               enumerate([ids["L"], ids["T"], ids["L"], "f" * 64])]
    (run / "privacy_decisions.json").write_text(json.dumps({"decisions": entries}))
    _restore(run, saved["loose"])
    with pytest.raises(OlderDecisionError, match=r"decision 3 of 4 \(" + ids["L"][:12] + r", recorded t3\)"):
        gate.require_clear(str(run))


def test_an_older_decision_moved_to_the_end_by_editing_the_history_is_refused_by_the_seq(tlt):
    # hostile: L's entry hand-moved last; the seq numbers no longer rise, so the edit is visible
    run, saved, ids = tlt
    path = run / "privacy_decisions.json"
    record = json.loads(path.read_text())
    entries = record["decisions"]
    record["decisions"] = [entries[0], entries[2], entries[1]]
    path.write_text(json.dumps(record))
    _restore(run, saved["loose"])
    with pytest.raises(UnrecordedDecisionError, match="seq"):
        gate.require_clear(str(run))


# --------------------------------------------------------------- the residual risk, stated
def test_known_limit_restoring_the_older_history_too_is_not_detected(tlt, site):
    """KNOWN LIMIT (docs/privacy-decision-history.md): the run folder is the only record.

    Whoever restores flags.json, graph_mapping.yaml AND privacy_decisions.json from the time L
    was newest makes L current again, and v2 (made under L) servable. This test asserts that the
    limit is what the kit does today, so a change in it is seen, not silently passed."""
    from lgk.steps.switch import rollback

    run, saved, ids = tlt
    _restore(run, saved["loose"])
    _restore(run, saved["loose_history"])
    assert _history_ids(run) == [ids["T"], ids["L"]]
    assert binding.stamp_id(gate.require_clear(str(run))) == ids["L"]  # not refused: the limit
    assert rollback(_config(run, version=3), 2) == "kit_v2"


def test_the_residual_risk_doc_names_the_limit():
    doc = (Path(__file__).resolve().parents[1] / "docs" / "privacy-decision-history.md").read_text()
    for claim in ("privacy_decisions.json", "OlderDecisionError", "UnrecordedDecisionError",
                  "test_known_limit_restoring_the_older_history_too_is_not_detected"):
        assert claim in doc


def test_decision_ids_are_the_binding_stamp_ids(tlt):
    run, _saved, ids = tlt
    assert decisions.decision_id(json.loads((run / "flags.json").read_text())["gate"]) == ids["T"]
    assert binding.stamp_id(gate.require_clear(str(run))) == ids["T"]


def test_the_gate_refuses_before_any_other_check_reads_on(tlt, monkeypatch):
    # the history check sits right after the stamp check: a later check never runs on L
    run, saved, _ids = tlt
    _restore(run, saved["loose"])
    ran = []
    monkeypatch.setattr(gate, "check_mapping", lambda *a, **k: ran.append("check_mapping"))
    with pytest.raises(OlderDecisionError):
        gate.require_clear(str(run))
    assert ran == []
