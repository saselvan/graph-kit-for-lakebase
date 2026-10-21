"""Owner decision 1 (2026-10-05, "a"): reapply runs the same privacy check as switch.

Before this, only `switch` refused a version not provisioned under the current privacy decision.
The nightly `reapply` re-serves its job's version; on an EMPTY served schema (no switch yet) its
served-version guard has nothing to compare, so it created the views for a version made under an
older decision. Now reapply calls `binding.require_servable` (the function switch calls) first,
before it connects, on every path: a direct call, the nightly task (refresh.tasks.run_task) and
the exported bundle's task (bundle_entry.main).

Hostile fixture: v1 is provisioned under decision L (loose), then the operator tightens to T
(the privacy step ran, so T is the current and newest decision) and nothing has been switched:
the served schema is empty, so the moved-site guard passes and only the new check can refuse.
"""

import json
import shutil
from pathlib import Path

import pytest

from lgk.privacy.binding import StaleBuildError
from tests.test_privacy_binding import (  # noqa: F401 - fixtures used by name
    _Conn, _config, _current, _loose_decision, _tighten, served, wh)
from tests.test_rollback import _build_provision, _nightly_reapply_parts, site  # noqa: F401


@pytest.fixture
def v1_under_loose_then_tightened(tmp_path, wh, site):
    _loose_decision(tmp_path)
    loose = _current(tmp_path)
    _build_provision(tmp_path, 1)  # v1 recorded under L
    _tighten(tmp_path)  # the privacy step ran: T is current and newest
    assert _current(tmp_path) != loose
    assert site.live is None  # nothing switched: the served schema is empty
    return tmp_path


def _direct(cfg):
    from lgk.steps.reapply import reapply

    reapply(cfg)


def _nightly(cfg):
    from lgk.refresh.tasks import run_task

    run_task("reapply", json.dumps(cfg))


def _bundle(cfg):
    """The exported bundle's job task: stage the bundled run files, then run_task("reapply")."""
    from lgk import bundle_entry

    run = Path(cfg["run_folder"])
    run_src = run.parent / f"{run.name}_bundle_run"
    shutil.copytree(run, run_src)
    # an SP bundle: `kit_schema` is the SP's own, and the workshop's schema is named apart from it
    bundled = {k: v for k, v in cfg.items() if k != "run_folder"}
    bundled["workshop_kit_schema"] = "kit_workshop"
    (run_src / "config.json").write_text(json.dumps(bundled))
    run_dir = run.parent / f"{run.name}_sp_run"
    bundle_entry.main(["--src", "unused", "--run-src", str(run_src), "--run-dir", str(run_dir),
                       "--task", "reapply", "--job-id", "42"])


PATHS = {"direct": _direct, "nightly": _nightly, "bundle": _bundle}


@pytest.mark.parametrize("path", sorted(PATHS))
def test_reapply_on_an_empty_served_schema_refuses_a_version_from_an_older_decision(
        v1_under_loose_then_tightened, site, monkeypatch, path):
    calls = _nightly_reapply_parts(monkeypatch)
    connected = []
    from lgk.steps import reapply as rp

    monkeypatch.setattr(rp, "connect_branch", lambda c: connected.append(c) or _Conn())
    with pytest.raises(StaleBuildError, match="kit_v1 was provisioned under a different privacy decision"):
        PATHS[path](_config(v1_under_loose_then_tightened, version=1))
    assert site.live is None and site.moves == []  # no served view was made
    assert calls == []  # no freshness guard, index or grant ran
    assert connected == []  # refused before it connected at all


@pytest.mark.parametrize("path", sorted(PATHS))
def test_reapply_serves_a_version_made_under_the_current_decision(
        v1_under_loose_then_tightened, site, monkeypatch, path):
    # control: v2 provisioned under T (current) on the same run folder is reapplied as before
    run = v1_under_loose_then_tightened
    _build_provision(run, 2)
    calls = _nightly_reapply_parts(monkeypatch)
    PATHS[path](_config(run, version=2))
    assert site.live == "kit_v2"
    assert calls == ["check_freshness", "_detect_unfloored_rows", "_reapply_indexes", "_setup_reader_role"]


def test_reapply_refuses_a_version_never_provisioned_by_this_run(tmp_path, wh, site, monkeypatch):
    # hostile: a job config naming a version with no record at all (not just a different stamp)
    from lgk.steps.reapply import reapply

    _loose_decision(tmp_path)
    _build_provision(tmp_path, 1)
    calls = _nightly_reapply_parts(monkeypatch)
    with pytest.raises(StaleBuildError, match="kit_v7 has no recorded privacy stamp"):
        reapply(_config(tmp_path, version=7))
    assert site.moves == [] and calls == []


def test_switch_and_reapply_call_the_same_check(monkeypatch, tmp_path, wh, site):
    # the one function: replacing binding.require_servable makes both refuse the same way
    from lgk.privacy import binding
    from lgk.steps.reapply import reapply
    from lgk.steps.switch import switch

    _loose_decision(tmp_path)
    _build_provision(tmp_path, 1)
    _nightly_reapply_parts(monkeypatch)
    seen = []

    def refuse(run_folder, version_schema):
        seen.append(version_schema)
        raise StaleBuildError("sentinel")

    monkeypatch.setattr(binding, "require_servable", refuse)
    for step in (switch, reapply):
        with pytest.raises(StaleBuildError, match="sentinel"):
            step(_config(tmp_path, version=1))
    assert seen == ["kit_v1", "kit_v1"] and site.moves == []
