"""The operator rollback recipe (SKILL.md "Roll back to an earlier version"), offline.

docs/rollback-case.md: a direct `switch(config, <previous version>)` is the rollback, and the
nightly `reapply` (frozen job config, `version` = the newer one) must not undo it overnight.

Hostile fixture: THREE versions under decisions T, L, T. v2 (the middle one, the newest earlier
version in versions.json) was made under the looser decision L; the current decision is T again
(the operator restored the tight files byte for byte AND ran the privacy step again: since
P6-decision a restore alone is refused, tests/test_privacy_decision_history.py). So the naive "the entry before the last"
is v2, and the switch must refuse it; v1 is the version the rollback may serve.

The served views are a stateful fake: switch_views moves them, live_version_schemas reads them
(the pg_depend reader is tested against Postgres in tests/test_versions.py).
"""

import json
import shutil
from pathlib import Path

import pytest

from lgk.privacy.binding import StaleBuildError
from lgk.runner import LaptopRunner, RunStatus
from tests.test_privacy_binding import (  # noqa: F401 - fixtures used by name
    _Conn, _config, _current, _loose_decision, _tighten, served, wh)

DECISION_FILES = ("flags.json", "graph_mapping.yaml")


class _Site:
    """What the served views read now, moved only by the one-transaction switch."""

    def __init__(self):
        self.live = None
        self.moves = []


@pytest.fixture
def site(monkeypatch, served):
    s = _Site()

    def switch_views(conn, served_schema, target, *a, **k):
        s.live = target
        s.moves.append(target)

    monkeypatch.setattr("lgk.versions.switch_views", switch_views)
    monkeypatch.setattr("lgk.versions.live_version_schemas",
                        lambda conn, served_schema: set() if s.live is None else {s.live})
    return s


def _save(run: Path, name: str) -> Path:
    keep = run / f"_decision_{name}"
    keep.mkdir()
    for f in DECISION_FILES:
        shutil.copy2(run / f, keep / f)
    return keep


def _restore(run: Path, keep: Path) -> None:
    """Restore a decision's files, then run the privacy step on them (the only way back to it)."""
    from lgk.steps.privacy import privacy

    for f in DECISION_FILES:
        shutil.copy2(keep / f, run / f)
    privacy({"run_folder": str(run)})


def _build_provision(run: Path, version: int) -> None:
    from lgk.steps.build import build
    from lgk.steps.provision import provision

    build(_config(run, version=version))
    provision(_config(run, version=version))


@pytest.fixture
def three_versions(tmp_path, wh, site):
    """v1 under T (served, then replaced), v2 under looser L (served), v3 under T (served now)."""
    from lgk.steps.switch import switch

    _loose_decision(tmp_path)
    loose = _save(tmp_path, "loose")
    _tighten(tmp_path)
    tight = _save(tmp_path, "tight")
    t_stamp = _current(tmp_path)

    _build_provision(tmp_path, 1)
    switch(_config(tmp_path, version=1))
    _restore(tmp_path, loose)
    l_stamp = _current(tmp_path)
    _build_provision(tmp_path, 2)
    switch(_config(tmp_path, version=2))
    _restore(tmp_path, tight)
    assert _current(tmp_path) == t_stamp != l_stamp  # the fixture really has two decisions
    _build_provision(tmp_path, 3)
    switch(_config(tmp_path, version=3))
    assert site.live == "kit_v3"
    return tmp_path


def _history(run: Path) -> list:
    return [h["live_version_schema"] for h in json.loads((run / "versions.json").read_text())["history"]]


def _nightly_reapply_parts(monkeypatch):
    """reapply's live parts recorded; its served-view step and the new guard run as shipped."""
    from lgk.steps import reapply as rp

    calls = []
    for name in ("check_freshness", "_detect_unfloored_rows", "_reapply_indexes", "_setup_reader_role"):
        monkeypatch.setattr(rp, name, lambda c, _n=name: calls.append(_n))
    monkeypatch.setattr(rp, "connect_branch", lambda c: _Conn(), raising=False)
    return calls


# ---------------------------------------------------------------- the recipe's sequence
def test_recipe_rolls_back_to_the_newest_earlier_version_made_under_the_current_decision(three_versions, site):
    from lgk.steps.switch import switch

    run = three_versions
    assert _history(run) == ["kit_v1", "kit_v2", "kit_v3"]
    # the entry before the last is v2: made under the looser decision, so it is refused
    with pytest.raises(StaleBuildError, match="kit_v2 was provisioned under a different"):
        switch(_config(run, version=3), 2)
    assert site.live == "kit_v3"
    from lgk.steps.switch import rollback

    with pytest.raises(StaleBuildError, match="kit_v2 was provisioned under a different"):
        rollback(_config(run, version=3), 2)  # the recipe's call refuses it the same way
    assert site.live == "kit_v3"
    assert rollback(_config(run, version=3), 1) == "kit_v1"
    assert site.live == "kit_v1"
    assert _history(run) == ["kit_v1", "kit_v2", "kit_v3", "kit_v1"]  # the rollback is recorded


def test_nightly_reapply_after_a_rollback_keeps_the_rolled_back_version(three_versions, site, monkeypatch):
    from lgk.steps.reapply import reapply
    from lgk.steps.switch import switch
    from lgk.versions import ServedVersionError

    run = three_versions
    switch(_config(run, version=3), 1)
    calls = _nightly_reapply_parts(monkeypatch)
    moves = list(site.moves)
    job_config = _config(run, version=3)  # the job was created for v3; its config is frozen
    with pytest.raises(ServedVersionError, match=r"read \['kit_v1'\], not kit_v3"):
        reapply(job_config)
    # the same with served_schema set explicitly (a guard gated on that key would skip it)
    with pytest.raises(ServedVersionError, match=r"read \['kit_v1'\], not kit_v3"):
        reapply({**job_config, "served_schema": "kit"})
    assert site.live == "kit_v1"
    assert site.moves == moves  # no view moved
    assert calls == []  # refused before the freshness guard, indexes or grants ran


def test_nightly_reapply_serving_its_own_version_still_reapplies(three_versions, site, monkeypatch):
    # control: no rollback, so the guard passes and reapply re-serves v3 as before
    from lgk.steps.reapply import reapply

    calls = _nightly_reapply_parts(monkeypatch)
    reapply(_config(three_versions, version=3))
    assert site.live == "kit_v3" and site.moves[-1] == "kit_v3"
    assert calls == ["check_freshness", "_detect_unfloored_rows", "_reapply_indexes", "_setup_reader_role"]


def test_reapply_refuses_a_site_switched_forward_past_the_job_version(three_versions, site, monkeypatch):
    # the same guard: a job made for v1 must not pull a site on v3 back to v1
    from lgk.steps.reapply import reapply
    from lgk.versions import ServedVersionError

    _nightly_reapply_parts(monkeypatch)
    with pytest.raises(ServedVersionError, match="not kit_v1"):
        reapply(_config(three_versions, version=1))
    assert site.live == "kit_v3"


# ---------------------------------------------------------------- the refusal, every path
def test_rollback_to_the_looser_version_is_refused_through_the_runner_too(three_versions, site):
    from lgk.steps.switch import switch

    run = three_versions
    (run / "run.json").unlink(missing_ok=True)  # a fresh run.json: the step really runs
    runner = LaptopRunner(_config(run, version=3))
    result = runner.run_step("rollback", lambda c: switch(c, 2))
    assert result.status == RunStatus.FAILED and "kit_v2 was provisioned under a different" in result.error
    assert site.live == "kit_v3"


def test_rollback_by_editing_config_version_is_refused(three_versions, site):
    from lgk.steps.switch import switch

    with pytest.raises(StaleBuildError, match="kit_v2 was provisioned under a different"):
        switch(_config(three_versions, version=2))  # no explicit version: config["version"] edited
    assert site.live == "kit_v3"


def test_reapply_before_any_switch_still_serves_the_job_version(tmp_path, wh, site, monkeypatch):
    # no served views yet: nothing to undo, so the guard passes (the job's first reapply)
    from lgk.steps.reapply import reapply

    _loose_decision(tmp_path)
    _build_provision(tmp_path, 1)
    _nightly_reapply_parts(monkeypatch)
    reapply(_config(tmp_path, version=1))
    assert site.live == "kit_v1"


class _ClosingConn(_Conn):
    closed = False

    def close(self):
        self.closed = True


@pytest.mark.parametrize("live", [{"kit_v1"}, {"other_v3"}, {"kit_old_v3"}])
def test_the_guard_refuses_any_other_schema_and_closes_its_connection(monkeypatch, live):
    # hostile: other_v3 / kit_old_v3 share the target's _v3 suffix but are not kit_v3
    from lgk import versions
    from lgk.steps import reapply as rp

    conn = _ClosingConn()
    monkeypatch.setattr(rp, "connect_branch", lambda c: conn)
    monkeypatch.setattr(versions, "live_version_schemas", lambda c, served_schema: set(live))
    cfg = {"graph_name": "g", "kit_schema": "kit", "version": 3}
    with pytest.raises(versions.ServedVersionError, match="not kit_v3"):
        rp._refuse_a_moved_site(cfg)
    assert conn.closed
    passing = _ClosingConn()
    monkeypatch.setattr(rp, "connect_branch", lambda c: passing)
    monkeypatch.setattr(versions, "live_version_schemas", lambda c, served_schema: {"kit_v3"})
    rp._refuse_a_moved_site(cfg)  # control: the job's own version passes
    assert passing.closed


def test_the_guard_reads_the_served_schema_and_refuses_a_mixed_site(monkeypatch):
    # hostile: served_schema differs from kit_schema, and the views read two versions at once
    from lgk import versions
    from lgk.steps import reapply as rp

    asked = []
    monkeypatch.setattr(rp, "connect_branch", lambda c: _Conn())
    monkeypatch.setattr(versions, "live_version_schemas",
                        lambda conn, served_schema: asked.append(served_schema) or {"kit_v1", "kit_v3"})
    with pytest.raises(versions.ServedVersionError, match=r"Served views in site read \['kit_v1', 'kit_v3'\]"):
        rp._refuse_a_moved_site({"graph_name": "g", "kit_schema": "kit", "served_schema": "site", "version": 3})
    assert asked == ["site"]


def test_served_history_lists_every_switch_oldest_first(three_versions, tmp_path_factory):
    from lgk.steps.switch import served_history, switch

    run = three_versions
    switch(_config(run, version=3), 1)
    assert served_history(_config(run, version=3)) == _history(run) == ["kit_v1", "kit_v2", "kit_v3", "kit_v1"]
    empty = tmp_path_factory.mktemp("no_switch")
    assert served_history(_config(empty)) == []  # before the first switch
    with pytest.raises(ValueError, match="run_folder"):
        served_history({"graph_name": "g"})  # never a silent '.' fallback


@pytest.mark.parametrize("version", [None, "1", True, 1.0])
def test_rollback_needs_the_version_number(three_versions, site, version):
    # hostile: switch(config) with no version re-serves config["version"] (the newer one)
    from lgk.steps.switch import rollback

    with pytest.raises(TypeError, match="version number"):
        rollback(_config(three_versions, version=3), version)
    assert site.live == "kit_v3"


def test_rollback_serves_exactly_the_named_version(three_versions, site):
    # hostile: never a silent substitute (e.g. "one before config['version']")
    from lgk.steps.switch import rollback

    run = three_versions
    assert rollback(_config(run, version=3), 1) == "kit_v1" and site.live == "kit_v1"
    assert rollback(_config(run, version=1), 3) == "kit_v3" and site.live == "kit_v3"  # forward again
    assert rollback(_config(run, version=3), 3) == "kit_v3" and site.moves[-1] == "kit_v3"
