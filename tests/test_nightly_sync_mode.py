"""Red tests (owner 2026-10-14): the nightly sync refuses every sync_mode but snapshot.

What this protects: v1 serves only snapshot synced tables. The nightly sync task
(`lgk.refresh.sync`) reads the mode from config, so an operator who edits a provisioned config to
"triggered" must be refused before any update starts, with the reason, never synced as TRIGGERED.

Contract under test:
- `sync(config, full_refresh)` and `sync_all(w, sql, config, full_refresh)` call
  `modes.require_v1_sync_mode(config)` first: a non-snapshot mode raises `UnsupportedSyncModeError`
  (its reason names Path B) before a live client is made, before the stamp check, before the synced
  tables are read, before `policy_name`, and before any update starts. The `full_refresh` argument
  never skips it.
- `modes.policy_name` maps as today: unset and "snapshot" give SNAPSHOT for every table; "triggered"
  gives TRIGGERED only for the tables in `modes.TRIGGERED_TABLES`, SNAPSHOT for the rest.

The live seams (`live_clients`, the stamp check, `synced_pairs`, `connect_branch`, the update runner,
the high-water read and write) are replaced by recorders; the mode check itself is the shipped code.
"""
from typing import Any, Dict, List

import pytest

from lgk.refresh import modes
from lgk.refresh import sync as sync_mod

CAT = "main"
FQ = f"{CAT}.lgk_v1_abc123"


def _config(**extra: Any) -> Dict[str, Any]:
    cfg = {
        "graph_name": "provider_graph",
        "lakebase_project": "proj",
        "lakebase_catalog": CAT,
        "kit_schema": "lgk_v1_abc123",
        "served_schema": "lgk_served_abc123",
        "branch_id": "t-x",
        "run_folder": "/Workspace/Users/someone@example.com/lgk_v1_abc123/run",
    }
    cfg.update(extra)
    return cfg


class _Conn:
    autocommit = False

    def close(self) -> None:
        pass


@pytest.fixture
def calls(monkeypatch) -> List[str]:
    """Record every live seam the sync task reaches, in order."""
    seen: List[str] = []

    def rec(name, value=None):
        def f(*a, **k):
            seen.append(name)
            return value
        return f

    monkeypatch.setattr(sync_mod, "live_clients", rec("live_clients", (object(), object())))
    monkeypatch.setattr(sync_mod, "_require_stamped_tables", rec("stamp_check"))
    monkeypatch.setattr(sync_mod, "synced_pairs",
                        rec("synced_pairs", [(f"{FQ}.nodes", "s"), (f"{FQ}.edges", "s"), (f"{FQ}.node_id_map", "s")]))
    monkeypatch.setattr(sync_mod, "connect_branch", rec("connect_branch", _Conn()))
    monkeypatch.setattr(sync_mod.mf, "read_sync_high_water", rec("read_high_water", None))
    monkeypatch.setattr(sync_mod, "_record_high_water", rec("record_high_water"))
    monkeypatch.setattr(sync_mod, "_run_adaptive", rec("run_updates"))
    real_policy = modes.policy_name
    monkeypatch.setattr(modes, "policy_name", lambda *a, **k: (seen.append("policy_name"), real_policy(*a, **k))[1])
    return seen


# hostile: a config provisioned as snapshot and edited to triggered afterwards; the case and space
# variants catch a check that normalises; "continuous" catches a deny-list of "triggered" only
NOT_SNAPSHOT = ["triggered", "Triggered", " triggered ", "continuous"]
# hostile: an explicit full_refresh skips policy_name today, so nothing reads the mode at all
FULL_REFRESH = [None, True, False]


@pytest.mark.parametrize("full_refresh", FULL_REFRESH, ids=str)
@pytest.mark.parametrize("mode", NOT_SNAPSHOT, ids=repr)
def test_the_nightly_sync_task_refuses_a_non_snapshot_mode_before_anything_runs(calls, mode, full_refresh):
    with pytest.raises(modes.UnsupportedSyncModeError, match="Path B"):
        sync_mod.sync(_config(sync_mode=mode), full_refresh)
    assert calls == [], f"reached {calls} before the refusal"


@pytest.mark.parametrize("full_refresh", FULL_REFRESH, ids=str)
@pytest.mark.parametrize("mode", NOT_SNAPSHOT, ids=repr)
def test_sync_all_refuses_a_non_snapshot_mode_before_the_tables_are_read(calls, mode, full_refresh):
    # a direct caller of sync_all, which skips sync(), is refused too
    with pytest.raises(modes.UnsupportedSyncModeError, match="Path B"):
        sync_mod.sync_all(object(), object(), _config(sync_mode=mode), full_refresh)
    assert calls == [], f"reached {calls} before the refusal"


def test_the_refusal_names_the_mode_and_the_reason(calls):
    with pytest.raises(modes.UnsupportedSyncModeError) as e:
        sync_mod.sync(_config(sync_mode="triggered"), None)
    assert "'triggered'" in str(e.value) and "Path B" in str(e.value) and "snapshot" in str(e.value)


@pytest.mark.parametrize("cfg", [_config(), _config(sync_mode="snapshot")], ids=["unset", "snapshot"])
def test_snapshot_and_unset_still_sync_every_table_with_a_full_refresh(calls, monkeypatch, cfg):
    # control: the check must not stop the v1 mode
    jobs: List[Any] = []
    monkeypatch.setattr(sync_mod, "_run_adaptive", lambda w, j, *a: (calls.append("run_updates"), jobs.extend(j)))
    sync_mod.sync(cfg, None)
    assert calls[:2] == ["live_clients", "stamp_check"] and "run_updates" in calls, calls
    assert jobs == [(f"{FQ}.nodes", True), (f"{FQ}.edges", True), (f"{FQ}.node_id_map", True)]


# --------------------------------------------------------------- policy_name maps as today (a pin)
TABLES = ["nodes", "edges", "node_id_map", "node_score", "node_group", "closure"]


@pytest.mark.parametrize("cfg", [_config(), _config(sync_mode="snapshot")], ids=["unset", "snapshot"])
def test_policy_name_is_snapshot_for_every_table_when_the_mode_is_snapshot_or_unset(cfg):
    assert {t: modes.policy_name(cfg, t) for t in TABLES} == {t: "SNAPSHOT" for t in TABLES}


def test_policy_name_is_triggered_only_for_the_triggered_tables():
    # hostile: tables outside the triggered set (node_id_map and the enrichment tables)
    got = {t: modes.policy_name(_config(sync_mode="triggered"), t) for t in TABLES}
    assert got == {"nodes": "TRIGGERED", "edges": "TRIGGERED", "node_id_map": "SNAPSHOT",
                   "node_score": "SNAPSHOT", "node_group": "SNAPSHOT", "closure": "SNAPSHOT"}


def test_policy_name_reads_the_triggered_set_never_a_fixed_list(monkeypatch):
    # varied vocabulary: a one-table set must move nodes to SNAPSHOT
    monkeypatch.setattr(modes, "TRIGGERED_TABLES", ("edges",))
    got = {t: modes.policy_name(_config(sync_mode="triggered"), t) for t in ("nodes", "edges")}
    assert got == {"nodes": "SNAPSHOT", "edges": "TRIGGERED"}
