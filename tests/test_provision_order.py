"""Provision proves before anything is served; advise runs between; switch is last.

Offline: provision's and switch's Postgres parts are recorded instead of run; the order, the
refusals and the runner wiring run as shipped. The live proof is tests/test_provision_order_live.py.
"""

import json
from pathlib import Path
from typing import Any, List

import pytest

from lgk.runner import LaptopRunner, RunFolder, RunStatus, execute_step

# require_sync_owner is faked here and proven in test_preflight.py; it must still run first
PROVISION_PARTS = ("require_sync_owner", "_refuse_existing_version", "_refuse_unbound_build", "_create_synced_tables",
                   "_mark_pg_version_schema", "_create_version_indexes", "_lock_version_from_reader", "_stage_version_functions",
                   "_prove_reader_isolated")
# Anything that would serve (or half-serve) a version. Provision must call none of these.
SERVING_PARTS = ("_create_served_views", "_setup_reader_role", "_serve_attribution", "serve")


@pytest.fixture
def calls(monkeypatch) -> List[Any]:
    from lgk.steps import provision as prov
    from lgk.steps import switch as sw

    got: List[Any] = []
    monkeypatch.setattr(prov, "require_clear", lambda rf: {})
    monkeypatch.setattr(sw.binding, "require_servable", lambda rf, target: "stamp")
    monkeypatch.setattr(prov.binding, "stamp_id", lambda clear: "stamp")
    for name in PROVISION_PARTS + SERVING_PARTS:
        monkeypatch.setattr(prov, name, lambda *a, _n=name, **k: got.append(_n))
    monkeypatch.setattr(prov, "connect_branch", lambda c: _Conn())
    return got


class _Conn:
    autocommit = False

    def close(self):
        pass


def _config(tmp_path: Path, **kw) -> dict:
    return {"graph_name": "g", "run_folder": str(tmp_path), "kit_schema": "kit", "version": 2, **kw}


class TestProvisionNeverServes:
    def test_provision_proves_then_records_and_serves_nothing(self, tmp_path, calls):
        from lgk.steps.provision import provision

        assert provision(_config(tmp_path)) is True
        assert calls == list(PROVISION_PARTS)
        bindings = json.loads((tmp_path / "privacy_bindings.json").read_text())
        assert bindings["versions"] == {"kit_v2": "stamp"}

    def test_failed_proof_raises_records_nothing_and_never_serves(self, tmp_path, calls, monkeypatch):
        # hostile: everything before the proof succeeds, so only the order keeps N+1 unserved
        from lgk.steps import provision as prov

        def leak(config):
            calls.append("_prove_reader_isolated")
            raise RuntimeError("reader has SELECT on raw kit_v2.nodes")

        monkeypatch.setattr(prov, "_prove_reader_isolated", leak)
        with pytest.raises(RuntimeError, match="raw kit_v2.nodes"):
            prov.provision(_config(tmp_path))
        assert not set(SERVING_PARTS) & set(calls), calls
        assert not (tmp_path / "privacy_bindings.json").exists() or \
            "versions" not in json.loads((tmp_path / "privacy_bindings.json").read_text())

    def test_reader_is_locked_out_of_the_new_version_before_the_proof(self, tmp_path, calls):
        from lgk.steps.provision import provision

        provision(_config(tmp_path))
        assert calls.index("_mark_pg_version_schema") < calls.index("_lock_version_from_reader") \
            < calls.index("_prove_reader_isolated")


class _Cursor:
    def __init__(self, log):
        self.log = log

    def execute(self, sql, params=None):
        self.log.append(sql)

    def fetchone(self):
        return (1,)  # reader exists

    def close(self):
        pass


class TestLockVersionFromReader:
    def test_lock_revokes_on_the_version_and_grants_no_view(self, monkeypatch):
        """Provision's reader step: no GRANT at all (the views are switch's), REVOKE on kit_v2."""
        from lgk.steps import provision as prov

        log: List[str] = []

        class Conn:
            def cursor(self):
                return _Cursor(log)

            def commit(self):
                pass

            def close(self):
                pass

        monkeypatch.setattr(prov, "connect_branch", lambda c: Conn())
        prov._lock_version_from_reader({"graph_name": "g", "kit_schema": "kit", "version": 2})
        assert not [s for s in log if s.startswith("GRANT")], log
        assert "REVOKE ALL ON kit_v2.nodes FROM reader" in log
        assert "REVOKE ALL ON SCHEMA kit_v2 FROM reader" in log


class TestSwitchIsLast:
    def _provisioned(self, tmp_path):
        from lgk.privacy import binding

        binding.record_version(str(tmp_path), "kit_v2", "stamp")

    def test_switch_defaults_to_the_config_version(self, tmp_path, calls, monkeypatch):
        from lgk.steps import switch as sw

        self._provisioned(tmp_path)
        moved = []
        monkeypatch.setattr(sw, "_move_views", lambda c, t: moved.append(t))
        assert sw.switch(_config(tmp_path)) == "kit_v2"
        assert moved == ["kit_v2"]
        live = json.loads((tmp_path / "versions.json").read_text())["live_version_schema"]
        assert live == "kit_v2"

    def test_switch_refuses_before_moving_when_the_reader_is_not_isolated(self, tmp_path, calls, monkeypatch):
        # hostile: the version IS recorded (provision passed), then a raw grant leaked
        from lgk.steps import provision as prov
        from lgk.steps import switch as sw

        self._provisioned(tmp_path)
        moved = []
        monkeypatch.setattr(sw, "_move_views", lambda c, t: moved.append(t))

        def leak(config):
            raise RuntimeError("PUBLIC has SELECT on raw kit_v2.edges")

        monkeypatch.setattr(prov, "_prove_reader_isolated", leak)
        with pytest.raises(RuntimeError, match="raw kit_v2.edges"):
            sw.switch(_config(tmp_path))
        assert moved == [] and not (tmp_path / "versions.json").exists()

    def test_reader_grants_follow_the_views_onto_the_switched_version(self, tmp_path, calls, monkeypatch):
        # hostile: config says version 2 but the switch is a rollback to 1; grants/revokes and
        # attribution must use the TARGET version, not config["version"]
        from lgk.privacy import binding
        from lgk.steps import provision as prov
        from lgk.steps import switch as sw

        binding.record_version(str(tmp_path), "kit_v1", "stamp")
        seen = []
        monkeypatch.setattr(sw, "_move_views", lambda c, t: None)
        monkeypatch.setattr(prov, "_setup_reader_role", lambda c: seen.append(c["version"]))
        sw.switch(_config(tmp_path), 1)
        assert seen == [1]


class TestRunnerFlow:
    def test_steps_order_is_provision_advise_switch_walks(self):
        steps = RunFolder.STEPS
        assert steps.index("provision") < steps.index("advise") < steps.index("switch") < steps.index("walks")
        assert steps.index("switch") == steps.index("provision") + 2

    def test_a_new_provision_reruns_advise_and_switch(self, tmp_path, calls, monkeypatch):
        """Hostile: run.json says advise and switch succeeded for the OLD version; a new
        provision must make them run again, or the new version would never be served."""
        from lgk.steps.provision import provision

        folder = RunFolder(tmp_path)
        state = json.loads((tmp_path / "run.json").read_text())
        for s in ("advise", "switch"):
            state[s] = {"status": "success", "error": None}
        (tmp_path / "run.json").write_text(json.dumps(state))
        assert execute_step(folder, _config(tmp_path), "provision", provision).status == RunStatus.SUCCESS
        assert folder.get_step_status("advise") == RunStatus.NOT_RUN
        assert folder.get_step_status("switch") == RunStatus.NOT_RUN

    def test_runner_flow_runs_advise_on_the_unserved_version_then_switches(self, tmp_path, calls, monkeypatch):
        from lgk.steps import advise as adv
        from lgk.steps import switch as sw
        from lgk.steps.advise import advise as real_advise
        from lgk.steps.provision import provision

        order: List[str] = []
        live: List[str] = []  # what the served views read, as the fake catalog sees it
        monkeypatch.setattr(sw, "_move_views", lambda c, t: (order.append("switch"), live.append(t)))

        def fake_advisor(conn, served, version, queries, params, **kw):
            order.append("advise")
            assert version not in live, "the advisor must run before the version is served"
            return {"proposals": [], "version_schema": version, "served_schema": served, "depths": {}}

        monkeypatch.setattr(adv, "connect_branch", lambda c: _Conn())
        monkeypatch.setattr(adv, "_walk_queries", lambda c, s: {})
        monkeypatch.setattr(adv.advisor, "advise", fake_advisor)
        monkeypatch.setattr(adv.advisor, "write_report", lambda rf, r: None)
        results = LaptopRunner(_config(tmp_path, advisor_seed_node_id="n1")).run_steps(
            [("provision", provision), ("advise", real_advise), ("switch", sw.switch)])
        assert [r.status for r in results] == [RunStatus.SUCCESS] * 3, [r.to_dict() for r in results]
        assert order == ["advise", "switch"] and live == ["kit_v2"]


class TestServeTimeGuards:
    """Prover survivors X1/X2: the serve-time raw REVOKE, and the bundle failing on a later step."""

    def test_serve_time_reader_setup_revokes_raw_on_the_target_version(self, monkeypatch):
        # hostile: config says v2, the switch target is v1; the REVOKE must name v1
        from lgk.steps import provision as prov

        log: List[str] = []

        class Conn:
            def cursor(self):
                return _Cursor(log)

            def commit(self):
                pass

            def close(self):
                pass

        monkeypatch.setattr(prov, "connect_branch", lambda c: Conn())
        prov._setup_reader_role({"graph_name": "g", "kit_schema": "kit", "version": 1})
        assert "REVOKE ALL ON kit_v1.nodes FROM reader" in log
        assert "GRANT SELECT ON kit.nodes_view TO reader" in log

    @pytest.mark.parametrize("failing", ["advise", "switch"])
    def test_bundle_task_fails_when_a_later_serving_step_fails(self, tmp_path, monkeypatch, failing):
        from lgk import bundle_entry

        order: List[str] = []

        def step(name):
            def run(config):
                order.append(name)
                if name == failing:
                    raise RuntimeError(f"{name} broke")
            return run

        monkeypatch.setattr(bundle_entry, "_serving_steps",
                            lambda: [(n, step(n)) for n in ("provision", "advise", "switch")])
        with pytest.raises(RuntimeError, match=f"{failing} failed: {failing} broke"):
            bundle_entry._run_provision({"run_folder": str(tmp_path)})
        if failing == "advise":
            assert "switch" not in order


class TestAdvisorSkipsWithoutReadGrants:
    """The bundle SP's advise must not fail the provision task when it
    may not READ the version's raw tables; any other error still fails it."""

    class _Raising:
        autocommit = False

        def __init__(self, exc):
            self.exc, self.closed = exc, 0

        def execute(self, sql, params=None):
            raise self.exc

        def close(self):
            self.closed += 1

    def _wire(self, monkeypatch, exc):
        from lgk.steps import advise as adv

        conn = self._Raising(exc)
        monkeypatch.setattr(adv, "connect_branch", lambda c: conn)
        monkeypatch.setattr(adv, "_walk_queries", lambda c, s: {"web": {1: "SELECT 1"}})
        return conn

    def test_no_read_grant_gives_a_skipped_report_and_the_step_succeeds(self, tmp_path, monkeypatch, caplog):
        import psycopg

        from lgk.steps.advise import advise

        conn = self._wire(monkeypatch, psycopg.errors.InsufficientPrivilege("permission denied for schema kit_v2"))
        with caplog.at_level("WARNING", logger="lgk.steps.advise"):
            report = advise(_config(tmp_path))
        assert any("advisor skipped" in r.getMessage() for r in caplog.records), caplog.text
        assert report["skipped"] == "advisor skipped: permission denied for schema kit_v2"
        written = json.loads((tmp_path / "advisor_report.json").read_text())
        assert written["skipped"] == report["skipped"] and written["proposals"] == []
        assert "advisor skipped: permission denied for schema kit_v2" in (tmp_path / "advisor_report.md").read_text()
        assert conn.closed == 1

    def test_no_read_grant_during_the_timings_is_skipped_too(self, tmp_path, monkeypatch):
        # hostile: the seed is given (no seed query); the denial comes from the advisor's own reads
        import psycopg

        from lgk.steps.advise import advise

        self._wire(monkeypatch, psycopg.errors.InsufficientPrivilege("permission denied for table edges"))
        assert advise(_config(tmp_path, advisor_seed_node_id="n1"))["skipped"].endswith("table edges")

    @pytest.mark.parametrize("exc", [RuntimeError("connection reset"), ValueError("kit_v2.edges is empty")])
    def test_any_other_error_still_fails_the_step(self, tmp_path, monkeypatch, exc):
        from lgk.steps.advise import advise

        conn = self._wire(monkeypatch, exc)
        with pytest.raises(type(exc), match=str(exc)):
            advise(_config(tmp_path))
        assert not (tmp_path / "advisor_report.json").exists()
        assert conn.closed == 1  # the connection is closed on the failure path too

    def _bundle(self, tmp_path, monkeypatch):
        from lgk import bundle_entry
        from lgk.steps.advise import advise

        order: List[str] = []
        monkeypatch.setattr(bundle_entry, "_serving_steps", lambda: [
            ("provision", lambda c: order.append("provision")), ("advise", advise),
            ("switch", lambda c: order.append("switch"))])
        return lambda: bundle_entry._run_provision(_config(tmp_path)), order

    def test_the_bundle_task_succeeds_and_switches_when_advise_is_skipped(self, tmp_path, monkeypatch):
        import psycopg

        self._wire(monkeypatch, psycopg.errors.InsufficientPrivilege("permission denied for schema kit_v2"))
        run, order = self._bundle(tmp_path, monkeypatch)
        run()
        assert order == ["provision", "switch"]

    def test_the_bundle_task_fails_on_any_other_advise_error(self, tmp_path, monkeypatch):
        self._wire(monkeypatch, RuntimeError("connection reset"))
        run, order = self._bundle(tmp_path, monkeypatch)
        with pytest.raises(RuntimeError, match="advise failed: connection reset"):
            run()
        assert order == ["provision"]
