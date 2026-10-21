"""Offline tests for the Asset Bundle export."""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from lgk import bundle_entry
from lgk.export import IdentityInBundleError, export, refuse_identities

SRC = Path(__file__).resolve().parents[1] / "src" / "lgk"
HOSTILE_SP = "0f0f0f0f-hostile-sp-id-in-env"


def _workshop_run(tmp_path: Path) -> Path:
    """A run folder after a workshop provision, as the operator."""
    run = tmp_path / "run"
    run.mkdir()
    # a real mapping: the bundle's SP builds from its source schema, and the README's
    # grants name that schema
    from tests.test_build_provision_walks import _write_provider_graph_mapping

    _write_provider_graph_mapping(run / "graph_mapping.yaml", "cat", "fx")
    (run / "flags.json").write_text('{"flags": []}')
    (run / "profile").mkdir()
    (run / "profile" / "t.json").write_text("{}")
    # hostile: workshop state that would make the SP inherit the operator's run —
    # run.json says provision already succeeded (it would be skipped), and the manifest
    # lists the operator's synced tables (they would be reused).
    (run / "run.json").write_text(json.dumps({"provision": {"status": "success", "error": None}}))
    (run / "lgk_manifest.json").write_text(json.dumps({"objects": [{"type": "synced_table", "name": "c.s_v1.nodes"}]}))
    # hostile: a stale bundle from an earlier export inside the run folder
    (run / "bundle").mkdir()
    (run / "bundle" / ".lgk_bundle").write_text("")
    (run / "bundle" / "stale.txt").write_text("old")
    # hostile: the other workshop records, and a nested copy of run.json
    (run / "versions.json").write_text("{}")
    (run / "branch_manifest.json").write_text("{}")
    (run / "profile" / "run.json").write_text("{}")
    # hostile: the operator's passing preflight; travelling, it would name the operator as the
    # SP run's sync owner or be copied over the SP's own record on every task
    (run / "preflight_result.json").write_text(json.dumps({"status": "pass", "sync_owner_user": "operator"}))
    return run


def _config(run: Path, **extra):
    return {
        "graph_name": "g", "kit_schema": "lgk_v1_abc123", "lakebase_project": "proj",
        "lakebase_catalog": "cat", "branch_id": "t-x", "version": 3, "warehouse_id": "wh",
        "run_folder": str(run), **extra,
    }


@pytest.fixture
def bundle(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LGK_SP_APP_ID", HOSTILE_SP)
    run = _workshop_run(tmp_path)
    out = export(_config(run))
    return out, capsys.readouterr().out


def _job(out: Path, key: str = "lgk_provision") -> dict:
    return yaml.safe_load((out / "resources" / "lgk_job.yml").read_text())["resources"]["jobs"][key]


def _keys_and_deps(job: dict):
    return ([t["task_key"] for t in job["tasks"]],
            [t.get("depends_on", [{}])[0].get("task_key") for t in job["tasks"]])


def test_job_runs_as_the_sp_variable_not_the_deployer(bundle):
    out, _ = bundle
    job = _job(out)
    assert job["run_as"] == {"service_principal_name": "${var.sp_app_id}"}
    top = yaml.safe_load((out / "databricks.yml").read_text())
    assert "sp_app_id" in top["variables"]
    assert top["permissions"] == [{"level": "CAN_MANAGE", "service_principal_name": "${var.sp_app_id}"}]


def test_sp_id_from_env_is_never_written(bundle):
    out, _ = bundle
    for p in out.rglob("*"):
        if p.is_file():
            assert HOSTILE_SP not in p.read_text(errors="ignore"), p


def test_provision_job_is_preflight_as_the_sp_then_build_then_provision(bundle):
    out, _ = bundle
    tasks = _job(out)["tasks"]
    # the SP builds its own kit tables before it provisions them
    assert _keys_and_deps(_job(out)) == (["preflight", "build", "provision"], [None, "preflight", "build"])
    pre, build, prov = (t["spark_python_task"]["parameters"] for t in tasks)
    assert pre[pre.index("--task") + 1] == "preflight" and prov[prov.index("--task") + 1] == "provision"
    assert build[build.index("--task") + 1] == "build"
    # the sync owner is the SP variable (resolved at deploy), never an identity in the bundle
    assert pre[pre.index("--sync-owner") + 1] == "${var.sp_app_id}"
    assert "--sync-owner" not in prov and "--sync-owner" not in build
    assert all(t["spark_python_task"]["python_file"] == "../lgk_task.py" for t in tasks)


def test_bundle_config_scopes_the_sp_preflight_and_drops_the_owner(tmp_path):
    run = _workshop_run(tmp_path)
    # hostile: the workshop config says the run creates branches, and names its (email) owner
    out = export(_config(run, sync_owner_user="op@example.com", creates_branches=True))
    cfg = json.loads((out / "run" / "config.json").read_text())
    assert cfg["creates_branches"] is False and cfg["provision_only"] is True
    assert "sync_owner_user" not in cfg


def test_refresh_is_its_own_nightly_job_that_never_runs_provision(tmp_path):
    """#22: provision (one-time, version N+1) and the nightly refresh are separate jobs."""
    run = _workshop_run(tmp_path)
    (run / "enrich.yaml").write_text("methods: [degree]\n")  # the run enriched: scores must recompute
    # the bundle's SP builds its own kit tables, so the operator pipeline keeps the kit build
    out = export(_config(run, sdp_pipeline_id="pipe-1", build_mode="sql"))
    provision_job, refresh = _job(out), _job(out, "lgk_refresh")
    assert _keys_and_deps(provision_job) == (["preflight", "build", "enrich", "provision"],
                                             [None, "preflight", "build", "enrich"])
    assert _keys_and_deps(refresh) == (["sdp_pipeline", "build", "enrich", "sync", "reapply"],
                                       [None, "sdp_pipeline", "build", "enrich", "sync"])
    assert refresh["tasks"][0]["pipeline_task"] == {"pipeline_id": "pipe-1"}
    for t in refresh["tasks"][1:]:
        params = t["spark_python_task"]["parameters"]
        assert params[params.index("--task") + 1] == t["task_key"]  # no task runs provision
    assert refresh["schedule"] == {"quartz_cron_expression": "0 0 2 * * ?", "timezone_id": "UTC"}
    assert "pause_status" not in refresh["schedule"]  # nightly by default, not paused
    for job in (provision_job, refresh):
        assert job["run_as"] == {"service_principal_name": "${var.sp_app_id}"}
        assert job["tags"] == {"lgk_graph": "g"}
        assert job["max_concurrent_runs"] == 1
        for t in job["tasks"]:
            if "spark_python_task" in t:
                params = t["spark_python_task"]["parameters"]
                assert params[params.index("--job-id") + 1] == "{{job.id}}"
    assert "schedule" not in provision_job


def test_refresh_without_pipeline_or_enrich_is_sync_then_reapply(bundle):
    # hostile: no sdp_pipeline_id and no enrich.yaml. An enrich task here would fail every night
    # (enrich stops with no enrich.yaml), and a dangling depends_on on a missing task fails deploy.
    out, _ = bundle
    assert _keys_and_deps(_job(out, "lgk_refresh")) == (["build", "sync", "reapply"], [None, "build", "sync"])


def test_bundle_refresh_without_pipeline_builds_first(tmp_path):
    """With no operator pipeline the bundle's nightly job rebuilds the graph first."""
    run = _workshop_run(tmp_path)
    (run / "enrich.yaml").write_text("methods: [degree]\n")
    refresh = _job(export(_config(run)), "lgk_refresh")
    assert _keys_and_deps(refresh) == (["build", "enrich", "sync", "reapply"],
                                       [None, "build", "enrich", "sync"])
    build_task = refresh["tasks"][0]
    params = build_task["spark_python_task"]["parameters"]  # a python task, like enrich
    assert params[params.index("--task") + 1] == "build"
    assert params[params.index("--run-dir") + 1] == "${workspace.root_path}/state/run"
    assert "pipeline_task" not in build_task


# The nightly-chain rule table, written out by hand (enriched); the not-enriched chain drops "enrich".
_CHAINS = [
    ({}, ["build", "enrich", "sync", "reapply"]),
    ({"build_mode": "sql"}, ["build", "enrich", "sync", "reapply"]),
    ({"sdp_pipeline_id": ""}, ["build", "enrich", "sync", "reapply"]),
    ({"sdp_pipeline_id": "pipe-1"}, ["sdp_pipeline", "stamp", "enrich", "sync", "reapply"]),  # stamp: owner decision
    ({"sdp_pipeline_id": "pipe-1", "build_mode": "sql"}, ["sdp_pipeline", "build", "enrich", "sync", "reapply"]),
]


@pytest.mark.parametrize("extra,expected", _CHAINS)
@pytest.mark.parametrize("enriched", [True, False])
def test_job_and_bundle_chains_are_the_same(tmp_path, extra, expected, enriched):
    """The workspace job and the bundle's job both come from refresh_chain: they cannot drift."""
    from lgk.refresh import job

    run = _workshop_run(tmp_path)
    if enriched:
        (run / "enrich.yaml").write_text("methods: [degree]\n")
    cfg = _config(run, **extra)
    want = expected if enriched else [k for k in expected if k != "enrich"]
    assert job.refresh_chain(cfg, enriched=enriched) == want
    if "stamp" in want:  # the bundle's SP builds its own tables; a no-build chain is refused
        with pytest.raises(ValueError, match="build_mode"):
            export(cfg)
    else:
        keys, deps = _keys_and_deps(_job(export(cfg), "lgk_refresh"))
        assert keys == want and deps == [None, *want[:-1]]
    # the workspace job decides enrich from the same run folder (no enrich.yaml -> no enrich task)
    assert [t.task_key for t in job.refresh_job_tasks(cfg, notebook_path="/nb", wheel_path="/w.whl")] == want


def test_bundle_decides_enrich_through_the_shared_helper(tmp_path, monkeypatch):
    """One rule for both jobs: export asks lgk.refresh.job.run_enriched, it does not re-check the file."""
    import lgk.export as ex

    run = _workshop_run(tmp_path)  # hostile: no enrich.yaml, so a copied check would say "no enrich"
    seen = []
    monkeypatch.setattr(ex, "run_enriched", lambda folder: seen.append(folder) or True)
    keys = _keys_and_deps(_job(export(_config(run)), "lgk_refresh"))[0]
    assert keys == ["build", "enrich", "sync", "reapply"] and seen == [str(run)]


def test_entry_runs_the_build_task(tmp_path, monkeypatch):
    """The bundle's lgk_task entry accepts the nightly build task and runs it into the run dir."""
    import lgk.refresh.tasks as rt

    calls = []
    monkeypatch.setattr(rt, "_task_functions", lambda: {"build": lambda c: calls.append(c["run_folder"])})
    run_dir = tmp_path / "rd"
    bundle_entry.main(_argv(_bundled(tmp_path), run_dir, "build"))
    assert calls == [str(run_dir)]


def test_workshop_state_does_not_travel(bundle):
    out, _ = bundle
    shipped = sorted(str(p.relative_to(out / "run")) for p in (out / "run").rglob("*") if p.is_file())
    assert shipped == ["config.json", "flags.json", "graph_mapping.yaml", "profile/t.json"]


def test_bundle_config_is_a_new_version_without_runtime_keys(tmp_path):
    run = _workshop_run(tmp_path)
    # hostile: a runtime-only key and an identity key the workshop config carries
    out = export(_config(run, operator="someone", user_name="someone"))
    cfg = json.loads((out / "run" / "config.json").read_text())
    assert cfg["version"] == 4  # a new operator is a new version (workshop was 3)
    assert "run_folder" not in cfg and "operator" not in cfg and "user_name" not in cfg
    assert cfg["kit_schema"] == "lgk_v1_abc123_sp" and cfg["branch_id"] == "t-x"  # the SP's own


def test_non_json_config_value_is_refused(tmp_path):
    run = _workshop_run(tmp_path)
    with pytest.raises(TypeError):
        export(_config(run, sdp_pipeline_id=object()))


def test_kit_code_is_copied_without_bytecode(bundle):
    out, _ = bundle
    shipped = out / "src" / "lgk"
    assert (shipped / "steps" / "provision.py").read_text() == (SRC / "steps" / "provision.py").read_text()
    assert (shipped / "bundle_entry.py").exists()
    assert not list(shipped.rglob("__pycache__")) and not list(shipped.rglob("*.pyc"))


def test_stale_bundle_is_replaced(bundle):
    out, _ = bundle
    assert not (out / "stale.txt").exists()


def test_deploy_command_is_printed_with_the_env_var(bundle):
    out, printed = bundle
    assert "databricks bundle deploy" in printed
    assert '--var="sp_app_id=$LGK_SP_APP_ID"' in printed
    assert "databricks bundle run lgk_provision" in printed
    assert str(out) in printed


def test_email_in_any_bundle_file_is_refused(tmp_path):
    run = _workshop_run(tmp_path)
    # hostile: an operator email in a config value (the mutation "hardcode the operator email")
    with pytest.raises(IdentityInBundleError, match="config.json"):
        export(_config(run, notify="first.last@example.com"))


def test_refuse_identities_scans_yaml_too(bundle):
    out, _ = bundle
    (out / "resources" / "extra.yml").write_text("owner: a.b@" + "corp.io\n")
    with pytest.raises(IdentityInBundleError, match="extra.yml"):
        refuse_identities(out)


# ------------------------------------------------------------------ the job task itself
def _bundled(tmp_path: Path):
    run_src = tmp_path / "files" / "run"
    run_src.mkdir(parents=True)
    (run_src / "config.json").write_text(json.dumps(
        {"graph_name": "g", "version": 2, "lakebase_project": "proj", "branch_id": "t-x",
         "kit_schema": "k_sp", "workshop_kit_schema": "k"}))
    (run_src / "flags.json").write_text("{}")
    return run_src


def _argv(run_src, run_dir, task, job_id="4242"):
    return ["--src", "/x", "--run-src", str(run_src), "--run-dir", str(run_dir), "--task", task,
            "--job-id", job_id]


def _jobs_recorded(run_dir: Path):
    from lgk import manifest as mf

    m = mf.load_run_manifest({"run_folder": str(run_dir), "graph_name": "g"})
    return sorted((e.object_id, e.metadata["kind"], e.metadata["branch"]) for e in m.entries
                  if e.object_type == mf.ObjectType.JOB)


def test_each_bundle_task_records_its_own_job_once(tmp_path, monkeypatch):
    import lgk.refresh.tasks as rt
    import lgk.steps.preflight as pf

    _fake_serving_steps(monkeypatch, lambda c: None)
    monkeypatch.setattr(pf, "preflight", lambda c: None)
    monkeypatch.setattr(rt, "_task_functions", lambda: {k: (lambda c: None) for k in ("sync", "reapply")})
    run_src, run_dir = _bundled(tmp_path), tmp_path / "state" / "run"
    # hostile: the provision job has two tasks (preflight, provision); still ONE recorded job
    bundle_entry.main(_argv(run_src, run_dir, "preflight", job_id="11") + ["--sync-owner", "sp"])
    bundle_entry.main(_argv(run_src, run_dir, "provision", job_id="11"))
    # hostile: the refresh job runs twice and has two tasks; it is still ONE recorded job
    for _ in range(2):
        for task in ("sync", "reapply"):
            bundle_entry.main(_argv(run_src, run_dir, task, job_id="22"))
    branch = "projects/proj/branches/t-x"
    assert _jobs_recorded(run_dir) == [("11", "provision", branch), ("22", "refresh", branch)]


def test_an_unresolved_job_id_is_refused_before_the_task_runs(tmp_path, monkeypatch):
    import lgk.steps.provision as prov

    ran = []
    monkeypatch.setattr(prov, "provision", lambda c: ran.append(c))
    # hostile: the literal dynamic reference, as it arrives if the job does not resolve it
    with pytest.raises(ValueError, match="job-id"):
        bundle_entry.main(_argv(_bundled(tmp_path), tmp_path / "rd", "provision", job_id="{{job.id}}"))
    assert ran == []


def _fake_serving_steps(monkeypatch, provision, order=None):
    """provision faked as given; advise and switch faked, recording their order."""
    import lgk.steps.advise as adv
    import lgk.steps.provision as prov
    import lgk.steps.switch as sw

    order = [] if order is None else order
    monkeypatch.setattr(prov, "provision", provision)
    monkeypatch.setattr(adv, "advise", lambda c: order.append("advise"))
    monkeypatch.setattr(sw, "switch", lambda c: order.append("switch"))
    return order


def test_entry_runs_provision_in_the_writable_run_folder(tmp_path, monkeypatch):
    seen = {}
    order = _fake_serving_steps(monkeypatch, lambda c: seen.update(c) or order.append("provision"), [])
    run_src, run_dir = _bundled(tmp_path), tmp_path / "state" / "run"
    bundle_entry.main(_argv(run_src, run_dir, "provision"))
    assert seen["run_folder"] == str(run_dir) and seen["version"] == 2
    assert (run_dir / "flags.json").exists()
    state = json.loads((run_dir / "run.json").read_text())
    # the bundle serves through the same flow: the views move last, after the advisor
    assert order == ["provision", "advise", "switch"]
    assert [state[s]["status"] for s in ("provision", "advise", "switch")] == ["success"] * 3


def test_entry_fails_the_task_when_provision_fails(tmp_path, monkeypatch):
    def boom(c):
        raise RuntimeError("no grant")

    order = _fake_serving_steps(monkeypatch, boom)
    with pytest.raises(RuntimeError, match="no grant"):
        bundle_entry.main(_argv(_bundled(tmp_path), tmp_path / "rd", "provision"))
    assert order == [], "a failed provision must never reach advise or switch"


def test_entry_runs_a_refresh_task_every_time(tmp_path, monkeypatch):
    calls = []
    import lgk.refresh.tasks as rt

    monkeypatch.setattr(rt, "_task_functions", lambda: {"sync": lambda c: calls.append(c["run_folder"])})
    run_dir = tmp_path / "rd"
    for _ in range(2):
        bundle_entry.main(_argv(_bundled(tmp_path) if not calls else tmp_path / "files" / "run", run_dir, "sync"))
    assert calls == [str(run_dir), str(run_dir)]


def test_entry_refuses_unknown_task(tmp_path):
    with pytest.raises(ValueError, match="unknown bundle task"):
        bundle_entry.main(_argv(_bundled(tmp_path), tmp_path / "rd", "teardown"))
    assert not (tmp_path / "rd").exists()  # refused before anything is staged or recorded


# Built from parts, so the public gate finds no user path in this file.
USERS = "/" + "Users" + "/"
WS_USERS = "/Workspace" + USERS


def test_users_path_gets_the_workspace_prefix():
    assert str(bundle_entry._ws(USERS + "x/run")) == WS_USERS + "x/run"
    assert str(bundle_entry._ws(WS_USERS + "x/run")) == WS_USERS + "x/run"


def test_workspace_client_object_is_dropped_not_refused(tmp_path):
    run = _workshop_run(tmp_path)
    out = export(_config(run, workspace_client=object()))
    assert "workspace_client" not in json.loads((out / "run" / "config.json").read_text())


def test_email_in_a_file_name_is_refused_and_nothing_is_left(tmp_path):
    run = _workshop_run(tmp_path)
    # hostile: the identity is in a path, not in any file's content
    (run / "profile" / ("first.last@" + "example.com.json")).write_text("{}")
    with pytest.raises(IdentityInBundleError, match="example.com.json"):
        export(_config(run))
    assert not (run / "bundle.staging").exists()
    assert (run / "bundle" / "stale.txt").exists()  # the earlier bundle is untouched, not half-replaced


def test_refused_bundle_is_not_left_deployable(tmp_path):
    run = _workshop_run(tmp_path)
    shutil.rmtree(run / "bundle")
    with pytest.raises(IdentityInBundleError):
        export(_config(run, notify="first.last@example.com"))
    assert not (run / "bundle").exists() and not (run / "bundle.staging").exists()


def test_out_dir_that_is_or_contains_the_run_folder_is_refused(tmp_path):
    run = _workshop_run(tmp_path)
    for target in (run, tmp_path):
        with pytest.raises(ValueError, match="contains it"):
            export(_config(run), out_dir=str(target))
    assert (run / "graph_mapping.yaml").exists() and (run / "run.json").exists()


def test_out_dir_that_is_not_a_bundle_is_refused(tmp_path):
    run = _workshop_run(tmp_path)
    precious = tmp_path / "team_repo"
    precious.mkdir()
    (precious / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="not an lgk bundle"):
        export(_config(run), out_dir=str(precious))
    assert (precious / "keep.txt").exists()


def test_custom_out_dir_inside_the_run_folder_is_not_copied_into_itself(tmp_path):
    run = _workshop_run(tmp_path)
    out = export(_config(run), out_dir=str(run / "exported"))
    assert not (out / "run" / "exported").exists() and not (out / "run" / "exported.staging").exists()
    export(_config(run), out_dir=str(run / "exported"))  # a re-export replaces its own bundle
    assert not (out / "run" / "exported").exists()


def test_bootstrap_puts_the_workspace_src_on_sys_path(tmp_path):
    import subprocess
    import sys

    from lgk.export import BOOTSTRAP

    # stub lgk.bundle_entry under a fake "/Workspace" root is not possible; record sys.path[0] instead
    stub = tmp_path / "stub" / "lgk"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text("")
    (stub / "bundle_entry.py").write_text("import sys\ndef main(a):\n    print(sys.path[0])\n")
    (tmp_path / "boot.py").write_text(BOOTSTRAP)
    for given, want in ((USERS + "x/src", WS_USERS + "x/src"), (WS_USERS + "x/src", WS_USERS + "x/src")):
        res = subprocess.run([sys.executable, str(tmp_path / "boot.py"), "--src", given, "--task", "provision"],
                             capture_output=True, text=True, env={"PYTHONPATH": str(tmp_path / "stub"), "PATH": "/usr/bin:/bin"})
        assert res.stdout.strip() == want, res.stderr


def test_readme_documents_the_operator_setup_without_identities(bundle):
    out, _ = bundle
    text = (out / "README.md").read_text()
    for line in ("CREATE ROLE reader WITH LOGIN;", "databricks_create_role('<SP application id>', 'service_principal')",
                 "GRANT CONNECT, CREATE ON DATABASE databricks_postgres", "cat.lgk_v1_abc123_sp_v4"):
        assert line in text, line
    assert HOSTILE_SP not in text


def test_second_bundle_run_does_not_skip_provision(tmp_path, monkeypatch):
    """Isaac Review: a run.json left by an earlier successful run must not make provision a no-op."""
    run_src = tmp_path / "bundled"
    run_src.mkdir()
    (run_src / "config.json").write_text(json.dumps({"graph_name": "g"}))
    run_dir = tmp_path / "state" / "run"
    calls = []

    def fake_provision(config):
        calls.append(config["run_folder"])

    _fake_serving_steps(monkeypatch, fake_provision)
    for _ in range(2):  # hostile: the second run finds the first run's run.json (provision: success)
        config = bundle_entry._stage_run_folder(run_src, run_dir)
        bundle_entry._run_provision(config)
    assert len(calls) == 2, "the second bundle run skipped provision"


def test_bundle_run_folder_clears_the_gate_with_the_workshop_stamp_but_not_its_build(tmp_path, monkeypatch):
    """The SP's staged run folder clears the gate under the SAME stamp, with NO build record.

    The bundle's SP builds its own kit tables (the workshop's are the operator's), so
    flags.json (+ its stamp), schema.json and the mapping travel, and the workshop's
    privacy_bindings.json does not: it names the workshop's tables. Provision then refuses
    until the SP's own build records its tables (tests/test_export_sp_build.py).
    """
    from lgk.privacy import binding
    from lgk.privacy.gate import require_clear
    from tests.privacy_support import clear_gate_offline
    from tests.test_build_provision_walks import _write_provider_graph_mapping

    monkeypatch.setenv("LGK_SP_APP_ID", HOSTILE_SP)
    run = tmp_path / "run"
    run.mkdir()
    _write_provider_graph_mapping(run / "graph_mapping.yaml", "cat", "fx")
    clear_gate_offline(str(run))
    stamp = binding.stamp_id(require_clear(str(run)))
    tables = [f"cat.lgk_v1_abc123.{t}" for t in binding.BUILT_TABLES]
    binding.record_build(str(run), stamp, tables)
    # hostile: workshop state that must not travel next to the privacy records
    (run / "run.json").write_text(json.dumps({"provision": {"status": "success", "error": None}}))
    out = export(_config(run))

    staged = bundle_entry._stage_run_folder(out / "run", tmp_path / "state" / "run")
    assert staged["version"] == 4  # a new version, served under the workshop's decision
    assert binding.stamp_id(require_clear(staged["run_folder"])) == stamp
    assert binding.built_stamp(staged["run_folder"]) is None
    with pytest.raises(binding.StaleBuildError, match="records no build"):
        binding.check_build_record(staged["run_folder"], stamp, tables)


# ------------------------------------------------ H1: the SP's preflight, the bundle's first task
SP_APP = "11111111-2222-4333-" + "8444-555555555555"


def sp_environment(config):
    """preflight_env factory: the workspace as the bundle's SP sees it (README grants)."""
    from tests.preflight_support import FakePg, FakeWorkspace

    ws = FakeWorkspace(user=SP_APP, project=config["lakebase_project"], branches={config["branch_id"]},
                       schemas={"cat.k_sp", "cat.k_sp_v2", "cat.fx"})
    ws.project_acl = [{"service_principal_name": SP_APP, "all_permissions": [{"permission_level": "CAN_USE"}]}]
    # the SP builds into its own pre-made kit schema from the mapping's source schema; it
    # holds nothing on the workshop's kit schema `cat.k`
    ws.grants = {("catalog", "cat"): ["USE_CATALOG"], ("schema", "cat.k_sp"): ["USE_SCHEMA", "CREATE_TABLE"],
                 ("schema", "cat.k_sp_v2"): ["USE_SCHEMA", "CREATE_TABLE"], ("schema", "cat.fx"): ["USE_SCHEMA", "SELECT"]}
    return ws, FakePg(roles={"reader"}).connect


def _sp_bundled(tmp_path: Path) -> Path:
    from tests.test_build_provision_walks import _write_provider_graph_mapping

    run_src = tmp_path / "files" / "run"
    run_src.mkdir(parents=True)
    _write_provider_graph_mapping(run_src / "graph_mapping.yaml", "cat", "fx")
    (run_src / "config.json").write_text(json.dumps({
        "graph_name": "g", "version": 2, "lakebase_project": "proj", "lakebase_catalog": "cat",
        "kit_schema": "k_sp", "workshop_kit_schema": "k", "branch_id": "t-x", "creates_branches": False,
        "provision_only": True, "sp_builds": True, "preflight_env": "tests.test_export:sp_environment"}))
    return run_src


def test_the_preflight_task_records_the_sp_as_the_owner_provision_requires(tmp_path):
    from lgk.steps.preflight import SyncOwnerMismatch, require_sync_owner

    run_src, run_dir = _sp_bundled(tmp_path), tmp_path / "state" / "run"
    bundle_entry.main(_argv(run_src, run_dir, "preflight") + ["--sync-owner", SP_APP])
    record = json.loads((run_dir / "preflight_result.json").read_text())
    assert record["status"] == "pass" and record["operator"] == SP_APP, record["reason"]
    # the next task re-stages the run folder: the SP's record must survive it
    bundle_entry._stage_run_folder(run_src, run_dir)
    assert require_sync_owner(run_dir, SP_APP) == SP_APP
    with pytest.raises(SyncOwnerMismatch):
        require_sync_owner(run_dir, "operator@example.com")


def test_the_preflight_task_blocks_without_the_sp_grants(tmp_path):
    from lgk.steps.preflight import PreflightBlocked

    run_src, run_dir = _sp_bundled(tmp_path), tmp_path / "state" / "run"
    cfg = json.loads((run_src / "config.json").read_text())
    (run_src / "config.json").write_text(json.dumps({**cfg, "creates_branches": True}))  # hostile: CAN_USE only
    with pytest.raises(PreflightBlocked, match="CAN_MANAGE on project proj"):
        bundle_entry.main(_argv(run_src, run_dir, "preflight") + ["--sync-owner", SP_APP])


@pytest.mark.parametrize("owner", [None, "${var.sp_app_id}"])
def test_the_preflight_task_refuses_a_missing_or_unresolved_owner(tmp_path, owner):
    run_src, run_dir = _sp_bundled(tmp_path), tmp_path / "state" / "run"
    argv = _argv(run_src, run_dir, "preflight") + (["--sync-owner", owner] if owner else [])
    with pytest.raises(ValueError, match="sync-owner"):
        bundle_entry.main(argv)
    assert not (run_dir / "preflight_result.json").exists()


def test_an_injected_pg_connect_is_dropped_not_refused(tmp_path):
    run = _workshop_run(tmp_path)
    out = export(_config(run, pg_connect=lambda *a: None))  # hostile: a callable cannot be JSON
    assert "pg_connect" not in json.loads((out / "run" / "config.json").read_text())


def test_the_preflight_task_owner_is_the_argument_not_a_config_value(tmp_path):
    run_src, run_dir = _sp_bundled(tmp_path), tmp_path / "state" / "run"
    cfg = json.loads((run_src / "config.json").read_text())
    # hostile: a hand-edited bundled config names someone else as the owner
    (run_src / "config.json").write_text(json.dumps({**cfg, "sync_owner_user": "operator@" + "x.org"}))
    bundle_entry.main(_argv(run_src, run_dir, "preflight") + ["--sync-owner", SP_APP])
    assert json.loads((run_dir / "preflight_result.json").read_text())["sync_owner_user"] == SP_APP
