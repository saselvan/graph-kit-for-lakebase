"""The exported bundle's service principal builds and owns its kit tables.

Live 2026-10-04 (two job runs): the bundle's nightly `build` ran as
the SP and CREATE OR REPLACEd the OPERATOR's kit tables: `PERMISSION_DENIED: User does not have
MODIFY on Table ...nodes`. CREATE OR REPLACE needs ownership or MANAGE, so no grant of MODIFY
fixes it. The bundle now points at a kit schema of the SP's own (`<kit>_sp` by default, created by
the operator), the provision job builds there as the SP before provision, and every nightly
CREATE OR REPLACE is on the SP's own tables. The workshop's kit tables stay the operator's.

The SP path keeps every privacy check (owner condition 1): its build passes the privacy gate,
and its provision refuses tables not built (by the SP, in the SP's schema) under the current stamp.
"""

import json
from pathlib import Path

import pytest
import yaml

from lgk import bundle_entry
from lgk.export import export
from lgk.privacy import binding
from lgk.privacy.gate import PrivacyGateError, require_clear
from tests.privacy_support import clear_gate_offline
from tests.test_build_provision_walks import _write_provider_graph_mapping

WORKSHOP_KIT = "lgk_v1_abc123"


def _workshop(tmp_path: Path, **gate) -> Path:
    """A workshop run folder after build + provision as the operator, privacy gate cleared."""
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    _write_provider_graph_mapping(run / "graph_mapping.yaml", "cat", "fx")
    clear_gate_offline(str(run))
    stamp = binding.stamp_id(require_clear(str(run)))
    # hostile: the workshop's build record names the WORKSHOP tables under the same stamp. If
    # it travelled, the SP's provision would accept a build the SP never made.
    binding.record_build(str(run), stamp, [f"cat.{WORKSHOP_KIT}.{t}" for t in binding.BUILT_TABLES])
    # hostile: the workshop's enrich report; travelling, it would make provision sync enrichment
    # tables the SP never computed (and be copied over the SP's own report on every task)
    (run / "enrich_report.json").write_text(json.dumps({"degree": {"rows": 3}}))
    return run


def _config(run: Path, **extra) -> dict:
    return {"graph_name": "g", "kit_schema": WORKSHOP_KIT, "lakebase_project": "proj",
            "lakebase_catalog": "cat", "branch_id": "t-x", "version": 3, "warehouse_id": "wh",
            "run_folder": str(run), **extra}


def _bundle_cfg(out: Path) -> dict:
    return json.loads((out / "run" / "config.json").read_text())


def _job(out: Path, key: str) -> dict:
    return yaml.safe_load((out / "resources" / "lgk_job.yml").read_text())["resources"]["jobs"][key]


def _keys_and_deps(job: dict):
    return ([t["task_key"] for t in job["tasks"]],
            [t.get("depends_on", [{}])[0].get("task_key") for t in job["tasks"]])


# ------------------------------------------------------------------ the SP's own kit schema
def test_the_bundle_builds_into_the_sps_own_kit_schema_not_the_workshops(tmp_path):
    out = export(_config(_workshop(tmp_path)))  # hostile: no served_schema in the workshop config
    cfg = _bundle_cfg(out)
    assert cfg["kit_schema"] == "lgk_v1_abc123_sp"
    assert cfg["workshop_kit_schema"] == WORKSHOP_KIT
    # the website keeps reading the schema the workshop served (it defaulted to the kit schema)
    assert cfg["served_schema"] == WORKSHOP_KIT
    assert "sp_kit_schema" not in cfg


def test_an_explicit_sp_kit_schema_is_used(tmp_path):
    cfg = _bundle_cfg(export(_config(_workshop(tmp_path), sp_kit_schema="team_graph", served_schema="web")))
    assert cfg["kit_schema"] == "team_graph" and cfg["workshop_kit_schema"] == WORKSHOP_KIT
    assert cfg["served_schema"] == "web" and "sp_kit_schema" not in cfg


@pytest.mark.parametrize("bad", [WORKSHOP_KIT, "", "lgk_v1_abc123_v3", "Bad-Name"])
def test_an_sp_kit_schema_that_is_the_workshops_or_invalid_is_refused(tmp_path, bad):
    run = _workshop(tmp_path)
    with pytest.raises(ValueError, match="sp_kit_schema"):
        export(_config(run, sp_kit_schema=bad))
    assert not (run / "bundle").exists() and not (run / "bundle.staging").exists()


def test_the_workshops_built_state_does_not_travel(tmp_path):
    out = export(_config(_workshop(tmp_path)))
    shipped = {p.name for p in (out / "run").rglob("*") if p.is_file()}
    assert binding.BINDINGS_FILE not in shipped and "enrich_report.json" not in shipped
    assert {"flags.json", "schema.json", "graph_mapping.yaml"} <= shipped  # the gate's inputs do travel


# --------------------------------------------------------------------- the provision job
def test_the_provision_job_builds_as_the_sp_before_provision(tmp_path):
    out = export(_config(_workshop(tmp_path)))
    assert _keys_and_deps(_job(out, "lgk_provision")) == (["preflight", "build", "provision"],
                                                          [None, "preflight", "build"])


def test_an_enriched_run_also_enriches_as_the_sp_before_provision(tmp_path):
    run = _workshop(tmp_path)
    (run / "enrich.yaml").write_text("methods: [degree]\n")
    out = export(_config(run))
    assert _keys_and_deps(_job(out, "lgk_provision")) == (["preflight", "build", "enrich", "provision"],
                                                          [None, "preflight", "build", "enrich"])
    for t in _job(out, "lgk_provision")["tasks"]:
        params = t["spark_python_task"]["parameters"]
        assert params[params.index("--task") + 1] == t["task_key"]


def test_a_chain_that_does_not_build_the_kit_tables_is_refused(tmp_path):
    """An operator SDP pipeline without `build_mode: sql` runs the stamp step: the SP would label
    tables nobody writes into its schema, and fail every night. Refused at export."""
    run = _workshop(tmp_path)
    with pytest.raises(ValueError, match="build_mode"):
        export(_config(run, sdp_pipeline_id="pipe-1"))
    assert not (run / "bundle").exists()
    # with the kit build kept, it exports (the operator's pipeline runs first)
    out = export(_config(run, sdp_pipeline_id="pipe-1", build_mode="sql"))
    assert _keys_and_deps(_job(out, "lgk_provision"))[0] == ["preflight", "build", "provision"]


# ------------------------------------------------------------ every task: never the workshop schema
ALL_TASKS = ("preflight", "provision", "build", "stamp", "enrich", "sync", "reapply")


def _bundled(tmp_path: Path, **cfg) -> Path:
    run_src = tmp_path / "files" / "run"
    run_src.mkdir(parents=True, exist_ok=True)
    base = {"graph_name": "g", "version": 2, "lakebase_project": "proj", "branch_id": "t-x",
            "lakebase_catalog": "cat", "kit_schema": "k_sp", "workshop_kit_schema": "k"}
    (run_src / "config.json").write_text(json.dumps({k: v for k, v in {**base, **cfg}.items() if v is not None}))
    return run_src


def _argv(run_src, run_dir, task, job_id="4242"):
    argv = ["--src", "/x", "--run-src", str(run_src), "--run-dir", str(run_dir), "--task", task, "--job-id", job_id]
    return argv + (["--sync-owner", "sp"] if task == "preflight" else [])


def _record_every_task(monkeypatch) -> list:
    import lgk.refresh.tasks as rt
    import lgk.steps.advise as adv
    import lgk.steps.preflight as pf
    import lgk.steps.provision as prov
    import lgk.steps.switch as sw

    ran = []
    monkeypatch.setattr(rt, "_task_functions", lambda: {k: (lambda c, k=k: ran.append(k)) for k in ALL_TASKS})
    monkeypatch.setattr(pf, "preflight", lambda c: ran.append("preflight"))
    monkeypatch.setattr(prov, "provision", lambda c: ran.append("provision"))
    monkeypatch.setattr(adv, "advise", lambda c: ran.append("advise"))
    monkeypatch.setattr(sw, "switch", lambda c: ran.append("switch"))
    return ran


@pytest.mark.parametrize("task", ALL_TASKS)
@pytest.mark.parametrize("cfg", [
    {"kit_schema": "k"},                      # hostile: a hand-edited config points at the workshop
    {"workshop_kit_schema": None},            # a bundle from before the SP built its own tables: re-export it
    {"kit_schema": None},                     # no kit schema at all: never a default
    {"kit_schema": "k_v2"},                   # the workshop's version schema
    {"workshop_kit_schema": ""},              # empty is not "none to protect"
    # prover round 1: Unity Catalog names are case-insensitive; a space or a type trick is no new name
    {"kit_schema": "K"},
    {"kit_schema": " k"},
    {"workshop_kit_schema": "K", "kit_schema": "k"},
    {"kit_schema": "k_v12"},                  # a two-digit workshop version schema
    {"kit_schema": ""},
    {"kit_schema": 7},
])
def test_every_bundle_task_refuses_a_config_on_the_workshop_kit_schema(tmp_path, monkeypatch, task, cfg):
    ran = _record_every_task(monkeypatch)
    with pytest.raises(ValueError, match="kit schema"):
        bundle_entry.main(_argv(_bundled(tmp_path, **cfg), tmp_path / "rd", task))
    assert ran == []
    assert not (tmp_path / "rd" / "lgk_manifest.json").exists()  # refused before the job is recorded


@pytest.mark.parametrize("task", ALL_TASKS)
def test_every_bundle_task_runs_on_the_sps_own_kit_schema(tmp_path, monkeypatch, task):
    """Positive control for the refusal above: the same tasks run with a good config."""
    ran = _record_every_task(monkeypatch)
    bundle_entry.main(_argv(_bundled(tmp_path), tmp_path / "rd", task))
    assert ran[0] == task


def test_build_called_directly_with_a_bundle_config_on_the_workshop_schema_is_refused(tmp_path):
    from lgk.steps.build import build

    run = _workshop(tmp_path)
    for cfg in ({"kit_schema": WORKSHOP_KIT, "workshop_kit_schema": WORKSHOP_KIT},
                {"kit_schema": f"{WORKSHOP_KIT}_v2", "workshop_kit_schema": WORKSHOP_KIT},
                {"kit_schema": WORKSHOP_KIT, "workshop_kit_schema": ""},
                {"kit_schema": WORKSHOP_KIT.upper(), "workshop_kit_schema": WORKSHOP_KIT}):
        with pytest.raises(ValueError, match="kit schema"):
            build({**_config(run), **cfg})


# ------------------------------------- condition 1: the SP's build and provision keep every gate
class _NoWorkspace:
    """Any WorkspaceClient() means a gate was passed: these tests must stop before it."""

    def __init__(self, *a, **k):
        raise AssertionError("reached the workspace: a privacy gate was skipped")


def _staged(tmp_path: Path, run: Path, **extra) -> dict:
    out = export(_config(run, **extra), out_dir=str(tmp_path / f"bundle{len(list(tmp_path.glob('bundle*')))}"))
    return bundle_entry._stage_run_folder(out / "run", tmp_path / "state" / "run")


def _edit_mapping(run_folder: str) -> None:
    path = Path(run_folder) / "graph_mapping.yaml"
    path.write_text(path.read_text().replace("properties: [specialty]", "properties: [specialty, name]"))


def _drop_stamp(run_folder: str) -> None:
    from lgk.privacy.gate import STAMP_KEY

    path = Path(run_folder) / "flags.json"
    data = json.loads(path.read_text())
    del data[STAMP_KEY]
    path.write_text(json.dumps(data))


def _lower_floor(run_folder: str) -> None:
    """The decision itself edited by hand (a lower privacy floor), the stamp left as it was."""
    path = Path(run_folder) / "flags.json"
    data = json.loads(path.read_text())
    data["floor"] = int(data["floor"]) - 1
    path.write_text(json.dumps(data))


def _edit_schema(run_folder: str) -> None:
    path = Path(run_folder) / "schema.json"
    path.write_text(path.read_text().replace("specialty", "specialty_text"))


@pytest.mark.parametrize("edit", [_edit_mapping, _drop_stamp, _lower_floor, _edit_schema])
def test_the_sps_build_task_refuses_inputs_changed_after_clearance(tmp_path, monkeypatch, edit):
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", _NoWorkspace)
    out = export(_config(_workshop(tmp_path)))
    edit(str(out / "run"))  # hostile: the deployed files are edited after the operator cleared the gate
    with pytest.raises(PrivacyGateError):
        bundle_entry.main(_argv(out / "run", tmp_path / "rd", "build"))


def _serve_offline(monkeypatch) -> list:
    """provision's owner and version checks pass offline; its stamp checks run for real."""
    import lgk.steps.provision as prov

    seen = []
    monkeypatch.setattr(prov, "require_sync_owner", lambda run_folder: "sp")
    monkeypatch.setattr(prov, "_refuse_existing_version", lambda config: None)
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", _NoWorkspace)
    return seen


def test_the_sps_provision_refuses_without_its_own_build(tmp_path, monkeypatch):
    """The workshop built under the same stamp, but its record does not travel: the SP must build."""
    from lgk.privacy.binding import StaleBuildError
    from lgk.steps.provision import provision

    _serve_offline(monkeypatch)
    staged = _staged(tmp_path, _workshop(tmp_path))
    with pytest.raises(StaleBuildError, match="records no build"):
        provision(staged)  # the step called directly
    with pytest.raises(RuntimeError, match="records no build"):
        bundle_entry._run_provision(staged)  # and through the bundle's provision task


def test_the_sps_provision_refuses_a_build_of_the_workshop_tables(tmp_path, monkeypatch):
    """Hostile: a build record under the CURRENT stamp, but of the workshop's tables."""
    from lgk.privacy.binding import StaleBuildError
    from lgk.steps.provision import provision

    _serve_offline(monkeypatch)
    staged = _staged(tmp_path, _workshop(tmp_path))
    stamp = binding.stamp_id(require_clear(staged["run_folder"]))
    binding.record_build(staged["run_folder"], stamp, [f"cat.{WORKSHOP_KIT}.{t}" for t in binding.BUILT_TABLES])
    with pytest.raises(StaleBuildError, match="did not make cat.lgk_v1_abc123_sp"):
        provision(staged)


def test_the_sps_provision_refuses_a_build_made_before_the_privacy_decision_changed(tmp_path, monkeypatch):
    from lgk.privacy.binding import StaleBuildError
    from lgk.steps.privacy import privacy
    from lgk.steps.provision import provision

    _serve_offline(monkeypatch)
    run = _workshop(tmp_path)
    staged = _staged(tmp_path, run)
    sp_tables = [f"cat.lgk_v1_abc123_sp.{t}" for t in binding.BUILT_TABLES]
    old = binding.stamp_id(require_clear(staged["run_folder"]))
    binding.record_build(staged["run_folder"], old, sp_tables)  # the SP built under the old decision
    # positive control: under the decision it was built with, provision passes every run-folder
    # check and goes on to read the Delta tables' own stamps (the workspace)
    with pytest.raises(AssertionError, match="reached the workspace"):
        provision(staged)
    privacy({"run_folder": str(run), "privacy_floor": 20})  # the operator tightens the decision
    assert binding.stamp_id(require_clear(str(run))) != old
    restaged = _staged(tmp_path, run)  # redeployed: the new flags.json, the SP's old build record
    assert binding.built_stamp(restaged["run_folder"]) == old
    with pytest.raises(StaleBuildError, match="different privacy decision"):
        provision(restaged)
    with pytest.raises(RuntimeError, match="different privacy decision"):
        bundle_entry._run_provision(restaged)


def test_the_sps_build_record_survives_the_next_task_staging(tmp_path):
    """The SP's own record is not overwritten by the bundled files on the next task."""
    staged = _staged(tmp_path, _workshop(tmp_path))
    binding.record_build(staged["run_folder"], "sp-stamp", ["cat.lgk_v1_abc123_sp.nodes"])
    again = bundle_entry._stage_run_folder(Path(staged["run_folder"]).parent.parent / "bundle0" / "run",
                                           Path(staged["run_folder"]))
    assert binding.built_stamp(again["run_folder"]) == "sp-stamp"


# ----------------------------------------------------------------- the one grant list (condition 3)
def _mapping(source_catalog="cat"):
    return type("M", (), {"source_catalog": source_catalog, "source_schema": "fx"})()


def test_the_grant_list_names_the_sps_schemas_and_the_source():
    from lgk import sp_grants

    cfg = {"lakebase_catalog": "cat", "kit_schema": "k_sp", "version": 2, "graph_name": "g"}
    got = [(sec, name, privs) for sec, name, privs, _ in sp_grants.uc_grants(sp_grants.targets(cfg, _mapping()))]
    # written out by hand (not read from the code's list): the same catalog is granted once
    assert got == [("CATALOG", "cat", ("USE CATALOG",)),
                   ("SCHEMA", "cat.k_sp", ("USE SCHEMA", "CREATE TABLE")),
                   ("SCHEMA", "cat.k_sp_v2", ("USE SCHEMA", "CREATE TABLE")),
                   ("SCHEMA", "cat.fx", ("USE SCHEMA", "SELECT"))]


def test_a_source_in_another_catalog_adds_use_catalog_on_it():
    from lgk import sp_grants

    cfg = {"lakebase_catalog": "cat", "kit_schema": "k_sp", "version": 2, "graph_name": "g"}
    got = [(sec, name) for sec, name, _, _ in sp_grants.uc_grants(sp_grants.targets(cfg, _mapping("gold")))]
    assert got == [("CATALOG", "cat"), ("CATALOG", "gold"), ("SCHEMA", "cat.k_sp"), ("SCHEMA", "cat.k_sp_v2"),
                   ("SCHEMA", "gold.fx")]


def test_the_operator_creates_the_sps_schemas_only():
    from lgk import sp_grants

    cfg = {"lakebase_catalog": "cat", "kit_schema": "k_sp", "version": 2, "graph_name": "g"}
    creates = [name for _, name, _, op in sp_grants.uc_grants(sp_grants.targets(cfg, _mapping())) if op]
    assert creates == ["cat.k_sp", "cat.k_sp_v2"]  # never the source schema, never a catalog


def test_the_setup_sql_is_exact():
    from lgk import sp_grants

    cfg = {"lakebase_catalog": "cat", "kit_schema": "k_sp", "version": 2, "graph_name": "g"}
    assert sp_grants.setup_sql(sp_grants.targets(cfg, _mapping()), "g", "sp-1") == [
        "CREATE SCHEMA IF NOT EXISTS cat.k_sp COMMENT 'lgk:graph=g';",
        "CREATE SCHEMA IF NOT EXISTS cat.k_sp_v2 COMMENT 'lgk:graph=g';",
        "GRANT USE CATALOG ON CATALOG cat TO `sp-1`;",
        "GRANT USE SCHEMA, CREATE TABLE ON SCHEMA cat.k_sp TO `sp-1`;",
        "GRANT USE SCHEMA, CREATE TABLE ON SCHEMA cat.k_sp_v2 TO `sp-1`;",
        "GRANT USE SCHEMA, SELECT ON SCHEMA cat.fx TO `sp-1`;",
    ]


def test_the_readme_is_the_grant_list_and_never_grants_the_workshop_schema(tmp_path):
    out = export(_config(_workshop(tmp_path)))
    text = (out / "README.md").read_text()
    for line in ("CREATE SCHEMA IF NOT EXISTS cat.lgk_v1_abc123_sp COMMENT 'lgk:graph=g';",
                 "CREATE SCHEMA IF NOT EXISTS cat.lgk_v1_abc123_sp_v4 COMMENT 'lgk:graph=g';",
                 "GRANT USE SCHEMA, CREATE TABLE ON SCHEMA cat.lgk_v1_abc123_sp TO `<SP application id>`;",
                 "GRANT USE SCHEMA, SELECT ON SCHEMA cat.fx TO `<SP application id>`;",
                 "GRANT USE CATALOG ON CATALOG cat TO `<SP application id>`;"):
        assert line in text, line
    assert "ON SCHEMA cat.lgk_v1_abc123 " not in text  # no grant of any kind on the workshop kit schema
    assert "cat.lgk_v1_abc123 stay the operator's" in text


# ------------------------------------------------------------ prover round 1 survivors
def test_the_bundle_config_scopes_the_sp_preflight_to_building(tmp_path):
    """The bundle's preflight checks the SP's grant list, not the old provision-only checks."""
    from lgk.steps import preflight as pf

    cfg = _bundle_cfg(export(_config(_workshop(tmp_path))))
    assert cfg["sp_builds"] is True
    sc = pf.scope_of({**cfg, "run_folder": str(tmp_path / "run")})
    assert pf._uc_check_names(sc) == ["UC catalog cat", "UC destination cat.lgk_v1_abc123_sp",
                                      "UC destination cat.lgk_v1_abc123_sp_v4", "UC source cat.fx"]


def test_a_refused_chain_writes_no_bundle_file(tmp_path, monkeypatch):
    import lgk.export as ex

    copied = []
    monkeypatch.setattr(ex, "_copy_kit_code", lambda dest: copied.append(dest))
    with pytest.raises(ValueError, match="build_mode"):
        export(_config(_workshop(tmp_path), sdp_pipeline_id="pipe-1"))
    assert copied == []


def test_setup_sql_refuses_an_invalid_graph_name():
    from lgk import sp_grants

    cfg = {"lakebase_catalog": "cat", "kit_schema": "k_sp", "version": 2, "graph_name": "g"}
    with pytest.raises(ValueError):
        sp_grants.setup_sql(sp_grants.targets(cfg, _mapping()), "Bad'G", "sp")


# ------------------------------------------------------------ prover round 2 survivors
@pytest.mark.parametrize("workshop", ["Lgk_W", "lgk-w", "lgk_w_v2"])
def test_an_invalid_workshop_name_is_refused_even_with_a_valid_kit(tmp_path, monkeypatch, workshop):
    ran = _record_every_task(monkeypatch)
    with pytest.raises(ValueError, match="kit schema"):
        bundle_entry.main(_argv(_bundled(tmp_path, workshop_kit_schema=workshop, kit_schema="lgk_w_sp"),
                                tmp_path / "rd", "build"))
    assert ran == [] and not (tmp_path / "rd" / "lgk_manifest.json").exists()


def test_a_kit_name_that_only_contains_a_workshop_version_name_is_accepted():
    from lgk.sp_grants import require_own_kit_schema

    assert require_own_kit_schema({"workshop_kit_schema": "lgk_w", "kit_schema": "lgk_w_v2_sp"}) == "lgk_w_v2_sp"


@pytest.mark.parametrize("cfg,message", [
    ({"kit_schema": "k_sp"}, "exported before"),
    ({"workshop_kit_schema": "k"}, "names no kit schema"),
])
def test_the_guard_says_why(cfg, message):
    from lgk.sp_grants import WorkshopKitSchemaError, require_own_kit_schema

    with pytest.raises(WorkshopKitSchemaError, match=message):
        require_own_kit_schema(cfg)
