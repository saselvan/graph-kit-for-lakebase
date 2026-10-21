"""Preflight: every check performs or reports not_checked; only all-ok passes;
provision refuses any identity but the recorded sync-job owner."""

import json
import uuid
from pathlib import Path

import pytest

from lgk.steps import preflight as pf
from lgk.steps.preflight import PreflightBlocked, SyncOwnerMismatch, preflight, require_sync_owner
from tests.preflight_support import OPERATOR, FakePg, FakeWorkspace

# an SP that is not the owner; its application id is made at run time, never written out
OTHER_SP = str(uuid.uuid4())


def _config(tmp_path: Path, ws=None, pg=None, **kw) -> dict:
    ws = ws or FakeWorkspace()
    pg = pg or FakePg()
    return {"graph_name": "g", "run_folder": str(tmp_path), "lakebase_project": "proj",
            "branch_id": "production",  # named explicitly: there is no default branch (owner "branch a")
            "lakebase_catalog": "cat", "kit_schema": "lgk_g", "sync_owner_user": OPERATOR,
            "workspace_client": ws, "pg_connect": pg.connect, **kw}


def _blocked(config: dict) -> dict:
    with pytest.raises(PreflightBlocked):
        preflight(config)
    return json.loads((Path(config["run_folder"]) / "preflight_result.json").read_text())


def _status(record: dict, name: str) -> str:
    (check,) = [c for c in record["checks"] if c["name"] == name]
    return check["status"]


def test_all_checks_pass_with_full_grants_and_record_the_owner(tmp_path):
    pg = FakePg()
    result = preflight(_config(tmp_path, pg=pg))
    assert result.status == "pass" and result.missing_permissions == []
    record = json.loads((tmp_path / "preflight_result.json").read_text())
    assert record["sync_owner_user"] == OPERATOR and record["operator"] == OPERATOR
    names = [c["name"] for c in record["checks"]]
    # written by hand: the default run asks CAN_USE (no shipped step creates a branch), checks both
    # UC schemas the kit claims, every role provision needs, and the Postgres schema switch creates
    assert names == ["identity", "sync-job owner", "sync mode", "project proj", "CAN_USE on project proj", "branch production",
                     "UC destination cat.lgk_g", "UC destination cat.lgk_g_v1", "synced tables in catalog cat",
                     "branch production reachable", "role reader", "role public_reader",
                     "role coordinator_reader", "Postgres schema lgk_g"]
    assert all(c["status"] == "ok" for c in record["checks"])
    assert pg.connected == [("proj", "production")] and pg.closed == 1


# ---------------------------------------------------------------- synced tables vs catalog
def test_synced_table_check_passes_on_catalog_grants_old_always_missing_bug(tmp_path):
    """The old check reported 'missing' unconditionally; with USE CATALOG + CREATE TABLE it passes."""
    record = json.loads(json.dumps(preflight(_config(tmp_path)).to_dict()))
    assert _status(record, "synced tables in catalog cat") == "ok"


def test_synced_table_check_reads_the_catalog_not_the_project(tmp_path):
    ws = FakeWorkspace()
    # hostile: full project rights and every privilege on ANOTHER catalog; this one lacks USE CATALOG
    ws.catalogs.add("other")
    ws.grants = {("catalog", "cat"): ["CREATE_SCHEMA", "CREATE_TABLE"], ("catalog", "other"): ["ALL_PRIVILEGES"]}
    record = _blocked(_config(tmp_path, ws=ws))
    assert _status(record, "synced tables in catalog cat") == "missing"
    (perm,) = [p for p in record["missing_permissions"] if "synced tables" in p["permission"]]
    assert perm["fix"] == f"GRANT USE CATALOG ON CATALOG cat TO `{OPERATOR}`;"


def test_catalog_owner_through_a_group_has_every_privilege(tmp_path):
    ws = FakeWorkspace(groups=("users", "data-eng"))
    ws.grants = {}
    ws.owners["cat"] = "data-eng"  # hostile: no grant at all, ownership via a group
    assert preflight(_config(tmp_path, ws=ws)).status == "pass"


# ------------------------------------------------------------------- destination schema
def test_existing_schema_needs_use_schema_and_create_table(tmp_path):
    ws = FakeWorkspace(schemas={"cat.lgk_g"})
    # hostile: the catalog grants CREATE SCHEMA, but the schema exists and lacks CREATE TABLE
    ws.grants[("schema", "cat.lgk_g")] = ["USE_SCHEMA"]
    record = _blocked(_config(tmp_path, ws=ws))
    (check,) = [c for c in record["checks"] if c["name"] == "UC destination cat.lgk_g"]
    assert check["status"] == "missing" and check["detail"].startswith("schema exists")
    assert check["fix"] == f"GRANT CREATE TABLE ON SCHEMA cat.lgk_g TO `{OPERATOR}`;"


def test_absent_schema_needs_create_schema_on_the_catalog_and_says_so(tmp_path):
    ws = FakeWorkspace()
    ws.grants = {("catalog", "cat"): ["USE_CATALOG", "CREATE_TABLE"]}  # hostile: no CREATE SCHEMA
    record = _blocked(_config(tmp_path, ws=ws))
    (check,) = [c for c in record["checks"] if c["name"] == "UC destination cat.lgk_g"]
    assert check["status"] == "missing" and check["detail"].startswith("schema absent; the kit creates it")
    ok = preflight(_config(tmp_path / "ok"))
    assert [c for c in ok.checks if c["name"] == "UC destination cat.lgk_g"][0]["detail"] == \
        "schema absent; the kit creates it"


# --------------------------------------------------------------------------- project/branch
def test_missing_project_is_reported_missing(tmp_path):
    record = _blocked(_config(tmp_path, ws=FakeWorkspace(project="elsewhere")))
    assert _status(record, "project proj") == "missing" and record["project_exists"] is False


def test_project_manage_through_group_counts_use_alone_does_not(tmp_path):
    ws = FakeWorkspace(groups=("users", "admins"))
    ws.project_acl = [{"user_name": OPERATOR, "all_permissions": [{"permission_level": "CAN_USE"}]},
                      {"group_name": "admins", "all_permissions": [{"permission_level": "CAN_MANAGE"}]}]
    assert preflight(_config(tmp_path / "a", ws=ws, creates_branches=True)).status == "pass"
    ws2 = FakeWorkspace()
    # hostile: CAN_MANAGE exists, but for another user with the same group-less ACL shape
    ws2.project_acl = [{"user_name": OPERATOR, "all_permissions": [{"permission_level": "CAN_USE"}]},
                       {"user_name": "other@example.com", "all_permissions": [{"permission_level": "CAN_MANAGE"}]}]
    assert _status(_blocked(_config(tmp_path / "b", ws=ws2, creates_branches=True)),
                   "CAN_MANAGE on project proj") == "missing"


def test_source_branch_must_exist_too(tmp_path):
    ws = FakeWorkspace(branches={"t-run"})
    # hostile: the run's own branch exists; its source branch does not
    record = _blocked(_config(tmp_path, ws=ws, branch_id="t-run", source_branch="test-base"))
    assert _status(record, "branch t-run") == "ok"
    assert _status(record, "branch test-base") == "missing"


def test_unreachable_branch_is_missing_and_role_not_checked(tmp_path):
    record = _blocked(_config(tmp_path, pg=FakePg(fails=True)))
    assert _status(record, "branch production reachable") == "missing"
    for name in ("role reader", "role public_reader", "role coordinator_reader", "Postgres schema lgk_g"):
        assert _status(record, name) == "not_checked"


# ---------------------------------------------------------------------------------- role
def test_role_absent_and_operator_cannot_create_it_is_missing(tmp_path):
    record = _blocked(_config(tmp_path, pg=FakePg(can_create_role=False)))
    assert _status(record, "role reader") == "missing"


def test_existing_role_passes_without_createrole(tmp_path):
    # hostile: the operator lacks CREATEROLE, but every role provision needs already exists
    pg = FakePg(roles={"reader", "public_reader", "coordinator_reader"}, can_create_role=False)
    assert preflight(_config(tmp_path, pg=pg)).status == "pass"


# ------------------------------------------------------------------- never vacuous (notebook)
def test_unreachable_workspace_is_not_checked_and_blocks(tmp_path):
    ws = FakeWorkspace()
    ws.down = True  # hostile: identity works, every API call fails with a non-permission error
    record = _blocked(_config(tmp_path, ws=ws))
    statuses = {c["name"]: c["status"] for c in record["checks"]}
    workspace = {n: s for n, s in statuses.items() if n not in ("identity", "sync-job owner", "sync mode")
                 and not n.endswith("reachable") and not n.startswith(("role ", "Postgres schema "))}
    assert len(workspace) == 6 and set(workspace.values()) == {"not_checked"}, workspace
    assert statuses["sync mode"] == "ok"  # reads config only, so a down workspace never hides it
    assert record["status"] == "blocked"


def test_no_client_on_the_notebook_reports_every_check_not_checked(tmp_path):
    """The notebook runner cannot send a client object; a factory that fails yields not_checked."""
    cfg = _config(tmp_path)
    del cfg["workspace_client"], cfg["pg_connect"]
    cfg["preflight_env"] = "tests.test_preflight:_broken_env"  # JSON-safe, crosses to the notebook
    record = _blocked(cfg)
    # "sync mode" reads config only, so it is still checked; every other check is not_checked
    others = [c for c in record["checks"] if c["name"] != "sync mode"]
    assert others and all(c["status"] == "not_checked" for c in others)
    assert [c["status"] for c in record["checks"] if c["name"] == "sync mode"] == ["ok"]
    assert len(record["missing_permissions"]) == len(others)


def _broken_env(config):
    raise RuntimeError("no workspace auth in this process")


def test_preflight_env_factory_is_used_when_no_client_is_injected(tmp_path):
    cfg = _config(tmp_path)
    del cfg["workspace_client"], cfg["pg_connect"]
    cfg["preflight_env"] = "tests.preflight_support:golden_preflight_env"
    cfg["lakebase_project"] = "proj"
    assert preflight(cfg).status == "pass"


def test_default_env_is_live(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(pf, "live_environment", lambda c: seen.append(c) or (None, None))
    cfg = _config(tmp_path)
    del cfg["workspace_client"], cfg["pg_connect"]
    _blocked(cfg)
    assert len(seen) == 1


def test_missing_sync_owner_is_refused(tmp_path):
    cfg = _config(tmp_path)
    del cfg["sync_owner_user"]
    with pytest.raises(KeyError):
        preflight(cfg)
    assert json.loads((tmp_path / "preflight_result.json").read_text())["status"] == "blocked"


# ------------------------------------------------------------------------------------- operator identity
def test_owner_check_passes_for_the_recorded_owner_case_insensitively(tmp_path):
    preflight(_config(tmp_path))
    assert require_sync_owner(tmp_path, OPERATOR.upper()) == OPERATOR


def test_owner_check_refuses_another_identity(tmp_path):
    preflight(_config(tmp_path))
    # hostile: the recorded owner as a prefix of the current identity
    with pytest.raises(SyncOwnerMismatch, match="sync-job owner"):
        require_sync_owner(tmp_path, OPERATOR + ".evil")
    with pytest.raises(SyncOwnerMismatch):
        require_sync_owner(tmp_path, OTHER_SP)  # an SP


def test_owner_check_refuses_without_a_passing_preflight(tmp_path):
    with pytest.raises(SyncOwnerMismatch, match="run preflight first"):
        require_sync_owner(tmp_path, OPERATOR)
    _blocked(_config(tmp_path, pg=FakePg(fails=True)))  # blocked record, owner recorded
    with pytest.raises(SyncOwnerMismatch, match="did not pass"):
        require_sync_owner(tmp_path, OPERATOR)


def test_provision_refuses_a_different_identity_before_any_work(tmp_path, monkeypatch):
    """Through the shipped provision: the identity check runs before anything is made."""
    from lgk.steps import provision as prov

    preflight(_config(tmp_path))
    calls = []
    monkeypatch.setattr(prov, "require_clear", lambda r: {})
    monkeypatch.setattr(prov.binding, "stamp_id", lambda clear: "stamp")
    monkeypatch.setattr(prov, "require_sync_owner",
                        lambda r: pf.require_sync_owner(r, "someone.else@example.com"))
    for name in ("_refuse_existing_version", "_refuse_unbound_build", "_create_synced_tables"):
        monkeypatch.setattr(prov, name, lambda *a, _n=name, **k: calls.append(_n))
    with pytest.raises(SyncOwnerMismatch):
        prov.provision({"graph_name": "g", "run_folder": str(tmp_path)})
    assert calls == []
    monkeypatch.setattr(prov, "require_sync_owner", lambda r: pf.require_sync_owner(r, OPERATOR))
    monkeypatch.setattr(prov, "_mark_pg_version_schema", lambda c: calls.append("mark"))
    monkeypatch.setattr(prov, "_create_version_indexes", lambda c: calls.append("indexes"))
    for name in ("_lock_version_from_reader", "_stage_version_functions", "_prove_reader_isolated"):
        monkeypatch.setattr(prov, name, lambda *a, _n=name, **k: calls.append(_n))
    monkeypatch.setattr(prov.binding, "record_version", lambda *a, **k: calls.append("record"))
    prov.provision({"graph_name": "g", "run_folder": str(tmp_path)})
    assert "_create_synced_tables" in calls and "record" in calls


# ------------------------------------------------------------- self-prover gap tests
def test_owner_other_than_the_operator_blocks(tmp_path):
    # hostile: a valid-looking owner whose grants preflight never checked
    record = _blocked(_config(tmp_path, sync_owner_user="b@example.com"))
    assert _status(record, "sync-job owner") == "missing"


@pytest.mark.parametrize("key", ["sync_owner_user", "lakebase_project", "lakebase_catalog", "branch_id"])
def test_empty_required_value_is_refused(tmp_path, key):
    with pytest.raises(KeyError):
        preflight(_config(tmp_path, **{key: ""}))
    assert json.loads((tmp_path / "preflight_result.json").read_text())["status"] == "blocked"


def test_owner_check_refuses_a_passing_record_with_no_owner(tmp_path):
    (tmp_path / "preflight_result.json").write_text(json.dumps({"status": "pass", "sync_owner_user": None}))
    with pytest.raises(SyncOwnerMismatch):
        require_sync_owner(tmp_path, "None")


def test_owner_check_default_reads_the_live_identity(tmp_path, monkeypatch):
    """No identity passed: the shipped default asks the workspace who runs this process."""
    import databricks.sdk

    from tests.preflight_support import FakeWorkspace as _W

    preflight(_config(tmp_path))
    monkeypatch.setattr(databricks.sdk, "WorkspaceClient", lambda: _W(user="someone.else@example.com"))
    with pytest.raises(SyncOwnerMismatch):
        require_sync_owner(tmp_path)
    monkeypatch.setattr(databricks.sdk, "WorkspaceClient", lambda: _W(user=OPERATOR))
    assert require_sync_owner(tmp_path) == OPERATOR


def test_synced_tables_need_use_catalog(tmp_path):
    ws = FakeWorkspace()
    ws.grants = {("catalog", "cat"): ["CREATE_SCHEMA", "CREATE_TABLE"]}  # hostile: no USE CATALOG
    record = _blocked(_config(tmp_path, ws=ws))
    (perm,) = [p for p in record["missing_permissions"] if "synced tables" in p["permission"]]
    assert perm["fix"] == f"GRANT USE CATALOG ON CATALOG cat TO `{OPERATOR}`;"


def test_existing_schema_needs_use_schema(tmp_path):
    ws = FakeWorkspace(schemas={"cat.lgk_g"})
    ws.grants[("schema", "cat.lgk_g")] = ["CREATE_TABLE"]  # hostile: CREATE TABLE without USE SCHEMA
    record = _blocked(_config(tmp_path, ws=ws))
    (c,) = [c for c in record["checks"] if c["name"] == "UC destination cat.lgk_g"]
    assert c["status"] == "missing" and c["fix"] == f"GRANT USE SCHEMA ON SCHEMA cat.lgk_g TO `{OPERATOR}`;"


def test_missing_catalog_is_missing(tmp_path):
    record = _blocked(_config(tmp_path, ws=FakeWorkspace(catalogs=set())))
    (c,) = [c for c in record["checks"] if c["name"] == "synced tables in catalog cat"]
    assert c["status"] == "missing" and c["detail"] == "catalog not found"


def test_denied_schema_read_is_not_checked_not_absent(tmp_path):
    from databricks.sdk.errors import PermissionDenied

    ws = FakeWorkspace()
    real = ws.api_client.do

    def do(method, path, query=None, body=None):
        if path == "/api/2.1/unity-catalog/schemas/cat.lgk_g":
            raise PermissionDenied("no BROWSE")  # hostile: denied, not absent
        return real(method, path, query=query, body=body)

    ws.api_client.do = do
    assert _status(_blocked(_config(tmp_path, ws=ws)), "UC destination cat.lgk_g") == "not_checked"


def test_query_failure_after_connect_is_missing_and_closes(tmp_path):
    pg = FakePg()

    class BadCursor:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a):
            raise RuntimeError("endpoint suspended mid-query")

    pg.cursor = lambda: BadCursor()
    record = _blocked(_config(tmp_path, pg=pg))
    assert _status(record, "branch production reachable") == "missing"
    assert _status(record, "role reader") == "not_checked" and pg.closed == 1


def test_unknown_identity_does_not_touch_postgres(tmp_path):
    ws, pg = FakeWorkspace(), FakePg()

    def boom():
        raise RuntimeError("token expired")

    ws.current_user.me = boom
    record = _blocked(_config(tmp_path, ws=ws, pg=pg))
    assert _status(record, "branch production reachable") == "not_checked"
    assert _status(record, "role reader") == "not_checked" and pg.connected == []


# ------------------------------------------- owner decision (H1): the bundle's SP, provision only
# an SP's user_name is its application id (assembled from parts, so no UUID sits in the source)
SP_ID = "-".join(["1" * 8, "2" * 4, "4333", "8444", "5" * 12])


def _sp_workspace(grants: dict = None) -> FakeWorkspace:
    """The exported bundle's SP as the operator set it up (bundle README): CAN_USE on the project,
    USE CATALOG, USE SCHEMA + SELECT on the kit schema, USE SCHEMA + CREATE TABLE on the
    pre-made version schema. No CAN_MANAGE, no catalog-level CREATE TABLE."""
    ws = FakeWorkspace(user=SP_ID, schemas={"cat.lgk_g", "cat.lgk_g_v2"})
    ws.project_acl = [{"service_principal_name": SP_ID, "all_permissions": [{"permission_level": "CAN_USE"}]},
                      # hostile: a human holds CAN_MANAGE; it must not count for the SP
                      {"user_name": OPERATOR, "all_permissions": [{"permission_level": "CAN_MANAGE"}]}]
    ws.grants = {("catalog", "cat"): ["USE_CATALOG"], ("schema", "cat.lgk_g"): ["USE_SCHEMA", "SELECT"],
                 ("schema", "cat.lgk_g_v2"): ["USE_SCHEMA", "CREATE_TABLE"], **(grants or {})}
    return ws


def _sp_config(tmp_path: Path, ws=None, **kw) -> dict:
    return _config(tmp_path, ws=ws or _sp_workspace(), sync_owner_user=SP_ID, version=2,
                   **{"creates_branches": False, "provision_only": True, **kw})


def test_the_bundle_sp_passes_with_can_use_and_schema_level_grants(tmp_path):
    result = preflight(_sp_config(tmp_path))
    assert result.status == "pass", result.reason
    names = [c["name"] for c in result.checks]
    assert names == ["identity", "sync-job owner", "sync mode", "project proj", "CAN_USE on project proj", "branch production",
                     "UC source cat.lgk_g", "UC destination cat.lgk_g_v2", "synced tables in catalog cat",
                     "branch production reachable", "role reader", "role public_reader",
                     "role coordinator_reader", "Postgres schema lgk_g"]
    assert json.loads((tmp_path / "preflight_result.json").read_text())["sync_owner_user"] == SP_ID


def test_a_run_that_creates_branches_still_needs_can_manage(tmp_path):
    # hostile: the same SP and grants; only the flag differs. CAN_USE must not be enough.
    record = _blocked(_sp_config(tmp_path, creates_branches=True))
    assert _status(record, "CAN_MANAGE on project proj") == "missing"
    assert "CAN_MANAGE" in [c for c in record["checks"] if c["name"] == "CAN_MANAGE on project proj"][0]["fix"]


def test_creates_branches_defaults_to_can_use(tmp_path):
    """No shipped step creates a branch: a config without the key asks CAN_USE (seen live 2026-10-05)."""
    cfg = _sp_config(tmp_path)
    del cfg["creates_branches"]
    result = preflight(cfg)
    assert result.status == "pass", result.reason
    names = [c["name"] for c in result.checks]
    assert "CAN_USE on project proj" in names and "CAN_MANAGE on project proj" not in names


def test_can_use_held_by_another_identity_does_not_count(tmp_path):
    ws = _sp_workspace()
    ws.project_acl = [{"service_principal_name": "other-sp", "all_permissions": [{"permission_level": "CAN_USE"}]}]
    assert _status(_blocked(_sp_config(tmp_path, ws=ws)), "CAN_USE on project proj") == "missing"


def test_can_manage_satisfies_a_can_use_run(tmp_path):
    ws = _sp_workspace()
    ws.project_acl = [{"service_principal_name": SP_ID, "all_permissions": [{"permission_level": "CAN_MANAGE"}]}]
    assert preflight(_sp_config(tmp_path, ws=ws)).status == "pass"


def test_provision_only_needs_the_version_schema_made_already(tmp_path):
    ws = _sp_workspace()
    ws.schemas = {"cat.lgk_g"}  # hostile: the catalog would let it create schemas; it may not
    ws.grants[("catalog", "cat")] = ["ALL_PRIVILEGES"]
    record = _blocked(_sp_config(tmp_path, ws=ws))
    (c,) = [c for c in record["checks"] if c["name"] == "UC destination cat.lgk_g_v2"]
    assert c["status"] == "missing" and "CREATE SCHEMA cat.lgk_g_v2" in c["fix"]


def test_provision_only_destination_needs_create_table(tmp_path):
    ws = _sp_workspace({("schema", "cat.lgk_g_v2"): ["USE_SCHEMA"]})
    record = _blocked(_sp_config(tmp_path, ws=ws))
    (c,) = [c for c in record["checks"] if c["name"] == "UC destination cat.lgk_g_v2"]
    assert c["status"] == "missing" and "CREATE TABLE" in c["fix"]


def test_provision_only_destination_needs_use_schema(tmp_path):
    ws = _sp_workspace({("schema", "cat.lgk_g_v2"): ["CREATE_TABLE"]})
    assert _status(_blocked(_sp_config(tmp_path, ws=ws)), "UC destination cat.lgk_g_v2") == "missing"


@pytest.mark.parametrize("privs", [["USE_SCHEMA"], ["SELECT"]])
def test_provision_only_source_needs_use_schema_and_select(tmp_path, privs):
    ws = _sp_workspace({("schema", "cat.lgk_g"): privs})
    assert _status(_blocked(_sp_config(tmp_path, ws=ws)), "UC source cat.lgk_g") == "missing"


def test_provision_only_source_must_exist(tmp_path):
    ws = _sp_workspace()
    ws.schemas = {"cat.lgk_g_v2"}
    assert _status(_blocked(_sp_config(tmp_path, ws=ws)), "UC source cat.lgk_g") == "missing"


def test_provision_only_catalog_needs_use_catalog(tmp_path):
    ws = _sp_workspace({("catalog", "cat"): []})
    assert _status(_blocked(_sp_config(tmp_path, ws=ws)), "synced tables in catalog cat") == "missing"


def test_a_building_run_needs_create_table_on_its_kit_schema_not_the_catalog(tmp_path):
    # hostile: the SP's catalog grants (USE CATALOG only) in a run that builds: the catalog check
    # passes (synced tables need CREATE TABLE on their schema only, seen live 2026-10-05); the kit
    # schema, with SELECT but no CREATE TABLE, blocks
    record = _blocked(_sp_config(tmp_path, provision_only=False))
    assert _status(record, "synced tables in catalog cat") == "ok"
    assert _status(record, "UC destination cat.lgk_g") == "missing"


# ---- live 2026-10-03: the CAN_USE SP may not read the project ACL; the credential proves CAN_USE
def test_can_use_is_proven_by_a_credential_when_the_acl_is_not_readable(tmp_path):
    ws = _sp_workspace()
    ws.acl_denied = True
    # hostile: a source branch too; the credential must be for the run's branch
    cfg = _sp_config(tmp_path, ws=ws, branch_id="t-sp", source_branch="production")
    ws.branches = {"t-sp", "production"}
    assert preflight(cfg).status == "pass"
    assert ws.credentials == ["projects/proj/branches/t-sp/endpoints/primary"]  # the run's branch


def test_a_denied_credential_is_missing_can_use(tmp_path):
    from databricks.sdk.errors import PermissionDenied

    ws = _sp_workspace()
    ws.acl_denied, ws.credential_error = True, PermissionDenied("no CAN_USE")
    assert _status(_blocked(_sp_config(tmp_path, ws=ws)), "CAN_USE on project proj") == "missing"


def test_a_failing_credential_call_is_not_checked(tmp_path):
    ws = _sp_workspace()
    ws.acl_denied, ws.credential_error = True, ConnectionError("timeout")
    assert _status(_blocked(_sp_config(tmp_path, ws=ws)), "CAN_USE on project proj") == "not_checked"


def test_a_branch_creating_run_never_falls_back_to_the_credential(tmp_path):
    # hostile: a credential WOULD be issued, but CAN_MANAGE cannot be shown by using the project
    ws = _sp_workspace()
    ws.acl_denied = True
    record = _blocked(_sp_config(tmp_path, ws=ws, creates_branches=True))
    assert _status(record, "CAN_MANAGE on project proj") == "not_checked" and ws.credentials == []


def test_reachability_and_role_use_the_run_branch_not_the_source_branch(tmp_path):
    pg = FakePg()
    ws = FakeWorkspace(branches={"b1", "production"})
    record = preflight(_config(tmp_path, ws=ws, pg=pg, branch_id="b1", source_branch="production"))
    assert pg.connected == [("proj", "b1")]
    assert "branch b1 reachable" in [c["name"] for c in record.checks]


def test_sdk_errors_match_by_class_name_not_identity():
    """Serverless notebooks can hold two copies of databricks.sdk; NotFound/PermissionDenied must
    still be recognised (live parity finding)."""
    from lgk.steps import preflight as pf

    NotFound = type("NotFound", (Exception,), {})          # hostile: a foreign copy of the class
    Sub = type("ResourceDoesNotExist", (NotFound,), {})     # subclass, as the SDK raises
    Denied = type("PermissionDenied", (Exception,), {})
    assert pf._is_not_found(Sub("missing")) and pf._is_not_found(NotFound("x"))
    assert pf._is_denied(Denied("no")) and not pf._is_denied(Sub("x"))
    assert not pf._is_not_found(ValueError("NotFound"))


# ---------------- the bundle's SP builds into its own kit schema
def _sp_build_workspace(grants: dict = None) -> FakeWorkspace:
    """The SP as SKILL/README set it up for building: its own pre-made kit schema and version
    schema (USE SCHEMA + CREATE TABLE), the mapping's source schema (USE SCHEMA + SELECT), USE
    CATALOG. Hostile: it also holds SELECT on the WORKSHOP kit schema, which must not count."""
    ws = FakeWorkspace(user=SP_ID, schemas={"cat.lgk_g", "cat.lgk_g_sp", "cat.lgk_g_sp_v2", "cat.gold"})
    ws.project_acl = [{"service_principal_name": SP_ID, "all_permissions": [{"permission_level": "CAN_USE"}]}]
    ws.grants = {("catalog", "cat"): ["USE_CATALOG"], ("schema", "cat.lgk_g"): ["USE_SCHEMA", "SELECT"],
                 ("schema", "cat.lgk_g_sp"): ["USE_SCHEMA", "CREATE_TABLE"],
                 ("schema", "cat.lgk_g_sp_v2"): ["USE_SCHEMA", "CREATE_TABLE"],
                 ("schema", "cat.gold"): ["USE_SCHEMA", "SELECT"], **(grants or {})}
    return ws


def _sp_build_config(tmp_path: Path, ws=None, mapping: bool = True, **kw) -> dict:
    from tests.test_build_provision_walks import _write_provider_graph_mapping

    if mapping:
        _write_provider_graph_mapping(tmp_path / "graph_mapping.yaml", "cat", "gold")
    return _config(tmp_path, ws=ws or _sp_build_workspace(), sync_owner_user=SP_ID, version=2,
                   **{"creates_branches": False, "provision_only": True, "sp_builds": True,
                      "kit_schema": "lgk_g_sp", "workshop_kit_schema": "lgk_g", **kw})


def _check(record: dict, name: str) -> dict:
    (c,) = [c for c in record["checks"] if c["name"] == name]
    return c


def test_the_building_sp_passes_with_its_own_schemas_and_the_source(tmp_path):
    result = preflight(_sp_build_config(tmp_path))
    assert result.status == "pass", result.reason
    names = [c["name"] for c in result.checks]
    # written by hand: the SP's grant list, in its order; the workshop kit schema is not checked
    assert names == ["identity", "sync-job owner", "sync mode", "project proj", "CAN_USE on project proj", "branch production",
                     "UC catalog cat", "UC destination cat.lgk_g_sp", "UC destination cat.lgk_g_sp_v2",
                     "UC source cat.gold", "branch production reachable", "role reader", "role public_reader",
                     "role coordinator_reader", "Postgres schema lgk_g_sp"]


@pytest.mark.parametrize("name,key,privs,fix", [
    ("UC destination cat.lgk_g_sp", ("schema", "cat.lgk_g_sp"), ["USE_SCHEMA"],
     f"GRANT CREATE TABLE ON SCHEMA cat.lgk_g_sp TO `{SP_ID}`;"),
    ("UC destination cat.lgk_g_sp", ("schema", "cat.lgk_g_sp"), ["CREATE_TABLE"],
     f"GRANT USE SCHEMA ON SCHEMA cat.lgk_g_sp TO `{SP_ID}`;"),
    ("UC destination cat.lgk_g_sp_v2", ("schema", "cat.lgk_g_sp_v2"), ["USE_SCHEMA"],
     f"GRANT CREATE TABLE ON SCHEMA cat.lgk_g_sp_v2 TO `{SP_ID}`;"),
    ("UC source cat.gold", ("schema", "cat.gold"), ["USE_SCHEMA"],
     f"GRANT SELECT ON SCHEMA cat.gold TO `{SP_ID}`;"),
    ("UC source cat.gold", ("schema", "cat.gold"), ["SELECT"],
     f"GRANT USE SCHEMA ON SCHEMA cat.gold TO `{SP_ID}`;"),
    ("UC catalog cat", ("catalog", "cat"), [],
     f"GRANT USE CATALOG ON CATALOG cat TO `{SP_ID}`;"),
])
def test_each_missing_sp_grant_blocks_with_its_grant(tmp_path, name, key, privs, fix):
    record = _blocked(_sp_build_config(tmp_path, ws=_sp_build_workspace({key: privs})))
    c = _check(record, name)
    assert c["status"] == "missing" and c["fix"] == fix, c


def test_an_absent_sp_kit_schema_blocks_and_the_operator_creates_it(tmp_path):
    ws = _sp_build_workspace({("catalog", "cat"): ["ALL_PRIVILEGES"]})  # hostile: it could create it itself
    ws.schemas.discard("cat.lgk_g_sp")
    c = _check(_blocked(_sp_build_config(tmp_path, ws=ws)), "UC destination cat.lgk_g_sp")
    assert c["status"] == "missing"
    assert c["fix"].startswith("as the operator: CREATE SCHEMA IF NOT EXISTS cat.lgk_g_sp COMMENT 'lgk:graph=g';")


def test_an_absent_source_schema_blocks_and_is_never_created(tmp_path):
    ws = _sp_build_workspace()
    ws.schemas.discard("cat.gold")
    c = _check(_blocked(_sp_build_config(tmp_path, ws=ws)), "UC source cat.gold")
    assert c["status"] == "missing" and "CREATE SCHEMA" not in c["fix"]


def test_a_building_sp_without_a_mapping_blocks(tmp_path):
    record = _blocked(_sp_build_config(tmp_path, mapping=False))
    assert _status(record, "UC grants of the service principal") == "missing"


def test_a_building_sp_on_the_workshop_kit_schema_blocks(tmp_path):
    # hostile: a config that would build into the workshop's kit schema, with every grant there
    ws = _sp_build_workspace({("schema", "cat.lgk_g"): ["ALL_PRIVILEGES"]})
    record = _blocked(_sp_build_config(tmp_path, ws=ws, kit_schema="lgk_g"))
    assert _status(record, "UC grants of the service principal") == "missing"



def test_a_building_sp_without_a_workshop_kit_schema_blocks(tmp_path):
    """Prover round 1: an older bundle config with sp_builds is blocked, never the old checks."""
    cfg = _sp_build_config(tmp_path)
    del cfg["workshop_kit_schema"]
    record = _blocked(cfg)
    names = [c["name"] for c in record["checks"]]
    assert _status(record, "UC grants of the service principal") == "missing"
    assert "synced tables in catalog cat" not in names and "UC source cat.lgk_g_sp" not in names


def test_a_denied_schema_lookup_is_not_checked_not_absent(tmp_path):
    from databricks.sdk.errors import PermissionDenied

    ws = _sp_build_workspace()
    real = ws.api_client.do

    def do(method, path, query=None, body=None):
        if path.endswith("/schemas/cat.lgk_g_sp"):
            raise PermissionDenied("no access")
        return real(method, path, query, body)

    ws.api_client.do = do
    c = _check(_blocked(_sp_build_config(tmp_path, ws=ws)), "UC destination cat.lgk_g_sp")
    assert c["status"] == "not_checked" and "CREATE SCHEMA" not in (c["fix"] or "")


# ------------- seen live 2026-10-05: a CAN USE identity, gaps 1-6 of preflight
DB = "databricks_postgres"
ROLES = ("reader", "public_reader", "coordinator_reader")


def _schema_level_workspace() -> FakeWorkspace:
    """The spike's grants for a building run: both UC schemas pre-made by the operator, USE SCHEMA +
    CREATE TABLE on each, and USE CATALOG only on the catalog (no CREATE SCHEMA, no CREATE TABLE)."""
    ws = FakeWorkspace(schemas={"cat.lgk_g", "cat.lgk_g_v1"})
    ws.grants = {("catalog", "cat"): ["USE_CATALOG"], ("schema", "cat.lgk_g"): ["USE_SCHEMA", "CREATE_TABLE"],
                 ("schema", "cat.lgk_g_v1"): ["USE_SCHEMA", "CREATE_TABLE"]}
    return ws


def test_gap2_premade_schemas_with_schema_level_create_table_pass(tmp_path):
    """The false block: the spike's build and provision ran with schema-level CREATE TABLE only."""
    result = preflight(_config(tmp_path, ws=_schema_level_workspace()))
    assert result.status == "pass", result.reason


def test_gap2_absent_kit_schema_without_create_schema_blocks(tmp_path):
    ws = _schema_level_workspace()
    ws.schemas.discard("cat.lgk_g")
    ws.grants[("catalog", "cat")] = ["USE_CATALOG", "CREATE_TABLE"]  # hostile: catalog CREATE TABLE is not CREATE SCHEMA
    c = _check(_blocked(_config(tmp_path, ws=ws)), "UC destination cat.lgk_g")
    assert c["status"] == "missing" and c["fix"] == f"GRANT CREATE SCHEMA ON CATALOG cat TO `{OPERATOR}`;"


def test_gap2_absent_version_schema_without_create_schema_blocks(tmp_path):
    """provision claims (creates) the UC version schema too: an absent one needs CREATE SCHEMA."""
    ws = _schema_level_workspace()
    ws.schemas.discard("cat.lgk_g_v1")  # hostile: the kit schema is fine; only the version schema is absent
    record = _blocked(_config(tmp_path, ws=ws))
    assert _status(record, "UC destination cat.lgk_g") == "ok"
    c = _check(record, "UC destination cat.lgk_g_v1")
    assert c["status"] == "missing" and c["fix"] == f"GRANT CREATE SCHEMA ON CATALOG cat TO `{OPERATOR}`;"


def test_gap2_existing_version_schema_needs_create_table(tmp_path):
    ws = _schema_level_workspace()
    ws.grants[("schema", "cat.lgk_g_v1")] = ["USE_SCHEMA"]
    ws.grants[("catalog", "cat")] = ["USE_CATALOG", "CREATE_SCHEMA", "CREATE_TABLE"]  # hostile: the catalog would do
    c = _check(_blocked(_config(tmp_path, ws=ws)), "UC destination cat.lgk_g_v1")
    assert c["status"] == "missing" and c["fix"] == f"GRANT CREATE TABLE ON SCHEMA cat.lgk_g_v1 TO `{OPERATOR}`;"


def test_gap2_version_schema_follows_the_config_version(tmp_path):
    ws = _schema_level_workspace()
    ws.schemas = {"cat.lgk_g", "cat.lgk_g_v1"}  # hostile: v1 is pre-made; the run is v3
    record = _blocked(_config(tmp_path, ws=ws, version=3))
    assert _status(record, "UC destination cat.lgk_g_v3") == "missing"


@pytest.mark.parametrize("absent", ROLES)
def test_gap3_each_missing_role_blocks_without_createrole(tmp_path, absent):
    # hostile: the other two roles exist; only one is absent
    pg = FakePg(roles=set(ROLES) - {absent}, can_create_role=False)
    record = _blocked(_config(tmp_path, pg=pg))
    for role in ROLES:
        assert _status(record, f"role {role}") == ("missing" if role == absent else "ok"), role
    c = _check(record, f"role {absent}")
    assert c["fix"].startswith(f"as the project owner, on branch production (or its parent): "
                               f"CREATE ROLE {absent} WITH LOGIN;"), c["fix"]


def test_gap3_reader_present_public_reader_missing_names_public_reader(tmp_path):
    record = _blocked(_config(tmp_path, pg=FakePg(roles={"reader", "coordinator_reader"}, can_create_role=False)))
    assert "role public_reader: missing" in record["reason"] and "role reader:" not in record["reason"]


def test_gap3_role_names_match_exactly(tmp_path):
    # hostile: case and quoting variants of every role exist; none of them is the role provision uses
    pg = FakePg(roles={"Reader", "PUBLIC_READER", '"coordinator_reader"', "reader "}, can_create_role=False)
    record = _blocked(_config(tmp_path, pg=pg))
    assert [_status(record, f"role {r}") for r in ROLES] == ["missing"] * 3


def test_gap3_createrole_passes_for_missing_roles(tmp_path):
    record = preflight(_config(tmp_path, pg=FakePg(roles=set(), can_create_role=True)))
    for role in ROLES:
        c = [c for c in record.checks if c["name"] == f"role {role}"][0]
        assert c["status"] == "ok" and c["detail"] == "absent; this identity can create it"


def test_gap4_no_create_on_the_database_blocks_with_all_roles_present(tmp_path):
    pg = FakePg(roles=set(ROLES), can_create_role=False, db_create=False)
    record = _blocked(_config(tmp_path, pg=pg))
    assert [_status(record, f"role {r}") for r in ROLES] == ["ok"] * 3
    c = _check(record, "Postgres schema lgk_g")
    assert c["status"] == "missing"
    assert c["fix"].startswith(f'as the project owner: GRANT CREATE ON DATABASE {DB} TO "{OPERATOR}";'), c["fix"]


def test_gap4_the_database_name_is_read_not_assumed(tmp_path):
    pg = FakePg(roles=set(ROLES), db_create=False, database="other_db")  # hostile: not databricks_postgres
    c = _check(_blocked(_config(tmp_path, pg=pg)), "Postgres schema lgk_g")
    assert c["fix"].startswith(f'as the project owner: GRANT CREATE ON DATABASE other_db TO "{OPERATOR}";')


def test_gap4_an_existing_served_schema_needs_create_on_it_not_on_the_database(tmp_path):
    """switch's claim_pg_schema creates the served schema only when it is absent."""
    ok = FakePg(roles=set(ROLES), db_create=False, schemas={"lgk_g": True})
    assert preflight(_config(tmp_path / "a", pg=ok)).status == "pass"
    denied = FakePg(roles=set(ROLES), db_create=True, schemas={"lgk_g": False})  # hostile: DB CREATE is no help
    c = _check(_blocked(_config(tmp_path / "b", pg=denied)), "Postgres schema lgk_g")
    assert c["status"] == "missing" and c["fix"].startswith(f'as the project owner: GRANT CREATE ON SCHEMA lgk_g TO "{OPERATOR}";')


def test_gap4_the_served_schema_is_the_configured_one(tmp_path):
    pg = FakePg(roles=set(ROLES), db_create=False, schemas={"lgk_g": True})  # hostile: the kit schema is creatable
    record = _blocked(_config(tmp_path, pg=pg, served_schema="web"))
    assert _status(record, "Postgres schema web") == "missing"


SP_SETUP = ("CREATE EXTENSION IF NOT EXISTS databricks_auth; "
            f"SELECT databricks_create_role('{SP_ID}', 'service_principal'); "
            "CREATE ROLE reader WITH LOGIN; CREATE ROLE public_reader WITH LOGIN; "
            "CREATE ROLE coordinator_reader WITH LOGIN; "
            f'GRANT CREATE ON DATABASE {DB} TO "{SP_ID}";')


def test_gap5_a_can_use_sp_gets_the_owner_setup_in_order(tmp_path):
    pg = FakePg(roles=set(), can_create_role=False, db_create=False, user=SP_ID)
    record = _blocked(_sp_config(tmp_path, pg=pg))
    for name in [f"role {r}" for r in ROLES] + ["Postgres schema lgk_g"]:
        c = _check(record, name)
        assert c["status"] == "missing" and c["fix"].endswith(
            f"One-time owner setup for {SP_ID}, in order: {SP_SETUP}"), (name, c["fix"])


def test_gap5_a_user_identity_is_quoted_and_typed_user(tmp_path):
    user = 'Jo.O"Neil@Example.com'  # hostile: mixed case, a double quote, an email not an app id
    pg = FakePg(roles=set(), can_create_role=False, user=user)
    c = _check(_blocked(_config(tmp_path, pg=pg)), "role reader")
    assert "SELECT databricks_create_role('Jo.O\"Neil@Example.com', 'user');" in c["fix"]
    assert 'GRANT CREATE ON DATABASE databricks_postgres TO "Jo.O""Neil@Example.com";' in c["fix"]


def test_gap5_an_unreachable_branch_gets_the_owner_setup(tmp_path):
    """No Postgres role: login is refused, so the reachability fix carries the whole setup."""
    ws = _sp_workspace()
    c = _check(_blocked(_sp_config(tmp_path, ws=ws, pg=FakePg(fails=True))), "branch production reachable")
    assert c["fix"].endswith(f"in order: {SP_SETUP}"), c["fix"]


def _all_missing(tmp_path):
    ws = FakeWorkspace(schemas={"cat.lgk_g"})
    ws.grants, ws.acl_denied = {}, True
    ws.credential_error = __import__("databricks.sdk.errors", fromlist=["x"]).PermissionDenied("no")
    pg = FakePg(roles=set(), can_create_role=False, db_create=False, schemas={"lgk_g": False})
    return ws, pg


@pytest.mark.parametrize("scenario", ["pass", "all_missing", "provision_only", "sp_builds"])
def test_gap6_preflight_only_reads(tmp_path, scenario):
    """Every statement preflight sends Postgres is a SELECT, every API call a GET (no create, alter, grant)."""
    import re

    if scenario == "pass":
        ws, pg = FakeWorkspace(), FakePg()
        preflight(_config(tmp_path, ws=ws, pg=pg))
    elif scenario == "all_missing":
        ws, pg = _all_missing(tmp_path)
        _blocked(_config(tmp_path, ws=ws, pg=pg))
    elif scenario == "provision_only":
        ws, pg = _sp_workspace(), FakePg(roles=set(), can_create_role=False, db_create=False)
        _blocked(_sp_config(tmp_path, ws=ws, pg=pg))
    else:
        ws, pg = _sp_build_workspace(), FakePg(roles=set(), can_create_role=False, db_create=False)
        _blocked(_sp_build_config(tmp_path, ws=ws, pg=pg))
    assert pg.sql and all(re.fullmatch(r"SELECT\s[^;]*", q) for q in pg.sql), pg.sql
    writes = re.compile(r"\b(CREATE|ALTER|GRANT|REVOKE|DROP|INSERT|UPDATE|DELETE|COMMENT|SET)\b", re.I)
    assert not [q for q in pg.sql if writes.search(q.replace("'CREATE'", ""))], pg.sql
    assert ws.api_client.methods and set(ws.api_client.methods) == {"GET"}


def test_gap3_preflight_checks_every_role_provision_ensures():
    """Drift guard: the roles preflight checks are the roles provision creates (and their SQL)."""
    from lgk.steps import provision

    assert set(pf.KIT_ROLES) == set(provision.ROLE_SETUP_SQL) == set(ROLES)
    setup = pf.owner_setup_sql(SP_ID)
    assert all(sql in setup for sql in provision.ROLE_SETUP_SQL.values())
