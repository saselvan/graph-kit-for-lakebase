"""Preflight step: confirm the environment, list each missing permission with its fix.

Checks, each reported as ok / missing / not_checked:
- identity: who runs this (the operator); sync-job owner: the operator is `sync_owner_user`.
- project: the Lakebase project exists; project permission: CAN_USE (default: no shipped step
  creates a branch, so a run works on an existing one), or CAN_MANAGE when a caller sets
  `creates_branches: true` (only that level may create branches).
- branch: the branch the run uses (and `source_branch`, when set) exists; reachable: Postgres
  on it answers as the identity.
- roles: `reader`, `public_reader` and `coordinator_reader` (all three provision ensures) each
  exist on the branch, or the identity can create roles (CREATEROLE or superuser).
- Postgres schema: the served schema exists with CREATE on it, or the identity has CREATE on the
  database so switch can create it (the Postgres version schema is made by the sync pipeline).
- UC destination: the kit schema (build writes its Delta tables there) and the version
  schema (provision's synced tables land there) each exist with USE SCHEMA + CREATE TABLE, or the
  catalog grants CREATE SCHEMA so the kit can create it (the detail says which). Live 2026-10-05:
  schema-level CREATE TABLE is all build and provision need; the catalog's CREATE TABLE is not.
- synced tables: USE CATALOG on the destination catalog.

Every check is a pure read (GET calls and SELECT statements): preflight creates, alters and
grants nothing.

A Postgres check that fails names the one statement its project owner runs, then the whole
one-time owner setup for this identity, in order:
the databricks_auth extension, `databricks_create_role`, the three roles, CREATE on the database.
Not checked: the workspace's `databricks-sql-access` entitlement the warehouse needs (an identity
can hold it through a group, which a non-admin cannot read reliably; the docs list it).

`provision_only` (default False; the exported bundle's SP): the run builds nothing and cannot
create schemas, so the kit schema is checked as a SOURCE (USE SCHEMA + SELECT), the version
schema the synced tables land in must already exist with USE SCHEMA + CREATE TABLE, and
the catalog needs USE CATALOG (CREATE TABLE is the destination schema's).

`sp_builds` (with provision_only; the exported bundle's SP): the SP also BUILDS, into a
kit schema of its own, so the UC checks are exactly lgk.sp_grants' list for this run (the one the
README and SKILL show): USE CATALOG, USE SCHEMA + CREATE TABLE on its pre-made kit and version
schemas, USE SCHEMA + SELECT on the mapping's source schema. The workshop's kit schema is not
checked (the SP never reads it); a config on it, or with no mapping, is blocked.

Never vacuous: a check that cannot be performed is `not_checked`, and only all-ok passes. A
blocked preflight writes its result and raises, so no later step runs (never fails halfway).
It records `sync_owner_user`; `provision` refuses any other identity.
"""

import importlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

OK, MISSING, NOT_CHECKED = "ok", "missing", "not_checked"
PASS, BLOCKED = "pass", "blocked"  # PreflightResult.status (preflight_result.json)
READER_ROLE = "reader"
# Every login role provision ensures (lgk.steps.provision ROLE_SETUP_SQL: reader + FUNCTION_ROLES)
KIT_ROLES = (READER_ROLE, "public_reader", "coordinator_reader")
DEFAULT_DATABASE = "databricks_postgres"  # the database lgk.walks.nb_runtime.connect opens
_APP_ID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
RESULT_FILE = "preflight_result.json"
DEFAULT_ENV = "lgk.steps.preflight:live_environment"


class PreflightBlocked(RuntimeError):
    """Preflight found a missing or unchecked precondition."""


class SyncOwnerMismatch(RuntimeError):
    """provision runs as an identity other than the recorded sync-job owner."""


@dataclass
class Check:
    name: str
    status: str  # OK | MISSING | NOT_CHECKED
    detail: str
    fix: str = ""


@dataclass
class PreflightResult:
    """Result of preflight validation."""

    status: str  # PASS or BLOCKED
    reason: Optional[str] = None
    missing_permissions: List[Dict[str, str]] = field(default_factory=list)
    project_exists: bool = False
    schema_creatable: bool = False
    sync_owner_user: Optional[str] = None
    operator: Optional[str] = None
    checks: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Identity:
    user_name: str
    groups: Tuple[str, ...]


# ------------------------------------------------------------------------- environment
def live_environment(config: Dict[str, Any]) -> Tuple[Any, Callable[[str, str], Any]]:
    """(workspace client, connect(project, branch)) as the identity running this process."""
    from databricks.sdk import WorkspaceClient

    from lgk.walks.nb_runtime import connect

    return WorkspaceClient(), connect


def _environment(config: Dict[str, Any]) -> Tuple[Any, Optional[Callable[[str, str], Any]]]:
    """The clients the checks use: injected objects, else the `preflight_env` factory
    (a JSON-safe `module:qualname`, so it crosses to the notebook runner), else live.
    A factory that fails yields no client: every check then reports not_checked."""
    if config.get("workspace_client") is not None:
        if config.get("pg_connect") is not None:
            return config["workspace_client"], config["pg_connect"]
        from lgk.walks.nb_runtime import connect

        return config["workspace_client"], connect
    ref = config.get("preflight_env", DEFAULT_ENV)
    try:
        module, qualname = ref.split(":", 1)
        return getattr(importlib.import_module(module), qualname)(config)
    except Exception as e:  # noqa: BLE001 - reported as not_checked, never as pass
        logger.warning(f"preflight cannot reach the workspace ({ref}): {e}")
        return None, None


# ------------------------------------------------------------------------------ helpers
def _is_sdk_error(e: Exception, name: str) -> bool:
    """Match an SDK error class by name across its MRO, not by identity: on a serverless notebook
    the raising client and this module can hold different copies of databricks.sdk (verified live:
    a NotFound failed isinstance there)."""
    return any(c.__name__ == name for c in type(e).__mro__)


def _is_not_found(e: Exception) -> bool:
    return _is_sdk_error(e, "NotFound")


def _is_denied(e: Exception) -> bool:
    return _is_sdk_error(e, "PermissionDenied")


def _not_checked(name: str, why: Any) -> Check:
    why = f"{type(why).__name__}: {why}" if isinstance(why, BaseException) else why
    return Check(name, NOT_CHECKED, f"not checked: {why}",
                 "re-run preflight where the workspace and the branch are reachable")


def _privileges(w: Any, securable: str, full_name: str, me: Identity) -> set:
    """Effective UC privileges of `me` (direct and through groups) on one securable; the
    owner (me or one of my groups) has them all."""
    info = w.api_client.do("GET", f"/api/2.1/unity-catalog/{securable}s/{full_name}")
    if info.get("owner") in (me.user_name, *me.groups):
        return {"ALL_PRIVILEGES"}
    d = w.api_client.do("GET", f"/api/2.1/unity-catalog/effective-permissions/{securable}/{full_name}",
                        query={"principal": me.user_name})
    return {p["privilege"] for a in d.get("privilege_assignments", []) for p in a.get("privileges", [])}


def _lacking(privs: set, needed: Iterable[str]) -> List[str]:
    return [] if "ALL_PRIVILEGES" in privs else [p for p in needed if p not in privs]


# -------------------------------------------------------------------------------- checks
def _check_identity(w: Any) -> Tuple[Check, Optional[Identity]]:
    if w is None:
        return _not_checked("identity", "no workspace client"), None
    try:
        me = w.current_user.me()
        groups = tuple(g.display for g in (me.groups or []) if getattr(g, "display", None))
        return Check("identity", OK, f"operator {me.user_name}"), Identity(me.user_name, groups)
    except Exception as e:  # noqa: BLE001
        return _not_checked("identity", e), None


def _check_project(w: Any, project: str) -> Check:
    name = f"project {project}"
    try:
        w.api_client.do("GET", f"/api/2.0/postgres/projects/{project}")
        return Check(name, OK, "exists")
    except Exception as e:  # noqa: BLE001
        if _is_not_found(e) or _is_denied(e):
            return Check(name, MISSING, f"not found or not visible: {e}",
                         f"create Lakebase project {project}, or ask its owner for CAN_USE")
        return _not_checked(name, e)


def _project_level(creates_branches: bool) -> Tuple[str, Tuple[str, ...]]:
    """(level named in the check, levels that satisfy it): CAN_MANAGE creates branches; a run on
    an existing branch needs CAN_USE, which CAN_MANAGE includes."""
    return ("CAN_MANAGE", ("CAN_MANAGE",)) if creates_branches else ("CAN_USE", ("CAN_USE", "CAN_MANAGE"))


def _can_use_by_credential(w: Any, project: str, branch: str, name: str) -> Check:
    """CAN_USE when the ACL is not readable: only CAN_MANAGE may read a project's ACL (the bundle
    SP, with CAN_USE, is denied it, live 2026-10-03). The positive proof is the use itself: a
    database credential for the branch's endpoint. Denied is missing; anything else not checked."""
    try:
        w.postgres.generate_database_credential(endpoint=f"projects/{project}/branches/{branch}/endpoints/primary")
    except Exception as e:  # noqa: BLE001
        if _is_denied(e):
            return Check(name, MISSING, f"denied a database credential on branch {branch}: {e}",
                         f"grant this identity CAN_USE on Lakebase project {project} (Permissions tab)")
        return _not_checked(name, e)
    return Check(name, OK, f"ACL readable only with CAN_MANAGE; a database credential on branch {branch} was issued")


def _check_project_permission(w: Any, project: str, me: Identity, creates_branches: bool = True,
                              branch: str = "production") -> Check:
    """CAN_MANAGE on the project when the run creates branches (and roles on them); else CAN_USE."""
    level, enough = _project_level(creates_branches)
    name = f"{level} on project {project}"
    try:
        acl = w.api_client.do("GET", f"/api/2.0/permissions/database-projects/{project}")
    except Exception as e:  # noqa: BLE001
        if not creates_branches and _is_denied(e):
            return _can_use_by_credential(w, project, branch, name)
        return _not_checked(name, e)
    mine = {p["permission_level"] for a in acl.get("access_control_list", [])
            if a.get("user_name") == me.user_name or a.get("service_principal_name") == me.user_name
            or a.get("group_name") in me.groups
            for p in a.get("all_permissions", [])}
    if mine & set(enough):
        return Check(name, OK, "granted")
    return Check(name, MISSING, f"{me.user_name} has {sorted(mine) or 'no permission'}",
                 f"grant {me.user_name} {level} on Lakebase project {project} (Permissions tab)")


def _check_branch(w: Any, project: str, branch: str) -> Check:
    name = f"branch {branch}"
    try:
        w.api_client.do("GET", f"/api/2.0/postgres/projects/{project}/branches/{branch}")
        return Check(name, OK, "exists")
    except Exception as e:  # noqa: BLE001
        if _is_not_found(e):
            return Check(name, MISSING, f"not found in project {project}",
                         f"create branch {branch} in project {project}, or set branch_id to an existing one")
        return _not_checked(name, e)


def _pg_ident(name: str) -> str:
    """A Postgres identifier, double-quoted (an email or an app id is not a bare name)."""
    return '"' + name.replace('"', '""') + '"'


def _pg_name(name: str) -> str:
    """A database or schema name as written in SQL: bare when it is a plain lower-case name."""
    from lgk import ident

    return name if ident.is_valid(name) else _pg_ident(name)


def _pg_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def owner_setup_sql(identity: str, database: str = DEFAULT_DATABASE) -> str:
    """The project owner's one-time Postgres setup for `identity`, in order (live, 2026-10-05: a
    CAN USE service principal ran every step after exactly this)."""
    kind = "service_principal" if _APP_ID.fullmatch(identity) else "user"
    return " ".join(["CREATE EXTENSION IF NOT EXISTS databricks_auth;",
                     f"SELECT databricks_create_role({_pg_literal(identity)}, {_pg_literal(kind)});",
                     *(f"CREATE ROLE {role} WITH LOGIN;" for role in KIT_ROLES),
                     f"GRANT CREATE ON DATABASE {_pg_name(database)} TO {_pg_ident(identity)};"])


def _owner_fix(statement: str, identity: str, database: str) -> str:
    return f"{statement} One-time owner setup for {identity}, in order: {owner_setup_sql(identity, database)}"


def _role_checks(cur: Any, branch: str, identity: str, database: str) -> List[Check]:
    """Each role provision ensures exists (exact name), or this identity may create roles."""
    cur.execute("SELECT rolcreaterole OR rolsuper FROM pg_roles WHERE rolname = current_user")
    row = cur.fetchone()
    can_create = bool(row and row[0])
    checks = []
    for role in KIT_ROLES:
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
        if cur.fetchone():
            checks.append(Check(f"role {role}", OK, "exists"))
        elif can_create:
            checks.append(Check(f"role {role}", OK, "absent; this identity can create it"))
        else:
            checks.append(Check(f"role {role}", MISSING, "absent, and this identity cannot create roles",
                                _owner_fix(f"as the project owner, on branch {branch} (or its parent): "
                                           f"CREATE ROLE {role} WITH LOGIN;", identity, database)))
    return checks


def _served_schema_check(cur: Any, served: str, identity: str, database: str) -> Check:
    """switch creates the served schema when it is absent (CREATE on the database), then creates
    views and functions in it (CREATE on the schema)."""
    name = f"Postgres schema {served}"
    cur.execute("SELECT has_schema_privilege(oid, 'CREATE') FROM pg_namespace WHERE nspname = %s", (served,))
    row = cur.fetchone()
    if row is not None:
        if row[0]:
            return Check(name, OK, "exists; this identity can create in it")
        return Check(name, MISSING, "exists, and this identity lacks CREATE on it",
                     _owner_fix(f"as the project owner: GRANT CREATE ON SCHEMA {_pg_name(served)} TO {_pg_ident(identity)};",
                                identity, database))
    cur.execute("SELECT has_database_privilege(current_database(), 'CREATE')")
    row = cur.fetchone()
    if row and row[0]:
        return Check(name, OK, f"absent; this identity can create it (CREATE on database {database})")
    return Check(name, MISSING, f"absent, and this identity lacks CREATE on database {database}",
                 _owner_fix(f"as the project owner: GRANT CREATE ON DATABASE {_pg_name(database)} TO {_pg_ident(identity)};",
                            identity, database))


def _pg_check_names(served: str) -> List[str]:
    return [f"role {r}" for r in KIT_ROLES] + [f"Postgres schema {served}"]


def _check_postgres(connect: Optional[Callable], project: str, branch: str, served: str,
                    me: Optional[Identity]) -> List[Check]:
    """Reachable as this identity; then the roles and the served schema (SELECTs only)."""
    reach = f"branch {branch} reachable"
    later = _pg_check_names(served)
    if connect is None:
        return [_not_checked(reach, "no Postgres connector")] + [_not_checked(n, "no connection") for n in later]
    try:
        conn = connect(project, branch)
    except Exception as e:  # noqa: BLE001
        who = me.user_name if me else "<identity>"
        return [Check(reach, MISSING, f"cannot connect as this identity: {e}",
                      f"start the branch's primary endpoint. If {who} has no Postgres role on {branch}, the "
                      f"project owner runs once, on the branch or its parent, in order: "
                      f"{owner_setup_sql(who, DEFAULT_DATABASE)}")] + \
               [_not_checked(n, "branch not reachable") for n in later]
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
            cur.execute("SELECT current_user, current_database()")
            identity, database = cur.fetchone()
            return ([Check(reach, OK, f"connected as {identity}")] + _role_checks(cur, branch, identity, database)
                    + [_served_schema_check(cur, served, identity, database)])
    except Exception as e:  # noqa: BLE001
        return [Check(reach, MISSING, f"query failed: {e}", "check the branch endpoint")] + \
               [_not_checked(n, e) for n in later]
    finally:
        conn.close()


def _check_uc_destination(w: Any, catalog: str, schema: str, me: Identity) -> Check:
    """R7 (i): USE SCHEMA + CREATE TABLE on an existing schema, or CREATE SCHEMA on the catalog."""
    name = f"UC destination {catalog}.{schema}"
    try:
        w.api_client.do("GET", f"/api/2.1/unity-catalog/schemas/{catalog}.{schema}")
        exists = True
    except Exception as e:  # noqa: BLE001
        if not _is_not_found(e):
            return _not_checked(name, e)
        exists = False
    try:
        if exists:
            lack = _lacking(_privileges(w, "schema", f"{catalog}.{schema}", me), ("USE_SCHEMA", "CREATE_TABLE"))
            on, mode = f"SCHEMA {catalog}.{schema}", "schema exists; the kit uses it"
        else:
            lack = _lacking(_privileges(w, "catalog", catalog, me), ("CREATE_SCHEMA",))
            on, mode = f"CATALOG {catalog}", "schema absent; the kit creates it"
    except Exception as e:  # noqa: BLE001
        return _not_checked(name, e)
    if not lack:
        return Check(name, OK, mode)
    grants = ", ".join(p.replace("_", " ") for p in lack)
    return Check(name, MISSING, f"{mode}; missing {grants}", f"GRANT {grants} ON {on} TO `{me.user_name}`;")


def _check_uc_source(w: Any, catalog: str, schema: str, me: Identity) -> Check:
    """provision_only: the kit schema holds the Delta sources the operator built; read them."""
    name = f"UC source {catalog}.{schema}"
    try:
        lack = _lacking(_privileges(w, "schema", f"{catalog}.{schema}", me), ("USE_SCHEMA", "SELECT"))
    except Exception as e:  # noqa: BLE001
        if _is_not_found(e):
            return Check(name, MISSING, "schema absent: nothing was built",
                         f"build the graph into {catalog}.{schema} as the operator first")
        return _not_checked(name, e)
    if not lack:
        return Check(name, OK, "USE SCHEMA + SELECT")
    grants = ", ".join(p.replace("_", " ") for p in lack)
    return Check(name, MISSING, f"missing {grants}", f"GRANT {grants} ON SCHEMA {catalog}.{schema} TO `{me.user_name}`;")


def _check_premade_destination(w: Any, catalog: str, schema: str, me: Identity) -> Check:
    """provision_only: the identity cannot create schemas, so the synced tables'
    destination schema must exist already, with USE SCHEMA + CREATE TABLE."""
    name = f"UC destination {catalog}.{schema}"
    try:
        lack = _lacking(_privileges(w, "schema", f"{catalog}.{schema}", me), ("USE_SCHEMA", "CREATE_TABLE"))
    except Exception as e:  # noqa: BLE001
        if _is_not_found(e):
            return Check(name, MISSING, "schema absent, and a provision-only run does not create schemas",
                         f"as the operator: CREATE SCHEMA {catalog}.{schema}; GRANT USE SCHEMA, CREATE TABLE "
                         f"ON SCHEMA {catalog}.{schema} TO `{me.user_name}`;")
        return _not_checked(name, e)
    if not lack:
        return Check(name, OK, "schema exists (made by the operator); the kit uses it")
    grants = ", ".join(p.replace("_", " ") for p in lack)
    return Check(name, MISSING, f"schema exists; missing {grants}",
                 f"GRANT {grants} ON SCHEMA {catalog}.{schema} TO `{me.user_name}`;")


def _check_synced_table_catalog(w: Any, catalog: str, me: Identity) -> Check:
    """R7 (ii): synced tables are UC tables in the destination catalog. Their CREATE TABLE is
    checked on the destination schema (live 2026-10-05: schema-level is enough), so the catalog
    needs USE CATALOG only."""
    name = f"synced tables in catalog {catalog}"
    needed = ("USE_CATALOG",)
    try:
        lack = _lacking(_privileges(w, "catalog", catalog, me), needed)
    except Exception as e:  # noqa: BLE001
        if _is_not_found(e):
            return Check(name, MISSING, "catalog not found", f"create catalog {catalog} or choose another")
        return _not_checked(name, e)
    if not lack:
        return Check(name, OK, " + ".join(p.replace("_", " ") for p in needed))
    grants = ", ".join(p.replace("_", " ") for p in lack)
    return Check(name, MISSING, f"missing {grants}", f"GRANT {grants} ON CATALOG {catalog} TO `{me.user_name}`;")


def _check_sync_owner(owner: str, me: Optional[Identity]) -> Check:
    """R5: preflight runs as the identity that will own the sync job, so the grants it checks
    are the owner's. A recorded owner that is not the operator would bind provision to an
    identity whose grants were never checked."""
    name = "sync-job owner"
    if me is None:
        return _not_checked(name, "identity unknown")
    if str(owner).casefold() == me.user_name.casefold():
        return Check(name, OK, f"{owner} runs preflight and will own the sync job")
    return Check(name, MISSING, f"sync_owner_user {owner!r} is not the operator {me.user_name!r}",
                 "run preflight (and provision) as the identity that will own the sync job")


def _check_sync_mode(config: Dict[str, Any]) -> Check:
    """Kit v1 supports only 'snapshot' sync mode (or unset, which defaults to 'snapshot')."""
    from lgk.refresh import modes

    name = "sync mode"
    try:
        modes.require_v1_sync_mode(config)
        return Check(name, OK, "snapshot")
    except modes.UnsupportedSyncModeError as e:
        return Check(name, MISSING, str(e),
                     'set sync_mode to "snapshot", or leave it unset')


@dataclass
class Scope:
    """What this run does, so preflight checks exactly what it needs (owner decision, H1)."""

    project: str
    catalog: str
    kit_schema: str
    version_schema: str
    branches: List[str]
    served_schema: str = ""
    creates_branches: bool = False
    provision_only: bool = False
    # sp_builds: (securable, full name, privileges, operator creates it) of lgk.sp_grants, or the
    # reason the list cannot be made (no mapping, the workshop's kit schema)
    sp_grants: Optional[List[Tuple[str, str, Tuple[str, ...], bool]]] = None
    sp_grants_error: Optional[str] = None
    graph_name: str = ""


SP_GRANTS_CHECK = "UC grants of the service principal"


def _sp_grant_list(config: Dict[str, Any]) -> Tuple[Optional[list], Optional[str]]:
    """lgk.sp_grants' list for this bundle config and its run's mapping, or why it cannot be made."""
    from lgk import sp_grants
    from lgk.mapping import load_mapping

    try:
        sp_grants.require_own_kit_schema(config)
        mapping = load_mapping(Path(config.get("run_folder", ".")) / "graph_mapping.yaml")
        return sp_grants.uc_grants(sp_grants.targets(config, mapping)), None
    except Exception as e:  # noqa: BLE001 — reported as a missing check, never a pass
        return None, f"{type(e).__name__}: {e}"


def scope_of(config: Dict[str, Any]) -> Scope:
    from lgk.versions import served_schema, version_schema

    schema = config.get("kit_schema", f"lgk_{config.get('graph_name', 'graph')}")
    from lgk.branch import BranchRef

    branch = BranchRef.from_config(config).branch
    grants, grants_error = _sp_grant_list(config) if config.get("sp_builds") else (None, None)
    return Scope(
        project=config["lakebase_project"], catalog=config["lakebase_catalog"], kit_schema=schema,
        version_schema=version_schema({**config, "kit_schema": schema}),
        served_schema=served_schema({**config, "kit_schema": schema}),
        branches=list(dict.fromkeys([branch] + ([config["source_branch"]] if config.get("source_branch") else []))),
        # default False: no shipped step creates a branch (only the test harness does)
        creates_branches=bool(config.get("creates_branches", False)),
        provision_only=bool(config.get("provision_only", False)),
        sp_grants=grants, sp_grants_error=grants_error, graph_name=str(config.get("graph_name", "")),
    )


def _sp_check_name(securable: str, full: str, privileges: Tuple[str, ...]) -> str:
    kind = "destination" if "CREATE TABLE" in privileges else "source" if "SELECT" in privileges else "catalog"
    return f"UC {kind} {full}"


def _check_sp_grant(w: Any, me: Identity, securable: str, full: str, privileges: Tuple[str, ...],
                    operator_creates: bool, graph_name: str) -> Check:
    """One entry of lgk.sp_grants: the securable exists (the operator made it) and the SP holds the privileges."""
    from lgk import sp_grants

    name = _sp_check_name(securable, full, privileges)
    try:
        lack = _lacking(_privileges(w, securable.lower(), full, me), [p.replace(" ", "_") for p in privileges])
    except Exception as e:  # noqa: BLE001
        if not _is_not_found(e):
            return _not_checked(name, e)
        grant = sp_grants.grant_sql(securable, full, privileges, me.user_name)
        if operator_creates:
            return Check(name, MISSING, f"{securable.lower()} absent; the operator creates it (the service "
                         "principal cannot create schemas)",
                         f"as the operator: {sp_grants.create_schema_sql(full, graph_name)} {grant}")
        return Check(name, MISSING, f"{securable.lower()} absent", f"as the operator: create or choose it; {grant}")
    if not lack:
        return Check(name, OK, " + ".join(privileges))
    missing = tuple(p.replace("_", " ") for p in lack)
    return Check(name, MISSING, f"missing {', '.join(missing)}", sp_grants.grant_sql(securable, full, missing, me.user_name))


def _uc_check_names(sc: Scope) -> List[str]:
    if sc.sp_grants is not None:
        return [_sp_check_name(sec, full, privs) for sec, full, privs, _ in sc.sp_grants]
    if sc.sp_grants_error is not None:
        return [SP_GRANTS_CHECK]
    if sc.provision_only:
        return [f"UC source {sc.catalog}.{sc.kit_schema}", f"UC destination {sc.catalog}.{sc.version_schema}"]
    return [f"UC destination {sc.catalog}.{sc.kit_schema}", f"UC destination {sc.catalog}.{sc.version_schema}"]


def _uc_checks(w: Any, me: Identity, sc: Scope) -> List[Check]:
    if sc.sp_grants is not None:
        return [_check_sp_grant(w, me, sec, full, privs, creates, sc.graph_name) for sec, full, privs, creates in sc.sp_grants]
    if sc.sp_grants_error is not None:
        return [Check(SP_GRANTS_CHECK, MISSING, sc.sp_grants_error,
                      "export the run again (the bundle names its own kit schema and carries the mapping)")]
    if sc.provision_only:
        return [_check_uc_source(w, sc.catalog, sc.kit_schema, me),
                _check_premade_destination(w, sc.catalog, sc.version_schema, me)]
    # build claims the kit schema and writes its Delta tables; provision claims the version
    # schema and creates the synced tables in it (lgk.manifest.claim_uc_schema: create if absent)
    return [_check_uc_destination(w, sc.catalog, sc.kit_schema, me),
            _check_uc_destination(w, sc.catalog, sc.version_schema, me)]


def _workspace_checks(w: Any, me: Optional[Identity], sc: Scope) -> List[Check]:
    """Every check that reads the workspace API; all not_checked without an identity."""
    level, _ = _project_level(sc.creates_branches)
    # sp_builds: the SP's grant list holds USE CATALOG on the destination catalog
    sp = sc.sp_grants is not None or sc.sp_grants_error is not None
    catalog_names = [] if sp else [f"synced tables in catalog {sc.catalog}"]
    names = ([f"project {sc.project}", f"{level} on project {sc.project}"] + [f"branch {b}" for b in sc.branches]
             + _uc_check_names(sc) + catalog_names)
    if me is None:
        return [_not_checked(n, "identity unknown") for n in names]
    catalog = [] if sp else [_check_synced_table_catalog(w, sc.catalog, me)]
    return ([_check_project(w, sc.project), _check_project_permission(w, sc.project, me, sc.creates_branches, sc.branches[0])]
            + [_check_branch(w, sc.project, b) for b in sc.branches]
            + _uc_checks(w, me, sc) + catalog)


def run_checks(config: Dict[str, Any]) -> Tuple[List[Check], Optional[Identity]]:
    """All preflight checks, in order. Pure reads: nothing is created."""
    sc = scope_of(config)
    w, connect = _environment(config)
    ident, me = _check_identity(w)
    checks = [ident, _check_sync_owner(config["sync_owner_user"], me), _check_sync_mode(config)] + _workspace_checks(w, me, sc)
    checks += _check_postgres(connect if me else None, sc.project, sc.branches[0], sc.served_schema, me)
    return checks, me


# ----------------------------------------------------------------------------------- step
def _result(checks: List[Check], me: Optional[Identity], owner: str) -> PreflightResult:
    failed = [c for c in checks if c.status != OK]
    by_name = {c.name: c for c in checks}
    return PreflightResult(
        status=BLOCKED if failed else PASS,
        reason=("; ".join(f"{c.name}: {c.status}" for c in failed) or None),
        missing_permissions=[{"permission": f"{c.name} ({c.status}): {c.detail}", "fix": c.fix} for c in failed],
        project_exists=any(n.startswith("project ") and c.status == OK for n, c in by_name.items()),
        schema_creatable=any(n.startswith("UC destination") and c.status == OK for n, c in by_name.items()),
        sync_owner_user=owner,
        operator=me.user_name if me else None,
        checks=[asdict(c) for c in checks],
    )


def preflight(config: Dict[str, Any]) -> PreflightResult:
    """Run every check; write run_folder/preflight_result.json; raise PreflightBlocked unless all ok.

    Config: lakebase_project, lakebase_catalog, sync_owner_user (required); kit_schema,
    branch_id (required: no default branch), source_branch, graph_name, version, served_schema;
    creates_branches (default False: CAN_USE; True asks CAN_MANAGE); provision_only (default False, see the module doc); workspace_client + pg_connect
    (injected), or preflight_env (`module:qualname` factory, default live_environment).
    """
    run_folder = Path(config.get("run_folder", "."))
    run_folder.mkdir(parents=True, exist_ok=True)
    for key in ("lakebase_project", "lakebase_catalog", "sync_owner_user", "branch_id"):
        if not config.get(key):
            _write_result(run_folder, PreflightResult(status=BLOCKED, reason=f"Missing required config key: {key}"))
            raise KeyError(f"Missing required config key: {key}")
    checks, me = run_checks(config)
    result = _result(checks, me, config["sync_owner_user"])
    _write_result(run_folder, result)
    if result.status != PASS:
        raise PreflightBlocked(f"preflight blocked: {result.reason}")
    logger.info("Preflight passed")
    return result


def _write_result(run_folder: Path, result: PreflightResult) -> None:
    (run_folder / RESULT_FILE).write_text(json.dumps(result.to_dict(), indent=2))


# ------------------------------------------------------------------------- provision
def require_sync_owner(run_folder: Any, current_identity: Optional[str] = None) -> str:
    """R5: refuse unless this process runs as the sync-job owner preflight recorded.

    Refuses a missing or blocked preflight record, too: the owner must have been asked.
    Databricks user names compare case-insensitively; anything else must match exactly.
    """
    path = Path(run_folder) / RESULT_FILE
    if not path.exists():
        raise SyncOwnerMismatch(f"no {RESULT_FILE} in {run_folder}: run preflight first")
    record = json.loads(path.read_text())
    owner = record.get("sync_owner_user")
    if record.get("status") != PASS or not owner:
        raise SyncOwnerMismatch(f"preflight did not pass or recorded no sync owner: {record.get('reason')}")
    if current_identity is None:
        from databricks.sdk import WorkspaceClient

        current_identity = WorkspaceClient().current_user.me().user_name
    if str(current_identity).casefold() != str(owner).casefold():
        raise SyncOwnerMismatch(
            f"provision runs as {current_identity!r}, but preflight recorded {owner!r} as the sync-job "
            "owner; synced-table access stays with the identity that creates them. Run as the owner, "
            "or re-run preflight with sync_owner_user set to this identity.")
    return owner
