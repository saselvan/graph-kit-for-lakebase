"""Scripted fakes for preflight: a workspace API and a branch Postgres, both answering like the
live services (paths, payload shapes and SDK error types read off the live reference project on 2026-10-03)."""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set

from databricks.sdk.errors import NotFound, PermissionDenied

OPERATOR = "operator@example.com"


class FakeApi:
    def __init__(self, ws: "FakeWorkspace"):
        self.ws = ws
        self.calls: List[str] = []
        self.methods: List[str] = []

    def do(self, method: str, path: str, query: Optional[Dict[str, Any]] = None, body: Any = None) -> Any:
        self.calls.append(path)
        self.methods.append(method)
        ws = self.ws
        if ws.down:
            raise ConnectionError("workspace unreachable")
        if path.startswith("/api/2.0/postgres/projects/"):
            rest = path[len("/api/2.0/postgres/projects/"):].split("/")
            if rest[0] != ws.project:
                raise NotFound("project id not found")
            if len(rest) == 3 and rest[2] not in ws.branches:
                raise NotFound("branch id not found")
            return {"name": path}
        if path.startswith("/api/2.0/permissions/database-projects/"):
            if ws.acl_denied:  # live: only CAN_MANAGE may read a project's ACL
                raise PermissionDenied("User does not have CAN_MANAGE permission on database project")
            return {"access_control_list": ws.project_acl}
        if path.startswith("/api/2.1/unity-catalog/effective-permissions/"):
            securable, name = path.split("/")[-2:]
            assert query == {"principal": ws.user}
            privs = ws.grants.get((securable, name), [])
            return {"privilege_assignments": [{"principal": ws.user, "privileges": [{"privilege": p} for p in privs]}]}
        for securable in ("catalog", "schema"):
            prefix = f"/api/2.1/unity-catalog/{securable}s/"
            if path.startswith(prefix):
                name = path[len(prefix):]
                known = ws.catalogs if securable == "catalog" else ws.schemas
                if name not in known:
                    raise NotFound(f"{securable} {name} does not exist")
                return {"name": name, "owner": ws.owners.get(name, "someone-else")}
        raise AssertionError(f"unexpected API call {method} {path}")


class FakeWorkspace:
    """Defaults: every check passes for OPERATOR."""

    def __init__(self, user: str = OPERATOR, groups: tuple = ("users",), project: str = "proj",
                 branches: Optional[Set[str]] = None, catalogs: Optional[Set[str]] = None,
                 schemas: Optional[Set[str]] = None):
        self.user, self.project = user, project
        self.branches = branches if branches is not None else {"production"}
        self.catalogs = catalogs if catalogs is not None else {"cat"}
        self.schemas = schemas if schemas is not None else set()
        self.owners: Dict[str, str] = {}
        self.grants: Dict[tuple, List[str]] = {("catalog", "cat"): ["USE_CATALOG", "CREATE_SCHEMA", "CREATE_TABLE"]}
        self.project_acl: List[Dict[str, Any]] = [
            {"user_name": user, "all_permissions": [{"permission_level": "CAN_MANAGE"}]}]
        self.down = False
        self.api_client = FakeApi(self)
        self.acl_denied = False
        self.credential_error: Optional[Exception] = None  # None: a credential is issued
        self.credentials: List[str] = []
        self.postgres = SimpleNamespace(generate_database_credential=self._credential)
        self.current_user = SimpleNamespace(me=lambda: SimpleNamespace(
            user_name=self.user, groups=[SimpleNamespace(display=g) for g in groups]))


    def _credential(self, endpoint: str) -> Any:
        self.credentials.append(endpoint)
        if self.credential_error is not None:
            raise self.credential_error
        return SimpleNamespace(token="never-" + "printed")


class FakeCursor:
    def __init__(self, pg: "FakePg"):
        self.pg, self._row = pg, None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        self.pg.sql.append(sql)
        if sql == "SELECT 1":
            self._row = (1,)
        elif sql == "SELECT current_user, current_database()":
            self._row = (self.pg.user, self.pg.database)
        elif "has_database_privilege(current_database(), 'CREATE')" in sql:
            self._row = (self.pg.db_create,)
        elif "has_schema_privilege(oid, 'CREATE') FROM pg_namespace WHERE nspname = %s" in sql:
            self._row = (self.pg.schemas[params[0]],) if params[0] in self.pg.schemas else None
        elif "FROM pg_roles WHERE rolname = %s" in sql:
            self._row = (1,) if params[0] in self.pg.roles else None
        elif "rolcreaterole OR rolsuper" in sql:
            self._row = (self.pg.can_create_role,)
        else:
            raise AssertionError(f"unexpected SQL {sql}")

    def fetchone(self):
        return self._row


class FakePg:
    """Defaults: every check passes for OPERATOR (CREATEROLE, CREATE on the database, no schema yet).
    `schemas`: Postgres schema name -> whether current_user may CREATE in it."""

    def __init__(self, roles: Set[str] = frozenset(), can_create_role: bool = True, fails: bool = False,
                 user: str = OPERATOR, database: str = "databricks_postgres", db_create: bool = True,
                 schemas: Optional[Dict[str, bool]] = None):
        self.roles, self.can_create_role, self.fails = set(roles), can_create_role, fails
        self.user, self.database, self.db_create = user, database, db_create
        self.schemas: Dict[str, bool] = dict(schemas or {})
        self.sql: List[str] = []
        self.connected: List[tuple] = []
        self.closed = 0

    def connect(self, project: str, branch: str) -> "FakePg":
        if self.fails:
            raise OSError("connection refused")
        self.connected.append((project, branch))
        return self

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def close(self) -> None:
        self.closed += 1


def golden_preflight_env(config: Dict[str, Any]):
    """The parity golden's preflight environment: every check performed, every check ok."""
    ws = FakeWorkspace(project=config["lakebase_project"], catalogs={config["lakebase_catalog"]})
    ws.grants = {("catalog", config["lakebase_catalog"]): ["ALL_PRIVILEGES"]}
    ws.user = config["sync_owner_user"]
    ws.project_acl = [{"user_name": ws.user, "all_permissions": [{"permission_level": "CAN_MANAGE"}]}]
    return ws, FakePg().connect


def preflight_as_operator(workspace_client: Any, config: Dict[str, Any]) -> Any:
    """Live fixtures: the REAL preflight, as the operator, into config["run_folder"] (the operator runs the real preflight).

    Every live caller that provisions runs this first, so provision's require_sync_owner reads
    a real passing record naming the operator; it is never faked or bypassed. Sets
    config["sync_owner_user"] (the operator) and returns the PreflightResult.
    """
    from lgk.steps.preflight import preflight

    config["sync_owner_user"] = workspace_client.current_user.me().user_name
    return preflight(dict(config, workspace_client=workspace_client))
