"""Offline: the reader role is used if it exists and never escalated for."""

import psycopg
import pytest

from lgk.steps.provision import READER_SETUP_SQL, ROLE_SETUP_SQL, ReaderRoleMissingError, _create_reader_role_pg


class FakeCursor:
    """Scripted Postgres: whether `reader` exists, and whether CREATE ROLE is allowed."""

    def __init__(self, exists: bool, may_create: bool):
        self.exists, self.may_create, self.sql = exists, may_create, []

    def execute(self, sql, params=None):
        self.sql.append(" ".join(sql.split()))
        if "CREATE ROLE" in sql.upper() and not self.may_create:
            raise psycopg.errors.InsufficientPrivilege("permission denied to create role")

    def fetchone(self):
        return (1,) if self.exists else None


def _creates(cur):
    return [s for s in cur.sql if "CREATE ROLE" in s.upper() or "DATABRICKS_CREATE_ROLE" in s.upper()]


def test_existing_reader_issues_no_create():
    # hostile: an identity that may NOT create roles — any CREATE would fail the SP's provision
    cur = FakeCursor(exists=True, may_create=False)
    assert _create_reader_role_pg(cur) is False
    assert _creates(cur) == []


def test_missing_reader_without_createrole_names_the_fix():
    cur = FakeCursor(exists=False, may_create=False)
    with pytest.raises(ReaderRoleMissingError) as err:
        _create_reader_role_pg(cur)
    assert "CREATE ROLE reader WITH LOGIN;" in str(err.value) and "Operator" in str(err.value)
    # never escalates: exactly one CREATE ROLE attempt, no GRANT of a privileged role
    assert _creates(cur) == ["CREATE ROLE reader WITH LOGIN;"]
    assert not [s for s in cur.sql if "GRANT" in s.upper() or "ALTER ROLE" in s.upper()]


def test_missing_reader_is_created_by_an_operator_that_may():
    cur = FakeCursor(exists=False, may_create=True)
    assert _create_reader_role_pg(cur) is True
    assert _creates(cur) == [ROLE_SETUP_SQL["reader"]]
    assert ROLE_SETUP_SQL["reader"] == "CREATE ROLE reader WITH LOGIN;"  # LOGIN only, no other attribute
    # the operator's one-time setup makes the two function roles too, also LOGIN only
    assert READER_SETUP_SQL == ("CREATE ROLE reader WITH LOGIN; CREATE ROLE public_reader WITH LOGIN; "
                                "CREATE ROLE coordinator_reader WITH LOGIN;")


@pytest.mark.parametrize("exists", [False, True])
def test_serve_time_role_setup_records_no_role_for_the_graph(tmp_path, exists):
    """A6 (a): `reader` is shared by every graph on the branch, so no graph's manifest owns it."""
    from lgk.manifest import MANIFEST_FILE
    from lgk.steps.provision import _ensure_function_roles, _ensure_reader_role

    config = {"graph_name": "g", "run_folder": str(tmp_path), "lakebase_project": "p", "branch_id": "t-x"}
    # hostile: the role is missing, so THIS run creates it (the pin then recorded it as the graph's)
    cur = FakeCursor(exists=exists, may_create=True)
    _ensure_reader_role(config, cur)
    _ensure_function_roles(config, cur)
    assert not (tmp_path / MANIFEST_FILE).exists()
    assert not [s for s in cur.sql if s.upper().startswith("COMMENT ON ROLE")]
    assert len(_creates(cur)) == (0 if exists else 3)
