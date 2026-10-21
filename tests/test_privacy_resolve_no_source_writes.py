"""R (persona round): resolve_flag never alters a source table; it writes only flags.json.

A persona's agent proposed ALTER/TRUNCATE on source tables around a privacy decision. The
decision itself (fold / drop / reclassify) is a record in the run folder, nothing else. This
test arms every way the kit reaches a warehouse or Postgres (the SDK client, the statement
API, `execute_sql` in every loaded lgk module, psycopg connect) with a recorder, resolves
every choice, and asserts no statement was issued and only flags.json changed.
"""

import hashlib
import json
import sys

import psycopg
import pytest

import lgk.uc_sql  # noqa: F401  (loaded, so the fixture arms its execute_sql)
from lgk.privacy.flags import OpenFlagError, resolve_flag
from lgk.steps.privacy import privacy

SOURCE_WORDS = ("ALTER", "UPDATE", "DROP", "COMMENT", "TRUNCATE", "DELETE", "INSERT", "CREATE")


def _schema(run_dir):
    col = lambda n: {"name": n, "type": "STRING"}  # noqa: E731
    (run_dir / "schema.json").write_text(json.dumps({
        "source_catalog": "c", "source_schema": "s", "tables": [
            {"name": "providers", "key_columns": ["provider_id"], "foreign_keys": [],
             "columns": [col("provider_id"), col("name"), col("patient_name")]},
            {"name": "encounters", "key_columns": ["patient_id", "provider_id"],
             "foreign_keys": [{"columns": ["provider_id"], "references": ["providers", "provider_id"]}],
             "columns": [col("patient_id"), col("provider_id"), col("member_id")]},
        ]}))


def _snapshot(run_dir):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(run_dir.iterdir()) if p.is_file()}


@pytest.fixture
def armed(monkeypatch):
    """Record (and refuse) every reach to a warehouse or Postgres."""
    calls = []

    def recorder(where):
        def record(*args, **kwargs):
            calls.append((where, args, kwargs))
            raise AssertionError(f"resolve_flag reached {where}: {args} {kwargs}")
        return record

    import databricks.sdk
    from databricks.sdk.service.sql import StatementExecutionAPI

    monkeypatch.setattr(databricks.sdk.WorkspaceClient, "__init__", recorder("WorkspaceClient"))
    monkeypatch.setattr(StatementExecutionAPI, "execute_statement", recorder("execute_statement"))
    monkeypatch.setattr(psycopg.Connection, "connect", recorder("psycopg.Connection.connect"))
    monkeypatch.setattr(psycopg, "connect", recorder("psycopg.connect"))
    for name, mod in list(sys.modules.items()):
        if name.startswith("lgk") and hasattr(mod, "execute_sql"):
            monkeypatch.setattr(mod, "execute_sql", recorder(f"{name}.execute_sql"))
    return calls


@pytest.mark.parametrize("table, column, choice, extra", [
    ("encounters", "patient_id", "fold", {}),
    ("encounters", "member_id", "drop", {}),
    ("providers", "patient_name", "reclassify", {"reason": "a test reason", "new_class": "none"}),
])
def test_resolve_flag_issues_no_statement_and_writes_only_flags_json(
        tmp_path, armed, table, column, choice, extra):
    _schema(tmp_path)
    with pytest.raises(OpenFlagError):
        privacy({"run_folder": str(tmp_path)})
    before = _snapshot(tmp_path)
    flag = resolve_flag(str(tmp_path), table, column, choice, chooser="op", **extra)
    after = _snapshot(tmp_path)
    assert armed == []
    assert {n for n in set(before) | set(after) if before.get(n) != after.get(n)} == {"flags.json"}
    # the change IS the decision, recorded where the gate reads it
    on_disk = [f for f in json.loads((tmp_path / "flags.json").read_text())["flags"]
               if (f["table"], f["column"]) == (table, column)]
    assert on_disk == [flag] and flag["choice"] == choice


def test_the_recorder_catches_a_statement(armed):
    """Non-vacuous: a COMMENT ON COLUMN through the kit's SQL helper is caught."""
    with pytest.raises(AssertionError, match="execute_sql"):
        lgk.uc_sql.execute_sql(None, "wh", "COMMENT ON COLUMN c.s.encounters.patient_id IS 'x'")
    assert any(any(w in str(a) for w in SOURCE_WORDS) for _w, a, _k in armed)
