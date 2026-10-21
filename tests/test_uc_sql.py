"""Seam-2: a failed UC statement must stop build, not pass silently."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lgk.uc_sql import StatementFailedError, execute_sql


def _resp(state, statement_id="s1", error=None, rows=None):
    return SimpleNamespace(
        statement_id=statement_id,
        status=SimpleNamespace(state=SimpleNamespace(value=state), error=error),
        result=SimpleNamespace(data_array=rows) if rows is not None else None,
    )


@pytest.mark.seam2
def test_build_raises_when_statement_failed(monkeypatch, tmp_path):
    w = MagicMock()
    w.statement_execution.execute_statement.return_value = _resp(
        "FAILED", error=SimpleNamespace(message="TABLE_OR_VIEW_NOT_FOUND")
    )
    w.schemas.get.return_value = SimpleNamespace(comment="lgk:graph=g")  # kit schema is ours
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: w)
    (tmp_path / "graph_mapping.yaml").write_text(
        "source_catalog: c\nsource_schema: s\n"
        "nodes:\n  p: {source_table: t, natural_key: k, label: l}\nedges: {}\n"
    )
    from lgk.steps.build import build
    from tests.privacy_support import clear_gate_offline

    clear_gate_offline(str(tmp_path))
    with pytest.raises(StatementFailedError, match="TABLE_OR_VIEW_NOT_FOUND"):
        build({"run_folder": str(tmp_path), "graph_name": "g", "lakebase_catalog": "c", "kit_schema": "k", "warehouse_id": "wh"})


@pytest.mark.seam2
def test_execute_sql_polls_until_terminal_then_returns_rows():
    w = MagicMock()
    w.statement_execution.execute_statement.return_value = _resp("RUNNING")
    w.statement_execution.get_statement.side_effect = [
        _resp("RUNNING"),
        _resp("SUCCEEDED", rows=[["7"]]),
    ]
    assert execute_sql(w, "wh", "SELECT 7", poll_s=0) == [["7"]]
    assert w.statement_execution.get_statement.call_count == 2


@pytest.mark.seam2
@pytest.mark.parametrize("state", ["CANCELED", "CLOSED"])
def test_execute_sql_raises_on_other_terminal_states(state):
    w = MagicMock()
    w.statement_execution.execute_statement.return_value = _resp(state)
    with pytest.raises(StatementFailedError, match=state):
        execute_sql(w, "wh", "SELECT 1")
