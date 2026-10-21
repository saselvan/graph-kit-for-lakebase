"""Seam-2: provision must refuse to reuse a synced table bound to another branch.

Branch deletion does not cascade to the UC synced table, so a re-run with the same
names finds an "already exists" table still pointing at a deleted branch.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lgk.steps import provision as prov

CONFIG = {
    "lakebase_catalog": "cat",
    "graph_name": "g",
    "kit_schema": "lgk_v1_x",
    "lakebase_project": "proj",
    "branch_id": "t-current",
    "warehouse_id": "wh",
}


def _client(bound_branch):
    w = MagicMock()
    w.statement_execution.execute_statement.return_value = SimpleNamespace(
        status=SimpleNamespace(state="SUCCEEDED"), result=None
    )
    # The UC kit and version schemas already carry this graph's marker.
    w.schemas.get.return_value = SimpleNamespace(comment="lgk:graph=g")
    w.postgres.create_synced_table.side_effect = Exception(
        "RESOURCE_ALREADY_EXISTS: Synced table already exists"
    )
    # Hostile: the stale table looks perfectly healthy (ONLINE), so only the
    # branch check can stop the reuse.
    w.postgres.get_synced_table.return_value = SimpleNamespace(
        spec=SimpleNamespace(branch=bound_branch),
        status=SimpleNamespace(detailed_state=SimpleNamespace(value="SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE")),
    )
    return w


@pytest.mark.seam2
def test_existing_synced_table_on_other_branch_raises(monkeypatch):
    w = _client("projects/proj/branches/t-deleted-old-run")
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: w)
    with pytest.raises(prov.StaleSyncedTableError, match="t-deleted-old-run"):
        prov._create_synced_tables(CONFIG)
    w.postgres.get_synced_table.assert_any_call(name="synced_tables/cat.lgk_v1_x_v1.nodes")


@pytest.mark.seam2
def test_existing_synced_table_with_unreadable_branch_raises(monkeypatch):
    w = _client(None)  # sparse GET: branch not echoed, reuse cannot be proven
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: w)
    with pytest.raises(prov.StaleSyncedTableError):
        prov._create_synced_tables(CONFIG)


@pytest.mark.seam2
def test_existing_synced_table_on_same_branch_is_reused(monkeypatch):
    w = _client("projects/proj/branches/t-current")
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: w)
    prov._create_synced_tables(CONFIG)  # no raise
    assert w.postgres.create_synced_table.call_count == 3
