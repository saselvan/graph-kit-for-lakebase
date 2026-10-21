"""Seam-1 gate for preflight: the real API and the real branch Postgres.

Moved out of tests/test_preflight.py so every live test lives in a *_live.py file.
"""

import uuid

import pytest

from lgk.steps.preflight import SyncOwnerMismatch, preflight, require_sync_owner

pytestmark = pytest.mark.seam1


def test_live_preflight_passes_for_the_operator_on_the_live_project(workspace_client, lakebase_project,
                                                                    lakebase_catalog, tmp_path):
    """Real API + real branch Postgres (production, read-only queries), with all defaults."""
    me = workspace_client.current_user.me().user_name
    cfg = {"graph_name": "preflight_live", "run_folder": str(tmp_path), "lakebase_project": lakebase_project,
           "branch_id": "production", "lakebase_catalog": lakebase_catalog, "sync_owner_user": me,
           "source_branch": "test-base"}
    result = preflight(cfg)
    assert result.status == "pass", result.reason
    # 15 checks: the 14 base checks plus "sync mode" (v1: snapshot only; unset here, so it passes)
    assert result.operator == me and len(result.checks) == 15
    (sync_mode,) = [c for c in result.checks if c["name"] == "sync mode"]
    assert sync_mode["status"] == "ok"
    require_sync_owner(tmp_path)  # the live identity is the recorded owner
    with pytest.raises(SyncOwnerMismatch):
        require_sync_owner(tmp_path, str(uuid.uuid4()))  # an SP id made at run time
