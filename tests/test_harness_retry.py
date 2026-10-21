"""Offline: BranchLeasor retries the transient branch-create failure seen under parallel live suites."""

import pytest

from lgk.harness import BranchLeasor


class FakeApi:
    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def do(self, method, path, query=None, body=None):
        self.calls += 1
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _leasor(tmp_path, monkeypatch, script):
    monkeypatch.setattr(BranchLeasor, "CREATE_BACKOFF_S", 0)
    monkeypatch.delenv("LGK_SEAM2", raising=False)
    api = FakeApi(script)
    w = type("W", (), {"api_client": api})()
    return BranchLeasor(workspace_client=w, project_id="p", test_base_branch="test-base",
                        run_folder=tmp_path, ttl_seconds=60), api


TRANSIENT = RuntimeError("The request could not be processed. Please try again later")


def test_transient_failures_are_retried(tmp_path, monkeypatch):
    leasor, api = _leasor(tmp_path, monkeypatch, [TRANSIENT, TRANSIENT, {"branch_id": "t-x"}])
    assert leasor.create_test_branch("x")["branch_id"] == "t-x" and api.calls == 3


def test_already_exists_after_a_transient_is_our_branch(tmp_path, monkeypatch):
    # hostile: the transient call DID create the branch, so the retry sees "already exists"
    leasor, api = _leasor(tmp_path, monkeypatch, [TRANSIENT, RuntimeError("Branch already exists")])
    assert leasor.create_test_branch("x")["branch_id"].startswith("t-x-") and api.calls == 2


def test_already_exists_without_a_transient_is_raised(tmp_path, monkeypatch):
    leasor, api = _leasor(tmp_path, monkeypatch, [RuntimeError("Branch already exists")])
    with pytest.raises(RuntimeError, match="already exists"):
        leasor.create_test_branch("x")
    assert api.calls == 1


def test_other_errors_and_exhausted_retries_are_raised(tmp_path, monkeypatch):
    leasor, api = _leasor(tmp_path, monkeypatch, [RuntimeError("PERMISSION_DENIED")])
    with pytest.raises(RuntimeError, match="PERMISSION_DENIED"):
        leasor.create_test_branch("x")
    leasor, api = _leasor(tmp_path, monkeypatch, [TRANSIENT] * BranchLeasor.CREATE_TRIES)
    with pytest.raises(RuntimeError, match="could not be processed"):
        leasor.create_test_branch("x")
    assert api.calls == BranchLeasor.CREATE_TRIES


# ---------------------------------------------------------------- mutation-prover additions (merge 698dd02)
def test_transient_backoff_grows_with_each_try(tmp_path, monkeypatch):
    leasor, api = _leasor(tmp_path, monkeypatch, [TRANSIENT, TRANSIENT, {"branch_id": "t-x"}])
    monkeypatch.setattr(BranchLeasor, "CREATE_BACKOFF_S", 7)  # hostile: non-zero, so a flat or no wait differs
    slept = []
    monkeypatch.setattr("lgk.harness.time.sleep", slept.append)
    leasor.create_test_branch("x")
    assert slept == [7, 14]


def test_endpoint_and_manifest_entry_name_the_created_branch_id(tmp_path, monkeypatch):
    from lgk.manifest import Manifest, ObjectType

    leasor, api = _leasor(tmp_path, monkeypatch, [{}])
    out = leasor.create_test_branch("My_Branch")  # hostile: the human name is not a legal id
    bid = out["branch_id"]
    assert bid != "My_Branch" and out["endpoint_name"] == f"projects/p/branches/{bid}/endpoints/primary"
    [entry] = [e for e in Manifest.load(tmp_path / "manifest.json").entries
               if e.object_type == ObjectType.BRANCH]
    assert (entry.object_id, entry.name, entry.location) == (
        bid, "My_Branch", f"lakebase:{out['endpoint_name']}")
    assert entry.metadata == {"source_branch": "projects/p/branches/test-base", "ttl_seconds": 60}


def test_two_creates_of_one_name_in_the_same_second_get_distinct_ids(tmp_path, monkeypatch):
    leasor, api = _leasor(tmp_path, monkeypatch, [{}, {}])
    monkeypatch.setattr("lgk.harness.time.time", lambda: 1_700_000_000)  # hostile: frozen clock
    assert leasor.create_test_branch("x")["branch_id"] != leasor.create_test_branch("x")["branch_id"]


# ---------------------------------------------------------------- platform operation cap (H3, owner decision B)
from databricks.sdk.errors.platform import Aborted  # noqa: E402

OP_CAP = Aborted("Your project has too many operations in progress. Please wait until they are complete, "
                 "and then try again. [TraceId: b1fd73bbc560ab63b31653006a81401c]")
KEYRING_44 = ValueError(
    "default auth: databricks-cli: cannot get access token: Error: cache: OS keyring unreachable: exit status 44 "
    "(set DATABRICKS_AUTH_STORAGE=plaintext or run `databricks auth login` to use file-based token storage).")


def test_operation_cap_on_create_is_retried_then_succeeds(tmp_path, monkeypatch):
    # hostile: the cap twice in a row, with the real message and trace id (seen at suite setup, t06-live logs)
    leasor, api = _leasor(tmp_path, monkeypatch, [OP_CAP, OP_CAP, {"branch_id": "t-x"}])
    assert leasor.create_test_branch("x")["branch_id"] == "t-x" and api.calls == 3


def test_operation_cap_backoff_grows_with_each_try(tmp_path, monkeypatch):
    leasor, api = _leasor(tmp_path, monkeypatch, [OP_CAP, OP_CAP, {"branch_id": "t-x"}])
    monkeypatch.setattr(BranchLeasor, "CREATE_BACKOFF_S", 5)  # hostile: non-zero, so a flat or no wait differs
    slept = []
    monkeypatch.setattr("lgk.harness.time.sleep", slept.append)
    leasor.create_test_branch("x")
    assert slept == [5, 10]


def test_already_exists_after_an_operation_cap_is_our_branch(tmp_path, monkeypatch):
    leasor, api = _leasor(tmp_path, monkeypatch, [OP_CAP, RuntimeError("Branch already exists")])
    assert leasor.create_test_branch("x")["branch_id"].startswith("t-x-") and api.calls == 2


def test_a_different_aborted_message_is_not_retried(tmp_path, monkeypatch):
    # hostile: same exception class as the cap, other message
    leasor, api = _leasor(tmp_path, monkeypatch, [Aborted("Operation was aborted by the server"), {}])
    with pytest.raises(Aborted, match="aborted by the server"):
        leasor.create_test_branch("x")
    assert api.calls == 1


def test_the_cap_message_on_another_error_class_is_not_retried(tmp_path, monkeypatch):
    # hostile: the cap text but not an Aborted (Aborted is an OSError; so is the keyring failure)
    leasor, api = _leasor(tmp_path, monkeypatch, [OSError("too many operations in progress"), {}])
    with pytest.raises(OSError, match="too many operations"):
        leasor.create_test_branch("x")
    assert api.calls == 1


def test_keyring_exit_44_is_not_retried(tmp_path, monkeypatch):
    for err in (KEYRING_44, OSError(str(KEYRING_44))):  # the SDK raises it as ValueError, wrapping an OSError
        leasor, api = _leasor(tmp_path, monkeypatch, [err, {}])
        with pytest.raises(type(err), match="exit status 44"):
            leasor.create_test_branch("x")
        assert api.calls == 1


def test_exhausted_operation_cap_retries_are_raised(tmp_path, monkeypatch):
    leasor, api = _leasor(tmp_path, monkeypatch, [OP_CAP] * BranchLeasor.CREATE_TRIES)
    with pytest.raises(Aborted, match="too many operations"):
        leasor.create_test_branch("x")
    assert api.calls == BranchLeasor.CREATE_TRIES


def _leased(tmp_path, monkeypatch, delete_script):
    leasor, api = _leasor(tmp_path, monkeypatch, [{"branch_id": "t-x"}, *delete_script])
    leasor.create_test_branch("x")
    return leasor, api


def test_operation_cap_on_delete_is_retried_then_succeeds(tmp_path, monkeypatch):
    leasor, api = _leased(tmp_path, monkeypatch, [OP_CAP, {}])
    assert leasor.delete_branch("t-x") is True and api.calls == 3
    assert leasor.manifest.entries == []


def test_delete_does_not_retry_other_aborted_or_keyring_errors(tmp_path, monkeypatch):
    for i, err in enumerate((Aborted("Operation was aborted by the server"), KEYRING_44)):
        (tmp_path / str(i)).mkdir()
        leasor, api = _leased(tmp_path / str(i), monkeypatch, [err, {}])
        assert leasor.delete_branch("t-x") is False and api.calls == 2
        assert [e.object_id for e in leasor.manifest.entries] == ["t-x"]


# ---------------------------------------------------------------- prover additions (H3 prover survivors)
from databricks.sdk.errors.platform import ResourceConflict  # noqa: E402


def test_exhausted_cap_retries_on_delete_report_failure_and_keep_the_entry(tmp_path, monkeypatch):
    leasor, api = _leased(tmp_path, monkeypatch, [OP_CAP] * BranchLeasor.CREATE_TRIES)
    assert leasor.delete_branch("t-x") is False
    assert api.calls == 1 + BranchLeasor.CREATE_TRIES
    assert [e.object_id for e in leasor.manifest.entries] == ["t-x"]


def test_operation_cap_backoff_on_delete_grows_with_each_try(tmp_path, monkeypatch):
    leasor, api = _leased(tmp_path, monkeypatch, [OP_CAP, OP_CAP, {}])
    monkeypatch.setattr(BranchLeasor, "CREATE_BACKOFF_S", 5)
    slept = []
    monkeypatch.setattr("lgk.harness.time.sleep", slept.append)
    assert leasor.delete_branch("t-x") is True and slept == [5, 10]


def test_delete_does_not_retry_the_cap_text_on_another_class_or_the_create_transient(tmp_path, monkeypatch):
    # hostile: the create-only transient, and the cap text on a non-Aborted OSError
    for i, err in enumerate((OSError("too many operations in progress"), TRANSIENT)):
        (tmp_path / str(i)).mkdir()
        leasor, api = _leased(tmp_path / str(i), monkeypatch, [err, {}])
        assert leasor.delete_branch("t-x") is False and api.calls == 2


def test_the_cap_text_on_the_parent_class_is_not_retried(tmp_path, monkeypatch):
    # hostile: ResourceConflict is Aborted's parent; only Aborted is the cap
    leasor, api = _leasor(tmp_path, monkeypatch, [ResourceConflict("too many operations in progress"), {}])
    with pytest.raises(ResourceConflict):
        leasor.create_test_branch("x")
    assert api.calls == 1


def test_the_cap_match_ignores_case(tmp_path, monkeypatch):
    leasor, api = _leasor(tmp_path, monkeypatch, [Aborted("Too Many Operations In Progress"), {"branch_id": "t-x"}])
    assert leasor.create_test_branch("x")["branch_id"] == "t-x" and api.calls == 2


def test_a_near_miss_cap_message_is_not_retried(tmp_path, monkeypatch):
    leasor, api = _leasor(tmp_path, monkeypatch, [Aborted("too many operations queued"), {}])
    with pytest.raises(Aborted, match="queued"):
        leasor.create_test_branch("x")
    assert api.calls == 1


def test_the_leasor_delete_uses_its_own_try_count(tmp_path, monkeypatch):
    # hostile: CREATE_TRIES differs from the shared helper's default (6)
    monkeypatch.setattr(BranchLeasor, "CREATE_TRIES", 3)
    leasor, api = _leased(tmp_path, monkeypatch, [OP_CAP] * 3 + [{}])
    assert leasor.delete_branch("t-x") is False
    assert api.calls == 1 + 3
