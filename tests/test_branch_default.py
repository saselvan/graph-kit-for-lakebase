"""Owner "branch a": a missing or empty branch_id is an error naming the key, never `production`.

Every branch path and connection goes through BranchRef.from_config, so the refusal there is the
one guard; the readers below (connect_branch, the manifest's pg_location, the refresh job record,
teardown's branch name) are checked to refuse too, so none of them keeps a private fallback.
"""

import pytest

from lgk.branch import BranchRef, MissingBranchError, connect_branch

# hostile: the key absent, present but empty, present but None, and whitespace only
BAD = [
    pytest.param({}, id="missing"),
    pytest.param({"branch_id": ""}, id="empty"),
    pytest.param({"branch_id": None}, id="none"),
    pytest.param({"branch_id": "   "}, id="whitespace"),
    # hostile: not a string at all (a number would build projects/p/branches/123)
    pytest.param({"branch_id": 123}, id="int"),
    pytest.param({"branch_id": 0}, id="zero"),
    pytest.param({"branch_id": ["t-b"]}, id="list"),
]


@pytest.mark.parametrize("extra", BAD)
def test_branch_ref_refuses_a_missing_or_empty_branch_id_naming_the_key(extra):
    with pytest.raises(MissingBranchError, match=r"config\['branch_id'\] is required"):
        BranchRef.from_config({"lakebase_project": "p", **extra})


@pytest.mark.parametrize("extra", BAD)
def test_no_reader_falls_back_to_production(extra, monkeypatch):
    import lgk.walks.nb_runtime as rt
    from lgk import manifest as mf

    calls = []
    monkeypatch.setattr(rt, "connect", lambda p, b: calls.append((p, b)))
    cfg = {"lakebase_project": "p", **extra}
    with pytest.raises(MissingBranchError):
        connect_branch(cfg)
    with pytest.raises(MissingBranchError):
        mf.pg_location(cfg)
    assert calls == []  # never connected anywhere, production least of all


def test_an_explicit_production_branch_is_still_allowed():
    """The refusal is of the missing key only: an operator who names production gets it."""
    assert BranchRef.from_config({"lakebase_project": "p", "branch_id": "production"}).path == \
        "projects/p/branches/production"


def test_an_explicit_branch_builds_the_path():
    ref = BranchRef.from_config({"lakebase_project": "p", "branch_id": "t-b"})
    assert (ref.path, ref.pg_location) == ("projects/p/branches/t-b", "postgres:projects/p/branches/t-b")


def test_missing_branch_error_is_a_value_error():
    """A config error, like every other refused config value in the kit."""
    assert issubclass(MissingBranchError, ValueError)
