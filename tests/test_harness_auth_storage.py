"""Offline: the live harness opts in to file-based CLI token storage only when asked (H3, owner decision A).

The OS keyring fails with exit 44 under many parallel live suites. A person who clones the repo keeps the
default keyring storage; only `LGK_LIVE_AUTH_STORAGE=plaintext` switches this process (and its CLI children)
to `DATABRICKS_AUTH_STORAGE=plaintext`. A value the user set is never overridden.
"""

import os
import subprocess
import sys

import pytest

import conftest
from lgk.harness import apply_live_auth_storage


def test_without_the_opt_in_the_variable_is_not_set():
    env = {"DATABRICKS_CONFIG_PROFILE": "placeholder-profile"}  # hostile: a live profile is configured, no opt-in
    assert apply_live_auth_storage(env) is False
    assert "DATABRICKS_AUTH_STORAGE" not in env


def test_with_the_opt_in_the_variable_is_set():
    env = {"LGK_LIVE_AUTH_STORAGE": "plaintext"}
    assert apply_live_auth_storage(env) is True
    assert env["DATABRICKS_AUTH_STORAGE"] == "plaintext"


def test_a_value_the_user_set_is_never_overridden():
    env = {"LGK_LIVE_AUTH_STORAGE": "plaintext", "DATABRICKS_AUTH_STORAGE": "secure"}  # hostile: user chose
    assert apply_live_auth_storage(env) is False
    assert env["DATABRICKS_AUTH_STORAGE"] == "secure"


def test_an_empty_user_value_is_still_the_users():
    env = {"LGK_LIVE_AUTH_STORAGE": "plaintext", "DATABRICKS_AUTH_STORAGE": ""}  # hostile: set but empty
    apply_live_auth_storage(env)
    assert env["DATABRICKS_AUTH_STORAGE"] == ""


@pytest.mark.parametrize("value", ["", "0", "false", "keyring", "PLAINTEXT "])
def test_an_unknown_opt_in_value_fails_loud(value):
    env = {"LGK_LIVE_AUTH_STORAGE": value}
    if value == "":
        assert apply_live_auth_storage(env) is False  # empty = not opted in
    else:
        with pytest.raises(ValueError, match="LGK_LIVE_AUTH_STORAGE"):
            apply_live_auth_storage(env)
    assert "DATABRICKS_AUTH_STORAGE" not in env


def test_the_default_target_is_this_process_environment(monkeypatch):
    monkeypatch.setenv("LGK_LIVE_AUTH_STORAGE", "plaintext")
    monkeypatch.delenv("DATABRICKS_AUTH_STORAGE", raising=False)
    apply_live_auth_storage()
    assert os.environ["DATABRICKS_AUTH_STORAGE"] == "plaintext"
    # a child process (the databricks CLI the SDK runs) inherits it
    out = subprocess.run([sys.executable, "-c", "import os; print(os.environ.get('DATABRICKS_AUTH_STORAGE'))"],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == "plaintext"


def test_the_root_conftest_applies_it_at_configure(monkeypatch):
    """Every live suite passes through the root conftest's pytest_configure."""
    monkeypatch.setenv("LGK_LIVE_AUTH_STORAGE", "plaintext")
    monkeypatch.delenv("DATABRICKS_AUTH_STORAGE", raising=False)
    cfg = type("Cfg", (), {"addinivalue_line": lambda self, *a: None})()
    conftest.pytest_configure(cfg)
    assert os.environ.get("DATABRICKS_AUTH_STORAGE") == "plaintext"


def test_the_root_conftest_leaves_it_unset_without_the_opt_in(monkeypatch):
    monkeypatch.delenv("LGK_LIVE_AUTH_STORAGE", raising=False)
    monkeypatch.delenv("DATABRICKS_AUTH_STORAGE", raising=False)
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "placeholder-profile")  # hostile: a live profile, no opt-in
    cfg = type("Cfg", (), {"addinivalue_line": lambda self, *a: None})()
    conftest.pytest_configure(cfg)
    assert "DATABRICKS_AUTH_STORAGE" not in os.environ
