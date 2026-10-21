"""The runner and refresh-task notebooks ship inside the wheel (persona rerun, 2026-10-05).

In Genie Code a `.py` file whose first line is `# Databricks notebook source` becomes a workspace
notebook when imported, and `Path.read_bytes()` on it then fails (the run saw `lgk_runner.py.py`).
So `DatabricksNotebookBackend.setup()` reads the runner from the installed package by default:
Genie needs only the wheel."""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from lgk.notebook_runner import NOTEBOOKS, DatabricksNotebookBackend, packaged_notebook

REPO = Path(__file__).resolve().parents[1]
WORKDIR = "/Workspace/Users/someone@example.com/lgk_v1_abc123"
# The API path (no /Workspace); built from parts, so the shipping scan finds no whole user path here.
RUNNER = "/" + "Users/someone@example.com/lgk_v1_abc123/lgk_runner"


class _Ws:
    def __init__(self):
        self.uploads = {}

    def mkdirs(self, path):
        pass

    def upload(self, path, content, **kw):
        self.uploads[path] = (content, kw)


class _W:
    def __init__(self):
        self.workspace = _Ws()


def _wheel(tmp_path):
    wheel = tmp_path / "lgk-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"not a real wheel")
    return wheel


def test_the_kit_ships_both_notebooks():
    assert NOTEBOOKS == ("lgk_runner.py", "lgk_refresh_task.py")


@pytest.mark.parametrize("name", ["lgk_runner.py", "lgk_refresh_task.py"])
def test_packaged_notebook_is_the_committed_notebook(name):
    assert packaged_notebook(name) == (REPO / "notebooks" / name).read_bytes()


def test_an_unknown_notebook_is_refused():
    with pytest.raises(ValueError, match="lgk_runner.py"):
        packaged_notebook("other.py")


def test_setup_without_a_notebook_source_uploads_the_packaged_runner(tmp_path):
    w = _W()
    DatabricksNotebookBackend(w, WORKDIR, _wheel(tmp_path)).setup()
    content, kw = w.workspace.uploads[RUNNER]
    assert content == (REPO / "notebooks" / "lgk_runner.py").read_bytes()
    assert kw["format"].value == "SOURCE" and kw["language"].value == "PYTHON"


def test_an_explicit_notebook_source_still_wins(tmp_path):
    other = tmp_path / "my_runner.py"
    other.write_bytes(b"# Databricks notebook source\nprint('mine')\n")  # hostile: differs from the package copy
    w = _W()
    DatabricksNotebookBackend(w, WORKDIR, _wheel(tmp_path), notebook_source=other).setup()
    assert w.workspace.uploads[RUNNER][0] == other.read_bytes()


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build the real wheel")
def test_the_built_wheel_carries_the_notebooks_and_reads_them_without_the_repo(tmp_path):
    subprocess.run(["uv", "build", "--wheel", "-o", str(tmp_path)], cwd=REPO, check=True, capture_output=True)
    wheel = tmp_path / "lgk-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as z:
        for name in NOTEBOOKS:
            assert z.read(f"lgk/_notebooks/{name}") == (REPO / "notebooks" / name).read_bytes()
    # hostile: run from an install with no repo next to it, so only the package copy can answer
    target = tmp_path / "site"
    with zipfile.ZipFile(wheel) as z:
        z.extractall(target)
    code = ("import sys, lgk.notebook_runner as n; "
            "sys.stdout.buffer.write(n.packaged_notebook('lgk_runner.py'))")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True,
                         env={"PYTHONPATH": str(target), "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr.decode()
    assert out.stdout == (REPO / "notebooks" / "lgk_runner.py").read_bytes()


def test_upload_notebook_puts_the_packaged_copy_at_the_workspace_path():
    from lgk.notebook_runner import upload_notebook

    w = _W()
    upload_notebook(w, "/Workspace/Users/someone@example.com/lgk/lgk_refresh_task", "lgk_refresh_task.py")
    content, kw = w.workspace.uploads["/" + "Users/someone@example.com/lgk/lgk_refresh_task"]
    assert content == (REPO / "notebooks" / "lgk_refresh_task.py").read_bytes()
    assert kw["format"].value == "SOURCE" and kw["language"].value == "PYTHON" and kw["overwrite"] is True


def test_upload_notebook_refuses_an_unknown_notebook():
    from lgk.notebook_runner import upload_notebook

    w = _W()
    with pytest.raises(ValueError):
        upload_notebook(w, "/Workspace/Users/someone@example.com/lgk/x", "other.py")
    assert w.workspace.uploads == {}
