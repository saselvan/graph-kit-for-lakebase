"""P2-runtime: the kit runs on the Python it needs, wherever a kit job runs it (offline).

A. every src/lgk and notebooks/ file parses on Python 3.10 and 3.11 (a persona run died on
   `SyntaxError: f-string: unmatched '('` at walks/sql.py:92 in a 3.10 serverless job);
B. the notebook runner's one-time run and the nightly refresh job pin the serverless
   environment (version 3 = Python 3.12) and every notebook task names it;
C. both notebooks stop at once, with the fix, on a Python older than 3.12;
D. a notebook run writes runtime.json (Python + environment) into the run folder;
E. a live client in the config never reaches the refresh job's parameters.
"""

import ast
import io
import json
import os
import shutil
import subprocess
import sys
import tokenize
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from databricks.sdk.service import jobs

from lgk.notebook_runner import DatabricksNotebookBackend, run_in_workspace, step_ref
from lgk.refresh import job
from tests import parity_support as ps

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = (REPO / "notebooks" / "lgk_runner.py", REPO / "notebooks" / "lgk_refresh_task.py")

# Independent reference, copied from the docs table (not from lgk.runtime):
# https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/ (read 2026-10-04)
DOCS_PYTHON_OF_ENVIRONMENT = {"1": "3.10.12", "2": "3.11.10", "3": "3.12.3", "4": "3.12.3",
                              "5": "3.12.3", "6": "3.12.3"}


def _kit_sources() -> List[Path]:
    return sorted((REPO / "src" / "lgk").rglob("*.py")) + sorted((REPO / "notebooks").glob("*.py"))


# ------------------------------------------------------------------------------------------ A
def pep701_only(source: str) -> List[str]:
    """f-string forms that parse only on Python >= 3.12 (PEP 701), found with the 3.12 tokenizer.

    Before 3.12 an f-string's replacement field could not hold: a string in the same quote as the
    enclosing (single-quoted) f-string, a backslash, or a comment. `ast.parse(feature_version=
    (3, 10))` does not check any of these (the 3.12 tokenizer accepts them), so this walks tokens.
    """
    found: List[str] = []
    stack: List[List[Any]] = []  # [quote, open braces] per enclosing f-string
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        in_expr = bool(stack) and stack[-1][1] > 0
        if tok.type in (tokenize.STRING, tokenize.FSTRING_START) and in_expr:
            body = tok.string.lstrip("rRbBuUfF")
            quote = body[:3] if body[:3] in ('"""', "'''") else body[:1]
            for enclosing, depth in stack:
                if depth > 0 and quote[0] == enclosing[0] and (len(enclosing) == 1 or quote == enclosing):
                    found.append(f"line {tok.start[0]}: nested {quote} string inside a {enclosing} f-string")
            if tok.type == tokenize.STRING and "\\" in tok.string:
                found.append(f"line {tok.start[0]}: backslash inside an f-string expression")
        if tok.type == tokenize.COMMENT and in_expr:
            found.append(f"line {tok.start[0]}: comment inside an f-string expression")
        if tok.type == tokenize.FSTRING_START:
            body = tok.string.lstrip("rRbBuUfF")
            stack.append([body if body in ('"""', "'''") else body[:1], 0])
        elif tok.type == tokenize.FSTRING_END:
            stack.pop()
        elif tok.type == tokenize.OP and stack and tok.string in "{}":
            stack[-1][1] += 1 if tok.string == "{" else -1
    return found


class TestParsesBefore312:
    # hostile: the exact line that broke the persona job, and the second 3.12-only line in the file
    BROKE = [
        '''x = f"IN ({', '.join(f"'{n}'" for n in names)})"\n''',
        '''y = (f"WITH {',\\n'.join(ctes)}\\n")\n''',
        '''z = f"{a  # a comment
}"\n''',
    ]

    @pytest.mark.parametrize("src", BROKE)
    def test_detector_flags_each_pep701_form(self, src):
        assert pep701_only(src) != []

    @pytest.mark.parametrize("src", [
        '''q = ", ".join(f"'{n}'" for n in names)\nx = f"IN ({q})"\n''',
        '''x = f"{d['k']}" + f'{d["k"]}' + f"""{d["k"]}"""\n''',  # different quotes were always legal
        '''x = f"{a:>{w}}"\n''',
    ])
    def test_detector_passes_310_legal_fstrings(self, src):
        assert pep701_only(src) == []

    def test_no_kit_source_uses_312_only_fstrings(self):
        bad = {str(p.relative_to(REPO)): v for p in _kit_sources() if (v := pep701_only(p.read_text()))}
        assert bad == {}

    @pytest.mark.parametrize("version", ["3.10", "3.11"])
    def test_every_kit_source_compiles_on_a_real_older_python(self, version):
        """The authoritative check: the older interpreter's own compiler (skipped only where absent)."""
        exe = shutil.which(f"python{version}")
        if exe is None:
            pytest.skip(f"python{version} not installed (pep701_only still runs)")
        files = [str(p) for p in _kit_sources()]
        code = ("import sys\nfor f in sys.argv[1:]:\n    try: compile(open(f).read(), f, 'exec')\n"
                "    except SyntaxError as e: print(f'{f}:{e.lineno}: {e.msg}')\n")
        out = subprocess.run([exe, "-c", code, *files], capture_output=True, text=True, timeout=120)
        assert (out.returncode, out.stdout, out.stderr) == (0, "", "")


# ------------------------------------------------------------------------------------------ B
def _docs_python(environment_version: str) -> str:
    return DOCS_PYTHON_OF_ENVIRONMENT[environment_version]


WORKDIR = "/Workspace/Users/me@example.com/lgk_v1_abc123"


def _backend(w: Any = None) -> DatabricksNotebookBackend:
    return DatabricksNotebookBackend(w, WORKDIR, Path("dist/lgk-0.1.0-py3-none-any.whl"),
                                     Path("notebooks/lgk_runner.py"))


def _succeeding_w(calls: List[Dict[str, Any]]) -> Any:
    run = SimpleNamespace(run_id=7, state=SimpleNamespace(result_state=jobs.RunResultState.SUCCESS),
                          run_page_url="u")
    waiter = SimpleNamespace(run_id=7, result=lambda timeout: run)
    return SimpleNamespace(jobs=SimpleNamespace(submit=lambda **kw: calls.append(kw) or waiter))


class TestRunnerPinsEnvironment:
    def test_submitted_run_pins_python_312_and_the_task_uses_it(self):
        calls: List[Dict[str, Any]] = []
        _backend(_succeeding_w(calls)).run_notebook({"steps": "[]", "config": "{}"})
        [kw] = calls
        envs = [e.as_dict() for e in kw["environments"]]
        [task] = [t.as_dict() for t in kw["tasks"]]
        assert len(envs) == 1
        assert task["environment_key"] == envs[0]["environment_key"]  # the task runs on the pin
        assert _docs_python(envs[0]["spec"]["environment_version"]).startswith("3.12")
        # the wheel is a dependency of that environment, from the work dir the backend uploaded to
        assert envs[0]["spec"]["dependencies"] == [f"{WORKDIR}/lgk-0.1.0-py3-none-any.whl"]
        params = task["notebook_task"]["base_parameters"]
        assert params["environment_version"] == envs[0]["spec"]["environment_version"]
        assert params["wheel"] == f"{WORKDIR}/lgk-0.1.0-py3-none-any.whl"
        assert "existing_cluster_id" not in task and "new_cluster" not in task  # still serverless


class TestRefreshJobPinsEnvironment:
    def test_every_notebook_task_names_the_pinned_environment(self, tmp_path):
        from tests.refresh_support import fake_client, seed_run_folder

        w, calls = fake_client(job_id=9)
        seed_run_folder(tmp_path, enriched=True)
        cfg = _refresh_config(run_folder=str(tmp_path), sdp_pipeline_id="p-1")
        job.create_refresh_job(cfg, notebook_path="/Workspace/Users/user@example.com/x/lgk_refresh_task",
                               wheel_path="/Workspace/Users/user@example.com/x/lgk-0.1.0-py3-none-any.whl", w=w)
        [kw] = calls
        envs = [e.as_dict() for e in kw["environments"]]
        assert len(envs) == 1
        assert _docs_python(envs[0]["spec"]["environment_version"]).startswith("3.12")
        assert envs[0]["spec"]["dependencies"] == ["/Workspace/Users/user@example.com/x/lgk-0.1.0-py3-none-any.whl"]
        tasks = [t.as_dict() for t in kw["tasks"]]
        notebook_tasks = [t for t in tasks if "notebook_task" in t]
        assert [t["task_key"] for t in notebook_tasks] == ["stamp", "enrich", "sync", "reapply"]
        assert {t["environment_key"] for t in notebook_tasks} == {envs[0]["environment_key"]}
        # hostile: a pipeline task cannot take an environment (the API refuses one)
        assert "environment_key" not in next(t for t in tasks if t["task_key"] == "sdp_pipeline")


# ------------------------------------------------------------------------------------------ C
def _require_python(notebook: Path):
    """The notebook's own require_python, executed from its source (it runs before the kit installs)."""
    tree = ast.parse(notebook.read_text())
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "require_python")
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "MIN_PYTHON" for t in n.targets))]
    ns: Dict[str, Any] = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(notebook), "exec"), ns)
    return ns["require_python"]


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
class TestNotebookStopsOnOldPython:
    # hostile: the exact edges (3.12.0 passes, 3.11.9 fails) and the persona's real 3.10.12
    @pytest.mark.parametrize("ok", [(3, 12, 0), (3, 12, 3), (3, 13, 1), (4, 0, 0)])
    def test_312_and_later_pass(self, notebook, ok):
        _require_python(notebook)(ok)

    @pytest.mark.parametrize("old", [(3, 11, 9), (3, 10, 12), (2, 7, 18)])
    def test_older_python_stops_with_the_fix(self, notebook, old):
        with pytest.raises(RuntimeError) as e:
            _require_python(notebook)(old)
        msg = str(e.value)
        assert ">= 3.12" in msg and ".".join(map(str, old)) in msg
        assert "environment version 3" in msg and "environment_key" in msg

    def test_the_wheel_install_keeps_the_python_pin(self, notebook):
        # the old waiver let a 3.10 job install the 3.12-only wheel and die later on a SyntaxError
        assert "--ignore-requires-python" not in notebook.read_text()

    def test_check_runs_first_on_the_real_python(self, notebook):
        src = notebook.read_text()
        call = src.index("\nrequire_python(sys.version_info)\n")
        assert call < src.index("dbutils.widgets") and call < src.index("pip")


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_really_stops_on_python_310(tmp_path, notebook):
    """# hostile: the real notebook on a real Python 3.10, through the parity driver: no traceback
    from deep inside the kit, the clear message, before any kit import."""
    exe = shutil.which("python3.10")
    if exe is None:
        pytest.skip("python3.10 not installed")
    out = subprocess.run([exe, "-c", ps._DRIVER, str(notebook), json.dumps({"wheel": ""})],
                         capture_output=True, text=True, timeout=60, cwd=str(tmp_path))
    assert out.returncode != 0
    assert "RuntimeError: lgk needs Python >= 3.12; this notebook runs Python 3.10." in out.stderr
    assert "lgk/" not in out.stderr  # stopped before anything of the kit was imported


# ------------------------------------------------------------------------------------------ D
class TestRuntimeRecord:
    def test_run_in_workspace_writes_python_and_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "client.3.6")
        run_dir = ps.seed_propose_golden(tmp_path / "run")
        cfg = dict(ps.golden_config(run_dir), run_folder=str(run_dir))
        run_in_workspace(json.dumps([["propose", step_ref(ps.propose)]]), json.dumps(cfg), "3")
        rec = json.loads((run_dir / "runtime.json").read_text())
        assert rec["python_version"] == sys.version
        assert rec["python_version_info"] == list(sys.version_info[:3])
        assert rec["environment_version"] == "client.3.6"
        assert rec["environment_version_requested"] == "3"
        assert rec["steps"] == ["propose"]

    def test_written_even_when_the_step_fails(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        cfg = dict(ps.golden_config(run_dir), run_folder=str(run_dir))
        out = json.loads(run_in_workspace(json.dumps([["fail", step_ref(ps.failing_step)]]), json.dumps(cfg)))
        assert out[0]["status"] == "failed"
        rec = json.loads((run_dir / "runtime.json").read_text())
        assert rec["environment_version"] is None and rec["environment_version_requested"] is None
        assert rec["python_version"] == sys.version

    def test_written_before_the_steps_run(self, tmp_path, monkeypatch):
        # hostile: a run whose executor itself dies must still leave the record behind
        import lgk.notebook_runner as nr

        def boom(*a, **k):
            raise RuntimeError("executor died")

        monkeypatch.setattr(nr, "execute_steps", boom)
        run_dir = tmp_path / "run"
        cfg = dict(ps.golden_config(run_dir), run_folder=str(run_dir))
        with pytest.raises(RuntimeError, match="executor died"):
            run_in_workspace(json.dumps([["propose", step_ref(ps.propose)]]), json.dumps(cfg), "3")
        assert json.loads((run_dir / "runtime.json").read_text())["environment_version_requested"] == "3"

    def test_the_notebook_passes_the_requested_version_through(self, tmp_path):
        """The real notebook source, through the parity driver, with the widget the backend sets."""
        run_dir = ps.seed_propose_golden(tmp_path / "run")
        cfg = dict(ps.golden_config(run_dir), run_folder=str(run_dir))
        params = {"wheel": "", "steps": json.dumps([["propose", step_ref(ps.propose)]]),
                  "config": json.dumps(cfg), "environment_version": "3"}
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO)]))
        out = subprocess.run([sys.executable, "-c", ps._DRIVER, str(NOTEBOOKS[0]), json.dumps(params)],
                             capture_output=True, text=True, timeout=120, cwd=str(tmp_path), env=env)
        assert out.returncode == 0, out.stderr[-2000:]
        assert json.loads((run_dir / "runtime.json").read_text())["environment_version_requested"] == "3"

    def test_notebook_run_through_the_runner_brings_it_back(self, tmp_path):
        from lgk.notebook_runner import NotebookRunner

        run_dir = ps.seed_propose_golden(tmp_path / "nb")
        backend = ps.LocalProcessBackend(tmp_path / "ws")
        NotebookRunner(ps.golden_config(run_dir), backend).run_steps(ps.PROPOSE_GOLDEN_STEPS)
        rec = json.loads((run_dir / "runtime.json").read_text())
        assert rec["environment_version"] == "client.parity-fake"  # the notebook process wrote it
        assert rec["steps"] == [name for name, _ in ps.PROPOSE_GOLDEN_STEPS]


# ------------------------------------------------------------------------------------------ E
class WorkspaceClient:  # same class name as the SDK's; not JSON-serializable, like the real one
    def __init__(self):
        self.config = SimpleNamespace(host="https://workspace.example.com")


def _refresh_config(**extra: Any) -> Dict[str, Any]:
    cfg = {"graph_name": "provider_graph", "lakebase_project": "proj", "lakebase_catalog": "cat",
           "kit_schema": "lgk_v1_abc123", "served_schema": "lgk_served_abc123", "branch_id": "t-x",
           "run_folder": "/Workspace/Users/user@example.com/lgk_v1_abc123/run", "warehouse_id": "wh", "version": 1}
    cfg.update(extra)
    return cfg


# the workspace copy of a laptop run folder (lgk.refresh.nightly_folder), next to the notebook
COPY = "/Workspace/Users/user@example.com/x/runs/provider_graph"


class TestRefreshJobDropsLiveClients:
    def _created_config(self, cfg) -> List[Dict[str, Any]]:
        from tests.refresh_support import fake_client, seed_run_folder

        seed_run_folder(Path(cfg["run_folder"]))
        w, calls = fake_client(job_id=1)
        job.create_refresh_job(cfg, notebook_path="/Workspace/Users/user@example.com/x/nb", wheel_path="/Workspace/Users/user@example.com/x/w.whl",
                               w=w)
        [kw] = calls
        json.dumps([t.as_dict() for t in kw["tasks"]])  # the whole job spec is JSON
        return [json.loads(t.notebook_task.base_parameters["config"]) for t in kw["tasks"] if t.notebook_task]

    def test_top_level_and_nested_clients_never_reach_the_job(self, caplog, tmp_path):
        cfg = _refresh_config(run_folder=str(tmp_path),
            workspace_client=WorkspaceClient(),
            pg_connect=lambda: None,
            # hostile: a client one level down under an innocent key, and two levels down in a list
            preflight={"w": WorkspaceClient(), "keep": 1, "deeper": [{"client": WorkspaceClient()}, "x"]},
            # hostile: a client as a list element itself (dropped, not kept as null), a non-string key
            clients=[WorkspaceClient(), "y"],
            odd_keys={1: "a", "k": 2},
            local_path=PurePosixPath("/Workspace/Users/user@example.com/x/run"),
            tables=("a", "b"),
        )
        configs = self._created_config(cfg)
        assert len(configs) == 3  # build, sync, reapply: every notebook task gets the clean config
        expected = dict(_refresh_config(run_folder=COPY), preflight={"keep": 1, "deeper": [{}, "x"]},
                        local_path="/Workspace/Users/user@example.com/x/run", tables=["a", "b"], clients=["y"],
                        odd_keys={"k": 2})
        assert all(c == expected for c in configs)
        assert "config['workspace_client'] (WorkspaceClient)" in caplog.text
        assert "config['pg_connect'] (function)" in caplog.text
        assert "config['clients'][0] (WorkspaceClient)" in caplog.text
        assert "config['odd_keys'][1] (non-string key)" in caplog.text
        assert "config['preflight']['w'] (WorkspaceClient)" in caplog.text
        assert "config['preflight']['deeper'][0]['client'] (WorkspaceClient)" in caplog.text

    def test_caller_config_is_left_as_it_was(self, tmp_path):
        client = WorkspaceClient()
        cfg = _refresh_config(run_folder=str(tmp_path), workspace_client=client, preflight={"w": client})
        self._created_config(cfg)
        assert cfg["workspace_client"] is client and cfg["preflight"]["w"] is client

    def test_a_clean_config_travels_unchanged(self, tmp_path):
        clean = _refresh_config(run_folder=str(tmp_path))
        assert self._created_config(dict(clean))[0] == dict(clean, run_folder=COPY)  # only the folder moves


def test_environment_pins_agree_across_kit_jobs():
    """The bundle's jobs (lgk.export) and the workspace jobs use the same pinned version."""
    from lgk import export, runtime

    assert export.ENVIRONMENT_VERSION == runtime.ENVIRONMENT_VERSION
