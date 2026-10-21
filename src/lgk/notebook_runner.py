"""Workspace-notebook runner: the second implementation of the runner seam.

The laptop runner calls `lgk.runner.execute_steps` in this process. The notebook runner sends the
same steps to a workspace notebook (notebooks/lgk_runner.py), which calls the same
`execute_steps` there through `run_in_workspace`. The runner only moves things across the
boundary: the run folder (push before, pull after), the config (as JSON, with `run_folder`
pointing at the workspace copy) and the steps (as importable `module:qualname` references).
It adds no step logic, so a golden run gives the same files through both runners.
"""

import importlib
import json
import logging
import re
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from lgk.runtime import ENVIRONMENT_KEY, ENVIRONMENT_VERSION, job_environment, write_runtime_record
from lgk.runner import (
    RunFolder,
    Runner,
    RunResult,
    RunStatus,
    StepFunc,
    StepList,
    execute_steps,
    resolve_run_dir,
)

logger = logging.getLogger(__name__)

_WORKDIR_RE = re.compile(r"/Workspace/Users/" + r"[^/]+/lgk_v1_[0-9a-f]{6}")

# The workspace notebooks ship inside the wheel as lgk/_notebooks/<name> (pyproject force-include),
# so a host with only the wheel (Genie Code) never reads a notebook from the workspace as a file.
NOTEBOOKS = ("lgk_runner.py", "lgk_refresh_task.py")


def packaged_notebook(name: str) -> bytes:
    """The bytes of a kit notebook: the installed package's copy, else the repo's notebooks/ copy
    (a source checkout, where src/lgk has no _notebooks folder)."""
    if name not in NOTEBOOKS:
        raise ValueError(f"no kit notebook {name!r}; the kit ships {', '.join(NOTEBOOKS)}")
    here = Path(__file__).resolve().parent
    for path in (here / "_notebooks" / name, here.parents[1] / "notebooks" / name):
        if path.is_file():
            return path.read_bytes()
    raise FileNotFoundError(f"{name} is in neither {here / '_notebooks'} nor the repo's notebooks/")


def upload_notebook(w: Any, workspace_path: str, name: str) -> None:
    """Upload the kit notebook `name` (packaged in the wheel) to `workspace_path` as a Python
    workspace notebook, replacing what is there. For a host whose CLI cannot read local files."""
    from databricks.sdk.service.workspace import ImportFormat, Language

    w.workspace.upload(_api_path(workspace_path), packaged_notebook(name), format=ImportFormat.SOURCE,
                       language=Language.PYTHON, overwrite=True)


# ------------------------------------------------------------------ step references
def step_ref(func: StepFunc) -> str:
    """`module:qualname` for a module-level step function; refuses anything else."""
    module, qualname = getattr(func, "__module__", None), getattr(func, "__qualname__", "")
    if not module or "<" in qualname or module == "__main__":
        raise ValueError(f"step {func!r} is not importable by module:qualname")
    if resolve_step(f"{module}:{qualname}") is not func:
        raise ValueError(f"step {func!r} is not importable by module:qualname")
    return f"{module}:{qualname}"


def resolve_step(ref: str) -> StepFunc:
    """Import the step function named by a `module:qualname` reference."""
    module_name, qualname = ref.split(":", 1)
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


# -------------------------------------------------------------- workspace-side entry
def run_in_workspace(steps_json: str, config_json: str, environment_version: str = "") -> str:
    """Notebook entry: run the steps in-process against config["run_folder"]; return results JSON.

    Writes `<run_folder>/runtime.json` first (Python and serverless environment of this run), so a
    run that fails on its runtime still says what it ran on.
    """
    config = json.loads(config_json)
    steps = [(name, resolve_step(ref)) for name, ref in json.loads(steps_json)]
    write_runtime_record(Path(config["run_folder"]), environment_version, [name for name, _ in steps])
    results = execute_steps(RunFolder(Path(config["run_folder"])), config, steps)
    return json.dumps([r.to_dict() for r in results])


# ------------------------------------------------------------------- laptop-side runner
class NotebookBackend(Protocol):
    """Where the notebook runs: a workspace (DatabricksNotebookBackend) or a local stand-in."""

    remote_run_dir: str

    def push_run_folder(self, local_dir: Path) -> None: ...

    def run_notebook(self, parameters: Dict[str, str]) -> None: ...

    def pull_run_folder(self, local_dir: Path) -> None: ...


def _remote_config(config: Dict[str, Any], remote_run_dir: str) -> str:
    """Config as JSON with run_folder repointed; refuses values that cannot cross as JSON."""
    remote = dict(config, run_folder=remote_run_dir)
    for key, value in remote.items():
        try:
            json.dumps(value)
        except TypeError as e:
            raise TypeError(f"config[{key!r}] cannot be sent to the notebook as JSON: {e}") from e
    return json.dumps(remote)


def _results_from_run_json(run_folder: RunFolder, steps: StepList) -> List[RunResult]:
    """RunResults for the attempted steps, read from the pulled run.json."""
    state = json.loads(run_folder.run_json.read_text())
    results = []
    for step_name, _ in steps:
        entry = state.get(step_name, {"status": RunStatus.NOT_RUN.value, "error": None})
        status = RunStatus(entry["status"])
        if status == RunStatus.NOT_RUN:
            break
        results.append(RunResult(step=step_name, status=status, error=entry.get("error")))
        if status == RunStatus.FAILED:
            break
    return results


class NotebookRunner(Runner):
    """Runner that executes steps in a workspace notebook (serverless, as the invoking user)."""

    def __init__(self, config: Dict[str, Any], backend: NotebookBackend):
        self.config = config
        self.backend = backend
        self.run_folder = RunFolder(resolve_run_dir(config))

    def run_step(self, step_name: str, step_func: StepFunc) -> RunResult:
        """Run one step through the notebook."""
        return self.run_steps([(step_name, step_func)])[0]

    def run_steps(self, steps: StepList) -> List[RunResult]:
        """Push the run folder, run all steps in one notebook run, pull the run folder back."""
        parameters = {
            "steps": json.dumps([[name, step_ref(func)] for name, func in steps]),
            "config": _remote_config(self.config, self.backend.remote_run_dir),
        }
        self.backend.push_run_folder(self.run_folder.path)
        self.backend.run_notebook(parameters)
        self.backend.pull_run_folder(self.run_folder.path)
        return _results_from_run_json(self.run_folder, steps)


# ---------------------------------------------------------------- Databricks backend
def _api_path(path: str) -> str:
    """Workspace API path format (uses `/Users/` placeholder in docstring, not a real path)."""
    return path[len("/Workspace") :] if path.startswith("/Workspace/") else path


class DatabricksNotebookBackend:
    """Runs notebooks/lgk_runner.py as a serverless one-time job run in a workspace dir.

    `workdir` is a FUSE path such as /Workspace/Users/<me>/lgk_v1_<6hex>. setup() uploads the
    toolkit wheel and the notebook there; cleanup() deletes the whole dir. With no
    `notebook_source`, setup() uploads the runner notebook packaged in the wheel.
    """

    def __init__(
        self, w: Any, workdir: str, wheel: Path, notebook_source: Optional[Path] = None,
        timeout_s: int = 1500,
    ):
        self.w = w
        self.workdir = workdir.rstrip("/")
        # cleanup() deletes workdir recursively: refuse anything but a kit-owned leaf dir,
        # so a bad path can never take out the user's workspace home.
        if not _WORKDIR_RE.fullmatch(self.workdir):
            raise ValueError(f"workdir must match {_WORKDIR_RE.pattern}, got {workdir!r}")
        self.wheel = Path(wheel)
        self.notebook_source = Path(notebook_source) if notebook_source is not None else None
        self.timeout_s = timeout_s
        self.remote_run_dir = f"{self.workdir}/run"
        self.remote_wheel = f"{self.workdir}/{self.wheel.name}"
        self.remote_notebook = f"{self.workdir}/lgk_runner"
        self.last_run_id = None

    def setup(self) -> None:
        """Upload the wheel and the notebook to the work dir."""
        from databricks.sdk.service.workspace import ImportFormat, Language

        ws = self.w.workspace
        ws.mkdirs(_api_path(self.workdir))
        ws.upload(_api_path(self.remote_wheel), self.wheel.read_bytes(),
                  format=ImportFormat.AUTO, overwrite=True)
        source = (self.notebook_source.read_bytes() if self.notebook_source is not None
                  else packaged_notebook("lgk_runner.py"))
        ws.upload(_api_path(self.remote_notebook), source,
                  format=ImportFormat.SOURCE, language=Language.PYTHON, overwrite=True)

    def cleanup(self) -> None:
        """Delete the work dir and everything uploaded into it."""
        from databricks.sdk.errors import NotFound

        try:
            self.w.workspace.delete(_api_path(self.workdir), recursive=True)
        except NotFound:
            pass

    def push_run_folder(self, local_dir: Path) -> None:
        """Replace the remote run folder with the local one."""
        from databricks.sdk.service.workspace import ImportFormat

        self.cleanup_run_folder()
        self.w.workspace.mkdirs(_api_path(self.remote_run_dir))
        for src in sorted(Path(local_dir).rglob("*")):
            if src.is_file():
                dst = f"{self.remote_run_dir}/{src.relative_to(local_dir).as_posix()}"
                self.w.workspace.mkdirs(_api_path(dst.rsplit("/", 1)[0]))
                self.w.workspace.upload(_api_path(dst), src.read_bytes(),
                                        format=ImportFormat.AUTO, overwrite=True)

    def cleanup_run_folder(self) -> None:
        """Delete the remote run folder if present."""
        from databricks.sdk.errors import NotFound

        try:
            self.w.workspace.delete(_api_path(self.remote_run_dir), recursive=True)
        except NotFound:
            pass

    def submit_spec(self, parameters: Dict[str, str]) -> Dict[str, Any]:
        """The jobs.submit arguments: one serverless notebook task on the pinned environment.

        With no environment a serverless notebook task runs environment version 1 (Python 3.10),
        which cannot import the kit; the job environment pins ENVIRONMENT_VERSION (lgk.runtime)
        and carries the wheel as a dependency.
        """
        from databricks.sdk.service import jobs

        task = jobs.SubmitTask(
            task_key="lgk_runner",
            environment_key=ENVIRONMENT_KEY,
            notebook_task=jobs.NotebookTask(
                notebook_path=_api_path(self.remote_notebook),
                base_parameters=dict(parameters, wheel=self.remote_wheel,
                                     environment_version=ENVIRONMENT_VERSION),
                source=jobs.Source.WORKSPACE,
            ),
        )
        return {"run_name": f"lgk-runner-{Path(self.workdir).name}", "tasks": [task],
                "environments": [job_environment([self.remote_wheel])]}

    def run_notebook(self, parameters: Dict[str, str]) -> None:
        """Submit a one-time serverless notebook run (no cluster) and wait; raise on failure."""
        from databricks.sdk.errors import OperationFailed
        from databricks.sdk.service import jobs

        waiter = self.w.jobs.submit(**self.submit_spec(parameters))
        self.last_run_id = waiter.run_id
        try:
            run = waiter.result(timeout=timedelta(seconds=self.timeout_s))
        except OperationFailed as e:
            raise RuntimeError(f"{e}; task error: {self._task_error(waiter.run_id)}") from e
        if run.state.result_state != jobs.RunResultState.SUCCESS:
            raise RuntimeError(
                f"notebook run {run.run_id} ended {run.state.result_state}: "
                f"{self._task_error(run.run_id)} ({run.run_page_url})"
            )

    def _task_error(self, run_id: int) -> str:
        """The notebook task's error text from the run output."""
        task = self.w.jobs.get_run(run_id).tasks[0]
        return str(self.w.jobs.get_run_output(task.run_id).error)

    def pull_run_folder(self, local_dir: Path) -> None:
        """Copy every file of the remote run folder into the local run folder."""
        from databricks.sdk.service.workspace import ObjectType

        root = _api_path(self.remote_run_dir)
        for info in self.w.workspace.list(root, recursive=True):
            if info.object_type != ObjectType.FILE:
                continue
            dst = Path(local_dir) / info.path[len(root) + 1 :]
            dst.parent.mkdir(parents=True, exist_ok=True)
            with self.w.workspace.download(info.path) as f:
                dst.write_bytes(f.read())
