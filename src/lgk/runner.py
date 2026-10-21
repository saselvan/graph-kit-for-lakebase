"""Runner seam: abstract interface for invoking toolkit steps."""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


class RunStatus(Enum):
    """Status of a toolkit step."""

    NOT_RUN = "not_run"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class RunResult:
    """Result of a single step invocation."""

    step: str
    status: RunStatus
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "step": self.step,
            "status": self.status.value,
            "error": self.error,
        }


class RunFolder:
    """Manages the run folder and run.json state file."""

    # All toolkit steps
    STEPS = [
        "preflight",
        "profile",
        "propose",
        "privacy",
        "build",
        "enrich",
        "provision",
        "advise",
        "switch",
        "walks",
        "app",
        "notebook",
        "teardown",
    ]

    def __init__(self, path: Path):
        """Initialize run folder and create run.json if needed."""
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.run_json = self.path / "run.json"
        if not self.run_json.exists():
            self._initialize_run_json()

    def _initialize_run_json(self) -> None:
        """Create run.json with all steps marked not_run."""
        state = {}
        for step in self.STEPS:
            state[step] = {
                "status": RunStatus.NOT_RUN.value,
                "error": None,
            }

        with open(self.run_json, "w") as f:
            json.dump(state, f, indent=2)

    def update_step(self, result: RunResult, timing: Optional[Dict[str, Any]] = None) -> None:
        """Update run.json with step result (plus started_at/finished_at/duration_s if given)."""
        with open(self.run_json) as f:
            state = json.load(f)

        state[result.step] = {
            "status": result.status.value,
            "error": result.error,
            **(timing or {}),
        }

        with open(self.run_json, "w") as f:
            json.dump(state, f, indent=2)

    def invalidate_steps(self, steps: List[str], reason: str) -> None:
        """Mark `steps` not_run (with `reason`) so the next run executes them again."""
        with open(self.run_json) as f:
            state = json.load(f)
        for step in steps:
            state[step] = {"status": RunStatus.NOT_RUN.value, "error": None, "invalidated": reason}
        with open(self.run_json, "w") as f:
            json.dump(state, f, indent=2)

    def get_step_status(self, step: str) -> RunStatus:
        """Read current step status from run.json."""
        with open(self.run_json) as f:
            state = json.load(f)

        # If step not in run.json, it hasn't been tracked yet (not_run)
        if step not in state:
            return RunStatus.NOT_RUN

        status_str = state[step]["status"]
        return RunStatus(status_str)


StepFunc = Callable[[Dict[str, Any]], Any]
StepList = List[Tuple[str, StepFunc]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def execute_step(
    run_folder: RunFolder, config: Dict[str, Any], step_name: str, step_func: StepFunc
) -> RunResult:
    """Run one step in this process, respecting idempotency from run.json.

    The single executor behind every runner: the laptop runner calls it directly, and the
    notebook runner's workspace notebook calls it inside the workspace. A succeeded step is
    skipped; a failed or not_run step runs. run.json records status, error and timing.
    A step marked `always_rerun` (the privacy gate) runs every time: its earlier pass says
    nothing about inputs edited since.
    """
    always = getattr(step_func, "always_rerun", False)
    if not always and run_folder.get_step_status(step_name) == RunStatus.SUCCESS:
        logger.info(f"Step {step_name} already succeeded, skipping")
        return RunResult(step=step_name, status=RunStatus.SUCCESS)

    started_at, t0 = _utc_now(), time.monotonic()
    try:
        logger.info(f"Running step: {step_name}")
        changed = step_func(config)
        result = RunResult(step=step_name, status=RunStatus.SUCCESS)
        _invalidate_dependents(run_folder, step_name, step_func, changed)
        logger.info(f"Step {step_name} completed successfully")
    except Exception as e:
        result = RunResult(step=step_name, status=RunStatus.FAILED, error=str(e))
        logger.error(f"Step {step_name} failed: {e}")

    timing = {
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_s": round(time.monotonic() - t0, 6),
    }
    run_folder.update_step(result, timing)
    return result


def _invalidate_dependents(run_folder: RunFolder, step_name: str, step_func: StepFunc,
                           changed: Any) -> None:
    """A step marked `invalidates_on_change` that returns True (its decision changed) marks
    those later steps not_run, so they run again instead of being skipped (the privacy step:
    a new decision rebuilds, never serves a build made under the old one)."""
    dependents = getattr(step_func, "invalidates_on_change", ())
    if changed is True and dependents:
        run_folder.invalidate_steps(list(dependents), f"{step_name} changed its decision")
        logger.info(f"Step {step_name} changed its decision; will rerun {', '.join(dependents)}")


def execute_steps(run_folder: RunFolder, config: Dict[str, Any], steps: StepList) -> List[RunResult]:
    """Run steps in order in this process; stop after the first failure."""
    results = []
    for step_name, step_func in steps:
        result = execute_step(run_folder, config, step_name, step_func)
        results.append(result)
        if result.status == RunStatus.FAILED:
            break
    return results


def resolve_run_dir(config: Dict[str, Any]) -> Path:
    """Run folder from config["run_folder"], else ./runs/<graph_name>."""
    if "run_folder" in config:
        return Path(config["run_folder"])
    return Path.cwd() / "runs" / config["graph_name"]


class Runner(ABC):
    """Abstract interface for invoking toolkit steps."""

    @abstractmethod
    def run_step(
        self,
        step_name: str,
        step_func: Callable[[Dict[str, Any]], Any],
    ) -> RunResult:
        """Run a single toolkit step.

        Args:
            step_name: Name of the step (e.g., "preflight")
            step_func: Function that performs the step, receives config dict

        Returns:
            RunResult with status and any error message
        """

    @abstractmethod
    def run_steps(self, steps: StepList) -> List[RunResult]:
        """Run (name, step_func) pairs in order, stopping after the first failure.

        Returns one RunResult per step attempted.
        """

    def invalidate_steps(self, steps: List[str], reason: str) -> None:
        """Mark `steps` not_run in the run folder's run.json so they run again (a new version).

        Both runners keep `run_folder` locally; the notebook runner pushes it on its next run."""
        self.run_folder.invalidate_steps(steps, reason)


class LaptopRunner(Runner):
    """Runner implementation for laptop/local execution: steps run in this process."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize laptop runner.

        Args:
            config: Configuration dict with at minimum:
                - graph_name: Name of the graph
                - run_folder: Path to run folder (optional, derived from graph_name)
        """
        self.config = config
        self.run_folder = RunFolder(resolve_run_dir(config))

    def run_step(self, step_name: str, step_func: StepFunc) -> RunResult:
        """Run a step, respecting idempotency from run.json."""
        return execute_step(self.run_folder, self.config, step_name, step_func)

    def run_steps(self, steps: StepList) -> List[RunResult]:
        """Run steps in order in this process, stopping after the first failure."""
        return execute_steps(self.run_folder, self.config, steps)
