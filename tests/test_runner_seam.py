"""Tests for runner seam interface and run folder state management."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lgk.runner import (
    RunResult,
    RunStatus,
    Runner,
    RunFolder,
    LaptopRunner,
)


class TestRunStatus:
    """Test the RunStatus enum and result tracking."""

    def test_status_values(self):
        """Verify RunStatus enum has expected values."""
        assert RunStatus.NOT_RUN.value == "not_run"
        assert RunStatus.IN_PROGRESS.value == "in_progress"
        assert RunStatus.SUCCESS.value == "success"
        assert RunStatus.FAILED.value == "failed"


class TestRunResult:
    """Test RunResult serialization."""

    def test_run_result_to_dict(self):
        """RunResult should serialize to dict for JSON storage."""
        result = RunResult(
            step="preflight",
            status=RunStatus.SUCCESS,
            error=None,
        )
        data = result.to_dict()
        assert data["step"] == "preflight"
        assert data["status"] == "success"
        assert data["error"] is None

    def test_run_result_with_error(self):
        """RunResult should capture error messages."""
        result = RunResult(
            step="build",
            status=RunStatus.FAILED,
            error="Schema not found",
        )
        data = result.to_dict()
        assert data["error"] == "Schema not found"


class TestRunFolder:
    """Test run folder initialization and run.json management."""

    def test_run_folder_creation(self):
        """RunFolder should initialize and create directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "run_001"
            run_folder = RunFolder(run_dir)

            assert run_folder.path == run_dir
            assert run_folder.path.exists()

    def test_run_json_init(self):
        """run.json should be created with all steps marked not_run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "run_001"
            run_folder = RunFolder(run_dir)

            run_json = run_folder.path / "run.json"
            assert run_json.exists()

            with open(run_json) as f:
                state = json.load(f)

            # All steps should be initialized as not_run
            expected_steps = [
                "preflight",
                "profile",
                "propose",
                "build",
                "enrich",
                "provision",
                "advise",  # the advisor on the new, not-yet-served version
                "switch",  # the served views move last
                "walks",
                "app",
                "teardown",
            ]
            for step in expected_steps:
                assert step in state
                assert state[step]["status"] == "not_run"
                assert state[step]["error"] is None

    def test_update_step_status(self):
        """RunFolder.update_step should update run.json atomically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "run_001"
            run_folder = RunFolder(run_dir)

            # Update a step to success
            result = RunResult(
                step="preflight",
                status=RunStatus.SUCCESS,
                error=None,
            )
            run_folder.update_step(result)

            run_json = run_folder.path / "run.json"
            with open(run_json) as f:
                state = json.load(f)

            assert state["preflight"]["status"] == "success"

    def test_read_step_status(self):
        """RunFolder.get_step_status should read current step status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "run_001"
            run_folder = RunFolder(run_dir)

            # Update step
            result = RunResult(
                step="profile",
                status=RunStatus.SUCCESS,
                error=None,
            )
            run_folder.update_step(result)

            # Read it back
            status = run_folder.get_step_status("profile")
            assert status == RunStatus.SUCCESS


class TestRunnerSeam:
    """Test the Runner interface for abstracting SDK/Connect/jobs."""

    def test_runner_is_abstract(self):
        """Runner should define the interface methods."""
        # Runner is an abstract base class
        runner = Mock(spec=Runner)
        runner.run_step = Mock(return_value=RunResult(
            step="preflight",
            status=RunStatus.SUCCESS,
        ))

        result = runner.run_step("preflight", {})
        assert result.step == "preflight"
        assert result.status == RunStatus.SUCCESS


class TestLaptopRunner:
    """Test the laptop runner implementation."""

    def test_laptop_runner_init(self):
        """LaptopRunner should initialize with config."""
        config = {
            "graph_name": "test_graph",
            "project_name": "test_project",
        }
        runner = LaptopRunner(config)
        assert runner.config["graph_name"] == "test_graph"

    def test_run_step_success(self):
        """LaptopRunner.run_step should execute a step and record status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "run_001"

            config = {
                "graph_name": "test_graph",
                "run_folder": str(run_dir),
            }
            runner = LaptopRunner(config)

            # Create a mock step function
            def mock_step_func(config):
                return {"status": "done", "data": "test"}

            # Run the step
            result = runner.run_step("preflight", mock_step_func)

            assert result.step == "preflight"
            assert result.status == RunStatus.SUCCESS

            # Verify run.json was updated
            run_folder = RunFolder(run_dir)
            status = run_folder.get_step_status("preflight")
            assert status == RunStatus.SUCCESS

    def test_run_step_failure(self):
        """LaptopRunner.run_step should catch errors and record them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "run_001"

            config = {
                "graph_name": "test_graph",
                "run_folder": str(run_dir),
            }
            runner = LaptopRunner(config)

            # Create a failing step function
            def failing_step_func(config):
                raise ValueError("Test error")

            # Run the step
            result = runner.run_step("preflight", failing_step_func)

            assert result.step == "preflight"
            assert result.status == RunStatus.FAILED
            assert "Test error" in result.error

            # Verify run.json was updated with error
            run_folder = RunFolder(run_dir)
            status = run_folder.get_step_status("preflight")
            assert status == RunStatus.FAILED

    def test_idempotent_run_skips_completed_step(self):
        """LaptopRunner should skip a step that already succeeded on re-run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "run_001"

            config = {
                "graph_name": "test_graph",
                "run_folder": str(run_dir),
            }
            runner = LaptopRunner(config)

            call_count = 0

            def counting_step_func(config):
                nonlocal call_count
                call_count += 1
                return {"status": "done"}

            # First run
            result1 = runner.run_step("preflight", counting_step_func)
            assert result1.status == RunStatus.SUCCESS
            assert call_count == 1

            # Second run (should be skipped)
            result2 = runner.run_step("preflight", counting_step_func)
            assert result2.status == RunStatus.SUCCESS
            assert call_count == 1  # Not called again

    def test_idempotent_run_retries_failed_step(self):
        """LaptopRunner should retry a step that previously failed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "run_001"

            config = {
                "graph_name": "test_graph",
                "run_folder": str(run_dir),
            }
            runner = LaptopRunner(config)

            call_count = 0

            def flaky_step_func(config):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ValueError("First attempt fails")
                return {"status": "done"}

            # First run (fails)
            result1 = runner.run_step("preflight", flaky_step_func)
            assert result1.status == RunStatus.FAILED
            assert call_count == 1

            # Second run (succeeds)
            result2 = runner.run_step("preflight", flaky_step_func)
            assert result2.status == RunStatus.SUCCESS
            assert call_count == 2


class TestRunSequence:
    """Test running multiple steps in sequence."""

    def test_run_sequence_idempotent(self):
        """Multiple steps should be idempotent: completed ones skipped, failed ones retried."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "run_001"

            config = {
                "graph_name": "test_graph",
                "run_folder": str(run_dir),
            }
            runner = LaptopRunner(config)

            call_sequence = []

            def step1_func(config):
                call_sequence.append("step1")
                return {"status": "done"}

            def step2_func(config):
                call_sequence.append("step2")
                if len(call_sequence) == 2:  # First time fails
                    raise ValueError("step2 failed")
                return {"status": "done"}

            def step3_func(config):
                call_sequence.append("step3")
                return {"status": "done"}

            # First full run - all steps called but step2 fails
            runner.run_step("step1", step1_func)
            runner.run_step("step2", step2_func)
            # Note: step3 is not called on first run if step2 fails (typical workflow engine behavior)

            assert call_sequence == ["step1", "step2"]
            run_folder = RunFolder(run_dir)
            assert run_folder.get_step_status("step1") == RunStatus.SUCCESS
            assert run_folder.get_step_status("step2") == RunStatus.FAILED
            assert run_folder.get_step_status("step3") == RunStatus.NOT_RUN

            # Resume run (step1 skipped, step2 retried, step3 runs)
            runner.run_step("step1", step1_func)
            runner.run_step("step2", step2_func)
            runner.run_step("step3", step3_func)

            assert call_sequence == ["step1", "step2", "step2", "step3"]
            assert run_folder.get_step_status("step1") == RunStatus.SUCCESS
            assert run_folder.get_step_status("step2") == RunStatus.SUCCESS
            assert run_folder.get_step_status("step3") == RunStatus.SUCCESS
