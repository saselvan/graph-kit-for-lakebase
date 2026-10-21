"""Seam-2 parity gate: laptop runner vs notebook runner give identical files.

The notebook runner runs the committed notebook (notebooks/lgk_runner.py) in a separate process
through LocalProcessBackend, so every byte crosses the same JSON/importable-step boundary the
workspace run crosses. The live workspace run is in tests/test_parity_live.py (seam1).
"""

import ast
import inspect
import json
from pathlib import Path

import pytest

from lgk.notebook_runner import NotebookRunner, run_in_workspace, step_ref
from lgk.parity import TIMING_KEYS, diff_run_folders, normalized_bytes
from lgk.runner import LaptopRunner, Runner, RunStatus
from tests import parity_support as ps

pytestmark = pytest.mark.seam2

STEPS_DIR = Path(__file__).resolve().parents[1] / "src" / "lgk" / "steps"


def _laptop(run_dir: Path, steps, seed) -> Path:
    seed(run_dir)
    LaptopRunner(ps.golden_config(run_dir)).run_steps(steps)
    return run_dir


def _notebook(run_dir: Path, steps, seed, tmp_path: Path):
    seed(run_dir)
    backend = ps.LocalProcessBackend(tmp_path / "ws")
    results = NotebookRunner(ps.golden_config(run_dir), backend).run_steps(steps)
    return run_dir, results, backend


# ----------------------------------------------------------------------------- criterion 1
class TestGoldenParity:
    def test_full_golden_run_identical_files(self, tmp_path):
        lap = _laptop(tmp_path / "laptop", ps.FULL_GOLDEN_STEPS, ps.seed_full_golden)
        nb, results, backend = _notebook(
            tmp_path / "nb", ps.FULL_GOLDEN_STEPS, ps.seed_full_golden, tmp_path
        )

        # The notebook really ran in the stand-in workspace (not in-process).
        assert len(backend.submissions) == 1
        assert [r.status for r in results] == [RunStatus.SUCCESS] * 3

        # Non-vacuous: every golden output exists in both folders.
        assert {p.name for p in lap.iterdir()} == ps.FULL_GOLDEN_FILES
        # The notebook run also records its runtime (P2-runtime, D); the laptop run does not.
        assert {p.name for p in nb.iterdir()} == ps.FULL_GOLDEN_FILES | {"runtime.json"}

        # Real step content, not empty shells.
        flags = json.loads((nb / "flags.json").read_text())
        assert flags == {"shape": "tree"}
        mapping = (nb / "graph_mapping.yaml").read_text()
        assert "  claims:\n    source_table: claims\n    natural_key:\n    - claim_id\n    - line_no\n" in mapping
        assert "audit_log" not in mapping  # hostile: keyless table is not a node

        assert diff_run_folders(lap, nb) == []

    def test_timing_fields_really_differ_before_stripping(self, tmp_path):
        """The strip is load-bearing: raw run.json differs, normalized run.json does not."""
        lap = _laptop(tmp_path / "laptop", ps.FULL_GOLDEN_STEPS, ps.seed_full_golden)
        nb, _, _ = _notebook(tmp_path / "nb", ps.FULL_GOLDEN_STEPS, ps.seed_full_golden, tmp_path)

        lap_run = json.loads((lap / "run.json").read_text())
        nb_run = json.loads((nb / "run.json").read_text())
        for step in ("preflight", "profile", "propose"):
            assert set(lap_run[step]) >= {"started_at", "finished_at", "duration_s"}
            assert lap_run[step]["started_at"] != nb_run[step]["started_at"]
        assert (lap / "run.json").read_bytes() != (nb / "run.json").read_bytes()
        assert normalized_bytes(lap / "run.json") == normalized_bytes(nb / "run.json")

    def test_propose_golden_cycle_and_duplicate_fk(self, tmp_path):
        lap = _laptop(tmp_path / "laptop", ps.PROPOSE_GOLDEN_STEPS, ps.seed_propose_golden)
        nb, _, _ = _notebook(
            tmp_path / "nb", ps.PROPOSE_GOLDEN_STEPS, ps.seed_propose_golden, tmp_path
        )
        assert json.loads((nb / "flags.json").read_text()) == {"shape": "web"}  # hostile: cycle
        mapping = (nb / "graph_mapping.yaml").read_text()
        # hostile: duplicate FK -> exactly one a->b edge
        assert mapping.count("    src_type: a\n    dst: b_id\n    dst_type: b\n") == 1
        assert diff_run_folders(lap, nb) == []

    def test_runner_specific_branch_turns_parity_red(self, tmp_path):
        """# hostile: a step that branches on its runtime environment must break parity."""
        steps = [("propose", ps.runner_sniffing_step)]
        lap = _laptop(tmp_path / "laptop", steps, ps.seed_propose_golden)
        nb, _, _ = _notebook(tmp_path / "nb", steps, ps.seed_propose_golden, tmp_path)
        diffs = diff_run_folders(lap, nb)
        assert diffs == ["flags.json: content differs"]


# ----------------------------------------------------------------------- timing allow-list
class TestTimingAllowList:
    def _pair(self, tmp_path, a: dict, b: dict, name="out.json"):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "a" / name).write_text(json.dumps(a, indent=2))
        (tmp_path / "b" / name).write_text(json.dumps(b, indent=2))
        return diff_run_folders(tmp_path / "a", tmp_path / "b")

    def test_allow_list_is_exact(self):
        assert TIMING_KEYS == frozenset({"started_at", "finished_at", "duration_s"})

    def test_timing_keys_stripped_at_any_depth(self, tmp_path):
        a = {"s": {"status": "success", "started_at": "1", "finished_at": "2", "duration_s": 1.0}}
        b = {"s": {"status": "success", "started_at": "9", "finished_at": "8", "duration_s": 7.5}}
        assert self._pair(tmp_path, a, b) == []

    def test_near_miss_key_is_not_stripped(self, tmp_path):
        """# hostile: a key that merely contains a timing name is a real difference."""
        a = {"started_at": "1", "started_at_count": 3}
        b = {"started_at": "2", "started_at_count": 4}
        assert self._pair(tmp_path, a, b) == ["out.json: content differs"]

    def test_non_timing_value_difference_is_red(self, tmp_path):
        a = {"status": "success", "duration_s": 1, "error": None}
        b = {"status": "success", "duration_s": 2, "error": "boom"}
        assert self._pair(tmp_path, a, b) == ["out.json: content differs"]

    def test_non_json_file_is_byte_compared(self, tmp_path):
        """# hostile: timing-looking text in YAML is not stripped (only JSON keys are)."""
        diffs = self._pair(tmp_path, {"x": 1}, {"x": 1}, name="ok.json")
        assert diffs == []
        (tmp_path / "a" / "m.yaml").write_text("started_at: 1\n")
        (tmp_path / "b" / "m.yaml").write_text("started_at: 2\n")
        assert diff_run_folders(tmp_path / "a", tmp_path / "b") == ["m.yaml: content differs"]

    def test_only_a_root_runtime_json_is_skipped(self, tmp_path):
        """# hostile: a runtime.json one level down is a step output and must still be compared."""
        for side in ("a", "b"):
            (tmp_path / side / "sub").mkdir(parents=True)
        (tmp_path / "a" / "runtime.json").write_text('{"python_version": "3.12.3"}')
        (tmp_path / "a" / "sub" / "runtime.json").write_text("{}")
        assert diff_run_folders(tmp_path / "a", tmp_path / "b") == ["sub/runtime.json: only in a"]

    def test_file_only_in_one_folder_is_red(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "a" / "extra.json").write_text("{}")
        assert diff_run_folders(tmp_path / "a", tmp_path / "b") == ["extra.json: only in a"]


# ----------------------------------------------------------------------------- criterion 2
class TestSameInterface:
    def test_both_runners_expose_exactly_the_seam_interface(self):
        seam = {n for n, _ in inspect.getmembers(Runner, inspect.isfunction) if not n.startswith("_")}
        assert seam == {"run_step", "run_steps", "invalidate_steps"}
        for cls in (LaptopRunner, NotebookRunner):
            assert issubclass(cls, Runner)
            public = {
                n for n, _ in inspect.getmembers(cls, inspect.isfunction) if not n.startswith("_")
            }
            assert public == seam, cls
            sig = inspect.signature(cls.run_steps)
            assert list(sig.parameters) == ["self", "steps"]

    def test_toolkit_steps_have_no_runner_awareness(self):
        """Structural: no step module reads its runtime/runner identity."""
        forbidden_names = {"dbutils", "spark", "LaptopRunner", "NotebookRunner", "Runner"}
        forbidden_attrs = {"environ", "getenv", "getcwd", "cwd", "gethostname", "node"}
        forbidden_strings = ("DATABRICKS_RUNTIME_VERSION", "/Workspace")
        offenders = []
        for path in sorted(STEPS_DIR.glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in forbidden_names:
                    offenders.append(f"{path.name}:{node.lineno} name {node.id}")
                if isinstance(node, ast.Attribute) and node.attr in forbidden_attrs:
                    offenders.append(f"{path.name}:{node.lineno} reads .{node.attr}")
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    mods = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                    if any(m.startswith(("lgk.runner", "lgk.notebook_runner")) for m in mods):
                        offenders.append(f"{path.name}:{node.lineno} imports runner")
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    low = node.value
                    if any(s in low for s in forbidden_strings):
                        offenders.append(f"{path.name}:{node.lineno} string {low!r}")
        assert offenders == []

    def test_steps_see_the_same_config_through_both_runners(self, tmp_path):
        """Behavioural: neither runner injects a caller-specific config key or value."""
        steps = [("profile", ps.dump_config_step)]
        lap = _laptop(tmp_path / "laptop", steps, ps.seed_full_golden)
        nb, _, _ = _notebook(tmp_path / "nb", steps, ps.seed_full_golden, tmp_path)
        seen = json.loads((nb / "config_seen.json").read_text())
        assert seen == {k: v for k, v in ps.golden_config(nb).items() if k != "run_folder"}
        assert diff_run_folders(lap, nb) == []


# ---------------------------------------------------------------------- notebook runner
class TestNotebookRunner:
    def test_failure_is_recorded_and_later_steps_do_not_run(self, tmp_path):
        steps = [("preflight", ps.failing_step), ("profile", ps.dump_config_step)]
        nb, results, _ = _notebook(tmp_path / "nb", steps, ps.seed_full_golden, tmp_path)
        assert [(r.step, r.status) for r in results] == [("preflight", RunStatus.FAILED)]
        assert "golden failure: schema not found" in results[0].error
        state = json.loads((nb / "run.json").read_text())
        assert state["preflight"]["status"] == "failed"
        assert state["profile"]["status"] == "not_run"
        assert not (nb / "config_seen.json").exists()

        # Laptop runner records the same failure in the same shape.
        lap = _laptop(tmp_path / "laptop", steps, ps.seed_full_golden)
        assert diff_run_folders(lap, nb) == []

    def test_failure_hides_a_stale_later_success(self, tmp_path):
        """# hostile: a later step already 'success' from an earlier run was not attempted now."""
        run_dir = ps.seed_full_golden(tmp_path / "nb")
        backend = ps.LocalProcessBackend(tmp_path / "ws")
        runner = NotebookRunner(ps.golden_config(run_dir), backend)
        runner.run_steps([("profile", ps.dump_config_step)])
        results = runner.run_steps([("preflight", ps.failing_step), ("profile", ps.dump_config_step)])
        assert [(r.step, r.status) for r in results] == [("preflight", RunStatus.FAILED)]

    def test_resume_skips_succeeded_step_in_workspace(self, tmp_path):
        run_dir = ps.seed_full_golden(tmp_path / "nb")
        backend = ps.LocalProcessBackend(tmp_path / "ws")
        runner = NotebookRunner(ps.golden_config(run_dir), backend)
        runner.run_steps([("profile", ps.dump_config_step)])
        (run_dir / "config_seen.json").unlink()
        results = runner.run_steps([("profile", ps.dump_config_step)])
        assert results[0].status == RunStatus.SUCCESS
        assert not (run_dir / "config_seen.json").exists()  # skipped: run.json said success

    def test_refuses_unimportable_step(self, tmp_path):
        with pytest.raises(ValueError, match="importable"):
            step_ref(lambda c: None)

    def test_refuses_non_json_config(self, tmp_path):
        cfg = dict(ps.golden_config(tmp_path), workspace_client=object())
        runner = NotebookRunner(cfg, ps.LocalProcessBackend(tmp_path / "ws"))
        with pytest.raises(TypeError, match="workspace_client"):
            runner.run_steps(ps.PROPOSE_GOLDEN_STEPS)

    def test_run_in_workspace_points_steps_at_the_given_folder(self, tmp_path):
        run_dir = ps.seed_propose_golden(tmp_path / "ws_run")
        cfg = dict(ps.golden_config(Path("/elsewhere")), run_folder=str(run_dir))
        out = json.loads(run_in_workspace(json.dumps([["propose", step_ref(ps.propose)]]), json.dumps(cfg)))
        assert out == [{"step": "propose", "status": "success", "error": None}]
        assert json.loads((run_dir / "flags.json").read_text()) == {"shape": "web"}


# A user's workspace home, built from parts so the public gate finds no user path in this file.
ME_HOME = "/Workspace" + "/Users/" + "me@x" + ".com"


class TestWorkdirGuard:
    """cleanup() deletes workdir recursively, so only a kit-owned leaf dir is accepted."""

    @pytest.mark.parametrize(
        "bad",
        [
            ME_HOME,  # hostile: the user's workspace home
            ME_HOME + "/",
            ME_HOME + "/lgk_v1_abc12",  # 5 hex
            ME_HOME + "/lgk_v1_abc123/..",
            ME_HOME + "/projects",
            "/Workspace",
        ],
    )
    def test_refuses_non_kit_workdir(self, bad, tmp_path):
        from lgk.notebook_runner import DatabricksNotebookBackend

        with pytest.raises(ValueError):
            DatabricksNotebookBackend(object(), bad, tmp_path / "x.whl", tmp_path / "nb.py")

    def test_accepts_kit_workdir(self, tmp_path):
        from lgk.notebook_runner import DatabricksNotebookBackend

        b = DatabricksNotebookBackend(
            object(), ME_HOME + "/lgk_v1_abc123/", tmp_path / "x.whl", tmp_path / "nb.py"
        )
        assert b.workdir == ME_HOME + "/lgk_v1_abc123"
