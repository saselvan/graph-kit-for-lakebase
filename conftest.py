"""Pytest configuration and fixtures for lakebase-graph-kit integration tests.

Provides:
- Seam-1 (`seam1` marker: live Lakebase integration test) and Seam-2 (`seam2` marker: mock/no-cloud unit test) test separation
- Isolated test branches with automatic cleanup
- Serialization markers for provision-type tests
- `record_metric`: the opt-in numbers recorder (LGK_METRICS_FILE; off by default)
- Live test opt-in: --require-live selects live tests; default run deselects them
"""

import functools
import logging
import os
import tempfile
import threading
from pathlib import Path

import pytest

from lgk.harness import BranchLeasor, SeamGuard, apply_live_auth_storage
from lgk.metrics import MetricsRecorder, git_head, metrics_path, metrics_run


class RequireLivePlugin:
    """Plugin to track skipped tests when --require-live is set.

    Uses per-session state via config.stash to avoid module-level mutable state.
    Registered only when --require-live is True and config has the option attribute.
    """

    # Class-level stash key instance (unique per class)
    STASH_KEY = pytest.StashKey()

    def __init__(self, config):
        """Initialize plugin with reference to config for stash access."""
        self.config = config
        config.stash[self.STASH_KEY] = []

    def pytest_collectreport(self, report):
        """Track collection-level skips (e.g., pytest.importorskip)."""
        if getattr(report, "outcome", None) != "skipped":
            return

        # Extract nodeid from report
        nodeid = getattr(report, "nodeid", None)
        if nodeid is None and hasattr(report, "collector"):
            nodeid = getattr(report.collector, "nodeid", None)
        if nodeid is None:
            nodeid = str(getattr(report, "fspath", "unknown"))

        self.config.stash[self.STASH_KEY].append(nodeid)

    def pytest_runtest_logreport(self, report):
        """Track runtime skips (excluding xfails).

        Tracks skips in all phases (setup, call, teardown) to catch marked skips
        like @pytest.mark.skip, which may skip in the setup phase.
        """
        # Track skips in any phase, but exclude xfails
        if report.skipped and not hasattr(report, "wasxfail"):
            self.config.stash[self.STASH_KEY].append(report.nodeid)

    def get_skipped(self) -> list:
        """Return the list of skipped test nodeids."""
        return self.config.stash.get(self.STASH_KEY, [])

    def pytest_sessionfinish(self, session):
        """Fail session if any tests were skipped."""
        if self.get_skipped():
            session.exitstatus = 1


def _get_require_live(config) -> bool:
    """Check if --require-live is set. Returns False if config lacks the option (e.g., fake config).

    Helper to safely read the flag without assuming config has getoption.
    """
    if not hasattr(config, "getoption"):
        return False
    try:
        return config.getoption("--require-live")
    except (AttributeError, ValueError):
        return False


def _cut_databricks_credentials():
    """Remove Databricks credentials from the process environment.

    In the default run (no --require-live), the test process must never reach
    a live Databricks workspace. This function removes all DATABRICKS_* env vars
    and sets DATABRICKS_CONFIG_FILE to a non-existent path inside a temporary
    directory, ensuring credential reads always fail safely. The config file
    is never created.
    """
    # Remove all DATABRICKS_* env vars except DATABRICKS_CONFIG_FILE
    keys_to_remove = [k for k in os.environ if k.startswith("DATABRICKS_") and k != "DATABRICKS_CONFIG_FILE"]
    for key in keys_to_remove:
        del os.environ[key]

    # Set DATABRICKS_CONFIG_FILE to a path that doesn't exist
    tmpdir = tempfile.mkdtemp()
    os.environ["DATABRICKS_CONFIG_FILE"] = os.path.join(tmpdir, "no-databrickscfg")


# Lakebase connection configuration
# NOTE: No personal or workspace names as defaults — use environment variables
LAKEBASE_PROFILE = os.environ.get("LAKEBASE_PROFILE")
LAKEBASE_PROJECT = os.environ.get("LAKEBASE_PROJECT")
LAKEBASE_CATALOG = os.environ.get("LAKEBASE_CATALOG")
LAKEBASE_TEST_BASE_BRANCH = "test-base"

# Global lock for serializing provision tests
_provision_lock = threading.Lock()


def pytest_addoption(parser):
    """Register --require-live flag: include live tests in the session."""
    parser.addoption(
        "--require-live",
        action="store_true",
        default=False,
        help="Include live tests in the run. Without this flag, live tests (seam1 or workspace_client) are deselected.",
    )


def pytest_configure(config):
    """Register custom markers; cut Databricks credentials in default runs; opt in to file token storage if LGK_LIVE_AUTH_STORAGE=plaintext.

    Register RequireLivePlugin only if config has the option attribute (not a fake config)
    and --require-live is True.
    """
    apply_live_auth_storage()

    # Cut Databricks credentials from the process environment in default runs (real configs only)
    if hasattr(config, "getoption") and not _get_require_live(config):
        _cut_databricks_credentials()

    config.addinivalue_line(
        "markers",
        "seam1: mark test as Seam-1 (live Lakebase integration test; deselected by default, include with --require-live)",
    )
    config.addinivalue_line(
        "markers",
        "seam2: mark test as Seam-2 (mock/no-cloud unit test)",
    )
    config.addinivalue_line(
        "markers",
        "provision: mark test as provision-type (must run serially with other provision tests)",
    )

    # Register the plugin only if require_live is True (and config has the option attribute)
    if _get_require_live(config):
        plugin = RequireLivePlugin(config)
        config.pluginmanager.register(plugin, "require_live")


def _is_live_test(item: pytest.Item) -> bool:
    """Check if a test is a live test (has seam1 marker or uses workspace_client fixture)."""
    # Check for seam1 marker (on test, class, or module)
    if item.get_closest_marker("seam1"):
        return True
    # Check if test uses workspace_client fixture
    return "workspace_client" in item.fixturenames


def pytest_collection_modifyitems(config, items):
    """Apply default marker, add autouse behaviors, and handle live test deselection.

    Without --require-live: deselect every live test (seam1 marker or workspace_client fixture).
    With --require-live: keep live tests in the session.
    """
    require_live = _get_require_live(config)

    # Identify live tests
    live_tests = []
    non_live_tests = []
    for item in items:
        if _is_live_test(item):
            live_tests.append(item)
        else:
            non_live_tests.append(item)

    # If --require-live is not set, deselect live tests
    if not require_live and live_tests:
        items[:] = non_live_tests
        config.hook.pytest_deselected(items=live_tests)
        # Print message about deselection
        print(f"live tests deselected: {len(live_tests)} tests (use --require-live to run them)")

    # Apply seam2 guard to seam2-marked tests
    for item in items:
        if item.get_closest_marker("seam2"):
            item.fixturenames.insert(0, "_seam2_guard_fixture")




def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print summary about skipped tests when --require-live is set.

    Retrieves the plugin (if registered) and prints skip summary.
    """
    plugin = config.pluginmanager.get_plugin("require_live")
    if plugin is None:
        return

    skipped = plugin.get_skipped()
    if skipped:
        terminalreporter.write_sep("=", "FAILURES")
        terminalreporter.write_line(
            f"--require-live: {len(skipped)} test(s) skipped; a live run must not skip",
            red=True,
        )
        for test_id in skipped:
            terminalreporter.write_line(f"  SKIPPED: {test_id}", red=True)


@pytest.fixture
def _seam2_guard_fixture():
    """Fixture to enable Seam-2 guard for tests marked with @pytest.mark.seam2."""
    with SeamGuard():
        yield


@pytest.fixture
def seam2_guard():
    """Fixture providing Seam-2 guard for explicit use in tests."""
    return SeamGuard()


@pytest.fixture(scope="session")
def lakebase_profile() -> str:
    """Lakebase connection profile name."""
    return LAKEBASE_PROFILE


@pytest.fixture(scope="session")
def lakebase_project() -> str:
    """Lakebase project ID."""
    return LAKEBASE_PROJECT


@pytest.fixture(scope="session")
def lakebase_catalog() -> str:
    """UC catalog for test artifacts."""
    return LAKEBASE_CATALOG


@pytest.fixture(scope="session")
def run_folder(tmp_path_factory) -> Path:
    """Create a run folder for manifest and artifacts."""
    folder = tmp_path_factory.mktemp("runs")
    return folder


@pytest.fixture(scope="session")
def workspace_client(request):
    """Create a Databricks WorkspaceClient for Lakebase integration.

    Requires DATABRICKS_HOST and DATABRICKS_TOKEN environment variables.
    Raises if Seam-2 is active (no cloud calls allowed).

    With --require-live: failures to build the client cause the test to fail.
    Without --require-live: failures cause the test to skip.
    """
    from lgk.harness import _seam2_guard_active

    if _seam2_guard_active():
        raise RuntimeError("Cannot create workspace client in Seam-2 test")

    require_live = _get_require_live(request.config)

    try:
        from databricks.sdk import WorkspaceClient

        client = WorkspaceClient(profile=LAKEBASE_PROFILE)
        return client
    except ImportError as e:
        if require_live:
            pytest.fail(f"databricks-sdk not installed (required for --require-live): {e}", pytrace=False)
        else:
            pytest.skip("databricks-sdk not installed (Seam-1 test)")
    except Exception as e:  # noqa: BLE001
        if require_live:
            pytest.fail(f"Failed to initialize WorkspaceClient (required for --require-live): {e}", pytrace=False)
        else:
            pytest.skip(f"Failed to initialize WorkspaceClient: {e}")


@pytest.fixture
def branch_leasor(workspace_client, lakebase_project, run_folder) -> BranchLeasor:
    """Provide a BranchLeasor for managing test branches.

    This fixture is only available for Seam-1 tests (marked with @pytest.mark.seam1).
    """
    return BranchLeasor(
        workspace_client=workspace_client,
        project_id=lakebase_project,
        test_base_branch=LAKEBASE_TEST_BASE_BRANCH,
        run_folder=run_folder,
        ttl_seconds=14400,  # 4 hours
    )


@pytest.fixture
def provision_lock():
    """Provide the global provision test lock for explicit locking."""
    return _provision_lock


def pytest_runtest_setup(item):
    """Hook to enforce serialization for provision tests."""
    if item.get_closest_marker("provision"):
        _provision_lock.acquire()


def pytest_runtest_teardown(item, nextitem):
    """Hook to release lock after provision tests."""
    if item.get_closest_marker("provision"):
        _provision_lock.release()


# ---------------------------------------------------------------------------- numbers recorder


@functools.lru_cache(maxsize=1)
def _tree_head() -> str:
    """HEAD of the tree under test (this conftest's repo), once per session and only when recording."""
    return git_head(Path(__file__).resolve().parent)


@functools.lru_cache(maxsize=None)
def _endpoint_cu(project: str, branch: str):
    """{"min", "max"} CU limits of the branch's primary endpoint: a read-only GET, cached per branch."""
    try:
        from databricks.sdk import WorkspaceClient

        status = WorkspaceClient(profile=LAKEBASE_PROFILE).postgres.get_endpoint(
            name=f"projects/{project}/branches/{branch}/endpoints/primary").status
        return {"min": status.autoscaling_limit_min_cu, "max": status.autoscaling_limit_max_cu}
    except Exception as e:  # noqa: BLE001 — context only; a failed read must not fail the live test
        logging.getLogger(__name__).warning(f"metrics: endpoint CU read failed for {branch}: {e}")
        return None


def metrics_recorder(nodeid: str, suite: str, environ=None) -> MetricsRecorder:
    """The recorder for one test: on only when LGK_METRICS_FILE is set in `environ` (default os.environ).

    Its `run` is LGK_METRICS_RUN (default `test`), validated here: a bad name fails the test at setup."""
    env = os.environ if environ is None else environ
    project = env.get("LAKEBASE_PROJECT") or None
    return MetricsRecorder(
        metrics_path(env), suite=suite, test=nodeid, commit=_tree_head, run=metrics_run(env), project=project,
        endpoint_cu=(lambda branch: _endpoint_cu(project, branch)) if project else None,
    )


@pytest.fixture
def record_metric(request) -> MetricsRecorder:
    """`record_metric(metric, value, unit, rows=, edges=, **context)` and `record_metric.latencies(...)`.

    rows/edges = the size of the graph the test provisioned (`graph["graph_size"]`); client defaults to laptop."""
    return metrics_recorder(request.node.nodeid, Path(str(request.node.fspath)).stem)
