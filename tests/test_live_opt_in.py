"""Live tests are opt-in: the repo conftest drives these inner pytest sessions.

Each test copies the repo's conftest.py into a pytester folder, writes small test files, and runs
pytest in a subprocess with no Databricks credentials reachable. The contract:

- A default run deselects every live test (a `seam1` mark on the test, class or module, or any
  test that uses the `workspace_client` fixture) and prints why, naming `--require-live`.
- `--require-live` selects them, and the session fails if any selected test skips.
- Under `--require-live`, missing credentials fail the test; they never skip it.
- A skip at collection (a module-level `pytest.skip` or `importorskip`) counts as a skip too.
- An xfail is not a skip: it is an expected failure the test itself declares.
- Credential cut: a default run removes every DATABRICKS_* variable and points DATABRICKS_CONFIG_FILE at a
  missing path, so a client an offline test builds has nothing to authenticate with. `--require-live` keeps them.
- Tripwire: on the real suite (`--collect-only`, no arguments), the set a default run deselects
  equals the tests in `tests/*_live.py`, plus the listed files and tests below. A live test outside
  that set fails it, and so does an offline test that got deselected.
- Static check: outside that set and conftest.py, no file under tests/ builds a WorkspaceClient,
  AccountClient or databricks.sql.connect, except the allow-listed offline fakes.
"""

import ast
import os
import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

ROOT = Path(__file__).resolve().parent.parent

LIVE_FILES = {
    "test_inner_live": """
import pytest

@pytest.mark.seam1
def test_marked():
    raise AssertionError("a seam1 test ran in a default session")

def test_uses_client_without_a_mark(workspace_client):
    raise AssertionError("a workspace_client test ran in a default session")

@pytest.mark.seam1
class TestMarkedClass:
    def test_in_class(self):
        raise AssertionError("a seam1 class ran in a default session")

def test_offline():
    assert 1 + 1 == 2
""",
    # hostile: a module-level mark, not a decorator.
    "test_inner_module_mark": """
import pytest

pytestmark = pytest.mark.seam1

def test_module_marked():
    raise AssertionError("a seam1 module ran in a default session")
""",
}


@pytest.fixture
def inner(pytester, monkeypatch):
    """A pytester folder with the repo conftest and no credentials reachable."""
    shutil.copy(ROOT / "conftest.py", pytester.path / "conftest.py")
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    for name in list(os.environ):
        if name.startswith(("DATABRICKS_", "LAKEBASE_", "LGK_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(pytester.path / "missing" / "databrickscfg"))
    return pytester


def _run(pytester, *args):
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", "-rA", *args)


def test_default_run_deselects_every_seam1_test_and_prints_why(inner):
    inner.makepyfile(**LIVE_FILES)
    result = _run(inner)
    result.assert_outcomes(passed=1, deselected=4)
    assert result.ret == 0
    out = result.stdout.str()
    assert "live tests deselected" in out and "--require-live" in out

    # hostile: the old way in (`-m seam1`) still runs no live test without --require-live.
    marked = _run(inner, "-m", "seam1")
    counts = marked.parseoutcomes()
    assert counts.get("failed", 0) == counts.get("passed", 0) == counts.get("errors", 0) == 0
    assert "--require-live" in marked.stdout.str()


def test_require_live_selects_every_live_test(inner):
    inner.makepyfile(test_inner_selected="""
import pytest

@pytest.mark.seam1
def test_marked():
    assert True

def test_offline():
    assert True
""")
    result = _run(inner, "--require-live")
    result.assert_outcomes(passed=2)
    assert result.ret == 0


def test_require_live_fails_when_any_selected_test_skips(inner):
    inner.makepyfile(test_inner_skips="""
import pytest

@pytest.mark.seam1
def test_live_skips_at_run_time():
    pytest.skip("no branch today")

def test_offline():
    assert True
""")
    result = _run(inner, "--require-live")
    assert result.parseoutcomes().get("passed", 0) == 1  # the session ran (not a usage error)
    assert result.ret != 0
    assert "--require-live" in result.stdout.str()

    # hostile: an offline skip marker counts too under --require-live ...
    inner.makepyfile(test_inner_skips="""
import pytest

@pytest.mark.skip(reason="offline skip")
def test_offline_skipped():
    assert True

def test_offline():
    assert True
""")
    offline = _run(inner, "--require-live")
    assert offline.parseoutcomes().get("passed", 0) == 1
    assert offline.ret != 0
    # ... but a default session with the same skip passes: only --require-live fails on a skip.
    assert _run(inner).ret == 0


def test_require_live_fails_when_a_module_skips_at_import(inner):
    # hostile: the skip happens at collection, so no test report carries it.
    inner.makepyfile(test_inner_modskip="""
import pytest

pytest.importorskip("lgk_no_such_module_for_this_test")

def test_never_collected():
    assert True
""", test_inner_ok="""
def test_offline():
    assert True
""")
    result = _run(inner, "--require-live")
    assert result.parseoutcomes().get("passed", 0) == 1
    assert result.ret != 0
    assert "--require-live" in result.stdout.str()
    assert _run(inner).ret == 0  # a default session with the same module skip passes


def test_require_live_does_not_count_an_xfail_as_a_skip(inner):
    inner.makepyfile(test_inner_xfail="""
import pytest

@pytest.mark.xfail(reason="known gap", strict=True)
def test_expected_to_fail():
    assert False

def test_offline():
    assert True
""")
    result = _run(inner, "--require-live")
    result.assert_outcomes(passed=1, xfailed=1)
    assert result.ret == 0


def test_require_live_with_no_credentials_fails_and_never_skips(inner):
    inner.makepyfile(test_inner_creds="""
import pytest

@pytest.mark.seam1
def test_marked_needs_client(workspace_client):
    assert workspace_client is not None

def test_unmarked_needs_client(workspace_client):
    assert workspace_client is not None
""")
    result = _run(inner, "--require-live")
    counts = result.parseoutcomes()
    assert result.ret != 0
    assert counts.get("skipped", 0) == 0, counts
    assert counts.get("failed", 0) + counts.get("errors", 0) + counts.get("error", 0) == 2, counts


# The probe checks what a client would read (env and config file), not a client build: building one
# reaches the network for the host, so it fails with or without credentials and proves nothing.
CREDENTIAL_PROBE = """
import os
from pathlib import Path


def test_the_session_has_no_databricks_credential():
    left = sorted(k for k in os.environ if k.startswith("DATABRICKS_") and k != "DATABRICKS_CONFIG_FILE")
    assert left == [], left
    assert not Path(os.environ["DATABRICKS_CONFIG_FILE"]).exists()


def test_a_child_process_inherits_the_cut():
    import subprocess, sys
    out = subprocess.run([sys.executable, "-c", "import os; print(sorted(k for k in os.environ if k.startswith('DATABRICKS_')))"],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == "['DATABRICKS_CONFIG_FILE']", out
"""


@pytest.fixture
def creds(inner, tmp_path, monkeypatch):
    """Credentials in env and file, in the shape a client reads. Synthetic and built at run time, so the
    shipped file holds no token or host literal for the public gate to find."""
    host = f"https://{secrets.token_hex(8)}.invalid"
    cfg = tmp_path / "databrickscfg"
    cfg.write_text(f"[DEFAULT]\nhost = {host}\ntoken = {secrets.token_hex(16)}\n")
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(cfg))
    monkeypatch.setenv("DATABRICKS_HOST", host)
    monkeypatch.setenv("DATABRICKS_TOKEN", secrets.token_hex(16))
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", secrets.token_hex(16))
    return cfg, host


def test_a_default_run_cuts_every_databricks_credential(inner, creds):
    # hostile: working credentials sit in env AND in the config file; an offline test and its child process see neither.
    cfg, _ = creds
    inner.makepyfile(test_inner_creds_cut=CREDENTIAL_PROBE)
    result = _run(inner)
    result.assert_outcomes(passed=2)
    assert cfg.exists()  # the cut points away from the file; it never deletes it


def test_require_live_keeps_the_credentials(inner, creds):
    cfg, host = creds
    inner.makepyfile(test_inner_creds_kept=f"""
import os

def test_kept():
    assert os.environ["DATABRICKS_HOST"] == {host!r}
    assert os.environ["DATABRICKS_CONFIG_FILE"] == {str(cfg)!r}
""")
    result = _run(inner, "--require-live")
    result.assert_outcomes(passed=1)
    assert result.ret == 0


def test_no_warehouse_env_var_means_no_live_run():
    """The live tests' warehouse comes from LGK_WAREHOUSE_ID only: no literal fallback, no ID in the file."""
    path = ROOT / "tests" / "test_build_provision_walks.py"
    text = path.read_text()
    values = [node.value for node in ast.parse(text).body
              if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "WAREHOUSE_ID"
                                                      for t in node.targets)]
    assert len(values) == 1
    assert ast.unparse(values[0]) == "os.environ.get('LGK_WAREHOUSE_ID')"
    assert re.search(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{16}(?![0-9A-Fa-f])", text) is None


def test_require_live_without_the_databricks_sdk_fails_and_never_skips(inner):
    # hostile: a local `databricks` package without `sdk` shadows the installed one, so the
    # fixture's import raises ImportError (a missing install), not an auth error.
    inner.mkpydir("databricks")
    inner.makepyfile(test_inner_nosdk="""
import pytest

@pytest.mark.seam1
def test_needs_client(workspace_client):
    assert workspace_client is not None
""")
    result = _run(inner, "--require-live")
    counts = result.parseoutcomes()
    assert result.ret != 0
    assert counts.get("skipped", 0) == 0, counts
    assert counts.get("failed", 0) + counts.get("errors", 0) + counts.get("error", 0) == 1, counts
    assert "databricks" in result.stdout.str()


# What a default run deselects, besides every tests/*_live.py (owner ruling A, 2026-10-13): two
# files that are live but not named *_live.py. A new live test goes in a *_live.py file (there is
# no single listed test; the live preflight test moved to tests/test_preflight_live.py).
LISTED_FILES = ("tests/test_build_provision_walks.py", "tests/test_golden_front_to_build.py")

RECORDER = """
import os

def pytest_deselected(items):
    with open(os.environ["TRIPWIRE_DESELECTED"], "a") as f:
        f.writelines(item.nodeid + "\\n" for item in items)

def pytest_collection_finish(session):
    with open(os.environ["TRIPWIRE_SELECTED"], "w") as f:
        f.writelines(item.nodeid + "\\n" for item in session.items)
"""


def _listed_file(path: str) -> bool:
    return (Path(path).parent == Path("tests") and path.endswith("_live.py")) or path in LISTED_FILES


def _listed(nodeid: str) -> bool:
    return _listed_file(nodeid.split("::")[0])


def _tripwire(root: Path, work: Path):
    """Collect the suite at `root` as a default run does; return (deselected, selected, stdout)."""
    (work / "lgk_tripwire_recorder.py").write_text(RECORDER)
    desel, sel = work / "deselected.txt", work / "selected.txt"
    desel.write_text("")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DATABRICKS_", "LAKEBASE_", "LGK_"))}
    env.update(DATABRICKS_CONFIG_FILE=str(work / "missing" / "databrickscfg"),
               PYTHONPATH=os.pathsep.join([str(work), str(ROOT / "src")]),
               TRIPWIRE_DESELECTED=str(desel), TRIPWIRE_SELECTED=str(sel))
    run = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                          "-p", "lgk_tripwire_recorder"], cwd=root, env=env, capture_output=True, text=True,
                         check=False)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    return set(desel.read_text().split()), set(sel.read_text().split()), run.stdout


def _mismatch(deselected: set, selected: set):
    """(deselected but not listed, listed but not deselected)."""
    listed = {i for i in deselected | selected if _listed(i)}
    return sorted(deselected - listed), sorted(listed - deselected)


def test_the_real_suite_deselects_exactly_the_listed_live_tests(tmp_path):
    deselected, selected, out = _tripwire(ROOT, tmp_path)
    assert deselected and selected  # the recorder saw the harness; empty == empty proves nothing
    assert f"live tests deselected: {len(deselected)} tests" in out
    unlisted, not_deselected = _mismatch(deselected, selected)
    assert unlisted == [], f"deselected by default but not in a *_live.py file or the list: {unlisted}"
    assert not_deselected == [], f"listed as live but runs by default: {not_deselected}"
    # every listed file exists, so a rename cannot empty the list unseen
    for entry in LISTED_FILES:
        assert any(i.startswith((entry + "::", entry + "[")) for i in deselected), entry
    # the live preflight test is in a *_live.py file, and no test of tests/test_preflight.py is live
    assert any(i.startswith("tests/test_preflight_live.py::") for i in deselected), sorted(deselected)
    assert not any(i.startswith("tests/test_preflight.py::") for i in deselected)


def _suite(inner, files: dict):
    (inner.path / "tests").mkdir(exist_ok=True)
    for name, text in files.items():
        (inner.path / "tests" / name).write_text(text)


SUITE = {
    "test_a_live.py": "import pytest\n\npytestmark = pytest.mark.seam1\n\ndef test_live():\n    pass\n",
    "test_build_provision_walks.py": "def test_walks(workspace_client):\n    pass\n",
    "test_offline.py": "def test_offline():\n    pass\n",
    "test_other.py": "def test_other():\n    pass\n",  # stays selected, so collection never comes up empty
}


def test_the_tripwire_fails_on_a_live_test_outside_the_listed_files(inner, tmp_path):
    _suite(inner, SUITE)
    assert _mismatch(*_tripwire(inner.path, tmp_path)[:2]) == ([], [])  # the clean suite passes
    # hostile: a live test in a file that is neither *_live.py nor listed.
    _suite(inner, {"test_offline.py": SUITE["test_offline.py"]
                   + "\nimport pytest\n\n" + "@pytest.mark.seam1\ndef test_live_elsewhere():\n    pass\n"})
    assert _mismatch(*_tripwire(inner.path, tmp_path)[:2]) == (["tests/test_offline.py::test_live_elsewhere"], [])


def test_the_tripwire_fails_on_an_offline_test_that_got_deselected(inner, tmp_path):
    # hostile: a broken harness also deselects an offline test (a second conftest stands in for it).
    _suite(inner, {**SUITE, "conftest.py": """
def pytest_collection_modifyitems(config, items):
    victims = [i for i in items if i.name == "test_offline"]
    items[:] = [i for i in items if i not in victims]
    config.hook.pytest_deselected(items=victims)
"""})
    assert _mismatch(*_tripwire(inner.path, tmp_path)[:2]) == (["tests/test_offline.py::test_offline"], [])
    # hostile: and the other way, a test in a *_live.py file that a default run still runs.
    _suite(inner, {"conftest.py": "", "test_b_live.py": "def test_offline_in_a_live_file():\n    pass\n"})
    assert _mismatch(*_tripwire(inner.path, tmp_path)[:2]) == ([], ["tests/test_b_live.py::test_offline_in_a_live_file"])


# A live client built outside the live set: the static half of the tripwire. The tripwire sees a
# test only through its mark and fixtures; a test that builds its own client has neither.
CLIENT_TARGETS = {"databricks.sdk.WorkspaceClient", "databricks.sdk.AccountClient", "databricks.sql.connect",
                  "databricks.sql.client.Connection"}
CLIENT_NAMES = ("WorkspaceClient", "AccountClient")  # any constructor of these names, whatever module
# Offline constructions on purpose: (file, the name the call resolves to) -> why it is offline.
CLIENT_ALLOW = {
    ("tests/test_runtime_pin.py", "<local>.WorkspaceClient"):
        "a fake class defined in the file with the SDK's name (not JSON-serializable, like the real one); "
        "it proves live clients never reach a refresh job spec",
}


def _is_client(q: str) -> bool:
    return q in CLIENT_TARGETS or q.rsplit(".", 1)[-1] in CLIENT_NAMES


def _resolve(bound: dict, d: list) -> set:
    return {".".join([b, *d[1:]]) for b in bound.get(d[0], {d[0]})}


def _bound_names(tree: ast.Module) -> dict:
    """Each local name -> the dotted names it can stand for: imports, the module's own defs, plain
    aliases (`W = WorkspaceClient`), and subclasses of a client (`class C(WorkspaceClient)`)."""
    bound: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                name, target = (a.asname, a.name) if a.asname else ((root := a.name.split(".")[0]), root)
                bound.setdefault(name, set()).add(target)
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                bound.setdefault(a.asname or a.name, set()).add(f"{node.module}.{a.name}")
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.setdefault(node.name, set()).add(f"<local>.{node.name}")
    for _ in range(3):  # a few rounds, for chains (A = B; B = WorkspaceClient)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and (d := _dotted(node.value))):
                bound.setdefault(node.targets[0].id, set()).update(q for q in _resolve(bound, d) if _is_client(q))
            elif isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if d := _dotted(base):
                        bound.setdefault(node.name, set()).update(q for q in _resolve(bound, d) if _is_client(q))
    return bound


def _dotted(func: ast.expr):
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    return [func.id, *reversed(parts)] if isinstance(func, ast.Name) else None


def _client_constructions(root: Path):
    """(calls that build a client outside the live set as 'path:line name', allow-list entries used)."""
    hits, used = [], set()
    for path in sorted((root / "tests").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if path.name == "conftest.py" or _listed_file(rel):
            continue
        tree = ast.parse(path.read_text(), filename=rel)
        bound = _bound_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # a client reached by name at run time: getattr(x, "WorkspaceClient")
            if (isinstance(func, ast.Name) and func.id == "getattr" and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant) and node.args[1].value in CLIENT_NAMES):
                hits.append(f"{rel}:{node.lineno} getattr(..., {node.args[1].value!r})")
            if not (d := _dotted(func)):
                # a client on an object that is not a plain name: import_module(...).WorkspaceClient()
                if isinstance(func, ast.Attribute) and func.attr in CLIENT_NAMES:
                    hits.append(f"{rel}:{node.lineno} <expr>.{func.attr}")
                continue
            for q in sorted(_resolve(bound, d)):
                if _is_client(q):
                    if (rel, q) in CLIENT_ALLOW:
                        used.add((rel, q))
                    else:
                        hits.append(f"{rel}:{node.lineno} {q}")
    return hits, used


def test_no_offline_test_file_builds_a_databricks_client():
    hits, used = _client_constructions(ROOT)
    assert hits == [], f"builds a client outside tests/*_live.py and the listed tests: {hits}"
    assert used == set(CLIENT_ALLOW), f"allow-list entries that match nothing: {set(CLIENT_ALLOW) - used}"


def test_the_client_check_catches_an_unmarked_test_that_builds_its_own_client(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    sneaky = "from databricks.sdk import WorkspaceClient\n\ndef test_x():\n    WorkspaceClient()\n"
    # hostile: an unmarked test with its own client; no mark and no fixture, so the tripwire misses it.
    (tests / "test_sneaky.py").write_text(sneaky)
    # hostile: every alias form of the three constructors.
    (tests / "test_alias.py").write_text(
        "import databricks.sdk as sdk\nfrom databricks import sql\nfrom databricks.sql import connect as c\n"
        "from databricks.sdk import AccountClient as A\n\ndef test_y():\n    sdk.WorkspaceClient(); sql.connect(); c(); A()\n")
    # the same code in a live file is fine. hostile: a seam1-marked test in a file that is not
    # *_live.py is not exempt: no single test is listed any more, so its client counts.
    (tests / "test_ok_live.py").write_text(sneaky)
    (tests / "test_preflight.py").write_text(
        "import pytest\nfrom databricks.sdk import WorkspaceClient\n\n"
        "@pytest.mark.seam1\ndef test_marked_live():\n    WorkspaceClient()\n")
    # hostile: the allow-listed file also builds the real SDK class under another name.
    (tests / "test_runtime_pin.py").write_text(
        "from databricks.sdk import WorkspaceClient as Real\n\nclass WorkspaceClient:\n    pass\n\n"
        "def test_z():\n    WorkspaceClient(); Real()\n")
    hits, used = _client_constructions(tmp_path)
    assert sorted(hits) == [
        "tests/test_alias.py:7 databricks.sdk.AccountClient",
        "tests/test_alias.py:7 databricks.sdk.WorkspaceClient",
        "tests/test_alias.py:7 databricks.sql.connect",
        "tests/test_alias.py:7 databricks.sql.connect",
        "tests/test_preflight.py:6 databricks.sdk.WorkspaceClient",
        "tests/test_runtime_pin.py:7 databricks.sdk.WorkspaceClient",
        "tests/test_sneaky.py:4 databricks.sdk.WorkspaceClient",
    ]
    assert used == {("tests/test_runtime_pin.py", "<local>.WorkspaceClient")}


# hostile: the shapes prover round 3 got past the first version of the check (2026-10-13).
EVASIONS = {
    "getattr": ("import databricks.sdk\n\ndef test_a():\n    getattr(databricks.sdk, 'WorkspaceClient')()\n",
                "tests/test_getattr.py:4 getattr(..., 'WorkspaceClient')"),
    "dunder_import": ("def test_a():\n    __import__('databricks.sdk').sdk.WorkspaceClient()\n",
                      "tests/test_dunder_import.py:2 <expr>.WorkspaceClient"),
    "import_module": ("import importlib\n\ndef test_a():\n    importlib.import_module('databricks.sdk').AccountClient()\n",
                      "tests/test_import_module.py:4 <expr>.AccountClient"),
    "assign_alias": ("from databricks.sdk import WorkspaceClient\n\nW = WorkspaceClient\nV = W\n\ndef test_a():\n    V()\n",
                     "tests/test_assign_alias.py:7 databricks.sdk.WorkspaceClient"),
    "subclass": ("from databricks.sdk import WorkspaceClient\n\nclass C(WorkspaceClient):\n    pass\n\ndef test_a():\n    C()\n",
                 "tests/test_subclass.py:7 databricks.sdk.WorkspaceClient"),
    "sql_client": ("from databricks.sql import client\n\ndef test_a():\n    client.Connection('h', 'p')\n",
                   "tests/test_sql_client.py:4 databricks.sql.client.Connection"),
}


@pytest.mark.parametrize("shape", sorted(EVASIONS))
def test_the_client_check_catches_indirect_constructions(tmp_path, shape):
    src, expected = EVASIONS[shape]
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / f"test_{shape}.py").write_text(src)
    hits, _ = _client_constructions(tmp_path)
    assert expected in hits, hits
