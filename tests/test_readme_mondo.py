"""Red tests for the README's MONDO step: load_mondo exists, parameters match signature, and placement is correct.

Each check reads the README using AST to parse Python blocks, reading the function signature from the live module.
Each check is a function, so hostile rows can feed it broken README text and prove it fails.
"""
import ast
import importlib
import inspect
import re
from pathlib import Path

import pytest

from lgk.loaders import mondo

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

QUICK_START_SECTION = re.compile(r"^## Quick start\s*$(.*?)(?=^## |\Z)", re.M | re.S)
PY_BLOCK = re.compile(r"^```python\n(.*?)^```", re.M | re.S)


def _read(path: Path) -> str:
    assert path.is_file(), f"{path.relative_to(ROOT)} does not exist"
    return path.read_text()


def _quick_start_blocks(text: str) -> list:
    """All ```python blocks in the Quick start section."""
    m = QUICK_START_SECTION.search(text)
    if not m:
        return []
    section = m.group(1)
    return PY_BLOCK.findall(section)


def _find_load_mondo_call(text: str) -> tuple:
    """Find load_mondo call in Quick start: (block, lineno, ast.Call node).

    Returns (None, -1, None) if not found.
    """
    for block in _quick_start_blocks(text):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                is_load_mondo = False
                if isinstance(node.func, ast.Name) and node.func.id == "load_mondo":
                    is_load_mondo = True
                elif isinstance(node.func, ast.Attribute) and node.func.attr == "load_mondo":
                    is_load_mondo = True

                if is_load_mondo:
                    return (block, node.lineno, node)

    return (None, -1, None)


def _find_mondo_step(text: str) -> str:
    """Backward compat: find load_mondo call and return as unparsed string."""
    block, lineno, call_node = _find_load_mondo_call(text)
    return ast.unparse(call_node) if call_node else ""


def _get_config_value(block: str, key: str) -> str:
    """Extract string value of config[key] from the block."""
    m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', block)
    return m.group(1) if m else ""


def _get_config_schema_node(block: str):
    """Get the AST node for config["source_schema"] from config dict."""
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key_node, val_node in zip(node.keys, node.values):
                if isinstance(key_node, ast.Constant) and key_node.value == "source_schema":
                    return val_node
    return None


# ================================================================== checks

def check_mondo_step_exists(text: str) -> None:
    """Quick start contains a load_mondo call."""
    block, lineno, call_node = _find_load_mondo_call(text)
    assert call_node is not None, "no load_mondo call in Quick start"


def check_load_mondo_function_exists(text: str) -> None:
    """Call names load_mondo from lgk.loaders.mondo."""
    block, lineno, call_node = _find_load_mondo_call(text)
    assert call_node is not None, "no load_mondo call"
    assert hasattr(mondo, "load_mondo"), "load_mondo not in module"


def check_call_parameters_match_signature(text: str) -> None:
    """Call parameters match load_mondo signature (no extra, missing, unknown)."""
    block, lineno, call_node = _find_load_mondo_call(text)
    assert call_node is not None, "no load_mondo call"

    positional = [ast.unparse(arg) for arg in call_node.args]
    keywords = {kw.arg: ast.unparse(kw.value) for kw in call_node.keywords}

    sig = inspect.signature(mondo.load_mondo)
    try:
        bound = sig.bind(*positional, **keywords)
    except TypeError as e:
        raise AssertionError(f"parameters don't match signature: {e}") from e


def check_mondo_before_runner_steps(text: str) -> None:
    """load_mondo call comes before runner steps in Quick start."""
    block, call_lineno, call_node = _find_load_mondo_call(text)
    assert call_node is not None, "load_mondo call not in Quick start"
    assert call_lineno > 0, "load_mondo line number invalid"

    try:
        tree = ast.parse(block)
    except SyntaxError:
        raise AssertionError("Quick start does not parse")

    runner_lineno = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "steps":
                    runner_lineno = node.lineno
                    break
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "run_steps":
                runner_lineno = node.lineno
                break

    # Block-local: runner steps in a later block than load_mondo fail here, not as an ordering error (still red)
    assert runner_lineno is not None, "runner steps not in Quick start"
    assert call_lineno < runner_lineno, "load_mondo after runner steps"


def check_schema_matches_config_source_schema(text: str) -> None:
    """Schema arg exactly references config["source_schema"] or matches its value."""
    block, lineno, call_node = _find_load_mondo_call(text)
    assert call_node is not None, "no load_mondo call"

    # Verify config defines source_schema
    config_schema_node = _get_config_schema_node(block)
    assert config_schema_node is not None, "config missing source_schema"

    # Find the schema argument by binding the call to the live signature, never by a typed position
    sig = inspect.signature(mondo.load_mondo)
    try:
        bound = sig.bind(*call_node.args, **{kw.arg: kw.value for kw in call_node.keywords})
    except TypeError as e:
        raise AssertionError(f"parameters don't match signature: {e}") from e
    schema_node = bound.arguments.get("schema")

    assert schema_node is not None, "schema param not in call"

    # Check: schema_node must reference config["source_schema"] directly
    # or be a string constant matching the config value

    # Case 1: Direct reference to config["source_schema"] (Subscript node)
    if isinstance(schema_node, ast.Subscript):
        if isinstance(schema_node.value, ast.Name) and schema_node.value.id == "config":
            if isinstance(schema_node.slice, ast.Constant) and schema_node.slice.value == "source_schema":
                return  # This is config["source_schema"]

    # Case 2: String constant equal to config's source_schema value
    if isinstance(schema_node, ast.Constant) and isinstance(schema_node.value, str):
        config_value = _get_config_value(block, "source_schema")
        if schema_node.value == config_value:
            return

    # Fail: schema doesn't match
    schema_unparsed = ast.unparse(schema_node)
    config_unparsed = ast.unparse(config_schema_node)
    raise AssertionError(f"schema {schema_unparsed!r} != config source_schema {config_unparsed!r}")


# ================================================================== real README

def test_readme_quick_start_has_load_mondo_step():
    """Quick start has load_mondo call."""
    check_mondo_step_exists(_read(README))


def test_readme_load_mondo_uses_real_function():
    """Call names load_mondo from module."""
    check_load_mondo_function_exists(_read(README))


def test_readme_load_mondo_parameters_match_signature():
    """Call parameters match signature."""
    check_call_parameters_match_signature(_read(README))


def test_readme_load_mondo_comes_before_runner_steps():
    """load_mondo before runner steps."""
    check_mondo_before_runner_steps(_read(README))


def test_readme_load_mondo_schema_matches_config_source_schema():
    """Schema references config source_schema."""
    check_schema_matches_config_source_schema(_read(README))


# ================================================================== GOOD_QUICKSTART

GOOD_QUICKSTART = (
    """## Quick start

```python
from databricks.sdk import WorkspaceClient
from lgk.runner import LaptopRunner
from lgk.notebook_runner import resolve_step
from lgk.loaders.mondo import load_mondo

config = {
    "run_folder": "./runs/my_graph",
    "lakebase_project": "<project>",
    "branch_id": "<branch>",
    "source_catalog": "<source_catalog>",
    "source_schema": "<source_schema>",
    "lakebase_catalog": "<lakebase_catalog>",
    "kit_schema": "lgk_my_graph",
    "served_schema": "lgk_my_graph",
    "sync_owner_user": "<operator_email>",
    "warehouse_id": "<warehouse>",
    "workspace_client": WorkspaceClient(),
}

load_mondo(config["workspace_client"], config["warehouse_id"],
           config["source_catalog"], config["source_schema"],
           config["run_folder"])

runner = LaptopRunner(config)

# Run the first few steps
steps = [
    ("preflight", resolve_step("lgk.steps.preflight:preflight")),
]

results = runner.run_steps(steps)
```

## Next
"""
)

# ================================================================== hostile tests

def test_hostile_missing_load_mondo_step_fails():
    """No load_mondo call fails."""
    bad = GOOD_QUICKSTART.replace(
        'load_mondo(config["workspace_client"], config["warehouse_id"],\n'
        '           config["source_catalog"], config["source_schema"],\n'
        '           config["run_folder"])\n\n', "")
    check_mondo_step_exists(GOOD_QUICKSTART)
    with pytest.raises(AssertionError):
        check_mondo_step_exists(bad)


def test_hostile_wrong_function_name_fails():
    """Different function name fails."""
    bad = GOOD_QUICKSTART.replace("load_mondo(", "load_mondo_data(")
    check_load_mondo_function_exists(GOOD_QUICKSTART)
    with pytest.raises(AssertionError):
        check_load_mondo_function_exists(bad)


def test_hostile_too_many_positional_args_fails():
    """Extra positional args fail."""
    bad = GOOD_QUICKSTART.replace(
        'load_mondo(config["workspace_client"], config["warehouse_id"],\n'
        '           config["source_catalog"], config["source_schema"],\n'
        '           config["run_folder"])',
        'load_mondo(config["workspace_client"], config["warehouse_id"],\n'
        '           config["source_catalog"], config["source_schema"],\n'
        '           config["run_folder"], None, "extra")')
    check_call_parameters_match_signature(GOOD_QUICKSTART)
    with pytest.raises(AssertionError):
        check_call_parameters_match_signature(bad)


def test_hostile_missing_required_arg_fails():
    """Missing required arg fails."""
    bad = GOOD_QUICKSTART.replace(
        'load_mondo(config["workspace_client"], config["warehouse_id"],\n'
        '           config["source_catalog"], config["source_schema"],\n'
        '           config["run_folder"])',
        'load_mondo(config["workspace_client"], config["warehouse_id"],\n'
        '           config["source_catalog"])')
    check_call_parameters_match_signature(GOOD_QUICKSTART)
    with pytest.raises(AssertionError):
        check_call_parameters_match_signature(bad)


def test_hostile_unknown_keyword_parameter_fails():
    """Unknown keyword param fails."""
    bad = GOOD_QUICKSTART.replace(
        'load_mondo(config["workspace_client"], config["warehouse_id"],\n'
        '           config["source_catalog"], config["source_schema"],\n'
        '           config["run_folder"])',
        'load_mondo(config["workspace_client"], config["warehouse_id"],\n'
        '           config["source_catalog"], config["source_schema"],\n'
        '           config["run_folder"], batch_size=100)')
    check_call_parameters_match_signature(GOOD_QUICKSTART)
    with pytest.raises(AssertionError):
        check_call_parameters_match_signature(bad)


def test_hostile_schema_with_suffix_fails():
    """Schema with expression suffix fails."""
    bad = GOOD_QUICKSTART.replace(
        'config["source_schema"],',
        'config["source_schema"] + "_v2",')
    check_schema_matches_config_source_schema(GOOD_QUICKSTART)
    with pytest.raises(AssertionError):
        check_schema_matches_config_source_schema(bad)


def test_hostile_schema_wrong_key_fails():
    """Schema with wrong config key fails."""
    bad = GOOD_QUICKSTART.replace(
        'config["source_schema"],',
        'config["source_schema_old"],')
    check_schema_matches_config_source_schema(GOOD_QUICKSTART)
    with pytest.raises(AssertionError):
        check_schema_matches_config_source_schema(bad)


def test_hostile_mondo_after_runner_steps_fails():
    """load_mondo after runner steps fails."""
    bad = GOOD_QUICKSTART.replace(
        'load_mondo(config["workspace_client"], config["warehouse_id"],\n'
        '           config["source_catalog"], config["source_schema"],\n'
        '           config["run_folder"])\n\n', "")
    bad = bad.replace(
        "results = runner.run_steps(steps)",
        'results = runner.run_steps(steps)\n\nload_mondo(config["workspace_client"], config["warehouse_id"],\n'
        '           config["source_catalog"], config["source_schema"],\n'
        '           config["run_folder"])')
    check_mondo_before_runner_steps(GOOD_QUICKSTART)
    with pytest.raises(AssertionError):
        check_mondo_before_runner_steps(bad)


def test_hostile_step_outside_quick_start_fails():
    """load_mondo outside Quick start fails."""
    bad = GOOD_QUICKSTART.replace(
        'load_mondo(config["workspace_client"], config["warehouse_id"],\n'
        '           config["source_catalog"], config["source_schema"],\n'
        '           config["run_folder"])\n', "")
    bad = bad + "\n## Later\n\n```python\n" + \
        'load_mondo(config["workspace_client"], config["warehouse_id"],\n' + \
        '           config["source_catalog"], config["source_schema"],\n' + \
        '           config["run_folder"])\n```\n'
    check_mondo_step_exists(GOOD_QUICKSTART)
    with pytest.raises(AssertionError):
        check_mondo_step_exists(bad)


def test_hostile_schema_position_is_read_from_the_live_signature(monkeypatch):
    """With `schema` moved last in the signature, the 4th argument is no longer the schema, so the check fails."""
    def moved(w, warehouse_id, catalog, run_folder, schema, doc=None):
        raise NotImplementedError("signature stand-in only")

    check_schema_matches_config_source_schema(GOOD_QUICKSTART)  # good passes with the real signature
    monkeypatch.setattr(mondo, "load_mondo", moved)
    with pytest.raises(AssertionError):
        check_schema_matches_config_source_schema(GOOD_QUICKSTART)


def test_good_schema_as_keyword_passes():
    """Schema as keyword param passes."""
    good = GOOD_QUICKSTART.replace(
        'load_mondo(config["workspace_client"], config["warehouse_id"],\n'
        '           config["source_catalog"], config["source_schema"],\n'
        '           config["run_folder"])',
        'load_mondo(config["workspace_client"], config["warehouse_id"],\n'
        '           config["source_catalog"], run_folder=config["run_folder"],\n'
        '           schema=config["source_schema"])')
    check_call_parameters_match_signature(good)
    check_schema_matches_config_source_schema(good)


def test_good_assignment_form_passes():
    """Assignment form passes."""
    good = GOOD_QUICKSTART.replace(
        'load_mondo(config["workspace_client"], config["warehouse_id"],\n'
        '           config["source_catalog"], config["source_schema"],\n'
        '           config["run_folder"])',
        'release = load_mondo(config["workspace_client"], config["warehouse_id"],\n'
        '                     config["source_catalog"], config["source_schema"],\n'
        '                     config["run_folder"])')
    check_load_mondo_function_exists(good)


# ================================================================== import resolution
# The Quick start is copied and pasted, so a call whose name is never imported fails for the first reader (NameError).

MONDO_MODULE = "lgk.loaders.mondo"


def _import_bindings(tree: ast.AST) -> dict:
    """Map each name bound by an import in the block to (lineno, module, attr).

    from lgk.loaders.mondo import load_mondo as lm -> {"lm": (n, "lgk.loaders.mondo", "load_mondo")}
    from lgk.loaders import mondo               -> {"mondo": (n, "lgk.loaders.mondo", None)}
    import lgk.loaders.mondo as m               -> {"m": (n, "lgk.loaders.mondo", None)}
    """
    bound = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                name = alias.asname or alias.name
                full = f"{node.module}.{alias.name}"
                # "from pkg import submodule" binds a module; otherwise an attribute
                if full == MONDO_MODULE:
                    bound[name] = (node.lineno, MONDO_MODULE, None)
                else:
                    bound[name] = (node.lineno, node.module, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    bound[alias.asname] = (node.lineno, alias.name, None)
    return bound


def check_load_mondo_is_imported(text: str) -> None:
    """The Quick start block that calls load_mondo imports it from lgk.loaders.mondo,
    above the call, and the import resolves against the live module."""
    block, call_lineno, call_node = _find_load_mondo_call(text)
    assert call_node is not None, "no load_mondo call in Quick start"

    tree = ast.parse(block)
    bindings = _import_bindings(tree)

    func = call_node.func
    if isinstance(func, ast.Name):
        name = func.id
        assert name in bindings, f"{name!r} is called but never imported in the Quick start block"
        lineno, module, attr = bindings[name]
        assert (module, attr) == (MONDO_MODULE, "load_mondo"), (
            f"{name!r} is imported from {module}.{attr}, not {MONDO_MODULE}.load_mondo")
    elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        name = func.value.id
        assert name in bindings, f"{name!r} is used but never imported in the Quick start block"
        lineno, module, attr = bindings[name]
        assert (module, attr) == (MONDO_MODULE, None), (
            f"{name!r} is bound to {module}{'.' + attr if attr else ''}, not the module {MONDO_MODULE}")
    else:
        raise AssertionError(f"load_mondo is called through an unsupported form: {ast.unparse(func)}")

    # Both line numbers are relative to the same parsed block
    assert lineno < call_lineno, "the load_mondo import comes after the call"

    # Bind to the live code, not just the spelling: the module path must import
    # and expose load_mondo, the same object the signature checks read.
    live = importlib.import_module(MONDO_MODULE)
    assert getattr(live, "load_mondo", None) is mondo.load_mondo, (
        f"{MONDO_MODULE}.load_mondo does not resolve to the live function")


def test_readme_load_mondo_is_imported():
    """The real Quick start imports load_mondo before calling it."""
    check_load_mondo_is_imported(_read(README))


_IMPORT_LINE = "from lgk.loaders.mondo import load_mondo\n"


def test_hostile_missing_import_fails():
    """Call kept, import removed: the pasted block would NameError."""
    assert _IMPORT_LINE in GOOD_QUICKSTART
    bad = GOOD_QUICKSTART.replace(_IMPORT_LINE, "")
    check_load_mondo_is_imported(GOOD_QUICKSTART)
    with pytest.raises(AssertionError):
        check_load_mondo_is_imported(bad)


def test_hostile_import_from_wrong_module_fails():
    """Same name, wrong module."""
    bad = GOOD_QUICKSTART.replace(_IMPORT_LINE, "from lgk.loaders.hpo import load_mondo\n")
    check_load_mondo_is_imported(GOOD_QUICKSTART)
    with pytest.raises(AssertionError):
        check_load_mondo_is_imported(bad)


def test_hostile_import_after_call_fails():
    """Import moved below the call: the call runs first and NameErrors."""
    bad = GOOD_QUICKSTART.replace(_IMPORT_LINE, "")
    bad = bad.replace("runner = LaptopRunner(config)",
                      _IMPORT_LINE + "runner = LaptopRunner(config)")
    check_load_mondo_is_imported(GOOD_QUICKSTART)
    with pytest.raises(AssertionError):
        check_load_mondo_is_imported(bad)


def test_hostile_import_aliased_away_fails():
    """Imported under another name, called as load_mondo: still unbound."""
    bad = GOOD_QUICKSTART.replace(_IMPORT_LINE, "from lgk.loaders.mondo import load_mondo as lm\n")
    check_load_mondo_is_imported(GOOD_QUICKSTART)
    with pytest.raises(AssertionError):
        check_load_mondo_is_imported(bad)


def test_good_module_import_form_passes():
    """`from lgk.loaders import mondo` + `mondo.load_mondo(...)` is a valid form."""
    good = GOOD_QUICKSTART.replace(_IMPORT_LINE, "from lgk.loaders import mondo\n")
    good = good.replace("load_mondo(config[", "mondo.load_mondo(config[", 1)
    check_load_mondo_is_imported(good)
