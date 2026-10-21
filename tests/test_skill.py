"""Acceptance tests for the committed SKILL.md interviewer.

The skill folder `skill/lakebase-graph-kit/` is the single source of truth. These tests read the
committed file and check it against the toolkit itself:

1. Frontmatter is exactly the common subset {name, description} (Agent Skills open standard).
2. Every step / interface name the skill uses resolves in the toolkit, and steps run only through
   a runner (`run_steps`), never by calling a step function directly.
3. `install_skill_to_both_hosts` copies the committed bytes to both discovery paths.

Each checker is also run on hostile variants of the real file, so a checker that cannot fail
fails here.
"""

import ast
import importlib
import re
from pathlib import Path

import pytest
import yaml

from lgk.interview import SKILL_FILE, install_skill_to_both_hosts
from lgk.notebook_runner import resolve_step
from lgk.runner import RunFolder

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED = REPO_ROOT / "skill" / "lakebase-graph-kit" / "SKILL.md"

# The documented interface a skill may import from. Anything else under lgk.* is internal.
INTERFACE = {
    "lgk.runner": {"LaptopRunner"},
    "lgk.notebook_runner": {"NotebookRunner", "DatabricksNotebookBackend", "resolve_step", "upload_notebook"},
}
# Public helpers that are not runner steps (no config-in signature) and may be called directly.
QUERY_HELPERS = {
    "lgk.steps.walks:get_walk_query",
    "lgk.steps.walks:get_lookup_query",  # the closure lookups
    "lgk.refresh.job:create_refresh_job",
    "lgk.steps.privacy:resolve_flag",
    "lgk.walks.sql:NodePredicate",  # the set walk's seed / stop predicate
    "lgk.export:export",  # the service-principal bundle
    "lgk.steps.retire:retire",  # each needs the typed graph name, and may run more than
    "lgk.steps.retire:retire_versions",  # once, so none is a run.json-tracked runner step
    "lgk.reaper:find_orphan_jobs",
    "lgk.reaper:find_orphans",
    "lgk.reaper:reap",
    "lgk.steps.switch:rollback",  # rollback: a switch back, called directly (the runner skips
    "lgk.steps.switch:served_history",  # a succeeded switch); the history read that picks the version
    "lgk.keys:open_orphans",  # the operator's orphan decisions, recorded in key_decisions.json
    "lgk.keys:resolve_orphans",
}
# Runner steps that run through run_steps but are not pre-listed in RunFolder.STEPS (run.json gets
# their entry on the first run): measure runs after switch.
UNLISTED_RUNNER_STEPS = {"measure"}
# Runner steps that also need the operator's typed graph name: resolved into a name, then run
# through run_steps as ("<step>", lambda cfg: <name>(cfg, ...)).
CONFIRMED_STEPS = {"lgk.steps.teardown:teardown"}
# The spec's step order, restricted to what is merged on this branch.
SPEC_ORDER = ["preflight", "profile", "propose", "privacy", "build", "enrich", "provision", "advise",
              "switch", "walks", "app", "notebook", "teardown"]

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)
REF_RE = re.compile(r"\blgk(?:\.\w+)+:\w+")
HEADING_RE = re.compile(r"^## Step \d+: `(\w+)`", re.M)
PY_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.S)
HOST_WORDS = re.compile(r"claude|genie|\.assistant", re.I)


# ------------------------------------------------------------------------------- checkers
def _steps_without_module(steps) -> list:
    """The step names that have no lgk.steps module (none may, at release)."""
    import importlib.util

    return [s for s in steps if importlib.util.find_spec(f"lgk.steps.{s}") is None]


def split_skill(text: str) -> tuple[dict, str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        raise AssertionError("SKILL.md has no leading --- frontmatter --- block")
    return yaml.safe_load(m.group(1)), text[m.end():]


def check_frontmatter(text: str, folder_name: str) -> None:
    meta, _ = split_skill(text)
    if set(meta) != {"name", "description"}:
        raise AssertionError(f"frontmatter keys must be exactly name+description, got {set(meta)}")
    if meta["name"] != folder_name:
        raise AssertionError(f"name {meta['name']!r} != folder {folder_name!r}")
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", meta["name"]) or len(meta["name"]) > 64:
        raise AssertionError(f"name {meta['name']!r} breaks the Agent Skills name rule")
    if not isinstance(meta["description"], str) or not 0 < len(meta["description"]) <= 1024:
        raise AssertionError("description must be a 1..1024 char string")


def check_host_agnostic(text: str) -> None:
    _, body = split_skill(text)
    hits = HOST_WORDS.findall(body)
    if hits:
        raise AssertionError(f"body names a host: {hits}")


def _check_ref(ref: str) -> None:
    module, name = ref.split(":", 1)
    if any(part.startswith("_") for part in module.split(".") + [name]):
        raise AssertionError(f"{ref} is an internal (underscore) name")
    resolve_step(ref)  # raises ImportError / AttributeError if it does not exist
    if module.startswith("lgk.steps.") and ref not in QUERY_HELPERS:
        step = module.rsplit(".", 1)[1]
        if name != step or step not in RunFolder.STEPS + sorted(UNLISTED_RUNNER_STEPS):
            raise AssertionError(f"{ref} is not a runner step (want lgk.steps.<s>:<s>)")


def _check_imports(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.startswith("lgk") for a in node.names):
            raise AssertionError("plain `import lgk...` bypasses the interface")
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("lgk"):
            allowed = INTERFACE.get(node.module)
            if allowed is None:
                raise AssertionError(f"import from internal module {node.module}")
            mod = importlib.import_module(node.module)
            for alias in node.names:
                if alias.name not in allowed or not hasattr(mod, alias.name):
                    raise AssertionError(f"{node.module}.{alias.name} is not in the interface")


def _parents(tree: ast.AST) -> dict:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _is_runner_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and (
        node.func.attr == "run_steps"
    )


def _check_resolve_calls(tree: ast.AST) -> list[str]:
    """Every resolve_step(<step ref>) must be the 2nd item of a ("<step>", ...) tuple in run_steps."""
    parents, run_names = _parents(tree), []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "resolve_step"):
            continue
        arg = node.args[0] if node.args else None
        if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
            raise AssertionError("resolve_step needs a literal module:qualname ref")
        _check_ref(arg.value)
        if arg.value in QUERY_HELPERS:
            continue
        step = arg.value.split(":")[1]
        if arg.value in CONFIRMED_STEPS:
            run_names.append(_check_confirmed_step(tree, parents, node, step))
            continue
        tup, lst = parents.get(node), None
        if isinstance(tup, ast.Tuple) and tup.elts[1:2] == [node]:
            lst = parents.get(tup)
        first = tup.elts[0] if isinstance(tup, ast.Tuple) else None
        if not (
            isinstance(first, ast.Constant) and first.value == step
            and isinstance(lst, ast.List) and _is_runner_call(parents.get(lst))
        ):
            raise AssertionError(f"step {arg.value} is not run through runner.run_steps")
        run_names.append(step)
    return run_names


def _check_confirmed_step(tree: ast.AST, parents: dict, call: ast.Call, step: str) -> str:
    """`<name> = resolve_step(ref)`, then ("<step>", lambda cfg: <name>(cfg, ...)) in run_steps."""
    assign = parents.get(call)
    if not (isinstance(assign, ast.Assign) and len(assign.targets) == 1
            and isinstance(assign.targets[0], ast.Name)):
        raise AssertionError(f"step {step} is not run through runner.run_steps (want `name = resolve_step(...)`)")
    name = assign.targets[0].id
    for lam in ast.walk(tree):
        body = getattr(lam, "body", None)
        if not (isinstance(lam, ast.Lambda) and isinstance(body, ast.Call)
                and getattr(body.func, "id", None) == name and lam.args.args and body.args
                and getattr(body.args[0], "id", None) == lam.args.args[0].arg):
            continue
        typed = body.args[1] if len(body.args) > 1 else None
        if not (isinstance(typed, ast.Constant) and isinstance(typed.value, str)):
            raise AssertionError(f"step {step}: the confirmation must be the name the operator typed "
                                 f"(a literal), never derived from the config")
        tup = parents.get(lam)
        lst = parents.get(tup)
        if (isinstance(tup, ast.Tuple) and tup.elts[1:2] == [lam]
                and isinstance(tup.elts[0], ast.Constant) and tup.elts[0].value == step
                and isinstance(lst, ast.List) and _is_runner_call(parents.get(lst))):
            return step
    raise AssertionError(f"step {step} is not run through runner.run_steps (no runner lambda calls {name})")


def check_interface(text: str) -> list[str]:
    """Resolve every toolkit name in the skill; return the step names it runs, in order."""
    for ref in REF_RE.findall(text):
        _check_ref(ref)
    run_names = []
    for block in PY_BLOCK_RE.findall(text):
        tree = ast.parse(block)
        _check_imports(tree)
        run_names += _check_resolve_calls(tree)
    for step in HEADING_RE.findall(text):
        if step not in RunFolder.STEPS:
            raise AssertionError(f"heading names step {step!r}, not in RunFolder.STEPS")
        if step not in run_names and not any(r.startswith(f"lgk.steps.{step}:") for r in QUERY_HELPERS):
            raise AssertionError(f"heading step {step!r} has no runner call or helper")
    return run_names


# ------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def skill_text() -> str:
    return COMMITTED.read_text(encoding="utf-8")


def _mutate(text: str, old: str, new: str) -> str:
    assert old in text, f"hostile fixture anchor {old!r} missing from SKILL.md"
    return text.replace(old, new, 1)


# ----------------------------------------------------------------- 1. frontmatter subset
class TestFrontmatter:
    def test_committed_frontmatter_is_exactly_name_and_description(self, skill_text):
        check_frontmatter(skill_text, COMMITTED.parent.name)
        meta, _ = split_skill(skill_text)
        assert meta["name"] == "lakebase-graph-kit"

    def test_host_specific_key_is_rejected(self, skill_text):
        # hostile: `allowed-tools` is a real, tolerated-by-one-host key, not a made-up one
        bad = _mutate(skill_text, "\n---\n", "\nallowed-tools: Bash Read\n---\n")
        with pytest.raises(AssertionError, match="exactly name"):
            check_frontmatter(bad, COMMITTED.parent.name)

    def test_name_must_match_folder(self, skill_text):
        # hostile: name differs only in case, so a case-folding compare would pass it
        bad = _mutate(skill_text, "name: lakebase-graph-kit", "name: Lakebase-Graph-Kit")
        with pytest.raises(AssertionError):
            check_frontmatter(bad, COMMITTED.parent.name)

    def test_body_is_host_agnostic(self, skill_text):
        check_host_agnostic(skill_text)
        # hostile: a host branch phrased in prose, not code
        bad = skill_text + "\nIf you are running in Genie Code, skip preflight.\n"
        with pytest.raises(AssertionError, match="names a host"):
            check_host_agnostic(bad)


# ------------------------------------------------------- 2. interface: names and the seam
class TestInterface:
    def test_runs_the_spec_steps_in_order_through_the_runner(self, skill_text):
        run_names = check_interface(skill_text)
        assert run_names[:9] == ["preflight", "profile", "propose", "privacy", "build", "enrich",
                                 "provision", "advise", "switch"]
        assert HEADING_RE.findall(skill_text) == SPEC_ORDER

    def test_every_ref_resolves_to_a_real_toolkit_object(self, skill_text):
        refs = set(REF_RE.findall(skill_text))
        assert "lgk.steps.provision:provision" in refs
        for ref in refs:
            assert callable(resolve_step(ref)), ref

    def test_nonexistent_step_ref_is_rejected(self, skill_text, monkeypatch):
        # never a skip (--require-live fails on any skip). Every RunFolder step has a module;
        # hostile: a step name that IS in RunFolder.STEPS but has no lgk.steps module, so only the
        # import, never the STEPS membership check, can reject the ref.
        assert _steps_without_module(RunFolder.STEPS) == []
        monkeypatch.setattr(RunFolder, "STEPS", [*RunFolder.STEPS, "nomodulestep"])
        assert _steps_without_module(RunFolder.STEPS) == ["nomodulestep"]
        bad = skill_text.replace("lgk.steps.provision:provision", "lgk.steps.nomodulestep:nomodulestep")
        with pytest.raises((ImportError, AttributeError)):
            check_interface(bad)

    def test_heading_for_unknown_step_is_rejected(self, skill_text):
        bad = _mutate(skill_text, "## Step 10: `walks`", "## Step 10: `serve`")
        with pytest.raises(AssertionError, match="not in RunFolder.STEPS"):
            check_interface(bad)

    def test_internal_function_ref_is_rejected(self, skill_text):
        # hostile: the internal really exists, so only the underscore rule can catch it
        bad = skill_text.replace(
            'resolve_step("lgk.steps.provision:provision")',
            'resolve_step("lgk.steps.provision:_create_synced_tables")',
        )
        with pytest.raises(AssertionError, match="internal"):
            check_interface(bad)

    def test_direct_import_of_a_step_is_rejected(self, skill_text):
        bad = _mutate(
            skill_text,
            "from lgk.runner import LaptopRunner\n",
            "from lgk.runner import LaptopRunner\nfrom lgk.steps.provision import provision\n",
        )
        with pytest.raises(AssertionError, match="internal module"):
            check_interface(bad)

    def test_direct_call_of_a_resolved_step_is_rejected(self, skill_text):
        # hostile: same public ref, resolved through the interface, but called outside run_steps
        bad = _mutate(
            skill_text,
            'runner.run_steps([("provision", resolve_step("lgk.steps.provision:provision"))])',
            'resolve_step("lgk.steps.provision:provision")(config)',
        )
        with pytest.raises(AssertionError, match="not run through"):
            check_interface(bad)

    def test_step_name_must_match_its_function(self, skill_text):
        # hostile: right function, wrong run.json slot - would record build's status as provision
        bad = _mutate(
            skill_text,
            '("provision", resolve_step("lgk.steps.provision:provision"))',
            '("build", resolve_step("lgk.steps.provision:provision"))',
        )
        with pytest.raises(AssertionError, match="not run through"):
            check_interface(bad)


# ------------------------------------------------------------------- 3. one file, two hosts
class TestInstall:
    def test_installer_reads_the_committed_file(self):
        assert SKILL_FILE.resolve() == COMMITTED.resolve()

    def test_both_hosts_get_the_committed_bytes(self, tmp_path):
        claude_home, genie_home = tmp_path / "dot_claude", tmp_path / "Users" / "me"
        # hostile: a stale older copy already sits at the Genie path and must be replaced
        stale = genie_home / ".assistant" / "skills" / "lakebase-graph-kit" / "SKILL.md"
        stale.parent.mkdir(parents=True)
        stale.write_text("---\nname: lakebase-graph-kit\ndescription: old\n---\n")

        paths = install_skill_to_both_hosts(claude_home, genie_home)

        claude_file = claude_home / "skills" / "lakebase-graph-kit" / "SKILL.md"
        assert paths == (claude_file, stale)
        committed = COMMITTED.read_bytes()
        assert claude_file.read_bytes() == committed
        assert stale.read_bytes() == committed

    def test_copy_is_byte_exact_for_non_ascii_and_crlf(self, tmp_path):
        # hostile: non-ASCII + CRLF + no trailing newline - a text-mode re-render changes these
        src_dir = tmp_path / "src" / "probe-skill"
        src_dir.mkdir(parents=True)
        src = src_dir / "SKILL.md"
        src.write_bytes("---\r\nname: probe-skill\r\ndescription: x → y\r\n---\r\nbody".encode())
        c, g = install_skill_to_both_hosts(tmp_path / "c", tmp_path / "g", skill_file=src)
        assert c == tmp_path / "c" / "skills" / "probe-skill" / "SKILL.md"
        assert g == tmp_path / "g" / ".assistant" / "skills" / "probe-skill" / "SKILL.md"
        assert c.read_bytes() == src.read_bytes() == g.read_bytes()


# ----------------------------------------------------------------- single source of truth
def test_no_skill_generator_left_in_the_toolkit():
    """The committed file is the only source: no renderer may produce a second SKILL.md."""
    import lgk.interview as interview

    src = Path(interview.__file__).read_text()
    # hostile shape: an escaped "---\\nname:" literal has no real newline, so look for the marker
    assert "---" not in src and "name:" not in src, "interview.py embeds SKILL.md content"
    assert not hasattr(interview, "InterviewSkill")


# ------------------------------------------------------------------- nightly refresh
class TestNightlyRefresh:
    def test_refresh_is_a_generated_job_not_a_rerun(self, skill_text):
        refs = set(REF_RE.findall(skill_text))
        assert "lgk.refresh.job:create_refresh_job" in refs
        check_interface(skill_text)

    def test_running_a_refresh_task_directly_is_rejected(self, skill_text):
        # hostile: the sync task really exists and resolves, so only the runner/helper rule
        # catches the skill doing the job's work itself
        bad = _mutate(
            skill_text,
            'resolve_step("lgk.refresh.job:create_refresh_job")',
            'resolve_step("lgk.refresh.sync:sync")',
        )
        with pytest.raises(AssertionError, match="not run through runner.run_steps"):
            check_interface(bad)


# --------------------------------------------------------- SKILL.md vs the code
SECTION_RE = re.compile(r"^## (.+)$", re.M)


def _section(text: str, title_start: str) -> str:
    """The body of the one `## <title_start>...` section, up to the next `## ` heading."""
    heads = [(m.start(), m.end(), m.group(1)) for m in SECTION_RE.finditer(text)]
    hits = [i for i, (_s, _e, t) in enumerate(heads) if t.startswith(title_start)]
    if len(hits) != 1:
        raise AssertionError(f"want one section starting {title_start!r}, got {len(hits)}")
    i = hits[0]
    return text[heads[i][1]:heads[i + 1][0] if i + 1 < len(heads) else len(text)]


def _interview_default(text: str, key: str) -> str:
    rows = [r for r in _section(text, "Interview: the config").splitlines()
            if r.startswith(f"| `{key}` |")]
    if len(rows) != 1:
        raise AssertionError(f"want one interview row for {key}, got {len(rows)}")
    return rows[0].strip().strip("|").split("|")[-1].strip()


def check_served_schema_row(text: str) -> None:
    """The served_schema default is the kit schema (versions.served_schema), and says so."""
    from lgk import versions

    cfg = {"graph_name": "g"}
    if versions.served_schema(cfg) != versions.kit_schema(cfg):
        raise AssertionError("versions.served_schema no longer defaults to the kit schema")
    default = _interview_default(text, "served_schema")
    if "kit schema" not in default or "`kit_schema`" not in default:
        raise AssertionError(f"served_schema default row says {default!r}, not the kit schema")
    if _interview_default(text, "kit_schema") != "`lgk_<graph_name>`" or "lgk_<graph_name>" not in default:
        raise AssertionError("served_schema row must name kit_schema's own default lgk_<graph_name>")


def check_enrich_step(text: str) -> None:
    """Enrich sits right after build and before provision, and says how its stop works."""
    heads = HEADING_RE.findall(text)
    if "enrich" not in heads or heads.index("enrich") != heads.index("build") + 1:
        raise AssertionError(f"enrich must come right after build, headings are {heads}")
    if heads.index("enrich") > heads.index("provision"):
        raise AssertionError("enrich must come before provision")
    body = _section(text, "Step 6: `enrich`")
    for must in ("enrich_menu.md", "enrich.yaml", "FileNotFoundError", "methods: []",
                 'resolve_step("lgk.steps.enrich:enrich")'):
        if must not in body:
            raise AssertionError(f"the enrich step does not say {must!r}")
    # The expected-stop line is said before the run (persona tests: an unexplained red error).
    said = body.find(ENRICH_EXPECTED_STOP)
    run = body.find('resolve_step("lgk.steps.enrich:enrich")')
    if said < 0 or said > run:
        raise AssertionError("the enrich step must give the expected-stop line before its run")
    if "**First run done when** you said that line before the run" not in " ".join(body.split()):
        raise AssertionError("the enrich step has no done-when for the expected-stop line")


ENRICH_EXPECTED_STOP = ('"Enrich stops once on purpose to show you its menu. The red error is expected; '
                        'nothing is wrong."')


# The text the skill had before every step shipped. Each paragraph naming one of these
# steps must not say it has no step.
NO_STEP_PHRASE = re.compile(r"\b(?:not a runner step|no runner step|(?:has|have) no step|no step|"
                            r"not available)\b", re.I)


def check_no_stale_step_text(text: str) -> None:
    """No "Not available yet" section, and no paragraph that says enrich or teardown has no step."""
    if any(t.startswith("Not available yet") for t in SECTION_RE.findall(text)):
        raise AssertionError("SKILL.md still has a 'Not available yet' section; every step has shipped")
    for para in re.split(r"\n\s*\n", text):
        names = re.findall(r"\b(enrich|teardown)\b", para, re.I)
        if names and NO_STEP_PHRASE.search(para):
            raise AssertionError(f"stale text says {sorted(set(names))} has no step: {para[:120]!r}")


def check_stated_signatures(text: str) -> None:
    """Every `lgk.x:f(a, b)` SKILL.md states names exactly f's required parameters, in order."""
    _check_stated_signatures(text)


def _check_stated_signatures(body: str) -> None:
    """Each `lgk.x:f(a, b)` the text states names exactly f's required parameters, in order."""
    import inspect

    for ref, args in re.findall(r"(lgk(?:\.\w+)+:\w+)\(([^)]*)\)", body):
        params = inspect.signature(resolve_step(ref)).parameters.values()
        required = [p.name for p in params if p.default is p.empty
                    and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        stated = [a.strip() for a in args.split(",") if a.strip()]
        if stated != required:
            raise AssertionError(f"{ref} is stated as ({', '.join(stated)}), the code needs {required}")


def check_function_roles(text: str) -> None:
    """Provision and switch name the reader role and both function roles (lgk.served_functions)."""
    from lgk.served_functions import FUNCTION_ROLES

    for step in ("provision", "switch"):
        body = _section(text, f"Step {SPEC_ORDER.index(step) + 1}: `{step}`")
        roles = re.findall(r"`(\w*reader)`", body)
        missing = [r for r in ("reader",) + FUNCTION_ROLES if r not in roles]
        if missing:
            raise AssertionError(f"the {step} step does not name the roles {missing}")


class TestSkillMatchesCode:
    def test_served_schema_default_row_matches_versions(self, skill_text):
        check_served_schema_row(skill_text)
        # hostile: the old "none" default, the exact text SKILL.md had before the served_schema default changed
        row = next(r for r in skill_text.splitlines() if r.startswith("| `served_schema` |"))
        bad = _mutate(skill_text, row, row.rsplit("|", 2)[0] + "| none |")
        with pytest.raises(AssertionError, match="not the kit schema"):
            check_served_schema_row(bad)
        # hostile: each half of the row's default dropped on its own
        for drop in ("`kit_schema`", "`lgk_<graph_name>`"):
            with pytest.raises(AssertionError):
                check_served_schema_row(_mutate(skill_text, row, row.replace(drop, "it")))

    def test_enrich_step_sits_between_build_and_provision(self, skill_text):
        check_enrich_step(skill_text)
        # hostile: the enrich section moved after provision, still with its runner call
        body = "## Step 6: `enrich`" + _section(skill_text, "Step 6: `enrich`")
        moved = skill_text.replace(body, "", 1).replace("## Step 8: `advise`", body + "## Step 8: `advise`", 1)
        with pytest.raises(AssertionError, match="right after build"):
            check_enrich_step(moved)
        body6 = _section(skill_text, "Step 6: `enrich`")
        # hostile: the expected-stop line moved after the run, or dropped
        line = ENRICH_EXPECTED_STOP
        moved_line = body6.replace(line, "it", 1).replace("print([r.to_dict() for r in results])",
                                                          "print([r.to_dict() for r in results])\n" + line, 1)
        for bad6 in (moved_line, body6.replace(line, "it", 1),
                     body6.replace("**First run done when**", "**Then**", 1)):
            with pytest.raises(AssertionError):
                check_enrich_step(skill_text.replace(body6, bad6, 1))
        for must in ("`methods: []`", "FileNotFoundError", "enrich_menu.md", "enrich.yaml",
                     'resolve_step("lgk.steps.enrich:enrich")'):
            # hostile: the section loses one fact at a time
            bad = skill_text.replace(body6, body6.replace(must, "it"), 1)
            with pytest.raises(AssertionError):
                check_enrich_step(bad)

    def test_no_stale_not_available_text(self, skill_text):
        """Replaces the early "teardown is not available" check: every step has shipped."""
        check_no_stale_step_text(skill_text)
        # hostile: the exact section SKILL.md had before every step shipped, re-added at the end
        with pytest.raises(AssertionError, match="Not available yet"):
            check_no_stale_step_text(skill_text + STALE_SECTION)
        # hostile: the stale claim as one sentence, in an otherwise current paragraph
        bad = _mutate(skill_text, "Whole-graph teardown drops exactly",
                      "Enrich and teardown have no step. Whole-graph teardown drops exactly")
        with pytest.raises(AssertionError, match="has no step"):
            check_no_stale_step_text(bad)

    @pytest.mark.parametrize("sentence", ["Teardown is not available.", "Enrich has no step.",
                                          "Teardown: no step.", "Enrich is not a runner step.",
                                          "Teardown has no runner step."])
    def test_each_stale_phrase_is_caught(self, skill_text, sentence):
        # hostile: each phrasing on its own paragraph, enrich and teardown each
        with pytest.raises(AssertionError, match="has no step"):
            check_no_stale_step_text(skill_text + "\n" + sentence + "\n")

    def test_stated_signatures_match_the_code(self, skill_text):
        check_stated_signatures(skill_text)
        # hostile: teardown stated with a signature the code does not have
        with pytest.raises(AssertionError, match="stated as"):
            check_stated_signatures(_mutate(skill_text, "teardown(config, confirm_graph_name)",
                                            "teardown(config)"))
        # hostile: retire's arguments in the wrong order (same names, so a set compare passes)
        with pytest.raises(AssertionError, match="stated as"):
            check_stated_signatures(_mutate(skill_text, "retire(config, version, confirm_graph_name)",
                                            "retire(config, confirm_graph_name, version)"))

    def test_provision_and_switch_name_the_function_roles(self, skill_text):
        check_function_roles(skill_text)
        # hostile: one role dropped from the switch step only
        switch = _section(skill_text, "Step 9: `switch`")
        bad = skill_text.replace(switch, switch.replace("`coordinator_reader`", "the coordinator"), 1)
        with pytest.raises(AssertionError, match="coordinator_reader"):
            check_function_roles(bad)
        # hostile: a role dropped from provision only, and `reader` dropped from provision
        provision = _section(skill_text, "Step 7: `provision`")
        for role in ("`public_reader`", "`reader`"):
            bad = skill_text.replace(provision, provision.replace(role, "the role"), 1)
            with pytest.raises(AssertionError, match="provision"):
                check_function_roles(bad)


# ================================================== the skill drives the whole flow
# The shipped flow, in order: every step and command of the toolkit. This list is
# the exam, written from the spec, never read from SKILL.md; the toolkit test below proves no
# shipped step module is missing from it.
SHIPPED_ORDER = ["preflight", "profile", "propose", "privacy", "build", "enrich", "provision",
                 "advise", "switch", "measure", "walks", "app", "notebook", "refresh_job", "export", "rollback", "retire",
                 "teardown", "reap"]
# The stage each public helper belongs to. A runner step is its own stage (its tuple name).
STAGE_OF_HELPER = {
    "lgk.steps.privacy:resolve_flag": "privacy",
    "lgk.keys:open_orphans": "profile",
    "lgk.keys:resolve_orphans": "profile",
    "lgk.steps.walks:get_walk_query": "walks",
    "lgk.steps.walks:get_lookup_query": "walks",
    "lgk.walks.sql:NodePredicate": "walks",
    "lgk.refresh.job:create_refresh_job": "refresh_job",
    "lgk.export:export": "export",
    "lgk.steps.switch:served_history": "rollback",
    "lgk.steps.switch:rollback": "rollback",
    "lgk.steps.retire:retire": "retire",
    "lgk.steps.retire:retire_versions": "retire",
    "lgk.reaper:find_orphan_jobs": "reap",
    "lgk.reaper:find_orphans": "reap",
    "lgk.reaper:reap": "reap",
}
# lgk.steps modules the skill never runs itself: tasks of the nightly refresh job (named there).
JOB_TASKS = {"reapply": "Nightly refresh"}
# The section before every step shipped, verbatim (hostile fixture for check_no_stale_step_text).
STALE_SECTION = """
## Not available yet

The run folder also lists `teardown`. It is not a runner step: the runner passes only the config,
and teardown also needs the graph name typed by the operator as a confirmation
(`lgk.steps.teardown:teardown(config, confirm_graph_name)`). Say so if the operator asks, and do
not try to do its work yourself.
"""
PY_BLOCK_POS_RE = re.compile(r"```python\n(.*?)```", re.S)
ANY_HEADING_RE = re.compile(r"^#{2,3} (.+)$", re.M)


def _stage_of(ref: str) -> str:
    return STAGE_OF_HELPER.get(ref) or ref.split(":", 1)[1]


def _collapse(stages: list) -> list:
    return [s for i, s in enumerate(stages) if i == 0 or stages[i - 1] != s]


def _resolve_refs_in_order(code: str) -> list:
    """The literal refs of every resolve_step(...) call in `code`, in source order."""
    calls = [n for n in ast.walk(ast.parse(code))
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "resolve_step"
             and n.args and isinstance(n.args[0], ast.Constant)]
    return [c.args[0].value for c in sorted(calls, key=lambda c: (c.lineno, c.col_offset))]


def skill_stages(text: str) -> list:
    """The stages SKILL.md runs, in its own order: each resolve_step ref of every python block."""
    refs = [r for block in PY_BLOCK_RE.findall(text) for r in _resolve_refs_in_order(block)]
    return _collapse([_stage_of(r) for r in refs])


def _step_modules() -> list:
    import pkgutil

    import lgk.steps

    return sorted(m.name for m in pkgutil.iter_modules(lgk.steps.__path__))


def check_toolkit_modules(text: str, modules: list, steps: list) -> None:
    """Every lgk.steps module is a skill stage (job tasks: named in their section); every
    run.json step is in SHIPPED_ORDER."""
    stages = skill_stages(text)
    for name in modules:
        if name in JOB_TASKS:
            if f"`{name}`" not in _section(text, JOB_TASKS[name]):
                raise AssertionError(f"job task {name} is not named under {JOB_TASKS[name]}")
        elif name not in stages or name not in SHIPPED_ORDER:
            raise AssertionError(f"step module {name} is not a skill stage")
    extra = [s for s in steps if s not in SHIPPED_ORDER]
    if extra:
        raise AssertionError(f"run.json steps {extra} are not in SHIPPED_ORDER")


def check_covers_shipped(text: str) -> None:
    run_names = check_interface(text)
    dups = sorted({s for s in run_names if run_names.count(s) > 1})
    if dups:
        raise AssertionError(f"SKILL.md runs the steps {dups} more than once")
    stages = skill_stages(text)
    if stages != SHIPPED_ORDER:
        missing = [s for s in SHIPPED_ORDER if s not in stages]
        raise AssertionError(f"SKILL.md runs {stages}, want {SHIPPED_ORDER} (missing {missing})")


class TestCoversTheShippedFlow:
    def test_skill_runs_every_shipped_stage_in_order(self, skill_text):
        check_interface(skill_text)
        check_covers_shipped(skill_text)

    def test_every_toolkit_step_module_is_a_stage(self, skill_text):
        modules = _step_modules()
        assert len(modules) >= 15, modules
        check_toolkit_modules(skill_text, modules, RunFolder.STEPS)
        # hostile: a new step module the skill does not run
        with pytest.raises(AssertionError, match="zzz_new"):
            check_toolkit_modules(skill_text, modules + ["zzz_new"], RunFolder.STEPS)
        # hostile: a run.json step the shipped order forgot
        with pytest.raises(AssertionError, match="serve"):
            check_toolkit_modules(skill_text, modules, RunFolder.STEPS + ["serve"])
        # hostile: the job task no longer named in its section
        nightly = _section(skill_text, "Nightly refresh")
        bad = skill_text.replace(nightly, nightly.replace("`reapply`", "the last task"), 1)
        with pytest.raises(AssertionError, match="reapply"):
            check_toolkit_modules(bad, modules, RunFolder.STEPS)
        # hostile: the job task named, but in another section
        moved = bad.replace("## Step 13: `teardown`", "Also `reapply`.\n\n## Step 13: `teardown`", 1)
        with pytest.raises(AssertionError, match="reapply"):
            check_toolkit_modules(moved, modules, RunFolder.STEPS)

    def test_a_step_run_twice_is_caught(self, skill_text):
        # hostile: the provision block duplicated right after itself (collapsing would hide it)
        block = 'results = runner.run_steps([("provision", resolve_step("lgk.steps.provision:provision"))])\n'
        with pytest.raises(AssertionError, match="more than once"):
            check_covers_shipped(_mutate(skill_text, block, block + block))

    @pytest.mark.parametrize("old,new,match", [
        ('teardown = resolve_step("lgk.steps.teardown:teardown")',
         'teardown, _ = resolve_step("lgk.steps.teardown:teardown"), 0', "not run through"),
        ('lambda cfg: teardown(cfg, "<graph name', 'lambda cfg: print(cfg, "<graph name', "not run through"),
        ('lambda cfg: teardown(cfg, "<graph name', 'lambda cfg: teardown(config, "<graph name', "not run through"),
        ('runner.run_steps([("teardown",', 'runner.plan_steps([("teardown",', "not run through"),
        ('("teardown", lambda cfg:', '("teardown", None, lambda cfg:', "not run through"),
        ('teardown(cfg, "<graph name, typed by the operator>")', 'teardown(cfg, cfg["graph_name"])', "typed"),
        ('teardown(cfg, "<graph name, typed by the operator>")', 'teardown(cfg, None)', "typed"),
        ('teardown(cfg, "<graph name, typed by the operator>")', 'teardown(cfg, f"{cfg[\'graph_name\']}")', "typed"),
        ('teardown = resolve_step("lgk.steps.teardown:teardown")',
         'teardown = other = resolve_step("lgk.steps.teardown:teardown")', "not run through"),
    ])
    def test_each_teardown_lambda_rule_can_fail(self, skill_text, old, new, match):
        with pytest.raises(AssertionError, match=match):
            check_interface(_mutate(skill_text, old, new))

    @pytest.mark.parametrize("section", ["Step 12: `notebook`", "Retire an old version", "Step 13: `teardown`",
                                         "Reap what a deleted branch left", "Step 6: `enrich`"])
    def test_dropping_a_stage_is_caught(self, skill_text, section):
        # hostile: one whole section removed, the rest of the file intact
        body = _section(skill_text, section)
        head = next(m.group(0) for m in SECTION_RE.finditer(skill_text) if m.group(1).startswith(section))
        with pytest.raises(AssertionError):
            check_covers_shipped(skill_text.replace(head + body, "", 1))

    def test_reordering_two_stages_is_caught(self, skill_text):
        # hostile: export and the refresh job swap places; every ref still resolves
        a, b = _section(skill_text, "Nightly refresh"), _section(skill_text, "Export")
        swapped = skill_text.replace(a, "\n@@A@@").replace(b, "\n@@B@@")
        swapped = swapped.replace("@@A@@", b).replace("@@B@@", a)
        check_interface(swapped)
        with pytest.raises(AssertionError, match="want"):
            check_covers_shipped(swapped)

    def test_renamed_helper_is_caught(self, skill_text):
        # hostile: a plausible but wrong helper name
        bad = _mutate(skill_text, 'resolve_step("lgk.steps.walks:get_lookup_query")',
                      'resolve_step("lgk.steps.walks:get_lookups_query")')
        with pytest.raises(AttributeError):
            check_interface(bad)

    def test_teardown_called_outside_the_runner_is_caught(self, skill_text):
        # hostile: the real teardown, with the typed name, but not through run_steps
        bad = _mutate(skill_text,
                      'results = runner.run_steps([("teardown", lambda cfg: teardown(cfg, '
                      '"<graph name, typed by the operator>"))])',
                      'results = teardown(config, "<graph name, typed by the operator>")')
        with pytest.raises(AssertionError, match="not run through"):
            check_interface(bad)
        # hostile: through the runner, but recorded under another step's name
        bad = _mutate(skill_text, '("teardown", lambda cfg: teardown(cfg,', '("switch", lambda cfg: teardown(cfg,')
        with pytest.raises(AssertionError, match="not run through"):
            check_interface(bad)


# ------------------------------------------------------- operator setup and features
def check_operator_setup(text: str) -> None:
    """The reader-role SQL and the export's service-principal setup, as the code has them."""
    from lgk import export
    from lgk.steps.provision import READER_SETUP_SQL

    if READER_SETUP_SQL not in _section(text, "Step 7: `provision`"):
        raise AssertionError("the provision step does not give the reader-role SQL verbatim")
    env = re.search(r"\$(\w+)", export.deploy_command(Path("b"))).group(1)
    body = _section(text, "Export")
    for must in (f"`{export.SP_VAR}`", f"`{env}`", "`CAN_USE`", "databricks_create_role",
                 "GRANT CONNECT, CREATE ON\nDATABASE databricks_postgres", f"`{export.JOB_KEY}`",
                 f"`{export.REFRESH_JOB_KEY}`"):
        if must not in body:
            raise AssertionError(f"the export section does not say {must!r}")


def _app_config_keys() -> set:
    import lgk.app.generate as gen

    return set(re.findall(r"(?:_app\(config\)|app)\.get\(\"(\w+)\"", Path(gen.__file__).read_text()))


def _feature_claims() -> list:
    """(section, needle, why): each needle is built from the code, so the text and code agree."""
    from lgk import mapping as mp
    from lgk import sync_limits as sl
    from lgk.privacy import binding, gate
    from lgk.steps import walks
    from lgk.walks import sql

    tmpl = Path(__file__).resolve().parents[1] / "src" / "lgk" / "app" / "template" / "backend" / "app"
    timeout = re.search(r"SET LOCAL statement_timeout = '(\d+)s'", (tmpl / "db.py.tmpl").read_text()).group(1)
    assert "/api/set-walk" in (tmpl / "main.py.tmpl").read_text()
    claims = [("Step 10: `walks`", f'"{d}"', "direction") for d in sql.DIRECTIONS]
    claims += [("Step 10: `walks`", f'"{k}"', "lookup kind") for k in walks.LOOKUP_KINDS]
    claims += [
        ("Step 10: `walks`", f"default {sql.MAX_SEEDS_DEFAULT}, at most {sql.MAX_SEEDS_CAP:,}", "seeds"),
        ("Step 10: `walks`", f"`{gate.MixedFoldSelectionError.__name__}`", "edge_types"),
        ("Step 10: `walks`", "`edge_types=[...]`", "edge_types"),
        ("Step 4: `privacy`", f"`{mp._EDGE_OPTIONAL_KEYS[0]}`", "fold"),
        ("Step 4: `privacy`", "`walk_keys`", "walk keys"),
        ("Step 4: `privacy`", f"at most {mp.MAX_WALK_KEYS}", "walk keys"),
        ("Step 5: `build`", "`weight`", "weight"),
        ("Step 5: `build`", "`JoinNotUniqueError`", "joins"),
        ("Step 3: `propose`", "`joins`", "joins"),
        ("Step 11: `app`", "`GET /api/set-walk", "set walk"),
        ("Step 11: `app`", f"{timeout} s statement timeout", "timeout"),
        ("Step 11: `app`", f"`SET LOCAL statement_timeout = '{timeout}s'`", "timeout"),
        ("Nightly refresh", "`enrich.yaml`", "enrich only when the run enriched"),
        ("Nightly refresh", f"`{binding.StaleBuildError.__name__}`", "stamp"),
        ("Step 7: `provision`", f"at most {sl.SYNCED_TABLES_PER_SOURCE} synced tables", "sync cap"),
        ("Step 7: `provision`", f"`sync_per_source_headroom` (config, default {sl.DEFAULT_SYNC_PER_SOURCE_HEADROOM})",
         "sync headroom"),
        ("Step 7: `provision`", f"`{sl.SyncLimitError.__name__}`", "sync cap"),
    ]
    return claims


def check_feature_claims(text: str) -> None:
    import inspect

    from lgk.steps import walks

    for section, needle, why in _feature_claims():
        if needle not in _section(text, section):
            raise AssertionError(f"{section} does not say {needle!r} ({why})")
    # every `name=` the walks section names is a parameter of a walk helper
    params = set(inspect.signature(walks.get_walk_query).parameters)
    params |= {"max_depth", "top_n", "method"}  # get_lookup_query's documented builder kwargs
    for kw in re.findall(r"`(\w+)=", _section(text, "Step 10: `walks`")):
        if kw not in params:
            raise AssertionError(f"the walks section names {kw}=, not a walk helper parameter")
    # every app.<key> the app section names is read by the app generator
    keys = _app_config_keys()
    for key in re.findall(r"`app\.(\w+)`", _section(text, "Step 11: `app`")):
        if key not in keys:
            raise AssertionError(f"the app section names app.{key}, which lgk.app.generate never reads")
    if "LGK_LIVE_AUTH_STORAGE" in text:
        raise AssertionError("LGK_LIVE_AUTH_STORAGE is for the test harness only")


class TestOperatorSetupAndFeatures:
    def test_operator_setup_matches_the_code(self, skill_text):
        check_operator_setup(skill_text)
        provision = _section(skill_text, "Step 7: `provision`")
        # hostile: one role left out of the setup SQL
        bad = skill_text.replace(provision, provision.replace(" CREATE ROLE public_reader WITH LOGIN;", ""), 1)
        with pytest.raises(AssertionError, match="reader-role SQL"):
            check_operator_setup(bad)
        # hostile: the SQL moved out of the provision step to the end of the file
        sql_line = "    CREATE ROLE reader WITH LOGIN; CREATE ROLE public_reader WITH LOGIN; CREATE ROLE coordinator_reader WITH LOGIN;\n"
        moved = skill_text.replace(provision, provision.replace(sql_line, ""), 1) + "\n" + sql_line
        with pytest.raises(AssertionError, match="reader-role SQL"):
            check_operator_setup(moved)
        export = _section(skill_text, "Export")
        for must in ("`sp_app_id`", "`LGK_SP_APP_ID`", "`CAN_USE`", "databricks_create_role",
                     "GRANT CONNECT, CREATE ON\nDATABASE databricks_postgres", "`lgk_provision`", "`lgk_refresh`"):
            # hostile: the SP setup loses one fact at a time
            bad = skill_text.replace(export, export.replace(must, "it"), 1)
            with pytest.raises(AssertionError, match="export section"):
                check_operator_setup(bad)

    def test_feature_claims_match_the_code(self, skill_text):
        check_feature_claims(skill_text)

    @pytest.mark.parametrize("claim", range(len(_feature_claims())))
    def test_each_feature_claim_can_fail(self, skill_text, claim):
        section, needle, _why = _feature_claims()[claim]
        body = _section(skill_text, section)
        bad = skill_text.replace(body, body.replace(needle, "it"), 1)
        with pytest.raises(AssertionError):
            check_feature_claims(bad)
        # hostile: the claim kept, but moved out of its section to the end of the file
        with pytest.raises(AssertionError):
            check_feature_claims(bad + "\n" + needle + "\n")

    @pytest.mark.parametrize("old,new", [
        ("`app.pguser`", "`app.pg_user`"),  # hostile: a plausible config key the app never reads
        ("`sort_method=\"degree\"`", "`sort=\"degree\"`"),  # hostile: a renamed walk keyword
    ])
    def test_misnamed_option_is_caught(self, skill_text, old, new):
        with pytest.raises(AssertionError, match="names"):
            check_feature_claims(_mutate(skill_text, old, new))

    def test_harness_only_env_is_refused(self, skill_text):
        with pytest.raises(AssertionError, match="test harness only"):
            check_feature_claims(skill_text + "\nSet LGK_LIVE_AUTH_STORAGE=plaintext.\n")


# ------------------------------------------- an offline scripted run of the skill's order
# The run reads SKILL.md's python blocks in order and executes them, as the interviewer would,
# with the real LaptopRunner on a temp run folder. Remote work is faked at the boundary: every
# resolve_step returns a stand-in that BINDS the skill's arguments to the real function's
# signature (a renamed helper or keyword fails), and WorkspaceClient refuses any call.
SKIPPED_BLOCKS = ("Notebook runner", "Running inside a Databricks workspace")  # the alternative runner; the scripted run is the laptop one
PLACEHOLDERS = {"<graph_name>": "providers", "<graph name, typed by the operator>": "providers"}


class _NoRemote:
    def __getattr__(self, name):
        raise AssertionError(f"remote call WorkspaceClient().{name} in the offline scripted run")


def skill_blocks(text: str) -> list:
    """(heading, code) for each python block, in order; heading = the nearest ##/### above it."""
    heads = [(m.start(), m.group(1)) for m in ANY_HEADING_RE.finditer(text)]
    out = []
    for m in PY_BLOCK_POS_RE.finditer(text):
        above = [h for pos, h in heads if pos < m.start()]
        out.append((above[-1] if above else "", m.group(1)))
    return out


def _fill(code: str, run_dir: Path) -> str:
    values = {**PLACEHOLDERS, "<run_folder>": str(run_dir)}
    return re.sub(r'"(<[^"]*>)"', lambda m: repr(values.get(m.group(1), m.group(1))), code)


def _stand_in_resolver(calls: list):
    import inspect

    from lgk import versions

    def fake(ref: str):
        target = resolve_step(ref)  # the real lookup: a renamed ref fails here
        if inspect.isclass(target):
            return target  # a pure value type (NodePredicate)
        sig = inspect.signature(target)

        def stand_in(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)  # TypeError when the skill's call does not fit
            calls.append(ref)
            if "confirm_graph_name" in bound.arguments:  # the real confirmation rule
                versions.confirm_graph_name(bound.arguments["config"], bound.arguments["confirm_graph_name"])
            return None

        return stand_in

    return fake


def scripted_run(text: str, run_dir: Path, monkeypatch) -> tuple:
    """Execute the skill's blocks in order; return (stages executed, run.json)."""
    import json

    import databricks.sdk
    import lgk.notebook_runner

    calls: list = []
    monkeypatch.setattr(databricks.sdk, "WorkspaceClient", _NoRemote)
    monkeypatch.setattr(lgk.notebook_runner, "resolve_step", _stand_in_resolver(calls))
    ns: dict = {"__name__": "skill_scripted_run"}
    for heading, code in skill_blocks(text):
        if heading.startswith(SKIPPED_BLOCKS):
            continue
        ns.pop("results", None)
        exec(compile(_fill(code, run_dir), f"SKILL.md: {heading}", "exec"), ns)
        failed = [r.to_dict() for r in ns.get("results") or [] if r.status.value == "failed"]
        if failed:  # the skill's rule: never go on after a failed step
            raise AssertionError(f"{heading}: {failed}")
    return _collapse([_stage_of(r) for r in calls]), json.loads((run_dir / "run.json").read_text())


class TestScriptedRun:
    def test_the_skill_order_runs_through_the_runner_to_teardown(self, skill_text, tmp_path, monkeypatch):
        executed, state = scripted_run(skill_text, tmp_path, monkeypatch)
        assert executed == skill_stages(skill_text)
        assert executed[-2:] == ["teardown", "reap"]
        # every runner step reached run.json, succeeded, and started in the skill's order
        ran = sorted((v["started_at"], k) for k, v in state.items() if "started_at" in v)
        runner_steps = [s for s in executed
                        if (s in RunFolder.STEPS or s in UNLISTED_RUNNER_STEPS) and s != "walks"]
        assert [k for _t, k in ran] == runner_steps
        assert runner_steps[-1] == "teardown"
        assert all(state[s]["status"] == "success" for s in runner_steps)

    def test_a_skill_without_the_teardown_block_does_not_reach_it(self, skill_text, tmp_path, monkeypatch):
        # hostile: the teardown section's code removed, its prose kept
        body = _section(skill_text, "Step 13: `teardown`")
        block = PY_BLOCK_POS_RE.search(body).group(0)
        executed, state = scripted_run(skill_text.replace(block, "", 1), tmp_path, monkeypatch)
        assert "teardown" not in executed and state["teardown"]["status"] == "not_run"

    def test_a_wrong_typed_name_stops_the_run(self, skill_text, tmp_path, monkeypatch):
        # hostile: the confirmation is a different graph's name
        bad = skill_text.replace('teardown(cfg, "<graph name, typed by the operator>")',
                                 'teardown(cfg, "<graph_name>s")', 1)
        with pytest.raises(AssertionError, match="teardown"):
            scripted_run(bad, tmp_path, monkeypatch)

    def test_a_remote_call_in_a_block_stops_the_run(self, skill_text, tmp_path, monkeypatch):
        # hostile: a block that does remote work itself instead of through a step
        bad = skill_text + "\n```python\nWorkspaceClient().jobs.list()\n```\n"
        with pytest.raises(AssertionError, match="remote call"):
            scripted_run(bad, tmp_path, monkeypatch)

    def test_a_call_that_does_not_fit_the_code_stops_the_run(self, skill_text, tmp_path, monkeypatch):
        # hostile: a keyword the helper does not take (a renamed parameter)
        bad = _mutate(skill_text, 'get_lookup_query("ancestors", served_schema=',
                      'get_lookup_query("ancestors", schema=')
        with pytest.raises(TypeError):
            scripted_run(bad, tmp_path, monkeypatch)


# ------------------------------------- P1-skilltext: the Genie persona-fix round (owner, 2026-10-04)
# Each claim is (criterion, where, needle). `where` is a section title start, "HEAD" (the text
# above the first `## ` heading: the top-level invariants) or "DESCRIPTION" (the frontmatter).
# Needles are compared on whitespace-flattened text, so a re-wrap does not break them, but a
# gutted sentence does. Needles that state code behaviour are built from the code below.
WORKSPACE_APP_PATH = "/Workspace/Users/<operator>/lgk-apps/<name>"
# exact and in order; the persona openings' own phrases are included (DETAILS.md section 4)
TRIGGER_PHRASES = ("Lakebase Graph Kit", "lakebase-graph-kit", "lgk", "serve a graph from Lakebase",
                   "find-a-doctor search", "shows related providers", "see which doctors work together",
                   "which doctors work together", "related providers graph")
GENERIC_TRIGGERS = {"graph", "graphs", "data", "table", "tables", "query", "sql", "schema", "network"}
# persona 5 (graph novice) transcript, docs/personas/results/5-graph-novice.md: kept verbatim,
# each bound to its own term (a swap of two quotes is caught)
TERM_PAIRS = (
    '**node**: "Nodes = the things (doctors, hospitals)"',
    '**edge**: "Edges = the connections between them"',
    '**hop**: "A “hop” is just a step',
    '**schema**: "A schema is just a folder',
    '**version**: version 1 is the "first version of this graph"',
    '**branch**: "a workspace within that server',
    '**fold**: "the toolkit counts how many patients',
    '**closure**: "everything below a hospital',
    '**degree**: "how many connections each doctor has"',
    '**PageRank**: "which doctors are most central"',
    '**components**: "which groups are connected"',
    '**similarity**: "which doctors are most alike"',
    "**Jaccard**: the similarity score: the neighbours two nodes share, divided by all the neighbours either has",
    "**DAG**: a graph with no cycle: following edges forward never leads back to where you started",
    "**p50 / p95**: the query time half of the runs beat, and the time 95 of 100 runs beat",
    "**label `lgk.privacy_stamp`**: a tag on the kit's tables that says which privacy decision they were built under",
    "**table_id**: the Unity Catalog id of a table; it changes when a table is dropped and made again",
    "**certified_version**: the last Delta version of the table whose data the kit checked",
)
PERSONA5_QUOTES = (
    "Nodes = the things (doctors, hospitals)",
    "Edges = the connections between them",
    "A “hop” is just a step — like following a line on a map from one dot to the next.",
    "A schema is just a folder that holds related tables.",
    "first version of this graph",
    "a workspace within that server, set up for you.",
    "the toolkit counts how many patients two doctors share, then throws away the IDs.",
    "everything below a hospital, at any depth",
    "how many connections each doctor has",
    "which doctors are most central",
    "which groups are connected",
    "which doctors are most alike",
)
# claims that were false and must never come back (flattened text)
FORBIDDEN = (
    "below the privacy floor fails the build",
    "below the floor fails the build",
    "| `branch_id` | Which branch of that project? | `production` |",
    "Show `<run_folder>/graph_mapping.yaml` and the graph shape (`tree` or `web`) in `<run_folder>/flags.json`.",
)
# contradictions of the fixes, matched on flattened, backtick-free, lower-case text (prover round 1)
CONTRADICTIONS = (
    r"fails? the build", r"build fails", r"build failure", r"must (?:also )?be a foreign key",
    r"foreign key is (?:also )?(?:needed|required)", r"search the workspace", r"(?:other|related|sibling) schemas",
    r"graph is a tree", r"unless you are confident", r"graph shape .{0,40}in .{0,40}flags\.json",
    r"drops? the column in unity catalog", r"left in place, unless", r"teardown drops them",
    r"(?:is|are) dropped too", r"accepted with a warning", r"run it again until",
    r"skip completion", r"most lines", r"never print status", r"any (?:graph|sql|data)\b.{0,20}question",
    r"unsure, use production", r"\bmay (?:also )?(?:search|browse|inspect)",
)


def _flat(s: str) -> str:
    return " ".join(s.split())


def _wheel_name() -> str:
    import tomllib

    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    return f"{project['name']}-{project['version']}-py3-none-any.whl"


def _persona_claims() -> list:
    from lgk.privacy import floor
    from lgk.served_functions import FUNCTION_ROLES

    wheel = _wheel_name()
    roles = ("reader",) + tuple(FUNCTION_ROLES)
    return [
        # 15. propose's shape is a guess; the measured shape is the top of enrich_menu.md
        ("15 shape", "Step 3: `propose`", "the `shape` in `<run_folder>/flags.json` (`tree` or `web`) is only a guess"),
        ("15 shape", "Step 3: `propose`", "made from the table-level draft before build"),
        ("15 shape", "Step 3: `propose`", "The measured shape is the first line of `<run_folder>/enrich_menu.md`"),
        ("15 shape", "Step 3: `propose`", "Never call the graph a tree (or a DAG) from this guess"),
        # 1. the two floors, defined apart, and no "fails the build"
        ("1 fold floor", "Terms", "**fold floor**: on a folded edge, the number of distinct people"),
        ("1 fold floor", "Terms", "A pair with fewer people than the floor is left out"),
        ("1 fold floor", "Terms", "`HAVING COUNT(DISTINCT <subject>) >= <floor>`. It is not a failure"),
        ("1 join floor", "Terms", "**join floor**: on a node `aggregates` join with a `subject`"),
        ("1 join floor", "Terms", "Below the floor that node's join outputs are left blank (NULL). It is not a failure"),
        ("1 floor value", "Terms", f"`privacy_floor`: default {floor.DEFAULT_FLOOR}. It can be raised and never lowered"),
        ("1 floor value", "Terms", f"(a value below {floor.DEFAULT_FLOOR} is refused)"),
        ("1 join floor", "Step 3: `propose`", "its outputs are blank for any node with fewer distinct patients than the privacy floor"),
        ("1 build", "Step 5: `build`", "a pair below the fold floor is left out"),
        ("1 build", "Step 5: `build`", "below the join floor is left blank (NULL), not a failure"),
        ("1 build", "Step 5: `build`", f"`{floor.FoldFloorError.__name__}` or `{floor.JoinFloorError.__name__}` only if"),
        # 2. fold eligibility, as _fold_allowed has it
        ("2 fold eligibility", "Step 4: `privacy`", "A composite key membership is enough: no foreign key and no patients table is needed."),
        ("2 fold eligibility", "Step 4: `privacy`", "a foreign-key column of its table, or one column of a composite key"),
        ("2 fold eligibility", "Step 4: `privacy`", "whole single-column key is never foldable"),
        ("2 fold eligibility", "Step 4: `privacy`", "Read `fold_allowed` in `flags.json` before you mention folding"),
        # 3. branch_id: no default, never production
        ("3 branch_id", "Interview: the config", "Always ask; offer no branch name. Never assume `production`"),
        # 4. app folder uploaded to the printed path before deploy
        ("4 app upload", "Step 11: `app`", "Before the deploy command, upload the app folder to that exact path"),
        ("4 app upload", "Step 11: `app`", f"(default `{WORKSPACE_APP_PATH}`, or `app.workspace_path`)"),
        ("4 app upload", "Step 11: `app`", "databricks workspace import-dir <app folder> <workspace path from the printed command>"),
        ("4 app upload", "Step 11: `app`", "databricks workspace list <workspace path from the printed command>"),
        ("4 app upload", "Step 11: `app`", "the workspace list of the printed `--source-code-path` shows the uploaded files"),
        # 5. the wheel, for the notebook runner and the refresh job
        ("5 wheel", "### Install the toolkit", "Every step imports the toolkit, so install it first."),
        ("5 wheel", "### Install the toolkit", "uv build --wheel"),
        ("5 wheel", "### Notebook runner", "databricks workspace list <work folder>"),
        ("5 wheel", "### Notebook runner", f"**Done when** that list shows `{wheel}` and `lgk_runner`"),
        ("5 wheel", "Nightly refresh", f"--file dist/{wheel} --format AUTO"),
        ("5 wheel", "Nightly refresh", "--file notebooks/lgk_refresh_task.py --language PYTHON --format SOURCE"),
        ("5 wheel", "Nightly refresh", f"wheel_path=\"/Workspace/Users/<operator>/lgk/{wheel}\""),
        # 6. trigger phrases (each one)
        *[("6 trigger", "DESCRIPTION", f'"{p}"') for p in TRIGGER_PHRASES],
        # 7. source boundary
        ("7 source boundary", "HEAD", "read only the catalog and schema the operator names"),
        ("7 source boundary", "HEAD", "Inspect nothing else"),
        ("7 source boundary", "HEAD", "Never reuse facts from earlier chats or from memory"),
        # 8. completion invariant + the Completion step
        ("8 completion", "HEAD", "Never say a step, or the whole run, is done unless its **Done when** line is met and you have shown the evidence"),
        ("8 completion", "HEAD", "say the step is NOT DONE and what is missing"),
        ("8 completion", "HEAD", "Always end with the [Completion](#completion) step"),
        ("8 completion", "Completion", "Say the run is done only when every line printed is DONE."),
        ("8 completion", "Completion", "Print every **Done when** line of each step you ran"),
        ("8 completion", "Completion", "each with its evidence"),
        ("8 completion", "Completion", "Mark any line whose evidence you have not shown as NOT DONE"),
        ("8 completion", "Completion", "Status: DONE | NOT DONE"),
        # 9. evidence rule, next to the invariant
        ("9 evidence", "HEAD", "Make a claim about latency"),
        ("9 evidence", "HEAD", "degree, edge direction, graph shape (tree, web, DAG) or a count only when you show"),
        ("9 evidence", "HEAD", "the exact query you ran and its output, both verbatim"),
        ("9 evidence", "HEAD", "If you have not shown them, do not make the claim: say it was not measured."),
        ("9 evidence", "HEAD", "This rule fails closed."),
        # 10. nightly refresh: scheduled, schedule shown
        ("10 schedule", "Nightly refresh", "The job is created with no schedule."),
        ("10 schedule", "Nightly refresh", "\"quartz_cron_expression\""),
        ("10 schedule", "Nightly refresh", "Ask the operator for the time and time zone of the nightly run, set the schedule, and show it"),
        ("10 schedule", "Nightly refresh", "databricks jobs get <job id> -o json"),
        ("10 schedule", "Nightly refresh", "the job exists with a schedule, and you have shown the job id and the schedule (`settings.schedule.quartz_cron_expression`"),
        ("10 config", "Nightly refresh", "k != \"workspace_client\""),
        # 11. jargon: verbatim persona-5 wording, first-use terms, inline before outputs
        *[("11 quote", "Terms", q) for q in PERSONA5_QUOTES],
        *[("11 term", "Terms", t) for t in TERM_PAIRS],
        ("11 first use", "Terms", "**label `lgk.privacy_stamp`**"),
        ("11 first use", "Terms", "**table_id**"),
        ("11 first use", "Terms", "**certified_version**"),
        ("11 first use", "Terms", "**DAG**: a graph with no cycle"),
        ("11 first use", "Terms", "**p50 / p95**"),
        ("11 first use", "Terms", "**Jaccard**: the similarity score"),
        ("11 inline privacy", "Step 4: `privacy`", "Before you show the privacy output, explain it in plain words"),
        ("11 inline privacy", "Step 4: `privacy`", "To fold is: \"the toolkit counts how many patients two doctors share, then throws away the IDs.\""),
        ("11 inline enrich", "Step 6: `enrich`", "Before you show the enrichment output"),
        ("11 inline enrich", "Step 6: `enrich`", "degree is \"how many connections each doctor has\""),
        ("11 inline advise", "Step 8: `advise`", "Before you show the advisor output, explain it in plain words"),
        ("11 inline advise", "Step 8: `advise`", "p50 is the query time half of the runs beat, p95 the time 95 of 100 runs beat"),
        # 12. advise on mixed graphs
        ("12 mixed advise", "Step 8: `advise`", "Advise fails on every graph that mixes folded and unfolded edge types: the step fails before it connects"),
        ("12 mixed advise", "Export: `lgk.export:export(config)`", "on a graph that mixes folded and unfolded edge types `lgk_provision` fails at advise and never switches: do not export such a graph."),
        ("12 care keys", "Step 4: `privacy`", "Node types and patient data. The privacy step fails with `CareKeyNodeError`, and names each node type it refuses, when:"),
        ("12 mixed advise", "Step 8: `advise`", "so do not run it again"),
        ("12 mixed advise", "Step 8: `advise`", "This is the one failed step you may go past."),
        ("12 mixed advise", "HEAD", "(the one exception is advise on a mixed graph; see its step)"),
        # 13. resolve_flag records only
        ("13 resolve_flag", "Step 4: `privacy`", "`resolve_flag` only records a decision"),
        ("13 resolve_flag", "Step 4: `privacy`", "It never alters the source tables: a `drop` drops nothing in Unity Catalog."),
        ("13 resolve_flag", "Step 4: `privacy`", '"drop" removes the column from the GRAPH mapping only; the source table is unchanged.'),
        # 14. teardown
        ("14 teardown", "Step 13: `teardown`", "teardown needs the graph name typed to confirm"),
        ("14 teardown", "Step 13: `teardown`", "it removes only what this graph's manifest says it created"),
        ("14 teardown", "Step 13: `teardown`", "(a different name is refused)"),
        ("14 teardown", "Step 13: `teardown`", "other graphs on the branch use them"),
        ("14 teardown", "Step 13: `teardown`", "the Postgres tables and schemas"),
        ("14 teardown call", "Step 13: `teardown`", "its second parameter is `confirm_graph_name` (there is no `confirm_name`)"),
        ("7 memory", "HEAD", "Ignore memory or notes from earlier chats."),
        ("7 memory", "HEAD", "If the operator has named no catalog and schema, ask; never pick one."),
        ("14 teardown", "Step 13: `teardown`",
         f"The shared roles `{roles[0]}`, `{roles[1]}` and `{roles[2]}` are left in place"),
        # 15. persona 6 (SRE): source tables read-only, consent per question, care after a stop
        ("15 read-only", "HEAD", "Source tables are read-only. Never run ALTER, CREATE, INSERT, UPDATE, DELETE, MERGE, DROP, TRUNCATE or GRANT"),
        ("15 read-only", "HEAD", "show the operator the exact statement, ask them to run it themselves, and wait until they say they ran it"),
        ("15 read-only", "Step 2: `profile`", "If the output shows the key, tell the operator the profile missed it and stop; do not change the tables."),
        ("15 read-only", "Step 2: `profile`", "show the operator the statement to run themselves (source tables are read-only for you)"),
        ("15 consent", "HEAD", "If the operator's reply is about something else, it is not a yes"),
        ("15 consent", "HEAD", "wait for one yes for that plan"),
        ("15 consent", "HEAD", "before a teardown, a retire, a reap, a job, an upload, an app, or any change to the plan"),
        ("17 workspace", "### Install the toolkit", "skip this section and the runner sections below: go to \"Running inside a Databricks workspace\""),
        ("17 workspace", "### Running inside a Databricks workspace", "Install first, before you write the config or make a runner: the restart below clears every Python variable."),
        ("17 workspace", "### Running inside a Databricks workspace", "After the restart, write the config"),
        ("17 workspace", "### Notebook runner", "Inside a Databricks workspace, give no `notebook_source`"),
        ("17 workspace", "Nightly refresh", "upload the task notebook with `upload_notebook`"),
        ("15 after stop", "HEAD", "After any stop (an error you cannot explain, a safety block, or the operator says stop), describe each next action before you run it and wait for a yes"),
        ("15 undo", "HEAD", "Before you offer to undo something, query its current state and show the output."),
        ("15 no early result", "HEAD", "Never state a step's result before the step has run"),
        ("15 no memory writes", "HEAD", "Never write facts about this run (names, schemas, branches, results) to memory or notes."),
        ("15 term", "Terms", "**gold tables**: the clean, final tables"),
        # 16. persona 2 (app developer): the set walk's one seed condition, endpoints from the contract
        ("16 set walk", "Step 10: `walks`", "A set walk takes exactly one seed condition (one walk key equal to one value) and at most one stop condition."),
        ("16 set walk", "Step 10: `walks`", "It cannot combine seed conditions"),
        ("16 set walk", "Step 11: `app` (app developers and architects only)", "It takes one seed condition and one stop condition only."),
        ("16 contract", "Step 11: `app` (app developers and architects only)", "Describe the app's endpoints to the operator only from the generated `DATA_CONTRACT.md`, after this step has run."),
    ]


def _where(text: str, where: str) -> str:
    meta, body = split_skill(text)
    if where == "DESCRIPTION":
        return meta["description"]
    if where == "HEAD":
        first = SECTION_RE.search(body)
        return body[:first.start()] if first else body
    if where.startswith("### "):
        heads = list(ANY_HEADING_RE.finditer(text))
        hits = [i for i, m in enumerate(heads) if m.group(0).startswith(where)]
        if len(hits) != 1:
            raise AssertionError(f"want one subsection {where!r}, got {len(hits)}")
        i = hits[0]
        return text[heads[i].end():heads[i + 1].start() if i + 1 < len(heads) else len(text)]
    return _section(text, where)


def _trigger_list(description: str) -> list:
    m = re.search(r"Trigger phrases include (.*)$", description)
    if not m:
        raise AssertionError("the description has no 'Trigger phrases include ...' list")
    return re.findall(r'"([^"]+)"', m.group(1))


def check_persona_fixes(text: str) -> None:
    flat = _flat(text)
    for bad in FORBIDDEN:
        if _flat(bad) in flat:
            raise AssertionError(f"SKILL.md still says {bad!r}")
    for crit, where, needle in _persona_claims():
        if _flat(needle) not in _flat(_where(text, where)):
            raise AssertionError(f"[{crit}] {where} does not say {needle!r}")
    plain = flat.replace("`", "").lower()
    for pattern in CONTRADICTIONS:
        hit = re.search(pattern, plain)
        if hit:
            raise AssertionError(f"SKILL.md contradicts a fix: {hit.group(0)!r} ({pattern})")
    # 3: the branch_id row offers no default and names production only to forbid it
    if _interview_default(text, "branch_id") != "none":
        raise AssertionError(f"branch_id default is {_interview_default(text, 'branch_id')!r}, want none")
    row = next(r for r in _section(text, "Interview: the config").splitlines() if r.startswith("| `branch_id` |"))
    if row.replace("Never assume `production`", "").count("production"):
        raise AssertionError(f"the branch_id row offers production: {row!r}")
    # every Done-when line and every command is required, never optional
    for line in re.findall(r"^\*\*Done when\*\*.*$", text, re.M):
        if re.search(r"optional|if time allows|if they want|if you can", line, re.I):
            raise AssertionError(f"a Done-when line is made optional: {line!r}")
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        for cmd in block.splitlines():
            if cmd.lstrip().startswith("#") or re.search(r"optional", cmd, re.I):
                raise AssertionError(f"a required command is commented out or optional: {cmd!r}")
    for section in ("### Notebook runner", "Step 11: `app`", "Nightly refresh"):
        if re.search(r"\boptional\b[^.]*\b(?:wheel|upload|schedule|import)", _where(text, section), re.I):
            raise AssertionError(f"{section} makes an upload, the wheel or the schedule optional")
    # 4: the upload comes before the Done-when line of the app step
    app = _section(text, "Step 11: `app`")
    if not 0 <= app.find("import-dir") < app.find("**Done when**"):
        raise AssertionError("the app upload must come before the app step's Done-when line")
    # 6: exactly the specific trigger phrases, in order; no catch-all
    desc = _where(text, "DESCRIPTION")
    got = _trigger_list(desc)
    if got != list(TRIGGER_PHRASES):
        raise AssertionError(f"trigger phrases {got} != {list(TRIGGER_PHRASES)}")
    generic = [p for p in got if p.lower().strip() in GENERIC_TRIGGERS]
    if generic or re.search(r"\bany (?:graph|sql|question|data)\b", desc, re.I):
        raise AssertionError(f"generic trigger phrases {generic or desc}")
    # 8: Completion is the last heading of any level, and its template holds the status line
    heads = ANY_HEADING_RE.findall(text)
    if not heads or heads[-1] != "Completion" or SECTION_RE.findall(text)[-1] != "Completion":
        raise AssertionError(f"the last section must be Completion, got {heads[-1:]}")
    fences = re.findall(r"```text\n(.*?)```", _section(text, "Completion"), re.S)
    if not any("Status: DONE | NOT DONE" in f for f in fences):
        raise AssertionError("Completion's text template has no 'Status: DONE | NOT DONE' line")
    # 10: the schedule the skill sets is a real, unpaused Quartz schedule
    import json as _json

    nightly = _section(text, "Nightly refresh")
    m = re.search(r"databricks jobs update --json '(.*?)'", nightly)
    if not m:
        raise AssertionError("Nightly refresh does not set the schedule with databricks jobs update")
    body = _json.loads(m.group(1).replace("<job id>", "0"))
    if "job_id" not in body:  # the CLI refuses a JOB_ID argument with --json (skill E2E, 2026-10-05)
        raise AssertionError("the jobs update JSON body names no job_id")
    sched = body["new_settings"]["schedule"]
    if not sched.get("quartz_cron_expression") or sched.get("pause_status") != "UNPAUSED" or not sched.get("timezone_id"):
        raise AssertionError(f"the schedule the skill sets is not a live Quartz schedule: {sched}")
    # 11: the explanation is the first thing after the step's run block (privacy, enrich), or the
    # paragraph right before it (advise): no "show X first" can slip in ahead of it
    for section, explain, after_block in (("Step 4: `privacy`", "Before you show the privacy output", True),
                                          ("Step 6: `enrich`", "- Before you show the enrichment output", True),
                                          ("Step 8: `advise`", "Before you show the advisor output", False)):
        body = _section(text, section)
        fence = body.find("```python")
        start = body.find(explain)
        if fence < 0 or start < 0:
            raise AssertionError(f"{section}: no run block or no plain-words explanation before output")
        if after_block:
            end = body.find("```", fence + 3) + 3
            if body[end:start].strip():
                raise AssertionError(f"{section}: the explanation must come before the output, right after the run block")
        else:
            gap = body[body.find("\n\n", start):fence]
            if gap.strip():
                raise AssertionError(f"{section}: the explanation must come before the output, right before the run block")
    for section, explain, output in (
        ("Step 4: `privacy`", "Before you show the privacy output", "Read the open flags in `flags.json`"),
        ("Step 6: `enrich`", "Before you show the enrichment output", "Show the operator the menu"),
        ("Step 8: `advise`", "Before you show the advisor output", 'results = runner.run_steps([("advise",'),
    ):
        body = _flat(_section(text, section))
        if body.find(explain) < 0 or body.find(output) < 0 or body.find(explain) > body.find(output):
            raise AssertionError(f"{section}: the plain-words explanation must come before {output!r}")


def _cut(text: str, where: str, needle: str) -> str:
    """`text` with `needle` (matched across any line wrap) replaced by "it" inside `where`."""
    region = _where(text, where)
    pattern = r"\s+".join(re.escape(w) for w in needle.split())
    cut, n = re.subn(pattern, "it", region, count=1)
    assert n == 1, f"hostile fixture anchor {needle!r} missing from {where}"
    return text.replace(region, cut, 1)


class TestPersonaFixes:
    def test_committed_skill_has_every_fix(self, skill_text):
        check_persona_fixes(skill_text)

    @pytest.mark.parametrize("claim", range(len(_persona_claims())))
    def test_each_fix_can_fail(self, skill_text, claim):
        crit, where, needle = _persona_claims()[claim]
        # hostile: the sentence gutted in place, the rest of the file intact
        gutted = _cut(skill_text, where, needle)
        with pytest.raises(AssertionError):
            check_persona_fixes(gutted)
        if where != "DESCRIPTION":
            # hostile: the claim kept, but moved out of its place to the end of the file
            # (a Completion claim is moved to the top: the end of the file IS Completion)
            moved = (gutted.replace("\n## Terms\n", f"\n{needle}\n\n## Terms\n", 1) if where == "Completion"
                     else gutted + "\n" + needle + "\n")
            with pytest.raises(AssertionError):
                check_persona_fixes(moved)

    @pytest.mark.parametrize("old,new,match", [
        # hostile: the reverted items, as the reviewer will revert them
        ("Which branch of that project? Always ask; offer no branch name. Never assume `production`. The toolkit "
         "has no default branch: it refuses a config with no `branch_id` (`MissingBranchError`), and preflight "
         "requires it, so always write the answer into the config. | none |",
         "Which branch of that project? | `production` |", "production"),
        ("a pair below the fold floor is left out (the fold\nSQL's `HAVING COUNT(DISTINCT ...) >= <floor>`), not a failure.",
         "a folded edge below the privacy floor fails the\nbuild.", "fails the build"),
        ('"which doctors work together", ', "", "which doctors work together"),
        ('"lgk", ', '"graph", ', "lgk"),  # a generic word for a specific one
    ])
    def test_reverted_item_is_caught(self, skill_text, old, new, match):
        with pytest.raises(AssertionError, match=match):
            check_persona_fixes(_mutate(skill_text, old, new))

    @pytest.mark.parametrize("extra", ['"data", ', '"graph ", ', '"knowledge graph", ', '"Postgres", '])
    def test_a_generic_trigger_word_is_refused(self, skill_text, extra):
        anchor = 'Trigger phrases include "'
        assert anchor in skill_text
        with pytest.raises(AssertionError, match="trigger phrases"):
            check_persona_fixes(skill_text.replace(anchor, anchor[:-1] + extra + '"', 1))

    def test_deleting_the_completion_step_is_caught(self, skill_text):
        body = _section(skill_text, "Completion")
        with pytest.raises(AssertionError):
            check_persona_fixes(skill_text.replace("## Completion" + body, "", 1))

    def test_a_section_after_completion_is_caught(self, skill_text):
        with pytest.raises(AssertionError, match="last section"):
            check_persona_fixes(skill_text + "\n## Afterword\n\nMore.\n")

    def test_deleting_the_head_invariants_is_caught(self, skill_text):
        head = _where(skill_text, "HEAD")
        start = head.index("**Completion invariant.**")
        end = head.index("You interview")
        with pytest.raises(AssertionError, match="HEAD"):
            check_persona_fixes(skill_text.replace(head[start:end], "", 1))

    @pytest.mark.parametrize("section,explain", [
        ("Step 4: `privacy`", "Before you show the privacy output"),
        ("Step 6: `enrich`", "Before you show the enrichment output"),
        ("Step 8: `advise`", "Before you show the advisor output"),
    ])
    def test_jargon_line_moved_after_its_output_is_caught(self, skill_text, section, explain):
        body = _section(skill_text, section)
        para = next(p for p in re.split(r"\n(?=\n|- )", body) if explain in p)
        bad = skill_text.replace(body, body.replace(para, "", 1) + para + "\n", 1)
        with pytest.raises(AssertionError, match="before"):
            check_persona_fixes(bad)

    # ---- the code facts the new text states (read the expression, not the name)
    def test_fold_sql_leaves_sub_floor_pairs_out_with_having(self):
        from lgk.privacy import floor

        for sql in (floor.fold_pair_sql("c.s.t", "patient_id", "a", "b"),
                    floor.fold_edges_sql("c.s.t", "patient_id", "a")):
            assert re.search(r"HAVING COUNT\(DISTINCT (?:\w+\.)?patient_id\) >= 11$", sql), sql
        # the join floor blanks the output, it does not raise
        expr = floor.join_output_expr("j0", "visits", True, None)
        assert expr == "CASE WHEN j0._lgk_subjects >= 11 THEN CAST(j0.visits AS STRING) END"
        with pytest.raises(ValueError):
            floor.checked_floor(10)  # never lowered
        assert floor.checked_floor(12) == 12  # raised

    def test_fold_allowed_needs_only_a_composite_key(self):
        from lgk.steps.privacy import _fold_allowed

        # hostile: no foreign key at all, the subject is one column of a composite key
        assert _fold_allowed({"key_columns": ["provider_id", "patient_id"], "foreign_keys": []}, "patient_id")
        assert _fold_allowed({"key_columns": [], "foreign_keys": [{"columns": ["patient_id"]}]}, "patient_id")
        assert not _fold_allowed({"key_columns": ["patient_id"], "foreign_keys": []}, "patient_id")
        assert not _fold_allowed({"key_columns": ["a", "b"], "foreign_keys": []}, "patient_id")

    def test_branch_has_no_fallback_so_the_skill_must_ask(self):
        from lgk.branch import BranchRef, MissingBranchError

        with pytest.raises(MissingBranchError, match="branch_id"):  # owner "branch a": never production
            BranchRef.from_config({"lakebase_project": "p"})

    def test_app_default_workspace_path_is_the_one_the_skill_names(self):
        from lgk.app.generate import _workspace_path, deploy_command

        assert _workspace_path({"operator_email": "<operator>"}, "<name>") == WORKSPACE_APP_PATH
        assert deploy_command("n", WORKSPACE_APP_PATH).endswith(f"--source-code-path {WORKSPACE_APP_PATH}")

    def test_notebook_backend_uploads_the_wheel_and_runner_named_in_the_skill(self, tmp_path):
        from lgk.notebook_runner import DatabricksNotebookBackend

        b = DatabricksNotebookBackend(object(), "/Workspace/Users/user@example.com/lgk_v1_abc123",
                                      tmp_path / _wheel_name(), tmp_path / "nb.py")
        assert b.remote_wheel.endswith("/" + _wheel_name()) and b.remote_notebook.endswith("/lgk_runner")

    def test_refresh_job_is_created_without_a_schedule_and_carries_json_config(self):
        import json

        from lgk.refresh import job

        seen = {}

        class _Jobs:
            def create(self, **kw):
                seen.update(kw)
                return type("R", (), {"job_id": 7})()

        cfg = {"graph_name": "g", "lakebase_project": "p", "lakebase_catalog": "c", "kit_schema": "lgk_g",
               "served_schema": "lgk_g", "branch_id": "t-x", "run_folder": "/Workspace/Users/user@example.com/lgk_g/run",
               "warehouse_id": "wh",
               "version": 1}
        orig = job.record_job
        job.record_job = lambda *a, **k: None
        try:
            assert job.create_refresh_job(cfg, "/nb", "/w.whl", w=type("W", (), {"jobs": _Jobs()})()) == 7
        finally:
            job.record_job = orig
        assert "schedule" not in seen and "trigger" not in seen  # so the skill must set one
        params = [t.notebook_task.base_parameters for t in seen["tasks"]]
        assert all(json.loads(p["config"])["graph_name"] == "g" for p in params)
        # The skill still drops workspace_client first; the job code also drops it (P2), so a
        # leftover client never reaches the job's JSON config.
        tasks = job.refresh_job_tasks(dict(cfg, workspace_client=object()), "/nb", "/w.whl")
        assert all("workspace_client" not in json.loads(t.notebook_task.base_parameters["config"])
                   for t in tasks)

    def test_advise_on_a_mixed_graph_fails_with_edge_types_required(self):
        from lgk.privacy.gate import _derive_selection

        edge = lambda fold: type("E", (), {"fold_subject": fold})()  # noqa: E731
        mixed = type("M", (), {"edges": {"shares": edge("patient_id"), "works_at": edge(None)}})()
        with pytest.raises(ValueError, match="edge_types is required"):
            _derive_selection(mixed, None)

    def test_resolve_flag_writes_only_flags_json(self, tmp_path):
        import json

        from lgk.privacy.flags import new_flag, write_flags_file
        from lgk.steps.privacy import resolve_flag

        write_flags_file(str(tmp_path), {"flags": [new_flag("encounters", "patient_id", "name", True)]})
        (tmp_path / "schema.json").write_text("{}")
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        resolve_flag(str(tmp_path), "encounters", "patient_id", "drop", chooser="operator@example.com")
        after = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        assert set(after) == set(before)
        assert [n for n in after if after[n] != before[n]] == ["flags.json"]
        assert json.loads(after["flags.json"])["flags"][0]["choice"] == "drop"
