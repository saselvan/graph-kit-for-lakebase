"""Freeze of skill/lakebase-graph-kit/SKILL.md against the code (stream S1, owner decision "freeze a").

The phrase tests in test_skill.py check that sentences EXIST. They let a reworded contradiction
pass (57 of 66 survived). This file adds two layers:

1. Snapshot. Each `##` section of SKILL.md (plus the frontmatter and the head before the first
   `##`) has its sha256 in tests/skill_snapshot.json. Any edit, to any section, fails here until
   the snapshot is regenerated ON PURPOSE:

       PYTHONPATH=$PWD/src uv run --no-sync python tests/test_skill_freeze.py --regen

   Regenerate only after every binding test below passes on the edited file and the audit doc
   (docs/skill-audit-*.md) records the edited line with the code file:line that proves it.

2. Bindings. Each fact class is read from SKILL.md and, separately, from the code (an import of
   the constant, a signature, or a call that shows the behaviour), and the two are compared:
   - numbers: every digit run in SKILL.md is a named group of a NUMBER binding, or is in the
     reasoned NUMBER_ALLOW list; an unbound number fails (check_number_coverage);
   - status words: every status word near "status" (and every backticked status word) is bound
     to RunStatus / the preflight result constants (check_status_words);
   - python: every call in a ```python block, and every inline call in backticks, binds to the
     real signature (inspect.signature(...).bind) and its literal values to the code's value sets
     (check_python_blocks, check_inline_code);
   - CLI: every `databricks ...` / `uv ...` command is parsed against CLI_SURFACE, the subcommands
     and flags of Databricks CLI v1.14.1 (checked against the installed CLI when there is one);
   - names: every backticked code name is a name or string in the kit's code (check_inline_code);
   - claims: sentences whose meaning the code decides (an error name, a verdict such as refused /
     succeeds / passes, a default, which lookup kind does what, an order, a count in words, the
     file a step writes) are bound to what the code DOES (check_claims). Every backticked error
     name, every positive verdict word and every number word must sit in a bound claim;
   - config keys: every key the skill names is one the code reads from a config (check_config_keys);
   - CLI flags a command needs are present (check_cli_required).
   Every backticked token must be claimed by one of these checks or by TOKEN_ALLOW (with a reason).

Every checker also runs on hostile variants of the real file (TestHostile), and on monkeypatched
code constants, so a checker that cannot fail fails here.
"""

import ast
import hashlib
import importlib
import inspect
import json
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL = REPO_ROOT / "skill" / "lakebase-graph-kit" / "SKILL.md"
SNAPSHOT = Path(__file__).resolve().parent / "skill_snapshot.json"
MIN_SECTIONS = 20  # the file has 23 `##` sections; an empty or truncated snapshot must fail

FENCE_RE = re.compile(r"^```(\w*)\n(.*?)^```\n", re.M | re.S)
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
TICK_RE = re.compile(r"`([^`\n]+)`")


# =============================================================================== sections
def sections(text: str) -> List[Tuple[str, int, int]]:
    """(title, start, end) for FRONTMATTER, HEAD and each `## ` section, in file order.

    A `## ` line inside a fenced block is not a heading.
    """
    m = re.match(r"\A---\n.*?\n---\n", text, re.S)
    if not m:
        raise AssertionError("SKILL.md has no frontmatter")
    fences = [(f.start(), f.end()) for f in FENCE_RE.finditer(text)]
    heads = [h for h in re.finditer(r"^## (.+)$", text, re.M)
             if not any(s <= h.start() < e for s, e in fences)]
    out = [("FRONTMATTER", 0, m.end()), ("HEAD", m.end(), heads[0].start() if heads else len(text))]
    for i, h in enumerate(heads):
        out.append((h.group(1), h.start(), heads[i + 1].start() if i + 1 < len(heads) else len(text)))
    return out


def section_span(text: str, title: str) -> Tuple[int, int]:
    hits = [(s, e) for t, s, e in sections(text) if t == title]
    if len(hits) != 1:
        raise AssertionError(f"want exactly one section {title!r}, got {len(hits)}")
    return hits[0]


def snapshot_of(text: str) -> Dict[str, Any]:
    secs = [{"title": t, "sha256": hashlib.sha256(text[s:e].encode()).hexdigest(),
             "lines": text[s:e].count("\n")} for t, s, e in sections(text)]
    return {"file_sha256": hashlib.sha256(text.encode()).hexdigest(), "sections": secs}


def check_snapshot(text: str, snap: Dict[str, Any]) -> None:
    if len(snap.get("sections", [])) < MIN_SECTIONS:
        raise AssertionError(f"snapshot holds {len(snap.get('sections', []))} sections (< {MIN_SECTIONS}): "
                             "an empty or truncated snapshot never passes")
    now = snapshot_of(text)
    want = [s["title"] for s in snap["sections"]]
    got = [s["title"] for s in now["sections"]]
    if want != got:
        raise AssertionError(f"SKILL.md sections changed: snapshot {want} != file {got}")
    changed = [n["title"] for n, o in zip(now["sections"], snap["sections"]) if n["sha256"] != o["sha256"]]
    if changed or now["file_sha256"] != snap["file_sha256"]:
        raise AssertionError(f"SKILL.md sections edited since the snapshot: {changed or ['(whole file)']}. "
                             "If on purpose: make every binding test pass, record the change in the audit "
                             "doc, then regenerate (see this module's docstring).")


# ======================================================================== code readers
def _wheel_name() -> str:
    import tomllib

    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    return f"{project['name']}-{project['version']}-py3-none-any.whl"


def _notebook_constant(name: str, notebook: str) -> Any:
    """A module-level constant of a workspace notebook (they import dbutils, so they are parsed)."""
    tree = ast.parse((REPO_ROOT / "notebooks" / notebook).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"notebooks/{notebook} has no constant {name}")


def _min_python() -> str:
    """The Python both workspace notebooks require; they must agree with each other and pyproject."""
    import tomllib

    pins = {_notebook_constant("MIN_PYTHON", n) for n in ("lgk_runner.py", "lgk_refresh_task.py")}
    if len(pins) != 1:
        raise AssertionError(f"the notebooks pin different Pythons: {pins}")
    pin = ".".join(str(p) for p in pins.pop())
    req = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["requires-python"]
    if req != f">={pin}":
        raise AssertionError(f"pyproject requires-python {req!r} != notebook pin >={pin}")
    return pin


def _sdk_pin() -> str:
    """The databricks-sdk floor the refresh task notebook upgrades to before it runs a task."""
    m = re.search(r'%pip install --quiet --upgrade "databricks-sdk>=([\d.]+)"',
                  (REPO_ROOT / "notebooks" / "lgk_refresh_task.py").read_text())
    if not m or "dbutils.library.restartPython()" not in (REPO_ROOT / "notebooks" / "lgk_refresh_task.py").read_text():
        raise AssertionError("lgk_refresh_task.py no longer upgrades databricks-sdk and restarts Python")
    return m.group(1)


def _app_timeout() -> str:
    """The statement timeout the generated app sets on every walk (from the app template)."""
    m = re.search(r"conn\.execute\(\"SET LOCAL statement_timeout = '(\d+)s'\"\)",
                  (REPO_ROOT / "src/lgk/app/template/backend/app/db.py.tmpl").read_text())
    if not m:
        raise AssertionError("the app template sets no SET LOCAL statement_timeout")
    return m.group(1)


def _default(ref: str, param: str) -> Any:
    mod, name = ref.split(":")
    return inspect.signature(getattr(importlib.import_module(mod), name)).parameters[param].default


def _lowest_valid(check: Callable[[int], Any]) -> int:
    """The lowest int in 0..2 that `check` accepts (it raises ValueError below it)."""
    for n in range(0, 3):
        try:
            check(n)
            return n
        except ValueError:
            continue
    raise AssertionError("no low value accepted")


def _workdir_hex() -> str:
    from lgk.notebook_runner import _WORKDIR_RE

    m = re.search(r"lgk_v(\d+)_\[0-9a-f\]\{(\d+)\}", _WORKDIR_RE.pattern)
    if not m:
        raise AssertionError(f"workdir pattern changed: {_WORKDIR_RE.pattern}")
    return m.group(1), m.group(2)


def _version_default() -> str:
    from lgk import versions

    return versions.version_schema({"kit_schema": "k", "graph_name": "g"}).rsplit("_v", 1)[1]


def _bundle_bump() -> int:
    from lgk.export import _bundle_config

    return _bundle_config({"version": 5, "graph_name": "g"})["version"] - 5


def _percentile_rank() -> int:
    """"the time N of 100 runs beat" is p95: the advisor's percentile of 1..100 at 95."""
    from lgk.advisor import percentile

    return int(percentile([float(i) for i in range(1, 101)], 95))


def _lookup_floor() -> int:
    from lgk.enrich.shape import MAX_DEPTH
    from lgk.walks.lookups import ancestors_sql

    return _lowest_valid(lambda n: ancestors_sql("s", n)) if MAX_DEPTH else 0


def _web_floor() -> int:
    from lgk.walks.sql import web_sql

    return _lowest_valid(lambda n: web_sql(n, "s", 25))


def _c(mod: str, name: str) -> Callable[[], Any]:
    """A constant read at check time (so a monkeypatched constant is seen)."""
    return lambda: getattr(importlib.import_module(mod), name)


FLOOR = _c("lgk.privacy.floor", "DEFAULT_FLOOR")
SL = "lgk.sync_limits"


# ===================================================================== number bindings
@dataclass(frozen=True)
class Bind:
    """In `section`, `pattern` matches at least once; each named group equals expect()[group].

    `expect` reads the code (never a retyped number). A group may be checked by a predicate
    instead: expect()[group] is then a callable(value) that raises AssertionError.
    """
    id: str
    section: str
    pattern: str
    expect: Callable[[], Dict[str, Any]]


def _within(lo: Callable[[], int], hi: Callable[[], int]) -> Callable[[str], None]:
    def check(v: str) -> None:
        if not lo() <= int(v) <= hi():
            raise AssertionError(f"example value {v} is outside the code's range {lo()}..{hi()}")
    return check


def _steps_in_order(text: str) -> None:
    from lgk.runner import RunFolder

    found = re.findall(r"^## Step (\d+): `(\w+)`", text, re.M)
    if [int(n) for n, _ in found] != list(range(1, len(RunFolder.STEPS) + 1)):
        raise AssertionError(f"step numbers {[n for n, _ in found]} are not 1..{len(RunFolder.STEPS)}")
    if [s for _, s in found] != list(RunFolder.STEPS):
        raise AssertionError(f"step headings {[s for _, s in found]} != RunFolder.STEPS {RunFolder.STEPS}")


WEB_MAX = _c("lgk.walks.sql", "WEB_MAX_DEPTH")
MEASURE = "Measure the served graph"


def _measure_depths() -> List[int]:
    """The depths measure checks at its default max depth (the served walk SQL it builds per depth)."""
    from lgk.steps.measure import DEFAULT_MAX_DEPTH
    from lgk.walks.sql import per_depth_sql

    return sorted(per_depth_sql("tree", DEFAULT_MAX_DEPTH, served_schema="s", top_n=25))
SHAPE_MAX = _c("lgk.enrich.shape", "MAX_DEPTH")

NUMBERS: Tuple[Bind, ...] = (
    # ---- Terms
    Bind("version default (terms)", "Terms", r"version (?P<v>\d+) is the \"first version of this graph\"",
         lambda: {"v": _version_default()}),
    Bind("privacy floor (terms)", "Terms",
         r"`privacy_floor`: default (?P<f>\d+)\. It can be raised and never\s+lowered \(a value below (?P<g>\d+) is refused\)",
         lambda: {"f": FLOOR(), "g": FLOOR()}),
    Bind("p95 rank (terms)", "Terms", r"the time (?P<n>\d+) of (?P<h>100) runs beat", lambda: {"n": _percentile_rank(), "h": 100}),
    # ---- Interview
    Bind("version default (table)", "Interview: the config", r"\(an existing version is refused\)\. \| `(?P<v>\d+)` \|",
         lambda: {"v": _version_default()}),
    Bind("version default (laptop block)", "Interview: the config", r"\"version\": (?P<v>\d+),", lambda: {"v": _version_default()}),
    Bind("workdir pattern", "Interview: the config", r"/lgk_v(?P<v>\d+)_<(?P<n>\d+) hex chars>",
         lambda: dict(zip(("v", "n"), _workdir_hex()))),
    Bind("wheel name", None, r"(?P<w>lgk-\d+\.\d+\.\d+-py\d-none-any\.whl)", lambda: {"w": _wheel_name()}),
    Bind("environment pin", "Interview: the config",
         r"serverless environment version `(?P<e>\d+)` \(Python (?P<p>\d+\.\d+)\)",
         lambda: {"e": _c("lgk.runtime", "ENVIRONMENT_VERSION")(), "p": _min_python()}),
    Bind("install sdk floor", "Interview: the config", r"pip install dist/lgk-0\.1\.0-py3-none-any\.whl \"databricks-sdk>=(?P<v>[\d.]+)\"",
         lambda: {"v": _runner_sdk_floor()}),
    Bind("workspace install sdk floor", "Interview: the config", r"%pip install <wheel path> \"databricks-sdk>=(?P<v>[\d.]+)\"",
         lambda: {"v": _runner_sdk_floor()}),
    Bind("install python floor", "Interview: the config", r"into a\s+Python (?P<p>\d+\.\d+) or newer",
         lambda: {"p": _min_python()}),
    Bind("python floor", "Interview: the config", r"stop on a Python older than (?P<p>\d+\.\d+)\.", lambda: {"p": _min_python()}),
    # ---- privacy
    Bind("fold floor (privacy)", "Step 4: `privacy`", r"fold\nfloor \(default (?P<f>\d+)\) is left out", lambda: {"f": FLOOR()}),
    Bind("walk keys max", "Step 4: `privacy`", r"may test by equality, at most (?P<n>\d+)\)",
         lambda: {"n": _c("lgk.mapping", "MAX_WALK_KEYS")()}),
    Bind("privacy floor (privacy)", "Step 4: `privacy`", r"the config: default (?P<f>\d+), it can be raised and never lowered",
         lambda: {"f": FLOOR()}),
    # ---- provision
    Bind("sync cap", "Step 7: `provision`",
         r"holds at most (?P<a>\d+) synced tables, .*?pass (?P<b>\d+) minus `sync_per_source_headroom` \(config, default (?P<h>\d+)\)\. The platform's own refusal of the\n(?P<n>\d+)st",
         lambda: {"a": _c(SL, "SYNCED_TABLES_PER_SOURCE")(), "b": _c(SL, "SYNCED_TABLES_PER_SOURCE")(),
                  "h": _c(SL, "DEFAULT_SYNC_PER_SOURCE_HEADROOM")(), "n": _c(SL, "SYNCED_TABLES_PER_SOURCE")() + 1}),
    # ---- advise
    Bind("advisor depth", "Step 8: `advise`", r"`advisor_max_depth` \((?P<d>\d+)\)", lambda: {"d": _c("lgk.steps.advise", "DEFAULT_MAX_DEPTH")()}),
    Bind("p95 rank (advise)", "Step 8: `advise`", r"p95 the time (?P<n>\d+) of (?P<h>100) runs beat", lambda: {"n": _percentile_rank(), "h": 100}),
    # ---- walks
    Bind("walk depth and top", "Step 10: `walks`",
         r"the depth \((?P<lo>\d+) to (?P<hi>\d+); each hop keeps the top (?P<top>\d+)\n",
         lambda: {"lo": _web_floor(), "hi": WEB_MAX(), "top": _default("lgk.steps.walks:get_walk_query", "top_n")}),
    Bind("walk privacy floor", "Step 10: `walks`", r"privacy floor \(default (?P<f>\d+), never", lambda: {"f": FLOOR()}),
    Bind("walk example depth", "Step 10: `walks`", r"get_walk_query\([^)]*?depth=(?P<d>\d+)", lambda: {"d": _within(lambda: 1, WEB_MAX)}),
    Bind("seed caps", "Step 10: `walks`", r"seeds \(default (?P<d>\d+), at most (?P<c>[\d,]+); more is an error",
         lambda: {"d": _c("lgk.walks.sql", "MAX_SEEDS_DEFAULT")(), "c": _c("lgk.walks.sql", "MAX_SEEDS_CAP")()}),
    Bind("lookup range", "Step 10: `walks`", r"levels \((?P<lo>\d+) to (?P<hi>\d+)\), the first `top_n` \(default (?P<top>\d+)\)",
         lambda: {"lo": _lookup_floor(), "hi": SHAPE_MAX(), "top": _default("lgk.walks.lookups:ancestors_sql", "top_n")}),
    Bind("lookup example", "Step 10: `walks`",
         r"up to (?P<q>\d+) levels\?\".*?max_depth=(?P<d>\d+)\)",
         lambda: {"q": _within(lambda: 1, SHAPE_MAX), "d": _within(lambda: 1, SHAPE_MAX)}),
    Bind("same group top", "Step 10: `walks`", r"\(`top_n` default (?P<t>\d+), by node id\)",
         lambda: {"t": _default("lgk.walks.lookups:same_group_sql", "top_n")}),
    Bind("jaccard top", "Step 10: `walks`", r"\(`top_n` default (?P<t>\d+), at most the\n(?P<k>\d+) that enrich keeps per node",
         lambda: {"t": _default("lgk.walks.lookups:similar_jaccard_sql", "top_n"), "k": _c("lgk.enrich.methods", "SIMILARITY_TOP_K")()}),
    # ---- app
    Bind("app walk", "Step 11: `app` (app developers and architects only)",
         r"`GET /api/walk`, default depth (?P<d>\d+), top (?P<t>\d+) per hop",
         lambda: {"d": _c("lgk.app.generate", "DEFAULT_DEPTH")(), "t": _c("lgk.app.generate", "DEFAULT_TOP_N")()}),
    Bind("app seeds", "Step 11: `app` (app developers and architects only)", r"at most (?P<n>\d+) seeds\)",
         lambda: {"n": _c("lgk.walks.sql", "MAX_SEEDS_DEFAULT")()}),
    Bind("app timeout", "Step 11: `app` (app developers and architects only)",
         r"with a (?P<a>\d+) s statement timeout \(`SET LOCAL statement_timeout = '(?P<b>\d+)s'`\)",
         lambda: {"a": _app_timeout(), "b": _app_timeout()}),
    # ---- notebook
    Bind("notebook defaults", "Step 12: `notebook` (data engineers)",
         r"`top_n` \((?P<t>\d+)\), `notebook_max_depth` \((?P<d>\d+)\)",
         lambda: {"t": _c("lgk.walks.notebook", "DEFAULT_TOP_N")(), "d": _c("lgk.walks.notebook", "DEFAULT_MAX_DEPTH")()}),
    # ---- nightly
    Bind("sync parallel and headroom", "Nightly refresh",
         r"`sync_max_parallel`\n\(default (?P<p>\d+)\)\..*?`app_connection_headroom`, default (?P<h>\d+)\)",
         lambda: {"p": _c(SL, "DEFAULT_SYNC_MAX_PARALLEL")(), "h": _c(SL, "DEFAULT_APP_CONNECTION_HEADROOM")()}),
    Bind("per-sync budget", "Nightly refresh",
         r"budgets (?P<f>\d+) connections plus a margin of (?P<m>\d+) per update \(when an earlier night saw more than (?P<g>\d+) sync\nconnections, that high-water mark plus (?P<n>\d+)\)",
         lambda: {"f": _c(SL, "CONNECTIONS_PER_SYNC_FLOOR")(), "m": _c(SL, "PER_SYNC_MARGIN")(),
                  "g": _c(SL, "CONNECTIONS_PER_SYNC_FLOOR")(), "n": _c(SL, "PER_SYNC_MARGIN")()}),
    Bind("reapply synced-version wait", "Nightly refresh",
         r"`reapply`\s+\(it\s+waits\s+up\s+to\s+(?P<s>\d+)\s+s\s+for\s+each\s+synced\s+table\s+to\s+report\s+its\s+synced\s+version",
         lambda: {"s": _c("lgk.refresh.freshness", "SYNCED_VERSION_WAIT_S")()}),
    # ---- measure (stream M)
    Bind("measure seeds", MEASURE, r"`measure_seeds` seeds \(default (?P<s>\d+)\)",
         lambda: {"s": _c("lgk.steps.measure", "DEFAULT_SEEDS")()}),
    Bind("measure samples and warmup", MEASURE,
         r"`measure_samples` runs \(default (?P<n>\d+), at least (?P<m>\d+)\) follow\s+`measure_warmup` runs \(default (?P<w>\d+)\)",
         lambda: {"n": _c("lgk.measure.compare", "DEFAULT_SAMPLES")(), "m": _c("lgk.advisor", "MIN_SAMPLES")(),
                  "w": _c("lgk.advisor", "DEFAULT_WARMUP")()}),
    Bind("measure depths", MEASURE, r"`measure_max_depth` \(default\s+(?P<d>\d+): depths (?P<lo>\d+) to (?P<hi>\d+)\)",
         lambda: {"d": _c("lgk.steps.measure", "DEFAULT_MAX_DEPTH")(), "lo": min(_measure_depths()),
                  "hi": max(_measure_depths())}),
    Bind("measure similar top", MEASURE, r"`measure_similar_top_n` \(default (?P<t>\d+)\)",
         lambda: {"t": _c("lgk.steps.measure", "DEFAULT_SIMILAR_TOP_N")()}),
    Bind("measure reference rows", MEASURE, r"`measure_reference_max_rows`\s+\(default (?P<r>[\d,]+) source rows",
         lambda: {"r": _c("lgk.measure.reference", "DEFAULT_MAX_ROWS")()}),
    # ---- export
    Bind("bundle version bump", "Export: `lgk.export:export(config)` (a bundle that provisions as a service principal)",
         r"\(`version` \+ (?P<n>\d+)\)", lambda: {"n": _bundle_bump()}),
)

# Numbers that are not kit constants. Each entry: (section or None for any, pattern, reason).
# The pattern must match; every number it spans is covered. Keep each pattern a literal.
NUMBER_ALLOW: Tuple[Tuple[Optional[str], str, str], ...] = (
    ("Interview: the config", r"\bpython3 ", "the interpreter command name, not a version"),
    (None, r"\bp50\b", "the percentile name; the advisor report's column is p50 (TestBehaviour)"),
    (None, r"\bp95\b", "the percentile name; the rank is bound by 'the time N of 100 runs beat'"),
    ("Terms", r"1 means the same neighbours, 0 means none in common", "the definition of Jaccard (a ratio in 0..1), not a kit constant"),
    (None, r"\bC(?:10|13|23|30b)\b", "the owner's criterion ids (lgk.steps.measure docstring), not values"),
    ("Nightly refresh", r"\bv1\b", "the kit's release name in the snapshot-only rule, not a value"),
    ("Nightly refresh", r"\"quartz_cron_expression\": \"0 0 2 \* \* \?\"",
     "an example schedule (02:00); the operator gives the real time, and the JSON keys are the Jobs API's"),
    ("Retire an old version: `lgk.steps.retire:retire(config, version, confirm_graph_name)`",
     r"retire\(config, 1, ", "an example old version; the call itself is bound by check_python_blocks"),
    ("Roll back to an earlier version: `lgk.steps.switch:rollback(config, version)`",
     r"rollback\(config, 1\)", "an example earlier version; the call itself is bound by check_python_blocks"),
)


def _structural_number_spans(text: str) -> List[Tuple[int, int]]:
    """Step heading numbers: bound by _steps_in_order (1..N in RunFolder.STEPS order)."""
    _steps_in_order(text)
    return [m.span(1) for m in re.finditer(r"^## Step (\d+): ", text, re.M)]


def _search_area(text: str, section: Optional[str]) -> Tuple[int, int]:
    return (0, len(text)) if section is None else section_span(text, section)


def _check_value(bind_id: str, group: str, got: str, want: Any) -> None:
    if callable(want):
        want(got)
        return
    norm = got.replace(",", "") if re.fullmatch(r"\d{1,3}(,\d{3})+", got) else got
    if norm != str(want):
        raise AssertionError(f"[{bind_id}] SKILL.md says {got!r} for {group}; the code says {want!r}")


def check_numbers(text: str, binds: Tuple[Bind, ...] = NUMBERS) -> List[Tuple[int, int]]:
    """Every NUMBER binding matches and agrees with the code; returns the covered group spans."""
    if not binds:
        raise AssertionError("no number bindings: an empty list never passes")
    spans = []
    for b in binds:
        s, e = _search_area(text, b.section)
        hits = list(re.finditer(b.pattern, text[s:e], re.S))
        if not hits:
            raise AssertionError(f"[{b.id}] the bound sentence is gone or reworded in {b.section!r}: /{b.pattern}/")
        want = b.expect()
        for m in hits:
            for g, v in m.groupdict().items():
                _check_value(b.id, g, v, want[g])
                spans.append((s + m.start(g), s + m.end(g)))
    return spans


def _allow_spans(text: str, allow) -> List[Tuple[int, int]]:
    spans = []
    for section, pattern, reason in allow:
        if not reason.strip():
            raise AssertionError(f"allowlist entry {pattern!r} has no reason")
        s, e = _search_area(text, section)
        hits = list(re.finditer(pattern, text[s:e]))
        if not hits:
            raise AssertionError(f"allowlist entry {pattern!r} matches nothing in {section!r}: remove it")
        spans += [(s + m.start(), s + m.end()) for m in hits]
    return spans


def check_number_coverage(text: str) -> None:
    """Every digit run in SKILL.md is a bound group, a step heading number, or allowlisted."""
    covered = check_numbers(text) + _structural_number_spans(text) + _allow_spans(text, NUMBER_ALLOW)
    unbound = [(text.count("\n", 0, m.start()) + 1, m.group()) for m in NUMBER_RE.finditer(text)
               if not any(s <= m.start() and m.end() <= e for s, e in covered)]
    if unbound:
        raise AssertionError(f"numbers in SKILL.md with no binding (line, number): {unbound}. "
                             "Bind each to its code constant in NUMBERS, or allowlist it with a reason.")


# ======================================================================== status words
STATUS_VOCAB = re.compile(r"\b(success|succeeded|successful|succeed|ok|passed|pass|blocked|failed|failure|fail|"
                          r"error|errored|not_run|in_progress|complete|completed|finished|done)\b", re.I)


def _run(attr: str) -> Callable[[], str]:
    return lambda: getattr(importlib.import_module("lgk.runner").RunStatus, attr).value


def _pf(attr: str) -> Callable[[], str]:
    return lambda: getattr(importlib.import_module("lgk.steps.preflight"), attr)


NEWV = "Serve a new version"
STATUS: Tuple[Bind, ...] = (
    Bind("skip rule", "HEAD", r"A step whose status is `?(?P<s>\w+)`? is skipped", lambda: {"s": _run("SUCCESS")()}),
    Bind("failed rule", "HEAD", r"If `status` is `?(?P<s>\w+)`?, show the `error`", lambda: {"s": _run("FAILED")()}),
    Bind("preflight blocked", "Step 1: `preflight`", r"If `status` is `?(?P<s>\w+)`?, show each entry", lambda: {"s": _pf("BLOCKED")()}),
    Bind("preflight done", "Step 1: `preflight`",
         r"`preflight_result\.json` is not `?(?P<b>\w+)`? and the step's status is `?(?P<s>\w+)`?\.",
         lambda: {"b": _pf("BLOCKED")(), "s": _run("SUCCESS")()}),
    Bind("sync owner blocked", "Interview: the config", r"preflight is `?(?P<b>\w+)`? and provision refuses",
         lambda: {"b": _pf("BLOCKED")()}),
    Bind("new version done", NEWV, r"\*\*Done when\*\* each of those steps shows status `?(?P<s>\w+)`? and",
         lambda: {"s": _run("SUCCESS")()}),
    Bind("build done", "Step 5: `build`", r"\*\*Done when\*\* the step's status is `?(?P<s>\w+)`? and", lambda: {"s": _run("SUCCESS")()}),
    Bind("provision done", "Step 7: `provision`", r"\*\*Done when\*\* the step's status is `?(?P<s>\w+)`? \(its `reader` proof",
         lambda: {"s": _run("SUCCESS")()}),
    Bind("teardown done", "Step 13: `teardown`", r"`run\.json` shows `teardown` with status `?(?P<s>\w+)`?\.",
         lambda: {"s": _run("SUCCESS")()}),
    Bind("completion status", "Completion", r"Status: (?P<a>DONE) \| (?P<b>NOT DONE)",
         lambda: {"a": "DONE", "b": "NOT DONE"}),
)
# Status-vocabulary words that are prose, not a status value. (section, pattern, reason)
STATUS_ALLOW: Tuple[Tuple[Optional[str], str, str], ...] = (
    (None, r"\*\*Done when\*\*", "the step-completion marker of this skill, not a status value"),
    (None, r"\bNOT DONE\b", "the Completion marker (bound in STATUS 'completion status')"),
    (None, r"\bevery line printed is DONE\b", "the Completion marker"),
    (None, r"\bis done\b", "prose about the run, governed by the Completion invariant"),
    (None, r"\brun is finished\b", "prose: the operator's run, not a status"),
    ("HEAD", r"show the `error`", "the RunResult field `error` (bound: TestBehaviour test_run_result_fields)"),
    ("Step 7: `provision`", r"proof passed", "prose about the reader proof; the status word after it is bound"),
)


def check_status_words(text: str, binds: Tuple[Bind, ...] = STATUS) -> None:
    """Every status word that is backticked, or within 60 characters after 'status', is bound."""
    covered = check_numbers(text, binds) + _allow_spans(text, STATUS_ALLOW)
    suspects = []
    for m in TICK_RE.finditer(text):
        if STATUS_VOCAB.fullmatch(m.group(1)):
            suspects.append((m.start(1), m.end(1), m.group(1)))
    for m in re.finditer(r"\bstatus(?:es)?\b", text, re.I):
        for w in STATUS_VOCAB.finditer(text, m.end(), min(len(text), m.end() + 60)):
            suspects.append((w.start(), w.end(), w.group()))
    unbound = sorted({(text.count("\n", 0, s) + 1, w) for s, e, w in suspects
                      if not any(cs <= s and e <= ce for cs, ce in covered)})
    if unbound:
        raise AssertionError(f"status words in SKILL.md with no binding (line, word): {unbound}")


# ====================================================================== python and calls
def _resolve(ref: str) -> Any:
    from lgk.notebook_runner import resolve_step

    try:
        return resolve_step(ref)
    except (ImportError, AttributeError, ValueError) as e:
        raise AssertionError(f"resolve_step({ref!r}) does not resolve in the code: {e}")


# Literal arguments that must be one of the code's values: (callable qualname, parameter) -> values.
def _domains() -> Dict[Tuple[str, str], Any]:
    from lgk.enrich.menu import V1_METHODS
    from lgk.keys import CHOICES as KEY_CHOICES
    from lgk.privacy.classify import CLASSES
    from lgk.privacy.flags import CHOICES
    from lgk.steps.walks import LOOKUP_KINDS
    from lgk.walks.sql import DIRECTIONS

    from lgk.steps.enrich import TABLE_OF

    scored = tuple(m for m, table in TABLE_OF.items() if table == "node_score")  # read via node_score_view
    assert set(scored) <= set(V1_METHODS)
    return {("get_walk_query", "direction"): DIRECTIONS, ("get_walk_query", "sort_method"): scored,
            ("get_lookup_query", "kind"): LOOKUP_KINDS, ("resolve_flag", "choice"): CHOICES,
            ("resolve_flag", "new_class"): CLASSES, ("resolve_orphans", "choice"): KEY_CHOICES}


# Calls a ```python block may make that are not kit code. name -> reason
EXTERNAL_CALLS = {"print": "builtin", "Path": "pathlib", "WorkspaceClient": "databricks-sdk client"}
# Attribute calls on plain values. (object name, method) -> reason
VALUE_METHODS = {("config", "pop"): "dict", ("config", "items"): "dict"}


def _expand_ellipsis(sig: inspect.Signature, args: List[ast.expr], kwargs: List[ast.keyword]) -> List[ast.expr]:
    """An inline `...` argument stands for the required positional arguments it hides."""
    dots = [i for i, a in enumerate(args) if isinstance(a, ast.Constant) and a.value is Ellipsis]
    if not dots:
        return args
    if len(dots) > 1:
        raise AssertionError("more than one `...` in one call")
    keyed = {k.arg for k in kwargs}
    required = [p for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD
                and p.default is inspect.Parameter.empty and p.name not in keyed]
    hidden = max(0, len(required) - (len(args) - 1))
    i = dots[0]
    return args[:i] + [ast.Constant(None)] * hidden + args[i + 1:]


def _bind(fn: Any, name: str, args: List[ast.expr], kwargs: List[ast.keyword], where: str) -> None:
    sig = inspect.signature(fn)
    args = _expand_ellipsis(sig, args, kwargs)
    placeholder = object()
    try:
        bound = sig.bind(*[placeholder for _ in args], **{k.arg: placeholder for k in kwargs if k.arg})
    except TypeError as e:
        raise AssertionError(f"{where}: {name}(...) does not fit the code's signature {sig}: {e}")
    values = dict(zip(bound.arguments, args))
    values.update({k.arg: k.value for k in kwargs if k.arg})
    for param, node in values.items():
        if param not in sig.parameters or sig.parameters[param].kind in (inspect.Parameter.VAR_KEYWORD,
                                                                          inspect.Parameter.VAR_POSITIONAL):
            continue
        ann = str(sig.parameters[param].annotation)
        listy = bool(re.search(r"\b(?:List|list|Sequence|Iterable)\b", ann)) and not ann.startswith("typing.Union[str,")
        if isinstance(node, (ast.List, ast.Tuple)) and ann != str(inspect.Parameter.empty) and not listy:
            raise AssertionError(f"{where}: {name}: a list is given for {param} ({ann})")
        if listy and not isinstance(node, (ast.List, ast.Tuple, ast.Starred)) and \
                isinstance(node, (ast.Name, ast.Constant)) and getattr(node, "value", 0) is not None:
            raise AssertionError(f"{where}: {name}: {param} takes a list ({ann}), the skill gives one value")
    for (fname, param), domain in _domains().items():
        node = values.get(param)
        if fname == name and isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in domain:
                raise AssertionError(f"{where}: {name}({param}={node.value!r}) is not one of the code's {tuple(domain)}")


class _Env(dict):
    """name -> (qualname, callable) for the names a python block binds to kit code."""


def _block_env(tree: ast.AST, env: _Env) -> None:
    from lgk.runner import RunResult

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("lgk"):
            mod = importlib.import_module(node.module)
            for a in node.names:
                if not hasattr(mod, a.name):
                    raise AssertionError(f"from {node.module} import {a.name}: no such name in the code")
                env[a.asname or a.name] = (a.name, getattr(mod, a.name))
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            v, target = node.value, node.targets[0].id
            if (isinstance(v, ast.Call) and getattr(v.func, "id", None) == "resolve_step"
                    and isinstance(v.args[0], ast.Constant)):
                obj = _resolve(v.args[0].value)
                env[target] = (v.args[0].value.rsplit(":", 1)[1].split(".")[-1], obj)
            elif isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in env:
                cls = env[v.func.id][1]
                if inspect.isclass(cls):
                    env[target] = (target, cls)  # an instance: its methods are checked
    env.setdefault("r", ("r", RunResult))  # `r` in `[r.to_dict() for r in results]` is a RunResult


def _check_call(node: ast.Call, env: _Env, where: str) -> None:
    f = node.func
    if isinstance(f, ast.Name):
        if f.id == "resolve_step":
            if isinstance(node.args[0], ast.Constant) and "<" not in node.args[0].value:
                _resolve(node.args[0].value)
            return
        if f.id in env:
            name, obj = env[f.id]
            _bind(obj, name, node.args, node.keywords, where)
            return
        if f.id in EXTERNAL_CALLS:
            return
        raise AssertionError(f"{where}: call to {f.id}(...) is not bound to kit code (no import or resolve_step)")
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        owner = f.value.id
        if owner in env:
            cls = env[owner][1]
            if not hasattr(cls, f.attr):
                raise AssertionError(f"{where}: {owner}.{f.attr}(...): {getattr(cls, '__name__', cls)} has no {f.attr}")
            _bind(getattr(cls, f.attr), f.attr, [ast.Constant(None)] + node.args if inspect.isclass(cls) else node.args,
                  node.keywords, where)
            return
        if (owner, f.attr) in VALUE_METHODS:
            return
        raise AssertionError(f"{where}: {owner}.{f.attr}(...) is not bound to kit code")
    if isinstance(f, ast.Call) and getattr(f.func, "id", None) == "resolve_step":
        ref = f.args[0].value
        _bind(_resolve(ref), ref.rsplit(":", 1)[1], node.args, node.keywords, where)
        return
    raise AssertionError(f"{where}: unbound call form {ast.dump(f)[:80]}")


def _config_keys(tree: ast.AST) -> List[str]:
    keys = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "config" for t in node.targets) \
                and isinstance(node.value, ast.Dict):
            keys += [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    return keys


def check_python_blocks(text: str) -> _Env:
    """Every call in every ```python block binds to the code; config keys are keys the code reads."""
    env = _Env()
    blocks = [m.group(2) for m in FENCE_RE.finditer(text) if m.group(1) == "python"]
    if len(blocks) < 10:
        raise AssertionError(f"only {len(blocks)} python blocks: the binding has nothing to check")
    for i, code in enumerate(blocks):
        tree = ast.parse(code)
        _block_env(tree, env)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                _check_call(node, env, f"python block {i + 1}")
        for key in _config_keys(tree):
            if key not in config_keys():
                raise AssertionError(f"python block {i + 1}: config key {key!r} is not read by the code")
    return env


# ---------------------------------------------------------------- inline code tokens
_CODE_NAMES: Optional[set] = None


def code_names() -> set:
    """Every identifier and every string constant in the kit's code (src/lgk, notebooks), plus
    the words of the app templates and the notebook file stems."""
    global _CODE_NAMES
    if _CODE_NAMES is None:
        names: set = set()
        for p in list((REPO_ROOT / "src" / "lgk").rglob("*.py")) + list((REPO_ROOT / "notebooks").glob("*.py")):
            for node in ast.walk(ast.parse(p.read_text())):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    names.add(node.value)
                    # a quoted name inside generated code (the walks notebook's widget names)
                    names |= set(re.findall(r"[\"'](\w+)[\"']", node.value))
                    names |= set(re.findall(r"(?:\$|`)(\w+)", node.value))  # $ENV_VAR, `name` in generated text
                elif isinstance(node, ast.Name):
                    names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    names.add(node.attr)
                elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    names.add(node.name)
                elif isinstance(node, ast.arg):
                    names.add(node.arg)
                elif isinstance(node, ast.keyword) and node.arg:
                    names.add(node.arg)
            names.add(p.stem)
        for p in (REPO_ROOT / "src" / "lgk").rglob("*.tmpl"):
            names |= set(re.findall(r"[A-Za-z_][\w.-]*", p.read_text()))
        _CODE_NAMES = names
    return _CODE_NAMES


_CODE_STRINGS: Optional[set] = None


def code_strings() -> set:
    """Every string constant in the kit's Python (src/lgk, notebooks): config keys are read by them."""
    global _CODE_STRINGS
    if _CODE_STRINGS is None:
        _CODE_STRINGS = {n.value for p in list((REPO_ROOT / "src" / "lgk").rglob("*.py"))
                         + list((REPO_ROOT / "notebooks").glob("*.py"))
                         for n in ast.walk(ast.parse(p.read_text()))
                         if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    return _CODE_STRINGS


def _in_code(word: str) -> bool:
    return word in code_names()


IDENT_RE = re.compile(r"[A-Za-z_]\w*")
DOTTED_RE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
FILE_RE = re.compile(r"(?:<\w+>/|dist/|notebooks/)?(?:[\w-]+/)*(?P<f>[\w.*-]+\.(?:json|yaml|md|py|whl))")
CALL_RE = re.compile(r"(?P<name>[\w.:]+)\((?P<args>.*?)\)(?:\((?P<args2>.*)\))?")
KW_RE = re.compile(r"(?P<k>\w+)=(?P<v>.+)")
YAML_RE = re.compile(r"(?P<k>\w+): (?P<v>.+)")
QUOTED_RE = re.compile(r"\"(?P<v>\w+)\"")
CLI_RE = re.compile(r"(?:databricks|uv) .+")
FLAG_RE = re.compile(r"--[\w-]+")
ROUTE_RE = re.compile(r"GET (?P<route>/api/[\w-]+)(?:\?(?P<q>.*))?")

# Inline calls that are SQL, not kit Python: name -> reason
SQL_CALLS = {"databricks_create_role": "the Lakebase Postgres function that makes an identity's role"}
# Code names that are not the kit's: name -> reason
NAME_ALLOW = {
    "pg_views": "the Postgres catalog view the Done-when query reads",
    "lambda": "the Python keyword of the teardown runner call",
    "providers": "an example graph name the operator may give",
    "settings": "the Jobs API job object field `settings` of `databricks jobs get`",
    "schedule": "the Jobs API field settings.schedule",
    "quartz_cron_expression": "the Jobs API schedule field (the operator's cron)",
    "timezone_id": "the Jobs API schedule field",
    "pause_status": "the Jobs API schedule field the rollback recipe sets to PAUSED",
    "metadata": "the run manifest entry field (lgk.manifest.ManifestEntry.metadata, an attribute)",
}
# Backticked text that is not a code name, call, command, number or status. (regex, reason)
TOKEN_ALLOW: Tuple[Tuple[str, str], ...] = (
    (r"<[^>]+>(?:\.<[^>]+>|_v<\w+>)?", "a placeholder the operator fills in"),
    (r"<[^>]+>\.nodes_view|SELECT count\(\*\) FROM <served_schema>\.nodes_view", "SQL over the served view (bound: SERVED_VIEWS)"),
    (r"(?:CREATE ROLE|CREATE TABLE|USE CATALOG|USE SCHEMA|GRANT CONNECT|GRANT reader)", "a SQL privilege/statement word"),
    (r"CONSTRAINT \.\.\. PRIMARY KEY|FOREIGN KEY \.\.\. REFERENCES", "Unity Catalog constraint SQL the operator declares"),
    (r"HAVING COUNT\(DISTINCT (?:\.\.\.|<subject>)\) >= <floor>", "the fold SQL, bound by test_skill.py test_fold_sql_leaves_sub_floor_pairs_out_with_having"),
    (r"SET LOCAL statement_timeout = '\d+s'", "the app's SQL, bound by NUMBERS 'app timeout'"),
    (r"\{stamp, table_id, certified_version\}", "the label fields (bound: TestBehaviour label fields)"),
    (r"\{\"tables\": \[\{\"name\": \"providers\"\}\]\}", "an example gold_tables.json (bound: TestBehaviour gold tables)"),
    (r"\[outbound\]", "the advise default directions (bound: TestBehaviour advise defaults)"),
    (r"advisor skipped: <reason>", "the advisor report text (bound: TestBehaviour advisor skipped)"),
    (r"edge_types is required(?: \.\.\.)?", "the error text (bound: test_skill.py test_advise_on_a_mixed_graph_fails_with_edge_types_required)"),
    (r"\$\{workspace\.root_path\}/state/run", "the bundle state path (bound: TestBehaviour bundle state path)"),
    (r"/Workspace/(?:Users/<operator>/(?:lgk-apps/<name>|lgk/runs/<graph_name>|lgk_v1_<6 hex chars>))?",
     "workspace paths (bound: NUMBERS workdir, test_skill.py app path, TestBehaviour nightly folder)"),
    (r"\./runs/<graph_name>", "the runner's default run folder (bound: TestBehaviour run folder default)"),
    (r"<notebook folder>/runs/<graph_name>", "the nightly copy default (bound: TestBehaviour nightly folder)"),
    (r"<run_folder>/app/<name>", "the app folder default (bound in test_skill.py feature claims)"),
    (r"lgk_<graph_name>", "the kit schema default (bound: TestBehaviour kit schema default)"),
    (r"<kit_schema>_v<(?:N|version)>", "the version schema name (bound: TestBehaviour version schema)"),
    (r"event_log_<pipeline id>", "the event-log name (bound: TestBehaviour event log name)"),
    (r"module:qualname", "the resolve_step reference form (bound: check_python_blocks resolve)"),
    (r"databricks-sdk", "the package the refresh task upgrades, then restarts Python (bound: TestBehaviour test_refresh_task_upgrades_the_sdk_first)"),
    (r"pip install", "the pip command the runner notebook runs (bound: TestBehaviour runner pip)"),
    (r"dist/(?:lgk-\*\.whl)?", "the uv build output folder"),
    (r"methods: \[(?:\.\.\.)?\]", "the enrich.yaml form (bound: TestBehaviour enrich methods)"),
    (r"subject: <its folded care-subject column>", "the aggregate join key (bound: lgk.mapping 'subject')"),
    (r"(?:\. At most | and |; the values are bound at run time as )", "prose between two code spans on one line"),
    (r"1|3", "a backticked number (bound by NUMBERS)"),
    (r"--source-code-path", "the flag of the printed deploy command (bound: TestBehaviour deploy command)"),
    (r"(?:<run_folder>/)?measure\.jsonl", "the measure record (lgk.measure.record.JSONL; bound: TestBehaviour measure files)"),
    (r"'(?:user|service_principal)'", "the databricks_create_role kinds (bound: CLAIMS 'owner setup kinds')"),
)


def _call_target(name: str, env: _Env) -> Optional[Tuple[str, Any]]:
    if ":" in name:
        return name.rsplit(":", 1)[1], _resolve(name)
    head, _, attr = name.partition(".")
    owner = None
    if head in env:
        owner = env[head][1]
        if not attr:
            return env[head]
    elif head == "backend":
        from lgk.notebook_runner import DatabricksNotebookBackend as owner
    elif head == "result":
        from lgk.runner import RunResult as owner
    if owner is None or not attr:
        return None
    if not hasattr(owner, attr):
        raise AssertionError(f"inline call {name}(...): {getattr(owner, '__name__', owner)} has no {attr}")
    fn = getattr(owner, attr)
    if inspect.isclass(owner):  # an instance method: bind self too
        fn = fn.__get__(object()) if inspect.isfunction(fn) else fn
    return attr, fn


def _parse_args(src: str) -> ast.Call:
    src = re.sub(r"<[^>]+>", "_P", src)
    return ast.parse(f"f({src})", mode="eval").body


def _check_inline_call(tok: str, env: _Env) -> bool:
    m = CALL_RE.fullmatch(tok)
    if not m or not re.fullmatch(r"[\w.:]+", m.group("name")):
        return False
    name = m.group("name")
    if name in SQL_CALLS:
        return True
    if name == "resolve_step" and m.group("args2") is not None:
        ref = ast.literal_eval(m.group("args"))
        target = (ref.rsplit(":", 1)[1], _resolve(ref))
        call = _parse_args(m.group("args2"))
    else:
        target = _call_target(name, env)
        if target is None:
            raise AssertionError(f"inline call `{tok}`: {name} is not bound to kit code")
        call = _parse_args(m.group("args"))
    if name.startswith("runner.") and "<step>" in tok:
        return True  # the run_steps form; its pieces are checked in the python blocks
    fn = target[1]
    if ":" in name and all(isinstance(a, ast.Name) for a in call.args) and not call.keywords:
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.default is inspect.Parameter.empty and p.kind == p.POSITIONAL_OR_KEYWORD]
        stated = [a.id for a in call.args]
        if stated != [p.name for p in params]:
            raise AssertionError(f"`{tok}` states parameters {stated}; the code's required parameters are "
                                 f"{[p.name for p in params]}")
        return True
    _bind(fn, target[0], call.args, call.keywords, f"inline `{tok}`")
    return True


def _kw_owner(env: _Env, key: str) -> List[str]:
    return [n for n, (q, obj) in env.items() if callable(obj) and not inspect.isclass(obj)
            and key in inspect.signature(obj).parameters]


def _check_kw(tok: str, env: _Env) -> bool:
    m = KW_RE.fullmatch(tok)
    if not m:
        return False
    owners = _kw_owner(env, m.group("k"))
    if not owners:
        raise AssertionError(f"`{tok}`: no kit function the skill calls takes {m.group('k')!r}")
    v = m.group("v")
    if re.fullmatch(r"\"\w+\"", v):
        for (fname, param), domain in _domains().items():
            if param == m.group("k") and fname in {env[o][0] for o in owners} and v.strip('"') not in domain:
                raise AssertionError(f"`{tok}`: {v} is not one of the code's {tuple(domain)}")
    return True


def _yaml_domains() -> Dict[str, Any]:
    from lgk.enrich.menu import V1_METHODS
    from lgk.refresh.job import BUILD_MODES
    from lgk.refresh.modes import SYNC_MODES

    return {"build_mode": BUILD_MODES, "sync_mode": SYNC_MODES, "serve_closure": ("true",),
            "methods": None, "subject": None}


def _check_yaml(tok: str) -> bool:
    m = YAML_RE.fullmatch(tok)
    if not m or not _in_code(m.group("k")):
        return False
    from lgk.enrich.menu import V1_METHODS

    k, v = m.group("k"), m.group("v").strip('"')
    domain = _yaml_domains().get(k, ())
    if k == "methods":
        listed = [x.strip() for x in v.strip("[]").split(",") if x.strip() and x.strip() != "..."]
        bad = [x for x in listed if x not in V1_METHODS]
        if bad:
            raise AssertionError(f"`{tok}`: {bad} not in the code's methods {V1_METHODS}")
    elif domain and v not in domain:
        raise AssertionError(f"`{tok}`: {v!r} is not one of the code's {tuple(domain)}")
    return True


def _check_quoted(tok: str) -> bool:
    m = QUOTED_RE.fullmatch(tok)
    if not m:
        return False
    from lgk.refresh.job import BUILD_MODES
    from lgk.refresh.modes import SYNC_MODES

    allowed = set(BUILD_MODES) | set(SYNC_MODES)
    for d in _domains().values():
        allowed |= set(d)
    if m.group("v") not in allowed:
        raise AssertionError(f"`{tok}` is not a value the code accepts anywhere the skill uses it")
    return True


def _check_route(tok: str) -> bool:
    m = ROUTE_RE.fullmatch(tok)
    if not m:
        return False
    main = (REPO_ROOT / "src/lgk/app/template/backend/app/main.py.tmpl").read_text()
    if f'@app.get("{m.group("route")}")' not in main:
        raise AssertionError(f"`{tok}`: the app template serves no {m.group('route')}")
    for q in re.findall(r"(\w+)=", m.group("q") or ""):
        if not re.search(rf"\b{q}\b", main):
            raise AssertionError(f"`{tok}`: the app route takes no {q!r}")
    return True


def _check_name(tok: str) -> bool:
    if IDENT_RE.fullmatch(tok) or DOTTED_RE.fullmatch(tok):
        parts = tok.split(".") if DOTTED_RE.fullmatch(tok) and not _in_code(tok) else [tok]
        bad = [p for p in parts if not _in_code(p) and p not in NAME_ALLOW]
        if bad:
            raise AssertionError(f"`{tok}`: {bad} is not a name or string in the kit's code")
        return True
    return False


def _check_file(tok: str) -> bool:
    m = FILE_RE.fullmatch(tok)
    if not m:
        return False
    f = m.group("f")
    if "*" in f or f == _wheel_name():
        return True  # the wheel name is bound by NUMBERS 'wheel name'
    if tok.startswith(("notebooks/", "dist/")):
        if tok.startswith("notebooks/") and not (REPO_ROOT / tok).is_file():
            raise AssertionError(f"`{tok}`: no such file in the toolkit repo")
        return True
    if not any(isinstance(s, str) and (s == f or s.endswith("/" + f)) for s in code_names()) \
            and not list((REPO_ROOT / "src" / "lgk").rglob(f + "*")):
        raise AssertionError(f"`{tok}`: the code names no file {f!r}")
    return True


def _inline_tokens(text: str) -> List[Tuple[int, str]]:
    """(line, token) for every backticked span outside fenced blocks."""
    fences = [(f.start(), f.end()) for f in FENCE_RE.finditer(text)]
    return [(text.count("\n", 0, m.start()) + 1, m.group(1)) for m in TICK_RE.finditer(text)
            if not any(s <= m.start() < e for s, e in fences)]


def check_inline_code(text: str) -> None:
    """Every backticked token is claimed by a binding check, or by TOKEN_ALLOW with a reason."""
    env = check_python_blocks(text)
    unclaimed = []
    for line, tok in _inline_tokens(text):
        if CLI_RE.fullmatch(tok):
            check_cli_command(tok, partial=True)
            continue
        if STATUS_VOCAB.fullmatch(tok):
            continue  # bound by check_status_words
        claimed = (_check_route(tok) or _check_inline_call(tok, env) or _check_kw(tok, env) or _check_yaml(tok)
                   or _check_quoted(tok) or _check_file(tok) or _check_name(tok)
                   or any(re.fullmatch(p, tok) for p, _ in TOKEN_ALLOW))
        if not claimed:
            unclaimed.append((line, tok))
    if unclaimed:
        raise AssertionError(f"backticked text with no binding (line, token): {unclaimed}")


# ================================================================================== CLI
# Databricks CLI v1.14.1 (`databricks <cmd> --help`, read 2026-10-04): (positional args, flags that
# take a value -> allowed values or None for free text, switch flags). TestCliSurface re-reads the
# installed CLI when there is one.
CLI_SURFACE: Dict[Tuple[str, ...], Tuple[int, Dict[str, Optional[set]], set]] = {
    ("databricks", "workspace", "list"): (1, {}, set()),
    ("databricks", "workspace", "mkdirs"): (1, {}, set()),
    ("databricks", "workspace", "import-dir"): (2, {}, {"--overwrite"}),
    ("databricks", "workspace", "import"): (1, {"--file": None, "--format": {"AUTO", "DBC", "HTML", "JUPYTER", "RAW",
                                                                               "R_MARKDOWN", "SOURCE"},
                                                "--language": {"PYTHON", "R", "SCALA", "SQL"}}, {"--overwrite"}),
    ("databricks", "workspace", "delete"): (1, {}, {"--recursive"}),
    ("databricks", "jobs", "update"): (0, {"--json": None}, set()),
    ("databricks", "jobs", "get"): (1, {"-o": {"json", "text"}}, set()),
    ("databricks", "jobs", "delete"): (1, {}, set()),
    ("databricks", "apps", "deploy"): (1, {"--source-code-path": None}, set()),
    ("databricks", "bundle", "deploy"): (0, {}, set()),
    ("uv", "build"): (0, {}, {"--wheel"}),
    ("python3", "-m", "pip", "install"): (2, {}, set()),  # the wheel and the laptop SDK
    ("python3", "-c"): (1, {}, set()),
}


def check_cli_command(cmd: str, partial: bool = False) -> None:
    """One command line: a known subcommand, the right positional count, known flags and values.

    `partial` (an inline mention such as `databricks apps deploy ...`) skips the count of
    positionals and allows a trailing `...`.
    """
    words = shlex.split(re.sub(r"<[^>]+>", "PLACEHOLDER", cmd))
    key = next((k for k in sorted(CLI_SURFACE, key=len, reverse=True) if tuple(words[:len(k)]) == k), None)
    if key is None:
        raise AssertionError(f"`{cmd}`: not a command of the CLI surface the skill may use")
    npos, valued, switches = CLI_SURFACE[key]
    rest, pos = words[len(key):], 0
    i = 0
    while i < len(rest):
        w = rest[i]
        if w in valued:
            if i + 1 >= len(rest):
                raise AssertionError(f"`{cmd}`: {w} needs a value")
            allowed = valued[w]
            if allowed is not None and rest[i + 1] not in allowed:
                raise AssertionError(f"`{cmd}`: {w} {rest[i + 1]} is not one of {sorted(allowed)}")
            i += 2
            continue
        if w.startswith("-"):
            if w not in switches:
                raise AssertionError(f"`{cmd}`: {key[-1]} has no flag {w}")
        elif not (partial and w == "..."):
            pos += 1
        i += 1
    if not partial and pos != npos:
        raise AssertionError(f"`{cmd}`: {' '.join(key)} takes {npos} positional argument(s), got {pos}")


def check_cli(text: str) -> None:
    """Every line of every ```bash block is a CLI_SURFACE command."""
    lines = [ln for m in FENCE_RE.finditer(text) if m.group(1) == "bash" for ln in m.group(2).splitlines() if ln.strip()]
    if len(lines) < 8:
        raise AssertionError(f"only {len(lines)} bash lines: the binding has nothing to check")
    for ln in lines:
        check_cli_command(ln)


def check_cli_paths(text: str) -> None:
    """The paths the bash blocks upload are the files the toolkit has and the code then uses."""
    body = _search_area(text, "Nightly refresh")
    sec = text[body[0]:body[1]]
    wheel = _wheel_name()
    for need in (f"--file dist/{wheel}", "--file notebooks/lgk_refresh_task.py",
                 f"/Workspace/Users/<operator>/lgk/{Path('notebooks/lgk_refresh_task.py').stem} --file",
                 f'notebook_path="/Workspace/Users/<operator>/lgk/{Path("notebooks/lgk_refresh_task.py").stem}"',
                 f'wheel_path="/Workspace/Users/<operator>/lgk/{wheel}"'):
        if need not in sec:
            raise AssertionError(f"Nightly refresh: the upload and the job no longer agree on {need!r}")
    if not (REPO_ROOT / "notebooks" / "lgk_refresh_task.py").is_file():
        raise AssertionError("notebooks/lgk_refresh_task.py is gone")


# ================================================================= claims (prover round 1)
# Sentences whose MEANING the code decides: an error name, a verdict (refused / succeeds /
# passes), a default, which kind does what, an order. Each pattern is the bound sentence; each
# group is compared with what the code DOES (a call that raises, a probe of a default), so a
# reworded or swapped claim fails even after the snapshot is regenerated.
def _raises(fn: Callable[[], Any]) -> str:
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - the type is the answer
        return type(e).__name__
    raise AssertionError("the probe raised nothing")


def _built_in(module: str, cls: str) -> str:
    """`cls` (imported from the code) is constructed in `module`'s source: that module raises it."""
    obj = None
    for m in ("lgk.privacy.gate", "lgk.privacy.floor", "lgk.steps.build", "lgk.steps.provision", "lgk.mapping",
              "lgk.privacy.binding", "lgk.refresh.label", "lgk.steps.preflight", "lgk.event_logs", "lgk.steps.enrich",
              "lgk.keys"):
        obj = obj or getattr(importlib.import_module(m), cls, None)
    obj = obj or (__builtins__[cls] if isinstance(__builtins__, dict) else getattr(__builtins__, cls, None))
    if obj is None or not inspect.isclass(obj) or not issubclass(obj, BaseException):
        raise AssertionError(f"{cls} is not an exception class of the code")
    if f"{cls}(" not in inspect.getsource(importlib.import_module(module)):
        raise AssertionError(f"{module} never raises {cls}")
    return cls


def _advise_skip_verdict() -> str:
    """'succeeds' when the InsufficientPrivilege handler of _advise_or_skip returns, never raises."""
    tree = ast.parse(inspect.getsource(importlib.import_module("lgk.steps.advise")._advise_or_skip).lstrip())
    hs = [h for h in ast.walk(tree) if isinstance(h, ast.ExceptHandler) and "InsufficientPrivilege" in ast.unparse(h.type)]
    if len(hs) != 1 or any(isinstance(n, ast.Raise) for n in ast.walk(hs[0])) or \
            not any(isinstance(n, ast.Return) for n in ast.walk(hs[0])):
        raise AssertionError("advise no longer returns a skipped report on InsufficientPrivilege")
    return "succeeds"


def _open_flag_verdict() -> str:
    from lgk.privacy import gate
    from lgk.privacy.flags import new_flag

    return "FAILS" if _raises(lambda: gate.check_flags({"flags": [new_flag("t", "patient_id", "name", True)]})) else "?"


def _reclassify_clears() -> Tuple[str, str]:
    """(the one reclassify class that clears a care-subject column under the care-key check, a class that does not)."""
    from lgk.privacy import classify, gate

    def cleared(new_class: str) -> bool:
        data = {"columns": [{"table": "providers", "column": "full_name", "class": classify.CARE_SUBJECT}],
                "flags": [{"table": "providers", "column": "full_name", "choice": "reclassify",
                           "new_class": new_class}]}
        return not gate._care_sets(data)[0]

    clears = [c for c in (classify.PUBLIC_PROFESSIONAL, classify.NONE) if cleared(c)]
    keeps = [c for c in (classify.PUBLIC_PROFESSIONAL, classify.NONE) if not cleared(c)]
    if len(clears) != 1 or len(keeps) != 1:
        raise AssertionError(f"reclassify clears {clears}, keeps {keeps}: not exactly one of each")
    return clears[0], keeps[0]


def _bare_stamp_verdict() -> str:
    from lgk.refresh.label import _label_problems

    now = type("I", (), {"table_id": "t1", "version": 3})()
    return "accepted" if _label_problems("c.s.nodes", "stamp-1", {"stamp": "stamp-1"}, "stamp-1", now) else "?"


def _folded_directions() -> Tuple[str, str]:
    from lgk.walks.sql import DIRECTIONS, web_sql

    ok = [d for d in DIRECTIONS if _try(lambda d=d: web_sql(1, "s", 25, folded=True, direction=d))]
    if len(ok) != 1:
        raise AssertionError(f"folded walks accept {ok}, not one direction")
    return ok[0], "refused"


def _try(fn: Callable[[], Any]) -> bool:
    try:
        fn()
        return True
    except ValueError:
        return False


def _sync_mode_default() -> str:
    from lgk.refresh.modes import SYNC_MODES, policy_name

    same = [m for m in SYNC_MODES if all(policy_name({"sync_mode": m}, t) == policy_name({}, t) for t in ("nodes", "edges"))]
    if len(same) != 1:
        raise AssertionError(f"no single default sync mode: {same}")
    return same[0]


def _closure_opt_in() -> str:
    from lgk.steps.provision import serve_closure

    if serve_closure({}) is not False or serve_closure({"serve_closure": True}) is not True:
        raise AssertionError("serve_closure is no longer opt-in")
    return "true"


def _kind_of(method: str) -> str:
    from lgk.steps.walks import _LOOKUP_METHOD

    kinds = [k for k, m in _LOOKUP_METHOD.items() if m == method]
    if len(kinds) != 1:
        raise AssertionError(f"lookup kinds for {method}: {kinds}")
    return kinds[0]


def _count_word(n: int) -> str:
    return {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}.get(n, str(n))


def _lookups_using(*methods: str) -> int:
    from lgk.steps.walks import _LOOKUP_METHOD

    return sum(1 for m in _LOOKUP_METHOD.values() if m in methods)


def _join_table_fk_count() -> str:
    from lgk.steps.propose import _is_join_table

    ok = [n for n in range(0, 5) if _is_join_table({"key_columns": []}, [("a", "b", "c")] * n, False)]
    if len(ok) != 1:
        raise AssertionError(f"join tables have {ok} foreign keys")
    return _count_word(ok[0])


def _advisor_files() -> Tuple[str, str]:
    import tempfile

    from lgk.advisor import skipped_report, write_report

    with tempfile.TemporaryDirectory() as d:
        write_report(d, skipped_report("s", "s_v1", "no grant"))
        names = sorted(p.name for p in Path(d).iterdir())
    if len(names) != 2:
        raise AssertionError(f"the advisor writes {names}")
    return tuple(sorted(names, key=lambda n: not n.endswith(".json")))


def _gold_tables_file() -> str:
    m = re.findall(r'_read_gold_tables\(run_folder / "([\w.]+)"\)',
                   inspect.getsource(importlib.import_module("lgk.steps.profile").profile))
    if len(m) != 1:
        raise AssertionError("profile reads no table-list file")
    return m[0]


def _reap_first() -> str:
    src = inspect.getsource(importlib.import_module("lgk.reaper").reap)
    if not src.index("delete_jobs(") < src.index("_reap_one("):
        raise AssertionError("reap no longer deletes the orphan jobs first")
    if "event_log" in (REPO_ROOT / "src/lgk/reaper.py").read_text():
        raise AssertionError("reap now handles event logs: the skill's 'Reap does not drop' is stale")
    return "jobs"


def _role_of(prefix: str) -> str:
    from lgk.served_functions import ROLE_OF

    roles = {r for f, r in ROLE_OF.items() if f.startswith(prefix)}
    if len(roles) != 1:
        raise AssertionError(f"functions {prefix}* have roles {roles}")
    return roles.pop()


def _mixed_selection_error() -> str:
    from lgk.privacy.gate import _derive_selection

    edge = lambda fold: type("E", (), {"fold_subject": fold})()  # noqa: E731
    mixed = type("M", (), {"edges": {"a": edge("p"), "b": edge(None)}})()
    return _raises(lambda: _derive_selection(mixed, None))


def _mixed_named_error() -> str:
    from lgk.privacy.gate import _derive_selection

    edge = lambda fold: type("E", (), {"fold_subject": fold})()  # noqa: E731
    mixed = type("M", (), {"edges": {"a": edge("p"), "b": edge(None)}})()
    return _raises(lambda: _derive_selection(mixed, ["a", "b"]))


ADV = "Step 8: `advise`"
WALKS = "Step 10: `walks`"
NIGHT = "Nightly refresh"
RETIRE = "Retire an old version: `lgk.steps.retire:retire(config, version, confirm_graph_name)`"
REAP = "Reap what a deleted branch left: `lgk.reaper:reap(config, confirm_graph_name)`"
ROLLBACK = "Roll back to an earlier version: `lgk.steps.switch:rollback(config, version)`"


def _served_version_error() -> str:
    """What the nightly reapply's guard raises when the views read another version (after a rollback)."""
    rows = type("R", (), {"fetchall": lambda self: [("k_v1",)]})()
    conn = type("C", (), {"execute": lambda self, sql, params=None: rows})()
    return _raises(lambda: importlib.import_module("lgk.versions").refuse_unless_serving(conn, "k", "k_v3"))


def _rollback_stale_error() -> str:
    """What the switch's stamp check raises for a version recorded under another decision."""
    import tempfile

    binding = importlib.import_module("lgk.privacy.binding")
    with tempfile.TemporaryDirectory() as run:
        binding.record_version(run, "k_v2", "looser")
        return _raises(lambda: binding.check_version(run, "k_v2", "current"))


def _history_last_is_served() -> str:
    """'last' when served_history ends with the version the latest switch recorded."""
    import tempfile

    sw = importlib.import_module("lgk.steps.switch")
    with tempfile.TemporaryDirectory() as run:
        for target in ("k_v1", "k_v3", "k_v2"):
            sw._record_switch({"run_folder": run}, target)
        return "last" if sw.served_history({"run_folder": run}) == ["k_v1", "k_v3", "k_v2"] else "?"


def _unrecorded_version_refused() -> str:
    """'provision' when the switch's stamp check refuses a version this run never recorded."""
    import tempfile

    binding = importlib.import_module("lgk.privacy.binding")
    with tempfile.TemporaryDirectory() as run:
        return "provision" if _raises(lambda: binding.check_version(run, "k_v9", "s")) == "StaleBuildError" else "?"


def _sync_only_job_version() -> str:
    """'only' when the nightly sync's tables are all in the job's own version schema."""
    import tempfile

    fr = importlib.import_module("lgk.refresh.freshness")
    with tempfile.TemporaryDirectory() as run:
        cfg = {"graph_name": "g", "kit_schema": "k", "lakebase_catalog": "c", "version": 3, "run_folder": run}
        dests = [d for d, _ in fr.synced_pairs(cfg, lambda st: [])]
    return "only" if dests and all(d.startswith("c.k_v3.") for d in dests) else "?"


def _version_is_second_argument() -> str:
    names = list(inspect.signature(importlib.import_module("lgk.steps.switch").rollback).parameters)
    return "second" if names[:2] == ["config", "version"] else "?"


def _paused() -> str:
    """The Jobs API value of a paused schedule (the databricks-sdk enum, not the kit)."""
    from databricks.sdk.service.jobs import PauseStatus

    return PauseStatus.PAUSED.value


def _runner_sdk_floor() -> str:
    """The databricks-sdk floor the runner notebook upgrades to (the laptop install uses the same)."""
    m = re.search(r'"databricks-sdk>=([\d.]+)"', (REPO_ROOT / "notebooks" / "lgk_runner.py").read_text())
    return m.group(1) if m else "?"


def _invalidated_steps_rerun(steps: List[str]) -> List[str]:
    """The steps that RUN (not skip) after RunFolder.invalidate_steps, though run.json said success."""
    import tempfile

    runner = importlib.import_module("lgk.runner")
    ran: List[str] = []
    with tempfile.TemporaryDirectory() as run:
        folder = runner.RunFolder(Path(run))
        for s in steps:
            folder.update_step(runner.RunResult(step=s, status=runner.RunStatus.SUCCESS))
        laptop = runner.LaptopRunner({"graph_name": "g", "run_folder": run})
        laptop.invalidate_steps(steps, "new version")
        for s in steps:
            laptop.run_step(s, lambda cfg, s=s: ran.append(s))
    return ran


def _runner_skip_verdict() -> str:
    """'succeeded' when the runner skips (never calls) a step whose run.json status is success."""
    import tempfile

    runner = importlib.import_module("lgk.runner")
    calls = []
    with tempfile.TemporaryDirectory() as run:
        folder = runner.RunFolder(Path(run))
        folder.update_step(runner.RunResult(step="switch", status=runner.RunStatus.SUCCESS))
        runner.execute_step(folder, {}, "switch", lambda c: calls.append(c))
    return "succeeded" if calls == [] else "?"
def _decision_history_errors() -> Dict[str, str]:
    """What the gate's history check raises: an older recorded decision, and a missing history.

    Also proves the order words: decisions are kept newest last, and a re-run of the privacy
    step on an older decision makes it the newest (it then passes)."""
    import tempfile

    decisions = importlib.import_module("lgk.privacy.decisions")
    gate = importlib.import_module("lgk.privacy.gate")
    older, newer = {"flags.json": "a"}, {"flags.json": "b"}
    with tempfile.TemporaryDirectory() as run:
        missing = _raises(lambda: gate.check_newest_decision(run, {"gate": older}))
        decisions.record_decision(run, decisions.decision_id(older))
        decisions.record_decision(run, decisions.decision_id(newer))
        got = {"older": _raises(lambda: gate.check_newest_decision(run, {"gate": older})), "missing": missing,
               "file": decisions.history_path(run).name}
        gate.check_newest_decision(run, {"gate": newer})  # newest last: the newest passes
        decisions.record_decision(run, decisions.decision_id(older))  # the privacy step runs again on it
        gate.check_newest_decision(run, {"gate": older})
        got["newest"] = "newest"
    return got


def _decision_writer() -> str:
    """'privacy step' when steps/privacy.py is the only kit module that records a decision."""
    src = REPO_ROOT / "src" / "lgk"
    writers = sorted(str(p.relative_to(src)) for p in src.rglob("*.py")
                     if "record_decision(" in p.read_text() and p.name != "decisions.py")
    return "privacy step" if writers == ["steps/privacy.py"] else "?"


def _same_words(want: Callable[[], str]) -> Callable[[str], None]:
    """A group predicate: the matched text equals want() up to line breaks."""
    def check(v: str) -> None:
        if " ".join(v.split()) != want():
            raise AssertionError(f"{v!r} != {want()!r}")
    return check


def _reapply_privacy_error() -> str:
    """switch's StaleBuildError, when reapply runs binding.require_servable before it connects."""
    rp = importlib.import_module("lgk.steps.reapply")
    binding = importlib.import_module("lgk.privacy.binding")
    reached = []

    class _Sentinel(Exception):
        pass

    def servable(run_folder, version_schema):
        reached.append(version_schema)
        raise _Sentinel

    real_servable, real_connect = binding.require_servable, rp.connect_branch
    binding.require_servable = servable
    rp.connect_branch = lambda c: reached.append("connect")
    try:
        try:
            rp.reapply({"graph_name": "g", "run_folder": "/nonexistent-lgk", "kit_schema": "k", "version": 3})
        except _Sentinel:
            pass
    finally:
        binding.require_servable, rp.connect_branch = real_servable, real_connect
    sw_src = inspect.getsource(importlib.import_module("lgk.steps.switch").switch)
    if reached != ["k_v3"] or "binding.require_servable(" not in sw_src:
        return "?"
    return _rollback_stale_error()
# ------------------------------------------------------- stream K: key checks (lgk.keys)
def _keys_probe() -> Dict[str, str]:
    """What lgk.keys and profile DO: a duplicate fails profile (after schema.json), orphans do not;
    an accept covers its count and fails on growth; fixed needs zero; build records drops."""
    import tempfile

    keys = importlib.import_module("lgk.keys")
    profile = importlib.import_module("lgk.steps.profile")
    end = lambda n: [{"edge_type": "e", "table": "t", "column": "fk", "node_type": "n", "orphan_rows": n}]  # noqa: E731
    got: Dict[str, str] = {}
    with tempfile.TemporaryDirectory() as run:
        orphans = {"tables": [{"name": "t", "key_columns": ["id"], "duplicate_key_rows": 0,
                               "foreign_keys": [{"columns": ["fk"], "orphan_rows": 3}]}]}
        profile._record_key_findings(run, orphans)  # orphans only: no raise
        got["profile_orphans"] = "fail"
        dup = {"tables": [{"name": "t", "key_columns": ["id"], "duplicate_key_rows": 2, "foreign_keys": []}]}
        got["profile_dup"] = _raises(lambda: profile._record_key_findings(run, dup))
        got["undecided"] = _raises(lambda: keys.apply_edge_orphans(run, end(3)))
        keys.resolve_orphans(run, "t", "fk", keys.CHOICES[0], chooser="op")
        keys.apply_edge_orphans(run, end(3))  # covered: the accept's count
        dropped = keys.read_key_decisions(run)["dropped"]
        got["dropped"] = keys.KEY_DECISIONS_FILE if dropped and dropped[0]["dropped_rows"] == 3 else "?"
        got["grown"] = _raises(lambda: keys.apply_edge_orphans(run, end(4)))
        keys.resolve_orphans(run, "t", "fk", keys.CHOICES[1], chooser="op")
        got["fixed_left"] = _raises(lambda: keys.apply_edge_orphans(run, end(1)))
        keys.apply_edge_orphans(run, end(0))
        got["fixed_zero"] = "zero"
    src = inspect.getsource(profile.profile)
    if not src.index("json.dump(schema") < src.index("_record_key_findings("):
        got["profile_dup"] = "? (raised before schema.json is written)"
    gate_src = "".join(inspect.getsource(importlib.import_module(m))
                       for m in ("lgk.privacy.gate", "lgk.privacy.decisions", "lgk.privacy.binding"))
    got["gate"] = "never" if keys.KEY_DECISIONS_FILE not in gate_src and "lgk.keys" not in gate_src else "?"
    return got


def _choice(i: int) -> Callable[[str], None]:
    def check(v: str) -> None:
        if v != importlib.import_module("lgk.keys").CHOICES[i]:
            raise AssertionError(f"{v!r} is not lgk.keys.CHOICES[{i}]")
    return check


# ----------------------------------------------------------- stream M: the measure step
def _measure_status(shape: str) -> str:
    """The status measure writes for `shape` when it cannot check it (no similarity; a mixed graph)."""
    import tempfile

    m = importlib.import_module("lgk.steps.measure")
    with tempfile.TemporaryDirectory() as run:
        return m._shape_plan(shape, m.settings({}), run, "s", 11, None)[2]


def _owner_setup_kinds() -> Tuple[str, str]:
    """(the kind owner_setup_sql gives a person, the kind it gives a service principal's app id)."""
    pf = importlib.import_module("lgk.steps.preflight")
    kind = lambda who: re.search(r"databricks_create_role\('[^']*', '(\w+)'\)", pf.owner_setup_sql(who)).group(1)  # noqa: E731
    return kind("someone@example.com"), kind(SP_ID)


SP_ID = "-".join("0" * n for n in (8, 4, 4, 4, 12))  # a UUID-shaped id built from parts (no gate exemption)


def check_owner_setup(text: str) -> None:
    """Step 1's ```sql block is lgk.steps.preflight.owner_setup_sql for a service principal,
    one statement per line, with `<id>` for the id (the fix preflight prints, verbatim)."""
    from lgk.steps.preflight import owner_setup_sql

    s, e = section_span(text, "Step 1: `preflight`")
    fences = [m.group(2) for m in FENCE_RE.finditer(text[s:e]) if m.group(1) == "sql"]
    want = owner_setup_sql(SP_ID).replace(SP_ID, "<id>").replace("; ", ";\n") + "\n"
    if fences != [want]:
        raise AssertionError(f"Step 1's sql block is not preflight.owner_setup_sql: {fences!r} != {[want]!r}")


EXPORT = "Export: `lgk.export:export(config)` (a bundle that provisions as a service principal)"


def _c13_timer(v: str) -> None:
    """The C13 bullet says what measure's headline is (owner ruling 2026-10-14): the client p95 on a
    warm connection (lgk.steps.measure; the report's `client p95` column), with prepared statements in the mode
    time_walk records (it sets prepare_threshold = None, so "off"). It never calls the headline a server-side timer."""
    flat = " ".join(v.split())
    timing = importlib.import_module("lgk.measure.timing")
    record_src = inspect.getsource(importlib.import_module("lgk.measure.record"))
    mode = "off" if "conn.prepare_threshold = None" in inspect.getsource(timing.time_walk) else "on"
    for must, code_has in (("client p95", "| client p95 |" in record_src),
                           ("warm connection", "warm connection" in (importlib.import_module("lgk.steps.measure").__doc__
                                                                     or "")),
                           ("prepared statements", "prepared statements" in record_src)):
        if not code_has:
            raise AssertionError(f"the code no longer names {must!r}: rebind the C13 bullet")
        if must not in flat:
            raise AssertionError(f"the C13 bullet does not say {must!r}")
    said = re.search(r"prepared statements\W+(on|off)\b", flat)
    if not said or said.group(1) != mode or mode not in timing.PREPARED_MODES:
        raise AssertionError(f"the C13 bullet does not say prepared statements {mode} (time_walk's mode)")
    if re.search(r"server[- ]side timer", flat, re.I):
        raise AssertionError("the C13 bullet calls the headline a server-side timer; it is the client timer")


C13_BIND = Bind("measure C13 timer", MEASURE, r"- C13 walk speed: (?P<b>.*?)(?=\n- |\n\n)", lambda: {"b": _c13_timer})

CLAIMS: Tuple[Bind, ...] = (
    # ---- error names: the class the code raises for exactly that case
    Bind("missing branch", "Interview: the config", r"refuses a config with no `branch_id` \(`(?P<e>\w+)`\)",
         lambda: {"e": _raises(lambda: importlib.import_module("lgk.branch").BranchRef.from_config({"lakebase_project": "p"}))}),
    Bind("sync owner", "Interview: the config", r"provision refuses \(`(?P<e>\w+)`\) for any other identity",
         lambda: {"e": _built_in("lgk.steps.preflight", "SyncOwnerMismatch")}),
    Bind("warehouse", "Interview: the config", r"refuse a config without it \(`(?P<e>\w+)`\)",
         lambda: {"e": _raises(lambda: importlib.import_module("lgk.uc_sql").warehouse_id({}))}),
    Bind("mapping error", "Step 3: `propose`", r"stops before it runs any SQL, with a `(?P<e>\w+)` that names the field",
         lambda: {"e": _raises(lambda: importlib.import_module("lgk.mapping").parse_mapping({"bogus": 1}))
                  if hasattr(importlib.import_module("lgk.mapping"), "parse_mapping") else _built_in("lgk.mapping", "MappingError")}),
    Bind("care key node", "Step 4: `privacy`", r"The privacy step fails with `(?P<e>\w+)`, and\s+names each node type it refuses",
         lambda: {"e": _built_in("lgk.privacy.gate", "CareKeyNodeError")}),
    Bind("join not unique", "Step 5: `build`", r"a\s+`columns` join whose key is not unique fails with `(?P<e>\w+)`",
         lambda: {"e": _built_in("lgk.steps.build", "JoinNotUniqueError")}),
    Bind("floor errors", "Step 5: `build`", r"fails with `(?P<a>\w+)` or `(?P<b>\w+)` only if a value below",
         lambda: {"a": _built_in("lgk.privacy.floor", "FoldFloorError"), "b": _built_in("lgk.privacy.floor", "JoinFloorError")}),
    Bind("enrich menu stop", "Step 6: `enrich`", r"the step\s+fails with a `(?P<e>\w+)` that names `enrich\.yaml`",
         lambda: {"e": _built_in("lgk.steps.enrich", "FileNotFoundError")}),
    Bind("reader role", "Step 7: `provision`", r"it fails with `(?P<e>\w+)` and the SQL to run",
         lambda: {"e": _built_in("lgk.steps.provision", "ReaderRoleMissingError")}),
    Bind("sync limit", "Step 7: `provision`", r"and fails with `(?P<e>\w+)` when one more would",
         lambda: {"e": _raises(lambda: importlib.import_module("lgk.sync_limits").assert_room({"nodes": 99}, ["nodes"]))}),
    Bind("advise mixed", ADV, r"it is refused\s+with a `(?P<e>\w+)` \(`edge_types is required \.\.\.`\)",
         lambda: {"e": _mixed_selection_error()}),
    Bind("notebook mixed", "Step 12: `notebook` (data engineers)", r"refused with a `(?P<e>\w+)` \(`edge_types is required \.\.\.`\)",
         lambda: {"e": _mixed_selection_error()}),
    Bind("walk mixed", WALKS, r"a mix of both is refused \(`(?P<e>\w+)`\)", lambda: {"e": _mixed_named_error()}),
    Bind("stale build", NIGHT, r"The sync refuses \(`(?P<e>\w+)`\) a table whose label does not match",
         lambda: {"e": _built_in("lgk.refresh.label", "StaleBuildError")}),
    # v1 supports snapshot only; preflight and provision refuse "triggered" with the v1 error, before any source check
    Bind("v1 sync mode", NIGHT, r"v1 supports only `\"(?P<v>\w+)\"`",
         lambda: {"v": importlib.import_module("lgk.refresh.modes").V1_SYNC_MODES[0]}),
    Bind("triggered refused", NIGHT, r"`sync_mode: \"triggered\"`[^.]*\brefused\b[^.]*\(`(?P<e>\w+)`\)",
         lambda: {"e": _raises(lambda: importlib.import_module("lgk.refresh.modes").require_v1_sync_mode(
             {"sync_mode": "triggered"}))}),
    Bind("nightly folder", NIGHT, r"else it refuses with `(?P<e>\w+)`, before\s+any job exists",
         lambda: {"e": _raises(lambda: importlib.import_module("lgk.refresh.nightly_folder").nightly_files("/nonexistent-lgk"))}),
    Bind("unknown pipeline", RETIRE, r"retire\s+refuses \(`(?P<e>\w+)`\) before it deletes anything",
         lambda: {"e": _built_in("lgk.event_logs", "UnknownPipelineError")}),
    Bind("nightly job per version", NIGHT, r"A job is made for (?P<v>one) version: the `version` in the config when you "
         r"create it\.", lambda: {"v": "one" if _sync_only_job_version() == "only" else "?"}),
    Bind("nightly job after a forward switch", NIGHT, r"After a switch to a\s+new version, that job's `reapply` task "
         r"refuses \(`(?P<e>\w+)`\), so each night would fail\.", lambda: {"e": _served_version_error()}),
    Bind("nightly job recreated per switch", NIGHT, r"After each switch to a new version, (?P<v>create) the refresh job "
         r"again with the new config", lambda: {"v": "create" if _served_version_error() == "ServedVersionError" else "?"}),
    Bind("rollback undone overnight", ROLLBACK, r"its `reapply` task refuses \(`(?P<e>\w+)`\) and does not move the\s+"
         r"served views back", lambda: {"e": _served_version_error()}),
    Bind("rollback pauses the job", ROLLBACK, r"First, if a nightly refresh job exists, (?P<v>pause) it\. The job was made "
         r"for the version served now\.", lambda: {"v": "pause" if _served_version_error() == "ServedVersionError" else "?"}),
    Bind("rollback pause value", ROLLBACK, r"\"pause_status\": \"(?P<p>\w+)\"\}\}\}'(?:.|\n)*`settings\.schedule\.pause_status` "
         r"as (?P<d>\w+)\.", lambda: {"p": _paused(), "d": _paused()}),
    Bind("rollback history last", ROLLBACK, r"the\s+(?P<v>last) entry is the version served now",
         lambda: {"v": _history_last_is_served()}),
    Bind("rollback middle may be refused", ROLLBACK, r"ask the operator which version to serve\.\nAn entry before the last may be a version made under another "
         r"privacy decision; the rollback\s+refuses it \(below\), so then choose an earlier one",
         lambda: {} if _rollback_stale_error() == "StaleBuildError" else {"x": None}),
    Bind("rollback version argument", ROLLBACK, r"Give the version number as the (?P<o>\w+) argument; do not "
         r"(?P<v>change) `version` in\s+the config", lambda: {"o": _version_is_second_argument(), "v": "change"}),
    Bind("rollback nightly fails", ROLLBACK, r"served views back, so each night would (?P<v>fail)\.",
         lambda: {"v": "fail" if _served_version_error() == "ServedVersionError" else "?"}),
    Bind("rollback unrecorded", ROLLBACK, r"and a version\s+this run did not (?P<v>provision)\.",
         lambda: {"v": _unrecorded_version_refused()}),
    Bind("rollback rows kept", ROLLBACK, r"the nightly sync refreshes (?P<v>only) the version\s+its job was made for",
         lambda: {"v": _sync_only_job_version()}),
    Bind("rollback old job deleted", ROLLBACK, r"create the\s+refresh job again for it \(Nightly refresh\), and "
         r"(?P<v>delete) the paused job\.", lambda: {"v": "delete"}),
    Bind("rollback job shown", ROLLBACK, r"```bash\ndatabricks jobs update --json '\{\"job_id\": <job id>, [^\n]*\ndatabricks jobs get <job id> "
         r"-o json\n```", lambda: {}),
    Bind("rollback stale", ROLLBACK, r"The rollback refuses \(`(?P<e>\w+)`\) a version provisioned under a privacy "
         r"decision other than\s+the current one", lambda: {"e": _rollback_stale_error()}),
    Bind("older decision", "Step 4: `privacy`", r"the app and the walks\) refuses \(`(?P<e>\w+)`\) a decision that\s+is "
         r"not the (?P<n>newest) one recorded", lambda: {"e": _decision_history_errors()["older"], "n": "newest"}),
    Bind("decision history file", "Step 4: `privacy`", r"The privacy step records each\s+decision it clears in "
         r"`<run_folder>/(?P<f>[\w.]+)`, the newest last\. Only the (?P<w>privacy\s+step) writes this file\.",
         lambda: {"f": _decision_history_errors()["file"], "w": _same_words(_decision_writer)}),
    Bind("unrecorded decision", "Step 4: `privacy`", r"and a missing or damaged\s+`privacy_decisions\.json`, are refused "
         r"too \(`(?P<e>\w+)`\)", lambda: {"e": _decision_history_errors()["missing"]}),
    Bind("privacy rerun makes newest", "Step 4: `privacy`", r"restore its files and run the privacy step again: it "
         r"records that decision as the (?P<n>\w+)\.", lambda: {"n": _decision_history_errors()["newest"]}),
    Bind("reapply privacy check", NIGHT, r"this, `reapply` refuses \(`(?P<e>\w+)`\) a version that was not provisioned "
         r"under the current\s+privacy decision: it is the same check as `switch`, and it applies also before the "
         r"first switch\.", lambda: {"e": _reapply_privacy_error()}),
    # ---- verdicts (a positive word is never free: see VERDICT_WORDS)
    Bind("new version: the runner skips a succeeded step", NEWV, r"The runner skips a step that already\s+(?P<v>succeeded), "
         r"so first mark the steps to run again\.", lambda: {"v": _runner_skip_verdict()}),
    Bind("new version: invalidated steps run again", NEWV, r"runner\.invalidate_steps\(\[(?P<l>[^\]]+)\]",
         lambda: {"l": ", ".join(f'"{s}"' for s in _invalidated_steps_rerun(
             ["build", "enrich", "provision", "advise", "switch"]))}),
    Bind("runner skips a succeeded switch", ROLLBACK, r"The runner skips a step that already\s+(?P<v>succeeded), so "
         r"call `rollback` directly", lambda: {"v": _runner_skip_verdict()}),
    Bind("advise skip succeeds", ADV, r"If this identity may not read\s+the version's raw tables, the report says "
         r"`advisor skipped: <reason>` and the step still (?P<v>succeeds)\.", lambda: {"v": _advise_skip_verdict()}),
    Bind("privacy fails open", "Step 4: `privacy`", r"The step writes `<run_folder>/flags\.json` and (?P<v>FAILS) while any "
         r"care-subject flag is\s+open, so the run stops before build\.", lambda: {"v": _open_flag_verdict()}),
    Bind("reclassify clears", "Step 4: `privacy`", r"Only a `reclassify` to `(?P<c>\w+)`\s+clears a column",
         lambda: {"c": _reclassify_clears()[0]}),
    Bind("reclassify none keeps", "Step 4: `privacy`", r"A `drop` or a `reclassify` to `(?P<n>\w+)` does\s+not clear it",
         lambda: {"n": _reclassify_clears()[1]}),
    Bind("known limit passes", "Step 4: `privacy`", r"a table with one row per person and no\s+care-subject column "
         r"(?P<v>passes) the gate\.", lambda: {"v": "passes" if importlib.import_module("lgk.privacy.gate")._care_source_text(
             "p", "people", {}) is None else "?"}),
    Bind("propose skip", "Step 3: `propose`", r"it is drafted only as a join-table edge, or skipped with a (?P<v>warning)\.",
         lambda: {"v": "warning" if "logger.warning" in inspect.getsource(importlib.import_module("lgk.steps.propose")) else "?"}),
    Bind("bare stamp", NIGHT, r"A bare stamp the pipeline sets itself is no longer (?P<v>accepted)\.",
         lambda: {"v": _bare_stamp_verdict()}),
    Bind("folded outbound only", WALKS, r"Folded edges are symmetric,\s+so they are walked (?P<d>\w+) only; any other "
         r"direction is (?P<r>\w+)\.", lambda: dict(zip(("d", "r"), _folded_directions()))),
    # ---- defaults
    Bind("sync mode default", NIGHT, r"leave `sync_mode` at its default `\"(?P<v>\w+)\"`", lambda: {"v": _sync_mode_default()}),
    Bind("walk direction default", WALKS, r"`direction=\"(?P<d>\w+)\"` \(the default",
         lambda: {"d": _default("lgk.steps.walks:get_walk_query", "direction")}),
    Bind("closure opt-in", "Step 7: `provision`", r"synced only when the config sets\s+`serve_closure: (?P<v>\w+)` "
         r"\(closure is opt-in for now\)", lambda: {"v": _closure_opt_in()}),
    Bind("advise defaults", ADV, r"`advisor_shapes` \(default (?P<a>\w+) and (?P<b>\w+); add `(?P<c>\w+)` to\s+measure the "
         r"set walk\), `advisor_directions` \(default `\[(?P<d>\w+)\]`\)",
         lambda: {"a": _c("lgk.steps.advise", "DEFAULT_SHAPES")()[0], "b": _c("lgk.steps.advise", "DEFAULT_SHAPES")()[1],
                  "c": next(s for s in _c("lgk.walks.sql", "SHAPES")() if s.endswith("_set")),
                  "d": _c("lgk.steps.advise", "DEFAULT_DIRECTIONS")()[0]}),
    # ---- which lookup kind does what
    Bind("same group kind", WALKS, r"`get_lookup_query\(\"(?P<k>\w+)\", \.\.\.\)` returns the other nodes of the node's "
         r"connected group", lambda: {"k": _kind_of("components")}),
    Bind("jaccard kind", WALKS, r"`get_lookup_query\(\"(?P<k>\w+)\", \.\.\.\)` returns the nodes\s+that share the most "
         r"neighbours", lambda: {"k": _kind_of("similarity")}),
    Bind("lookup methods", WALKS, r"(?P<n>Two) more lookups need only `(?P<a>\w+)` or `(?P<b>\w+)` in enrich",
         lambda: {"n": _count_word(_lookups_using("components", "similarity")).capitalize(),
                  "a": "components", "b": "similarity"} if _lookups_using("components", "similarity") == 2 else {}),
    Bind("closure lookups", WALKS, r"(?P<n>\w+) lookups read `closure_view`: the (?P<a>\w+) or the (?P<b>\w+) of a",
         lambda: {"n": _count_word(_lookups_using("closure")), "a": [k for k in ("ancestors", "descendants")
                  if _kind_of_closure(k)][0], "b": [k for k in ("ancestors", "descendants") if _kind_of_closure(k)][1]}),
    # ---- counts written as words
    Bind("join table fks", "Step 3: `propose`", r"a table with no key and exactly (?P<n>\w+) foreign keys",
         lambda: {"n": _join_table_fk_count()}),
    Bind("function roles", "Step 9: `switch`", r"The (?P<n>\w+) function roles read the graph only through the",
         lambda: {"n": _count_word(len(_c("lgk.served_functions", "FUNCTION_ROLES")()))}),
    Bind("role duties", "Step 9: `switch`", r"`(?P<p>\w+)` may run the public directory lookups, and\s+`(?P<c>\w+)` "
         r"may run the privacy-floored walks", lambda: {"p": _role_of("directory_"), "c": _role_of("coordinator_")}),
    Bind("reader roles", EXPORT, r"and the (?P<n>\w+) reader roles \(the SQL of the provision step\)",
         lambda: {"n": _count_word(len(_c("lgk.steps.teardown", "SHARED_ROLES")()))}),
    # ---- files a step reads or writes
    Bind("advisor files", ADV, r"and writes `(?P<a>[\w.]+)` and\s+`(?P<b>[\w.]+)` \(p50/p95 before and after\)",
         lambda: dict(zip(("a", "b"), _advisor_files()))),
    Bind("gold tables file", "Step 2: `profile`", r"To profile only some tables, write their names to\s+`<run_folder>/(?P<f>[\w.]+)`",
         lambda: {"f": _gold_tables_file()}),
    Bind("provision does not serve", "Step 7: `provision`", r"(?P<v>It does not serve) the new version yet: the website "
         r"keeps reading the old one\.", lambda: {"v": _provision_does_not_serve()}),
    # ---- key checks (stream K)
    Bind("profile duplicate key", "Step 2: `profile`", r"A duplicate key fails profile with `(?P<e>\w+)`, after it writes "
         r"`schema\.json`\.", lambda: {"e": _keys_probe()["profile_dup"]}),
    Bind("profile orphans", "Step 2: `profile`", r"Orphans do not (?P<v>fail) profile\.",
         lambda: {"v": _keys_probe()["profile_orphans"]}),
    Bind("orphan call choice", "Step 2: `profile`", r'resolve_orphans\("<run_folder>", "<table>", "<foreign key column>", '
         r'"(?P<c>\w+)"', lambda: {"c": _choice(0)}),
    Bind("accept covers its count", "Step 2: `profile`", r'- `"(?P<a>\w+)"`: build leaves those rows out of `edges` and '
         r"records how many\. An (?P<b>accept) covers only\s+the count it was made on\. If the orphans grow, build stops "
         r"with `(?P<e>\w+)` until the\s+operator decides again\.",
         lambda: {"a": _choice(0), "b": _choice(0), "e": _keys_probe()["grown"]}),
    Bind("fixed needs zero", "Step 2: `profile`", r'- `"(?P<f>\w+)"`: the operator fixed the data\. Build then requires '
         r"(?P<z>zero) orphans", lambda: {"f": _choice(1), "z": _keys_probe()["fixed_zero"]}),
    Bind("key decisions apart from privacy", "Step 2: `profile`", r"`(?P<k>[\w.]+)` is not `flags\.json`: a key decision "
         r"(?P<v>never) changes the privacy gate\.", lambda: {"k": _c("lgk.keys", "KEY_DECISIONS_FILE")(),
                                                            "v": _keys_probe()["gate"]}),
    Bind("build key errors", "Step 5: `build`", r"It fails with `(?P<d>\w+)` on a duplicate natural key\s+of a node type\. "
         r"It fails with `(?P<o>\w+)` on orphans that no decision covers: a foreign key\s+with no decision, a `\"(?P<f>\w+)\"` "
         r"decision while orphans remain, or more orphans than an (?P<a>accept)\s+covers\.",
         lambda: {"d": _built_in("lgk.steps.build", "DuplicateKeyError"), "o": _keys_probe()["undecided"],
                  "f": _choice(1) if _keys_probe()["fixed_left"] == "OrphanKeyError" else "?", "a": _choice(0)}),
    Bind("build accept drops", "Step 5: `build`", r"With an (?P<a>accept), build leaves those rows out\s+of `edges` and "
         r"records the count in `(?P<f>[\w.]+)`\.", lambda: {"a": _choice(0), "f": _keys_probe()["dropped"]}),
    # ---- the measure step (stream M)
    C13_BIND,
    Bind("measure shapes", MEASURE, r"`measure_shapes` \(default (?P<a>\w+), (?P<b>\w+) and (?P<c>\w+)\)",
         lambda: dict(zip("abc", _c("lgk.steps.measure", "DEFAULT_SHAPES")()))),
    Bind("measure first serve steps", MEASURE, r"the sum of the step times in `run\.json`, (?P<a>\w+) to (?P<b>\w+)\.",
         lambda: {"a": _c("lgk.measure.record", "FIRST_SERVE_STEPS")()[0], "b": _c("lgk.measure.record", "FIRST_SERVE_STEPS")()[-1]}),
    Bind("measure freshness", MEASURE, r"C30b freshness: (?P<s>\w+), with the reason \"(?P<r>[^\"]+)\"",
         lambda: {"s": _c("lgk.measure.record", "FRESHNESS")()["status"],
                  "r": _same_words(lambda: _c("lgk.measure.record", "FRESHNESS")()["reason"])}),
    Bind("measure unchecked shapes", MEASURE, r"The similar shape is checked only when enrich\s+computed similarity; else "
         r"its line says `(?P<s>\w+)`\. On a graph that mixes folded and unfolded\s+edge types, the web shape says `(?P<w>\w+)`\.",
         lambda: {"s": _measure_status("similar"), "w": _measure_status("web")}),
    # ---- preflight (stream P)
    Bind("owner setup kinds", "Step 1: `preflight`", r"For a person, write `'(?P<u>\w+)'` in place\s+of `'(?P<s>\w+)'`",
         lambda: dict(zip("us", _owner_setup_kinds()))),
    # ---- orders
    Bind("reap order", REAP, r"`reap` deletes\s+the orphan (?P<a>\w+) first, then each orphan's synced table, UC table and "
         r"recorded pipeline\. Reap does\s+not drop the pipelines' event-log tables\.", lambda: {"a": _reap_first()}),
)


def _provision_does_not_serve() -> str:
    """provision never builds or moves a served view: that is switch (versions.view_statements)."""
    src = inspect.getsource(importlib.import_module("lgk.steps.provision"))
    if "view_statements" in src or "lgk.steps.switch import" in src or "steps.switch" in src.replace("(lgk.steps.switch)", ""):
        raise AssertionError("provision now touches the served views")
    return "It does not serve"


def _kind_of_closure(kind: str) -> bool:
    from lgk.steps.walks import _LOOKUP_METHOD

    return _LOOKUP_METHOD.get(kind) == "closure"


ERROR_TOKEN = re.compile(r"`([A-Z]\w*(?:Error|Mismatch))`")
VERDICT_WORDS = re.compile(r"\b(succeeds?|succeeded|passes|passed|accepted|accepts?|warning|allowed|allows|ignored|"
                           r"tolerated|kept anyway)\b", re.I)
NUMBER_WORDS = re.compile(r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|hundred)\b", re.I)
# (section or None, pattern, reason)
CLAIM_ALLOW: Tuple[Tuple[Optional[str], str, str], ...] = (
    (None, r"patients two doctors share", "the owner's verbatim fold quote (Terms; test_skill.py PERSONA5_QUOTES)"),
    ("Terms", r"the neighbours two nodes share", "Jaccard is defined for a pair of nodes"),
    ("Step 4: `privacy`", r"a key of two or more columns", "the definition of a composite key"),
    ("Step 4: `privacy`", r"a key that links two other things", "a fold joins a pair (src, dst)"),
    (WALKS, r"two hops out, stopping at hospitals", "an example question (depth is bound by NUMBERS)"),
    (WALKS, r"so below a hub the two can differ", "'the two' = the lookup and the tree walk just named"),
    ("Step 4: `privacy`", r"`flags\.json` after it passed", "prose: 'it' = the privacy step's earlier run; the rerun rule"),
    (NIGHT, r"keys the cleared mapping allows", "the property allowlist (bound: lgk.refresh.allowlist, TestBehaviour stamp)"),
    ("Step 4: `privacy`", r"until the privacy step passes again", "prose: the rerun rule; the verdict is bound by 'privacy fails open'"),
    ("Step 13: `teardown`", r"the runner `lambda` passes the typed name", "'passes' = hands over an argument"),
    ("Step 7: `provision`", r"its `reader` proof passed", "prose: the proof; the status after it is bound"),
)


def check_claims(text: str) -> List[Tuple[int, int]]:
    """Every CLAIM matches and agrees with what the code does; error names, verdict words and
    number words outside a bound claim fail."""
    spans = check_numbers(text, CLAIMS) + _allow_spans(text, CLAIM_ALLOW)
    loose = []
    for rx, kind in ((ERROR_TOKEN, "error name"), (VERDICT_WORDS, "verdict word"), (NUMBER_WORDS, "number word")):
        for m in rx.finditer(text):
            s, e = m.span(1) if rx is ERROR_TOKEN else m.span()
            if text.startswith("---", 0) and s < sections(text)[1][1]:
                continue  # frontmatter description is the trigger text (test_skill.py checks it)
            if not any(cs <= s and e <= ce for cs, ce in spans):
                loose.append((text.count("\n", 0, s) + 1, kind, text[s:e]))
    if loose:
        raise AssertionError(f"claims with no binding (line, kind, text): {loose}")
    return spans


# ------------------------------------------------------------------- config keys
_CONFIG_KEYS: Optional[set] = None


def config_keys() -> set:
    """Keys the code reads from a config: `<config>.get("k")`, `<config>["k"]`, `config_int(config, "k")`."""
    global _CONFIG_KEYS
    if _CONFIG_KEYS is None:
        keys: set = set()
        files = list((REPO_ROOT / "src" / "lgk").rglob("*.py")) + list((REPO_ROOT / "notebooks").glob("*.py"))
        for p in files:
            for n in ast.walk(ast.parse(p.read_text())):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("get", "pop", "setdefault") \
                        and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                    owner = getattr(n.func.value, "id", None) or getattr(n.func.value, "attr", None) or ""
                    if (isinstance(n.func.value, ast.Call) and getattr(n.func.value.func, "id", None) == "_app") \
                            or (owner == "app" and p.name == "generate.py"):
                        keys.add("app." + n.args[0].value)  # lgk.app.generate reads config["app"] through _app()
                    elif "config" in owner or owner in ("cfg", "app"):
                        keys.add(n.args[0].value)
                elif isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                    owner = getattr(n.value, "id", None) or ""
                    if "config" in owner or owner in ("cfg", "app"):
                        keys.add(n.slice.value)
                elif isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", None)) in CONFIG_READERS \
                        and len(n.args) > 1 and isinstance(n.args[1], ast.Constant):
                    keys.add(n.args[1].value)
        _CONFIG_KEYS = keys
    return _CONFIG_KEYS


# Helpers that read `config[key]` as (config, "key", ...): lgk.uc_sql.config_int, lgk.steps.measure._positive
CONFIG_READERS = ("config_int", "_positive")
# Backticked snake_case words in a sentence that says "config" but that are not config keys.
CONFIG_SENTENCE_ALLOW = {
    "closure": "an enrich method name (lgk.enrich.menu.V1_METHODS)",
    "workspace_client": "the live client the skill tells the operator to keep OUT of a JSON config",
    "teardown": "a step name", "lambda": "the Python keyword of the teardown runner call",
    "web_set": "an advisor_shapes VALUE (lgk.walks.sql.SHAPES; bound by CLAIMS 'advise defaults')",
    "closure_view": "a served view name (lgk.versions.OPTIONAL_SERVED_VIEWS)",
    "max_depth": "a get_lookup_query keyword (bound by check_python_blocks), not a config key",
}
# Non-config keys of the yaml-ish `key: value` tokens: key -> file
YAML_NON_CONFIG = {"methods": "enrich.yaml (lgk.steps.enrich)", "subject": "a graph_mapping.yaml join (lgk.mapping)"}


def check_config_keys(text: str) -> None:
    """Every config key the skill names is a key the code reads."""
    keys = config_keys()
    if len(keys) < 30:
        raise AssertionError(f"only {len(keys)} config keys found in the code: the reader is broken")
    bad = []
    s, e = section_span(text, "Interview: the config")
    for k in re.findall(r"^\| `(\w+)` \|", text[s:e], re.M):
        if k not in keys:
            bad.append(("interview table", k))
    prose = FENCE_RE.sub(lambda m: "\n" * m.group().count("\n"), text)
    # a sentence ends at ". " or ".\n"; a blank line, a list item or a table row starts a new one
    for chunk in re.split(r"(?<=\.)\s+|\n\n|\n(?=- |\|)", prose):
        if not re.search(r"\bconfig\b", chunk):
            continue
        for tok in re.findall(r"`([a-z_][a-z0-9_.]*)`", chunk):
            if tok in CONFIG_SENTENCE_ALLOW or ("." in tok and not tok.startswith("app.")):
                continue
            if tok not in keys:
                bad.append(("sentence", tok))
    for m in TICK_RE.finditer(text):
        y = YAML_RE.fullmatch(m.group(1))
        if y and y.group("k") not in keys and y.group("k") not in YAML_NON_CONFIG:
            bad.append(("yaml", m.group(1)))
    if bad:
        raise AssertionError(f"config keys the code never reads: {bad}")


# Positionals the help lists that must move into the --json body: with --json the CLI refuses
# them ("when --json flag is specified, no positional arguments are allowed"; skill E2E, 2026-10-05).
JSON_BODY_POSITIONALS: Dict[Tuple[str, ...], int] = {("databricks", "jobs", "update"): 1}

CLI_REQUIRED: Dict[Tuple[str, ...], Tuple[set, str]] = {
    ("databricks", "workspace", "delete"): ({"--recursive"}, "the copy folder holds files: without it the delete fails DIRECTORY_NOT_EMPTY"),
    ("databricks", "workspace", "import-dir"): ({"--overwrite"}, "a re-upload must replace the earlier files"),
    ("databricks", "workspace", "import"): ({"--file", "--format", "--overwrite"}, "a local file, its format, and re-upload"),
    ("databricks", "jobs", "update"): ({"--json"}, "the job id and the schedule are the JSON body: the CLI refuses a JOB_ID argument with --json"),
    ("databricks", "jobs", "get"): ({"-o"}, "the Done-when reads the JSON settings"),
    ("uv", "build"): ({"--wheel"}, "the runner and the job install the wheel"),
}


def check_cli_required(text: str) -> None:
    lines = [ln for m in FENCE_RE.finditer(text) if m.group(1) == "bash" for ln in m.group(2).splitlines() if ln.strip()]
    for ln in lines:
        words = shlex.split(re.sub(r"<[^>]+>", "PLACEHOLDER", ln))
        for key, (need, reason) in CLI_REQUIRED.items():
            if tuple(words[:len(key)]) == key and not need <= set(words):
                raise AssertionError(f"`{ln}`: needs {sorted(need - set(words))} ({reason})")


def check_notebook_source(text: str) -> None:
    """The runner notebook the skill uploads is the one the backend names (and exists)."""
    from lgk.notebook_runner import DatabricksNotebookBackend

    found = re.findall(r'notebook_source=Path\("([^"]+)"\)', text)
    if len(found) != 1:
        raise AssertionError(f"want one notebook_source in the skill, got {found}")
    remote = DatabricksNotebookBackend(object(), "/Workspace/Users/user@example.com/lgk_v1_abcdef", Path("w.whl"),
                                       Path(found[0])).remote_notebook.rsplit("/", 1)[1]
    if Path(found[0]).stem != remote or not (REPO_ROOT / found[0]).is_file():
        raise AssertionError(f"notebook_source {found[0]} is not the runner notebook {remote}")
    s, e = section_span(text, "Interview: the config")
    if f"`{found[0]}`" not in text[s:e]:
        raise AssertionError("the runner notebook named in the prose is not the one uploaded")


# ==================================================== the service principal's grants
def check_sp_grants(text: str) -> None:
    """The export section's ```sql block is the code's one grant list (lgk.sp_grants), verbatim.

    Rendered with the placeholders, so a grant added, dropped, reordered or reworded in the code
    or in the skill fails here; the README (lgk.export.operator_setup), the SP's preflight and
    the live test read the same list. The prose also names the project permission and the
    Postgres grant from that module, and says the kit never grants on a source schema."""
    from lgk import sp_grants

    s, e = section_span(text, EXPORT)
    section = text[s:e]
    fences = [m.group(2) for m in FENCE_RE.finditer(section) if m.group(1) == "sql"]
    want = "\n".join(sp_grants.setup_sql(sp_grants.PLACEHOLDERS, "<graph_name>", sp_grants.SP_PLACEHOLDER)) + "\n"
    if fences != [want]:
        raise AssertionError(f"the export section's sql block is not lgk.sp_grants.setup_sql: {fences!r} != {[want]!r}")
    flat = " ".join(section.split())
    for must in (f"`{sp_grants.PROJECT_PERMISSION}` on the Lakebase project", sp_grants.PG_GRANT,
                 "The kit never grants anything on a source schema"):
        if must not in flat:
            raise AssertionError(f"the export section does not say {must!r}")


# ============================================================================== tests
@pytest.fixture(scope="module")
def text() -> str:
    return SKILL.read_text()


@pytest.fixture(scope="module")
def snap() -> Dict[str, Any]:
    return json.loads(SNAPSHOT.read_text())


class TestFreeze:
    def test_snapshot_matches(self, text, snap):
        check_snapshot(text, snap)

    def test_every_number_is_bound(self, text):
        check_number_coverage(text)

    def test_every_status_word_is_bound(self, text):
        check_status_words(text)

    def test_every_python_call_binds(self, text):
        check_python_blocks(text)

    def test_every_backticked_token_is_bound(self, text):
        check_inline_code(text)

    def test_every_cli_command_parses(self, text):
        check_cli(text)
        check_cli_paths(text)

    def test_step_headings_follow_the_runner(self, text):
        _steps_in_order(text)

    def test_every_claim_matches_what_the_code_does(self, text):
        check_claims(text)

    def test_every_config_key_is_read_by_the_code(self, text):
        check_config_keys(text)

    def test_cli_commands_carry_their_required_flags(self, text):
        check_cli_required(text)

    def test_sp_grants_are_the_code_list(self, text):
        check_sp_grants(text)

    def test_owner_setup_is_the_preflight_fix(self, text):
        check_owner_setup(text)

    def test_notebook_source_is_the_runner_notebook(self, text):
        check_notebook_source(text)

    def test_installed_copies_are_byte_identical(self, tmp_path):
        from lgk.interview import install_skill_to_both_hosts

        for dest in install_skill_to_both_hosts(tmp_path / "claude", tmp_path / "genie"):
            assert Path(dest).read_bytes() == SKILL.read_bytes()


class TestBehaviour:
    """The code facts the TOKEN_ALLOW reasons point at (read the expression, not the name)."""

    def test_run_folder_and_kit_schema_defaults(self):
        from lgk import versions
        from lgk.runner import resolve_run_dir

        assert resolve_run_dir({"graph_name": "g"}) == Path.cwd() / "runs" / "g"
        assert versions.kit_schema({"graph_name": "g"}) == "lgk_g"
        assert versions.version_schema({"kit_schema": "k", "graph_name": "g", "version": 3}) == "k_v3"

    def test_refresh_task_upgrades_the_sdk_first(self, text):
        src = (REPO_ROOT / "notebooks/lgk_refresh_task.py").read_text()
        assert _sdk_pin() and src.index("databricks-sdk") < src.index("restartPython") < src.index("pip\", \"install")
        s, e = section_span(text, "Nightly refresh")
        assert "first upgrades `databricks-sdk` and restarts Python, then installs the toolkit wheel" in " ".join(text[s:e].split())

    def test_run_result_fields(self):
        from lgk.runner import RunResult, RunStatus

        assert set(RunResult("s", RunStatus.FAILED, "e").to_dict()) == {"step", "status", "error"}

    def test_nightly_folder_default_and_files(self):
        from lgk.refresh import nightly_folder as nf

        nb = "/Workspace/Users/<operator>/lgk/lgk_refresh_task"
        assert nf.workspace_run_folder({"graph_name": "<graph_name>"}, nb) == "/Workspace/Users/<operator>/lgk/runs/<graph_name>"
        assert nf.REQUIRED_FILES == ("flags.json", "schema.json", "graph_mapping.yaml", "privacy_decisions.json",
                                     "lgk_manifest.json")
        assert nf.OPTIONAL_FILES == ("enrich.yaml", "enrich_report.json", "privacy_bindings.json")

    def test_skill_lists_the_nightly_files_the_code_copies(self, text):
        from lgk.refresh import nightly_folder as nf

        s, e = section_span(text, "Nightly refresh")
        flat = " ".join(text[s:e].split())
        req = ", ".join(f"`{n}`" for n in nf.REQUIRED_FILES[:-1]) + f" and `{nf.REQUIRED_FILES[-1]}`"
        opt = ", ".join(f"`{n}`" for n in nf.OPTIONAL_FILES[:-1]) + f" and `{nf.OPTIONAL_FILES[-1]}`"
        assert f"holds {req} (else it refuses with `{nf.NightlyFolderError.__name__}`" in flat
        assert f"and {opt} when they exist" in flat

    def test_skill_names_the_key_file_the_nightly_job_copies(self, text, tmp_path):
        import json as _json

        from lgk.keys import KEY_DECISIONS_FILE
        from lgk.refresh import nightly_folder as nf

        assert nf.KEY_FILES == (KEY_DECISIONS_FILE,)
        s, e = section_span(text, "Nightly refresh")
        flat = " ".join(text[s:e].split())
        assert f"`create_refresh_job` also copies `{KEY_DECISIONS_FILE}`" in flat
        assert f"When `schema.json` records orphans and `{KEY_DECISIONS_FILE}` is missing, it refuses the same way" in flat
        for name in nf.REQUIRED_FILES:
            (tmp_path / name).write_text("{}")
        (tmp_path / "schema.json").write_text(_json.dumps(
            {"tables": [{"name": "t", "foreign_keys": [{"columns": ["fk"], "orphan_rows": 2}]}]}))
        with pytest.raises(nf.NightlyFolderError, match=KEY_DECISIONS_FILE):
            nf.nightly_files(tmp_path)  # the same refusal as a missing required file
        (tmp_path / KEY_DECISIONS_FILE).write_text("{}")
        assert KEY_DECISIONS_FILE in nf.nightly_files(tmp_path)

    def test_measure_files(self, text):
        from lgk.measure import record

        s, e = section_span(text, MEASURE)
        sec = text[s:e]
        assert f"`<run_folder>/{record.JSONL}`" in sec and f"`<run_folder>/{record.REPORT}`" in sec
        assert f"**Done when** `<run_folder>/{record.JSONL}` exists" in sec
        md = record.report_markdown([{"metric": "walk_correctness", "shape": "web", "depth": 1, "seeds_checked": 2,
                                      "seeds_equal": 2, "seeds_different": 0, "status": "checked"}])
        assert "| seeds checked | seeds equal | seeds different |" in md  # the C10 table the Done-when names

    def test_runtime_record_fields(self, tmp_path, text):
        from lgk.runtime import RUNTIME_FILE, runtime_record

        rec = runtime_record("3", ["profile"])
        s, e = section_span(text, "Interview: the config")
        sec = text[s:e]
        assert f"`<run_folder>/{RUNTIME_FILE}`" in sec
        for k in rec:
            assert f"`{k}`" in sec, k
        assert rec["python_version"] == sys.version and "(`sys.version`)" in sec

    def test_event_log_name_and_pipeline_lookup(self):
        from lgk.event_logs import event_log_table, pipeline_id

        assert event_log_table("c.s.t", "a-b") == "c.s.event_log_a_b"

        class _T:
            def get(self, name):
                return type("I", (), {"pipeline_id": "p-1", "properties": {}})()

        assert pipeline_id(type("W", (), {"tables": _T()})(), "c.s.t", None) == "p-1"
        assert pipeline_id(None, "c.s.t", "rec") == "rec"

    def test_label_fields(self):
        from lgk.privacy.binding import LABEL_FIELDS, STAMP_PROPERTY

        assert set(LABEL_FIELDS) == {"stamp", "table_id", "certified_version"} and STAMP_PROPERTY == "lgk.privacy_stamp"

    def test_advise_defaults_and_skip_text(self):
        from lgk.steps import advise

        assert advise.DEFAULT_DIRECTIONS == ("outbound",) and advise.DEFAULT_SHAPES == ("web", "tree")
        assert "advisor skipped" in (REPO_ROOT / "src/lgk/advisor.py").read_text() + \
            (REPO_ROOT / "src/lgk/steps/advise.py").read_text()

    def test_bundle_state_path(self):
        assert "${workspace.root_path}/state/run" in (REPO_ROOT / "src/lgk/export.py").read_text()

    def test_deploy_command_flag(self):
        from lgk.app.generate import deploy_command

        assert deploy_command("n", "/p") == "databricks apps deploy n --source-code-path /p"

    def test_runner_notebook_pip_installs_the_wheel(self):
        assert "pip" in (REPO_ROOT / "notebooks/lgk_runner.py").read_text()

    def test_enrich_methods_form_and_gold_tables_form(self, tmp_path):
        from lgk.steps.profile import _read_gold_tables

        (tmp_path / "g.json").write_text('{"tables": [{"name": "providers"}]}')
        assert _read_gold_tables(tmp_path / "g.json") == [{"name": "providers"}]

    def test_teardown_order_is_the_skill_order(self, text):
        """The teardown sentence lists the phases in the order teardown() calls them."""
        src = inspect.getsource(importlib.import_module("lgk.steps.teardown").teardown)
        calls = re.findall(r"ops\.(delete_jobs|revoke_shared_roles|drop_views|delete_synced_tables|drop_event_logs|"
                           r"drop_pg|delete_branches|drop_uc)\(", src)
        words = {"delete_jobs": "the recorded jobs", "revoke_shared_roles": "grants revoked",
                 "drop_views": "the served views and functions", "delete_synced_tables": "the synced tables",
                 "drop_event_logs": "the event-log table", "drop_pg": "the Postgres tables and schemas",
                 "delete_branches": "any recorded branch", "drop_uc": "the Unity Catalog tables and schemas"}
        s, e = section_span(text, "Step 13: `teardown`")
        flat = " ".join(text[s:e].split())
        pos = [flat.find(words[c]) for c in calls]
        assert -1 not in pos and pos == sorted(pos), list(zip(calls, pos))
        assert "recorded roles" not in flat  # ROLE entries are skipped (plan_from_manifest)

    def test_retire_order_is_the_skill_order(self, text):
        src = inspect.getsource(importlib.import_module("lgk.steps.retire")._drop_versions)
        calls = re.findall(r"(_delete_synced_tables|_drop_event_logs|_drop_pg_schema|_drop_uc_schema)\(", src)
        words = {"_delete_synced_tables": "recorded synced tables and their", "_drop_event_logs": "the event-log table",
                 "_drop_pg_schema": "the Postgres version schema", "_drop_uc_schema": "the Unity Catalog version schema"}
        s, e = section_span(text, "Retire an old version: `lgk.steps.retire:retire(config, version, confirm_graph_name)`")
        flat = " ".join(text[s:e].split())
        pos = [flat.find(words[c]) for c in calls]
        assert -1 not in pos and pos == sorted(pos), list(zip(calls, pos))

    def test_job_records_its_run_folder_and_teardown_keeps_it(self):
        """Open issue: the copy folder is in the JOB entry's metadata.run_folder; teardown never deletes it.
        When teardown starts deleting it, this test fails and the SKILL's operator step must go."""
        job_src = inspect.getsource(importlib.import_module("lgk.refresh.job").create_refresh_job)
        assert 'record_job(config, job_id, "refresh", run_folder=folder)' in job_src
        td = (REPO_ROOT / "src/lgk/steps/teardown.py").read_text()
        assert "workspace.delete" not in td and "run_folder" not in td

    def test_triggered_needs_a_streaming_table_and_the_kit_writes_plain_delta(self):
        from lgk.refresh import modes

        class _T:
            def get(self, full_name):
                return type("I", (), {"table_type": "MANAGED"})()

        with pytest.raises(modes.TriggeredSourceError):
            modes.assert_streaming_source(type("W", (), {"tables": _T()})(), "c.s.nodes")
        assert modes.policy_name({}, "nodes") == "SNAPSHOT"  # the default the skill tells the operator to keep
        assert "USING DELTA" in (REPO_ROOT / "src/lgk/steps/build.py").read_text()

    def test_stamp_check_has_no_skip(self):
        src = inspect.getsource(importlib.import_module("lgk.refresh.sync")._require_stamped_tables)
        body = src.split('"""')[2]
        assert "check_labels" in body and "return" not in body and " if " not in body

    def test_branch_id_is_refused_when_missing(self):
        from lgk.branch import BranchRef, MissingBranchError

        for cfg in ({"lakebase_project": "p"}, {"lakebase_project": "p", "branch_id": " "}):
            with pytest.raises(MissingBranchError):
                BranchRef.from_config(cfg)
        assert "branch_id" in inspect.getsource(importlib.import_module("lgk.steps.preflight").preflight)

    def test_care_source_rule_and_known_limit(self):
        from lgk.privacy import gate

        assert gate._care_source_text("p", "patients", {"patients": ["city"]})
        assert gate._care_source_text("p", "people", {}) is None  # the known limit: no flag, no refusal


# =============================================================================== hostile
def _mut(text: str, old: str, new: str, count: int = 1) -> str:
    assert old in text, f"hostile fixture anchor not found: {old!r}"
    return text.replace(old, new, count)


HOSTILE = [
    # (id, old, new, checker) -- reworded contradictions and planted unbound facts
    ("status succeeded for success", "status is `success` and `nodes`", "status is `succeeded` and `nodes`", check_status_words),
    ("status word unbackticked", "with status `success`.", "with status succeeded.", check_status_words),
    ("teardown status completed", "with status `success`.", "with status `completed`.", check_status_words),
    ("new status sentence", "## Step 6: `enrich`", "Its status is `ok`.\n\n## Step 6: `enrich`", check_status_words),
    ("floor 11 -> 12", "`privacy_floor`: default 11.", "`privacy_floor`: default 12.", check_number_coverage),
    ("floor 11 -> 10 (walks)", "privacy floor (default 11, never", "privacy floor (default 10, never", check_number_coverage),
    ("seeds 200 -> 201", "seeds (default 200, at most", "seeds (default 201, at most", check_number_coverage),
    ("seed cap 1,000 -> 1,001", "at most 1,000;", "at most 1,001;", check_number_coverage),
    ("headroom 2 -> 3", "(config, default 2)", "(config, default 3)", check_number_coverage),
    ("21st -> 22st", "21st is reported", "22st is reported", check_number_coverage),
    ("walk keys 4 -> 5", "at most 4) must", "at most 5) must", check_number_coverage),
    ("app timeout 5 -> 10", "with a 5 s statement timeout", "with a 10 s statement timeout", check_number_coverage),
    ("per-sync 18 -> 20", "budgets 18 connections", "budgets 20 connections", check_number_coverage),
    ("env 3 -> 2", "environment version `3`", "environment version `2`", check_number_coverage),
    ("python 3.12 -> 3.11", "older than 3.12.", "older than 3.11.", check_number_coverage),
    ("wheel 0.1.0 -> 0.2.0", "`lgk-0.1.0-py3-none-any.whl` and `lgk_runner`", "`lgk-0.2.0-py3-none-any.whl` and `lgk_runner`",
     check_number_coverage),
    ("example depth 2 -> 9", 'get_walk_query("<start node id>", depth=2', 'get_walk_query("<start node id>", depth=9',
     check_number_coverage),
    ("lookup range 100 -> 99", "levels (1 to 100)", "levels (1 to 99)", check_number_coverage),
    ("bound sentence reworded", "may test by equality, at most 4)", "may test by equality, up to 4)", check_number_coverage),
    ("new unbound number", "## Step 6: `enrich`", "Enrich retries 3 times.\n\n## Step 6: `enrich`", check_number_coverage),
    ("new unbound number in a code block", "print(job_id)", "print(job_id, 7)", check_number_coverage),
    ("step renumbered", "## Step 6: `enrich`", "## Step 7: `enrich`", check_number_coverage),
    ("cli flag renamed", "import-dir <app folder>", "importdir <app folder>", check_cli),
    ("cli flag invented", "--overwrite\ndatabricks workspace list <workspace path", "--force\ndatabricks workspace list <workspace path",
     check_cli),
    ("cli format value", "--format SOURCE", "--format NOTEBOOK", check_cli),
    ("cli positional dropped", "databricks workspace list <work folder>", "databricks workspace list", check_cli),
    ("cli jobs get output", "-o json", "-o yaml", check_cli),
    ("python kwarg renamed", "chooser=\"<operator email>\")", "choser=\"<operator email>\")", check_python_blocks),
    ("python ref renamed", '"lgk.reaper:find_orphans"', '"lgk.reaper:find_orphan"', check_python_blocks),
    ("python direction value", 'direction="outbound")', 'direction="upstream")', check_python_blocks),
    ("python lookup kind", 'get_lookup_query("ancestors"', 'get_lookup_query("parents"', check_python_blocks),
    ("python unknown call", "print(bundle_dir)", "print(bundle_dir)\ndeploy(bundle_dir)", check_python_blocks),
    ("config key renamed", '"sync_owner_user": "<sync_owner_user>"', '"sync_owner": "<sync_owner_user>"', check_python_blocks),
    ("inline signature param", "teardown(config, confirm_graph_name)`:", "teardown(config, confirm_name)`:",
     check_inline_code),
    ("inline kw choice", '"reclassify", chooser=...', '"rename", chooser=...', check_inline_code),
    ("inline kw name", '`sort_method="degree"`', '`sort_by="degree"`', check_inline_code),
    ("inline kw value", '`direction="outbound"`', '`direction="forward"`', check_inline_code),
    ("config key renamed inline", "`sync_per_source_headroom`", "`sync_headroom`", check_inline_code),
    ("error class renamed", "(`MissingBranchError`)", "(`NoBranchError`)", check_inline_code),
    ("yaml value", '`build_mode: "sql"`', '`build_mode: "pipeline"`', check_inline_code),
    ("quoted value", '`"inbound"`', '`"upstream"`', check_inline_code),
    ("route renamed", "`GET /api/walk`", "`GET /api/walks`", check_inline_code),
    ("file renamed", "`<run_folder>/runtime.json`", "`<run_folder>/run_env.json`", check_inline_code),
    ("new unbound token", "## Step 6: `enrich`", "Use `@@weird@@`.\n\n## Step 6: `enrich`", check_inline_code),
    # prover round 1 survivors (binding layer, after a careless snapshot regen), now closed
    ("M1 verdict flipped", "and the step still succeeds.", "and the step fails.", check_claims),
    ("M2 fails -> passes", "and FAILS while any care-subject flag is", "and passes with a warning while any care-subject flag is",
     check_claims),
    ("M3 number word", "exactly two foreign keys", "exactly three foreign keys", check_claims),
    ("M6 required flag dropped", "--recursive\n", "\n", check_cli_required),
    ("M9 positional order", "(config, [<versions>], <typed name>)", "(config, <typed name>, [<versions>])", check_inline_code),
    ("M10b lookup kinds swapped", 'get_lookup_query("same_group", ...)` returns', 'get_lookup_query("similar_jaccard", ...)` returns',
     check_claims),
    ("M12 config dict key", '"branch_id": "<branch_id>"', '"branch": "<branch_id>"', check_python_blocks),
    ("M14 yaml key", "the config set\n`serve_closure: true`", "the config set\n`closure: true`", check_config_keys),
    ("M15 optional config key", "`advisor_seed_node_id`", "`seed_node_id`", check_config_keys),
    ("M16 config key in prose", "(`sdp_pipeline_id`)", "(`pipeline_id`)", check_config_keys),
    ("M17 default value", 'at its default `"snapshot"`', 'at its default `"triggered"`', check_claims),
    ("M18b refused -> accepted", "any other direction is refused.", "any other direction is accepted.", check_claims),
    ("M19b refuses -> accepts", "The sync refuses (`StaleBuildError`)", "The sync accepts with a warning (`StaleBuildError`)", check_claims),
    ("M19 error swapped", "and fails with `SyncLimitError` when", "and fails with `StaleBuildError` when", check_claims),
    ("M21 file of the wrong step", "writes `advisor_report.json` and", "writes `enrich_report.json` and", check_claims),
    ("M22 wrong notebook", 'notebook_source=Path("notebooks/lgk_runner.py")', 'notebook_source=Path("notebooks/lgk_refresh_task.py")',
     check_notebook_source),
    ("M23 wrong file", "write their names to\n`<run_folder>/gold_tables.json`", "write their names to\n`<run_folder>/schema.json`",
     check_claims),
    ("M24 roles swapped", "`public_reader` may run the public directory lookups, and\n`coordinator_reader`",
     "`coordinator_reader` may run the public directory lookups, and\n`public_reader`", check_claims),
    ("M25 negation removed", "It does not serve the new version yet", "It serves the new version at once", check_claims),
    ("M26 order swapped", "the orphan jobs first", "the orphan synced tables first", check_claims),
    ("M27 negation flipped", "Reap does\nnot drop the pipelines' event-log tables.", "Reap also\ndrops the pipelines' event-log tables.",
     check_claims),
    ("M28 unscored sort method", 'or `"pagerank"`', 'or `"similarity"`', check_inline_code),
    ("M33 one -> two", "one edge\nfor each join table", "two edges\nfor each join table", check_claims),
    ("M35 contradiction inserted", "the report says `advisor skipped: <reason>` and the step still",
     "the step fails and the report says `advisor skipped: <reason>` and the step still", check_claims),
]


HOSTILE += [
    # the service principal's grant list is bound to lgk.sp_grants
    ("SP grant dropped", "GRANT USE SCHEMA, SELECT ON SCHEMA <source_catalog>.<source_schema> TO `<SP application id>`;\n", "",
     check_sp_grants),
    ("SP privilege swapped", "GRANT USE SCHEMA, CREATE TABLE ON SCHEMA <lakebase_catalog>.<sp_kit_schema> TO",
     "GRANT USE SCHEMA, MODIFY ON SCHEMA <lakebase_catalog>.<sp_kit_schema> TO", check_sp_grants),
    ("SP workshop schema", "ON SCHEMA <lakebase_catalog>.<sp_kit_schema> TO", "ON SCHEMA <lakebase_catalog>.<kit_schema> TO",
     check_sp_grants),
    ("SP project level", "`CAN_USE` on the Lakebase project", "`CAN_MANAGE` on the Lakebase project", check_sp_grants),
    # stream M: the measure step
    ("measure seeds 20 -> 25", "`measure_seeds` seeds (default 20)", "`measure_seeds` seeds (default 25)", check_number_coverage),
    ("measure samples 200 -> 100", "(default 200, at least 20)", "(default 100, at least 20)", check_number_coverage),
    ("measure min samples 20 -> 10", "(default 200, at least 20)", "(default 200, at least 10)", check_number_coverage),
    ("measure warmup 3 -> 5", "`measure_warmup` runs (default 3)", "`measure_warmup` runs (default 5)", check_number_coverage),
    ("measure depths 1..4 -> 0..4", "4: depths 1 to 4)", "4: depths 0 to 4)", check_number_coverage),
    ("measure rows 5,000,000 -> 1,000,000", "(default 5,000,000 source", "(default 1,000,000 source", check_number_coverage),
    ("measure planted number", "A rerun replaces both files.", "A rerun replaces both files in 30 s.", check_number_coverage),
    ("measure shapes swapped", "(default web, tree and similar)", "(default tree, web and similar)", check_claims),
    ("measure first serve to advise", "preflight to switch.\n", "preflight to advise.\n", check_claims),
    ("measure freshness reason", "untested from inside a workspace", "tested from inside a workspace", check_claims),
    ("measure unchecked swapped", "else its line says `not_served`", "else its line says `not_checked`", check_claims),
    # stream K: key checks
    ("profile dup error renamed", "A duplicate key fails profile with `DuplicateKeyError`",
     "A duplicate key fails profile with `OrphanKeyError`", check_claims),
    ("profile orphans fail", "Orphans do not fail profile.", "Orphans fail profile.", check_claims),
    ("accept covers growth", "If the orphans grow, build stops with `OrphanKeyError`",
     "If the orphans grow, build stops with `DuplicateKeyError`", check_claims),
    ("fixed and accept swapped", '- `"fixed"`: the operator fixed', '- `"accept"`: the operator fixed', check_claims),
    ("fixed needs some", "Build then requires zero orphans", "Build then requires few orphans", check_claims),
    ("build errors swapped", "It fails with `DuplicateKeyError` on a duplicate natural key",
     "It fails with `OrphanKeyError` on a duplicate natural key", check_claims),
    ("keys file renamed", "records the count in `key_decisions.json`.", "records the count in `flags.json`.", check_claims),
    ("orphan choice unknown", '"<foreign key column>", "accept"', '"<foreign key column>", "keep"', check_python_blocks),
    # stream P: preflight's owner setup
    ("owner setup role dropped", "CREATE ROLE public_reader WITH LOGIN;\n", "", check_owner_setup),
    ("owner setup grant widened", 'GRANT CREATE ON DATABASE databricks_postgres TO "<id>";',
     'GRANT ALL ON DATABASE databricks_postgres TO "<id>";', check_owner_setup),
    ("owner setup kinds swapped", "write `'user'` in place\nof `'service_principal'`",
     "write `'service_principal'` in place\nof `'user'`", check_claims),
]


class TestHostile:
    @pytest.mark.parametrize("mid,old,new,checker", HOSTILE, ids=[h[0] for h in HOSTILE])
    def test_mutation_is_caught(self, text, snap, mid, old, new, checker):
        bad = _mut(text, old, new)
        with pytest.raises(AssertionError):
            checker(bad)
        with pytest.raises(AssertionError):
            check_snapshot(bad, snap)  # and the freeze sees every edit

    @pytest.mark.parametrize("old,new", [
        ("advise fails; go on", "advise fails; go on"),  # placeholder replaced below
    ][:0] + [
        ("only folded types gives people counts", "only folded types gives edge counts"),
        ("Folded edges are symmetric,\nso they are walked outbound only", "Folded edges are symmetric,\nso they are walked inbound only"),
        ("It can be raised and never\n  lowered", "It can be raised and\n  lowered"),
        ("Teardown deletes\nthe job, but it does NOT delete", "Teardown deletes\nthe job, and it deletes"),
    ])
    def test_prose_contradiction_is_caught_by_the_snapshot(self, text, snap, old, new):
        with pytest.raises(AssertionError, match="edited since the snapshot"):
            check_snapshot(_mut(text, old, new), snap)

    def test_empty_snapshot_fails(self, text):
        with pytest.raises(AssertionError, match="never passes"):
            check_snapshot(text, {"sections": [], "file_sha256": ""})

    def test_section_added_fails(self, text, snap):
        with pytest.raises(AssertionError, match="sections changed"):
            check_snapshot(text + "\n## Afterword\n\nMore.\n", snap)

    def test_empty_binding_list_fails(self, text):
        with pytest.raises(AssertionError, match="never passes"):
            check_numbers(text, ())

    def test_reasonless_allow_entry_fails(self, text):
        with pytest.raises(AssertionError, match="no reason"):
            _allow_spans(text, ((None, "p50", " "),))

    @pytest.mark.parametrize("mod,name,value", [
        ("lgk.privacy.floor", "DEFAULT_FLOOR", 12),
        ("lgk.sync_limits", "DEFAULT_SYNC_PER_SOURCE_HEADROOM", 3),
        ("lgk.sync_limits", "CONNECTIONS_PER_SYNC_FLOOR", 20),
        ("lgk.sync_limits", "SYNCED_TABLES_PER_SOURCE", 25),
        ("lgk.walks.sql", "MAX_SEEDS_DEFAULT", 100),
        ("lgk.walks.sql", "MAX_SEEDS_CAP", 500),
        ("lgk.walks.sql", "WEB_MAX_DEPTH", 3),
        ("lgk.mapping", "MAX_WALK_KEYS", 5),
        ("lgk.runtime", "ENVIRONMENT_VERSION", "4"),
        ("lgk.enrich.methods", "SIMILARITY_TOP_K", 30),
        ("lgk.app.generate", "DEFAULT_TOP_N", 20),
        ("lgk.steps.advise", "DEFAULT_MAX_DEPTH", 5),
        ("lgk.steps.measure", "DEFAULT_SEEDS", 30),
        ("lgk.steps.measure", "DEFAULT_MAX_DEPTH", 3),
        ("lgk.steps.measure", "DEFAULT_SIMILAR_TOP_N", 5),
        ("lgk.measure.compare", "DEFAULT_SAMPLES", 100),
        ("lgk.advisor", "MIN_SAMPLES", 30),
        ("lgk.advisor", "DEFAULT_WARMUP", 5),
        ("lgk.measure.reference", "DEFAULT_MAX_ROWS", 1_000_000),
    ])
    def test_code_constant_change_is_caught(self, text, monkeypatch, mod, name, value):
        monkeypatch.setattr(importlib.import_module(mod), name, value)
        with pytest.raises(AssertionError):
            check_number_coverage(text)

    @pytest.mark.parametrize("mod,name,value", [
        ("lgk.keys", "CHOICES", ("fixed", "accept")),  # hostile: the two choices swap meaning
        ("lgk.keys", "KEY_DECISIONS_FILE", "orphans.json"),
        ("lgk.steps.measure", "DEFAULT_SHAPES", ("tree", "web", "similar")),
        ("lgk.measure.record", "FRESHNESS", {"status": "pending", "reason": "measured nightly"}),
        ("lgk.measure.record", "FIRST_SERVE_STEPS", ("preflight", "advise")),
    ])
    def test_code_claim_change_is_caught(self, text, monkeypatch, mod, name, value):
        monkeypatch.setattr(importlib.import_module(mod), name, value)
        with pytest.raises(Exception):
            check_claims(text)

    def test_owner_setup_code_change_is_caught(self, text, monkeypatch):
        pf = importlib.import_module("lgk.steps.preflight")
        monkeypatch.setattr(pf, "KIT_ROLES", ("reader",))
        with pytest.raises(AssertionError):
            check_owner_setup(text)

    @pytest.mark.parametrize("attr,value", [("SUCCESS", "succeeded"), ("FAILED", "failure")])
    def test_status_constant_change_is_caught(self, text, monkeypatch, attr, value):
        from lgk import runner

        class Fake:
            pass

        fake = type("RS", (), {a: type("V", (), {"value": getattr(runner.RunStatus, a).value})()
                               for a in ("SUCCESS", "FAILED", "NOT_RUN", "IN_PROGRESS")})
        setattr(fake, attr, type("V", (), {"value": value})())
        monkeypatch.setattr(runner, "RunStatus", fake)
        with pytest.raises(AssertionError):
            check_status_words(text)

    def test_preflight_constant_change_is_caught(self, text, monkeypatch):
        from lgk.steps import preflight

        monkeypatch.setattr(preflight, "BLOCKED", "refused")
        with pytest.raises(AssertionError):
            check_status_words(text)

    def test_signature_default_change_is_caught(self, text, monkeypatch):
        from lgk.walks import lookups

        orig = lookups.similar_jaccard_sql
        monkeypatch.setattr(lookups, "similar_jaccard_sql", lambda served_schema, top_n=12, top_k=20: orig(served_schema, top_n, top_k))
        with pytest.raises(AssertionError):
            check_number_coverage(text)

    def test_signature_rename_is_caught(self, text, monkeypatch):
        from lgk.steps import privacy

        def resolve_flag(run_folder, table, column, choice, who=None, reason=None, new_class=None):
            pass

        monkeypatch.setattr(privacy, "resolve_flag", resolve_flag)
        with pytest.raises(AssertionError):
            check_python_blocks(text)

    def test_walk_direction_default_change_is_caught(self, text, monkeypatch):  # prover C1
        from lgk.steps import walks

        orig = walks.get_walk_query
        monkeypatch.setattr(walks, "get_walk_query", lambda start_node_id, depth, served_schema, privacy_floor=None,
                            top_n=25, folded=None, run_folder=None, sort_method=None, direction="both",
                            edge_types=None, seed=None, stop=None, max_seeds=200: orig)
        with pytest.raises(AssertionError):
            check_claims(text)

    def test_closure_default_on_is_caught(self, text, monkeypatch):  # prover C3
        from lgk.steps import provision

        monkeypatch.setattr(provision, "serve_closure", lambda config: config.get("serve_closure", True))
        with pytest.raises(AssertionError):
            check_claims(text)

    def test_app_timeout_is_read_from_the_executed_statement(self, tmp_path, monkeypatch):  # prover C10
        tmpl = REPO_ROOT / "src/lgk/app/template/backend/app/db.py.tmpl"
        src = tmpl.read_text()
        assert src.count("conn.execute(\"SET LOCAL statement_timeout = '") == 1
        fake = tmp_path / "src/lgk/app/template/backend/app"
        fake.mkdir(parents=True)
        (fake / "db.py.tmpl").write_text(src.replace("conn.execute(\"SET LOCAL statement_timeout = '5s'\")",
                                                     "conn.execute(\"SET LOCAL statement_timeout = '30s'\")"))
        monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
        assert _app_timeout() == "30"  # the docstring's '5s' no longer decides

    def test_app_template_timeout_change_is_caught(self, text, monkeypatch):
        monkeypatch.setattr(sys.modules[__name__], "_app_timeout", lambda: "10")
        with pytest.raises(AssertionError):
            check_number_coverage(text)


class TestCliSurface:
    @pytest.mark.skipif(shutil.which("databricks") is None, reason="no Databricks CLI on PATH")
    @pytest.mark.parametrize("key", [k for k in CLI_SURFACE if k[0] == "databricks"])
    def test_surface_matches_the_installed_cli(self, key):
        out = subprocess.run([*key, "--help"], capture_output=True, text=True, timeout=30).stdout
        usage = re.search(r"Usage:\n\s+(.*)", out).group(1)
        flags = set(re.findall(r"^\s+(?:(-\w), )?(--[\w-]+)", out, re.M))
        known = {f for pair in flags for f in pair if f}
        npos, valued, switches = CLI_SURFACE[key]
        assert set(valued) | switches <= known, (key, known)
        listed = len(re.findall(r"\b[A-Z_]{2,}\b", usage.replace("[flags]", "")))
        assert listed == npos + JSON_BODY_POSITIONALS.get(key, 0), usage



# ---- the C13 bullet (owner ruling 2026-10-14)
OLD_C13_BULLET = ("- C13 walk speed: p50 and p95 for each shape and depth, from the server-side timer (EXPLAIN ANALYZE); "
                  "excludes planning, result transfer and network. `measure_samples` runs (default 200, at least 20) follow\n"
                  "  `measure_warmup` runs (default 3).")  # the bullet at 6b3456d
GOOD_C13_BULLET = ("- C13 walk speed: the client p95 for each shape and depth, timed on a warm connection with prepared\n"
                   "  statements off; Postgres execution time (EXPLAIN ANALYZE) is one part of the means breakdown.\n"
                   "  `measure_samples` runs (default 200, at least 20) follow `measure_warmup` runs (default 3).")


def _with_c13(text: str, bullet: str) -> str:
    m = re.search(C13_BIND.pattern, text, re.S)
    assert m, "no C13 bullet in SKILL.md"
    return text[:m.start()] + bullet + text[m.end():]


class TestC13Bullet:
    def test_near_miss_passes(self, text):
        check_numbers(_with_c13(text, GOOD_C13_BULLET), (C13_BIND,))  # a line break inside "prepared statements"

    @pytest.mark.parametrize("old,new", [
        (GOOD_C13_BULLET, OLD_C13_BULLET),                                           # the bullet at 6b3456d
        ("the client p95", "the p95"),                                               # the headline not named
        ("timed on a warm connection", "timed"),                                     # the connection not named
        ("prepared\n  statements off", "prepared\n  statements on"),                 # the wrong mode
        ("with prepared\n  statements off", "with caching off"),                     # prepared statements not named
        ("Postgres execution time", "the server-side timer"),                        # the old headline claim
    ])
    def test_hostile_bullet_fails(self, text, old, new):
        assert old in GOOD_C13_BULLET or old == GOOD_C13_BULLET
        bad = OLD_C13_BULLET if old == GOOD_C13_BULLET else GOOD_C13_BULLET.replace(old, new)
        with pytest.raises(AssertionError):
            check_numbers(_with_c13(text, bad), (C13_BIND,))

    def test_bullet_is_a_claim(self):
        assert C13_BIND in CLAIMS


if __name__ == "__main__" and "--regen" in sys.argv:
    SNAPSHOT.write_text(json.dumps(snapshot_of(SKILL.read_text()), indent=1) + "\n")
    print(f"wrote {SNAPSHOT} ({len(snapshot_of(SKILL.read_text())['sections'])} sections)")
