"""The public docs agree with the code.

Each check reads a shipped file and compares it with a value read from the code (the runner's step list, the
sync modes, the PageRank stop rule, the measure label), never with a copy typed here. Each check is a function,
so the hostile rows below can feed it a broken text and prove it fails.
"""
import hashlib
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

from lgk import metrics
from lgk.enrich import methods
from lgk.loaders import mondo
from lgk.measure import timing
from lgk.refresh import modes
from lgk.runner import RunFolder

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SKILL = ROOT / "skill/lakebase-graph-kit/SKILL.md"
KNOWN_LIMITS = ROOT / "docs/KNOWN_LIMITS.md"
LICENSE = ROOT / "LICENSE"
NOTICE = ROOT / "NOTICE"
PYPROJECT = ROOT / "pyproject.toml"
DATA_FLOW = "docs/readme-assets/data-flow.svg"

STEP_ITEM = re.compile(r"^\s*(\d+)\.\s+\*\*`?(\w+)`?\*\*", re.M)
OLD_C13_LABEL = "excludes network and connection time"  # the old C13 label, replaced by the client p95 label


def _j(*parts) -> str:
    """Join parts at run time, so a hostile reference or number is never written whole in this file."""
    return "".join(str(p) for p in parts)


def _read(path: Path) -> str:
    assert path.is_file(), f"{path.relative_to(ROOT)} does not exist"
    return path.read_text()


def _paragraphs(text: str):
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


# ------------------------------------------------------------------ checks (each raises AssertionError)

def check_step_list(text: str, steps) -> None:
    """The numbered `N. **step**` items, in file order, are the runner's steps, numbered 1..len(steps)."""
    items = STEP_ITEM.findall(text)
    assert [s for _, s in items] == list(steps), f"step list {[s for _, s in items]} != runner {list(steps)}"
    assert [int(n) for n, _ in items] == list(range(1, len(steps) + 1)), "steps are not numbered 1..N in order"


AFTER_SWITCH = re.compile(r"\bafter\s+`?switch\b|\bswitch`?\s*(?:,|;|->|→)?\s*(?:and\s+)?then\s+`?measure\b"
                          r"|\bswitch`?\s*(?:,|->|→)\s*`?measure\b", re.I)
BEFORE_SWITCH = re.compile(r"\bmeasure`?\s*(?:,|->|→)\s*(?:then\s+)?`?switch\b|\bmeasure\b[^.\n]{0,40}\bbefore\s+`?switch\b",
                           re.I)


def check_measure_after_switch(text: str) -> None:
    """`measure` is its own command after switch: a paragraph naming measure says it comes after switch, no
    sentence puts it before switch, and the doc names both measure files."""
    after = [p for p in _paragraphs(text) if re.search(r"\bmeasure\b", p, re.I) and AFTER_SWITCH.search(p)]
    assert after, "no paragraph places measure after switch"
    bad = BEFORE_SWITCH.search(text)
    assert not bad, f"measure placed before switch: {bad.group(0)!r}"
    for f in ("measure.jsonl", "measure_report.md"):
        assert f in text, f"{f} is not named"


def check_c13_label(text: str, label: str) -> None:
    """Every line that names C13 or the execution time uses the measure step's exact label; the old label is gone."""
    assert OLD_C13_LABEL not in text, "the replaced C13 label is still used"
    rows = [ln for ln in text.splitlines() if re.match(r"\s*\|\s*C13\s*\|", ln)]
    assert rows, "no C13 row"
    for ln in rows:
        assert label in ln, f"C13 row lacks the exact label: {ln.strip()[:120]!r}"


def check_c10_reference(text: str) -> None:
    """The C10 row names the reference the measure step uses now (S2: computed in Python), not a Spark one."""
    rows = [ln for ln in text.splitlines() if re.match(r"\s*\|\s*C10\s*\|", ln)]
    assert rows, "no C10 row"
    for ln in rows:
        assert "Spark reference" not in ln and "computed in Python" in ln, f"C10 row: {ln.strip()[:120]!r}"


def check_triggered_says_v1_refuses(text: str, error_name: str) -> None:
    """A paragraph that names triggered also says v1 supports snapshot only and names the refusal's error."""
    for p in _paragraphs(text):
        if re.search(r"\btriggered\b", p, re.I):
            low = p.lower()
            assert "v1" in low and "snapshot" in low and error_name in p, \
                f"offers triggered without the v1 refusal: {p.strip()[:120]!r}"


PRIVATE_REFS = [
    (re.compile(r"(?<![\w&])#\d+\b"), "an issue number"),
    (re.compile(r"\bT\d{1,3}[a-z]?\b"), "a ticket id"),
    (re.compile(r"\bF\d{2}\b"), "a follow-up id"),
    (re.compile(r"\b(?:issue|ticket|PR|follow-up)\s*#?\d+", re.I), "an issue or ticket number"),
    (re.compile(r"\b[A-Z]\d+ rule\b"), "a rule id"),
    (re.compile(re.escape("STUBS" + ".md")), "the private stubs file"),
]


def check_no_private_refs(text: str) -> None:
    for n, line in enumerate(text.splitlines(), 1):
        for pat, what in PRIVATE_REFS:
            assert not pat.search(line), f"line {n} holds {what}"


def check_sync_modes(text: str, v1_modes, all_modes, error_name: str) -> None:
    """KNOWN_LIMITS states the v1 rule as the code defines it: each v1 mode is named as supported, every quoted mode
    name is one the code knows, the refusal names its error, and the platform's ALTER refusal is never cited."""
    paras = [p for p in _paragraphs(text) if "sync_mode" in p]
    assert paras, "no paragraph on sync_mode"
    body = "\n".join(paras)
    for m in v1_modes:
        assert re.search(rf"[`\"]{m}[`\"]", body), f"v1 mode {m!r} is not named"
    for name in re.findall(r"[`\"]([a-z_]+)[`\"]", body):
        assert name in all_modes or name == "sync_mode", f"{name!r} is not a sync mode the code knows"
    assert error_name in body, "the refusal's error is not named"
    assert not re.search(r"\bALTER\b|S" r"-alter|SET TBLPROPERTIES", text), "cites the platform's ALTER refusal"


def check_pagerank_rule(text: str, max_iter: int, tol: float, damping: float) -> None:
    """The PageRank stop rule and the final_l1 label and bound, read from the enrich code."""
    paras = [p for p in _paragraphs(text) if "PageRank" in p]
    assert paras, "no PageRank paragraph"
    body = "\n".join(paras)
    assert re.search(rf"\b{max_iter} iterations\b", body), f"the {max_iter}-iteration cap is not stated"
    floats = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?e-\d+\b", body)]
    assert tol in floats, f"the stop tolerance {tol:g} is not stated"
    assert "L1 change in the last iteration" in body, "final_l1 is not labelled"
    assert f"{damping:g}/{1 - damping:.2f}" in body, "the final_l1 bound is not stated"
    body = check_pagerank_run(body, max_iter, damping)
    rest = re.sub(rf"\b{max_iter} iterations\b|{re.escape(f'{damping:g}/{1 - damping:.2f}')}|\bL1\b|\bfinal_l1\b",
                  " ", body)
    rest = re.sub(r"\b\d+(?:\.\d+)?e-\d+\b", lambda m: " " if float(m.group(0)) == tol else m.group(0), rest)
    assert not re.search(r"\d", rest), "a per-graph number is stated before the dated run"


def _ph(name: str) -> str:
    return "{{" + "GK-" + name + "}}"


# The per-graph PageRank sentence (owner 2026-10-20): placeholders until the dated run, then the fill's values. The
# final L1 change and the bound have 3 significant digits and a short exponent; the bound is damping/(1 - damping)
# times the final L1 change; a capped run stopped at the iteration cap.
PAGERANK_RUN = ("On the {scale} test graph, PageRank {stop}: the L1 change in the last iteration was {l1}, so each "
                "score is within {bound} of the converged answer.")
GROUPED = r"[1-9]\d{0,2}(?:,\d{3})*"
SHORT_FLOAT = r"[1-9](?:\.\d*[1-9])?e-[1-9]\d*"


def check_pagerank_run(body: str, max_iter: int, damping: float) -> str:
    """The per-graph sentence is in the PageRank paragraph; returns the paragraph without it."""
    said = " ".join(body.split())
    held = PAGERANK_RUN.format(scale=_ph("C17-kl-scale"), stop=_ph("C17-kl-stop"), l1=_ph("C17-kl-final-l1"),
                               bound=_ph("C17-kl-bound"))
    if held in said:
        return said.replace(held, " ")
    filled = re.escape(PAGERANK_RUN).replace(r"\{scale\}", rf"{GROUPED}-node, {GROUPED}-edge").replace(
        r"\{stop\}", r"(?:converged|stopped at the (?P<it>\d+)-iteration cap without converging)").replace(
        r"\{l1\}", rf"(?P<l1>{SHORT_FLOAT})").replace(r"\{bound\}", rf"(?P<bound>{SHORT_FLOAT})")
    m = re.search(filled, said)
    assert m, f"the PageRank paragraph has no per-graph sentence {PAGERANK_RUN!r}"
    assert m.group("it") is None or int(m.group("it")) == max_iter, f"a capped run stops at {max_iter} iterations"
    want = damping / (1 - damping) * float(m.group("l1"))
    assert abs(float(m.group("bound")) - want) <= 0.01 * want, "the bound is not damping/(1 - damping) times final_l1"
    return said[:m.start()] + " " + said[m.end():]


# Owner ruling 2026-10-14: LICENSE is the apache.org text unchanged, appendix included; the holder line goes in NOTICE.
# SHA-256 of https://www.apache.org/licenses/LICENSE-2.0.txt, read 2026-10-14.
APACHE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
APACHE_MARKS = ("Apache License", "Version 2.0, January 2004",
                "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION", "END OF TERMS AND CONDITIONS",
                "APPENDIX: How to apply the Apache License to your work.", "Copyright [yyyy] [name of copyright owner]")
HOLDER = "Copyright 2026 Databricks, Inc."


def check_license(data: bytes, sha256: str = APACHE_SHA256) -> None:
    """LICENSE is the Apache-2.0 text byte for byte: no holder line filled in, the appendix kept."""
    text = data.decode("utf-8")
    missing = [m for m in APACHE_MARKS if m not in text]
    assert not missing, f"not the verbatim Apache-2.0 text: missing {missing}"
    holders = [ln.strip() for ln in text.splitlines() if re.match(r"\s*Copyright\b", ln)
               and "[name of copyright owner]" not in ln]
    assert not holders, f"LICENSE stays verbatim; the holder line goes in NOTICE: {holders}"
    assert hashlib.sha256(data).hexdigest() == sha256, "LICENSE differs from the apache.org text (SHA-256)"


def dependency_names(data: dict):
    """Every package pyproject installs or builds with: dependencies, every optional group, dependency groups and
    the build requirements."""
    reqs = list(data["project"].get("dependencies", []))
    for group in data["project"].get("optional-dependencies", {}).values():
        reqs += group
    for group in data.get("dependency-groups", {}).values():
        reqs += [r for r in group if isinstance(r, str)]
    reqs += data.get("build-system", {}).get("requires", [])
    return sorted({re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", r).group(0) for r in reqs})


NOTICE_ITEM = re.compile(r"Third-party material: ([^.\n]+)\.")


def check_notice(text: str, deps, screen, license_url: str) -> None:
    """NOTICE: the holder line once, in the first paragraph; then only third-party material copied into the repo,
    one `Third-party material: <name>.` paragraph each, each named in the dataset screen; MONDO with its licence;
    no installed dependency (those are not redistributed)."""
    lines = [ln.strip() for ln in text.splitlines()]
    assert lines.count(HOLDER) == 1, f"the holder line {HOLDER!r} is not there exactly once"
    other = [ln for ln in lines if ln != HOLDER and re.search(r"Databricks\s*,?\s*Inc|Copyright\b.*Databricks", ln)]
    assert not other, f"another Databricks holder line: {other}"
    paras = _paragraphs(text)
    assert HOLDER in paras[0], "the holder line is not in the first paragraph"
    items = []
    for p in paras[1:]:
        m = NOTICE_ITEM.match(p.strip())
        assert m, f"a NOTICE paragraph is not a third-party item: {p.strip()[:80]!r}"
        items.append((m.group(1).strip(), p))
    mondo_paras = [p for name, p in items if "MONDO" in name]
    assert mondo_paras and "CC BY 4.0" in mondo_paras[0] and license_url in mondo_paras[0], "MONDO attribution missing"
    if screen is not None:  # the dataset screen does not ship: docs/release/test_release_docs.py passes it
        for name, _ in items:
            assert name in screen, f"NOTICE names {name!r}, which the dataset screen does not"
    for dep in deps:
        assert not re.search(rf"(?<![\w-]){re.escape(dep)}(?![\w-])", text, re.I), \
            f"NOTICE names the installed dependency {dep!r}"


# The search for code that writes literal rows (owner ruling 2026-10-14: the screen lists what a search of the
# shipping code finds, and quotes the search). The shipping code roots are the manifest's code section.
SHIP_CODE = ("src", "tests", "notebooks", "conftest.py", "tools/checks", "tools/package_kit.py",
             "tools/metrics_summary.py")
ROW_SEARCH = r'VALUES *([({"]|\\n|$)'  # SQL VALUES then a row, a brace, a quote, an escaped newline or the line end
SEARCH_CMD = f"git grep -l -E '{ROW_SEARCH}' -- " + " ".join(SHIP_CODE)
SCREEN_PATH = re.compile(r"`((?:src|tests|notebooks|tools)/[\w./-]+|conftest\.py)`")


def search_rows(root: Path):
    """Run SEARCH_CMD (read-only) in root: the shipping files that hold literal rows."""
    p = subprocess.run(["git", "-C", str(root), "grep", "-l", "-E", ROW_SEARCH, "--", *SHIP_CODE],
                       capture_output=True, text=True)
    assert p.returncode in (0, 1), f"the row search failed: {p.stderr.strip()[:200]}"
    return sorted(p.stdout.split())


def loader_paths(root: Path):
    return sorted(str(p.relative_to(root)) for p in (root / "src/lgk/loaders").glob("*.py") if p.name != "__init__.py")


def check_dataset_screen(text: str, found, loaders, readme: str, linked=None) -> None:
    """The screen quotes the search in a fenced block, lists every path it finds and every loader, plus MONDO and
    its licence; every path it names exists; every generator the README names is in it. `linked` maps a linked file
    to the paths whose rows live there: the screen links it, never names those paths, and the link covers them."""
    fenced = re.findall(r"```[^\n]*\n(.*?)```", text, re.S)
    assert any(SEARCH_CMD in block for block in fenced), "the screen does not quote the search in a fenced block"
    for target, paths in (linked or {}).items():
        assert re.search(r"\]\(" + re.escape(target) + r"\)", text), f"the screen does not link {target}"
        still = sorted(set(SCREEN_PATH.findall(text)) & set(paths))
        assert not still, f"the screen still names {still}; their rows live in {target}"
    named = set(SCREEN_PATH.findall(text)) | {p for paths in (linked or {}).values() for p in paths}
    missing = sorted((set(found) | set(loaders)) - named)
    assert not missing, f"the screen does not list {missing}"
    for must in ("MONDO", "CC BY 4.0", "src/lgk/loaders/mondo.py", "src/lgk/attribution.py", "tests/test_mondo.py"):
        assert must in text, f"{must} is not in the dataset screen"
    for path in named:
        assert (ROOT / path).exists(), f"{path} named in the screen does not exist"
    in_readme = {p for p in SCREEN_PATH.findall(readme) if p in set(found) | set(loaders)}
    in_readme |= {f"src/lgk/loaders/{m}.py" for m in re.findall(r"\blgk\.loaders\.(\w+)", readme)}
    assert not in_readme - named, f"the README names generators the screen does not list: {sorted(in_readme - named)}"


# ---- README numbers (owner ruling 2026-10-14): the claims gate's allow file binds each of these to a test below.
def check_readme_numbers(text: str, n_steps: int, max_iter: int, tol: float, py_min: str, spdx: str) -> None:
    steps = [int(n) for n in re.findall(r"\b(\d+) steps\b", text)]
    assert steps, "the README does not state the step count"
    assert all(n == n_steps for n in steps), f"step counts {steps} != {n_steps} (RunFolder.STEPS)"
    iters = [int(n) for n in re.findall(r"\b(\d+) iterations\b", text)]
    assert all(n == max_iter for n in iters), f"iteration counts {iters} != the PageRank cap {max_iter}"
    tols = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?e-\d+\b", text)]
    assert all(x == tol for x in tols), f"tolerances {tols} != the PageRank tolerance {tol:g}"
    pys = re.findall(r"\bPython (\d+\.\d+)\b", text)
    assert all(v == py_min for v in pys), f"Python versions {pys} != requires-python {py_min}"
    lics = re.findall(r"\bApache-(\d+\.\d+)\b|\bApache License,? Version (\d+\.\d+)\b", text)
    assert all((a or b) == spdx.split("-")[1] for a, b in lics) and spdx == "Apache-2.0", \
        f"licence versions {lics} != pyproject {spdx}"


def _requires_python_min(data: dict) -> str:
    return re.search(r">=\s*(\d+\.\d+)", data["project"]["requires-python"]).group(1)


def _spdx(data: dict) -> str:
    lic = data["project"].get("license")
    return lic if isinstance(lic, str) else (lic or {}).get("text")


# ---- README words (owner ruling 2026-10-14): say what the recorder does, never a guarantee about health data, and
# no banned word of the post register. The words are assembled from parts, so this file passes the repo-wide check.
_HEALTH = "P" + "HI"
README_BANNED = re.compile(r"\b(?:" + "|".join([r"cust" + r"omers?", r"P" + r"OCs?", r"real" + r"\s+" + r"data",
                                                 _HEALTH, "protected" + r"\s+health\s+" + "information"]) + r")\b",
                           re.I)


def check_readme_metrics_text(text: str, refused: str) -> None:
    bad = [n for n, line in enumerate(text.splitlines(), 1) if README_BANNED.search(line)]
    assert not bad, f"README lines {bad} hold a banned word"
    for p in _paragraphs(text):
        if "LGK_METRICS_FILE" in p:
            assert refused in p, f"a paragraph on LGK_METRICS_FILE does not name the refusal ({refused})"


def check_pyproject_license(data: dict) -> None:
    lic = data["project"].get("license")
    value = lic if isinstance(lic, str) else (lic or {}).get("text")
    assert value == "Apache-2.0", f"license field is {lic!r}"
    assert [a.get("name") for a in data["project"].get("authors", [])] == ["Databricks"], "authors changed"


# ------------------------------------------------------------------ the shipped files

def test_readme_step_list_equals_runner_steps():
    check_step_list(_read(README), RunFolder.STEPS)


@pytest.mark.parametrize("doc", [README, SKILL], ids=lambda p: p.name)
def test_every_doc_puts_measure_after_switch_as_its_own_command(doc):
    text = _read(doc)
    check_measure_after_switch(text)
    assert "measure" not in [s for _, s in STEP_ITEM.findall(text)], "measure is listed as a runner step"


def test_c13_uses_the_measure_label():
    # the SA manual's C13 and C10 rows are checked in docs/release/test_release_docs.py (it does not ship)
    assert OLD_C13_LABEL not in _read(SKILL), "the replaced C13 label is still used"


def test_known_limits_has_no_private_refs():
    check_no_private_refs(_read(KNOWN_LIMITS))
    assert "docs/KNOWN_LIMITS.md" in _read(README), "the README does not link KNOWN_LIMITS.md"


@pytest.mark.parametrize("doc", [README, SKILL], ids=lambda p: p.name)
def test_shipping_docs_have_no_private_refs(doc):
    check_no_private_refs(_read(doc))


def test_known_limits_names_the_supported_sync_mode_from_modes():
    text = _read(KNOWN_LIMITS)
    check_sync_modes(text, modes.V1_SYNC_MODES, modes.SYNC_MODES, modes.UnsupportedSyncModeError.__name__)
    check_pagerank_rule(text, methods.PAGERANK_MAX_ITER, methods.PAGERANK_TOLERANCE, methods.PAGERANK_DAMPING)


@pytest.mark.parametrize("doc", [README, SKILL], ids=lambda p: p.name)
def test_no_doc_offers_triggered_without_saying_v1_refuses_it(doc):
    check_triggered_says_v1_refuses(_read(doc), modes.UnsupportedSyncModeError.__name__)


def test_notice_holds_the_holder_and_only_copied_third_party_material():
    # each item NOTICE names is also checked against the dataset screen in docs/release/test_release_docs.py
    check_notice(_read(NOTICE), dependency_names(tomllib.loads(_read(PYPROJECT))), None, mondo.LICENSE_URL)


def test_license_is_the_apache_text_verbatim():
    assert LICENSE.is_file(), "LICENSE does not exist"
    check_license(LICENSE.read_bytes())
    check_pyproject_license(tomllib.loads(_read(PYPROJECT)))


def test_readme_step_count_equals_runner_steps():
    data = tomllib.loads(_read(PYPROJECT))
    check_readme_numbers(_read(README), len(RunFolder.STEPS), methods.PAGERANK_MAX_ITER, methods.PAGERANK_TOLERANCE,
                         _requires_python_min(data), _spdx(data))


def test_readme_pagerank_numbers_equal_methods():
    text = _read(README)
    for n in re.findall(r"\b(\d+) iterations\b", text):
        assert int(n) == methods.PAGERANK_MAX_ITER, f"{n} iterations != the cap in src/lgk/enrich/methods.py"
    for x in re.findall(r"\b\d+(?:\.\d+)?e-\d+\b", text):
        assert float(x) == methods.PAGERANK_TOLERANCE, f"{x} != the tolerance in src/lgk/enrich/methods.py"


def test_readme_python_and_licence_equal_pyproject():
    data = tomllib.loads(_read(PYPROJECT))
    text = _read(README)
    assert all(v == _requires_python_min(data) for v in re.findall(r"\bPython (\d+\.\d+)\b", text))
    lics = re.findall(r"\bApache-(\d+\.\d+)\b|\bApache License,? Version (\d+\.\d+)\b", text)
    assert all((a or b) == "2.0" for a, b in lics) and _spdx(data) == "Apache-2.0"


def test_readme_says_what_the_metrics_recorder_refuses():
    check_readme_metrics_text(_read(README), metrics.MetricRefused.__name__)


def test_readme_is_the_release_readme():
    text = _read(README)
    assert "skeleton" not in text.lower(), "the README still calls itself a skeleton"
    assert DATA_FLOW in text and (ROOT / DATA_FLOW).is_file(), "the data-flow diagram is missing"
    for heading in ("Quick start", "Prerequisites"):
        assert re.search(rf"^##+ {heading}\b", text, re.M), f"no {heading} section"


# ------------------------------------------------------------------ hostile rows: each check must fail

GOOD_STEPS = "\n".join(f"{i}. **{s}** - x" for i, s in enumerate(RunFolder.STEPS, 1))


def _swap(text, a, b):
    return text.replace(f"**{a}**", "@@").replace(f"**{b}**", f"**{a}**").replace("@@", f"**{b}**")


@pytest.mark.parametrize("bad", [
    _swap(GOOD_STEPS, "provision", "enrich"),                         # two steps swapped
    "\n".join(GOOD_STEPS.splitlines()[:-1]),                          # 12 of 13
    GOOD_STEPS.replace("9. **switch**", "9. **switch**\n10. **measure**"),  # measure listed as a step
    GOOD_STEPS.replace("1. **preflight**", "2. **preflight**"),       # numbering broken
], ids=["swapped", "one-missing", "measure-as-step", "numbering"])
def test_hostile_step_list_fails(bad):
    check_step_list(GOOD_STEPS, RunFolder.STEPS)  # the good list passes
    with pytest.raises(AssertionError):
        check_step_list(bad, RunFolder.STEPS)


def test_hostile_measure_before_switch_fails():
    good = "Run measure after switch. It writes measure.jsonl and measure_report.md."
    check_measure_after_switch(good)
    for bad in ("Run measure, then switch. It writes measure.jsonl and measure_report.md.",
                "Run measure after switch. It writes measure.jsonl and measure_report.md. Or run measure before "
                "switch.",
                "Run measure after switch.",  # no output files
                "Run measure. It writes measure.jsonl and measure_report.md."):  # never placed after switch
        with pytest.raises(AssertionError):
            check_measure_after_switch(bad)


def test_hostile_c13_and_c10_rows_fail():
    good = f"| C13 | Walk speed; breakdown: {timing.EXECUTION_LABEL} |\n| C10 | a reference computed in Python |"
    check_c13_label(good, timing.EXECUTION_LABEL)
    check_c10_reference(good)
    with pytest.raises(AssertionError):
        check_c13_label(f"| C13 | timed on the server ({OLD_C13_LABEL}) |", timing.EXECUTION_LABEL)
    with pytest.raises(AssertionError):
        check_c13_label("| C13 | Postgres execution time (EXPLAIN ANALYZE) |", timing.EXECUTION_LABEL)  # label cut
    with pytest.raises(AssertionError):
        check_c10_reference("| C10 | Walk answers equal a Spark reference |")


@pytest.mark.parametrize("bad", ["Set `sync_mode: TRIGGERED` for an incremental sync.",
                                 "Triggered sync is supported in v1 for streaming sources.",
                                 "With sync_mode triggered, v1 refuses: use snapshot."])  # no error named
def test_hostile_triggered_offer_fails(bad):
    err = modes.UnsupportedSyncModeError.__name__
    check_triggered_says_v1_refuses(f"v1 supports `snapshot` only; `triggered` raises {err}.", err)
    with pytest.raises(AssertionError):
        check_triggered_says_v1_refuses(bad, err)


@pytest.mark.parametrize("bad", [_j("See #", 15, " for the fix."), _j("Done in T", 12, "."),
                                 _j("Tracked as follow-up F", "05", "."), _j("Closed by issue ", 24, "."),
                                 _j("Listed in STUBS", ".md."), _j("acceptance tests (issue #", 4, ")"),
                                 _j("Node types and patient data (the A", 5, " rule)."), _j("set walk (T", 13, "b)")],
                         ids=["hash-number", "ticket", "follow-up", "issue-n", "stubs", "issue-hash", "rule-id",
                              "ticket-suffix"])
def test_hostile_private_ref_fails(bad):
    check_no_private_refs(_j("No private reference here; HTML &#", 160, "; is fine. A1 is a cell, not a rule."))
    with pytest.raises(AssertionError):
        check_no_private_refs(bad)


GOOD_KL = (f'v1 supports `sync_mode` "snapshot" only (unset means "snapshot"); preflight and provision refuse every '
           f"other value with {modes.UnsupportedSyncModeError.__name__}.\n\n"
           "PageRank stops after at most 50 iterations, or when the L1 change is below 1e-6. final_l1 is the "
           "L1 change in the last iteration; the distance to the fixed point can be up to 0.85/0.15 times larger. "
           + PAGERANK_RUN.format(scale=_ph("C17-kl-scale"), stop=_ph("C17-kl-stop"), l1=_ph("C17-kl-final-l1"),
                                 bound=_ph("C17-kl-bound")))
FILLED_KL = GOOD_KL.split("On the ")[0] + PAGERANK_RUN.format(
    scale="1,000,000-node, 1,519,226-edge", stop="stopped at the 50-iteration cap without converging",
    l1="2.74e-6", bound="1.55e-5")


def test_hostile_known_limits_rows_fail(monkeypatch):
    err = modes.UnsupportedSyncModeError.__name__
    check_sync_modes(GOOD_KL, modes.V1_SYNC_MODES, modes.SYNC_MODES, err)
    check_pagerank_rule(GOOD_KL, methods.PAGERANK_MAX_ITER, methods.PAGERANK_TOLERANCE, methods.PAGERANK_DAMPING)
    for bad in (GOOD_KL.replace('only (unset', 'or "continuous" (unset'),           # a mode the code does not know
                GOOD_KL.replace('"snapshot"', "the default"),                        # the v1 mode not named
                GOOD_KL + "\n\nThe platform refuses the ALTER on a streaming table.",  # the ALTER refusal cited
                GOOD_KL.replace(f" with {err}", "")):                                # the error not named
        with pytest.raises(AssertionError):
            check_sync_modes(bad, modes.V1_SYNC_MODES, modes.SYNC_MODES, err)
    # read from the code, not typed here: a changed cap or tolerance fails the same text
    with pytest.raises(AssertionError):
        check_pagerank_rule(GOOD_KL, 40, methods.PAGERANK_TOLERANCE, methods.PAGERANK_DAMPING)
    with pytest.raises(AssertionError):
        check_pagerank_rule(GOOD_KL, methods.PAGERANK_MAX_ITER, 1e-5, methods.PAGERANK_DAMPING)
    with pytest.raises(AssertionError):
        check_pagerank_rule(GOOD_KL.replace("L1 change in the last iteration", "final error"),
                            methods.PAGERANK_MAX_ITER, methods.PAGERANK_TOLERANCE, methods.PAGERANK_DAMPING)
    # the per-graph numbers are the owner's sentence, as placeholders or filled (owner 2026-10-20)
    check_pagerank_rule(FILLED_KL, methods.PAGERANK_MAX_ITER, methods.PAGERANK_TOLERANCE, methods.PAGERANK_DAMPING)
    converged = FILLED_KL.replace("stopped at the 50-iteration cap without converging", "converged")
    check_pagerank_rule(converged, methods.PAGERANK_MAX_ITER, methods.PAGERANK_TOLERANCE, methods.PAGERANK_DAMPING)
    for bad in (GOOD_KL.split(" On the ")[0],                                                      # no per-graph sentence
                GOOD_KL + _j(" On the sample graph it stopped after ", 37, " iterations."),      # an iteration count
                GOOD_KL + _j(" Its final_l1 was ", "3.2e-", 7, "."),                               # a measured final_l1
                GOOD_KL + _j(" The graph has ", 1200, " nodes."),                                  # any other number
                FILLED_KL.replace("2.74e-6", _j("2.74e-0", 6)),                                     # a long exponent
                FILLED_KL.replace("1.55e-5", _j("1.6e-", 5)),                                       # a bound off the formula
                FILLED_KL.replace("50-iteration", _j(40, "-iteration")),                            # a cap the code lacks
                FILLED_KL.replace("1,000,000-node", _j(1000000, "-node")),                          # no separators
                FILLED_KL.replace("1,519,226-edge", "1,519,226-edge (1.5M)"),                       # a number beside it
                FILLED_KL.replace("stopped at the 50-iteration cap without converging", "converged") + _j(
                    " It ran ", 37, " iterations."),                                               # a count after it
                GOOD_KL.replace(_ph("C17-kl-stop"), _ph("C17-pr-converged"))):                     # another placeholder
        with pytest.raises(AssertionError):
            check_pagerank_rule(bad, methods.PAGERANK_MAX_ITER, methods.PAGERANK_TOLERANCE, methods.PAGERANK_DAMPING)


# A stand-in for the Apache text (the real text is not copied into this file): the check takes the hash to compare.
APACHE_STANDIN = "\n".join(APACHE_MARKS) + "\n"
_STANDIN_SHA = hashlib.sha256(APACHE_STANDIN.encode()).hexdigest()


@pytest.mark.parametrize("bad", [
    APACHE_STANDIN + HOLDER + "\n",                                                   # a holder line added
    APACHE_STANDIN.replace("[yyyy] [name of copyright owner]", "2026 Databricks Inc"),  # the placeholder filled in
    APACHE_STANDIN.split("APPENDIX:")[0],                                             # the appendix cut
    APACHE_STANDIN.replace("Version 2.0", "Version 1.1"),                              # another version
    "MIT License\n\n" + APACHE_STANDIN,                                               # another licence on top
    APACHE_STANDIN + " ",                                                             # one byte changed
], ids=["holder-added", "placeholder-filled", "no-appendix", "version", "mit", "one-byte"])
def test_hostile_license_fails(bad):
    check_license(APACHE_STANDIN.encode(), _STANDIN_SHA)
    with pytest.raises(AssertionError):
        check_license(bad.encode(), _STANDIN_SHA)


GOOD_SCREEN_ITEM = "MONDO"
GOOD_NOTICE = (f"lakebase-graph-kit\n{HOLDER}\n\nThird-party material: MONDO. The Mondo Disease Ontology, used under "
               f"CC BY 4.0 ({mondo.LICENSE_URL}).\n")


@pytest.mark.parametrize("bad", [
    GOOD_NOTICE.replace(HOLDER, "Copyright 2026 Databricks Inc"),                       # holder spelled differently
    GOOD_NOTICE.replace(HOLDER, HOLDER + "\n" + HOLDER),                                # holder twice
    GOOD_NOTICE + "\nCopyright Databricks, Inc. and others\n",                         # a second holder line
    GOOD_NOTICE + "\nThird-party material: networkx. BSD licence.\n",                  # an installed dependency
    GOOD_NOTICE + "\nThird-party material: Gene Ontology. CC BY 4.0.\n",               # an item the screen lacks
    GOOD_NOTICE + "\nThis product also bundles a font.\n",                             # an unlabelled paragraph
    GOOD_NOTICE.replace(mondo.LICENSE_URL, "https://example.com/licence"),              # the MONDO licence URL lost
    f"lakebase-graph-kit\n{HOLDER}\n",                                                 # MONDO missing
], ids=["holder-spelling", "holder-twice", "second-holder", "dependency", "not-in-screen", "unlabelled",
        "licence-url", "no-mondo"])
def test_hostile_notice_fails(bad):
    deps = dependency_names(tomllib.loads(_read(PYPROJECT)))
    assert "networkx" in deps, "the hostile row needs networkx to be a pyproject dependency"
    check_notice(GOOD_NOTICE, deps, GOOD_SCREEN_ITEM, mondo.LICENSE_URL)
    with pytest.raises(AssertionError):
        check_notice(bad, deps, GOOD_SCREEN_ITEM, mondo.LICENSE_URL)


def _good_screen(found, loaders):
    rows = "\n".join(f"| `{p}` | hand-written rows | clean |" for p in sorted(set(found) | set(loaders)))
    return (f"```\n{SEARCH_CMD}\n```\n\nMONDO, CC BY 4.0: `src/lgk/loaders/mondo.py`, `src/lgk/attribution.py`, "
            f"`tests/test_mondo.py`.\n\n{rows}\n")


def test_hostile_dataset_screen_fails():
    found, loaders = search_rows(ROOT), loader_paths(ROOT)
    assert found and "src/lgk/loaders/mondo.py" in loaders  # the search finds files today, so the rows below bite
    good = _good_screen(found, loaders)
    check_dataset_screen(good, found, loaders, "")
    for bad, readme in ((good.replace(f"```\n{SEARCH_CMD}\n```", SEARCH_CMD), ""),     # the search not fenced
                        (good.replace(SEARCH_CMD, SEARCH_CMD.replace(" notebooks", "")), ""),  # a different search
                        (good.replace(f"| `{found[0]}` |", "| (dropped) |"), ""),          # a found file not listed
                        (good.replace("MONDO, CC BY 4.0", "An ontology"), ""),              # MONDO dropped
                        (good + "\n`src/lgk/loaders/gone.py`\n", ""),                       # a path that does not exist
                        (good.replace("| `src/lgk/loaders/mondo.py` | hand-written rows | clean |\n", "")
                         .replace("`src/lgk/loaders/mondo.py`, ", ""), "")):                # a loader not listed
        with pytest.raises(AssertionError):
            check_dataset_screen(bad, found, loaders, readme)
    # rows that live in a linked file: the link covers them, and the screen may not also name them
    moved = good.replace(f"| `{found[0]}` | hand-written rows | clean |", "").replace("MONDO, CC", "[moved](moved.md)"
                                                                                       "\n\nMONDO, CC")
    check_dataset_screen(moved, found, loaders, "", {"moved.md": (found[0],)})                # near miss
    for bad in (moved.replace("[moved](moved.md)", "moved.md"),                                 # not linked
                moved + f"\nSee also `{found[0]}`.\n",                                         # still named
                moved.replace("(moved.md)", "(other.md)")):                                      # links another file
        with pytest.raises(AssertionError):
            check_dataset_screen(bad, found, loaders, "", {"moved.md": (found[0],)})
    with pytest.raises(AssertionError):
        check_dataset_screen(moved, found, loaders, "")                                         # no link map: missing
    # a generator the README names must be in the screen
    with pytest.raises(AssertionError):
        check_dataset_screen(good.replace(f"| `{found[0]}` |", "| (dropped) |"), found, loaders,
                             f"The example graph is written by `{found[0]}`.")


GOOD_README_NUMBERS = ("The runner has 13 steps. PageRank stops after at most 50 iterations or below 1e-6. "
                       "Needs Python 3.12. Licensed under Apache-2.0.")


@pytest.mark.parametrize("bad", [
    GOOD_README_NUMBERS.replace("13 steps", _j(12, " steps")),                  # a wrong step count
    GOOD_README_NUMBERS.replace("The runner has 13 steps. ", ""),               # no step count
    GOOD_README_NUMBERS.replace("50 iterations", _j(40, " iterations")),        # a wrong cap
    GOOD_README_NUMBERS.replace("1e-6", _j("1e-", 5)),                          # a wrong tolerance
    GOOD_README_NUMBERS.replace("Python 3.12", _j("Python 3.", 11)),            # a wrong Python
    GOOD_README_NUMBERS.replace("Apache-2.0", _j("Apache-", "1.1")),            # a wrong licence version
], ids=["steps", "no-steps", "cap", "tolerance", "python", "licence"])
def test_hostile_readme_numbers_fail(bad):
    args = (13, 50, 1e-6, "3.12", "Apache-2.0")
    check_readme_numbers(GOOD_README_NUMBERS, *args)
    with pytest.raises(AssertionError):
        check_readme_numbers(bad, *args)
    # read from the code, not typed here: the good text fails a changed step count
    with pytest.raises(AssertionError):
        check_readme_numbers(GOOD_README_NUMBERS, 14, *args[1:])


def test_hostile_readme_metrics_text_fails():
    refused = metrics.MetricRefused.__name__
    good = (f"Set `LGK_METRICS_FILE=<path>`. The recorder refuses ({refused}) a value that is not a finite number "
            "and a context key outside its fixed list.")
    check_readme_metrics_text(good, refused)
    for bad in ("Set `LGK_METRICS_FILE=<path>`. Values must be numbers and context keys come from a fixed list, "
                + _j("so no ", _HEALTH, " or token can reach the file."),            # the old sentence (a guarantee)
                "Set `LGK_METRICS_FILE=<path>`. Values must be numbers.",          # the refusal not named
                good + _j(" It never holds protected", " health information."),  # the guarantee, spelled out
                good + _j("\n\nThe kit turns a cust", "omer's gold schema into a graph."),   # README.md:7's word
                good + _j("\n\nAll code is free of cust", "omer references."),             # README.md:104's word
                good + _j("\n\nTry it on your real", " data first."),                       # another banned word
                good + _j("\n\nRun a P", "OC on it.")):                                     # another banned word
        with pytest.raises(AssertionError):
            check_readme_metrics_text(bad, refused)


@pytest.mark.parametrize("lic", [{"text": "Databricks"}, "Apache 2.0", {"text": "MIT"}, None])
def test_hostile_pyproject_license_fails(lic):
    check_pyproject_license({"project": {"license": "Apache-2.0", "authors": [{"name": "Databricks"}]}})
    check_pyproject_license({"project": {"license": {"text": "Apache-2.0"}, "authors": [{"name": "Databricks"}]}})
    with pytest.raises(AssertionError):
        check_pyproject_license({"project": {"license": lic, "authors": [{"name": "Databricks"}]}})
    with pytest.raises(AssertionError):
        check_pyproject_license({"project": {"license": "Apache-2.0", "authors": [{"name": "Someone"}]}})


# ------------------------------------------------------------------ review round 2026-10-14: README errors
# Four errors a reviewer found in the helper's README that no check above caught: a Quick start against an API that
# does not exist, an availability claim the switch step does not make, release statuses nobody read from a page, and
# a link to a file that does not ship. Each check reads the code or the ship list, never a copy typed here.

import ast  # noqa: E402
import importlib  # noqa: E402

PY_BLOCK = re.compile(r"^```python\n(.*?)^```", re.M | re.S)


def check_python_blocks(text: str) -> None:
    """Every ```python block compiles; every `lgk` import resolves (module and name); every method called on an
    instance of an imported class is an attribute of that class."""
    blocks = PY_BLOCK.findall(text)
    for block in blocks:
        tree = ast.parse(block)  # SyntaxError fails the check too
        names = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] == "lgk":
                        try:
                            importlib.import_module(a.name)
                        except ImportError as e:
                            raise AssertionError(f"README imports {a.name}, which does not exist") from e
            elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "lgk":
                try:
                    mod = importlib.import_module(node.module)
                except ImportError as e:
                    raise AssertionError(f"README imports from {node.module}, which does not exist") from e
                for a in node.names:
                    assert hasattr(mod, a.name), f"{node.module} has no {a.name}"
                    names[a.asname or a.name] = getattr(mod, a.name)
        instances = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name) and isinstance(names.get(node.value.func.id), type)):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        instances[t.id] = names[node.value.func.id]
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id in instances):
                cls = instances[node.func.value.id]
                assert hasattr(cls, node.func.attr), f"{cls.__name__} has no method {node.func.attr}"


# switch.py repoints the served views in one transaction and records the switch; it states nothing about
# availability, so the README may not either. The list is narrow on purpose: words that claim uptime.
AVAILABILITY_CLAIM = re.compile(r"\bdown-?time\b|\b(?:no|without)\s+interruption\b|\binstant(?:ly|aneous(?:ly)?)?\b"
                                r"|\bseamless(?:ly)?\b|\balways\s+(?:up|available|online)\b", re.I)


def check_switch_claims(text: str) -> None:
    bad = [n for n, line in enumerate(text.splitlines(), 1)
           if re.search(r"\bswitch\b", line, re.I) and AVAILABILITY_CLAIM.search(line)]
    assert not bad, f"README lines {bad} claim an availability property for switch that the code does not state"


# Release statuses live on the release-only feature page, where each one was read from a page; the README states none.
STATUS_WORD = re.compile(r"\bGA\b|\b(?i:generally\s+available|public\s+preview|private\s+preview|beta)\b")


def check_no_release_status(text: str) -> None:
    bad = [n for n, line in enumerate(text.splitlines(), 1) if STATUS_WORD.search(line)]
    assert not bad, f"README lines {bad} state a release status"


LINK = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|\bsrc=\"([^\"]+)\"")


def manifest_entries(text: str):
    """The ship list: the first field of each non-comment line; an entry ending in `/` ships everything under it."""
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line.split()[0])
    return out


def ships(path: str, entries) -> bool:
    return any(path == e or (e.endswith("/") and path.startswith(e)) for e in entries)


def relative_links(text: str):
    """Every relative link or image path in the text (anchors and URLs with a scheme left out), in order."""
    out = []
    for m in LINK.finditer(text):
        target = (m.group(1) or m.group(2)).split("#", 1)[0]
        if not target or re.match(r"[a-z][a-z0-9+.-]*:", target, re.I):  # an anchor, or a URL with a scheme
            continue
        out.append(target.removeprefix("./"))
    return out


def check_links_ship(text: str, entries) -> None:
    """Every relative link or image path in the README names a file the export ships."""
    bad = [t for t in relative_links(text) if not ships(t, entries)]
    assert not bad, f"README links files that do not ship: {bad}"


def test_readme_python_blocks_use_the_real_api():
    check_python_blocks(_read(README))


def test_readme_claims_nothing_for_switch_that_the_code_does_not_state():
    check_switch_claims(_read(README))


def test_readme_states_no_release_status():
    check_no_release_status(_read(README))


def test_every_readme_link_target_exists():
    # in the public repo every file present ships; the manifest check is in docs/release/test_release_docs.py
    missing = [t for t in relative_links(_read(README)) if not (ROOT / t).exists()]
    assert not missing, f"README links files that do not exist: {missing}"


GOOD_PY = ("```python\nfrom lgk.runner import LaptopRunner\n\n"
           "runner = LaptopRunner({\"graph_name\": \"g\", \"run_folder\": \"/tmp/r\"})\n"
           "runner.run_steps([])\n```\n")


@pytest.mark.parametrize("bad", [
    # the helper's Quick start: a config module that does not exist, and two methods LaptopRunner lacks
    "```python\nfrom lgk.runner import LaptopRunner\nfrom lgk.config import Config\n\nconfig = Config(graph_name=\"g\")\n"
    "runner = LaptopRunner(config)\nrunner.run_all()\nrunner.print_status()\n```\n",
    GOOD_PY.replace("lgk.runner", "lgk.runnr"),                       # a misspelt module
    GOOD_PY.replace("import LaptopRunner", "import LaptopRunner, CloudRunner"),  # a name the module lacks
    GOOD_PY.replace("runner.run_steps([])", "runner.run_all()"),      # a method that does not exist
    GOOD_PY.replace("import LaptopRunner", "import LaptopRunner as R").replace("= LaptopRunner(", "= R(")
    .replace("run_steps", "start"),                                   # an alias still binds the class
    GOOD_PY.replace("runner.run_steps([])", "runner.run_steps([]"),   # does not compile
    "```python\nimport lgk.nosuch\n```\n",                            # a plain import of a missing module
])
def test_hostile_python_block_fails(bad):
    check_python_blocks(GOOD_PY)
    check_python_blocks("```bash\nlgk.config run_all\n```\n")         # near miss: not a python block
    with pytest.raises((AssertionError, SyntaxError)):
        check_python_blocks(bad)


@pytest.mark.parametrize("bad", [
    "9. **switch** — Activates the new version and deactivates the old (zero downtime)",  # the helper's line
    "9. **switch** — moves readers to the new version instantly",
    "9. **switch** — a seamless cut-over",
    "Run switch: no interruption for readers.",
])
def test_hostile_switch_claim_fails(bad):
    check_switch_claims("9. **switch** — repoints the served views to the new version in one transaction")  # near miss
    check_switch_claims("The app has no downtime page.")   # near miss: a line that does not name switch
    with pytest.raises(AssertionError):
        check_switch_claims(bad)


@pytest.mark.parametrize("bad", [
    "| Feature | Release Status |\n|---|---|\n| Lakebase (Postgres on Databricks) | GA |",  # the helper's table
    "Databricks Apps (Public Preview)",
    "Lakebase is generally available.",
    "This needs a beta feature.",
    "Private preview only.",
])
def test_hostile_release_status_fails(bad):
    check_no_release_status("- Lakebase: https://docs.databricks.com/aws/en/oltp/\n- Unity Catalog, with a gauge")  # near miss
    with pytest.raises(AssertionError):
        check_no_release_status(bad)


@pytest.mark.parametrize("bad", [
    "See [Feature Status](docs/release/feature-status.md).",      # the helper's link: release/ does not ship
    "See [x](docs/release/x.md#top).",                            # an anchor does not make it ship
    "See [manual](./docs/SA_PILOT_MANUAL.md).",                   # a doc outside the ship list
    "<img src=\"docs/other/d.svg\">",                             # an HTML image
])
def test_hostile_link_fails(bad):
    entries = manifest_entries("# c\nREADME.md\ndocs/KNOWN_LIMITS.md\ndocs/readme-assets/\nsrc/ sha256:" + "0" * 64 + "\n")
    check_links_ship("[limits](docs/KNOWN_LIMITS.md) ![d](docs/readme-assets/data-flow.svg) [web](https://x.org/a.md)"
                     " [top](#quick-start)", entries)                # near misses
    with pytest.raises(AssertionError):
        check_links_ship(bad, entries)


# ================================================== owner review round (2026-10-14, S5 review items 1-10)

from lgk import runtime  # noqa: E402

# ---- each README step line states only what that step module's own summary states.
# A step's summary is the first paragraph of src/lgk/steps/<step>.py's docstring. IDs in parentheses (requirement and
# story numbers) are dropped first, so the check holds after S3 removes them. A line passes when each of its content
# words, after STOP_WORDS go and plurals and verb endings fold, is a word of that summary (or of its EXTRA_SUMMARY).
STOP_WORDS = frozenset("a an the and or of to in on at for from with by its it each one as into that which is are be "
                       "then them they this these".split())
# A word no summary holds but a line may use for any step. Each entry needs its reason; keep the list short.
ALLOW_WORDS = {
    "step": "every step line names a step",
}
# A word a line may NOT use unless that step's summary holds it, whatever ALLOW_WORDS says.
ABSOLUTE_WORDS = frozenset("all every always never automatic automatically performance zero instant instantly "
                           "seamless seamlessly".split())
# advise: the module's summary names the advisor; lgk.advisor's summary says what it does (propose and apply).
# privacy: the module's summary names the gate; lgk.privacy.flags and lgk.privacy.classify say what it writes and does.
EXTRA_SUMMARY = {"advise": ("lgk.advisor",), "privacy": ("lgk.privacy.flags", "lgk.privacy.classify")}
PRIVACY_TERMS = ("flags.json", "classif")  # the privacy line says what the step writes and that it classifies columns
# build runs SQL statements on the SQL warehouse (lgk.steps.build calls execute_sql); it never runs a pipeline, so the
# build line names none, whatever its summary says (owner 2026-10-15).
BUILD_NOT = re.compile(r"\bSDP\b|\bpipelines?\b|\bdeclarative\b", re.I)
ADVISE_VERBS = ("propose", "apply")  # the advisor builds what it proposes: advisor.apply (CREATE INDEX IF NOT EXISTS)
# switch: these words are allowed only while tests/test_rollback.py exists and has a test that pins rollback.
ROLLBACK_WORDS = frozenset("rollback roll back rolled earlier repoint repoints".split())
ROLLBACK_TESTS = ROOT / "tests/test_rollback.py"
ID_PAREN = re.compile(r"\((?:[A-Z]\d|user stor)[^)]*\)")


def _forms(word: str) -> set:
    """The word plus its forms without a plural or verb ending: tables/table, applied/apply, indexes/index."""
    w = word.lower()
    out = {w}
    if w.endswith(("ies", "ied")) and len(w) > 4:
        out.add(w[:-3] + "y")
    for end in ("es", "s", "ed", "d", "ing"):
        if w.endswith(end) and len(w) - len(end) >= 3:
            out.add(w[: -len(end)])
            if end == "ing":
                out.add(w[:-3] + "e")
    return out


def content_words(text: str):
    """Lower-case words of text, in order, without STOP_WORDS and one-letter tokens."""
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 1 and w not in STOP_WORDS]


def module_summary(module: str) -> str:
    """The first paragraph of a module's docstring (read from source, never imported), IDs dropped."""
    path = ROOT / "src" / Path(*module.split(".")).with_suffix(".py")
    doc = ast.get_docstring(ast.parse(path.read_text())) or ""
    return ID_PAREN.sub("", " ".join(doc.split("\n\n", 1)[0].split()))


def step_summaries(steps) -> dict:
    return {s: " ".join([module_summary(f"lgk.steps.{s}")] + [module_summary(m) for m in EXTRA_SUMMARY.get(s, ())])
            for s in steps}


def rollback_is_pinned(path: Path = ROLLBACK_TESTS) -> bool:
    """tests/test_rollback.py exists and defines a test whose name pins rollback."""
    if not path.is_file():
        return False
    tree = ast.parse(path.read_text())
    return any(isinstance(n, ast.FunctionDef) and n.name.startswith("test_") and "roll" in n.name for n in tree.body)


STEP_LINE = re.compile(r"^\s*\d+\.\s+\*\*`?(\w+)`?\*\*\s*(?:—|–|-|:)\s*(.*)$", re.M)


def check_step_lines(text: str, summaries: dict, rollback_pinned: bool, allow: dict = None) -> None:
    """Each `N. **step** — line` uses only words its step's summary holds (see STOP_WORDS, ALLOW_WORDS)."""
    allow = ALLOW_WORDS if allow is None else allow
    lines = {m.group(1): m.group(2) for m in STEP_LINE.finditer(text)}
    assert set(lines) == set(summaries), f"step lines {sorted(lines)} != steps {sorted(summaries)}"
    bad = {}
    for step, line in lines.items():
        words = content_words(line)
        summary = set().union(*(_forms(w) for w in content_words(summaries[step])))
        extra = []
        if not words:
            extra.append("(no words)")
        for w in words:
            if _forms(w) & summary:
                continue
            if w in ABSOLUTE_WORDS:
                extra.append(w)
            elif step == "switch" and w in ROLLBACK_WORDS:
                if not rollback_pinned:
                    extra.append(f"{w} (no rollback test)")
            elif w not in allow:
                extra.append(w)
        if step == "advise":
            for verb in ADVISE_VERBS:
                if not any(verb in _forms(w) for w in words):
                    extra.append(f"(does not say {verb})")
        if step == "privacy":
            extra += [f"(does not say {term})" for term in PRIVACY_TERMS if term not in line.lower()]
        if step == "build" and BUILD_NOT.search(line):
            extra.append("(names a pipeline; build runs SQL statements on the SQL warehouse)")
        if extra:
            bad[step] = extra
    assert not bad, f"README step lines say what their step's summary does not: {bad}"


def test_each_readme_step_line_states_only_its_step_summary():
    check_step_lines(_read(README), step_summaries(RunFolder.STEPS), rollback_is_pinned())


def test_the_switch_rollback_clause_is_pinned_by_the_rollback_tests():
    # the README's rollback clause rests on these tests; they must exist and name rollback
    assert rollback_is_pinned(), "tests/test_rollback.py is missing or has no rollback test"
    assert "def rollback(" in _read(ROOT / "src/lgk/steps/switch.py"), "switch has no rollback entry point"


# The summaries at 6b3456d, frozen, so the hostile rows below do not move when a docstring is reworded.
FROZEN_SUMMARIES = {
    "walks": "Walks: the public query helpers over the served views.",
    "teardown": "Whole-graph teardown: drop exactly what the run manifest names.",
    "enrich": "Enrich step: compute the enrichments the user ticked on the fixed menu (Spark batch).",
    "app": "App step: generate the search app folder and print its one deploy command.",
    "advise": "Advise step: the index advisor on the run's NEW version, before the switch. Index advisor: read the "
              "plan, propose indexes, apply them, re-time.",
    "switch": "Switch step: point the served views at one version, in one transaction.",
    "privacy": "Privacy step: the gate between the interview and build. flags.json: one flag per care-subject column, "
               "resolved one by one. Column classification for the privacy gate.",
    "build": "Build step: SDP to create nodes/edges Delta tables from graph_mapping.yaml.",  # the summary at 9d78d73
}
GOOD_LINES = {  # near misses: plural, verb ending and order differ from the summary
    "walks": "The public query helpers over the served views",
    "teardown": "Drops exactly what the run manifest names",
    "enrich": "Computes the enrichments ticked on the fixed menu (Spark batch)",
    "app": "Generates the search app folder and prints its one deploy command",
    "advise": "Runs the index advisor on the new version before the switch: proposes indexes and applies them",
    "switch": "Points the served views at one version, in one transaction; a rollback points them back at an earlier "
              "version",
    "privacy": "The gate between the interview and build: column classification, then `flags.json`, one flag per "
               "care-subject column, resolved one by one",
    "build": "Creates the `nodes` and `edges` Delta tables from `graph_mapping.yaml`",
}


def _lines(lines: dict) -> str:
    return "\n".join(f"{i}. **{s}** — {t}" for i, (s, t) in enumerate(lines.items(), 1))


def test_step_line_near_misses_pass():
    check_step_lines(_lines(GOOD_LINES), FROZEN_SUMMARIES, True)
    with_ids = dict(FROZEN_SUMMARIES, app=_j("App step (U", 33, ", U", 34, "): generate the search app folder and "
                                                                           "print its one deploy command."))
    check_step_lines(_lines(GOOD_LINES), {s: ID_PAREN.sub("", t) for s, t in with_ids.items()}, True)


@pytest.mark.parametrize("step,bad", [
    ("walks", "Generates hop queries and performance tests"),            # the README's line at 6b3456d
    ("teardown", "Cleans up all created objects"),                       # the README's line at 6b3456d
    ("enrich", "Computes scores (PageRank, degree, components, similarity, closure) and materialized views"),
    ("app", "(Optional) Generates a simple Databricks App"),             # the README's line at 6b3456d
    ("advise", "Analyzes query performance and recommends indexes"),     # the README's line at 6b3456d
    ("advise", "Runs the index advisor on the new version and proposes indexes"),  # proposes, never says it applies
    ("teardown", "Drops every object the run manifest names"),           # an absolute the summary lacks
    ("walks", "The public query helpers over the served views"
              ", always"),                                               # an absolute, even when allowed below
    ("switch", ""),                                                      # an empty line
    ("privacy", "The gate between the interview and build"),            # the README's line at 9d78d73: says no output
    ("privacy", "The gate between the interview and build; writes `flags.json`"),   # never says it classifies
    ("privacy", "Column classification for the gate between the interview and build"),  # never names flags.json
    ("build", "Creates the `nodes` and `edges` Delta tables from `graph_mapping.yaml` (SDP)"),  # line at 9d78d73
    ("build", "Creates the `nodes` and `edges` Delta tables from `graph_mapping.yaml` (Spark Declarative Pipelines)"),
])
def test_hostile_step_line_fails(step, bad):
    lines = dict(GOOD_LINES, **{step: bad})
    with pytest.raises(AssertionError):
        check_step_lines(_lines(lines), FROZEN_SUMMARIES, True)
    with pytest.raises(AssertionError):  # an allow list never admits an absolute or a word the line adds
        check_step_lines(_lines(lines), FROZEN_SUMMARIES, True, dict(ALLOW_WORDS, always="x", every="x"))


def test_hostile_switch_rollback_clause_without_its_test_fails(tmp_path):
    with pytest.raises(AssertionError):
        check_step_lines(_lines(GOOD_LINES), FROZEN_SUMMARIES, False)
    assert not rollback_is_pinned(tmp_path / "missing.py")
    (tmp_path / "t.py").write_text("def test_something_else():\n    assert 1\n")
    assert not rollback_is_pinned(tmp_path / "t.py")
    (tmp_path / "r.py").write_text("def test_rollback_serves_the_named_version():\n    assert 1\n")
    assert rollback_is_pinned(tmp_path / "r.py")


def test_hostile_step_line_set_fails():
    with pytest.raises(AssertionError):  # a step with no line
        check_step_lines(_lines({k: v for k, v in GOOD_LINES.items() if k != "app"}), FROZEN_SUMMARIES, True)


# ---- no internal word in the README.
INTERNAL_WORDS = re.compile(r"\b" + _j("de", "ck") + r"s?\b", re.I)


def check_no_internal_words(text: str) -> None:
    bad = [n for n, line in enumerate(text.splitlines(), 1) if INTERNAL_WORDS.search(line)]
    assert not bad, f"README lines {bad} use an internal word"


def test_readme_has_no_internal_words():
    check_no_internal_words(_read(README))


@pytest.mark.parametrize("bad", [_j("for citing in a ", "de", "ck"), _j("Slide ", "De", "cks"), _j("the ", "DE", "CK")])
def test_hostile_internal_word_fails(bad):
    check_no_internal_words(_j("the declared shape; checked ", "de", "cking"))  # near misses: other words
    with pytest.raises(AssertionError):
        check_no_internal_words(bad)


# ---- a License section and one support line.
# SUPPORT_LINE is the owner's example wording. PENDING OWNER CONFIRMATION (2026-10-14 ruling): change it here
# and in the README together when the owner confirms the exact text.
SUPPORT_LINE = "Provided for exploration; not formally supported by Databricks."
NOT_PRODUCT = re.compile(_j("not ", "a ", "Databricks ", "product"), re.I)  # never written about code Databricks owns


def _section(text: str, title: str) -> str:
    m = re.search(rf"^## {re.escape(title)}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    return m.group(1) if m else ""


def check_license_section(text: str, spdx: str = "Apache-2.0") -> None:
    sec = _section(text, "License")
    assert sec.strip(), "the README has no '## License' section"
    assert spdx in sec, f"the License section does not name {spdx}"
    links = set(relative_links(sec))
    for target in ("LICENSE", "NOTICE"):
        assert target in links, f"the License section does not link {target}"
    assert SUPPORT_LINE in sec.splitlines() or SUPPORT_LINE in " ".join(sec.split()), "the support line is missing"
    assert not NOT_PRODUCT.search(text), "the README says the kit is not a Databricks product"


def test_readme_has_the_license_section_and_support_line():
    check_license_section(_read(README), _spdx(tomllib.loads(_read(PYPROJECT))))


GOOD_LICENSE = (f"# Kit\n\n## License\n\nApache-2.0; see [LICENSE](./LICENSE) and [NOTICE](NOTICE).\n\n{SUPPORT_LINE}\n"
                "\n## Next\n")


@pytest.mark.parametrize("bad", [
    GOOD_LICENSE.replace("## License", "## Licence notes"),                    # no License section
    GOOD_LICENSE.replace("Apache-2.0", "Apache 2"),                            # the licence not named by its id
    GOOD_LICENSE.replace(" and [NOTICE](NOTICE)", ""),                         # NOTICE not linked
    GOOD_LICENSE.replace("[LICENSE](./LICENSE)", "LICENSE"),                   # LICENSE named, not linked
    GOOD_LICENSE.replace(SUPPORT_LINE, "Provided as is."),                     # the support line changed
    GOOD_LICENSE.replace("\n## Next\n", "\n## Next\n" + SUPPORT_LINE + "\n").replace(f"\n\n{SUPPORT_LINE}\n", "\n"),
    GOOD_LICENSE + _j("\nThis is not a Databricks ", "product.\n"),            # the forbidden phrase
])
def test_hostile_license_section_fails(bad):
    check_license_section(GOOD_LICENSE)  # near miss: ./LICENSE and a following section
    with pytest.raises(AssertionError):
        check_license_section(bad)


# ---- items 5-7: README Prerequisites. One item per platform feature the kit uses, each with its own docs page, no
# release status (check_no_release_status), no "Lakeflow", and no Spark item unless the kit imports pyspark.
# docs/release/test_release_docs.py checks that the release-only feature page has the same features.
FEATURES = {  # the owner's list (2026-10-14): name -> a pattern the feature's name must match
    "Lakebase projects and branches": r"\bprojects\b.*\bbranches\b",
    "branch expiry": r"\bbranch\w*\b.*\bexpir",
    "synced tables": r"\bsynced\s+tables?\b",
    "Postgres roles for the reader": r"\bpostgres\w*\s+roles?\b",
    "SQL warehouses": r"\bSQL\s+warehouses?\b",
    "serverless notebooks": r"\bserverless\b.*\bnotebooks?\b",
    "Databricks Apps (optional)": r"\bDatabricks\s+Apps\b.*\boptional\b",
}
SERVERLESS_USES = ("notebook runner", "measure")  # what the serverless notebook row names
DOCS_URL = re.compile(r"https://docs\.databricks\.com/aws/en/[\w./#-]*")
NOT_THE_PAGE_NAME = re.compile(r"\bLakeflow\b", re.I)  # synced tables are a Lakebase feature
SPARK_ITEM = re.compile(r"\b(?:py)?spark\b", re.I)
PYSPARK_IMPORT = r"^\s*(import|from)\s+pyspark\b"


def kit_imports_pyspark(root: Path) -> bool:
    """git grep (read-only) of the kit's code for a pyspark import."""
    p = subprocess.run(["git", "-C", str(root), "grep", "-l", "-E", PYSPARK_IMPORT, "--", "src", "notebooks"],
                       capture_output=True, text=True)
    assert p.returncode in (0, 1), f"the pyspark search failed: {p.stderr.strip()[:200]}"
    return bool(p.stdout.strip())


def match_features(names, what: str) -> dict:
    """FEATURES name -> the one entry of names that matches it; every feature matches exactly one entry."""
    out = {}
    for feature, rx in FEATURES.items():
        hits = [n for n in names if re.search(rx, n, re.I)]
        assert len(hits) == 1, f"{what}: {feature!r} matches {len(hits)} entries: {hits}"
        out[feature] = hits[0]
    return out


def prerequisite_items(text: str):
    """(item text, link URL or None) per bullet of the README's Prerequisites section."""
    sec = _section(text, "Prerequisites")
    out = []
    for line in re.findall(r"^\s*[-*]\s+(.+)$", sec, re.M):
        url = re.search(r"\]\((https?://[^)\s]+)\)", line)
        name = re.sub(r"\]\([^)]*\)", "]", line)  # the name: the bullet without its link target
        out.append((name, url.group(1) if url and DOCS_URL.fullmatch(url.group(1)) else None))
    return out


def check_feature_names(items, pyspark: bool, what: str) -> dict:
    """items: (name, url). Each feature once, each with its own docs page; no Lakeflow; Spark only if imported."""
    assert items, f"{what}: no features listed"
    for name, url in items:
        assert url, f"{what}: {name!r} has no https://docs.databricks.com/aws/en/ link"
        assert not NOT_THE_PAGE_NAME.search(name), f"{what}: {name!r} names synced tables by another product"
        assert pyspark or not SPARK_ITEM.search(name), f"{what}: {name!r} lists Spark, which the kit does not import"
    urls = [u.split("#", 1)[0].rstrip("/") for _, u in items]
    assert len(set(urls)) == len(urls), f"{what}: two features share one page: {sorted(urls)}"
    found = match_features([n for n, _ in items], what)
    serverless = found["serverless notebooks"]
    for use in SERVERLESS_USES:
        assert use in serverless.lower(), f"{what}: the serverless notebooks entry does not name the {use}"
    return found


def test_readme_prerequisites_name_each_feature_with_its_page():
    text = _read(README)
    check_feature_names(prerequisite_items(text), kit_imports_pyspark(ROOT), "README Prerequisites")
    check_no_release_status(_section(text, "Prerequisites"))


GOOD_PREREQ = ("## Prerequisites\n\n"
               "- [Lakebase projects and branches](https://docs.databricks.com/aws/en/oltp/projects/manage-branches)\n"
               "- [Branch expiry](https://docs.databricks.com/aws/en/oltp/projects/branch-expiration)\n"
               "- [Synced tables](https://docs.databricks.com/aws/en/oltp/projects/sync-tables)\n"
               "- [Postgres roles for the reader](https://docs.databricks.com/aws/en/oltp/projects/roles)\n"
               "- [SQL warehouses](https://docs.databricks.com/aws/en/compute/sql-warehouse/)\n"
               "- [Serverless notebooks](https://docs.databricks.com/aws/en/compute/serverless/notebooks) (the "
               "notebook runner and the in-region measure)\n"
               "- [Databricks Apps](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/) (optional)\n"
               "- [Unity Catalog](https://docs.databricks.com/aws/en/data-governance/unity-catalog/)\n\n## Next\n")


@pytest.mark.parametrize("old,new", [
    ("- [Branch expiry](https://docs.databricks.com/aws/en/oltp/projects/branch-expiration)\n", ""),  # a feature gone
    ("[Synced tables]", "[Lakeflow synced tables]"),                                    # the helper's name
    ("/oltp/projects/roles", "/oltp/projects/sync-tables"),                              # two features, one page
    ("https://docs.databricks.com/aws/en/compute/sql-warehouse/", "https://example.org/w"),  # not a docs page
    (" (optional)", ""),                                                                 # Apps not marked optional
    ("the notebook runner and ", ""),                                                    # the runner not named
    ("## Next", "- [PySpark](https://docs.databricks.com/aws/en/spark)\n\n## Next"),     # Spark, not imported
    ("## Next", "- [Lakebase projects and branches, again](https://docs.databricks.com/aws/en/oltp/)\n\n## Next"),
])
def test_hostile_prerequisites_fail(old, new):
    assert old in GOOD_PREREQ
    check_feature_names(prerequisite_items(GOOD_PREREQ), False, "near miss")  # an extra feature (UC) is fine
    with pytest.raises(AssertionError):
        check_feature_names(prerequisite_items(GOOD_PREREQ.replace(old, new)), False, "hostile")


def test_a_spark_item_passes_only_when_the_kit_imports_pyspark():
    with_spark = GOOD_PREREQ.replace("## Next", "- [PySpark](https://docs.databricks.com/aws/en/spark)\n\n## Next")
    check_feature_names(prerequisite_items(with_spark), True, "imports pyspark")
    with pytest.raises(AssertionError):
        check_feature_names(prerequisite_items(with_spark), False, "no import")


def test_hostile_prerequisites_status_fails():
    with pytest.raises(AssertionError):
        check_no_release_status(_section(GOOD_PREREQ.replace("(optional)", "(optional, Public Preview)"),
                                         "Prerequisites"))


# ---- items 8-10: KNOWN_LIMITS.
# The tested scale is a placeholder from the claims register (docs/release/post-claims.md) until the dated run fills
# it; after the fill it is a value in the register's fill-table format. This file ships, so it builds every
# placeholder from parts (_ph, above) and holds none whole. Only the fill gate proves where a filled value came
# from (owner 2026-10-15); these checks prove its form.

REGISTERED_SCALE = ((_ph("C13-scale"), _ph("C13-cu")), (_ph("C23-scale"), _ph("C23-cu")))
PLACEHOLDER = re.compile(r"\{\{GK-[\w-]+\}\}")
# The fill table's formats: "{nodes:,} nodes, {edges:,} edges" (thousands separators, owner 2026-10-20), and for CU one value when min equals max, else
# "<min>-<max> CU". CU numbers are in the ":g" form: no leading or trailing zero.
SCALE_VALUE = r"[1-9]\d{0,2}(?:,\d{3})* nodes, [1-9]\d{0,2}(?:,\d{3})* edges"
CU_NUMBER = r"(?:0|[1-9]\d*)(?:\.\d*[1-9])?"
CU_VALUE = rf"(?P<lo>{CU_NUMBER})(?:-(?P<hi>{CU_NUMBER}))? CU"


def check_scale_section(text: str) -> None:
    sec = _section(text, "Scale")
    assert sec.strip(), "KNOWN_LIMITS has no '## Scale' section"
    said = " ".join(sec.split())
    filled = re.fullmatch(re.escape("Measured on a ") + SCALE_VALUE + re.escape(" graph at ") + CU_VALUE + r"\.", said)
    if filled:
        lo, hi = filled.group("lo"), filled.group("hi")
        assert hi is None or float(lo) < float(hi), f"a CU range must go up; equal ends print as one value: {said!r}"
        return  # filled, in the register's format
    held = set(PLACEHOLDER.findall(sec))
    registered = {p for pair in REGISTERED_SCALE for p in pair}
    assert held <= registered, f"unregistered placeholders: {sorted(held - registered)}"
    assert not re.search(r"\d", PLACEHOLDER.sub("", sec)), "the Scale section types a number; use the placeholders"
    # the placeholders hold their units, so the sentence adds none (owner 2026-10-15)
    assert said in {SCALE_SENTENCE.format(size=s, cu=c) for s, c in REGISTERED_SCALE}, \
        f"the Scale section is not one sentence {SCALE_SENTENCE!r} with the scale and CU of one run: {said!r}"


SCALE_SENTENCE = "Measured on a {size} graph at {cu}."


GOOD_SCALE = f"## Scale\n\nMeasured on a {_ph('C23-scale')} graph at {_ph('C23-cu')}.\n\n## Next\n"
FILLED_SCALE = "## Scale\n\nMeasured on a 1,200 nodes, 3,400 edges graph at 1-4 CU.\n\n## Next\n"


@pytest.mark.parametrize("bad", [
    GOOD_SCALE.replace("## Scale", "## Size"),                                 # no Scale section
    GOOD_SCALE.replace(_ph("C23-cu"), _ph("C13-cu")),                          # size and CU of different runs
    GOOD_SCALE.replace(" at " + _ph("C23-cu"), ""),                            # no CU
    GOOD_SCALE.replace(_ph("C23-scale"), _ph("C99-scale")),                    # an unregistered placeholder
    GOOD_SCALE.replace(_ph("C23-scale"), "1,000,000 nodes"),                   # a typed number
    GOOD_SCALE.replace(_ph("C23-cu") + ".", _ph("C23-cu") + " (at 4 CU)."),    # a typed number beside the placeholders
    GOOD_SCALE.replace(_ph("C23-cu") + ".", _ph("C23-cu") + " compute units."),  # a unit the placeholder holds already
    GOOD_SCALE.replace("graph at", "graph nodes and edges at"),                # the 9d78d73 sentence: units twice
    GOOD_SCALE.replace(_ph("C23-cu") + ".", _ph("C23-cu") + ". It was fast."),  # a second sentence
    FILLED_SCALE.replace("1,200 nodes, 3,400 edges", "1200 nodes, 3400 edges"),  # a filled value with no separators
    FILLED_SCALE.replace("1,200 nodes", "12,00 nodes"),                         # separators in the wrong places
    FILLED_SCALE.replace("1-4 CU", "4-4 CU"),                                  # a range whose ends are equal
    FILLED_SCALE.replace("1-4 CU", "4.0 CU"),                                  # one value with a trailing zero
    FILLED_SCALE.replace("1-4 CU", "1-4.0 CU"),                                # a range end with a trailing zero
    FILLED_SCALE.replace("1-4 CU", "4-1 CU"),                                  # a reversed range
    FILLED_SCALE.replace("1-4 CU", "04 CU"),                                   # a leading zero
    FILLED_SCALE.replace("1-4 CU", "4CU"),                                     # no space before the unit
    FILLED_SCALE.replace(", 3,400 edges", ""),                                  # a filled size with no edges
    FILLED_SCALE.replace("1-4 CU.", "1-4 CU. It was fast."),                   # a second sentence after a fill
    FILLED_SCALE.replace("1-4 CU", _ph("C23-cu")),                             # half filled
])
def test_hostile_scale_section_fails(bad):
    check_scale_section(GOOD_SCALE)
    check_scale_section(GOOD_SCALE.replace("C23", "C13"))  # near miss: the other registered run
    check_scale_section(FILLED_SCALE)                       # near miss: filled in the register's format
    with pytest.raises(AssertionError):
        check_scale_section(bad)


# The CU value has two filled forms (owner 2026-10-18): one value when the endpoint's min equals its max, else the
# range "<min>-<max> CU". Numbers are in the fill's ":g" form: no trailing zero.
@pytest.mark.parametrize("cu", ["4 CU", "1-4 CU", "0.5-4 CU", "0.5 CU", "16 CU", "2-16 CU"])
def test_a_filled_scale_section_takes_either_cu_form(cu):
    check_scale_section(FILLED_SCALE.replace("1-4 CU", cu))


def test_known_limits_has_the_scale_placeholders():
    check_scale_section(_read(KNOWN_LIMITS))


# The reference limit: measure's reference (lgk.measure.reference: the warehouse extracts ids, the walks run in Python in the client).
# The checked size is the dated run's scale placeholder, or its filled value in the register's format, never a
# typed number of another form (owner 2026-10-15).
SCALE_PLACEHOLDERS = tuple(size for size, _ in REGISTERED_SCALE)


def check_reference_limit(text: str) -> None:
    hits = [p for p in _paragraphs(text) if "reference" in p.lower()]
    flat = [" ".join(p.split()) for p in hits]

    def sized(p):
        return any(s in p for s in SCALE_PLACEHOLDERS) or re.search(SCALE_VALUE + " graph", p)

    def bare(p):
        return re.sub(SCALE_VALUE, "", PLACEHOLDER.sub("", p))

    ok = [p for p in flat if "Python" in p and re.search(r"\bclient\b", p)
          and sized(p) and set(PLACEHOLDER.findall(p)) <= set(SCALE_PLACEHOLDERS)
          and re.search(r"\bnot beyond\b", p) and not re.search(r"\d", bare(p))]
    assert ok, ("KNOWN_LIMITS does not say: the reference walks in Python in the client, checked at the dated run's "
                "scale, not beyond")


GOOD_REF = ("Measure's reference reads node ids and edges into the client and walks them in Python. It was checked on a "
            f"{_ph('C13-scale')} graph, not beyond.")
FILLED_REF = GOOD_REF.replace(_ph("C13-scale"), "1,200 nodes, 3,400 edges")


@pytest.mark.parametrize("bad", [
    GOOD_REF.replace("Python", "SQL"),                    # not a Python walk
    GOOD_REF.replace("into the client and walks them", "and walks them on the warehouse"),  # not the client
    GOOD_REF.replace(_ph("C13-scale") + " graph", "large graph"),  # no size
    GOOD_REF.replace(", not beyond", ""),                 # no limit
    GOOD_REF.replace(_ph("C13-scale"), "10^6-node"),      # the 9d78d73 form: a typed size, not the dated run's
    GOOD_REF.replace("graph, not", "graph (10^6 nodes), not"),  # a typed size beside the placeholder
    GOOD_REF.replace(_ph("C13-scale"), _ph("C99-scale")),   # an unregistered placeholder
    GOOD_REF.replace(_ph("C13-scale"), _ph("C13-cu")),      # a registered placeholder that is not a size
    GOOD_REF.replace("reference", "check"),               # not the reference's limit
    FILLED_REF.replace("1,200 nodes, 3,400 edges", "1,200 nodes"),      # a filled value in another format
    FILLED_REF.replace("1,200 nodes, 3,400 edges", "1200 nodes, 3400 edges"),  # a filled value with no separators
    FILLED_REF.replace("graph, not", "graph (10^6 nodes), not"),       # a typed size beside a filled value
])
def test_hostile_reference_limit_fails(bad):
    check_reference_limit(GOOD_REF)
    check_reference_limit(GOOD_REF.replace("C13", "C23"))  # near miss: the other registered run's scale
    check_reference_limit(FILLED_REF)                      # near miss: filled in the register's format
    with pytest.raises(AssertionError):
        check_reference_limit(bad)


def test_this_file_holds_no_placeholder_whole():
    """The export refuses a whole placeholder in any shipping file, and this file ships."""
    assert not PLACEHOLDER.search(Path(__file__).read_text())


def test_known_limits_states_the_reference_limit():
    check_reference_limit(_read(KNOWN_LIMITS))


# The notebook Python: KNOWN_LIMITS says only what the kit does about the notebook Python: it pins the serverless environment
# version (lgk.runtime.ENVIRONMENT_VERSION) and the notebook refuses an older Python. A "3.10" line needs its docs page
# and a date. The kit no longer installs with --ignore-requires-python (tests/test_runtime_pin.py), so a line that says
# it does is wrong.
DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b")


def check_python_runtime(text: str, env_version: str, waiver_in_code: bool) -> None:
    for n, line in enumerate(text.splitlines(), 1):
        if re.search(r"\b3\.10\b", line):
            assert DOCS_URL.search(line) and DATE.search(line), f"line {n} names Python 3.10 with no docs page and date"
        if "--ignore-requires-python" in line:
            assert waiver_in_code, f"line {n} names an install flag the kit does not use"
    m = re.search(r"\benvironment version\W+(\w+)", text, re.I)
    assert m, "KNOWN_LIMITS does not name the serverless environment version the kit pins"
    assert m.group(1) == env_version, f"KNOWN_LIMITS says environment version {m.group(1)}; the kit pins {env_version}"


def _waiver_in_code(root: Path) -> bool:
    return any("--ignore-requires-python" in p.read_text() for p in [*(root / "src").rglob("*.py"),
                                                                       *(root / "notebooks").glob("*.py")])


GOOD_PY_RUNTIME = ("Notebook jobs run on serverless environment version `3` (Python 3.12); the notebook stops with a fix "
                   "message on an older Python.")


@pytest.mark.parametrize("bad", [
    "3. **Serverless Python 3.10**: notebooks run Python 3.10. " + GOOD_PY_RUNTIME,          # 3.10, no page or date
    GOOD_PY_RUNTIME + " Install with `pip install --ignore-requires-python`.",                # a flag the kit dropped
    GOOD_PY_RUNTIME.replace("environment version `3`", "environment version `2`"),           # not the pinned version
    GOOD_PY_RUNTIME.replace("environment version `3`", "a pinned environment"),              # the version not named
])
def test_hostile_python_runtime_fails(bad):
    check_python_runtime(GOOD_PY_RUNTIME, "3", False)
    check_python_runtime(GOOD_PY_RUNTIME + " Python 3.10 (https://docs.databricks.com/aws/en/release-notes/serverless/"
                         "environment-version/, checked 2026-10-14) cannot run the kit.", "3", False)  # near miss
    with pytest.raises(AssertionError):
        check_python_runtime(bad, "3", False)


def test_known_limits_says_only_what_the_kit_does_about_the_notebook_python():
    check_python_runtime(_read(KNOWN_LIMITS), runtime.ENVIRONMENT_VERSION, _waiver_in_code(ROOT))


# ------------------------------------------------------------------ README comment above load_mondo


def check_readme_load_mondo_comment(text: str) -> None:
    """The line directly above 'load_mondo(' in the README Quick start python block must be the exact comment.

    load_mondo writes three tables and a volume into source_schema, so a reader who runs the Quick start against a
    schema that already holds tables of those names loses them. The comment says so before the call. The table set
    itself is pinned by tests/test_mondo.py::test_load_creates_volume_then_keyed_tables_from_read_files.
    """
    expected_comment = "# Optional starter data: writes three MONDO tables and a volume into source_schema. Use a new, empty schema, or skip this step to use your own tables."

    # Find the Quick start section
    match = re.search(r"## Quick start\s*\n```python\n(.*?)\n```", text, re.DOTALL)
    assert match, "Quick start python block not found in README"

    block = match.group(1)
    lines = block.splitlines()

    # Find the line starting with "load_mondo"
    load_mondo_indices = [i for i, ln in enumerate(lines) if ln.strip().startswith("load_mondo(")]
    assert load_mondo_indices, "load_mondo( not found in Quick start block"

    load_mondo_idx = load_mondo_indices[0]
    assert load_mondo_idx > 0, "load_mondo( is the first line of the Quick start; no line above it"

    # The line directly above must be the comment
    line_above = lines[load_mondo_idx - 1].rstrip()
    assert line_above == expected_comment, \
        f"Line above load_mondo does not match.\n" \
        f"Expected: {expected_comment!r}\n" \
        f"Got: {line_above!r}"


def test_readme_load_mondo_comment():
    """RED: The line above load_mondo( in README Quick start is the exact starter-data comment."""
    check_readme_load_mondo_comment(_read(README))


W21_COMMENT = ("# Optional starter data: writes three MONDO tables and a volume into source_schema. Use a new, "
               "empty schema, or skip this step to use your own tables.")
W21_CALL = ('load_mondo(config["workspace_client"], config["warehouse_id"],\n'
            '           config["source_catalog"], config["source_schema"],\n'
            '           config["run_folder"])\n')
W21_GOOD = ("## Quick start\n\n```python\nfrom lgk.loaders.mondo import load_mondo\n\nconfig = {}\n\n"
            + W21_COMMENT + "\n" + W21_CALL + "\nrunner = LaptopRunner(config)\n```\n\n## Next\n")


def test_good_readme_load_mondo_comment_passes():
    """The good block passes, so each hostile row below fails for its own reason."""
    check_readme_load_mondo_comment(W21_GOOD)


@pytest.mark.parametrize("bad", [
    W21_GOOD.replace(W21_COMMENT + "\n", ""),                                         # no comment
    W21_GOOD.replace("three MONDO tables", "two MONDO tables"),                       # "two tables"
    W21_GOOD.replace(" and a volume", ""),                                            # "and a volume" dropped
    W21_GOOD.replace(W21_COMMENT + "\n", W21_COMMENT + "\n\n"),                        # one line higher (blank line)
    W21_GOOD.replace(W21_COMMENT + "\n", W21_COMMENT + "\nrelease = None\n"),          # another line between
    W21_GOOD.replace(W21_COMMENT, W21_COMMENT.replace("tables.", "tables")),          # one other byte
    ("## Setup\n\n```python\n" + W21_COMMENT + "\n" + W21_CALL + "```\n\n"           # the comment in another block
     + W21_GOOD.replace(W21_COMMENT + "\n", "")),
], ids=["none", "two-tables", "no-volume", "blank-line", "line-between", "other-byte", "other-block"])
def test_hostile_readme_load_mondo_comment_fails(bad):
    with pytest.raises(AssertionError):
        check_readme_load_mondo_comment(bad)
