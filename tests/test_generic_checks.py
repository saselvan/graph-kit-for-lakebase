"""Red tests for the public generic leak gate (`tools/checks/generic_gate.py`) and the public CI
workflow (`.github/workflows/public-ci.yml`).

This file ships, and public CI runs it. So it never imports anything under the private release folder, and every
planted host, path, email, token or ID below is assembled from parts at run time: no planted string sits in a
shipping file, and the gate passes this file with no exemption (attacker: the gate flags itself, or an exemption
hides a real hit).

Contract (pinned by the coordinator's interface note):
- `scan_text(text, relpath, exceptions=None) -> [(line, kind)]` (None = `EXCEPTIONS`),
  `scan_tree(root) -> [(relpath, line, kind)]`, `EXCEPTIONS = [(relpath, exact value)]`; kinds host, path, email,
  token, secret, workspace_id, uuid.
- CLI `generic_gate.py <root>`: `<relpath>:<line>\t<kind>` per hit, a last `# ` line; exit 0 clean, 1 hits,
  2 usage; never the matched text.
- Accepted forms only: a workspace user-path template segment (`<...>`, `{...}`, `$VAR`, `${VAR}`) with no
  non-example email in it, an example.com/example.org email as the segment, a bare workspace user root, emails at
  example.com, example.org and users.noreply.github.com, and the one pinned exception (file + exact value).
"""
import ast
import importlib.util
import re
import secrets
import struct
import subprocess
import sys
import tomllib
import uuid
import zlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "tools" / "checks" / "generic_gate.py"
WORKFLOW = REPO / ".github" / "workflows" / "public-ci.yml"

# Parts. Nothing below is a real host, name, email, token or ID; each is assembled when the test runs.
WS = "/" + "Work" + "space" + "/" + "Users" + "/"
LAPTOP = "/" + "Us" + "ers" + "/"
LINUX_HOME = "/" + "ho" + "me" + "/"
AT = "@"
NAME = "jq" + "x"
CORP = "corp" + ".test"
EXAMPLE = "example" + ".com"
DOMAINS = (
    "cloud." + "databricks" + ".com",
    "azure" + "databricks" + ".net",
    "gcp." + "databricks" + ".com",
    "dev." + "databricks" + ".com",
    "databricks" + "apps" + ".com",
)


_NOREPLY = "users." + "noreply." + "github" + ".com"


def _corp_email():
    return NAME + AT + CORP


def _example_email():
    return "op" + AT + EXAMPLE


def _dbc_host():
    return "db" + "c-" + secrets.token_hex(4) + "-" + secrets.token_hex(2)


@pytest.fixture(scope="module")
def gate():
    assert GATE.is_file(), "tools/checks/generic_gate.py does not exist yet"
    spec = importlib.util.spec_from_file_location("lgk_generic_gate_under_test", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _kinds(gate, text, relpath="src/lgk/x.py"):
    return {k for _, k in gate.scan_text(text, relpath, exceptions=[])}


# ---------------------------------------------------------------- planted leaks: each must be a hit

def _flagged_cases():
    e = _corp_email()
    return [
        # (case id, text, kinds any one of which must be reported)
        ("ws-path-name", WS + NAME + ".doe/lgk", {"path"}),
        ("ws-path-word-me", WS + "me/lgk", {"path"}),                      # hostile: a word, not a template
        ("ws-path-lower-case", WS.lower() + NAME, {"path"}),               # hostile: case
        ("ws-path-name-after-braces", WS + "{" + NAME + "}.doe/lgk", {"path"}),
        ("ws-path-corp-email", WS + e + "/lgk", {"path"}),
        ("ws-template-braces-corp-email", WS + "{" + e + "}/", {"path", "email"}),  # hostile: "@" in a template
        ("ws-template-angle-corp-email", WS + "<" + e + ">/", {"path", "email"}),   # hostile
        ("ws-fstring-holding-corp-email", WS + "{lookup('" + e + "')}/x", {"path", "email"}),  # inside accepted span
        ("laptop-path", LAPTOP + NAME + "/repos/x", {"path"}),
        ("linux-home-path", LINUX_HOME + NAME + "/x", {"path"}),
        ("laptop-path-upper", LAPTOP.upper() + NAME, {"path"}),
        ("corp-email", "mail " + e, {"email"}),
        ("corp-email-upper", "mail " + e.upper(), {"email"}),
        # owner ruling 2026-10-14: a GitHub noreply address holds a username; only the export commit's pinned
        # author (commit metadata, checked by S7) may hold one, never file content
        ("noreply-in-content", "by 1+" + NAME + AT + _NOREPLY, {"email"}),
        ("noreply-upper", ("1+" + NAME + AT + _NOREPLY).upper(), {"email"}),
        ("noreply-in-template", WS + "{1+" + NAME + AT + _NOREPLY + "}/x", {"path", "email"}),
        *[(f"host-sub-{i}", "https://a1." + d + "/x", {"host"}) for i, d in enumerate(DOMAINS)],
        *[(f"host-bare-{i}", "see " + d + " docs", {"host"}) for i, d in enumerate(DOMAINS)],  # audit breadth
        ("host-dbc", "https://" + _dbc_host() + ".cloud." + "databricks" + ".com", {"host"}),
        ("host-dbc-bare-upper", _dbc_host().upper(), {"host"}),
        ("token-dapi", "x = " + "da" + "pi" + secrets.token_hex(16), {"token"}),
        ("token-dose-upper", ("do" + "se" + secrets.token_hex(16)).upper(), {"token"}),
        ("secret-assignment", "pass" + "word" + ' = "' + secrets.token_hex(12) + '"', {"secret"}),
        ("workspace-id", "https://h/?" + "o=" + str(10 ** 15 + secrets.randbelow(10 ** 15)), {"workspace_id"}),
        ("uuid", "id " + str(uuid.uuid4()), {"uuid"}),
        ("uuid-upper", "id " + str(uuid.uuid4()).upper(), {"uuid"}),
    ]


@pytest.mark.parametrize("case", _flagged_cases(), ids=lambda c: c[0])
def test_generic_gate_flags_every_planted_leak(gate, case):
    _, text, want = case
    got = _kinds(gate, text)
    assert got & want, f"{case[0]}: no hit of kind {sorted(want)} (got {sorted(got)})"


# ---------------------------------------------------------------- accepted forms: nothing else is allowed

def _accepted_cases():
    ex = _example_email()
    return [
        ("ws-angle-template", WS + "<user>/lgk/x.whl"),
        ("ws-brace-template", WS + "{operator}/lgk"),
        ("ws-dollar-var", WS + "$" + "USER/lgk"),
        ("ws-dollar-brace-var", WS + "${" + "USER}/lgk"),
        ("ws-fstring-expression", "f'" + WS + "{w.current_user.me().user_name}/x'"),
        ("ws-example-email", WS + ex + "/lgk"),
        ("ws-brace-example-email", WS + "{" + ex + "}/lgk"),
        ("ws-bare-no-segment", 'root = "' + WS + '"'),                      # names nobody
        ("emails-example", "a" + AT + "example" + ".org " + ex),
        ("plain-code", "def walk(node_id):\n    return node_id + 1\n"),
    ]


@pytest.mark.parametrize("case", _accepted_cases(), ids=lambda c: c[0])
def test_generic_gate_accepts_only_the_allow_listed_forms(gate, case):
    _, text = case
    assert gate.scan_text(text, "src/lgk/x.py", exceptions=[]) == []


def test_an_email_inside_a_flagged_user_path_counts_once_as_the_path(gate):
    hits = gate.scan_text(WS + _corp_email() + "/x", "src/lgk/x.py", exceptions=[])
    assert [k for _, k in hits] == ["path"]


def test_scan_text_reports_one_based_line_numbers(gate):
    text = "a = 1\n\nb = '" + LAPTOP + NAME + "'\n"
    assert gate.scan_text(text, "src/lgk/x.py", exceptions=[]) == [(3, "path")]


# ---------------------------------------------------------------- the pinned exception: file + exact value only

def test_the_one_pinned_exception_is_the_pyproject_author_address(gate):
    authors = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["authors"]
    emails = [a["email"] for a in authors if "email" in a]
    assert gate.EXCEPTIONS == [("pyproject.toml", emails[0])], "one exception: (relpath, exact value), no line number"
    assert emails[0] not in GATE.read_text(), "the value is assembled from parts in the gate's source"


def test_the_real_pyproject_passes_with_the_default_exception(gate):
    assert gate.scan_text((REPO / "pyproject.toml").read_text(), "pyproject.toml") == []


def test_the_exception_matches_by_file_and_exact_value_never_by_line_number(gate):
    e = NAME + secrets.token_hex(3) + AT + CORP
    exc = [("pyproject.toml", e)]
    on_line_3 = '[project]\nname = "x"\nauthors = [{email = "' + e + '"}]\n'
    on_line_9 = "\n" * 6 + on_line_3                                         # hostile: the value moved lines
    assert gate.scan_text(on_line_3, "pyproject.toml", exceptions=exc) == []
    assert gate.scan_text(on_line_9, "pyproject.toml", exceptions=exc) == []
    # hostile: the same value in another file is a hit
    assert gate.scan_text(on_line_3, "README.md", exceptions=exc) == [(3, "email")]
    # hostile: a different address in pyproject.toml is a hit
    other = "z" + e
    assert gate.scan_text(on_line_3.replace(e, other), "pyproject.toml", exceptions=exc) == [(3, "email")]


# ---------------------------------------------------------------- tree scan: binaries, dot-files, run output

def _png_with_text(text):
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"tEXt", b"Comment\x00" + text.encode())
            + chunk(b"IDAT", zlib.compress(b"\x00\x00")) + chunk(b"IEND", b""))


def test_scan_tree_reads_binaries_dot_files_and_run_output(gate, tmp_path):
    host = "a1." + DOMAINS[0]
    (tmp_path / "docs" / "readme-assets").mkdir(parents=True)
    (tmp_path / "docs" / "readme-assets" / "flow.png").write_bytes(_png_with_text(host))     # attacker 7
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "cfg").write_text("owner: " + _corp_email() + "\n")              # dot-directory
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    run = ('{"host": "https://' + host + '", "run_as": "' + _corp_email() + '",\n'
           ' "url": "https://h/?' + "o=" + "1" * 16 + '"}\n')
    (tmp_path / "tests" / "fixtures" / "run.json").write_text(run)                           # attacker 2, 14
    (tmp_path / "junit.xml").write_text('<testsuite hostname="' + host + '"/>\n')             # attacker 14
    got = {}
    for rel, line, kind in gate.scan_tree(tmp_path):
        got.setdefault(rel, set()).add(kind)
    assert "host" in got.get("docs/readme-assets/flow.png", set())
    assert "email" in got.get(".hidden/cfg", set())
    assert {"host", "email", "workspace_id"} <= got.get("tests/fixtures/run.json", set())
    assert "host" in got.get("junit.xml", set())


def test_scan_tree_skips_only_the_git_directory(gate, tmp_path):
    host = "a1." + DOMAINS[1]
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("url = https://" + host + "\n")
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "x.yml").write_text("url: https://" + host + "\n")  # hostile: dot-dir
    rels = {rel for rel, _, _ in gate.scan_tree(tmp_path)}
    assert rels == {".github/workflows/x.yml"}


def test_a_commit_message_file_is_scanned(gate, tmp_path):
    (tmp_path / "COMMIT_EDITMSG").write_text("release 1.0\n\nAuthor: " + _corp_email() + "\n")  # attacker 8
    assert [(r, k) for r, _, k in gate.scan_tree(tmp_path)] == [("COMMIT_EDITMSG", "email")]


def test_a_pyproject_url_naming_a_workspace_host_is_a_hit(gate):
    text = '[project.urls]\nHomepage = "https://a1.' + DOMAINS[0] + '/x"\n'                    # attacker 9
    assert gate.scan_text(text, "pyproject.toml", exceptions=[]) == [(2, "host")]


# ---------------------------------------------------------------- CLI: path, line and kind; never the text

def _cli(*args):
    return subprocess.run([sys.executable, "-I", str(GATE), *map(str, args)], capture_output=True, text=True,
                          timeout=120)


def test_cli_prints_path_line_and_kind_never_the_matched_text(tmp_path):
    assert GATE.is_file(), "tools/checks/generic_gate.py does not exist yet"
    planted = ["a1." + DOMAINS[4], _corp_email(), "da" + "pi" + secrets.token_hex(16), LAPTOP + NAME + "q"]
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("ok = 1\n" + "\n".join("v = '" + p + "'" for p in planted) + "\n")
    r = _cli(tmp_path)
    assert r.returncode == 1, r.stderr[-400:]
    lines = r.stdout.splitlines()
    assert lines[-1].startswith("# ")
    assert sorted(lines[:-1]) == sorted(["src/a.py:2\thost", "src/a.py:3\temail", "src/a.py:4\ttoken",
                                         "src/a.py:5\tpath"])
    out = r.stdout + r.stderr
    for p in planted:
        for form in (p, p.lower(), p.upper(), p.split("/")[-1]):
            assert form not in out, "the gate printed matched text"


def test_cli_exit_codes_clean_and_usage(tmp_path):
    assert GATE.is_file(), "tools/checks/generic_gate.py does not exist yet"
    (tmp_path / "a.py").write_text("x = 1\n")
    clean = _cli(tmp_path)
    assert clean.returncode == 0 and clean.stdout.splitlines()[-1].startswith("# ")
    assert _cli(tmp_path / "missing").returncode == 2
    assert _cli().returncode == 2


# ---------------------------------------------------------------- the gate passes its own files, no exemption

def test_the_gate_passes_its_own_files_and_this_test_file(gate, tmp_path):
    # the export ships tracked files only, so scan the gate's tracked files (a local __pycache__ holds
    # compiled constants and is never shipped; the gate must still scan any __pycache__ it is given)
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "tools/checks"], capture_output=True,
                             text=True, check=True).stdout.split()
    assert tracked, "no tracked gate files"
    for rel in tracked:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes((REPO / rel).read_bytes())
    assert gate.scan_tree(tmp_path) == []
    assert gate.scan_text(Path(__file__).read_text(), "tests/test_generic_checks.py", exceptions=[]) == []
    assert gate.scan_text(WORKFLOW.read_text(), ".github/workflows/public-ci.yml", exceptions=[]) == []


def test_neither_the_gate_nor_this_file_imports_the_private_release_folder():
    private = ("docs" + "/" + "release", "ship" + "_scan", "scan" + "_reconcile")
    for path in (GATE, Path(__file__)):
        tree = ast.parse(path.read_text())
        names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert not [m for m in names if any(p.replace("/", ".") in m or p in m for p in private)], path.name


# ---------------------------------------------------------------- public CI

def _top_level_blocks(text):
    """{top-level key: [inline value if any] + indented child lines}, comments dropped."""
    blocks, key = {}, None
    for raw in text.splitlines():
        line = raw.split(" #")[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        if not line[0].isspace():
            k, _, rest = line.partition(":")
            key = k.strip().strip("'\"")
            blocks[key] = [rest.strip()] if rest.strip() else []
        elif key is not None:
            blocks[key].append(line)
    return blocks


def _child_keys(lines):
    inline = [ln for ln in lines if not ln[:1].isspace()]
    if inline:
        return {t.strip().strip("'\"") for t in inline[0].strip("[]").split(",") if t.strip()}
    indent = min(len(ln) - len(ln.lstrip()) for ln in lines)
    return {ln.strip().split(":")[0].strip("- '\"") for ln in lines if len(ln) - len(ln.lstrip()) == indent}


def test_public_ci_is_pull_request_only_read_only_no_secrets_actions_pinned_to_sha():
    assert WORKFLOW.is_file(), ".github/workflows/public-ci.yml does not exist yet"
    text = WORKFLOW.read_text()
    blocks = _top_level_blocks(text)
    on = blocks.get("on", blocks.get("true"))
    assert on is not None, "no top-level on:"
    assert _child_keys(on) == {"pull_request"}, f"triggers {sorted(_child_keys(on))}; pull_request only"
    assert "pull_request" + "_target" not in text
    perms = blocks.get("permissions")
    assert perms is not None and [ln.strip() for ln in perms] == ["contents: read"], "permissions: contents: read"
    code = "\n".join(ln.split("#")[0] for ln in text.splitlines())
    assert not re.search(r"\bwrite\b", code, re.I), "no write permission anywhere"
    assert "secrets" not in code.lower(), "no secrets"
    assert "DATABRICKS_" not in code.upper() and "--require-live" not in code, "no live runs"
    uses = [u.strip("'\"") for u in re.findall(r"^\s*(?:-\s*)?uses:\s*(\S+)", text, re.M)]
    assert uses, "the workflow uses at least actions/checkout"
    pinned = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+" + "@" + "[0-9a-f]{40}")
    assert [u for u in uses if not pinned.fullmatch(u)] == [], "every action pinned to a 40-hex commit SHA"
    assert "tools/checks/generic_gate.py" in text, "CI runs the generic gate"
    assert re.search(r"\bpytest\b", text), "CI runs the offline tests"


# Each action's tag and that tag's commit, resolved 2026-10-14 with
# `gh api repos/actions/<action>/commits/refs/tags/<tag> --jq .sha`. A made-up 40-hex string passes the shape check
# above and fails on GitHub; a pin change is a reviewed change to this table.
REAL_PINS = {"actions/checkout": ("11d5960a326750d5838078e36cf38b85af677262", "v4"),
             "actions/setup-python": ("a26af69be951a213d495a4c3e4e4022e16d87065", "v5")}


def test_every_action_is_pinned_to_the_real_commit_of_the_tag_its_comment_names():
    text = WORKFLOW.read_text()
    uses = re.findall(r"^\s*(?:-\s*)?uses:", text, re.M)
    found = re.findall(r"^\s*(?:-\s*)?uses:\s*([^@\s]+)@([0-9a-f]{40})\s+#\s*(v\d+)\s*$", text, re.M)
    assert len(found) == len(uses), "every uses: line is <action>@<40 hex>  # <tag>"
    assert {a: (s, t) for a, s, t in found} == REAL_PINS


def test_ci_installs_the_test_dependencies_before_it_runs_pytest():
    # hostile: setup-python gives a bare interpreter; `python3 -m pytest` there fails with "No module named pytest"
    runs = [ln.split("run:", 1)[1].strip() for ln in WORKFLOW.read_text().splitlines() if "run:" in ln]
    install = [i for i, r in enumerate(runs) if re.search(r"\bpip install\b.*\.\[dev\]", r)]
    tests = [i for i, r in enumerate(runs) if re.search(r"\bpytest\b", r)]
    assert install and tests and install[0] < tests[0], runs


@pytest.mark.parametrize("text", [
    "/" + "Work" + "space/Users/{" + _corp_email() + "," + _example_email() + "}/x",   # an example beside a real one
    "/" + "Work" + "space/Users/<" + _example_email() + " " + _corp_email() + ">/x",
    "/" + "Work" + "space/Users/{op" + AT + EXAMPLE + "." + CORP + "}/x",              # an example look-alike domain
    "contact: op" + AT + EXAMPLE + "." + CORP,
    "/" + "Work" + "space" + "/" + "Users" + "/" + "op" + AT + EXAMPLE + "x",                     # hostile: one more letter, no dot
    "/" + "Work" + "space" + "/" + "Users" + "/" + "op" + AT + EXAMPLE + "x/lgk",
], ids=["mixed-braces", "mixed-angle", "look-alike-in-template", "look-alike-bare", "suffix-letter",
        "suffix-letter-then-path"])
def test_an_example_email_never_covers_a_real_one_or_a_look_alike_domain(gate, text):
    assert _kinds(gate, text), "the gate passed a real address because an example address was near it"


def test_scan_tree_scans_pycache(gate, tmp_path):
    # hostile: compiled constants in a .pyc keep string literals, so a name or an address survives there
    (tmp_path / "src" / "__pycache__").mkdir(parents=True)
    (tmp_path / "src" / "__pycache__" / "x.cpython-311.pyc").write_bytes(
        b"\x00\x01" + ("owner = " + _corp_email()).encode() + b"\x00")
    assert {rel for rel, _, _ in gate.scan_tree(tmp_path)} == {"src/__pycache__/x.cpython-311.pyc"}
