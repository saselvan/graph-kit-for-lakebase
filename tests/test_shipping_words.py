"""Red tests for the shipping files: no banned word, and no leak the public generic gate flags.

What this protects: the public repo's text. The kit is published as a general tool; its files never call its users
by the sales words the post register bans, and never carry a host, path, email, token or ID the generic gate
flags.

What must never happen:
- A banned word (built from parts below) reaches any file the export ships, in code, docs, comments or tests.
- The health-data acronym (`P` + `HI`) appears outside a reviewed line. A reviewed line is pinned by file and exact
  line text, never by line number, and describes what the kit does about that data, never data the kit handled.
- A private reference reaches a shipping file: an issue number (`#` and digits, or the word and digits), a follow-up
  ID (`F` and two digits), a ticket ID (`T` and two digits), a rule ID (a letter and digits next to the word), or the
  private stubs file's name. The public repo has none of those issues, tickets or rules, so the reference points
  nowhere. The forms are in the hostile rows below.
- A file the export ships fails the public generic gate (`tools/checks/generic_gate.py`).

This file ships, and public CI runs it. Every banned word and every pinned line is assembled from parts at run
time, so this file passes its own check with no exemption. The set of shipping files is the export's own list
(`export_files` of the private release tools, at HEAD) when those tools are present, and every tracked file when
they are not (the public repo is the export, so every tracked file there ships).
"""
import ast
import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "tools" / "checks" / "generic_gate.py"
EXPORT_TOOLS = REPO / "tools" / ("rel" + "ease") / "deny_gate.py"

# Parts. The banned words of the post register, assembled when the test runs.
P = "P" + "HI"
_WORDS = [r"cust" + r"omers?", r"P" + r"OCs?", r"real" + r"\s+" + r"data", P]
BAN = re.compile(r"\b(?:" + "|".join(_WORDS) + r")\b", re.IGNORECASE)

# Reviewed lines (owner rulings 2026-10-14): (file, exact line text). Reword by default; pin a line only where no
# plainer word is exact. Each pin describes what the kit does about the health data, never data the kit handled, and
# allows that one acronym only; any other banned word fails. No line needs a pin today: each was reworded to say what
# the code does (what a file can hold, what is dropped).
PINS = []


def word_hits(files, pins):
    """`path:line` of every banned word not on a pinned line, plus `path\tstale pin` for a pin that matches no line.

    `files` is {relpath: text}. A pinned line allows only the acronym: a pinned text that also holds another banned
    word is never allowed.
    """
    allowed = {(f, text) for f, text in pins
               if all(m.group(0).upper() == P for m in BAN.finditer(text))}
    hits, used = [], set()
    for relpath in sorted(files):
        for n, line in enumerate(files[relpath].splitlines(), 1):
            if not BAN.search(line):
                continue
            if (relpath, line) in allowed:
                used.add((relpath, line))
                continue
            hits.append(f"{relpath}:{n}")
    hits += [f"{f}\tstale pin" for f, text in pins if (f, text) not in used]
    return hits


# Private-reference forms (owner ruling 2026-10-14), assembled from parts so this file passes its own check.
_REFS = [
    r"\b[Ii]ssues?\s*#?\s*\d+\b",                    # the word and a number, with or without the hash
    r"(?<![\w&#/\"'=])(?<!:\s)" + "#" + r"\d+\b",       # a bare number; never a character reference, URL or colour
    r"\b" + "F" + r"\d{2}\b",                         # follow-up IDs
    r"\b" + "T" + r"\d{2}\b",                         # ticket IDs; a timestamp's T has no word break before it
    r"\b[A-Z]\d+[a-z]?\s+" + "rule" + r"\b",            # rule IDs, the ID first
    r"\b" + "rule" + r"\s+[A-Z]\d+[a-z]?\b",            # rule IDs, the word first
    r"\b(?i:" + "stubs" + r")\.md\b",                  # the private stubs file, in any case
    # Owner ruling 2026-10-15: every form below points to a document that does not ship (the spec, the plan, the
    # spike results, the owner's decisions and review lists). Seam-N stays: the shipping conftest defines it.
    r"(?<![\"'\w])[" + "UR" + r"]\d{1,3}[a-z]?\b(?![\"'])",  # requirement IDs; a quoted data value passes
    r"\b(?i:" + "it" + r"em)\s+\d+[a-z]?\b",                 # review-list items, any case
    r"\b" + "S-" + r"[a-z]+\b",                          # named spikes
    r"\b" + "S" + r"\d+(?:\.\d+|-\d+)\b",                  # numbered spikes
    # Owner ruling 2026-10-15: a lower-case spike ID inside a test name got past the form above (no word break
    # after "_", and "_" is not a separator there). Any case, any of "._-", inside an identifier too.
    r"(?<![A-Za-z0-9])[" + "Ss" + r"]\d+[._-]\d+(?![0-9])",
    r"\b" + "spi" + r"ke\s+\d+\b",                        # "spike" and a number (also a dated spike)
    r"\b[" + "DI" + r"]\d{1,2}\b",                         # owner decision IDs
    r"\b[" + "QENW" + r"]\d{1,2}[a-z]?\b",                 # plan question, evidence, note and work IDs
    r"\b[A-Z]\s\d[a-z]\b",                               # a letter and a section, as in a review's part 2b
    r"\b" + "pl" + r"an\s+[A-Z]\d+\b",                     # "plan" and an ID
    r"\b" + "user stor" + r"(?:y|ies)\s+\d+",                # user story numbers
    r"(?<!\{\{)\b" + "GK" + r"-C\d+\b",                     # claim-register IDs; a {{...}} placeholder is filled
]
REF = re.compile("|".join(f"(?:{r})" for r in _REFS))


def ref_hits(files):
    """`path:line` of every line holding a private reference, in file order. `files` is {relpath: text}."""
    return [f"{relpath}:{n}" for relpath in sorted(files)
            for n, line in enumerate(files[relpath].splitlines(), 1) if REF.search(line)]


def shipping_files():
    """The files the export ships, at HEAD."""
    if EXPORT_TOOLS.is_file():
        spec = importlib.util.spec_from_file_location("lgk_export_tools_for_words", EXPORT_TOOLS)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.export_files(REPO, "HEAD")
    out = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True, check=True).stdout
    return sorted(p for p in out.splitlines() if p)


def _read(relpath):
    return (REPO / relpath).read_bytes().decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def shipped():
    files = shipping_files()
    assert len(files) > 100, len(files)  # hostile: an empty or tiny list passes every check below
    return {f: _read(f) for f in files if (REPO / f).is_file()}


# ---------------------------------------------------------------- the real shipping files

def test_no_banned_word_in_any_shipping_file(shipped):
    hits = word_hits(shipped, PINS)
    assert hits == [], f"{len(hits)} lines: " + ", ".join(hits)


def test_no_private_reference_in_any_shipping_file(shipped):
    hits = ref_hits(shipped)
    assert hits == [], f"{len(hits)} lines: " + ", ".join(hits)


def test_the_shipping_files_pass_the_public_generic_gate(shipped):
    spec = importlib.util.spec_from_file_location("lgk_generic_gate_for_words", GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    hits = [f"{f}:{n}\t{kind}" for f in sorted(shipped) for n, kind in gate.scan_text(shipped[f], f)]
    assert hits == [], f"{len(hits)} hits: " + ", ".join(hits)  # path:line and kind only, never the text


MANIFEST = REPO / "tools" / ("rel" + "ease") / "manifest.txt"


def manifest_files(manifest_text: str, tracked) -> set:
    """The tracked files a ship list names, read apart from the export's own code: an entry ending in "/" ships every
    file under it, any other entry ships that one file."""
    entries = [ln.strip() for ln in manifest_text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    assert entries, "the ship list names nothing"
    return {f for f in tracked if any(f == e or (e.endswith("/") and f.startswith(e)) for e in entries)}


def check_reads_every_shipping_file(read, expected) -> None:
    missing, extra = sorted(set(expected) - set(read)), sorted(set(read) - set(expected))
    assert not missing, f"the word check skips {len(missing)} shipping files, e.g. {missing[:3]}"
    assert not extra, f"the word check reads {len(extra)} files the export does not ship, e.g. {extra[:3]}"


def test_the_word_check_reads_every_shipping_file_in_the_manifest(shipped):
    # owner 2026-10-15: the checks must provably read every file the ship list names (a shrunk list must go red)
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.split()
    if MANIFEST.is_file():
        expected = manifest_files(MANIFEST.read_text(), tracked)
    else:  # the public repo is the export: every tracked file ships
        expected = set(tracked)
    check_reads_every_shipping_file(shipped, {f for f in expected if (REPO / f).is_file()})


_TRACKED = ["src/lgk/a.py", "notebooks/n.py", "README.md", "tools/rel" + "ease/x.py", "docs/private.md"]
_LIST = "# code\nsrc/\nnotebooks/\nREADME.md\n"


@pytest.mark.parametrize("read", [
    ["src/lgk/a.py", "README.md"],                                        # notebooks/ skipped
    ["src/lgk/a.py", "notebooks/n.py"],                                   # one file entry skipped
    ["src/lgk/a.py", "notebooks/n.py", "README.md", "docs/private.md"],   # a file the list does not ship
])
def test_hostile_shrunk_or_grown_file_list_fails(read):
    expected = manifest_files(_LIST, _TRACKED)
    assert expected == {"src/lgk/a.py", "notebooks/n.py", "README.md"}
    check_reads_every_shipping_file(["README.md", "notebooks/n.py", "src/lgk/a.py"], expected)  # near miss: order
    with pytest.raises(AssertionError):
        check_reads_every_shipping_file(read, expected)


def test_this_file_is_a_shipping_file_and_passes_its_own_check(shipped):
    me = Path(__file__).resolve().relative_to(REPO).as_posix()
    assert me in shipped
    assert word_hits({me: shipped[me]}, []) == []
    assert ref_hits({me: shipped[me]}) == []


# ---------------------------------------------------------------- the checker (hostile rows)

PIN_FILE = "src/lgk/x.py"
PIN_TEXT = f"No {P} leaves this module."
PIN = [(PIN_FILE, PIN_TEXT)]


def test_a_pinned_line_moved_to_another_line_number_still_passes():
    for body in (f"{PIN_TEXT}\n", f"a\nb\nc\n{PIN_TEXT}\n"):
        assert word_hits({PIN_FILE: body}, PIN) == []


def test_the_pinned_text_in_another_file_fails():
    assert word_hits({PIN_FILE: PIN_TEXT, "src/lgk/y.py": f"x\n{PIN_TEXT}\n"}, PIN) == ["src/lgk/y.py:2"]


@pytest.mark.parametrize("changed", [
    f"No {P} leaves this file.",           # one word changed
    f"No {P} leaves this module. ",        # trailing space: exact text only
    f"    No {P} leaves this module.",     # indentation: exact text only
    f"No {P.lower()} leaves this module.",  # the case of the word itself
])
def test_a_pinned_line_with_any_change_fails(changed):
    assert word_hits({PIN_FILE: changed}, PIN) == [f"{PIN_FILE}:1", f"{PIN_FILE}\tstale pin"]


def test_a_pin_that_matches_no_line_fails_as_stale():
    assert word_hits({PIN_FILE: "nothing here\n"}, PIN) == [f"{PIN_FILE}\tstale pin"]


def test_a_pin_allows_only_the_acronym_never_another_banned_word():
    text = f"No {P} from a " + "cust" + "omer leaves this module."
    assert word_hits({PIN_FILE: text}, [(PIN_FILE, text)]) == [f"{PIN_FILE}:1", f"{PIN_FILE}\tstale pin"]


@pytest.mark.parametrize("line", [
    "# This file is " + "cust" + "omer-editable",   # hyphenated (graph_mapping.yaml:3 at the base)
    "Cust" + "omers pick a menu item.",              # plural, capitalised
    "a " + "P" + "OC criterion",
    "two " + "p" + "ocs ran",
    "it never reads " + "real" + "  " + "data",       # two spaces
    "REAL" + " " + "DATA",
    "a " + P + " row",
    "(" + P + ")",
])
def test_every_form_of_a_banned_word_is_a_hit(line):
    assert word_hits({"docs/x.md": f"ok\n{line}\n"}, []) == ["docs/x.md:2"]


@pytest.mark.parametrize("line", ["graphical output", "a typhoon", "epoch 3", "customs form", "realdata_column"])
def test_a_banned_word_inside_another_word_is_not_a_hit(line):
    assert word_hits({"docs/x.md": line}, []) == []


def test_hits_are_reported_as_path_and_line_in_file_order():
    files = {"b.md": f"{P}\n", "a.md": f"x\n{P}\ny\n{P}\n"}
    assert word_hits(files, []) == ["a.md:2", "a.md:4", "b.md:1"]


# ---------------------------------------------------------------- the private-reference check (hostile rows)

H, N = "#", "rule"


@pytest.mark.parametrize("line", [
    "is" + "sue " + "24 is open",                  # "issue" and a number
    "(Is" + "sue " + H + "4)",                     # capitalised, with the hash
    "Is" + "sue " + "20 criterion 3",              # capitalised, no hash: only the word form catches it
    "is" + "sues " + H + "7 and " + H + "9",       # plural
    "the app generator (" + H + "15, " + "U" + "33)",         # a bare issue number in brackets
    "Owner decision " + H + "8: the stamp",         # after a word
    H + "25 replaces the check",                   # at the start of a line
    "advise (follow-up " + "F" + "05)",            # a follow-up ID
    "fold (" + "T" + "12)",                         # a ticket ID
    "T" + "05 x " + "T" + "07 merge",              # two ticket IDs
    "(the " + "A" + "5 " + N + ")",                 # a rule ID
    "Run the " + "X" + "3 " + N + " for a view",    # another rule ID
    "see " + N + " " + "A" + "5",                   # the other order
    "(see " + "STUBS" + ".md)",                     # the private stubs file
    "stubs" + ".md limitation 5",                   # any case is not allowed for the file name
    # owner ruling 2026-10-15
    "privacy step (" + "U" + "11-" + "U" + "17, " + "R" + "12)",   # requirement IDs
    "the builder (" + "R" + "4a)",                  # a requirement ID with a letter
    "fold edges (" + "It" + "em 4)",                # a review-list item
    "see " + "it" + "em 3 of the review",           # lowercase
    "(" + "It" + "em 3b rule table)",               # an item with a letter
    "confirmed by " + "S-" + "types",               # a named spike
    "the 20 cap (" + "S" + "2.2)",                  # a numbered spike, dotted
    "a streaming source (" + "S" + "8-0)",          # a numbered spike, dashed
    "as " + "spi" + "ke 14 showed",                 # spike and a number
    "the dated " + "spi" + "ke 2026-10-05",         # a dated spike
    "18 + 2 per sync (" + "I" + "1)",               # an owner decision
    "no cap (" + "D" + "12 = b)",                   # another decision
    "the export (" + "Q" + "6)",                    # a plan question
    "the evidence (" + "E" + "4)",                  # plan evidence
    "the stop rule (" + "N" + "3)",                 # a plan note
    "the live opt-in (" + "W" + "3t)",              # a plan work item with a letter
    "binding (" + "A" + " 2b)",                     # a letter and a section
    "the advisor (" + "pl" + "an " + "X" + "4)",    # plan and an ID
    "teardown (" + "user stor" + "ies 46)",         # user stories
    "bound to " + "GK" + "-C10",                    # a claim-register ID
    "def test_" + "s4" + "_2_fold_cost(g):",         # a lower-case spike ID in a test name
    "x_" + "S8" + "_0 = 1",                          # upper case, inside an identifier
    "the " + "s2" + ".3 closure",                    # lower case, dotted
    "run " + "s14" + "-1 again",                     # lower case, dashed, two digits
])
def test_every_private_reference_form_is_a_hit(line):
    assert ref_hits({"src/lgk/x.py": f"ok\n{line}\n"}) == ["src/lgk/x.py:2"]


@pytest.mark.parametrize("line", [
    "the tissue 4 sample",                         # "issue" inside another word
    "an issue with 4 rows",                        # "issue" not followed by a number
    "colour " + H + "1a2b3c",                      # a colour with letters
    'fill="' + H + '000000"',                      # an all-digit colour in an attribute
    "color: " + H + "333;",                        # a CSS colour
    "&" + H + "39;",                               # an HTML character reference
    "https://example.com/page/" + H + "3",          # a URL fragment
    "# 25 rows",                                   # a comment hash with a space
    "F" + "5 refreshes the page",                  # one digit
    "0x" + "F" + "05",                             # inside a hex number
    "at 2026-10-14" + "T" + "12:00:00Z",           # a timestamp's T
    "T" + "100 rows",                              # three digits
    "a 5 " + N + " of thumb",                       # a number, not an ID
    "the " + N + " is simple",                      # no ID
    "test stubs live in tests/",                   # the word, not the file
    # owner ruling 2026-10-15
    'rows = ["' + "U" + '9", "' + "R" + '2"]',      # a quoted data value (byte-order rows)
    "Measured on a {{" + "GK" + "-C23-scale}} graph",  # a placeholder the dated run fills
    "@pytest.mark.seam1",                          # the live-test marker
    "a " + "Seam" + "-1 test",                      # the shipping conftest defines Seam-N
    "colour " + H + "D1E2F3",                       # a hex colour
    "at 2026-10-14T12:00:00Z",                     # a timestamp
    "an item with 3 rows",                         # "item" not followed by a number
    "items[0] and items 3",                        # the plural is not the form
    "W3C markup",                                  # letters run on, no word break
    "the " + "R" + "2x7 bucket",                   # digits, a letter, then more digits
    "UTF-8 and A 23",                              # a letter, a space and two digits
    "plan B",                                      # plan and a letter alone
    "the spike in latency",                        # spike with no number
    "s3://bucket/key_1",                           # an object-store URL
    "run_s3_bucket = 1",                           # digits then a letter, not a second number
    "pos4_2 and abs1.5",                           # a letter before the s
    "python3.12 and s3.amazonaws.com",             # a letter before the digit; a letter after the dot
])
def test_a_near_miss_is_not_a_private_reference(line):
    assert ref_hits({"src/lgk/x.py": line}) == []


def test_private_reference_hits_are_reported_as_path_and_line_in_file_order():
    ref = "(" + H + "4)"
    files = {"b.md": f"{ref}\n", "a.md": f"x\n{ref}\ny\n{ref}\n"}
    assert ref_hits(files) == ["a.md:2", "a.md:4", "b.md:1"]


# ---------------------------------------------------------------- NPI fixtures (owner ruling 2026-10-14)
# What this protects: real providers. Two fixture NPIs were registered to individual providers in the public NPI
# registry. A file that handles NPIs builds its values at run time from parts, so no 10-digit literal in it can be a
# real provider's number. A 10-digit literal in a file that never names NPIs (an epoch second, a branch suffix) is
# not a provider ID and passes.
NPI_WORD = re.compile(r"(?<![a-z])npis?(?![a-z])", re.IGNORECASE)  # also NPI_VALUES, bad_npi
TEN_DIGITS = re.compile(r"(?<!\d)\d{10}(?!\d)")
NPI_FIXTURES = ("tests/test_privacy.py", "tests/test_privacy_identifiers.py")


def npi_literal_hits(files):
    """`path:line` of every 10-digit literal in an NPI fixture file, in file order. `files` is {relpath: text}.

    Owner ruling 2026-10-15: the rule covers only the two NPI fixture files (NPI_FIXTURES)."""
    return [f"{relpath}:{n}" for relpath in sorted(files) if relpath in NPI_FIXTURES
            for n, line in enumerate(files[relpath].splitlines(), 1) if TEN_DIGITS.search(line)]


def test_no_ten_digit_literal_in_an_npi_fixture_file(shipped):
    for fixture in NPI_FIXTURES:  # hostile: a fixture dropped from the shipping list, or renamed away from NPIs
        assert fixture in shipped and NPI_WORD.search(shipped[fixture]), fixture
    hits = npi_literal_hits(shipped)
    assert hits == [], f"{len(hits)} lines hold a 10-digit literal in an NPI fixture file: {hits}"


@pytest.mark.parametrize("text", [
    'NPI_VALUES = ["' + "9" * 10 + '"]',                       # a literal list of NPIs
    "npi = " + "1" * 10,                                         # a bare literal next to the word
    "the npis\n\nbad = ['" + "2" * 10 + "']",                    # the word on another line of the same file
])
@pytest.mark.parametrize("fixture", NPI_FIXTURES)
def test_a_ten_digit_literal_in_an_npi_file_is_a_hit(text, fixture):
    assert npi_literal_hits({fixture: text})  # hostile rows: each must be caught, in each fixture file


@pytest.mark.parametrize("files", [
    {NPI_FIXTURES[0]: 'npi = make_npi("12345" + "6789")'},     # built from parts at run time
    {"tests/x.py": "npi = " + "1" * 10},                         # a file that names NPIs but is not a fixture
    {"tests/x.py": "sync_start_time=_ts(" + "1" * 10 + ")"},     # a timestamp in a file that never names NPIs
    {NPI_FIXTURES[1]: "the npi check digit: " + "1" * 9},      # nine digits
    {"tests/x.py": "unpinned = " + "1" * 10},                     # the letters inside another word
])
def test_a_value_built_from_parts_or_outside_an_npi_file_is_not_a_hit(files):
    assert npi_literal_hits(files) == []


# ---------------------------------------------------------------- Seam-N is defined (owner ruling 2026-10-15)
# What this protects: Seam-N ships (the conftest marker help names it), so a reader must find what it means. CONTEXT.md
# defines each Seam-N that conftest.py registers, and names its marker; it defines no Seam the conftest lacks.
CONFTEST = REPO / "conftest.py"
CONTEXT = REPO / "CONTEXT.md"
_MARKER = re.compile(r'"(seam\d+): mark test as (Seam-\d+)\b')
_ENTRY = re.compile(r"^\*\*(Seam-\d+)\*\*:\n((?:.+\n?)+)", re.M)


def registered_seams(conftest: str) -> dict:
    """{Seam-N: marker} from conftest.py's addinivalue_line marker text."""
    return {seam: marker for marker, seam in _MARKER.findall(conftest)}


def check_seams_defined(context: str, seams: dict) -> None:
    assert len(seams) >= 2, f"conftest registers {len(seams)} Seam markers"  # hostile: a parse that finds none
    entries = {term: body for term, body in _ENTRY.findall(context)}
    for seam, marker in seams.items():
        assert seam in entries, f"CONTEXT.md has no **{seam}**: entry"
        assert f"`{marker}`" in entries[seam], f"CONTEXT.md's {seam} entry does not name the marker `{marker}`"
    extra = sorted(set(entries) - set(seams))
    assert not extra, f"CONTEXT.md defines {extra}, which conftest.py does not register"


def test_context_defines_every_seam_conftest_registers():
    check_seams_defined(_read("CONTEXT.md"), registered_seams(_read("conftest.py")))


_SEAMS = {"Seam-1": "seam1", "Seam-2": "seam2"}
_GOOD_CTX = ("## Tests\n\n**Seam-1**:\nA live test, marked `seam1`; it runs only with --require-live.\n_Avoid_: x\n\n"
             "**Seam-2**:\nA test that reaches no cloud, marked `seam2`.\n_Avoid_: y\n")


@pytest.mark.parametrize("bad", [
    _GOOD_CTX.replace("**Seam-2**:", "**Seam two**:"),                      # one Seam not defined
    _GOOD_CTX.replace("`seam1`", "`live`"),                                 # the entry names another marker
    _GOOD_CTX.replace("`seam2`", "seam2"),                                  # the marker not named as code
    _GOOD_CTX + "\n**Seam-3**:\nA staging test, marked `seam3`.\n",       # a Seam the conftest does not register
])
def test_hostile_seam_definitions_fail(bad):
    check_seams_defined(_GOOD_CTX, _SEAMS)                                  # near miss: the good text passes
    with pytest.raises(AssertionError):
        check_seams_defined(bad, _SEAMS)


def test_the_seams_are_read_from_conftest_never_typed():
    seams = registered_seams(_read("conftest.py"))
    assert seams == _SEAMS  # the conftest registers exactly these today; the check above reads them from the file
    with pytest.raises(AssertionError):
        check_seams_defined("", {})  # hostile: no markers found is a failure, never a pass


# ---------------------------------------------------------------- the build summary (owner ruling 2026-10-15)
# What this protects: the README's step lines are read from each step module's summary. The build step runs SQL
# statements on the SQL warehouse (`execute_sql` with the run's warehouse id); it creates no pipeline. Its summary
# must say so and must not name SDP, which only an operator's own optional refresh pipeline uses.
BUILD = "src/lgk/steps/build.py"


def module_summary(source: str) -> str:
    """The first paragraph of a module docstring, on one line (as tests/test_readme_steps.py reads it)."""
    doc = ast.get_docstring(ast.parse(source)) or ""
    return " ".join(doc.split("\n\n", 1)[0].split())


def check_build_summary(source: str) -> None:
    summary = module_summary(source)
    assert not re.search(r"\bSDP\b|Declarative Pipelines?|\bpipelines?\b", summary, re.I), \
        f"the build summary names a pipeline: {summary!r}"
    assert re.search(r"\bSQL statements?\b", summary) and re.search(r"\bSQL warehouse\b", summary), \
        f"the build summary does not say it runs SQL statements on the SQL warehouse: {summary!r}"
    tree = ast.parse(source)
    calls = {n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
             for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "execute_sql" in calls, "the build step no longer calls execute_sql; update its summary with the code"


def test_the_build_summary_says_sql_on_the_warehouse():
    check_build_summary(_read(BUILD))


_GOOD_BUILD = ('"""Build step: runs SQL statements on the SQL warehouse to create the nodes and edges Delta tables '
               'from graph_mapping.yaml.\n\nMore.\n"""\ndef build(c):\n    execute_sql(w, wh, s)\n')


@pytest.mark.parametrize("bad", [
    _GOOD_BUILD.replace("runs SQL statements on the SQL warehouse to create", "SDP to create"),  # names SDP
    _GOOD_BUILD.replace("on the SQL warehouse", "in a Spark Declarative Pipelines pipeline"),     # names a pipeline
    _GOOD_BUILD.replace("graph_mapping.yaml.", "graph_mapping.yaml (SDP)."),                     # SQL and SDP
    _GOOD_BUILD.replace("Delta tables", "Delta tables through a pipeline"),                      # SQL and a pipeline
    _GOOD_BUILD.replace(" on the SQL warehouse", ""),                                            # no warehouse
    _GOOD_BUILD.replace("runs SQL statements", "runs"),                                          # no SQL statements
    _GOOD_BUILD.replace("execute_sql(w, wh, s)", "pass"),                                        # code runs no SQL
])
def test_hostile_build_summary_fails(bad):
    check_build_summary(_GOOD_BUILD)
    check_build_summary(_GOOD_BUILD.replace("\n\nMore.", "\n\nAn operator SDP pipeline may run first."))  # near
    # miss: SDP after the first paragraph is not the summary
    with pytest.raises(AssertionError):
        check_build_summary(bad)
