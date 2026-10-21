"""No telemetry and no personal or organisation names in src/ or the exported bundle.

The kit talks to the network only through the Databricks SDK and psycopg. Every other network
or analytics module is refused, by import (static) and by what importing the kit does (runtime).
Names are matched by sha256 of a lower-cased word, so this file carries no name itself.
"""

import ast
import getpass
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

BANNED_MODULES = frozenset({
    "requests", "urllib", "urllib2", "urllib3", "httpx", "http", "socket", "aiohttp", "websocket",
    "websockets", "grpc", "smtplib", "ftplib", "telnetlib", "xmlrpc", "pycurl", "sentry_sdk", "raven",
    "posthog", "segment", "analytics", "mixpanel", "amplitude", "opentelemetry", "datadog", "ddtrace",
    "newrelic", "statsd", "bugsnag", "rollbar", "honeybadger", "logtail", "heap",
})
# sha256(lower(word)) of the operator's and organisations' names (no plain names in the repo).
NAME_HASHES = frozenset({
    "90cc33a41b541af2c1964e3e10a46088cbdedf63031efaa35d588a698c91193f",
    "1fbeac9be6c0b3cf44c6185ca6b5ec2d002e56b87e4a9bbff6c80dad93b69934",
    "49b1b01cc7e397676b6f128a254e8cfc12db0adfb90e7c2fc0c5908e094f6381",
    "9078d0ad69d27e9052ce113faaf17c2d2a230e58a9172a76d74fa8ddbae63dfd",
    "bae3632ffa8b538a0930a410b2607cdba6bf8354798655a4fc925af32ef085d6",
    "a27d4c91b0366afaee1e091159b673d8f160810ccc82dc77c4a82f8ec38eaac4",
    "74831bf043c2f48a337f1ed5dd585b189b9b0e0abbbfd13e67d35a2e702cac86",
})
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
WORD_RE = re.compile(r"[A-Za-z]{3,}")


def _top(name: str) -> str:
    return name.split(".")[0]


def banned_imports(source: str) -> List[str]:
    """Banned modules a Python source imports: import, from-import, __import__, importlib."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if _top(a.name) in BANNED_MODULES]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if _top(node.module) in BANNED_MODULES:
                found.append(node.module)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            arg = node.args[0].value
            if name in ("__import__", "import_module") and isinstance(arg, str) and _top(arg) in BANNED_MODULES:
                found.append(arg)
    return found


def _py_files(*roots: Path) -> Iterable[Path]:
    for root in roots:
        yield from (p for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts)


def _text_files(*roots: Path) -> Iterable[Path]:
    for root in roots:
        yield from (p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix != ".pyc")


def _offenders(roots) -> List[str]:
    return [f"{p}: {m}" for p in _py_files(*roots) for m in banned_imports(p.read_text())]


# ------------------------------------------------------------------- the generated bundle
@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    from lgk.export import export

    run = tmp_path_factory.mktemp("np_run")
    from tests.test_build_provision_walks import _write_provider_graph_mapping

    _write_provider_graph_mapping(run / "graph_mapping.yaml", "c", "fx")  # the SP builds from it
    (run / "flags.json").write_text("{}")
    # build_mode sql: the bundle's SP builds its own kit tables; the operator pipeline runs first
    cfg = {"graph_name": "g", "kit_schema": "lgk_v1_000000", "lakebase_project": "p", "build_mode": "sql",
           "lakebase_catalog": "c", "version": 1, "run_folder": str(run), "sdp_pipeline_id": "pipe"}
    return export(cfg)


# ------------------------------------------------------------------- telemetry ------
def test_detector_catches_every_import_shape():
    # hostile: each way a phone-home could be imported
    src = (
        "import urllib.request\nfrom requests import post\nimport posthog as ph\n"
        "__import__('httpx')\nimport importlib\nimportlib.import_module('sentry_sdk.client')\n"
        "from . import socket\nfrom .http import helper\nimport json\n"
    )
    assert banned_imports(src) == ["urllib.request", "requests", "posthog", "httpx", "sentry_sdk.client"]


# The ONLY allowed network import: the MONDO loader downloads the public ontology file the
# operator asks for. It is a data download, not telemetry; the test below pins its host.
DATA_DOWNLOADERS = {"lgk/loaders/mondo.py": {"urllib.request"}}


def _allowed(entry: str) -> bool:
    path, _, module = entry.rpartition(": ")
    return any(path.endswith(f) and module in mods for f, mods in DATA_DOWNLOADERS.items())


def test_no_network_module_in_src_or_bundle(bundle):
    assert [e for e in _offenders([SRC, ROOT / "notebooks", bundle]) if not _allowed(e)] == []


def test_the_only_download_goes_to_the_mondo_purl():
    """The allowed downloader may reach exactly one host, the official MONDO purl."""
    from lgk.loaders import mondo

    tree = ast.parse((SRC / "lgk" / "loaders" / "mondo.py").read_text())
    opens = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", getattr(n.func, "id", "")) == "urlopen"]
    assert len(opens) == 1, "the MONDO loader must open exactly one URL"
    assert mondo.MONDO_URL == "https://purl.obolibrary.org/obo/mondo.json"


PROBE = r"""
import pkgutil, importlib, socket, sys, json
pkg_name = sys.argv[1]
calls = []
def trap(name):
    def f(*a, **k):
        calls.append(name)
        raise OSError("import-time network call: " + name)
    return f
socket.socket.connect = trap("connect")
socket.create_connection = trap("create_connection")
socket.getaddrinfo = trap("getaddrinfo")
pkg = importlib.import_module(pkg_name)
mods = [m.name for m in pkgutil.walk_packages(pkg.__path__, pkg_name + ".")]
errors = {}
for m in mods:
    try:
        importlib.import_module(m)
    except ModuleNotFoundError as e:
        if e.name != "pyspark":  # workspace-only modules import pyspark; anything else is an error
            errors[m] = repr(e)
    except Exception as e:
        errors[m] = repr(e)
print(json.dumps({"calls": calls, "errors": errors, "modules": len(mods)}))
"""


def _probe(pythonpath: Path, package: str) -> dict:
    """Import every module of `package` with the socket layer trapped (Python-level only:
    a C extension connecting at import time would not be seen)."""
    out = subprocess.run([sys.executable, "-c", PROBE, package], capture_output=True, text=True, cwd=ROOT,
                         env={"PYTHONPATH": str(pythonpath), "PATH": "/usr/bin:/bin"}, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_socket_trap_sees_each_connection_kind(tmp_path):
    # hostile: one module per way a phone-home opens the network at import time
    pkg = tmp_path / "phonehome"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "a.py").write_text("import socket\nsocket.getaddrinfo('example.com', 80)\n")
    (pkg / "b.py").write_text("import socket\nsocket.create_connection(('127.0.0.1', 9))\n")
    (pkg / "c.py").write_text("import socket\nsocket.socket().connect(('127.0.0.1', 9))\n")
    # hostile: a module that dies before its network code must be reported, not hidden
    (pkg / "d.py").write_text("import nonexistent_mod_xyz\n")
    (pkg / "e.py").write_text("raise RuntimeError('broken at import')\n")
    res = _probe(tmp_path, "phonehome")
    assert sorted(res["calls"]) == ["connect", "create_connection", "getaddrinfo"]
    assert "phonehome.d" in res["errors"] and "phonehome.e" in res["errors"]


def test_importing_the_kit_opens_no_connection():
    res = _probe(SRC, "lgk")
    assert res["modules"] > 20
    assert res["calls"] == [], res
    assert res["errors"] == {}, res


# ------------------------------------------------------------------- names ------
def _identity_words() -> List[str]:
    """The operator's login and git identity on this machine, split into words."""
    words = set(WORD_RE.findall(getpass.getuser().lower()))  # the scan's own tokenizer
    for key in ("user.email", "user.name"):
        try:
            val = subprocess.run(["git", "config", key], capture_output=True, text=True, cwd=ROOT).stdout
        except OSError:
            val = ""
        words |= set(WORD_RE.findall(val.lower()))
    return sorted(w for w in words if len(w) >= 4 and w not in {"root", "runner", "user", "databricks", "users"})


def _hits(roots, words: List[str]) -> List[str]:
    hits = []
    hashes = NAME_HASHES | {hashlib.sha256(w.encode()).hexdigest() for w in words}
    for p in _text_files(*roots):
        text = p.name + "\n" + p.read_text(errors="ignore")
        if EMAIL_RE.search(text):
            hits.append(f"{p}: email {EMAIL_RE.search(text).group(0)}")
        for w in set(WORD_RE.findall(text.lower())):
            if hashlib.sha256(w.encode()).hexdigest() in hashes:
                hits.append(f"{p}: name")
    return hits


def test_name_scanner_finds_a_planted_identity(tmp_path):
    # hostile: the operator's own login, an email, and a word hidden inside an identifier
    words = _identity_words() or ["zzplanted"]
    (tmp_path / "a.yml").write_text(f"owner: {words[0]}\n")
    (tmp_path / "b.py").write_text("x = 'first.last@example.com'\n")
    hits = _hits([tmp_path], words)
    assert any("a.yml: name" in h for h in hits) and any("b.py: email" in h for h in hits)


def test_local_identity_words_match_on_their_own(tmp_path):
    # hostile: a word NOT in the hashed denylist, so only the local-identity path can find it
    (tmp_path / "c.yml").write_text("owner: zzplantedname\n")
    assert _hits([tmp_path], ["zzplantedname"]) == [f"{tmp_path / 'c.yml'}: name"]
    assert _hits([tmp_path], []) == []


def test_identity_words_use_the_scan_tokenizer(monkeypatch):
    # hostile: a noreply-style login with digits and '+', which a split on [._-@] keeps glued
    monkeypatch.setattr(getpass, "getuser", lambda: "1234+someoperator")
    assert "someoperator" in _identity_words()


def test_email_in_a_file_name_is_found(tmp_path):
    (tmp_path / ("first.last" + "@example.com.json")).write_text("{}")
    assert any("email" in h for h in _hits([tmp_path], []))


def test_no_identity_in_src_or_bundle(bundle):
    assert _hits([SRC, ROOT / "notebooks", bundle], _identity_words()) == []


def test_bundle_templates_the_deployer_identity(bundle):
    top = (bundle / "databricks.yml").read_text()
    assert "${workspace.current_user.userName}" in top
