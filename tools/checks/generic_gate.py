#!/usr/bin/env python3
"""Public generic leak gate: scans source code and binaries for secrets, workspace hosts,
paths, emails, tokens, workspace IDs and UUIDs. Patterns are at least as broad as the publish audit's
commands; accepted forms are an explicit allow-list. Never imports anything under docs/release/.
"""
import re
import sys
import tomllib
from pathlib import Path

KINDS = ("host", "path", "email", "token", "secret", "workspace_id", "uuid")

# Assembled from parts so the gate passes itself with no exemption.
_DOMAINS = (
    "cloud." + "databricks" + ".com",
    "azure" + "databricks" + ".net",
    "gcp." + "databricks" + ".com",
    "dev." + "databricks" + ".com",
    "databricks" + "apps" + ".com",
)
_EXAMPLE_ORG = "example" + ".org"
_EXAMPLE_COM = "example" + ".com"

# Pinned exception: file + exact value, no line number.
_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
try:
    _PYPROJECT_DATA = tomllib.loads(_PYPROJECT.read_text())
    _AUTHOR_EMAIL = _PYPROJECT_DATA.get("project", {}).get("authors", [{}])[0].get("email", "")
except Exception:
    _AUTHOR_EMAIL = ""

EXCEPTIONS = [("pyproject.toml", _AUTHOR_EMAIL)] if _AUTHOR_EMAIL else []


def _make_host_pattern():
    """Build host pattern from domains."""
    domains_escaped = "|".join(re.escape(d) for d in _DOMAINS)
    return re.compile(
        r"(?:https?://)?[\w-]*\.(?:" + domains_escaped + r")(?:[:\"/'\s,;]|$)"
        r"|(?:https?://)?(?:" + domains_escaped + r")(?:[:\"/'\s,;]|$)"
        r"|dbc-[0-9a-f]{8}-[0-9a-f]{4}(?:\.cloud\.databricks\.com)?(?:[:/,;]|$)",
        re.IGNORECASE
    )


_HOST_PAT = _make_host_pattern()
# Build common parts first
_WS_PREFIX = "/" + "Work" + "space" + "/" + "Users" + "/"
_USERS = "/" + "Users" + "/"
_HOME = "/" + "home" + "/"
# Path pattern: built from parts to avoid self-flagging
_PATH_PAT = re.compile(_USERS + r"[A-Za-z0-9._%-]+(?:[/\\\s'\"]|$)|" + _HOME + r"[A-Za-z0-9._%-]+(?:[/\\\s'\"]|$)|" + _WS_PREFIX + r"[^\s'\"]*", re.IGNORECASE)
_EMAIL_PAT = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.IGNORECASE)
_TOKEN_PAT = re.compile(r"(?:da|do)(?:pi|se)[0-9a-f]{32}", re.IGNORECASE)
_SECRET_PAT = re.compile(r"(?:api[_-]?key|token|secret|password)\s*[:=]\s*['\"][\w\-._!@#$%^&*()+=~`|:;<>?,./]{12,}['\"]", re.IGNORECASE)
_WS_ID_PAT = re.compile(r"[?&]o=\d{10,}", re.IGNORECASE)
_UUID_PAT = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
_OK_EMAIL_PAT = re.compile(r"@(?:example\.com|example\.org)$", re.IGNORECASE)
_LOOK_ALIKE_PAT = re.compile(r"@example\.(?:com|org)\.[A-Za-z0-9.-]+", re.IGNORECASE)


def _is_example_email(email_text):
    """Check if an email is a valid example email (not a look-alike)."""
    if _LOOK_ALIKE_PAT.search(email_text):
        return False
    return bool(re.search(r"@(?:example\.com|example\.org)(?:\b|$|[^A-Za-z0-9.-])", email_text, re.IGNORECASE))


def _all_emails_in_segment_are_example(segment_content):
    """Check if ALL emails in a segment are example.com/example.org."""
    # Find all email-like patterns in the content
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", segment_content)
    if not emails:
        return True  # No emails, so it's acceptable
    # All emails must be example emails
    return all(_is_example_email(e) for e in emails)


def _is_accepted_ws_path(path_text):
    """Check if a workspace user path is accepted."""
    # Find the position of /Workspace/Users/
    idx = path_text.lower().find(_WS_PREFIX.lower())
    if idx == -1:
        return False

    after_prefix = path_text[idx + len(_WS_PREFIX):]

    # Accept bare endpoint (names nobody)
    if not after_prefix or after_prefix[0] in ("\"", "'", "/", "\\", " "):
        return True

    # Check if segment starts with a template: <...>, {...}, $VAR, ${VAR}
    # Templates are accepted if they don't contain a non-example email
    template_patterns = [
        (r"^<[^>]*>(?:[/\\\"\'\s]|$)", r"^<([^>]*)>"),          # angle brackets
        (r"^\{[^}]*\}(?:[/\\\"\'\s]|$)", r"^\{([^}]*)?\}"),    # braces (including f-strings)
        (r"^\$[A-Za-z_]\w*(?:[/\\\"\'\s]|$)", r"^\$([A-Za-z_]\w*)"),  # $VAR
        (r"^\$\{[A-Za-z_]\w*\}(?:[/\\\"\'\s]|$)", r"^\$\{([A-Za-z_]\w*)\}"),  # ${VAR}
    ]

    for match_pattern, extract_pattern in template_patterns:
        if re.match(match_pattern, after_prefix, re.IGNORECASE):
            # Extract just the template content
            match = re.match(extract_pattern, after_prefix)
            if match:
                template_content = match.group(1) if match.lastindex else ""
                # If template contains @, check that ALL emails are example emails
                if "@" in template_content:
                    if not _all_emails_in_segment_are_example(template_content):
                        # Template contains a non-example email or look-alike - this is a leak
                        return False
            return True

    # Check if segment starts with example.com or example.org email (bare, not in template)
    if re.match(r"[A-Za-z0-9._%+-]+@(?:example\.com|example\.org)(?:[/\\\"\'\s]|$)", after_prefix, re.IGNORECASE):
        return True

    return False


def scan_text(text, relpath, exceptions=None):
    """Scan text for leaks. Return [(line, kind)] sorted, never including accepted forms."""
    if exceptions is None:
        exceptions = EXCEPTIONS

    # Build exception set for fast lookup
    exc_set = set((rel, val) for rel, val in exceptions)

    hits = []
    path_flagged = set()  # Track which line numbers have flagged paths

    for line_no, line in enumerate(text.split("\n"), 1):
        # Track which kinds we've already reported on this line
        reported_kinds = set()

        # Check host pattern
        if "host" not in reported_kinds:
            for match in _HOST_PAT.finditer(line):
                matched_text = match.group(0)
                if (relpath, matched_text) not in exc_set:
                    hits.append((line_no, "host"))
                    reported_kinds.add("host")
                break

        # Check path pattern
        if "path" not in reported_kinds:
            for match in _PATH_PAT.finditer(line):
                matched_text = match.group(0)
                # Special handling for /Workspace/Users/ paths
                if _is_accepted_ws_path(matched_text):
                    continue
                if (relpath, matched_text) not in exc_set:
                    hits.append((line_no, "path"))
                    reported_kinds.add("path")
                    path_flagged.add(line_no)
                break

        # Check email pattern
        if "email" not in reported_kinds:
            for match in _EMAIL_PAT.finditer(line):
                matched_text = match.group(0)
                # Skip if it's an allowed email domain
                if _OK_EMAIL_PAT.search(matched_text):
                    continue

                # Skip if email is inside a flagged path (email counted only as path)
                if line_no in path_flagged:
                    continue

                # Skip if it's inside an accepted /Workspace/Users/ path
                path_match = _PATH_PAT.search(line)
                if path_match and _is_accepted_ws_path(path_match.group(0)):
                    if match.start() >= path_match.start() and match.start() < path_match.end():
                        continue

                if (relpath, matched_text) not in exc_set:
                    hits.append((line_no, "email"))
                    reported_kinds.add("email")
                break

        # Check token pattern
        if "token" not in reported_kinds:
            for match in _TOKEN_PAT.finditer(line):
                matched_text = match.group(0)
                if (relpath, matched_text) not in exc_set:
                    hits.append((line_no, "token"))
                    reported_kinds.add("token")
                break

        # Check secret pattern
        if "secret" not in reported_kinds:
            for match in _SECRET_PAT.finditer(line):
                matched_text = match.group(0)
                if (relpath, matched_text) not in exc_set:
                    hits.append((line_no, "secret"))
                    reported_kinds.add("secret")
                break

        # Check workspace_id pattern
        if "workspace_id" not in reported_kinds:
            for match in _WS_ID_PAT.finditer(line):
                matched_text = match.group(0)
                if (relpath, matched_text) not in exc_set:
                    hits.append((line_no, "workspace_id"))
                    reported_kinds.add("workspace_id")
                break

        # Check uuid pattern
        if "uuid" not in reported_kinds:
            for match in _UUID_PAT.finditer(line):
                matched_text = match.group(0)
                if (relpath, matched_text) not in exc_set:
                    hits.append((line_no, "uuid"))
                    reported_kinds.add("uuid")
                break

    # Return sorted, unique tuples
    result = []
    seen = set()
    for line_no, kind in hits:
        key = (line_no, kind)
        if key not in seen:
            result.append(key)
            seen.add(key)

    return sorted(result)


def scan_tree(root):
    """Scan tree for leaks. Return [(relpath, line, kind)] for every file except .git, including binaries."""
    root = Path(root)
    if not root.is_dir():
        return []

    results = []
    for fpath in root.rglob("*"):
        if fpath.is_file() and ".git" not in fpath.parts:
            relpath = fpath.relative_to(root).as_posix()

            # Read file, handling binaries.
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                # Try latin-1 as fallback for binaries.
                try:
                    text = fpath.read_bytes().decode("latin-1", errors="replace")
                except Exception:
                    continue

            for line_no, kind in scan_text(text, relpath):
                results.append((relpath, line_no, kind))

    return sorted(results)


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: generic_gate.py <root>\n")
        sys.exit(2)

    root = Path(sys.argv[1])
    if not root.is_dir():
        sys.stderr.write(f"error: {root} is not a directory\n")
        sys.exit(2)

    results = scan_tree(root)
    for relpath, line, kind in results:
        print(f"{relpath}:{line}\t{kind}")

    print(f"# {len(results)} hits in {len(set(r[0] for r in results))} files")
    sys.exit(1 if results else 0)


if __name__ == "__main__":
    main()
