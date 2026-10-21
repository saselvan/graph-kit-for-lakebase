"""The one identifier rule for every name the kit splices into SQL (Spark or Postgres).

A name is ONE lower-case plain identifier: `^[a-z_][a-z0-9_]*$`, at most 63 bytes.
Lower case only, because Postgres folds unquoted names to lower case: `Foo` in a mapping
would be built as `foo` and then read back under a different name. 63 bytes, because
Postgres silently truncates longer names (two long names could collapse to one).

Every caller (mapping, versions, walks SQL, advisor, privacy floor, MONDO loader, app
queries) goes through `check` or `qualified`; none keeps its own pattern.
"""

import re
from typing import Any

PATTERN = r"[a-z_][a-z0-9_]*"
MAX_BYTES = 63

_NAME = re.compile(PATTERN)


class IdentifierError(ValueError):
    """A name is not one lower-case plain SQL identifier."""


def _reason(value: Any) -> str:
    if not isinstance(value, str):
        return f"must be a string, got {type(value).__name__}"
    if value != value.lower() and _NAME.fullmatch(value.lower()):
        return (f"must be lower case (Postgres folds unquoted names to lower case); "
                f"write {value.lower()!r}")
    if len(value.encode()) > MAX_BYTES:
        return f"must be at most {MAX_BYTES} bytes (Postgres truncates longer names)"
    return f"must match ^{PATTERN}$"


def is_valid(value: Any) -> bool:
    return (isinstance(value, str) and _NAME.fullmatch(value) is not None
            and len(value.encode()) <= MAX_BYTES)


def check(value: Any, what: str) -> str:
    """`value` if it is one lower-case plain identifier, else IdentifierError naming `what`."""
    if not is_valid(value):
        raise IdentifierError(f"{what} {_reason(value)}, got {value!r}")
    return value


def qualified(value: Any, what: str, min_parts: int = 1, max_parts: int = 3) -> str:
    """A dotted name (`a`, `a.b`, `a.b.c`) whose every part passes `check`."""
    parts = value.split(".") if isinstance(value, str) else None
    if parts is None or not min_parts <= len(parts) <= max_parts:
        raise IdentifierError(
            f"{what} must be {min_parts}..{max_parts} dot-separated identifiers, got {value!r}")
    for part in parts:
        check(part, what)
    return value
