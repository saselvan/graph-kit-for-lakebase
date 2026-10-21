"""flags.json: one flag per care-subject column, resolved one by one.

A flag records: table, column, class, detected_by (name / values / both), choice, chooser,
reason, timestamp, plus `fold_allowed` (whether fold may be offered) and, for a reclassify,
`new_class`. A flag with no choice is OPEN, and an open flag stops the run.

The choices:
- fold: only for a care-subject key in an edge or join table. The column reaches Lakebase
  only as folded, weighted edges at or above the floor (lgk.privacy.floor).
- drop: the column never reaches the mapping.
- reclassify: to public_professional or none, with a typed reason.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from lgk.privacy.classify import CARE_SUBJECT, NONE, PUBLIC_PROFESSIONAL

CHOICES = ("fold", "drop", "reclassify")
_NO_FOLD_WHY = (
    "is a care-subject property of a node, not a key in an edge or join table, so it can "
    "only be dropped or reclassified: folding needs a key that links two other things."
)


class OpenFlagError(RuntimeError):
    """At least one care-subject flag has no choice yet; the run stops before build."""


class FlagChoiceError(ValueError):
    """A resolution that the rules do not allow."""


def flags_path(run_folder: str) -> Path:
    return Path(run_folder) / "flags.json"


def read_flags_file(run_folder: str) -> Dict[str, Any]:
    path = flags_path(run_folder)
    return json.loads(path.read_text()) if path.exists() else {}


def write_flags_file(run_folder: str, data: Dict[str, Any]) -> None:
    flags_path(run_folder).write_text(json.dumps(data, indent=2))


def is_open(flag: Dict[str, Any]) -> bool:
    return flag.get("choice") is None


def open_flags(run_folder: str) -> List[Dict[str, Any]]:
    return [f for f in read_flags_file(run_folder).get("flags", []) if is_open(f)]


def new_flag(table: str, column: str, detected_by: str, fold_allowed: bool) -> Dict[str, Any]:
    return {
        "table": table, "column": column, "class": CARE_SUBJECT, "detected_by": detected_by,
        "fold_allowed": fold_allowed, "choice": None, "chooser": None, "reason": None,
        "timestamp": None,
    }


def _check_choice(flag: Dict[str, Any], choice: str, chooser: str, reason: Optional[str],
                  new_class: Optional[str]) -> None:
    if choice not in CHOICES:
        raise FlagChoiceError(f"choice must be one of {CHOICES}, got {choice!r}")
    if not isinstance(chooser, str) or not chooser.strip():
        raise FlagChoiceError("a chooser (who decided) is required")
    where = f"{flag['table']}.{flag['column']}"
    if choice == "fold" and not flag["fold_allowed"]:
        raise FlagChoiceError(f"{where} {_NO_FOLD_WHY}")
    if choice == "reclassify":
        if not isinstance(reason, str) or not reason.strip():
            raise FlagChoiceError(f"reclassifying {where} needs a typed reason")
        if new_class not in (PUBLIC_PROFESSIONAL, NONE):
            raise FlagChoiceError(
                f"reclassify {where} to {PUBLIC_PROFESSIONAL!r} or {NONE!r}, got {new_class!r}"
            )


def resolve_flag(run_folder: str, table: str, column: str, choice: str, chooser: str,
                 reason: Optional[str] = None, new_class: Optional[str] = None) -> Dict[str, Any]:
    """Record one decision on one flag in flags.json; returns the updated flag."""
    data = read_flags_file(run_folder)
    matches = [f for f in data.get("flags", []) if (f["table"], f["column"]) == (table, column)]
    if not matches:
        raise KeyError(f"no flag for {table}.{column} in {flags_path(run_folder)}")
    flag = matches[0]
    _check_choice(flag, choice, chooser, reason, new_class)
    flag.update(choice=choice, chooser=chooser.strip(), reason=reason,
                timestamp=datetime.now(timezone.utc).isoformat())
    if choice == "reclassify":
        flag["new_class"] = new_class
    write_flags_file(run_folder, data)
    return flag
