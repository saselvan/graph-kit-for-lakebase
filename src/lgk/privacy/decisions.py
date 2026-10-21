"""The record of past privacy decisions (owner decision 2026-10-05): privacy_decisions.json.

A decision is what the privacy step passed: the stamp in flags.json, whose hashes cover
flags.json, graph_mapping.yaml and schema.json. `decision_id` is the one string for it (the same
string binding.stamp_id gives). The privacy step appends each decision it passes, newest last:

    {"decisions": [{"seq": 1, "decision": "<id>", "at": "<UTC ISO>"}, ...]}

Only the privacy step writes this file (`record_decision`). The gate (lgk.privacy.gate,
`check_newest_decision`) reads it and refuses any decision that is not the newest entry. So
restoring an older flags.json + graph_mapping.yaml does not make the older decision current;
running the privacy step again on them does (it records them as the newest).

This module has no kit imports, so the gate and the binding can both use it.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

DECISIONS_FILE = "privacy_decisions.json"


class HistoryFormatError(ValueError):
    """privacy_decisions.json is not a list of rising {seq, decision} entries."""


def decision_id(recorded_stamp: Dict[str, Any]) -> str:
    """One string for a stamp (flags.json's `gate` dict)."""
    return hashlib.sha256(json.dumps(recorded_stamp, sort_keys=True).encode()).hexdigest()


def history_path(run_folder: str) -> Path:
    return Path(run_folder) / DECISIONS_FILE


def _checked(entries: Any) -> List[Dict[str, Any]]:
    if not isinstance(entries, list) or not entries:
        raise HistoryFormatError("'decisions' must be a non-empty list")
    for i, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or not isinstance(entry.get("decision"), str) or not entry["decision"]:
            raise HistoryFormatError(f"entry {i} has no decision id")
        if entry.get("seq") != i or isinstance(entry.get("seq"), bool):
            raise HistoryFormatError(f"entry {i} has seq {entry.get('seq')!r}: seq must rise 1, 2, 3, ...")
    return entries


def read_history(run_folder: str) -> List[Dict[str, Any]]:
    """The recorded decisions, oldest first. FileNotFoundError if absent, HistoryFormatError if malformed."""
    path = history_path(run_folder)
    if not path.exists():
        raise FileNotFoundError(str(path))
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise HistoryFormatError(f"not JSON: {exc}") from exc
    if not isinstance(record, dict):
        raise HistoryFormatError("the file must hold an object with 'decisions'")
    return _checked(record.get("decisions"))


def record_decision(run_folder: str, decision: str) -> None:
    """Append `decision` as the newest (the privacy step only, after it stamped flags.json).

    No entry is added when `decision` already is the newest. A missing file starts a new
    history; a malformed one raises (it is never silently replaced).
    """
    path = history_path(run_folder)
    entries = read_history(run_folder) if path.exists() else []
    if entries and entries[-1]["decision"] == decision:
        return
    entries.append({"seq": len(entries) + 1, "decision": decision,
                    "at": datetime.now(tz=UTC).isoformat()})
    path.write_text(json.dumps({"decisions": entries}, indent=2))
