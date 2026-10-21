"""Runner parity: identical run-folder files after removing timing and timestamp fields.

The allow-list is exact. Only JSON object keys named in TIMING_KEYS are removed, at any depth.
Every other byte must match: other JSON keys and values, key order, and every non-JSON file
byte for byte. The one runner-specific file, runtime.json at the run-folder root (the notebook
runner's record of its Python and serverless environment), is not a step output and is skipped.
"""

import json
from pathlib import Path
from typing import Any, List

from lgk.runtime import RUNTIME_FILE

TIMING_KEYS = frozenset({"started_at", "finished_at", "duration_s"})
RUNNER_RECORD_FILES = frozenset({RUNTIME_FILE})  # root of the run folder only


def _strip_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_timing(v) for k, v in value.items() if k not in TIMING_KEYS}
    if isinstance(value, list):
        return [_strip_timing(v) for v in value]
    return value


def normalized_bytes(path: Path) -> bytes:
    """File content with timing keys removed (JSON) or unchanged (anything else)."""
    raw = Path(path).read_bytes()
    if Path(path).suffix != ".json":
        return raw
    return json.dumps(_strip_timing(json.loads(raw)), indent=2).encode()


def _relative_files(root: Path) -> set:
    files = {str(p.relative_to(root)) for p in Path(root).rglob("*") if p.is_file()}
    return files - RUNNER_RECORD_FILES


def diff_run_folders(a: Path, b: Path) -> List[str]:
    """Differences between two run folders after normalization; [] means parity."""
    files_a, files_b = _relative_files(a), _relative_files(b)
    diffs = [f"{f}: only in a" for f in sorted(files_a - files_b)]
    diffs += [f"{f}: only in b" for f in sorted(files_b - files_a)]
    for rel in sorted(files_a & files_b):
        if normalized_bytes(Path(a) / rel) != normalized_bytes(Path(b) / rel):
            diffs.append(f"{rel}: content differs")
    return diffs
