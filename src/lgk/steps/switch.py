"""Switch step: point the served views at one version, in one transaction.

The same call is the first serve after provision (no live version yet: the served schema
and views are created), a switch to a new version, and the rollback (a switch back to N-1).
It is the last step of a run: provision -> advise -> switch.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from lgk import versions
from lgk.privacy import binding
from lgk.privacy.gate import required_run_folder
from lgk.steps import provision as prov

logger = logging.getLogger(__name__)


def switch(config: Dict[str, Any], version: Optional[int] = None) -> str:
    """Repoint every served view at version `version` (default config["version"]); return its schema.

    Refused unless the privacy gate is clear, `version` was provisioned under the current
    privacy decision (a rollback to a version made under a looser one is refused), and the
    reader is still isolated from every version's raw data. Only then do the views move, in
    one transaction; the reader's view grants and the attribution notice follow.
    """
    run_folder = required_run_folder(config)
    cfg = config if version is None else {**config, "version": version}
    served = versions.served_schema(cfg)
    target = versions.version_schema(cfg)
    binding.require_servable(run_folder, target)
    prov._prove_reader_isolated(cfg)
    _move_views(cfg, target)
    prov._setup_reader_role(cfg)
    prov._serve_attribution(cfg)
    _record_switch(cfg, target)
    logger.info(f"Served schema {served} now reads {target}")
    return target


def _move_views(config: Dict[str, Any], target: str) -> None:
    """The one-transaction switch (lgk.versions.switch_views), with the served schema claimed."""
    conn = prov.connect_branch(config)
    conn.autocommit = True
    try:
        prov.serve(conn, config, target)
    finally:
        conn.close()


def rollback(config: Dict[str, Any], version: int) -> str:
    """The operator's rollback: `switch` back to the earlier `version`, which must be named.

    Not a runner step (the runner skips a switch that already succeeded). The version has no
    default: `switch(config)` alone would re-serve config["version"], the newer one. Every
    refusal of `switch` holds (gate, privacy stamp, reader isolation).
    """
    if isinstance(version, bool) or not isinstance(version, int):
        raise TypeError(f"rollback needs the version number to serve, got {version!r}")
    return switch(config, version)


def served_history(config: Dict[str, Any]) -> List[str]:
    """The version schemas the switches served, oldest first (run_folder/versions.json); [] before any.

    A record for people choosing a rollback target: what is live now is read from pg_depend.
    """
    path = Path(required_run_folder(config)) / "versions.json"
    if not path.exists():
        return []
    return [entry["live_version_schema"] for entry in json.loads(path.read_text())["history"]]


def _record_switch(config: Dict[str, Any], target: str) -> None:
    """Write run_folder/versions.json for the history record.

    Never consulted to decide what is live: retire/teardown read pg_depend instead.
    """
    path = Path(config["run_folder"]) / "versions.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    history = record.get("history", [])
    history.append({"live_version_schema": target, "at": datetime.now(tz=UTC).isoformat()})
    path.write_text(json.dumps({"live_version_schema": target, "history": history}, indent=2))
