"""Sync mode per table and the rule for triggered sources.

- First run (`sync_mode` "snapshot", the default): every table is SNAPSHOT, so a first run
  needs no change to the operator's source tables.
- Nightly (`sync_mode` "triggered"): nodes and edges are TRIGGERED (they change a little each
  night); every other table is SNAPSHOT (recomputed in full).
- A triggered synced table builds only from a STREAMING TABLE. From a materialized view it is
  accepted at create time and then goes ONLINE_PIPELINE_FAILED ("Failed to resolve flow").
  So the source type is checked before anything is created.

Kit v1: only "snapshot" is supported. A config with "triggered", "continuous", or any other
value is refused. The message names the unsupported mode, the value, and why v1 cannot support it
(streaming tables cannot take the kit's label; Path B—the label set through the pipeline
definition—is not in v1). For case variants of "snapshot" (e.g. "Snapshot"), the message uses the
lgk.ident form: `must be lower case ...; write 'snapshot'`.
"""

from typing import Any, Dict

SYNC_MODES = ("snapshot", "triggered")
TRIGGERED_TABLES = ("nodes", "edges")
V1_SYNC_MODES = ("snapshot",)


class UnsupportedSyncModeError(ValueError):
    """config["sync_mode"] is not one v1 supports (v1: "snapshot", or unset)."""


class TriggeredSourceError(ValueError):
    """A triggered synced table was asked for over a source that is not a streaming table."""


def require_v1_sync_mode(config: Dict[str, Any]) -> str:
    """"snapshot" if `sync_mode` is absent from config or exactly "snapshot"; else raise
    UnsupportedSyncModeError with the reason. Absent means key not in config (not even with None).
    No normalising: no lower-casing, no strip."""
    if "sync_mode" not in config:
        return "snapshot"
    mode = config["sync_mode"]
    if mode == "snapshot":
        return "snapshot"
    if isinstance(mode, str) and mode.lower() == "snapshot":
        raise UnsupportedSyncModeError(
            f"config['sync_mode'] = {mode!r}, but must be lower case and exactly 'snapshot'; write 'snapshot'."
        )
    if mode is None or mode == "":
        raise UnsupportedSyncModeError(
            f"config['sync_mode'] = {mode!r} is empty: remove the key for snapshot, or write 'snapshot'."
        )
    raise UnsupportedSyncModeError(
        f"config['sync_mode'] = {mode!r}, but v1 supports only 'snapshot' (unset is also 'snapshot'). "
        f"A pipeline-owned streaming table cannot take the kit's label, and Path B (the label set through "
        f"the pipeline definition) is not in v1. Set sync_mode to 'snapshot', or leave it unset."
    )


def policy_name(config: Dict[str, Any], table: str) -> str:
    """'SNAPSHOT' or 'TRIGGERED' for `table` under config["sync_mode"]."""
    mode = config.get("sync_mode", "snapshot")
    if mode not in SYNC_MODES:
        raise ValueError(f"sync_mode must be one of {SYNC_MODES} (never continuous), got {mode!r}")
    return "TRIGGERED" if mode == "triggered" and table in TRIGGERED_TABLES else "SNAPSHOT"


def assert_streaming_source(w: Any, source_table: str) -> None:
    """Raise TriggeredSourceError unless UC says `source_table` is a STREAMING_TABLE."""
    table_type = w.tables.get(full_name=source_table).table_type
    kind = str(getattr(table_type, "value", table_type))
    if kind != "STREAMING_TABLE":
        raise TriggeredSourceError(
            f"{source_table} is a {kind}, not a streaming table. A triggered synced table builds "
            f"only from a streaming table (from a materialized view it goes "
            f"ONLINE_PIPELINE_FAILED, 'Failed to resolve flow'). Make the source an SDP streaming "
            f"table, or use sync_mode 'snapshot'."
        )
