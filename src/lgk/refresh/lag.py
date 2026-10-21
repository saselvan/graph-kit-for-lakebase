"""Freshness lag of one synced table after a sync (deck claim C30b), read live.

lag = sync completion time - source Delta commit time of the data the synced table now holds, in
minutes (`lgk.metrics.freshness_lag_minutes` holds the definition and refuses a negative lag).

Live sources (SDK 0.146):
- completion: `status.last_sync.sync_end_time` (a protobuf Timestamp; populated). Never
  `status.last_sync_time`, which is always None.
- source commit: `status.last_sync.delta_table_sync_info.delta_commit_time` at `.delta_commit_version`,
  cross-checked against the `timestamp` of that version in DESCRIBE HISTORY of the source table.
The reading must belong to THIS update: the table is ONLINE_NO_PENDING_UPDATE and its sync ended at or
after `not_before_s` (when the night started), else it is refused.
"""

from datetime import datetime
from typing import Any, Callable, List, NamedTuple, Optional

from lgk.metrics import freshness_lag_minutes

Sql = Callable[[str], List[List[Any]]]

COMMIT_TIME_TOLERANCE_S = 1.0  # API commit time vs DESCRIBE HISTORY (ms precision either side)
CLOCK_SKEW_TOLERANCE_S = 5.0  # max difference between local and control plane clock, for the "this update" check


class LagUnavailable(RuntimeError):
    """The synced table's status does not hold a usable reading for this update."""


class SyncReading(NamedTuple):
    version: int
    commit_s: float
    sync_end_s: float


def epoch_seconds(value: Any) -> Optional[float]:
    """Epoch seconds from a protobuf-style Timestamp (seconds, nanos), a zone-aware datetime or an ISO string
    with a zone (Z or an offset); a naive one is refused."""
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if hasattr(value, "seconds"):
        return float(value.seconds) + float(getattr(value, "nanos", 0) or 0) / 1e9
    if isinstance(value, datetime):
        if value.tzinfo is None:  # a naive time would be read in the laptop's zone: refuse it
            raise LagUnavailable(f"naive datetime {value.isoformat()} has no zone")
        return value.timestamp()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise LagUnavailable(f"cannot read a time from {type(value).__name__}")


def sync_reading(w: Any, dest_id: str) -> SyncReading:
    """(synced source version, its commit time, sync end time) from the synced table's live status."""
    status = w.postgres.get_synced_table(name=f"synced_tables/{dest_id}").status
    state = str(getattr(getattr(status, "detailed_state", None), "value", ""))
    if "ONLINE_NO_PENDING_UPDATE" not in state:
        raise LagUnavailable(f"{dest_id} is {state or 'in no state'}, not ONLINE_NO_PENDING_UPDATE")
    last = getattr(status, "last_sync", None)
    info = getattr(last, "delta_table_sync_info", None)
    version = getattr(info, "delta_commit_version", None)
    commit_s = epoch_seconds(getattr(info, "delta_commit_time", None))
    end_s = epoch_seconds(getattr(last, "sync_end_time", None))
    if version is None or commit_s is None or end_s is None:
        raise LagUnavailable(f"{dest_id}: last_sync lacks version/commit time/end time "
                             f"({version}, {commit_s}, {end_s})")
    return SyncReading(int(version), commit_s, end_s)


def history_commit_s(sql: Sql, source_table: str, version: int) -> float:
    """The commit time (epoch s) of `version` in DESCRIBE HISTORY of the source; columns by name."""
    rows = sql(f"SELECT unix_millis(timestamp) FROM (DESCRIBE HISTORY {source_table}) WHERE version = {int(version)}")
    if not rows:
        raise LagUnavailable(f"{source_table} has no history row for version {version}")
    return int(rows[0][0]) / 1000.0


def table_lag_minutes(w: Any, sql: Sql, dest_id: str, source_table: str, not_before_s: float) -> float:
    """Freshness lag in minutes of one synced table, for the update that started at `not_before_s`."""
    reading = sync_reading(w, dest_id)
    if reading.sync_end_s < not_before_s - CLOCK_SKEW_TOLERANCE_S:
        raise LagUnavailable(f"{dest_id}: last sync ended at {reading.sync_end_s}, before this update "
                             f"started ({not_before_s})")
    history_s = history_commit_s(sql, source_table, reading.version)
    if abs(history_s - reading.commit_s) > COMMIT_TIME_TOLERANCE_S:
        raise LagUnavailable(f"{dest_id}: API commit time {reading.commit_s} != DESCRIBE HISTORY {history_s} "
                             f"for {source_table} v{reading.version}")
    return freshness_lag_minutes(reading.sync_end_s, reading.commit_s)
