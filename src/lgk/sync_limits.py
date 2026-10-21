"""Sync limits: the per-source synced-table cap and the sync's connections.

- A source table holds at most SYNCED_TABLES_PER_SOURCE synced tables (facts file, "Branch reset,
  permissions, synced-table rules"). Each kit version syncs every table again, so the count grows
  with versions. `per_source_counts` reads it from UC (the version schemas), not from the
  synced-table GET (NotFound once a branch is deleted, while the UC object stays) and not from the
  manifest (misses versions the bundle's SP made from another run folder, contract A11).
- Adaptive sync parallelism. One sync holds 2 Sync-Driver connections for the
  whole sync plus an executor burst of up to 16: a peak of CONNECTIONS_PER_SYNC_FLOOR = 18, and two at
  once add up. Before each new sync the nightly sync reads `free_slots`
  live on the branch (max_connections - superuser_reserved_connections - connections now - app
  headroom; nothing is hard-coded, the limit follows the compute size) and runs
  min(sync_max_parallel, free // per_sync) at once; per_sync = max(18, recorded high-water) + 2.

Known gap: synced tables made from the same source outside `<kit>_v*` are not counted.
"""

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

Sql = Callable[[str], List[List[Any]]]

SYNCED_TABLES_PER_SOURCE = 20
CONNECTIONS_PER_SYNC_FLOOR = 18  # 2 drivers + 16 executors at 10^6 rows
PER_SYNC_MARGIN = 2  # per_sync = max(floor, high-water) + 2

DEFAULT_SYNC_MAX_PARALLEL = 2
DEFAULT_SYNC_PER_SOURCE_HEADROOM = 2
DEFAULT_APP_CONNECTION_HEADROOM = 20


class SyncLimitError(RuntimeError):
    """A sync would pass a platform limit (per-source synced tables, or branch connections)."""


class SourceCounts(Dict[str, int]):
    """table name -> number of version schemas holding it; `.schemas` names those schemas."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.schemas: Dict[str, List[str]] = {}


def config_int(config: Dict[str, Any], key: str, default: int, minimum: int) -> int:
    """config[key] (default `default`) as an int >= minimum, else SyncLimitError."""
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SyncLimitError(f"config {key} must be an int >= {minimum}, got {value!r}")
    return value


def per_source_counts(sql: Sql, catalog: str, kit: str) -> SourceCounts:
    """For each table name, how many `<kit>_v<N>` schemas in UC hold it."""
    version = re.compile(rf"{re.escape(kit)}_v\d+", re.IGNORECASE)
    schemas = [row[0] for row in sql(f"SHOW SCHEMAS IN {catalog} LIKE '{kit}_v*'")]
    counts = SourceCounts()
    for schema in sorted(s for s in schemas if version.fullmatch(s)):
        for row in sql(f"SHOW TABLES IN {catalog}.{schema}"):
            counts[row[1]] = counts.get(row[1], 0) + 1
            counts.schemas.setdefault(row[1], []).append(schema)
    return counts


def assert_room(counts: Dict[str, int], tables: Iterable[str], limit: int = SYNCED_TABLES_PER_SOURCE,
                headroom: int = DEFAULT_SYNC_PER_SOURCE_HEADROOM) -> None:
    """Raise SyncLimitError if one more synced table of any of `tables` would pass limit - headroom."""
    full = [t for t in tables if counts.get(t, 0) + 1 > limit - headroom]
    if not full:
        return
    found = getattr(counts, "schemas", {})
    detail = "; ".join(f"{t}: {counts.get(t, 0)} version schemas ({', '.join(found.get(t, []))})" for t in full)
    raise SyncLimitError(
        f"A new synced table would pass the per-source cap of {limit} minus headroom {headroom} "
        f"for {detail}: retire a version or run `lgk.reaper.reap`, then re-run provision."
    )


# The platform's create-time refusal of the 21st synced table of one source.
_CAP_REFUSAL = re.compile(r"Cannot create more than \d+ synced database table\(s\) per source table", re.IGNORECASE)


def raise_if_per_source_cap(error: BaseException, source_table: str) -> None:
    """Re-raise the platform's per-source cap refusal of `error` as the kit's SyncLimitError.

    The authoritative check: the UC pre-check (`assert_room`) only sees `<kit>_v*` schemas.
    """
    if _CAP_REFUSAL.search(str(error)):
        raise SyncLimitError(
            f"Source {source_table} is at the platform's synced-table cap ({error}): "
            f"retire a version or run `lgk.reaper.reap`, then re-run provision."
        ) from error


# Connections now, WITHOUT pg_monitor: a plain role sees every session's usename but a NULL
# backend_type/state (facts file), so count every named session. That also counts the named
# background workers, which over-counts (the safe side). With pg_monitor an exact
# `backend_type = 'client backend'` count is possible; the kit does not grant pg_monitor.
CONNECTIONS_NOW_SQL = "SELECT count(*) FROM pg_stat_activity WHERE usename IS NOT NULL"
# The sync's own sessions (role databricks_writer_<n>, application Databricks-Sync-Driver/-Executor).
SYNC_WRITERS_SQL = ("SELECT count(*) FROM pg_stat_activity "
                    "WHERE usename LIKE 'databricks_writer%' OR application_name LIKE 'Databricks-Sync-%'")


@dataclass(frozen=True)
class FreeSlots:
    """One live read of the branch's connection room; `free` is what the next sync may use."""

    max_connections: int
    reserved: int  # superuser_reserved_connections
    now: int  # named sessions now (CONNECTIONS_NOW_SQL)
    headroom: int  # app_connection_headroom

    @property
    def free(self) -> int:
        return self.max_connections - self.reserved - self.now - self.headroom

    def describe(self) -> str:
        return (f"max_connections {self.max_connections} - superuser_reserved_connections {self.reserved} "
                f"- connections now {self.now} - app headroom {self.headroom} = {self.free} free slots")


def _read_int(conn: Any, query: str, what: str) -> int:
    raw = conn.execute(query).fetchone()[0]
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise SyncLimitError(f"could not read {what} on the branch (got {raw!r})") from None


def free_slots(conn: Any, headroom: int) -> FreeSlots:
    """max_connections - superuser_reserved_connections - connections now - headroom, read live."""
    return FreeSlots(
        max_connections=_read_int(conn, "SHOW max_connections", "max_connections"),
        reserved=_read_int(conn, "SHOW superuser_reserved_connections", "superuser_reserved_connections"),
        now=_read_int(conn, CONNECTIONS_NOW_SQL, "the connection count"),
        headroom=headroom,
    )


def sync_writers(conn: Any) -> int:
    """How many sync sessions (all syncs on the branch) are connected now."""
    return _read_int(conn, SYNC_WRITERS_SQL, "the sync writer count")


def per_sync(high_water: Optional[int]) -> int:
    """Connections to budget for one sync: max(CONNECTIONS_PER_SYNC_FLOOR, high-water) + PER_SYNC_MARGIN."""
    return max(CONNECTIONS_PER_SYNC_FLOOR, high_water or 0) + PER_SYNC_MARGIN


def parallel_now(max_parallel: int, slots: FreeSlots, need: int) -> int:
    """How many syncs may run at once now: min(sync_max_parallel, free // per_sync), never below 0."""
    return min(max_parallel, max(slots.free, 0) // need)


def refuse_no_room(slots: FreeSlots, need: int, high_water: Optional[int]) -> None:
    """Not even one sync fits and none is running: raise SyncLimitError with the numbers."""
    raise SyncLimitError(
        f"no room for one sync on the branch: {slots.describe()}; one sync needs {need} "
        f"(max({CONNECTIONS_PER_SYNC_FLOOR}, high-water {high_water}) + {PER_SYNC_MARGIN}). "
        f"Lower app_connection_headroom, wait for the app to release connections, or raise the compute size."
    )
