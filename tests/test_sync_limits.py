"""Offline tests for the sync limits.

- Provision refuses a synced table that would pass the per-source cap (20) minus the headroom
  (default 2), counting version schemas from UC (SHOW SCHEMAS / SHOW TABLES), never the manifest.
- The nightly sync runs at most `sync_max_parallel` (default 2) pipeline updates at once, and (adaptive
  parallelism) before each start at most free // per_sync, free = max_connections -
  superuser_reserved_connections - named sessions now - app headroom, per_sync = max(18, high-water) + 2;
  it refuses only when not even one fits and nothing is running.

Fakes stand in for the SDK API objects, the warehouse and the Postgres connection only.
"""

import re
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from databricks.sdk.service import postgres as pg

from lgk import manifest as mf
from lgk import sync_limits
from lgk.refresh import sync
from lgk.steps import provision as prov

CAT = "cat"
KIT = "lgk_v1_x"
BUILT = ("nodes", "edges", "node_id_map")


def _provision_config(tmp_path, **extra: Any) -> Dict[str, Any]:
    cfg = {
        "lakebase_catalog": CAT,
        "graph_name": "g",
        "kit_schema": KIT,
        "lakebase_project": "proj",
        "branch_id": "t-current",
        "warehouse_id": "wh",
        "run_folder": str(tmp_path),
    }
    cfg.update(extra)
    return cfg


def _provision_w():
    w = MagicMock()
    w.statement_execution.execute_statement.return_value = SimpleNamespace(
        status=SimpleNamespace(state="SUCCEEDED"), result=None
    )
    w.schemas.get.return_value = SimpleNamespace(comment="lgk:graph=g")
    w.postgres.get_synced_table.return_value = SimpleNamespace(
        spec=SimpleNamespace(branch="projects/proj/branches/t-current"),
        status=SimpleNamespace(pipeline_id="p",
                               detailed_state=SimpleNamespace(value="SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE")),
    )
    return w


def _uc(version_schemas: Dict[str, List[str]], seen: List[str]):
    """A warehouse answering SHOW SCHEMAS LIKE and SHOW TABLES from `version_schemas`."""
    def execute_sql(w, wh, statement):
        seen.append(statement)
        if statement == f"SHOW SCHEMAS IN {CAT} LIKE '{KIT}_v*'":
            return [[s] for s in version_schemas]
        m = re.fullmatch(rf"SHOW TABLES IN {CAT}\.(\S+)", statement)
        if m:
            return [[m.group(1), t, "false"] for t in version_schemas.get(m.group(1), [])]
        return []
    return execute_sql


def _older_versions(n: int, first: int = 2) -> Dict[str, List[str]]:
    return {f"{KIT}_v{i}": list(BUILT) for i in range(first, first + n)}


# ------------------------------------------------------------- provision: per-source room
@pytest.mark.parametrize(
    "existing, orphans, refused",
    [
        (19, 0, True),   # 19 + 1 > 20 - 2
        (17, 0, False),  # boundary: 17 + 1 == 18, allowed
        (18, 0, True),   # first refused count
        # hostile: an orphan version schema (in UC, absent from this run's manifest: made by
        # the bundle's SP from another run folder, or its branch was deleted) counts.
        (17, 1, True),
    ],
)
def test_provision_refuses_at_limit_minus_headroom(tmp_path, monkeypatch, existing, orphans, refused):
    cfg = _provision_config(tmp_path)
    recorded = _older_versions(existing)
    uc = dict(recorded)
    uc.update(_older_versions(orphans, first=2 + existing))
    # hostile: this run's own (just claimed, empty) version schema is listed too, and a schema
    # the LIKE pattern matches that is not a version schema holds the same table names.
    uc[f"{KIT}_v1"] = []
    uc[f"{KIT}_vault"] = list(BUILT)
    for schema in recorded:  # the manifest knows only the non-orphan versions
        for t in BUILT:
            mf.record(cfg, mf.ObjectType.SYNCED_TABLE, f"{CAT}.{schema}.{t}", "uc")
    seen: List[str] = []
    w = _provision_w()
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: w)
    monkeypatch.setattr(prov, "execute_sql", _uc(uc, seen))
    if refused:
        with pytest.raises(sync_limits.SyncLimitError) as e:
            prov._create_synced_tables(cfg)
        msg = str(e.value)
        assert "retire a version or run `lgk.reaper.reap`" in msg
        assert f"{KIT}_v2" in msg and f"{KIT}_v{1 + existing + orphans}" in msg
        assert f"{KIT}_vault" not in msg
        assert w.postgres.create_synced_table.call_count == 0  # refused before the first create
    else:
        prov._create_synced_tables(cfg)
        assert w.postgres.create_synced_table.call_count == 3
    assert f"SHOW SCHEMAS IN {CAT} LIKE '{KIT}_v*'" in seen


def test_the_headroom_comes_from_config(tmp_path, monkeypatch):
    # hostile: 19 existing would be refused under the default headroom 2; headroom 0 allows it
    cfg = _provision_config(tmp_path, sync_per_source_headroom=0)
    w = _provision_w()
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: w)
    monkeypatch.setattr(prov, "execute_sql", _uc(_older_versions(19), []))
    prov._create_synced_tables(cfg)
    assert w.postgres.create_synced_table.call_count == 3


def test_enrichment_tables_about_to_be_synced_count_too(tmp_path, monkeypatch):
    # hostile: only node_score is at the cap; the built tables have room
    cfg = _provision_config(tmp_path)
    (tmp_path / "enrich_report.json").write_text('{"degree": {"rows": 1}}')  # degree wrote node_score
    uc = {f"{KIT}_v{i}": ["node_score"] for i in range(2, 20)}  # 18 versions hold node_score
    seen: List[str] = []
    base = _uc(uc, seen)

    def execute_sql(w, wh, statement):
        if statement == f"SHOW TABLES IN {CAT}.{KIT}":
            return [[KIT, t, "false"] for t in (*BUILT, "node_score")]
        return base(w, wh, statement)

    w = _provision_w()
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: w)
    monkeypatch.setattr(prov, "execute_sql", execute_sql)
    with pytest.raises(sync_limits.SyncLimitError, match="node_score"):
        prov._create_synced_tables(cfg)
    assert w.postgres.create_synced_table.call_count == 0


def test_per_source_counts_counts_version_schemas_per_table():
    seen: List[str] = []
    uc = {f"{KIT}_v3": ["nodes", "edges"], f"{KIT}_v12": ["nodes"], f"{KIT}_vault": ["nodes"],
          f"{KIT}_v2x": ["nodes"]}
    counts = sync_limits.per_source_counts(lambda s: _uc(uc, seen)(None, None, s), CAT, KIT)
    assert dict(counts) == {"nodes": 2, "edges": 1}
    assert counts.schemas == {"nodes": [f"{KIT}_v12", f"{KIT}_v3"], "edges": [f"{KIT}_v3"]}
    # the non-version schemas are not even listed for their tables
    assert f"SHOW TABLES IN {CAT}.{KIT}_vault" not in seen


def test_assert_room_names_only_the_tables_at_the_cap():
    counts = sync_limits.SourceCounts({"nodes": 18, "edges": 3})
    counts.schemas = {"nodes": [f"{KIT}_v{i}" for i in range(2, 20)], "edges": [f"{KIT}_v2"]}
    with pytest.raises(sync_limits.SyncLimitError) as e:
        sync_limits.assert_room(counts, ["edges", "nodes"])
    assert "nodes" in str(e.value) and "edges" not in str(e.value)
    sync_limits.assert_room(counts, ["edges", "brand_new"])  # unknown table counts 0


@pytest.mark.parametrize("bad", [-1, True, "2", 1.5, None])
def test_a_bad_per_source_headroom_is_refused(tmp_path, monkeypatch, bad):
    cfg = _provision_config(tmp_path, sync_per_source_headroom=bad)
    w = _provision_w()
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: w)
    monkeypatch.setattr(prov, "execute_sql", _uc({}, []))
    with pytest.raises(sync_limits.SyncLimitError, match="sync_per_source_headroom"):
        prov._create_synced_tables(cfg)
    assert w.postgres.create_synced_table.call_count == 0


# ------------------------------------------------------------- refresh: bounded parallel sync
def _sync_config(tmp_path, **extra: Any) -> Dict[str, Any]:
    cfg = {
        "graph_name": "provider_graph",
        "lakebase_project": "proj",
        "lakebase_catalog": CAT,
        "kit_schema": "lgk_v1_abc123",
        "branch_id": "t-x",
        "run_folder": str(tmp_path),
        "warehouse_id": "wh",
        "version": 1,
    }
    cfg.update(extra)
    # only a reported method's table syncs; degree writes node_score, not node_group
    (tmp_path / "enrich_report.json").write_text('{"degree": {"rows": 1}}')
    for t in ("nodes", "edges", "node_id_map", "node_score", "node_group"):
        mf.record(cfg, mf.ObjectType.SYNCED_TABLE, f"{CAT}.lgk_v1_abc123_v1.{t}", "uc")
    return cfg


def _show_kit(*names: str):
    return lambda s: [["lgk_v1_abc123", n, "false"] for n in names]


class _OverlapW:
    """Fake w: records how many updates are running at once (started, not yet seen terminal)."""

    def __init__(self, polls_until_done: int = 4):
        self.lock = threading.Lock()
        self.running = 0
        self.max_running = 0
        self.started: List[str] = []
        self.polls: Dict[str, int] = {}
        self.events: List[str] = []
        self.polls_until_done = polls_until_done
        self.postgres = SimpleNamespace(get_synced_table=self._get_synced_table)
        self.pipelines = SimpleNamespace(start_update=self._start, get_update=self._get_update)

    @staticmethod
    def _get_synced_table(name):
        return pg.SyncedTable(status=pg.SyncedTableSyncedTableStatus(pipeline_id=f"pipe-{name.split('.')[-1]}"))

    def _start(self, pipeline_id, full_refresh):
        with self.lock:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
            self.started.append(pipeline_id)
            self.events.append(f"start {pipeline_id}")
        return SimpleNamespace(update_id=f"u-{pipeline_id}")

    def _get_update(self, pipeline_id, update_id):
        with self.lock:
            n = self.polls[pipeline_id] = self.polls.get(pipeline_id, 0) + 1
            state = "COMPLETED" if n >= self.polls_until_done else "RUNNING"
            if state == "COMPLETED":
                self.running -= 1
                self.events.append(f"done {pipeline_id}")
        return SimpleNamespace(update=SimpleNamespace(state=SimpleNamespace(value=state)))


WRITER = ("databricks_writer_77", "Databricks-Sync-Executor")
# Without pg_monitor a plain role sees usename + application_name but NULL backend_type (facts file),
# so every scripted row carries backend_type None.
_PREDICATES = {
    "": lambda u, a: True,
    "usename IS NOT NULL": lambda u, a: u is not None,
    "backend_type = 'client backend'": lambda u, a: False,
    "usename LIKE 'databricks_writer%' OR application_name LIKE 'Databricks-Sync-%'":
        lambda u, a: (u or "").startswith("databricks_writer") or (a or "").startswith("Databricks-Sync-"),
}


class _Conn:
    """Branch connection fake: SHOW settings, and counts over a scripted pg_stat_activity.

    named: app sessions now (a list = one value per free-slot read, the last repeats); unnamed:
    background workers (usename NULL); per_running: sync sessions per update running on `w`.
    """

    def __init__(self, max_connections: Any, reserved: Any = "0", named: Any = 0, unnamed: int = 0,
                 w: Any = None, per_running: int = 0):
        self.max_connections, self.reserved = max_connections, reserved
        self.named = named if isinstance(named, list) else [named]
        self.unnamed, self.w, self.per_running = unnamed, w, per_running
        self.queries: List[str] = []
        self.reads = 0  # free-slot reads (the named-session count queries)
        self.closed = False
        self.autocommit = False

    def _rows(self) -> List[tuple]:
        app = [("app_user", "directory_app")] * self.named[min(self.reads, len(self.named) - 1)]
        running = self.w.running if self.w is not None else 0
        return app + [(None, None)] * self.unnamed + [WRITER] * (running * self.per_running)

    def _count(self, query: str) -> int:
        where = query.split(" WHERE ", 1)[1] if " WHERE " in query else ""
        assert where in _PREDICATES, f"unscripted predicate: {where!r}"
        n = sum(1 for u, a in self._rows() if _PREDICATES[where](u, a))
        if where == "usename IS NOT NULL":
            self.reads += 1
        return n

    def execute(self, query):
        self.queries.append(query)
        if query == "SHOW max_connections":
            value = self.max_connections
        elif query == "SHOW superuser_reserved_connections":
            value = self.reserved
        else:
            assert query.startswith("SELECT count(*) FROM pg_stat_activity"), query
            value = self._count(query)
        return SimpleNamespace(fetchone=lambda: (value,))

    def close(self):
        self.closed = True


ROOM_READ = ["SHOW max_connections", "SHOW superuser_reserved_connections",
             "SELECT count(*) FROM pg_stat_activity WHERE usename IS NOT NULL"]


@pytest.fixture
def quick_waits(monkeypatch):
    real_sleep = time.sleep
    # a short real sleep: updates stay RUNNING long enough for the next one to start
    monkeypatch.setattr(sync.time, "sleep", lambda s: real_sleep(0.02))
    monkeypatch.setattr(sync, "_wait_synced_table_online", lambda w, dest: None)
    monkeypatch.setattr(sync, "SAMPLE_INTERVAL_S", 0.005)


ALL4 = ("nodes", "edges", "node_id_map", "node_score")


@pytest.mark.parametrize("parallel_cfg, expected_max", [({}, 2), ({"sync_max_parallel": 3}, 3)])
def test_sync_all_never_has_more_than_max_parallel_running(tmp_path, monkeypatch, quick_waits,
                                                          parallel_cfg, expected_max):
    conn = _Conn("1000")
    monkeypatch.setattr(sync, "connect_branch", lambda cfg: conn)
    w = _OverlapW()
    # hostile: four synced tables (enrichment included), more than either bound
    sync.sync_all(w, _show_kit(*ALL4, "_enrich_deg"), _sync_config(tmp_path, **parallel_cfg))
    assert sorted(w.started) == sorted(f"pipe-{t}" for t in ALL4)  # every update ran
    assert w.max_running == expected_max  # bounded, and actually parallel up to the bound
    assert w.running == 0


def test_sync_all_refuses_when_the_branch_cannot_hold_the_sync_connections(tmp_path, monkeypatch, quick_waits):
    conn = _Conn("39")  # 39 - 0 - 0 - 20 = 19 free; one sync needs max(18, none) + 2 = 20
    monkeypatch.setattr(sync, "connect_branch", lambda cfg: conn)
    w = _OverlapW()
    with pytest.raises(sync_limits.SyncLimitError, match=r"max_connections 39.*= 19 free slots.*needs 20"):
        sync.sync_all(w, _show_kit(*ALL4), _sync_config(tmp_path))
    assert w.started == []  # checked before the first start
    assert conn.queries == ROOM_READ
    assert conn.closed


def test_sync_all_allows_exactly_enough_connections(tmp_path, monkeypatch, quick_waits):
    monkeypatch.setattr(sync, "connect_branch", lambda cfg: _Conn("40"))  # 20 free = per_sync 20
    w = _OverlapW()
    sync.sync_all(w, _show_kit(*ALL4), _sync_config(tmp_path))
    assert len(w.started) == 4


@pytest.mark.parametrize("extra, max_conn, refused", [
    ({"sync_max_parallel": 3}, "39", True),         # 39 - 20 = 19 free < one sync (20)
    ({"app_connection_headroom": 0}, "20", False),  # 20 - 0 = 20 free = one sync
    ({"app_connection_headroom": 40}, "59", True),  # 59 - 40 = 19 free < one sync
])
def test_the_connection_room_reads_parallel_and_headroom_from_config(tmp_path, monkeypatch, quick_waits,
                                                                     extra, max_conn, refused):
    monkeypatch.setattr(sync, "connect_branch", lambda cfg: _Conn(max_conn))
    w = _OverlapW()
    if refused:
        with pytest.raises(sync_limits.SyncLimitError):
            sync.sync_all(w, _show_kit(*ALL4), _sync_config(tmp_path, **extra))
        assert w.started == []
    else:
        sync.sync_all(w, _show_kit(*ALL4), _sync_config(tmp_path, **extra))
        assert len(w.started) == 4


def test_an_unreadable_max_connections_fails_loudly(tmp_path, monkeypatch, quick_waits):
    monkeypatch.setattr(sync, "connect_branch", lambda cfg: _Conn("unknown"))
    w = _OverlapW()
    with pytest.raises(sync_limits.SyncLimitError, match="max_connections"):
        sync.sync_all(w, _show_kit(*ALL4), _sync_config(tmp_path))
    assert w.started == []


@pytest.mark.parametrize("key, bad", [
    ("sync_max_parallel", 0), ("sync_max_parallel", True), ("sync_max_parallel", "2"),
    ("app_connection_headroom", -1), ("app_connection_headroom", 2.5),
])
def test_bad_sync_parallelism_config_is_refused_before_any_start(tmp_path, monkeypatch, quick_waits, key, bad):
    monkeypatch.setattr(sync, "connect_branch", lambda cfg: _Conn("1000"))
    w = _OverlapW()
    with pytest.raises(sync_limits.SyncLimitError, match=key):
        sync.sync_all(w, _show_kit(*ALL4), _sync_config(tmp_path, **{key: bad}))
    assert w.started == []


def test_a_failed_update_stops_new_starts(tmp_path, monkeypatch, quick_waits):
    monkeypatch.setattr(sync, "connect_branch", lambda cfg: _Conn("1000"))
    w = _OverlapW()
    real = w._get_update

    def get_update(pipeline_id, update_id):
        if pipeline_id == "pipe-nodes":
            with w.lock:
                w.running -= 1
            return SimpleNamespace(update=SimpleNamespace(state=SimpleNamespace(value="FAILED")))
        return real(pipeline_id, update_id)

    w.pipelines.get_update = get_update
    cfg = _sync_config(tmp_path, sync_max_parallel=1)
    with pytest.raises(sync.SyncUpdateError, match="pipe-nodes.*FAILED"):
        sync.sync_all(w, _show_kit(*ALL4), cfg)
    assert w.started == ["pipe-nodes"]  # parallel 1: nothing else started after the failure


# ------------------------------------------------------------- provision: the platform's own refusal
@pytest.mark.parametrize("platform_text", [
    "INVALID_PARAMETER_VALUE: Cannot create more than 20 synced database table(s) per source table.",
    "cannot create more than 25 synced database table(s) per source table",  # hostile: case and cap vary
])
def test_the_platforms_21st_table_refusal_becomes_the_kits_sync_limit_error(tmp_path, monkeypatch, platform_text):
    # hostile: the UC pre-check sees room (synced tables made by hand outside <kit>_v*, a gap seen
    # live), so only the platform's create-time refusal (verbatim text) catches the cap.
    cfg = _provision_config(tmp_path)
    w = _provision_w()
    w.postgres.create_synced_table.side_effect = Exception(platform_text)
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: w)
    monkeypatch.setattr(prov, "execute_sql", _uc(_older_versions(3), []))
    with pytest.raises(sync_limits.SyncLimitError) as e:
        prov._create_synced_tables(cfg)
    msg = str(e.value)
    assert f"{CAT}.{KIT}.nodes" in msg  # names the source at its cap
    assert "retire a version or run `lgk.reaper.reap`" in msg
    assert platform_text in msg  # the platform's own words are kept
    assert w.postgres.get_synced_table.call_count == 0  # never treated as "already exists" / waited on


def test_another_create_error_is_not_mistaken_for_the_cap(tmp_path, monkeypatch):
    # hostile: an error that mentions "20" and "synced" but is not the cap refusal
    cfg = _provision_config(tmp_path)
    w = _provision_w()
    w.postgres.create_synced_table.side_effect = Exception("PERMISSION_DENIED: 20 synced table grants missing")
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: w)
    monkeypatch.setattr(prov, "execute_sql", _uc({}, []))
    with pytest.raises(Exception, match="PERMISSION_DENIED") as e:
        prov._create_synced_tables(cfg)
    assert not isinstance(e.value, sync_limits.SyncLimitError)


def test_per_source_counts_matches_version_schemas_whatever_their_case():
    # hostile: UC returns schema names lower-cased while the configured kit has capitals
    uc = {f"lgk_v1_x_v{i}": ["nodes"] for i in range(2, 5)}
    rows = lambda s: ([[n] for n in uc] if s.startswith("SHOW SCHEMAS")  # noqa: E731
                      else [[s.rsplit(".", 1)[1], "nodes", "false"]])
    counts = sync_limits.per_source_counts(rows, CAT, "LGK_v1_X")
    assert dict(counts) == {"nodes": 3}


def test_the_nightly_sync_entry_cannot_skip_the_connection_room(tmp_path, monkeypatch, quick_waits):
    """The live task entry `sync.sync` goes through the room check (safety checks cannot be skipped)."""
    w, conn = _OverlapW(), _Conn("39")  # 19 free < 20
    monkeypatch.setattr(sync, "live_clients", lambda cfg: (w, _show_kit(*ALL4)))
    monkeypatch.setattr(sync, "connect_branch", lambda cfg: conn)
    monkeypatch.setattr(sync, "_require_stamped_tables", lambda sql, cfg, w: None)  # the stamp check runs first; not under test here
    with pytest.raises(sync_limits.SyncLimitError, match="max_connections"):
        sync.sync(_sync_config(tmp_path))
    assert w.started == [] and conn.queries == ROOM_READ


# ------------------------------------------------------------- adaptive parallelism from live free slots
import json  # noqa: E402
from pathlib import Path  # noqa: E402


def _write_high_water(cfg: Dict[str, Any], value: Any) -> None:
    """Put the high-water field straight into the run manifest JSON (not through the code under test)."""
    path = Path(cfg["run_folder"]) / "lgk_manifest.json"
    data = json.loads(path.read_text())
    data["sync_connection_high_water"] = value
    path.write_text(json.dumps(data))


def _manifest_json(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads((Path(cfg["run_folder"]) / "lgk_manifest.json").read_text())


def test_a_small_branch_runs_the_syncs_one_at_a_time(tmp_path, monkeypatch, quick_waits, caplog):
    w = _OverlapW()
    # 60 - 7 reserved - 3 app - 20 headroom = 30 free: one sync (20) fits; once it runs its 18
    # sessions leave 12. hostile: sync_max_parallel 3 allows more than the branch holds
    conn = _Conn("60", reserved="7", named=3, w=w, per_running=18)
    monkeypatch.setattr(sync, "connect_branch", lambda cfg: conn)
    with caplog.at_level("INFO", logger="lgk.refresh.sync"):
        sync.sync_all(w, _show_kit(*ALL4), _sync_config(tmp_path, sync_max_parallel=3))
    assert sorted(w.started) == sorted(f"pipe-{t}" for t in ALL4)
    assert w.max_running == 1
    assert ("sync: parallel 1 now (min(sync_max_parallel 3, free 30 // per sync 20)); 0 running; "
            "max_connections 60 - superuser_reserved_connections 7 - connections now 3 - app headroom 20 "
            "= 30 free slots") in caplog.text
    assert conn.autocommit is True and conn.closed


def test_a_busy_app_refuses_before_any_start_with_the_numbers(tmp_path, monkeypatch, quick_waits):
    w = _OverlapW()
    # hostile: realistic 0.5-1 CU branch, app holding 180: 225 - 7 - 180 - 20 = 18 < 20
    conn = _Conn("225", reserved="7", named=180, w=w, per_running=18)
    monkeypatch.setattr(sync, "connect_branch", lambda cfg: conn)
    with pytest.raises(sync_limits.SyncLimitError) as e:
        sync.sync_all(w, _show_kit(*ALL4), _sync_config(tmp_path))
    assert w.started == []
    msg = str(e.value)
    for part in ("max_connections 225", "superuser_reserved_connections 7", "connections now 180",
                 "app headroom 20", "= 18 free slots", "one sync needs 20", "high-water None"):
        assert part in msg, (part, msg)


@pytest.mark.parametrize("high_water, expected_max", [(None, 2), (40, 1)])
def test_the_high_water_mark_raises_per_sync(tmp_path, monkeypatch, quick_waits, high_water, expected_max):
    w = _OverlapW()
    # 100 - 7 - 3 - 20 = 70 free, constant (per_running 0): per_sync 20 -> 3 fit (bounded to 2);
    # high-water 40 -> per_sync 42 -> 1 fits
    conn = _Conn("100", reserved="7", named=3, w=w)
    monkeypatch.setattr(sync, "connect_branch", lambda cfg: conn)
    cfg = _sync_config(tmp_path)
    if high_water is not None:
        _write_high_water(cfg, high_water)
    sync.sync_all(w, _show_kit(*ALL4), cfg)
    assert w.max_running == expected_max and len(w.started) == 4


@pytest.mark.parametrize("free, expected_max", [(39, 1), (40, 2)])
def test_per_sync_is_the_floor_plus_two_when_the_high_water_is_below_it(tmp_path, monkeypatch, quick_waits,
                                                                        free, expected_max):
    w = _OverlapW()
    conn = _Conn(str(free + 20), w=w)  # constant free slots (no sync sessions scripted)
    monkeypatch.setattr(sync, "connect_branch", lambda cfg: conn)
    cfg = _sync_config(tmp_path)
    _write_high_water(cfg, 5)  # hostile: a recorded peak below the floor must not lower per_sync
    sync.sync_all(w, _show_kit(*ALL4), cfg)
    assert w.max_running == expected_max  # 39 // 20 = 1, 40 // 20 = 2


def test_the_room_is_rechecked_before_each_start_and_grows(tmp_path, monkeypatch, quick_waits):
    w = _OverlapW()
    # app 15 for the first two reads (60 - 15 - 20 = 25: one fits), then it releases them (40: two fit)
    conn = _Conn("60", named=[15, 15, 0], w=w)
    monkeypatch.setattr(sync, "connect_branch", lambda cfg: conn)
    sync.sync_all(w, _show_kit(*ALL4), _sync_config(tmp_path))
    first = w.started[0]
    assert w.events.index(f"done {first}") < w.events.index(f"start {w.started[1]}")  # alone at first
    assert w.max_running == 2  # then two at once, after the re-check
    assert conn.reads >= 4  # one read per start at least


def test_the_room_is_rechecked_between_syncs_and_refuses_when_it_shrinks(tmp_path, monkeypatch, quick_waits):
    w = _OverlapW()
    # 40 free for the first two reads; after the first sync ends the app takes 100 sessions
    conn = _Conn("60", named=[0, 0, 100], w=w)
    monkeypatch.setattr(sync, "connect_branch", lambda cfg: conn)
    with pytest.raises(sync_limits.SyncLimitError, match="connections now 100"):
        sync.sync_all(w, _show_kit(*ALL4), _sync_config(tmp_path, sync_max_parallel=1))
    assert w.started == ["pipe-nodes"] and w.running == 0


def test_connections_now_counts_every_named_session_only(tmp_path, monkeypatch, quick_waits):
    # hostile: 30 background workers with no usename, and (no pg_monitor) a NULL backend_type everywhere
    conn = _Conn("60", named=20, unnamed=30)
    slots = sync_limits.free_slots(conn, 20)
    assert (slots.max_connections, slots.reserved, slots.now, slots.free) == (60, 0, 20, 20)
    assert "SELECT count(*) FROM pg_stat_activity WHERE usename IS NOT NULL" in conn.queries
    w = _OverlapW()
    monkeypatch.setattr(sync, "connect_branch", lambda cfg: _Conn("60", named=20, unnamed=30, w=w))
    sync.sync_all(w, _show_kit(*ALL4), _sync_config(tmp_path))  # 20 free: exactly one at a time
    assert len(w.started) == 4 and w.max_running == 1


def test_the_nights_peak_is_kept_as_the_high_water_mark(tmp_path, monkeypatch, quick_waits):
    cfg = _sync_config(tmp_path, sync_max_parallel=1)
    w = _OverlapW()
    monkeypatch.setattr(sync, "connect_branch", lambda c: _Conn("1000", w=w, per_running=25))
    sync.sync_all(w, _show_kit(*ALL4), cfg)
    assert _manifest_json(cfg)["sync_connection_high_water"] == 25
    assert len(_manifest_json(cfg)["entries"]) == 5  # the records are kept
    # hostile: a lighter night never lowers the mark
    w2 = _OverlapW()
    monkeypatch.setattr(sync, "connect_branch", lambda c: _Conn("1000", w=w2, per_running=10))
    sync.sync_all(w2, _show_kit(*ALL4), cfg)
    assert _manifest_json(cfg)["sync_connection_high_water"] == 25
    # and the next night budgets 25 + 2: 70 - 20 = 50 free holds one sync of 27, not two
    w3 = _OverlapW()
    monkeypatch.setattr(sync, "connect_branch", lambda c: _Conn("70", w=w3))
    sync.sync_all(w3, _show_kit(*ALL4), dict(cfg, sync_max_parallel=2))
    assert w3.max_running == 1


def test_a_failing_sampler_never_fails_the_night(tmp_path, monkeypatch, quick_waits, caplog):
    w = _OverlapW()

    class _NoWriters(_Conn):
        def execute(self, query):
            if "databricks_writer" in query:
                raise RuntimeError("permission denied for pg_stat_activity")
            return super().execute(query)

    monkeypatch.setattr(sync, "connect_branch", lambda c: _NoWriters("1000", w=w))
    cfg = _sync_config(tmp_path)
    with caplog.at_level("WARNING", logger="lgk.refresh.sync"):
        sync.sync_all(w, _show_kit(*ALL4), cfg)
    assert len(w.started) == 4
    assert "sync_connection_high_water" not in _manifest_json(cfg)
    assert "sampler stopped" in caplog.text


@pytest.mark.parametrize("bad", ["40", -1, True, 2.5])
def test_a_corrupt_high_water_is_refused_before_any_start(tmp_path, monkeypatch, quick_waits, bad):
    w = _OverlapW()
    monkeypatch.setattr(sync, "connect_branch", lambda c: _Conn("1000", w=w))
    cfg = _sync_config(tmp_path)
    _write_high_water(cfg, bad)
    with pytest.raises(ValueError, match="sync_connection_high_water"):
        sync.sync_all(w, _show_kit(*ALL4), cfg)
    assert w.started == []


# ------------------------------------------------------------- adaptive parallelism: prover survivors
@pytest.mark.parametrize("which", ["SHOW superuser_reserved_connections", "usename IS NOT NULL"])
def test_an_unreadable_reserved_or_connection_count_refuses(tmp_path, monkeypatch, quick_waits, which):
    w = _OverlapW()

    class _Unreadable(_Conn):
        def execute(self, query):
            if which in query:
                self.queries.append(query)
                return SimpleNamespace(fetchone=lambda: (None,))
            return super().execute(query)

    monkeypatch.setattr(sync, "connect_branch", lambda c: _Unreadable("1000", w=w))  # roomy otherwise
    with pytest.raises(sync_limits.SyncLimitError, match="could not read"):
        sync.sync_all(w, _show_kit(*ALL4), _sync_config(tmp_path))
    assert w.started == []


def test_two_overlapping_syncs_record_one_syncs_peak_not_their_sum(tmp_path, monkeypatch, quick_waits):
    w = _OverlapW(polls_until_done=8)
    # hostile: defaults (2 at once), 18 sessions per running update: the branch sees 36 at once
    monkeypatch.setattr(sync, "connect_branch", lambda c: _Conn("1000", w=w, per_running=18))
    cfg = _sync_config(tmp_path)
    sync.sync_all(w, _show_kit(*ALL4), cfg)
    assert w.max_running == 2
    assert _manifest_json(cfg)["sync_connection_high_water"] == 18


def test_the_peak_follows_sessions_that_rise_during_the_night(tmp_path, monkeypatch, quick_waits):
    w = _OverlapW(polls_until_done=8)

    class _Rising(_Conn):
        samples = 0

        def _count(self, query):
            if "databricks_writer" in query:
                _Rising.samples += 1
                self.per_running = 10 if _Rising.samples <= 3 else 30
            return super()._count(query)

    monkeypatch.setattr(sync, "connect_branch", lambda c: _Rising("1000", w=w, per_running=10))
    cfg = _sync_config(tmp_path, sync_max_parallel=1)
    sync.sync_all(w, _show_kit(*ALL4), cfg)
    assert _manifest_json(cfg)["sync_connection_high_water"] == 30


def test_a_sampler_that_fails_after_one_good_read_records_nothing(tmp_path, monkeypatch, quick_waits, caplog):
    w = _OverlapW(polls_until_done=8)

    class _OnceThenFail(_Conn):
        writer_reads = 0

        def execute(self, query):
            if "databricks_writer" in query:
                _OnceThenFail.writer_reads += 1
                if _OnceThenFail.writer_reads > 1:
                    raise RuntimeError("connection reset")
            return super().execute(query)

    monkeypatch.setattr(sync, "connect_branch", lambda c: _OnceThenFail("1000", w=w, per_running=25))
    cfg = _sync_config(tmp_path, sync_max_parallel=1)
    with caplog.at_level("WARNING", logger="lgk.refresh.sync"):
        sync.sync_all(w, _show_kit(*ALL4), cfg)
    assert len(w.started) == 4
    assert "sync_connection_high_water" not in _manifest_json(cfg)
    assert caplog.text.count("sampler stopped") == 1
    assert _OnceThenFail.writer_reads == 2  # no writer read after the failure


def test_a_failed_night_still_records_what_the_sampler_saw(tmp_path, monkeypatch, quick_waits):
    w = _OverlapW(polls_until_done=8)
    real = w._get_update

    def get_update(pipeline_id, update_id):
        r = real(pipeline_id, update_id)
        if pipeline_id == "pipe-edges" and r.update.state.value == "COMPLETED":
            return SimpleNamespace(update=SimpleNamespace(state=SimpleNamespace(value="FAILED")))
        return r

    w.pipelines.get_update = get_update
    monkeypatch.setattr(sync, "connect_branch", lambda c: _Conn("1000", w=w, per_running=25))
    cfg = _sync_config(tmp_path, sync_max_parallel=1)
    with pytest.raises(sync.SyncUpdateError, match="pipe-edges"):
        sync.sync_all(w, _show_kit(*ALL4), cfg)
    assert _manifest_json(cfg)["sync_connection_high_water"] == 25


def test_a_short_update_frees_its_slot_while_a_long_one_runs(tmp_path, monkeypatch, quick_waits):
    w = _OverlapW()
    long_polls = {"pipe-nodes": 40}
    real = w._get_update

    def get_update(pipeline_id, update_id):
        with w.lock:
            n = w.polls.get(pipeline_id, 0)
        if pipeline_id in long_polls and n + 1 < long_polls[pipeline_id]:
            with w.lock:
                w.polls[pipeline_id] = n + 1
            return SimpleNamespace(update=SimpleNamespace(state=SimpleNamespace(value="RUNNING")))
        return real(pipeline_id, update_id)

    w.polls_until_done = 2
    w.pipelines.get_update = get_update
    monkeypatch.setattr(sync, "connect_branch", lambda c: _Conn("1000", w=w))
    sync.sync_all(w, _show_kit("nodes", "edges", "node_id_map"), _sync_config(tmp_path))
    # hostile: one long update; the third starts as soon as the short second one ends
    assert w.events.index("start pipe-node_id_map") < w.events.index("done pipe-nodes")


@pytest.mark.parametrize("high_water, free, needs", [(40, 41, 42), (60, 61, 62)])
def test_the_refusal_names_the_recorded_high_water(tmp_path, monkeypatch, quick_waits, high_water, free, needs):
    w = _OverlapW()
    monkeypatch.setattr(sync, "connect_branch", lambda c: _Conn(str(free + 20), w=w))
    cfg = _sync_config(tmp_path)
    _write_high_water(cfg, high_water)
    with pytest.raises(sync_limits.SyncLimitError) as e:
        sync.sync_all(w, _show_kit(*ALL4), cfg)
    assert f"high-water {high_water}" in str(e.value) and f"needs {needs}" in str(e.value)
    assert w.started == []


def test_a_recorded_high_water_of_zero_is_valid(tmp_path, monkeypatch, quick_waits):
    w = _OverlapW()
    monkeypatch.setattr(sync, "connect_branch", lambda c: _Conn("40", w=w))  # 20 free = per_sync 20
    cfg = _sync_config(tmp_path)
    _write_high_water(cfg, 0)  # hostile: a night whose sampler saw no sync session
    sync.sync_all(w, _show_kit(*ALL4), cfg)
    assert len(w.started) == 4


def test_the_shipped_sample_interval_catches_an_executor_burst():
    # a live run saw a 16-executor burst over several 0.6 s samples; a long period misses it
    assert 0 < sync.SAMPLE_INTERVAL_S <= 5
