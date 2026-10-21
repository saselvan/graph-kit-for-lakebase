"""Offline: which enrichment tables provision syncs, and the stamp checks on them.

- The sync set is "a method this run's enrich_report.json lists writes it" (never "the name is
  present in UC"); closure also needs `serve_closure: true` (opt-in).
- provision refuses an enrichment table whose own stamp, or whose enrich record, is not the
  current privacy decision's.
- The per-source room check and the bounded refresh sync count every table of the set.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from lgk import manifest as mf
from lgk.privacy import binding
from tests.label_support import kit_label
from lgk.steps import provision as prov

CAT, KIT = "c", "k"
EVERY_METHOD = {"pagerank": {}, "degree": {}, "components": {}, "similarity": {}, "closure": {}}
# hostile: UC holds EVERY enrichment table (an earlier run wrote them all), whatever this run did
UC_TABLES = ("nodes", "edges", "node_id_map", "node_score", "node_group", "similar_to", "closure",
             "seed_score", "_enrich_deg")


def _cfg(tmp_path: Path, report: Any = None, **extra: Any) -> Dict[str, Any]:
    if report is not None:
        (tmp_path / "enrich_report.json").write_text(json.dumps(
            {"shape": {}, "menu": [], **{m: {} for m in report}}))
    return {"graph_name": "g", "lakebase_project": "p", "lakebase_catalog": CAT, "kit_schema": KIT,
            "branch_id": "t-x", "warehouse_id": "wh", "version": 2, "run_folder": str(tmp_path), **extra}


@pytest.fixture
def created(monkeypatch) -> List[str]:
    """provision's synced-table creation, recorded (UC answers SHOW TABLES with every table)."""
    log: List[str] = []
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: object())
    monkeypatch.setattr(prov.mf, "claim_uc_schema", lambda *a: False)
    monkeypatch.setattr(prov, "execute_sql", lambda w, wh, s: [[KIT, t, "false"] for t in UC_TABLES]
                        if s.startswith("SHOW TABLES") else [])
    monkeypatch.setattr(prov, "_create_one_synced_table",
                        lambda w, c, t, src, dest, pk, br: log.append((t, tuple(pk))))
    monkeypatch.setattr(prov, "_wait_synced_table_online", lambda w, dest: None)
    monkeypatch.setattr(prov, "_record_synced_table", lambda c, w, dest, br: None)
    monkeypatch.setattr(prov, "_assert_per_source_room",
                        lambda w, wh, c, tables: log.append(("room", tuple(tables))))
    return log


def _synced(log) -> List[str]:
    return [t for t, _pk in log if t != "room"]


def test_provision_syncs_only_reported_enrichment_tables(tmp_path, created):
    # hostile: the report lists similarity and degree only; node_group and closure are in UC
    prov._create_synced_tables(_cfg(tmp_path, ["similarity", "degree"]))
    assert _synced(created) == ["nodes", "edges", "node_id_map", "node_score", "similar_to"]


def test_the_menu_offering_a_method_is_not_running_it(tmp_path, created):
    # hostile (prover P23): a real report lists the full menu, every row available; only degree ran
    menu = [{"method": m, "table": t, "available": True, "reason": ""} for m, t in
            (("pagerank", "node_score"), ("degree", "node_score"), ("components", "node_group"),
             ("similarity", "similar_to"), ("closure", "closure"))]
    (tmp_path / "enrich_report.json").write_text(json.dumps({"shape": {}, "menu": menu, "degree": {}}))
    prov._create_synced_tables({**_cfg(tmp_path), "serve_closure": True})
    assert _synced(created) == ["nodes", "edges", "node_id_map", "node_score"]


def test_provision_syncs_no_enrichment_without_a_report(tmp_path, created):
    prov._create_synced_tables(_cfg(tmp_path))
    assert _synced(created) == ["nodes", "edges", "node_id_map"]


def test_each_enrichment_table_syncs_with_its_unique_primary_key(tmp_path, created):
    prov._create_synced_tables(_cfg(tmp_path, EVERY_METHOD, serve_closure=True))
    assert dict((t, pk) for t, pk in created if t != "room") == {
        "nodes": ("node_id",), "edges": ("edge_id",), "node_id_map": ("node_id",),
        "node_score": ("node_id", "method"), "node_group": ("node_id", "method"),
        "similar_to": ("src_id", "dst_id"), "closure": ("ancestor_id", "descendant_id")}


@pytest.mark.parametrize("serve, synced", [(None, False), (False, False), (True, True)])
def test_closure_is_opt_in(tmp_path, created, serve, synced):
    extra = {} if serve is None else {"serve_closure": serve}
    prov._create_synced_tables(_cfg(tmp_path, ["closure"], **extra))
    assert ("closure" in _synced(created)) is synced


@pytest.mark.parametrize("bad", ["true", 1, None])
def test_a_non_bool_serve_closure_is_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="serve_closure"):
        prov.enrichment_sync_set(_cfg(tmp_path, ["closure"], serve_closure=bad))


def test_the_room_check_counts_all_seven_tables(tmp_path, created):
    """The per-source check sees every table about to sync, the 4 enrichment ones included."""
    prov._create_synced_tables(_cfg(tmp_path, EVERY_METHOD, serve_closure=True))
    room = [tables for t, tables in created if t == "room"]
    assert room == [("nodes", "edges", "node_id_map", "node_score", "node_group", "similar_to", "closure")]
    assert created[0][0] == "room"  # before the first create


# ------------------------------------------------------------------------ stamp checks
class _Stamps:
    """A warehouse answering SHOW TBLPROPERTIES with a per-table stamp (None = no property)."""

    def __init__(self, stamps: Dict[str, Any]):
        self.stamps = stamps

    def __call__(self, w, wh, statement):
        table = statement.removeprefix("SHOW TBLPROPERTIES ")
        found = self.stamps[table]
        return [] if found is None else [[binding.STAMP_PROPERTY, found]]


def _bound(tmp_path, report, enrich_stamp="S", table_stamps=None, enrich_tables=None, **extra):
    cfg = _cfg(tmp_path, report, **extra)
    built = [f"{CAT}.{KIT}.{t}" for t in binding.BUILT_TABLES]
    binding.record_build(str(tmp_path), "S", built)
    enriched = enrich_tables if enrich_tables is not None else [
        f"{CAT}.{KIT}.{t}" for t in ("node_score", "node_group", "similar_to", "closure")]
    if enrich_stamp is not None:
        binding.record_enrich(str(tmp_path), enrich_stamp, enriched, list(report))
    stamps = {t: kit_label("S", t) for t in built}  # the kit tables carry the JSON label
    stamps.update({f"{CAT}.{KIT}.{t}": "S" for t in ("node_score", "node_group", "similar_to", "closure")})
    stamps.update(table_stamps or {})
    return cfg, _Stamps(stamps)


@pytest.fixture
def no_ws(monkeypatch):
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: object())


def test_provision_accepts_enrichment_under_the_current_stamp(tmp_path, monkeypatch, no_ws):
    cfg, stamps = _bound(tmp_path, ["degree", "similarity"])
    monkeypatch.setattr(prov, "execute_sql", stamps)
    prov._refuse_unbound_build(cfg, str(tmp_path), "S", {})


@pytest.mark.parametrize("found", ["OLD", None])
def test_provision_refuses_enrichment_from_another_stamp(tmp_path, monkeypatch, no_ws, found):
    # hostile: the run folder's records are all current; only node_score's own property is old/absent
    cfg, stamps = _bound(tmp_path, ["degree"], table_stamps={f"{CAT}.{KIT}.node_score": found})
    monkeypatch.setattr(prov, "execute_sql", stamps)
    with pytest.raises(binding.StaleBuildError, match="c.k.node_score"):
        prov._refuse_unbound_build(cfg, str(tmp_path), "S", {})


def test_an_unserved_enrichment_tables_stamp_is_not_checked(tmp_path, monkeypatch, no_ws):
    # hostile: closure carries an old stamp, but serve_closure is off, so it is not synced
    cfg, stamps = _bound(tmp_path, ["closure", "degree"], table_stamps={f"{CAT}.{KIT}.closure": "OLD"})
    monkeypatch.setattr(prov, "execute_sql", stamps)
    prov._refuse_unbound_build(cfg, str(tmp_path), "S", {})


def test_provision_refuses_an_enrich_record_from_another_stamp(tmp_path, monkeypatch, no_ws):
    # hostile: every table property is current; only the run folder's enrich record is old
    cfg, stamps = _bound(tmp_path, ["degree"], enrich_stamp="OLD")
    monkeypatch.setattr(prov, "execute_sql", stamps)
    with pytest.raises(binding.StaleBuildError, match="different privacy decision"):
        prov._refuse_unbound_build(cfg, str(tmp_path), "S", {})


def test_provision_refuses_enrichment_with_no_enrich_record(tmp_path, monkeypatch, no_ws):
    cfg, stamps = _bound(tmp_path, ["degree"], enrich_stamp=None)
    monkeypatch.setattr(prov, "execute_sql", stamps)
    with pytest.raises(binding.StaleBuildError, match="records no enrich run"):
        prov._refuse_unbound_build(cfg, str(tmp_path), "S", {})


def test_provision_refuses_a_table_the_enrich_record_does_not_list(tmp_path, monkeypatch, no_ws):
    # hostile: the report lists similarity, the record (same stamp) wrote only node_score
    cfg, stamps = _bound(tmp_path, ["degree", "similarity"], enrich_tables=[f"{CAT}.{KIT}.node_score"])
    monkeypatch.setattr(prov, "execute_sql", stamps)
    with pytest.raises(binding.StaleBuildError, match="c.k.similar_to"):
        prov._refuse_unbound_build(cfg, str(tmp_path), "S", {})


def test_no_enrichment_needs_no_enrich_record(tmp_path, monkeypatch, no_ws):
    cfg, stamps = _bound(tmp_path, [], enrich_stamp=None)
    monkeypatch.setattr(prov, "execute_sql", stamps)
    prov._refuse_unbound_build(cfg, str(tmp_path), "S", {})


# --------------------------------------------------------------- refresh uses the same set
def _recorded(cfg, tables):
    for t in tables:
        mf.record(cfg, mf.ObjectType.SYNCED_TABLE, f"{CAT}.{KIT}_v2.{t}", "uc")


def test_the_refresh_covers_every_synced_enrichment_table(tmp_path):
    from lgk.refresh.freshness import synced_pairs

    cfg = _cfg(tmp_path, EVERY_METHOD, serve_closure=True)
    _recorded(cfg, ("nodes", "edges", "node_id_map", "node_score", "node_group", "similar_to", "closure"))
    pairs = synced_pairs(cfg, lambda s: [[KIT, t, "false"] for t in UC_TABLES])
    assert [d.rsplit(".", 1)[1] for d, _ in pairs] == [
        "nodes", "edges", "node_id_map", "closure", "node_group", "node_score", "similar_to"]
    assert all(src == f"{CAT}.{KIT}.{d.rsplit('.', 1)[1]}" for d, src in pairs)


def test_the_refresh_skips_a_recorded_table_the_report_no_longer_lists(tmp_path):
    from lgk.refresh.freshness import enrichment_tables

    # hostile: node_group IS recorded as synced and IS in UC, but this run's report has no components
    cfg = _cfg(tmp_path, ["degree"])
    _recorded(cfg, ("node_score", "node_group"))
    assert enrichment_tables(lambda s: [[KIT, t, "false"] for t in UC_TABLES], cfg) == ["node_score"]


def test_sync_all_updates_every_served_enrichment_table(tmp_path, monkeypatch):
    from lgk.refresh import sync

    cfg = _cfg(tmp_path, EVERY_METHOD, serve_closure=True)
    _recorded(cfg, ("nodes", "edges", "node_id_map", "node_score", "node_group", "similar_to", "closure"))
    from tests.test_refresh import _RoomyConn

    monkeypatch.setattr(sync, "connect_branch", lambda c: _RoomyConn())  # room for every sync
    started: List[str] = []
    monkeypatch.setattr(sync, "_sync_one", lambda w, dest, full: started.append(dest.rsplit(".", 1)[1]))
    sync.sync_all(object(), lambda s: [[KIT, t, "false"] for t in UC_TABLES], cfg)
    assert sorted(started) == sorted(("nodes", "edges", "node_id_map", "node_score", "node_group",
                                      "similar_to", "closure"))


def test_the_enrichment_tables_are_the_optional_views_tables():
    from lgk import versions

    assert tuple(prov.ENRICHMENT_SYNCED_TABLES) == versions.OPTIONAL_TABLES


# ------------------------------------------------------- reader lock-out and reapply indexes
class _Cursor:
    """A cursor answering to_regclass from `present` (schema.table names); records statements."""

    def __init__(self, present):
        self.present, self.log, self._row = set(present), [], None

    def execute(self, sql, params=None):
        if sql.startswith("SELECT to_regclass"):
            self._row = (params[0] if params[0] in self.present else None,)
            return
        if sql.startswith("SELECT 1 FROM pg_roles"):
            self._row = (1,)
            return
        self.log.append(sql)

    def fetchone(self):
        return self._row

    def close(self):
        pass


def _cursor_conn(monkeypatch, cur):
    conn = type("C", (), {"cursor": lambda self: cur, "commit": lambda self: None, "close": lambda self: None})()
    monkeypatch.setattr(prov, "connect_branch", lambda c: conn)


def test_provision_revokes_reader_on_every_enrichment_table_present(monkeypatch):
    # hostile: node_group and closure are present, node_score is not
    cur = _Cursor({"k_v2.node_group", "k_v2.closure"})
    _cursor_conn(monkeypatch, cur)
    prov._lock_version_from_reader({"graph_name": "g", "kit_schema": "k", "version": 2})
    revokes = [s for s in cur.log if s.startswith("REVOKE ALL ON k_v2.")]
    assert revokes == [f"REVOKE ALL ON k_v2.{t} FROM reader"
                       for t in ("nodes", "edges", "node_id_map", "node_group", "closure")]
    assert not [s for s in cur.log if s.startswith("GRANT")]


def test_serve_time_reader_setup_grants_each_optional_view_present(monkeypatch):
    cur = _Cursor({"k.node_score_view"})
    _cursor_conn(monkeypatch, cur)
    prov._setup_reader_role({"graph_name": "g", "kit_schema": "k", "version": 1})
    grants = [s for s in cur.log if s.startswith("GRANT SELECT")]
    assert grants == [f"GRANT SELECT ON k.{v} TO reader"
                      for v in ("nodes_view", "edges_view", "node_id_map_view", "node_score_view")]


class _IndexConn:
    def __init__(self, tables):
        self.tables, self.log = tables, []

    def transaction(self):
        import contextlib

        return contextlib.nullcontext()

    def execute(self, sql, params=None):
        self.log.append(sql)
        rows = [(t,) for t in self.tables] if "FROM pg_class" in sql else []
        return type("R", (), {"fetchall": lambda self: rows})()


def test_reapply_indexes_only_the_tables_the_version_holds():
    from lgk.steps.reapply import ensure_indexes

    # hostile: closure is absent (greyed or not served); node_group and similar_to are present
    conn = _IndexConn(["nodes", "edges", "node_id_map", "node_group", "similar_to"])
    ensure_indexes(conn, "k_v2")
    made = [s for s in conn.log if s.startswith("CREATE INDEX")]
    assert made == [
        "CREATE INDEX IF NOT EXISTS lgk_edges_src_node_id_idx ON k_v2.edges (src_node_id)",
        "CREATE INDEX IF NOT EXISTS lgk_edges_dst_node_id_idx ON k_v2.edges (dst_node_id)",
        # node_group (node_id) and similar_to (src_id) are the PK's leading columns
        "CREATE INDEX IF NOT EXISTS lgk_node_group_group_id_idx ON k_v2.node_group (group_id)",
    ]


def test_reapply_indexes_closure_both_ways_when_present():
    from lgk.steps.reapply import ensure_indexes

    conn = _IndexConn(["edges", "closure"])
    ensure_indexes(conn, "k_v1")
    # closure_pkey (ancestor_id, descendant_id) serves the ancestor_id way
    assert [s.split(" ON ")[1] for s in conn.log if s.startswith("CREATE INDEX") and "closure" in s] == [
        "k_v1.closure (descendant_id)"]


def test_serve_runs_the_reader_grant_block_inside_the_switch(monkeypatch, tmp_path):
    """serve hands the reader role to the one switch transaction (X3 (4)); every view the switch
    may make, optional ones included, is recorded for teardown."""
    got = {}
    monkeypatch.setattr("lgk.manifest.claim_pg_schema", lambda *a, **k: False)
    monkeypatch.setattr(prov, "_ensure_function_roles", lambda config, cursor: None)
    monkeypatch.setattr(prov.sf, "staged_signatures", lambda conn, served: [])
    monkeypatch.setattr(prov.sf, "switch_statements", lambda *a, **k: [])
    monkeypatch.setattr(prov.sf, "served_functions", lambda config, target: [])  # the gate has its own test
    monkeypatch.setattr("lgk.versions.switch_views", lambda conn, s, t, fsql, **k: got.update(k) or [])
    conn = type("C", (), {"cursor": lambda self: None})()
    cfg = {"graph_name": "g", "kit_schema": "k", "served_schema": "srv", "run_folder": str(tmp_path),
           "lakebase_project": "p", "branch_id": "t-x"}
    prov.serve(conn, cfg, "k_v2")
    assert got["reader"] == "reader"
    views = [e.object_id for e in mf.load_run_manifest(cfg).entries if e.object_type == mf.ObjectType.VIEW]
    assert views == [f"srv.{v}" for v in ("nodes_view", "edges_view", "node_id_map_view", "node_score_view",
                                          "node_group_view", "similar_to_view", "closure_view")]


# ------------------------------------------------- final review S4: `cleared` is required
def test_refuse_unbound_build_needs_cleared(tmp_path, monkeypatch, no_ws):
    """No default that disables the fold check: omitting `cleared` is a TypeError, before SQL."""
    cfg, _stamps = _bound(tmp_path, [])
    monkeypatch.setattr(prov, "execute_sql", lambda *a, **k: pytest.fail("SQL ran"))
    with pytest.raises(TypeError, match="cleared"):
        prov._refuse_unbound_build(cfg, str(tmp_path), "S")  # type: ignore[call-arg]


@pytest.mark.parametrize("cleared", [None, "flags", []])
def test_refuse_unbound_build_refuses_a_non_dict_cleared(tmp_path, monkeypatch, no_ws, cleared):
    # hostile: an explicit None would have meant "no fold type" under the old `cleared or {}`
    cfg, _stamps = _bound(tmp_path, [])
    monkeypatch.setattr(prov, "execute_sql", lambda *a, **k: pytest.fail("SQL ran"))
    with pytest.raises(TypeError, match="cleared gate data"):
        prov._refuse_unbound_build(cfg, str(tmp_path), "S", cleared)
