"""Offline: key checks in profile and build (UC keys are informational, so the kit checks the data).

1. Duplicate primary keys fail, in profile (declared PK) and in build (each node type's natural
   key, re-checked because the nightly job rebuilds without profile). Build no longer keeps a
   silent winner per duplicate key.
2. Orphans (FK values with no matching row) are the operator's decision, recorded in
   key_decisions.json (never flags.json: a data-quality decision must not re-lock privacy). Build
   fails on any edge end the CURRENT data shows with orphans that no matching `accept` covers.
3. Messages and files carry counts only, never a key value.

The reference counts come from tests/test_keys_support.RowWarehouse (Python over plain rows).
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml

from tests.test_build_sql import _col, pass_privacy, run_build
from tests.test_keys_support import RowWarehouse

# hostile: a recognisable key value; it must never reach an error message or a written file.
LEAK = "ZZLEAK4471"

SCHEMA_TABLES = [
    {"name": "providers", "key_columns": ["provider_id"], "foreign_keys": [],
     "columns": [_col("provider_id"), _col("provider_name"), _col("specialty")]},
    {"name": "referrals", "key_columns": ["referral_id"],
     "foreign_keys": [{"columns": ["from_provider_id"], "references": ["providers", "provider_id"]},
                      {"columns": ["to_provider_id"], "references": ["providers", "provider_id"]}],
     "columns": [_col("referral_id"), _col("from_provider_id"), _col("to_provider_id")]},
    {"name": "sites", "key_columns": ["org_id", "site_no"], "foreign_keys": [],
     "columns": [_col("org_id"), _col("site_no"), _col("site_name")]},
]

MAPPING: Dict[str, Any] = {
    "source_catalog": "main", "source_schema": "gold",
    "nodes": {
        "provider": {"source_table": "providers", "natural_key": "provider_id", "label": "provider_name",
                     "properties": ["specialty"]},
        "site": {"source_table": "sites", "natural_key": ["org_id", "site_no"], "label": "site_name",
                 "properties": []},
    },
    "edges": {
        "refers": {"source_table": "referrals", "src": "from_provider_id", "src_type": "provider",
                   "dst": "to_provider_id", "dst_type": "provider"},
    },
}


def _provider(pid: Any, name: str = "Dr X") -> Dict[str, Any]:
    return {"provider_id": pid, "provider_name": name, "specialty": "Cardiology"}


def _referral(rid: str, src: Any, dst: Any) -> Dict[str, Any]:
    return {"referral_id": rid, "from_provider_id": src, "to_provider_id": dst}


def tables(n_providers: int = 5) -> Dict[str, Dict[str, Any]]:
    """Clean data: unique keys, every FK value matches a provider."""
    providers = [_provider(f"P{i}") for i in range(n_providers)]
    referrals = [_referral(f"R{i}", f"P{i}", f"P{(i + 1) % n_providers}") for i in range(n_providers)]
    # hostile: a composite key where EACH column alone repeats but the pair is unique (must pass)
    sites = [{"org_id": "O1", "site_no": "1", "site_name": "a"}, {"org_id": "O1", "site_no": "2", "site_name": "b"},
             {"org_id": "O2", "site_no": "1", "site_name": "c"}]
    return {
        "providers": {"columns": [("provider_id", "STRING"), ("provider_name", "STRING"), ("specialty", "STRING")],
                      "rows": providers, "pk": ["provider_id"], "fks": []},
        "referrals": {"columns": [("referral_id", "STRING"), ("from_provider_id", "STRING"),
                                  ("to_provider_id", "STRING")],
                      "rows": referrals, "pk": ["referral_id"],
                      "fks": [("fk_from", ["from_provider_id"], "providers", ["provider_id"]),
                              ("fk_to", ["to_provider_id"], "providers", ["provider_id"])]},
        "sites": {"columns": [("org_id", "STRING"), ("site_no", "STRING"), ("site_name", "STRING")],
                  "rows": sites, "pk": ["org_id", "site_no"], "fks": []},
    }


def with_orphans(t: Dict[str, Any], n: int = 2) -> Dict[str, Any]:
    """hostile: referrals.to_provider_id has `n` orphans (a referral to a provider outside the
    directory); referrals.from_provider_id, the second FK on the same table, has none."""
    t["referrals"]["rows"] += [_referral(f"RX{i}", "P0", f"{LEAK}{i}") for i in range(n)]
    return t


def run_folder(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "schema.json").write_text(json.dumps(
        {"source_catalog": "main", "source_schema": "gold", "tables": SCHEMA_TABLES}))
    (tmp_path / "graph_mapping.yaml").write_text(yaml.safe_dump(MAPPING, sort_keys=False))
    pass_privacy(tmp_path, folds=())
    return tmp_path


def build_with(tmp_path: Path, monkeypatch, t: Dict[str, Any], **kw: Any) -> RowWarehouse:
    wh = RowWarehouse(t, **kw)
    run_build(tmp_path, monkeypatch, wh)
    return wh


def run_text(folder: Path) -> str:
    return "\n".join(p.read_text(errors="replace") for p in folder.rglob("*") if p.is_file())


def _nodes_replaced(wh: RowWarehouse) -> bool:
    return any("CREATE OR REPLACE TABLE main.lgk_care.nodes" in s for s in wh.statements)


# =================================================================== build: duplicate keys
def test_clean_data_builds_with_no_silent_dedup_and_a_pair_unique_composite_key(tmp_path, monkeypatch):
    rf = run_folder(tmp_path)
    wh = build_with(rf, monkeypatch, tables())
    inserts = [s for s in wh.statements if "INSERT INTO main.lgk_care.nodes" in s]
    assert len(inserts) == 2
    assert not any("ROW_NUMBER" in s or "rn = 1" in s for s in inserts)
    assert sorted(wh.nodes) == sorted([f"provider::P{i}" for i in range(5)]
                                      + ['site::["O1","1"]', 'site::["O1","2"]', 'site::["O2","1"]'])
    assert (rf / "privacy_bindings.json").exists()  # the build finished and bound its stamp


def test_a_duplicate_natural_key_fails_build_before_nodes_is_replaced(tmp_path, monkeypatch):
    from lgk.keys import DuplicateKeyError

    rf = run_folder(tmp_path)
    t = tables()
    # hostile: ONE duplicate at the key grain (P3 twice, a recognisable value); every other row is clean
    t["providers"]["rows"][3]["provider_id"] = LEAK
    t["providers"]["rows"].append(_provider(LEAK, "Dr Y"))
    with pytest.raises(DuplicateKeyError) as exc:
        build_with(rf, monkeypatch, t)
    msg = str(exc.value)
    assert "provider" in msg and "providers" in msg and "1 duplicate key row" in msg
    assert LEAK not in msg and LEAK not in run_text(rf)


def test_a_duplicate_fails_before_any_kit_table_is_replaced(tmp_path, monkeypatch):
    from lgk.keys import DuplicateKeyError

    rf = run_folder(tmp_path)
    t = tables()
    t["providers"]["rows"].append(_provider("P2", "Dr Dup"))
    wh = RowWarehouse(t)
    with pytest.raises(DuplicateKeyError):
        run_build(rf, monkeypatch, wh)
    assert not _nodes_replaced(wh)
    assert not any("CREATE OR REPLACE TABLE" in s for s in wh.statements)
    assert not (rf / "privacy_bindings.json").exists()


def test_a_composite_key_whose_pair_repeats_fails(tmp_path, monkeypatch):
    from lgk.keys import DuplicateKeyError

    rf = run_folder(tmp_path)
    t = tables()
    # hostile: the pair (O1, 2) repeats
    t["sites"]["rows"].append({"org_id": "O1", "site_no": "2", "site_name": "dup"})
    with pytest.raises(DuplicateKeyError, match=r"site.*sites.*1 duplicate key row"):
        build_with(rf, monkeypatch, t)


def test_an_int_and_a_string_rendering_the_same_are_one_key(tmp_path, monkeypatch):
    """hostile: a differently typed key cell — int 7 and string '7' get the same node id."""
    from lgk.keys import DuplicateKeyError

    rf = run_folder(tmp_path)
    t = tables()
    t["providers"]["rows"] += [_provider(7), _provider("7")]
    with pytest.raises(DuplicateKeyError, match="1 duplicate key row"):
        build_with(rf, monkeypatch, t)


def test_scale_and_noise_the_exact_duplicate_count_is_reported(tmp_path, monkeypatch):
    """hostile: scale — 3000 providers, 4 extra rows on 3 keys scattered, reordered input."""
    from lgk.keys import DuplicateKeyError

    rf = run_folder(tmp_path)
    t = tables(3000)
    t["providers"]["rows"] += [_provider("P10"), _provider("P2999"), _provider("P2999"), _provider("P1500")]
    t["providers"]["rows"].reverse()
    with pytest.raises(DuplicateKeyError, match=r"\b4 duplicate key rows"):
        build_with(rf, monkeypatch, t)


def test_a_null_key_row_is_dropped_not_a_duplicate_and_a_null_fk_is_not_an_orphan(tmp_path, monkeypatch):
    rf = run_folder(tmp_path)
    t = tables()
    t["providers"]["rows"] += [_provider(None, "No key"), _provider(None, "No key 2")]
    t["referrals"]["rows"].append(_referral("RN", None, "P1"))
    wh = build_with(rf, monkeypatch, t)
    assert "provider::None" not in wh.nodes and len([n for n in wh.nodes if n.startswith("provider::")]) == 5


def test_a_duplicate_written_after_the_check_is_caught_by_the_read_back(tmp_path, monkeypatch):
    """hostile: a writer races the build — the duplicate lands after the pre-check."""
    from lgk.keys import DuplicateKeyError

    rf = run_folder(tmp_path)
    with pytest.raises(DuplicateKeyError, match=r"nodes.*1 duplicate node id"):
        build_with(rf, monkeypatch, tables(), after_check={"providers": [_provider("P1", "late")]})
    assert not (rf / "privacy_bindings.json").exists()


# =================================================================== build: orphans
def test_orphans_with_no_decision_fail_build_naming_only_that_fk(tmp_path, monkeypatch):
    from lgk.keys import OrphanKeyError

    rf = run_folder(tmp_path)
    wh = RowWarehouse(with_orphans(tables()))
    with pytest.raises(OrphanKeyError) as exc:
        run_build(rf, monkeypatch, wh)
    msg = str(exc.value)
    assert "referrals.to_provider_id" in msg and "2 rows" in msg
    assert "from_provider_id" not in msg
    assert not any("CREATE OR REPLACE TABLE main.lgk_care.edges" in s for s in wh.statements)
    assert LEAK not in msg and LEAK not in run_text(rf)
    from lgk.keys import open_orphans

    assert [(o["table"], o["columns"], o["orphan_rows"]) for o in open_orphans(str(rf))] == [
        ("referrals", ["to_provider_id"], 2)]


def test_an_accepted_decision_builds_and_records_the_dropped_count(tmp_path, monkeypatch):
    from lgk.keys import OrphanKeyError, open_orphans, read_key_decisions, resolve_orphans

    rf = run_folder(tmp_path)
    with pytest.raises(OrphanKeyError):
        build_with(rf, monkeypatch, with_orphans(tables()))
    d = resolve_orphans(str(rf), "referrals", "to_provider_id", "accept", chooser="op",
                        reason="referrals to providers outside the directory")
    assert (d["choice"], d["chooser"], d["orphan_rows"]) == ("accept", "op", 2)
    assert open_orphans(str(rf)) == []
    build_with(rf, monkeypatch, with_orphans(tables()))
    dropped = read_key_decisions(str(rf))["dropped"]
    assert dropped == [{"edge_type": "refers", "table": "referrals", "column": "to_provider_id",
                        "node_type": "provider", "dropped_rows": 2}]
    assert LEAK not in run_text(rf)
    assert (rf / "privacy_bindings.json").exists()


def test_an_accepted_count_that_grows_fails(tmp_path, monkeypatch):
    from lgk.keys import OrphanKeyError, resolve_orphans

    rf = run_folder(tmp_path)
    with pytest.raises(OrphanKeyError):
        build_with(rf, monkeypatch, with_orphans(tables(), 2))
    resolve_orphans(str(rf), "referrals", "to_provider_id", "accept", chooser="op")
    # hostile: the accepted 2 became 3 overnight
    with pytest.raises(OrphanKeyError, match=r"referrals\.to_provider_id.*3 rows.*accepted 2"):
        build_with(rf, monkeypatch, with_orphans(tables(), 3))
    # a smaller count stays covered
    build_with(rf, monkeypatch, with_orphans(tables(), 1))


def test_reordered_input_gives_the_same_orphan_count(tmp_path, monkeypatch):
    from lgk.keys import OrphanKeyError

    rf = run_folder(tmp_path)
    t = with_orphans(tables(), 2)
    t["referrals"]["rows"].reverse()
    t["providers"]["rows"].reverse()
    with pytest.raises(OrphanKeyError, match="2 rows"):
        build_with(rf, monkeypatch, t)


def test_a_typed_fk_matches_by_its_rendering(tmp_path, monkeypatch):
    """hostile: a differently typed FK cell — int 3 matches provider '3'; int 3 does not match '03'."""
    from lgk.keys import OrphanKeyError

    rf = run_folder(tmp_path)
    t = tables()
    t["providers"]["rows"] += [_provider("3"), _provider("03")]
    t["referrals"]["rows"].append(_referral("RT", 3, "P1"))
    build_with(rf, monkeypatch, t)  # 3 renders '3': a node, no orphan
    t["providers"]["rows"] = [r for r in t["providers"]["rows"] if r["provider_id"] != "3"]
    with pytest.raises(OrphanKeyError, match=r"referrals\.from_provider_id.*1 row"):
        build_with(rf, monkeypatch, t)


def test_a_fixed_decision_with_orphans_left_fails(tmp_path, monkeypatch):
    from lgk.keys import OrphanKeyError, resolve_orphans

    rf = run_folder(tmp_path)
    with pytest.raises(OrphanKeyError):
        build_with(rf, monkeypatch, with_orphans(tables()))
    resolve_orphans(str(rf), "referrals", "to_provider_id", "fixed", chooser="op")
    with pytest.raises(OrphanKeyError, match="fixed"):
        build_with(rf, monkeypatch, with_orphans(tables()))
    build_with(rf, monkeypatch, tables())  # the data really fixed: builds


def _accepted(tmp_path: Path, monkeypatch) -> Path:
    from lgk.keys import OrphanKeyError, resolve_orphans

    rf = run_folder(tmp_path)
    with pytest.raises(OrphanKeyError):
        build_with(rf, monkeypatch, with_orphans(tables(), 2))
    resolve_orphans(str(rf), "referrals", "to_provider_id", "accept", chooser="op")
    return rf


@pytest.mark.parametrize("forged", [99, "2", True, -1, None])
def test_a_hand_edited_decision_count_is_refused(tmp_path, monkeypatch, forged):
    from lgk.keys import KEY_DECISIONS_FILE, OrphanKeyError

    rf = _accepted(tmp_path, monkeypatch)
    path = rf / KEY_DECISIONS_FILE
    data = json.loads(path.read_text())
    data["decisions"][-1]["orphan_rows"] = forged
    path.write_text(json.dumps(data))
    with pytest.raises(OrphanKeyError):
        build_with(rf, monkeypatch, with_orphans(tables(), 3 if forged == 99 else 2))


def test_a_deleted_decisions_file_fails_and_a_malformed_one_fails_loudly(tmp_path, monkeypatch):
    from lgk.keys import KEY_DECISIONS_FILE, KeyDecisionsFormatError, OrphanKeyError

    rf = _accepted(tmp_path, monkeypatch)
    (rf / KEY_DECISIONS_FILE).write_text("{not json")
    with pytest.raises(KeyDecisionsFormatError):
        build_with(rf, monkeypatch, with_orphans(tables(), 2))
    (rf / KEY_DECISIONS_FILE).unlink()
    with pytest.raises(OrphanKeyError):
        build_with(rf, monkeypatch, with_orphans(tables(), 2))


def test_a_decision_for_another_fk_does_not_cover(tmp_path, monkeypatch):
    from lgk.keys import KEY_DECISIONS_FILE, OrphanKeyError

    rf = _accepted(tmp_path, monkeypatch)
    path = rf / KEY_DECISIONS_FILE
    data = json.loads(path.read_text())
    data["decisions"][-1]["columns"] = ["from_provider_id"]
    path.write_text(json.dumps(data))
    with pytest.raises(OrphanKeyError):
        build_with(rf, monkeypatch, with_orphans(tables(), 2))


# =================================================================== resolve / open helpers
def test_resolve_refuses_bad_input(tmp_path, monkeypatch):
    from lgk.keys import OrphanKeyError, resolve_orphans

    rf = run_folder(tmp_path)
    with pytest.raises(OrphanKeyError):
        build_with(rf, monkeypatch, with_orphans(tables()))
    with pytest.raises(ValueError, match="choice"):
        resolve_orphans(str(rf), "referrals", "to_provider_id", "ignore", chooser="op")
    with pytest.raises(ValueError, match="chooser"):
        resolve_orphans(str(rf), "referrals", "to_provider_id", "accept", chooser="  ")
    with pytest.raises(KeyError, match="from_provider_id"):  # no orphans observed there
        resolve_orphans(str(rf), "referrals", "from_provider_id", "accept", chooser="op")


def test_a_key_decision_does_not_touch_the_privacy_files(tmp_path, monkeypatch):
    from lgk.keys import OrphanKeyError, resolve_orphans

    rf = run_folder(tmp_path)
    with pytest.raises(OrphanKeyError):
        build_with(rf, monkeypatch, with_orphans(tables()))
    before = {n: (rf / n).read_bytes() for n in ("flags.json", "schema.json", "graph_mapping.yaml")}
    resolve_orphans(str(rf), "referrals", "to_provider_id", "accept", chooser="op")
    assert {n: (rf / n).read_bytes() for n in before} == before


# =================================================================== profile (UC mode)
def _profile(tmp_path: Path, monkeypatch, t: Dict[str, Any]) -> Dict[str, Any]:
    from lgk.steps import profile as p

    wh = RowWarehouse(t)
    monkeypatch.setattr(p, "execute_sql", wh)
    tmp_path.mkdir(parents=True, exist_ok=True)
    p.profile({"run_folder": str(tmp_path), "warehouse_id": "wh", "workspace_client": object(),
               "source_catalog": "main", "source_schema": "gold"})
    return json.loads((tmp_path / "schema.json").read_text())


def _by_name(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {t["name"]: t for t in schema["tables"]}


def test_profile_records_key_counts_and_orphans_per_fk(tmp_path, monkeypatch):
    t = with_orphans(tables(), 2)
    t["providers"]["rows"].append(_provider(None))
    s = _by_name(_profile(tmp_path, monkeypatch, t))
    assert (s["providers"]["duplicate_key_rows"], s["providers"]["null_key_rows"]) == (0, 1)
    assert s["sites"]["duplicate_key_rows"] == 0  # pair-unique composite passes
    fks = {tuple(fk["columns"]): fk["orphan_rows"] for fk in s["referrals"]["foreign_keys"]}
    assert fks == {("from_provider_id",): 0, ("to_provider_id",): 2}
    from lgk.keys import open_orphans

    assert [(o["table"], o["columns"], o["orphan_rows"]) for o in open_orphans(str(tmp_path))] == [
        ("referrals", ["to_provider_id"], 2)]
    assert LEAK not in run_text(tmp_path)


def test_profile_fails_on_a_duplicate_key_after_writing_the_counts(tmp_path, monkeypatch):
    from lgk.keys import DuplicateKeyError

    t = tables()
    t["providers"]["rows"] += [_provider(LEAK), _provider(LEAK), _provider(LEAK)]
    t["providers"]["rows"].reverse()  # hostile: reordered input
    with pytest.raises(DuplicateKeyError) as exc:
        _profile(tmp_path, monkeypatch, t)
    msg = str(exc.value)
    assert "providers" in msg and "2 duplicate key rows" in msg and "sites" not in msg
    assert LEAK not in msg and LEAK not in run_text(tmp_path)
    s = _by_name(json.loads((tmp_path / "schema.json").read_text()))
    assert s["providers"]["duplicate_key_rows"] == 2 and s["referrals"]["duplicate_key_rows"] == 0


def test_profile_fails_on_a_repeated_composite_pair(tmp_path, monkeypatch):
    from lgk.keys import DuplicateKeyError

    t = tables()
    t["sites"]["rows"].append({"org_id": "O2", "site_no": "1", "site_name": "again"})
    with pytest.raises(DuplicateKeyError, match=r"sites.*1 duplicate key row"):
        _profile(tmp_path, monkeypatch, t)


def test_profile_counts_a_typed_orphan_by_rendering(tmp_path, monkeypatch):
    """hostile: a differently typed FK cell (int 3 vs provider '3' matches; '03' does not)."""
    t = tables()
    t["providers"]["rows"].append(_provider("03"))
    t["referrals"]["rows"] += [_referral("RT", 3, "P1"), _referral("RU", 4, "P1")]
    t["providers"]["rows"].append(_provider("4"))
    s = _by_name(_profile(tmp_path, monkeypatch, t))
    fks = {tuple(fk["columns"]): fk["orphan_rows"] for fk in s["referrals"]["foreign_keys"]}
    assert fks[("from_provider_id",)] == 1


def test_a_reprofile_with_more_orphans_reopens_an_accepted_fk(tmp_path, monkeypatch):
    from lgk.keys import open_orphans, resolve_orphans

    _profile(tmp_path, monkeypatch, with_orphans(tables(), 2))
    resolve_orphans(str(tmp_path), "referrals", "to_provider_id", "accept", chooser="op")
    assert open_orphans(str(tmp_path)) == []
    _profile(tmp_path, monkeypatch, with_orphans(tables(), 4))
    (o,) = open_orphans(str(tmp_path))
    assert (o["orphan_rows"], o["accepted_rows"]) == (4, 2)


# =================================================================== nightly copy
def _nightly_folder(tmp_path: Path, orphans: int) -> Path:
    from lgk.manifest import MANIFEST_FILE
    from lgk.privacy.decisions import DECISIONS_FILE

    for name in ("flags.json", "graph_mapping.yaml", DECISIONS_FILE, MANIFEST_FILE):
        (tmp_path / name).write_text("{}")
    tbls = copy.deepcopy(SCHEMA_TABLES)
    for fk in tbls[1]["foreign_keys"]:
        fk["orphan_rows"] = 0
    tbls[1]["foreign_keys"][1]["orphan_rows"] = orphans
    (tmp_path / "schema.json").write_text(json.dumps({"tables": tbls}))
    return tmp_path


def test_nightly_requires_key_decisions_when_profile_saw_orphans(tmp_path):
    from lgk.keys import KEY_DECISIONS_FILE
    from lgk.refresh import nightly_folder as nf

    folder = _nightly_folder(tmp_path, 2)
    with pytest.raises(nf.NightlyFolderError, match=KEY_DECISIONS_FILE):
        nf.nightly_files(folder)
    (folder / KEY_DECISIONS_FILE).write_text("{}")
    assert KEY_DECISIONS_FILE in nf.nightly_files(folder)


def test_nightly_copies_key_decisions_when_present_and_does_not_need_it_without_orphans(tmp_path):
    from lgk.keys import KEY_DECISIONS_FILE
    from lgk.refresh import nightly_folder as nf

    folder = _nightly_folder(tmp_path, 0)
    assert KEY_DECISIONS_FILE not in nf.nightly_files(folder)
    (folder / KEY_DECISIONS_FILE).write_text("{}")
    assert KEY_DECISIONS_FILE in nf.nightly_files(folder)


def test_nightly_upload_removes_a_stale_key_decisions_copy(tmp_path):
    from lgk.keys import KEY_DECISIONS_FILE
    from lgk.refresh import nightly_folder as nf

    deleted: List[str] = []

    class _WS:
        def mkdirs(self, p): pass
        def upload(self, p, data, format=None, overwrite=None): pass
        def delete(self, p): deleted.append(p)

    class _W:
        workspace = _WS()

    folder = _nightly_folder(tmp_path, 0)
    nf.upload_run_files(_W(), folder, nf.nightly_files(folder), "/Workspace/x/run")
    assert any(p.endswith(f"/{KEY_DECISIONS_FILE}") for p in deleted)
