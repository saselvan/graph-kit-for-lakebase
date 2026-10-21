"""Seam-1: the FULL MONDO release through the kit with no code change.

The loader writes gold tables into a per-run `lgk_v1_<6hex>` schema (via a UC Volume and
read_files), then the laptop runner runs the shipped steps on them: profile -> propose ->
privacy -> build -> provision. A tree walk on the served views is compared with references
built ONLY from sources the kit does not touch:
- the raw mondo.json, parsed here with its own few lines (not the loader), and
- the Delta gold table mondo_is_a through the Delta node_id_map (Spark joins).

Hostile real-world noise in the full file: ~4k obsolete terms, ~35k is_a links to other
ontologies, polyhierarchy (terms with several parents: diamonds), labels with commas,
quotes and non-ASCII characters, and 32k nodes in one node type.
"""

import json
import re
import secrets
import tempfile
from pathlib import Path
from typing import Any, Dict, Set

import pytest

from lgk.app.queries import attribution_query
from lgk.enrich.shape import measure_shape
from lgk.harness import BranchLeasor
from lgk.loaders.mondo import download, load_mondo
from lgk.runner import LaptopRunner, RunStatus
from lgk.steps.build import build
from lgk.steps.privacy import privacy
from lgk.steps.profile import profile
from lgk.steps.propose import propose
from lgk.steps.advise import advise
from lgk.steps.preflight import preflight
from lgk.steps.provision import provision
from lgk.steps.switch import switch
from lgk.uc_sql import execute_sql
from lgk.walks.sql import tree_sql
from tests.test_build_provision_walks import WAREHOUSE_ID, _pg, _teardown_live

pytestmark = [pytest.mark.seam1, pytest.mark.provision]

CACHE = Path(tempfile.gettempdir()) / "lgk18" / "mondo.json"
SEED = "MONDO:0005015"  # diabetes mellitus: a well-known term with many subtypes
OBO = "http://purl.obolibrary.org/obo/"


def _raw_reference(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Independent of the loader: kept ids, the release date, and the seed's children."""
    g = doc["graphs"][0]
    live = {n["id"] for n in g["nodes"] if n["id"].startswith(f"{OBO}MONDO_") and n.get("type") == "CLASS"
            and n.get("lbl") and not (n.get("meta") or {}).get("deprecated")}
    obsolete = {n["id"] for n in g["nodes"] if n["id"].startswith(f"{OBO}MONDO_")
                and (n.get("meta") or {}).get("deprecated")}
    seed_iri = OBO + SEED.replace(":", "_")
    kids = {e["sub"] for e in g["edges"] if e["pred"] == "is_a" and e["obj"] == seed_iri and e["sub"] in live}
    date = re.search(r"releases/(\d{4}-\d{2}-\d{2})/", g["meta"]["version"]).group(1)
    to_curie = lambda iri: iri[len(OBO):].replace("_", ":")  # noqa: E731
    return {"n_live": len(live), "release": f"v{date}", "children": {to_curie(k) for k in kids},
            "an_obsolete": to_curie(sorted(obsolete)[0])}


@pytest.fixture(scope="module")
def mondo_run(workspace_client, lakebase_project, lakebase_catalog):
    w, suffix = workspace_client, secrets.token_hex(3)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    doc = download(dest=CACHE)
    leasor = BranchLeasor(workspace_client=w, project_id=lakebase_project, test_base_branch="test-base",
                          run_folder=Path(tempfile.mkdtemp()), ttl_seconds=14400)
    config: Dict[str, Any] = {
        "graph_name": "mondo", "lakebase_project": lakebase_project, "lakebase_catalog": lakebase_catalog,
        "kit_schema": f"lgk_v1_{suffix}_kit", "served_schema": f"lgk_served_{suffix}",
        "fixture_schema": f"lgk_v1_{suffix}", "source_catalog": lakebase_catalog,
        "source_schema": f"lgk_v1_{suffix}", "warehouse_id": WAREHOUSE_ID, "workspace_client": w,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            config["run_folder"] = str(Path(tmpdir) / f"mondo_{suffix}")
            config["branch_id"] = leasor.create_test_branch(f"mondo-{suffix}")["branch_id"]
            rel = load_mondo(w, WAREHOUSE_ID, lakebase_catalog, config["source_schema"], config["run_folder"],
                             doc=doc)
            config["sync_owner_user"] = w.current_user.me().user_name
            results = LaptopRunner(config).run_steps([
                ("preflight", preflight), ("profile", profile), ("propose", propose), ("privacy", privacy),
                ("build", build), ("provision", provision), ("advise", advise), ("switch", switch)])
            yield config, rel, results, _raw_reference(doc)
        finally:
            _teardown_live(w, leasor, config)


def _q(w: Any, sql: str):
    return execute_sql(w, WAREHOUSE_ID, sql)


def test_every_step_succeeds_with_no_code_change(mondo_run):
    _, _, results, _ = mondo_run
    assert [(r.step, r.status) for r in results] == [
        (s, RunStatus.SUCCESS)
        for s in ("preflight", "profile", "propose", "privacy", "build", "provision", "advise", "switch")
    ], [r.to_dict() for r in results]


def test_advisor_ran_on_the_synced_version_with_defaults(mondo_run):
    """#21: the shipped flow's advisor, all defaults, on REAL synced (partitioned) tables.
    Every proposal is on the version's parent table and is either applied or reported."""
    from lgk import versions

    config, _, _, _ = mondo_run
    report = json.loads((Path(config["run_folder"]) / "advisor_report.json").read_text())
    print("ADVISOR_PROPOSALS", json.dumps(report["proposals"]))
    v = versions.version_schema(config)
    assert report["version_schema"] == v
    for p in report["proposals"]:
        assert p["schema"] == v and (p["applied"] or p["not_applied"]), p


def test_privacy_gate_passes_legitimately_with_no_care_flags(mondo_run):
    config, _, _, _ = mondo_run
    flags = json.loads((Path(config["run_folder"]) / "flags.json").read_text())
    assert flags["flags"] == [] and "gate" in flags
    assert {c["table"] for c in flags["columns"]} >= {"mondo_terms", "mondo_is_a"}


def test_graph_is_mondo_terms_only_and_matches_the_raw_file(mondo_run, workspace_client):
    config, rel, _, raw = mondo_run
    w, cat, kit = workspace_client, config["lakebase_catalog"], config["kit_schema"]
    types = _q(w, f"SELECT node_type, COUNT(*) FROM {cat}.{kit}.nodes GROUP BY node_type")
    assert [(t, int(n)) for t, n in types] == [("mondo_terms", raw["n_live"])]
    assert raw["n_live"] > 30_000  # the full release, not a subset
    n_edges = int(_q(w, f"SELECT COUNT(*) FROM {cat}.{kit}.edges")[0][0])
    assert n_edges == len(rel.is_a) == int(_q(w, f"SELECT COUNT(*) FROM {cat}.{config['source_schema']}.mondo_is_a")[0][0])
    assert n_edges > 40_000
    # hostile: an obsolete term is not a node
    assert _q(w, f"SELECT COUNT(*) FROM {cat}.{kit}.node_id_map WHERE natural_key = '{raw['an_obsolete']}'")[0][0] == "0"


def test_built_graph_is_tree_shaped(mondo_run, workspace_client):
    """The kit's own data-level shape check: MONDO's is_a hierarchy is acyclic."""
    config, _, _, _ = mondo_run
    fq = f"{config['lakebase_catalog']}.{config['kit_schema']}"
    shape = measure_shape(lambda s: _q(workspace_client, s), fq)
    assert shape.is_dag and not shape.too_deep, shape


def _ids(w: Any, config: Dict[str, Any], keys: Set[str]) -> Set[str]:
    cat, kit = config["lakebase_catalog"], config["kit_schema"]
    inlist = ", ".join(f"'{k}'" for k in sorted(keys))
    return {str(r[0]) for r in _q(w, f"SELECT node_id FROM {cat}.{kit}.node_id_map WHERE natural_key IN ({inlist})")}


def test_tree_walk_depth1_equals_raw_and_spark_references(mondo_run, workspace_client):
    config, _, _, raw = mondo_run
    w, cat, src, kit = workspace_client, config["lakebase_catalog"], config["source_schema"], config["kit_schema"]
    assert len(raw["children"]) >= 5  # non-vacuous
    want_raw = _ids(w, config, raw["children"])
    want_spark = {str(r[0]) for r in _q(w,
        f"SELECT m.node_id FROM {cat}.{src}.mondo_is_a e JOIN {cat}.{kit}.node_id_map m "
        f"ON m.natural_key = e.child_id WHERE e.parent_id = '{SEED}'")}
    assert want_raw == want_spark and len(want_raw) == len(raw["children"])
    (seed_id,) = _ids(w, config, {SEED})
    with _pg(config) as conn:
        rows = conn.execute(tree_sql(depth=1, served_schema=config["served_schema"], top_n=10_000),
                            {"start_node_id": seed_id}).fetchall()
    assert {str(r[0]) for r in rows} == want_raw
    assert {int(r[1]) for r in rows} == {1}


def test_tree_walk_depth3_equals_spark_bfs(mondo_run, workspace_client):
    """# hostile: polyhierarchy — a term under two reached parents must appear once, at its min depth."""
    config, _, _, _ = mondo_run
    w, cat, src, kit = workspace_client, config["lakebase_catalog"], config["source_schema"], config["kit_schema"]
    levels, seen, frontier = {}, {SEED}, {SEED}
    for d in (1, 2, 3):
        inlist = ", ".join(f"'{k}'" for k in sorted(frontier))
        frontier = {str(r[0]) for r in _q(w, f"SELECT DISTINCT child_id FROM {cat}.{src}.mondo_is_a "
                                             f"WHERE parent_id IN ({inlist})")} - seen
        seen |= frontier
        levels.update({k: d for k in frontier})
    inlist = ", ".join(f"'{k}'" for k in sorted(levels))
    key_by_id = {str(i): str(k) for i, k in _q(w, f"SELECT node_id, natural_key FROM "
                                                  f"{cat}.{kit}.node_id_map WHERE natural_key IN ({inlist})")}
    (seed_id,) = _ids(w, config, {SEED})
    with _pg(config) as conn:
        rows = conn.execute(tree_sql(depth=3, served_schema=config["served_schema"], top_n=10_000),
                            {"start_node_id": seed_id}).fetchall()
    got = {key_by_id.get(str(n), str(n)): int(d) for n, d, _ in rows}
    assert len(got) == len(rows)  # each node once
    assert got == levels and max(levels.values()) == 3


def test_served_attribution_is_the_exact_cc_by_string(mondo_run, workspace_client):
    config, rel, _, raw = mondo_run
    exact = (f"Mondo Disease Ontology (MONDO), release {raw['release']}, © the Monarch Initiative. "
             "Source: https://mondo.monarchinitiative.org/ — used under CC BY 4.0 "
             "(https://creativecommons.org/licenses/by/4.0/). Citation: \"Mondo: integrating disease "
             "terminology across communities,\" Genetics (2025), https://doi.org/10.1093/genetics/iyaf215.")
    served = config["served_schema"]
    with _pg(config) as conn:
        # the notice is the VIEW's comment; the schema's comment stays the ownership marker
        comment = conn.execute("SELECT obj_description(%s::regclass, 'pg_class')",
                               (f"{served}.source_attribution",)).fetchone()[0]
        marker = conn.execute("SELECT obj_description(oid, 'pg_namespace') FROM pg_namespace "
                              "WHERE nspname = %s", (served,)).fetchone()[0]
        assert marker == "lgk:graph=mondo"
        conn.execute("GRANT reader TO CURRENT_USER")
        conn.execute("SET ROLE reader")  # what the website reads
        text, version, lic, cite, source, changes = conn.execute(attribution_query(served)).fetchone()
    assert text == exact
    assert comment == exact
    assert (lic, cite, source) == ("https://creativecommons.org/licenses/by/4.0/",
                                   "https://doi.org/10.1093/genetics/iyaf215", "https://mondo.monarchinitiative.org/")
    assert changes.startswith("Derived from MONDO")
    # The recorded release equals the parsed one, in UC and in the served layer.
    recorded = _q(workspace_client, f"SELECT release_version FROM {config['lakebase_catalog']}."
                                    f"{config['source_schema']}.mondo_release")
    assert version == rel.release_version == raw["release"] == recorded[0][0]


def test_whole_graph_teardown_of_a_mondo_run_leaves_nothing(mondo_run, workspace_client):
    """#22: the recorded attribution view lets teardown drop the served schema. Runs LAST."""
    from lgk import manifest as mf
    from lgk.steps.teardown import teardown
    from lgk.versions import version_schema

    config, _, _, _ = mondo_run
    w, cat = workspace_client, config["lakebase_catalog"]
    entries = mf.load_run_manifest(config).entries
    synced = [e.object_id for e in entries if e.object_type == mf.ObjectType.SYNCED_TABLE]
    assert f"{config['served_schema']}.source_attribution" in {
        e.object_id for e in entries if e.object_type == mf.ObjectType.VIEW}
    assert len(synced) >= 3

    teardown(config, "mondo")

    with _pg(config) as conn:
        for schema in (config["served_schema"], version_schema(config)):
            assert conn.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (schema,)).fetchone() is None, schema
    for full in synced:
        from databricks.sdk.errors import NotFound

        with pytest.raises(NotFound):
            w.postgres.get_synced_table(name=f"synced_tables/{full}")
    for schema in (config["kit_schema"], version_schema(config)):
        assert not _q(w, f"SHOW SCHEMAS IN {cat} LIKE '{schema}'"), schema
