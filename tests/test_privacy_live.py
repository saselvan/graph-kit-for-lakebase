"""Seam-1 privacy gates, on one module-scoped test branch.

The tables are built directly in branch Postgres (no synced tables are needed: the floor and
the reader's privileges are Postgres facts). Fixture:

- `<kit>_fx.visits(patient, provider)`: care-subject visits. Provider 1 shares exactly 10
  (floor-1), 11 (floor) and 12 (floor+1) distinct patients with providers 2, 3 and 4.
  hostile: every patient shared with provider 2 visits provider 1 TWICE, so a COUNT(*) fold
  sees 20 rows behind (1,2) and would serve a pair that only 10 people stand behind.
- `<kit>_v1`: the version tables, with folded edges 1->2/3/4 weighted 10/11/12, and
  hostile: 1->5 with a NULL weight (a folded walk must hide it, never serve it unweighted).
- `<kit>_v2`: hostile: an older version still on the branch, holding an extra raw table.
"""

import json
import logging
import uuid
from typing import Any, Dict

import psycopg
import pytest

from lgk import versions
from lgk.privacy import floor, probe
from lgk.steps import provision
from lgk.steps.walks import get_walk_query
from tests.test_build_provision_walks import WAREHOUSE_ID, _teardown_live
from tests.test_versions_live import _connect_with_retry

logger = logging.getLogger(__name__)

SHARED = {2: 10, 3: 11, 4: 12}  # provider -> distinct patients shared with provider 1
EDGES = [(12, 1, 2, 10), (13, 1, 3, 11), (14, 1, 4, 12), (15, 1, 5, None)]


def _visits() -> list:
    rows = []
    for other, n in SHARED.items():
        for i in range(n):
            patient = f"MRN{other}{i:05d}"
            rows += [(patient, 1), (patient, other)]
            if other == 2:
                rows.append((patient, 1))  # hostile duplicate visit
    return rows


@pytest.fixture(scope="module")
def pgraph(workspace_client, lakebase_project, lakebase_catalog, tmp_path_factory):
    from lgk.harness import BranchLeasor

    suffix = uuid.uuid4().hex[:6]
    run_folder = tmp_path_factory.mktemp("privacy_run")
    leasor = BranchLeasor(
        workspace_client=workspace_client, project_id=lakebase_project,
        test_base_branch="test-base", run_folder=run_folder, ttl_seconds=14400,
    )
    kit = f"lgk_v1_{suffix}"
    config: Dict[str, Any] = {
        "graph_name": "privacy_test", "lakebase_project": lakebase_project,
        "lakebase_catalog": lakebase_catalog, "kit_schema": kit,
        "served_schema": f"{kit}_served", "fixture_schema": f"{kit}_fx", "version": 1,
        "warehouse_id": WAREHOUSE_ID, "run_folder": str(run_folder),
    }
    try:
        config["branch_id"] = leasor.create_test_branch(f"privacy-{suffix}")["branch_id"]
        conn = _connect_with_retry(config)
        conn.autocommit = True
        with conn:
            _create_fixture(conn, config)
            versions.switch_views(conn, versions.served_schema(config), versions.version_schema(config))
        provision._setup_reader_role(config)
        yield config
    finally:
        _teardown_live(workspace_client, leasor, config)


def _create_fixture(conn: Any, config: Dict[str, Any]) -> None:
    fx, v1, v2 = config["fixture_schema"], versions.version_schema(config), versions.version_schema(config, 2)
    with conn.transaction():
        conn.execute(f"CREATE SCHEMA {fx}")
        conn.execute(f"CREATE TABLE {fx}.visits (patient text, provider bigint)")
        for row in _visits():
            conn.execute(f"INSERT INTO {fx}.visits VALUES (%s, %s)", row)
        conn.execute(f"CREATE SCHEMA {v1}")
        conn.execute(f"CREATE TABLE {v1}.nodes (node_id text PRIMARY KEY, label text)")
        conn.execute(f"CREATE TABLE {v1}.edges (edge_id bigint PRIMARY KEY, src_node_id text, "
                     f"dst_node_id text, weight bigint)")
        conn.execute(f"CREATE TABLE {v1}.node_id_map (node_id text PRIMARY KEY, natural_key text)")
        for n in range(1, 6):
            conn.execute(f"INSERT INTO {v1}.nodes VALUES (%s, %s)", (str(n), f"provider {n}"))
            conn.execute(f"INSERT INTO {v1}.node_id_map VALUES (%s, %s)", (str(n), f"p{n}"))
        for row in EDGES:
            # node ids are text, as in the real synced tables (build writes STRING ids)
            conn.execute(f"INSERT INTO {v1}.edges VALUES (%s, %s, %s, %s)",
                         (row[0], str(row[1]), str(row[2]), row[3]))
        conn.execute(f"CREATE SCHEMA {v2}")
        conn.execute(f"CREATE TABLE {v2}.seed_score (node_id bigint, score float8)")


def _as_reader(conn: Any) -> None:
    conn.execute("GRANT reader TO CURRENT_USER")
    conn.execute("SET ROLE reader")


# ------------------------------------------------------------------- floor: the fold itself
class TestFoldFloor:
    @pytest.mark.seam1
    def test_fold_serves_only_pairs_at_or_above_the_floor(self, pgraph):
        sql = floor.fold_edges_sql(f"{pgraph['fixture_schema']}.visits", "patient", "provider")
        with _connect_with_retry(pgraph) as conn:
            got = {(int(a), int(b), int(w)) for a, b, w in conn.execute(sql).fetchall()}
        # floor-1 (10 people, 20 rows) is hidden; floor and floor+1 are served, both directions
        assert got == {(1, 3, 11), (3, 1, 11), (1, 4, 12), (4, 1, 12)}
        assert all(w >= floor.DEFAULT_FLOOR for _, _, w in got)

    @pytest.mark.seam1
    def test_raised_fold_floor_hides_the_floor_pair(self, pgraph):
        sql = floor.fold_edges_sql(f"{pgraph['fixture_schema']}.visits", "patient", "provider", 12)
        with _connect_with_retry(pgraph) as conn:
            got = {(int(a), int(b), int(w)) for a, b, w in conn.execute(sql).fetchall()}
        assert got == {(1, 4, 12), (4, 1, 12)}


# ---------------------------------------------------------- floor: the served walk counts
class TestWalkFloor:
    def _walk(self, config: Dict[str, Any], **kw) -> Dict[int, Any]:
        sql = get_walk_query("unused", depth=1, served_schema=config["served_schema"], **kw)
        conn = _connect_with_retry(config)
        try:
            _as_reader(conn)  # served counts are what the website's reader sees
            rows = conn.execute(sql, {"start_node_id": "1"}).fetchall()
            return {int(node_id): count for _hop, node_id, count, _score in rows}
        finally:
            conn.rollback()
            conn.close()

    @pytest.mark.seam1
    def test_folded_walk_serves_only_counts_at_or_above_the_default_floor(self, pgraph):
        got = self._walk(pgraph, folded=True)  # default floor: no argument
        assert got == {3: 11, 4: 12}  # 10 (floor-1) and NULL weight hidden

    @pytest.mark.seam1
    def test_folded_walk_with_a_raised_floor(self, pgraph):
        assert self._walk(pgraph, folded=True, privacy_floor=12) == {4: 12}

    @pytest.mark.seam1
    def test_unfolded_walk_is_not_blanked_by_the_floor(self, pgraph):
        # hostile: the old HAVING COUNT(*) >= 11 counted parallel edges (always 1) and
        # returned nothing for a directory graph with no care-subject data
        got = self._walk(pgraph, folded=False)
        assert set(got) == {2, 3, 4, 5}


# ------------------------------------------------------------------- the reader probe
class TestReaderProbe:
    @pytest.mark.seam1
    def test_provisioned_reader_is_isolated(self, pgraph):
        provision._prove_reader_isolated(pgraph)  # raises on any violation
        with _connect_with_retry(pgraph) as conn:
            assert probe.reader_violations(conn, pgraph["kit_schema"]) == []
            # non-vacuous: the probe really looked at every raw table, in both versions
            raw = probe.raw_relations(conn, pgraph["kit_schema"])
        v1, v2 = versions.version_schema(pgraph), versions.version_schema(pgraph, 2)
        assert {(v1, "nodes"), (v1, "edges"), (v1, "node_id_map"), (v2, "seed_score")} <= set(raw)

    def _violations_after(self, config: Dict[str, Any], *statements: str) -> list:
        conn = _connect_with_retry(config)
        try:
            for s in statements:
                conn.execute(s)
            return probe.reader_violations(conn, config["kit_schema"])
        finally:
            conn.rollback()
            conn.close()

    @pytest.mark.seam1
    def test_table_grant_masked_by_schema_usage_is_caught(self, pgraph):
        v1 = versions.version_schema(pgraph)
        # hostile: the reader has no USAGE on v1, so a SELECT is denied - yet the table grant exists
        conn = _connect_with_retry(pgraph)
        try:
            conn.execute(f"GRANT SELECT ON {v1}.edges TO reader")
            _as_reader(conn)
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                with conn.transaction():
                    conn.execute(f"SELECT * FROM {v1}.edges")
            conn.execute("RESET ROLE")
            found = probe.reader_violations(conn, pgraph["kit_schema"])
        finally:
            conn.rollback()
            conn.close()
        # the table-level grant itself, not only its column-level shadow
        assert f"reader has SELECT on raw {v1}.edges" in found, found

    @pytest.mark.seam1
    def test_public_grant_on_a_raw_table_is_caught(self, pgraph):
        v1 = versions.version_schema(pgraph)
        found = self._violations_after(pgraph, f"GRANT SELECT ON {v1}.node_id_map TO PUBLIC")
        assert any("PUBLIC" in v and f"{v1}.node_id_map" in v for v in found), found

    @pytest.mark.seam1
    def test_column_grant_on_a_raw_table_is_caught(self, pgraph):
        v1 = versions.version_schema(pgraph)
        # hostile: a column-level grant leaves has_table_privilege false
        found = self._violations_after(pgraph, f"GRANT SELECT (label) ON {v1}.nodes TO reader")
        assert f"reader has column-level SELECT on raw {v1}.nodes" in found, found

    @pytest.mark.seam1
    def test_grant_on_an_older_versions_extra_table_is_caught(self, pgraph):
        v2 = versions.version_schema(pgraph, 2)
        # hostile: not one of the three synced tables, and not the current version
        found = self._violations_after(pgraph, f"GRANT SELECT ON {v2}.seed_score TO reader")
        assert any(f"{v2}.seed_score" in v for v in found), found

    @pytest.mark.seam1
    def test_membership_in_another_role_is_caught(self, pgraph):
        other = f"{pgraph['kit_schema']}_other"
        found = self._violations_after(pgraph, f"CREATE ROLE {other}", f"GRANT {other} TO reader")
        assert any("member" in v and other in v for v in found), found

    @pytest.mark.seam1
    def test_public_execute_on_a_raw_function_is_caught(self, pgraph):
        v1 = versions.version_schema(pgraph)
        # hostile: a new function gets EXECUTE for PUBLIC by default, with no GRANT at all
        found = self._violations_after(
            pgraph, f"CREATE FUNCTION {v1}.leak() RETURNS bigint LANGUAGE sql AS 'SELECT 1'")
        assert f"PUBLIC has EXECUTE on raw function {v1}.leak()" in found, found

    @pytest.mark.seam1
    def test_reader_execute_on_a_raw_function_is_caught(self, pgraph):
        v1 = versions.version_schema(pgraph)
        found = self._violations_after(
            pgraph, f"CREATE FUNCTION {v1}.leak2(x int) RETURNS int LANGUAGE sql AS 'SELECT x'",
            f"REVOKE EXECUTE ON FUNCTION {v1}.leak2(int) FROM PUBLIC",
            f"GRANT EXECUTE ON FUNCTION {v1}.leak2(int) TO reader")
        assert f"reader has EXECUTE on raw function {v1}.leak2(x integer)" in found, found
        assert not any("PUBLIC" in v and "leak2" in v for v in found), found

    @pytest.mark.seam1
    def test_grant_on_a_raw_sequence_is_caught(self, pgraph):
        v1 = versions.version_schema(pgraph)
        found = self._violations_after(
            pgraph, f"CREATE SEQUENCE {v1}.leak_seq", f"GRANT USAGE ON SEQUENCE {v1}.leak_seq TO reader")
        assert f"reader has USAGE on raw sequence {v1}.leak_seq" in found, found


# ------------------------------------------------- served tables bound to the decision
class TestProvisionRefusesStaleDeltaStamp:
    """Real build on real Delta tables, then the operator tightens: provision must refuse.

    hostile: the member LABEL is `patient_name` (reclassified to none, the looser decision),
    so the old nodes table shows names; after it is dropped, that table must never be synced.
    """

    @pytest.mark.seam1
    def test_tighten_after_build_refuses_the_old_delta_tables(self, pgraph, workspace_client,
                                                              tmp_path):
        import yaml
        from databricks.sdk.errors import NotFound

        from lgk.privacy import binding
        from lgk.privacy.flags import OpenFlagError, resolve_flag
        from lgk.privacy.gate import require_clear
        from lgk.steps.build import build
        from lgk.steps.privacy import privacy
        from lgk.uc_sql import execute_sql
        from tests.privacy_support import declared_schema_from_mapping

        cat, kit = pgraph["lakebase_catalog"], pgraph["kit_schema"]
        src = f"{kit}_src"
        config = {**pgraph, "run_folder": str(tmp_path), "version": 3}
        q = lambda s: execute_sql(workspace_client, WAREHOUSE_ID, s)  # noqa: E731
        # a node is never sourced from a table holding a care-subject
        # column, so patient_name lives in member_notes (declared in schema.json, never read
        # by the mapping); the decision on it is what tightens
        mapping = {"source_catalog": cat, "source_schema": src,
                   "nodes": {"member": {"source_table": "members", "natural_key": "row_key",
                                        "label": "display_name"}}, "edges": {}}
        try:
            q(f"CREATE SCHEMA IF NOT EXISTS {cat}.{src}")
            q(f"CREATE OR REPLACE TABLE {cat}.{src}.members (row_key STRING, display_name STRING)")
            q(f"INSERT INTO {cat}.{src}.members VALUES ('k1', 'Team A'), ('k2', 'Team B')")
            (tmp_path / "graph_mapping.yaml").write_text(yaml.safe_dump(mapping))
            declared_schema_from_mapping(str(tmp_path))
            schema = json.loads((tmp_path / "schema.json").read_text())
            schema["tables"].append({"name": "member_notes", "key_columns": ["note_key"],
                                     "foreign_keys": [],
                                     "columns": [{"name": "note_key", "type": "STRING"},
                                                 {"name": "patient_name", "type": "STRING"}]})
            (tmp_path / "schema.json").write_text(json.dumps(schema))
            with pytest.raises(OpenFlagError):
                privacy(config)
            resolve_flag(str(tmp_path), "member_notes", "patient_name", "reclassify", chooser="op",
                         reason="looser decision under test", new_class="none")
            privacy(config)
            build(config)  # real Delta tables, stamped with the looser decision
            loose = binding.stamp_id(require_clear(str(tmp_path)))
            props = {r[0]: r[1] for r in q(f"SHOW TBLPROPERTIES {cat}.{kit}.nodes")}
            # non-vacuous: build really stamped (the label is the JSON {stamp, table_id, certified_version})
            assert json.loads(props[binding.STAMP_PROPERTY])["stamp"] == loose

            # the operator tightens: drop patient_name, move the label off it
            resolve_flag(str(tmp_path), "member_notes", "patient_name", "drop", chooser="op")
            mapping["nodes"]["member"]["label"] = "row_key"
            (tmp_path / "graph_mapping.yaml").write_text(yaml.safe_dump(mapping))
            privacy(config)
            tight = binding.stamp_id(require_clear(str(tmp_path)))
            assert tight != loose

            # the real preflight as the operator, so the binding is what refuses
            from tests.preflight_support import preflight_as_operator

            preflight_as_operator(workspace_client, config)
            with pytest.raises(binding.StaleBuildError, match="different privacy decision"):
                provision.provision(config)
            # hostile: the run folder claims a fresh build; only the Delta property tells
            binding.record_build(str(tmp_path), tight, provision._kit_tables(config))
            with pytest.raises(binding.StaleBuildError, match=r"\.nodes \(stale stamp\)"):
                provision.provision(config)
            # a table from before the binding carries no property: refused, not trusted
            for t in ("nodes", "edges", "node_id_map"):
                q(f"ALTER TABLE {cat}.{kit}.{t} UNSET TBLPROPERTIES ('{binding.STAMP_PROPERTY}')")
            with pytest.raises(binding.StaleBuildError, match=r"\.edges \(no lgk.privacy_stamp\)"):
                provision.provision(config)
            # refused before anything was made: no UC destination schema for version 3
            with pytest.raises(NotFound):
                workspace_client.schemas.get(f"{cat}.{versions.version_schema(config)}")
        finally:
            q(f"DROP SCHEMA IF EXISTS {cat}.{src} CASCADE")
