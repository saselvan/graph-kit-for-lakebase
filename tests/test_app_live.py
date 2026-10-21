"""Seam-1 acceptance for the generated app: its query layer, run AS the reader role
over real served views on a test branch, equals a Spark reference. No Apps deploy happens.

The version tables are built directly in branch Postgres (no synced tables are needed: the
walk and the reader's privileges are Postgres facts), served through the real one-transaction
view switch, and the reader role is set up by the real provision helper. The app is generated
with ALL defaults (depth 2, top 25, PGHOST and the operator read live), then its own
`db.OAuthConnection`, `reader.configure_reader` and `walk.run_walk` are run.

Fixture (one node type `provider`, plus one `hospital`):
- seeds s0..s5 link to `HUB`, to 4 of w00..w29, to `DrSmith`; even seeds also to `a::b|c`.
- # hostile: `HUB` has 40 out-links (top 25 cuts it; every leaf ties on support 1 -> node_id).
- # hostile: s0 -> w00 twice (two edge rows): support 2, a duplicate at the aggregation grain.
- # hostile: self-loop s1 -> s1, and w_j -> s_(j%6) cycles back to the seeds.
- # hostile: keys with separators (`a::b|c`), camelCase (`DrSmith`), and `shared` as BOTH a
  provider and a hospital key (the start lookup must refuse it without a type).

The Spark reference runs on the SQL warehouse over a VALUES copy of the same edge rows, with
ROW_NUMBER windows per hop. It never sees the Postgres SQL.

Set walk: the mapping declares `npi` as a provider walk key, and nodes carry a jsonb
`properties` map, so the generated app serves GET /api/set-walk. Set-walk fixture rows:
- seeds: s0, s1, s2 have npi `SEED`. # hostile: s3 has `SEED ` (trailing space; equality is exact).
- # hostile: the hospital `shared` has npi `SEED` too (wrong type, same value: never a seed).
- # hostile: HUB has npi `STOP`: matched, and its 40 leaves (reachable only through it) never appear.
- # hostile: support sums across seeds (HUB and DrSmith are reached from all three seeds).
"""

import hashlib
import importlib.util
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import json
from contextlib import contextmanager

import psycopg
import pytest
import yaml

from lgk import versions
from lgk.app.generate import generate_app
from lgk.steps import provision
from lgk.uc_sql import execute_sql
from tests.privacy_support import clear_gate_offline
from tests.test_build_provision_walks import WAREHOUSE_ID, _teardown_live, served_graph_size
from tests.test_privacy_gate import PROVIDER_MAPPING, _write_mapping
from tests.test_versions_live import _connect_with_retry

SEEDS = [f"s{i}" for i in range(6)]
WEB = [f"w{j:02d}" for j in range(30)]
LEAVES = [f"L{k:02d}" for k in range(40)]
DEFAULT_DEPTH, DEFAULT_TOP_N = 2, 25  # the app's defaults; the test passes no app settings


def _nid(kind: str, key: str) -> str:
    return "n" + hashlib.md5(f"{kind}::{key}".encode()).hexdigest()[:15]


SET_SEEDS = ["s0", "s1", "s2"]  # npi = SEED_NPI
SEED_NPI, STOP_NPI = "SEED", "STOP"
_NPI = {"s0": SEED_NPI, "s1": SEED_NPI, "s2": SEED_NPI, "s3": SEED_NPI + " ", "HUB": STOP_NPI}
APP_MAPPING = {**PROVIDER_MAPPING,
               "nodes": {"provider": {**PROVIDER_MAPPING["nodes"]["provider"], "walk_keys": ["npi"]}}}


def _nodes() -> List[Tuple[str, str, str, str]]:
    """(node_id, node_type, natural_key, label)."""
    keys = SEEDS + WEB + LEAVES + ["HUB", "DrSmith", "a::b|c", "shared"]
    rows = [(_nid("provider", k), "provider", k, f"Provider {k}") for k in keys]
    return rows + [(_nid("hospital", "shared"), "hospital", "shared", "Hospital shared")]


def _properties(node: Tuple[str, str, str, str]) -> str:
    """The node's properties map as jsonb text (npi only where the fixture sets one)."""
    _id, kind, key, _label = node
    npi = SEED_NPI if kind == "hospital" else _NPI.get(key)  # hostile: a hospital with the seed value
    return json.dumps({} if npi is None else {"npi": npi})


def _edges() -> List[Tuple[str, str]]:
    """(src node_id, dst node_id), duplicates kept on purpose."""
    p = lambda k: _nid("provider", k)  # noqa: E731
    pairs = []
    for i, s in enumerate(SEEDS):
        pairs += [(p(s), p("HUB")), (p(s), p("DrSmith"))]
        pairs += [(p(s), p(WEB[(i * 3 + k) % 30])) for k in range(4)]
        if i % 2 == 0:
            pairs.append((p(s), p("a::b|c")))
    pairs.append((p("s0"), p("w00")))  # hostile: duplicate edge row
    pairs.append((p("s1"), p("s1")))  # hostile: self-loop
    pairs += [(p("HUB"), p(leaf)) for leaf in LEAVES]  # hostile: hub
    for j, w in enumerate(WEB):
        pairs += [(p(w), p(WEB[(j + 1) % 30])), (p(w), p(WEB[(j * 7) % 30])), (p(w), p(SEEDS[j % 6]))]
    pairs += [(p("DrSmith"), _nid("hospital", "shared")), (p("a::b|c"), p("HUB"))]
    return pairs


@pytest.fixture(scope="module")
def served_graph(workspace_client, lakebase_project, lakebase_catalog, tmp_path_factory):
    from lgk.harness import BranchLeasor

    suffix = uuid.uuid4().hex[:6]
    run_folder = tmp_path_factory.mktemp("app_run")
    leasor = BranchLeasor(
        workspace_client=workspace_client, project_id=lakebase_project,
        test_base_branch="test-base", run_folder=run_folder, ttl_seconds=14400,
    )
    kit = f"lgk_v1_{suffix}"
    config: Dict[str, Any] = {
        "graph_name": "app_test", "lakebase_project": lakebase_project,
        "lakebase_catalog": lakebase_catalog, "kit_schema": kit,
        "served_schema": f"{kit}_served", "fixture_schema": f"{kit}_fx", "version": 1,
        "warehouse_id": WAREHOUSE_ID, "run_folder": str(run_folder),
    }
    try:
        config["branch_id"] = leasor.create_test_branch(f"app-{suffix}")["branch_id"]
        conn = _connect_with_retry(config)
        conn.autocommit = True
        with conn:
            _create_version(conn, versions.version_schema(config))
            versions.switch_views(conn, versions.served_schema(config), versions.version_schema(config))
        provision._setup_reader_role(config)
        provision._prove_reader_isolated(config)
        with _connect_with_retry(config) as conn:
            conn.execute("GRANT reader TO CURRENT_USER")  # the app's identity can become reader
        _write_mapping(run_folder, APP_MAPPING)
        clear_gate_offline(str(run_folder))
        config["graph_size"] = served_graph_size(config)  # what every metric line is measured at
        yield config
    finally:
        _teardown_live(workspace_client, leasor, config)


def _create_version(conn: Any, v1: str) -> None:
    with conn.transaction():
        conn.execute(f"CREATE SCHEMA {v1}")
        conn.execute(f"CREATE TABLE {v1}.nodes (node_id text PRIMARY KEY, node_type text, "
                     "natural_key text, label text, properties jsonb)")
        conn.execute(f"CREATE TABLE {v1}.edges (edge_id bigint PRIMARY KEY, src_node_id text, "
                     "dst_node_id text, weight bigint)")
        conn.execute(f"CREATE TABLE {v1}.node_id_map (node_id text PRIMARY KEY, node_type text, "
                     "natural_key text)")
        with conn.cursor() as cur:
            cur.executemany(f"INSERT INTO {v1}.nodes VALUES (%s, %s, %s, %s, %s::jsonb)",
                            [(*n, _properties(n)) for n in _nodes()])
            cur.executemany(f"INSERT INTO {v1}.node_id_map VALUES (%s, %s, %s)",
                            [n[:3] for n in _nodes()])
            cur.executemany(f"INSERT INTO {v1}.edges VALUES (%s, %s, %s, NULL)",
                            [(i, s, d) for i, (s, d) in enumerate(_edges())])


# ------------------------------------------------------------------------ Spark reference
def spark_reference(w: Any, seeds: List[str], depth: int, top_n: int) -> Dict[str, list]:
    """{seed key: [(hop, node_id, support), ...]} by hop then rank (support desc, node_id)."""
    edges = ", ".join(f"('{s}', '{d}')" for s, d in _edges())
    starts = ", ".join(f"('{k}', '{_nid('provider', k)}')" for k in seeds)
    ctes = [f"e AS (SELECT * FROM VALUES {edges} AS e(src, dst))",
            f"h0 AS (SELECT * FROM VALUES {starts} AS h0(seed, node_id))"]
    for i in range(1, depth + 1):
        seen = " UNION ALL ".join(f"SELECT seed, node_id FROM h{j}" for j in range(i))
        ctes.append(
            f"h{i} AS (SELECT seed, node_id, support, rn FROM (SELECT c.*, ROW_NUMBER() OVER "
            f"(PARTITION BY c.seed ORDER BY c.support DESC, c.node_id) AS rn FROM ("
            f"SELECT f.seed, e.dst AS node_id, COUNT(*) AS support FROM h{i - 1} f "
            f"JOIN e ON e.src = f.node_id GROUP BY f.seed, e.dst) c "
            f"LEFT ANTI JOIN ({seen}) s ON s.seed = c.seed AND s.node_id = c.node_id) "
            f"WHERE rn <= {top_n})")
    union = " UNION ALL ".join(f"SELECT seed, {i} AS hop, rn, node_id, support FROM h{i}"
                               for i in range(1, depth + 1))
    out: Dict[str, list] = {k: [] for k in seeds}
    rows = execute_sql(w, WAREHOUSE_ID, f"WITH {', '.join(ctes)} {union}")
    for seed, hop, _rn, node, sup in sorted(rows, key=lambda r: (r[0], int(r[1]), int(r[2]))):
        out[seed].append((int(hop), str(node), int(sup)))
    return out


def spark_set_reference(w: Any, seeds: List[str], stops: List[str], depth: int, top_n: int) -> list:
    """[(hop, node_id, support, matched)] of the set walk, from the set-walk semantics: one
    frontier for all seeds (no PARTITION BY), only unmatched nodes expand, `matched` after the cut.
    `seeds` / `stops` are node ids (chosen by the caller from the fixture, not by any SQL)."""
    edges = ", ".join(f"('{s}', '{d}')" for s, d in _edges())
    stop_rows = ", ".join(f"('{n}')" for n in stops) or "(NULL)"
    ctes = [f"e AS (SELECT * FROM VALUES {edges} AS e(src, dst))",
            f"st AS (SELECT * FROM VALUES {stop_rows} AS st(node_id))",
            "h0 AS (SELECT node_id, false AS matched FROM VALUES "
            + ", ".join(f"('{n}')" for n in seeds) + " AS h0(node_id))"]
    for i in range(1, depth + 1):
        seen = " UNION ALL ".join(f"SELECT node_id FROM h{j}" for j in range(i))
        ctes.append(
            f"h{i} AS (SELECT k.node_id, k.support, k.rn, st.node_id IS NOT NULL AS matched FROM ("
            f"SELECT c.*, ROW_NUMBER() OVER (ORDER BY c.support DESC, c.node_id) AS rn FROM ("
            f"SELECT e.dst AS node_id, COUNT(*) AS support FROM h{i - 1} f JOIN e ON e.src = f.node_id "
            f"WHERE NOT f.matched GROUP BY e.dst) c LEFT ANTI JOIN ({seen}) s ON s.node_id = c.node_id"
            f") k LEFT JOIN st ON st.node_id = k.node_id WHERE k.rn <= {top_n})")
    union = " UNION ALL ".join(f"SELECT {i} AS hop, rn, node_id, support, matched FROM h{i}"
                               for i in range(1, depth + 1))
    rows = execute_sql(w, WAREHOUSE_ID, f"WITH {', '.join(ctes)} {union}")
    return [(int(h), str(n), int(sup), str(m).lower() == "true")
            for h, _rn, n, sup, m in sorted(rows, key=lambda r: (int(r[0]), int(r[1])))]


# ------------------------------------------------------------------- the generated app
def _load_backend(folder: str):
    """Import the generated backend/app package under a unique name; return (db, reader, walk)."""
    pkg = f"genapp_{uuid.uuid4().hex[:8]}"
    root = Path(folder) / "backend" / "app"
    spec = importlib.util.spec_from_file_location(pkg, root / "__init__.py",
                                                  submodule_search_locations=[str(root)])
    sys.modules[pkg] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules[pkg])
    return tuple(importlib.import_module(f"{pkg}.{m}") for m in ("db", "reader", "walk"))


class _OneConnPool:
    """psycopg_pool's connection() contract on one real connection: the block's transaction is
    committed when it ends, rolled back on an error (psycopg_pool is not a kit dev dependency)."""

    def __init__(self, conn):
        self.conn = conn

    @contextmanager
    def connection(self):
        try:
            yield self.conn
        except BaseException:
            self.conn.rollback()
            raise
        else:
            self.conn.commit()


@pytest.fixture(scope="module")
def app_folder(served_graph):
    """Generated with ALL app defaults: no app settings, PGHOST and operator read live."""
    return generate_app(served_graph)["app_folder"]


@pytest.fixture
def reader_conn(app_folder, served_graph, workspace_client, monkeypatch):
    """A connection made by the app's OWN OAuthConnection from its app.yaml env, configured by
    its own configure_reader. PGUSER is the operator here (the deployed app uses its SP)."""
    db, reader, walk = _load_backend(app_folder)
    env = {e["name"]: e["value"] for e in yaml.safe_load((Path(app_folder) / "app.yaml").read_text())["env"]}
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PGUSER", workspace_client.current_user.me().user_name)
    conn = db.OAuthConnection.connect(db.conninfo())
    try:
        reader.configure_reader(conn)
        yield conn, walk
    finally:
        conn.close()


@pytest.fixture
def reader_app(app_folder, reader_conn):
    """(the generated db module, a one-connection pool of the reader connection, walk)."""
    conn, walk = reader_conn
    db = sys.modules[walk.__name__.rsplit(".", 1)[0] + ".db"]
    return db, _OneConnPool(conn), walk


class TestGeneratedAppLive:
    @pytest.mark.seam1
    def test_app_yaml_points_at_the_test_branch(self, app_folder, served_graph):
        env = {e["name"]: e["value"] for e in yaml.safe_load((Path(app_folder) / "app.yaml").read_text())["env"]}
        assert env["ENDPOINT_NAME"] == (f"projects/{served_graph['lakebase_project']}/branches/"
                                        f"{served_graph['branch_id']}/endpoints/primary")
        assert env["PGHOST"] and "." in env["PGHOST"]

    @pytest.mark.seam1
    def test_app_walk_as_reader_equals_spark_reference(self, reader_conn, workspace_client, served_graph,
                                                        record_metric):
        conn, walk = reader_conn
        assert conn.execute("SELECT current_user").fetchone()[0] == "reader"
        assert (walk.DEPTH, walk.TOP_N) == (DEFAULT_DEPTH, DEFAULT_TOP_N)
        ref = spark_reference(workspace_client, SEEDS, DEFAULT_DEPTH, DEFAULT_TOP_N)
        # Non-vacuous: the hub is cut at hop 2, and the duplicate edge gives support 2.
        assert all(len([r for r in ref[k] if r[0] == 2]) == DEFAULT_TOP_N for k in SEEDS)
        assert (1, _nid("provider", "w00"), 2) in ref["s0"]
        labels = {n[0]: (n[1], n[2], n[3]) for n in _nodes()}
        samples = []
        for k in SEEDS:
            t0 = time.perf_counter()
            got = walk.run_walk(conn, k)
            samples.append(time.perf_counter() - t0)
            assert got["start"] == {"node_id": _nid("provider", k), "node_type": "provider", "natural_key": k}
            assert [(r["hop"], r["node_id"], r["support"]) for r in got["rows"]] == ref[k], k
            assert all((r["node_type"], r["natural_key"], r["label"]) == labels[r["node_id"]]
                       for r in got["rows"])
        record_metric.latencies("app.walk", samples, branch=served_graph["branch_id"], depth=DEFAULT_DEPTH,
                                role="reader", **served_graph["graph_size"])

    @pytest.mark.seam1
    def test_start_lookup_through_served_views(self, reader_conn):
        conn, walk = reader_conn
        with pytest.raises(walk.AmbiguousStart):
            walk.run_walk(conn, "shared")  # hostile: provider AND hospital
        got = walk.run_walk(conn, "shared", "hospital")
        assert got["start"]["node_id"] == _nid("hospital", "shared") and got["rows"] == []
        assert walk.run_walk(conn, "a::b|c")["rows"][0]["natural_key"] == "HUB"
        with pytest.raises(walk.StartNotFound):
            walk.run_walk(conn, "no-such-key")

    @pytest.mark.seam1
    def test_app_session_is_denied_on_raw_version_tables(self, reader_conn, served_graph):
        conn, _walk = reader_conn
        v1 = versions.version_schema(served_graph)
        for table in versions.VERSION_TABLES:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                with conn.transaction():
                    conn.execute(f"SELECT 1 FROM {v1}.{table} LIMIT 1")
        assert conn.execute("SHOW default_transaction_read_only").fetchone()[0] == "on"


class TestAppSetWalkAndTimeoutLive:
    """The app's walk session timeout and its set walk, as the reader."""

    @pytest.mark.seam1
    def test_walk_session_runs_with_a_5s_timeout_that_ends_with_the_walk(self, reader_app):
        db, pool, _walk = reader_app
        with pool.connection() as conn:
            default = conn.execute("SHOW statement_timeout").fetchone()[0]
        assert default != "5s"  # non-vacuous: the session default is not already 5 s
        with db.walk_session(pool) as conn:
            assert conn.execute("SHOW statement_timeout").fetchone()[0] == "5s"
        with pool.connection() as conn:  # SET LOCAL: gone after the walk's transaction
            assert conn.execute("SHOW statement_timeout").fetchone()[0] == default
        with pytest.raises(psycopg.errors.QueryCanceled):
            with db.walk_session(pool) as conn:
                conn.execute("SELECT pg_sleep(7)")

    @pytest.mark.seam1
    def test_set_walk_as_reader_equals_spark_set_reference(self, reader_app, workspace_client):
        db, pool, walk = reader_app
        assert set(walk.SET_WALK_SQL) == {("provider.npi", None), ("provider.npi", "provider.npi")}
        seeds = [_nid("provider", k) for k in SET_SEEDS]  # s3 ("SEED ") and the hospital excluded
        hub = _nid("provider", "HUB")
        for stop_key, stops in ((None, []), ("provider.npi", [hub])):
            ref = spark_set_reference(workspace_client, seeds, stops, DEFAULT_DEPTH, DEFAULT_TOP_N)
            with db.walk_session(pool) as conn:
                got = walk.run_set_walk(conn, "provider.npi", SEED_NPI, stop_key,
                                        STOP_NPI if stop_key else None)
            assert [(r["hop"], r["node_id"], r["support"], r["matched"]) for r in got["rows"]] == ref
            # non-vacuous: support sums over the 3 seeds; with the stop, HUB is matched and its
            # leaves are absent; without it, leaves appear at hop 2
            assert (1, hub, 3, bool(stops)) in ref
            leaves = {_nid("provider", leaf) for leaf in LEAVES}
            assert bool(leaves & {r[1] for r in ref}) is (not stops)
