"""Offline tests for the MONDO loader, the attribution contract, and the no-code-change
front half (profile -> propose -> privacy) on MONDO-shaped gold tables.

Hostile fixture (MINI below), each row breaks a lazy loader:
- # hostile: an obsolete term (MONDO:0000002, meta.deprecated) that has a child AND a parent
- # hostile: a duplicate is_a edge (MONDO:0000003 is_a MONDO:0000001, twice)
- # hostile: an is_a to a non-MONDO class (MONDO:0000004 is_a HP:0000118)
- # hostile: a MONDO IRI that is a PROPERTY, not a CLASS; an unlabelled MONDO class
- # hostile: a non-is_a edge (part_of) between two kept terms
- # hostile: a label with a comma, quotes and non-ASCII characters
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from lgk import attribution as attr
from lgk.loaders import mondo
from lgk.runner import LaptopRunner, RunStatus
from lgk.steps.privacy import privacy
from lgk.steps.profile import profile
from lgk.steps.propose import propose
from lgk.mapping import load_mapping

OBO = "http://purl.obolibrary.org/obo/"
VERSION_IRI = "http://purl.obolibrary.org/obo/mondo/releases/2026-09-01/mondo.owl"
ODD_LABEL = 'Sjögren "dry", eye (type 2)'


def _cls(n: str, lbl: Any = None, deprecated: bool = False, typ: str = "CLASS") -> Dict[str, Any]:
    node: Dict[str, Any] = {"id": f"{OBO}{n}", "type": typ}
    if lbl is not None:
        node["lbl"] = lbl
    if deprecated:
        node["meta"] = {"deprecated": True}
    return node


def _isa(sub: str, obj: str, pred: str = "is_a") -> Dict[str, str]:
    return {"sub": f"{OBO}{sub}", "pred": pred, "obj": f"{OBO}{obj}"}


MINI = {"graphs": [{
    "id": "http://purl.obolibrary.org/obo/mondo.owl",
    "meta": {"version": VERSION_IRI, "basicPropertyValues": [
        {"pred": "http://purl.org/dc/terms/license", "val": "http://creativecommons.org/licenses/by/4.0/"}]},
    "nodes": [
        _cls("MONDO_0000004", ODD_LABEL),  # reordered input: sorted output is promised
        _cls("MONDO_0000001", "disease"),
        _cls("MONDO_0000002", "obsolete thing", deprecated=True),  # hostile: obsolete
        _cls("MONDO_0000003", "cancer"),
        _cls("MONDO_0000005"),  # hostile: unlabelled
        _cls("MONDO_0000006", "has_mondo_prop", typ="PROPERTY"),  # hostile: not a CLASS
        _cls("HP_0000118", "Phenotypic abnormality"),
    ],
    "edges": [
        _isa("MONDO_0000003", "MONDO_0000001"),
        _isa("MONDO_0000003", "MONDO_0000001"),  # hostile: duplicate is_a
        _isa("MONDO_0000004", "MONDO_0000003"),
        _isa("MONDO_0000004", "MONDO_0000001"),
        _isa("MONDO_0000004", "HP_0000118"),  # hostile: non-MONDO target
        _isa("MONDO_0000002", "MONDO_0000001"),  # obsolete child
        _isa("MONDO_0000003", "MONDO_0000002"),  # obsolete parent
        _isa("MONDO_0000005", "MONDO_0000001"),  # unlabelled child
        _isa("MONDO_0000003", "MONDO_0000004", pred=f"{OBO}BFO_0000050"),  # hostile: part_of, no is_a twin
    ],
}]}

EXACT_ATTRIBUTION = (
    "Mondo Disease Ontology (MONDO), release v2026-09-01, © the Monarch Initiative. "
    "Source: https://mondo.monarchinitiative.org/ — used under CC BY 4.0 "
    "(https://creativecommons.org/licenses/by/4.0/). Citation: \"Mondo: integrating disease "
    "terminology across communities,\" Genetics (2025), https://doi.org/10.1093/genetics/iyaf215."
)


# ------------------------------------------------------------------------------ parsing


def test_parse_keeps_labelled_non_obsolete_mondo_classes_sorted():
    rel = mondo.parse_mondo(MINI)
    assert rel.terms == [("MONDO:0000001", "disease"), ("MONDO:0000003", "cancer"),
                         ("MONDO:0000004", ODD_LABEL)]
    assert rel.n_obsolete == 1


def test_parse_is_a_parent_to_child_deduped_and_mondo_only():
    rel = mondo.parse_mondo(MINI)
    assert rel.is_a == [("MONDO:0000001", "MONDO:0000003"), ("MONDO:0000001", "MONDO:0000004"),
                        ("MONDO:0000003", "MONDO:0000004")]
    assert rel.n_external_is_a == 1


def test_parse_release_version_from_version_iri():
    rel = mondo.parse_mondo(MINI)
    assert (rel.release_version, rel.version_iri) == ("v2026-09-01", VERSION_IRI)
    assert rel.license_url == "http://creativecommons.org/licenses/by/4.0/"


def test_parse_release_follows_the_iri_not_a_constant():
    doc = json.loads(json.dumps(MINI))
    doc["graphs"][0]["meta"]["version"] = VERSION_IRI.replace("2026-09-01", "2031-01-15")
    assert mondo.parse_mondo(doc).release_version == "v2031-01-15"


@pytest.mark.parametrize("meta", [{}, {"version": "http://purl.obolibrary.org/obo/mondo.owl"},
                                  {"version": "http://purl.obolibrary.org/obo/mondo/2026-09-01/mondo.owl"}])
def test_parse_refuses_a_release_with_no_dated_version_iri(meta):
    doc = json.loads(json.dumps(MINI))
    doc["graphs"][0]["meta"] = meta
    with pytest.raises(mondo.MondoParseError, match="versionIRI"):
        mondo.parse_mondo(doc)


def test_parse_refuses_a_document_with_no_graphs():
    with pytest.raises(mondo.MondoParseError, match="no graphs"):
        mondo.parse_mondo({"graphs": []})


def test_parse_accepts_subclassof_iri_as_is_a():
    doc = json.loads(json.dumps(MINI))
    doc["graphs"][0]["edges"] = [_isa("MONDO_0000004", "MONDO_0000003",
                                      pred="http://www.w3.org/2000/01/rdf-schema#subClassOf")]
    assert mondo.parse_mondo(doc).is_a == [("MONDO:0000003", "MONDO:0000004")]


# ------------------------------------------------------------------------------ attribution


def test_attribution_text_is_the_exact_cc_by_string():
    assert mondo.attribution_text("v2026-09-01") == EXACT_ATTRIBUTION


def test_attribution_record_carries_the_parsed_release():
    rec = mondo.attribution(mondo.parse_mondo(MINI))
    assert rec["text"] == EXACT_ATTRIBUTION
    assert rec["release_version"] == "v2026-09-01"
    assert attr.validate(rec)["changes"].startswith("Derived from MONDO")


def _good() -> Dict[str, str]:
    return mondo.attribution(mondo.parse_mondo(MINI))


@pytest.mark.parametrize("field", attr.REQUIRED)
@pytest.mark.parametrize("bad", [None, "", "   ", 7])
def test_validate_refuses_missing_or_blank_required_field(field, bad):
    rec = _good()
    if bad is None:
        rec.pop(field)
    else:
        rec[field] = bad
    with pytest.raises(attr.AttributionError, match=field):
        attr.validate(rec)


@pytest.mark.parametrize("field", ["release_version", "license_url", "citation_url"])
def test_validate_refuses_text_that_omits_a_recorded_field(field):
    rec = _good()
    rec["text"] = rec["text"].replace(rec[field], "")
    with pytest.raises(attr.AttributionError, match=field):
        attr.validate(rec)


def test_validate_refuses_non_string_changes_and_defaults_missing_changes():
    rec = _good()
    rec["changes"] = ["x"]
    with pytest.raises(attr.AttributionError, match="changes"):
        attr.validate(rec)
    rec.pop("changes")
    assert attr.validate(rec)["changes"] == ""


def test_write_then_read_attribution_roundtrip(tmp_path):
    assert attr.read_attribution(str(tmp_path)) is None
    attr.write_attribution(str(tmp_path), _good())
    assert attr.read_attribution(str(tmp_path))["text"] == EXACT_ATTRIBUTION


def test_write_attribution_refuses_a_broken_record(tmp_path):
    with pytest.raises(attr.AttributionError):
        attr.write_attribution(str(tmp_path), {**_good(), "text": "MONDO data"})
    assert not (tmp_path / attr.FILE).exists()


def test_switch_serves_attribution_after_the_views_and_reader_grants(monkeypatch):
    """#21: attribution lives in the served schema, so it is served by switch, not provision."""
    from lgk.steps import provision as prov
    from lgk.steps import switch as sw

    calls: List[str] = []
    monkeypatch.setattr(sw.binding, "require_servable", lambda *a: calls.append("require_servable"))
    monkeypatch.setattr(sw, "_move_views", lambda c, t: calls.append("_move_views"))
    monkeypatch.setattr(sw, "_record_switch", lambda c, t: calls.append("_record_switch"))
    for name in ("_prove_reader_isolated", "_setup_reader_role", "_serve_attribution"):
        monkeypatch.setattr(prov, name, lambda *a, _n=name, **k: calls.append(_n))
    sw.switch({"graph_name": "mondo", "run_folder": ".", "version": 1})
    assert calls == ["require_servable", "_prove_reader_isolated", "_move_views", "_setup_reader_role",
                     "_serve_attribution", "_record_switch"]


def test_read_attribution_validates_an_edited_file(tmp_path):
    attr.write_attribution(str(tmp_path), _good())
    data = json.loads((tmp_path / attr.FILE).read_text())
    data["text"] = "MONDO data"  # hostile: someone trims the notice after the loader wrote it
    (tmp_path / attr.FILE).write_text(json.dumps(data))
    with pytest.raises(attr.AttributionError):
        attr.read_attribution(str(tmp_path))


def test_provision_serves_attribution_only_when_the_run_has_one(tmp_path, monkeypatch):
    from lgk.steps import provision as prov

    served: List[Any] = []
    monkeypatch.setattr(prov, "connect_branch", lambda c: type("C", (), {"close": lambda s: None})())
    monkeypatch.setattr(prov, "serve_attribution", lambda conn, c, r: served.append((c["served_schema"], r)))
    config = {"graph_name": "mondo", "served_schema": "lgk_served_abc123", "run_folder": str(tmp_path)}
    prov._serve_attribution(config)
    assert served == []
    attr.write_attribution(str(tmp_path), _good())
    prov._serve_attribution(config)
    ((schema, record),) = served
    assert schema == "lgk_served_abc123"
    assert record["text"] == EXACT_ATTRIBUTION and record["release_version"] == "v2026-09-01"


class _RecordingConn:
    """psycopg connection boundary: records each statement rendered as SQL text."""

    def __init__(self):
        self.statements: List[str] = []

    def transaction(self):
        import contextlib

        return contextlib.nullcontext()

    def execute(self, statement, *a):
        from psycopg import sql as psql

        if isinstance(statement, psql.Composable):
            statement = "".join(_render(statement))
        self.statements.append(statement)


def _render(c):
    from psycopg import sql as psql

    if isinstance(c, psql.Composed):
        return [x for part in c for x in _render(part)]
    if isinstance(c, psql.Identifier):
        return [".".join(f'"{s}"' for s in c._obj)]
    if isinstance(c, psql.Literal):
        return ["'" + str(c._obj).replace("'", "''") + "'"]
    return [c.as_string(None) if isinstance(c, psql.SQL) else str(c)]


def test_serve_attribution_records_the_view_and_keeps_the_schema_marker(tmp_path):
    """#22: teardown drops only recorded objects and checks the served schema's marker.

    So the attribution view is recorded, and its notice goes on the VIEW: a COMMENT ON SCHEMA
    would overwrite the `lgk:graph=` marker and teardown (and every re-serve) would refuse.
    """
    from lgk import manifest as mf

    config = {"graph_name": "mondo", "served_schema": "lgk_served_abc123", "run_folder": str(tmp_path),
              "lakebase_project": "proj", "branch_id": "t-m"}
    conn = _RecordingConn()
    attr.serve_attribution(conn, config, attr.validate(_good()))
    assert not any("COMMENT ON SCHEMA" in s for s in conn.statements)
    assert any(s.startswith('COMMENT ON VIEW "lgk_served_abc123"."source_attribution"') for s in conn.statements)
    assert 'GRANT SELECT ON "lgk_served_abc123"."source_attribution" TO reader' in conn.statements
    views = [(e.object_id, e.location) for e in mf.load_run_manifest(config).entries
             if e.object_type == mf.ObjectType.VIEW]
    assert views == [("lgk_served_abc123.source_attribution", "postgres:projects/proj/branches/t-m")]


def test_attribution_query_reads_the_served_view():
    from lgk.app.queries import attribution_query

    q = attribution_query("lgk_served_abc123")
    assert q == ("SELECT text, release_version, license_url, citation_url, source_url, changes "
                 "FROM lgk_served_abc123.source_attribution")
    with pytest.raises(ValueError):
        attribution_query("x; DROP SCHEMA y")


# ------------------------------------------------------------------------------ loading


class _FakeFiles:
    def __init__(self):
        self.uploads: Dict[str, bytes] = {}

    def upload(self, path, data, overwrite=False):
        assert overwrite
        self.uploads[path] = data.read()


def _fake_load(monkeypatch, tmp_path):
    statements: List[str] = []
    monkeypatch.setattr(mondo, "execute_sql", lambda w, wh, s: statements.append(s) or [])
    w = type("W", (), {})()
    w.files = _FakeFiles()
    rel = mondo.load_mondo(w, "wh", "cat_a", "lgk_v1_abc123", str(tmp_path), doc=MINI)
    return rel, statements, w.files.uploads


def _rows(blob: bytes) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in blob.decode("utf-8").splitlines()]


def test_load_stages_exact_rows_to_the_volume(monkeypatch, tmp_path):
    _, _, up = _fake_load(monkeypatch, tmp_path)
    base = "/Volumes/cat_a/lgk_v1_abc123/mondo_raw"
    assert sorted(up) == [f"{base}/is_a.json", f"{base}/release.json", f"{base}/terms.json"]
    assert _rows(up[f"{base}/terms.json"]) == [
        {"mondo_id": "MONDO:0000001", "label": "disease"},
        {"mondo_id": "MONDO:0000003", "label": "cancer"},
        {"mondo_id": "MONDO:0000004", "label": ODD_LABEL}]
    assert _rows(up[f"{base}/is_a.json"]) == [
        {"parent_id": "MONDO:0000001", "child_id": "MONDO:0000003"},
        {"parent_id": "MONDO:0000001", "child_id": "MONDO:0000004"},
        {"parent_id": "MONDO:0000003", "child_id": "MONDO:0000004"}]
    (release,) = _rows(up[f"{base}/release.json"])
    assert release["release_version"] == "v2026-09-01"
    assert release["text"] == EXACT_ATTRIBUTION
    assert (release["n_terms"], release["n_is_a"], release["n_obsolete_excluded"],
            release["n_external_is_a_excluded"]) == (3, 3, 1, 1)


def test_load_creates_volume_then_keyed_tables_from_read_files(monkeypatch, tmp_path):
    _, sql, _ = _fake_load(monkeypatch, tmp_path)
    joined = "\n".join(sql)
    assert sql[0] == "CREATE SCHEMA IF NOT EXISTS cat_a.lgk_v1_abc123"
    assert sql[1] == "CREATE VOLUME IF NOT EXISTS cat_a.lgk_v1_abc123.mondo_raw"
    assert "PRIMARY KEY (mondo_id)" in joined
    assert joined.count("read_files('/Volumes/cat_a/lgk_v1_abc123/mondo_raw/") == 3
    # parent FK sorts before child FK by name: profile orders FKs by constraint name
    assert re.search(r"CONSTRAINT (\w+) FOREIGN KEY \(parent_id\)", joined).group(1) < \
        re.search(r"CONSTRAINT (\w+) FOREIGN KEY \(child_id\)", joined).group(1)
    assert "parquet.`" not in joined  # serverless blocks direct path queries
    # load_mondo creates exactly three tables: mondo_terms, mondo_is_a, mondo_release
    table_names = set(re.findall(r"CREATE OR REPLACE TABLE\s+\w+\.\w+\.(\w+)", joined))
    assert table_names == {"mondo_terms", "mondo_is_a", "mondo_release"}, \
        f"Expected tables {{'mondo_terms', 'mondo_is_a', 'mondo_release'}} but got {table_names}"


def test_load_writes_the_run_attribution(monkeypatch, tmp_path):
    _fake_load(monkeypatch, tmp_path)
    assert attr.read_attribution(str(tmp_path))["release_version"] == "v2026-09-01"


@pytest.mark.parametrize("catalog,schema", [("cat_a", "x; DROP"), ("Cat-A", "lgk_v1_abc123")])
def test_load_refuses_unsafe_identifiers(monkeypatch, catalog, schema):
    monkeypatch.setattr(mondo, "execute_sql", lambda *a: pytest.fail("SQL ran"))
    with pytest.raises(ValueError):
        mondo.load_mondo(object(), "wh", catalog, schema, "/nonexistent", doc=MINI)


def test_download_reads_a_cached_file(tmp_path):
    p = tmp_path / "mondo.json"
    p.write_text(json.dumps(MINI))
    assert mondo.download(dest=p) == MINI


# ------------------------------------------------------------------------------ no code change


def _mondo_gold_tables(run_folder: Path) -> None:
    run_folder.mkdir(parents=True, exist_ok=True)
    (run_folder / "gold_tables.json").write_text(json.dumps({"tables": [
        {"name": "mondo_terms", "key_columns": ["mondo_id"]},
        {"name": "mondo_is_a", "key_columns": [], "foreign_keys": [
            {"columns": ["parent_id"], "references": ["mondo_terms", "mondo_id"]},
            {"columns": ["child_id"], "references": ["mondo_terms", "mondo_id"]}]},
        {"name": "mondo_release", "key_columns": []},
    ]}))


def test_front_half_runs_on_mondo_with_no_code_change(tmp_path):
    """profile (declared) -> propose -> privacy through the laptop runner, MONDO tables only."""
    run = tmp_path / "mondo_run"
    _mondo_gold_tables(run)
    config = {"graph_name": "mondo", "run_folder": str(run),
              "source_catalog": "cat_a", "source_schema": "lgk_v1_abc123"}
    results = LaptopRunner(config).run_steps(
        [("profile", profile), ("propose", propose), ("privacy", privacy)])
    assert [(r.step, r.status) for r in results] == [
        ("profile", RunStatus.SUCCESS), ("propose", RunStatus.SUCCESS),
        ("privacy", RunStatus.SUCCESS)], [r.to_dict() for r in results]
    m = load_mapping(run / "graph_mapping.yaml")
    assert sorted(m.nodes) == ["mondo_terms"]
    assert m.nodes["mondo_terms"].natural_key == "mondo_id"
    ((name, e),) = m.edges.items()
    assert (e.source_table, e.src, e.src_type, e.dst, e.dst_type) == (
        "mondo_is_a", "parent_id", "mondo_terms", "child_id", "mondo_terms")
    flags = json.loads((run / "flags.json").read_text())
    assert flags["flags"] == []  # no care-subject columns: the real gate passes, nothing resolved
    assert "gate" in flags


def test_no_doctor_or_provider_literals_in_the_kit_steps():
    """Grep guard: the toolkit steps carry no domain vocabulary (the kit is general)."""
    steps = Path(__file__).resolve().parents[1] / "src" / "lgk" / "steps"
    hits = [f"{p.name}:{i}" for p in sorted(steps.glob("*.py"))
            for i, line in enumerate(p.read_text().splitlines(), 1)
            if re.search(r"provider|doctor|physician", line, re.I)]
    assert hits == []
