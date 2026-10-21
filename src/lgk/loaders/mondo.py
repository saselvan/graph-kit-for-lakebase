"""MONDO loader: the public Mondo Disease Ontology as gold Delta tables.

This runs BEFORE the kit, so the kit's rule stays "gold tables in". It is not a toolkit
step: the kit then reads these tables like any other gold schema, with no MONDO code.

Source: the OBOGraph JSON release (`graphs[0]` with `nodes`, `edges`, `meta`).

Decisions (documented, each tested):
- A term is a MONDO `CLASS` node (`/MONDO_` IRI) with a label. **Obsolete terms
  (`meta.deprecated`) are excluded**: they are retired and must not be served as diseases.
- An edge is an `is_a` (subClassOf) whose child AND parent are kept terms. **is_a links to
  non-MONDO classes (other ontologies) and to obsolete or unlabelled terms are excluded**,
  so every edge resolves to two served nodes (no orphans). Duplicate is_a pairs collapse.
- Edges run parent -> child, so a tree walk from a term lists its descendants.
- The release version comes from the ontology's versionIRI
  (`.../mondo/releases/2026-09-01/mondo.owl` -> `v2026-09-01`); it is recorded in the
  `mondo_release` table and in the attribution the served layer shows (CC BY 4.0).

Tables written to `<catalog>.<schema>` (a per-run `lgk_v1_<6hex>` schema):
- `mondo_terms (mondo_id PK, label)`
- `mondo_is_a (parent_id FK, child_id FK)` - a keyless two-FK join table, so the kit's
  propose drafts one edge type from it
- `mondo_release (release_version, version_iri, license_url, ...)` - keyless, not a node
"""

import io
import json
import logging
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lgk import ident
from lgk.attribution import write_attribution
from lgk.uc_sql import execute_sql

logger = logging.getLogger(__name__)

MONDO_URL = "https://purl.obolibrary.org/obo/mondo.json"
SOURCE_URL = "https://mondo.monarchinitiative.org/"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
CITATION_URL = "https://doi.org/10.1093/genetics/iyaf215"
VOLUME = "mondo_raw"

_OBO = "http://purl.obolibrary.org/obo/"
_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
_RELEASE = re.compile(r"/releases/(\d{4}-\d{2}-\d{2})/")


class MondoParseError(ValueError):
    """The file is not a MONDO OBOGraph release the loader can read."""


@dataclass
class MondoRelease:
    release_version: str
    version_iri: str
    license_url: Optional[str]
    terms: List[Tuple[str, str]] = field(default_factory=list)  # (mondo_id, label)
    is_a: List[Tuple[str, str]] = field(default_factory=list)  # (parent_id, child_id)
    n_obsolete: int = 0
    n_external_is_a: int = 0


# ----------------------------------------------------------------------------- parsing


def curie(iri: str) -> str:
    """http://purl.obolibrary.org/obo/MONDO_0000001 -> MONDO:0000001."""
    if iri.startswith(_OBO):
        return iri[len(_OBO):].replace("_", ":", 1)
    return iri


def release_version(meta: Dict[str, Any]) -> Tuple[str, str]:
    """(v<date>, versionIRI) from the graph meta; raises if the IRI carries no release date."""
    iri = str(meta.get("version") or "")
    m = _RELEASE.search(iri)
    if not m:
        raise MondoParseError(f"no release date in the ontology versionIRI {iri!r}")
    return f"v{m.group(1)}", iri


def _license(meta: Dict[str, Any]) -> Optional[str]:
    for bpv in meta.get("basicPropertyValues", []) or []:
        if str(bpv.get("pred", "")).endswith("/license"):
            return bpv.get("val")
    return None


def _terms(nodes: List[Dict[str, Any]]) -> Tuple[List[Tuple[str, str]], int]:
    """Kept (mondo_id, label) terms, sorted, and the number of obsolete terms excluded."""
    kept: Dict[str, str] = {}
    obsolete = 0
    for n in nodes:
        iri = str(n.get("id", ""))
        if "/MONDO_" not in iri or n.get("type") != "CLASS":
            continue
        if (n.get("meta") or {}).get("deprecated"):
            obsolete += 1
            continue
        if n.get("lbl"):
            kept[curie(iri)] = n["lbl"]
    return sorted(kept.items()), obsolete


def _is_a(edges: List[Dict[str, Any]], kept: set) -> Tuple[List[Tuple[str, str]], int]:
    """Distinct (parent, child) is_a pairs between kept terms, and the count of MONDO-child
    is_a links whose parent is outside MONDO (excluded)."""
    pairs = set()
    external = 0
    for e in edges:
        if e.get("pred") not in ("is_a", _SUBCLASS):
            continue
        child, parent = curie(str(e.get("sub", ""))), curie(str(e.get("obj", "")))
        if child not in kept:
            continue
        if not parent.startswith("MONDO:"):
            external += 1
            continue
        if parent in kept:
            pairs.add((parent, child))
    return sorted(pairs), external


def parse_mondo(doc: Dict[str, Any]) -> MondoRelease:
    """Parse an OBOGraph MONDO document into kept terms, is_a edges and its release."""
    graphs = doc.get("graphs") or []
    if not graphs:
        raise MondoParseError("no graphs in the document")
    g = graphs[0]
    meta = g.get("meta") or {}
    version, iri = release_version(meta)
    terms, obsolete = _terms(g.get("nodes", []) or [])
    is_a, external = _is_a(g.get("edges", []) or [], {t for t, _ in terms})
    return MondoRelease(version, iri, _license(meta), terms, is_a, obsolete, external)


# ----------------------------------------------------------------------------- attribution


def attribution_text(release: str) -> str:
    """The CC BY 4.0 attribution: Title, Author, Source, License and citation."""
    return (
        f"Mondo Disease Ontology (MONDO), release {release}, © the Monarch Initiative. "
        f"Source: {SOURCE_URL} — used under CC BY 4.0 ({LICENSE_URL}). "
        "Citation: \"Mondo: integrating disease terminology across communities,\" "
        f"Genetics (2025), {CITATION_URL}."
    )


CHANGES_NOTE = ("Derived from MONDO: obsolete terms and is_a links to non-MONDO classes were "
                "removed, and the is_a hierarchy is served as a graph.")


def attribution(rel: MondoRelease) -> Dict[str, str]:
    """The attribution record the kit serves (lgk.attribution contract)."""
    return {"text": attribution_text(rel.release_version), "changes": CHANGES_NOTE,
            "release_version": rel.release_version, "license_url": LICENSE_URL,
            "citation_url": CITATION_URL, "source_url": SOURCE_URL}


# ----------------------------------------------------------------------------- loading


def download(url: str = MONDO_URL, dest: Optional[Path] = None) -> Dict[str, Any]:
    """Fetch the release JSON (or read `dest` if it already exists, else save it there)."""
    if dest is not None and Path(dest).exists():
        return json.loads(Path(dest).read_text())
    with urllib.request.urlopen(url, timeout=300) as resp:  # noqa: S310 - fixed https URL
        raw = resp.read()
    if dest is not None:
        Path(dest).write_bytes(raw)
    return json.loads(raw)


def _ndjson(rows: List[Dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def _stage(w: Any, volume_path: str, rel: MondoRelease) -> Dict[str, str]:
    """Upload terms, edges and the release row as NDJSON files to the UC Volume."""
    files = {
        "terms": [{"mondo_id": i, "label": lbl} for i, lbl in rel.terms],
        "is_a": [{"parent_id": p, "child_id": c} for p, c in rel.is_a],
        "release": [{**attribution(rel), "version_iri": rel.version_iri,
                     "embedded_license": rel.license_url, "n_terms": len(rel.terms),
                     "n_is_a": len(rel.is_a), "n_obsolete_excluded": rel.n_obsolete,
                     "n_external_is_a_excluded": rel.n_external_is_a}],
    }
    paths = {}
    for name, rows in files.items():
        path = f"{volume_path}/{name}.json"
        w.files.upload(path, io.BytesIO(_ndjson(rows)), overwrite=True)
        paths[name] = path
    return paths


def _create_tables(sql, fq: str, paths: Dict[str, str]) -> None:
    """Delta tables from the staged files, with declared PK/FK constraints for profile."""
    sql(f"CREATE OR REPLACE TABLE {fq}.mondo_terms (mondo_id STRING NOT NULL, label STRING, "
        "CONSTRAINT mondo_terms_pk PRIMARY KEY (mondo_id)) USING DELTA")
    sql(f"INSERT INTO {fq}.mondo_terms SELECT mondo_id, label FROM read_files("
        f"'{paths['terms']}', format => 'json', schema => 'mondo_id STRING, label STRING')")
    # FK constraint names order parent before child: profile lists FKs by name, and propose
    # takes the first as the edge source, so edges run parent -> child.
    sql(f"CREATE OR REPLACE TABLE {fq}.mondo_is_a (parent_id STRING, child_id STRING, "
        f"CONSTRAINT fk_a_parent FOREIGN KEY (parent_id) REFERENCES {fq}.mondo_terms, "
        f"CONSTRAINT fk_b_child FOREIGN KEY (child_id) REFERENCES {fq}.mondo_terms) USING DELTA")
    sql(f"INSERT INTO {fq}.mondo_is_a SELECT parent_id, child_id FROM read_files("
        f"'{paths['is_a']}', format => 'json', schema => 'parent_id STRING, child_id STRING')")
    sql(f"CREATE OR REPLACE TABLE {fq}.mondo_release USING DELTA AS SELECT * FROM read_files("
        f"'{paths['release']}', format => 'json')")


def load_mondo(w: Any, warehouse_id: str, catalog: str, schema: str, run_folder: str,
               doc: Optional[Dict[str, Any]] = None) -> MondoRelease:
    """Parse MONDO (downloaded unless `doc` is given), stage it to a UC Volume in
    `catalog.schema`, and create the gold Delta tables there.

    Always writes the run's attribution.json (required: CC BY data is never served without
    its notice), which provision serves.
    """
    ident.check(catalog, "catalog")
    ident.check(schema, "schema")
    rel = parse_mondo(doc if doc is not None else download())
    fq = f"{catalog}.{schema}"
    sql = lambda s: execute_sql(w, warehouse_id, s)  # noqa: E731
    sql(f"CREATE SCHEMA IF NOT EXISTS {fq}")
    sql(f"CREATE VOLUME IF NOT EXISTS {fq}.{VOLUME}")
    _create_tables(sql, fq, _stage(w, f"/Volumes/{catalog}/{schema}/{VOLUME}", rel))
    write_attribution(run_folder, attribution(rel))
    logger.info(f"MONDO {rel.release_version}: {len(rel.terms)} terms, {len(rel.is_a)} is_a "
                f"({rel.n_obsolete} obsolete, {rel.n_external_is_a} external is_a excluded)")
    return rel
