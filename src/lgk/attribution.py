"""Source attribution: a licence notice the served layer shows with the data.

A gold source under an attribution licence (for example MONDO under CC BY 4.0) ships an
`attribution.json` in the run folder. The kit knows nothing about the source: provision
serves whatever record is there, in a reader-visible view (with the text as the view's
COMMENT), and the app reads it back with `lgk.app.queries.attribution_query`. The served
schema's own COMMENT stays the kit's ownership marker: teardown and every re-serve
check it, so the notice must never overwrite it. The view is recorded in the run manifest,
so whole-graph teardown drops it with the other served views.

Contract (every value a non-empty string):
- text: the exact attribution string; it must contain the release version, the licence
  URL and the citation URL, so the shown notice can never drift from the recorded fields
- release_version, license_url, citation_url, source_url
- changes (optional): how the served data differs from the source
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

FILE = "attribution.json"
VIEW = "source_attribution"
REQUIRED = ("text", "release_version", "license_url", "citation_url", "source_url")
COLUMNS = REQUIRED + ("changes",)
_IN_TEXT = ("release_version", "license_url", "citation_url")


class AttributionError(ValueError):
    """attribution.json breaks the contract."""


def validate(record: Dict[str, Any]) -> Dict[str, str]:
    """The record with every column present (changes defaults to ''); raises on a break."""
    for key in REQUIRED:
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            raise AttributionError(f"attribution field '{key}' must be a non-empty string, got {value!r}")
    for key in _IN_TEXT:
        if record[key] not in record["text"]:
            raise AttributionError(f"attribution text does not contain its {key} {record[key]!r}")
    changes = record.get("changes", "")
    if not isinstance(changes, str):
        raise AttributionError(f"attribution field 'changes' must be a string, got {changes!r}")
    return {**{k: record[k] for k in REQUIRED}, "changes": changes}


def write_attribution(run_folder: str, record: Dict[str, Any]) -> Path:
    path = Path(run_folder) / FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validate(record), indent=2, ensure_ascii=False))
    return path


def read_attribution(run_folder: str) -> Optional[Dict[str, str]]:
    """The validated record, or None when the run has no attribution.json."""
    path = Path(run_folder) / FILE
    if not path.exists():
        return None
    return validate(json.loads(path.read_text()))


def serve_attribution(conn: Any, config: Dict[str, Any], record: Dict[str, str]) -> None:
    """In one transaction: the attribution view (security_invoker off) with the text as its
    COMMENT, and SELECT for `reader`; then the view is recorded for teardown.
    Values are bound as SQL literals, never formatted."""
    from psycopg import sql

    from lgk import manifest as mf
    from lgk.versions import served_schema

    served = served_schema(config)
    cols = sql.SQL(", ").join(
        sql.SQL("{}::text AS {}").format(sql.Literal(record[c]), sql.Identifier(c)) for c in COLUMNS)
    view = sql.Identifier(served, VIEW)
    with conn.transaction():
        conn.execute(sql.SQL("CREATE OR REPLACE VIEW {} WITH (security_invoker=off) AS SELECT {}")
                     .format(view, cols))
        conn.execute(sql.SQL("COMMENT ON VIEW {} IS {}").format(view, sql.Literal(record["text"])))
        conn.execute(sql.SQL("GRANT SELECT ON {} TO reader").format(view))
    mf.record(config, mf.ObjectType.VIEW, f"{served}.{VIEW}", mf.pg_location(config))
