"""Test helpers that pass the privacy gate LEGITIMATELY: profile, then the real privacy step.

Nothing here disables, mocks or stamps the gate. The privacy step runs as shipped; if the
fixture holds a care-subject column, it raises OpenFlagError and the test must resolve it.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from lgk.mapping import load_mapping
from lgk.privacy.flags import OpenFlagError, open_flags, resolve_flag
from lgk.steps.privacy import privacy


def declared_schema_from_mapping(run_folder: str) -> None:
    """Offline stand-in for profile: schema.json lists every column the mapping carries.

    Name-only verdicts (no patterns), exactly what the declared profile mode gives, and the
    mapping's source as the profiled source (declared mode records the configured source).
    """
    mapping = load_mapping(Path(run_folder) / "graph_mapping.yaml")
    cols: Dict[str, list] = {}
    for n in mapping.nodes.values():
        keys = n.natural_key if isinstance(n.natural_key, list) else [n.natural_key]
        cols.setdefault(n.source_table, []).extend([*keys, n.label, *n.properties])
    for e in mapping.edges.values():
        cols.setdefault(e.source_table, []).extend([e.src, e.dst])
    tables = [
        {"name": t, "key_columns": [], "foreign_keys": [],
         "columns": [{"name": c, "type": "STRING"} for c in dict.fromkeys(names)]}
        for t, names in cols.items()
    ]
    (Path(run_folder) / "schema.json").write_text(json.dumps({
        "source_catalog": mapping.source_catalog, "source_schema": mapping.source_schema,
        "tables": tables}))


def clear_gate_offline(run_folder: str) -> None:
    declared_schema_from_mapping(run_folder)
    privacy({"run_folder": run_folder})


def clear_gate_live(config: Dict[str, Any], workspace_client: Any,
                    drops: Iterable[Tuple[str, str]] = (),
                    reclassify: Iterable[Tuple[str, str]] = ()) -> None:
    """Live: profile the mapping's real source schema in UC (values -> rates only), then privacy.

    `drops` are the care-subject columns the caller expects the gate to flag and decides to
    drop (each must really be flagged); any other open flag still fails the run.
    """
    from lgk.steps.profile import profile

    mapping = load_mapping(Path(config["run_folder"]) / "graph_mapping.yaml")
    profile({**config, "source_catalog": mapping.source_catalog,
             "source_schema": mapping.source_schema, "workspace_client": workspace_client})
    resolve_expected_drops(config, drops, reclassify)


def resolve_expected_drops(config: Dict[str, Any], drops: Iterable[Tuple[str, str]],
                           reclassify: Iterable[Tuple[str, str]] = ()) -> None:
    """Run privacy; drop (or reclassify as public professional) exactly the expected flags
    (each must be flagged), then run it again."""
    drops, reclassify = list(drops), list(reclassify)
    if not drops and not reclassify:
        privacy(config)
        return
    try:
        privacy(config)
    except OpenFlagError:
        pass
    flagged = {(f["table"], f["column"]) for f in open_flags(config["run_folder"])}
    assert set(drops) | set(reclassify) <= flagged, \
        f"expected flags {drops + reclassify}, gate flagged {sorted(flagged)}"
    for table, column in drops:
        resolve_flag(config["run_folder"], table, column, "drop", chooser="lgk-live-test")
    for table, column in reclassify:
        resolve_flag(config["run_folder"], table, column, "reclassify", chooser="lgk-live-test",
                     reason="a provider's scheduling flag, not data about a patient",
                     new_class="public_professional")
    privacy(config)
