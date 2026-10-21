"""Test support for the kit-table label (owner decision: the nightly stamp step).

The label on nodes, edges and node_id_map is the JSON {stamp, table_id, certified_version}
(lgk.privacy.binding.label_json). Fakes that model no Delta history and no UC table (the
build-SQL and binding fakes) stand in for the UC/Delta identity read only: every kit table is
"tid:<fq>" at version 1 whose data is at version 1. The label check itself is never stubbed here.
"""

from typing import Any


def stub_table_identities(monkeypatch: Any) -> None:
    """Answer the label module's UC table_id + Delta history read for fakes that model neither."""
    from lgk.refresh import label

    monkeypatch.setattr(label, "read_identity", lambda w, sql, fq: label.TableIdentity(f"tid:{fq}", 1, 1))


def kit_label(stamp: str, fq: str) -> str:
    """The label a kit table built under `stamp` carries, as stub_table_identities sees the table."""
    from lgk.privacy import binding

    return binding.label_json(stamp, f"tid:{fq}", 1)
