"""Manifest tracking for all created objects and the per-graph ownership marker.

Every object the kit creates is recorded, at the moment it is made, in the run folder's
`lgk_manifest.json` (one record call per created object). The run-folder copy is what
teardown and the reaper read: Postgres-side records would die with a branch.

Every schema the kit creates or adopts carries a marker comment holding the graph name
(`COMMENT ON SCHEMA` in Postgres, the schema comment in UC). It is checked before any
drop, so two graphs that share an explicit kit_schema can never drop each other's objects.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ObjectType(Enum):
    """Types of objects the kit creates."""

    BRANCH = "branch"
    SCHEMA = "schema"
    PG_SCHEMA = "pg_schema"
    UC_SCHEMA = "uc_schema"
    ROLE = "role"
    SYNCED_TABLE = "synced_table"
    VIEW = "view"
    FUNCTION = "function"  # a served or staged function; object_id = "<schema>.<name>(<arg types>)"
    PIPELINE = "pipeline"
    UC_TABLE = "uc_table"
    JOB = "job"  # a Lakeflow job (the refresh job, the bundle's jobs); object_id = str(job_id)


@dataclass
class ManifestEntry:
    """Single entry in the manifest."""

    object_type: ObjectType
    object_id: str
    name: str
    created_at: datetime
    location: str  # Where the object lives (catalog.schema.table, postgres:role, etc)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entry to dict for JSON storage."""
        return {
            "object_type": self.object_type.value,
            "object_id": self.object_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "location": self.location,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManifestEntry":
        """Deserialize entry from dict."""
        return cls(
            object_type=ObjectType(data["object_type"]),
            object_id=data["object_id"],
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            location=data["location"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class Manifest:
    """Manifest of all objects created by a kit run."""

    graph_name: str
    entries: List[ManifestEntry] = field(default_factory=list)
    # The most sync sessions per running sync the nightly sampler has seen
    # (sessions / updates running, rounded up); refresh budgets max(18, this) + 2 per sync.
    sync_connection_high_water: Optional[int] = None
    # Per kit table, the label written on it ({stamp, table_id, certified_version});
    # the sync cross-checks the table's label against it.
    table_labels: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_entry(self, entry: ManifestEntry) -> None:
        """Add an object to the manifest."""
        self.entries.append(entry)
        logger.info(
            f"Manifest: added {entry.object_type.value} '{entry.object_id}' at {entry.location}"
        )

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        path = Path(path)
        data = {
            "graph_name": self.graph_name,
            "entries": [entry.to_dict() for entry in self.entries],
        }
        if self.sync_connection_high_water is not None:
            data["sync_connection_high_water"] = self.sync_connection_high_water
        if self.table_labels:
            data["table_labels"] = self.table_labels

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Manifest saved to {path} with {len(self.entries)} entries")

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Load manifest from JSON file."""
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        manifest = cls(graph_name=data["graph_name"],
                       sync_connection_high_water=data.get("sync_connection_high_water"),
                       table_labels=dict(data.get("table_labels") or {}))
        for entry_data in data.get("entries", []):
            manifest.add_entry(ManifestEntry.from_dict(entry_data))

        logger.info(f"Manifest loaded from {path} with {len(manifest.entries)} entries")
        return manifest

    def get_teardown_plan(self) -> List[ManifestEntry]:
        """Get objects in reverse creation order for safe teardown.

        Dependencies go first: synced tables before schemas, etc.
        """
        # Order dependencies: pipelines/synced tables first, then roles, then schemas
        # Higher priority value means delete earlier (more dependent = delete first)
        priority = {
            ObjectType.JOB: 20,  # first: nothing may re-sync during teardown
            ObjectType.PIPELINE: 10,
            ObjectType.SYNCED_TABLE: 10,
            ObjectType.UC_TABLE: 5,
            ObjectType.VIEW: 4,
            ObjectType.FUNCTION: 4,
            ObjectType.ROLE: 3,
            ObjectType.SCHEMA: 1,
            ObjectType.BRANCH: 1,
        }

        # Sort by priority (highest first) then by creation time (newest to oldest)
        sorted_entries = sorted(
            self.entries,
            key=lambda e: (-priority.get(e.object_type, 0), -e.created_at.timestamp()),
        )

        return sorted_entries


# ============================================================================
# Run-folder manifest of one graph
# ============================================================================

# Not `manifest.json`: the test harness keeps its own manifest there and rewrites it
# from memory, which would erase the kit's records.
MANIFEST_FILE = "lgk_manifest.json"


def _manifest_path(config: Dict[str, Any]) -> Path:
    return Path(config["run_folder"]) / MANIFEST_FILE


def _checked_graph(manifest: Manifest, config: Dict[str, Any], path: Path) -> Manifest:
    from lgk.versions import NotOursError

    if manifest.graph_name != config["graph_name"]:
        raise NotOursError(
            f"{path} is the manifest of graph {manifest.graph_name!r}, not {config['graph_name']!r}"
        )
    return manifest


def load_run_manifest(config: Dict[str, Any]) -> Manifest:
    """The graph's run-folder manifest. Raises FileNotFoundError if nothing was recorded."""
    path = _manifest_path(config)
    if not path.exists():
        raise FileNotFoundError(f"No kit manifest at {path}: nothing was recorded for this run")
    return _checked_graph(Manifest.load(path), config, path)


def record(
    config: Dict[str, Any],
    object_type: ObjectType,
    object_id: str,
    location: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Record one object the kit just created. Recording the same object again is a no-op."""
    if not config.get("run_folder"):
        logger.warning(f"No run_folder: {object_type.value} {object_id} is NOT recorded for teardown")
        return
    path = _manifest_path(config)
    if path.exists():
        manifest = _checked_graph(Manifest.load(path), config, path)
    else:
        manifest = Manifest(graph_name=config["graph_name"])
    if any(e.object_type == object_type and e.object_id == object_id for e in manifest.entries):
        return
    manifest.add_entry(
        ManifestEntry(
            object_type=object_type,
            object_id=object_id,
            name=object_id.rsplit(".", 1)[-1],
            created_at=datetime.now(tz=UTC),
            location=location,
            metadata=metadata or {},
        )
    )
    manifest.save(path)


def read_sync_high_water(config: Dict[str, Any]) -> Optional[int]:
    """The run's recorded sync connection high-water mark, or None if none was recorded yet."""
    if not config.get("run_folder") or not _manifest_path(config).exists():
        return None
    value = load_run_manifest(config).sync_connection_high_water
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError(f"{_manifest_path(config)}: sync_connection_high_water must be an int >= 0, got {value!r}")
    return value


def record_sync_high_water(config: Dict[str, Any], peak: int) -> int:
    """Keep max(recorded, peak) as the run's sync connection high-water mark; returns the kept value."""
    if not config.get("run_folder"):
        logger.warning(f"No run_folder: sync connection peak {peak} is NOT recorded")
        return peak
    path = _manifest_path(config)
    manifest = (_checked_graph(Manifest.load(path), config, path) if path.exists()
                else Manifest(graph_name=config["graph_name"]))
    kept = max(read_sync_high_water(config) or 0, peak)
    manifest.sync_connection_high_water = kept
    manifest.save(path)
    return kept


def record_table_labels(config: Dict[str, Any], labels: Dict[str, Dict[str, Any]]) -> None:
    """Record the label just written on each kit table (replacing that table's earlier record)."""
    if not config.get("run_folder"):
        logger.warning(f"No run_folder: the labels of {sorted(labels)} are NOT recorded; the sync will refuse them")
        return
    path = _manifest_path(config)
    manifest = (_checked_graph(Manifest.load(path), config, path) if path.exists()
                else Manifest(graph_name=config["graph_name"]))
    manifest.table_labels.update({fq: dict(label) for fq, label in labels.items()})
    manifest.save(path)


def read_table_labels(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{fq table: label} recorded in the run manifest; empty when nothing was recorded."""
    if not config.get("run_folder") or not _manifest_path(config).exists():
        return {}
    return load_run_manifest(config).table_labels


def pg_location(config: Dict[str, Any]) -> str:
    """The manifest location of a Postgres object on the run's branch."""
    from lgk.branch import BranchRef

    return BranchRef.from_config(config).pg_location


# The tag every kit-made job carries: teardown deletes a recorded job only if it holds this graph.
JOB_TAG = "lgk_graph"


def job_tags(graph_name: str) -> Dict[str, str]:
    from lgk.versions import _checked_name

    return {JOB_TAG: _checked_name(graph_name, "graph_name")}


# ============================================================================
# Ownership marker
# ============================================================================


def marker(graph_name: str) -> str:
    """The comment every kit schema carries: exactly this graph's name."""
    from lgk.versions import _checked_name

    return f"lgk:graph={_checked_name(graph_name, 'graph_name')}"


def refuse_unless_marked(where: str, comment: Optional[str], graph_name: str) -> None:
    """Raise NotOursError unless `comment` is this graph's marker."""
    from lgk.versions import NotOursError

    if comment != marker(graph_name):
        raise NotOursError(
            f"{where} is not marked as graph {graph_name!r} (its marker is {comment!r}); refusing to touch it"
        )


PG_SCHEMA_COMMENT_SQL = (
    "SELECT obj_description(oid, 'pg_namespace') FROM pg_namespace WHERE nspname = %s"
)


def claim_pg_schema(conn: Any, schema: str, graph_name: str, adopt_unmarked: bool = False) -> bool:
    """Create-and-mark a Postgres schema, or prove an existing one is this graph's.

    Returns True when the schema was created here. An existing schema must carry this
    graph's marker; `adopt_unmarked` marks a schema made for this run by someone else
    (the synced-table pipeline makes the version schema). Commits its own transaction.
    """
    with conn.transaction():
        row = conn.execute(PG_SCHEMA_COMMENT_SQL, (schema,)).fetchone()
        if row is None:
            conn.execute(f"CREATE SCHEMA {schema}")
        elif not (adopt_unmarked and row[0] is None):
            refuse_unless_marked(f"Postgres schema {schema}", row[0], graph_name)
            return False
        conn.execute(f"COMMENT ON SCHEMA {schema} IS '{marker(graph_name)}'")
    return row is None


def uc_schema_comment(w: Any, full_schema: str) -> tuple:
    """(exists, comment) of a UC schema."""
    from databricks.sdk.errors import NotFound

    try:
        return True, w.schemas.get(full_schema).comment
    except NotFound:
        return False, None


def claim_uc_schema(w: Any, warehouse_id: str, catalog: str, schema: str, graph_name: str) -> bool:
    """Create-and-mark a UC schema, or prove an existing one is this graph's. True if created."""
    from lgk.uc_sql import execute_sql

    full = f"{catalog}.{schema}"
    exists, comment = uc_schema_comment(w, full)
    if not exists:
        execute_sql(w, warehouse_id, f"CREATE SCHEMA IF NOT EXISTS {full} COMMENT '{marker(graph_name)}'")
        _, comment = uc_schema_comment(w, full)  # a concurrent creator wins; prove it was us
    refuse_unless_marked(f"UC schema {full}", comment, graph_name)
    return not exists
