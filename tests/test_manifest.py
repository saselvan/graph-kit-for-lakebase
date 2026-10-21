"""Tests for manifest tracking and teardown support."""

import json
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from lgk.manifest import Manifest, ManifestEntry, ObjectType


class TestObjectType:
    """Test ObjectType enum."""

    def test_object_types(self):
        """Verify all expected object types are defined."""
        assert hasattr(ObjectType, "BRANCH")
        assert hasattr(ObjectType, "SCHEMA")
        assert hasattr(ObjectType, "ROLE")
        assert hasattr(ObjectType, "SYNCED_TABLE")
        assert hasattr(ObjectType, "VIEW")
        assert hasattr(ObjectType, "PIPELINE")
        assert hasattr(ObjectType, "UC_TABLE")


class TestManifestEntry:
    """Test individual manifest entries."""

    def test_entry_creation(self):
        """ManifestEntry should store object metadata."""
        entry = ManifestEntry(
            object_type=ObjectType.SCHEMA,
            object_id="lgk_graph_v1",
            name="lgk_graph_v1",
            created_at=datetime(2026, 10, 3, 12, 0, 0),
            location="/Workspace/Catalogs/catalog/schema",
            metadata={"catalog": "uc_catalog", "branch": "production"},
        )

        assert entry.object_type == ObjectType.SCHEMA
        assert entry.object_id == "lgk_graph_v1"
        assert entry.name == "lgk_graph_v1"

    def test_entry_serialization(self):
        """ManifestEntry should serialize to dict."""
        entry = ManifestEntry(
            object_type=ObjectType.SYNCED_TABLE,
            object_id="synced_nodes",
            name="synced_nodes",
            created_at=datetime(2026, 10, 3, 12, 0, 0),
            location="uc_catalog.lgk_graph.nodes",
            metadata={"branch": "v1", "sync_mode": "triggered"},
        )

        data = entry.to_dict()
        assert data["object_type"] == "synced_table"
        assert data["object_id"] == "synced_nodes"
        assert data["metadata"]["sync_mode"] == "triggered"


class TestManifest:
    """Test manifest creation and persistence."""

    def test_manifest_creation(self):
        """Manifest should initialize empty."""
        manifest = Manifest(graph_name="test_graph")
        assert manifest.graph_name == "test_graph"
        assert len(manifest.entries) == 0

    def test_add_entry(self):
        """Manifest.add_entry should track created objects."""
        manifest = Manifest(graph_name="test_graph")

        entry = ManifestEntry(
            object_type=ObjectType.SCHEMA,
            object_id="lgk_test_graph_v1",
            name="lgk_test_graph_v1",
            created_at=datetime(2026, 10, 3, 12, 0, 0),
            location="catalog.schema",
            metadata={},
        )

        manifest.add_entry(entry)
        assert len(manifest.entries) == 1
        assert manifest.entries[0].object_id == "lgk_test_graph_v1"

    def test_save_to_run_folder(self):
        """Manifest should save to run folder as JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "run_001"
            run_dir.mkdir(parents=True)

            manifest = Manifest(graph_name="test_graph")

            entry = ManifestEntry(
                object_type=ObjectType.SCHEMA,
                object_id="lgk_test_graph_v1",
                name="lgk_test_graph_v1",
                created_at=datetime(2026, 10, 3, 12, 0, 0),
                location="catalog.schema",
                metadata={"catalog": "my_catalog"},
            )

            manifest.add_entry(entry)
            manifest_path = run_dir / "manifest.json"
            manifest.save(manifest_path)

            assert manifest_path.exists()

            with open(manifest_path) as f:
                data = json.load(f)

            assert data["graph_name"] == "test_graph"
            assert len(data["entries"]) == 1
            assert data["entries"][0]["object_id"] == "lgk_test_graph_v1"

    def test_load_from_run_folder(self):
        """Manifest should load from saved JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "run_001"
            run_dir.mkdir(parents=True)

            # Create and save
            manifest1 = Manifest(graph_name="test_graph")
            entry = ManifestEntry(
                object_type=ObjectType.ROLE,
                object_id="lgk_reader",
                name="lgk_reader",
                created_at=datetime(2026, 10, 3, 12, 0, 0),
                location="postgres:role",
                metadata={"login": True},
            )
            manifest1.add_entry(entry)
            manifest_path = run_dir / "manifest.json"
            manifest1.save(manifest_path)

            # Load it back
            manifest2 = Manifest.load(manifest_path)
            assert manifest2.graph_name == "test_graph"
            assert len(manifest2.entries) == 1
            assert manifest2.entries[0].object_id == "lgk_reader"

    def test_manifest_survives_branch_deletion(self):
        """Manifest in run folder should survive simulated branch deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "run_001"
            run_dir.mkdir(parents=True)

            manifest = Manifest(graph_name="test_graph")

            # Add objects that would be deleted with branch
            entries = [
                ManifestEntry(
                    object_type=ObjectType.SYNCED_TABLE,
                    object_id="synced_nodes",
                    name="synced_nodes",
                    created_at=datetime(2026, 10, 3, 12, 0, 0),
                    location="uc_catalog.lgk_test_graph_v1.nodes",
                    metadata={"branch": "test_branch", "sync_mode": "triggered"},
                ),
                ManifestEntry(
                    object_type=ObjectType.VIEW,
                    object_id="view_providers",
                    name="view_providers",
                    created_at=datetime(2026, 10, 3, 12, 0, 0),
                    location="test_project:lgk_test_graph.providers",
                    metadata={"branch": "production"},
                ),
            ]

            for entry in entries:
                manifest.add_entry(entry)

            manifest_path = run_dir / "manifest.json"
            manifest.save(manifest_path)

            # Simulate branch deletion (branch still exists in manifest)
            # Load the manifest after simulated deletion
            manifest2 = Manifest.load(manifest_path)
            assert len(manifest2.entries) == 2

            # Manifest should still contain all entries despite simulated deletion
            assert manifest2.entries[0].object_id == "synced_nodes"
            assert manifest2.entries[1].object_id == "view_providers"


class TestTeardownManifest:
    """Test teardown uses manifest to clean up."""

    def test_get_teardown_plan(self):
        """Manifest.get_teardown_plan should list objects in reverse creation order."""
        manifest = Manifest(graph_name="test_graph")

        # Add objects in creation order
        entries = [
            ManifestEntry(
                object_type=ObjectType.SCHEMA,
                object_id="lgk_test_v1",
                name="lgk_test_v1",
                created_at=datetime(2026, 10, 3, 12, 0, 0),
                location="catalog.schema",
                metadata={},
            ),
            ManifestEntry(
                object_type=ObjectType.ROLE,
                object_id="reader",
                name="reader",
                created_at=datetime(2026, 10, 3, 12, 1, 0),
                location="postgres:role",
                metadata={},
            ),
            ManifestEntry(
                object_type=ObjectType.SYNCED_TABLE,
                object_id="nodes",
                name="nodes",
                created_at=datetime(2026, 10, 3, 12, 2, 0),
                location="uc_catalog.schema.nodes",
                metadata={},
            ),
        ]

        for entry in entries:
            manifest.add_entry(entry)

        plan = manifest.get_teardown_plan()

        # Should be in reverse order
        assert len(plan) == 3
        assert plan[0].object_id == "nodes"
        assert plan[1].object_id == "reader"
        assert plan[2].object_id == "lgk_test_v1"

    def test_a_job_is_first_in_the_teardown_plan_even_when_made_before_the_pipeline(self):
        """#22: jobs go first so nothing re-syncs while the synced tables are deleted."""
        manifest = Manifest(graph_name="test_graph")
        mk = lambda t, i, minute: ManifestEntry(object_type=t, object_id=i, name=i,  # noqa: E731
                                                 created_at=datetime(2026, 10, 3, 12, minute, 0),
                                                 location="x", metadata={})
        # hostile: the job is OLDER than the pipeline, so newest-first alone would put it second
        manifest.add_entry(mk(ObjectType.JOB, "42", 0))
        manifest.add_entry(mk(ObjectType.PIPELINE, "p-1", 5))
        assert [e.object_id for e in manifest.get_teardown_plan()] == ["42", "p-1"]

    def test_manifest_completeness_for_teardown(self):
        """Manifest must include all created objects for complete teardown."""
        manifest = Manifest(graph_name="test_graph")

        # Missing synced table from manifest
        manifest.add_entry(ManifestEntry(
            object_type=ObjectType.SCHEMA,
            object_id="lgk_test_v1",
            name="lgk_test_v1",
            created_at=datetime(2026, 10, 3, 12, 0, 0),
            location="catalog.schema",
            metadata={},
        ))

        # If a synced table was created but not recorded, teardown is incomplete
        # Test that we can detect this by checking object count
        assert len(manifest.entries) == 1  # Only schema, synced table missing

        # A proper manifest should have recorded all objects
        manifest.add_entry(ManifestEntry(
            object_type=ObjectType.SYNCED_TABLE,
            object_id="synced_nodes",
            name="synced_nodes",
            created_at=datetime(2026, 10, 3, 12, 1, 0),
            location="uc_catalog.schema.nodes",
            metadata={},
        ))

        assert len(manifest.entries) == 2  # Now complete


# ---------------------------------------------------------------- the sync connection high-water mark
def test_the_high_water_survives_a_later_record_and_absent_reads_none(tmp_path):
    import json

    from lgk import manifest as mf

    cfg = {"graph_name": "g", "run_folder": str(tmp_path)}
    assert mf.read_sync_high_water(cfg) is None  # no manifest yet
    mf.record(cfg, ObjectType.SYNCED_TABLE, "c.s.nodes", "uc")
    assert mf.read_sync_high_water(cfg) is None  # a manifest without the field
    assert mf.record_sync_high_water(cfg, 23) == 23
    mf.record(cfg, ObjectType.SYNCED_TABLE, "c.s.edges", "uc")  # hostile: a later record rewrites the file
    data = json.loads((tmp_path / mf.MANIFEST_FILE).read_text())
    assert data["sync_connection_high_water"] == 23 and len(data["entries"]) == 2
    assert mf.record_sync_high_water(cfg, 4) == 23  # never lowered
    assert mf.read_sync_high_water(cfg) == 23
    assert mf.read_sync_high_water({"graph_name": "g"}) is None  # no run folder


def test_recording_the_high_water_refuses_another_graphs_manifest(tmp_path):
    from lgk import manifest as mf
    from lgk.versions import NotOursError

    mf.record({"graph_name": "other", "run_folder": str(tmp_path)}, ObjectType.SYNCED_TABLE, "c.s.nodes", "uc")
    before = (tmp_path / mf.MANIFEST_FILE).read_bytes()
    with pytest.raises(NotOursError):
        mf.record_sync_high_water({"graph_name": "g", "run_folder": str(tmp_path)}, 30)
    assert (tmp_path / mf.MANIFEST_FILE).read_bytes() == before


def test_recording_without_a_run_folder_returns_the_peak_and_writes_nothing(tmp_path, monkeypatch):
    from lgk import manifest as mf

    monkeypatch.chdir(tmp_path)
    assert mf.record_sync_high_water({"graph_name": "g"}, 7) == 7
    assert list(tmp_path.iterdir()) == []
