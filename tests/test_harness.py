"""Tests for test harness: Seam-1 (live) and Seam-2 (mock) separation."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from lgk.harness import (
    BranchLeasor,
    SeamGuard,
    _seam2_guard_active,
    set_seam2_guard,
)
from lgk.manifest import Manifest, ManifestEntry, ObjectType


class TestSeamGuard:
    """Test Seam-2 guard mechanism (no cloud calls allowed in Seam-2 tests)."""

    def test_seam2_guard_initially_off(self):
        """Seam-2 guard should start disabled."""
        # Reset to known state
        set_seam2_guard(False)
        assert not _seam2_guard_active()

    def test_seam2_guard_context_manager(self):
        """SeamGuard context manager should enable/disable guard."""
        set_seam2_guard(False)  # Start disabled

        with SeamGuard():
            assert _seam2_guard_active()  # Enabled inside

        assert not _seam2_guard_active()  # Disabled after

    def test_seam2_guard_preserves_state(self):
        """SeamGuard should restore previous state on exit."""
        set_seam2_guard(True)  # Start enabled

        with SeamGuard():
            assert _seam2_guard_active()  # Still enabled (context restores)

        assert _seam2_guard_active()  # Still enabled after

    def test_seam2_guard_nested(self):
        """Nested SeamGuard contexts should work correctly."""
        set_seam2_guard(False)

        with SeamGuard():
            assert _seam2_guard_active()

            with SeamGuard():
                assert _seam2_guard_active()

            assert _seam2_guard_active()

        assert not _seam2_guard_active()


class TestBranchLeasorManifest:
    """Test BranchLeasor manifest-based cleanup."""

    def test_branch_leasor_loads_or_creates_manifest(self, tmp_path):
        """BranchLeasor should load existing manifest or create new one."""
        # Mock workspace client
        class MockClient:
            pass

        run_folder = tmp_path / "runs"
        run_folder.mkdir()

        leasor = BranchLeasor(
            workspace_client=MockClient(),
            project_id="test_project",
            test_base_branch="test-base",
            run_folder=run_folder,
        )

        assert leasor.manifest.graph_name == "test_harness"
        assert len(leasor.manifest.entries) == 0

    def test_branch_leasor_saves_manifest(self, tmp_path):
        """BranchLeasor should save manifest to run folder."""
        class MockClient:
            pass

        run_folder = tmp_path / "runs"
        run_folder.mkdir()

        leasor = BranchLeasor(
            workspace_client=MockClient(),
            project_id="test_project",
            test_base_branch="test-base",
            run_folder=run_folder,
        )

        # Manually add an entry
        entry = ManifestEntry(
            object_type=ObjectType.BRANCH,
            object_id="test_branch",
            name="test_branch",
            created_at=datetime.now(tz=UTC),
            location="lakebase:test_location",
            metadata={},
        )
        leasor.manifest.add_entry(entry)
        leasor._save_manifest()

        # Verify manifest was saved
        manifest_path = run_folder / "manifest.json"
        assert manifest_path.exists()

        # Load it back
        loaded_manifest = Manifest.load(manifest_path)
        assert len(loaded_manifest.entries) == 1
        assert loaded_manifest.entries[0].object_id == "test_branch"


@pytest.mark.seam2
class TestSeam2NoCloudCalls:
    """Test that Seam-2 tests cannot make cloud calls."""

    def test_seam2_guard_prevents_branch_leasor_creation(self):
        """Creating BranchLeasor with Seam-2 guard active should use mock client."""
        # Inside a Seam-2 test, the guard is already active
        assert _seam2_guard_active()

        # Creating a mock workspace client should work
        class MockClient:
            pass

        # BranchLeasor creation itself is fine; cloud calls fail
        leasor = BranchLeasor(
            workspace_client=MockClient(),
            project_id="test_project",
            test_base_branch="test-base",
            run_folder=Path("/tmp"),
        )

        # But attempting to create a branch should fail
        with pytest.raises(RuntimeError, match="Cannot create Lakebase branch in Seam-2"):
            leasor.create_test_branch("test_branch")


# (Removed: TestSeam1ProvisionationTest, two `assert True` tests that passed with the provision
# marker and lock deleted. The lock is now proven in tests/test_provision_serialization.py.)


@pytest.mark.seam2
class TestSeam2UnitTest:
    """Test that Seam-2 tests can run without cloud calls."""

    def test_seam2_unit_test_with_guard(self, seam2_guard):
        """Seam-2 test can use guard to verify no cloud calls."""
        # Inside Seam-2 test, guard should already be active
        assert _seam2_guard_active()

    def test_seam2_manifest_operations_allowed(self, tmp_path):
        """Seam-2 tests can use Manifest without cloud calls."""
        manifest_path = tmp_path / "manifest.json"

        manifest = Manifest(graph_name="test_graph")
        entry = ManifestEntry(
            object_type=ObjectType.SCHEMA,
            object_id="test_schema",
            name="test_schema",
            created_at=datetime.now(tz=UTC),
            location="postgres:schema",
            metadata={},
        )
        manifest.add_entry(entry)
        manifest.save(manifest_path)

        # Verify save worked
        assert manifest_path.exists()

        # Load it back
        loaded = Manifest.load(manifest_path)
        assert len(loaded.entries) == 1
        assert loaded.entries[0].object_id == "test_schema"
