"""Tests for test harness branch lifecycle and cleanup.

These tests verify the complete cycle of creating, using, and cleaning up test branches.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock

from lgk.harness import BranchLeasor, set_seam2_guard
from lgk.manifest import Manifest, ManifestEntry, ObjectType


class MockWorkspaceClient:
    """Mock WorkspaceClient for testing."""

    def __init__(self):
        """Initialize with mock API client."""
        self.api_client = Mock()
        self.api_client.do = MagicMock()


class TestBranchCreationAndCleanup:
    """Test branch creation and manifest-based cleanup."""

    def test_create_branch_records_in_manifest(self, tmp_path):
        """Creating a branch should record it in the manifest."""
        mock_client = MockWorkspaceClient()
        mock_client.api_client.do.return_value = {
            "branch_id": "test_branch_001",
        }

        set_seam2_guard(False)  # Enable cloud calls for this test
        leasor = BranchLeasor(
            workspace_client=mock_client,
            project_id="test_project",
            test_base_branch="test-base",
            run_folder=tmp_path,
            ttl_seconds=14400,
        )

        # Create a branch
        branch_info = leasor.create_test_branch("test_branch_001")

        # Verify it was created
        assert branch_info["branch_id"] == "test_branch_001"
        assert branch_info["branch_name"] == "test_branch_001"
        assert len(leasor.manifest.entries) == 1

        # Verify manifest was saved
        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()

    def test_cleanup_all_deletes_branches_via_manifest(self, tmp_path):
        """Cleanup should delete all branches recorded in manifest."""
        mock_client = MockWorkspaceClient()
        mock_client.api_client.do.return_value = {
            "branch_id": "test_branch_001",
        }

        set_seam2_guard(False)
        leasor = BranchLeasor(
            workspace_client=mock_client,
            project_id="test_project",
            test_base_branch="test-base",
            run_folder=tmp_path,
            ttl_seconds=14400,
        )

        # Create a branch
        leasor.create_test_branch("test_branch_001")
        assert len(leasor.manifest.entries) == 1

        # Reset the mock to track DELETE calls
        mock_client.api_client.do.reset_mock()
        mock_client.api_client.do.return_value = None

        # Cleanup
        deleted_count = leasor.cleanup_all_test_branches()

        # Verify cleanup happened
        assert deleted_count == 1
        mock_client.api_client.do.assert_called_once()

        # Verify call was DELETE
        call_args = mock_client.api_client.do.call_args
        assert call_args[0][0] == "DELETE"

    def test_cleanup_removes_from_manifest(self, tmp_path):
        """Cleanup should remove deleted branches from manifest."""
        mock_client = MockWorkspaceClient()
        mock_client.api_client.do.return_value = {
            "branch_id": "test_branch_001",
        }

        set_seam2_guard(False)
        leasor = BranchLeasor(
            workspace_client=mock_client,
            project_id="test_project",
            test_base_branch="test-base",
            run_folder=tmp_path,
        )

        # Create a branch
        leasor.create_test_branch("test_branch_001")
        assert len(leasor.manifest.entries) == 1

        # Reset mock and cleanup
        mock_client.api_client.do.reset_mock()
        leasor.cleanup_all_test_branches()

        # Verify manifest is now empty
        assert len(leasor.manifest.entries) == 0

        # Verify manifest was saved
        manifest_path = tmp_path / "manifest.json"
        loaded_manifest = Manifest.load(manifest_path)
        assert len(loaded_manifest.entries) == 0


class TestManifestSurvivesBranchDeletion:
    """Test that manifest survives (and enables recovery from) manual branch deletion."""

    def test_manifest_records_branches_before_deletion(self, tmp_path):
        """Manifest should record branches before they're deleted externally."""
        mock_client = MockWorkspaceClient()
        mock_client.api_client.do.return_value = {
            "branch_id": "test_branch_001",
        }

        set_seam2_guard(False)
        leasor = BranchLeasor(
            workspace_client=mock_client,
            project_id="test_project",
            test_base_branch="test-base",
            run_folder=tmp_path,
        )

        # Create a branch
        leasor.create_test_branch("test_branch_001")

        # Save manifest to disk
        manifest_path = tmp_path / "manifest.json"
        leasor._save_manifest()

        # Simulate external branch deletion by clearing leasor's manifest
        leasor.manifest.entries = []

        # But we can reload from disk and see what was created
        saved_manifest = Manifest.load(manifest_path)
        assert len(saved_manifest.entries) == 1
        assert saved_manifest.entries[0].object_id == "test_branch_001"

    def test_cleanup_fails_gracefully_if_branch_deleted(self, tmp_path):
        """Cleanup should handle the case where branch was already deleted."""
        mock_client = MockWorkspaceClient()
        mock_client.api_client.do.side_effect = [
            {"branch_id": "test_branch_001"},  # Create call
            Exception("Branch not found"),  # Delete call fails
        ]

        set_seam2_guard(False)
        leasor = BranchLeasor(
            workspace_client=mock_client,
            project_id="test_project",
            test_base_branch="test-base",
            run_folder=tmp_path,
        )

        # Create a branch
        leasor.create_test_branch("test_branch_001")
        assert len(leasor.manifest.entries) == 1

        # Attempt cleanup (branch was already deleted externally)
        deleted_count = leasor.cleanup_all_test_branches()

        # Should report as failed (returned False), so deleted_count is 0
        assert deleted_count == 0

        # Manifest should still have the entry (cleanup failed)
        assert len(leasor.manifest.entries) == 1


class TestBranchTTLRecorded:
    """Test that branch TTL is recorded in manifest for reference."""

    def test_branch_ttl_in_metadata(self, tmp_path):
        """Branch TTL should be stored in manifest metadata."""
        mock_client = MockWorkspaceClient()
        mock_client.api_client.do.return_value = {
            "branch_id": "test_branch_001",
        }

        set_seam2_guard(False)
        leasor = BranchLeasor(
            workspace_client=mock_client,
            project_id="test_project",
            test_base_branch="test-base",
            run_folder=tmp_path,
            ttl_seconds=14400,
        )

        # Create a branch
        leasor.create_test_branch("test_branch_001")

        # Check manifest entry
        entry = leasor.manifest.entries[0]
        assert entry.object_type == ObjectType.BRANCH
        assert entry.metadata["ttl_seconds"] == 14400


class TestMultipleBranchesCleanup:
    """Test cleanup of multiple branches in correct order."""

    def test_cleanup_multiple_branches(self, tmp_path):
        """Cleanup should delete multiple branches."""
        mock_client = MockWorkspaceClient()

        # Create call returns (incremented for each branch)
        create_responses = [
            {"branch_id": "test_branch_001"},
            {"branch_id": "test_branch_002"},
            {"branch_id": "test_branch_003"},
        ]

        # Delete calls succeed
        delete_responses = [None, None, None]

        responses = create_responses + delete_responses
        mock_client.api_client.do.side_effect = responses

        set_seam2_guard(False)
        leasor = BranchLeasor(
            workspace_client=mock_client,
            project_id="test_project",
            test_base_branch="test-base",
            run_folder=tmp_path,
        )

        # Create 3 branches
        for i in range(1, 4):
            leasor.create_test_branch(f"test_branch_{i:03d}")

        assert len(leasor.manifest.entries) == 3

        # Cleanup all
        deleted_count = leasor.cleanup_all_test_branches()

        assert deleted_count == 3
        assert len(leasor.manifest.entries) == 0


class TestCleanupFiltersByBranchType:
    """Test that cleanup only deletes ObjectType.BRANCH entries (hostile fixture).

    This test verifies the object_type filter is necessary and not a no-op.
    Mutation: removing the ObjectType.BRANCH check should cause this test to fail.
    """

    def test_cleanup_ignores_non_branch_entries(self, tmp_path):
        """Cleanup should only delete BRANCH objects, not all entries."""
        mock_client = MockWorkspaceClient()
        mock_client.api_client.do.return_value = None  # DELETE calls succeed

        set_seam2_guard(False)
        leasor = BranchLeasor(
            workspace_client=mock_client,
            project_id="test_project",
            test_base_branch="test-base",
            run_folder=tmp_path,
        )

        # Manually add a branch entry (what create_test_branch would add)
        branch_entry = ManifestEntry(
            object_type=ObjectType.BRANCH,  # HOSTILE: This entry should be deleted by cleanup
            object_id="test_branch_001",
            name="test_branch_001",
            created_at=datetime.now(tz=timezone.utc),
            location="lakebase:test_location",
            metadata={},
        )
        leasor.manifest.add_entry(branch_entry)

        # Manually add a non-branch schema entry (hostile fixture)
        schema_entry = ManifestEntry(
            object_type=ObjectType.SCHEMA,
            object_id="lgk_graph_schema",
            name="lgk_graph_schema",
            created_at=datetime.now(tz=timezone.utc),
            location="uc_catalog.schema",
            metadata={},  # HOSTILE: No branch_type marker — should NOT be deleted
        )
        leasor.manifest.add_entry(schema_entry)

        # Manually add a role entry (another non-branch)
        role_entry = ManifestEntry(
            object_type=ObjectType.ROLE,
            object_id="lgk_reader",
            name="lgk_reader",
            created_at=datetime.now(tz=timezone.utc),
            location="postgres:role",
            metadata={},  # HOSTILE: No branch_type marker — should NOT be deleted
        )
        leasor.manifest.add_entry(role_entry)

        assert len(leasor.manifest.entries) == 3

        # Cleanup only test branches
        deleted_count = leasor.cleanup_all_test_branches()

        # Should delete only the branch, not the schema or role
        assert deleted_count == 1

        # Verify only one DELETE was called (branch deletion)
        assert mock_client.api_client.do.call_count == 1
        call_args = mock_client.api_client.do.call_args
        assert call_args[0][0] == "DELETE"  # DELETE method
        assert "test_branch_001" in call_args[0][1]  # branch ID in URL

        # Verify non-branch entries are still in manifest
        assert len(leasor.manifest.entries) == 2
        object_ids = [e.object_id for e in leasor.manifest.entries]
        assert "lgk_graph_schema" in object_ids  # Schema survived
        assert "lgk_reader" in object_ids  # Role survived
