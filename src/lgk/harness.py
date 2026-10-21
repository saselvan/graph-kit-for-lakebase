"""Test harness for lakebase-graph-kit integration tests.

Provides fixtures for:
- Isolated child branches with automatic cleanup
- Seam-1 (live) vs Seam-2 (mock) test separation
- Serialization of provision-type tests
"""

import logging
import os
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lgk.branch import BranchRef, delete_branch_retrying_cap, is_operation_cap
from lgk.manifest import Manifest, ManifestEntry, ObjectType

logger = logging.getLogger(__name__)

LIVE_AUTH_STORAGE_OPT_IN = "LGK_LIVE_AUTH_STORAGE"


def apply_live_auth_storage(environ: Any = None) -> bool:
    """Opt-in: use the Databricks CLI's file token storage for this process and its children.

    Under many parallel live suites the OS keyring fails ("OS keyring unreachable: exit status 44").
    Only `LGK_LIVE_AUTH_STORAGE=plaintext` sets `DATABRICKS_AUTH_STORAGE=plaintext` (the CLI then keeps
    tokens in ~/.databricks/token-cache.json, 0600, outside the repo). Off by default; a
    `DATABRICKS_AUTH_STORAGE` the user already set is never overridden. Returns True if it set it.
    """
    env = os.environ if environ is None else environ
    choice = env.get(LIVE_AUTH_STORAGE_OPT_IN, "")
    if choice == "":
        return False
    if choice != "plaintext":
        raise ValueError(f"{LIVE_AUTH_STORAGE_OPT_IN} must be 'plaintext' or unset, got {choice!r}")
    if "DATABRICKS_AUTH_STORAGE" in env:
        return False
    env["DATABRICKS_AUTH_STORAGE"] = "plaintext"
    return True


_is_operation_cap = is_operation_cap  # the one implementation is lgk.branch (teardown shares it)


def _safe_branch_id(branch_name: str) -> str:
    """t-<name>-<time>-<hex>: a legal Lakebase branch id (^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$).

    Underscores etc. are illegal, so the human name is sanitised; the suffix makes it unique.
    """
    safe = re.sub(r"[^a-z0-9-]+", "-", branch_name.lower()).strip("-")
    suffix = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    return f"t-{safe}-{suffix}"[:63].rstrip("-")


class BranchLeasor:
    """Manages test branch lifecycle with automatic cleanup via manifest."""

    def __init__(
        self,
        workspace_client: Any,
        project_id: str,
        test_base_branch: str,
        run_folder: Path,
        ttl_seconds: int = 14400,  # 4 hours
    ):
        """Initialize branch leaser.

        Args:
            workspace_client: Databricks WorkspaceClient
            project_id: Lakebase project ID (e.g., "<project-name>")
            test_base_branch: Name of test-base branch to branch from
            run_folder: Path to run folder where manifest lives
            ttl_seconds: Time to live for created branch (default 4 hours)
        """
        self.workspace_client = workspace_client
        self.project_id = project_id
        self.test_base_branch = test_base_branch
        self.run_folder = Path(run_folder)
        self.ttl_seconds = ttl_seconds
        self.manifest = self._load_or_create_manifest()

    def _load_or_create_manifest(self) -> Manifest:
        """Load manifest from run folder or create new one."""
        manifest_path = self.run_folder / "manifest.json"
        if manifest_path.exists():
            return Manifest.load(manifest_path)
        else:
            return Manifest(graph_name="test_harness")

    # Concurrent branch creates (parallel live suites) get a transient "The request could not
    # be processed" (also seen on a first call, lakebase-platform-facts) or the project operation
    # cap (Aborted "too many operations in progress"). Both are retried with backoff; the branch id
    # is unique per call, so "already exists" after a transient means it is ours. Auth errors
    # (keyring exit 44) are never retried.
    CREATE_TRIES = 6
    CREATE_BACKOFF_S = 10

    def _post_branch(self, branch_name: str, branch_id: str, source_branch: str) -> dict[str, Any]:
        """Create the branch via REST (the SDK create_branch(ttl=...) is broken), retrying transients."""
        transient = False
        for attempt in range(1, self.CREATE_TRIES + 1):
            try:
                return self.workspace_client.api_client.do(
                    "POST",
                    f"/api/2.0/postgres/projects/{self.project_id}/branches",
                    query={"branch_id": branch_id},
                    body={"spec": {"source_branch": source_branch, "ttl": f"{self.ttl_seconds}s"}},
                ) or {}
            except Exception as e:  # noqa: BLE001
                text = str(e).lower()
                if transient and "already exists" in text:
                    return {"branch_id": branch_id}
                retryable = "could not be processed" in text or _is_operation_cap(e)
                if not retryable or attempt == self.CREATE_TRIES:
                    logger.error(f"Failed to create branch {branch_name}: {e}")
                    raise
                transient = True
                logger.warning(f"Branch create {branch_id} transient failure (try {attempt}): {e}")
                time.sleep(self.CREATE_BACKOFF_S * attempt)
        raise AssertionError("unreachable")

    def create_test_branch(self, branch_name: str) -> dict[str, Any]:
        """Create a child branch from test-base with TTL; record it in the manifest.

        Returns a dict with branch_id, branch_name, endpoint_name and ttl_seconds. Raises if
        the Seam-2 guard is active or branch creation fails.
        """
        if _seam2_guard_active():
            raise RuntimeError("Cannot create Lakebase branch in Seam-2 test")
        source_branch = BranchRef(self.project_id, self.test_base_branch).path
        safe_branch_id = _safe_branch_id(branch_name)
        response = self._post_branch(branch_name, safe_branch_id, source_branch)
        branch_id = response.get("branch_id", safe_branch_id)
        endpoint_name = f"{BranchRef(self.project_id, branch_id).path}/endpoints/primary"
        self._record_branch(branch_id, branch_name, endpoint_name, source_branch)
        logger.info(f"Created test branch: {branch_name}")
        return {
            "branch_id": branch_id,
            "branch_name": branch_name,
            "endpoint_name": endpoint_name,
            "ttl_seconds": self.ttl_seconds,
        }

    def _record_branch(self, branch_id: str, branch_name: str, endpoint_name: str,
                       source_branch: str) -> None:
        """Record the branch in the harness manifest, for cleanup."""
        self.manifest.add_entry(ManifestEntry(
            object_type=ObjectType.BRANCH,
            object_id=branch_id,
            name=branch_name,
            created_at=datetime.now(tz=UTC),
            location=f"lakebase:{endpoint_name}",
            metadata={"source_branch": source_branch, "ttl_seconds": self.ttl_seconds},
        ))
        self._save_manifest()

    def delete_branch(self, branch_id: str) -> bool:
        """Delete a test branch and remove from manifest.

        Args:
            branch_id: ID of branch to delete

        Returns:
            True if successful, False if branch not found
        """
        if _seam2_guard_active():
            raise RuntimeError("Cannot delete Lakebase branch in Seam-2 test")

        try:
            self._delete_with_cap_retry(branch_id)
            logger.info(f"Deleted branch: {branch_id}")

            # Remove from manifest
            self.manifest.entries = [
                e for e in self.manifest.entries if e.object_id != branch_id
            ]
            self._save_manifest()

            return True
        except Exception as e:  # noqa: BLE001
            # API call failed; log and continue cleanup
            logger.error(f"Failed to delete branch {branch_id}: {e}")
            return False

    def _delete_with_cap_retry(self, branch_id: str) -> None:
        """DELETE the branch, retrying only the project operation cap with backoff."""
        delete_branch_retrying_cap(self.workspace_client, self.project_id, branch_id,
                                   tries=self.CREATE_TRIES, backoff_s=self.CREATE_BACKOFF_S)

    def cleanup_all_test_branches(self) -> int:
        """Delete all test branches recorded in manifest.

        Returns:
            Number of branches cleaned up
        """
        deleted_count = 0
        test_branches = [e for e in self.manifest.entries if e.object_type == ObjectType.BRANCH]

        for entry in test_branches:
            if self.delete_branch(entry.object_id):
                deleted_count += 1

        return deleted_count

    def _save_manifest(self) -> None:
        """Save manifest to run folder."""
        manifest_path = self.run_folder / "manifest.json"
        self.manifest.save(manifest_path)


# Global flag for Seam-2 guard
_seam2_guard_flag = False


def _seam2_guard_active() -> bool:
    """Check if Seam-2 guard is active (no cloud calls allowed)."""
    return _seam2_guard_flag


def set_seam2_guard(active: bool) -> None:
    """Set Seam-2 guard state (for test fixtures)."""
    global _seam2_guard_flag
    _seam2_guard_flag = active


class SeamGuard:
    """Context manager to enforce Seam-2 (no-cloud) behavior."""

    def __init__(self):
        """Initialize guard."""
        self.previous_state = _seam2_guard_active()

    def __enter__(self):
        """Enable guard on entry."""
        set_seam2_guard(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous state on exit."""
        set_seam2_guard(self.previous_state)
        return False
