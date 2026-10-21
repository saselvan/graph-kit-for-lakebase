"""The run's Lakebase branch: one reference type and one public Postgres connector.

`BranchRef.from_config` is the only place that reads `lakebase_project` / `branch_id`; every
branch path and every branch connection goes through it. There is no default branch (owner
"branch a"): a missing, empty or blank branch_id is refused naming the key, so no step can land
on `production` because a config forgot to say which branch.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


class MissingBranchError(ValueError):
    """The config names no branch: the kit never picks one (no production fallback)."""


@dataclass(frozen=True)
class BranchRef:
    project: str
    branch: str  # no default anywhere: from_config refuses a config without one

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BranchRef":
        branch = config.get("branch_id")
        if not isinstance(branch, str) or not branch.strip():
            raise MissingBranchError(
                f"config['branch_id'] is required (got {branch!r}): name the Lakebase branch to use; "
                "the kit never falls back to a default branch")
        return cls(config["lakebase_project"], branch)

    @property
    def path(self) -> str:
        """projects/<project>/branches/<branch> (the API resource name)."""
        return f"projects/{self.project}/branches/{self.branch}"

    @property
    def pg_location(self) -> str:
        """The manifest location of a Postgres object on this branch."""
        return f"postgres:{self.path}"


def connect_branch(config: Dict[str, Any]) -> Any:
    """Open a psycopg connection to the run's branch Postgres, as the operator.

    Lakebase pattern (verified live): host = the branch's primary endpoint host,
    user = the operator's Databricks identity, password = a fresh database credential.
    Not the workspace host, not user "token" — those fail "External authorization failed".
    The one implementation lives in lgk.walks.nb_runtime.connect, which the walks notebook
    carries verbatim.
    """
    from lgk.walks.nb_runtime import connect

    ref = BranchRef.from_config(config)
    return connect(ref.project, ref.branch)


# The per-project cap on concurrent control-plane operations (lakebase-platform-facts, 2026-10-04).
CAP_TRIES = 6
CAP_BACKOFF_S = 10


def is_operation_cap(error: Exception) -> bool:
    """Aborted "too many operations in progress": the project's operation cap (retryable)."""
    from databricks.sdk.errors.platform import Aborted

    return isinstance(error, Aborted) and "too many operations in progress" in str(error).lower()


def delete_branch_retrying_cap(w: Any, project: str, branch_id: str, tries: int = CAP_TRIES,
                               backoff_s: float = CAP_BACKOFF_S) -> None:
    """DELETE a branch over REST, retrying only the operation cap (backoff_s x try); other errors raise."""
    for attempt in range(1, tries + 1):
        try:
            w.api_client.do("DELETE", f"/api/2.0/postgres/projects/{project}/branches/{branch_id}")
            return
        except Exception as e:  # noqa: BLE001
            if not is_operation_cap(e) or attempt == tries:
                raise
            logger.warning(f"Branch delete {branch_id} hit the operation cap (try {attempt}): {e}")
            time.sleep(backoff_s * attempt)
