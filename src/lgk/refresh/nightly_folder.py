"""The nightly job's run folder (owner "bugs a"): a workspace copy of the run files its tasks read.

The refresh job runs serverless, where an operator's laptop path does not exist. So when the
config's run_folder is not a /Workspace path, create_refresh_job copies the files the nightly
tasks read into a workspace folder and points the job's config at it. A /Workspace run_folder is
used as is. The caller's config is never changed.

What the tasks read and write (lgk.refresh.tasks: build, stamp, enrich, sync, reapply):
- every task: flags.json, schema.json, graph_mapping.yaml (the privacy gate, require_clear,
  hashes all three; it also reads privacy_decisions.json, the privacy step's record of past
  decisions, and refuses a decision that is not the newest one there; build reads schema.json again for the join types; sync, stamp and reapply
  read the mapping for the floor, allowlist, walk-key indexes and served functions);
- lgk_manifest.json: read by sync (table labels, connection high water, synced tables via
  freshness.synced_pairs) and reapply; written by build/stamp (labels, UC records), enrich (UC
  records), sync (high water) and reapply (served schema, views, functions);
- enrich.yaml (enrich's choice, read by enrich), enrich_report.json (written by enrich; read by
  sync and reapply for the enrichment tables served), privacy_bindings.json (read-modify-written
  by build and enrich): copied when present.
- key_decisions.json (lgk.keys: the operator's orphan decisions, read and appended by build):
  copied when present, and REQUIRED when schema.json records an FK with orphans (profile saw
  them, so the nightly build needs the same decisions). It is kept out of REQUIRED_FILES and
  OPTIONAL_FILES (KEY_FILES): it is required only in that case.
Nothing else is copied (an app folder, a bundle, notebooks, run.json stay on the laptop).

State rule: the workspace copy is the run folder of record for the nightly chain (its labels,
high water and bindings move night by night there). The laptop folder stays the record for the
operator's steps: the nightly tasks re-record only objects provision and switch already
recorded (same ids; manifest.record is a no-op for a known id), so teardown from the laptop sees
every object. Creating the job again re-seeds the workspace copy from the laptop. No two-way sync.
"""

from pathlib import Path, PurePosixPath
from typing import Any, Dict, List

from lgk.keys import KEY_DECISIONS_FILE
from lgk.manifest import MANIFEST_FILE
from lgk.notebook_runner import _api_path
from lgk.privacy.decisions import DECISIONS_FILE

WORKSPACE_PREFIX = "/Workspace/"
REQUIRED_FILES = ("flags.json", "schema.json", "graph_mapping.yaml", DECISIONS_FILE, MANIFEST_FILE)
OPTIONAL_FILES = ("enrich.yaml", "enrich_report.json", "privacy_bindings.json")
KEY_FILES = (KEY_DECISIONS_FILE,)  # required when schema.json records orphans, else copied when present


class NightlyFolderError(ValueError):
    """The run folder cannot seed the nightly job (no folder, or a file the tasks read is missing)."""


def is_workspace_path(path: Any) -> bool:
    return str(path).startswith(WORKSPACE_PREFIX)


def _fuse_path(path: str) -> str:
    """The /Workspace/... form of a workspace path given either way."""
    return path if is_workspace_path(path) else f"/Workspace{path}"


def workspace_run_folder(config: Dict[str, Any], notebook_path: str) -> str:
    """config['workspace_run_folder'], else <notebook dir>/runs/<graph_name>; always a /Workspace path."""
    chosen = config.get("workspace_run_folder")
    if chosen is None:
        chosen = f"{_fuse_path(str(PurePosixPath(notebook_path).parent))}/runs/{config['graph_name']}"
    if not is_workspace_path(chosen) or chosen.rstrip("/") == WORKSPACE_PREFIX.rstrip("/"):
        raise NightlyFolderError(f"workspace_run_folder must be a folder under /Workspace/, got {chosen!r}")
    return chosen.rstrip("/")


def nightly_files(run_folder: Any) -> List[str]:
    """The run files to copy: every required one (else NightlyFolderError naming each) + optional present."""
    if not run_folder:
        raise NightlyFolderError("config['run_folder'] is required: the nightly tasks read the run files there")
    folder = Path(run_folder)
    missing = [name for name in REQUIRED_FILES if not (folder / name).is_file()]
    if missing:
        raise NightlyFolderError(
            f"{folder} lacks {', '.join(missing)}: the nightly tasks read them; run the steps that "
            "write them (profile, propose, privacy, build, provision) before creating the refresh job")
    has_keys = (folder / KEY_DECISIONS_FILE).is_file()
    if not has_keys and _profiled_orphans(folder / "schema.json"):
        raise NightlyFolderError(
            f"{folder} lacks {KEY_DECISIONS_FILE}, but schema.json records foreign keys with orphan rows: "
            "the nightly build needs the operator's decisions (lgk.keys.resolve_orphans); run profile "
            "and build again in this run folder before creating the refresh job")
    return (list(REQUIRED_FILES) + [name for name in OPTIONAL_FILES if (folder / name).is_file()]
            + ([KEY_DECISIONS_FILE] if has_keys else []))


def _profiled_orphans(schema_file: Path) -> bool:
    """True when schema.json records any foreign key with orphan_rows > 0 (a non-count is True: fail closed)."""
    import json

    tables = json.loads(schema_file.read_text()).get("tables", [])
    for table in tables:
        for fk in table.get("foreign_keys", []):
            n = fk.get("orphan_rows", 0)
            if not (isinstance(n, int) and not isinstance(n, bool) and n == 0):
                return True
    return False


def upload_run_files(w: Any, run_folder: Any, names: List[str], dest: str) -> None:
    """Copy `names` from the local run folder to `dest`; an optional file absent locally is removed there.

    Removing it keeps a re-seeded copy from serving a stale enrich_report.json or enrich.yaml.
    """
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.workspace import ImportFormat

    w.workspace.mkdirs(_api_path(dest))
    for name in names:
        w.workspace.upload(_api_path(f"{dest}/{name}"), (Path(run_folder) / name).read_bytes(),
                           format=ImportFormat.AUTO, overwrite=True)
    for name in OPTIONAL_FILES + KEY_FILES:
        if name not in names:
            try:
                w.workspace.delete(_api_path(f"{dest}/{name}"))
            except NotFound:
                pass
