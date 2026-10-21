"""Offline helpers for the refresh job: a seeded laptop run folder and a fake Workspace files API."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from lgk.manifest import MANIFEST_FILE


def seed_run_folder(folder: Path, graph_name: str = "provider_graph", enriched: bool = False) -> Path:
    """The run files a provisioned graph's folder holds (contents distinct per file, so a swap shows)."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "flags.json").write_text('{"flags": [], "kind": "flags"}')
    (folder / "schema.json").write_text('{"tables": {}, "kind": "schema"}')
    (folder / "graph_mapping.yaml").write_text("nodes: {}\nedges: {}\n# mapping\n")
    (folder / "privacy_decisions.json").write_text('{"decisions": [], "kind": "decisions"}')
    (folder / MANIFEST_FILE).write_text(json.dumps({"graph_name": graph_name, "entries": []}))
    (folder / "privacy_bindings.json").write_text('{"build": {"stamp": "s1"}}')
    if enriched:
        (folder / "enrich.yaml").write_text("methods: [degree]\n")
        (folder / "enrich_report.json").write_text('{"degree": {}}')
    return folder


class FakeWorkspaceFiles:
    """w.workspace: mkdirs / upload / delete, keeping {api path: bytes} and the call log."""

    def __init__(self) -> None:
        self.files: Dict[str, bytes] = {}
        self.dirs: set = set()
        self.log: List[Any] = []

    def mkdirs(self, path: str) -> None:
        self.log.append(("mkdirs", path))
        self.dirs.add(path.rstrip("/"))

    def upload(self, path: str, content: bytes, format: Any = None, overwrite: bool = False) -> None:
        """Like the API: the parent folder must exist, and an existing file needs overwrite=True."""
        from databricks.sdk.errors import NotFound, ResourceAlreadyExists
        from databricks.sdk.service.workspace import ImportFormat

        self.log.append(("upload", path))
        if path.rsplit("/", 1)[0] not in self.dirs:
            raise NotFound(f"parent folder of {path} does not exist")
        if path in self.files and not overwrite:
            raise ResourceAlreadyExists(path)
        assert format == ImportFormat.AUTO, format  # a run file is a plain file, never a notebook source
        self.files[path] = content

    def delete(self, path: str, recursive: bool = False) -> None:
        from databricks.sdk.errors import NotFound

        self.log.append(("delete", path))
        if path not in self.files:
            raise NotFound(path)
        del self.files[path]


def fake_client(job_id: int = 42):
    """(w, jobs.create calls): a client with a recording jobs.create and a FakeWorkspaceFiles."""
    calls: List[Dict[str, Any]] = []
    w = SimpleNamespace(jobs=SimpleNamespace(create=lambda **kw: calls.append(kw) or SimpleNamespace(job_id=job_id)),
                        workspace=FakeWorkspaceFiles())
    return w, calls
