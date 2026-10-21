"""Build the kit zip: the skill, the toolkit wheel and the two workspace notebooks.

The zip keeps the repo's own layout, so every path in SKILL.md (`dist/...`, `notebooks/...`) works
from the unzipped folder:

    lgk-kit-<commit>/SKILL.md
    lgk-kit-<commit>/dist/lgk-0.1.0-py3-none-any.whl
    lgk-kit-<commit>/notebooks/lgk_runner.py
    lgk-kit-<commit>/notebooks/lgk_refresh_task.py

    uv run --no-sync python tools/package_kit.py --out /tmp

The zip is byte-reproducible: the same commit gives the same bytes on any machine.
- The wheel is built from `git archive HEAD` (not the working tree), so untracked files, local
  mtimes and local chmods never reach it; `SOURCE_DATE_EPOCH` = the commit time pins the wheel's
  entry dates, and `[build-system] requires` pins the exact hatchling.
- Every outer zip entry is written through a ZipInfo with fixed fields: date 1980-01-01 00:00:00
  (the earliest a zip can hold; fixed, so it never depends on a file's mtime or on the commit),
  mode 0o644 regular file, create_system 3 (Unix), stored uncompressed (ZIP_STORED, so no zlib
  version can change the bytes), in sorted path order.
- The wheel carries its build id: `src/lgk/_build.py` with `COMMIT = "<full sha of HEAD>"` is
  written into the archived tree before `uv build` (never into the repo; a committed copy is
  refused). The sha is a fixed input of the commit, so the bytes stay reproducible.

Refuses (exit 2) a dirty tree (the zip must be one commit), a wheel that is not the one SKILL.md
names, and any packed file that holds a personal workspace path (`/Users/<name>@`): the zip must not
hold paths to the operator's workspace.
"""

import argparse
import re
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional

REPO = Path(__file__).resolve().parent.parent
WHEEL_NAME = "lgk-0.1.0-py3-none-any.whl"
NOTEBOOKS = ("lgk_runner.py", "lgk_refresh_task.py")
PERSONAL_PATH_RE = re.compile(rb"/Users/[\w.+-]+@")
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)  # fixed: never a file mtime, never the build time
ZIP_FILE_MODE = 0o100644  # regular file, rw-r--r--, whatever the source file's mode or umask
ZIP_CREATE_SYSTEM = 3  # Unix, on every host (zipfile would write 0 on Windows)
# The build id: written into the archived tree only (never into the repo), so the wheel carries the
# commit it was built from (lgk.measure.record reads it). A fixed input of the commit: reproducible.
BUILD_MODULE = "src/lgk/_build.py"


class PackageRefused(ValueError):
    """The kit zip would not be one clean, shareable commit."""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def _export_head(repo: Path, dst: Path) -> None:
    """Write the committed tree of HEAD into `dst`: only tracked files, with the commit's content."""
    tar = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", "HEAD"],
                         capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(dst, filter="data")


def build_module_text(commit: str) -> str:
    """The text of src/lgk/_build.py for `commit` (the full sha)."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PackageRefused(f"the build id must be a full commit sha, got {commit!r}")
    return f'"""Written by tools/package_kit.py: the commit this wheel was built from."""\n\nCOMMIT = "{commit}"\n'


def _write_build_module(src: Path, commit: str) -> None:
    path = src / BUILD_MODULE
    path.write_text(build_module_text(commit))
    path.chmod(0o644)  # never the umask's mode


def _build_wheel(repo: Path, out: Path) -> Path:
    """Build the wheel from HEAD, with SOURCE_DATE_EPOCH = the commit time and the commit in lgk._build."""
    src = out / "src"
    _export_head(repo, src)
    _write_build_module(src, _git(repo, "rev-parse", "HEAD"))
    env = {**os.environ, "SOURCE_DATE_EPOCH": _git(repo, "log", "-1", "--format=%ct")}
    subprocess.run(["uv", "build", "--wheel", "-o", str(out)], cwd=src, env=env, check=True,
                   capture_output=True)
    return out / WHEEL_NAME


def _zip_entry(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_DATE_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = ZIP_CREATE_SYSTEM
    info.external_attr = ZIP_FILE_MODE << 16
    return info


def kit_members(repo: Path, wheel: Path) -> Dict[str, Path]:
    """{path inside the kit folder: source file}, in the layout SKILL.md expects."""
    if wheel.name != WHEEL_NAME:
        raise PackageRefused(f"wheel {wheel.name} is not the {WHEEL_NAME} that SKILL.md names")
    members = {"SKILL.md": repo / "skill" / "lakebase-graph-kit" / "SKILL.md", f"dist/{WHEEL_NAME}": wheel}
    members.update({f"notebooks/{n}": repo / "notebooks" / n for n in NOTEBOOKS})
    for inside, src in members.items():
        if not src.is_file():
            raise PackageRefused(f"{inside}: {src} does not exist")
        if inside.endswith(".whl"):
            with zipfile.ZipFile(src) as whl:
                for name in whl.namelist():
                    if PERSONAL_PATH_RE.search(whl.read(name)):
                        raise PackageRefused(f"{inside}: {name} holds a personal workspace path")
        elif PERSONAL_PATH_RE.search(src.read_bytes()):
            raise PackageRefused(f"{inside} holds a personal workspace path")
    return members


def package(repo: Path, out: Path, wheel: Optional[Path] = None) -> Path:
    """Write `<out>/lgk-kit-<commit>.zip` from a clean tree and return its path."""
    if _git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise PackageRefused("the tree has uncommitted changes: the kit must be one commit")
    if _git(repo, "ls-files", BUILD_MODULE):
        raise PackageRefused(f"{BUILD_MODULE} is committed: it is written at packaging time only, "
                             "so a committed copy would name a stale commit")
    commit = _git(repo, "rev-parse", "--short", "HEAD")
    with tempfile.TemporaryDirectory() as tmp:
        members = kit_members(repo, wheel or _build_wheel(repo, Path(tmp)))
        top = f"lgk-kit-{commit}"
        path = out / f"{top}.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
            for inside in sorted(members):
                z.writestr(_zip_entry(f"{top}/{inside}"), members[inside].read_bytes())
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        print(package(REPO, args.out))
    except PackageRefused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
