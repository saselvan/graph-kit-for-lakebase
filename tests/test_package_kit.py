"""tools/package_kit.py: the kit zip holds exactly the skill, the wheel and the two notebooks, in the
layout SKILL.md's paths expect, from one clean commit, with no personal workspace path."""

import importlib.util
import os
import re
import shutil
import time
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("package_kit", REPO / "tools" / "package_kit.py")
pk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pk)


def _repo(tmp_path: Path, skill: bytes = b"# skill\nrun from /Workspace/Users/<operator>/lgk\n") -> Path:
    repo = tmp_path / "repo"
    (repo / "skill" / "lakebase-graph-kit").mkdir(parents=True)
    (repo / "notebooks").mkdir()
    (repo / "skill" / "lakebase-graph-kit" / "SKILL.md").write_bytes(skill)
    for n in pk.NOTEBOOKS:
        (repo / "notebooks" / n).write_text(f"# {n}\n")
    (repo / "notebooks" / "extra.py").write_text("# not shipped\n")  # hostile: a file the kit must leave out
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@example.com", "-c", "user.name=t",
                    "commit", "-q", "--no-verify", "-m", "fixture"], check=True)
    return repo


def _wheel(tmp_path: Path, body: bytes = b"x = 1\n", name: str = pk.WHEEL_NAME) -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("lgk/__init__.py", body)
    return path


def test_the_kit_holds_exactly_the_skill_wheel_and_notebooks_in_the_skill_layout(tmp_path):
    repo, wheel = _repo(tmp_path), _wheel(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    path = pk.package(repo, out, wheel)
    commit = pk._git(repo, "rev-parse", "--short", "HEAD")
    assert path.name == f"lgk-kit-{commit}.zip"
    with zipfile.ZipFile(path) as z:
        names = sorted(z.namelist())
        top = f"lgk-kit-{commit}"
        assert names == sorted([f"{top}/SKILL.md", f"{top}/dist/{pk.WHEEL_NAME}",
                                f"{top}/notebooks/lgk_runner.py", f"{top}/notebooks/lgk_refresh_task.py"])
        assert z.read(f"{top}/SKILL.md") == (repo / "skill/lakebase-graph-kit/SKILL.md").read_bytes()
        assert z.read(f"{top}/dist/{pk.WHEEL_NAME}") == wheel.read_bytes()


def test_a_dirty_tree_is_refused(tmp_path):
    repo = _repo(tmp_path)
    (repo / "notebooks" / "lgk_runner.py").write_text("# edited, not committed\n")
    with pytest.raises(pk.PackageRefused, match="uncommitted"):
        pk.package(repo, tmp_path, _wheel(tmp_path))


def test_a_personal_path_in_the_skill_is_refused(tmp_path):
    repo = _repo(tmp_path, skill=b"# skill\nsee /" + b"Users/someone.name@" + b"databricks.com/lgk\n")
    with pytest.raises(pk.PackageRefused, match="SKILL.md holds a personal workspace path"):
        pk.package(repo, tmp_path, _wheel(tmp_path))


def test_a_personal_path_inside_the_wheel_is_refused(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(pk.PackageRefused, match="personal workspace path"):
        pk.package(repo, tmp_path, _wheel(tmp_path, body=b'P = "/' + b'Users/a.b@example.com/x"\n'))


def test_a_wheel_the_skill_does_not_name_is_refused(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(pk.PackageRefused, match="SKILL.md names"):
        pk.package(repo, tmp_path, _wheel(tmp_path, name="lgk-0.2.0-py3-none-any.whl"))


def test_the_skill_names_the_wheel_and_notebooks_the_kit_ships():
    text = (REPO / "skill" / "lakebase-graph-kit" / "SKILL.md").read_text()
    assert f"dist/{pk.WHEEL_NAME}" in text
    for n in pk.NOTEBOOKS:
        assert f"notebooks/{n}" in text
    assert all((REPO / "notebooks" / n).is_file() for n in pk.NOTEBOOKS)


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build the real wheel")
def test_the_real_wheel_holds_no_personal_path(tmp_path):
    wheel = pk._build_wheel(REPO, tmp_path)
    pk.kit_members(REPO, wheel)  # raises on a personal path in the skill, the notebooks or the wheel


# ------------------------------------------------------------------ byte-reproducible zip

def _noise(repo: Path, wheel: Path, stamp: int, mode: int) -> None:
    """Hostile: new mtimes and modes on every packed source (and an untracked file), between builds."""
    srcs = [repo / "skill/lakebase-graph-kit/SKILL.md", wheel] + [repo / "notebooks" / n for n in pk.NOTEBOOKS]
    for src in srcs:
        os.utime(src, (stamp, stamp))
        src.chmod(mode)
    (repo / "notebooks" / "untracked.py").write_text(f"# {stamp}\n")


def _fixture_ignoring_modes(tmp_path: Path) -> Path:
    repo = _repo(tmp_path)
    # A chmod is noise here, not an edit: git must not call the tree dirty for it.
    subprocess.run(["git", "-C", str(repo), "config", "core.fileMode", "false"], check=True)
    return repo


@pytest.mark.seam2
def test_two_builds_are_byte_identical_despite_mtimes_modes_and_umask(tmp_path):
    repo, wheel = _fixture_ignoring_modes(tmp_path), _wheel(tmp_path)
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    out1.mkdir(), out2.mkdir()
    _noise(repo, wheel, 1_600_000_000, 0o644)
    first = pk.package(repo, out1, wheel).read_bytes()
    _noise(repo, wheel, 1_900_000_000, 0o755)
    old = os.umask(0o077)
    try:
        second = pk.package(repo, out2, wheel).read_bytes()
    finally:
        os.umask(old)
    assert first == second


@pytest.mark.seam2
def test_every_entry_has_fixed_date_mode_system_storage_and_order(tmp_path):
    repo, wheel = _fixture_ignoring_modes(tmp_path), _wheel(tmp_path)
    _noise(repo, wheel, 1_900_000_000, 0o755)
    with zipfile.ZipFile(pk.package(repo, tmp_path, wheel)) as z:
        infos = z.infolist()
    names = [i.filename for i in infos]
    assert names == sorted(names) and len(names) == 4
    for i in infos:
        assert i.date_time == (1980, 1, 1, 0, 0, 0), i
        assert i.external_attr >> 16 == 0o100644, (i.filename, oct(i.external_attr >> 16))
        assert i.create_system == 3, i
        assert i.compress_type == zipfile.ZIP_STORED, i


def _clone_head(tmp_path: Path, name: str) -> Path:
    clone = tmp_path / name
    subprocess.run(["git", "clone", "-q", f"file://{REPO}", str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "config", "core.fileMode", "false"], check=True)
    return clone


@pytest.mark.seam2
def test_the_real_kit_is_byte_identical_across_two_builds_with_local_noise(tmp_path):
    # The real wheel build (uv build + the pinned hatchling) from a clone of this repo's HEAD.
    clone = _clone_head(tmp_path, "clone")
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    out1.mkdir(), out2.mkdir()
    first = pk.package(clone, out1)
    # hostile, in the working tree only: touched mtimes, an exec bit, an untracked module.
    for f in (clone / "src" / "lgk").rglob("*.py"):
        os.utime(f, (1_900_000_000, 1_900_000_000))
    (clone / "src/lgk/advisor.py").chmod(0o755)
    (clone / "src/lgk/zz_untracked.py").write_text("X = 1\n")
    second = pk.package(clone, out2)
    assert first.read_bytes() == second.read_bytes()

    commit_time = int(pk._git(clone, "log", "-1", "--format=%ct"))
    pin = re.search(r'requires = \["hatchling==([\d.]+)"\]', (clone / "pyproject.toml").read_text())
    assert pin, "hatchling is not pinned to one exact version"
    with zipfile.ZipFile(second) as kit:
        whl_name = next(n for n in kit.namelist() if n.endswith(".whl"))
        whl_bytes = kit.read(whl_name)
    whl_path = tmp_path / "w.whl"
    whl_path.write_bytes(whl_bytes)
    with zipfile.ZipFile(whl_path) as whl:
        assert f"Generator: hatchling {pin.group(1)}" in whl.read("lgk-0.1.0.dist-info/WHEEL").decode()
        infos = {i.filename: i for i in whl.infolist()}
    assert "lgk/zz_untracked.py" not in infos
    assert infos["lgk/advisor.py"].external_attr >> 16 == 0o100644
    # A zip date holds even seconds only (DOS time), so compare against the commit second floored to 2.
    assert infos["lgk/advisor.py"].date_time == time.gmtime(commit_time - commit_time % 2)[:6]


# ------------------------------------------------------------------ the build id in the wheel

FULL_SHA = "0123456789abcdef0123456789abcdef01234567"


def test_the_build_module_text_carries_the_full_commit_and_refuses_a_short_one():
    ns: dict = {}
    exec(pk.build_module_text(FULL_SHA), ns)
    assert ns["COMMIT"] == FULL_SHA
    with pytest.raises(pk.PackageRefused, match="full commit sha"):
        pk.build_module_text(FULL_SHA[:7])  # hostile: the short sha the zip name uses


def test_the_repo_never_tracks_the_build_module():
    assert pk._git(REPO, "ls-files", pk.BUILD_MODULE) == ""


def test_a_committed_build_module_is_refused(tmp_path):
    repo = _repo(tmp_path)
    (repo / "src" / "lgk").mkdir(parents=True)
    (repo / pk.BUILD_MODULE).write_text(f'COMMIT = "{FULL_SHA}"\n')  # hostile: a stale id in git
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@example.com", "-c", "user.name=t",
                    "commit", "-q", "-m", "stale build id"], check=True)
    with pytest.raises(pk.PackageRefused, match="stale commit"):
        pk.package(repo, tmp_path, _wheel(tmp_path))


def _record_build(monkeypatch, module) -> dict:
    import sys

    from lgk.measure import record

    monkeypatch.delitem(sys.modules, "lgk._build", raising=False)
    if module is not None:
        monkeypatch.setitem(sys.modules, "lgk._build", module)
    return record.build_identity()


def test_record_reads_the_packaged_commit_first(monkeypatch):
    import types

    head = pk._git(REPO, "rev-parse", "HEAD")
    b = _record_build(monkeypatch, types.SimpleNamespace(COMMIT=FULL_SHA))
    assert FULL_SHA != head and b["commit"] == FULL_SHA


def test_a_checkout_without_the_build_module_records_git_head(monkeypatch):
    import types

    head = pk._git(REPO, "rev-parse", "HEAD")
    assert not (REPO / pk.BUILD_MODULE).exists()
    assert _record_build(monkeypatch, None)["commit"] == head
    # hostile: a build module that holds no full sha is not a build id
    assert _record_build(monkeypatch, types.SimpleNamespace(COMMIT="abc1234"))["commit"] == head


@pytest.mark.seam2
@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build the real wheel")
def test_the_packaged_wheel_carries_the_commit_and_measure_records_it(tmp_path):
    import sys

    clone = _clone_head(tmp_path, "clone")
    head = pk._git(clone, "rev-parse", "HEAD")
    with zipfile.ZipFile(pk.package(clone, tmp_path)) as kit:
        whl_bytes = kit.read(next(n for n in kit.namelist() if n.endswith(".whl")))
    (tmp_path / "w.whl").write_bytes(whl_bytes)
    site = tmp_path / "site"
    with zipfile.ZipFile(tmp_path / "w.whl") as whl:
        ns: dict = {}
        exec(whl.read("lgk/_build.py").decode(), ns)
        whl.extractall(site)
    assert ns["COMMIT"] == head
    # the installed wheel (no .git near it) records the packaged commit
    out = subprocess.run([sys.executable, "-c", "from lgk.measure import record; "
                          "print(record.build_identity()['commit'])"],
                         cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(site)},
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == head
