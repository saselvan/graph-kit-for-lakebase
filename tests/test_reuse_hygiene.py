"""No organisation names or hardcoded schemas in src (lgk.hashing and its tests were removed:
unused, and its ids never matched Spark's xxhash64)."""

import re
from pathlib import Path

import pytest



# A personal catalog is a string literal shaped <user>_<workspace>_catalog. Catalogs come from config.
PERSONAL_CATALOG = re.compile(r"""["'][a-z0-9]+_[a-z0-9]+(?:_[a-z0-9]+)*_catalog["']""")
# The real organisation code and schema are on the owner's private deny-list (owner ruling 2026-10-15), so the export
# gate refuses them in every shipping file. This file names neither: its values are synthetic, built at run time.
ORG_CODE = "zq" + "x"
ORG_SCHEMA = "_".join((ORG_CODE, "dm", "test"))
FORBIDDEN = (ORG_SCHEMA, ORG_CODE)


def src_hits(src_dir, values, pattern):
    """Path (from src_dir) of every .py file that holds one of `values` or a `pattern` match, in path order."""
    out = []
    for py_file in sorted(src_dir.rglob("*.py")):
        content = py_file.read_text()
        if any(v in content for v in values) or pattern.search(content):
            out.append(py_file.relative_to(src_dir).as_posix())
    return out


class TestReusedCodeHygiene:
    """Test that reused code contains no organisation or personal names."""

    @pytest.fixture
    def repo_root(self):
        """Return the repo root."""
        return Path(__file__).parent.parent

    def test_all_src_files_no_hardcoded_schemas(self, repo_root):
        """No .py under src/lgk holds a personal catalog name or an organisation value; both come from config."""
        assert src_hits(repo_root / "src" / "lgk", FORBIDDEN, PERSONAL_CATALOG) == []

    def test_a_planted_value_or_catalog_is_a_hit(self, tmp_path):
        # hostile: the scan must find each kind; a scan that reads no file, or checks only one kind, passes the test
        # above on a clean tree and fails here
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "a.py").write_text(f'SCHEMA = "{ORG_SCHEMA}"\n')
        (tmp_path / "b.py").write_text("CATALOG = '" + "_".join(("jdoe", "ws1", "catalog")) + "'\n")
        (tmp_path / "c.py").write_text(f"# the {ORG_CODE} team\n")
        (tmp_path / "d.py").write_text('catalog = config["catalog"]\n')  # near miss: a name from config
        assert src_hits(tmp_path, FORBIDDEN, PERSONAL_CATALOG) == ["b.py", "c.py", "pkg/a.py"]
