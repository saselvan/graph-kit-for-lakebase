"""The built wheel declares its runtime dependencies: every step imports databricks.sdk, so a wheel
installed without the dev extra must pull it in."""

import re
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _requires_dist(wheel: Path) -> list:
    with zipfile.ZipFile(wheel) as z:
        meta = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        text = z.read(meta).decode()
    return re.findall(r"^Requires-Dist: (.+)$", text, flags=re.M)


@pytest.mark.seam2
def test_the_wheel_requires_databricks_sdk_at_runtime(tmp_path):
    subprocess.run(["uv", "build", "--wheel", "-o", str(tmp_path)], cwd=REPO, check=True,
                   capture_output=True)
    (wheel,) = tmp_path.glob("lgk-*.whl")
    runtime = [r for r in _requires_dist(wheel) if "extra ==" not in r]
    # The reference is written here, not read from pyproject.toml: the check must not follow the code.
    assert any(re.fullmatch(r"databricks-sdk\s*>=\s*0\.81(\.0)?", r.strip()) for r in runtime), runtime
