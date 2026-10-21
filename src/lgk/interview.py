"""Install the committed interviewer skill into both skill hosts.

The skill lives in one committed folder, `skill/lakebase-graph-kit/` (its `SKILL.md` is the single
source of truth; nothing here generates it). Both hosts read the Agent Skills open standard, so
the same bytes go to both discovery paths:

- `<claude_home>/skills/<name>/SKILL.md`             (claude_home is normally `~/.claude`)
- `<genie_home>/.assistant/skills/<name>/SKILL.md`   (genie_home is a workspace user folder)
"""

import shutil
from pathlib import Path

SKILL_NAME = "lakebase-graph-kit"
SKILL_FILE = Path(__file__).resolve().parents[2] / "skill" / SKILL_NAME / "SKILL.md"


def install_skill_to_both_hosts(
    claude_home: Path, genie_home: Path, skill_file: Path = SKILL_FILE
) -> tuple[Path, Path]:
    """Copy `skill_file` byte-for-byte into both hosts' skill folders; return the two copies.

    The skill folder's name is the skill name. Existing copies are overwritten.
    """
    skill_file = Path(skill_file)
    name = skill_file.parent.name
    targets = (
        Path(claude_home) / "skills" / name / "SKILL.md",
        Path(genie_home) / ".assistant" / "skills" / name / "SKILL.md",
    )
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(skill_file, target)
    return targets
