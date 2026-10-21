"""Notebook step: write the data-engineer walks notebook to the run folder."""

from typing import Any, Dict

from lgk.walks.notebook import write_notebook


def notebook(config: Dict[str, Any]) -> str:
    """Write `<run_folder>/walks_notebook.py` (refused unless the privacy gate is clear)."""
    path = write_notebook(config)
    print(f"walks notebook: {path}")
    return str(path)
