"""Run a generated Databricks source-format notebook locally, cell by cell, with a dbutils stub.

Like the parity LocalProcessBackend, `# MAGIC` lines (markdown, %pip) are dropped and the rest
of each cell is executed as Python. Cells share one namespace, so a single cell can be re-run
after the others, exactly as an engineer re-runs a cell in the workspace.
"""

import json
from typing import Any, Dict, List, Optional

SEPARATOR = "# COMMAND ----------"
HEADER = "# Databricks notebook source"


class NotebookExit(Exception):
    """dbutils.notebook.exit was called."""


class _Widgets:
    def __init__(self, overrides: Dict[str, str]):
        self.overrides, self.values = dict(overrides), {}

    def text(self, name: str, default: str, label: Optional[str] = None) -> None:
        self.values[name] = self.overrides.get(name, default)

    def get(self, name: str) -> str:
        return self.values[name]


class _NotebookUtils:
    def __init__(self):
        self.exit_value: Optional[str] = None

    def exit(self, value: str) -> None:
        self.exit_value = value
        raise NotebookExit()


class DbutilsStub:
    def __init__(self, widgets: Optional[Dict[str, str]] = None):
        self.widgets = _Widgets(widgets or {})
        self.notebook = _NotebookUtils()


def split_cells(source: str) -> List[str]:
    lines = source.splitlines()
    assert lines[0] == HEADER, f"not a Databricks source notebook: {lines[0]!r}"
    return [c.strip("\n") for c in "\n".join(lines[1:]).split(SEPARATOR)]


def _python(cell: str) -> str:
    return "\n".join(line for line in cell.splitlines() if not line.startswith("# MAGIC"))


class LocalNotebook:
    def __init__(self, source: str, widgets: Optional[Dict[str, str]] = None):
        self.cells = split_cells(source)
        self.dbutils = DbutilsStub(widgets)
        self.ns: Dict[str, Any] = {"dbutils": self.dbutils, "__name__": "__main__"}

    def run_cell(self, i: int) -> None:
        exec(compile(_python(self.cells[i]), f"<cell {i}>", "exec"), self.ns)

    def walk_cells(self) -> List[int]:
        return [i for i, c in enumerate(self.cells) if "= walk_cell(conn," in c]

    def run_until_last(self) -> None:
        """Every cell but the last (which closes the connection and exits)."""
        for i in range(len(self.cells) - 1):
            self.run_cell(i)

    def finish(self) -> Dict[str, Any]:
        try:
            self.run_cell(len(self.cells) - 1)
        except NotebookExit:
            pass
        assert self.dbutils.notebook.exit_value is not None, "notebook never called exit"
        return json.loads(self.dbutils.notebook.exit_value)

    def run_all(self) -> Dict[str, Any]:
        self.run_until_last()
        return self.finish()
