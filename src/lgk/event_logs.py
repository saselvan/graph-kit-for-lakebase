"""The sync pipelines' event-log tables: which ones are ours, and dropping them (teardown and retire).

Each sync pipeline writes a MANAGED Delta table `event_log_<pipeline id, '-' -> '_'>` into its
synced table's DESTINATION UC schema. Those tables are ours only through the
pipeline id of a synced table this graph recorded; a name pattern (`event_log_%`) is never
enough, so another graph's or pipeline's event log in the same schema stays foreign.

The pipeline id is the one the run manifest recorded, else the one Unity Catalog holds for the
synced table: `tables.get(<synced table>).pipeline_id`, a TableInfo field (live, 2026-10-04: it
is NOT in `.properties`, which hold source_table, database_branch_id, ...; tables.get still
answers after the branch is gone. With neither, the caller refuses before deleting anything.
"""

import logging
from typing import Any, List, Mapping, Optional

from lgk import manifest as mf
from lgk.versions import NotOursError

logger = logging.getLogger(__name__)


class UnknownPipelineError(NotOursError):
    """A recorded synced table's pipeline id is not recorded and cannot be looked up."""


def event_log_table(synced_table: str, pipeline_id: str) -> str:
    """`<cat>.<dest schema>.event_log_<pipeline id with '_' for '-'>`."""
    dest_schema = synced_table.rsplit(".", 1)[0]
    return f"{dest_schema}.event_log_{pipeline_id.replace('-', '_')}"


def pipeline_id(w: Any, synced_table: str, recorded: Optional[str]) -> Optional[str]:
    """The synced table's pipeline id: the recorded one, else Unity Catalog's.

    None only when the synced table AND its destination UC schema are gone (no event log can be
    left there). Raises UnknownPipelineError when the schema exists but the id cannot be found.
    """
    if recorded:
        return str(recorded)
    from databricks.sdk.errors import NotFound

    try:
        info = w.tables.get(synced_table)
    except NotFound:
        if not mf.uc_schema_comment(w, synced_table.rsplit(".", 1)[0])[0]:
            logger.info(f"Synced table {synced_table} and its UC schema are gone; no event log to drop")
            return None
        raise UnknownPipelineError(
            f"Synced table {synced_table}: the run manifest records no pipeline_id and Unity Catalog "
            f"no longer has the table, so its sync event log cannot be told from a stranger's; refusing "
            f"to delete anything. Add its pipeline_id to lgk_manifest.json, or drop the event log by hand.")
    found = getattr(info, "pipeline_id", None)
    if not found:
        raise UnknownPipelineError(
            f"Synced table {synced_table}: the run manifest records no pipeline_id and Unity Catalog "
            f"names none, so its sync event log cannot be told from a stranger's; refusing to delete anything.")
    logger.info(f"Synced table {synced_table}: pipeline_id {found} read from Unity Catalog (not recorded)")
    return str(found)


def own_event_logs(w: Any, synced: Mapping[str, Optional[str]]) -> List[str]:
    """One event-log table per pipeline of `synced` (synced table -> recorded pipeline id), in order."""
    found = []
    for table, recorded in synced.items():
        pid = pipeline_id(w, table, recorded)
        if pid is not None:
            found.append(event_log_table(table, pid))
    return list(dict.fromkeys(found))


def drop_event_logs(w: Any, warehouse: str, tables: List[str]) -> None:
    """After the pipelines are gone: exactly these event-log tables, and no other."""
    from lgk.uc_sql import execute_sql

    for table in tables:
        execute_sql(w, warehouse, f"DROP TABLE IF EXISTS {table}")
    logger.info(f"Dropped sync pipeline event logs {tables}")
