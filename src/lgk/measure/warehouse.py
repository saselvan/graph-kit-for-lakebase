"""Every row of a warehouse statement, or an error, through the Databricks SQL connector.

`lgk.uc_sql.execute_sql` returns the first result chunk only, which is enough for the kit's
counts but would silently truncate the reference's id pulls. The Statement Execution API's INLINE
result is capped at 25 MiB, which the reference passes at 10^6 nodes (live, 2026-10-06: "Inline
byte limit exceeded"). The SQL connector (a kit dependency) pages any size of result itself, so
the kit imports no network module. Its telemetry is turned off explicitly.

Values come back as the Statement Execution API's JSON_ARRAY gave them: text, `true`/`false`
for booleans, None for NULL, so the reference reads the same rows either way.
"""

from typing import Any, Callable, List, Optional

from lgk.uc_sql import StatementFailedError

BATCH_ROWS = 100_000


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def fetch_all(w: Any, warehouse_id: str, statement: str,
              connect: Optional[Callable[..., Any]] = None) -> List[List[Optional[str]]]:
    """All rows of `statement` on `warehouse_id`, as the identity of `w`; StatementFailedError otherwise."""
    if connect is None:
        from databricks import sql

        connect = sql.connect
    host = str(w.config.host).split("://", 1)[-1].rstrip("/")
    try:
        conn = connect(server_hostname=host, http_path=f"/sql/1.0/warehouses/{warehouse_id}",
                       credentials_provider=lambda: w.config.authenticate, enable_telemetry=False)
    except Exception as e:
        raise StatementFailedError(f"measure statement FAILED to connect: {type(e).__name__}: {str(e)[:300]}") from e
    rows: List[List[Optional[str]]] = []
    try:
        cursor = conn.cursor()
        cursor.execute(statement)
        while True:
            batch = cursor.fetchmany(BATCH_ROWS)
            if not batch:
                break
            rows.extend([_text(v) for v in row] for row in batch)
        cursor.close()
    except Exception as e:
        # the message, never the statement text: it names source tables only, but keep it short
        raise StatementFailedError(f"measure statement FAILED: {type(e).__name__}: {str(e)[:300]}") from e
    finally:
        conn.close()
    return rows
