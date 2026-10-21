"""Run a UC SQL statement on a warehouse and fail loudly.

The statement-execution API does not raise when a statement fails, and with a
wait_timeout it returns while the statement is still running. This helper waits
until the statement reaches a terminal state and raises unless it SUCCEEDED.
"""

import time
from typing import Any, Dict, List

_RUNNING = {"PENDING", "RUNNING"}


class MissingWarehouseError(ValueError):
    """The run config names no SQL warehouse (there is no default warehouse)."""


def warehouse_id(config: Dict[str, Any]) -> str:
    """The run's SQL warehouse id, from `config["warehouse_id"]`. Raises if missing or empty."""
    value = config.get("warehouse_id")
    if not isinstance(value, str) or not value.strip():
        raise MissingWarehouseError(
            f"config has no warehouse_id (got {value!r}): set warehouse_id to the id of the SQL "
            f"warehouse the kit runs UC statements on; there is no default warehouse")
    return value


class StatementFailedError(RuntimeError):
    """A UC statement ended FAILED, CANCELED or CLOSED (or never finished)."""


def _state(resp: Any) -> str:
    state = resp.status.state
    return str(getattr(state, "value", state))


def execute_sql(
    w: Any, warehouse_id: str, statement: str, timeout_s: int = 900, poll_s: float = 2.0
) -> List[List[Any]]:
    """Execute `statement`, wait until it finishes, and return its rows (data_array).

    Raises StatementFailedError on FAILED/CANCELED/CLOSED, or if it is still running
    after `timeout_s`.
    """
    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id, statement=statement, wait_timeout="50s"
    )
    deadline = time.monotonic() + timeout_s
    while _state(resp) in _RUNNING:
        if time.monotonic() > deadline:
            raise StatementFailedError(
                f"Statement still {_state(resp)} after {timeout_s}s: {statement[:200]}"
            )
        time.sleep(poll_s)
        resp = w.statement_execution.get_statement(resp.statement_id)

    state = _state(resp)
    if state != "SUCCEEDED":
        err = getattr(resp.status, "error", None)
        msg = getattr(err, "message", None) or err
        raise StatementFailedError(f"Statement {state}: {msg} :: {statement[:200]}")
    result = getattr(resp, "result", None)
    return (getattr(result, "data_array", None) or []) if result else []
