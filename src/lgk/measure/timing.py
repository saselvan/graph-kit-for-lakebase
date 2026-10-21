"""Client-side latency for walks on a warm connection: in-region timing from a job.

The measure step times each (shape, depth) walk with:
1. Warmup runs (not timed, not counted).
2. Samples client-timed walks (time.perf_counter before execute, after fetchall), in ms.
3. Samples SELECT 1 round-trip times, in ms.
4. Samples EXPLAIN (ANALYZE, TIMING OFF) runs: planning and execution times (ms).

The breakdown: client mean = SELECT 1 mean + planning mean + execution mean + remainder.
The remainder is "result transfer and client overhead"; a remainder below -(5% of the client mean
+ 2 combined standard errors of the four means) is refused, as is an execution p95 above the client p95.

The exact label is "Postgres execution time (EXPLAIN ANALYZE); excludes planning, result transfer
and network" (C13).
"""

import json
import math
import time
from typing import Any, Callable, Dict, List, Optional

from lgk import advisor

EXECUTION_LABEL = "Postgres execution time (EXPLAIN ANALYZE); excludes planning, result transfer and network"
REMAINDER_LABEL = "result transfer and client overhead"
REMAINDER_TOLERANCE = 0.05  # fraction of the client mean
REMAINDER_SE_FACTOR = 2  # combined standard errors of the four means
# Allowed values for prepared_statements mode
PREPARED_MODES = frozenset(("on", "off"))


class TimingRefused(ValueError):
    """A timing that violates the consistency checks."""


def _standard_error(values: List[float]) -> float:
    """Standard error of the mean: sample std dev (n-1) / sqrt(n)."""
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std_dev = math.sqrt(variance)
    return std_dev / math.sqrt(n)


def breakdown(client_ms: List[float], select1_ms: List[float], planning_ms: List[float],
              execution_ms: List[float]) -> Dict[str, Any]:
    """Breakdown of client time into components, with consistency checks.

    Args:
        client_ms: Per-sample client times (walk + transfer + client overhead), in ms.
        select1_ms: Per-sample SELECT 1 round-trip times, in ms.
        planning_ms: Per-sample EXPLAIN planning times, in ms.
        execution_ms: Per-sample EXPLAIN execution times, in ms.

    Returns:
        Dict with n_samples, client_p95_ms, client_mean_ms, execution_p95_ms, execution_mean_ms,
        planning_mean_ms, select1_rtt_mean_ms, remainder_mean_ms.

    Raises:
        TimingRefused: if fewer than 2 samples, if remainder mean < -(REMAINDER_TOLERANCE * client mean
            + REMAINDER_SE_FACTOR * combined SE), or if execution_p95_ms > client_p95_ms.
    """
    n_samples = len(client_ms)
    if n_samples < 2:
        raise TimingRefused(f"fewer than 2 samples ({n_samples}); cannot calculate standard error")

    client_mean = sum(client_ms) / n_samples
    select1_mean = sum(select1_ms) / n_samples
    planning_mean = sum(planning_ms) / n_samples
    execution_mean = sum(execution_ms) / n_samples

    client_p95 = advisor.percentile(client_ms, 95)
    execution_p95 = advisor.percentile(execution_ms, 95)

    client_se = _standard_error(client_ms)
    select1_se = _standard_error(select1_ms)
    planning_se = _standard_error(planning_ms)
    execution_se = _standard_error(execution_ms)
    combined_se = math.sqrt(client_se ** 2 + select1_se ** 2 + planning_se ** 2 + execution_se ** 2)

    remainder_mean = client_mean - (select1_mean + planning_mean + execution_mean)

    # Check: remainder must be above the negative tolerance plus the combined standard errors
    tolerance_pct = int(REMAINDER_TOLERANCE * 100)
    tolerance_ms = REMAINDER_TOLERANCE * client_mean
    se_allowance_ms = REMAINDER_SE_FACTOR * combined_se
    if remainder_mean < -(tolerance_ms + se_allowance_ms):
        raise TimingRefused(f"remainder {remainder_mean:.3f} ms is below -({tolerance_pct}% of client mean "
                           f"{client_mean:.3f} ms = {tolerance_ms:.3f} ms + {REMAINDER_SE_FACTOR} standard errors "
                           f"= {se_allowance_ms:.3f} ms) = -{tolerance_ms + se_allowance_ms:.3f} ms")

    # Check: execution p95 must not exceed client p95
    if execution_p95 > client_p95:
        raise TimingRefused(f"execution p95 {execution_p95:.3f} ms exceeds client p95 "
                           f"{client_p95:.3f} ms")

    return {
        "n_samples": n_samples,
        "client_p95_ms": client_p95,
        "client_mean_ms": client_mean,
        "client_mean_se_ms": client_se,
        "execution_p95_ms": execution_p95,
        "execution_mean_ms": execution_mean,
        "execution_mean_se_ms": execution_se,
        "planning_mean_ms": planning_mean,
        "planning_mean_se_ms": planning_se,
        "select1_rtt_mean_ms": select1_mean,
        "select1_rtt_mean_se_ms": select1_se,
        "remainder_mean_ms": remainder_mean,
    }


def time_walk(conn: Any, sql: str, params: Dict[str, Any], warmup: int, samples: int,
              clock: Callable[[], float] = time.perf_counter) -> Dict[str, Any]:
    """Time a walk on the given connection (never opens a new one).

    Args:
        conn: Postgres connection (must support conn.execute(sql, params).fetchall()).
        sql: The walk SQL.
        params: Parameters to bind (e.g., {"start_node_id": seed}).
        warmup: Number of plain (untimed) runs before sampling.
        samples: Number of timed runs to collect.
        clock: A clock function (default time.perf_counter). Must return seconds.

    Returns:
        Dict from breakdown(...): per-sample means and p95s, plus consistency checks.

    Raises:
        TimingRefused: if consistency checks fail.
    """
    # Disable psycopg 3's server-side prepared statements (cache after 5 runs).
    # This ensures both timed walks and each EXPLAIN plan every time.
    conn.prepare_threshold = None

    # Warmup: plain runs, not timed, not counted
    for _ in range(warmup):
        conn.execute(sql, params).fetchall()

    client_ms: List[float] = []
    select1_ms: List[float] = []
    planning_ms: List[float] = []
    execution_ms: List[float] = []
    prepared_statements: Optional[str] = None

    for sample_idx in range(samples):
        # Read the prepare_threshold at the first timed run (before the clock starts on sample 0)
        if sample_idx == 0:
            threshold = conn.prepare_threshold
            prepared_statements = "off" if threshold is None else "on"

        # Timed walk
        t0 = clock()
        conn.execute(sql, params).fetchall()
        t1 = clock()
        client_ms.append((t1 - t0) * 1000)

        # SELECT 1 round-trip
        t0 = clock()
        conn.execute("SELECT 1").fetchall()
        t1 = clock()
        select1_ms.append((t1 - t0) * 1000)

        # EXPLAIN (ANALYZE, TIMING OFF, FORMAT JSON)
        explain_sql = f"EXPLAIN (ANALYZE, TIMING OFF, FORMAT JSON) {sql}"
        result = conn.execute(explain_sql, params).fetchone()
        plan_doc = json.loads(result[0]) if isinstance(result[0], str) else result[0]
        top_plan = plan_doc[0] if isinstance(plan_doc, list) else plan_doc

        # Refuse if either time field is missing (never silently read as 0)
        if "Planning Time" not in top_plan:
            raise TimingRefused("Planning Time field missing from EXPLAIN output")
        if "Execution Time" not in top_plan:
            raise TimingRefused("Execution Time field missing from EXPLAIN output")

        planning_ms.append(top_plan["Planning Time"])
        execution_ms.append(top_plan["Execution Time"])

    result = breakdown(client_ms, select1_ms, planning_ms, execution_ms)
    result["prepared_statements"] = prepared_statements
    return result
