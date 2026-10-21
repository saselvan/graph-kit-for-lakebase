"""The refresh job's sync task: run each synced table's managed pipeline once and wait.

The pipeline id is on the synced table's status. Snapshot tables get a full refresh
(that is what a snapshot sync is);
triggered tables get an incremental update. The mode is read from config (modes.policy_name),
not from the synced table, whose GET returns a sparse spec without the scheduling policy.

Before the first update, the sync task checks the privacy gate is clear and that every
built Delta table carries the current privacy stamp (binding.check_table_stamps, as provision
does). Every refresh chain requires the stamp to be valid: the stamp is the label
{stamp, table_id, certified_version}, written by the kit build or by
the chain's "stamp" task (lgk.refresh.operator_stamp), and the sync requires it to match the run
manifest, the current cleared stamp, the current UC table_id and the Delta history
(lgk.refresh.label.check_labels); a mismatch is refused, never skipped.

Adaptive parallelism: before EACH new update the task reads the branch's
free connection slots (`sync_limits.free_slots`) and starts it only while running <
min(sync_max_parallel, free // per_sync). It refuses (SyncLimitError) only when not even one
fits and nothing is running. While updates run, the same connection samples the sync sessions;
the night's peak is kept in the run manifest as the high-water mark that raises per_sync.
"""

import logging
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from lgk import manifest as mf
from lgk import sync_limits as sl
from lgk.branch import connect_branch
from lgk.privacy import binding, floor
from lgk.privacy.gate import fold_policy, read_run_mapping, require_clear, required_run_folder
from lgk.refresh import label, modes
from lgk.refresh.freshness import Sql, live_clients, synced_pairs
from lgk.steps.provision import _wait_synced_table_online

logger = logging.getLogger(__name__)

_DONE = {"COMPLETED", "FAILED", "CANCELED"}
SAMPLE_INTERVAL_S = 2.0  # the sync-session sampler's period while updates run


class SyncUpdateError(RuntimeError):
    """A synced table's pipeline update did not complete."""


def _wait_update_done(w: Any, pipeline_id: str, update_id: str, max_wait_s: int = 1200) -> None:
    deadline, state = time.monotonic() + max_wait_s, ""
    while time.monotonic() < deadline:
        update = w.pipelines.get_update(pipeline_id=pipeline_id, update_id=update_id).update
        state = str(getattr(update.state, "value", update.state))
        if state in _DONE:
            if state != "COMPLETED":
                raise SyncUpdateError(f"pipeline {pipeline_id} update {update_id} ended {state}")
            return
        time.sleep(10)
    raise TimeoutError(f"pipeline {pipeline_id} update {update_id} still {state} after {max_wait_s}s")


def sync_all(w: Any, sql: Sql, config: Dict[str, Any], full_refresh: Optional[bool] = None) -> None:
    """Sync every synced table (version and enrichment), as many at once as the branch has room for.

    Dispatcher (adaptive parallelism): read the bounds and the budget per sync, then run the updates
    adaptively on one held branch connection, then record the peak sync sessions seen.
    """
    modes.require_v1_sync_mode(config)
    parallel = sl.config_int(config, "sync_max_parallel", sl.DEFAULT_SYNC_MAX_PARALLEL, 1)
    headroom = sl.config_int(config, "app_connection_headroom", sl.DEFAULT_APP_CONNECTION_HEADROOM, 0)
    jobs = [(dest_id, _full_refresh(config, dest_id, full_refresh)) for dest_id, _ in synced_pairs(config, sql)]
    high_water = mf.read_sync_high_water(config)
    budget = _Budget(parallel, headroom, sl.per_sync(high_water), high_water)
    conn = connect_branch(config)
    conn.autocommit = True  # each read its own transaction: pg_stat_activity is snapshotted per transaction
    sampler = _WriterSampler(conn)
    try:
        _run_adaptive(w, jobs, budget, conn, sampler)
    finally:
        conn.close()
        _record_high_water(config, sampler)


def _full_refresh(config: Dict[str, Any], dest_id: str, full_refresh: Optional[bool]) -> bool:
    table = dest_id.rsplit(".", 1)[1]
    return modes.policy_name(config, table) == "SNAPSHOT" if full_refresh is None else full_refresh


@dataclass(frozen=True)
class _Budget:
    max_parallel: int  # sync_max_parallel
    headroom: int  # app_connection_headroom
    per_sync: int  # max(18, high_water) + 2
    high_water: Optional[int]


class _WriterSampler:
    """Light sampler of the sync sessions on the branch; a failed read disables it, never the sync.

    `peak` is a PER-SYNC peak: the sessions seen divided by the updates running then (rounded up).
    All syncs together never exceed running x (one sync's peak), so this never over-states one sync
    (no false ratchet of the never-lowered mark); it can under-state one bursting sync beside idle
    ones, which the floor of 18 covers, and any sample with one update running is exact.
    """

    def __init__(self, conn: Any) -> None:
        self.conn = conn
        self.peak: Optional[int] = None
        self.failed = False

    def sample(self, running: int) -> None:
        if self.failed:
            return
        try:
            seen = sl.sync_writers(self.conn)
        except Exception as e:  # noqa: BLE001 - the sampler must never fail a night
            self.failed = True
            logger.warning(f"sync: the connection sampler stopped ({e}); no high-water is recorded tonight")
            return
        one = -(-seen // max(running, 1))  # ceil
        self.peak = one if self.peak is None else max(self.peak, one)


def _record_high_water(config: Dict[str, Any], sampler: _WriterSampler) -> None:
    """Keep the night's peak sync sessions in the run manifest (a failure only warns)."""
    if sampler.failed or sampler.peak is None:
        return
    try:
        kept = mf.record_sync_high_water(config, sampler.peak)
        logger.info(f"sync: peak {sampler.peak} sync connections per sync tonight; high-water now {kept}")
    except Exception as e:  # noqa: BLE001 - do not mask the night's own result
        logger.warning(f"sync: could not record the sync connection high-water {sampler.peak}: {e}")


def _room_now(conn: Any, budget: _Budget, running: int) -> int:
    """Read the free slots live and log the parallelism they allow (the gate before each start)."""
    slots = sl.free_slots(conn, budget.headroom)
    allowed = sl.parallel_now(budget.max_parallel, slots, budget.per_sync)
    logger.info(f"sync: parallel {allowed} now (min(sync_max_parallel {budget.max_parallel}, "
                f"free {slots.free} // per sync {budget.per_sync})); {running} running; {slots.describe()}")
    if allowed == 0 and running == 0:
        sl.refuse_no_room(slots, budget.per_sync, budget.high_water)
    return allowed


def _sync_one(w: Any, dest_id: str, full: bool) -> None:
    """Start one synced table's update, wait for it to complete, then for the table to be online."""
    pipeline_id = w.postgres.get_synced_table(name=f"synced_tables/{dest_id}").status.pipeline_id
    update = w.pipelines.start_update(pipeline_id=pipeline_id, full_refresh=full)
    logger.info(f"sync: {dest_id} pipeline {pipeline_id} update {update.update_id} full={full}")
    _wait_update_done(w, pipeline_id, update.update_id)
    _wait_synced_table_online(w, dest_id)


def _wait_one(running: Set[Future], errors: List[BaseException], sampler: _WriterSampler) -> Set[Future]:
    """Sample the sync sessions until at least one running update ends; return those still running."""
    while True:
        done, rest = wait(running, timeout=SAMPLE_INTERVAL_S, return_when=FIRST_COMPLETED)
        sampler.sample(max(len(rest), 1))  # updates still running (the ended one's sessions are going)
        errors.extend(f.exception() for f in done if f.exception() is not None)
        if done:
            return set(rest)


def _run_adaptive(w: Any, jobs: List[Tuple[str, bool]], budget: _Budget, conn: Any,
                  sampler: _WriterSampler) -> None:
    """Start each update only while running < the parallelism the branch allows now (re-read each time).

    No room and nothing running: refuse. No room but updates running: wait for one, re-check.
    After the first failure no new update starts; the first error is raised once all have ended.
    """
    pending, running, errors = list(jobs), set(), []  # type: ignore[var-annotated]
    with ThreadPoolExecutor(max_workers=budget.max_parallel) as pool:
        while pending and not errors:
            if len(running) < _room_now(conn, budget, len(running)):
                dest_id, full = pending.pop(0)
                running.add(pool.submit(_sync_one, w, dest_id, full))
            else:
                running = _wait_one(running, errors, sampler)
        while running:
            running = _wait_one(running, errors, sampler)
    if errors:
        raise errors[0]


def _require_stamped_tables(sql: Sql, config: Dict[str, Any], w: Any) -> None:
    """The privacy gate must be clear, and the kit tables must carry its stamp, on every chain.

    Whoever rebuilt nodes/edges/node_id_map (the kit build, or an operator's
    pipeline that stamps them with lgk.refresh.operator_stamp), a table without the current
    stamp raises StaleBuildError; there is no skip. The fold floor on the source Delta `edges`
    and the join floor on the source Delta `nodes` are checked too: the
    operator's pipeline is exactly the writer the kit's HAVING and CASE did not filter.
    """
    from lgk.refresh.job import refresh_chain

    refresh_chain(config, enriched=False)  # refuses a config the job cannot run (unknown build_mode)
    cleared = require_clear(required_run_folder(config))
    stamp = binding.stamp_id(cleared)
    fq = f"{config['lakebase_catalog']}.{config['kit_schema']}"
    fold_floor, fold_types = fold_policy(cleared)
    floor.assert_fold_floor(sql, f"{fq}.edges", fold_types, fold_floor)
    # Check the same constraints on the source Delta `nodes` for every floored node join
    floor.assert_join_floor(sql, f"{fq}.nodes", read_run_mapping(required_run_folder(config)), fold_floor)
    label.check_labels(w, sql, config, [f"{fq}.{t}" for t in binding.BUILT_TABLES], stamp)


def sync(config: Dict[str, Any], full_refresh: Optional[bool] = None) -> None:
    """The sync task, with a live client and warehouse: the stamp check, then every update."""
    modes.require_v1_sync_mode(config)
    w, sql = live_clients(config)
    _require_stamped_tables(sql, config, w)
    sync_all(w, sql, config, full_refresh)
