"""Offline tests for the measure step's client timing: per walk, the in-region client
p95 on a warm connection is the headline; the breakdown adds up in MEANS, because p95s do not add.

client mean = SELECT 1 mean + planning mean + execution mean + remainder ("result transfer and client
overhead"). A remainder below the stated negative tolerance is refused, and so is an execution p95
above the client p95. The parts' p95s are never required to sum below the client p95.

Fakes only: a scripted connection advances a fake clock by a set duration per statement, so every
expected number below is written by hand, never computed by the code under test.
"""

import json
from pathlib import Path

import pytest

from lgk.measure import record
from lgk.steps import measure as step

try:
    from lgk.measure import timing
except ImportError:  # red first: each test fails on its own until lgk.measure.timing exists
    timing = None
from tests.test_measure import EDGE_ROWS, NODE_ROWS, FakeWarehouse, ServedConn, _R
from tests.test_measure_step import _answers, _by, _config, _lines

EXECUTION_LABEL = "Postgres execution time (EXPLAIN ANALYZE); excludes planning, result transfer and network"
REMAINDER_LABEL = "result transfer and client overhead"
OLD_LABEL = "excludes network and connection time"


class TimedConn(ServedConn):
    """ServedConn with scripted durations: a walk advances the fake clock by the next `walk_s`
    (seconds), SELECT 1 by `rtt_s`, and EXPLAIN (ANALYZE, TIMING OFF) answers the next `plans`
    (planning ms, execution ms) pair. A walk or EXPLAIN beyond the script fails the test, unless a
    default is given."""

    def __init__(self, walk_s=None, plans=None, default_walk_s=None, default_plan=None, **kw):
        super().__init__(**kw)
        self.walk_s, self.plans = list(walk_s or []), list(plans or [])
        self.default_walk_s, self.default_plan = default_walk_s, default_plan

    def walk_seconds(self):
        if self.walk_s:
            return self.walk_s.pop(0)
        assert self.default_walk_s is not None, "more walk runs than the script holds"
        return self.default_walk_s

    def plan_times(self):
        if self.plans:
            return self.plans.pop(0)
        assert self.default_plan is not None, "more EXPLAIN runs than the script holds"
        return self.default_plan


class PreparingConn(TimedConn):
    """TimedConn with psycopg 3's `prepare_threshold` (default 5); records its value at every statement."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.prepare_threshold, self.seen = 5, []

    def execute(self, sql, params=None):
        self.seen.append((sql, self.prepare_threshold))
        return super().execute(sql, params)


class StuckConn(TimedConn):
    """The setting is left on: an assignment to `prepare_threshold` does not take; it stays 5."""

    @property
    def prepare_threshold(self):
        return 5

    @prepare_threshold.setter
    def prepare_threshold(self, value):
        pass


class ReArmingConn(PreparingConn):
    """Something on the connection turns auto-prepare back on (5) during the first walk, the warm-up."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.rearmed = False

    def execute(self, sql, params=None):
        if sql == WALK and not self.rearmed:
            self.prepare_threshold, self.rearmed = 5, True
        return super().execute(sql, params)


WALK = "walk-sql"
SEED = {"start_node_id": "n4"}


def _ms(values):
    return [v / 1000.0 for v in values]


# ---------------------------------------------------------------------------- the timer


class TestTimeWalk:
    def test_client_p95_is_the_nearest_rank_p95_of_the_timed_walks_not_the_p50(self):
        # hostile: 20 distinct client times, so p50 (10 ms) and p95 (19 ms) differ
        conn = TimedConn(walk_s=_ms([1.0]) + _ms(range(20, 0, -1)), plans=[(0.1, 0.5)] * 20)  # reordered
        got = timing.time_walk(conn, WALK, SEED, warmup=1, samples=20, clock=conn.now)
        assert got["n_samples"] == 20
        assert got["client_p95_ms"] == pytest.approx(19.0)
        assert got["client_mean_ms"] == pytest.approx(10.5)

    def test_the_cold_first_run_is_a_warm_up_and_never_a_sample(self):
        # hostile: the first walk on the connection is cold (500 ms); every warm one takes 4 ms
        conn = TimedConn(walk_s=_ms([500.0] + [4.0] * 20), plans=[(0.5, 2.0)] * 20)
        got = timing.time_walk(conn, WALK, SEED, warmup=1, samples=20, clock=conn.now)
        assert got["client_mean_ms"] == pytest.approx(4.0)
        assert got["client_p95_ms"] == pytest.approx(4.0)

    def test_select1_runs_once_per_sample_on_the_walks_own_connection(self):
        conn = TimedConn(walk_s=_ms([4.0] * 21), plans=[(0.5, 2.0)] * 20, rtt_s=0.0015)
        got = timing.time_walk(conn, WALK, SEED, warmup=1, samples=20, clock=conn.now)
        assert [s for s, _ in conn.log].count("SELECT 1") == 20
        assert got["select1_rtt_mean_ms"] == pytest.approx(1.5)

    def test_planning_and_execution_are_read_apart_from_explain(self):
        # hostile: planning folded into execution would read 2.5 / 0
        conn = TimedConn(walk_s=_ms([4.0] * 21), plans=[(0.5, 2.0)] * 20)
        got = timing.time_walk(conn, WALK, SEED, warmup=1, samples=20, clock=conn.now)
        assert got["planning_mean_ms"] == pytest.approx(0.5)
        assert got["execution_mean_ms"] == pytest.approx(2.0)
        assert got["execution_p95_ms"] == pytest.approx(2.0)
        timed = [s for s, _ in conn.log if s.startswith("EXPLAIN (ANALYZE, TIMING OFF")]
        assert len(timed) == 20 and all(s.endswith(WALK) for s in timed)

    def test_the_seed_is_bound_as_a_parameter_on_every_walk(self):
        conn = TimedConn(walk_s=_ms([4.0] * 21), plans=[(0.5, 2.0)] * 20)
        timing.time_walk(conn, WALK, SEED, warmup=1, samples=20, clock=conn.now)
        walks = [(s, p) for s, p in conn.log if WALK in s]
        assert walks and all(p == SEED for _, p in walks)

    def test_the_clock_is_a_real_clock_by_default(self):
        # hostile: a constant clock would give every walk 0 ms; here each walk really takes >= 2 ms
        import time

        class SleepConn(TimedConn):
            def execute(self, sql, params=None):
                if sql == WALK:
                    time.sleep(0.002)
                return super().execute(sql, params)

        conn = SleepConn(walk_s=[0.0] * 21, plans=[(0.0, 0.0)] * 20, rtt_s=0.0)
        got = timing.time_walk(conn, WALK, SEED, warmup=1, samples=20)
        assert got["n_samples"] == 20 and got["client_mean_ms"] >= 2.0

    def test_auto_prepare_is_off_on_the_timing_connection_before_the_first_timed_run(self):
        # hostile: psycopg 3's default prepare_threshold is 5, so after the 5th run the timed walks would
        # use a server-side prepared (cached) plan while each EXPLAIN plans fresh. Off: both plan every time.
        conn = PreparingConn(walk_s=_ms([4.0] * 21), plans=[(0.5, 2.0)] * 20)
        timing.time_walk(conn, WALK, SEED, warmup=1, samples=20, clock=conn.now)
        assert conn.seen and all(threshold is None for _, threshold in conn.seen), conn.seen[:3]

    def test_the_mode_is_read_from_the_connection_never_from_a_constant(self):
        # owner ruling: the record says "off" only if prepare_threshold is None at the first timed run
        conn = PreparingConn(walk_s=_ms([4.0] * 21), plans=[(0.5, 2.0)] * 20)
        assert timing.time_walk(conn, WALK, SEED, warmup=1, samples=20, clock=conn.now)["prepared_statements"] == "off"
        # hostile: the setting is left on (the assignment does not take), so a constant "off" would be false
        stuck = StuckConn(walk_s=_ms([4.0] * 21), plans=[(0.5, 2.0)] * 20)
        assert timing.time_walk(stuck, WALK, SEED, warmup=1, samples=20, clock=stuck.now)["prepared_statements"] == "on"

    def test_the_mode_is_the_setting_at_the_first_timed_run_not_before_the_warm_up(self):
        # hostile: the setting is turned back on during the warm-up; a reader that checks before the warm-up says "off"
        conn = ReArmingConn(walk_s=_ms([4.0] * 21), plans=[(0.5, 2.0)] * 20)
        got = timing.time_walk(conn, WALK, SEED, warmup=1, samples=20, clock=conn.now)
        walks = [threshold for sql, threshold in conn.seen if sql == WALK]
        at_first_timed_run = walks[1]  # walks[0] is the warm-up
        assert got["prepared_statements"] == ("off" if at_first_timed_run is None else "on"), walks[:3]


# ---------------------------------------------------------------------------- the breakdown (means)


class TestBreakdown:
    def test_means_add_up_with_the_remainder_as_transfer_and_client_overhead(self):
        got = timing.breakdown(client_ms=[4.0] * 20, select1_ms=[1.0] * 20, planning_ms=[0.5] * 20,
                               execution_ms=[2.0] * 20)
        assert got["client_mean_ms"] == pytest.approx(4.0)
        assert got["remainder_mean_ms"] == pytest.approx(0.5)
        assert timing.REMAINDER_LABEL == REMAINDER_LABEL

    def test_parts_p95s_summing_above_the_client_p95_pass_when_the_means_add_up(self):
        # hostile: each part spikes in two different rounds, so p95s 5 + 3 + 6 = 14 > client p95 10,
        # while the means (1.4 + 0.75 + 4.2 = 6.35) sit under the client mean 10. p95s do not add.
        select1 = [1.0] * 18 + [5.0, 5.0]
        planning = [3.0, 3.0] + [0.5] * 18
        execution = [4.0] * 9 + [6.0, 6.0] + [4.0] * 9
        got = timing.breakdown(client_ms=[10.0] * 20, select1_ms=select1, planning_ms=planning,
                               execution_ms=execution)
        assert got["client_p95_ms"] == pytest.approx(10.0)
        assert got["execution_p95_ms"] == pytest.approx(6.0)
        assert got["remainder_mean_ms"] == pytest.approx(3.65)

    def test_a_small_negative_remainder_is_reported_as_it_is_never_clamped(self):
        # parts' means 3 + 1 + 6.4 = 10.4 over a client mean of 10: -0.4 ms, inside 5 % of 10
        got = timing.breakdown(client_ms=[10.0] * 20, select1_ms=[3.0] * 20, planning_ms=[1.0] * 20,
                               execution_ms=[6.4] * 20)
        assert got["remainder_mean_ms"] == pytest.approx(-0.4)

    def test_a_remainder_below_the_negative_tolerance_is_refused(self):
        # parts' means 3 + 1 + 6.6 = 10.6 over a client mean of 10: -0.6 ms, beyond 5 % of 10
        with pytest.raises(timing.TimingRefused, match="remainder"):
            timing.breakdown(client_ms=[10.0] * 20, select1_ms=[3.0] * 20, planning_ms=[1.0] * 20,
                             execution_ms=[6.6] * 20)

    def test_the_tolerance_is_the_stated_five_percent_of_the_client_mean(self):
        assert timing.REMAINDER_TOLERANCE == 0.05
        # hostile: the same -0.6 ms overshoot is inside 5 % of a 20 ms client mean
        got = timing.breakdown(client_ms=[20.0] * 20, select1_ms=[3.0] * 20, planning_ms=[1.0] * 20,
                               execution_ms=[16.6] * 20)
        assert got["remainder_mean_ms"] == pytest.approx(-0.6)

    # The four means come from separate sample sets, so their sum carries their combined standard
    # error, sqrt(SE_c^2 + SE_s^2 + SE_p^2 + SE_e^2); a remainder is refused only beyond the stated
    # tolerance plus two combined standard errors. Client 3,5 x 10: sd sqrt(20/19), SE 0.229416;
    # allowance 5 % of 4 + 2 x 0.229416 = 0.658831 ms (one SE: 0.429416; three SEs: 0.888247).
    NOISY_CLIENT = [3.0, 5.0] * 10

    def test_a_remainder_past_five_percent_but_inside_two_standard_errors_passes_unclamped(self):
        # parts 1 + 0.5 + 3.05 = 4.55 over a client mean of 4: -0.55 ms, past one SE, inside two
        got = timing.breakdown(client_ms=self.NOISY_CLIENT, select1_ms=[1.0] * 20, planning_ms=[0.5] * 20,
                               execution_ms=[3.05] * 20)
        assert got["remainder_mean_ms"] == pytest.approx(-0.55)

    def test_a_remainder_past_two_standard_errors_is_refused(self):
        # parts 1 + 0.5 + 3.25 = 4.75: -0.75 ms, past two SEs (0.658831), inside three (0.888247)
        with pytest.raises(timing.TimingRefused, match="remainder"):
            timing.breakdown(client_ms=self.NOISY_CLIENT, select1_ms=[1.0] * 20, planning_ms=[0.5] * 20,
                             execution_ms=[3.25] * 20)

    def test_the_allowance_takes_every_parts_standard_error_not_only_the_clients(self):
        # hostile: the client is steady (SE 0); execution 2.25,3.25 x 10 has SE 0.5 sqrt(20/19)/sqrt(20)
        # = 0.114708, so the allowance is 0.2 + 2 x 0.114708 = 0.429416 and -0.25 ms passes
        got = timing.breakdown(client_ms=[4.0] * 20, select1_ms=[1.0] * 20, planning_ms=[0.5] * 20,
                               execution_ms=[2.25, 3.25] * 10)
        assert got["remainder_mean_ms"] == pytest.approx(-0.25)

    # Every part noisy (SE = half the spread x sqrt(20/19) / sqrt(20)): client 0.229416, select 1
    # 0.114708, planning and execution 0.057354. In quadrature: sqrt(0.052632 + 0.013158 + 2 x 0.003289)
    # = 0.269013, allowance 0.2 + 2 x 0.269013 = 0.738026 ms. Summed linearly instead: 0.2 + 2 x 0.458832
    # = 1.117664; with only the client and execution SEs: 0.2 + 2 x 0.236476 = 0.672952.
    ALL_NOISY = {"client_ms": [3.0, 5.0] * 10, "select1_ms": [0.5, 1.5] * 10, "planning_ms": [0.25, 0.75] * 10}

    def test_the_standard_errors_add_in_quadrature_not_linearly(self):
        # -0.9 ms: inside the linear sum, past the quadrature allowance
        with pytest.raises(timing.TimingRefused, match="remainder"):
            timing.breakdown(execution_ms=[3.15, 3.65] * 10, **self.ALL_NOISY)

    def test_the_select1_and_planning_standard_errors_count_too(self):
        # -0.7 ms: past the client-and-execution-only allowance, inside the full one
        got = timing.breakdown(execution_ms=[2.95, 3.45] * 10, **self.ALL_NOISY)
        assert got["remainder_mean_ms"] == pytest.approx(-0.7)

    def test_the_allowance_is_the_stated_two_standard_errors(self):
        assert timing.REMAINDER_SE_FACTOR == 2

    def test_execution_p95_above_the_client_p95_is_refused(self):
        # hostile: the means add up (1 + 0.5 + 4.8 = 6.3 < 10); only the execution p95 (12) is wrong
        with pytest.raises(timing.TimingRefused, match="execution p95"):
            timing.breakdown(client_ms=[10.0] * 20, select1_ms=[1.0] * 20, planning_ms=[0.5] * 20,
                             execution_ms=[4.0] * 18 + [12.0, 12.0])

    def test_the_execution_label_is_exact(self):
        assert timing.EXECUTION_LABEL == EXECUTION_LABEL

    def test_each_mean_has_its_standard_error_with_the_sample_formula(self):
        # SE = sample standard deviation (n - 1) / sqrt(n). Values worked by hand:
        # client 1,3,1,3: sd sqrt(4/3), SE sqrt(4/3)/2 = 0.57735 (hostile: the population formula gives 0.5)
        got = timing.breakdown(client_ms=[1.0, 3.0, 1.0, 3.0], select1_ms=[0.5] * 4,
                               planning_ms=[0.1, 0.3, 0.1, 0.3], execution_ms=[0.4, 0.8, 0.4, 0.8])
        assert got["n_samples"] == 4
        assert got["client_mean_se_ms"] == pytest.approx(0.577350, abs=1e-6)
        assert got["select1_rtt_mean_se_ms"] == pytest.approx(0.0)
        assert got["planning_mean_se_ms"] == pytest.approx(0.0577350, abs=1e-7)
        assert got["execution_mean_se_ms"] == pytest.approx(0.115470, abs=1e-6)
        assert got["remainder_mean_ms"] == pytest.approx(0.7)

    @pytest.mark.parametrize("n", [0, 1])
    def test_fewer_than_two_samples_is_refused_never_a_zero_standard_error(self, n):
        # hostile: one sample has no spread to measure; an SE of 0 would read as an exact mean
        with pytest.raises(timing.TimingRefused, match="samples"):
            timing.breakdown(client_ms=[4.0] * n, select1_ms=[1.0] * n, planning_ms=[0.5] * n,
                             execution_ms=[2.0] * n)


# ---------------------------------------------------------------------------- the step


def _timed_deps(conn, connects):
    wh = FakeWarehouse({"nodes:prov": NODE_ROWS, "edges:aff": EDGE_ROWS})

    def connect(cfg):
        connects.append(cfg)
        return conn

    return step.MeasureDeps(connect=connect, warehouse_sql=lambda cfg: wh,
                            endpoint_cu=lambda cfg: {"min": 0.5, "max": 4.0}, clock=conn.now)


def _timed_conn(**kw):
    return TimedConn(answers=_answers(), default_walk_s=0.004, default_plan=(0.5, 2.0), rtt_s=0.001, **kw)


class TestMeasureStepTiming:
    def test_each_walk_record_has_client_p95_execution_p95_planning_ms_and_select1_rtt_ms(self, tmp_path):
        cfg = _config(tmp_path)
        conn = _timed_conn()
        step.run_measure(cfg, _timed_deps(conn, []))
        lines = _lines(cfg["run_folder"])
        for shape in ("web", "tree"):
            [x] = _by(lines, "walk_latency", shape, 1)
            assert x["unit"] == "ms" and x["timing"] == "client" and x["n_samples"] == 20
            assert x["value"] == pytest.approx(4.0) and x["client_p95_ms"] == pytest.approx(4.0)
            assert x["client_mean_ms"] == pytest.approx(4.0)
            assert x["execution_p95_ms"] == pytest.approx(2.0) and x["execution_mean_ms"] == pytest.approx(2.0)
            assert x["planning_mean_ms"] == pytest.approx(0.5)
            assert x["select1_rtt_mean_ms"] == pytest.approx(1.0)
            assert x["remainder_mean_ms"] == pytest.approx(0.5)
            for se in ("client_mean_se_ms", "select1_rtt_mean_se_ms", "planning_mean_se_ms", "execution_mean_se_ms"):
                assert x[se] == pytest.approx(0.0), se  # every scripted run takes the same time
            assert x["prepared_statements"] == "off"
        # one line per walk: the old server-only p50/p95 lines are gone
        assert not [x for x in lines if x["metric"].startswith("walk_latency.")]

    def test_the_line_records_the_mode_the_timer_saw_never_the_constant(self, tmp_path):
        # hostile: the setting is left on, so a line that copies a module constant would say "off"
        cfg = _config(tmp_path)
        conn = StuckConn(answers=_answers(), default_walk_s=0.004, default_plan=(0.5, 2.0), rtt_s=0.001)
        step.run_measure(cfg, _timed_deps(conn, []))
        lines = _by(_lines(cfg["run_folder"]), "walk_latency")
        assert lines and all(x["prepared_statements"] == "on" for x in lines)
        md = (Path(cfg["run_folder"]) / record.REPORT).read_text()
        assert "prepared statements: on" in md and "prepared statements: off" not in md

    def test_select1_and_the_walk_use_the_same_connection(self, tmp_path):
        # hostile: a timer that opens a new connection per query (one connect per SELECT 1)
        cfg = _config(tmp_path)
        conn, connects = _timed_conn(), []
        step.run_measure(cfg, _timed_deps(conn, connects))
        assert len(connects) == 1
        assert [s for s, _ in conn.log].count("SELECT 1") == 2 * 20  # web d1 + tree d1, 20 samples each

    def test_label_is_exact_in_record_and_report(self, tmp_path):
        cfg = _config(tmp_path)
        step.run_measure(cfg, _timed_deps(_timed_conn(), []))
        for x in _by(_lines(cfg["run_folder"]), "walk_latency"):
            assert x["execution_label"] == EXECUTION_LABEL
            assert x["remainder_label"] == REMAINDER_LABEL
        md = (Path(cfg["run_folder"]) / record.REPORT).read_text()
        assert EXECUTION_LABEL in md and REMAINDER_LABEL in md and "client p95" in md
        assert OLD_LABEL not in md  # hostile: the label right in the record and wrong in the report

    def test_the_step_docstring_carries_the_label(self):
        assert EXECUTION_LABEL in step.__doc__ and OLD_LABEL not in step.__doc__

    def test_a_refused_breakdown_writes_nothing(self, tmp_path):
        # hostile: execution (9 ms) above the client time (4 ms) of every walk
        cfg = _config(tmp_path)
        conn = TimedConn(answers=_answers(), default_walk_s=0.004, default_plan=(0.5, 9.0), rtt_s=0.001)
        with pytest.raises(timing.TimingRefused):
            step.run_measure(cfg, _timed_deps(conn, []))
        assert not (Path(cfg["run_folder"]) / record.JSONL).exists()

    def test_measure_only_reads(self, tmp_path):
        # guard: every statement the step sends is a read
        cfg = _config(tmp_path)
        conn = _timed_conn()
        step.run_measure(cfg, _timed_deps(conn, []))
        assert conn.log and all(s.lstrip().upper().startswith(("SELECT", "WITH", "EXPLAIN")) for s, _ in conn.log)


class _DropsAPlanField(TimedConn):
    """EXPLAIN (ANALYZE) answers without one of its two time fields."""

    def __init__(self, drop, **kw):
        super().__init__(**kw)
        self.drop = drop

    def execute(self, sql, params=None):
        if not sql.startswith("EXPLAIN (ANALYZE, TIMING OFF"):
            return super().execute(sql, params)
        self.log.append((sql, params))
        doc = {"Plan": {"Node Type": "Result"}, "Planning Time": 0.5, "Execution Time": 2.0}
        del doc[self.drop]
        return _R([([doc],)])


@pytest.mark.parametrize("drop", ["Planning Time", "Execution Time"])
def test_an_explain_without_a_time_field_is_refused_never_read_as_zero(drop):
    # hostile: a missing field read as 0 ms hides planning (or execution) and inflates the remainder
    conn = _DropsAPlanField(drop, walk_s=_ms([4.0] * 21))
    with pytest.raises(timing.TimingRefused, match=drop):
        timing.time_walk(conn, WALK, SEED, warmup=1, samples=20, clock=conn.now)


# ---------------------------------------------------------------------------- the record


class TestRecordLabels:
    COMMON = {"build": {"version": "0.1.0", "commit": None, "source_sha256": "ab12"}, "nodes": 7,
              "edges": 8, "endpoint_cu": {"min": 0.5, "max": 4.0}, "ts": "2026-10-05T00:00:00Z"}

    def _line(self, **kw):
        return {"metric": "walk_latency", "value": 4.0, "unit": "ms", "shape": "web", "depth": 1,
                "n_samples": 200, "timing": "client", "client_p95_ms": 4.0, "client_mean_ms": 3.5,
                "execution_p95_ms": 2.0, "execution_mean_ms": 1.8, "planning_mean_ms": 0.4,
                "select1_rtt_mean_ms": 1.0, "remainder_mean_ms": 0.3, "execution_label": EXECUTION_LABEL,
                "remainder_label": REMAINDER_LABEL, "client_mean_se_ms": 0.11, "select1_rtt_mean_se_ms": 0.02,
                "planning_mean_se_ms": 0.03, "execution_mean_se_ms": 0.07, "prepared_statements": "off",
                **self.COMMON, **kw}

    def test_a_walk_latency_line_is_accepted(self):
        json.loads(record.check_line(self._line()))

    @pytest.mark.parametrize("key,value", [
        ("execution_label", "Postgres execution time (EXPLAIN ANALYZE); excludes network"),  # one clause lost
        ("execution_label", "Dr Jane Smith"),                                                # free text
        ("remainder_label", "network"),
        ("client_p95_ms", "4.0"),                                                            # not a number
        ("prepared_statements", "unknown"),                  # owner ruling: only "on" or "off" is a mode
        ("prepared_statements", "OFF"),                      # exact words only
        ("prepared_statements", ""),
        ("execution_mean_se_ms", "0.07"),                                                    # not a number
    ])
    def test_a_changed_label_or_a_non_number_is_refused(self, key, value):
        with pytest.raises(record.RecordRefused):
            record.check_line(self._line(**{key: value}))

    def test_the_report_shows_the_means_breakdown_that_adds_up(self):
        # the means 1.0 + 0.4 + 1.8 + 0.3 = 3.5 (client mean), so a reader can add them up from the report.
        # hostile: execution mean 1.8 differs from execution p95 2.0, so a p95 in a mean column fails
        md = record.report_markdown([self._line()])
        rows = [l for l in md.splitlines() if l.startswith("| web | 1 |")]
        [header] = [l for l in md.splitlines() if "client p95" in l and l.startswith("|")]
        assert len(rows) == 1
        assert "client mean" in header and "execution mean" in header and "remainder" in header
        cells = [c.strip() for c in rows[0].strip("|").split("|")]
        for v in (3.5, 1.8, 0.4, 1.0, 0.3, 4.0, 2.0):
            assert record._fmt(v) in cells, (v, cells)

    def test_the_report_shows_each_mean_standard_error_and_that_prepared_statements_are_off(self):
        # hostile: SE values differ from every mean and from each other, so a swapped column fails
        md = record.report_markdown([self._line()])
        [row] = [l for l in md.splitlines() if l.startswith("| web | 1 |")]
        [header] = [l for l in md.splitlines() if "client p95" in l and l.startswith("|")]
        for name in ("client mean SE", "select 1 mean SE", "planning mean SE", "execution mean SE"):
            assert name in header, (name, header)
        cells = [c.strip() for c in row.strip("|").split("|")]
        heads = [c.strip() for c in header.strip("|").split("|")]
        for name, v in (("client mean SE", 0.11), ("select 1 mean SE", 0.02), ("planning mean SE", 0.03),
                        ("execution mean SE", 0.07), ("n", 200)):
            assert cells[heads.index(name)] == record._fmt(v), (name, cells)
        assert "prepared statements: off" in md

    @pytest.mark.parametrize("key", ["prepared_statements", "client_mean_se_ms", "select1_rtt_mean_se_ms",
                                     "planning_mean_se_ms", "execution_mean_se_ms"])
    def test_a_walk_latency_line_without_its_mode_or_a_standard_error_is_refused(self, key):
        # hostile: an old or hand-made line without the key would still get "prepared statements: off"
        line = self._line()
        del line[key]
        with pytest.raises(record.RecordRefused, match=key):
            record.check_line(line)

    def test_the_report_states_the_mode_the_record_holds_never_a_fixed_string(self):
        # hostile: a report that prints "off" whatever the line says
        line = self._line()
        del line["prepared_statements"]
        assert "prepared statements: off" not in record.report_markdown([line])

    def test_a_line_recorded_with_prepared_statements_on_is_accepted_and_reported_on(self):
        # owner ruling: the record says "on" when the setting is left on; it is recorded, never rewritten
        line = self._line(prepared_statements="on")
        assert json.loads(record.check_line(line))["prepared_statements"] == "on"
        md = record.report_markdown([line])
        assert "prepared statements: on" in md and "prepared statements: off" not in md

    def test_mixed_modes_are_shown_per_row_never_one_mode_for_all(self):
        # hostile: the report reads the mode from the first line only, so a tree walk timed with auto-prepare
        # on would sit under "prepared statements: off"
        web, tree = self._line(), self._line(shape="tree", prepared_statements="on")
        md = record.report_markdown([web, tree])
        assert "prepared statements: off" not in md and "prepared statements: on" not in md
        [header] = [l for l in md.splitlines() if "client p95" in l and l.startswith("|")]
        heads = [c.strip() for c in header.strip("|").split("|")]
        assert "prepared statements" in heads, heads
        for shape, mode in (("web", "off"), ("tree", "on")):
            [row] = [l for l in md.splitlines() if l.startswith(f"| {shape} | 1 |")]
            cells = [c.strip() for c in row.strip("|").split("|")]
            assert cells[heads.index("prepared statements")] == mode, (shape, cells)
