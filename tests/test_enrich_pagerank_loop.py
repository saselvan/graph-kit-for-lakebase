"""Offline tests for PageRank's stop rule (L1 < 1e-6, cap 50).

The loop control is Python; the math is Spark SQL (proven live in test_enrich_live.py).
A scripted SQL runner feeds the L1 values the loop sees, so the stop rule is tested exactly.
"""

import pytest

from lgk.enrich.methods import run_pagerank


class ScriptedSql:
    """Answers the node-count query and replays a scripted L1 sequence."""

    def __init__(self, l1_values):
        self.l1_values = list(l1_values)
        self.l1_calls = 0
        self.statements = []

    def __call__(self, sql):
        self.statements.append(sql)
        if "L1_CHANGE" in sql:
            value = self.l1_values[min(self.l1_calls, len(self.l1_values) - 1)]
            self.l1_calls += 1
            return [[str(value)]]
        if sql.lstrip().upper().startswith("SELECT COUNT(*)"):
            return [["10"]]
        return []


@pytest.mark.seam2
def test_stops_at_cap_50_when_never_converging():
    sql = ScriptedSql([1.0])
    out = run_pagerank(sql, "c.s")
    assert (out["iterations"], out["converged"], out["final_l1"]) == (50, False, 1.0)
    assert sql.l1_calls == 50


@pytest.mark.seam2
def test_stops_on_first_l1_strictly_below_1e_6():
    # hostile: exactly 1e-6 must NOT count as converged (the rule is L1 < 1e-6).
    sql = ScriptedSql([1e-3, 1e-6, 9e-7, 1e-9])
    out = run_pagerank(sql, "c.s")
    assert (out["iterations"], out["converged"], out["final_l1"]) == (3, True, 9e-7)


@pytest.mark.seam2
def test_writes_last_computed_ranks_to_node_score():
    # hostile: an ODD iteration count, so the newest ranks are in _enrich_pr_b, not the
    # table the loop started from. Deleting the ping-pong swap goes red here.
    sql = ScriptedSql([1e-3, 1e-3, 1e-9])
    run_pagerank(sql, "c.s")
    inserts = [s for s in sql.statements if "INSERT INTO c.s.node_score" in s]
    assert len(inserts) == 1
    assert "FROM c.s._enrich_pr_b" in inserts[0] and "'pagerank'" in inserts[0]
