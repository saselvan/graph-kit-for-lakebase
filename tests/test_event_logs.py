"""Offline tests for lgk.event_logs: which sync event-log tables are this graph's (B1/B2, persona round 2).

The pipeline id is the recorded one, else Unity Catalog's (`tables.get(...).properties`), else
refused. Fakes stand in for the workspace client only.
"""

from types import SimpleNamespace

import pytest
from databricks.sdk.errors import NotFound

from lgk import event_logs as el
from lgk import versions

CAT = "cat"
A_PID = "-".join(["a1b2c3d4", "0000", "4000", "8000", "0000000000aa"])
B_PID = "-".join(["0d1e2f30", "0000", "4000", "8000", "00000000beef"])
WRONG_PID = "-".join(["ffffffff", "ffff", "4fff", "8fff", "ffffffffffff"])


class FakeUc:
    """tables.get answers from `pids` (full name -> pipeline id, or None for no property); NotFound otherwise."""

    def __init__(self, pids=None, schemas=(f"{CAT}.lgk_k_v1",)):
        self.pids = pids or {}
        self.got = []
        live = set(schemas)

        def get_table(full):
            self.got.append(full)
            if full not in self.pids:
                raise NotFound(f"Table '{full}' does not exist.")
            pid = self.pids[full]
            # as live (2026-10-04): the id is a TableInfo field. hostile: a wrong id under properties
            # catches an implementation reading `.properties["pipeline_id"]`
            return SimpleNamespace(pipeline_id=pid, properties={"source_table": "x", "pipeline_id": WRONG_PID})

        def get_schema(full):
            if full not in live:
                raise NotFound(f"Schema '{full}' does not exist.")
            return SimpleNamespace(comment="lgk:graph=g")

        self.tables = SimpleNamespace(get=get_table)
        self.schemas = SimpleNamespace(get=get_schema)


def test_event_log_name_replaces_every_hyphen():
    # hostile: a hyphenated id (persona 1's leftover), the dest schema is the synced table's schema
    assert el.event_log_table(f"{CAT}.lgk_care_v1.edges", "-".join(["c3d4e5f6", "0000", "4000", "8000", "0000000000c3"])) == \
        f"{CAT}.lgk_care_v1.event_log_c3d4e5f6_0000_4000_8000_0000000000c3"


def test_the_recorded_id_wins_and_uc_is_not_asked():
    # hostile: UC would answer a DIFFERENT id, so an always-look-up implementation names the wrong log
    w = FakeUc({f"{CAT}.lgk_k_v1.nodes": WRONG_PID})
    assert el.pipeline_id(w, f"{CAT}.lgk_k_v1.nodes", A_PID) == A_PID
    assert w.got == []


def test_a_missing_recorded_id_is_read_from_unity_catalog():
    w = FakeUc({f"{CAT}.lgk_k_v1.edges": B_PID})
    # hostile: provision recorded pipeline_id None (status had none at create time)
    assert el.own_event_logs(w, {f"{CAT}.lgk_k_v1.edges": None}) == [
        f"{CAT}.lgk_k_v1.event_log_0d1e2f30_0000_4000_8000_00000000beef"]
    assert w.got == [f"{CAT}.lgk_k_v1.edges"]


@pytest.mark.parametrize("uc", [
    FakeUc({}),                                 # hostile: UC no longer has the table, its schema still exists
    FakeUc({f"{CAT}.lgk_k_v1.edges": None}),    # hostile: UC has the table but names no pipeline
])
def test_no_recorded_id_and_no_lookup_refuses_naming_the_table(uc):
    with pytest.raises(el.UnknownPipelineError, match=r"cat\.lgk_k_v1\.edges") as err:
        el.own_event_logs(uc, {f"{CAT}.lgk_k_v1.nodes": A_PID, f"{CAT}.lgk_k_v1.edges": None})
    assert isinstance(err.value, versions.NotOursError)  # callers' "refuse, delete nothing" path


def test_a_table_whose_schema_is_gone_has_no_event_log_left():
    # hostile: a re-run after the UC version schema was already dropped; nothing can remain there
    w = FakeUc({}, schemas=())
    assert el.own_event_logs(w, {f"{CAT}.lgk_k_v1.edges": None}) == []


def test_two_synced_tables_of_one_pipeline_name_one_event_log():
    # hostile: the same pipeline, once recorded and once looked up
    w = FakeUc({f"{CAT}.lgk_k_v1.edges": A_PID})
    assert el.own_event_logs(w, {f"{CAT}.lgk_k_v1.nodes": A_PID, f"{CAT}.lgk_k_v1.edges": None}) == [
        f"{CAT}.lgk_k_v1.event_log_a1b2c3d4_0000_4000_8000_0000000000aa"]


def test_one_event_log_per_pipeline_in_each_destination_schema_in_order():
    w = FakeUc({})
    got = el.own_event_logs(w, {f"{CAT}.lgk_k_v1.nodes": A_PID, f"{CAT}.lgk_k_v2.nodes": B_PID})
    assert got == [f"{CAT}.lgk_k_v1.event_log_a1b2c3d4_0000_4000_8000_0000000000aa",
                   f"{CAT}.lgk_k_v2.event_log_0d1e2f30_0000_4000_8000_00000000beef"]
