"""The provision lock (conftest): held for the whole of a @provision test, free otherwise.

Replaces the earlier log-only tests (`assert True`, `lock is not None`), which passed with
the conftest hooks deleted. These go red if `pytest_runtest_setup` stops acquiring the lock
for provision-marked tests, or if `pytest_runtest_teardown` stops releasing it.
"""

import pytest


@pytest.mark.provision
def test_a_provision_test_runs_holding_the_lock(provision_lock):
    assert provision_lock.locked()


def test_an_unmarked_test_runs_without_the_lock(provision_lock):
    # Runs after the provision test above (file order): the lock was released at its teardown.
    assert not provision_lock.locked()


@pytest.mark.provision
class TestMarkedClass:
    def test_a_class_marker_also_takes_the_lock(self, provision_lock):
        assert provision_lock.locked()


def test_released_again_after_a_marked_class(provision_lock):
    assert not provision_lock.locked()
