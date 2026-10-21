"""A decision stamped by hand, for tests of the gate checks AFTER the stamp and history checks.

`forge` is what a person who edits the run folder by hand can do: recompute the stamp with
gate.stamp AND append that stamp to privacy_decisions.json as the newest decision (the residual
risk in docs/privacy-decision-history.md: the run folder is the only record). The later gate
checks (open flags, leaks, pins, classes) must still refuse such a decision on their own, so
those tests forge both files and assert the later check's error.

A hand stamp WITHOUT the history entry is refused earlier, by UnrecordedDecisionError
(tests/test_privacy_decision_history.py).
"""

from typing import Any, Dict

from lgk.privacy import binding, decisions, gate
from lgk.privacy.flags import write_flags_file


def forge(run_folder: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Write `data` with a hand-computed stamp to flags.json and record it as the newest decision."""
    stamped = gate.stamp(str(run_folder), data)
    write_flags_file(str(run_folder), stamped)
    decisions.record_decision(str(run_folder), binding.stamp_id(stamped))
    return stamped
