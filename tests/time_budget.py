"""Per-case wall-clock budgets for the unit gate (TASK-1189, TASK-1457).

A unit case is judged on setup + call + teardown. Tests marked
``real_process`` spawn a real binary (bash, node, git), whose start-up on a
cold, shared CI runner is outside the test's control: a node launcher that
takes 0.1s on echo took 3.3s on a hosted runner. They get their own, wider
budget so the unit budget can stay tight, and both stay enforced.

A passing case that breaches its budget is run once more and judged on the
faster run (TASK-1650). A hang, a fixed sleep or a real slowdown repeats and
still fails; a one-off stall of the runner (a cold node start once took 8s on
a hosted runner whose usual worst case is under 1.5s) does not. The retry
never changes a test's outcome, only which duration the budget sees.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, NamedTuple, Sequence

REAL_PROCESS = "real_process"
# user_properties key on a case's teardown report; travels with xdist reports.
RETIME = "time_budget_retime"


def case_budget(unit_budget: float | None, real_process_budget: float | None,
                is_real_process: bool) -> float | None:
    """The budget one case is held to; None when no budget is enforced."""
    if unit_budget is None:
        return None
    if is_real_process and real_process_budget is not None:
        return real_process_budget
    return unit_budget


def needs_retime(reports: Sequence, budget: float | None) -> bool:
    """A case that passed every phase but took at least its budget."""
    return (budget is not None and bool(reports) and all(r.passed for r in reports)
            and sum(r.duration for r in reports) >= budget)


def retimed_duration(reports: Sequence) -> float:
    """The retry's duration; a retry that didn't pass leaves the breach standing."""
    if not reports or not all(r.passed for r in reports):
        return math.inf
    return sum(r.duration for r in reports)


class Breach(NamedTuple):
    label: str
    budget: float
    slow: list[tuple[str, float]]  # slowest first

    def lines(self) -> list[str]:
        return [f"{self.label} time budget exceeded ({self.budget}s including fixture setup/teardown):",
                *(f"  {duration:.3f}s {nodeid}" for nodeid, duration in self.slow)]


def breaches(durations: Mapping[str, float], real_process: Iterable[str],
             unit_budget: float | None, real_process_budget: float | None) -> list[Breach]:
    """Every budget a case met or exceeded; ``real_process_budget`` None means the unit budget."""
    if unit_budget is None:
        return []
    marked = frozenset(real_process)
    process_budget = unit_budget if real_process_budget is None else real_process_budget
    found = []
    for label, budget, in_scope in (("Unit", unit_budget, lambda nodeid: nodeid not in marked),
                                    ("Real-process", process_budget, lambda nodeid: nodeid in marked)):
        slow = sorted(((nodeid, duration) for nodeid, duration in durations.items()
                       if in_scope(nodeid) and duration >= budget), key=lambda pair: -pair[1])
        if slow:
            found.append(Breach(label, budget, slow))
    return found


def slowest_real_process(durations: Mapping[str, float], real_process: Iterable[str]) -> tuple[str, float] | None:
    """The slowest marked case, printed every gated run so the budget can be set from data."""
    marked = [(nodeid, durations[nodeid]) for nodeid in set(real_process) if nodeid in durations]
    return max(marked, key=lambda pair: pair[1]) if marked else None
