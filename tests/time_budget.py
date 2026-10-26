"""Per-case wall-clock budgets for the unit gate (TASK-1189, TASK-1457).

A unit case is judged on setup + call + teardown. Tests marked
``real_process`` spawn a real binary (bash, node, git), whose start-up on a
cold, shared CI runner is outside the test's control: a node launcher that
takes 0.1s on echo took 3.3s on a hosted runner. They get their own, wider
budget so the unit budget can stay tight, and both stay enforced.
"""

from __future__ import annotations

from typing import Iterable, Mapping, NamedTuple

REAL_PROCESS = "real_process"


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
