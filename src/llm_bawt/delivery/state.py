"""Did a delivery attempt run, and what legacy status does that project to.

Pure: no I/O. This is the single owner of the design's §4.2 evidence
classification and §4.3 status projection (TASK-1306, C1 = TASK-2480).

Only positive evidence counts. Missing evidence is never proof that an
attempt did not run: a run end, a teardown or an empty retry list leaves an
attempt with no STARTED and no REJECTED UNCERTAIN, not NOT_RUN.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class EventKind(str, Enum):
    CLAIMED = "CLAIMED"
    ACCEPTED = "ACCEPTED"
    STARTED = "STARTED"
    REJECTED = "REJECTED"
    INTERRUPTED = "INTERRUPTED"
    # Exactly one terminal per attempt.
    DONE = "DONE"
    NOT_RUN = "NOT_RUN"
    UNCERTAIN = "UNCERTAIN"


TERMINAL_KINDS = frozenset({EventKind.DONE, EventKind.NOT_RUN, EventKind.UNCERTAIN})


class ItemKind(str, Enum):
    START = "START"
    ADDITION = "ADDITION"
    USER_ADDITION = "USER_ADDITION"


class RunClass(str, Enum):
    RAN = "RAN"
    NOT_RUN = "NOT_RUN"
    UNCERTAIN = "UNCERTAIN"
    LIVE = "LIVE"  # the attempt's run hasn't ended; nothing to decide yet
    CONFLICT = "CONFLICT"


OUTCOME_LEGACY = "legacy"
UNKNOWN_PREFIX = "outcome_unknown: "


def unknown_error(reason: str) -> str:
    return f"{UNKNOWN_PREFIX}{reason}"


def is_conflict(kinds: Iterable[EventKind | str]) -> bool:
    """STARTED alongside a not-run proof, or more than one terminal."""
    seen = [EventKind(k) for k in kinds]
    terminals = [k for k in seen if k in TERMINAL_KINDS]
    if len(terminals) > 1:
        return True
    started = EventKind.STARTED in seen
    return started and (EventKind.REJECTED in seen or EventKind.NOT_RUN in seen)


def classify(kinds: Iterable[EventKind | str], *, run_ended: bool, finished: bool = False) -> RunClass:
    """§4.2 classification of one attempt's evidence.

    ``run_ended``: the run (or epoch) this attempt belonged to is over.
    ``finished``: that run completed normally rather than being killed or torn down.
    """
    seen = {EventKind(k) for k in kinds}
    if is_conflict(seen):
        return RunClass.CONFLICT
    if EventKind.STARTED in seen:
        if not run_ended:
            return RunClass.LIVE
        return RunClass.RAN if finished else RunClass.UNCERTAIN
    if EventKind.REJECTED in seen or EventKind.NOT_RUN in seen:
        return RunClass.NOT_RUN
    if not seen:
        # Never claimed: nothing could have run.
        return RunClass.NOT_RUN
    if not run_ended:
        return RunClass.LIVE
    return RunClass.UNCERTAIN


@dataclass(frozen=True)
class Projection:
    status: str
    last_error: str | None = None


def project(
    item: ItemKind | str | None,
    kinds: Iterable[EventKind | str],
    *,
    cancelled: bool = False,
    failed_reason: str | None = None,
    unknown_reason: str | None = None,
    done_ok: bool = False,
    live: bool = False,
) -> Projection | None:
    """§4.3 rows 1–6 and 9; first match wins.

    Returns None for rows 7–8 (a live attempt): C1 keeps today's
    DISPATCHING/STEERING there, so the caller leaves status unchanged.
    """
    seen = [EventKind(k) for k in kinds]
    kind = ItemKind(item) if item else None
    if is_conflict(seen):
        return Projection("FAILED", unknown_error("evidence_conflict"))
    if kind in (ItemKind.ADDITION, ItemKind.USER_ADDITION) and EventKind.STARTED in seen:
        return Projection("DELIVERED")
    if kind is ItemKind.START and done_ok:
        return Projection("DELIVERED")
    if cancelled:
        return Projection("CANCELLED")
    if failed_reason is not None:
        return Projection("FAILED", failed_reason)
    if unknown_reason is not None:
        return Projection("FAILED", unknown_error(unknown_reason))
    if live:
        return None
    return Projection("QUEUED")


__all__ = [
    "EventKind", "ItemKind", "OUTCOME_LEGACY", "Projection", "RunClass", "TERMINAL_KINDS",
    "UNKNOWN_PREFIX", "classify", "is_conflict", "project", "unknown_error",
]
