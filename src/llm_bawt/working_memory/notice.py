"""The working-memory change notice (docs/WORKING_MEMORY.md 9).

The pure half reads a TASK-1318 :class:`PinDecision` conservatively: what the
harness may hold (``P``, ``missing``), the bounded unverified floor ``F`` and
the notice episode ``N``. :func:`prepare_notice` adds the reads and the one
session-row write, and returns the notice line when it fires.

Ordering per 9.3 and 13: the caller runs this after the pin transaction has
committed and before the turn's command is published, so the mark (with the
``compaction_gen`` it saw) commits before the harness can compact the notice
away. The mark is best effort: a turn that dies after the mark loses that
notice until the next external change or boundary (OI-R2, Nick's call).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from . import metrics
from .policy import WorkingMemoryPolicy
from .prompts import notice_text
from .render import EMPTY, OK, SECTION_NAME
from .session_state import SUPPRESSED, compaction_gen_of, mode_of, read_state, update_state
from .store import WorkingMemoryStore

logger = logging.getLogger(__name__)

NONE = -1  # NULL in the spec's max() arithmetic


# -- pure reading of a pin decision (9.2) ------------------------------------

def entry_version(sections: Any) -> int | None:
    """A section entry's version, or None when it is missing.

    Missing means absent, or a state other than ok/empty (unavailable,
    suppressed), or no usable version.
    """
    entry = sections.get(SECTION_NAME) if isinstance(sections, dict) else None
    if not isinstance(entry, dict) or entry.get("state") not in (OK, EMPTY):
        return None
    version = entry.get("version")
    return version if isinstance(version, int) and not isinstance(version, bool) and version >= 0 else None


def _floor_parts(floor: Any) -> tuple[int | None, bool] | None:
    if not isinstance(floor, dict):
        return None
    version = floor.get("min_version")
    version = version if isinstance(version, int) and not isinstance(version, bool) and version >= 0 else None
    return version, bool(floor.get("may_be_missing"))


def _lowest(*versions: int | None) -> int | None:
    present = [v for v in versions if v is not None]
    return min(present) if present else None


def has_candidate(decision) -> bool:
    """Whether the harness may hold an unverified re-pin's fallback render.

    Every unverified decision has a candidate, even when the pointer recorded
    no ``candidate_sections``: a turn that rendered without the section
    (working memory disabled or refused for the bot) stores none, yet its
    fallback bytes are what an auto-compaction adopts. That candidate's entry
    reads as missing (Loopy 2a r1 B1).
    """
    return decision.snapshot_unverified or decision.candidate_sections is not None


def next_floor(floor: Any, decision) -> dict[str, Any] | None:
    """``F`` after this decision: fold while unverified, clear on a verified kept turn.

    Folding only lowers ``min_version`` or sets ``may_be_missing``, so a
    staged boundary that is never confirmed can cost an extra notice, never
    a lost one.
    """
    if has_candidate(decision):
        current = _floor_parts(floor) or (None, False)
        s = entry_version(decision.pinned_sections)
        c = entry_version(decision.candidate_sections)
        return {
            "min_version": _lowest(current[0], s, c),
            "may_be_missing": current[1] or s is None or c is None,
        }
    if not decision.is_boundary and not decision.snapshot_unverified:
        return None
    return floor if isinstance(floor, dict) else None


@dataclass(frozen=True)
class Holding:
    """What the harness may hold: lowest present version ``p`` (or -1), and ``missing``."""

    p: int
    missing: bool


def holding(decision, floor: Any) -> Holding:
    s = entry_version(decision.pinned_sections)
    candidate = has_candidate(decision)
    c = entry_version(decision.candidate_sections) if candidate else None
    parts = _floor_parts(floor)
    f_version, f_missing = parts if parts is not None else (None, False)
    lowest = _lowest(s, c, f_version)
    return Holding(
        p=NONE if lowest is None else lowest,
        missing=s is None or (candidate and c is None) or f_missing,
    )


def episode_version(state: dict[str, Any], decision) -> int | None:
    """``N``: the notice version, only within this pin episode and with no compaction since (9.2, 9.4)."""
    notice = state.get("notice")
    if not isinstance(notice, dict) or notice.get("pin_turn_id") != decision.pin_turn_id:
        return None
    marked_gen = notice.get("compaction_gen")
    marked_gen = marked_gen if isinstance(marked_gen, int) and not isinstance(marked_gen, bool) else 0
    if compaction_gen_of(state) > marked_gen:
        return None
    version = notice.get("version")
    return version if isinstance(version, int) and not isinstance(version, bool) else None


def threshold(hold: Holding, n: int | None) -> int:
    return max(hold.p, NONE if n is None else n)


def fires(hold: Holding, n: int | None, external_max: int | None) -> bool:
    """Rule 1: an external change above ``T``. Rule 2: missing and no notice this episode."""
    if external_max is not None and external_max > threshold(hold, n):
        return True
    return hold.missing and n is None


@dataclass(frozen=True)
class Evaluation:
    floor: dict[str, Any] | None
    hold: Holding
    n: int | None
    t: int


def evaluate(state: dict[str, Any], decision) -> Evaluation:
    floor = next_floor(state.get("unverified_floor"), decision)
    hold = holding(decision, floor)
    n = episode_version(state, decision)
    return Evaluation(floor, hold, n, threshold(hold, n))


def apply_floor(state: dict[str, Any], floor: dict[str, Any] | None) -> dict[str, Any]:
    if floor is None:
        state.pop("unverified_floor", None)
    else:
        state["unverified_floor"] = floor
    return state


# -- the turn hook (9.1-9.3) -------------------------------------------------

def prepare_notice(engine, *, bot_id: str, user_id: str, session_id: str, decision,
                   eligible: bool) -> str | None:
    """Fold the floor and decide the notice for one started turn with a pin decision.

    ``eligible`` is the caller's turn-kind check (not a maintenance command,
    not a scheduler turn). Boundary turns fold the floor but never notice.
    Returns the notice line when it fires, after its mark has committed.
    Never raises: a failure logs and the turn continues without a notice.
    """
    if decision is None:
        return None
    try:
        # describe() reads the tags first: a private bot is refused before any row.
        if not WorkingMemoryPolicy(engine).describe(bot_id).tools_allowed:
            return None
        state = read_state(engine, session_id)
        if state is None or mode_of(state) == SUPPRESSED:
            return None
        store = WorkingMemoryStore(engine)
        may_fire = eligible and not decision.is_boundary

        def external_after(t: int, conn=None) -> tuple[int | None, int]:
            if not may_fire:
                return None, 0
            return store.external_changes(bot_id, user_id, after=t, session_id=session_id, conn=conn)

        # Unlocked pre-check: most turns change nothing and take no lock.
        first = evaluate(state, decision)
        external, _ = external_after(first.t)
        if first.floor == state.get("unverified_floor") and not (may_fire and fires(first.hold, first.n, external)):
            logger.debug("working memory notice for %s: P=%s N=%s T=%s fired=False",
                         bot_id, first.hold.p, first.n, first.t)
            return None
        fired = _mark(engine, session_id, decision, may_fire, external_after)
    except Exception:
        metrics.record_notice_failure(bot_id)
        logger.warning("working memory notice failed for %s session %s; the turn continues without it",
                       bot_id, session_id, exc_info=True)
        return None
    if not fired:
        return None
    metrics.record_notice(bot_id)
    return notice_text(bot_id)


def _mark(engine, session_id: str, decision, may_fire: bool, external_after) -> bool:
    """One session-row transaction: re-evaluate under the lock, write ``F`` and the notice.

    The decision is re-made from the locked state, with the external-change
    read at the locked threshold; that read takes no lock (13) and runs on
    this transaction's connection, so a mark holding its session row never
    waits on the pool for a second connection. The notice field is written
    only when it fires; a floor-only write leaves it untouched.
    """
    fired = False

    def mutate(conn, state: dict[str, Any]) -> dict[str, Any]:
        nonlocal fired
        locked = evaluate(state, decision)
        apply_floor(state, locked.floor)
        external, current = external_after(locked.t, conn)
        fired = may_fire and fires(locked.hold, locked.n, external)
        logger.debug("working memory notice: P=%s N=%s T=%s external=%s fired=%s",
                     locked.hold.p, locked.n, locked.t, external, fired)
        if fired:
            state["notice"] = {
                "version": max(NONE if locked.n is None else locked.n,
                               NONE if external is None else external, current),
                "compaction_gen": compaction_gen_of(state),
                "pin_turn_id": decision.pin_turn_id,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        return state

    with engine.begin() as conn:
        if update_state(conn, session_id, lambda state: mutate(conn, state)) is None:
            return False
    return fired


__all__ = [
    "Evaluation", "Holding", "NONE", "apply_floor", "entry_version", "episode_version", "evaluate",
    "fires", "has_candidate", "holding", "next_floor", "prepare_notice", "threshold",
]
