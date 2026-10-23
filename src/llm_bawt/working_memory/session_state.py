"""Working-memory state on ``sessions.session_metadata`` (docs/WORKING_MEMORY.md 5.4).

Everything lives under one key, ``session_metadata.working_memory``:

* ``mode``: ``pinned`` or ``suppressed``. Written by the rotate transaction on
  the successor (10.1). Absent means pinned.
* ``notice``: ``{version, compaction_gen, pin_turn_id, at}`` (9.3).
* ``compaction_gen``: an integer, absent meaning 0 (9.4).
* ``unverified_floor``: ``{min_version, may_be_missing}`` while the pin is
  unverified (9.2).

Every write locks only the session row, reads the current object under that
lock and replaces the ``working_memory`` key with the merged result, so a write
never drops a sibling key it didn't mean to change (13). No working-memory
namespace lock is ever taken here.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

KEY = "working_memory"
PINNED = "pinned"
SUPPRESSED = "suppressed"
MODES = (PINNED, SUPPRESSED)

Mutation = Callable[[dict[str, Any]], dict[str, Any]]


def require_mode(mode: Any) -> str:
    """Validate a rotate caller's mode; there is no default (10.1)."""
    if mode not in MODES:
        raise ValueError(f"working_memory_mode must be one of {MODES}, got {mode!r}")
    return mode


def mode_from_flag(include_working_memory: bool | None) -> str:
    """Only an explicit ``include_working_memory=false`` suppresses (N1)."""
    return SUPPRESSED if include_working_memory is False else PINNED


def successor_metadata(mode: str) -> dict[str, Any]:
    """The value a rotate writes under ``KEY`` on the successor session."""
    return {"mode": require_mode(mode)}


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def state_of(session_metadata: Any) -> dict[str, Any]:
    """The ``working_memory`` object of a metadata value, or ``{}``."""
    value = _metadata(session_metadata).get(KEY)
    return dict(value) if isinstance(value, dict) else {}


def mode_of(state: dict[str, Any]) -> str:
    return SUPPRESSED if state.get("mode") == SUPPRESSED else PINNED


def compaction_gen_of(state: dict[str, Any]) -> int:
    value = state.get("compaction_gen")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def read_state(engine, session_id: str) -> dict[str, Any] | None:
    """Unlocked read; None when the session row doesn't exist."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT session_metadata FROM sessions WHERE id = :sid"), {"sid": session_id},
        ).first()
    return None if row is None else state_of(row[0])


def session_mode(engine, session_id: str | None) -> str:
    """The session's mode; a missing session or no session id is ``pinned``."""
    if not session_id:
        return PINNED
    state = read_state(engine, session_id)
    return mode_of(state or {})


def update_state(conn, session_id: str, mutate: Mutation) -> dict[str, Any] | None:
    """Lock the session row, merge ``mutate``'s result, write only if it changed.

    Runs on the caller's connection and transaction. ``mutate`` gets a copy of
    the current object and returns the new one. Returns the object now stored,
    or None when the session row doesn't exist.
    """
    row = conn.execute(
        text("SELECT session_metadata FROM sessions WHERE id = :sid FOR UPDATE"), {"sid": session_id},
    ).first()
    if row is None:
        return None
    current = state_of(row[0])
    updated = mutate(dict(current))
    if updated != current:
        conn.execute(text("""
            UPDATE sessions SET session_metadata = jsonb_set(
                COALESCE(session_metadata::jsonb, '{}'::jsonb), '{working_memory}', CAST(:state AS jsonb)
            ) WHERE id = :sid
        """), {"state": json.dumps(updated), "sid": session_id})
    return updated


def bump_compaction_gen(engine, session_id: str) -> int | None:
    """One successful compaction recorded: ``compaction_gen += 1`` (9.4)."""
    def mutate(state: dict[str, Any]) -> dict[str, Any]:
        state["compaction_gen"] = compaction_gen_of(state) + 1
        return state

    with engine.begin() as conn:
        state = update_state(conn, session_id, mutate)
    return None if state is None else state["compaction_gen"]


def compaction_succeeded(usage: Any) -> bool:
    return (
        isinstance(usage, dict)
        and usage.get("context_action") == "compact"
        and usage.get("context_action_outcome") == "success"
    )


def bump_after_finalize(ctx, usage: Any) -> None:
    """Finalization hook: bump the generation after a recorded compaction.

    Runs for every non-protected bot whose backend pins, opted in or not, so a
    compaction while working memory is disabled still re-arms an earlier
    notice (9.4). A failure is logged and costs only that re-arm.
    """
    if not compaction_succeeded(usage):
        return
    from .policy import is_protected

    try:
        if is_protected(str(ctx.bot_id or "")):
            return
        backend = getattr(getattr(ctx.llm_bawt, "client", None), "_backend", None)
        if not getattr(backend, "pins_system_prompt", False):
            return
        session_id = str((ctx.thread_binding or {}).get("thread_session_id") or "").strip()
        if not session_id:
            return
        bump_compaction_gen(ctx.svc._turn_log_store.engine, session_id)
    except Exception:
        logger.warning(
            "working memory compaction_gen bump failed for turn %s; the notice re-arm for this "
            "compaction is lost", getattr(ctx, "turn_log_id", None), exc_info=True,
        )


__all__ = [
    "KEY", "MODES", "PINNED", "SUPPRESSED", "bump_after_finalize", "bump_compaction_gen",
    "compaction_gen_of", "compaction_succeeded", "mode_from_flag", "mode_of", "read_state",
    "require_mode", "session_mode", "state_of", "successor_metadata", "update_state",
]
