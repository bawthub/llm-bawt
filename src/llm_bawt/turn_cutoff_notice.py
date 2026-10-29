"""Tell an agent its previous turn was cut off by the system (TASK-1738).

When a bridge restart (host shutdown) or the app's inactivity watchdog kills a
turn mid-tool-call, the SDK records the killed call as a user rejection ("STOP
what you are doing and wait for the user"). The resumed session still holds
that text, so the next turn reads as a human refusal and the bot halts.

The first turn after such a cut-off in the same agent session gets the
``agents.turn_cutoff_notice`` registry prompt in front of its message. The
cause is read from the cut-off turn's own row: the bridge-stamped
``abort_cause`` (an explicit user Stop never qualifies) or the watchdog's
error text. Only the immediately preceding ended turn counts, so the notice
is shown once.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from agent_bridge.abort_contract import HOST_SHUTDOWN
from agent_bridge.subscriber import INACTIVITY_TIMEOUT_MARKER

logger = logging.getLogger(__name__)

TURN_CUTOFF_NOTICE_KEY = "agents.turn_cutoff_notice"

RESTART_CAUSE = "a bridge restart"
WATCHDOG_CAUSE = "the app's inactivity watchdog"


@dataclass(frozen=True)
class TurnCutoff:
    cause: str
    ended_at: datetime | None


def cutoff_of(row: Mapping[str, Any] | None) -> TurnCutoff | None:
    """The system cut-off that ended ``row``, or ``None`` for any other end."""
    if not row or row.get("ended_at") is None:
        return None
    from .inter_bot_delivery_retry import _RowView, restart_abort_cause

    status = row.get("status")
    ended_at = row.get("ended_at")
    if status == "aborted":
        # One owner of "the host killed this turn" (a user Stop never counts).
        if restart_abort_cause(_RowView(row)) == HOST_SHUTDOWN:
            return TurnCutoff(RESTART_CAUSE, ended_at)
        return None
    if status == "error" and INACTIVITY_TIMEOUT_MARKER in str(row.get("error_text") or ""):
        return TurnCutoff(WATCHDOG_CAUSE, ended_at)
    return None


_PREVIOUS_TURN_SQL = """
    SELECT status, ended_at, request_json, error_text
    FROM turn_logs
    WHERE bot_id = :bot_id AND agent_session_key = :session_key
      AND ended_at IS NOT NULL
      AND (agent_request_id IS NULL OR agent_request_id <> :request_id)
    ORDER BY ended_at DESC
    LIMIT 1
"""


def previous_turn_cutoff(engine, *, bot_id: str, session_key: str, request_id: str) -> TurnCutoff | None:
    """Whether the last ended turn of this bot's agent session was cut off."""
    from sqlalchemy import text as sa_text

    with engine.connect() as conn:
        row = conn.execute(sa_text(_PREVIOUS_TURN_SQL), {
            "bot_id": bot_id, "session_key": session_key, "request_id": request_id,
        }).mappings().first()
    return cutoff_of(row)


def render_notice(cutoff: TurnCutoff, bot_id: str | None) -> str:
    from .prompt_registry import PromptResolver, get_prompt_resolver

    ended = cutoff.ended_at
    if isinstance(ended, datetime):
        ended = ended.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    variables = {"cause": cutoff.cause, "ended_at": ended or "an unknown time"}
    try:
        text = get_prompt_resolver().render(
            key=TURN_CUTOFF_NOTICE_KEY, variables=variables,
            scope_type="bot" if bot_id else "global", scope_id=bot_id,
        )
        if text.strip():
            return text
    except Exception as error:  # a DB failure falls back to the registered default
        logger.warning("turn cut-off notice resolve failed, using its default: %s", error)
    definition = PromptResolver.definition_for(TURN_CUTOFF_NOTICE_KEY)
    if definition is None:
        raise RuntimeError(f"{TURN_CUTOFF_NOTICE_KEY} is not registered")
    return definition.default_body().format(**variables)


def with_cutoff_notice(prompt: str, *, engine, bot_id: str | None, session_key: str,
                       request_id: str) -> str:
    """``prompt``, prefixed with the notice when the previous turn was cut off.

    Never blocks a turn: a lookup failure sends the prompt unchanged.
    """
    if engine is None or not bot_id or not session_key:
        return prompt
    try:
        cutoff = previous_turn_cutoff(engine, bot_id=bot_id, session_key=session_key,
                                      request_id=request_id)
    except Exception as error:
        logger.warning("turn cut-off lookup failed for %s: %s", session_key, error)
        return prompt
    if cutoff is None:
        return prompt
    logger.info("Previous turn of %s was cut off by %s; prefixing notice", session_key, cutoff.cause)
    return f"{render_notice(cutoff, bot_id)}\n\n{prompt}"
