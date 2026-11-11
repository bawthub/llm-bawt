"""Tell an agent the user is "in the room" (TASK-1692).

While an operator freeze with ``admit_user`` covers a bot, only the user's own
chat reaches it; other bots' messages, task dispatch and schedules queue until
thaw. Each turn the bot runs then gets the ``agents.room_freeze_notice``
registry prompt in front of its message: iterate with the user, edit in place,
and don't commit, gate, request review or file cards unless the user's message
asks. Agent prompts are pinned per session, so this is per turn, not a system
prompt section.
"""
from __future__ import annotations

import logging
import re

from agent_bridge.room_freeze import ASKED, LOCKED

logger = logging.getLogger(__name__)

ROOM_FREEZE_NOTICE_KEY = "agents.room_freeze_notice"


def in_room(engine, bot_id: str) -> bool:
    """Whether a live "in the room" freeze covers ``bot_id`` now."""
    from .service.turn_gates import room_gate_ids

    with engine.connect() as conn:
        return bool(room_gate_ids(conn, bot_id))


def render_notice(bot_id: str | None) -> str:
    from .prompt_registry import PromptResolver, get_prompt_resolver

    try:
        text = get_prompt_resolver().render(
            key=ROOM_FREEZE_NOTICE_KEY, variables={},
            scope_type="bot" if bot_id else "global", scope_id=bot_id,
        )
        if text.strip():
            return text
    except Exception as error:  # a DB failure falls back to the registered default
        logger.warning("room freeze notice resolve failed, using its default: %s", error)
    definition = PromptResolver.definition_for(ROOM_FREEZE_NOTICE_KEY)
    if definition is None:
        raise RuntimeError(f"{ROOM_FREEZE_NOTICE_KEY} is not registered")
    return definition.default_body()


_ASKS_TO_COMMIT = re.compile(r"\b(commit|push|ship)\w*\b", re.IGNORECASE)


def room_freeze_state(prompt: str, *, engine, bot_id: str | None) -> str | None:
    """``None`` outside a room freeze, else :data:`LOCKED` or :data:`ASKED`.

    Read before any notice is prefixed, so only the message itself counts.
    An inter-bot delivery never runs under a room freeze (it waits at the
    claim), so a turn running now carries the user's message or a system
    continuation; a continuation's text names no commit, so it stays locked.
    A lookup failure answers ``None``: a DB fault never blocks a turn.
    """
    if engine is None or not bot_id:
        return None
    try:
        if not in_room(engine, bot_id):
            return None
    except Exception as error:
        logger.warning("room freeze lookup failed for %s: %s", bot_id, error)
        return None
    return ASKED if _ASKS_TO_COMMIT.search(prompt or "") else LOCKED


def with_room_freeze_notice(prompt: str, state: str | None, *, bot_id: str | None,
                            reset_action: str | None = None) -> str:
    """``prompt``, prefixed with the notice when ``state`` is a room freeze.

    A reset or bridge command (/new, /clear, /compact) is left alone, since a
    prefix would turn the command into plain text.
    """
    from .turn_cutoff_notice import is_session_command

    if state is None or is_session_command(prompt, reset_action):
        return prompt
    return f"{render_notice(bot_id)}\n\n{prompt}"


__all__ = ["ROOM_FREEZE_NOTICE_KEY", "in_room", "render_notice", "room_freeze_state",
           "with_room_freeze_notice"]
