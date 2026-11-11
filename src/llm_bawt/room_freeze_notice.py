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

from agent_bridge.room_freeze import ASKED, LOCKED, UNKNOWN

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


# An affirmative ask: a clause that opens (after an optional lead-in) with the
# verb itself, e.g. "commit it", "ok, go ahead and push", "please ship that".
# A mention ("the last commit"), question ("why did you push?") or prohibition
# ("do not commit") never matches, so it stays locked.
_LEAD_IN = r"(?:(?:ok(?:ay)?|yes|yep|sure|great|good|cool|now|then|and|so|please|go ahead and|you can|feel free to)[\s,]+)*"
# The verb must take an object from this short list or end the clause, so a
# noun phrase ("commit message needs fixing", "push notifications") never counts.
_OBJECT = (r"(?:it|that|this|them|these|those|everything|all|now|"
           r"(?:the|your|my|these|those) (?:changes?|work|fix(?:es)?|edits?|branch|commits?)|"
           r"(?:and|then) (?:commit|push|ship))")
_ASK_CLAUSE = re.compile(rf"^\s*{_LEAD_IN}(?:commit|push|ship)(?:\s+{_OBJECT}\b.*|\s*)$",
                         re.IGNORECASE)
_CLAUSE_SPLIT = re.compile(r"[.!;\n]+|,\s*(?:then|and)\s+|\s+(?:then|and then)\s+")
_NEGATION = re.compile(r"\b(?:not|don'?t|do not|never|no|stop|hold off|wait)\b", re.IGNORECASE)


def asks_to_commit(prompt: str) -> bool:
    """Whether ``prompt`` affirmatively asks to commit, push or ship.

    Any negation anywhere in the message withholds the grant, so a later
    "actually, don't" wins over an earlier ask.
    """
    if _NEGATION.search(prompt or ""):
        return False
    return any("?" not in clause and _ASK_CLAUSE.search(clause)
               for clause in _CLAUSE_SPLIT.split(prompt or ""))


def room_freeze_state(prompt: str, *, engine, bot_id: str | None) -> str | None:
    """``None`` outside a room freeze, else :data:`LOCKED` or :data:`ASKED`.

    Read before any notice is prefixed, so only the message itself counts.
    An inter-bot delivery never runs under a room freeze (it waits at the
    claim), so a turn running now carries the user's message or a system
    continuation; a continuation's text asks nothing, so it stays locked.
    A lookup failure answers :data:`UNKNOWN`: the turn runs without a notice,
    but the git hook still denies commit/push.
    """
    if engine is None or not bot_id:
        return None
    try:
        if not in_room(engine, bot_id):
            return None
    except Exception as error:
        logger.warning("room freeze lookup failed for %s, locking git: %s", bot_id, error)
        return UNKNOWN
    return ASKED if asks_to_commit(prompt) else LOCKED


def with_room_freeze_notice(prompt: str, state: str | None, *, bot_id: str | None,
                            reset_action: str | None = None) -> str:
    """``prompt``, prefixed with the notice when ``state`` is a room freeze.

    A reset or bridge command (/new, /clear, /compact) is left alone, since a
    prefix would turn the command into plain text.
    """
    from .turn_cutoff_notice import is_session_command

    if state in (None, UNKNOWN) or is_session_command(prompt, reset_action):
        return prompt
    return f"{render_notice(bot_id)}\n\n{prompt}"


__all__ = ["ROOM_FREEZE_NOTICE_KEY", "asks_to_commit", "in_room", "render_notice", "room_freeze_state",
           "with_room_freeze_notice"]
