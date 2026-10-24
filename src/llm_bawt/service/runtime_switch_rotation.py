"""Start a fresh conversation when a bot's runtime family changes.

A bot's active thread carries runtime-specific state: ``agent_session_keys``
(the SDK transcript to resume) for agent harnesses, and a chat-shaped history
for direct chat. When a profile save moves the bot to a different runtime
family (chat <-> agent, or between agent bridges), resuming that state is
wrong. Example: Nova went agent -> chat -> agent while the same thread stayed
active, and the agent turn resumed a days-old SDK transcript from the first
agent stint.

The fix is the same non-destructive rotation ``/new`` uses: archive the
active thread(s) and open fresh ones, so the next agent turn cold-starts and
seeds per ``history_scope``. Model switches inside a family (e.g. the
``claude-code`` <-> ``claude-proxy`` pair) do NOT rotate; Claude SDK
transcripts are model-agnostic and are meant to resume.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from sqlalchemy import text

from ..bot_types import durable_agent_backend
from ..bot_session_state import BotSessionState
from ..memory.session_reset_lineage import SessionResetLineage
from ..working_memory.session_state import PINNED
from .turn_admission import lock_target

log = logging.getLogger(__name__)

RUNTIME_SWITCH_SOURCE = "runtime_switch"


def thread_switched_event(
    bot_id: str, user_id: str, new_session_id: str, source: str,
) -> dict[str, Any]:
    """TASK-257 ``thread_switched`` payload. Single builder for every emitter."""
    return {
        "_type": "thread_switched",
        "bot_id": bot_id,
        "user_id": user_id,
        "new_session_id": new_session_id,
        "source": source,
        "ts": time.time(),
    }


def runtime_family(bot: Any) -> str:
    """The runtime family a bot profile runs on: an agent bridge slug or ``chat``."""
    return durable_agent_backend(bot) or "chat"


def runtime_family_changed(before: Any | None, after: Any) -> bool:
    """True only for an update (not a create) that crosses runtime families."""
    return before is not None and runtime_family(before) != runtime_family(after)


def rotate_active_threads(engine, bot_id: str) -> list[tuple[str | None, str]]:
    """Rotate every user's active thread for ``bot_id``.

    Returns ``(user_id, new_session_id)`` pairs. All scopes rotate atomically
    under the bot admission lock, through the common lineage primitive. Only
    the successor of the pointed-to predecessor may move ``last_session_id``;
    iterating other users must not make the last loop iteration the active bot
    context. A null or explicitly archived pointer is not guessed or replaced.
    Provider binding remains lazy and there is no reset-confirmation message.
    """
    rotated: list[tuple[str | None, str]] = []
    with engine.begin() as conn:
        lock_target(conn, bot_id)
        accepted = BotSessionState.resolve(conn, bot_id=bot_id)
        active = conn.execute(text("""
            SELECT id, user_id FROM sessions
            WHERE bot_id=:bot AND status='active' AND ended_at IS NULL
            ORDER BY user_id, id
        """), {"bot": bot_id}).mappings().all()
        pointed_owner = next(
            (row for row in active if accepted and row["id"] == accepted["id"]), None,
        )
        users = dict.fromkeys(row["user_id"] for row in active)
        for user_id in users:
            new_id = str(uuid.uuid4())
            SessionResetLineage.rotate(
                None, connection=conn, bot_id=bot_id, user_id=user_id, new_id=new_id,
                accept_pointer=bool(pointed_owner and pointed_owner["user_id"] == user_id),
                # A runtime switch is not a suppression request; it pins (N1).
                working_memory_mode=PINNED,
                context_lifecycle={"action": RUNTIME_SWITCH_SOURCE},
            )
            rotated.append((user_id, new_id))
    if rotated:
        log.info(
            "Runtime family changed for bot=%s: rotated %d active thread(s)",
            bot_id, len(rotated),
        )
    return rotated
