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
from ..memory.session_reset_lineage import SessionResetLineage

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

    Returns ``(user_id, new_session_id)`` pairs. Each rotation is the shared
    ``/new`` primitive (archive + successor link in one transaction), so
    lineage and approval continuation behave exactly as after ``/new``.
    """
    with engine.connect() as conn:
        users = conn.execute(text("""
            SELECT DISTINCT user_id FROM sessions
            WHERE bot_id=:bot AND status='active' AND ended_at IS NULL
        """), {"bot": bot_id}).scalars().all()
    rotated: list[tuple[str | None, str]] = []
    for user_id in users:
        new_id = str(uuid.uuid4())
        SessionResetLineage.rotate(engine, bot_id=bot_id, user_id=user_id, new_id=new_id)
        rotated.append((user_id, new_id))
    if rotated:
        log.info(
            "Runtime family changed for bot=%s: rotated %d active thread(s)",
            bot_id, len(rotated),
        )
    return rotated
