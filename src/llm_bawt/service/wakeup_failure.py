"""Tell a bot when its wake-up failed instead of leaving it stopped (TASK-2436).

A one-time wake-up is often a bot's only wake signal. When its occurrence ends
failed or unknown, the run's completion receipt only reaches the bot on some
later turn, so nothing restarts it. One system notice delivery per recipient
fixes that: the target bot, the bot that created the wake-up and the task's
owner at creation time, each once.

The notice is a notification, never a retry. A failure caused by the wake-up's
own content or a persistent provider error would fail again, and a wake-up that
partly ran must not replay. The notice is an ordinary delivery, not a schedule,
so it cannot loop back into this path.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

FAILED_NOTICE_KEY = "schedule.wakeup_failed"
FAILED_STATES = frozenset({"failed", "unknown"})
ERROR_CHARS = 300
PROMPT_CHARS = 400
DEFAULT_FAILED_NOTICE = (
    "[Wake-up failed — notification, not a retry]\n"
    "A wake-up scheduled by a bot did not complete, and it was not retried or "
    "replayed. The JSON below gives the schedule, task, state, error and the "
    "start of the prompt it carried. State \"unknown\" means the turn may have "
    "partly run, so check task evidence before repeating any side effect. If "
    "that prompt still carries work that is yours and isn't finished or handed "
    "off, do it now or schedule a new wake-up; otherwise no action or reply is "
    "needed."
)


def recipients(snapshot: dict) -> list[str]:
    """Bots to tell, deduplicated, or none for a wake-up no bot created."""
    if not snapshot.get("creator_bot_id"):
        return []
    found = [snapshot.get(key) for key in ("bot_id", "creator_bot_id", "task_owner_bot_id")]
    return list(dict.fromkeys(str(bot).strip() for bot in found if bot and str(bot).strip()))


def failed_notice(snapshot: dict, run_id: str, state: str, error: str | None,
                  recipient: str) -> str:
    """The notice body for one recipient of a failed wake-up."""
    from .wakeup_staleness import notice_body

    prompt = str(snapshot.get("prompt") or "")
    facts = {
        "state": state,
        "error": str(error or "")[:ERROR_CHARS],
        "schedule": str(snapshot.get("name") or "")[:120],
        "task_id": snapshot.get("task_id"),
        "target_bot_id": snapshot.get("bot_id"),
        "occurrence_id": run_id,
        "scheduled_for": snapshot.get("scheduled_for"),
        "prompt_excerpt": prompt[:PROMPT_CHARS],
        "prompt_truncated": len(prompt) > PROMPT_CHARS,
        "retried": False,
    }
    body = notice_body(FAILED_NOTICE_KEY, DEFAULT_FAILED_NOTICE, recipient)
    return body + "\n" + json.dumps(facts, ensure_ascii=False, sort_keys=True, default=str)
