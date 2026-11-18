"""Skip one-time task-tagged bot wake-ups whose task has moved on (TASK-2056).

A bot's wake-up ("check on TASK-N later") is stale when, by the time it fires,
the task is finished, in REVIEW, archived, or was handed off: the target
owned it when the wake-up was made and no longer does. Recurring and
user-created schedules are exempt: only one-time wake-ups with a proven
creator bot and a task are checked.

Task state lives in BawtHub, so it is read outside the outbox locks. Any
lookup failure fails open: the wake-up fires as before.

A skipped wake-up is not silent (TASK-2294): its creator bot gets one system
notice with the reason and the original prompt, so work the wake-up was also
carrying (a window, a follow-up) is not stranded.
"""
from __future__ import annotations

import logging
from urllib.parse import quote

import httpx

from ..bawthub_auth import service_auth

logger = logging.getLogger(__name__)

_LOOKUP_TIMEOUT_S = 3.0
SKIPPED_NOTICE_KEY = "schedule.wakeup_skipped"
DEFAULT_SKIPPED_NOTICE = (
    "[Wake-up skipped: a one-time wake-up you scheduled did not run because its "
    "task moved on. The JSON below gives the reason, the task and the prompt it "
    "would have sent. If that prompt still carries work that isn't finished or "
    "handed off, do it now or reschedule it untagged; otherwise no action or "
    "reply is needed.]"
)
_DONE = {"COMPLETED": "completed", "CANCELLED": "cancelled", "REVIEW": "in REVIEW"}


def checked(snapshot: dict, kind: str) -> bool:
    """Whether an occurrence is subject to the stale check at all."""
    return bool(
        kind == "scheduled"
        and snapshot.get("creator_bot_id")
        and snapshot.get("task_id")
        and (snapshot.get("timing") or {}).get("kind") == "once"
    )


def stale_reason(snapshot: dict, task: dict | None) -> str | None:
    """Why this wake-up should not fire, or None to fire it."""
    if not isinstance(task, dict):
        return None
    ref = task.get("shortId") or snapshot.get("task_id")
    status = str(task.get("status") or "").upper()
    if task.get("archivedAt"):
        return f"Skipped: {ref} is archived"
    if status in _DONE:
        return f"Skipped: {ref} is {_DONE[status]}"
    # Only a hand-off skips: the target owned the task when the wake-up was
    # made and no longer does. A contributor's wake-up (reviewer, integrator)
    # on someone else's task keeps firing.
    then = str(snapshot.get("task_owner_bot_id") or "").strip().lower()
    owner = str(task.get("agentBotId") or "").strip().lower()
    target = str(snapshot.get("bot_id") or "").strip().lower()
    if then and then == target and owner != target:
        return f"Skipped: {ref} is now owned by {owner or 'nobody'}, not {target}"
    return None


def fetch_task(task_ref: str) -> dict | None:
    """The task from BawtHub, or None when it can't be read (fail open)."""
    from ..mcp_server.task_api import API_PREFIX, BASE_URL

    try:
        with httpx.Client(base_url=BASE_URL, timeout=_LOOKUP_TIMEOUT_S, auth=service_auth()) as client:
            response = client.get(f"{API_PREFIX}/tasks/{quote(task_ref, safe='')}")
        if response.status_code != 200:
            return None
        body = response.json()
        return body if isinstance(body, dict) else None
    except Exception:
        logger.warning("Wake-up task lookup failed for %s", task_ref, exc_info=True)
        return None


def wakeup_stale_reason(snapshot: dict, kind: str) -> str | None:
    if not checked(snapshot, kind):
        return None
    return stale_reason(snapshot, fetch_task(str(snapshot["task_id"])))


def skipped_notice(snapshot: dict, run_id: str, reason: str) -> str:
    """The notice body for the creator of a skipped wake-up."""
    import json

    body = DEFAULT_SKIPPED_NOTICE
    creator = snapshot.get("creator_bot_id")
    try:
        from ..prompt_registry import get_prompt_resolver

        resolved = get_prompt_resolver().resolve(
            SKIPPED_NOTICE_KEY, scope_type="bot", scope_id=creator,
        )
        if resolved and (resolved.body or "").strip():
            body = resolved.body
    except Exception as error:  # a registry failure must not drop the notice
        logger.warning("Prompt %s resolve failed, using seed: %s", SKIPPED_NOTICE_KEY, error)
    facts = {
        "reason": reason,
        "task_id": snapshot.get("task_id"),
        "occurrence_id": run_id,
        "scheduled_for": snapshot.get("scheduled_for"),
        "prompt": snapshot.get("prompt"),
    }
    return body + "\n" + json.dumps(facts, ensure_ascii=False, sort_keys=True, default=str)
