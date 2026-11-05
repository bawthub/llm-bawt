"""Skip one-time task-tagged bot wake-ups whose task has moved on (TASK-2056).

A bot's wake-up ("check on TASK-N later") is stale when, by the time it fires,
the task is finished, in REVIEW, archived, or was handed off: the target
owned it when the wake-up was made and no longer does. Recurring and
user-created schedules are exempt: only one-time wake-ups with a proven
creator bot and a task are checked.

Task state lives in BawtHub, so it is read outside the outbox locks. Any
lookup failure fails open: the wake-up fires as before.
"""
from __future__ import annotations

import logging
from urllib.parse import quote

import httpx

from ..bawthub_auth import service_auth

logger = logging.getLogger(__name__)

_LOOKUP_TIMEOUT_S = 3.0
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
