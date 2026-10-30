"""``schedule_wakeup``: durable bot wake-ups over prompt schedules (TASK-1719).

Harness timers (Claude Code ``CronCreate``) live in one SDK session and die on
restart, ``/new`` or compaction. This tool writes a row in the durable
prompt-schedule system (``/v1/prompt-schedules``, the Schedules page) instead,
so the wake-up survives all of those.

The owning user and the creating bot come from the signed turn, never from
arguments: an unsigned client can't schedule, and a bot can't schedule as
another bot. Wake-ups are ordinary schedules owned by the caller's user (they
show on the Schedules page) whose name carries ``wakeup:<creator>:``. The
server proves and persists that creator (``creator_bot_id``), and runs are
authored by it (TASK-1790); list and cancel only see wake-ups the caller
created or that target the caller, never the user's own hand-made schedules.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Annotated, Any, Literal
from urllib.parse import quote
from uuid import uuid4

from pydantic import Field

from llm_bawt.bot_access import BotAccessError
from llm_bawt.bot_tags import BehaviorTag, BotTags
from .registry import mcp

logger = logging.getLogger(__name__)

_APP_BASE_URL = os.getenv("LLM_BAWT_APP_BASE_URL", "http://localhost:8642").rstrip("/")
_TIMEOUT_S = 30.0
_MISFIRE_GRACE_S = 604800
_TASK_RE = re.compile(r"^TASK-\d+$")
Identifier = Annotated[str, Field(min_length=1, max_length=64)]


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _error(error: str, message: str, **extra: Any) -> str:
    return _dump({"error": error, "message": message, **extra})


def _now() -> datetime:
    return datetime.now(dt_timezone.utc)


def _caller(arguments: dict):
    from .bot_access_identity import trusted_caller
    try:
        caller = trusted_caller("schedule_wakeup", arguments)
    except BotAccessError:
        return None
    if caller is None or not caller.bot_id or not caller.user_id:
        return None
    return caller


def _target_refusal(target: str, caller_bot: str) -> str | None:
    if target == caller_bot:
        return None
    tags = BotTags().snapshot()
    if not tags.available:
        return "bot policy is unavailable; try again"
    if target not in tags.known_bots:
        return f"unknown bot {target!r}"
    if tags.has(target, BehaviorTag.PRIVATE):
        return f"{target!r} is private; only it can schedule its own wake-ups"
    return None


async def _request(method: str, path: str, user: str, body: dict | None = None,
                   params: dict | None = None):
    import httpx

    async with httpx.AsyncClient() as client:
        return await client.request(method, f"{_APP_BASE_URL}{path}", json=body, timeout=_TIMEOUT_S,
                                    params={"user": user, **(params or {})})


async def _call(method: str, path: str, user: str, body: dict | None = None,
                params: dict | None = None, *, unknown: str) -> tuple[Any, str | None]:
    try:
        response = await _request(method, path, user, body, params)
    except Exception:
        logger.exception("schedule_wakeup call failed: %s %s", method, path)
        return None, _error("unavailable", unknown)
    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text}
    if response.status_code >= 400:
        detail = payload.get("detail") if isinstance(payload, dict) else payload
        return None, _error("rejected", str(detail), status_code=response.status_code)
    return payload, None


def _creator(schedule: dict) -> str | None:
    """The bot that created this wake-up, or None if it isn't provably one.

    The server proves it once at create (``proven_wakeup_creator``) and
    persists it; the user-editable name is never consulted here (TASK-1790).
    """
    return schedule.get("creator_bot_id") or None


def _visible(schedule: dict, bot: str) -> bool:
    creator = _creator(schedule)
    return creator is not None and bot in {creator, schedule.get("bot_id")}


def _summary(schedule: dict) -> dict:
    return {key: schedule.get(key) for key in (
        "id", "bot_id", "name", "description", "next_run_at", "lifecycle", "timing")}


def _timing(at: str | None, in_minutes: int | None, cron: str | None,
            timezone_name: str) -> tuple[dict | None, str | None]:
    if sum(value is not None for value in (at, in_minutes, cron)) != 1:
        return None, "give exactly one of at, in_minutes or cron"
    if cron is not None:
        return {"kind": "cron", "cron_expression": cron, "timezone": timezone_name}, None
    if in_minutes is not None:
        when = _now() + timedelta(minutes=in_minutes)
    else:
        try:
            when = datetime.fromisoformat(at.strip().replace("Z", "+00:00"))
        except ValueError:
            return None, "at must be an ISO-8601 time with a UTC offset"
        if when.tzinfo is None:
            return None, "at needs a UTC offset (e.g. 2026-10-30T07:17:00-04:00)"
    return {"kind": "once", "once_at": when.isoformat(), "timezone": timezone_name}, None


@mcp.tool(name="schedule_wakeup")
async def schedule_wakeup(
    action: Literal["create", "list", "cancel"] = "create",
    message: Annotated[str | None, Field(max_length=20000)] = None,
    target_bot_id: Annotated[str | None, Field(max_length=64)] = None,
    at: Annotated[str | None, Field(max_length=64)] = None,
    in_minutes: Annotated[int | None, Field(ge=1, le=525600)] = None,
    cron: Annotated[str | None, Field(max_length=120)] = None,
    timezone: Annotated[str, Field(min_length=1, max_length=64)] = "UTC",
    task_id: Annotated[str | None, Field(max_length=32)] = None,
    label: Annotated[str | None, Field(max_length=80)] = None,
    idempotency_key: Annotated[str | None, Field(max_length=128)] = None,
    schedule_id: Annotated[str | None, Field(max_length=64)] = None,
) -> str:
    """Durable wake-up: a future message to a bot that survives restarts, /new and compaction.

    create: message plus one of at (ISO with offset), in_minutes or cron
    (recurring, in timezone). target_bot_id defaults to you. Runs in a fresh
    context, no memory augment/extract; a run missed by up to 7 days fires
    once on recovery. Reusing idempotency_key returns the original.
    list / cancel(schedule_id): wake-ups you created or that target you.
    """
    arguments = {"action": action, "message": message, "target_bot_id": target_bot_id, "at": at,
                 "in_minutes": in_minutes, "cron": cron, "timezone": timezone,
                 "task_id": task_id, "label": label, "idempotency_key": idempotency_key,
                 "schedule_id": schedule_id}
    caller = _caller(arguments)
    if caller is None:
        return _error("unauthorized", "a signed bot turn is required")
    bot, user = caller.bot_id.strip().lower(), caller.user_id.strip().lower()
    from llm_bawt.service.prompt_schedule_management import (
        WAKEUP_PREFIX, schedule_identifier, wakeup_create_key,
    )

    if action == "list":
        payload, error = await _call("GET", "/v1/prompt-schedules", user,
                                     params={"search": WAKEUP_PREFIX, "lifecycle": "active", "limit": 100},
                                     unknown="the schedule list is unavailable; try again")
        if error:
            return error
        mine = [_summary(s) for s in payload.get("schedules", []) if _visible(s, bot)]
        return _dump({"wakeups": mine})

    if action == "cancel":
        if not schedule_id or not schedule_id.strip():
            return _error("invalid_request", "schedule_id is required")
        path = f"/v1/prompt-schedules/{quote(schedule_id.strip(), safe='')}"
        current, error = await _call("GET", path, user, unknown="the schedule is unavailable; try again")
        if error:
            return error
        if not _visible(current, bot):
            return _error("forbidden", "only wake-ups you created or that target you can be cancelled")
        payload, error = await _call("DELETE", path, user,
                                     unknown="the cancel's outcome is unknown; check action=list")
        return error or _dump(_summary(payload))

    if not message or not message.strip():
        return _error("invalid_request", "message is required")
    target = (target_bot_id or bot).strip().lower()
    refusal = _target_refusal(target, bot)
    if refusal:
        return _error("refused", refusal)
    task = task_id.strip().upper() if task_id and task_id.strip() else None
    if task and not _TASK_RE.match(task):
        return _error("invalid_request", "task_id must look like TASK-123")
    timing, problem = _timing(at, in_minutes, cron, timezone)
    if problem:
        return _error("invalid_request", problem)
    replayable = bool(idempotency_key and idempotency_key.strip())
    nonce = (hashlib.sha256(idempotency_key.strip().encode()).hexdigest()[:16]
             if replayable else uuid4().hex[:16])
    key = wakeup_create_key(bot, nonce)
    if replayable:
        # A retry returns the original even when in_minutes moved once_at.
        existing, _ = await _call("GET", f"/v1/prompt-schedules/{schedule_identifier(user, key)}", user,
                                  unknown="")
        if existing is not None:
            return _dump({**_summary(existing), "duplicate": True})
    header = f"[Scheduled wake-up from {bot}" + (f", {task}" if task else "") + "]"
    name = f"{WAKEUP_PREFIX}{bot}:{nonce}:{(label or task or 'wake-up').strip()}"[:120]
    body = {
        "name": name, "description": f"task_id={task}" if task else "",
        "prompt": f"{header}\n\n{message.strip()}", "bot_id": target, "target_mode": "specific",
        "timing": timing, "clear_context": True, "augment_memory": False, "extract_memory": False,
        # A late wake-up beats none: missed runs fire on recovery for up to
        # the API's maximum grace (7 days), not the 1h UI default.
        "missed_policy": "run_latest", "misfire_grace_seconds": _MISFIRE_GRACE_S,
        "idempotency_key": key,
    }
    payload, error = await _call("POST", "/v1/prompt-schedules", user, body,
                                 unknown="the create's outcome is unknown; check action=list")
    return error or _dump(_summary(payload))
