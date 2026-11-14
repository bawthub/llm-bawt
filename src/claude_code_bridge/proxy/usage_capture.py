"""Capture passive Codex plan-usage fallback from response headers.

The app normally refreshes limits through the official ``/wham/usage``
endpoint. Real ``/responses`` calls also carry equivalent ``x-codex-*``
headers, so the bridge passively saves them as a fallback when that dedicated
endpoint is temporarily unavailable. We peek those headers off the SDK stream
object (already buffered — no extra request, no body consumed) and stash a
canonical snapshot in Redis for the app's ``/v1/usage`` endpoint.

This must NEVER affect inference: ``schedule_capture`` is synchronous, reads
the (already-present) headers inline, and fire-and-forgets only the Redis
write. All errors are swallowed.

Redis key is shared with the reader at
``src/llm_bawt/service/usage/adapters/openai_chatgpt.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

# Shared contract with the app-side usage adapter. Keep both in lockstep.
from llm_bawt.shared.chatgpt_usage import account_scope, snapshot_key

logger = logging.getLogger(__name__)

_TTL_SECONDS = 7 * 24 * 3600  # drop the snapshot if no codex turn for a week

_redis = None
_tasks: set = set()


def _client():
    global _redis
    if _redis is None:
        import redis.asyncio as redis

        url = (
            os.getenv("REDIS_URL")
            or os.getenv("LLM_BAWT_REDIS_URL")
            or "redis://redis:6379/0"
        )
        _redis = redis.from_url(
            url,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=2,
            socket_connect_timeout=2,
        )
    return _redis


def _i(headers, key):
    v = headers.get(key)
    if v in (None, ""):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _b(headers, key):
    v = headers.get(key)
    if v is None:
        return None
    return str(v).strip().lower() in ("true", "1", "yes")


def _snapshot_from_headers(headers) -> dict | None:
    # Require at least one codex usage signal before publishing.
    if (
        headers.get("x-codex-primary-used-percent") is None
        and headers.get("x-codex-plan-type") is None
    ):
        return None
    return {
        "captured_at": int(time.time()),
        "plan_type": headers.get("x-codex-plan-type"),
        "active_limit": headers.get("x-codex-active-limit"),
        "primary": {
            "used_percent": _i(headers, "x-codex-primary-used-percent"),
            "window_minutes": _i(headers, "x-codex-primary-window-minutes"),
            "reset_at": _i(headers, "x-codex-primary-reset-at"),
            "reset_after_seconds": _i(headers, "x-codex-primary-reset-after-seconds"),
        },
        "secondary": {
            "used_percent": _i(headers, "x-codex-secondary-used-percent"),
            "window_minutes": _i(headers, "x-codex-secondary-window-minutes"),
            "reset_at": _i(headers, "x-codex-secondary-reset-at"),
            "reset_after_seconds": _i(headers, "x-codex-secondary-reset-after-seconds"),
        },
        "credits": {
            "has_credits": _b(headers, "x-codex-credits-has-credits"),
            "balance": headers.get("x-codex-credits-balance") or None,
            "unlimited": _b(headers, "x-codex-credits-unlimited"),
        },
    }


async def _publish(snap: dict) -> None:
    try:
        scope = snap.get("account_scope")
        if not scope:
            return
        await _client().set(snapshot_key(scope), json.dumps(snap), ex=_TTL_SECONDS)
    except Exception as e:  # noqa: BLE001
        logger.debug("codex usage snapshot publish skipped: %s", e)


def schedule_capture(stream) -> None:
    """Peek rate-limit headers off an openai ``AsyncStream`` and publish them
    to Redis without blocking inference. Safe to call on every turn."""
    try:
        resp = getattr(stream, "response", None)
        if resp is None:
            return
        snap = _snapshot_from_headers(resp.headers)
        if snap is None:
            return
        # Bind to the request that produced these headers, never credentials
        # that may have changed while its stream was running.
        scope = account_scope(resp.request.headers.get("ChatGPT-Account-Id"))
        if not scope:
            return
        snap["account_scope"] = scope
        task = asyncio.create_task(_publish(snap))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
    except Exception as e:  # noqa: BLE001
        logger.debug("codex usage snapshot capture skipped: %s", e)
