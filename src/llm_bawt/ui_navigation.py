"""Autopilot navigation: a bot moves its user's open app view (TASK-2626).

The ``ui_navigate`` MCP tool publishes one ``ui_navigate`` event on the
bot/user stream that every open BawtHub tab already reads, then waits briefly
for a tab to report what it did. A tab answers through the app's ack route,
which pushes onto a per-request Redis list the tool is blocked on. Nothing
polls; an unanswered request simply means no tab showed it.

Outcomes: ``navigated`` (a wide tab with that bot's chat open moved its app
pane), ``offered`` (a tab showing the chat left the user in place; the tool
card in chat is the link) and ``no_client``.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from urllib.parse import unquote

EVENT_TYPE = "ui_navigate"
NAVIGATED = "navigated"
OFFERED = "offered"
NO_CLIENT = "no_client"
CLIENT_OUTCOMES = frozenset({NAVIGATED, OFFERED})

MAX_PATH = 512
MAX_REASON = 160
#: First answer wait, then a short extra wait so a tab that navigated wins
#: over one that only offered the link.
FIRST_ACK_SECONDS = 4.0
UPGRADE_ACK_SECONDS = 1.0
#: Request and ack keys outlive the wait so a late ack is accepted and dropped.
KEY_TTL_SECONDS = 60
#: Tabs ignore a replayed event older than this (stream resume after sleep).
#: Not the safety boundary (client clocks drift): a tab moves only after its
#: "navigated" ack is accepted, which stops once the tool has answered.
EVENT_TTL_SECONDS = 30

_REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f\\]")


def request_key(request_id: str) -> str:
    return f"ui_nav:req:{request_id}"


def ack_key(request_id: str) -> str:
    return f"ui_nav:ack:{request_id}"


def valid_request_id(value: str) -> bool:
    return bool(_REQUEST_ID.match(value or ""))


def clean_path(raw: str) -> str | None:
    """Return a same-origin app path, or None for anything else.

    Accepts ``/route``, optionally with ``?query`` and ``#anchor``. Refuses
    absolute and protocol-relative URLs, backslashes, control characters and
    the ``/api`` namespace (not a page).
    """
    path = (raw or "").strip()
    if not path or len(path) > MAX_PATH or not path.startswith("/"):
        return None
    if path.startswith("//") or _CONTROL.search(path):
        return None
    route = re.split(r"[?#]", path, maxsplit=1)[0]
    # Classify the route the browser will actually commit: refuse dot and
    # empty segments (raw or percent-encoded) and encoded separators, so no
    # prefix check can be walked out of (``/tasks/../api``).
    if re.search(r"%2f|%5c", route, re.I):
        return None
    try:
        decoded = unquote(route, errors="strict")
    except UnicodeDecodeError:
        return None
    segments = decoded.split("/")[1:]
    if any(s in (".", "..") for s in segments) or "" in segments[:-1] or _CONTROL.search(decoded):
        return None
    if decoded == "/api" or decoded.startswith("/api/"):
        return None
    return path


def new_event(*, bot_id: str, user_id: str, path: str, reason: str | None) -> dict:
    return {
        "_type": EVENT_TYPE,
        "request_id": uuid.uuid4().hex,
        "bot_id": bot_id,
        "user_id": user_id,
        "path": path,
        "reason": reason,
        "ts": time.time(),
        "expires_in": EVENT_TTL_SECONDS,
    }


def owner_value(bot_id: str, user_id: str) -> str:
    return json.dumps([bot_id, user_id])


async def publish_and_wait(client, event: dict) -> str:
    """Publish ``event`` and return the best outcome a tab reported."""
    from agent_bridge.user_events import publish_user_event

    request_id = event["request_id"]
    await client.set(request_key(request_id),
                     owner_value(event["bot_id"], event["user_id"]), ex=KEY_TTL_SECONDS)
    await publish_user_event(client, event["bot_id"], event["user_id"],
                             json.dumps(event, ensure_ascii=False))
    outcomes: list[str] = []
    first = await client.blpop([ack_key(request_id)], timeout=FIRST_ACK_SECONDS)
    if first:
        outcomes.append(first[1])
        if first[1] != NAVIGATED:
            upgrade = await client.blpop([ack_key(request_id)], timeout=UPGRADE_ACK_SECONDS)
            if upgrade:
                outcomes.append(upgrade[1])
    # Close the request before answering. A tab claims "navigated" with its ack
    # *before* it moves, and record_ack refuses once the request is gone, so a
    # tab that wakes later cannot move the user after this reports no move.
    # Acks accepted before the close are drained here and counted.
    await client.delete(request_key(request_id))
    outcomes.extend(await client.lrange(ack_key(request_id), 0, -1))
    await client.delete(ack_key(request_id))
    if NAVIGATED in outcomes:
        return NAVIGATED
    return OFFERED if outcomes else NO_CLIENT


async def record_ack(client, *, request_id: str, bot_id: str, user_id: str, outcome: str) -> bool:
    """Push one tab's outcome; False when the request is unknown, closed or not this user's.

    WATCH makes the owner check and the push atomic against the tool closing
    the request: an ack either lands before the close (and is counted) or is
    refused, and a refused "navigated" claim means the tab must not move.
    """
    from redis.exceptions import WatchError

    key = request_key(request_id)
    async with client.pipeline(transaction=True) as pipe:
        try:
            await pipe.watch(key)
            if await pipe.get(key) != owner_value(bot_id, user_id):
                return False
            pipe.multi()
            pipe.rpush(ack_key(request_id), outcome)
            pipe.expire(ack_key(request_id), KEY_TTL_SECONDS)
            await pipe.execute()
        except WatchError:
            return False
    return True
