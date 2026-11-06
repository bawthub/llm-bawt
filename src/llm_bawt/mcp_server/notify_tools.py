"""``notify_user``: push to the user's phone through Home Assistant (TASK-1531).

The caller must be a signed turn (or approved/service) identity for
``bot_id``: no credential, one that doesn't open, or one for another bot is
refused rather than trusting the ``bot_id`` argument. Private and
unknown bots are refused, and an unreadable tag snapshot refuses everyone.
Every call, refused or not, writes one ``notify_log`` audit row. A push that
reached Home Assistant also gets one BawtHub bell entry (TASK-2002).
"""
from __future__ import annotations

import asyncio
from typing import Annotated, Literal

import httpx
from pydantic import Field

from llm_bawt.bot_access import BotAccessError
from llm_bawt.bot_tags import BehaviorTag, BotTags
from llm_bawt.integrations import notify as notify_module
from llm_bawt.integrations import notify_bell
from llm_bawt.integrations.notify import Notifier
from llm_bawt.integrations.notify_store import NotifyStore
from llm_bawt.utils.config import Config
from .registry import mcp

Identifier = Annotated[str, Field(min_length=1, max_length=64)]


def _store(config) -> NotifyStore:
    from llm_bawt.utils.db import get_shared_engine
    return NotifyStore(get_shared_engine(config))


def _http() -> httpx.AsyncClient:
    return httpx.AsyncClient()


def _bell_http() -> httpx.AsyncClient:
    return notify_bell.bell_http()


def _check(bot_id: str, arguments: dict):
    """``(refusal, caller)``: why this caller may not notify as ``bot_id``, or None."""
    from .bot_access_identity import trusted_caller
    try:
        caller = trusted_caller("notify_user", arguments)
    except BotAccessError:
        return "refused:caller_unverified", None
    if caller is None:
        # A push reaches a person's phone: an unsigned client never gets to
        # speak as a bot by naming it.
        return "refused:caller_unverified", None
    if caller.bot_id != bot_id:
        return "refused:caller_mismatch", None
    tags = BotTags().snapshot()
    if not tags.available:
        return "refused:policy_unavailable", None
    if bot_id not in tags.known_bots:
        return "refused:unknown_bot", None
    if tags.has(bot_id, BehaviorTag.PRIVATE):
        return "refused:private_bot", None
    return None, caller


async def _record_bell(receipt: dict, *, bot_id: str, user_id: str | None, message: str,
                       title: str | None, urgency: str, link: str | None) -> None:
    """Best-effort bell entry for a delivered push; never changes the receipt."""
    if not notify_bell.delivered(str(receipt.get("result", ""))) or not receipt.get("id"):
        return
    if not user_id:
        notify_bell.logger.warning("notify bell %s not recorded: no caller user", receipt["id"])
        return
    try:
        sender = await asyncio.to_thread(notify_module.sender_name, bot_id)
        async with _bell_http() as http:
            await notify_bell.record_push_bell(
                http, push_id=receipt["id"], bot_id=bot_id, user_id=user_id,
                title=notify_module.push_title(bot_id, title, sender), message=message,
                urgency=urgency, link=link, result=receipt["result"])
    except Exception as error:  # the push already happened
        notify_bell.logger.warning("notify bell %s not recorded: %s", receipt["id"],
                                   type(error).__name__)


@mcp.tool()
async def notify_user(
    bot_id: Identifier,
    message: Annotated[str, Field(min_length=1, max_length=1000)],
    title: Annotated[str | None, Field(max_length=120)] = None,
    urgency: Literal["normal", "high"] = "normal",
    link: Annotated[str | None, Field(max_length=2000)] = None,
    dedupe_key: Annotated[str | None, Field(max_length=128)] = None,
) -> dict:
    """Push a phone notification to the user through Home Assistant."""
    if not message.strip():
        raise ValueError("message must not be blank")
    bot_id = bot_id.strip().lower()
    arguments = {"bot_id": bot_id, "message": message, "title": title, "urgency": urgency,
                 "link": link, "dedupe_key": dedupe_key}
    config = Config()
    store = await asyncio.to_thread(_store, config)
    refusal, caller = await asyncio.to_thread(_check, bot_id, arguments)
    if refusal:
        row = await asyncio.to_thread(store.record, bot_id, urgency, message, dedupe_key, refusal)
        return {"result": refusal, "id": row}
    async with _http() as http:
        receipt = await Notifier(config, store, http).notify(
            bot_id, message, title=title, urgency=urgency, link=link, dedupe_key=dedupe_key)
    await _record_bell(receipt, bot_id=bot_id, user_id=getattr(caller, "user_id", None),
                       message=message, title=title, urgency=urgency, link=link)
    return receipt


NotificationKind = Literal["bot_reply", "push", "approval", "agent_task", "alert", "other"]


def _caller_user(arguments: dict) -> str | None:
    """The signed caller's user: whose bell this reads. None when unverified."""
    from .bot_access_identity import trusted_caller
    try:
        caller = trusted_caller("notifications_list", arguments)
    except BotAccessError:
        return None
    return getattr(caller, "user_id", None) if caller is not None else None


@mcp.tool()
async def notifications_list(
    bot_id: Identifier,
    kind: NotificationKind | None = None,
    since: str | None = None,
    limit: Annotated[int, Field(ge=1, le=100)] = 20,
) -> dict:
    """Read what the user's notification bell recorded (TASK-2066). Read-only."""
    arguments = {"bot_id": bot_id, "kind": kind, "since": since, "limit": limit}
    user_id = await asyncio.to_thread(_caller_user, arguments)
    if not user_id:
        return {"error": "caller_unverified"}
    params = {"user": user_id, "limit": str(limit)}
    for key, value in (("kind", kind), ("since", since)):
        if value:
            params[key] = value
    async with _bell_http() as http:
        response = await http.get("/internal/notifications", params=params)
    if not response.is_success:
        try:
            detail = response.json().get("error")
        except ValueError:
            detail = None
        return {"error": detail or f"HTTP {response.status_code}"}
    return response.json()
