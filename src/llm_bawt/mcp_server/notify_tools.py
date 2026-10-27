"""``notify_user``: push to the user's phone through Home Assistant (TASK-1531).

The caller is the signed turn (or approved/service) identity when one is
present; a supplied credential that doesn't open, or one for another bot,
is refused rather than trusted as the ``bot_id`` argument. Private and
unknown bots are refused, and an unreadable tag snapshot refuses everyone.
Every call, refused or not, writes one ``notify_log`` audit row.
"""
from __future__ import annotations

import asyncio
from typing import Annotated, Literal

import httpx
from pydantic import Field

from llm_bawt.bot_access import BotAccessError
from llm_bawt.bot_tags import BehaviorTag, BotTags
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


def _refusal(bot_id: str, arguments: dict) -> str | None:
    """Why this caller may not notify as ``bot_id``, or None."""
    from .bot_access_identity import trusted_caller
    try:
        caller = trusted_caller("notify_user", arguments)
    except BotAccessError:
        return "refused:caller_unverified"
    if caller is not None and caller.bot_id != bot_id:
        return "refused:caller_mismatch"
    tags = BotTags().snapshot()
    if not tags.available:
        return "refused:policy_unavailable"
    if bot_id not in tags.known_bots:
        return "refused:unknown_bot"
    if tags.has(bot_id, BehaviorTag.PRIVATE):
        return "refused:private_bot"
    return None


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
    refusal = await asyncio.to_thread(_refusal, bot_id, arguments)
    if refusal:
        row = await asyncio.to_thread(store.record, bot_id, urgency, message, dedupe_key, refusal)
        return {"result": refusal, "id": row}
    async with _http() as http:
        return await Notifier(config, store, http).notify(
            bot_id, message, title=title, urgency=urgency, link=link, dedupe_key=dedupe_key)
