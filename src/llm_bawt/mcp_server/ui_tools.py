"""``ui_navigate``: move the user's open app view to a page (TASK-2626).

The caller is the signed caller, never the ``bot_id`` argument alone; the
request goes only to that caller's user. See :mod:`llm_bawt.ui_navigation` for
the event and acknowledgement contract.
"""
from __future__ import annotations

import asyncio
import logging

from llm_bawt import ui_navigation as nav
from llm_bawt.bot_access import BotAccessError

from .registry import mcp

log = logging.getLogger(__name__)

_RESULTS = {
    nav.NAVIGATED: "The user's app view now shows {path}.",
    nav.OFFERED: "The user was not moved; your chat shows them a link to {path}.",
    nav.NO_CLIENT: "No open app window took the request; the user was not moved. "
                   "The link stays in your chat.",
    nav.UNCONFIRMED: "An app window accepted the request but did not confirm the move; "
                     "don't assume the user now sees {path}. The link stays in your chat.",
}


def _caller(bot_id: str, arguments: dict):
    from .bot_access_identity import trusted_caller
    try:
        caller = trusted_caller("ui_navigate", arguments)
    except BotAccessError:
        return "caller_unverified", None
    if caller is None or not getattr(caller, "user_id", None):
        return "caller_unverified", None
    if caller.bot_id != bot_id:
        return "caller_mismatch", None
    return None, caller


async def _publish(event: dict) -> str:
    import redis.asyncio as aioredis

    from llm_bawt.utils.config import Config

    url = Config().REDIS_URL
    if not url:
        raise RuntimeError("live events unavailable")
    client = aioredis.from_url(url, decode_responses=True, socket_connect_timeout=2.0)
    try:
        return await nav.publish_and_wait(client, event)
    finally:
        await client.aclose()


@mcp.tool()
async def ui_navigate(bot_id: str, path: str, reason: str | None = None) -> dict:
    """Move the user's app view to an in-app page beside your chat."""
    # The signed caller check binds the call exactly as made, before cleanup.
    arguments = {"bot_id": bot_id, "path": path, "reason": reason}
    bot_id = (bot_id or "").strip().lower()
    reason = (reason or "").strip() or None
    if not bot_id or len(bot_id) > 64:
        return {"error": "bot_id must be your bot slug"}
    clean = nav.clean_path(path)
    if clean is None:
        return {"error": "path must be an in-app path such as /tasks?view=board"}
    if reason and len(reason) > nav.MAX_REASON:
        return {"error": f"reason must be at most {nav.MAX_REASON} characters"}
    refusal, caller = await asyncio.to_thread(_caller, bot_id, arguments)
    if refusal:
        return {"error": refusal}
    event = nav.new_event(bot_id=bot_id, user_id=caller.user_id, path=clean, reason=reason)
    try:
        outcome = await _publish(event)
    except Exception:
        log.warning("ui_navigate: live publish failed", exc_info=True)
        return {"error": "live events unavailable; the user was not moved"}
    return {"outcome": outcome, "path": clean, "reason": reason,
            "message": _RESULTS[outcome].format(path=clean)}
