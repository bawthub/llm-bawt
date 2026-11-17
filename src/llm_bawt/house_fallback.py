"""Tell the user a house was down and new work ran in the mansion (TASK-2077).

Called by :func:`agent_bridge.houses.route_new_command` at most once per
house/backend outage (Redis ``SET NX`` dedupe there). Best-effort: a failed
notice never blocks the send, which has already been routed to the mansion.
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def push_system_notice(*, bot_id: str, user_id: str | None, title: str, message: str,
                             dedupe_key: str, urgency: str = "high") -> dict:
    """Push through the user's notification settings and, once delivered,
    record it in their bell. Returns the push receipt."""
    import httpx

    from .integrations import notify as notify_module
    from .integrations import notify_bell
    from .integrations.notify import Notifier
    from .integrations.notify_store import NotifyStore
    from .utils.config import Config
    from .utils.db import get_shared_engine

    config = Config()
    store = await asyncio.to_thread(lambda: NotifyStore(get_shared_engine(config)))
    async with httpx.AsyncClient() as http:
        receipt = await Notifier(config, store, http).notify(
            bot_id, message, title=title, urgency=urgency, link=None, dedupe_key=dedupe_key)
    if not user_id or not notify_bell.delivered(str(receipt.get("result", ""))) \
            or not receipt.get("id"):
        return receipt
    async with notify_bell.bell_http() as http:
        await notify_bell.record_push_bell(
            http, push_id=receipt["id"], bot_id=bot_id, user_id=user_id,
            title=notify_module.push_title(bot_id, title, None), message=message,
            urgency=urgency, link=None, result=receipt["result"])
    return receipt


def fallback_notifier(user_id: str | None):
    """A ``FallbackNotifier`` that pushes to the user and records a bell entry."""

    async def notify(house: str, backend: str, bot_id: str) -> None:
        await push_system_notice(
            bot_id=bot_id or "system", user_id=user_id, title=f"House {house} is down",
            message=(f"{bot_id or 'A bot'}'s new work is running in the mansion because house "
                     f"{house} ({backend}) stopped checking in. Turns already running there "
                     "were not moved."),
            dedupe_key=f"house-fallback:{house}:{backend}")

    return notify


def house_fallback_for(config: dict):
    """The fallback notice for a housed bot's config; None for the mansion."""
    if not config.get("house"):
        return None
    return fallback_notifier(str(config.get("user_id") or "") or None)
