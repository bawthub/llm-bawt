"""Strict bot -> endpoint -> usage access-path resolution.

A model vendor or harness is never evidence of a subscription credential.
Legacy bots without an endpoint binding report unavailable instead of borrowing
another route's account usage.
"""

from __future__ import annotations

from typing import Any


def bot_access_path(config: Any, bot_id: str) -> str | None:
    """Return the bot's current catalog access-path key, or None when unresolved."""
    from ...bots import BotManager

    try:
        bot = BotManager(config).get_bot(bot_id)
        if bot is None:
            return None
        catalog = config.ensure_model_catalog()
        endpoint_id = getattr(bot, "endpoint_id", None)
        if endpoint_id is not None:
            endpoint = catalog.resolve_endpoint(endpoint_id, harness=getattr(bot, "harness", None))
        else:
            # A full endpoint ref is unambiguous; a legacy bare alias isn't.
            endpoint = catalog.endpoint_by_ref(getattr(bot, "default_model", None))
        return endpoint.access_path.key if endpoint is not None else None
    except Exception:  # noqa: BLE001 — missing profile/catalog is not Claude OAuth
        return None
