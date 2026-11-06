"""House tenancy lifecycle: /new vacates a house, which then recycles (TASK-2080).

A bot's ``house`` (``bot_profiles.house``) is its tenancy. A new session ends
it unless the reset asked to keep it (``keep_house``) or the profile pins it
(``house_pinned``). The idle janitor reaches this through its own scheduled
/new. Eviction happens inside the reset turn, after the bot's previous turn has
finished, so it never interrupts a running turn of this bot.

When the last tenant leaves, the app asks the house to restart itself over the
house's own command stream (:func:`agent_bridge.houses.request_recycle`). The
house drains any turn still admitted there before it exits, and Docker's
restart policy brings it back on a fresh container. The app needs no Docker
access. A house that never comes back has no lease, so new work falls back to
the mansion as before.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import text

from .bot_profile_changes import ProfileWriteConflict

logger = logging.getLogger(__name__)

SYSTEM_ACTOR = "system:house-lease"

_TENANTS_SQL = text("SELECT slug FROM bot_profiles WHERE house = :house ORDER BY slug")


def keep_house_requested(service: Any, delivery_id: str | None) -> bool:
    """Whether this reset's persisted intent asked to keep the house."""
    if not delivery_id:
        return False
    dispatcher = getattr(service, "_inter_bot_dispatcher", None)
    store = getattr(dispatcher, "store", None)
    if store is None:
        return False
    record = store.get(delivery_id)
    raw = ((getattr(record, "metadata", None) or {}).get("new_session_intent")
           if record is not None else None)
    return isinstance(raw, dict) and raw.get("keep_house") is True


def evict_for_reset(service: Any, llm_bawt: Any, bot_id: str, *,
                    delivery_id: str | None) -> str | None:
    """Vacate ``bot_id``'s house for this reset; return the house vacated.

    No-op for a mansion bot, a pinned bot, or a reset that keeps its house.
    The reset turn itself then runs in the mansion. A failure is logged and
    leaves the tenancy as it was: a reset never fails because of its house.
    """
    slug = (bot_id or "").strip().lower()
    cached = getattr(getattr(llm_bawt, "bot", None), "house", None)
    if not (isinstance(cached, str) and cached.strip()):
        # Profile writes invalidate this instance, so a mansion bot is known
        # here without a database read on every reset.
        return None
    try:
        from .service.dependencies import get_bot_profile_store

        store = get_bot_profile_store(service.config)
        row = store.get(slug)
        house = (getattr(row, "house", None) or "").strip() if row is not None else ""
        if not house or getattr(row, "house_pinned", False):
            return None
        if keep_house_requested(service, delivery_id):
            logger.info("bot %s keeps house %s through this reset", slug, house)
            return None
        store.upsert({
            "slug": slug, "house": None, "house_pinned": False,
            # Rechecked under the bot lock: a pin or house change that landed
            # after the read above refuses the eviction (ProfileFieldsChanged).
            "expected_fields": {"house": house, "house_pinned": False},
            "_changed_by": SYSTEM_ACTOR, "change_note": "new session vacated the house",
        }, require_existing=True)
    except ProfileWriteConflict as conflict:
        logger.info("bot %s tenancy changed during eviction (%s); kept", slug, conflict)
        return None
    except Exception:
        logger.exception("house eviction failed for %s; tenancy unchanged", slug)
        return None
    logger.info("bot %s vacated house %s on a new session", slug, house)
    _refresh_bot(service, llm_bawt, slug)
    if tenants(service.config, house):
        logger.info("house %s still has tenants; not recycling", house)
        return house
    request_recycle(service.config, house)
    return house


def tenants(config: Any, house: str) -> list[str]:
    from .utils.db import get_shared_engine

    with get_shared_engine(config).connect() as conn:
        return list(conn.execute(_TENANTS_SQL, {"house": house}).scalars())


def _refresh_bot(service: Any, llm_bawt: Any, slug: str) -> None:
    """Make the vacated profile authoritative for this and later turns."""
    try:
        from .bots import _check_reload, invalidate_bots_cache

        invalidate_bots_cache()
        _check_reload()
        invalidate = getattr(service, "invalidate_bot_instances", None)
        if callable(invalidate):
            invalidate(slug)
        bot_config = getattr(getattr(llm_bawt, "client", None), "_bot_config", None)
        if isinstance(bot_config, dict):
            # The reset turn's own command is published from this instance.
            bot_config.pop("house", None)
        _publish(service.config, lambda client: _profile_changed(client, slug))
    except Exception:
        logger.exception("bot %s cache refresh after house eviction failed", slug)


def _profile_changed(client: Any, slug: str) -> None:
    from agent_bridge.publisher import UNIFIED_EVENTS_PREFIX, UNIFIED_STREAM_MAXLEN

    client.xadd(
        f"{UNIFIED_EVENTS_PREFIX}system:__config__",
        {"payload": json.dumps({"_type": "bot_profile_changed", "bot_id": slug})},
        maxlen=UNIFIED_STREAM_MAXLEN, approximate=True,
    )


def request_recycle(config: Any, house: str) -> list[str]:
    """Ask a vacated house to restart itself; best-effort, returns backends asked."""
    from agent_bridge.houses import commands_stream, lease_key, normalize_house, recycle_command

    from .house_tenancy import RECYCLE_MARKER_TTL_S, recycle_marker_key

    house = normalize_house(house)
    if not house:
        return []

    def send(client: Any) -> list[str]:
        asked = []
        prefix = lease_key(house, "")
        for key in client.scan_iter(match=f"{prefix}*"):
            backend = str(key)[len(prefix):]
            epoch = client.get(key)
            if backend and epoch:
                # Before the request: no self-move claims a house about to
                # restart at this epoch (TASK-2118). Its next epoch is free.
                client.set(recycle_marker_key(house, backend), str(epoch),
                           ex=RECYCLE_MARKER_TTL_S)
                client.xadd(commands_stream(house), recycle_command(house, backend, str(epoch)))
                asked.append(backend)
        return asked

    asked = _publish(config, send) or []
    logger.info("house %s recycle requested (backends=%s)", house, ",".join(asked) or "none live")
    return asked


def _publish(config: Any, action):
    try:
        import redis

        client = redis.Redis.from_url(
            config.REDIS_URL, decode_responses=True,
            socket_timeout=2.0, socket_connect_timeout=2.0,
        )
        try:
            return action(client)
        finally:
            client.close()
    except Exception:
        logger.exception("house lease Redis publish failed")
        return None
