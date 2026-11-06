"""Where each bot's new turns run (TASK-2116).

Mirrors the dispatcher (``agent_bridge.houses.route_new_command``): a bot with
a ``house`` whose lease for its backend is live runs in that house; otherwise
its backend's mansion bridge consumes the shared command stream. Chat bots
run inside the app. The model or provider never decides this: a GPT bot on
the ``claude-code`` backend runs on ``claude-code-bridge``.

``location`` is one of:

* ``{"runtime": "claude-code-bridge"}`` / ``{"runtime": "codex-bridge"}``
* ``{"runtime": "house", "house": "h1"}`` (``"pinned": true`` when house_pinned)
* ``{"runtime": "app"}`` for chat bots
* a housed bot whose house has no live lease reports its mansion bridge plus
  ``"house": "h1", "house_live": false`` (new turns fall back there); when the
  lease can't be read, ``"runtime": "house"`` with ``"house_live": null``.

Only housed agent bots cost a read: one Redis ``GET`` each, none otherwise.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from agent_bridge.houses import lease_key, normalize_house

from .bot_types import normalize_bot_type

log = logging.getLogger(__name__)

BRIDGES = {"claude-code": "claude-code-bridge", "codex": "codex-bridge",
           "local": "local-model-bridge"}

LeaseReader = Callable[[str], Awaitable[Any]]


def _bridge(backend: str) -> str:
    """The service running this backend; an unmapped backend reports its own name."""
    return BRIDGES.get(backend, backend)


def location_runtimes(location: dict) -> set[str]:
    """Every name a ``location`` filter may match: runtime, and house id."""
    names = {location["runtime"]}
    if location.get("house") and location.get("house_live") is not False:
        names.add(location["house"])
    return names


def tenancy(bot_type: Any, agent_backend: Any, house: Any) -> tuple[str, str] | None:
    """``(backend, house)`` the dispatcher routes by; None for a chat bot."""
    backend = (agent_backend or "").strip().lower()
    if normalize_bot_type(bot_type, backend) != "agent" or not backend:
        return None
    try:
        return backend, normalize_house(house)
    except ValueError:
        return backend, ""  # the dispatcher sends an invalid house to the mansion


def location_of(bot_type: Any, agent_backend: Any, house: Any, pinned: Any,
                live: bool | None) -> dict:
    """One bot's location given its house lease liveness (None: unknown)."""
    routed = tenancy(bot_type, agent_backend, house)
    if routed is None:
        return {"runtime": "app"}
    backend, house = routed
    if not house:
        return {"runtime": _bridge(backend)}
    pin = {"pinned": True} if pinned else {}
    if live is False:
        return {"runtime": _bridge(backend), "house": house, "house_live": False, **pin}
    if live is None:
        return {"runtime": "house", "house": house, "house_live": None, **pin}
    return {"runtime": "house", "house": house, **pin}


async def resolve_locations(rows: Iterable[Any], read_lease: LeaseReader | None
                            ) -> dict[str, dict]:
    """``{slug: location}``; ``read_lease(key)`` is a Redis GET (None: unreadable)."""
    out: dict[str, dict] = {}
    for row in rows:
        routed = tenancy(getattr(row, "bot_type", None), getattr(row, "agent_backend", None),
                         getattr(row, "house", None))
        live: bool | None = None
        if routed and routed[1] and read_lease is not None:
            try:
                live = bool(await read_lease(lease_key(routed[1], routed[0])))
            except Exception:
                log.warning("bot location: lease for %s unreadable", routed[1], exc_info=True)
        out[row.slug] = location_of(getattr(row, "bot_type", None),
                                    getattr(row, "agent_backend", None),
                                    getattr(row, "house", None),
                                    getattr(row, "house_pinned", False), live)
    return out


async def read_locations(rows: list[Any]) -> dict[str, dict]:
    """``resolve_locations`` with the app's Redis, opened only if a bot is housed."""
    if not any(getattr(row, "house", None) for row in rows):
        return await resolve_locations(rows, None)
    client = None
    try:
        import redis.asyncio as aioredis

        from .utils.config import Config

        url = Config().REDIS_URL
        client = aioredis.from_url(url, decode_responses=True) if url else None
    except Exception:
        log.warning("bot location: redis unavailable", exc_info=True)
    try:
        return await resolve_locations(rows, client.get if client is not None else None)
    finally:
        if client is not None:
            await client.aclose()
