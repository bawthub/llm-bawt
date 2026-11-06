"""House up/down/recreating from lease keyspace events (TASK-2120).

A house bridge holds ``agent:house:<H>:lease:<backend>`` (TTL 30s, renewed
every 10s, value = its epoch). The app subscribes to Redis keyspace events on
those keys, so no one polls:

* ``set``: a bridge took or renewed the lease. A house not up, or up under a
  different epoch (restarted between heartbeats), comes ``up``.
* ``del``: a vacated house released its lease to recycle: ``recreating``.
* ``expired``: the bridge stopped renewing (crash, stop): ``down``.

Each house transition sends ``house_state_changed`` and, for every tenant,
``bot_location_changed`` (``house_down_fallback`` going down/recreating,
``house_up`` coming back): the dispatcher's routing changed with the lease.

Keyspace events are fire-and-forget, so each (re)subscribe reads the leases
once and emits whatever changed while nobody listened. The watcher turns on the
keyspace flags it needs (``K$gx``) and keeps any already configured.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

LEASE_RE = re.compile(r"^agent:house:([a-z0-9][a-z0-9-]{0,30}):lease:([a-z0-9_-]+)$")
#: K keyspace channel, $ string commands (SET), g generic (DEL), x expired.
REQUIRED_FLAGS = "K$gx"
RECONNECT_SECONDS = 5.0

UP, DOWN, RECREATING, UNKNOWN = "up", "down", "recreating", "unknown"


def merge_flags(current: str) -> str:
    """``current`` plus :data:`REQUIRED_FLAGS`, in a stable order."""
    if "A" in current:
        wanted = set(current) | {"K"}
    else:
        wanted = set(current) | set(REQUIRED_FLAGS)
    return "".join(sorted(wanted))


class HouseStates:
    """Per-house lease view; every method returns ``[(house, from, to)]``."""

    def __init__(self) -> None:
        self.epochs: dict[str, dict[str, str]] = {}
        self.state: dict[str, str] = {}

    def _set_state(self, house: str, new: str) -> list[tuple[str, str | None, str]]:
        old = self.state.get(house)
        if old == new:
            return []
        self.state[house] = new
        return [(house, old, new)]

    def on_set(self, house: str, backend: str, epoch: str | None) -> list:
        if not epoch:
            return []
        epochs = self.epochs.setdefault(house, {})
        previous = epochs.get(backend)
        epochs[backend] = epoch
        if previous is not None and previous != epoch and self.state.get(house) == UP:
            # Restarted between heartbeats: nothing released or expired.
            return self._set_state(house, RECREATING) + self._set_state(house, UP)
        return self._set_state(house, UP)

    def on_gone(self, house: str, backend: str, released: bool) -> list:
        epochs = self.epochs.get(house, {})
        epochs.pop(backend, None)
        if epochs:  # another backend still holds this house
            return []
        return self._set_state(house, RECREATING if released else DOWN)

    def reseed(self, leases: dict[tuple[str, str], str]) -> list:
        """Reconcile with a fresh lease read; houses that vanished are ``down``."""
        out: list = []
        live = {house for house, _ in leases}
        for house in list(self.state):
            if house not in live and self.state[house] == UP:
                self.epochs.pop(house, None)
                out += self._set_state(house, DOWN)
        for (house, backend), epoch in sorted(leases.items()):
            out += self.on_set(house, backend, epoch)
        return out


def _tenants(config: Any, house: str) -> list[Any]:
    from sqlalchemy import text

    from .utils.db import get_shared_engine

    with get_shared_engine(config).connect() as conn:
        return list(conn.execute(text(
            "SELECT slug, bot_type, agent_backend, house, house_pinned "
            "FROM bot_profiles WHERE house = :house ORDER BY slug"), {"house": house}))


def frames_for(transitions: list, tenants_of) -> list[dict]:
    """Frames for house transitions; ``tenants_of(house)`` lists profile rows."""
    from .bot_location import location_of
    from .campus_events import BOT_LOCATION_CHANGED, HOUSE_STATE_CHANGED, ROUTE

    frames: list[dict] = []
    for house, old, new in transitions:
        rows = tenants_of(house) if old is not None else []
        frames.append({"_type": HOUSE_STATE_CHANGED, "house_id": house, "from": old,
                       "to": new, "tenants": [r.slug for r in rows]})
        if old is None:
            continue  # first sighting: nothing a client could have shown moved
        for row in rows:
            args = (row.bot_type, row.agent_backend, row.house, row.house_pinned)
            before = location_of(*args, old == UP)
            after = location_of(*args, new == UP)
            if before != after:
                frames.append({"_type": BOT_LOCATION_CHANGED, "kind": ROUTE, "bot_id": row.slug,
                               "from": before, "to": after,
                               "reason": "house_up" if new == UP else "house_down_fallback"})
    return frames


class HouseWatcher:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.states = HouseStates()

    async def _emit(self, transitions: list) -> None:
        if not transitions:
            return
        from .campus_events import publish_frames

        def run() -> None:
            publish_frames(frames_for(transitions, lambda h: _tenants(self.config, h)),
                           self.config)

        try:
            await asyncio.to_thread(run)
        except Exception:
            log.warning("house watcher emit failed", exc_info=True)

    async def _read_leases(self, client: Any) -> dict[tuple[str, str], str]:
        leases: dict[tuple[str, str], str] = {}
        async for key in client.scan_iter(match="agent:house:*:lease:*", count=200):
            match = LEASE_RE.match(key)
            if match:
                epoch = await client.get(key)
                if epoch:
                    leases[(match.group(1), match.group(2))] = epoch
        return leases

    async def handle(self, client: Any, channel: str, op: str) -> None:
        key = channel.split(":", 1)[1] if channel.startswith("__keyspace@") else channel
        match = LEASE_RE.match(key)
        if not match:
            return
        house, backend = match.groups()
        if op == "set":
            transitions = self.states.on_set(house, backend, await client.get(key))
        elif op == "del":
            transitions = self.states.on_gone(house, backend, released=True)
        elif op == "expired":
            transitions = self.states.on_gone(house, backend, released=False)
        else:
            return
        await self._emit(transitions)

    async def run(self) -> None:
        import redis.asyncio as aioredis

        url = getattr(self.config, "REDIS_URL", None)
        if not url:
            return
        first = True
        while True:
            client = aioredis.from_url(url, decode_responses=True)
            pubsub = client.pubsub()
            try:
                flags = (await client.config_get("notify-keyspace-events")).get(
                    "notify-keyspace-events", "")
                if merge_flags(flags) != "".join(sorted(flags)):
                    await client.config_set("notify-keyspace-events", merge_flags(flags))
                db = client.connection_pool.connection_kwargs.get("db", 0)
                await pubsub.psubscribe(f"__keyspace@{db}__:agent:house:*:lease:*")
                transitions = self.states.reseed(await self._read_leases(client))
                # The first read only learns the pool; later ones report what changed.
                if not first:
                    await self._emit(transitions)
                first = False
                async for message in pubsub.listen():
                    if message.get("type") == "pmessage":
                        await self.handle(client, message["channel"], message["data"])
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("house watcher disconnected; resubscribing", exc_info=True)
            finally:
                try:
                    await pubsub.aclose()
                    await client.aclose()
                except Exception:
                    pass
            await asyncio.sleep(RECONNECT_SECONDS)
