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

The same subscription covers ``agent:house:<H>:preview`` (TASK-2131), which a
house bridge sets only when its preview changes, deletes when it stops and
lets expire when the house dies: each sends ``house_preview_changed``.

Keyspace events are fire-and-forget, so each (re)subscribe reads the leases
once and emits whatever changed while nobody listened. The watcher turns on the
keyspace flags it needs (``K$gx``) and keeps any already configured.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

LEASE_RE = re.compile(r"^agent:house:([a-z0-9][a-z0-9-]{0,30}):lease:([a-z0-9_-]+)$")
PREVIEW_RE = re.compile(r"^agent:house:([a-z0-9][a-z0-9-]{0,30}):preview$")
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


def parse_preview(raw: str | None) -> dict | None:
    """A stored preview value, or None when absent or malformed."""
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


async def read_previews(client: Any) -> dict[str, dict]:
    """Every published house preview, by house (snapshot and resubscribe)."""
    found: dict[str, dict] = {}
    async for key in client.scan_iter(match="agent:house:*:preview", count=200):
        match = PREVIEW_RE.match(key)
        if match:
            value = parse_preview(await client.get(key))
            if value is not None:
                found[match.group(1)] = value
    return found


def preview_frames(changes: list[tuple[str, dict | None]]) -> list[dict]:
    from .campus_events import HOUSE_PREVIEW_CHANGED

    return [{"_type": HOUSE_PREVIEW_CHANGED, "house_id": house, "preview": preview}
            for house, preview in changes]


def _tenants(config: Any, house: str) -> list[Any]:
    from sqlalchemy import text

    from .utils.db import get_shared_engine

    with get_shared_engine(config).connect() as conn:
        return list(conn.execute(text(
            "SELECT slug, bot_type, agent_backend, house, house_pinned, updated_at "
            "FROM bot_profiles WHERE house = :house ORDER BY slug"), {"house": house}))


def frames_for(transitions: list, tenants_of, shown=lambda slug: True) -> list[dict]:
    """Frames for house transitions; ``tenants_of(house)`` lists profile rows.

    A house seen for the first time (``old`` None) is reported as coming from
    ``down``: a client's snapshot without its lease showed it, and routed its
    tenants, that way. Hidden (private/inactive) tenants are only counted.
    """
    from .bot_location import location_of
    from .campus_events import BOT_LOCATION_CHANGED, HOUSE_STATE_CHANGED, ROUTE, version_of

    frames: list[dict] = []
    for house, old, new in transitions:
        old = DOWN if old is None else old
        if old == new:
            continue
        rows = tenants_of(house)
        visible = [r for r in rows if shown(r.slug)]
        frames.append({"_type": HOUSE_STATE_CHANGED, "house_id": house, "from": old,
                       "to": new, "tenants": [r.slug for r in visible],
                       "hidden_tenants": len(rows) - len(visible)})
        for row in visible:
            args = (row.bot_type, row.agent_backend, row.house, row.house_pinned)
            before = location_of(*args, old == UP)
            after = location_of(*args, new == UP)
            if before != after:
                frames.append({"_type": BOT_LOCATION_CHANGED, "kind": ROUTE, "bot_id": row.slug,
                               "from": before, "to": after, "version": version_of(row),
                               "reason": "house_up" if new == UP else "house_down_fallback"})
    return frames


class HouseWatcher:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.states = HouseStates()
        #: house -> its preview value; a house with none is absent.
        self.previews: dict[str, dict] = {}

    def _set_preview(self, house: str, value: dict | None) -> list[tuple[str, dict | None]]:
        if self.previews.get(house) == value:
            return []
        if value is None:
            self.previews.pop(house, None)
        else:
            self.previews[house] = value
        return [(house, value)]

    def reseed_previews(self, found: dict[str, dict]) -> list[tuple[str, dict | None]]:
        out: list[tuple[str, dict | None]] = []
        for house in sorted(set(self.previews) - set(found)):
            out += self._set_preview(house, None)
        for house, value in sorted(found.items()):
            out += self._set_preview(house, value)
        return out

    async def _emit_previews(self, changes: list[tuple[str, dict | None]]) -> None:
        if not changes:
            return
        from .campus_events import publish_frames

        try:
            await asyncio.to_thread(publish_frames, preview_frames(changes), self.config)
        except Exception:
            log.warning("house watcher preview emit failed", exc_info=True)


    async def _emit(self, transitions: list) -> None:
        if not transitions:
            return
        from .campus_events import publish_frames, visibility

        def run() -> None:
            shown = visibility(self.config)
            publish_frames(frames_for(transitions, lambda h: _tenants(self.config, h), shown),
                           self.config)

        try:
            await asyncio.to_thread(run)
        except Exception:
            log.warning("house watcher emit failed", exc_info=True)

    async def _ensure_flags(self, client: Any) -> None:
        flags = (await client.config_get("notify-keyspace-events")).get(
            "notify-keyspace-events", "")
        if merge_flags(flags) != "".join(sorted(flags)):
            await client.config_set("notify-keyspace-events", merge_flags(flags))

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
        preview = PREVIEW_RE.match(key)
        if preview:
            house = preview.group(1)
            if op == "set":
                await self._emit_previews(
                    self._set_preview(house, parse_preview(await client.get(key))))
            elif op in ("del", "expired"):
                await self._emit_previews(self._set_preview(house, None))
            return
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
                db = client.connection_pool.connection_kwargs.get("db", 0)
                await pubsub.psubscribe(f"__keyspace@{db}__:agent:house:*:lease:*",
                                        f"__keyspace@{db}__:agent:house:*:preview")
                async for message in pubsub.listen():
                    kind = message.get("type")
                    if kind == "psubscribe":
                        # Every established subscription, including redis-py's
                        # own resubscribe after a reconnect: re-apply the flags
                        # (a restarted Redis loses them) and re-read the leases.
                        await self._ensure_flags(client)
                        # Two patterns: two confirmations per (re)subscribe.
                        # The second rereads nothing new, so it emits nothing.
                        # Read everything before touching either baseline: a
                        # failed read must leave both intact so the next
                        # subscription still reports what changed.
                        leases = await self._read_leases(client)
                        current_previews = await read_previews(client)
                        transitions = self.states.reseed(leases)
                        previews = self.reseed_previews(current_previews)
                        # The first read only learns the pool; later ones report changes.
                        if not first:
                            await self._emit(transitions)
                            await self._emit_previews(previews)
                        first = False
                    elif kind == "pmessage":
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
