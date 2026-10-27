"""Per-window UI fan-out of the unified event streams (TASK-1582).

Each browser window holds one SSE connection identified by its ``consumer_id``.
A window reads its streams with plain ``XREAD`` from a cursor the server keeps
in the Redis hash ``ui:cursor:{consumer_id}`` (one field per stream key, the
last id delivered), so a reconnect resumes where the previous connection left
off, including events published while no connection was open.

The hash's ``owner`` field fences the cursor. Each connection claims it with a
fresh random token at connect; the newest connection always wins. Every cursor
write is a compare-and-set on that token, so an older connection that is still
alive upstream (a frozen mobile socket, a proxy hop that never closed) cannot
move the cursor. It sees the refusal, stops yielding and exits. ``XREAD`` is
non-destructive, so a lingering connection takes nothing from the live one.

This replaced per-window consumer groups (``ui:{consumer_id}`` with one fixed
consumer name). With groups, a lingering connection kept competing for the
group's messages, and the live connection received about half of them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from typing import Any, AsyncIterator, Iterable

from redis.exceptions import ResponseError

from .publisher import GLOBAL_CONFIG_STREAM, UNIFIED_EVENTS_PREFIX

logger = logging.getLogger(__name__)

UI_CURSOR_PREFIX = "ui:cursor:"
# Resume window. A window away longer than this reconnects at the stream tail;
# the client's long-gap history reconcile covers what it missed. Matches the
# idle limit the retired consumer groups were swept at.
UI_CURSOR_TTL_S = 1800
# How often an idle connection confirms it still owns the cursor (and refreshes
# the TTL). A superseded idle connection exits within this interval.
OWNER_CHECK_INTERVAL_S = 5.0
# Consumer-group prefix of the retired per-window groups. Each is destroyed by
# its own window once migrated into the cursor; nothing sweeps them.
LEGACY_UI_GROUP_PREFIX = "ui:"

READ_COUNT = 100
READ_BLOCK_MS = 2000
_EMPTY_STREAM_ID = "0-0"

# KEYS[1] = cursor hash; ARGV[1] = token; ARGV[2] = ttl seconds.
# Takes ownership and returns the stored fields (including the new owner).
CLAIM_SCRIPT = """
redis.call('HSET', KEYS[1], 'owner', ARGV[1])
redis.call('EXPIRE', KEYS[1], ARGV[2])
return redis.call('HGETALL', KEYS[1])
"""

# KEYS[1] = cursor hash; ARGV[1] = token; ARGV[2] = ttl seconds;
# ARGV[3..] = stream_key, id pairs. Writes only while the token owns the hash.
# With no pairs it is the owner check / TTL refresh. Returns 1 or 0 (superseded).
ADVANCE_SCRIPT = """
if redis.call('HGET', KEYS[1], 'owner') ~= ARGV[1] then return 0 end
for i = 3, #ARGV, 2 do redis.call('HSET', KEYS[1], ARGV[i], ARGV[i + 1]) end
redis.call('EXPIRE', KEYS[1], ARGV[2])
return 1
"""

# Same arguments as ADVANCE_SCRIPT, but only sets fields that are absent, so
# seeding a start point never moves a cursor that is already set. Returns the
# resulting fields, or 0 when superseded.
SEED_SCRIPT = """
if redis.call('HGET', KEYS[1], 'owner') ~= ARGV[1] then return 0 end
for i = 3, #ARGV, 2 do redis.call('HSETNX', KEYS[1], ARGV[i], ARGV[i + 1]) end
redis.call('EXPIRE', KEYS[1], ARGV[2])
return redis.call('HGETALL', KEYS[1])
"""

_OWNER_FIELD = "owner"
_MAX_SEQ = 18446744073709551615


def unified_stream_keys(
    bot_ids: Iterable[str],
    user_id: str,
    *,
    include_global: bool = False,
) -> dict[str, str]:
    """Map each unified stream key a window reads to its bot id."""
    keys = {f"{UNIFIED_EVENTS_PREFIX}{bid}:{user_id}": bid for bid in bot_ids}
    if include_global:
        keys[GLOBAL_CONFIG_STREAM] = "system"
    return keys


def stream_id_key(stream_id: str) -> tuple[int, int]:
    """Order Redis stream ids numerically (``ms-seq``), not as strings."""
    ms, _, seq = stream_id.partition("-")
    return int(ms), int(seq or 0)


def stream_id_before(stream_id: str) -> str:
    """The largest stream id strictly below ``stream_id`` (an XREAD start point)."""
    ms, seq = stream_id_key(stream_id)
    if seq:
        return f"{ms}-{seq - 1}"
    if ms:
        return f"{ms - 1}-{_MAX_SEQ}"
    return _EMPTY_STREAM_ID


class UiStreamReader:
    """One SSE connection's read of its window's unified streams."""

    def __init__(
        self,
        redis: Any,
        consumer_id: str,
        stream_keys: Iterable[str],
        *,
        cursor_ttl_s: int = UI_CURSOR_TTL_S,
        owner_check_s: float = OWNER_CHECK_INTERVAL_S,
        token: str | None = None,
    ) -> None:
        self._redis = redis
        self.cursor_key = f"{UI_CURSOR_PREFIX}{consumer_id}"
        self._legacy_group = f"{LEGACY_UI_GROUP_PREFIX}{consumer_id}"
        self._streams = list(dict.fromkeys(stream_keys))
        self._ttl_s = int(cursor_ttl_s)
        self._owner_check_s = owner_check_s
        self.token = token or secrets.token_hex(16)
        self._cursors: dict[str, str] = {}
        self._catchup_until: dict[str, tuple[int, int]] = {}
        self.superseded = False

    async def claim(self) -> None:
        """Take the cursor from any older connection and load the resume point.

        A stream with no stored cursor resumes from the window's retired
        consumer group when one exists (its oldest unacked entry, else its
        last-delivered id), and otherwise starts at the current tail. The start
        point is persisted at once, so every event published after ``claim``
        returns is delivered, even across a reconnect before the first event.
        """
        raw = await self._redis.eval(CLAIM_SCRIPT, 1, self.cursor_key, self.token, self._ttl_s)
        stored = dict(zip(raw[::2], raw[1::2])) if raw else {}
        stored.pop(_OWNER_FIELD, None)
        seeds: dict[str, str] = {}
        migrated: list[str] = []
        for sk in self._streams:
            tail = await self._tail(sk)
            self._catchup_until[sk] = stream_id_key(tail)
            if sk in stored:
                self._cursors[sk] = stored[sk]
                continue
            legacy = await self._legacy_resume_point(sk)
            if legacy is not None:
                migrated.append(sk)
            seeds[sk] = legacy if legacy is not None else tail
        if seeds:
            raw = await self._redis.eval(
                SEED_SCRIPT, 1, self.cursor_key, self.token, self._ttl_s, *_pairs(seeds),
            )
            if not isinstance(raw, list):
                self._mark_superseded()
                return
            current = dict(zip(raw[::2], raw[1::2]))
            for sk, seed in seeds.items():
                self._cursors[sk] = current.get(sk, seed)
        for sk in migrated:
            try:
                await self._redis.xgroup_destroy(sk, self._legacy_group)
            except Exception:
                logger.debug("Failed to retire group %s on %s", self._legacy_group, sk)

    async def _legacy_resume_point(self, stream_key: str) -> str | None:
        """Where the retired ``ui:{consumer_id}`` group would have resumed.

        ``None`` only when the stream or the group is proven absent. Any other
        failure raises, so ``claim`` fails closed: nothing is seeded and the
        group is kept, and the next connect retries the migration.
        """
        try:
            groups = await self._redis.xinfo_groups(stream_key)
        except ResponseError as exc:
            if _is_absent(exc):
                return None
            raise
        for group in groups:
            if group.get("name") != self._legacy_group:
                continue
            if int(group.get("pending") or 0):
                try:
                    summary = await self._redis.xpending(stream_key, self._legacy_group)
                except ResponseError as exc:
                    if _is_absent(exc):
                        return None
                    raise
                oldest = (summary or {}).get("min")
                if not oldest:
                    raise RuntimeError(
                        f"legacy group {self._legacy_group} on {stream_key} reports pending "
                        "entries without an oldest id"
                    )
                return stream_id_before(oldest)
            return group.get("last-delivered-id") or _EMPTY_STREAM_ID
        return None

    async def events(self, *, timeout_s: float = 86400) -> AsyncIterator[dict | None]:
        """Yield events in id order; ``None`` on an idle read (heartbeat tick).

        Ends when superseded by a newer connection for the same consumer_id,
        or after ``timeout_s`` without an event.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        next_owner_check = loop.time() + self._owner_check_s
        while not self.superseded:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            try:
                results = await self._redis.xread(
                    dict(self._cursors),
                    count=READ_COUNT,
                    block=max(1, min(int(remaining * 1000), READ_BLOCK_MS)),
                )
            except Exception:
                logger.exception("Redis XREAD error on %s", self._streams)
                await asyncio.sleep(1)
                continue

            if not results:
                if loop.time() >= next_owner_check:
                    next_owner_check = loop.time() + self._owner_check_s
                    if not await self._advance({}):
                        return
                yield None
                continue

            # Confirm ownership before handing a batch downstream, so a
            # superseded connection never emits another event.
            if not await self._advance({}):
                return
            next_owner_check = loop.time() + self._owner_check_s

            batch = [
                (msg_id, stream_name, fields)
                for stream_name, messages in results
                for msg_id, fields in messages
            ]
            batch.sort(key=lambda m: stream_id_key(m[0]))
            for msg_id, stream_name, fields in batch:
                deadline = loop.time() + timeout_s
                payload_str = fields.get("payload", "{}")
                try:
                    data = json.loads(payload_str)
                except Exception:
                    data = {"raw": payload_str}
                catchup = self._catchup_until.get(stream_name)
                data["_replayed"] = catchup is not None and stream_id_key(msg_id) <= catchup
                yield data
                # Persist only after the event was handed off, like the old XACK.
                self._cursors[stream_name] = msg_id
                if not await self._advance({stream_name: msg_id}):
                    return

    async def _tail(self, stream_key: str) -> str:
        last = await self._redis.xrevrange(stream_key, count=1)
        return last[0][0] if last else _EMPTY_STREAM_ID

    async def _advance(self, cursors: dict[str, str]) -> bool:
        ok = await self._redis.eval(
            ADVANCE_SCRIPT, 1, self.cursor_key, self.token, self._ttl_s, *_pairs(cursors),
        )
        if not int(ok or 0):
            self._mark_superseded()
            return False
        return True

    def _mark_superseded(self) -> None:
        if not self.superseded:
            logger.info("UI stream %s superseded by a newer connection", self.cursor_key)
        self.superseded = True


def _is_absent(exc: ResponseError) -> bool:
    """Redis proved the stream (``no such key``) or group (``NOGROUP``) absent."""
    msg = str(exc)
    return "no such key" in msg.lower() or msg.startswith("NOGROUP")


def _pairs(cursors: dict[str, str]) -> list[str]:
    return [x for sk, msg_id in cursors.items() for x in (sk, msg_id)]
