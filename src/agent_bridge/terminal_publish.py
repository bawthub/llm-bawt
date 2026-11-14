"""Acknowledged, idempotent terminal publication on the existing UI stream.

The caller supplies the winning DB snapshot and its immutable deadline in Unix
epoch *seconds*. This module neither creates connections nor changes payloads.
Redis TIME, not the application clock, decides whether a new publication is
still allowed. A retained receipt wins even after that deadline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any

from redis.asyncio import Redis

from .publisher import UNIFIED_EVENTS_PREFIX, UNIFIED_STREAM_MAXLEN

TERMINAL_DEDUPE_TTL_SECONDS = 7 * 24 * 60 * 60
_DEDUPE_PREFIX = "agent:terminal_publication:"

# Redis scripts isolate commands but do NOT roll back writes on a later error.
# Check arguments, key types and available stream-ID space before either XADD.
# Operational failures (e.g. Redis loss/OOM) still propagate; this is not a
# cross-system exactly-once guarantee.
_PUBLISH_TERMINAL_LUA = r"""
if #KEYS ~= 2 or #ARGV ~= 5 then
    return redis.error_reply('terminal publication: expected 2 keys and 5 arguments')
end
if KEYS[1] == '' or KEYS[2] == '' or KEYS[1] == KEYS[2] then
    return redis.error_reply('terminal publication: invalid keys')
end
local deadline = tonumber(ARGV[1])
local maxlen = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
if not deadline or deadline ~= deadline or deadline < 0 or deadline == math.huge then
    return redis.error_reply('terminal publication: invalid deadline epoch seconds')
end
if not string.match(ARGV[2], '^[1-9]%d*$')
    or not maxlen or maxlen < 2 or maxlen ~= math.floor(maxlen)
    or maxlen > 2147483647 then
    return redis.error_reply('terminal publication: invalid MAXLEN')
end
if not string.match(ARGV[3], '^[1-9]%d*$')
    or not ttl or ttl < 1 or ttl ~= math.floor(ttl) or ttl > 2147483647 then
    return redis.error_reply('terminal publication: invalid receipt TTL')
end
local function valid_event(payload, kind)
    if string.sub(payload, 1, 1) ~= '{' then return false end
    local ok, event = pcall(cjson.decode, payload)
    return ok and type(event) == 'table' and event['_type'] == kind
end
if not valid_event(ARGV[4], 'turn_complete') then
    return redis.error_reply('terminal publication: invalid completion JSON')
end
local has_tail = ARGV[5] ~= ''
if has_tail and not valid_event(ARGV[5], 'tts_delta') then
    return redis.error_reply('terminal publication: invalid TTS tail JSON')
end

local receipt_type = redis.call('TYPE', KEYS[2]).ok
if receipt_type ~= 'none' and receipt_type ~= 'string' then
    return redis.error_reply('WRONGTYPE terminal publication receipt must be a string')
end
local original = redis.call('GET', KEYS[2])
if original then
    return {'already_published', original}
end

local stream_type = redis.call('TYPE', KEYS[1]).ok
if stream_type ~= 'none' and stream_type ~= 'stream' then
    return redis.error_reply('WRONGTYPE terminal publication destination must be a stream')
end
local now = redis.call('TIME')
local now_epoch_seconds = tonumber(now[1]) + tonumber(now[2]) / 1000000
if now_epoch_seconds >= deadline then
    return {'expired', ''}
end

if stream_type == 'stream' then
    local info = redis.call('XINFO', 'STREAM', KEYS[1])
    for i = 1, #info, 2 do
        if info[i] == 'last-generated-id' then
            local ms, seq = string.match(info[i + 1], '^(%d+)%-(%d+)$')
            -- Compare strings: uint64 stream IDs exceed Lua's exact integers.
            if ms == '18446744073709551615'
                and (seq == '18446744073709551615'
                     or (has_tail and seq == '18446744073709551614')) then
                return redis.error_reply('terminal publication: stream ID space exhausted')
            end
            break
        end
    end
end

if has_tail then
    redis.call('XADD', KEYS[1], 'MAXLEN', '~', ARGV[2], '*', 'payload', ARGV[5])
end
local completion_id = redis.call(
    'XADD', KEYS[1], 'MAXLEN', '~', ARGV[2], '*', 'payload', ARGV[4]
)
redis.call('SET', KEYS[2], completion_id, 'EX', ARGV[3])
return {'published', completion_id}
"""


class TerminalPublishStatus(str, Enum):
    PUBLISHED = "published"
    ALREADY_PUBLISHED = "already_published"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class TerminalPublishResult:
    status: TerminalPublishStatus
    completion_stream_id: str | None


def _validate_identity(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")
    # redis-py encodes strings only when executing the command. Do this before
    # invoking Redis so an invalid Unicode identity never reaches the script.
    value.encode("utf-8")


def terminal_publication_key(bot_id: str, user_id: str, turn_id: str) -> str:
    """Stable, delimiter-safe receipt identity for a bot/user/turn tuple."""
    for name, value in (("bot_id", bot_id), ("user_id", user_id), ("turn_id", turn_id)):
        _validate_identity(name, value)
    identity = json.dumps([bot_id, user_id, turn_id], ensure_ascii=False)
    return _DEDUPE_PREFIX + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _encode_event(
    event: dict[str, Any],
    *,
    kind: str,
    bot_id: str,
    user_id: str,
    turn_id: str,
) -> str:
    if not isinstance(event, dict):
        raise TypeError(f"{kind} must be a dictionary")
    expected = {"_type": kind, "bot_id": bot_id, "user_id": user_id, "turn_id": turn_id}
    for field, value in expected.items():
        if event.get(field) != value:
            raise ValueError(f"{kind}.{field} must match {value!r}")
    # No default=str, timestamp injection or re-enrichment: retries must retain
    # precisely the frozen, JSON-compatible snapshot. Reject NaN/Infinity too.
    payload = json.dumps(event, ensure_ascii=False, allow_nan=False)
    payload.encode("utf-8")
    return payload


async def publish_terminal(
    redis: Redis,
    *,
    bot_id: str,
    user_id: str,
    turn_id: str,
    publication_deadline_epoch: float,
    completion: dict[str, Any],
    tts_tail: dict[str, Any] | None = None,
) -> TerminalPublishResult:
    """Publish a frozen tail/completion once, returning the Redis acknowledgment.

    ``publication_deadline_epoch`` is the immutable DB deadline in Unix epoch
    seconds, NOT a duration or milliseconds. An expired result writes nothing.
    A retry returns the original completion ID without refreshing its seven-day
    receipt TTL, including when that retry arrives after the deadline.

    Payloads are serialized before the first await and never mutated. The caller
    must supply the same committed snapshot/deadline on every retry. Connection,
    timeout, Redis response and cancellation errors propagate unchanged; none is
    converted into an expired result. The caller owns the Redis connection.
    """
    receipt_key = terminal_publication_key(bot_id, user_id, turn_id)
    if isinstance(publication_deadline_epoch, bool) or not isinstance(
        publication_deadline_epoch, (int, float)
    ):
        raise TypeError("publication_deadline_epoch must be epoch seconds")
    try:
        deadline = float(publication_deadline_epoch)
    except OverflowError as exc:
        raise ValueError("publication_deadline_epoch must be finite") from exc
    if not math.isfinite(deadline) or deadline < 0:
        raise ValueError("publication_deadline_epoch must be finite and nonnegative")
    completion_json = _encode_event(
        completion, kind="turn_complete", bot_id=bot_id, user_id=user_id, turn_id=turn_id
    )
    tail_json = "" if tts_tail is None else _encode_event(
        tts_tail, kind="tts_delta", bot_id=bot_id, user_id=user_id, turn_id=turn_id
    )
    response = await redis.eval(
        _PUBLISH_TERMINAL_LUA,
        2,
        f"{UNIFIED_EVENTS_PREFIX}{bot_id}:{user_id}",
        receipt_key,
        repr(deadline),
        UNIFIED_STREAM_MAXLEN,
        TERMINAL_DEDUPE_TTL_SECONDS,
        completion_json,
        tail_json,
    )
    raw_status, raw_id = response
    status = TerminalPublishStatus(
        raw_status.decode("utf-8") if isinstance(raw_status, bytes) else raw_status
    )
    stream_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else raw_id
    return TerminalPublishResult(
        status=status,
        completion_stream_id=None if status is TerminalPublishStatus.EXPIRED else stream_id,
    )
