"""Bounded, read-only proof of a complete retained run for headless recovery.

This is not subscribe_run: it cannot launch inference or replay live effects.
Redis transport failures propagate; they are never evidence of an absent run.
A caller may fold events only when disposition is COMPLETE. All other outcomes
require another observation or the durable-partial incomplete-recovery path.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json

from redis.exceptions import ResponseError

from .events import AgentEvent, AgentEventKind
from .publisher import RUN_STREAM_PREFIX
from .run_contract import DONE_FIELD, RunAccepted, RunTerminal


class ReplayDisposition(str, Enum):
    COMPLETE = "complete"
    PENDING = "pending"
    MISSING = "missing"
    TRIMMED = "trimmed"
    INVALID = "invalid"
    LIMIT = "limit"


@dataclass(frozen=True)
class RetainedRun:
    disposition: ReplayDisposition
    reason: str
    events: tuple[AgentEvent, ...] = ()
    terminal: RunTerminal | None = None


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, dict):
        return {_decode(key): _decode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_decode(item) for item in value]
    return value


async def read_retained_run(
    redis, *, request_id: str, session_key: str, backend: str,
    boot_id: str | None = None, max_entries: int = 10_000,
    max_payload_bytes: int = 16 * 1024 * 1024,
) -> RetainedRun:
    """Read one atomic snapshot; never wait on or mutate the bridge run stream.

    MULTI/EXEC keeps XINFO's trim proof and XRANGE's contents at one instant.
    Count and payload bounds are explicit LIMIT results, not truncated success.
    A live stream without its terminal remains PENDING for a bounded caller retry.
    """
    if not all(isinstance(value, str) and value.strip() for value in (request_id, session_key, backend)):
        raise ValueError("replay requires pinned request, session and backend")
    if type(max_entries) is not int or not 1 <= max_entries <= 10_000:
        raise ValueError("replay max_entries must be between 1 and 10000")
    if type(max_payload_bytes) is not int or not 1 <= max_payload_bytes <= 16 * 1024 * 1024:
        raise ValueError("replay max_payload_bytes must be between 1 and 16 MiB")
    key = f"{RUN_STREAM_PREFIX}{request_id}"
    async with redis.pipeline(transaction=True) as pipe:
        pipe.xinfo_stream(key)
        pipe.xrange(key, min="-", max="+", count=max_entries + 1)
        info, rows = await pipe.execute(raise_on_error=False)
    if isinstance(info, ResponseError):
        if str(info).lower() == "no such key" and rows == []:
            return RetainedRun(ReplayDisposition.MISSING, "run_stream_missing")
        raise info
    if isinstance(rows, Exception):
        raise rows
    try:
        info, rows = _decode(info), _decode(rows)
        length, added = info["length"], info["entries-added"]
        if type(length) is not int or type(added) is not int or length < 0 or added < length:
            raise ValueError("invalid stream counters")
        if added != length or info.get("max-deleted-entry-id", "0-0") != "0-0":
            return RetainedRun(ReplayDisposition.TRIMMED, "run_stream_trimmed_or_deleted")
        if length > max_entries:
            return RetainedRun(ReplayDisposition.LIMIT, "run_stream_entry_limit")
        if not rows:
            return RetainedRun(ReplayDisposition.PENDING, "run_stream_empty")
        if len(rows) != length or rows[0][0] != info["first-entry"][0] or rows[-1][0] != info["last-generated-id"]:
            raise ValueError("stream snapshot does not match counters")
        events = []
        terminal = None
        seen_ids = set()
        payload_bytes = 0
        for index, (_, fields) in enumerate(rows):
            if DONE_FIELD in fields:
                if fields[DONE_FIELD] != "1" or index != len(rows) - 1:
                    raise ValueError("terminal sentinel is not the final entry")
                terminal = RunTerminal.from_fields(fields)
                if terminal is None or terminal.request_id != request_id or terminal.backend != backend:
                    raise ValueError("missing or mismatched terminal identity")
                if boot_id is not None and terminal.boot_id != boot_id:
                    raise ValueError("terminal boot differs from durable acceptance")
                continue
            payload = fields["payload"]
            payload_bytes += len(payload.encode("utf-8"))
            if payload_bytes > max_payload_bytes:
                return RetainedRun(ReplayDisposition.LIMIT, "run_stream_payload_limit")
            data = json.loads(payload)
            # Live readers tolerate unknown kinds. Recovery cannot silently
            # ignore a newer content-affecting kind and call the answer complete.
            AgentEventKind(data["kind"])
            event = AgentEvent.from_dict(data)
            if event.kind == AgentEventKind.RUN_TERMINAL:
                raise ValueError("terminal must come from the sentinel")
            if event.session_key != session_key or not event.event_id or event.event_id in seen_ids:
                raise ValueError("wrong session or duplicate event identity")
            seen_ids.add(event.event_id)
            if event.provider is not None and event.provider != backend:
                raise ValueError("event backend differs from pinned backend")
            if event.kind == AgentEventKind.RUN_ACCEPTED:
                accepted = RunAccepted.from_dict(event.raw)
                if accepted.request_id != request_id or accepted.backend != backend:
                    raise ValueError("acceptance identity mismatch")
                if boot_id is not None and accepted.boot_id != boot_id:
                    raise ValueError("acceptance boot mismatch")
            events.append(event)
        if terminal is None:
            return RetainedRun(ReplayDisposition.PENDING, "run_stream_not_terminal")
        for event in events:
            if event.kind == AgentEventKind.RUN_ACCEPTED and event.raw["boot_id"] != terminal.boot_id:
                raise ValueError("acceptance and terminal boots differ")
        return RetainedRun(ReplayDisposition.COMPLETE, "retained_terminal_run", tuple(events), terminal)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return RetainedRun(ReplayDisposition.INVALID, "run_stream_invalid_or_unprovable")
