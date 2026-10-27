"""OpenClaw WS->SSE bridge endpoint."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import List

from agent_bridge.subscriber import RedisSubscriber

from ..dependencies import get_service
from ..logging import get_service_logger

router = APIRouter()
log = get_service_logger(__name__)


# Keep well under the bawthub client watchdog (useUnifiedEventStream.ts).
HEARTBEAT_INTERVAL_S = 15.0


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.get("/v1/ws", tags=["OpenClaw Bridge"])
async def openclaw_ws_bridge(
    session_key: str = Query(None, description="Session key to stream (legacy)"),
    bot_id: List[str] = Query(None, description="Bot ID(s) for unified event stream (repeatable)"),
    user_id: str = Query(None, description="User ID for unified event stream"),
    consumer_id: str = Query(None, description="Consumer ID for durable consumer group"),
    config_only: bool = Query(False, description="Subscribe only to global configuration changes"),
):
    """Expose events as SSE via Redis subscriber.

    Modes:
    - Legacy: ``?session_key=main`` — streams OpenClaw session events (XREAD, no resume)
    - Unified: ``?bot_id=X&bot_id=Y&user_id=Z&consumer_id=UUID`` — durable consumer group
      subscribing to one or more bots. Falls back to legacy if consumer_id is absent.
    - Configuration only: ``?config_only=true&consumer_id=UUID`` — global profile
      updates without claiming a user identity or subscribing to any turns.
    """
    service = get_service()

    redis_sub = getattr(service, "_redis_subscriber", None)
    if redis_sub is None:
        raise HTTPException(status_code=503, detail="OpenClaw bridge not enabled (no Redis subscriber)")

    # Each SSE connection gets its OWN dedicated Redis client (see
    # _unified_event_stream / _legacy_event_stream). Hand the stream the URL,
    # not the shared singleton — the singleton's pool must not absorb stranded
    # connections from every mobile reconnect.
    redis_url = redis_sub._redis_url

    if config_only:
        if not consumer_id or session_key or bot_id or user_id:
            raise HTTPException(status_code=400, detail="config_only requires consumer_id and no user or bot scope")
        bot_id, user_id = [], ""

    use_unified = config_only or bool(bot_id and user_id and consumer_id)
    if not use_unified and not session_key:
        raise HTTPException(
            status_code=400,
            detail="Provide session_key, bot_id+user_id+consumer_id, or config_only+consumer_id",
        )

    if use_unified:
        return StreamingResponse(
            _unified_event_stream(redis_url, bot_id, user_id, consumer_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Legacy session-based streaming
    return StreamingResponse(
        _legacy_event_stream(redis_url, session_key),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _unified_event_stream(redis_url: str, bot_ids: list[str], user_id: str, consumer_id: str):
    """Resumable event stream for one UI window, over one or more bots.

    Reads by the window's server-held cursor (``agent_bridge.ui_stream``): the
    newest connection for a consumer_id owns it, and an older connection that
    is still open upstream stops at its next read instead of competing for
    events (TASK-1582).

    Uses a DEDICATED RedisSubscriber per SSE connection, closed in finally.

    Why per-connection: the reader issues a blocking XREAD that
    holds a pool connection. When the client disconnects (mobile lock screen,
    tab close, network drop), Starlette cancels this task mid-read and redis-py
    strands that connection in whatever pool it was using. If that were the
    shared app singleton's pool, every reconnect would leak one connection until
    the pool exhausted (MaxConnectionsError) — hanging the app and dropping
    turn_complete events so the UI stays "in turn". A private client per
    connection confines the strand to this stream, and the finally close()
    reclaims it. Reconnects clean up instead of building up.
    """
    sub = RedisSubscriber(redis_url)
    await sub.connect()
    # Claim the cursor BEFORE announcing "hello". A stream without a stored
    # cursor is pinned to its current tail, so every event published from this
    # point on is delivered. A downstream consumer (e.g. the bawthub chat TTS
    # driver) can therefore treat "hello" as a real "you are subscribed, safe to
    # fire a turn" signal — closing the race where a turn published between
    # connect and subscribe produced no audio.
    reader = sub.ui_stream(bot_ids, user_id, consumer_id, include_global=True)
    await reader.claim()
    yield _sse(
        "hello",
        {
            "bot_ids": bot_ids,
            "user_id": user_id,
            "consumer_id": consumer_id,
            "mode": "unified" if bot_ids else "config-only",
            "connected": True,
            "ts": datetime.now(timezone.utc).isoformat(),
        },
    )

    # Named heartbeat event (not an SSE ``: comment``): EventSource never
    # surfaces comments, so only a named event lets the browser's watchdog
    # tell a live-but-quiet stream from a silently dead one (a proxy hop that
    # keeps the client socket open after the upstream died).
    last_ping = asyncio.get_running_loop().time()
    try:
        async for event_data in reader.events(timeout_s=86400):
            if event_data is not None:
                replayed = event_data.pop("_replayed", False)
                yield _sse("event", {**event_data, "replayed": replayed})

            now = asyncio.get_running_loop().time()
            if now - last_ping >= HEARTBEAT_INTERVAL_S:
                yield _sse("heartbeat", {"ts": datetime.now(timezone.utc).isoformat()})
                last_ping = now
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.exception("/v1/ws unified stream error")
        yield _sse("error", {"error": str(e)})
    finally:
        await sub.close()


async def _legacy_event_stream(redis_url: str, session_key: str):
    """Legacy session-based event stream (no resume).

    Dedicated RedisSubscriber per connection, closed in finally — same
    strand-on-disconnect reasoning as _unified_event_stream.
    """
    sub = RedisSubscriber(redis_url)
    await sub.connect()
    yield _sse(
        "hello",
        {
            "session_key": session_key,
            "mode": "legacy",
            "connected": True,
            "ts": datetime.now(timezone.utc).isoformat(),
        },
    )

    last_ping = asyncio.get_running_loop().time()
    try:
        async for evt in sub.subscribe(session_key):
            payload = {
                "id": evt.db_id,
                "event_id": evt.event_id,
                "session_key": evt.session_key,
                "run_id": evt.run_id,
                "kind": evt.kind.value,
                "origin": evt.origin,
                "text": evt.text,
                "tool_name": evt.tool_name,
                "timestamp": evt.timestamp.isoformat() if evt.timestamp else None,
                "seq": evt.seq,
                "raw": evt.raw,
            }
            yield _sse("event", payload)

            now = asyncio.get_running_loop().time()
            if now - last_ping > 25:
                yield ": ping\n\n"
                last_ping = now
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.exception("/v1/ws legacy stream error")
        yield _sse("error", {"error": str(e)})
    finally:
        await sub.close()
