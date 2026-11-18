"""Saved audio for spoken assistant replies (TASK-1115).

Producer routes (service token only) let the speech backend reserve a turn,
bind its assistant message and upload the finished WAV. Reader routes are
owner-scoped by the ``user_id`` query field and serve the audio with single-range
support (``Range``/``If-Range``, 206/416) plus ``HEAD`` and ``If-None-Match``
so a browser ``<audio>`` element can seek natively.

Each state change publishes a ``response_audio`` event on the owner's
existing per-bot user event stream; clients refresh from that, from
``app:resync`` and from the bounded lookup below. No polling.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...media.object_store import BlobBackendUnavailable
from ...media.response_audio import (
    MAX_AUDIO_BYTES,
    STATUS_FAILED,
    TERMINAL_STATUSES,
    ResponseAudioConflict,
    ResponseAudioStore,
    get_response_audio_store,
    public_view,
)
from ...media.store import MediaAssetNotFound, get_media_store
from .service_auth import read_body, require_service_token

logger = logging.getLogger(__name__)

router = APIRouter(tags=["response-audio"])
_PRODUCER = [Depends(require_service_token)]

#: Bounded lookup size for reconnect / history hydration.
MAX_LOOKUP_IDS = 100
WAV_MIME = "audio/wav"
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


def _audio_store() -> ResponseAudioStore:
    store = get_response_audio_store()
    if store is None:
        raise HTTPException(status_code=503, detail="Response audio storage unavailable")
    return store


#: The owner travels as the ``user_id`` query field so BawtHub's generic
#: proxy, which binds declared user carriers, fills it with the signed-in user.
_OWNER_QUERY = Query(description="Owner user id; BawtHub's proxy sets it to the request's user")


def _entity(user_id: Optional[str]) -> str:
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=401, detail="user_id is required")
    return user_id.strip()


def _publish(row: dict) -> None:
    """Best-effort status event on the owner's existing user stream."""
    try:
        import redis as _redis

        from agent_bridge.user_events import publish_user_event

        from ...utils.config import config

        client = _redis.Redis.from_url(
            config.REDIS_URL, decode_responses=True, socket_timeout=1.0, socket_connect_timeout=1.0,
        )
        try:
            payload = {"_type": "response_audio", "bot_id": row["bot_id"],
                       "turn_id": row["turn_id"], **public_view(row)}
            publish_user_event(client, row["bot_id"], row["owner_user_id"], json.dumps(payload))
        finally:
            client.close()
    except Exception as e:  # nonfatal: resync and lookup still converge
        logger.warning("response_audio event publish failed: %s", e)


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------


class ReserveBody(BaseModel):
    turn_id: str = Field(min_length=1, max_length=128)
    bot_id: str = Field(min_length=1, max_length=64)
    owner_user_id: str = Field(min_length=1, max_length=100)
    session_id: Optional[str] = Field(default=None, max_length=64)
    assistant_message_id: Optional[str] = Field(default=None, max_length=36)


class BindBody(BaseModel):
    owner_user_id: str = Field(min_length=1, max_length=100)
    assistant_message_id: str = Field(min_length=1, max_length=36)


class FailBody(BaseModel):
    lease_token: str = Field(min_length=1)
    error: str = Field(default="audio not saved", max_length=500)


@router.post("/v1/media/response-audio/reservations", dependencies=_PRODUCER)
async def reserve(request: Request):
    body = await read_body(request, ReserveBody)
    try:
        res = await run_in_threadpool(_audio_store().reserve, **body.model_dump())
    except ResponseAudioConflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    if res.lease_token:
        await run_in_threadpool(_publish, res.row)
    return {"reservation": public_view(res.row), "lease_token": res.lease_token}


@router.post("/v1/media/response-audio/reservations/{turn_id}/bind", dependencies=_PRODUCER)
async def bind(turn_id: str, request: Request):
    body = await read_body(request, BindBody)
    try:
        row = await run_in_threadpool(_audio_store().bind_message, turn_id=turn_id, **body.model_dump())
    except ResponseAudioConflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    if row is None:
        raise HTTPException(status_code=404, detail="Reservation not found")
    await run_in_threadpool(_publish, row)
    return {"reservation": public_view(row)}


@router.put("/v1/media/response-audio/reservations/{turn_id}/audio", dependencies=_PRODUCER)
async def upload_audio(
    turn_id: str,
    request: Request,
    lease_token: str = Query(min_length=1),
    status: str = Query(default="finalized"),
    duration_ms: int = Query(ge=0),
    owner_user_id: str = Query(min_length=1, max_length=100),
):
    """Store the WAV and finalize. Repeating the same upload is a no-op."""
    if status not in TERMINAL_STATUSES or status == STATUS_FAILED:
        raise HTTPException(status_code=400, detail="status must be finalized or partial")
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="empty body")
    if len(raw) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="audio exceeds the 10 minute cap")
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise HTTPException(status_code=415, detail="expected a WAV body")
    try:
        asset = await run_in_threadpool(
            get_media_store().upload, raw, WAV_MIME, "tts_response", owner_user_id,
            filename="reply.wav", kind="file",
        )
    except BlobBackendUnavailable:
        raise HTTPException(status_code=503, detail="Media storage backend unavailable")
    try:
        row = await run_in_threadpool(
            _audio_store().finalize, turn_id=turn_id, lease_token=lease_token, status=status,
            media_asset_id=asset.id, duration_ms=duration_ms, size_bytes=len(raw),
        )
    except ResponseAudioConflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    if row is None:
        # Reply deleted mid-save: the asset stays unreferenced for the GC.
        raise HTTPException(status_code=410, detail="Reply no longer exists")
    await run_in_threadpool(_publish, row)
    return {"reservation": public_view(row)}


@router.post("/v1/media/response-audio/reservations/{turn_id}/fail", dependencies=_PRODUCER)
async def fail(turn_id: str, request: Request):
    body = await read_body(request, FailBody)
    try:
        row = await run_in_threadpool(
            _audio_store().finalize, turn_id=turn_id, lease_token=body.lease_token,
            status=STATUS_FAILED, media_asset_id=None, error=body.error,
        )
    except ResponseAudioConflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    if row is None:
        raise HTTPException(status_code=410, detail="Reply no longer exists")
    await run_in_threadpool(_publish, row)
    return {"reservation": public_view(row)}


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


@router.get("/v1/media/response-audio")
async def lookup(
    message_ids: str = Query(description="Comma-separated assistant message ids"),
    user_id: str = _OWNER_QUERY,
):
    owner = _entity(user_id)
    ids = [i for i in (s.strip() for s in message_ids.split(",")) if i]
    if len(ids) > MAX_LOOKUP_IDS:
        raise HTTPException(status_code=400, detail=f"at most {MAX_LOOKUP_IDS} ids")
    rows = await run_in_threadpool(_audio_store().get_many_for_messages, ids, owner)
    return {"items": {mid: public_view(row) for mid, row in rows.items()}}


def parse_range(header: Optional[str], size: int) -> tuple[int, int] | None | str:
    """``(start, end)`` inclusive, ``None`` for no/ignored range, ``"416"`` if unsatisfiable."""
    if not header:
        return None
    m = _RANGE_RE.match(header.strip())
    if not m:
        return None  # multi-range or malformed: serve the whole body
    first, last = m.group(1), m.group(2)
    if not first and not last:
        return None
    if not first:  # suffix range
        n = int(last)
        if n == 0:
            return "416"
        return (max(0, size - n), size - 1)
    start = int(first)
    if start >= size:
        return "416"
    end = min(int(last), size - 1) if last else size - 1
    if end < start:
        return None
    return (start, end)


@router.api_route("/v1/media/response-audio/{assistant_message_id}", methods=["GET", "HEAD"])
async def serve(
    assistant_message_id: str,
    request: Request,
    user_id: str = _OWNER_QUERY,
):
    owner = _entity(user_id)
    row = await run_in_threadpool(_audio_store().get_for_message, assistant_message_id, owner)
    if row is None or not public_view(row)["available"]:
        raise HTTPException(status_code=404, detail="No saved audio for this reply")
    media = get_media_store()
    asset = await run_in_threadpool(media.stat, row["media_asset_id"])
    if asset is None:
        raise HTTPException(status_code=404, detail="Saved audio is missing")
    etag = f'"{asset.sha256}"'
    size = int(asset.size_bytes)
    base = {
        "ETag": etag,
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, max-age=31536000, immutable",
        "X-Content-Type-Options": "nosniff",
        "Content-Disposition": 'inline; filename="reply.wav"',
    }
    inm = request.headers.get("if-none-match")
    if inm and (etag in [c.strip() for c in inm.split(",")] or inm.strip() == "*"):
        return Response(status_code=304, headers=base)

    rng = parse_range(request.headers.get("range"), size)
    if_range = request.headers.get("if-range")
    if rng is not None and if_range and if_range.strip() != etag:
        rng = None  # stale validator: full body
    if rng == "416":
        return Response(status_code=416, headers={**base, "Content-Range": f"bytes */{size}"})

    if request.method == "HEAD":
        if rng is None:
            return Response(status_code=200, media_type=WAV_MIME, headers={**base, "Content-Length": str(size)})
        start, end = rng
        return Response(status_code=206, media_type=WAV_MIME, headers={
            **base, "Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(end - start + 1)})

    try:
        data, _mime = await run_in_threadpool(media.read_variant, row["media_asset_id"], "original")
    except (MediaAssetNotFound, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Saved audio is missing")
    except BlobBackendUnavailable:
        raise HTTPException(status_code=503, detail="Media storage backend unavailable")
    size = len(data)
    if rng is None:
        return Response(content=data, media_type=WAV_MIME, headers=base)
    start, end = rng
    end = min(end, size - 1)
    return Response(
        content=data[start:end + 1], status_code=206, media_type=WAV_MIME,
        headers={**base, "Content-Range": f"bytes {start}-{end}/{size}"},
    )
