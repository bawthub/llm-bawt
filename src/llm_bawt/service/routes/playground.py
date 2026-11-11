"""Playground REST (TASK-2190): list, open once, read with the view token, close.

LAN trust boundary, like the other owner routes: BawtHub proxies these with
the signed-in user as ``user``. Every route is scoped to that user. Listing
returns metadata only; content is reachable only through the token minted by
the single successful open, and only until the artifact expires.

HTML is returned as plain text (BawtHub renders it in a sandboxed ``srcdoc``
frame), and every content response except PDF carries ``CSP: sandbox`` and
``nosniff`` so nothing executes in the app origin if opened directly.
"""
from __future__ import annotations

import logging
from urllib.parse import quote

from fastapi import APIRouter, Header, HTTPException, Query, Response

from ...media.object_store import BlobBackendUnavailable
from ...playground.service import get_playground, view_timeout_minutes
from ...playground.store import OpenRefused, public_view
from ..dependencies import get_service
from .sessions import _resolve_user

router = APIRouter(prefix="/v1/playground", tags=["Playground"])
logger = logging.getLogger(__name__)

_TEXT_KINDS = {"html", "markdown", "text"}
_NO_STORE = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


def _playground():
    try:
        return get_playground(get_service().config)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="playground unavailable") from exc


def _row_or_404(playground, artifact_id: str, user_id: str) -> dict:
    row = playground.store.get(artifact_id, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="not_found")
    return row


@router.get("")
def list_artifacts(
    user: str | None = Query(None),
    bot: str | None = Query(None, max_length=64),
    status: str | None = Query(None, pattern="^(unopened|expired)$"),
    limit: int = Query(200, ge=1, le=1000),
):
    user_id = _resolve_user(user)
    playground = _playground()
    rows = playground.store.list(user_id, bot_id=bot, status=status, limit=limit)
    return {
        "items": [public_view(row) for row in rows],
        "senders": playground.store.senders(user_id),
        "view_timeout_minutes": view_timeout_minutes(get_service().config),
    }


@router.get("/{artifact_id}")
def get_artifact(artifact_id: str, user: str | None = Query(None)):
    user_id = _resolve_user(user)
    return public_view(_row_or_404(_playground(), artifact_id, user_id))


@router.post("/{artifact_id}/open")
def open_artifact(artifact_id: str, user: str | None = Query(None)):
    """Mark opened and mint the view token. A second open expires the artifact (410)."""
    user_id = _resolve_user(user)
    playground = _playground()
    try:
        row, token = playground.open(artifact_id, user_id)
    except OpenRefused as refused:
        raise HTTPException(status_code=404 if refused.code == "not_found" else 410,
                            detail=refused.code) from None
    return {
        "artifact": public_view(row),
        "token": token,
        "content_path": f"/v1/playground/{artifact_id}/content?token={quote(token)}",
    }


@router.get("/{artifact_id}/content")
def artifact_content(
    artifact_id: str,
    token: str = Query(..., min_length=1, max_length=128),
    user: str | None = Query(None),
    range_header: str | None = Header(None, alias="Range"),
):
    user_id = _resolve_user(user)
    playground = _playground()
    row = _row_or_404(playground, artifact_id, user_id)
    if row["expired_at"] is not None:
        raise HTTPException(status_code=410, detail="expired")
    if playground.store.viewable(artifact_id, user_id, token) is None:
        raise HTTPException(status_code=403, detail="invalid_token")
    headers = dict(_NO_STORE)
    if row["mime"] != "application/pdf":
        headers["Content-Security-Policy"] = "sandbox"
    headers["X-Playground-Kind"] = row["kind"]
    try:
        content = playground.read(row, range_header)
    except FileNotFoundError:
        raise HTTPException(status_code=410, detail="expired") from None
    except ValueError:
        raise HTTPException(status_code=416, detail="bad_range") from None
    except BlobBackendUnavailable:
        raise HTTPException(status_code=503, detail="storage_unavailable") from None
    if isinstance(content, str):
        return Response(content=content.encode("utf-8"), headers=headers,
                        media_type="text/plain; charset=utf-8")
    mime = "text/plain; charset=utf-8" if row["kind"] in _TEXT_KINDS else row["mime"]
    disposition = "attachment" if row["kind"] == "file" else "inline"
    if row.get("filename"):
        disposition += f"; filename*=UTF-8''{quote(row['filename'])}"
    headers["Content-Disposition"] = disposition
    headers["Accept-Ranges"] = "bytes"
    status = 200
    if content.partial:
        status = 206
        headers["Content-Range"] = f"bytes {content.start}-{content.end}/{content.total_size}"
    return Response(content=content.data, status_code=status, headers=headers, media_type=mime)


@router.post("/{artifact_id}/close")
def close_artifact(artifact_id: str, user: str | None = Query(None)):
    """Expire now: content and blob deleted, History row kept. Idempotent."""
    user_id = _resolve_user(user)
    row = _playground().expire(artifact_id, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="not_found")
    return public_view(row)
