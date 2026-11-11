"""Playground REST (TASK-2190, mailbox model TASK-2195).

LAN trust boundary, like the other owner routes: BawtHub proxies these with
the signed-in user as ``user``. Every route is scoped to that user. Listing
returns metadata only. Opening marks an artifact read and mints a view token
for its content; it can be reopened any time until it is deleted forever.
Archive, Trash and read state are ``/state`` actions; only ``/delete`` (from
Trash) and the trash purge remove content.

HTML is returned as plain text (BawtHub renders it in a sandboxed ``srcdoc``
frame), and every content response except PDF carries ``CSP: sandbox`` and
``nosniff`` so nothing executes in the app origin if opened directly.
"""
from __future__ import annotations

import logging
from urllib.parse import quote

from fastapi import APIRouter, Body, Header, HTTPException, Query, Response

from ...media.object_store import BlobBackendUnavailable
from ...playground.service import get_playground, trash_days
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
    view: str | None = Query(None, pattern="^(inbox|archive|trash|all)$"),
    q: str | None = Query(None, max_length=200),
    limit: int = Query(200, ge=1, le=1000),
):
    """Default view is the Inbox, so unread counts from it follow the Inbox."""
    user_id = _resolve_user(user)
    playground = _playground()
    rows = playground.store.list(user_id, bot_id=bot, status=status, view=view,
                                 query=(q or "").strip() or None, limit=limit)
    return {
        "items": [public_view(row) for row in rows],
        "senders": playground.store.senders(user_id),
        "trash_days": trash_days(get_service().config),
    }


@router.get("/{artifact_id}")
def get_artifact(artifact_id: str, user: str | None = Query(None)):
    user_id = _resolve_user(user)
    return public_view(_row_or_404(_playground(), artifact_id, user_id))


@router.post("/{artifact_id}/open")
def open_artifact(artifact_id: str, user: str | None = Query(None)):
    """Open or reopen: marks it read and mints a view token. Deleted forever: 410."""
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


PREVIEW_TEXT_CHARS = {"text": 1200, "markdown": 1200, "html": 200_000}
PREVIEW_IMAGE_BYTES = 8 * 1024 * 1024


@router.get("/{artifact_id}/preview")
def artifact_preview(artifact_id: str, user: str | None = Query(None)):
    """A chat-card preview (TASK-2195); it does not mark the artifact read.

    Images up to ``PREVIEW_IMAGE_BYTES``; html/markdown/text truncated. Other
    kinds have no preview (415); deleted forever is 410. Same sandbox headers
    as content.
    """
    user_id = _resolve_user(user)
    playground = _playground()
    row = _row_or_404(playground, artifact_id, user_id)
    if row["expired_at"] is not None:
        raise HTTPException(status_code=410, detail="expired")
    kind = row["kind"]
    if kind not in _TEXT_KINDS and not (kind == "image" and row["size_bytes"] <= PREVIEW_IMAGE_BYTES):
        raise HTTPException(status_code=415, detail="no_preview")
    headers = {**_NO_STORE, "Content-Security-Policy": "sandbox", "X-Playground-Kind": kind}
    try:
        content = playground.read(row, None)
    except FileNotFoundError:
        raise HTTPException(status_code=410, detail="expired") from None
    except BlobBackendUnavailable:
        raise HTTPException(status_code=503, detail="storage_unavailable") from None
    if kind in _TEXT_KINDS:
        text = content if isinstance(content, str) else content.data.decode("utf-8", "replace")
        return Response(content=text[:PREVIEW_TEXT_CHARS[kind]].encode("utf-8"), headers=headers,
                        media_type="text/plain; charset=utf-8")
    return Response(content=content.data, headers=headers, media_type=row["mime"])


@router.post("/{artifact_id}/close")
def close_artifact(artifact_id: str, user: str | None = Query(None)):
    """Closing the viewer keeps the artifact (TASK-2195); kept for old clients."""
    user_id = _resolve_user(user)
    return public_view(_row_or_404(_playground(), artifact_id, user_id))


@router.post("/{artifact_id}/state")
def set_state(
    artifact_id: str,
    action: str = Body(..., embed=True, pattern="^(read|unread|archive|unarchive|trash|restore)$"),
    user: str | None = Query(None),
):
    """Mailbox actions: read/unread, archive/unarchive, trash/restore."""
    user_id = _resolve_user(user)
    row = _playground().store.set_state(artifact_id, user_id, action)
    if row is None:
        raise HTTPException(status_code=404, detail="not_found")
    return public_view(row)


@router.post("/{artifact_id}/delete")
def delete_forever(artifact_id: str, user: str | None = Query(None)):
    """Delete forever, only from Trash: content and blob go, a tombstone stays."""
    user_id = _resolve_user(user)
    playground = _playground()
    row = _row_or_404(playground, artifact_id, user_id)
    if row["expired_at"] is None and row["trashed_at"] is None:
        raise HTTPException(status_code=409, detail="not_in_trash")
    return public_view(playground.expire(artifact_id, user_id))
