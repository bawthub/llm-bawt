"""HTTP routes for the content-addressed media store (TASK-224 / TASK-847).

Endpoints
---------

::

    POST   /v1/uploads
        body: multipart `file=...`  OR
              application/json {"data_url": "...", "filename": "..."}
        query: ?source=chat_upload | tool_generated | agent_attachment
        auth:  X-Entity-Id header (owner scoping)
        returns: asset metadata + variant URLs + public_url (see
                 ``asset_to_upload_response_dict``).

    GET    /v1/uploads/{asset_id}[?download=1] -> original bytes
                                                 (WebP for images, verbatim
                                                 for files; ?download=1 forces
                                                 Content-Disposition: attachment)
    GET    /v1/uploads/{asset_id}/thumb        -> 256px variant (images only)
    GET    /v1/uploads/{asset_id}/preview      -> 1024px variant (images only)
    DELETE /v1/uploads/{asset_id}              -> owner-only delete

These routes wrap :class:`MediaStore` (TASK-223). The store handles
normalization, dedup, variant generation, and disk layout; this module
adds the HTTP surface: auth, content negotiation, caching headers, and
error mapping.

Auth model
----------

- ``POST`` requires the ``X-Entity-Id`` header — the value is recorded as
  ``owner_user_id`` and used by ``DELETE`` to gate access.
- ``DELETE`` requires ``X-Entity-Id`` to match the asset's owner; mismatch
  returns ``403 Forbidden``.
- ``GET`` is *unauthenticated* by design. Asset IDs are ``ma_<ulid>`` and
  the underlying paths are sha256-derived (unguessable), which mirrors the
  bawthub ``Upload`` model's behaviour. If you want stronger ACLs, do not
  put the asset URL in a place where unauthenticated callers can see it.

Limits
------

- Images (``image/jpeg`` / ``png`` / ``gif`` / ``webp``): 15 MB max raw —
  the store normalises down to a 1568px-cap WebP, so anything bigger is
  misuse.
- Every other MIME (TASK-847) is stored verbatim as a ``file`` asset, capped
  by ``LLM_BAWT_MAX_FILE_UPLOAD_BYTES`` (default 100 MB). There is no MIME
  allowlist any more; the LAN boundary is the trust boundary (see
  ``reference/security-model.md``). A handful of browser-active types
  (``text/html``, ``image/svg+xml``, …) are always served as
  ``Content-Disposition: attachment`` so they never execute in the
  BawtHub origin.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import re
from typing import Literal, Optional
from urllib.parse import quote

from fastapi import APIRouter, File, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from ...media import (
    MediaAssetNotFound,
    MediaStore,
    asset_to_upload_response_dict,
    get_media_store,
)
from ...media.asset_kinds import (
    IMAGE_KIND,
    IMAGE_MIME_TYPES,
    OCTET_STREAM,
    AssetKind,
    UnsupportedVariant,
    kind_for_mime,
    normalize_mime,
)
from ...media.object_store import BlobBackendUnavailable

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Uploads"])


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Hard cap on raw *image* upload size. We normalise down to ~200-400 KB
#: WebP, so anything above this is either a misuse of the API or an attack.
MAX_RAW_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB

#: Default cap for non-image files (TASK-847). Override with
#: ``LLM_BAWT_MAX_FILE_UPLOAD_BYTES``. The body is buffered in memory, so
#: keep this in the "large log / small video" range, not "disk image".
DEFAULT_MAX_FILE_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

#: MIME types routed through the image pipeline. Kept under the historic
#: name for callers that imported it; everything else is now a ``file``.
ACCEPTED_MIME_TYPES = IMAGE_MIME_TYPES

#: Fallback MIME when nothing better is known (images always report WebP).
RESPONSE_MIME = "image/webp"

#: MIME types that browsers will *execute* if served inline from our origin.
#: Always forced to ``Content-Disposition: attachment``.
FORCE_ATTACHMENT_MIMES = frozenset(
    {
        OCTET_STREAM,
        "text/html",
        "application/xhtml+xml",
        "image/svg+xml",
        "application/javascript",
        "text/javascript",
    }
)

#: Longest filename we will store / echo in Content-Disposition.
MAX_FILENAME_CHARS = 200

#: One year, immutable — the URL is content-addressed so the bytes can
#: never change. Browsers can cache aggressively.
CACHE_CONTROL = "public, max-age=31536000, immutable"

#: Data URL regex: ``data:<mime>;base64,<payload>``. We require base64
#: encoding (the common case) and surface anything else as 400.
_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[\w.+/-]+);base64,(?P<payload>[A-Za-z0-9+/=\s]+)$"
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _store() -> MediaStore:
    """Resolve the process-wide :class:`MediaStore`.

    Pulled out as a tiny wrapper so tests can monkeypatch a different
    accessor if they want isolation from the singleton.
    """
    return get_media_store()


def _require_entity_id(x_entity_id: Optional[str]) -> str:
    """Return the entity ID or raise 401 if missing.

    The existing service has no centralised auth middleware; we enforce
    presence here so producers know to wire the header. Treat empty or
    whitespace-only values the same as missing.
    """
    if not x_entity_id or not x_entity_id.strip():
        raise HTTPException(
            status_code=401,
            detail="X-Entity-Id header is required",
        )
    return x_entity_id.strip()


# ---------------------------------------------------------------------------
# Body parsing
# ---------------------------------------------------------------------------


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    """Return ``(raw_bytes, declared_mime)`` from a ``data:<mime>;base64,<...>`` URL.

    Raises 400 on malformed input. We trust the declared MIME only enough
    to gate it through the 415 check — the store re-derives the real
    format from the bytes via Pillow.
    """
    match = _DATA_URL_RE.match(data_url.strip())
    if not match:
        raise HTTPException(
            status_code=400,
            detail="data_url must be of form 'data:<mime>;base64,<payload>'",
        )
    declared_mime = match.group("mime").lower()
    payload = re.sub(r"\s+", "", match.group("payload"))
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as e:  # ValueError or binascii.Error
        raise HTTPException(status_code=400, detail=f"data_url base64 decode failed: {e}")
    return raw, declared_mime


def max_file_upload_bytes() -> int:
    """Cap for ``file``-kind uploads; env-tunable, read per request."""
    raw = os.environ.get("LLM_BAWT_MAX_FILE_UPLOAD_BYTES", "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            logger.warning("LLM_BAWT_MAX_FILE_UPLOAD_BYTES=%r is not an int; using default", raw)
    return DEFAULT_MAX_FILE_UPLOAD_BYTES


def _sanitize_filename(name: object) -> Optional[str]:
    """Reduce a caller-supplied filename to a safe display name, or ``None``.

    Basename only (both separators), control characters stripped, whitespace
    collapsed, capped at :data:`MAX_FILENAME_CHARS` while keeping the
    extension. Never used to build a storage path — keys are sha-derived —
    so this is about ``Content-Disposition`` hygiene and UI display.
    """
    if not isinstance(name, str):
        return None
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    base = "".join(ch for ch in base if ch.isprintable() and ch not in '"\x7f')
    base = re.sub(r"\s+", " ", base).strip().strip(".")
    if not base:
        return None
    if len(base) > MAX_FILENAME_CHARS:
        stem, dot, ext = base.rpartition(".")
        if dot and 0 < len(ext) <= 16 and stem:
            keep = MAX_FILENAME_CHARS - len(ext) - 1
            base = f"{stem[:keep]}.{ext}"
        else:
            base = base[:MAX_FILENAME_CHARS]
    return base


def _resolve_mime(declared_mime: Optional[str], filename: Optional[str]) -> str:
    """Pick the MIME we trust for kind routing + storage.

    The declared type wins when it is specific. ``curl -F file=@x.pdf`` and
    hand-rolled clients frequently send ``application/octet-stream`` (or
    nothing), so in that case fall back to the filename extension. Unknown
    → ``application/octet-stream``.
    """
    mime = normalize_mime(declared_mime)
    if (not mime or mime == OCTET_STREAM) and filename:
        guessed, _ = mimetypes.guess_type(filename, strict=False)
        if guessed:
            mime = guessed.lower()
    return mime or OCTET_STREAM


def _check_size(raw: bytes, kind: AssetKind) -> None:
    """Reject uploads over the per-kind cap (413)."""
    limit = MAX_RAW_UPLOAD_BYTES if kind is IMAGE_KIND else max_file_upload_bytes()
    if len(raw) > limit:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Upload too large: {len(raw)} bytes > {limit} byte limit "
                f"for kind={kind.name}"
            ),
        )


def _content_disposition(disposition: str, filename: str) -> str:
    """RFC 6266 header value with an ASCII fallback + UTF-8 ``filename*``."""
    ascii_name = filename.encode("ascii", "ignore").decode("ascii").replace('"', "") or "download"
    value = f'{disposition}; filename="{ascii_name}"'
    if ascii_name != filename:
        value += f"; filename*=UTF-8''{quote(filename)}"
    return value


def _display_filename(asset, mime: str, variant: str) -> str:
    """Filename for Content-Disposition: stored name, else ``<id><ext>``."""
    stored = getattr(asset, "filename", None) if asset is not None else None
    asset_id = getattr(asset, "id", None) or "asset"
    if getattr(asset, "kind", "image") == "image":
        # Images are always re-encoded to WebP; the stored filename (if any)
        # describes the *source*, so swap the extension and tag the variant.
        stem = stored.rsplit(".", 1)[0] if stored else asset_id
        suffix = "" if variant == "original" else f"-{variant}"
        return f"{stem}{suffix}.webp"
    if stored:
        return stored
    ext = mimetypes.guess_extension(mime, strict=False) or ""
    return f"{asset_id}{ext}"


# ---------------------------------------------------------------------------
# POST /v1/uploads
# ---------------------------------------------------------------------------


@router.post("/v1/uploads")
async def upload_asset(
    request: Request,
    source: str = Query(
        default="chat_upload",
        description="Provenance tag; one of chat_upload | tool_generated | agent_attachment",
    ),
    x_entity_id: Optional[str] = Header(default=None, alias="X-Entity-Id"),
    file: Optional[UploadFile] = File(default=None),
):
    """Accept an image upload via multipart or JSON-with-data-URL.

    Two body shapes are supported (we sniff the content-type):

    - ``multipart/form-data`` with a ``file`` part. Standard browser form
      upload path.
    - ``application/json`` with ``{"data_url": "data:image/png;base64,...",
      "filename": "<optional>"}``. Used by paste/clipboard flows that
      already have a data URL in hand and don't want to round-trip
      through multipart.

    Returns the canonical upload-response dict — see
    :func:`asset_to_upload_response_dict` for the exact shape.
    """
    entity_id = _require_entity_id(x_entity_id)

    content_type = (request.headers.get("content-type") or "").lower()

    raw: bytes
    declared_mime: str

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        data_url = body.get("data_url")
        if not isinstance(data_url, str) or not data_url:
            raise HTTPException(
                status_code=400,
                detail="JSON body must include 'data_url' (string)",
            )
        raw, declared_mime = _decode_data_url(data_url)
        filename = _sanitize_filename(body.get("filename"))
    elif file is not None:
        # FastAPI parses multipart for us; ``file.content_type`` may be
        # missing on hand-crafted requests — ``_resolve_mime`` then falls
        # back to the filename extension.
        raw = await file.read()
        declared_mime = (file.content_type or OCTET_STREAM).lower()
        filename = _sanitize_filename(file.filename)
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "Provide either a multipart 'file' field or a JSON body with 'data_url'"
            ),
        )

    mime = _resolve_mime(declared_mime, filename)
    kind = kind_for_mime(mime)
    _check_size(raw, kind)

    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload body")

    try:
        asset = _store().upload(
            raw_bytes=raw,
            original_mime=mime,
            source=source,
            owner_user_id=entity_id,
            filename=filename,
            kind=kind,
        )
    except ValueError as e:
        # ``source`` not in ALLOWED_SOURCES surfaces here.
        raise HTTPException(status_code=400, detail=str(e))
    except BlobBackendUnavailable as e:
        # Garage / S3 unreachable. Chat itself keeps working; the upload
        # path just refuses cleanly so the frontend can retry.
        logger.error("MediaStore.upload: storage backend unavailable: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Media storage backend unavailable; try again shortly",
        )
    except Exception as e:
        # The text of an unexpected error can name the DB host and user
        # (psycopg2 connect errors do), so it goes to the log only (TASK-1561).
        logger.exception("MediaStore.upload failed")
        raise HTTPException(
            status_code=500, detail=f"upload failed ({type(e).__name__}); see the app log"
        )

    return JSONResponse(asset_to_upload_response_dict(asset))


# ---------------------------------------------------------------------------
# GET /v1/uploads/{asset_id}[/variant]
# ---------------------------------------------------------------------------


def _etag_matches(if_none_match: Optional[str], etag: str) -> bool:
    """Return True if any tag in an ``If-None-Match`` header matches ``etag``.

    Per RFC 7232 the header is a comma-separated list of entity-tags or the
    literal ``*``. We compare strong-only (we never emit a weak ETag) and
    tolerate optional surrounding whitespace.
    """
    if not if_none_match:
        return False
    candidates = [c.strip() for c in if_none_match.split(",")]
    if "*" in candidates:
        return True
    return etag in candidates


def _serve_variant(
    asset_id: str,
    variant: str,
    if_none_match: Optional[str] = None,
    *,
    download: bool = False,
) -> Response:
    """Read a variant from the store and wrap it in a cacheable response.

    Common helper for the three GET endpoints; consolidates the headers so
    the original / thumb / preview routes can't drift on caching policy.
    Honours ``If-None-Match`` by returning 304 *before* reading the blob —
    the DB-only ``stat`` call is enough to compute the ETag.

    TASK-847: every response carries ``Content-Disposition`` built from the
    stored filename. ``download=True`` (``?download=1``) or a MIME in
    :data:`FORCE_ATTACHMENT_MIMES` switches it from ``inline`` to
    ``attachment``.
    """
    store = _store()

    # Compute ETag from the asset row first so a conditional request never
    # has to read disk. ``stat`` is a single SELECT; if the asset is gone
    # we fall through to ``read_variant`` for a consistent 404 path.
    asset = store.stat(asset_id)
    etag = f'"{asset.sha256}"' if asset is not None else None

    mutable = bool(asset is not None and asset.storage_key)
    cache_control = "no-store" if mutable else CACHE_CONTROL
    if not mutable and etag and _etag_matches(if_none_match, etag):
        return Response(
            status_code=304,
            headers={"ETag": etag, "Cache-Control": CACHE_CONTROL},
        )

    try:
        data, mime = store.read_variant(asset_id, variant)  # type: ignore[arg-type]
    except MediaAssetNotFound:
        raise HTTPException(status_code=404, detail=f"Asset {asset_id!r} not found")
    except FileNotFoundError:
        # Row exists but blob is gone — treat as 404 from the client's
        # perspective; the actionable detail is logged for ops.
        logger.error("Asset %s has DB row but missing blob (variant=%s)", asset_id, variant)
        raise HTTPException(status_code=404, detail=f"Asset {asset_id!r} blob missing")
    except BlobBackendUnavailable as e:
        logger.error("upload serve: backend unavailable (asset=%s variant=%s): %s", asset_id, variant, e)
        raise HTTPException(
            status_code=503,
            detail="Media storage backend unavailable; try again shortly",
        )
    except UnsupportedVariant as e:
        # The asset exists but this rendition doesn't (thumb of a PDF).
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    mime = mime or RESPONSE_MIME
    disposition = "attachment" if (download or mime in FORCE_ATTACHMENT_MIMES) else "inline"
    headers = {
        "Cache-Control": cache_control,
        "Content-Disposition": _content_disposition(
            disposition, _display_filename(asset, mime, variant)
        ),
    }
    if etag and not mutable:
        headers["ETag"] = etag

    return Response(content=data, media_type=mime, headers=headers)


@router.get("/v1/uploads")
def list_assets(
    q: str = Query(default="", max_length=200),
    kind: Optional[Literal["image", "file"]] = None,
    source: Optional[Literal["chat_upload", "tool_generated", "agent_attachment"]] = None,
    limit: int = Query(default=48, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Tenant media registry for the admin browser; deletion remains owner-scoped.

    The internal service is LAN-trusted, like the existing asset GET routes.
    No S3 credentials or raw blob listing are exposed.
    """
    from ...media.serializers import asset_row_to_attachment_dict, public_urls

    store = _store()
    if store.db is None or getattr(store.db, "engine", None) is None:
        raise HTTPException(status_code=503, detail="Media registry unavailable")
    page = store.db.browse(q=q, kind=kind, source=source, limit=limit, offset=offset)
    page["items"] = [
        {**asset_row_to_attachment_dict(row),
         "source": row["source"], "owner_user_id": row.get("owner_user_id"),
         "created_at": row["created_at"], "storage_key": row.get("storage_key"),
         "public_url": public_urls(row["id"], row.get("kind") or "image").get("url")}
        for row in page["items"]
    ]
    return page


@router.get("/v1/uploads/{asset_id}")
async def get_original(
    asset_id: str,
    if_none_match: Optional[str] = Header(default=None, alias="If-None-Match"),
    download: bool = Query(
        default=False,
        description="Force Content-Disposition: attachment (save-as instead of inline view)",
    ),
) -> Response:
    """Return the original: 1568px-cap WebP for images, verbatim bytes for files."""
    return _serve_variant(asset_id, "original", if_none_match, download=download)


@router.get("/v1/uploads/{asset_id}/thumb")
async def get_thumb(
    asset_id: str,
    if_none_match: Optional[str] = Header(default=None, alias="If-None-Match"),
) -> Response:
    """Return the 256px thumb variant (Q80). Images only — 404 for files."""
    return _serve_variant(asset_id, "thumb", if_none_match)


@router.get("/v1/uploads/{asset_id}/preview")
async def get_preview(
    asset_id: str,
    if_none_match: Optional[str] = Header(default=None, alias="If-None-Match"),
) -> Response:
    """Return the 1024px preview variant (Q82). Images only — 404 for files."""
    return _serve_variant(asset_id, "preview", if_none_match)


# ---------------------------------------------------------------------------
# DELETE /v1/uploads/{asset_id}
# ---------------------------------------------------------------------------


@router.delete("/v1/uploads/{asset_id}", status_code=204)
async def delete_asset(
    asset_id: str,
    x_entity_id: Optional[str] = Header(default=None, alias="X-Entity-Id"),
) -> Response:
    """Owner-only delete. Removes every blob for the asset + the DB row.

    Returns ``204 No Content`` on success. Mismatched owner is ``403``;
    unknown asset is ``404``.
    """
    entity_id = _require_entity_id(x_entity_id)
    store = _store()

    asset = store.stat(asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail=f"Asset {asset_id!r} not found")

    # ``owner_user_id`` is nullable for tool-generated assets. Per the
    # TASK-224 spec, owner-NULL assets are deletable by any authenticated
    # caller (so an agent that produced an image can clean up after itself
    # without knowing a synthetic owner id). Owner-set assets require an
    # exact match — 403 (not 404) so the caller knows the asset exists but
    # they can't touch it.
    if asset.owner_user_id is not None and asset.owner_user_id != entity_id:
        raise HTTPException(
            status_code=403,
            detail="X-Entity-Id does not match asset owner",
        )

    try:
        store.delete(asset_id)
    except BlobBackendUnavailable as e:
        logger.error("MediaStore.delete: backend unavailable for %s: %s", asset_id, e)
        raise HTTPException(
            status_code=503,
            detail="Media storage backend unavailable; try again shortly",
        )
    except Exception as e:
        logger.exception("MediaStore.delete failed for %s", asset_id)
        raise HTTPException(status_code=500, detail=f"delete failed: {e}")

    return Response(status_code=204)
