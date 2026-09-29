"""Shared dict-shape helpers for media-attachment JSON payloads.

Used by:

- ``POST /v1/uploads`` response (TASK-224, this module's primary caller).
- ``/v1/chat/completions`` persistence layer (TASK-225) — when an LLM call
  includes an image attachment, the row written to ``{bot}_messages.attachments``
  uses the same shape.
- ``/v1/history`` responses (TASK-226) — history JSON returns the attachment
  dict embedded in each message.
- bawthub frontend renderers via API docs and TypeScript types.

Keep this stable. **Other code reads it**, so do not rename keys or change
URL shapes without updating every downstream consumer first.

Two functions, two shapes:

- :func:`asset_to_attachment_dict` — the *minimum* envelope every API
  surface ships. Goes into the ``attachments`` array on chat/history rows.
- :func:`asset_to_upload_response_dict` — superset returned by
  ``POST /v1/uploads`` only; adds ``sha256``, ``size_bytes``, and
  ``original_mime_type`` so the uploader can dedup client-side, show file
  size, and render a "you uploaded a JPEG; we store WebP" hint.

Both forms share the same ``urls`` block so any consumer can use a single
helper to pick a variant URL.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable

from .asset_kinds import IMAGE_KIND, kind_by_name
from .assets import MediaAsset

logger = logging.getLogger(__name__)


#: Default user-facing origin for clickable asset links (TASK-847). Mirrors
#: ``Config.PUBLIC_ORIGIN``; read straight from the environment here so the
#: serializers stay import-light and usable without a Config instance.
DEFAULT_PUBLIC_ORIGIN = "https://app.bawthub.com"

#: BawtHub's same-origin proxy prefix for llm-bawt ``/v1/uploads/...``.
PUBLIC_UPLOADS_PREFIX = "/api/chat/uploads"


def public_origin() -> str:
    """User-facing BawtHub origin, or ``""`` when public links are disabled."""
    return (os.environ.get("LLM_BAWT_PUBLIC_ORIGIN", DEFAULT_PUBLIC_ORIGIN) or "").rstrip("/")


def variant_urls(asset_id: str, kind: str = "image") -> dict[str, str]:
    """Relative ``/v1/uploads`` URLs for an asset.

    Images keep the pre-TASK-847 ``thumb`` / ``preview`` / ``original`` trio.
    Files expose ``original`` (inline) plus ``download`` (forces
    ``Content-Disposition: attachment``) — there is no thumb/preview to
    point at, and emitting broken URLs would only confuse renderers.
    """
    base = f"/v1/uploads/{asset_id}"
    if kind_by_name(kind) is IMAGE_KIND:
        return {
            "thumb": f"{base}/thumb",
            "preview": f"{base}/preview",
            "original": base,
        }
    return {
        "original": base,
        "download": f"{base}?download=1",
    }


def public_urls(asset_id: str, kind: str = "image", origin: str | None = None) -> dict[str, str]:
    """Clickable BawtHub URLs for an asset: ``{"url": ..., "download": ...}``.

    Routed through BawtHub's ``/api/chat/uploads`` proxy so the browser hits
    the same origin (and auth) it already uses for chat thumbnails. Empty
    when no public origin is configured.
    """
    o = public_origin() if origin is None else origin.rstrip("/")
    if not o:
        return {}
    base = f"{o}{PUBLIC_UPLOADS_PREFIX}/{asset_id}"
    out = {"url": base}
    if kind_by_name(kind) is not IMAGE_KIND:
        out["download"] = f"{base}?download=1"
    return out


def _attachment_dict(
    *,
    asset_id: str,
    kind: str | None,
    mime_type: str | None,
    width: int | None,
    height: int | None,
    filename: str | None,
    size_bytes: int | None,
) -> dict:
    kind_name = kind_by_name(kind).name
    return {
        "asset_id": asset_id,
        "kind": kind_name,
        "mime_type": mime_type,
        "width": width,
        "height": height,
        "filename": filename,
        "size_bytes": size_bytes,
        "urls": variant_urls(asset_id, kind_name),
    }


def asset_to_attachment_dict(asset: MediaAsset) -> dict:
    """Return the canonical 'attachment' dict used on every API surface.

    This is the shape persisted in ``{bot}_messages.attachments`` and the
    shape returned inside ``/v1/history`` messages. Keep field names and
    URL structure stable — bawthub renderers read this directly.

    TASK-847 additions are purely additive for images: ``kind`` now reflects
    the row (``image`` | ``file``), and ``filename`` / ``size_bytes`` ride
    along so file chips can render without a second round-trip.
    """
    return _attachment_dict(
        asset_id=asset.id,
        kind=getattr(asset, "kind", None),
        mime_type=asset.mime_type,
        width=asset.width,
        height=asset.height,
        filename=getattr(asset, "filename", None),
        size_bytes=asset.size_bytes,
    )


def asset_to_upload_response_dict(asset: MediaAsset) -> dict:
    """Shape returned by ``POST /v1/uploads``. Superset of attachment dict.

    Adds ``sha256`` (for client-side dedup checks), ``size_bytes`` (for
    file-size display), and ``original_mime_type`` (the pre-normalization
    MIME the client sent — useful for "you uploaded a JPEG; we store WebP"
    UX) on top of the canonical attachment envelope.
    """
    base = asset_to_attachment_dict(asset)
    base["sha256"] = asset.sha256
    base["size_bytes"] = asset.size_bytes
    base["original_mime_type"] = asset.original_mime_type
    # TASK-847: clickable BawtHub links so an agent (or the composer) can
    # hand the user a URL that opens in a browser, not just an internal
    # http://app:8642 path.
    pub = public_urls(asset.id, base["kind"])
    base["public_url"] = pub.get("url")
    base["public_download_url"] = pub.get("download")
    return base


def asset_row_to_attachment_dict(row: dict[str, Any]) -> dict:
    """Same canonical attachment dict, but from a DB-row mapping.

    ``MediaAssetStore`` returns plain dicts (not ``MediaAsset`` SQLModel
    instances), so the history-enrichment path on TASK-226 needs an
    overload that takes a mapping. Keep this in sync with
    :func:`asset_to_attachment_dict` — both shapes are wire-identical.
    """
    return _attachment_dict(
        asset_id=row.get("id"),
        kind=row.get("kind"),
        mime_type=row.get("mime_type"),
        width=row.get("width"),
        height=row.get("height"),
        filename=row.get("filename"),
        size_bytes=row.get("size_bytes"),
    )


def _abs_url(path: str, origin: str) -> str:
    """Prefix a relative ``/v1/uploads/...`` path with ``origin``.

    Empty ``origin`` returns the path unchanged so callers degrade to
    relative URLs (still meaningful to same-origin consumers).
    """
    if not origin:
        return path
    return f"{origin.rstrip('/')}{path}"


def _fmt_size(size_bytes: Any) -> str | None:
    """Human-readable byte size (``124.3 KB``), or ``None`` if unknown."""
    try:
        n = int(size_bytes)
    except (TypeError, ValueError):
        return None
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def build_agent_image_manifest(
    refs: Iterable[dict[str, Any]],
    asset_store,
    origin: str = "",
    public_origin_override: str | None = None,
) -> str:
    """Render a plain-text 'Attached Files' manifest for agent backends.

    TASK-391 — when a chat turn carrying image attachments is dispatched
    to an agent backend (Claude Code / Codex / OpenClaw), the model can
    *see* the image bytes inline, but its shell/tools cannot fetch the
    same asset (blob: URLs are browser-local). This helper turns the tiny
    persisted refs ``[{"asset_id": "ma_xxx", "kind": "image"}, ...]`` into
    a curlable manifest that gets appended to the agent-visible prompt.

    Each ref is resolved against ``asset_store`` (``get_many``/``get_by_id``
    -> ``media_assets`` row) and rendered with its ``mime_type``,
    dimensions, size, and absolute variant URLs. URLs are absolutized with
    ``origin`` (e.g. ``http://app:8642``) so a tool running in a sibling
    container can curl them; an empty ``origin`` emits relative
    ``/v1/uploads/...`` paths unchanged.

    Returns ``""`` when ``refs`` is empty or nothing resolves — callers
    should skip injection on an empty string.

    Args:
        refs: Iterable of ``{"asset_id", "kind"}`` dicts (order preserved,
            duplicates collapsed).
        asset_store: Anything exposing ``get_by_id(asset_id) -> dict | None``
            and optionally ``get_many(ids) -> list[dict]`` (preferred —
            single round-trip).
        origin: Absolute base URL for curlable links, or "" for relative.
        public_origin_override: User-facing origin for the ``public:`` line
            (TASK-847). ``None`` reads ``LLM_BAWT_PUBLIC_ORIGIN``; ``""``
            suppresses the line.
    """
    ids: list[str] = []
    seen: set[str] = set()
    for ref in refs or []:
        if not isinstance(ref, dict):
            continue
        aid = ref.get("asset_id")
        if aid and aid not in seen:
            seen.add(aid)
            ids.append(aid)
    if not ids:
        return ""

    rows: dict[str, dict[str, Any]] = {}
    getter_many = getattr(asset_store, "get_many", None)
    if callable(getter_many):
        try:
            for r in getter_many(ids) or []:
                if r and r.get("id"):
                    rows[r["id"]] = r
        except Exception as e:
            logger.warning("build_agent_image_manifest: get_many failed: %s", e)
    if not rows:
        for aid in ids:
            try:
                r = asset_store.get_by_id(aid)
            except Exception as e:
                logger.warning(
                    "build_agent_image_manifest: get_by_id(%s) failed: %s", aid, e
                )
                r = None
            if r:
                rows[aid] = r

    lines: list[str] = []
    n = 0
    n_images = 0
    for aid in ids:
        row = rows.get(aid)
        if not row:
            continue
        n += 1
        att = asset_row_to_attachment_dict(row)
        urls = att["urls"]
        is_image = att["kind"] == "image"
        if is_image:
            n_images += 1
        meta_bits = [
            f"asset_id={att['asset_id']}",
            f"type={att.get('mime_type') or ('image' if is_image else 'file')}",
        ]
        if att.get("filename"):
            meta_bits.append(f"name={att['filename']}")
        if att.get("width") and att.get("height"):
            meta_bits.append(f"{att['width']}x{att['height']}")
        size = _fmt_size(row.get("size_bytes"))
        if size:
            meta_bits.append(size)
        lines.append(f"{n}. " + "  ".join(meta_bits))
        lines.append(f"   original: {_abs_url(urls['original'], origin)}")
        if is_image:
            lines.append(f"   preview:  {_abs_url(urls['preview'], origin)}")
            lines.append(f"   thumb:    {_abs_url(urls['thumb'], origin)}")
        pub = public_urls(att["asset_id"], att["kind"], public_origin_override)
        if pub.get("url"):
            lines.append(f"   public:   {pub['url']}")

    if not lines:
        return ""

    if n_images == n:
        header = (
            f"[Attached Images] The user attached {n} image(s) to this message. "
            "You can see them inline; your tools can fetch the same assets by "
            "curling these URLs (HTTP GET, no auth on the internal network). "
            "The `public:` link is the one to paste back to the user:"
        )
    else:
        header = (
            f"[Attached Files] The user attached {n} file(s) to this message "
            f"({n_images} image(s) visible inline). Your tools can fetch them by "
            "curling the `original` URLs (HTTP GET, no auth on the internal "
            "network). The `public:` link is the one to paste back to the user:"
        )
    return header + "\n" + "\n".join(lines)


def attachment_refs(attachments: Iterable[Any] | None) -> list[dict[str, Any]]:
    """Reduce attachment envelopes (or refs) to persisted ``{asset_id, kind}`` refs.

    Inverse of :func:`enrich_attachments_for_messages`: durable rows keep only
    the tiny ref and are re-enriched on read, so URLs never go stale. Order is
    preserved, duplicate asset ids collapse, entries without an id drop out.
    """
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in attachments or []:
        if not isinstance(item, dict):
            continue
        aid = item.get("asset_id")
        if not isinstance(aid, str) or not aid or aid in seen:
            continue
        seen.add(aid)
        refs.append({"asset_id": aid, "kind": item.get("kind") or "image"})
    return refs


def enrich_attachments_for_messages(
    messages: Iterable[dict[str, Any]],
    asset_store,
) -> None:
    """Resolve tiny ``attachments`` refs into full URL-block dicts in place.

    TASK-226 — history endpoints persist a tiny shape per row:
    ``[{"asset_id": "ma_xxx", "kind": "image"}, ...]``. Outbound HTTP
    consumers want the canonical attachment envelope including
    ``mime_type``/``width``/``height``/``urls`` so the frontend can
    render thumbnails without a second round-trip per message.

    This helper:

    1. Collects every distinct ``asset_id`` referenced by ``messages``.
    2. Runs a single ``SELECT * FROM media_assets WHERE id = ANY(:ids)``
       through ``asset_store`` — O(1) regardless of page size.
    3. Rewrites each message's ``attachments`` list to the full shape,
       preserving order.

    Missing asset IDs (e.g. deleted blobs) drop out of the rewritten
    list and trigger a single warning log per call. The message keeps
    ``attachments=[]`` rather than failing so partial deletes can never
    take down the whole history response.

    The ``attachments`` key is always set on every message after this
    call (even on messages that had no refs to begin with), so frontend
    code can iterate without defensive checks.

    Args:
        messages: Iterable of message dicts. Mutated in place.
        asset_store: Anything exposing ``get_by_id(asset_id) -> dict | None``
            **and** (optionally) ``get_many(ids) -> list[dict]``. The
            former is required; the batched variant is preferred when
            available.
    """
    # Materialize to a list so we can iterate twice (collect ids, rewrite).
    msg_list = list(messages)

    seen_ids: list[str] = []
    seen_set: set[str] = set()
    for msg in msg_list:
        for ref in msg.get("attachments") or []:
            if not isinstance(ref, dict):
                continue
            aid = ref.get("asset_id")
            if aid and aid not in seen_set:
                seen_set.add(aid)
                seen_ids.append(aid)

    # Batch-load. Fall back to per-id lookups if the store doesn't expose
    # a batch helper (the production MediaAssetStore does not at TASK-226
    # time, so we go per-id but in a single connection cycle below).
    asset_map: dict[str, dict[str, Any]] = {}
    if seen_ids:
        getter_many = getattr(asset_store, "get_many", None)
        if callable(getter_many):
            try:
                rows = getter_many(seen_ids) or []
                for r in rows:
                    if r and r.get("id"):
                        asset_map[r["id"]] = r
            except Exception as e:
                logger.warning(
                    "asset_store.get_many failed (%s); falling back to per-id lookup",
                    e,
                )

        if not asset_map:
            for aid in seen_ids:
                try:
                    row = asset_store.get_by_id(aid)
                except Exception as e:
                    logger.warning("asset_store.get_by_id(%s) failed: %s", aid, e)
                    row = None
                if row:
                    asset_map[aid] = row

    missing: list[str] = [aid for aid in seen_ids if aid not in asset_map]
    if missing:
        logger.warning(
            "history attachments: %s asset id(s) not in media_assets (deleted?): %s",
            len(missing),
            ",".join(missing[:5]),
        )

    for msg in msg_list:
        raw_refs = msg.get("attachments") or []
        resolved: list[dict[str, Any]] = []
        for ref in raw_refs:
            if not isinstance(ref, dict):
                continue
            aid = ref.get("asset_id")
            row = asset_map.get(aid) if aid else None
            if row:
                resolved.append(asset_row_to_attachment_dict(row))
            # else: silently drop unresolved refs (already logged above)
        msg["attachments"] = resolved
