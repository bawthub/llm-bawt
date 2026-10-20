"""Attachment preparation for streaming chat requests.

Extracted from :mod:`chat_streaming` so the streaming coordinator owns turn
orchestration rather than object-store and multimodal decoding details.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

from ..media import MediaAssetNotFound, get_media_store
from .chat_message_text import message_content_text


async def prepare_stream_request_attachments(
    request: Any,
    *,
    user_id: str,
    log: Any,
) -> tuple[str, list[dict], list[dict], Any | None]:
    """Return prompt, LLM image bytes, durable refs, and the media store.

    Only the trailing user message is examined. Legacy inline data URLs are
    passed to the model and best-effort uploaded; asset IDs are resolved to
    preview bytes while retaining their canonical durable references.
    """
    user_prompt = ""
    user_attachments: list[dict] = []
    attachments_to_persist: list[dict] = []
    try:
        media_store = get_media_store()
    except Exception as exc:
        media_store = None
        log.error(
            "MediaStore unavailable — attachments disabled for this turn: %s",
            exc,
        )

    for message in reversed(request.messages):
        if message.role != "user":
            continue

        user_prompt = message_content_text(message.content) or ""
        if isinstance(message.content, list):
            for part in message.content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if not url.startswith("data:"):
                        continue
                    try:
                        header, data = url.split(",", 1)
                        mime = header.split(":")[1].split(";")[0]
                    except Exception:
                        continue
                    user_attachments.append({"mimeType": mime, "content": data})
                    try:
                        if media_store is None:
                            raise RuntimeError("MediaStore unavailable")
                        asset = media_store.upload(
                            raw_bytes=base64.b64decode(data, validate=False),
                            original_mime=mime,
                            source="chat_upload",
                            owner_user_id=user_id,
                        )
                        attachments_to_persist.append(
                            {"asset_id": asset.id, "kind": "image"}
                        )
                    except Exception as exc:
                        log.warning(
                            "TASK-225: failed to auto-upload legacy inline image: %s",
                            exc,
                        )

        requested_ids = list(getattr(message, "attachment_ids", None) or [])
        for asset_id in requested_ids:
            if not isinstance(asset_id, str) or not asset_id.strip():
                continue
            asset_id = asset_id.strip()
            if media_store is None:
                log.error(
                    "TASK-225: dropping attachment_id %s — MediaStore unavailable",
                    asset_id,
                )
                continue
            # TASK-847: non-image assets (PDF, log, zip, …) are persisted as
            # refs but never inlined — there is no preview rendition and
            # vision input only takes images. Agent backends still learn
            # about them through the attachment manifest built in
            # turn_stream_worker (curlable + public URLs).
            try:
                meta = await asyncio.to_thread(media_store.stat, asset_id)
            except Exception as exc:
                log.warning("TASK-847: stat failed for attachment_id=%s: %s", asset_id, exc)
                meta = None
            if meta is not None and getattr(meta, "kind", "image") != "image":
                attachments_to_persist.append({"asset_id": asset_id, "kind": meta.kind})
                continue
            try:
                data_url = await asyncio.to_thread(
                    media_store.read_preview_as_data_url, asset_id
                )
            except MediaAssetNotFound:
                log.warning(
                    "TASK-225: attachment_id not found in media_assets: %s",
                    asset_id,
                )
                continue
            except FileNotFoundError as exc:
                log.warning(
                    "TASK-225: preview blob missing for asset_id=%s, "
                    "falling back to original: %s",
                    asset_id,
                    exc,
                )
                try:
                    data_url = await asyncio.to_thread(
                        media_store.read_original_as_data_url, asset_id
                    )
                except Exception as fallback_exc:
                    log.error(
                        "TASK-225: original fallback also failed for "
                        "asset_id=%s: %s",
                        asset_id,
                        fallback_exc,
                    )
                    continue
            except Exception as exc:
                log.warning(
                    "TASK-225: MediaStore.read_preview failed for asset_id=%s: %s",
                    asset_id,
                    exc,
                )
                continue

            try:
                header, payload = data_url.split(",", 1)
                mime = header.split(":")[1].split(";")[0]
            except Exception:
                continue
            user_attachments.append({"mimeType": mime, "content": payload})
            attachments_to_persist.append({"asset_id": asset_id, "kind": "image"})
        break

    return user_prompt, user_attachments, attachments_to_persist, media_store
