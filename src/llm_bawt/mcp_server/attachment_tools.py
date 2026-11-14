"""``attachment_get``: load an uploaded image into the turn (TASK-471).

A bot is often handed an image only as a reference: a chat upload id
(``ma_…``), a chat upload URL, a task-UI serve path
(``/api/uploads/serve/originals/…``) or a task whose card carries images.
The public origin sits behind sign-in, so guessing a host to fetch the path
fails. This tool resolves the reference server-side and returns the image as
an MCP image block, which works in every harness and in headless runs.

- Chat uploads are read from llm-bawt's own media store (the preview variant,
  which is what vision inlining uses elsewhere).
- Task-UI uploads are fetched from BawtHub's serve route on its internal
  address (``BAWTHUB_UPLOADS_INTERNAL_URL``), which needs no sign-in.
- A task reference returns that card's images, up to ``max_images``.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

import httpx
from mcp.server.mcpserver import Image

from .media_generation_tools import _sniff_image_format
from .server import mcp
from .task_api import api_get
from .task_images import ImageRefError, parse_image_ref

_TASK_REF = re.compile(r"TASK-\d+", re.IGNORECASE)
#: BawtHub's internal web address; its upload serve route needs no sign-in.
_DEFAULT_UPLOADS_URL = "http://frontend-prod:3000"
#: Refuse anything larger: an image block this big is not useful in context.
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGES_CAP = 8


def _uploads_base() -> str:
    return (os.getenv("BAWTHUB_UPLOADS_INTERNAL_URL", "").strip() or _DEFAULT_UPLOADS_URL).rstrip("/")


def _read_chat_upload(asset_id: str) -> bytes:
    from llm_bawt.media import get_media_store
    from llm_bawt.media.asset_kinds import UnsupportedVariant
    from llm_bawt.media.object_store import BlobBackendUnavailable

    store = get_media_store()
    try:
        data, _mime = store.read_variant(asset_id, "preview")
    except UnsupportedVariant as exc:
        raise ImageRefError(f"upload {asset_id} is not an image") from exc
    except LookupError as exc:
        raise ImageRefError(f"unknown upload id: {asset_id}") from exc
    except FileNotFoundError as exc:
        raise ImageRefError(f"upload {asset_id} has no stored image") from exc
    except BlobBackendUnavailable as exc:
        raise ImageRefError(f"upload storage is unavailable for {asset_id}") from exc
    return data


async def _fetch_task_upload(path: str) -> bytes:
    url = f"{_uploads_base()}{path}"
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=False) as client:
        resp = await client.get(url)
    if resp.status_code == 404:
        raise ImageRefError(f"no stored upload at {path}")
    if resp.status_code != 200:
        raise ImageRefError(f"BawtHub returned {resp.status_code} for {path}")
    if not resp.headers.get("content-type", "").startswith("image/"):
        raise ImageRefError(f"{path} is not an image")
    return resp.content


async def _load(ref: str) -> bytes:
    asset_id, stored = parse_image_ref(ref)
    if asset_id:
        data = await asyncio.to_thread(_read_chat_upload, asset_id)
    else:
        data = await _fetch_task_upload(stored)
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageRefError(f"{ref} is larger than {MAX_IMAGE_BYTES} bytes")
    return data


async def _task_image_refs(task_ref: str) -> list[str]:
    task = await api_get(f"/tasks/{task_ref.upper()}")
    urls = task.get("imageUrls") if isinstance(task, dict) else None
    if not isinstance(urls, list):
        raise ImageRefError(f"{task_ref.upper()} returned no image list")
    return [url for url in urls if isinstance(url, str)]


@mcp.tool(name="attachment_get", structured_output=False)
async def attachment_get(ref: str, max_images: int = 4) -> list[Any]:
    """Load an uploaded image into this turn so you can see it.

    ``ref``: a chat upload id (``ma_…``), an upload URL or path (chat upload
    or ``/api/uploads/serve/…``, with or without the BawtHub host), or a
    task (``TASK-N``) to load that card's images (in stored order, at most
    ``max_images``, capped at 8). No host or sign-in needed.
    Returns image blocks plus a short info entry; a reference that can't be
    loaded comes back as ``{"ref", "error"}`` without failing the others.
    """
    value = (ref or "").strip()
    limit = max(1, min(int(max_images or 1), MAX_IMAGES_CAP))
    try:
        if _TASK_REF.fullmatch(value):
            refs = await _task_image_refs(value)
            if not refs:
                return [{"ref": value.upper(), "images": 0, "note": "this task has no images"}]
        else:
            refs = [value]
    except ImageRefError as exc:
        return [{"ref": value, "error": str(exc)}]
    except httpx.HTTPError as exc:
        return [{"ref": value, "error": f"task lookup failed: {exc}"}]

    blocks: list[Any] = []
    loaded: list[dict[str, Any]] = []
    for item in refs[:limit]:
        try:
            data = await _load(item)
        except ImageRefError as exc:
            loaded.append({"ref": item, "error": str(exc)})
            continue
        except httpx.HTTPError as exc:
            loaded.append({"ref": item, "error": f"fetch failed: {exc}"})
            continue
        blocks.append(Image(data=data, format=_sniff_image_format(data)))
        loaded.append({"ref": item, "bytes": len(data)})
    info: dict[str, Any] = {"ref": value, "images": len(blocks), "items": loaded}
    if len(refs) > limit:
        info["omitted"] = len(refs) - limit
    return [*blocks, info]
