"""Image attachments for the task MCP tools (TASK-1526).

BawtHub stores a task's images as ``imageUrls``: origin-relative paths the
card renders and the agent prompt turns into fetchable links. A bot names an
image by chat upload id (``ma_…``) or by any URL form of an upload; every
chat upload is stored as ``/api/chat/uploads/<id>``, the form BawtHub's own
upload proxy and ``proxifyUploadUrl`` produce. Images already uploaded through
BawtHub's task UI (``/api/uploads/serve/…``) are kept as they are.

BawtHub's PATCH replaces the whole array, so an update reads the list, edits
it and re-reads it just before writing. A list that changed in between is a
conflict and nothing is written. The window between that re-read and the
PATCH is not closed: BawtHub has no compare-and-set for ``imageUrls``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Iterable
from urllib.parse import urlsplit

from llm_bawt.media.asset_kinds import IMAGE_KIND, kind_by_name
from llm_bawt.media.serializers import PUBLIC_UPLOADS_PREFIX

logger = logging.getLogger(__name__)

_ASSET_ID = re.compile(r"ma_[0-9A-Z]{26}")
_UPLOAD_PATH = re.compile(rf"(?:/v1/uploads|{re.escape(PUBLIC_UPLOADS_PREFIX)})/({_ASSET_ID.pattern})(?:/[a-z]+)?/?")
_TASK_UPLOAD_PREFIX = "/api/uploads/serve/"
#: BawtHub's own public domain: absolute links to it are stored origin-relative.
_PRODUCT_DOMAIN = "bawthub.com"


class ImageRefError(ValueError):
    """A reference that names no usable upload."""


def _first_party(host: str) -> bool:
    host = host.lower()
    return host == _PRODUCT_DOMAIN or host.endswith(f".{_PRODUCT_DOMAIN}")


def parse_image_ref(ref: Any) -> tuple[str | None, str]:
    """``(asset_id, stored_url)`` for one reference; ``asset_id`` is None for a task-UI upload.

    Raises ImageRefError for anything that is not an upload id or upload URL.
    """
    if not isinstance(ref, str) or not ref.strip():
        raise ImageRefError("image reference must be a non-empty string")
    value = ref.strip()
    if _ASSET_ID.fullmatch(value):
        return value, f"{PUBLIC_UPLOADS_PREFIX}/{value}"
    parts = urlsplit(value)
    if parts.scheme not in ("", "http", "https") or (parts.scheme and not parts.netloc):
        raise ImageRefError(f"not an upload id or upload URL: {ref!r}")
    match = _UPLOAD_PATH.fullmatch(parts.path)
    if match:
        # The id is checked against the upload store, so the host doesn't matter.
        asset_id = match.group(1)
        return asset_id, f"{PUBLIC_UPLOADS_PREFIX}/{asset_id}"
    if parts.path.startswith(_TASK_UPLOAD_PREFIX) and ".." not in parts.path:
        if parts.netloc and not _first_party(parts.hostname or ""):
            raise ImageRefError(f"not a BawtHub upload URL: {ref!r}")
        return None, parts.path
    raise ImageRefError(f"not an upload id or upload URL: {ref!r}")


def _key(url: str) -> str:
    """Compare stored entries by upload, so a legacy absolute form matches its id."""
    try:
        return parse_image_ref(url)[1]
    except ImageRefError:
        return url


def _lookup_assets(asset_ids: list[str]) -> dict[str, dict[str, Any]]:
    from llm_bawt.media import get_media_store

    db = get_media_store().db
    if db is None:
        raise ImageRefError("the upload registry is unavailable")
    return {row["id"]: row for row in db.get_many(asset_ids)}


async def resolve_image_refs(refs: Iterable[Any]) -> list[str]:
    """Stored URLs for refs being attached; every chat upload must exist and be an image."""
    parsed = [parse_image_ref(ref) for ref in refs]
    asset_ids = _unique(asset_id for asset_id, _ in parsed if asset_id)
    if asset_ids:
        rows = await asyncio.to_thread(_lookup_assets, asset_ids)
        unknown = [asset_id for asset_id in asset_ids if asset_id not in rows]
        if unknown:
            raise ImageRefError(f"unknown upload id(s): {unknown}")
        not_images = [asset_id for asset_id in asset_ids
                      if kind_by_name(rows[asset_id].get("kind")) is not IMAGE_KIND]
        if not_images:
            raise ImageRefError(f"not image uploads: {not_images}")
    return _unique(url for _, url in parsed)


def edit_image_urls(current: list[str], add: list[str], remove: Iterable[Any]) -> list[str]:
    """``current`` plus ``add`` minus ``remove``, order kept; repeating an edit changes nothing.

    ``add`` holds stored URLs (``resolve_image_refs``). ``remove`` may name an
    upload in any form, or repeat an entry exactly as stored; removing an
    upload that no longer exists is allowed.
    """
    dropped = {_key(ref) if isinstance(ref, str) else ref for ref in remove}
    seen: dict[str, str] = {}
    for url in [*current, *add]:
        seen.setdefault(_key(url), url)
    return [url for key, url in seen.items() if key not in dropped and url not in dropped]


def _trigger_attachments(bot_id: str, message_id: str) -> list[dict[str, Any]]:
    from .server import _get_storage

    backend = _get_storage().get_backend(bot_id)
    return backend.get_attachments_for_message_ids([message_id]).get(message_id, [])


async def turn_image_urls() -> list[str]:
    """Stored URLs of the images on the signed turn's triggering message.

    The turn comes only from the signed capability, never from arguments.
    Uploads that were deleted since, or aren't images, are skipped.
    Raises TaskTurnContextError without a trusted turn.
    """
    from .task_association import current_task_turn_context

    context = current_task_turn_context()
    refs = await asyncio.to_thread(_trigger_attachments, context.bot_id, context.trigger_message_id)
    asset_ids = _unique(ref["asset_id"] for ref in refs
                        if isinstance(ref, dict) and isinstance(ref.get("asset_id"), str)
                        and _ASSET_ID.fullmatch(ref["asset_id"]))
    if not asset_ids:
        return []
    rows = await asyncio.to_thread(_lookup_assets, asset_ids)
    return [f"{PUBLIC_UPLOADS_PREFIX}/{asset_id}" for asset_id in asset_ids
            if asset_id in rows and kind_by_name(rows[asset_id].get("kind")) is IMAGE_KIND]


async def linked_turn_images(associate: bool, attach: bool) -> tuple[list[str], str | None]:
    """``(urls, error)`` for auto-attaching the linked turn's images (TASK-1526).

    Nothing when the turn isn't being linked or the caller opted out. A lookup
    failure never blocks the task write; it comes back as ``error``. Without a
    trusted turn the association itself reports that, so no error here.
    """
    if not (associate and attach):
        return [], None
    from llm_bawt.task_turn_context import TaskTurnContextError

    try:
        return await turn_image_urls(), None
    except TaskTurnContextError:
        return [], None
    except Exception as error:  # noqa: BLE001 - reported, the task write goes on
        logger.warning("turn image lookup failed: %s", error)
        return [], f"could not read this turn's images: {error}"


class ImageListChanged(ImageRefError):
    """The task's image list changed between the read and the write."""


def newly_added(urls: Iterable[str], before: Iterable[str]) -> list[str]:
    """The ``urls`` that weren't already in ``before`` (compared by upload)."""
    had = {_key(url) for url in before}
    return [url for url in urls if _key(url) not in had]


async def image_urls_patch(read_task, task_id: str, add_refs: list[Any] | None,
                           remove_refs: list[Any] | None,
                           turn_urls: Iterable[str] = ()) -> tuple[list[str] | None, list[str]]:
    """``(imageUrls for the PATCH or None when unchanged, the list as read)``.

    ``turn_urls`` are already-checked stored URLs from the linked turn.
    ``read_task`` is the task GET. The list is re-read just before returning;
    if it changed since the first read, ImageListChanged is raised and the
    caller writes nothing.
    """
    add = [*await resolve_image_refs(add_refs or []), *turn_urls]
    current = _image_urls(await read_task(task_id))
    wanted = edit_image_urls(current, add, remove_refs or [])
    if wanted == current:
        return None, current
    if _image_urls(await read_task(task_id)) != current:
        raise ImageListChanged("the task's images changed while this update was prepared; "
                               "nothing was saved, re-read and retry")
    return wanted, current


def _image_urls(task: Any) -> list[str]:
    urls = task.get("imageUrls") if isinstance(task, dict) else None
    if not isinstance(urls, list):
        raise ImageRefError("the task API returned no imageUrls for this task")
    return [url for url in urls if isinstance(url, str)]


def _unique(urls: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(urls))
