"""Send, open, read, close and sweep Playground artifacts.

Bytes from an existing media asset are copied into Playground's own blob at
send time, so expiry never touches the source asset or a chat attachment that
shares its bytes. Inline text is kept in the row until expiry.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any

from ..media.asset_kinds import normalize_mime
from ..media.object_store import BlobBackend, BlobNotFound, BlobRange
from .store import (
    KINDS,
    PlaygroundStore,
    blob_key_for,
    new_artifact_id,
)

logger = logging.getLogger(__name__)

TRASH_SETTING = "playground_trash_days"
DEFAULT_TRASH_DAYS = 30
#: Inline HTML/Markdown/text cap; anything bigger should be a file asset.
MAX_INLINE_BYTES = 1024 * 1024
INLINE_FORMATS = {
    "html": ("html", "text/html"),
    "markdown": ("markdown", "text/markdown"),
    "text": ("text", "text/plain"),
}


class PlaygroundError(ValueError):
    """A send the caller can fix: bad source, too large, unknown asset."""


def kind_for_mime(mime: str) -> str:
    mime = normalize_mime(mime)
    if mime == "application/pdf":
        return "pdf"
    if mime == "text/html":
        return "html"
    if mime in ("text/markdown", "text/x-markdown"):
        return "markdown"
    head = mime.split("/", 1)[0]
    if head in ("image", "video", "audio"):
        return head
    if head == "text" or mime in ("application/json", "application/xml"):
        return "text"
    return "file"


@dataclass(frozen=True)
class Source:
    kind: str
    mime: str
    filename: str | None
    data: bytes | None = None
    inline: str | None = None

    @property
    def size(self) -> int:
        if self.data is not None:
            return len(self.data)
        return len((self.inline or "").encode("utf-8"))


def source_from_inline(content: str, fmt: str) -> Source:
    if fmt not in INLINE_FORMATS:
        raise PlaygroundError(f"format must be one of {sorted(INLINE_FORMATS)}")
    if not content or not content.strip():
        raise PlaygroundError("content must not be blank")
    if len(content.encode("utf-8")) > MAX_INLINE_BYTES:
        raise PlaygroundError(f"inline content is over {MAX_INLINE_BYTES} bytes; send it as a file")
    kind, mime = INLINE_FORMATS[fmt]
    return Source(kind=kind, mime=mime, filename=None, inline=content)


def source_from_asset(media_store, asset_id: str, max_bytes: int) -> Source:
    """Copy an existing ``ma_…`` asset's original bytes."""
    asset = media_store.stat(asset_id)
    if asset is None:
        raise PlaygroundError(f"unknown asset_id: {asset_id}")
    if asset.size_bytes and asset.size_bytes > max_bytes:
        raise PlaygroundError(f"asset is over the {max_bytes}-byte limit")
    try:
        data, mime = media_store.read_variant(asset_id, "original")
    except (LookupError, FileNotFoundError) as exc:
        raise PlaygroundError(f"asset {asset_id} has no stored bytes") from exc
    if len(data) > max_bytes:
        raise PlaygroundError(f"asset is over the {max_bytes}-byte limit")
    mime = normalize_mime(mime) or "application/octet-stream"
    return Source(kind=kind_for_mime(mime), mime=mime, filename=asset.filename, data=data)


class Playground:
    """Playground operations over one store and one blob backend."""

    def __init__(self, store: PlaygroundStore, backend: BlobBackend):
        self.store = store
        self.backend = backend

    def send(self, *, user_id: str, bot_id: str, title: str, note: str | None,
             source: Source) -> dict[str, Any]:
        assert source.kind in KINDS
        artifact_id = new_artifact_id()
        blob_key = None
        if source.data is not None:
            blob_key = blob_key_for(artifact_id)
            self.backend.put(blob_key, source.data, source.mime)
        row = {
            "id": artifact_id, "user_id": user_id, "sender_bot_id": bot_id,
            "title": title, "note": note, "kind": source.kind, "mime": source.mime,
            "filename": source.filename, "size_bytes": source.size,
            "blob_key": blob_key, "inline_content": source.inline,
        }
        try:
            return self.store.insert(row)
        except Exception:
            if blob_key:
                self._delete_blob_quietly(blob_key)
            raise

    def open(self, artifact_id: str, user_id: str):
        """``(row, token)``: open or reopen; marks it read and keeps it."""
        return self.store.open(artifact_id, user_id)

    def read(self, row: dict[str, Any], range_header: str | None) -> BlobRange | str:
        """The live content: inline text, or a (ranged) blob read."""
        if row.get("inline_content") is not None:
            return row["inline_content"]
        if not row.get("blob_key"):
            raise FileNotFoundError(row["id"])
        try:
            return self.backend.get_range(row["blob_key"], range_header)
        except BlobNotFound as exc:
            raise FileNotFoundError(row["id"]) from exc

    def expire(self, artifact_id: str, user_id: str | None = None, *,
               in_trash: bool = False, trashed_before: datetime | None = None,
               ) -> dict[str, Any] | None:
        """Delete forever: clear content, delete the blob, keep a tombstone row.

        Eligibility (``in_trash``/``trashed_before``) is enforced atomically by
        the store; the blob is deleted only when this call expired the row.
        """
        row, changed = self.store.mark_expired(artifact_id, user_id, in_trash=in_trash,
                                               trashed_before=trashed_before)
        if changed and row is not None and row.get("blob_key"):
            self._delete_blob(row["id"], row["blob_key"])
            row = {**row, "blob_key": None}
        return row

    def sweep(self, trash_days: float) -> dict[str, int]:
        """Delete forever what has sat in Trash ``trash_days``; retry blob deletes."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=trash_days)
        purged = 0
        for artifact_id in self.store.trashed_before(cutoff):
            row, changed = self.store.mark_expired(artifact_id, trashed_before=cutoff)
            if changed:
                purged += 1
                if row is not None and row.get("blob_key"):
                    self._delete_blob(row["id"], row["blob_key"])
        deleted = failed = 0
        for item in self.store.pending_blob_deletes():
            if self._delete_blob(item["id"], item["blob_key"]):
                deleted += 1
            else:
                failed += 1
        return {"purged": purged, "blobs_deleted": deleted, "blob_delete_failures": failed}

    def _delete_blob(self, artifact_id: str, blob_key: str) -> bool:
        try:
            self.backend.delete(blob_key)
        except Exception as exc:  # the sweep retries while blob_key stays set
            logger.warning("playground blob delete failed for %s: %s", artifact_id,
                           type(exc).__name__)
            return False
        self.store.clear_blob_key(artifact_id, blob_key)
        return True

    def _delete_blob_quietly(self, blob_key: str) -> None:
        try:
            self.backend.delete(blob_key)
        except Exception as exc:
            logger.warning("playground orphan blob %s not deleted: %s", blob_key,
                           type(exc).__name__)


_lock = threading.Lock()
_instance: Playground | None = None


def get_playground(config=None) -> Playground:
    """Process-wide Playground on the shared engine and the media blob backend."""
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is None:
            from ..media import get_media_store
            from ..utils.config import Config
            from ..utils.db import get_shared_engine

            engine = get_shared_engine(config or Config())
            if engine is None:
                raise RuntimeError("playground needs the database")
            _instance = Playground(PlaygroundStore(engine), get_media_store().backend)
    return _instance


def trash_days(config) -> float:
    """Days an artifact stays in Trash before it is deleted forever."""
    from ..runtime_setting_resolution import resolve_global_runtime_setting

    try:
        value = float(resolve_global_runtime_setting(config, TRASH_SETTING, DEFAULT_TRASH_DAYS))
    except (TypeError, ValueError):
        value = DEFAULT_TRASH_DAYS
    return value if value > 0 else DEFAULT_TRASH_DAYS
