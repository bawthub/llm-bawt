"""Backend-pluggable blob manager for generated media (videos, thumbnails, images).

Content-addressed storage using SHA256 hashes. No binary data in the database;
keys are organized with a two-level shard prefix that mirrors :class:`MediaStore`.

Layout (key namespace, identical for FS and S3 backends)::

  videos/{sha256[:2]}/{sha256[2:4]}/{sha256}.mp4
  thumbnails/{sha256[:2]}/{sha256[2:4]}/{sha256}.jpg
  images/{sha256[:2]}/{sha256[2:4]}/{sha256}.png

For the FS backend keys join under ``MEDIA_STORAGE_PATH``; for the S3 backend
the factory prepends a ``media/`` prefix so MediaStore + MediaStorage can share
one bucket without colliding (see :mod:`llm_bawt.media.object_store`).

Backend selection (TASK-266): ``fs`` by default. Flip to S3/Garage by setting
``LLM_BAWT_STORAGE_BACKEND=s3`` along with the matching ``LLM_BAWT_S3_*``
credentials.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from .object_store import (
    BlobBackend,
    BlobNotFound,
    BlobRange,
    FsBlobBackend,
    get_blob_backend,
)

logger = logging.getLogger(__name__)

DEFAULT_STORAGE_PATH = "/app/storage/media"


# ---------------------------------------------------------------------------
# MIME guesses for upload — used as Content-Type on the backend write.
# Reading back goes through BlobRange.content_type which is set by the backend.
# ---------------------------------------------------------------------------

_UPLOAD_MIME_BY_EXT = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _mime_for_extension(extension: str) -> str:
    return _UPLOAD_MIME_BY_EXT.get(extension.lower(), "application/octet-stream")


# ---------------------------------------------------------------------------
# MediaStorage
# ---------------------------------------------------------------------------


class MediaStorage:
    """Content-addressed media blob manager backed by a :class:`BlobBackend`.

    Construction performs no network I/O — S3 clients are built lazily on
    first use, so the app boots cleanly even when Garage is down.

    For backwards compatibility the constructor still accepts a ``base_path``
    (or honors ``MEDIA_STORAGE_PATH``); it's used to root the FS backend and
    as the source for the optional read-fallback during cutover.
    """

    def __init__(
        self,
        base_path: str | None = None,
        backend: BlobBackend | None = None,
    ):
        self.base_path = Path(
            base_path or os.environ.get("MEDIA_STORAGE_PATH", DEFAULT_STORAGE_PATH)
        )
        if backend is None:
            # S3 credentials: the stored object-store connection, per
            # operation (TASK-1476).
            backend = get_blob_backend("media_storage", fs_root=self.base_path)
        self.backend = backend
        logger.info(
            "MediaStorage initialized: base_path=%s backend=%s",
            self.base_path, type(self.backend).__name__,
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, data: bytes, subdir: str, extension: str) -> tuple[str, str]:
        """Write data to content-addressed storage.

        Args:
            data: Raw bytes to store.
            subdir: Subdirectory name (e.g. 'videos', 'thumbnails').
            extension: File extension including dot (e.g. '.mp4', '.jpg').

        Returns:
            ``(relative_path, sha256_hex)`` — ``relative_path`` is the
            backend key (a forward-slash string), suitable for storing
            in the DB and passing back to :meth:`read_range` /
            :meth:`delete` / :meth:`exists`.
        """
        sha256_hex = hashlib.sha256(data).hexdigest()
        rel_path = self._build_relative_path(sha256_hex, subdir, extension)
        self.backend.put(rel_path, data, _mime_for_extension(extension))
        logger.debug(
            "Wrote %d bytes to %s (sha256=%s)", len(data), rel_path, sha256_hex[:12],
        )
        return rel_path, sha256_hex

    async def write_async(self, data: bytes, subdir: str, extension: str) -> tuple[str, str]:
        """Async wrapper around :meth:`write` (runs in thread pool)."""
        return await asyncio.to_thread(self.write, data, subdir, extension)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, relative_path: str) -> bytes:
        """Return the full body at ``relative_path``.

        :raises FileNotFoundError: no object at ``relative_path``.
        :raises BlobBackendUnavailable: backend unreachable.
        """
        try:
            return self.backend.get(relative_path)
        except BlobNotFound as e:
            raise FileNotFoundError(f"Media file not found: {relative_path}") from e

    def read_range(self, relative_path: str, range_header: str | None) -> BlobRange:
        """Return a (possibly Range-bounded) read for streaming responses.

        Pass through the HTTP ``Range`` header verbatim — both FS and S3
        backends honor the same ``bytes=…`` spec. The returned
        :class:`BlobRange` carries the bytes, content range, and content
        type the route handler needs to construct a 200 or 206 response.

        :raises FileNotFoundError: no object at ``relative_path``.
        :raises ValueError: malformed or unsatisfiable Range.
        :raises BlobBackendUnavailable: backend unreachable.
        """
        try:
            return self.backend.get_range(relative_path, range_header)
        except BlobNotFound as e:
            raise FileNotFoundError(f"Media file not found: {relative_path}") from e

    def resolve(self, relative_path: str) -> Path:
        """Resolve a relative path to an absolute filesystem path.

        **FS backend only.** Raises :class:`RuntimeError` if the active
        backend is S3 — there's no local-disk equivalent. Callers that
        previously used ``resolve()`` to feed ``FileResponse`` should
        migrate to :meth:`read_range` + ``StreamingResponse``.

        Args:
            relative_path: Backend key (forward-slash relative path).

        :raises FileNotFoundError: if the file does not exist.
        :raises RuntimeError: if the active backend is not FS.
        """
        if not isinstance(self.backend, FsBlobBackend):
            raise RuntimeError(
                "MediaStorage.resolve() requires the FS backend "
                "(LLM_BAWT_STORAGE_BACKEND=fs). For S3 mode, use read_range() "
                "and stream the BlobRange data through StreamingResponse."
            )
        abs_path = self.base_path / relative_path
        if not abs_path.is_file():
            raise FileNotFoundError(f"Media file not found: {abs_path}")
        return abs_path

    def exists(self, relative_path: str) -> bool:
        """``True`` iff the backend has an object at ``relative_path``."""
        return self.backend.exists(relative_path)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, relative_path: str) -> bool:
        """Delete the object at ``relative_path``. Idempotent.

        Returns ``True`` if the object existed before the call, ``False``
        otherwise. The pre-check exists for the legacy contract — callers
        that don't care about pre-existence can ignore the return.

        :raises BlobBackendUnavailable: backend unreachable.
        """
        existed = self.backend.exists(relative_path)
        self.backend.delete(relative_path)
        if existed:
            logger.debug("Deleted %s", relative_path)
        return existed

    async def delete_async(self, relative_path: str) -> bool:
        """Async wrapper around :meth:`delete`."""
        return await asyncio.to_thread(self.delete, relative_path)

    # ------------------------------------------------------------------
    # Poster frame extraction
    # ------------------------------------------------------------------

    async def extract_poster_frame(self, video_path: str) -> Optional[str]:
        """Extract the first frame of a video as a JPEG thumbnail.

        Backend-aware: ffmpeg needs a real file on disk to read from, so
        in S3 mode we download the video to a tempfile, run ffmpeg, then
        upload the extracted frame via the backend. In FS mode we skip
        the download and run ffmpeg straight against the resolved path.

        Args:
            video_path: Backend key for the video.

        Returns:
            Backend key for the thumbnail, or ``None`` on failure.
        """
        # Pull the video bytes through the backend regardless of mode —
        # we need them anyway to compute the sha for the thumbnail name
        # and to dedup-skip if we've already extracted from these bytes.
        try:
            video_data = await asyncio.to_thread(self.backend.get, video_path)
        except BlobNotFound:
            logger.error("Cannot extract poster frame: video not found at %s", video_path)
            return None
        except Exception as e:
            logger.error("Cannot extract poster frame: backend read failed for %s: %s", video_path, e)
            return None

        sha256_hex = hashlib.sha256(video_data).hexdigest()
        thumb_rel = self._build_relative_path(sha256_hex, "thumbnails", ".jpg")

        # Dedup: if the same video bytes have already produced a thumbnail,
        # the backend will have it under the same key — skip the work.
        try:
            if self.backend.exists(thumb_rel):
                logger.debug("Thumbnail already exists: %s", thumb_rel)
                return thumb_rel
        except Exception as e:
            # exists() can fail on a wedged backend; fall through and try
            # the extract anyway — the subsequent put() will surface the
            # real error if the backend is truly down.
            logger.warning("Thumbnail exists() probe failed for %s: %s", thumb_rel, e)

        try:
            thumb_bytes = await self._run_ffmpeg_poster(video_data)
        except Exception as e:
            logger.error("ffmpeg poster extraction failed: %s", e)
            return None

        if thumb_bytes is None:
            return None

        try:
            await asyncio.to_thread(
                self.backend.put, thumb_rel, thumb_bytes, "image/jpeg",
            )
        except Exception as e:
            logger.error("Failed to write extracted poster frame to %s: %s", thumb_rel, e)
            return None

        logger.debug("Extracted poster frame to %s", thumb_rel)
        return thumb_rel

    async def _run_ffmpeg_poster(self, video_data: bytes) -> Optional[bytes]:
        """Run ffmpeg against ``video_data`` to extract a single JPEG frame.

        ffmpeg can read stdin if invoked with ``-i pipe:0``, but its
        seek/probe behavior on streamed input is unreliable for some
        containers (notably some MP4 variants). Writing the video to a
        tempfile first matches the behavior we got from the FS-only
        version verbatim — same ffmpeg invocation, just sourced from a
        temp path instead of the canonical store path.

        Returns the JPEG bytes, or ``None`` if ffmpeg failed.
        """
        # Write video to a tempfile in a known scratch dir. Best-effort
        # cleanup via try/finally — tempfile.NamedTemporaryFile with
        # delete=True doesn't work cleanly here because we need to close
        # the handle before ffmpeg opens it (Windows portability isn't a
        # concern but the close-before-spawn pattern is clearer).
        with tempfile.TemporaryDirectory(prefix="llm_bawt_poster_") as scratch:
            scratch_path = Path(scratch)
            video_tmp = scratch_path / "in.bin"
            thumb_tmp = scratch_path / "out.jpg"
            video_tmp.write_bytes(video_data)

            try:
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-i", str(video_tmp),
                    "-vframes", "1", "-f", "image2",
                    "-y",  # overwrite
                    str(thumb_tmp),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate()
            except FileNotFoundError:
                logger.warning("ffmpeg not found; cannot extract poster frame")
                return None

            if proc.returncode != 0:
                logger.error(
                    "ffmpeg poster extraction failed (rc=%d): %s",
                    proc.returncode, stderr.decode(errors="replace")[:500],
                )
                return None

            try:
                return thumb_tmp.read_bytes()
            except FileNotFoundError:
                logger.error("ffmpeg returned 0 but no thumbnail file was produced")
                return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_relative_path(sha256_hex: str, subdir: str, extension: str) -> str:
        """Build the content-addressed backend key.

        Pattern: ``{subdir}/{sha256[:2]}/{sha256[2:4]}/{sha256}{extension}``.

        Returns a forward-slash string (not a :class:`~pathlib.Path`) so
        callers can pass it directly to backend methods regardless of
        the OS path separator.
        """
        return f"{subdir}/{sha256_hex[:2]}/{sha256_hex[2:4]}/{sha256_hex}{extension}"
