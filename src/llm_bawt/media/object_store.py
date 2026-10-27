"""Object-storage abstraction for llm-bawt media blobs (TASK-266).

Why this exists
---------------
Today MediaStore and MediaStorage talk directly to the local filesystem.
We want to move those bytes to Garage S3 (self-hosted, on Unraid) without
touching DB schemas, HTTP routes, GC logic, or clients — and we want the
migration to be a config flip, not a code branch in every reader.

This module is the seam: an env-driven backend factory and two
implementations of the same protocol.

Backends
--------
``FsBlobBackend``
    Current on-disk behavior. Absorbs the NFS ESTALE retry helpers from
    :mod:`llm_bawt.media.store` so the same hardening applies to every
    caller that goes through the backend.

``S3BlobBackend``
    boto3-backed, path-style (Garage requires it). Lazy client init: no
    network I/O during construction, so the app boots cleanly even when
    Garage is down.

``FallbackReadBlobBackend``
    Cutover safety net. Wraps an S3 primary; if a GET / GET-Range / EXISTS
    misses, we ask the FS fallback. Writes and deletes go to the primary
    only. Enable by passing ``fallback_fs=True`` to the factory during the
    migration window; turn it off once you've confirmed every blob lives
    in S3.

Object-key model
----------------
Callers pass forward-slash keys identical to today's relative paths
(``originals/aa/bb/<sha>.webp``, ``videos/aa/bb/<sha>.mp4``, …). For the
FS backend the key joins onto ``fs_root``. For the S3 backend the factory
prepends a per-subsystem prefix so MediaStore and MediaStorage can share a
single bucket without colliding:

    media_store    → ``blobs/<key>``  (MediaStore — chat upload variants)
    media_storage  → ``media/<key>``  (MediaStorage — generated videos / images)

This is the same shape ``rclone copy`` will reproduce when we move the
~33 MB of existing blobs across. See TASK-266 spec for the migration
runbook.

Error mapping
-------------
``BlobNotFound``           — 404 / NoSuchKey on the backend.
``BlobBackendUnavailable`` — connection error, 5xx, auth failure. Route
                             handlers should map this to HTTP 503 JSON.

Both classes subclass standard stdlib exceptions so ``except IOError`` and
``except LookupError`` still catch them, but new code should prefer the
named exceptions.
"""

from __future__ import annotations

import errno
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BlobNotFound(LookupError):
    """No object exists at the given key. Maps to HTTP 404."""


class BlobBackendUnavailable(IOError):
    """Backend cannot be reached or returned an unexpected error.

    Connection refused, 5xx, auth failure, transport error — anything the
    caller couldn't have prevented. Route handlers should map this to
    HTTP 503 JSON; background jobs should log and skip.
    """


# ---------------------------------------------------------------------------
# Range result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlobRange:
    """Result of a read that may have been bounded by a ``Range`` header.

    ``partial=False`` means "we returned the whole object" — caller should
    send HTTP 200. ``partial=True`` means we honored a byte range — caller
    sends HTTP 206 with ``Content-Range: bytes {start}-{end}/{total_size}``.
    """

    data: bytes
    start: int        # inclusive byte offset of first returned byte
    end: int          # inclusive byte offset of last returned byte
    total_size: int   # size of the entire object on the backend
    content_type: str
    partial: bool


# ---------------------------------------------------------------------------
# NFS resilience (absorbed from store.py — see that module for context)
# ---------------------------------------------------------------------------
#
# Production llm-bawt bind-mounts MEDIA_ROOT onto an NFS4 share from
# Unraid. Stale-file-handle errors (errno 116, ESTALE) surface whenever
# the local NFS attribute cache gets out of sync with the server: mover
# relocations, server restarts, autofs remounts, etc. The blob layout is
# content-addressed (sha256), so any write is idempotent — the right
# strategy on ESTALE is "flush the parent attr cache, sleep, retry."

#: Retry delays for ESTALE backoff. Total budget ~1.55s.
_ESTALE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)


def _is_estale(e: OSError) -> bool:
    """True iff ``e`` is the NFS stale-file-handle error."""
    return getattr(e, "errno", None) == errno.ESTALE


def _flush_parent_attr_cache(path: Path) -> None:
    """Best-effort: force the NFS client to re-validate ``path.parent``."""
    try:
        path.parent.stat()
    except OSError:
        pass


def _path_exists_nfs_safe(path: Path) -> Optional[bool]:
    """``Path.exists()`` that survives NFS ESTALE.

    Returns ``True`` / ``False`` for a confident answer, ``None`` when the
    filesystem keeps refusing to give one. ``None`` is "don't know,
    proceed defensively" — for a content-addressed store that means
    "write anyway", which is always safe.
    """
    last_err: Optional[OSError] = None
    for attempt, delay in enumerate(_ESTALE_RETRY_DELAYS, start=1):
        try:
            return path.exists()
        except OSError as e:
            if not _is_estale(e):
                raise
            last_err = e
            logger.warning(
                "NFS ESTALE on exists(%s) attempt %d/%d; flushing parent attr cache",
                path, attempt, len(_ESTALE_RETRY_DELAYS),
            )
            _flush_parent_attr_cache(path)
            time.sleep(delay)
    logger.error(
        "NFS ESTALE persisted on exists(%s) after %d retries; "
        "treating as 'unknown' (last error: %s)",
        path, len(_ESTALE_RETRY_DELAYS), last_err,
    )
    return None


def _write_idempotent(path: Path, data: bytes) -> None:
    """Atomic write with NFS ESTALE retry. Skips if ``path`` already exists.

    The tempfile name embeds pid+thread id so concurrent writers can't
    clobber each other mid-write. ``os.replace`` is atomic on POSIX.
    """
    exists = _path_exists_nfs_safe(path)
    if exists is True:
        return

    tmp = path.with_suffix(
        f"{path.suffix}.tmp.{os.getpid()}.{threading.get_ident()}"
    )

    def _do_write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(data)
        os.replace(tmp, path)

    last_err: Optional[OSError] = None
    for attempt, delay in enumerate(_ESTALE_RETRY_DELAYS, start=1):
        try:
            _do_write()
            return
        except OSError as e:
            if not _is_estale(e):
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
            last_err = e
            logger.warning(
                "NFS ESTALE on write(%s) attempt %d/%d; flushing parent attr cache",
                path, attempt, len(_ESTALE_RETRY_DELAYS),
            )
            _flush_parent_attr_cache(path)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            time.sleep(delay)

    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    assert last_err is not None
    raise last_err


# ---------------------------------------------------------------------------
# Range helpers
# ---------------------------------------------------------------------------


def _parse_byte_range(header: str, total: int) -> tuple[int, int]:
    """Parse a single ``Range: bytes=…`` header.

    Supports ``bytes=start-end``, ``bytes=start-``, and ``bytes=-suffix``.
    Multi-range requests collapse to the first range — clients seek video,
    they don't fetch segmented downloads through this surface.

    Raises ``ValueError`` for malformed or unsatisfiable ranges; callers
    should map that to HTTP 416 Requested Range Not Satisfiable.
    """
    if not header.lower().startswith("bytes="):
        raise ValueError(f"unsupported Range header: {header!r}")
    spec = header[6:].split(",", 1)[0].strip()
    if "-" not in spec:
        raise ValueError(f"malformed Range: {header!r}")
    lo, hi = spec.split("-", 1)
    try:
        if not lo and hi:
            n = int(hi)
            if n <= 0:
                raise ValueError(f"suffix length must be positive: {header!r}")
            start = max(total - n, 0)
            end = total - 1
        elif lo and not hi:
            start = int(lo)
            end = total - 1
        elif lo and hi:
            start = int(lo)
            end = min(int(hi), total - 1)
        else:
            raise ValueError(f"malformed Range: {header!r}")
    except ValueError as e:
        raise ValueError(f"malformed Range numbers: {header!r}") from e
    if start < 0 or start > end or start >= total:
        raise ValueError(f"unsatisfiable Range: {header!r} for size={total}")
    return start, end


_MIME_BY_EXT = {
    "webp": "image/webp",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "mp4": "video/mp4",
    "webm": "video/webm",
    "mov": "video/quicktime",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
}


def _guess_mime_from_key(key: str) -> str:
    """Cheap MIME guess from file extension.

    Used by the FS backend (no metadata) and as a fallback for S3 GETs
    that come back without a ``Content-Type``. Callers that have richer
    info (e.g. the DB row's ``mime_type``) should pass that instead and
    not rely on the backend's guess.
    """
    if "." not in key:
        return "application/octet-stream"
    ext = key.rsplit(".", 1)[-1].lower()
    return _MIME_BY_EXT.get(ext, "application/octet-stream")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class BlobBackend(Protocol):
    """The minimal interface every storage backend exposes.

    Implementations are expected to be **thread-safe** — the FastAPI app
    serves concurrent requests on a single backend instance.

    Construction must not perform network I/O (boto3 clients are built
    lazily in ``S3BlobBackend``), so a misconfigured Garage doesn't keep
    the app from booting.
    """

    def put(self, key: str, data: bytes, content_type: str) -> None:
        """Store ``data`` at ``key``. Idempotent for content-addressed keys.

        :raises BlobBackendUnavailable: backend unreachable or refused.
        """

    def replace(self, key: str, data: bytes, content_type: str) -> None:
        """Atomically replace one named object, even if its key already exists."""

    def get(self, key: str) -> bytes:
        """Return the full body at ``key``.

        :raises BlobNotFound: no object at ``key``.
        :raises BlobBackendUnavailable: backend unreachable.
        """

    def get_range(self, key: str, range_header: str | None) -> BlobRange:
        """Return a (possibly Range-bounded) read.

        When ``range_header`` is ``None`` returns the whole object
        (``partial=False``). When set, honors the ``bytes=…`` spec and
        returns ``partial=True``.

        :raises BlobNotFound: no object at ``key``.
        :raises ValueError: ``range_header`` is malformed or unsatisfiable.
        :raises BlobBackendUnavailable: backend unreachable.
        """

    def delete(self, key: str) -> None:
        """Remove ``key``. **Idempotent**: missing keys are not an error.

        :raises BlobBackendUnavailable: backend unreachable.
        """

    def exists(self, key: str) -> bool:
        """``True`` iff the backend has an object at ``key``.

        :raises BlobBackendUnavailable: backend unreachable.
        """


# ---------------------------------------------------------------------------
# FS backend
# ---------------------------------------------------------------------------


class FsBlobBackend:
    """Local-filesystem backend with NFS ESTALE hardening.

    ``root`` is the directory the keys hang off of. Each key is interpreted
    as a forward-slash relative path beneath ``root``.

    Construction tries to ``mkdir(parents=True, exist_ok=True)`` but tolerates
    failure — the root may live on an NFS mount that's currently flapping.
    Subsequent I/O calls will surface the real error.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(
                "FsBlobBackend root unavailable (%s): %s — operations will "
                "fail until the mount recovers",
                self.root, e,
            )

    # ------------------------------------------------------------------

    def _abs(self, key: str) -> Path:
        """Resolve ``key`` to an absolute path beneath ``root``.

        Rejects absolute keys and ``..`` segments — a backend caller
        should never be able to escape the root via a crafted key.
        """
        if not key:
            raise ValueError("blob key must be non-empty")
        if key.startswith("/"):
            raise ValueError(f"blob key must be relative, got {key!r}")
        parts = Path(key).parts
        if ".." in parts:
            raise ValueError(f"blob key may not contain '..': {key!r}")
        return self.root / key

    # ------------------------------------------------------------------

    def put(self, key: str, data: bytes, content_type: str) -> None:
        del content_type  # FS layer has no use for the MIME hint
        _write_idempotent(self._abs(key), data)

    def replace(self, key: str, data: bytes, content_type: str) -> None:
        import tempfile

        path = self._abs(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
                temporary = Path(out.name)
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def get(self, key: str) -> bytes:
        path = self._abs(key)
        # exists() is the NFS-hardened check; read_bytes() can still race
        # with a concurrent delete, so we catch FileNotFoundError below too.
        existence = _path_exists_nfs_safe(path)
        if existence is False:
            raise BlobNotFound(key)
        try:
            return path.read_bytes()
        except FileNotFoundError as e:
            raise BlobNotFound(key) from e

    def get_range(self, key: str, range_header: str | None) -> BlobRange:
        path = self._abs(key)
        try:
            total = path.stat().st_size
        except FileNotFoundError as e:
            raise BlobNotFound(key) from e

        if range_header:
            start, end = _parse_byte_range(range_header, total)
            partial = True
        else:
            start, end = 0, max(total - 1, 0)
            partial = False

        length = end - start + 1 if total > 0 else 0
        with path.open("rb") as f:
            f.seek(start)
            data = f.read(length)

        return BlobRange(
            data=data,
            start=start,
            end=end,
            total_size=total,
            content_type=_guess_mime_from_key(key),
            partial=partial,
        )

    def delete(self, key: str) -> None:
        path = self._abs(key)
        try:
            path.unlink()
        except FileNotFoundError:
            return  # NoSuchKey == success

        # Clean up empty shard prefixes so we don't leak thousands of
        # empty <aa>/<bb> dirs after GC sweeps. Mirror the loop in
        # MediaStore.delete().
        parent = path.parent
        try:
            while parent != self.root and parent.is_dir():
                parent.rmdir()  # raises if not empty — that's our break
                parent = parent.parent
        except OSError:
            pass

    def exists(self, key: str) -> bool:
        return _path_exists_nfs_safe(self._abs(key)) is True


# ---------------------------------------------------------------------------
# S3 backend
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class S3Config:
    """Connection + auth bundle for an S3-compatible backend.

    Defaults target Garage; raise the timeouts only if you're running
    against a remote provider with higher latency.
    """

    endpoint: str
    region: str
    bucket: str
    access_key: str = field(repr=False)
    secret_key: str = field(repr=False)
    #: TCP connect timeout. Short on purpose — a wedged Garage shouldn't
    #: tie up app workers.
    connect_timeout: float = 5.0
    #: Per-request read timeout. Larger to accommodate the slowest
    #: legitimate object (small generated videos, MBs at most for our use).
    read_timeout: float = 30.0
    #: boto3 retry budget. Standard mode adds exponential backoff between
    #: attempts; we keep the count low so a real outage surfaces quickly.
    max_attempts: int = 3


class S3ConfigSource(Protocol):
    """Supplies the :class:`S3Config` to use now; raises when none is usable."""

    def current(self) -> S3Config: ...


class S3BlobBackend:
    """S3-compatible backend, lazily-initialized boto3 client.

    Designed for Garage:

    - ``addressing_style="path"`` — Garage rejects virtual-host style.
    - ``signature_version="s3v4"`` — required by Garage and modern S3.
    - Short timeouts — see :class:`S3Config`.

    The boto3 client is built on first property access, behind a lock.
    Until then, instantiating this class does no network I/O and never
    fails because Garage is down.

    ``cfg`` is either a fixed :class:`S3Config` or a source with a
    ``current()`` method (TASK-1476: the stored ``object-store`` connection,
    see :mod:`llm_bawt.media.object_store_credentials`). A source is asked
    before every operation; when its answer changes (rotated keys, a new
    endpoint or bucket) the client is rebuilt, so no restart is needed.
    """

    def __init__(self, cfg: S3Config | S3ConfigSource, key_prefix: str = ""):
        self._source = None if isinstance(cfg, S3Config) else cfg
        self._static_cfg = cfg if isinstance(cfg, S3Config) else None
        # Always end with "/" so concat is safe; empty stays empty.
        if key_prefix and not key_prefix.endswith("/"):
            key_prefix = key_prefix + "/"
        self.key_prefix = key_prefix
        self._client_lock = threading.Lock()
        # (client, the config it was built from), published as one tuple so
        # a reader can never see a client paired with another config.
        self._pair: tuple[Any, S3Config] | None = None

    @property
    def cfg(self) -> S3Config:
        """The config to use now. A source raises when none is usable."""
        if self._source is not None:
            return self._source.current()
        return self._static_cfg

    # ------------------------------------------------------------------
    # Lazy client construction
    # ------------------------------------------------------------------

    @property
    def client(self):
        """Return the boto3 S3 client, building on first access."""
        return self._connection()[0]

    def _connection(self):
        """The client and the config it was built from, as one consistent pair.

        A fixed config builds once (double-checked locking). A source's
        config is re-read each call and the client is rebuilt when it differs
        from the one the current client was built with. The pair is read once
        and published as one tuple, so a concurrent rebuild can't hand this
        caller a new client with its old config.
        """
        cfg = self.cfg
        pair = self._pair
        if pair is not None and (self._source is None or pair[1] == cfg):
            return pair
        with self._client_lock:
            pair = self._pair
            if pair is None or (self._source is not None and pair[1] != cfg):
                pair = (self._build_client(cfg), cfg)
                self._pair = pair
            return pair

    def _build_client(self, cfg: S3Config):
        # Local import — keeps boto3 off the hot path for processes that
        # never touch S3 and lets the app boot even if the package is
        # missing on a dev box.
        import boto3
        from botocore.config import Config as BotoConfig

        client = boto3.client(
            "s3",
            endpoint_url=cfg.endpoint,
            region_name=cfg.region,
            aws_access_key_id=cfg.access_key,
            aws_secret_access_key=cfg.secret_key,
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                connect_timeout=cfg.connect_timeout,
                read_timeout=cfg.read_timeout,
                retries={
                    "max_attempts": cfg.max_attempts,
                    "mode": "standard",
                },
            ),
        )
        logger.info(
            "S3BlobBackend client built: endpoint=%s region=%s bucket=%s prefix=%r",
            cfg.endpoint, cfg.region, cfg.bucket, self.key_prefix,
        )
        return client

    def _full_key(self, key: str) -> str:
        if not key:
            raise ValueError("blob key must be non-empty")
        if key.startswith("/"):
            raise ValueError(f"blob key must be relative, got {key!r}")
        return self.key_prefix + key

    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------

    @staticmethod
    def _is_not_found(err) -> bool:
        """True if ``err`` is the S3 'key doesn't exist' shape.

        Garage / boto3 may surface this as ``NoSuchKey``, ``404``, or just
        an HTTP 404 with no body — we cover all three.
        """
        # Late import so this module loads without botocore in the env.
        from botocore.exceptions import ClientError

        if not isinstance(err, ClientError):
            return False
        code = err.response.get("Error", {}).get("Code")
        status = err.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in {"NoSuchKey", "404", "NotFound"} or status == 404

    @staticmethod
    def _is_invalid_range(err) -> bool:
        """True if ``err`` is the S3 416-equivalent."""
        from botocore.exceptions import ClientError

        if not isinstance(err, ClientError):
            return False
        code = err.response.get("Error", {}).get("Code")
        status = err.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in {"InvalidRange", "InvalidArgument"} or status == 416

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def put(self, key: str, data: bytes, content_type: str) -> None:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            client, cfg = self._connection()
            client.put_object(
                Bucket=cfg.bucket,
                Key=self._full_key(key),
                Body=data,
                ContentType=content_type,
            )
        except ClientError as e:
            raise BlobBackendUnavailable(f"S3 PUT {key!r} failed: {e}") from e
        except BotoCoreError as e:
            raise BlobBackendUnavailable(f"S3 PUT {key!r} transport error: {e}") from e

    def replace(self, key: str, data: bytes, content_type: str) -> None:
        # S3 PUT publishes the whole replacement object atomically.
        self.put(key, data, content_type)

    def get(self, key: str) -> bytes:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            client, cfg = self._connection()
            resp = client.get_object(
                Bucket=cfg.bucket,
                Key=self._full_key(key),
            )
        except ClientError as e:
            if self._is_not_found(e):
                raise BlobNotFound(key) from e
            raise BlobBackendUnavailable(f"S3 GET {key!r} failed: {e}") from e
        except BotoCoreError as e:
            raise BlobBackendUnavailable(f"S3 GET {key!r} transport error: {e}") from e

        return resp["Body"].read()

    def get_range(self, key: str, range_header: str | None) -> BlobRange:
        from botocore.exceptions import BotoCoreError, ClientError

        kwargs: dict = {"Key": self._full_key(key)}
        if range_header:
            # Forward the client's Range header verbatim. boto3 / S3 will
            # surface InvalidRange / 416 if it's unsatisfiable.
            kwargs["Range"] = range_header

        try:
            client, cfg = self._connection()
            resp = client.get_object(Bucket=cfg.bucket, **kwargs)
        except ClientError as e:
            if self._is_not_found(e):
                raise BlobNotFound(key) from e
            if self._is_invalid_range(e):
                raise ValueError(f"unsatisfiable Range: {range_header!r}") from e
            raise BlobBackendUnavailable(f"S3 GET {key!r} failed: {e}") from e
        except BotoCoreError as e:
            raise BlobBackendUnavailable(f"S3 GET {key!r} transport error: {e}") from e

        body = resp["Body"].read()
        content_type = resp.get("ContentType") or _guess_mime_from_key(key)
        content_range = resp.get("ContentRange") or ""
        content_length = resp.get("ContentLength") or len(body)

        if content_range and content_range.lower().startswith("bytes "):
            # "bytes 0-1023/2048" → start=0 end=1023 total=2048
            rng, _, total_str = content_range[len("bytes "):].partition("/")
            start_s, _, end_s = rng.partition("-")
            try:
                start = int(start_s)
                end = int(end_s)
                total = int(total_str)
            except ValueError:
                start, end, total = 0, content_length - 1, content_length
            partial = True
        else:
            start = 0
            end = max(content_length - 1, 0)
            total = content_length
            partial = False

        return BlobRange(
            data=body,
            start=start,
            end=end,
            total_size=total,
            content_type=content_type,
            partial=partial,
        )

    def delete(self, key: str) -> None:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            client, cfg = self._connection()
            client.delete_object(
                Bucket=cfg.bucket,
                Key=self._full_key(key),
            )
        except ClientError as e:
            if self._is_not_found(e):
                return  # idempotent: missing is success
            raise BlobBackendUnavailable(f"S3 DELETE {key!r} failed: {e}") from e
        except BotoCoreError as e:
            raise BlobBackendUnavailable(f"S3 DELETE {key!r} transport error: {e}") from e

    def exists(self, key: str) -> bool:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            client, cfg = self._connection()
            client.head_object(
                Bucket=cfg.bucket,
                Key=self._full_key(key),
            )
            return True
        except ClientError as e:
            if self._is_not_found(e):
                return False
            raise BlobBackendUnavailable(f"S3 HEAD {key!r} failed: {e}") from e
        except BotoCoreError as e:
            raise BlobBackendUnavailable(f"S3 HEAD {key!r} transport error: {e}") from e


# ---------------------------------------------------------------------------
# Fallback wrapper
# ---------------------------------------------------------------------------


class FallbackReadBlobBackend:
    """Cutover safety net: S3 primary, FS read-fallback on 404.

    Writes and deletes go to the primary only — we're cutting *over* to S3,
    not running dual-write. Reads fall back to FS so a partially-migrated
    state (some blobs on S3, some still on disk) still serves cleanly.

    Drop this wrapper once you've confirmed every blob lives in S3.
    """

    def __init__(self, primary: BlobBackend, fs_fallback: BlobBackend):
        self.primary = primary
        self.fs_fallback = fs_fallback

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.primary.put(key, data, content_type)

    def replace(self, key: str, data: bytes, content_type: str) -> None:
        self.primary.replace(key, data, content_type)

    def get(self, key: str) -> bytes:
        try:
            return self.primary.get(key)
        except BlobNotFound:
            logger.debug("blob fallback: GET %s missed primary, trying fs", key)
            return self.fs_fallback.get(key)

    def get_range(self, key: str, range_header: str | None) -> BlobRange:
        try:
            return self.primary.get_range(key, range_header)
        except BlobNotFound:
            logger.debug("blob fallback: GET-Range %s missed primary, trying fs", key)
            return self.fs_fallback.get_range(key, range_header)

    def delete(self, key: str) -> None:
        # Delete from primary only. The FS side is going away — its leftover
        # blobs will be swept by the cutover cleanup step in the runbook.
        self.primary.delete(key)

    def exists(self, key: str) -> bool:
        return self.primary.exists(key) or self.fs_fallback.exists(key)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


#: S3 key prefix per subsystem. Both subsystems share one bucket per app.
_S3_KEY_PREFIXES: dict[str, str] = {
    "media_store": "blobs/",     # chat upload variants (originals, thumb_256, preview_1024)
    "media_storage": "media/",   # generated media (videos, thumbnails, images)
    "tool_results": "toolblobs/",  # TASK-594: overflow tool-call results (content-addressed, transient)
    "turn_diffs": "diffs/",  # TASK-661: per-turn changed-file before/after bytes (content-addressed)
}

#: Recognized values for ``LLM_BAWT_STORAGE_BACKEND``.
BackendKind = Literal["fs", "s3"]


def _env_backend_kind() -> str:
    return (os.environ.get("LLM_BAWT_STORAGE_BACKEND") or "fs").strip().lower()


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def s3_config_from_env() -> S3Config | None:
    """Build an :class:`S3Config` from env vars, or ``None`` if incomplete.

    Required vars: ``LLM_BAWT_S3_ENDPOINT``, ``LLM_BAWT_S3_REGION``,
    ``LLM_BAWT_S3_BUCKET``, ``LLM_BAWT_S3_ACCESS_KEY``,
    ``LLM_BAWT_S3_SECRET_KEY``.

    Callers should treat ``None`` as "S3 not configured" — combined with
    ``LLM_BAWT_STORAGE_BACKEND=fs`` (the default) this is a no-op.

    Deprecated (TASK-1476): only the legacy fallback of
    :func:`~llm_bawt.media.object_store_credentials.resolve_s3_config`,
    used while no ``object-store`` connection is stored. Don't pass its
    result to :func:`get_blob_backend`; that would bypass the record.
    """
    endpoint = os.environ.get("LLM_BAWT_S3_ENDPOINT")
    region = os.environ.get("LLM_BAWT_S3_REGION")
    bucket = os.environ.get("LLM_BAWT_S3_BUCKET")
    access_key = os.environ.get("LLM_BAWT_S3_ACCESS_KEY")
    secret_key = os.environ.get("LLM_BAWT_S3_SECRET_KEY")
    if not all([endpoint, region, bucket, access_key, secret_key]):
        return None
    return S3Config(
        endpoint=endpoint,
        region=region,
        bucket=bucket,
        access_key=access_key,
        secret_key=secret_key,
    )


def get_blob_backend(
    kind: str,
    *,
    fs_root: Path | None = None,
    s3_cfg: S3Config | None = None,
    fallback_fs: bool | None = None,
) -> BlobBackend:
    """Construct a backend for the named subsystem.

    Parameters
    ----------
    kind
        ``"media_store"`` or ``"media_storage"``. Selects the S3 key
        prefix (``blobs/`` or ``media/`` respectively) so both subsystems
        share one bucket without colliding.
    fs_root
        Directory the FS backend hangs keys off of. Required when
        ``LLM_BAWT_STORAGE_BACKEND=fs`` (the default). Required as the
        fallback source if ``fallback_fs=True``.
    s3_cfg
        A fixed :class:`S3Config` for the S3 backend (tests, tools). Leave
        it unset in the app: the backend then uses the stored
        ``object-store`` connection, re-read per operation, and the
        ``LLM_BAWT_S3_*`` env vars only while no record exists.
    fallback_fs
        Wraps the S3 primary so 404 GETs retry from ``fs_root``. Defaults
        to the value of ``LLM_BAWT_S3_FALLBACK_FS`` (boolean env var),
        otherwise ``False``. Only meaningful when the primary is S3.

    Returns
    -------
    BlobBackend
        Constructed without network I/O — the boto3 client is built lazily.
    """
    if kind not in _S3_KEY_PREFIXES:
        raise ValueError(
            f"unknown blob backend kind: {kind!r}; "
            f"expected one of {list(_S3_KEY_PREFIXES)}"
        )

    backend_kind = _env_backend_kind()
    if fallback_fs is None:
        fallback_fs = _env_bool("LLM_BAWT_S3_FALLBACK_FS", default=False)

    if backend_kind == "fs":
        if fs_root is None:
            raise ValueError("FS backend selected but fs_root not provided")
        return FsBlobBackend(fs_root)

    if backend_kind == "s3":
        source: S3Config | S3ConfigSource
        if s3_cfg is not None:
            source = s3_cfg
        else:
            # TASK-1476: the stored object-store connection, re-read per
            # operation (env vars only while no record exists).
            from .object_store_credentials import default_s3_config_source

            source = default_s3_config_source()
        primary = S3BlobBackend(source, key_prefix=_S3_KEY_PREFIXES[kind])
        if fallback_fs:
            if fs_root is None:
                raise ValueError("fallback_fs=True requires fs_root for the fallback")
            return FallbackReadBlobBackend(primary, FsBlobBackend(fs_root))
        return primary

    raise ValueError(
        f"unknown LLM_BAWT_STORAGE_BACKEND value: {backend_kind!r} "
        f"(expected 'fs' or 's3')"
    )
