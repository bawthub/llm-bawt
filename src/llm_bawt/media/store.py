"""MediaStore — normalized, content-addressed asset storage (TASK-223 / TASK-847).

This is the single entry point for bytes flowing into llm-bawt's chat asset
store: chat uploads, tool-generated images, agent attachments, and (since
TASK-847) arbitrary files agents want to hand the user a link to.

Two asset *kinds*, one store
----------------------------
Per-kind policy lives in :mod:`llm_bawt.media.asset_kinds`; MediaStore is
kind-agnostic and only orchestrates dedup → blob writes → DB row.

``image`` (jpeg / png / gif / webp uploads)
    Pipeline on upload:

    1. ``Pillow.Image.open(BytesIO(raw_bytes))``
    2. Strip EXIF / GPS / ICC profile / XMP — we rebuild the image from its
       RGB(A) pixel buffer, which leaves no metadata behind by construction.
    3. Convert ``CMYK`` / ``P`` palette modes to ``RGB``; keep ``RGBA``.
    4. If ``max(w, h) > 1568``: ``image.thumbnail((1568, 1568), LANCZOS)``.
    5. Encode WebP, ``quality=85``, ``method=6`` → the stored "original".
    6. ``sha256(stored_original_bytes)`` — computed **after** normalization,
       so two upload paths that produce the same normalized buffer dedup.

    From the same normalized buffer: ``thumb_256`` (fit 256, Q80) and
    ``preview_1024`` (fit 1024, Q82 — the vision-model feed size).

    Why a 1568px cap? Anthropic's vision API downsizes anything over ~1.15 MP
    server-side, so extra pixels only inflate the upload and the token bill.
    Measured on real 4K screenshots: ~8 MB PNG → ~200–400 KB WebP, ~2,500–
    3,000 → ~1,200 vision tokens, no visible quality loss.

``file`` (everything else)
    Stored verbatim — bytes, MIME, and the caller's filename are preserved.
    ``sha256(raw_bytes)`` keys dedup. One ``original`` variant; ``thumb`` /
    ``preview`` raise :class:`~llm_bawt.media.asset_kinds.UnsupportedVariant`.

Storage layout
--------------
``<MEDIA_ROOT>`` (env: ``LLM_BAWT_MEDIA_ROOT``, default
``/var/lib/llm-bawt/media/blobs``; S3 backend prefixes ``blobs/``)::

    originals/<aa>/<bb>/<sha256>.webp
    thumb_256/<aa>/<bb>/<sha256>.webp
    preview_1024/<aa>/<bb>/<sha256>.webp
    files/<aa>/<bb>/<sha256>

The ``<aa>/<bb>`` shard prefix keeps any single directory from growing past
a few thousand entries. Same convention as :class:`llm_bawt.media.storage.MediaStorage`.

Database
--------
Metadata lives in the ``media_assets`` table managed by
:class:`llm_bawt.media.assets.MediaAssetStore` (TASK-222). MediaStore wraps
that Store for inserts / lookups / deletes — it doesn't talk to Postgres
directly. The ``sha256`` UNIQUE constraint there is the source of truth
for dedup; we do an explicit ``get_by_sha256`` first to avoid the bytes
write entirely when we already have the asset. The row's ``kind`` column
selects the strategy on read / delete.
"""

from __future__ import annotations

import base64
import hashlib
import re
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from .asset_kinds import (  # noqa: F401  (constants re-exported for back-compat)
    ALLOWED_KINDS,
    FILE_KIND,
    FILES_DIR,
    IMAGE_KIND,
    IMAGE_MIME_TYPES,
    MAX_LONG_EDGE,
    ORIGINAL_WEBP_QUALITY,
    PREVIEW_MAX,
    PREVIEW_WEBP_QUALITY,
    THUMB_MAX,
    THUMB_WEBP_QUALITY,
    VARIANT_DIRS,
    VARIANT_MIME,
    WEBP_METHOD,
    AssetKind,
    UnsupportedVariant,
    _encode_variant,
    _normalize_to_original,
    kind_by_name,
    kind_for_mime,
    kind_for_row,
    normalize_mime,
    shard_key,
)
from .assets import MediaAsset, MediaAssetStore, new_asset_id

# NFS ESTALE helpers + the blob-backend abstraction are canonical in
# ``object_store``. We re-export the helpers here for back-compat — older
# call sites (and tests that pre-date TASK-266) reach into
# ``llm_bawt.media.store`` for them. New code should import directly
# from ``llm_bawt.media.object_store``.
#
# Note: monkeypatching ``store._ESTALE_RETRY_DELAYS`` does NOT change the
# retry budget seen by ``_write_idempotent`` — that function lives in
# ``object_store`` and resolves the name in its own module namespace.
# Tests that need to shorten the retry budget should patch
# ``object_store._ESTALE_RETRY_DELAYS`` instead.
from .object_store import (  # noqa: F401  (re-exported for back-compat)
    BlobBackend,
    BlobBackendUnavailable,
    BlobNotFound,
    FsBlobBackend,
    S3Config,
    _ESTALE_RETRY_DELAYS,
    _flush_parent_attr_cache,
    _is_estale,
    _path_exists_nfs_safe,
    _write_idempotent,
    get_blob_backend,
    s3_config_from_env,
)

logger = logging.getLogger(__name__)


DEFAULT_MEDIA_ROOT = Path("/var/lib/llm-bawt/media/blobs")

ALLOWED_SOURCES = ("chat_upload", "tool_generated", "agent_attachment")

Variant = Literal["original", "thumb", "preview"]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MediaAssetNotFound(LookupError):
    """Raised by :meth:`MediaStore.read_variant` when no DB row matches the id."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shard_key(variant_subdir: str, sha256_hex: str) -> str:
    """Back-compat alias for the image-variant key layout.

    Pre-TASK-847 callers built ``<subdir>/<aa>/<bb>/<sha>.webp`` through this
    helper; the canonical implementation is :func:`asset_kinds.shard_key`.
    """
    return shard_key(variant_subdir, sha256_hex, ".webp")


def _row_to_asset(row: dict) -> MediaAsset:
    """Coerce a raw DB-row dict into the :class:`MediaAsset` SQLModel.

    ``MediaAssetStore`` returns ``dict[str, Any]`` from RETURNING; we want
    the typed model in the public API so downstream code can rely on
    attribute access.
    """
    return MediaAsset(**row)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class MediaStore:
    """High-level facade over a blob backend + ``media_assets`` row.

    Construct once per process via :func:`get_media_store`. The class is
    cheap to instantiate (no network I/O — S3 clients are built lazily by
    :class:`S3BlobBackend`), so tests can build their own with a
    ``tmp_path`` root or a custom backend.

    Backend selection (TASK-266): defaults to ``fs`` rooted at
    ``LLM_BAWT_MEDIA_ROOT`` (or the legacy ``DEFAULT_MEDIA_ROOT``). Flip
    to S3/Garage by setting ``LLM_BAWT_STORAGE_BACKEND=s3`` along with
    the matching ``LLM_BAWT_S3_*`` credentials; the FS root then becomes
    a fallback source if ``LLM_BAWT_S3_FALLBACK_FS=true`` (cutover only).
    """

    def __init__(
        self,
        root: Path | None = None,
        db: MediaAssetStore | None = None,
        backend: BlobBackend | None = None,
    ):
        if root is None:
            env_root = os.environ.get("LLM_BAWT_MEDIA_ROOT")
            root = Path(env_root) if env_root else DEFAULT_MEDIA_ROOT
        self.root = Path(root)
        if backend is None:
            # Default backend: env-driven (``fs`` unless STORAGE_BACKEND=s3).
            # Construction performs no network I/O — boto3 client is lazy.
            # S3 credentials: the stored object-store connection, per
            # operation (TASK-1476).
            backend = get_blob_backend("media_store", fs_root=self.root)
        self.backend = backend
        self.db = db

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload(
        self,
        raw_bytes: bytes,
        original_mime: str,
        source: str,
        owner_user_id: Optional[str],
        expires_at: Optional[datetime] = None,
        *,
        filename: Optional[str] = None,
        kind: AssetKind | str | None = None,
        replace_key: str | None = None,
    ) -> MediaAsset:
        """Prepare, deduplicate, and persist an upload of any kind.

        ``replace_key`` explicitly opts file uploads into one mutable named slot:
        successive uploads replace its bytes and reuse its asset ID/row. Callers
        must serialize usage (including consumers); normal uploads stay immutable.

        ``kind`` defaults to :func:`kind_for_mime` — the four Pillow-safe
        image MIMEs take the normalising image pipeline, everything else is
        stored verbatim as a ``file``. Pass ``kind="file"`` to force verbatim
        storage of an image (e.g. an SVG or a PNG whose metadata must
        survive).

        Dedup is keyed on the kind's sha256 (post-normalization for images,
        raw bytes for files). If a row already exists with that sha, we
        short-circuit: no re-encode, no rewrite, just return the existing
        asset. Blob writes are idempotent too — if the DB row is gone but
        the blobs are still there we skip nothing but corrupt nothing.
        """
        if source not in ALLOWED_SOURCES:
            raise ValueError(
                f"source must be one of {ALLOWED_SOURCES!r}, got {source!r}"
            )
        strategy = self._resolve_kind(kind, original_mime)
        if replace_key is not None:
            # Internal opt-in, never accepted from the public upload route.
            # Namespace prevents overwriting immutable content-addressed blobs.
            if (not re.fullmatch(r"[a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_.-]+)*", replace_key)
                    or any(p in {".", ".."} for p in replace_key.split("/"))):
                raise ValueError("replace_key must be a safe relative object name")
            if strategy.name != "file" or expires_at is not None:
                raise ValueError("Named replacement requires a file without expiration")
            if self.db is None:
                raise ValueError("Named replacement requires the asset registry")
            prepared = strategy.prepare(raw_bytes, original_mime)
            key = "named/" + replace_key
            row = self.db.replace_named(
                storage_key=key, sha256=prepared.sha256, mime_type=prepared.mime_type,
                original_mime_type=normalize_mime(original_mime) or None,
                size_bytes=prepared.size_bytes, filename=filename, source=source,
                owner_user_id=owner_user_id,
                write_blob=lambda: self.backend.replace(key, raw_bytes, prepared.mime_type),
            )
            return _row_to_asset(row)

        # Step 1: prepare. For images this is the only place we hold the
        # full decoded image; everything else just shuffles bytes around.
        prepared = strategy.prepare(raw_bytes, original_mime)
        sha256_hex = prepared.sha256

        # Step 2: dedup hit? Return early before touching the backend.
        # ``MediaAssetStore.insert`` would dedup internally, but checking
        # first skips the blob writes.
        #
        # Self-heal: if the row exists but any of its blobs are gone
        # (manual cleanup, partial restore, container reset that wiped the
        # bind-mount), we cannot just return the row — reads would 404.
        # Re-write the blobs from the bytes in hand and return the existing
        # row, so the caller's asset_id stays stable across the heal.
        _heal_existing_row: dict | None = None
        if self.db is not None:
            existing = self.db.get_by_sha256(sha256_hex)
            if existing is not None:
                # Treat "backend can't confirm" as "not intact" so we take
                # the heal path and rewrite — safer than returning a row
                # whose blobs we can't verify (NFS ESTALE ambiguity on FS,
                # transient blips on S3; a real outage surfaces on put()).
                blobs_intact = all(
                    self._backend_exists_safe(blob.key) for blob in prepared.blobs
                )
                if blobs_intact:
                    logger.debug(
                        "MediaStore upload dedup hit: sha=%s -> id=%s",
                        sha256_hex[:12],
                        existing["id"],
                    )
                    return _row_to_asset(existing)
                logger.warning(
                    "MediaStore upload dedup hit with missing blobs — self-healing: sha=%s -> id=%s",
                    sha256_hex[:12],
                    existing["id"],
                )
                # Fall through to the idempotent writes below. The DB row
                # stays; insert() would race the sha unique constraint, so
                # we skip it on the heal path.
                _heal_existing_row = existing

        # Step 3: write every blob through the backend. Content-addressed
        # keys make the writes idempotent, so racing writers can never
        # corrupt anything.
        for blob in prepared.blobs:
            self.backend.put(blob.key, blob.data, blob.mime)

        # Step 4: register in Postgres. ``insert`` is itself dedup-safe —
        # if two callers race on the same bytes, the second one gets the
        # first one's row back instead of an IntegrityError.
        if _heal_existing_row is not None:
            return _row_to_asset(_heal_existing_row)
        if self.db is not None:
            row = self.db.insert(
                sha256=sha256_hex,
                mime_type=prepared.mime_type,
                original_mime_type=normalize_mime(original_mime) or None,
                size_bytes=prepared.size_bytes,
                width=prepared.width,
                height=prepared.height,
                source=source,
                owner_user_id=owner_user_id,
                expires_at=expires_at,
                kind=strategy.name,
                filename=filename,
            )
            return _row_to_asset(row)

        # No DB configured (test path) — synthesize an ephemeral asset so
        # the contract still holds. Production always has a DB.
        return MediaAsset(
            id=new_asset_id(),
            sha256=sha256_hex,
            mime_type=prepared.mime_type,
            original_mime_type=normalize_mime(original_mime) or None,
            size_bytes=prepared.size_bytes,
            width=prepared.width,
            height=prepared.height,
            kind=strategy.name,
            filename=filename,
            source=source,
            owner_user_id=owner_user_id,
            expires_at=expires_at,
        )

    def upload_file(
        self,
        raw_bytes: bytes,
        mime_type: str,
        source: str,
        owner_user_id: Optional[str],
        *,
        filename: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> MediaAsset:
        """Store bytes verbatim as a ``file`` asset regardless of MIME (TASK-847)."""
        return self.upload(
            raw_bytes,
            mime_type,
            source,
            owner_user_id,
            expires_at,
            filename=filename,
            kind=FILE_KIND,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_variant(
        self,
        asset_id: str,
        variant: Variant,
    ) -> tuple[bytes, str]:
        """Return ``(bytes, mime_type)`` for the requested variant.

        :raises MediaAssetNotFound: if no DB row matches ``asset_id``.
        :raises FileNotFoundError: if the row exists but the blob has been
            evicted from the backend (e.g. volume restored without DB sync,
            S3 key deleted out-of-band).
        :raises BlobBackendUnavailable: if the backend can't be reached
            at all (network / 5xx). Route handlers should map this to 503.
        :raises UnsupportedVariant: if this asset's kind has no such variant
            (``thumb`` / ``preview`` on a ``file``; any unknown name). It is a
            ``ValueError`` subclass for pre-TASK-847 callers.
        """
        row = self._require_row(asset_id)
        strategy = kind_for_row(row)
        strategy.require_variant(variant)
        key = row.get("storage_key") or strategy.blob_key(variant, row["sha256"])
        try:
            data = self.backend.get(key)
        except BlobNotFound as e:
            # Preserve the legacy contract: callers expect FileNotFoundError
            # for "DB row exists but blob is gone". 404-mapping is the same
            # on the route side regardless of source.
            raise FileNotFoundError(
                f"MediaStore blob missing for asset={asset_id} variant={variant} key={key}"
            ) from e
        if row.get("storage_key") and hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise BlobBackendUnavailable("Named asset replacement is not yet consistent; retry")
        # Image rows store VARIANT_MIME; file rows store the declared MIME.
        # Legacy rows always have mime_type set, so the fallback is only
        # for hand-built test fakes.
        return data, row.get("mime_type") or VARIANT_MIME

    def read_original_as_data_url(self, asset_id: str) -> str:
        """Return the original variant as a ``data:<mime>;base64,...`` URL.

        Convenience for the LLM inlining path (TASK-225). Always returns
        the cap-bounded original — callers that want the smaller preview
        should use :meth:`read_preview_as_data_url`.
        """
        data, mime = self.read_variant(asset_id, "original")
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def read_preview_as_data_url(self, asset_id: str) -> str:
        """Return the 1024px preview variant as a ``data:image/webp;base64,...`` URL.

        Preferred for LLM vision inlining: ~55% fewer pixels than the 1568px
        original (1024² vs 1568²) at Q82, cutting the per-image token bill with
        only a modest quality loss. Callers that need maximum fidelity (e.g.
        reading dense fine print) should use :meth:`read_original_as_data_url`.
        Images only — raises :class:`UnsupportedVariant` for ``file`` assets.
        """
        data, mime = self.read_variant(asset_id, "preview")
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def stat(self, asset_id: str) -> MediaAsset | None:
        """Return the asset row without reading any bytes, or ``None``."""
        if self.db is None:
            return None
        row = self.db.get_by_id(asset_id)
        return _row_to_asset(row) if row is not None else None

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, asset_id: str) -> None:
        """Remove every blob for the asset's kind + the DB row. Idempotent.

        Order: blobs first, then DB row. If the process dies between the
        two, the next ``upload`` of the same content will overwrite
        nothing (idempotent backend put) and re-insert the row — the
        system self-heals.

        Backend deletes are idempotent: missing keys are not an error.
        Empty shard-dir cleanup (the ``<aa>/<bb>`` fan-out) lives inside
        :meth:`FsBlobBackend.delete`; the S3 backend has no equivalent
        notion of an empty prefix.

        :raises BlobBackendUnavailable: if the backend can't be reached
            at all. The DB row is left in place so the GC pass can retry.
        """
        if self.db is None:  # tests
            return
        row = self.db.get_by_id(asset_id)
        if row is None:
            return  # nothing to do

        strategy = kind_for_row(row)
        keys = [row["storage_key"]] if row.get("storage_key") else strategy.all_keys(row["sha256"])
        for key in keys:
            self.backend.delete(key)

        self.db.delete(asset_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_kind(kind: AssetKind | str | None, mime: str) -> AssetKind:
        if isinstance(kind, AssetKind):
            return kind
        if kind:
            return kind_by_name(kind)
        return kind_for_mime(mime)

    def _backend_exists_safe(self, key: str) -> bool:
        """Like ``backend.exists(key)`` but swallows backend-unavailable.

        Used by the dedup-heal check: if we can't tell whether a blob is
        intact, assume it isn't and force a re-write. The subsequent
        ``put`` will surface the real error if the backend is genuinely
        down — we just don't want a transient blip to confuse the heal
        decision into a false "intact" answer.
        """
        try:
            return self.backend.exists(key)
        except BlobBackendUnavailable as e:
            logger.warning(
                "backend exists() unavailable during dedup-heal check for key=%s: %s",
                key, e,
            )
            return False

    def _require_row(self, asset_id: str) -> dict:
        if self.db is None:
            raise MediaAssetNotFound(
                f"MediaStore has no DB attached; cannot resolve asset_id={asset_id!r}"
            )
        row = self.db.get_by_id(asset_id)
        if row is None:
            raise MediaAssetNotFound(f"No media_assets row for id={asset_id!r}")
        return row


# ---------------------------------------------------------------------------
# Process-wide accessor
# ---------------------------------------------------------------------------

_singleton: MediaStore | None = None


def get_media_store() -> MediaStore:
    """Return the process-wide :class:`MediaStore`, building on first call.

    Wires up the shared :class:`MediaAssetStore` against the active
    :class:`Config` so route handlers can grab a Store without DI plumbing.
    Mirrors :func:`llm_bawt.service.dependencies._get_or_build_store` in
    spirit — a single Store per process, lazy-built.
    """
    global _singleton
    if _singleton is None:
        from ..utils.config import Config  # late import to avoid cycle

        config = Config()
        try:
            db = MediaAssetStore(config)
        except Exception as e:
            # No DB? Construct anyway — tests + early boot can still use
            # the on-disk path. Log loudly so prod doesn't silently lose
            # metadata.
            logger.warning("MediaStore initialized without DB: %s", e)
            db = None
        _singleton = MediaStore(db=db)
    return _singleton


def reset_media_store() -> None:
    """Drop the singleton. Tests only — don't call in production."""
    global _singleton
    _singleton = None
