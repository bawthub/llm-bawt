"""Asset-kind strategies for :class:`llm_bawt.media.store.MediaStore` (TASK-847).

MediaStore was image-only through TASK-266: every upload went through the
Pillow → WebP pipeline and produced three variants. TASK-847 opens the store
to arbitrary file types so agents can hand the user a link to *any* artifact
(PDF, log, zip, csv, …) they produce.

Rather than sprinkling ``if kind == "file"`` through ``upload`` /
``read_variant`` / ``delete``, everything that differs per kind lives on a
strategy object:

``ImageAssetKind`` (``kind='image'``)
    The pre-existing behaviour, byte-for-byte: strip metadata, cap at 1568px,
    encode WebP Q85, derive ``thumb`` (256) + ``preview`` (1024). sha256 is
    computed on the *normalised* original so identical screenshots dedup.

``FileAssetKind`` (``kind='file'``)
    Opaque bytes stored verbatim under ``files/<aa>/<bb>/<sha>`` with the
    declared MIME preserved. One ``original`` variant only; asking for
    ``thumb`` / ``preview`` raises :class:`UnsupportedVariant`, which the
    HTTP layer maps to 404.

MediaStore stays kind-agnostic: it asks the strategy to *prepare* raw bytes
into ``(sha256, mime, size, dims, blobs)``, writes the blobs, and records the
row. On read / delete it resolves the strategy from the row's ``kind`` column
and asks it for the backend keys. Adding a third kind (say, ``video`` with a
poster frame) is a new subclass + one registry entry — no store changes.

Selecting a kind for an upload
------------------------------
:func:`kind_for_mime` routes the four image MIMEs Pillow handles reliably
(jpeg/png/gif/webp) to ``image`` and everything else — including exotic image
types like SVG/HEIC that would either fail or lose meaning in the WebP
pipeline — to ``file``. Callers can force a kind explicitly.
"""

from __future__ import annotations

import hashlib
import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image tuning constants — kept here (not in store.py) so the strategy that
# uses them is self-contained. ``store`` re-exports them for back-compat.
# ---------------------------------------------------------------------------

#: Longest-edge cap. Stays under Anthropic's ~1.15 MP recommendation for
#: any aspect ratio you'll see in practice.
MAX_LONG_EDGE = 1568

#: Quality settings for the WebP encoder. Q85 is the screenshot/photo sweet
#: spot; Q80/Q82 for derived variants where we want a smaller payload.
ORIGINAL_WEBP_QUALITY = 85
THUMB_WEBP_QUALITY = 80
PREVIEW_WEBP_QUALITY = 82

#: WebP encode effort. ``method=6`` is the maximum — slow on encode (we
#: only do this once per asset thanks to dedup) but produces materially
#: smaller files vs the default ``method=4``.
WEBP_METHOD = 6

THUMB_MAX = (256, 256)
PREVIEW_MAX = (1024, 1024)

#: Three on-disk subdirectories, one per image variant.
VARIANT_DIRS: dict[str, str] = {
    "original": "originals",
    "thumb": "thumb_256",
    "preview": "preview_1024",
}

VARIANT_MIME = "image/webp"

#: Subdirectory for opaque (non-image) files.
FILES_DIR = "files"

#: MIME types routed through the image pipeline. Anything else is a file.
IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})

OCTET_STREAM = "application/octet-stream"


class UnsupportedVariant(ValueError):
    """The requested variant does not exist for this asset kind.

    Subclasses ``ValueError`` so pre-TASK-847 callers that caught the old
    "variant must be one of …" ``ValueError`` keep working; the uploads route
    maps it to 404 (the asset exists, that rendition of it does not).
    """


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedBlob:
    """One backend object to write for an asset."""

    variant: str
    key: str
    data: bytes
    mime: str


@dataclass(frozen=True)
class PreparedAsset:
    """Everything MediaStore needs to persist an upload, kind-independent."""

    sha256: str
    mime_type: str
    size_bytes: int
    width: int | None
    height: int | None
    blobs: tuple[PreparedBlob, ...]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_mime(mime: str | None) -> str:
    """``'Image/PNG; charset=x'`` → ``'image/png'``; empty/None → ``''``."""
    if not mime:
        return ""
    return mime.split(";")[0].strip().lower()


def shard_key(subdir: str, sha256_hex: str, suffix: str = "") -> str:
    """Return ``<subdir>/<aa>/<bb>/<sha><suffix>``.

    Backend-agnostic relative key. For the FS backend this joins under
    ``MediaStore.root``; for S3 the factory prepends ``blobs/`` so MediaStore
    + MediaStorage can share one bucket without colliding. The two-level
    shard mirrors :class:`llm_bawt.media.storage.MediaStorage` so anyone
    walking the tree sees the same fan-out everywhere.
    """
    return f"{subdir}/{sha256_hex[:2]}/{sha256_hex[2:4]}/{sha256_hex}{suffix}"


def _normalize_to_original(raw_bytes: bytes) -> tuple[bytes, Image.Image, int, int]:
    """Run the full image normalization pipeline on raw bytes.

    Returns ``(stored_original_webp_bytes, normalized_image, width, height)``
    where ``normalized_image`` is a Pillow Image kept in memory so the
    caller can derive the thumb / preview variants without decoding the
    WebP we just produced.
    """
    src = Image.open(io.BytesIO(raw_bytes))

    # Step 1: strip ALL metadata — EXIF, ICC profile, XMP, comments. The
    # canonical way to do this in Pillow is to reconstruct the image from
    # just its pixel buffer.
    src.load()  # force-decode before we touch .mode / .size

    # Apply EXIF orientation to the pixels before the metadata is dropped,
    # so phone photos keep displaying upright.
    from PIL import ImageOps

    src = ImageOps.exif_transpose(src) or src

    # Step 2: collapse exotic color modes. Preserve alpha, including palette
    # (P) and colour-key ("transparency" info) transparency.
    if src.mode == "RGBA":
        target_mode = "RGBA"
    elif src.mode in ("LA", "PA") or "transparency" in src.info:
        src = src.convert("RGBA")
        target_mode = "RGBA"
    else:
        src = src.convert("RGB")
        target_mode = "RGB"

    # Rebuild a clean image from just the pixel buffer — this IS the
    # EXIF / ICC strip.
    stripped = Image.frombytes(target_mode, src.size, src.tobytes())

    # Step 3: downscale if needed (aspect-preserving, in place).
    if max(stripped.size) > MAX_LONG_EDGE:
        stripped.thumbnail((MAX_LONG_EDGE, MAX_LONG_EDGE), Image.LANCZOS)

    # Step 4: encode WebP. Single frame — animated GIFs collapse to their
    # first frame (intentional).
    buf = io.BytesIO()
    stripped.save(
        buf,
        format="WEBP",
        quality=ORIGINAL_WEBP_QUALITY,
        method=WEBP_METHOD,
        exif=b"",
        icc_profile=None,
    )
    original_bytes = buf.getvalue()
    return original_bytes, stripped, stripped.size[0], stripped.size[1]


def _encode_variant(normalized: Image.Image, max_size: tuple[int, int], quality: int) -> bytes:
    """Derive a fit-inside WebP variant from the already-normalized image."""
    img = normalized.copy()
    img.thumbnail(max_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(
        buf,
        format="WEBP",
        quality=quality,
        method=WEBP_METHOD,
        exif=b"",
        icc_profile=None,
    )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------


class AssetKind(ABC):
    """Per-kind policy: how bytes are prepared and where they live."""

    #: Value stored in ``media_assets.kind``.
    name: ClassVar[str]
    #: Servable variant names, ``original`` first.
    variants: ClassVar[tuple[str, ...]]

    @abstractmethod
    def prepare(self, raw: bytes, declared_mime: str) -> PreparedAsset:
        """Turn raw upload bytes into the blobs + metadata to persist."""

    @abstractmethod
    def blob_key(self, variant: str, sha256_hex: str) -> str:
        """Backend key for ``variant`` of the asset with this sha.

        :raises UnsupportedVariant: if this kind has no such variant.
        """

    def all_keys(self, sha256_hex: str) -> list[str]:
        """Every backend key this kind writes for one asset."""
        return [self.blob_key(v, sha256_hex) for v in self.variants]

    def require_variant(self, variant: str) -> None:
        if variant not in self.variants:
            raise UnsupportedVariant(
                f"variant {variant!r} not available for kind={self.name!r}; "
                f"expected one of {list(self.variants)!r}"
            )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<AssetKind {self.name}>"


class ImageAssetKind(AssetKind):
    """Normalised WebP + thumb/preview — the original MediaStore contract."""

    name = "image"
    variants = ("original", "thumb", "preview")

    def blob_key(self, variant: str, sha256_hex: str) -> str:
        self.require_variant(variant)
        return shard_key(VARIANT_DIRS[variant], sha256_hex, ".webp")

    def prepare(self, raw: bytes, declared_mime: str) -> PreparedAsset:
        original_bytes, normalized, width, height = _normalize_to_original(raw)
        sha256_hex = hashlib.sha256(original_bytes).hexdigest()
        # Variants derive from the same normalized buffer (NOT re-decoding
        # the WebP we just produced — that would stack lossy compression).
        thumb_bytes = _encode_variant(normalized, THUMB_MAX, THUMB_WEBP_QUALITY)
        preview_bytes = _encode_variant(normalized, PREVIEW_MAX, PREVIEW_WEBP_QUALITY)
        blobs = (
            PreparedBlob("original", self.blob_key("original", sha256_hex), original_bytes, VARIANT_MIME),
            PreparedBlob("thumb", self.blob_key("thumb", sha256_hex), thumb_bytes, VARIANT_MIME),
            PreparedBlob("preview", self.blob_key("preview", sha256_hex), preview_bytes, VARIANT_MIME),
        )
        return PreparedAsset(
            sha256=sha256_hex,
            mime_type=VARIANT_MIME,
            size_bytes=len(original_bytes),
            width=width,
            height=height,
            blobs=blobs,
        )


class FileAssetKind(AssetKind):
    """Opaque bytes, stored verbatim. One variant, MIME preserved."""

    name = "file"
    variants = ("original",)

    def blob_key(self, variant: str, sha256_hex: str) -> str:
        self.require_variant(variant)
        # No extension on the key: it must be reconstructible from the sha
        # alone (dedup means two different filenames can share one blob).
        return shard_key(FILES_DIR, sha256_hex)

    def prepare(self, raw: bytes, declared_mime: str) -> PreparedAsset:
        mime = normalize_mime(declared_mime) or OCTET_STREAM
        sha256_hex = hashlib.sha256(raw).hexdigest()
        return PreparedAsset(
            sha256=sha256_hex,
            mime_type=mime,
            size_bytes=len(raw),
            width=None,
            height=None,
            blobs=(PreparedBlob("original", self.blob_key("original", sha256_hex), raw, mime),),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

IMAGE_KIND = ImageAssetKind()
FILE_KIND = FileAssetKind()

KINDS: dict[str, AssetKind] = {k.name: k for k in (IMAGE_KIND, FILE_KIND)}

#: Allowed values for ``media_assets.kind``.
ALLOWED_KINDS: tuple[str, ...] = tuple(KINDS)


def kind_by_name(name: str | None) -> AssetKind:
    """Resolve a ``media_assets.kind`` value. ``None``/'' → image (legacy rows).

    :raises ValueError: for an unknown kind name.
    """
    if not name:
        return IMAGE_KIND
    try:
        return KINDS[name]
    except KeyError:
        raise ValueError(f"unknown asset kind {name!r}; expected one of {ALLOWED_KINDS!r}") from None


def kind_for_row(row: dict) -> AssetKind:
    """Strategy for a ``media_assets`` row dict (pre-TASK-847 rows have no ``kind``)."""
    return kind_by_name(row.get("kind"))


def kind_for_mime(mime: str | None) -> AssetKind:
    """Pick the upload strategy for a declared MIME type."""
    return IMAGE_KIND if normalize_mime(mime) in IMAGE_MIME_TYPES else FILE_KIND
