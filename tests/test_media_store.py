"""Unit tests for :class:`llm_bawt.media.store.MediaStore` (TASK-223).

These tests run with real Pillow operations on synthesized images — we
trust the library to do the right thing and just check the *contract*
MediaStore promises (cap, EXIF-free, alpha preserved, dedup, all three
variants on disk, idempotent delete).

The DB is mocked via a tiny in-memory fake of :class:`MediaAssetStore` so
the suite stays a pure unit test — Postgres integration is covered by
TASK-224. ``tmp_path`` keeps every test in its own MEDIA_ROOT so they
can't trip over each other.
"""

from __future__ import annotations

import errno
import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pytest
from PIL import Image

from llm_bawt.media.store import (
    MAX_LONG_EDGE,
    MediaAssetNotFound,
    MediaStore,
    VARIANT_DIRS,
    VARIANT_MIME,
)


# ---------------------------------------------------------------------------
# In-memory DB fake
# ---------------------------------------------------------------------------


class FakeMediaAssetStore:
    """Drop-in stand-in for :class:`MediaAssetStore`.

    Only implements the four methods MediaStore uses: ``get_by_sha256``,
    ``get_by_id``, ``insert``, ``delete``. Keeps rows in a plain dict so
    we can assert dedup behavior without spinning up Postgres.
    """

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.by_sha: dict[str, str] = {}

    def get_by_sha256(self, sha256: str) -> Optional[dict[str, Any]]:
        rid = self.by_sha.get(sha256)
        return dict(self.rows[rid]) if rid else None

    def get_by_id(self, asset_id: str) -> Optional[dict[str, Any]]:
        row = self.rows.get(asset_id)
        return dict(row) if row else None

    def insert(self, **kwargs: Any) -> dict[str, Any]:
        existing = self.get_by_sha256(kwargs["sha256"])
        if existing is not None:
            return existing
        from llm_bawt.media.assets import new_asset_id

        rid = kwargs.pop("asset_id", None) or new_asset_id()
        row = {
            "id": rid,
            "created_at": datetime.now(timezone.utc),
            **kwargs,
        }
        self.rows[rid] = row
        self.by_sha[row["sha256"]] = rid
        return dict(row)

    def delete(self, asset_id: str) -> bool:
        row = self.rows.pop(asset_id, None)
        if row is None:
            return False
        self.by_sha.pop(row["sha256"], None)
        return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> MediaStore:
    """Build a MediaStore with a tmp_path root + in-memory fake DB."""
    return MediaStore(root=tmp_path, db=FakeMediaAssetStore())


def _png_bytes(width: int, height: int, mode: str = "RGB", color=(128, 64, 200)) -> bytes:
    """Return a synthesized PNG of the given size + mode."""
    if mode == "RGBA":
        color = (*color[:3], 200) if len(color) == 3 else color
    img = Image.new(mode, (width, height), color)
    # Draw a checkerboard so the encoder actually has something to compress
    # (a flat color hits a degenerate WebP that's not representative).
    pixels = img.load()
    block = 32
    for y in range(0, height, block):
        for x in range(0, width, block):
            if ((x // block) + (y // block)) % 2 == 0:
                if mode == "RGBA":
                    pixels[x, y] = (255, 255, 255, 200)
                else:
                    pixels[x, y] = (255, 255, 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_with_gps_exif() -> bytes:
    """Synthesize a JPEG that carries a hand-rolled EXIF APP1 segment.

    We don't pull in piexif — instead we hand-build a minimal valid TIFF
    + EXIF IFD by hand. That keeps the test dep-free. The payload is
    nonsense to a strict EXIF reader but Pillow happily parses enough of
    it to populate ``Image.info['exif']``, which is all this test needs:
    proof that EXIF *was present* before normalization and *is not present*
    after.
    """
    img = Image.new("RGB", (320, 240), (50, 100, 150))

    # Hand-rolled EXIF blob: 'Exif\0\0' header + minimal TIFF + IFD with a
    # single tag (Artist=0x013B) pointing at an ASCII string. Pillow
    # exposes this verbatim via Image.info["exif"] so we can check
    # presence/absence without needing a real EXIF parser.
    exif_payload = (
        b"Exif\x00\x00"
        b"II*\x00"  # little-endian TIFF magic, IFD offset 8
        b"\x08\x00\x00\x00"
        b"\x01\x00"  # 1 IFD entry
        b"\x3B\x01"  # Tag = Artist (0x013B)
        b"\x02\x00"  # type = ASCII
        b"\x05\x00\x00\x00"  # count = 5 bytes
        b"NICK\x00"  # inline value (fits in 4? extended)
        b"\x00\x00\x00\x00"  # next IFD = 0
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif_payload, quality=90)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_normalize_4k_png_downscales_to_1568_webp(store: MediaStore) -> None:
    """A 3840x2160 PNG should normalize to a WebP whose longest edge is
    capped at 1568px and whose sha256 is different from the raw input."""
    import hashlib

    raw = _png_bytes(3840, 2160)
    raw_sha = hashlib.sha256(raw).hexdigest()

    asset = store.upload(
        raw_bytes=raw,
        original_mime="image/png",
        source="chat_upload",
        owner_user_id="user-1",
    )

    # Sha changed — proves normalization actually ran.
    assert asset.sha256 != raw_sha
    # Mime is WebP regardless of input.
    assert asset.mime_type == "image/webp"
    assert asset.original_mime_type == "image/png"
    # Cap applied: longest edge is exactly MAX_LONG_EDGE (1568), aspect preserved.
    assert max(asset.width, asset.height) == MAX_LONG_EDGE
    assert asset.width == 1568  # 3840:2160 -> 16:9 -> width is the long edge
    assert asset.height == 882  # 1568 * (2160/3840), rounded by PIL

    # Round-trip: decode the stored original and confirm size + format.
    data, mime = store.read_variant(asset.id, "original")
    assert mime == "image/webp"
    decoded = Image.open(io.BytesIO(data))
    assert decoded.format == "WEBP"
    assert max(decoded.size) == MAX_LONG_EDGE


def test_dedup_same_image_returns_same_asset_id(store: MediaStore) -> None:
    """Uploading the same bytes twice must yield one DB row + the same id."""
    raw = _png_bytes(800, 600)

    first = store.upload(raw, "image/png", "chat_upload", owner_user_id="user-1")
    second = store.upload(raw, "image/png", "chat_upload", owner_user_id="user-2")

    assert first.id == second.id
    assert first.sha256 == second.sha256
    # Exactly one row in the fake DB.
    assert len(store.db.rows) == 1  # type: ignore[union-attr]


def test_strips_exif(store: MediaStore) -> None:
    """A JPEG carrying GPS EXIF must come out with no EXIF segment.

    JPEG carries EXIF in an APP1 segment that starts with 0xFFE1. After
    normalization to WebP we have an even simpler check: WebP has no
    APP1 markers at all, and Pillow's reader exposes ``.info`` which
    should not contain ``"exif"``.
    """
    raw = _jpeg_with_gps_exif()

    # Sanity: the source actually has EXIF before we strip it.
    src = Image.open(io.BytesIO(raw))
    src.load()
    assert "exif" in src.info, "fixture broken: source JPEG should have EXIF"

    asset = store.upload(raw, "image/jpeg", "chat_upload", owner_user_id="user-1")
    data, _ = store.read_variant(asset.id, "original")

    # WebP doesn't use APP1 markers, but Pillow surfaces EXIF (when present)
    # through ``.info["exif"]``. Either no key or empty bytes is acceptable.
    out = Image.open(io.BytesIO(data))
    out.load()
    assert out.info.get("exif", b"") in (b"", None), (
        f"EXIF leaked through normalization: {out.info.get('exif')!r}"
    )

    # And belt-and-suspenders: the raw WebP bytes don't contain the
    # APP1-style EXIF\0\0 prefix that JPEGs use.
    assert b"\xff\xe1" not in data, "WebP output contains a JPEG APP1 marker"


def test_alpha_preserved(store: MediaStore) -> None:
    """An RGBA PNG must come out as RGBA (alpha intact)."""
    raw = _png_bytes(400, 300, mode="RGBA")

    asset = store.upload(raw, "image/png", "chat_upload", owner_user_id="user-1")
    data, _ = store.read_variant(asset.id, "original")

    decoded = Image.open(io.BytesIO(data))
    decoded.load()
    assert decoded.mode == "RGBA", f"expected RGBA, got {decoded.mode}"


def test_all_three_variants_exist_on_disk(store: MediaStore, tmp_path: Path) -> None:
    """After upload, originals/thumb_256/preview_1024 each have a .webp file."""
    raw = _png_bytes(2000, 1500)

    asset = store.upload(raw, "image/png", "chat_upload", owner_user_id="user-1")
    sha = asset.sha256

    for subdir in VARIANT_DIRS.values():
        path = store.root / subdir / sha[:2] / sha[2:4] / f"{sha}.webp"
        assert path.is_file(), f"missing variant blob: {path}"
        assert path.stat().st_size > 0


def test_read_variant_returns_correct_mime(store: MediaStore) -> None:
    """All three variants return ``image/webp``."""
    raw = _png_bytes(1600, 900)
    asset = store.upload(raw, "image/png", "chat_upload", owner_user_id="user-1")

    for variant in ("original", "thumb", "preview"):
        data, mime = store.read_variant(asset.id, variant)  # type: ignore[arg-type]
        assert mime == VARIANT_MIME
        assert len(data) > 0
        decoded = Image.open(io.BytesIO(data))
        assert decoded.format == "WEBP"


def test_thumb_and_preview_are_size_bounded(store: MediaStore) -> None:
    """Derived variants honor their respective caps."""
    raw = _png_bytes(2400, 1800)
    asset = store.upload(raw, "image/png", "chat_upload", owner_user_id="user-1")

    thumb_bytes, _ = store.read_variant(asset.id, "thumb")
    thumb = Image.open(io.BytesIO(thumb_bytes))
    assert max(thumb.size) <= 256

    preview_bytes, _ = store.read_variant(asset.id, "preview")
    preview = Image.open(io.BytesIO(preview_bytes))
    assert max(preview.size) <= 1024


def test_delete_removes_blobs_and_row(store: MediaStore) -> None:
    """``delete`` removes all 3 paths + the DB row; subsequent ``stat`` is None."""
    raw = _png_bytes(800, 600)
    asset = store.upload(raw, "image/png", "chat_upload", owner_user_id="user-1")

    # Sanity: everything is there.
    sha = asset.sha256
    paths = [
        store.root / subdir / sha[:2] / sha[2:4] / f"{sha}.webp"
        for subdir in VARIANT_DIRS.values()
    ]
    assert all(p.is_file() for p in paths)
    assert store.stat(asset.id) is not None

    # Delete.
    store.delete(asset.id)

    assert all(not p.exists() for p in paths)
    assert store.stat(asset.id) is None

    # Idempotent: a second delete is a no-op.
    store.delete(asset.id)  # should not raise


def test_read_original_as_data_url_round_trip(store: MediaStore) -> None:
    """The data URL round-trips back to identical bytes."""
    import base64

    raw = _png_bytes(640, 480)
    asset = store.upload(raw, "image/png", "chat_upload", owner_user_id="user-1")

    url = store.read_original_as_data_url(asset.id)
    assert url.startswith("data:image/webp;base64,")

    blob = base64.b64decode(url.split(",", 1)[1])
    original_bytes, _ = store.read_variant(asset.id, "original")
    assert blob == original_bytes


def test_unknown_asset_id_raises(store: MediaStore) -> None:
    """``read_variant`` on a non-existent id raises :class:`MediaAssetNotFound`."""
    with pytest.raises(MediaAssetNotFound):
        store.read_variant("ma_does_not_exist", "original")


def test_invalid_source_rejected(store: MediaStore) -> None:
    """Source must be one of the three allowed enum values."""
    raw = _png_bytes(100, 100)
    with pytest.raises(ValueError):
        store.upload(raw, "image/png", "garbage_source", owner_user_id="user-1")


# ---------------------------------------------------------------------------
# NFS resilience — ESTALE handling on the bind-mounted media root
# ---------------------------------------------------------------------------
#
# Production mounts MEDIA_ROOT over NFS4 with ``actimeo=600``. The kernel
# can hand back ``errno 116 ESTALE`` from any of: ``Path.exists`` (via
# ``stat``), ``mkdir``, ``write_bytes``, ``os.replace``. These tests
# simulate those failures via ``monkeypatch`` to prove the upload
# self-heals instead of bubbling a 500 to the chat client.


def _make_estale(path: Any = "fake") -> OSError:
    """Build an :class:`OSError` shaped like a real NFS ESTALE."""
    return OSError(errno.ESTALE, "Stale file handle", str(path))


def test_write_idempotent_retries_on_estale_then_succeeds(
    store: MediaStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient ESTALE on ``Path.exists`` must not fail the upload.

    Models the real failure mode from the production log: NFS attribute
    cache hands back a stale handle, then a retry (after a fresh
    parent-dir stat) succeeds. The upload should complete and persist
    all three blob variants.
    """
    import llm_bawt.media.object_store as store_mod

    # Pretend the very first ``exists()`` call ESTALE-fails, then
    # subsequent calls behave normally. Drives ``_path_exists_nfs_safe``
    # into its retry path.
    real_exists = Path.exists
    call_count = {"n": 0}

    def fake_exists(self: Path) -> bool:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _make_estale(self)
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    # Make the retry delays instant so the test stays fast.
    monkeypatch.setattr(store_mod, "_ESTALE_RETRY_DELAYS", (0.0, 0.0, 0.0))

    raw = _png_bytes(400, 300)
    asset = store.upload(raw, "image/png", "chat_upload", owner_user_id="user-1")

    # Upload completed and all three blob variants made it to disk.
    for subdir in VARIANT_DIRS.values():
        shard = (
            store.root
            / subdir
            / asset.sha256[:2]
            / asset.sha256[2:4]
            / f"{asset.sha256}.webp"
        )
        assert shard.exists(), f"variant {subdir} missing on disk after ESTALE retry"


def test_write_idempotent_persistent_estale_on_exists_falls_through_to_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``exists`` keeps ESTALE-ing, we must still try the write.

    Content-addressing makes blind re-writes safe — failing the upload
    is strictly worse than re-writing identical bytes. This test
    confirms ``_path_exists_nfs_safe`` returns ``None`` after exhausting
    retries and the write path runs anyway.
    """
    import llm_bawt.media.object_store as store_mod

    monkeypatch.setattr(store_mod, "_ESTALE_RETRY_DELAYS", (0.0, 0.0))

    # Every ``Path.exists`` raises ESTALE — we must not crash.
    def always_estale(self: Path) -> bool:
        raise _make_estale(self)

    monkeypatch.setattr(Path, "exists", always_estale)

    target = tmp_path / "shard" / "ab" / "cd" / "deadbeef.webp"
    # Must not raise — the write path runs because ``exists`` returned None.
    store_mod._write_idempotent(target, b"hello world")
    assert target.read_bytes() == b"hello world"


def test_write_idempotent_estale_on_write_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ESTALE on the actual write (``write_bytes`` / ``os.replace``) retries.

    The whole write sequence is wrapped in the same ESTALE retry budget,
    so a transient failure during ``write_bytes`` or ``os.replace`` is
    not user-visible.
    """
    import llm_bawt.media.object_store as store_mod

    monkeypatch.setattr(store_mod, "_ESTALE_RETRY_DELAYS", (0.0, 0.0, 0.0))

    # First call to ``os.replace`` ESTALE-fails; second one succeeds.
    real_replace = os.replace
    call_count = {"n": 0}

    def fake_replace(src: Any, dst: Any) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _make_estale(dst)
        return real_replace(src, dst)

    monkeypatch.setattr(store_mod.os, "replace", fake_replace)

    target = tmp_path / "shard" / "ab" / "cd" / "blob.webp"
    store_mod._write_idempotent(target, b"payload")
    assert target.read_bytes() == b"payload"
    assert call_count["n"] == 2  # one failure + one success


def test_write_idempotent_non_estale_oserror_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only ESTALE is retried; other errors (EACCES, ENOSPC) must surface.

    Catching everything would hide real bugs. ``_path_exists_nfs_safe``
    and the write loop both narrow to ``errno.ESTALE``.
    """
    import llm_bawt.media.object_store as store_mod

    def raise_eacces(self: Path) -> bool:
        raise OSError(errno.EACCES, "Permission denied", str(self))

    monkeypatch.setattr(Path, "exists", raise_eacces)

    target = tmp_path / "shard" / "blob.webp"
    with pytest.raises(OSError) as ei:
        store_mod._write_idempotent(target, b"x")
    assert ei.value.errno == errno.EACCES


def test_upload_dedup_check_treats_estale_as_not_intact(
    store: MediaStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dedup-check ESTALE must NOT short-circuit the upload to a row whose
    blobs we couldn't verify — instead it falls through to the heal path
    that re-writes the blobs. Better to do an extra write than to hand
    the caller back an asset_id whose disk state is unknowable.
    """
    import llm_bawt.media.object_store as store_mod

    monkeypatch.setattr(store_mod, "_ESTALE_RETRY_DELAYS", (0.0,))

    # Seed the store with an asset.
    raw = _png_bytes(400, 300)
    first = store.upload(raw, "image/png", "chat_upload", owner_user_id="user-1")

    # Wipe one of the blobs to force the heal path — but make ``exists``
    # ESTALE so the heal-check itself can't verify intact-ness.
    for subdir in VARIANT_DIRS.values():
        for f in (store.root / subdir).rglob("*.webp"):
            f.unlink()

    # Make every ``Path.exists`` ESTALE — the dedup-check should treat
    # the unknown answer as "not intact" and rebuild.
    def always_estale(self: Path) -> bool:
        raise _make_estale(self)

    # Selectively ESTALE only during the upload's intact-check window —
    # but our retry helper exhausts and returns None, so this lets the
    # write path run, which has its own ESTALE handling.
    monkeypatch.setattr(Path, "exists", always_estale)

    second = store.upload(raw, "image/png", "chat_upload", owner_user_id="user-2")
    # Same asset_id preserved (heal path returns the existing row).
    assert second.id == first.id

    # And the blobs are back on disk after the heal.
    monkeypatch.undo()
    for subdir in VARIANT_DIRS.values():
        shard = (
            store.root
            / subdir
            / first.sha256[:2]
            / first.sha256[2:4]
            / f"{first.sha256}.webp"
        )
        assert shard.exists(), f"variant {subdir} missing after heal-on-ESTALE"


# ---------------------------------------------------------------------------
# TASK-847: non-image ``file`` kind
# ---------------------------------------------------------------------------

from llm_bawt.media.asset_kinds import (  # noqa: E402
    FILES_DIR,
    FILE_KIND,
    IMAGE_KIND,
    UnsupportedVariant,
    kind_for_mime,
)


def _file_key(store: MediaStore, sha: str) -> Path:
    return store.root / FILES_DIR / sha[:2] / sha[2:4] / sha


def test_kind_for_mime_routes_images_vs_files() -> None:
    assert kind_for_mime("image/png") is IMAGE_KIND
    assert kind_for_mime("Image/JPEG; charset=x") is IMAGE_KIND
    assert kind_for_mime("application/pdf") is FILE_KIND
    assert kind_for_mime("image/svg+xml") is FILE_KIND  # not Pillow-safe → verbatim
    assert kind_for_mime("") is FILE_KIND
    assert kind_for_mime(None) is FILE_KIND


def test_upload_file_stores_bytes_verbatim(store: MediaStore) -> None:
    """A PDF goes under files/<aa>/<bb>/<sha> untouched, mime + name kept."""
    raw = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n" * 20
    asset = store.upload(
        raw, "application/pdf", "agent_attachment", owner_user_id="nick", filename="report.pdf"
    )
    assert asset.kind == "file"
    assert asset.mime_type == "application/pdf"
    assert asset.original_mime_type == "application/pdf"
    assert asset.filename == "report.pdf"
    assert asset.size_bytes == len(raw)
    assert asset.width is None and asset.height is None
    import hashlib

    assert asset.sha256 == hashlib.sha256(raw).hexdigest()

    blob = _file_key(store, asset.sha256)
    assert blob.exists()
    assert blob.read_bytes() == raw
    # No image variants were produced.
    for subdir in VARIANT_DIRS.values():
        assert not list((store.root / subdir).rglob("*")) if (store.root / subdir).exists() else True

    data, mime = store.read_variant(asset.id, "original")
    assert data == raw
    assert mime == "application/pdf"


def test_upload_file_octet_stream_default_and_mime_normalised(store: MediaStore) -> None:
    a = store.upload(b"\x00\x01\x02", "", "agent_attachment", owner_user_id=None, filename="blob.bin")
    assert a.mime_type == "application/octet-stream"
    b = store.upload(b"a,b\n1,2\n", "TEXT/CSV; charset=utf-8", "chat_upload", owner_user_id="u")
    assert b.mime_type == "text/csv"
    assert b.filename is None


def test_upload_file_dedups_by_raw_sha(store: MediaStore) -> None:
    raw = b"same bytes, different names"
    first = store.upload(raw, "text/plain", "agent_attachment", owner_user_id="nick", filename="a.txt")
    second = store.upload(raw, "text/plain", "agent_attachment", owner_user_id="nick", filename="b.txt")
    assert first.id == second.id
    # First name wins — the row is content-addressed.
    assert second.filename == "a.txt"
    assert len(store.db.rows) == 1  # type: ignore[union-attr]


def test_upload_file_self_heals_missing_blob(store: MediaStore) -> None:
    raw = b"heal me" * 100
    first = store.upload(raw, "text/plain", "agent_attachment", owner_user_id="nick", filename="h.txt")
    blob = _file_key(store, first.sha256)
    blob.unlink()
    second = store.upload(raw, "text/plain", "agent_attachment", owner_user_id="nick", filename="h.txt")
    assert second.id == first.id
    assert blob.exists() and blob.read_bytes() == raw


def test_file_thumb_and_preview_unsupported(store: MediaStore) -> None:
    asset = store.upload(b"log line\n" * 10, "text/plain", "agent_attachment", owner_user_id="nick", filename="x.log")
    with pytest.raises(UnsupportedVariant):
        store.read_variant(asset.id, "thumb")
    with pytest.raises(UnsupportedVariant):
        store.read_variant(asset.id, "preview")
    # Still a ValueError for pre-TASK-847 callers.
    with pytest.raises(ValueError):
        store.read_preview_as_data_url(asset.id)
    # Unknown variant names are rejected for images too.
    img = store.upload(_png_bytes(64, 64), "image/png", "chat_upload", owner_user_id="u")
    with pytest.raises(ValueError):
        store.read_variant(img.id, "bogus")  # type: ignore[arg-type]


def test_delete_file_removes_blob_and_row(store: MediaStore) -> None:
    asset = store.upload(b"bye" * 50, "application/zip", "agent_attachment", owner_user_id="nick", filename="z.zip")
    blob = _file_key(store, asset.sha256)
    assert blob.exists()
    store.delete(asset.id)
    assert not blob.exists()
    assert store.stat(asset.id) is None
    store.delete(asset.id)  # idempotent


def test_forced_file_kind_keeps_png_verbatim(store: MediaStore) -> None:
    """kind='file' bypasses the WebP pipeline even for a PNG."""
    raw = _png_bytes(100, 80)
    asset = store.upload_file(raw, "image/png", "agent_attachment", owner_user_id="nick", filename="raw.png")
    assert asset.kind == "file"
    assert asset.mime_type == "image/png"
    data, mime = store.read_variant(asset.id, "original")
    assert data == raw and mime == "image/png"


def test_image_upload_rows_carry_kind_image(store: MediaStore) -> None:
    """The image path is unchanged and now labels itself."""
    asset = store.upload(_png_bytes(300, 200), "image/png", "chat_upload", owner_user_id="u", filename="shot.png")
    assert asset.kind == "image"
    assert asset.mime_type == VARIANT_MIME
    assert asset.filename == "shot.png"
    for subdir in VARIANT_DIRS.values():
        assert (store.root / subdir / asset.sha256[:2] / asset.sha256[2:4] / f"{asset.sha256}.webp").exists()


def test_legacy_row_without_kind_reads_as_image(store: MediaStore) -> None:
    """Pre-TASK-847 rows have no ``kind`` key — they must resolve to image keys."""
    asset = store.upload(_png_bytes(50, 50), "image/png", "chat_upload", owner_user_id="u")
    row = store.db.rows[asset.id]  # type: ignore[union-attr]
    row.pop("kind", None)
    row.pop("filename", None)
    data, mime = store.read_variant(asset.id, "thumb")
    assert mime == VARIANT_MIME and data[:4] == b"RIFF"
    store.delete(asset.id)
    assert store.stat(asset.id) is None


def test_thumbnail_mime_type_follows_the_thumbnail_not_the_generation():
    from llm_bawt.service.routes.media import thumbnail_mime_type
    assert thumbnail_mime_type("thumbnails/03/76/abc.jpg") == "image/jpeg"
    assert thumbnail_mime_type("images/aa/bb/abc.png") == "image/png"
    assert thumbnail_mime_type("images/aa/bb/abc.webp") == "image/webp"
    assert thumbnail_mime_type("videos/aa/bb/abc.mp4") == "image/jpeg"
