"""HTTP route tests for ``/v1/uploads*`` (TASK-224).

These tests build a minimal FastAPI app that mounts only the uploads
router, point the route's :func:`get_media_store` accessor at a fresh
:class:`MediaStore` backed by an in-memory DB fake, and exercise the
endpoints with :class:`fastapi.testclient.TestClient`.

We keep the suite hermetic — no Postgres, no real filesystem outside
``tmp_path``, no MCP server. The DB fake is the same shape used by
``tests/test_media_store.py`` so the two suites can't drift.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from llm_bawt.media.store import MediaStore
from llm_bawt.service.routes import uploads as uploads_routes


# ---------------------------------------------------------------------------
# In-memory DB fake (mirrors tests/test_media_store.py — keep in sync)
# ---------------------------------------------------------------------------


class FakeMediaAssetStore:
    """Drop-in stand-in for :class:`MediaAssetStore`. See note in test_media_store.py."""

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
def media_store(tmp_path: Path) -> MediaStore:
    """A fresh MediaStore rooted at ``tmp_path`` with a fake in-memory DB."""
    return MediaStore(root=tmp_path, db=FakeMediaAssetStore())


@pytest.fixture
def client(media_store: MediaStore, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient against a minimal app with the uploads router mounted.

    We monkeypatch the route module's :func:`_store` accessor so every
    request resolves to the same per-test :class:`MediaStore`. Avoids
    leaking the singleton across tests and lets us swap roots cleanly.
    """

    def _stub_store() -> MediaStore:
        return media_store

    monkeypatch.setattr(uploads_routes, "_store", _stub_store)

    app = FastAPI()
    app.include_router(uploads_routes.router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _png_bytes(width: int = 640, height: int = 480, color=(200, 50, 100)) -> bytes:
    """Synthesize a small PNG with a checkerboard so WebP has work to do."""
    img = Image.new("RGB", (width, height), color)
    pixels = img.load()
    block = 16
    for y in range(0, height, block):
        for x in range(0, width, block):
            if ((x // block) + (y // block)) % 2 == 0:
                pixels[x, y] = (255, 255, 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _png_data_url(width: int = 640, height: int = 480) -> str:
    raw = _png_bytes(width, height)
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# POST /v1/uploads
# ---------------------------------------------------------------------------


def test_post_multipart_upload_happy_path(client: TestClient) -> None:
    """POST multipart returns 200 + the canonical upload-response shape."""
    raw = _png_bytes(800, 600)
    resp = client.post(
        "/v1/uploads",
        headers={"X-Entity-Id": "user-1"},
        files={"file": ("test.png", raw, "image/png")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Required fields.
    assert body["asset_id"].startswith("ma_")
    assert body["mime_type"] == "image/webp"
    assert body["original_mime_type"] == "image/png"
    assert body["kind"] == "image"
    assert body["sha256"] and len(body["sha256"]) == 64
    assert body["size_bytes"] > 0
    assert body["width"] and body["height"]

    # URLs reference the same asset_id under all three variants.
    asset_id = body["asset_id"]
    assert body["urls"]["original"] == f"/v1/uploads/{asset_id}"
    assert body["urls"]["thumb"] == f"/v1/uploads/{asset_id}/thumb"
    assert body["urls"]["preview"] == f"/v1/uploads/{asset_id}/preview"


def test_post_data_url_happy_path_and_dedups_against_multipart(client: TestClient) -> None:
    """JSON data-url upload of identical bytes hits the dedup path."""
    raw = _png_bytes(800, 600)
    # First, multipart.
    r1 = client.post(
        "/v1/uploads",
        headers={"X-Entity-Id": "user-1"},
        files={"file": ("test.png", raw, "image/png")},
    )
    assert r1.status_code == 200
    asset_id_1 = r1.json()["asset_id"]

    # Then JSON-with-data-url of the same source bytes — should dedup to
    # the SAME asset id since the normalized sha is identical.
    data_url = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    r2 = client.post(
        "/v1/uploads",
        headers={"X-Entity-Id": "user-1", "Content-Type": "application/json"},
        json={"data_url": data_url, "filename": "from-paste.png"},
    )
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["asset_id"] == asset_id_1, "dedup should yield the same asset_id"
    assert body["sha256"] == r1.json()["sha256"]


def test_post_non_pillow_image_mime_stored_as_file(client: TestClient) -> None:
    """TASK-847: ``image/svg+xml`` no longer 415s — it bypasses the WebP
    pipeline and is stored verbatim as a ``file`` (served as attachment)."""
    resp = client.post(
        "/v1/uploads",
        headers={"X-Entity-Id": "user-1"},
        files={"file": ("evil.svg", b"<svg></svg>", "image/svg+xml")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kind"] == "file"
    assert body["mime_type"] == "image/svg+xml"
    get = client.get(body["urls"]["original"])
    assert get.content == b"<svg></svg>"
    assert get.headers["content-disposition"].startswith("attachment;")


def test_post_rejects_oversized_raw_bytes(client: TestClient) -> None:
    """A payload over the 15MB raw cap returns 413."""
    # 16 MB of incompressible-ish bytes (we never get to Pillow — the
    # size check fires first, so a valid image isn't required).
    big = b"\x89PNG\r\n\x1a\n" + b"A" * (16 * 1024 * 1024)
    resp = client.post(
        "/v1/uploads",
        headers={"X-Entity-Id": "user-1"},
        files={"file": ("big.png", big, "image/png")},
    )
    assert resp.status_code == 413, resp.text


def test_post_missing_entity_id_returns_401(client: TestClient) -> None:
    """No ``X-Entity-Id`` header → 401."""
    raw = _png_bytes(64, 64)
    resp = client.post(
        "/v1/uploads",
        files={"file": ("x.png", raw, "image/png")},
    )
    assert resp.status_code == 401


def test_post_neither_file_nor_data_url_returns_400(client: TestClient) -> None:
    """Empty JSON body returns 400 (no file, no data_url)."""
    resp = client.post(
        "/v1/uploads",
        headers={"X-Entity-Id": "user-1", "Content-Type": "application/json"},
        json={},
    )
    assert resp.status_code == 400


def test_post_malformed_data_url_returns_400(client: TestClient) -> None:
    """A non-data-URL string in ``data_url`` returns 400."""
    resp = client.post(
        "/v1/uploads",
        headers={"X-Entity-Id": "user-1", "Content-Type": "application/json"},
        json={"data_url": "not-a-data-url"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /v1/uploads/{asset_id}[/variant]
# ---------------------------------------------------------------------------


def test_get_original_returns_webp_bytes_with_headers(client: TestClient) -> None:
    """The original variant is served with ETag + Cache-Control + image/webp."""
    raw = _png_bytes(1600, 900)
    post = client.post(
        "/v1/uploads",
        headers={"X-Entity-Id": "user-1"},
        files={"file": ("hd.png", raw, "image/png")},
    )
    asset_id = post.json()["asset_id"]
    sha = post.json()["sha256"]

    resp = client.get(f"/v1/uploads/{asset_id}")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/webp")
    assert resp.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert resp.headers["etag"] == f'"{sha}"'

    # Body actually decodes as WebP.
    img = Image.open(io.BytesIO(resp.content))
    assert img.format == "WEBP"


def test_get_thumb_is_smaller_than_original(client: TestClient) -> None:
    """The ``/thumb`` variant has shorter long-edge than the original."""
    raw = _png_bytes(1600, 900)
    post = client.post(
        "/v1/uploads",
        headers={"X-Entity-Id": "user-1"},
        files={"file": ("hd.png", raw, "image/png")},
    )
    asset_id = post.json()["asset_id"]

    orig = client.get(f"/v1/uploads/{asset_id}")
    thumb = client.get(f"/v1/uploads/{asset_id}/thumb")
    assert thumb.status_code == 200
    assert thumb.headers["content-type"].startswith("image/webp")

    orig_img = Image.open(io.BytesIO(orig.content))
    thumb_img = Image.open(io.BytesIO(thumb.content))
    assert max(thumb_img.size) <= 256
    assert max(thumb_img.size) < max(orig_img.size)


def test_get_preview_is_middle_sized(client: TestClient) -> None:
    """The ``/preview`` variant sits between thumb and original."""
    raw = _png_bytes(1600, 900)
    post = client.post(
        "/v1/uploads",
        headers={"X-Entity-Id": "user-1"},
        files={"file": ("hd.png", raw, "image/png")},
    )
    asset_id = post.json()["asset_id"]

    resp = client.get(f"/v1/uploads/{asset_id}/preview")
    assert resp.status_code == 200
    img = Image.open(io.BytesIO(resp.content))
    assert max(img.size) <= 1024
    assert max(img.size) > 256  # bigger than the thumb


def test_get_unknown_asset_returns_404(client: TestClient) -> None:
    """GET on a bogus id is 404, not 500."""
    resp = client.get("/v1/uploads/ma_does_not_exist")
    assert resp.status_code == 404


def test_get_etag_round_trip_returns_304(client: TestClient) -> None:
    """GET returns ETag; second GET with matching If-None-Match yields 304."""
    raw = _png_bytes(400, 300)
    post = client.post(
        "/v1/uploads",
        headers={"X-Entity-Id": "user-1"},
        files={"file": ("rt.png", raw, "image/png")},
    )
    asset_id = post.json()["asset_id"]
    sha = post.json()["sha256"]

    first = client.get(f"/v1/uploads/{asset_id}")
    assert first.status_code == 200
    etag = first.headers["etag"]
    assert etag == f'"{sha}"'

    # Conditional re-request with matching ETag → 304 + empty body.
    second = client.get(
        f"/v1/uploads/{asset_id}",
        headers={"If-None-Match": etag},
    )
    assert second.status_code == 304
    assert second.headers["etag"] == etag
    assert second.headers["cache-control"] == "public, max-age=31536000, immutable"
    # 304 responses MUST NOT include a body.
    assert second.content == b""

    # Non-matching ETag should still return 200 with bytes.
    miss = client.get(
        f"/v1/uploads/{asset_id}",
        headers={"If-None-Match": '"deadbeef"'},
    )
    assert miss.status_code == 200
    assert miss.content == first.content

    # Wildcard '*' in If-None-Match also yields 304 (per RFC 7232).
    star = client.get(
        f"/v1/uploads/{asset_id}",
        headers={"If-None-Match": "*"},
    )
    assert star.status_code == 304


# ---------------------------------------------------------------------------
# DELETE /v1/uploads/{asset_id}
# ---------------------------------------------------------------------------


def test_delete_happy_path_returns_204(client: TestClient) -> None:
    """Owner can delete their own asset; subsequent GET is 404."""
    raw = _png_bytes(400, 300)
    post = client.post(
        "/v1/uploads",
        headers={"X-Entity-Id": "user-1"},
        files={"file": ("x.png", raw, "image/png")},
    )
    asset_id = post.json()["asset_id"]

    resp = client.delete(
        f"/v1/uploads/{asset_id}",
        headers={"X-Entity-Id": "user-1"},
    )
    assert resp.status_code == 204

    # Now gone.
    follow = client.get(f"/v1/uploads/{asset_id}")
    assert follow.status_code == 404


def test_delete_wrong_owner_returns_403(client: TestClient) -> None:
    """A different X-Entity-Id can't delete someone else's asset."""
    raw = _png_bytes(400, 300)
    post = client.post(
        "/v1/uploads",
        headers={"X-Entity-Id": "user-1"},
        files={"file": ("x.png", raw, "image/png")},
    )
    asset_id = post.json()["asset_id"]

    resp = client.delete(
        f"/v1/uploads/{asset_id}",
        headers={"X-Entity-Id": "user-2"},
    )
    assert resp.status_code == 403

    # Asset still there.
    follow = client.get(f"/v1/uploads/{asset_id}")
    assert follow.status_code == 200


def test_delete_missing_asset_returns_404(client: TestClient) -> None:
    """DELETE on an unknown id is 404."""
    resp = client.delete(
        "/v1/uploads/ma_nope",
        headers={"X-Entity-Id": "user-1"},
    )
    assert resp.status_code == 404


def test_delete_requires_entity_id_header(client: TestClient) -> None:
    """No header → 401, even before we try to look up the asset."""
    resp = client.delete("/v1/uploads/ma_anything")
    assert resp.status_code == 401


def test_delete_owner_null_asset_allowed_for_any_entity(
    client: TestClient, media_store: MediaStore
) -> None:
    """Tool-generated assets (owner_user_id=None) are deletable by anyone.

    Per TASK-224 spec — agents that generated an image without a human
    owner need a way to GC their own outputs without knowing a synthetic
    owner id. Owner-set assets remain strictly owner-only (covered by
    :func:`test_delete_wrong_owner_returns_403`).
    """
    asset = media_store.upload(
        raw_bytes=_png_bytes(200, 150),
        original_mime="image/png",
        source="tool_generated",
        owner_user_id=None,
    )

    resp = client.delete(
        f"/v1/uploads/{asset.id}",
        headers={"X-Entity-Id": "some-random-agent"},
    )
    assert resp.status_code == 204

    follow = client.get(f"/v1/uploads/{asset.id}")
    assert follow.status_code == 404


# ---------------------------------------------------------------------------
# TASK-847: arbitrary file types + public links
# ---------------------------------------------------------------------------


def test_post_pdf_multipart_stored_as_file(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BAWT_PUBLIC_ORIGIN", "https://app.example.test")
    raw = b"%PDF-1.4 fake pdf body " * 50
    resp = client.post(
        "/v1/uploads?source=agent_attachment",
        headers={"X-Entity-Id": "nick"},
        files={"file": ("Quarterly Report.pdf", raw, "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kind"] == "file"
    assert body["mime_type"] == "application/pdf"
    assert body["filename"] == "Quarterly Report.pdf"
    assert body["size_bytes"] == len(raw)
    assert body["width"] is None and body["height"] is None
    assert body["urls"] == {
        "original": f"/v1/uploads/{body['asset_id']}",
        "download": f"/v1/uploads/{body['asset_id']}?download=1",
    }
    assert body["public_url"] == f"https://app.example.test/api/chat/uploads/{body['asset_id']}"
    assert body["public_download_url"] == body["public_url"] + "?download=1"

    get = client.get(f"/v1/uploads/{body['asset_id']}")
    assert get.status_code == 200
    assert get.content == raw
    assert get.headers["content-type"].startswith("application/pdf")
    assert get.headers["content-disposition"] == 'inline; filename="Quarterly Report.pdf"'
    assert get.headers["etag"] == f'"{body["sha256"]}"'

    dl = client.get(f"/v1/uploads/{body['asset_id']}?download=1")
    assert dl.status_code == 200
    assert dl.headers["content-disposition"].startswith('attachment; filename="Quarterly Report.pdf"')

    # No image renditions for a file.
    assert client.get(f"/v1/uploads/{body['asset_id']}/thumb").status_code == 404
    assert client.get(f"/v1/uploads/{body['asset_id']}/preview").status_code == 404


def test_post_text_file_json_data_url_with_filename(client: TestClient) -> None:
    raw = "hello, world\nline two\n".encode()
    data_url = "data:text/plain;base64," + base64.b64encode(raw).decode("ascii")
    resp = client.post(
        "/v1/uploads",
        headers={"X-Entity-Id": "nick"},
        json={"data_url": data_url, "filename": "../../etc/notes\x00.txt"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kind"] == "file"
    assert body["filename"] == "notes.txt"  # basename + control chars stripped
    get = client.get(body["urls"]["original"])
    assert get.content == raw
    assert get.headers["content-type"].startswith("text/plain")


def test_post_octet_stream_guesses_mime_from_extension(client: TestClient) -> None:
    """``curl -F file=@x.csv`` sends octet-stream — the extension should win."""
    resp = client.post(
        "/v1/uploads",
        headers={"X-Entity-Id": "nick"},
        files={"file": ("data.csv", b"a,b\n1,2\n", "application/octet-stream")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["mime_type"] == "text/csv"


def test_post_octet_stream_png_extension_takes_image_pipeline(client: TestClient) -> None:
    resp = client.post(
        "/v1/uploads",
        headers={"X-Entity-Id": "nick"},
        files={"file": ("shot.png", _png_bytes(320, 200), "application/octet-stream")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kind"] == "image"
    assert body["mime_type"] == "image/webp"
    assert set(body["urls"]) == {"thumb", "preview", "original"}


def test_unknown_binary_forced_to_attachment(client: TestClient) -> None:
    resp = client.post(
        "/v1/uploads",
        headers={"X-Entity-Id": "nick"},
        files={"file": ("mystery", b"\x00\x01\x02\x03" * 10, "application/octet-stream")},
    )
    assert resp.status_code == 200, resp.text
    aid = resp.json()["asset_id"]
    get = client.get(f"/v1/uploads/{aid}")
    assert get.headers["content-disposition"].startswith("attachment;")


def test_html_and_svg_never_served_inline(client: TestClient) -> None:
    for name, mime, payload in (
        ("page.html", "text/html", b"<script>alert(1)</script>"),
        ("logo.svg", "image/svg+xml", b"<svg xmlns='http://www.w3.org/2000/svg'/>"),
    ):
        resp = client.post(
            "/v1/uploads", headers={"X-Entity-Id": "nick"}, files={"file": (name, payload, mime)}
        )
        assert resp.status_code == 200, resp.text
        get = client.get(resp.json()["urls"]["original"])
        assert get.headers["content-disposition"].startswith("attachment;"), name


def test_file_size_cap_is_separate_from_image_cap(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BAWT_MAX_FILE_UPLOAD_BYTES", "1024")
    ok = client.post(
        "/v1/uploads", headers={"X-Entity-Id": "nick"}, files={"file": ("s.txt", b"x" * 1024, "text/plain")}
    )
    assert ok.status_code == 200
    too_big = client.post(
        "/v1/uploads", headers={"X-Entity-Id": "nick"}, files={"file": ("b.txt", b"x" * 1025, "text/plain")}
    )
    assert too_big.status_code == 413
    assert "kind=file" in too_big.json()["detail"]
    # Images are still governed by the 15 MB cap, not the file cap.
    img = client.post(
        "/v1/uploads", headers={"X-Entity-Id": "nick"}, files={"file": ("p.png", _png_bytes(200, 200), "image/png")}
    )
    assert img.status_code == 200


def test_image_response_has_content_disposition_and_public_url(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_BAWT_PUBLIC_ORIGIN", "https://app.example.test/")
    resp = client.post(
        "/v1/uploads", headers={"X-Entity-Id": "nick"}, files={"file": ("cat photo.jpg", _png_bytes(100, 100), "image/png")}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kind"] == "image"
    assert body["filename"] == "cat photo.jpg"
    assert body["public_url"] == f"https://app.example.test/api/chat/uploads/{body['asset_id']}"
    assert body["public_download_url"] is None
    get = client.get(body["urls"]["thumb"])
    assert get.headers["content-disposition"] == 'inline; filename="cat photo-thumb.webp"'
    orig = client.get(body["urls"]["original"])
    assert orig.headers["content-disposition"] == 'inline; filename="cat photo.webp"'


def test_public_url_omitted_when_origin_blank(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BAWT_PUBLIC_ORIGIN", "")
    resp = client.post(
        "/v1/uploads", headers={"X-Entity-Id": "nick"}, files={"file": ("n.txt", b"n", "text/plain")}
    )
    assert resp.status_code == 200
    assert resp.json()["public_url"] is None


def test_unicode_filename_gets_rfc5987_fallback(client: TestClient) -> None:
    resp = client.post(
        "/v1/uploads", headers={"X-Entity-Id": "nick"}, files={"file": ("résumé.txt", b"cv", "text/plain")}
    )
    assert resp.status_code == 200, resp.text
    get = client.get(resp.json()["urls"]["original"])
    cd = get.headers["content-disposition"]
    assert cd.startswith('inline; filename="rsum.txt"')
    assert "filename*=UTF-8''r%C3%A9sum%C3%A9.txt" in cd


def test_delete_file_asset(client: TestClient) -> None:
    resp = client.post(
        "/v1/uploads", headers={"X-Entity-Id": "nick"}, files={"file": ("d.zip", b"PK\x03\x04junk", "application/zip")}
    )
    aid = resp.json()["asset_id"]
    assert client.delete(f"/v1/uploads/{aid}", headers={"X-Entity-Id": "nick"}).status_code == 204
    assert client.get(f"/v1/uploads/{aid}").status_code == 404


def test_post_unexpected_error_does_not_echo_its_text(
    client: TestClient, media_store: MediaStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DB error names its host and user; the 500 body must not (TASK-1561)."""
    leak = 'connection to server at "postgres" (192.0.2.7) failed for user "tenant_user"'

    def _fail(**_kwargs: Any) -> None:
        raise RuntimeError(leak)

    monkeypatch.setattr(media_store, "upload", _fail)
    resp = client.post(
        "/v1/uploads", headers={"X-Entity-Id": "user-1"}, files={"file": ("a.txt", b"hi", "text/plain")}
    )

    assert resp.status_code == 500
    assert resp.json() == {"detail": "upload failed (RuntimeError); see the app log"}
