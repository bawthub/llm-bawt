"""attachment_get resolves upload references to image blocks (TASK-471)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from mcp.server.fastmcp import Image

from llm_bawt.mcp_server import attachment_tools
from llm_bawt.mcp_server.task_images import ImageRefError

PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 32
SERVE = "/api/uploads/serve/originals/10/92/" + "a" * 64 + ".png"
ASSET = "ma_" + "0" * 26


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def fetched(monkeypatch):
    calls: list[str] = []

    async def fetch(path: str) -> bytes:
        calls.append(path)
        if path.endswith("missing.png"):
            raise ImageRefError(f"no stored upload at {path}")
        return PNG

    monkeypatch.setattr(attachment_tools, "_fetch_task_upload", fetch)
    monkeypatch.setattr(attachment_tools, "_read_chat_upload", lambda asset_id: PNG)
    return calls


def test_relative_serve_path_returns_image(fetched):
    out = _run(attachment_tools.attachment_get(SERVE))
    assert isinstance(out[0], Image)
    assert out[-1]["images"] == 1
    assert fetched == [SERVE]


def test_public_host_url_is_fetched_by_path_only(fetched):
    _run(attachment_tools.attachment_get("https://app.bawthub.com" + SERVE))
    assert fetched == [SERVE]


def test_foreign_host_refused(fetched):
    out = _run(attachment_tools.attachment_get("https://evil.example" + SERVE))
    assert out[-1]["images"] == 0 and "error" in out[-1]["items"][0]
    assert fetched == []


def test_chat_upload_id_reads_media_store(fetched):
    out = _run(attachment_tools.attachment_get(ASSET))
    assert isinstance(out[0], Image) and fetched == []


def test_task_ref_loads_card_images_with_limit_and_per_item_errors(monkeypatch, fetched):
    urls = [SERVE, SERVE.replace("a" * 64 + ".png", "missing.png"), f"/api/chat/uploads/{ASSET}"]

    async def api_get(path: str, **_: Any):
        assert path == "/tasks/TASK-7"
        return {"imageUrls": urls}

    monkeypatch.setattr(attachment_tools, "api_get", api_get)
    out = _run(attachment_tools.attachment_get("task-7", max_images=2))
    info = out[-1]
    assert info["images"] == 1 and info["omitted"] == 1
    assert "error" in info["items"][1]


def test_chat_storage_outage_is_an_item_error(monkeypatch):
    """A chat-media outage fails only that item; the others still load (review cmw7o8ovv)."""
    from llm_bawt.media.object_store import BlobBackendUnavailable

    class Store:
        def read_variant(self, asset_id, variant):
            raise BlobBackendUnavailable("down")

    async def fetch(path: str) -> bytes:
        return PNG

    async def api_get(path: str, **_: Any):
        return {"imageUrls": [SERVE, f"/api/chat/uploads/{ASSET}", SERVE]}

    monkeypatch.setattr("llm_bawt.media.get_media_store", lambda: Store())
    monkeypatch.setattr(attachment_tools, "_fetch_task_upload", fetch)
    monkeypatch.setattr(attachment_tools, "api_get", api_get)
    out = _run(attachment_tools.attachment_get("TASK-9"))
    info = out[-1]
    assert info["images"] == 2
    assert "unavailable" in info["items"][1]["error"]


def test_task_without_images(monkeypatch, fetched):
    async def api_get(path: str, **_: Any):
        return {"imageUrls": []}

    monkeypatch.setattr(attachment_tools, "api_get", api_get)
    out = _run(attachment_tools.attachment_get("TASK-8"))
    assert out == [{"ref": "TASK-8", "images": 0, "note": "this task has no images"}]


def test_fetch_uses_internal_base_and_rejects_non_images(monkeypatch):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path.endswith("text.png"):
            return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html>")
        return httpx.Response(200, headers={"content-type": "image/png"}, content=PNG)

    real = httpx.AsyncClient
    monkeypatch.setattr(attachment_tools.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setenv("BAWTHUB_UPLOADS_INTERNAL_URL", "http://internal:1/")
    assert _run(attachment_tools._fetch_task_upload(SERVE)) == PNG
    assert seen == ["http://internal:1" + SERVE]
    with pytest.raises(ImageRefError):
        _run(attachment_tools._fetch_task_upload("/api/uploads/serve/text.png"))
