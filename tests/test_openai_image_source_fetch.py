"""Reference images are fetched without the OpenAI API key (TASK-2263)."""

from __future__ import annotations

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llm_bawt.media.clients import openai_images
from llm_bawt.media.clients.openai_images import OpenAIImageClient

PNG = b"\x89PNG\r\n\x1a\nfake"


@pytest.fixture
def reference_host():
    seen: list[dict[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            seen.append({k.lower(): v for k, v in self.headers.items()})
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/ref.png")
                self.end_headers()
                return
            body = PNG * (10 if self.path == "/big.png" else 1)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()


def _resolve(url: str):
    async def run():
        client = OpenAIImageClient(api_key="sk-SYNTHETIC-KEY")
        try:
            return await client._resolve_source_image(url)
        finally:
            await client.close()

    return asyncio.run(run())


def test_reference_fetch_sends_no_provider_authorization(reference_host):
    base, seen = reference_host

    image = _resolve(f"{base}/ref.png")

    assert image.data == PNG and image.mime_type == "image/png"
    assert len(seen) == 1
    assert "authorization" not in seen[0]
    assert not any("SYNTHETIC" in v for v in seen[0].values())


def test_reference_fetch_does_not_follow_redirects(reference_host):
    base, seen = reference_host

    with pytest.raises(Exception):
        _resolve(f"{base}/redirect")

    assert len(seen) == 1


def test_reference_fetch_enforces_size_cap(reference_host, monkeypatch):
    base, _seen = reference_host
    monkeypatch.setattr(openai_images, "SOURCE_IMAGE_MAX_BYTES", len(PNG) * 2)

    with pytest.raises(ValueError, match="limit"):
        _resolve(f"{base}/big.png")


@pytest.mark.parametrize("url", ["/v1/files/x", "file:///etc/passwd", "ftp://host/x.png"])
def test_reference_fetch_rejects_non_http_urls(url):
    with pytest.raises(ValueError, match="http"):
        _resolve(url)
