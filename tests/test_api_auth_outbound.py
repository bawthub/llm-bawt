import urllib.request

import httpx
import pytest

from llm_bawt.api_auth import client, outbound

KEY = "house.abc123"


@pytest.fixture
def keyed(tmp_path, monkeypatch):
    path = tmp_path / "api-key"
    path.write_text(KEY + "\n")
    monkeypatch.setenv(client.FILE_ENV, str(path))
    monkeypatch.setenv(outbound.HOSTS_ENV, "10.9.9.9:8642")
    outbound.install()
    outbound.install()  # idempotent


@pytest.mark.parametrize("url,expected", [
    ("http://app:8642/v1/bots", True),
    ("http://localhost:8642/x", True),
    ("http://[::1]:8642/x", True),
    ("http://10.9.9.9:8642/x", True),
    ("http://app:8001/mcp", False),
    ("https://api.openai.com/v1", False),
    ("http://app.evil:8642/x", False),
])
def test_targets_only_app_hosts(keyed, url, expected):
    assert outbound.targets_app(url) is expected


def _seen(url, headers=None):
    captured = {}

    def handler(request):
        captured.update(request.headers)
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        http.get(url, headers=headers)
    return captured


def test_httpx_adds_key_only_for_app(keyed):
    assert _seen("http://app:8642/v1/bots")["x-llm-bawt-key"] == KEY
    assert "x-llm-bawt-key" not in _seen("https://api.openai.com/v1/models")
    assert _seen("http://app:8642/v1", {"X-LLM-Bawt-Key": "mine"})["x-llm-bawt-key"] == "mine"


@pytest.mark.asyncio
async def test_async_httpx_adds_key(keyed):
    captured = {}

    def handler(request):
        captured.update(request.headers)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await http.get("http://app:8642/v1/bots")
    assert captured["x-llm-bawt-key"] == KEY


def test_urllib_handler_adds_key_only_for_app(keyed):
    handler = outbound._UrllibHandler()
    req = handler.http_request(urllib.request.Request("http://app:8642/v1/x"))
    assert req.get_header("X-llm-bawt-key") == KEY
    other = handler.http_request(urllib.request.Request("https://example.com/"))
    assert other.get_header("X-llm-bawt-key") is None


def test_no_key_file_adds_nothing(keyed, monkeypatch):
    monkeypatch.delenv(client.FILE_ENV)
    assert "x-llm-bawt-key" not in _seen("http://app:8642/v1/bots")
