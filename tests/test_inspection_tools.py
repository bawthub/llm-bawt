import asyncio
import json
import time
from dataclasses import replace

import httpx
import pytest
from mcp.server.mcpserver import Image

from llm_bawt import inspection_attestation
from llm_bawt.mcp_server import inspection_tools
from llm_bawt.mcp_server.catalog import CatalogFastMCP
from llm_bawt.task_turn_context import TaskTurnContext, TaskTurnContextError

ORIGIN = "https://tenant.example"
TURN = TaskTurnContext(
    session_id="f5d5175f-d118-4f42-8aef-db9c681ee036", turn_id="turn-" + "1" * 32,
    trigger_message_id="c4e9028f-8fb8-41b6-8fd6-d84c6b685159", bot_id="worker-1", user_id="owner-1",
    issued_at=1_700_000_000, inspection=("account-1", ORIGIN, 1_700_000_000),
)
PNG = b"\x89PNG\r\n\x1a\n"


class Body(httpx.AsyncByteStream):
    def __init__(self, data: bytes):
        self.data = data

    async def __aiter__(self):
        yield self.data


@pytest.fixture
def requests(monkeypatch):
    monkeypatch.setenv("BAWTHUB_INSPECTION_INTERNAL_URL", "http://frontend:3002")
    monkeypatch.setenv("BAWTHUB_INSPECTION_ORIGIN", ORIGIN)
    monkeypatch.setattr(inspection_attestation, "_key", b"\x2a" * 32)
    def current_turn():
        now = int(time.time())
        return replace(TURN, issued_at=now, inspection=("account-1", ORIGIN, now))

    monkeypatch.setattr(inspection_tools, "current_task_turn_context", current_turn)
    monkeypatch.setattr(inspection_tools, "service_auth", lambda: httpx.Auth())
    original_client = httpx.AsyncClient
    seen = []

    def register(status, content_type, data, extra_headers=None):
        async def handle(request):
            seen.append(request)
            return httpx.Response(status, headers={"Content-Type": content_type, **(extra_headers or {})}, stream=Body(data))

        def client(**kwargs):
            assert kwargs["follow_redirects"] is False and kwargs["trust_env"] is False
            return original_client(transport=httpx.MockTransport(handle), **kwargs)

        monkeypatch.setattr(inspection_tools.httpx, "AsyncClient", client)
        return seen

    return register


def test_snapshot_only_posts_verified_identity_to_fixed_internal_path(requests):
    seen = requests(200, "image/png", PNG)
    assert isinstance(asyncio.run(inspection_tools.inspect_bawthub_tasks("phone")), Image)
    [request] = seen
    assert str(request.url) == "http://frontend:3002/internal/inspection/tasks"
    assert request.method == "POST"
    assert "cookie" not in request.headers
    assert len(request.headers[inspection_attestation.INSPECTION_MAC_HEADER]) == 64
    payload = json.loads(request.content)
    assert (payload["accountId"], payload["botId"], payload["userId"], payload["viewport"]) == (
        "account-1", "worker-1", "owner-1", "phone",
    )


def test_without_verified_owner_chat_provenance_does_not_call_frontend(monkeypatch, requests):
    seen = requests(200, "image/png", PNG)

    def no_turn():
        raise TaskTurnContextError("untrusted")

    monkeypatch.setattr(inspection_tools, "current_task_turn_context", no_turn)
    assert asyncio.run(inspection_tools.inspect_bawthub_tasks()) == {"status": "authorization_required"}
    monkeypatch.setattr(inspection_tools, "current_task_turn_context", lambda: replace(TURN, inspection=None))
    assert asyncio.run(inspection_tools.inspect_bawthub_tasks()) == {"status": "authorization_required"}
    monkeypatch.setattr(inspection_tools, "current_task_turn_context", lambda: TURN)
    monkeypatch.setenv("BAWTHUB_INSPECTION_ORIGIN", "https://another.example")
    assert asyncio.run(inspection_tools.inspect_bawthub_tasks()) == {"status": "wrong_origin"}
    assert seen == []


@pytest.mark.parametrize("status,content_type,data,headers,expected", [
    (200, "image/png", PNG + b"x" * (2 * 1024 * 1024), {}, "unavailable"),
    (200, "image/png", b"not a PNG", {}, "unavailable"),
    (200, "image/png", PNG, {"Content-Encoding": "gzip"}, "unavailable"),
    (403, "application/json", b'{"status":"expired","details":"not returned"}', {}, "expired"),
    (403, "application/json", b'{"status":"unrecognized internal error"}', {}, "unavailable"),
    (307, "application/json", b"ignored", {"Location": "http://other.example"}, "unavailable"),
])
def test_response_is_bounded_and_sanitized(requests, status, content_type, data, headers, expected):
    seen = requests(status, content_type, data, headers)
    assert asyncio.run(inspection_tools.inspect_bawthub_tasks()) == {"status": expected}
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_inspection_catalog_is_opt_in_and_returns_mcp_image(monkeypatch, requests):
    from mcp import Client
    from mcp.types import ImageContent, TextContent

    server = CatalogFastMCP("fixture")
    monkeypatch.delenv("BAWTHUB_INSPECTION_KEY_EXPORT_FILE", raising=False)
    inspection_tools.register_inspection_tool(server)
    assert await server.list_tools() == []

    monkeypatch.setenv("BAWTHUB_INSPECTION_KEY_EXPORT_FILE", "/run/fixture/key")
    inspection_tools.register_inspection_tool(server)
    requests(200, "image/png", PNG)
    async with Client(server, mode="legacy") as session:
        [tool] = (await session.list_tools()).tools
        assert tool.name == "inspect_bawthub_tasks"
        assert "Account → Bot inspection" in tool.description
        assert set(tool.input_schema["properties"]) == {"viewport"}
        assert tool.input_schema["properties"]["viewport"]["enum"] == ["phone", "desktop"]
        from scripts.mcp_catalog_report import measure
        budget = measure([tool.model_dump(by_alias=True, exclude_none=True)], "o200k_base")
        assert budget["descriptions_tokens"] <= 45
        assert budget["full_definitions_tokens"] <= 94
        result = await session.call_tool(tool.name, {"viewport": "phone"})
        assert not result.is_error
        [image] = result.content
        assert isinstance(image, ImageContent) and image.mime_type == "image/png"
        requests(403, "application/json", b'{"status":"revoked"}')
        denied = await session.call_tool(tool.name, {})
        assert not denied.is_error
        [status] = denied.content
        assert isinstance(status, TextContent) and json.loads(status.text) == {"status": "revoked"}
