"""TASK-2610: the remote MCP gateway for agent-bridge bots."""
from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager

import httpx
import mcp.types as types
import pytest
from cryptography.fernet import Fernet

from llm_bawt import task_turn_context as codec
from llm_bawt.integrations.mcp_servers import gateway as gw
from llm_bawt.integrations.mcp_servers.service import ResolvedServer

SECRET = "sample-access-token"


class FakeServers:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def resolve_one(self, bot, slug):
        self.calls.append((bot, slug))
        if self.fail:
            raise RuntimeError("db down")
        if (bot, slug) != ("reader", "alpha"):
            return None
        return ResolvedServer("alpha", "Alpha", "https://mcp.example.com/mcp", "streamable_http",
                              {"Authorization": f"Bearer {SECRET}"})


class FakeSession:
    fail_list = False

    async def list_tools(self, params=None):
        if self.fail_list:
            raise RuntimeError(f"GET https://mcp.example.com/mcp?key={SECRET} failed")
        if params is None:  # two pages: the gateway must follow the cursor
            return types.ListToolsResult(tools=[types.Tool(name="lookup", inputSchema={"type": "object"})],
                                         next_cursor="page-2")
        assert params.cursor == "page-2"
        return types.ListToolsResult(tools=[types.Tool(name="second", inputSchema={"type": "object"})])

    async def call_tool(self, name, arguments):
        if name == "boom":
            raise RuntimeError(f"transport failed with {SECRET}")
        return types.CallToolResult(content=[types.TextContent(type="text", text=f"{name}:{arguments}")])


@pytest.fixture
def signer(monkeypatch):
    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(codec, "_get_fernet", lambda: fernet)


@pytest.fixture
def fake_upstream(monkeypatch):
    opened = []

    @asynccontextmanager
    async def open_session(url, transport, headers):
        opened.append((url, transport, headers))
        yield FakeSession()

    monkeypatch.setattr(gw, "open_session", open_session)
    return opened


def capability(bot="reader"):
    return codec.mint_task_turn_context(
        session_id=str(uuid.uuid4()), turn_id="turn-" + uuid.uuid4().hex,
        trigger_message_id=str(uuid.uuid4()), bot_id=bot, user_id="owner")


async def inner(scope, receive, send):
    if scope["type"] == "lifespan":
        return
    await send({"type": "http.response.start", "status": 299, "headers": []})
    await send({"type": "http.response.body", "body": b"inner"})


class Exposure:
    def __init__(self, fail=False):
        self.fail, self.turns = fail, []

    def __call__(self, turn):
        if self.fail:
            raise RuntimeError("ledger down")
        self.turns.append(turn.bot_id)


def app(servers, expose=None):
    return gw.WithRemoteMcp(inner, gw.RemoteMcpGateway(lambda: servers, expose=expose or Exposure()))


async def rpc(client, path, method, params=None, headers=None):
    return await client.post(path, headers={
        "accept": "application/json, text/event-stream", "content-type": "application/json",
        **(headers or {})}, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})


@asynccontextmanager
async def client_for(application):
    async with application.gateway.manager.run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application),
                                     base_url="http://testserver") as client:
            yield client


@pytest.mark.asyncio
async def test_other_paths_go_to_the_inner_app(signer):
    async with client_for(app(FakeServers())) as client:
        response = await client.get("/mcp")
    assert response.status_code == 299 and response.text == "inner"


@pytest.mark.asyncio
async def test_refuses_bad_paths_missing_or_invalid_capability(signer):
    servers = FakeServers()
    async with client_for(app(servers)) as client:
        assert (await rpc(client, "/remote-mcp/Bad Slug/mcp", "tools/list")).status_code == 404
        assert (await rpc(client, "/remote-mcp/alpha/other", "tools/list")).status_code == 404
        missing = await rpc(client, "/remote-mcp/alpha/mcp", "tools/list")
        assert missing.status_code == 401 and missing.json() == {"error": "turn_capability_required"}
        forged = await rpc(client, "/remote-mcp/alpha/mcp", "tools/list",
                           headers={codec.TASK_TURN_CONTEXT_HEADER: "forged"})
        assert forged.status_code == 401
    assert servers.calls == []  # nothing resolved without a proven bot


@pytest.mark.asyncio
async def test_ungranted_and_unavailable_storage(signer, fake_upstream):
    async with client_for(app(FakeServers())) as client:
        other = await rpc(client, "/remote-mcp/alpha/mcp", "tools/list",
                          headers={codec.TASK_TURN_CONTEXT_HEADER: capability("writer")})
        assert other.status_code == 404 and other.json() == {"error": "not_available"}
    async with client_for(app(FakeServers(fail=True))) as client:
        down = await rpc(client, "/remote-mcp/alpha/mcp", "tools/list",
                         headers={codec.TASK_TURN_CONTEXT_HEADER: capability()})
        assert down.status_code == 503
    assert fake_upstream == []


@pytest.mark.asyncio
async def test_proxies_list_and_call_with_the_server_credential(signer, fake_upstream):
    headers = {codec.TASK_TURN_CONTEXT_HEADER: capability()}
    exposure = Exposure()
    async with client_for(app(FakeServers(), exposure)) as client:
        listed = await rpc(client, "/remote-mcp/alpha/mcp", "tools/list", headers=headers)
        called = await rpc(client, "/remote-mcp/alpha/mcp", "tools/call",
                           {"name": "lookup", "arguments": {"q": "x"}}, headers=headers)
        failed = await rpc(client, "/remote-mcp/alpha/mcp", "tools/call",
                           {"name": "boom", "arguments": {}}, headers=headers)
    assert listed.status_code == 200, listed.text
    assert [t["name"] for t in listed.json()["result"]["tools"]] == ["lookup", "second"]
    assert "nextCursor" not in listed.json()["result"]
    labelled = called.json()["result"]
    assert json.loads(labelled["content"][0]["text"]) == {
        "untrusted_content": True, "source": "mcp:alpha", "data": "lookup:{'q': 'x'}"}
    assert labelled["_meta"]["content_provenance"]["source"] == "mcp:alpha"
    assert exposure.turns == ["reader"] * 3  # list and both calls, before upstream data
    result = failed.json()["result"]
    assert result["isError"] is True and SECRET not in failed.text
    assert all(h == {"Authorization": f"Bearer {SECRET}"} for _, _, h in fake_upstream)
    assert SECRET not in listed.text + called.text


@pytest.mark.asyncio
async def test_nothing_upstream_without_recorded_exposure(signer, fake_upstream):
    headers = {codec.TASK_TURN_CONTEXT_HEADER: capability()}
    async with client_for(app(FakeServers(), Exposure(fail=True))) as client:
        listed = await rpc(client, "/remote-mcp/alpha/mcp", "tools/list", headers=headers)
        called = await rpc(client, "/remote-mcp/alpha/mcp", "tools/call",
                           {"name": "lookup", "arguments": {}}, headers=headers)
    assert "error" in listed.json() and "result" not in listed.json()
    assert called.json()["result"]["isError"] is True
    assert fake_upstream == []


@pytest.mark.asyncio
async def test_list_failure_is_a_fixed_redacted_error(signer, fake_upstream, monkeypatch, caplog):
    monkeypatch.setattr(FakeSession, "fail_list", True)
    headers = {codec.TASK_TURN_CONTEXT_HEADER: capability()}
    with caplog.at_level(logging.DEBUG):
        async with client_for(app(FakeServers())) as client:
            listed = await rpc(client, "/remote-mcp/alpha/mcp", "tools/list", headers=headers)
    assert listed.json()["error"]["message"] == gw.LIST_FAILED
    assert SECRET not in listed.text and "mcp.example.com" not in listed.text
    assert SECRET not in caplog.text and "mcp.example.com" not in caplog.text


@pytest.mark.asyncio
async def test_lifespan_runs_the_gateway_session_manager():
    application = app(FakeServers())
    messages = iter([{"type": "lifespan.startup"}])
    ran = []

    async def inner_lifespan(scope, receive, send):
        ran.append(application.gateway.manager._task_group is not None)

    application.inner = inner_lifespan
    await application({"type": "lifespan"}, lambda: next(messages), None)
    assert ran == [True]
