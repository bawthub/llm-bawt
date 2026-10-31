"""Installed-CLI MCP outage/reconnect proof with disposable HTTP + SSE services.

No operational app, provider credentials, bot transcript or paid inference is used.
RUN_MCP_RECONNECT_SDK_PROBE=1 is set by ``make test-sdk-probe``.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import uuid
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from mcp.server.fastmcp import FastMCP

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ResultMessage
from claude_agent_sdk.types import ToolResultBlock, UserMessage
from claude_code_bridge.active_run import ClaudeActiveRun

pytestmark = [pytest.mark.integration, pytest.mark.sdk_probe, pytest.mark.skipif(
    os.environ.get("RUN_MCP_RECONNECT_SDK_PROBE") != "1",
    reason="opt-in disposable SDK MCP reconnect probe",
)]

MODEL = "claude-sonnet-4-6"


def _wire(*, tool: bool):
    events = [("message_start", {"type": "message_start", "message": {
        "id": "msg_" + uuid.uuid4().hex[:12], "type": "message", "role": "assistant",
        "model": MODEL, "content": [], "stop_reason": None,
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }})]
    if tool:
        block = {"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:12],
                 "name": "mcp__probe__ping", "input": {}}
        initial = {**block, "input": {}}
        delta = {"type": "input_json_delta", "partial_json": "{}"}
    else:
        initial = {"type": "text", "text": ""}
        delta = {"type": "text_delta", "text": "probe complete"}
    events += [
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                  "content_block": initial}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                  "delta": delta}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {
            "stop_reason": "tool_use" if tool else "end_turn"},
            "usage": {"output_tokens": 5}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return StreamingResponse(iter([
        "".join(f"event: {name}\ndata: {json.dumps(body)}\n\n" for name, body in events),
    ]), media_type="text/event-stream")


class _McpService:
    def __init__(self, transport: str):
        self.transport = transport
        self.port: int | None = None
        self.server = None
        self.running = None
        self.sock = None
        self.calls = 0

    async def start(self):
        fixture = FastMCP("probe", json_response=True, stateless_http=True)

        @fixture.tool(name="ping")
        def ping() -> str:
            self.calls += 1
            return "PONG_FROM_RESTARTED_SERVER"

        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", self.port or 0))
        self.port = sock.getsockname()[1]
        self.sock = sock
        app = fixture.streamable_http_app() if self.transport == "http" else fixture.sse_app()
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        server.install_signal_handlers = lambda: None
        self.server = server
        self.running = asyncio.create_task(server.serve(sockets=[sock]))
        while not server.started:
            if self.running.done():
                self.running.result()
            await asyncio.sleep(0.01)

    async def stop(self):
        if self.server is None:
            return
        # Simulate a process loss, including its open SSE/HTTP connections.
        for connection in tuple(self.server.server_state.connections):
            connection.transport.close()
        self.server.should_exit = True
        await asyncio.wait_for(self.running, timeout=5)
        self.sock.close()
        self.server = self.running = self.sock = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/{'mcp' if self.transport == 'http' else 'sse'}"


def _status_of(reply):
    return next((server for server in reply.get("mcpServers", [])
                 if server.get("name") == "probe"), {})


@pytest.mark.parametrize("transport", ["sse", "http"])
def test_cli_restores_mcp_catalog_mid_turn_after_local_server_restarts(tmp_path: Path, transport):
    async def run():
        provider = FastAPI()
        first_request = asyncio.Event()
        offline = asyncio.Event()
        second_request = asyncio.Event()
        allow_second = asyncio.Event()
        invocations = []

        @provider.post("/v1/messages")
        async def respond(request: Request):
            body = await request.json()
            if not body.get("stream"):
                return {"id": "msg_validate", "type": "message", "role": "assistant",
                        "model": MODEL, "content": [{"type": "text", "text": "ok"}],
                        "stop_reason": "end_turn", "usage": {"input_tokens": 1,
                                                       "output_tokens": 1}}
            invocations.append(body)
            if len(invocations) == 1:
                first_request.set()
                await offline.wait()
                return _wire(tool=True)
            if len(invocations) == 2:
                second_request.set()
                await allow_second.wait()
                return _wire(tool=True)
            if len(invocations) == 3:
                return _wire(tool=False)
            raise AssertionError("unexpected model call; tool was replayed")

        service = _McpService(transport)
        await service.start()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        provider_server = uvicorn.Server(uvicorn.Config(provider, log_level="error", lifespan="off"))
        provider_server.install_signal_handlers = lambda: None
        serving = asyncio.create_task(provider_server.serve(sockets=[sock]))
        while not provider_server.started:
            if serving.done():
                serving.result()
            await asyncio.sleep(0.01)
        home, work = tmp_path / "home", tmp_path / "work"
        (home / ".claude").mkdir(parents=True)
        work.mkdir()
        client = ClaudeSDKClient(options=ClaudeAgentOptions(
            model=MODEL, cwd=str(work), env={
                "HOME": str(home), "CLAUDE_CONFIG_DIR": str(home / ".claude"),
                "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{sock.getsockname()[1]}",
                "ANTHROPIC_AUTH_TOKEN": "local-probe", "ANTHROPIC_API_KEY": "",
                "CLAUDE_CODE_OAUTH_TOKEN": "",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_ENABLE_TELEMETRY": "0", "MCP_TOOL_TIMEOUT": "3000",
            },
            mcp_servers={"probe": {"type": transport, "url": service.url}},
            strict_mcp_config=True, setting_sources=[], permission_mode="bypassPermissions",
            system_prompt="Call the local ping tool twice, then finish.", max_turns=4,
        ))
        statuses = []
        original_status = client.get_mcp_status

        async def record_status():
            reply = await original_status()
            statuses.append(_status_of(reply).get("status"))
            return reply

        client.get_mcp_status = record_status
        reconnects = []
        reconnect_outcomes = []
        first_reconnect_finished = asyncio.Event()
        original_reconnect = client.reconnect_mcp_server

        async def record_reconnect(name):
            reconnects.append(name)
            try:
                result = await original_reconnect(name)
            except Exception as exc:
                reconnect_outcomes.append(type(exc).__name__)
                raise
            else:
                reconnect_outcomes.append("success")
                return result
            finally:
                first_reconnect_finished.set()

        client.reconnect_mcp_server = record_reconnect
        live = ClaudeActiveRun(client=client, request_id="probe")
        first_failed = asyncio.Event()
        results = []

        async def consume():
            async for message in live.messages():
                if isinstance(message, UserMessage):
                    for block in message.content:
                        if isinstance(block, ToolResultBlock):
                            results.append(block)
                            if block.is_error:
                                first_failed.set()
                if isinstance(message, ResultMessage):
                    return message
            raise AssertionError("SDK run ended without a result")

        try:
            await client.connect()
            await client.query("Run the fixture ping twice.")
            consuming = asyncio.create_task(consume())
            await first_request.wait()
            for _ in range(20):
                initial = _status_of(await client.get_mcp_status())
                if initial.get("status") == "connected":
                    break
                await asyncio.sleep(.25)
            assert initial.get("status") == "connected", initial
            await service.stop()
            offline.set()
            await first_failed.wait()
            await second_request.wait()
            # Keep the service offline until the monitor itself attempts a
            # reconnect: HTTP can become failed; SSE can remain pending without
            # tools. No test-side reconnect is allowed in this proof.
            await first_reconnect_finished.wait()
            missing = _status_of(await client.get_mcp_status())
            assert missing.get("status") in {"failed", "pending"}, missing
            assert not missing.get("tools"), missing
            assert reconnects == ["probe"], reconnects
            assert reconnect_outcomes[0] != "success", reconnect_outcomes
            await service.start()  # same URL, new FastMCP process state
            # The monitor's control requests run while a model request is open.
            # It must observe failure, recover and repopulate this same CLI.
            for _ in range(90):
                status = _status_of(await client.get_mcp_status())
                if status.get("status") == "connected" and status.get("tools"):
                    break
                await asyncio.sleep(.5)
            else:
                raise AssertionError(f"MCP catalog did not recover: {statuses}")
            allow_second.set()
            final = await consuming
            assert not final.is_error, final
            assert len(results) == 2, results
            assert results[0].is_error
            assert not results[1].is_error
            assert "PONG_FROM_RESTARTED_SERVER" in str(results[1].content)
            assert service.calls == 1
            assert reconnects == ["probe", "probe"], reconnects  # both from monitor
            assert reconnect_outcomes[-1] == "success", reconnect_outcomes
            assert missing["status"] in statuses and "connected" in statuses, statuses
        finally:
            live.mark_completed()
            await client.disconnect()
            provider_server.should_exit = True
            provider_server.force_exit = True
            await asyncio.wait_for(serving, timeout=5)
            sock.close()
            await service.stop()

    # timing-ok: bounds a real CLI/subprocess hang; success is event/result based.
    asyncio.run(asyncio.wait_for(run(), timeout=100))
