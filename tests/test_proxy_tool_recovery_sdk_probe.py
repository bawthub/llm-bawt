"""Offline CLI contract: an unavailable carrier produces a recoverable tool error."""
from __future__ import annotations

import asyncio
import os
import socket

import pytest
import uvicorn
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ResultMessage
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from claude_code_bridge.proxy.tool_recovery import CARRIER_NAME, ToolArgumentGenerationError, recovery_frames
from tests.test_claude_result_origin_sdk_probe import MODEL, _wire
from tests.test_proxy_truncated_tools import frame

pytestmark = [pytest.mark.integration, pytest.mark.sdk_probe, pytest.mark.skipif(
    os.environ.get("RUN_WORKER_LIVENESS_SDK_PROBE") != "1", reason="opt-in isolated SDK probe",
)]


@pytest.mark.anyio
@pytest.mark.parametrize("scenario", ["carrier", "empty_max_tokens"])
async def test_unknown_carrier_returns_error_and_continues_without_execution(tmp_path, scenario):
    home, work = tmp_path / "home", tmp_path / "work"
    (home / ".claude").mkdir(parents=True)
    work.mkdir()
    app = FastAPI()
    bodies = []
    results = []
    carrier = CARRIER_NAME

    @app.post("/v1/messages")
    async def respond(request: Request):
        body = await request.json()
        bodies.append(body)
        if scenario == "empty_max_tokens":
            return _wire([], "max_tokens")
        for message in body.get("messages", []):
            content = message.get("content", [])
            if isinstance(content, list):
                results.extend(block for block in content if block.get("type") == "tool_result"
                               and block.get("tool_use_id") == "toolu_stall_fixture")
        if results:
            return _wire([{"type": "text", "text": "RECOVERED_WITHOUT_EXECUTION"}])
        assert carrier not in {tool["name"] for tool in body.get("tools", [])}
        start = frame("message_start", message={
            "id": "msg_fixture", "type": "message", "role": "assistant", "model": MODEL,
            "content": [], "stop_reason": None, "usage": {"input_tokens": 10, "output_tokens": 0},
        })
        recovery = ToolArgumentGenerationError("Bash", 1543, "ChatGPTEventTimeout: upstream whitespace stall")
        return StreamingResponse(iter([start, *recovery_frames(
            recovery, index=0, call_id="toolu_stall_fixture",
        )]), media_type="text/event-stream")

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    server.install_signal_handlers = lambda: None
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        if serving.done():
            serving.result()
        await asyncio.sleep(.01)
    client = ClaudeSDKClient(options=ClaudeAgentOptions(
        model=MODEL, cwd=str(work), env={
            "HOME": str(home), "CLAUDE_CONFIG_DIR": str(home / ".claude"),
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{sock.getsockname()[1]}",
            "ANTHROPIC_AUTH_TOKEN": "local-probe", "ANTHROPIC_API_KEY": "",
            "CLAUDE_CODE_OAUTH_TOKEN": "", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_ENABLE_TELEMETRY": "0", "ANTHROPIC_SMALL_FAST_MODEL": MODEL,
        }, mcp_servers={}, strict_mcp_config=True, setting_sources=[],
        permission_mode="bypassPermissions", system_prompt="Local protocol fixture.", max_turns=3,
    ))
    try:
        await client.connect()
        await client.query("Recover the simulated tool argument failure.")
        # timing-ok: bounds a real isolated SDK subprocess, not a performance assertion.
        async with asyncio.timeout(30):
            messages = [message async for message in client.receive_response()]
        if scenario == "carrier":
            assert results and results[-1].get("is_error") is True
            assert any(isinstance(message, ResultMessage) and not message.is_error for message in messages)
            assert len(bodies) <= 3
        else:
            terminal = next(message for message in messages if isinstance(message, ResultMessage))
            assert terminal.is_error
            assert "output token maximum" in str(terminal.result) + str(terminal.errors)
            assert len(bodies) > 1
            # The CLI manufactures continuation requests, changing the body
            # fingerprint even though no usable model output has arrived.
            assert bodies[0]["messages"] != bodies[-1]["messages"]
        assert not list(work.iterdir())
    finally:
        await client.disconnect()
        server.should_exit = True
        await serving
        sock.close()
