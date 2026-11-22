"""Opt-in fresh SDK turn against loopback MCP and a scripted upstream."""
import asyncio
from contextlib import asynccontextmanager
import json
import os
import socket
from uuid import uuid4

from claude_agent_sdk import ClaudeSDKClient, ResultMessage
from cryptography.fernet import Fernet
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
import uvicorn

from claude_code_bridge.approval_ops import ClaudeApprovalMixin
from claude_code_bridge.bootstrap import _normalize_mcp_settings
from claude_code_bridge.send_stream import ClaudeStreamMixin
from llm_bawt import task_turn_context
from llm_bawt.mcp_server import host_access_tools, server
from llm_bawt.mcp_server.registry import HTTP_APP_OPTIONS
from llm_bawt.mcp_server.task_association import TaskTurnCapabilityMiddleware
from llm_bawt.ops.host_access import create_host
from llm_bawt.utils import db

pytestmark = [pytest.mark.integration, pytest.mark.sdk_probe, pytest.mark.skipif(
    os.environ.get("RUN_HOST_ACCESS_SDK_PROBE") != "1", reason="opt-in isolated SDK host inventory probe",
)]

MODEL = "claude-sonnet-4-6"
INVENTORY = "mcp__bawthub__host_access_list"


@asynccontextmanager
async def _loopback(app):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        service = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        task = asyncio.create_task(service.serve(sockets=[sock]))
        try:
            async with asyncio.timeout(5):
                while not service.started:
                    if task.done():
                        task.result()
                    await asyncio.sleep(.01)
            yield f"http://127.0.0.1:{sock.getsockname()[1]}"
        finally:
            service.should_exit = True
            await asyncio.wait_for(task, 5)


def _wire(block, delta, stop):
    events = [
        ("message_start", {"message": {"id": "msg_" + uuid4().hex, "type": "message", "role": "assistant",
                                      "model": MODEL, "content": [], "stop_reason": None,
                                      "usage": {"input_tokens": 10, "output_tokens": 1}}}),
        ("content_block_start", {"index": 0, "content_block": block}),
        ("content_block_delta", {"index": 0, "delta": delta}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": stop}, "usage": {"output_tokens": 5}}),
        ("message_stop", {}),
    ]
    return StreamingResponse(iter([
        "".join(f"event: {kind}\ndata: {json.dumps(dict(body, type=kind))}\n\n" for kind, body in events),
    ]), media_type="text/event-stream")


@pytest.mark.asyncio
async def test_fresh_default_turn_discovers_and_calls_inventory(tmp_path, monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    create_host(engine, dict(label="Build box", host="build.example.test", port=2222, login_user="builder"))
    monkeypatch.setattr(host_access_tools, "get_shared_engine", lambda _config: engine)
    monkeypatch.setattr(db, "get_shared_engine", lambda _config: None)  # no policy DB needed
    signer = Fernet(Fernet.generate_key())
    monkeypatch.setattr(task_turn_context, "_get_fernet", lambda: signer)
    capability = task_turn_context.mint_for_agent_turn(
        backend="claude-code", session_id=str(uuid4()), turn_id="turn-" + uuid4().hex,
        trigger_message_id=str(uuid4()), bot_id="first-run", user_id="fixture-owner",
    )
    assert capability

    provider = FastAPI()
    requests = []

    @provider.post("/v1/messages")
    async def respond(request: Request):
        body = await request.json()
        if not body.get("stream"):
            return {"id": "msg_validate", "type": "message", "role": "assistant", "model": MODEL,
                    "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1}}
        requests.append(body)
        if len(requests) == 1:
            assert INVENTORY in json.dumps(body), "SDK did not advertise the default tool"
            return _wire({"type": "tool_use", "id": "toolu_host_access", "name": INVENTORY, "input": {}},
                         {"type": "input_json_delta", "partial_json": "{}"}, "tool_use")
        return _wire({"type": "text", "text": ""}, {"type": "text_delta", "text": "Inventory received."}, "end_turn")

    async def observe(*_args, **_kwargs):
        return {}

    app = server.BotAccessMiddleware(TaskTurnCapabilityMiddleware(server.mcp.streamable_http_app(**HTTP_APP_OPTIONS)))
    async with _loopback(app) as mcp_url, _loopback(provider) as provider_url:
        monkeypatch.setenv("CLAUDE_CODE_BAWTHUB_MCP_URL", f"{mcp_url}/mcp")
        default_servers = _normalize_mcp_settings({})["mcpServers"]
        # Exercise the stock BawtHub connector without starting unrelated MCP servers.
        class FixtureBridge(ClaudeApprovalMixin, ClaudeStreamMixin):
            _backend_name = "claude-code"
            _permission_mode = "bypassPermissions"
            _add_dirs = []

            def _publish_event(self, *_args, **_kwargs):
                pass  # isolated fixture has no Redis event stream

            def _make_external_failure_hook(self):
                return observe

        bridge = FixtureBridge()
        bridge._mcp_servers = {"bawthub": default_servers["bawthub"]}
        bridge._cwd = str(tmp_path)
        pre_tool_use = bridge._make_pre_tool_use_hook(
            request_id="fixture-request", session_key="first-run:fixture-owner", seq_holder=[0],
            task_turn_capability=capability,
        )
        options = bridge._build_agent_options(
            model=MODEL, system_prompt=None, disallowed_tools=[], resume_id=None, sdk_env={
                "HOME": str(tmp_path), "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude"),
                "ANTHROPIC_BASE_URL": provider_url, "ANTHROPIC_AUTH_TOKEN": "fixture-only",
                "ANTHROPIC_API_KEY": "", "CLAUDE_CODE_OAUTH_TOKEN": "",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
            }, settings_path=None, bot_effort=None, bot_max_turns=3, can_use_tool_cb=None,
            pre_tool_use_cb=pre_tool_use, post_tool_use_cb=observe, stderr=None, task_turn_capability=capability,
        )
        options.setting_sources = []
        options.strict_mcp_config = True
        options.cli_path = os.environ.get("COMPACT_PROBE_CLI")
        prompt, done = ClaudeStreamMixin._make_prompt_input("Which SSH hosts are registered here?")
        client = ClaudeSDKClient(options=options)
        try:
            async with asyncio.timeout(25):
                await client.connect(prompt=prompt)
                results = [item async for item in client.receive_response() if isinstance(item, ResultMessage)]
        finally:
            done.set()
            await client.disconnect()

    assert len(results) == 1 and not results[0].is_error
    assert len(requests) == 2
    replies = [block for message in requests[1]["messages"] if isinstance(message["content"], list)
               for block in message["content"] if block.get("type") == "tool_result"
               and block.get("tool_use_id") == "toolu_host_access"]
    assert len(replies) == 1 and not replies[0].get("is_error"), replies
    assert "build.example.test" in json.dumps(replies[0]) and "registered_only" in json.dumps(replies[0])
    engine.dispose()
