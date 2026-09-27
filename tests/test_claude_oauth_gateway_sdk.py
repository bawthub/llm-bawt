"""Opt-in installed CLI proof: fake upstream, isolated HOME, one local Read.

RUN_CLAUDE_OAUTH_SDK_PROBE=1 python -m pytest tests/test_claude_oauth_gateway_sdk.py -q
No live provider credentials, traffic, bot sessions or refreshes are used.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from claude_code_bridge.proxy import claude_oauth as gateway

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CLAUDE_OAUTH_SDK_PROBE") != "1", reason="opt-in installed CLI probe",
)


def wire(block, *, tool=False):
    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_fixture", "type": "message", "role": "assistant", "model": "claude-sonnet-4-6",
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 1},
        }}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": (
            {**block, "input": {}} if tool else {"type": "text", "text": ""}
        )}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": (
            {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
            if tool else {"type": "text_delta", "text": block["text"]}
        )}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {
            "stop_reason": "tool_use" if tool else "end_turn", "stop_sequence": None,
        }, "usage": {"output_tokens": 10}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return b"".join(f"event: {name}\ndata: {json.dumps(body)}\n\n".encode() for name, body in events)


class Stream(httpx.AsyncByteStream):
    def __init__(self, body):
        self.body = body

    async def __aiter__(self):
        yield self.body


def test_native_cli_recovers_after_tool_without_reexecuting_it(tmp_path, monkeypatch):
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, ResultMessage, query

    home = tmp_path / "home"
    home.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    fixture = work / "fixture.txt"
    fixture.write_text("read exactly once")
    requests = []
    broker_calls = []
    reads = []
    state = {"token": "a"}

    def broker(**kwargs):
        broker_calls.append(kwargs)
        return state["token"], None

    monkeypatch.setattr(gateway, "_fetch_broker_token", broker)

    async def upstream(req):
        requests.append(req)
        n = len(requests)
        if n == 1:
            body = wire({"type": "tool_use", "id": "tool_fixture", "name": "Read",
                         "input": {"file_path": str(fixture)}}, tool=True)
            return httpx.Response(200, stream=Stream(body), headers={"content-type": "text/event-stream"})
        if n == 2:
            state["token"] = "b"  # rotation between resolution and upstream authentication
            body = b'{"type":"error","error":{"type":"authentication_error","message":"OAuth access token has been revoked."}}'
            return httpx.Response(401, stream=Stream(body), headers={"content-type": "application/json"})
        assert n == 3, "SDK unexpectedly replayed the turn/request"
        return httpx.Response(200, stream=Stream(wire({"type": "text", "text": "Recovered after one Read."})),
                              headers={"content-type": "text/event-stream"})

    async def pre_tool(input_data, tool_use_id, context):
        reads.append(input_data["tool_name"])
        return {}

    async def run():
        proxy = gateway.ClaudeOAuthGateway(httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
        app = FastAPI()
        app.include_router(gateway.router)
        app.state.claude_oauth_gateway = proxy
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        server.install_signal_handlers = lambda: None
        task = asyncio.create_task(server.serve(sockets=[sock]))
        results = []
        try:
            while not server.started:
                if task.done():
                    task.result()
                    raise RuntimeError("fixture server exited")
                await asyncio.sleep(0.01)
            options = ClaudeAgentOptions(
                model="claude-sonnet-4-6", cwd=str(work),
                env={
                    "HOME": str(home), "CLAUDE_CONFIG_DIR": str(home / ".claude"),
                    "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}/claude-oauth",
                    "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-local-fixture",
                    "ANTHROPIC_AUTH_TOKEN": "", "ANTHROPIC_API_KEY": "",
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
                },
                tools=["Read"], allowed_tools=["Read"],
                hooks={"PreToolUse": [HookMatcher(matcher="Read", hooks=[pre_tool])]},
                mcp_servers={}, strict_mcp_config=True, setting_sources=[],
                system_prompt="Local protocol fixture. Read the specified file once.",
                max_turns=3,
            )
            async for message in query(prompt=f"Read {fixture} once then finish.", options=options):
                if isinstance(message, ResultMessage):
                    results.append(message)
            assert len(results) == 1
            assert not results[0].is_error, results[0].result
            assert "Recovered after one Read." in results[0].result
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, timeout=5)
            sock.close()
            await proxy.close()

    asyncio.run(asyncio.wait_for(run(), timeout=60))
    assert reads == ["Read"]
    assert len(requests) == 3
    assert requests[1].content == requests[2].content
    assert [r.headers["authorization"] for r in requests] == ["Bearer a", "Bearer a", "Bearer b"]
    assert broker_calls == [{}, {}, {"force": True, "rejected_token_sha256": hashlib.sha256(b"a").hexdigest()}]
    history = json.loads(requests[-1].content)["messages"]
    assert any(b.get("type") == "tool_result" for m in history for b in m.get("content", []) if isinstance(b, dict))
    assert "oauth-2025-04-20" in requests[-1].headers.get("anthropic-beta", "")
