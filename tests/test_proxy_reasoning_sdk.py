"""Opt-in local CLI proof: no provider traffic, credentials, tools or bot sessions.

RUN_PROXY_SDK_PROBE=1 python -m pytest tests/test_proxy_reasoning_sdk.py -q
"""
import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from claude_code_bridge.proxy.reasoning import ReasoningCodec
from claude_code_bridge.proxy.stream import responses_to_anthropic_sse
from claude_code_bridge.proxy.chatgpt_transport import _event_object

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_PROXY_SDK_PROBE") != "1", reason="opt-in installed CLI probe",
)


def test_cli_persists_and_replays_opaque_reasoning_on_resume(tmp_path):
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    codec = ReasoningCodec.for_account("openai_chatgpt", "test", "http://local", "fake")
    item = {"type": "reasoning", "id": "rs_local_probe", "summary": [],
            "encrypted_content": "opaque-local-fixture"}
    requests = []

    async def frames():
        async def upstream():
            for event in [
                {"type": "response.output_item.added", "item": {"type": "reasoning", "id": item["id"]}},
                {"type": "response.output_item.done", "item": item},
                {"type": "response.output_text.delta", "delta": "Local fixture answer."},
                {"type": "response.completed", "response": {"status": "completed", "usage": None}},
            ]:
                yield _event_object(event)
        return b"".join([frame async for frame in responses_to_anthropic_sse(
            upstream(), "openai_chatgpt/test", reasoning_codec=codec,
        )])
    wire = asyncio.run(frames())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.startswith("/v1/messages") and "count_tokens" not in self.path:
                requests.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(wire)))
                self.end_headers()
                self.wfile.write(wire)
            else:
                payload = b'{"input_tokens":100}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / "isolated-home"
    home.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    env = {
        "HOME": str(home), "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{server.server_port}",
        "ANTHROPIC_AUTH_TOKEN": "local-fixture", "ANTHROPIC_API_KEY": "",
        "CLAUDE_CODE_OAUTH_TOKEN": "", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
    }

    async def run():
        sid = None
        for prompt in ("First local fixture turn", "Second local fixture turn"):
            options = ClaudeAgentOptions(
                model="openai_chatgpt/test", cwd=str(work), env=env,
                tools=[], mcp_servers={}, strict_mcp_config=True, setting_sources=[],
                system_prompt="Local protocol fixture. No tools.",
                max_turns=1, resume=sid,
            )
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    assert not message.is_error, message.result
                    sid = message.session_id
        return sid
    try:
        assert asyncio.run(asyncio.wait_for(run(), timeout=60))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert len(requests) >= 2
    signatures = [block.get("signature") for message in requests[-1]["messages"]
                  for block in message.get("content", []) if isinstance(block, dict)
                  and block.get("type") == "thinking"]
    assert any(codec.decode(signature) == item for signature in signatures)
