"""TASK-1879 disposable local-provider SDK/CLI ordering probe (no operational credentials)."""
from __future__ import annotations

import asyncio
import json
import os
import socket
import uuid
from pathlib import Path

import pytest
import uvicorn
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ResultMessage
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from claude_code_bridge.active_run import ClaudeActiveRun

pytestmark = [pytest.mark.integration, pytest.mark.sdk_probe, pytest.mark.skipif(
    os.environ.get("RUN_WORKER_LIVENESS_SDK_PROBE") != "1", reason="opt-in isolated SDK probe",
)]
MODEL = "claude-sonnet-4-6"  # protocol fixture, not runtime model preference


def _wire(blocks, stop="end_turn"):
    events = [("message_start", {"type": "message_start", "message": {
        "id": "msg_" + uuid.uuid4().hex[:8], "type": "message", "role": "assistant",
        "model": MODEL, "content": [], "stop_reason": None,
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }})]
    for index, block in enumerate(blocks):
        if block["type"] == "text":
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        events.extend([
            ("content_block_start", {"type": "content_block_start", "index": index, "content_block": start}),
            ("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta}),
            ("content_block_stop", {"type": "content_block_stop", "index": index}),
        ])
    events.extend([
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop},
                           "usage": {"output_tokens": 5}}),
        ("message_stop", {"type": "message_stop"}),
    ])
    return StreamingResponse(iter([
        "".join(f"event: {event}\ndata: {json.dumps(body)}\n\n" for event, body in events),
    ]), media_type="text/event-stream")


@pytest.mark.anyio
async def test_local_cli_background_bash_result_order(tmp_path: Path):
    home, work = tmp_path / "home", tmp_path / "work"
    (home / ".claude").mkdir(parents=True)
    work.mkdir()
    marker, finished = tmp_path / "allow", tmp_path / "finished"
    marker2, finished2 = tmp_path / "allow2", tmp_path / "finished2"
    notification_seen, release_notification = asyncio.Event(), asyncio.Event()
    release_notification.set()
    block_notification = False
    app = FastAPI()

    @app.post("/v1/messages")
    async def respond(request: Request):
        body = await request.json()
        if not body.get("stream"):
            return {"id": "msg_validate", "type": "message", "role": "assistant",
                    "model": MODEL, "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}
        content = json.dumps(body.get("messages", [])[-1:])
        if "START_BACKGROUND" in content:
            second_job = "START_BACKGROUND2" in content
            allow, done = (marker2, finished2) if second_job else (marker, finished)
            command = (
                f"while [ ! -f {allow} ]; do sleep 0.05; done; "
                f"printf task_finished > {done}"
            )
            return _wire([{"type": "tool_use", "id": "toolu_probe_bg", "name": "Bash", "input": {
                "command": command, "description": "isolated background completion",
                "run_in_background": True,
            }}], "tool_use")
        if "FOLLOW_INTERRUPT" in content:
            return _wire([{"type": "text", "text": "FOLLOW_INTERRUPT_DONE"}])
        if "<task-notification>" in content:
            notification_seen.set()
            if block_notification:
                await release_notification.wait()
            return _wire([{"type": "text", "text": "NOTIFY_DONE"}])
        if "FOLLOW" in content:
            return _wire([{"type": "text", "text": "FOLLOW_DONE"}])
        return _wire([{"type": "text", "text": "STARTED"}])

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    server.install_signal_handlers = lambda: None
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        if serving.done():
            serving.result()
        await asyncio.sleep(0.01)
    client = ClaudeSDKClient(options=ClaudeAgentOptions(
        model=MODEL, cwd=str(work), env={
            "HOME": str(home), "CLAUDE_CONFIG_DIR": str(home / ".claude"),
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{sock.getsockname()[1]}",
            "ANTHROPIC_AUTH_TOKEN": "local-probe", "ANTHROPIC_API_KEY": "",
            "CLAUDE_CODE_OAUTH_TOKEN": "", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_ENABLE_TELEMETRY": "0", "ANTHROPIC_SMALL_FAST_MODEL": MODEL,
        }, mcp_servers={}, strict_mcp_config=True, setting_sources=[],
        permission_mode="bypassPermissions", system_prompt="Local probe fixture.", max_turns=4,
    ))
    frames = []
    live = None
    try:
        await client.connect()
        query = client._query
        original = query.receive_messages

        async def capture():
            async for frame in original():
                frames.append((str(frame.get("type")), str(frame.get("subtype")),
                               frame.get("origin"), str(frame.get("result", ""))[:50]))
                yield frame

        query.receive_messages = capture
        live = ClaudeActiveRun(client=client, request_id="local-probe")

        async def turn(prompt):
            await client.query(prompt)
            seen = []
            async for message in live.messages():
                seen.append(message)
                if isinstance(message, ResultMessage):
                    return seen
            return seen

        first = await asyncio.wait_for(turn("START_BACKGROUND"), timeout=25)
        marker.touch()

        async def wait_for_file(path):
            while not path.exists():
                await asyncio.sleep(.02)

        await asyncio.wait_for(wait_for_file(finished), timeout=10)
        await asyncio.wait_for(notification_seen.wait(), timeout=10)
        live.mark_completed()
        live = ClaudeActiveRun(client=client, request_id="normal-follow")
        second = await asyncio.wait_for(turn("FOLLOW"), timeout=25)
        assert [(m.result, m.origin) for m in first if isinstance(m, ResultMessage)] == [
            ("STARTED", None)
        ]
        assert [(m.result, m.origin) for m in second if isinstance(m, ResultMessage)] == [
            ("FOLLOW_DONE", None)
        ]
        normal_results = [frame for frame in frames if frame[0] == "result"]
        assert normal_results[-2:] == [
            ("result", "success", {"kind": "task-notification"}, "NOTIFY_DONE"),
            ("result", "success", None, "FOLLOW_DONE"),
        ]

        # A second real background Bash completion is held at the fake provider,
        # allowing a host query and now-steer while the injected turn runs.
        live.mark_completed()
        live = ClaudeActiveRun(client=client, request_id="create-interrupt-job")
        block_notification = True
        release_notification.clear()
        notification_seen.clear()
        created = await asyncio.wait_for(turn("START_BACKGROUND2"), timeout=25)
        marker2.touch()
        await asyncio.wait_for(wait_for_file(finished2), timeout=10)
        await asyncio.wait_for(notification_seen.wait(), timeout=10)
        live.mark_completed()
        live = ClaudeActiveRun(client=client, request_id="interrupt-notification")
        await client.query("FOLLOW_INTERRUPT")

        async def consume_follow():
            seen = []
            async for message in live.messages():
                if isinstance(message, ResultMessage):
                    seen.append((message.result, message.origin, message.subtype))
                    if not live.consume_replaced_result(message):
                        return seen
            return seen

        consuming = asyncio.create_task(consume_follow())
        await live.steer("replacement", priority="now", origin="user",
                         framed_message="NOW[replacement]")
        release_notification.set()
        # timing-ok: this bounds a real SDK subprocess hang; success is the result sequence.
        third = await asyncio.wait_for(consuming, timeout=20)
        assert [(m.result, m.origin) for m in created if isinstance(m, ResultMessage)] == [
            ("STARTED", None)
        ]
        assert third == [
            (None, {"kind": "task-notification"}, "error_during_execution"),
            ("FOLLOW_INTERRUPT_DONE", None, "success"),
        ]
        assert live._finishing is True
        assert live._expected_interrupt_results == 0
        assert [frame[:3] for frame in frames if frame[0] == "result"][-2:] == [
            ("result", "error_during_execution", {"kind": "task-notification"}),
            ("result", "success", None),
        ]
    finally:
        release_notification.set()
        marker.touch()
        marker2.touch()
        if live is not None:
            live.mark_completed()
        await client.disconnect()
        server.should_exit = True
        await asyncio.wait_for(serving, timeout=5)
        sock.close()
