"""TASK-1148 opt-in proof against the installed Claude Code CLI (no credentials).

A fake localhost Anthropic provider drives a parent that spawns two agents,
then resumes both with serial SendMessage calls. The first resumed worker's
model request never produces output. The real TurnWatchdog + WorkerLiveness
must stop only that worker, the parent must receive the liveness note and
continue, the queued sibling must run, and the turn must finish cleanly.
A stalled *spawned* Agent remains a parent-terminal failure (unchanged).

RUN_WORKER_LIVENESS_SDK_PROBE=1 (set by ``make test-sdk-probe``).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher, ResultMessage
from claude_agent_sdk.types import (
    TaskNotificationMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    ToolResultBlock,
    UserMessage,
)
from claude_code_bridge.event_ops import ClaudeEventMixin
from claude_code_bridge.subagent_usage import SubagentUsage
from claude_code_bridge.turn_watchdog import TurnWatchdog, TurnWatchdogTimeout
from claude_code_bridge.worker_liveness import WorkerLiveness

pytestmark = [pytest.mark.integration, pytest.mark.sdk_probe, pytest.mark.skipif(
    os.environ.get("RUN_WORKER_LIVENESS_SDK_PROBE") != "1", reason="opt-in isolated CLI probe",
)]

MARK = "PARENT_MARKER_1148"
MODEL = "claude-sonnet-4-6"


def _sse(blocks, stop="end_turn"):
    events = [("message_start", {"type": "message_start", "message": {
        "id": "msg_" + uuid.uuid4().hex[:8], "type": "message", "role": "assistant", "model": MODEL,
        "content": [], "stop_reason": None, "usage": {"input_tokens": 10, "output_tokens": 1}}})]
    for i, block in enumerate(blocks):
        if block["type"] == "text":
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        events += [
            ("content_block_start", {"type": "content_block_start", "index": i, "content_block": start}),
            ("content_block_delta", {"type": "content_block_delta", "index": i, "delta": delta}),
            ("content_block_stop", {"type": "content_block_stop", "index": i}),
        ]
    events += [("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop},
                                  "usage": {"output_tokens": 5}}),
               ("message_stop", {"type": "message_stop"})]
    return StreamingResponse(
        iter(["".join(f"event: {n}\ndata: {json.dumps(b)}\n\n" for n, b in events)]),
        media_type="text/event-stream",
    )


def _stall():
    async def body():
        yield "event: message_start\ndata: " + json.dumps({"type": "message_start", "message": {
            "id": "msg_stall", "type": "message", "role": "assistant", "model": MODEL, "content": [],
            "stop_reason": None, "usage": {"input_tokens": 10, "output_tokens": 1}}}) + "\n\n"
        for _ in range(120):  # keepalives only: the observed upstream no-output stall
            await asyncio.sleep(1)
            yield 'event: ping\ndata: {"type": "ping"}\n\n'
    return StreamingResponse(body(), media_type="text/event-stream")


def _flat(content):
    if isinstance(content, str):
        return content
    return "\n".join(
        b.get("text", "") if b.get("type") == "text" else json.dumps(b.get("content"))
        for b in content if isinstance(b, dict)
    )


def _provider(*, stall_spawn: bool, parent_requests: list[str]):
    app = FastAPI()

    @app.post("/v1/messages")
    async def respond(request: Request):
        body = await request.json()
        msgs = body.get("messages", [])
        last = _flat(msgs[-1]["content"]).strip() if msgs else ""
        if not body.get("stream"):
            return {"id": "msg_x", "type": "message", "role": "assistant", "model": MODEL,
                    "content": [{"type": "text", "text": "OK"}], "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1}}
        if MARK in json.dumps(body.get("system", "")):
            parent_requests.append(json.dumps(msgs[-1:]))
            if last.endswith("SPAWN"):
                return _sse([{"type": "tool_use", "id": f"toolu_spawn{x}", "name": "Agent", "input": {
                    "description": f"worker {x}", "prompt": f"WORKER_{x}",
                    "subagent_type": "general-purpose"}} for x in "AB"], "tool_use")
            if last.endswith("RESUME"):
                ids = re.findall(r"agentId: ([A-Za-z0-9_-]+)", "\n".join(_flat(m["content"]) for m in msgs))
                return _sse([{"type": "tool_use", "id": f"toolu_send{x}", "name": "SendMessage", "input": {
                    "to": agent, "message": f"PING_{x}", "summary": f"ping {x}"}}
                    for x, agent in zip("AB", ids[:2])], "tool_use")
            return _sse([{"type": "text", "text": "PARENT_DONE"}])
        if "PING_A" in last or (stall_spawn and last.endswith("WORKER_A")):
            return _stall()
        return _sse([{"type": "text", "text": "B_PONG" if "PING_B" in last else "worker ready"}])

    return app


async def _session(tmp_path: Path, *, stall_spawn: bool):
    parent_requests: list[str] = []
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(
        _provider(stall_spawn=stall_spawn, parent_requests=parent_requests),
        log_level="error", lifespan="off",
    ))
    serve = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.01)
    home, work = tmp_path / "home", tmp_path / "work"
    (home / ".claude").mkdir(parents=True)
    work.mkdir()
    liveness = WorkerLiveness()
    hook = ClaudeEventMixin._make_post_tool_use_hook(
        SimpleNamespace(), session_key="probe", screenshot_artifacts_by_tool_use_id={},
        worker_liveness=liveness,
    )
    env = {
        "HOME": str(home), "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{sock.getsockname()[1]}",
        "ANTHROPIC_AUTH_TOKEN": "probe", "CLAUDE_CODE_OAUTH_TOKEN": "", "ANTHROPIC_API_KEY": "",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
        "ANTHROPIC_SMALL_FAST_MODEL": MODEL, "CLAUDE_CODE_SUBAGENT_MODEL": MODEL,
        # Same as the bridge: workers run inline, SendMessage resumes serialise.
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    }
    client = ClaudeSDKClient(options=ClaudeAgentOptions(
        model=MODEL, cwd=str(work), env=env, mcp_servers={}, strict_mcp_config=True,
        setting_sources=[], permission_mode="bypassPermissions",
        system_prompt=f"{MARK} test parent",
        hooks={"PostToolUse": [HookMatcher(hooks=[hook])]},
    ))
    stopped: list[str] = []

    async def stop_task(task_id: str) -> None:
        stopped.append(task_id)
        await client.stop_task(task_id)

    async def turn(prompt: str) -> list:
        usage = SubagentUsage()
        watchdog = TurnWatchdog(
            idle_timeout=4.0, health_interval=0.5, tool_grace=2.0,
            resolve_parent=usage.host_parent, liveness=liveness, stop_task=stop_task,
        )
        await client.query(prompt)
        stream = client.receive_response()
        seen = []
        while True:
            try:
                msg = await watchdog.receive_next(stream, request_id=prompt, session_key="probe")
            except StopAsyncIteration:
                return seen
            if isinstance(msg, TaskStartedMessage):
                usage.register_task(msg.tool_use_id, msg.task_id, msg.description or "")
            seen.append(msg)

    async def close():
        await client.disconnect()
        server.should_exit = True
        await asyncio.wait_for(serve, timeout=5)

    await client.connect()
    return SimpleNamespace(turn=turn, close=close, stopped=stopped, liveness=liveness,
                           parent_requests=parent_requests)


def _results(seen):
    return {b.tool_use_id: b for m in seen if isinstance(m, UserMessage) and isinstance(m.content, list)
            for b in m.content if isinstance(b, ToolResultBlock)}


def test_stalled_resume_fails_its_call_and_the_parent_completes(tmp_path):
    async def run():
        probe = await _session(tmp_path, stall_spawn=False)
        try:
            assert isinstance((await probe.turn("SPAWN"))[-1], ResultMessage)
            seen = await asyncio.wait_for(probe.turn("RESUME"), timeout=90)
        finally:
            await probe.close()
        return probe, seen

    probe, seen = asyncio.run(run())
    starts = {m.tool_use_id: m.task_id for m in seen if isinstance(m, TaskStartedMessage)}
    # Serial resumes: A ran and was stopped; queued B started only afterwards.
    assert set(starts) == {"toolu_sendA", "toolu_sendB"}
    assert probe.stopped == [starts["toolu_sendA"]]
    order = [getattr(m, "tool_use_id", None) for m in seen if isinstance(m, TaskStartedMessage)]
    assert order == ["toolu_sendA", "toolu_sendB"]
    terminal = {(m.task_id, m.status) for m in seen
                if isinstance(m, (TaskUpdatedMessage, TaskNotificationMessage)) and m.status}
    assert (starts["toolu_sendA"], "killed") in terminal or (starts["toolu_sendA"], "stopped") in terminal
    assert probe.liveness.stop_status(starts["toolu_sendA"], "stopped")[0] == "timed_out"
    assert probe.liveness.stop_for_tool("toolu_sendB") is None

    results = _results(seen)
    assert "interrupted" in str(results["toolu_sendA"].content)
    assert "B_PONG" in str(results["toolu_sendB"].content)
    # The parent model was told why A failed, then finished the turn.
    assert any("Bridge worker liveness" in req for req in probe.parent_requests)
    final = seen[-1]
    assert isinstance(final, ResultMessage) and not final.is_error
    assert final.result == "PARENT_DONE"


def test_stalled_spawned_agent_is_still_a_parent_terminal_failure(tmp_path):
    async def run():
        probe = await _session(tmp_path, stall_spawn=True)
        try:
            with pytest.raises(TurnWatchdogTimeout) as raised:
                await asyncio.wait_for(probe.turn("SPAWN"), timeout=60)
        finally:
            await probe.close()
        return probe, raised.value

    probe, error = asyncio.run(run())
    assert error.phase == "tool_running"
    assert probe.stopped == []
