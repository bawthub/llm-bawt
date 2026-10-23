"""Opt-in installed-CLI probe for the session system-prompt pin (TASK-1318).

llm-bawt pins an agent's system prompt per session because the Claude Code
CLI records the prompt when a session starts and keeps it on resume until a
manual compaction. This probe fails when that stops being true, or when the
bridge turns on bare mode (``--bare`` / ``CLAUDE_CODE_SIMPLE=1``), which
changes how the CLI builds its prompt.

RUN_SYSTEM_PROMPT_PIN_SDK_PROBE=1 (set by ``make test-sdk-probe``). Fake
localhost Anthropic upstream, isolated HOME/CLAUDE_CONFIG_DIR, real bridge
option builder. No credentials, no provider traffic, no bot sessions.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from claude_agent_sdk import ClaudeSDKClient, ResultMessage
from claude_agent_sdk.types import PermissionResultAllow
from claude_code_bridge.send_stream import ClaudeStreamMixin

pytestmark = [pytest.mark.integration, pytest.mark.sdk_probe, pytest.mark.skipif(
    os.environ.get("RUN_SYSTEM_PROMPT_PIN_SDK_PROBE") != "1", reason="opt-in isolated CLI probe",
)]

MODEL = "claude-sonnet-4-6"
MARKS = ("PIN-V1", "PIN-V2", "PIN-V3", "PIN-V4", "PIN-V5")
FILLER = "Stable persona filler line.\n" * 40
BARE_ENV = "CLAUDE_CODE_SIMPLE"


def prompt(mark: str) -> str:
    return f"## About the User\nmarker: {mark}\n\n{FILLER}"


def marks(value) -> list[str]:
    text = json.dumps(value, ensure_ascii=False)
    return [mark for mark in MARKS if mark in text]


def wire(model: str, reply: str) -> str:
    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_pin_probe", "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": reply}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": 1}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(body)}\n\n" for name, body in events)


def bridge_options(tmp: Path, base_url: str, system_prompt: str, resume_id: str | None):
    """Options from the bridge's own builder, pointed at the isolated fixture."""
    home, work = tmp / "home", tmp / "work"
    home.mkdir(exist_ok=True)
    work.mkdir(exist_ok=True)
    env = {
        "HOME": str(home), "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        "ANTHROPIC_BASE_URL": base_url, "ANTHROPIC_AUTH_TOKEN": "isolated-probe",
        "ANTHROPIC_API_KEY": "", "CLAUDE_CODE_OAUTH_TOKEN": "",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
    }
    bridge = SimpleNamespace(_mcp_servers={}, _cwd=str(work), _permission_mode="default", _add_dirs=[])

    async def allow(*_args, **_kwargs):
        return PermissionResultAllow()

    async def observe(*_args, **_kwargs):
        return {}

    options = ClaudeStreamMixin._build_agent_options(
        bridge, model=MODEL, system_prompt=system_prompt, disallowed_tools=[],
        resume_id=resume_id, sdk_env=env, settings_path=None, bot_effort=None,
        bot_max_turns=1, can_use_tool_cb=allow, pre_tool_use_cb=observe,
        post_tool_use_cb=observe, stderr=None,
    )
    options.setting_sources = []
    options.strict_mcp_config = True
    options.cli_path = os.environ.get("COMPACT_PROBE_CLI")
    return options, home


def assert_not_bare(options) -> None:
    extra = {str(key).lstrip("-") for key in (options.extra_args or {})}
    assert "bare" not in extra, "bridge enabled --bare; the session prompt pin no longer holds"
    for source in (options.env or {}, os.environ):
        assert str(source.get(BARE_ENV, "")).strip() in ("", "0"), f"{BARE_ENV} is on"


def test_cli_freezes_the_system_prompt_until_a_manual_compaction(tmp_path):
    requests: list[dict] = []
    app = FastAPI()

    @app.post("/v1/messages")
    async def respond(request: Request):
        body = await request.json()
        requests.append(body)
        if not body.get("stream"):
            return JSONResponse({"id": "msg_pin_probe", "type": "message", "role": "assistant",
                                 "model": body.get("model", MODEL),
                                 "content": [{"type": "text", "text": "ok"}],
                                 "stop_reason": "end_turn",
                                 "usage": {"input_tokens": 1, "output_tokens": 1}})
        return StreamingResponse(iter([wire(body.get("model", MODEL), "ok")]),
                                 media_type="text/event-stream")

    @app.post("/v1/messages/count_tokens")
    async def count(_request: Request):
        return {"input_tokens": 100}

    # (prompt the bridge passes, user message, prompt the CLI must keep)
    plan = [
        ("PIN-V1", "turn one", "PIN-V1"),      # cold start records V1
        ("PIN-V2", "turn two", "PIN-V1"),      # resume ignores a changed prompt
        ("PIN-V3", "/compact", "PIN-V1"),      # the compaction call keeps V1
        ("PIN-V4", "turn four", "PIN-V4"),     # first turn after compaction re-records
        ("PIN-V5", "turn five", "PIN-V4"),     # and the new prompt is frozen again
    ]

    async def run():
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        observed = []
        try:
            while not server.started:
                if task.done():
                    task.result()
                await asyncio.sleep(.01)
            base_url = f"http://127.0.0.1:{sock.getsockname()[1]}"
            session_id = None
            home = None
            for passed, message, _kept in plan:
                options, home = bridge_options(tmp_path, base_url, prompt(passed), session_id)
                assert_not_bare(options)
                before = len(requests)
                content, done = ClaudeStreamMixin._make_prompt_input(message)
                client = ClaudeSDKClient(options=options)  # one client per turn, as the bridge does
                results = []
                try:
                    await client.connect(prompt=content)
                    async for item in client.receive_response():
                        if isinstance(item, ResultMessage):
                            results.append(item)
                finally:
                    done.set()
                    await client.disconnect()
                assert len(results) == 1 and not results[0].is_error, results
                session_id = results[0].session_id
                observed.append([marks(body.get("system")) for body in requests[before:]])
            return observed, home
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, timeout=5)
            sock.close()

    observed, home = asyncio.run(asyncio.wait_for(run(), timeout=180))
    for (passed, message, kept), systems in zip(plan, observed):
        assert systems, f"{message!r}: no model call reached the fixture"
        assert all(found == [kept] for found in systems), (
            f"{message!r} passed {passed}: CLI sent system prompts {systems}, expected only {kept}"
        )

    transcripts = list((home / ".claude" / "projects").rglob("*.jsonl"))
    assert len(transcripts) == 1, transcripts
    rows = [json.loads(line) for line in transcripts[0].read_text().splitlines()]
    assert sum(row.get("subtype") == "compact_boundary" for row in rows) == 1
