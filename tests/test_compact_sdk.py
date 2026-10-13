"""Opt-in installed-CLI compaction probes with disposable transcripts.

RUN_COMPACT_SDK_PROBE=1 runs a local fake-provider protocol test (no credentials).
COMPACT_PROBE_MODEL=provider/model additionally enables the real-provider comparison
using the existing proxy/broker. Never resumes a bot or accesses live transcripts.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import time
import uuid
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher, ResultMessage
from claude_code_bridge.bootstrap import _ensure_compact_hook
from claude_code_bridge.compact_policy import apply_compact_default, is_bare_compact
from claude_code_bridge.compact_prompt_lookup import COMPACT_PROMPT_KEY, fetch_compact_prompt
from claude_code_bridge.send_stream import ClaudeStreamMixin
from llm_bawt.chat_prompt_defaults import COMPACT_SUMMARY_DEFAULT

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_COMPACT_SDK_PROBE") != "1", reason="opt-in isolated CLI probe",
)

FACTS = (
    "Active goal: fix TASK-4242 Unicode export without deleting history. "
    "Exact file: src/export/naïve.py; identifier: export_7f3a. "
    "Authorization: implement and test only; NO commit, push, restart or deployment. "
    "Unfinished: add timeout regression and review duplicate callback ordering. "
    "Verification: 17 unit tests passed; integration NOT verified. "
    "Durable receipt: docs/receipts/TASK-4242.md; dependency TASK-4243 is blocked."
)
SUMMARY = "Continuity: " + FACTS


def seed(root: Path):
    home, work = root / "home", root / "work"
    home.mkdir(parents=True)
    work.mkdir()
    config = home / ".claude"
    project = config / "projects" / re.sub(r"[^a-zA-Z0-9]", "-", str(work))
    project.mkdir(parents=True)
    session = str(uuid.uuid4())
    transcript = project / f"{session}.jsonl"
    parent = None
    entries = []
    for n in range(40):
        for role in ("user", "assistant"):
            ident = str(uuid.uuid4())
            text = (
                f"Iteration {n}: {FACTS}\n"
                "Previously inspected logs, discussed implementation alternatives, "
                "and confirmed that obsolete approaches should not be repeated. " * 4
            )
            message = {"role": role, "content": text}
            if role == "assistant":
                message.update(model="<synthetic>", type="message", stop_reason="end_turn",
                               content=[{"type": "text", "text": text}],
                               usage={"input_tokens": 10, "output_tokens": 10})
            entries.append({
                "parentUuid": parent, "uuid": ident, "type": role, "message": message,
                "sessionId": session, "cwd": str(work), "isSidechain": False,
                "version": "2.1.283", "userType": "external",
                "timestamp": "2026-10-13T12:00:00.000Z",
            })
            parent = ident
    transcript.write_text("".join(json.dumps(e) + "\n" for e in entries))
    return home, work, config, session, transcript


async def compact(root, base_url, model, command, *, policy=True, hook_enabled=None,
                  prompt_api_url=None):
    # A benchmark baseline must disable BOTH independent sources of guidance.
    if hook_enabled is None:
        hook_enabled = policy
    home, work, config, session, transcript = seed(root)
    hooks = []

    async def precompact(data, _tool, _context):
        hooks.append(data)
        return {}  # Observe only; no invented PreCompact instruction output.

    env = {
        "HOME": str(home), "CLAUDE_CONFIG_DIR": str(config),
        "ANTHROPIC_BASE_URL": base_url, "ANTHROPIC_AUTH_TOKEN": "isolated-probe",
        "CLAUDE_CODE_OAUTH_TOKEN": "", "ANTHROPIC_API_KEY": "",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "LLM_BAWT_API_URL": prompt_api_url or "",
    }
    settings_path = root / "hook-settings.json"
    if hook_enabled:
        settings_path.write_text(json.dumps(_ensure_compact_hook({})))
    options = ClaudeAgentOptions(
        model=model, cwd=str(work), resume=session, env=env,
        tools=[], mcp_servers={}, strict_mcp_config=True, setting_sources=[],
        settings=str(settings_path) if hook_enabled else None,
        system_prompt="Summarize the isolated fixture; preserve operational continuity.",
        hooks={"PreCompact": [HookMatcher(hooks=[precompact])]},
        cli_path=os.environ.get("COMPACT_PROBE_CLI"),
    )
    guidance = None
    if policy and is_bare_compact(command):
        guidance = await asyncio.to_thread(fetch_compact_prompt, prompt_api_url)
    prompt, done = ClaudeStreamMixin._make_prompt_input(
        apply_compact_default(command, guidance) if policy else command,
    )
    client = ClaudeSDKClient(options=options)
    messages = []
    start = time.monotonic()
    try:
        await client.connect(prompt=prompt)
        async for message in client.receive_response():
            messages.append(message)
    finally:
        done.set()
        await client.disconnect()
    seconds = time.monotonic() - start
    rows = [json.loads(line) for line in transcript.read_text().splitlines()]
    boundaries = [r for r in rows if r.get("subtype") == "compact_boundary"]
    summaries = [r for r in rows if r.get("isCompactSummary")]
    summary = summaries[-1]["message"]["content"] if summaries else ""
    if isinstance(summary, list):
        summary = "\n".join(b.get("text", "") for b in summary)
    return {
        "seconds": round(seconds, 3), "chars": len(summary), "summary": summary,
        "metadata": boundaries[-1].get("compactMetadata", {}) if boundaries else {},
        "hooks": hooks, "results": [m for m in messages if isinstance(m, ResultMessage)],
        "statuses": [getattr(m, "data", {}) for m in messages],
    }


def wire(text):
    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_fixture", "type": "message", "role": "assistant",
            "model": "claude-sonnet-4-6", "content": [], "stop_reason": None,
            "usage": {"input_tokens": 12000, "output_tokens": 1},
        }}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                           "usage": {"output_tokens": 123}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(body)}\n\n" for name, body in events)


@pytest.mark.parametrize("command,failure,policy,hook_enabled", [
    pytest.param("/compact\n  ", False, True, True, id="bare-with-preflight-and-hook"),
    pytest.param("/compact\n保留 exact IDs\nWrite 15000 tokens.  ", False, True, True,
                 id="custom-bypasses-both"),
    pytest.param("/compact", True, True, True, id="provider-failure"),
    pytest.param("/compact", False, False, False, id="baseline-without-both-defaults"),
    pytest.param("/compact", False, False, True, id="bare-hook-only"),
])
def test_installed_cli_receives_policy_and_persists_complete_summary(
    tmp_path, command, failure, policy, hook_enabled,
):
    requests = []
    # Deliberately over the default target: policy must never chop model output.
    complete_summary = SUMMARY + "\nExact retained continuity." * 2000
    app = FastAPI()
    prompt_state = {"body": COMPACT_SUMMARY_DEFAULT, "reads": 0}

    @app.get(f"/v1/prompts/{COMPACT_PROMPT_KEY}")
    async def current_prompt():
        prompt_state["reads"] += 1
        return {"key": COMPACT_PROMPT_KEY, "body": prompt_state["body"],
                "source": "db_override", "updated_at": "edit-1"}

    @app.post("/v1/messages")
    async def respond(request: Request):
        requests.append(await request.json())
        if failure:
            return JSONResponse(status_code=400, content={"type": "error", "error": {
                "type": "invalid_request_error", "message": "isolated compact fixture failure",
            }})
        return StreamingResponse(iter([wire(complete_summary)]), media_type="text/event-stream")

    async def run():
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            while not server.started:
                if task.done():
                    task.result()
                await asyncio.sleep(.01)
            fixture_url = f"http://127.0.0.1:{sock.getsockname()[1]}"
            result = await compact(tmp_path, fixture_url, "claude-sonnet-4-6", command,
                                   policy=policy, hook_enabled=hook_enabled,
                                   prompt_api_url=fixture_url)
            assert len(result["results"]) == 1
            if failure:
                assert not result["metadata"]
                assert not result["summary"]
                assert any(s.get("compact_result") == "failed" for s in result["statuses"])
            else:
                assert result["metadata"], result["statuses"]
                assert complete_summary in result["summary"]
                assert not result["results"][0].is_error
            assert result["hooks"][0]["trigger"] == "manual"
            sent = apply_compact_default(command, COMPACT_SUMMARY_DEFAULT) if policy else command
            expected_instructions = sent.removeprefix("/compact").strip()
            assert (result["hooks"][0]["custom_instructions"] or "") == expected_instructions
            inputs = json.dumps(requests, ensure_ascii=False)
            assert inputs.count(COMPACT_SUMMARY_DEFAULT) == (
                1 if command.strip() == "/compact" and (policy or hook_enabled) else 0
            )
            assert prompt_state["reads"] == (
                1 if command.strip() == "/compact" and (policy or hook_enabled) else 0
            )
            if "保留 exact IDs" in command:
                assert "保留 exact IDs" in inputs
                assert "Write 15000 tokens." in inputs
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, timeout=5)
            sock.close()

    asyncio.run(asyncio.wait_for(run(), timeout=90))


def test_installed_cli_auto_compaction_receives_hook_default(tmp_path):
    """Exercise the actual command hook on a fake usage-triggered auto compact."""
    requests = []
    app = FastAPI()
    edited_guidance = "AUTO EDIT: preserve TASK-4242 and all unverified integration details."
    prompt_reads = []

    @app.get(f"/v1/prompts/{COMPACT_PROMPT_KEY}")
    async def current_prompt():
        prompt_reads.append(True)
        return {"key": COMPACT_PROMPT_KEY, "body": edited_guidance,
                "source": "db_override", "updated_at": "edit-2"}

    @app.post("/v1/messages")
    async def respond(request: Request):
        body = await request.json()
        requests.append(body)
        if not body.get("stream"):
            return {"id": "msg_validate", "type": "message", "role": "assistant",
                    "model": body["model"], "content": [{"type": "text", "text": "OK"}],
                    "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}
        text = wire(SUMMARY)
        if len(requests) == 1:
            text = text.replace('"input_tokens": 12000', '"input_tokens": 180000')
        return StreamingResponse(iter([text]), media_type="text/event-stream")

    async def run():
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            while not server.started:
                if task.done():
                    task.result()
                await asyncio.sleep(.01)
            home, work, config, session, transcript = seed(tmp_path)
            rows = [json.loads(line) for line in transcript.read_text().splitlines()]
            rows[-1]["message"]["usage"] = {"input_tokens": 180000, "output_tokens": 100}
            transcript.write_text("".join(json.dumps(row) + "\n" for row in rows))
            settings = tmp_path / "settings.json"
            settings.write_text(json.dumps(_ensure_compact_hook({})))
            env = {
                "HOME": str(home), "CLAUDE_CONFIG_DIR": str(config),
                "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{sock.getsockname()[1]}",
                "ANTHROPIC_AUTH_TOKEN": "isolated-probe", "CLAUDE_CODE_OAUTH_TOKEN": "",
                "ANTHROPIC_API_KEY": "", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_ENABLE_TELEMETRY": "0", "ANTHROPIC_SMALL_FAST_MODEL": "claude-sonnet-4-6",
                "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "50000",
                "LLM_BAWT_API_URL": f"http://127.0.0.1:{sock.getsockname()[1]}",
            }
            options = ClaudeAgentOptions(
                model="claude-sonnet-4-6", cwd=str(work), resume=session, env=env,
                tools=[], mcp_servers={}, strict_mcp_config=True, setting_sources=[],
                settings=str(settings),
                system_prompt="Preserve continuity in the isolated fixture.",
                cli_path=os.environ.get("COMPACT_PROBE_CLI"),
            )
            client = ClaudeSDKClient(options=options)
            try:
                await client.connect()
                await client.query("Continue the current work with a short status.")
                first = [m async for m in client.receive_response()]
                await client.query("Continue with the next status.")
                second = [m async for m in client.receive_response()]
            finally:
                await client.disconnect()
            rows = [json.loads(line) for line in transcript.read_text().splitlines()]
            assert any(row.get("subtype") == "compact_boundary" for row in rows)
            assert all(not m.is_error for m in first + second if isinstance(m, ResultMessage))
            assert sum(edited_guidance in json.dumps(req) for req in requests) == 1
            assert len(prompt_reads) == 1
            assert all(COMPACT_SUMMARY_DEFAULT not in json.dumps(req) for req in requests)
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, timeout=5)
            sock.close()

    asyncio.run(asyncio.wait_for(run(), timeout=90))


@pytest.mark.skipif(
    not (os.environ.get("COMPACT_PROBE_MODEL") and os.environ.get("LLM_BAWT_API_URL")),
    reason="explicit live probe model and app prompt URL required",
)
def test_real_provider_summary_size_and_continuity(tmp_path):
    from claude_code_bridge.proxy.app import ProxyServer

    async def run():
        proxy = ProxyServer()
        await proxy.start()
        try:
            results = {}
            for name, command, policy in [
                ("baseline", "/compact", False),
                ("default", "/compact", True),
                ("custom", "/compact Write a detailed summary of at least 3,500 tokens. "
                 "Preserve all operational facts and include CUSTOM-LONG-OVERRIDE.", True),
            ]:
                result = await compact(tmp_path / name, proxy.base_url,
                                       os.environ["COMPACT_PROBE_MODEL"], command,
                                       policy=policy, prompt_api_url=os.environ["LLM_BAWT_API_URL"])
                results[name] = {k: v for k, v in result.items() if k not in {"results", "statuses", "hooks"}}
                assert result["metadata"], result["statuses"]
                for marker in ("TASK-4242", "export_7f3a", "TASK-4243", "17", "docs/receipts/TASK-4242.md"):
                    assert marker in result["summary"], (name, marker)
                assert "restart" in result["summary"].lower()
                assert "timeout" in result["summary"].lower()
                assert "integration" in result["summary"].lower()
                assert "CUSTOM-LONG-OVERRIDE" in result["summary"] if name == "custom" else True
            report = tmp_path / "compact-measurements.json"
            report.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            print(f"\nCOMPACT_MEASUREMENTS={report}")
            for name, result in results.items():
                print(name, {k: v for k, v in result.items() if k != "summary"})
            assert results["default"]["chars"] < results["baseline"]["chars"]
            assert results["custom"]["chars"] > results["default"]["chars"]
        finally:
            await proxy.stop()

    asyncio.run(asyncio.wait_for(run(), timeout=600))
