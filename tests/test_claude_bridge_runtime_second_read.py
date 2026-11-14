"""TASK-2286: second-read fixes in the Claude bridge runtime."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from claude_agent_sdk import AssistantMessage, TextBlock

from agent_bridge.events import AgentEventKind
from claude_code_bridge import skill_hooks
from claude_code_bridge.bootstrap import bootstrap_claude_home
from claude_code_bridge.session_ops import ClaudeSessionMixin
from claude_code_bridge.skill_hooks import git_gate, record
from tests.test_claude_code_send_boundaries import MODEL, SESSION, _SDKStream, _SendHarness

SEED = '[{"role":"user","content":"retained context"}]'


def _reset_turn():
    return {
        "request_id": "turn-request", "session_key": SESSION,
        "model": MODEL, "message": "Do the supplied task",
        "thread_session_id": "fresh-thread", "thread_resume_id": "stale-sdk",
        "trigger_message_id": "message-turn-request",
        "bridge_reset_action": "reset_follow_up",
        "inject_messages": SEED,
    }


# Skill markers are per context, not shared with sub-agents.
def test_skill_marker_is_not_shared_between_parent_and_sub_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(skill_hooks, "MARKER_DIR", tmp_path)
    push = {"session_id": "s1", "tool_input": {"command": "git push"}}
    record({"session_id": "s1", "tool_input": {"skill": "git-commits"}}, {})
    assert git_gate(push, {}) is None
    assert git_gate({**push, "agent_id": "sub-1"}, {}) is not None
    record({"session_id": "s2", "agent_id": "sub-2", "tool_input": {"skill": "git-commits"}}, {})
    assert git_gate({**push, "session_id": "s2", "agent_id": "sub-2"}, {}) is None
    assert git_gate({**push, "session_id": "s2"}, {}) is not None
    assert git_gate({**push, "agent_id": "sub-1"},
                    {skill_hooks.INJECTED_SKILLS_ENV: "git-commits"}) is None


# The Playwright URL follows the env after first boot.
def test_bootstrap_overwrites_a_stale_playwright_url(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    path = home / ".claude" / "settings.json"
    path.write_text(json.dumps({"mcpServers": {"playwright": {"type": "http", "url": "http://old/mcp"}}}))
    (tmp_path / "skills").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CODE_SKILLS_PATH", str(tmp_path / "skills"))
    monkeypatch.setenv("CLAUDE_CODE_PLAYWRIGHT_MCP_URL", "http://new:8931/mcp")
    bootstrap_claude_home()
    assert json.loads(path.read_text())["mcpServers"]["playwright"]["url"] == "http://new:8931/mcp"


# A failed tool-context fetch is not cached.
def test_failed_mcp_tool_context_fetch_is_retried(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"body": "override for {bot_slug}"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "claude_code_bridge.session_ops.httpx.AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    bridge = object.__new__(ClaudeSessionMixin)
    bridge._app_api_url = "http://app"

    async def run():
        first = await bridge._get_mcp_tool_context("al", "nick")
        second = await bridge._get_mcp_tool_context("al", "nick")
        third = await bridge._get_mcp_tool_context("al", "nick")
        return first, second, third

    first, second, third = asyncio.run(run())
    assert "override" not in first
    assert second == third == "override for al"
    assert len(calls) == 2


# A thread-binding failure is reported by the seed.
def test_seed_reports_a_failed_thread_binding():
    from agent_bridge.continuity import SeedFraming
    bridge = object.__new__(ClaudeSessionMixin)
    bridge._write_seed_transcript = Mock()
    bridge._set_thread_session = AsyncMock(return_value=False)
    bridge._record_continuity_receipt = AsyncMock()
    result = asyncio.run(bridge._seed_new_session(
        "al", "model", json.loads(SEED), "dest", framing=SeedFraming(briefing="b", ack="a")))
    assert result["seeded"] and result["persisted"] is False


def test_unpersisted_seed_is_bound_again_at_turn_init(monkeypatch):
    from claude_agent_sdk import SystemMessage

    class InitSDK(_SDKStream):
        async def receive_messages(self):
            yield SystemMessage(subtype="init", data={"session_id": "seeded-sdk", "model": MODEL})
            yield AssistantMessage(content=[TextBlock(text="done")], model=MODEL)

    async def run():
        harness = _SendHarness()
        harness._seed_new_session = AsyncMock(return_value={
            "seeded": True, "clean_start": False, "session_id": "seeded-sdk", "persisted": False})
        harness._set_thread_session = AsyncMock(return_value=True)
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: InitSDK())
        harness._build_agent_options = Mock(return_value=None)
        await harness._handle_send(_reset_turn(), "send-turn-request", harness.redis)
        harness._set_thread_session.assert_awaited_once()
        assert harness._set_thread_session.await_args.args[2] == "seeded-sdk"
    asyncio.run(asyncio.wait_for(run(), timeout=5))


# A reset turn ending at EOF still sends its ack.
def test_reset_follow_up_eof_fallback_carries_the_reset_ack(monkeypatch):
    class EofSDK(_SDKStream):
        async def receive_messages(self):
            yield AssistantMessage(content=[TextBlock(text="done")], model=MODEL)

    async def run():
        harness = _SendHarness()
        harness._seed_new_session = AsyncMock(return_value={
            "seeded": True, "clean_start": False, "session_id": "seeded-sdk"})
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: EofSDK())
        harness._build_agent_options = Mock(return_value=None)
        await harness._handle_send(_reset_turn(), "send-turn-request", harness.redis)
        done = [e for e in harness.events if e.kind == AgentEventKind.ASSISTANT_DONE]
        assert len(done) == 1
        ack = done[0].raw["new_session_ack"]
        assert ack["provider_session_id"] == "seeded-sdk" and ack["clean_start"] is False
        assert ack["seeded_count"] == 1
    asyncio.run(asyncio.wait_for(run(), timeout=5))


# A fresh-session retry re-seeds instead of dropping the seed.
@pytest.mark.parametrize("reseed_ok", [True, False])
def test_fresh_retry_of_a_seeded_reset_reseeds(monkeypatch, reseed_ok):
    attempts = []

    class CrashThenAnswer(_SDKStream):
        async def receive_messages(self):
            if len(attempts) == 1:
                raise RuntimeError("CLI subprocess failed with exit code 1")
            yield AssistantMessage(content=[TextBlock(text="done")], model=MODEL)

    def client(**kw):
        sdk = CrashThenAnswer()
        attempts.append(sdk)
        return sdk

    async def run():
        harness = _SendHarness()
        harness._uses_claude_oauth_gateway = lambda: True
        second = ({"seeded": True, "clean_start": False, "session_id": "reseeded-sdk"}
                  if reseed_ok else {"seeded": False, "reason": "seed write failed: disk"})
        harness._seed_new_session = AsyncMock(side_effect=[
            {"seeded": True, "clean_start": False, "session_id": "seeded-sdk"}, second])
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", client)
        harness._build_agent_options = Mock(return_value=None)
        await harness._handle_send(_reset_turn(), "send-turn-request", harness.redis)
        assert harness._seed_new_session.await_count == 2
        acks = [e.raw.get("new_session_ack") for e in harness.events if e.raw.get("new_session_ack")]
        if reseed_ok:
            assert len(attempts) == 2
            assert acks and acks[0]["provider_session_id"] == "reseeded-sdk"
        else:
            assert len(attempts) == 1 and not acks
            errors = [e for e in harness.events if e.kind == AgentEventKind.ERROR]
            assert errors and "seed failed on retry" in errors[0].text
    asyncio.run(asyncio.wait_for(run(), timeout=5))
