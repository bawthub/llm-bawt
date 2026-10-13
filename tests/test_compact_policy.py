"""TASK-1071: custom compaction instructions always bypass default guidance."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from claude_code_bridge.compact_policy import apply_compact_default, is_bare_compact
from llm_bawt.chat_prompt_defaults import COMPACT_SUMMARY_DEFAULT
from claude_code_bridge.send_preflight import prepare_send
from claude_code_bridge.send_stream import ClaudeStreamMixin


BARE = ["/compact", "/compact   ", "/compact\n\t", "  /compact\r\n", "/compact "]
CUSTOM = [
    "/compact keep IDs",
    "/compact x",
    "/compact Write a huge 15,000-token summary. Do not be concise.",
    "/compact\nKeep TASK-1071 exactly.\n保留全部细节 🐕\n\t ",
    "  /compact\t  first line\r\nsecond line  ",
    "/compact é — exact Unicode\n",
]
ORDINARY = [
    "Mention /compact in the docs", "What does /compact do?", "`/compact`",
    "/compactness", "/compact-custom", "/compact: detail", "/COMPACT", "",
    "hello\n/compact", "/compact/", "/compact🐕",
]


@pytest.mark.parametrize("message", BARE)
def test_bare_command_gets_only_default(message):
    result = apply_compact_default(message, COMPACT_SUMMARY_DEFAULT)
    assert result == f"{message.rstrip()} {COMPACT_SUMMARY_DEFAULT}"
    assert "2,000 tokens" in result
    assert apply_compact_default(result) == result


@pytest.mark.parametrize("message", CUSTOM + ORDINARY)
def test_custom_and_non_commands_are_byte_identical(message):
    assert apply_compact_default(message, COMPACT_SUMMARY_DEFAULT) == message
    assert COMPACT_SUMMARY_DEFAULT not in apply_compact_default(message, COMPACT_SUMMARY_DEFAULT)


def _preflight(message):
    bridge = SimpleNamespace(
        _trigger_message_ids={},
        _preprocess_new_command=AsyncMock(side_effect=lambda text, **kw: text),
        _publish_event=Mock(), _publisher=Mock(),
    )
    fields = {"message": message, "request_id": "compact-test", "model": "fixture-model"}

    async def run():
        with patch("claude_code_bridge.send_preflight.fetch_compact_prompt",
                   return_value=COMPACT_SUMMARY_DEFAULT) as lookup:
            prepared = await prepare_send(
                bridge, fields, "redis-id", SimpleNamespace(xack=AsyncMock()),
            )
            assert lookup.call_count == int(is_bare_compact(message))
            return prepared

    return run()


@pytest.mark.parametrize("message", BARE + CUSTOM + [m for m in ORDINARY if m])
def test_preflight_to_sdk_prompt_preserves_policy_and_source(message):
    async def run():
        prepared = await _preflight(message)
        assert prepared.request.message == message  # history/attribution unchanged
        prompt, done = ClaudeStreamMixin._make_prompt_input(prepared.message)
        frame = await anext(prompt)
        assert frame["message"]["content"] == apply_compact_default(message, COMPACT_SUMMARY_DEFAULT)
        done.set()
        with pytest.raises(StopAsyncIteration):
            await anext(prompt)

    asyncio.run(run())


def test_mcp_compact_reaches_same_default_without_changing_delivery_contract(monkeypatch):
    from llm_bawt.mcp_server import server  # noqa: F401 — initialize registration first
    from llm_bawt.mcp_server import inter_bot_tools

    captured = []

    async def enqueue(**kwargs):
        captured.append(kwargs)
        prepared = await _preflight(kwargs["message"])
        return {"prompt": prepared.message}

    monkeypatch.setattr(inter_bot_tools, "agent_context_health", AsyncMock(return_value={
        "backend": "claude-code", "capabilities": {"compact": True},
    }))
    monkeypatch.setattr(inter_bot_tools, "_enqueue_durable", enqueue)
    result = asyncio.run(inter_bot_tools.agent_context_compact(
        "fixture-bot", "compact-test-key", sender_bot_id="fixture-sender",
    ))
    assert result["prompt"] == f"/compact {COMPACT_SUMMARY_DEFAULT}"
    assert captured[0]["prefer_steer"] is False
    assert captured[0]["message"] == "/compact"
    assert captured[0]["max_tokens"] is None
    assert captured[0]["message_kind"] == "CONTEXT_MAINTENANCE"
    assert captured[0]["idempotency_key"] == "compact-test-key"


def test_no_global_auto_compact_guidance_or_cap(monkeypatch):
    """Automatic compaction must not inherit a default from unrelated turns."""
    harness = ClaudeStreamMixin()
    harness._proxy_base_url = "http://fixture"
    env = harness._build_sdk_env(
        use_proxy=True, model="fixture/model", subagent_model=None,
        force_refresh=False, bot_id="fixture", session_key="fixture",
        thread_session_id="fixture", request_id="fixture", compact_threshold=250000,
    )
    assert COMPACT_SUMMARY_DEFAULT not in str(env)
    assert "2000" not in env.values()
    assert "2,000" not in env.values()
    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "250000"
