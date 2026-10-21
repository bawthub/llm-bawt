"""Structured reset commands stay request-local across Redis and bridge preflight."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_bridge.command_publisher import CommandPublisherMixin
from claude_code_bridge.send_preflight import prepare_send
from claude_code_bridge.send_result import ClaudeResultMixin
from claude_code_bridge.session_ops import ClaudeSessionMixin
from llm_bawt.clients.agent_backend_client import AgentBackendClient
from llm_bawt.models.message import Message


@pytest.mark.parametrize("backend,action,message", [
    ("claude-code", "reset_only", " /new\n"),
    ("claude-code", "reset_follow_up", "Do the task; this is not a /new command."),
    ("codex", "reset_only", "/new"),
    ("codex", "reset_follow_up", "Do the task; this is not a /new command."),
])
def test_reset_action_is_serialized_beside_unmodified_message(backend, action, message):
    seen = []

    class Redis:
        async def xadd(self, _stream, fields, **_kwargs):
            seen.append(fields)
            return "1-0"

    publisher = CommandPublisherMixin()
    publisher._pub_redis = Redis()
    asyncio.run(publisher.send_command(
        "bot:test", message, "req_test", backend=backend,
        bridge_reset_action=action,
    ))
    assert seen[0]["message"] == message
    assert seen[0]["bridge_reset_action"] == action


@pytest.mark.parametrize("backend,action", [
    ("codex", "unsupported"),
    ("openclaw", "reset_only"),
])
def test_publisher_rejects_unsupported_reset_action(backend, action):
    publisher = CommandPublisherMixin()
    publisher._pub_redis = SimpleNamespace(xadd=AsyncMock())
    with pytest.raises(ValueError, match="Unsupported agent bridge reset action"):
        asyncio.run(publisher.send_command(
            "bot:test", "/new", "req_test", backend=backend,
            bridge_reset_action=action,
        ))
    publisher._pub_redis.xadd.assert_not_awaited()


@pytest.mark.parametrize("action,message,expected", [
    ("reset_only", " /new\n", None),
    ("reset_follow_up", "Do the task; this is not a /new command.",
     "Do the task; this is not a /new command."),
])
def test_bridge_reset_action_resets_without_rewriting_follow_up(action, message, expected):
    events = []

    class Bridge(ClaudeSessionMixin):
        _publisher = SimpleNamespace(publish_run_done=lambda *_: events.append("done"))

        def _publish_session_reset_unified(self, *_args, **_kwargs):
            events.append("reset")

        async def _seed_new_session(self, *_args, **_kwargs):
            events.append("seed")
            return {"clean_start": True, "session_id": None}

        def _format_seed_ack(self, *_args):
            return "ack"

        def _seed_usage_estimate(self, *_args, **_kwargs):
            return {}

        def _publish_event(self, *_args, **kwargs):
            events.append(kwargs)

    redis = SimpleNamespace(xack=AsyncMock())
    bridge = Bridge()
    result = asyncio.run(bridge._preprocess_new_command(
        message, reset_action=action, explicit_thread=False,
        bot_slug="test", session_key="bot:test", request_id="req_test",
        model="fixture", context_window=None, inject_messages=[],
        thread_session_id="fresh-thread", msg_id="1-0", async_redis=redis,
    ))
    assert result == expected
    assert events[0] == "reset"
    if action == "reset_only":
        assert events[1] == "seed"
        assert events[2]["extra_raw"]["new_session_ack"]["thread_session_id"] == "fresh-thread"
        redis.xack.assert_awaited_once()
    else:
        assert events == ["reset"]
        redis.xack.assert_not_called()


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_agent_client_reset_action_is_request_local(stream):
    seen = []

    class Backend:
        async def chat_full(self, _prompt, config):
            seen.append(dict(config))
            return SimpleNamespace(text="ok", tool_calls=[])

        def stream_raw(self, _prompt, config, **_kwargs):
            seen.append(dict(config))
            yield "ok"

    client = AgentBackendClient.__new__(AgentBackendClient)
    client._backend = Backend()
    client._bot_config = {"bot_id": "test"}
    client._with_question_wait = lambda config: config
    if stream:
        assert list(client.stream_raw(
            [Message(role="user", content="Do task")],
            bridge_reset_action="reset_follow_up",
        )) == ["ok"]
    else:
        assert client.query(
            [Message(role="user", content="Do task")],
            bridge_reset_action="reset_follow_up",
        ) == "ok"
    assert seen[0]["bridge_reset_action"] == "reset_follow_up"
    assert "bridge_reset_action" not in client._bot_config


def test_preflight_recognizes_structured_follow_up_without_command_text():
    fields = {"request_id": "req", "session_key": "bot:test", "bot_id": "test",
              "model": "fixture", "message": "Do task", "thread_session_id": "new",
              "bridge_reset_action": "reset_follow_up"}
    bridge = SimpleNamespace(
        _trigger_message_ids={},
        _preprocess_new_command=AsyncMock(return_value="Do task"),
    )
    result = asyncio.run(prepare_send(bridge, fields, "1-0", SimpleNamespace()))
    assert result.reset_requested is True
    assert result.message == "Do task"
    assert bridge._preprocess_new_command.await_args.kwargs["reset_action"] == "reset_follow_up"


def test_successful_follow_up_done_carries_reset_evidence_not_replacement_text():
    events = []

    class Bridge(ClaudeResultMixin):
        def _compute_result_usage(self, *_args, **_kwargs):
            return {}, None, None

        def _publish_event(self, *_args, **kwargs):
            events.append(kwargs)

    evidence = {"thread_session_id": "fresh-thread", "provider_session_id": "sdk-fresh",
                "clean_start": False}
    asyncio.run(Bridge()._finalize_result_message(
        SimpleNamespace(text="actual task answer"), request_id="req", session_key="bot:test",
        seq=0, text_parts=[], assistant_snapshot_text="", api_retry_count=0,
        api_last_error=None, actual_model="fixture", model="fixture",
        bot_context_window=None, latest_assistant_usage=None,
        latest_stream_usage=None, native_context_usage=None, compact_status=None,
        compact_error_msg=None, turn_session_id="sdk-fresh", turn_screenshot_assets=[],
        reset_ack=evidence,
    ))
    assert events[0]["text"] == "actual task answer"
    assert events[0]["extra_raw"] == {"new_session_ack": evidence}


def test_preflight_rejects_structured_reset_on_explicit_thread():
    fields = {"request_id": "req", "session_key": "bot:test", "bot_id": "test",
              "model": "fixture", "message": "hello", "explicit_thread": "1",
              "thread_session_id": "archived", "bridge_reset_action": "reset_follow_up"}
    with pytest.raises(ValueError, match="Invalid structured reset"):
        asyncio.run(prepare_send(SimpleNamespace(_trigger_message_ids={}),
                                 fields, "1-0", SimpleNamespace()))
