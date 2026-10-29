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


SKILLS = {"refs": ["llm-bawt"], "tokens": 4200}


@pytest.mark.parametrize("action,skills,expected", [
    ("reset_only", SKILLS, SKILLS),
    ("reset_follow_up", SKILLS, SKILLS),
    ("reset_only", None, None),  # unknown: no field, so the ack claims nothing
    ("reset_only", {"refs": [], "tokens": 0}, None),
    (None, SKILLS, None),  # never on an ordinary turn
])
def test_reset_skills_ride_only_a_reset_command(action, skills, expected):
    """TASK-1750: the bridge ack names the same skills as the app receipt."""
    import json

    from agent_bridge.continuity import RESET_SKILLS_FIELD, reset_skills_from_field

    seen = []

    class Redis:
        async def xadd(self, _stream, fields, **_kwargs):
            seen.append(fields)
            return "1-0"

    publisher = CommandPublisherMixin()
    publisher._pub_redis = Redis()
    asyncio.run(publisher.send_command(
        "bot:test", "/new", "req_test", backend="codex",
        bridge_reset_action=action, reset_skills=skills,
    ))
    raw = seen[0].get(RESET_SKILLS_FIELD)
    assert (json.loads(raw) if raw else None) == expected
    assert reset_skills_from_field(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "not json", "[]", '{"refs": "x"}', '{"refs": []}'])
def test_malformed_reset_skills_field_is_absent(raw):
    from agent_bridge.continuity import reset_skills_from_field

    assert reset_skills_from_field(raw) is None


@pytest.mark.parametrize("backend,action", [
    ("codex", "unsupported"),
    ("local", "reset_only"),
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


@pytest.mark.parametrize("stats,skills,text", [
    ({"clean_start": True, "session_id": None}, SKILLS,
     "Session reset. Injected skills: llm-bawt (~4.2k tokens). Ready for a new conversation."),
    ({"clean_start": True, "session_id": None}, None, "Session reset. Ready for a new conversation."),
    ({"seeded": True, "session_id": "sdk", "message_count": 3, "approx_tokens": 900}, SKILLS,
     "Session reset. Seeded 3 recent messages (~900 tokens). "
     "Injected skills: llm-bawt (~4.2k tokens). Continuity restored."),
])
def test_claude_bare_new_ack_names_the_reset_skills(stats, skills, text):
    events = []

    class Bridge(ClaudeSessionMixin):
        def _publish_session_reset_unified(self, *_args, **_kwargs):
            pass

        async def _seed_new_session(self, *_args, **_kwargs):
            return stats

        def _seed_usage_estimate(self, *_args, **_kwargs):
            return {}

        def _publish_event(self, *_args, **kwargs):
            events.append(kwargs)

    asyncio.run(Bridge()._preprocess_new_command(
        "/new", reset_action="reset_only", explicit_thread=False,
        bot_slug="test", session_key="bot:test", request_id="req_test",
        model="fixture", context_window=None, inject_messages=[],
        thread_session_id="fresh-thread", msg_id="1-0",
        async_redis=SimpleNamespace(xack=AsyncMock()), reset_skills=skills,
    ))
    assert events[0]["text"] == text


def test_claude_request_parses_the_reset_skills_field():
    import json

    from agent_bridge.continuity import RESET_SKILLS_FIELD
    from claude_code_bridge.send_request import SendRequest

    fields = {"request_id": "r", "session_key": "bot:test", "message": "/new",
              "bridge_reset_action": "reset_only", "thread_session_id": "s",
              RESET_SKILLS_FIELD: json.dumps(SKILLS)}
    assert SendRequest.from_fields(fields).reset_skills == SKILLS


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
    client._with_turn_settings = lambda config: config
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


def test_stream_client_forwards_reset_skills_with_the_reset():
    seen = []

    class Backend:
        def stream_raw(self, _prompt, config, **_kwargs):
            seen.append(dict(config))
            yield "ok"

    client = AgentBackendClient.__new__(AgentBackendClient)
    client._backend = Backend()
    client._bot_config = {"bot_id": "test"}
    client._with_turn_settings = lambda config: config
    list(client.stream_raw([Message(role="user", content="/new")],
                           bridge_reset_action="reset_only", reset_skills=SKILLS))
    list(client.stream_raw([Message(role="user", content="hi")], reset_skills=SKILLS))
    assert seen[0]["reset_skills"] == SKILLS
    assert "reset_skills" not in seen[1]


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
        latest_stream_usage=None, cost_baseline_usd=0.0, native_context_usage=None,
        compact_status=None, compact_error_msg=None, turn_session_id="sdk-fresh",
        turn_screenshot_assets=[],
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
