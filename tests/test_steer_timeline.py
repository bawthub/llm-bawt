"""TASK-1067: accepted mid-turn messages reach the timeline before completion."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from llm_bawt.entity_presentation import EntityPresentationResolver
from llm_bawt.message_authorship import AuthorReference
from llm_bawt.service.steer_timeline import SteerTimeline


@pytest.fixture
def timeline():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as conn:
        conn.execute(text("""CREATE TABLE messages (
            id TEXT, bot_id TEXT, content TEXT, timestamp REAL, session_id TEXT,
            author_entity_type TEXT, author_entity_id TEXT,
            interrupt_source_message_id TEXT, interrupt_content_offset INTEGER
        )"""))
    service = SimpleNamespace(_turn_log_store=SimpleNamespace(engine=engine))
    result = SteerTimeline(service, bot_id="target", user_id="nick", turn_id="turn-live", message_id="message-steer")
    yield result
    engine.dispose()


def persist(timeline, offset=12):
    with timeline.engine.begin() as conn:
        conn.execute(text("""INSERT INTO messages VALUES (
            'message-steer', 'target', 'Message from bot sender: ready', 123,
            'session-target', 'bot', 'sender', 'trigger', :offset
        )"""), {"offset": offset})


def test_infers_live_offset_without_waiting_for_turn_end(timeline):
    subscriber = SimpleNamespace(turn_text_offset=AsyncMock(return_value=8))
    assert asyncio.run(timeline.content_offset(subscriber, None)) == 8
    subscriber.turn_text_offset.assert_awaited_once_with("target", "nick", "turn-live")


def test_browser_offset_is_preserved_without_stream_lookup(timeline):
    subscriber = SimpleNamespace(turn_text_offset=AsyncMock())
    assert asyncio.run(timeline.content_offset(subscriber, 0)) == 0
    subscriber.turn_text_offset.assert_not_awaited()


def test_delivery_retry_cannot_move_existing_anchor(timeline):
    persist(timeline, offset=12)
    subscriber = SimpleNamespace(turn_text_offset=AsyncMock())
    assert asyncio.run(timeline.content_offset(subscriber, 999)) == 12
    subscriber.turn_text_offset.assert_not_awaited()


def test_unknown_turn_position_never_invents_an_anchor(timeline):
    subscriber = SimpleNamespace(turn_text_offset=AsyncMock(return_value=None))
    assert asyncio.run(timeline.content_offset(subscriber, None)) is None


def test_publishes_canonical_message_to_recipient_not_sender(timeline, monkeypatch):
    persist(timeline)
    author = {"entity_type": "bot", "entity_id": "sender", "display_name": "Sender"}
    monkeypatch.setattr(EntityPresentationResolver, "resolve_many_safe", lambda self, refs: {("bot", "sender"): author})
    subscriber = SimpleNamespace(publish_tool_event=AsyncMock(return_value="stream-id"))
    asyncio.run(timeline.publish(subscriber))
    subscriber.publish_tool_event.assert_awaited_once_with("target", "nick", {
        "_type": "steer_message", "bot_id": "target", "user_id": "nick",
        "turn_id": "turn-live", "message_id": "message-steer", "session_id": "session-target",
        "content": "Message from bot sender: ready", "author": author,
        "interrupt_source_message_id": "trigger", "interrupt_content_offset": 12, "ts": 123.0,
    })


def test_publish_failure_does_not_undo_persisted_message(timeline, monkeypatch):
    persist(timeline)
    monkeypatch.setattr(EntityPresentationResolver, "resolve_many_safe", lambda self, refs: {("bot", "sender"): {}})
    subscriber = SimpleNamespace(publish_tool_event=AsyncMock(side_effect=RuntimeError("offline")))
    asyncio.run(timeline.publish(subscriber))
    assert timeline._message()["interrupt_content_offset"] == 12


@pytest.mark.parametrize(
    ("author", "steer_request_id", "persisted_author"),
    [
        (AuthorReference.bot("sender"), "steer_delivery_test", ("bot", "sender")),
        # Public /v1/chat/steer passes no author: the composer message is the
        # turn owner's. NULL authorship hid live user corrections from recall.
        (None, None, ("user", "nick")),
    ],
)
def test_active_steer_persists_anchor_then_publishes_without_ending_turn(
    monkeypatch, author, steer_request_id, persisted_author,
):
    from llm_bawt.agent_backends import agent_bridge
    from llm_bawt.bots import BotManager
    from llm_bawt.service.routes import chat

    calls = []
    turn = SimpleNamespace(id="turn-live", bot_id="target", user_id="nick", ended_at=None,
                           status="streaming", agent_session_key="session-key", agent_request_id="req",
                           trigger_message_id="trigger")
    memory = SimpleNamespace(get_message_by_id=lambda id: {"session_id": "session-target"},
                             add_message=lambda *args: calls.append(("persist", args)))
    service = SimpleNamespace(config=object(), _turn_log_store=SimpleNamespace(get_turn=lambda id: turn),
                              get_memory_client=lambda *args: memory)
    subscriber = SimpleNamespace(send_steer=AsyncMock(return_value={"ok": True}))
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: subscriber)
    monkeypatch.setattr(BotManager, "__init__", lambda *args: None)
    monkeypatch.setattr(BotManager, "get_bot", lambda *args: SimpleNamespace(agent_backend="claude-code"))

    class Timeline:
        def __init__(self, service, **kwargs):
            assert kwargs["bot_id"] == "target"

        async def content_offset(self, sub, requested):
            assert requested is None
            return 12

        async def publish(self, sub):
            calls.append(("publish", turn.ended_at))

    monkeypatch.setattr("llm_bawt.service.steer_timeline.SteerTimeline", Timeline)
    monkeypatch.setattr(chat, "_persist_interrupt_anchor", lambda *args, **kwargs: calls.append(("anchor", kwargs)) or True)
    result = asyncio.run(chat.steer_active_turn(service, chat.ChatSteerRequest(
        turn_id="turn-live", bot_id="target", user_id="nick", message_id="message-steer", message="ready",
    ), steer_request_id=steer_request_id, author=author))
    assert result.persisted is True
    assert [call[0] for call in calls] == ["persist", "anchor", "publish"]
    assert calls[0][1][-2:] == persisted_author
    assert calls[1][1]["content_offset"] == 12
    assert calls[1][1]["source_message_id"] == "trigger"
    assert calls[2] == ("publish", None)


def test_delivery_steer_sends_wire_suffix_but_persists_clean_text(monkeypatch):
    """The reply-channel note reaches the bridge only; history keeps the message."""
    from llm_bawt.agent_backends import agent_bridge
    from llm_bawt.bots import BotManager
    from llm_bawt.service.routes import chat

    persisted = []
    turn = SimpleNamespace(id="turn-live", bot_id="target", user_id="nick", ended_at=None,
                           status="streaming", agent_session_key="session-key", agent_request_id="req",
                           trigger_message_id="trigger")
    memory = SimpleNamespace(get_message_by_id=lambda id: {"session_id": "session-target"},
                             add_message=lambda *args: persisted.append(args))
    service = SimpleNamespace(config=object(), _turn_log_store=SimpleNamespace(get_turn=lambda id: turn),
                              get_memory_client=lambda *args: memory)
    subscriber = SimpleNamespace(send_steer=AsyncMock(return_value={"ok": True}))
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: subscriber)
    monkeypatch.setattr(BotManager, "__init__", lambda *args: None)
    monkeypatch.setattr(BotManager, "get_bot", lambda *args: SimpleNamespace(agent_backend="claude-code"))

    class Timeline:
        def __init__(self, service, **kwargs):
            pass

        async def content_offset(self, sub, requested):
            return None

        async def publish(self, sub):
            pass

    monkeypatch.setattr("llm_bawt.service.steer_timeline.SteerTimeline", Timeline)
    asyncio.run(chat.steer_active_turn(service, chat.ChatSteerRequest(
        turn_id="turn-live", bot_id="target", user_id="nick", message_id="message-steer",
        message="Message from bot 'sender': verdict?",
    ), steer_request_id="steer_delivery_test", author=AuthorReference.bot("sender"),
        wire_suffix="[Reply channel: async delivery.]"))

    sent = subscriber.send_steer.await_args.kwargs["message"]
    assert sent == "Message from bot 'sender': verdict?\n\n[Reply channel: async delivery.]"
    assert len(persisted) == 1
    assert "Message from bot 'sender': verdict?" in persisted[0]
    assert not any("Reply channel" in str(arg) for arg in persisted[0])


@pytest.mark.parametrize("steer_request_id", [None, "steer_delivery_test"])
def test_run_finishing_is_a_definite_409_without_composer_retries(monkeypatch, steer_request_id):
    """TASK-1344: the run already produced its final result; it will not become
    steerable again, so the composer loop must not spin and callers get a 409
    they route to a fresh turn."""
    from fastapi import HTTPException

    from llm_bawt.agent_backends import agent_bridge
    from llm_bawt.bots import BotManager
    from llm_bawt.service.routes import chat

    turn = SimpleNamespace(id="turn-live", bot_id="target", user_id="nick", ended_at=None,
                           status="streaming", agent_session_key="session-key", agent_request_id="req",
                           trigger_message_id="trigger")
    service = SimpleNamespace(config=object(), _turn_log_store=SimpleNamespace(get_turn=lambda id: turn),
                              get_memory_client=lambda *args: None)
    subscriber = SimpleNamespace(send_steer=AsyncMock(return_value={"ok": False, "error": "run_finishing"}))
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: subscriber)
    monkeypatch.setattr(BotManager, "__init__", lambda *args: None)
    monkeypatch.setattr(BotManager, "get_bot", lambda *args: SimpleNamespace(agent_backend="claude-code"))

    with pytest.raises(HTTPException) as raised:
        asyncio.run(chat.steer_active_turn(service, chat.ChatSteerRequest(
            turn_id="turn-live", bot_id="target", user_id="nick", message_id="message-steer", message="late",
        ), steer_request_id=steer_request_id))

    assert (raised.value.status_code, raised.value.detail) == (409, "run_finishing")
    assert subscriber.send_steer.await_count == 1
