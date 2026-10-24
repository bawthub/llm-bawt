"""Every rotation origin passes its working-memory mode (docs/WORKING_MEMORY.md 10.1, N1).

Only an explicit ``include_working_memory=false`` on a session or context API,
or a bot delivery resetting without history, suppresses. Everything else pins.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from llm_bawt.memory.session_reset_lineage import SessionResetLineage
from llm_bawt.message_authorship import AuthorReference
from llm_bawt.service.session_lifecycle.execution import NewSessionExecution
from llm_bawt.service.session_lifecycle.intent import (
    Admission, Continuity, NewSessionIntent, Origin,
)
from llm_bawt.service.session_lifecycle.lifecycle import NewSessionLifecycle
from llm_bawt.working_memory.session_state import mode_from_flag, require_mode


@pytest.fixture
def sessions():
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        conn.execute(text("""CREATE TABLE sessions (
            id TEXT PRIMARY KEY, bot_id TEXT, user_id TEXT, status TEXT,
            started_at TIMESTAMP, ended_at TIMESTAMP, archived_at TIMESTAMP, session_metadata TEXT
        )"""))
        conn.execute(text("""INSERT INTO sessions (id, bot_id, user_id, status, session_metadata)
            VALUES ('old', 'test-bot', 'test-user', 'active', '{}')"""))
    yield engine
    engine.dispose()


def metadata(engine, session_id):
    with engine.connect() as conn:
        raw = conn.execute(text("SELECT session_metadata FROM sessions WHERE id=:id"),
                           {"id": session_id}).scalar_one_or_none()
    return None if raw is None else json.loads(raw)


def rotate(engine, new_id="next", **kwargs):
    return SessionResetLineage.rotate(engine, bot_id="test-bot", user_id="test-user",
                                      new_id=new_id, **kwargs)


def test_flag_and_mode_helpers():
    assert mode_from_flag(None) == "pinned"
    assert mode_from_flag(True) == "pinned"
    assert mode_from_flag(False) == "suppressed"
    assert require_mode("suppressed") == "suppressed"
    for bad in (None, "", "PINNED", True):
        with pytest.raises(ValueError):
            require_mode(bad)


def test_rotate_requires_a_mode(sessions):
    with pytest.raises(TypeError, match="working_memory_mode"):
        rotate(sessions)
    with pytest.raises(ValueError, match="working_memory_mode"):
        rotate(sessions, working_memory_mode="off")
    # A refused mode writes nothing: the predecessor stays active, no successor.
    assert metadata(sessions, "next") is None
    with sessions.connect() as conn:
        assert conn.execute(text("SELECT status FROM sessions WHERE id='old'")).scalar_one() == "active"


@pytest.mark.parametrize("mode", ["pinned", "suppressed"])
def test_rotate_writes_the_mode_on_the_successor_only(sessions, mode):
    assert rotate(sessions, working_memory_mode=mode) is True
    assert metadata(sessions, "next")["working_memory"] == {"mode": mode}
    assert "working_memory" not in metadata(sessions, "old")


def test_keyed_replay_keeps_the_original_mode(sessions):
    assert rotate(sessions, working_memory_mode="suppressed", idempotency_key="delivery-1") is True
    assert rotate(sessions, working_memory_mode="pinned", idempotency_key="delivery-1") is False
    assert metadata(sessions, "next")["working_memory"] == {"mode": "suppressed"}


def _intent(**overrides):
    options = dict(
        bot_id="test-bot", user_id="test-user", author=AuthorReference.user("test-user"),
        continuity=Continuity.CONFIGURED, admission=Admission.WAIT,
        origin=Origin.CHAT_COMMAND, idempotency_key="delivery-1",
    )
    options.update(overrides)
    return NewSessionIntent(**options)


def _api(origin=Origin.CONTEXT_API, continuity=Continuity.CLEAN, **overrides):
    return _intent(origin=origin, continuity=continuity, admission=Admission.SKIP, **overrides)


@pytest.mark.parametrize(("intent", "mode"), [
    (_intent(), "pinned"),  # typed /new
    (_intent(continuity=Continuity.CLEAN), "pinned"),  # typed clean /new is not a suppression
    (_intent(origin=Origin.SCHEDULER, author=AuthorReference.bot("system"), admission=Admission.SKIP,
             expected_session_id="s1", expected_usage_turn_id="t1", job_id="j1", run_id="r1"), "pinned"),
    (_intent(origin=Origin.BOT_DELIVERY, continuity=Continuity.RETAIN), "pinned"),
    (_intent(origin=Origin.BOT_DELIVERY, continuity=Continuity.CLEAN), "suppressed"),
    (_api(), "pinned"),  # API clean reset without the flag
    (_api(include_working_memory=True), "pinned"),
    (_api(include_working_memory=False), "suppressed"),
    (_api(continuity=Continuity.RETAIN, include_working_memory=False), "suppressed"),
    (_api(origin=Origin.SESSION_API, include_working_memory=False), "suppressed"),
])
def test_intent_mode_by_origin(intent, mode):
    assert intent.working_memory_mode == mode


@pytest.mark.parametrize("changes", [
    {"include_working_memory": False},  # typed /new has no such flag
    {"include_working_memory": False, "origin": Origin.BOT_DELIVERY},
])
def test_flag_outside_session_or_context_api_is_refused(changes):
    with pytest.raises(ValueError, match="include_working_memory"):
        _intent(**changes)


@pytest.mark.parametrize("value", ["false", 0, 1])
def test_non_boolean_flag_is_refused(value):
    with pytest.raises(ValueError, match="include_working_memory"):
        _api(include_working_memory=value)


def test_encoded_intent_keeps_old_byte_shape_and_round_trips_the_flag():
    assert "include_working_memory" not in NewSessionLifecycle._encode(_api())
    encoded = NewSessionLifecycle._encode(_api(include_working_memory=False))
    assert encoded["include_working_memory"] is False
    decoded = NewSessionLifecycle._decode(json.loads(json.dumps(encoded)))
    assert decoded.working_memory_mode == "suppressed"


@pytest.mark.parametrize(("metadata", "policy", "mode"), [
    (None, "reset_without_history", "suppressed"),
    ({}, "reset_retain_history", "pinned"),
    (None, None, "pinned"),
    # The persisted intent decides over the delivery's policy, both ways.
    ({"new_session_intent": NewSessionLifecycle._encode(
        _api(continuity=Continuity.RETAIN, include_working_memory=False))},
     "reset_retain_history", "suppressed"),
    ({"new_session_intent": NewSessionLifecycle._encode(_api())},
     "reset_without_history", "pinned"),
])
def test_claimed_delivery_mode(metadata, policy, mode):
    assert NewSessionLifecycle.working_memory_mode(metadata, policy) == mode


def _service(record):
    store = SimpleNamespace(get=lambda delivery_id: record if delivery_id == "delivery-1" else None)
    return SimpleNamespace(_inter_bot_dispatcher=SimpleNamespace(store=store))


def test_typed_new_mode_comes_from_the_durable_delivery():
    clean = SimpleNamespace(metadata={}, session_policy="reset_without_history")
    assert NewSessionExecution.working_memory_mode(_service(clean), None) == "pinned"
    assert NewSessionExecution.working_memory_mode(SimpleNamespace(), "delivery-1") == "pinned"
    assert NewSessionExecution.working_memory_mode(_service(clean), "delivery-1") == "suppressed"
    typed = SimpleNamespace(metadata={"new_session_intent": NewSessionLifecycle._encode(
        _intent(continuity=Continuity.CLEAN))}, session_policy="reset_without_history")
    assert NewSessionExecution.working_memory_mode(_service(typed), "delivery-1") == "pinned"
    with pytest.raises(ValueError, match="missing"):
        NewSessionExecution.working_memory_mode(_service(clean), "delivery-gone")


@pytest.mark.parametrize(("flag", "mode"), [(None, "pinned"), (True, "pinned"), (False, "suppressed")])
def test_sessions_rotate_tool_maps_the_flag(monkeypatch, flag, mode):
    # session_tools registers on server's `mcp`; importing it first is circular.
    from llm_bawt.mcp_server import server  # noqa: F401
    from llm_bawt.mcp_server import session_tools

    seen = {}

    async def rotate_session(**kwargs):
        seen.update(kwargs)
        return "next"

    monkeypatch.setattr(session_tools, "_get_storage", lambda: SimpleNamespace(rotate_session=rotate_session))
    kwargs = {} if flag is None else {"include_working_memory": flag}
    assert asyncio.run(session_tools.rotate_session(bot_id="test-bot", user_id="test-user", **kwargs)) == "next"
    assert seen["working_memory_mode"] == mode


@pytest.mark.parametrize(("mode", "flag"), [("pinned", True), ("suppressed", False)])
def test_server_client_sends_the_flag_explicitly(mode, flag):
    from llm_bawt.mcp_server.client import MemoryClient

    client = MemoryClient(SimpleNamespace(), bot_id="test-bot", user_id="test-user",
                          server_url="http://testserver")
    calls = []
    client._call_server = lambda tool, args: calls.append((tool, args)) or "next"
    assert client.rotate_session("delivery-1", working_memory_mode=mode) == "next"
    assert calls == [("sessions_rotate", {
        "bot_id": "test-bot", "user_id": "test-user", "accept_pointer": False,
        "include_working_memory": flag, "key": "delivery-1",
    })]
    with pytest.raises(ValueError, match="working_memory_mode"):
        client.rotate_session(working_memory_mode="off")
    assert len(calls) == 1


def _route(monkeypatch):
    from llm_bawt.service.routes import sessions

    seen = []

    async def rotate_session(**kwargs):
        seen.append(kwargs)
        return "next"

    monkeypatch.setattr(sessions, "get_effective_bot_id", lambda bot: bot)
    monkeypatch.setattr(sessions, "_resolve_user", lambda user: user)
    monkeypatch.setattr(sessions, "get_storage", lambda: SimpleNamespace(rotate_session=rotate_session))
    return sessions, seen


@pytest.mark.parametrize(("flag", "mode"), [(None, "pinned"), (True, "pinned"), (False, "suppressed")])
def test_activating_session_route_passes_the_flag_mode(monkeypatch, flag, mode):
    sessions, seen = _route(monkeypatch)
    body = sessions.CreateSessionRequest(bot_id="test-bot", user="test-user",
                                         include_working_memory=flag)
    assert asyncio.run(sessions.create_session(body=body)).session_id == "next"
    assert seen == [{"bot_id": "test-bot", "user_id": "test-user", "working_memory_mode": mode}]


def test_non_activating_session_refuses_suppression(monkeypatch):
    from fastapi import HTTPException

    sessions, seen = _route(monkeypatch)
    body = sessions.CreateSessionRequest(bot_id="test-bot", user="test-user",
                                         activate=False, include_working_memory=False)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(sessions.create_session(body=body))
    assert refused.value.status_code == 422
    assert seen == []
