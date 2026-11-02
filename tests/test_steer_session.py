"""Durable composer intent is fenced before bridge or maintenance acceptance."""
import asyncio
from copy import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from llm_bawt.service.routes import chat
from llm_bawt.service.steer_session import owned_turn_sessions


@pytest.fixture
def rig(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sessions (id TEXT,bot_id TEXT,user_id TEXT,status TEXT)")
        conn.exec_driver_sql("CREATE TABLE messages (id TEXT,bot_id TEXT,session_id TEXT)")
        conn.exec_driver_sql("CREATE TABLE turn_logs (id TEXT,bot_id TEXT,user_id TEXT,trigger_message_id TEXT)")
        conn.exec_driver_sql("INSERT INTO sessions VALUES ('durable-A','target','nick','archived')")
        conn.exec_driver_sql("INSERT INTO messages VALUES ('trigger','target','durable-A')")
        conn.exec_driver_sql("INSERT INTO turn_logs VALUES ('turn','target','nick','trigger')")
    turn = SimpleNamespace(id="turn", bot_id="target", user_id="nick", trigger_message_id="trigger",
                           status="streaming", ended_at=None, user_prompt="working",
                           agent_session_key="target:nick", agent_request_id="req-A",
                           agent_backend="claude-code", agent_steer_caps="next")
    memory = SimpleNamespace(add_message=Mock())
    service = SimpleNamespace(config=object(), get_memory_client=lambda *_: memory,
                              _turn_log_store=SimpleNamespace(engine=engine, get_turn=lambda _: copy(turn)))
    subscriber = SimpleNamespace(send_steer=AsyncMock(return_value={"ok": True}))
    monkeypatch.setattr("llm_bawt.agent_backends.agent_bridge.get_agent_subscriber", lambda: subscriber)
    monkeypatch.setattr(chat, "_steer_framing", lambda *_a, **_kw: {})
    timeline = SimpleNamespace(content_offset=AsyncMock(return_value=None), publish=AsyncMock())
    monkeypatch.setattr("llm_bawt.service.steer_timeline.SteerTimeline", lambda *_a, **_kw: timeline)
    from llm_bawt.service.maintenance_queue import MaintenanceQueue
    enqueue_user = MaintenanceQueue.enqueue_user
    queue = AsyncMock(return_value=dict(ok=True, detail="queued", turn_id="queued", message_id="message",
                                       session_id="durable-A"))
    monkeypatch.setattr("llm_bawt.service.maintenance_queue.MaintenanceQueue.enqueue_user", queue)
    yield SimpleNamespace(engine=engine, turn=turn, service=service, memory=memory, sub=subscriber,
                          queue=queue, enqueue_user=enqueue_user)
    engine.dispose()


def request(**kw):
    return chat.ChatSteerRequest(**(dict(turn_id="turn", bot_id="target", user_id="nick",
                                       message_id="message", message="correct", session_id="durable-A") | kw))


def run(rig, **kw):
    return asyncio.run(chat.steer_active_turn(rig.service, request(**kw)))


@pytest.mark.parametrize("explicit", [True, False])
def test_same_session_and_omitted_intent_return_canonical_durable_session(rig, explicit):
    result = run(rig, session_id="durable-A" if explicit else None)
    assert result.ok and result.persisted and result.session_id == "durable-A"
    assert rig.sub.send_steer.await_args.kwargs["session_key"] == "target:nick"
    assert rig.sub.send_steer.await_args.kwargs["target_request_id"] == "req-A"
    assert rig.memory.add_message.call_args.args[2] == "durable-A"


@pytest.mark.parametrize("maintenance", [False, True])
def test_cross_session_is_definitive_before_any_effect(rig, maintenance):
    with pytest.raises(HTTPException) as caught:
        run(rig, session_id="other-thread", message="/compact" if maintenance else "correct")
    assert (caught.value.status_code, caught.value.detail) == (409, "session_mismatch")
    rig.sub.send_steer.assert_not_awaited()
    rig.queue.assert_not_awaited()
    rig.memory.add_message.assert_not_called()


@pytest.mark.parametrize("change", ["missing_trigger", "missing_session", "foreign_bot", "foreign_user", "deleted"])
def test_unowned_or_missing_trigger_session_fails_closed(rig, change):
    with rig.engine.begin() as conn:
        sql = {"missing_trigger": "DELETE FROM messages", "missing_session": "DELETE FROM sessions",
               "foreign_bot": "UPDATE sessions SET bot_id='foreign'",
               "foreign_user": "UPDATE sessions SET user_id='foreign'",
               "deleted": "UPDATE sessions SET status='deleted'"}[change]
        conn.exec_driver_sql(sql)
    with pytest.raises(HTTPException) as caught:
        run(rig)
    assert (caught.value.status_code, caught.value.detail) == (409, "session_unavailable")
    rig.sub.send_steer.assert_not_awaited()
    rig.memory.add_message.assert_not_called()


@pytest.mark.parametrize("change", ["ended", "request", "trigger", "durable_session", "owner"])
def test_async_preparation_race_cannot_retarget_or_dispatch(rig, monkeypatch, change):
    def framing(*_a, **_kw):
        if change == "ended":
            rig.turn.ended_at, rig.turn.status = "ended", "ok"
        elif change == "request":
            rig.turn.agent_request_id = "req-B"
        elif change == "trigger":
            rig.turn.trigger_message_id = "new-trigger"
        elif change == "owner":
            rig.turn.user_id = "someone-else"
        else:
            with rig.engine.begin() as conn:
                conn.exec_driver_sql("INSERT INTO sessions VALUES ('durable-B','target','nick','active')")
                conn.exec_driver_sql("UPDATE messages SET session_id='durable-B'")
        return {}
    monkeypatch.setattr(chat, "_steer_framing", framing)
    with pytest.raises(HTTPException) as caught:
        run(rig)
    assert caught.value.status_code == 409
    rig.sub.send_steer.assert_not_awaited()
    rig.memory.add_message.assert_not_called()


def test_accepted_persistence_does_not_reread_a_new_destination(rig):
    async def accepted(**_):
        with rig.engine.begin() as conn:
            conn.exec_driver_sql("UPDATE messages SET session_id='changed-after-acceptance'")
        return {"ok": True}

    rig.sub.send_steer.side_effect = accepted
    result = run(rig)
    assert result.persisted and result.session_id == "durable-A"
    assert rig.memory.add_message.call_args.args[2] == "durable-A"


def test_no_active_retry_revalidates_original_target_before_second_rpc(rig, monkeypatch):
    async def ended(**_):
        rig.turn.ended_at, rig.turn.status = "ended", "ok"
        return {"ok": False, "error": "no_active_run"}

    rig.sub.send_steer.side_effect = ended
    monkeypatch.setattr(chat.asyncio, "sleep", AsyncMock())
    with pytest.raises(HTTPException, match="Turn is no longer active"):
        run(rig)
    assert rig.sub.send_steer.await_count == 1
    rig.memory.add_message.assert_not_called()


def test_already_ended_public_target_is_refused_without_rpc(rig):
    rig.turn.ended_at = "ended"
    with pytest.raises(HTTPException, match="Turn is no longer active"):
        run(rig)
    rig.sub.send_steer.assert_not_awaited()


def test_successful_maintenance_ack_contains_canonical_session(rig):
    result = run(rig, message="/compact")
    assert result.session_id == "durable-A" and result.turn_id == "queued"
    rig.queue.assert_awaited_once()
    rig.sub.send_steer.assert_not_awaited()


def test_active_session_enrichment_does_not_drop_unknown_or_stale_targets(rig):
    assert owned_turn_sessions(rig.engine, [rig.turn]) == {"turn": "durable-A"}
    stale = copy(rig.turn)
    stale.trigger_message_id = "other"
    assert owned_turn_sessions(rig.engine, [stale]) == {}
    with rig.engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM sessions")
    assert owned_turn_sessions(rig.engine, [rig.turn]) == {}


def test_lookup_failure_is_unavailable_not_a_dispatch_retry(rig):
    with rig.engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE sessions")
    with pytest.raises(HTTPException) as caught:
        run(rig)
    assert caught.value.status_code == 503
    rig.sub.send_steer.assert_not_awaited()
    rig.memory.add_message.assert_not_called()


@pytest.mark.parametrize("change", [None, "selected", "binding"])
def test_real_maintenance_enqueue_uses_and_revalidates_owned_destination(rig, monkeypatch, change):
    import uuid
    from llm_bawt.service.maintenance_queue import MaintenanceQueue

    payloads = {}

    def enqueue(**kwargs):
        payloads.update(kwargs)
        return SimpleNamespace(id="delivery", turn_id="successor", user_message_id=kwargs["user_message_id"]), False

    store = SimpleNamespace(engine=rig.engine, enqueue=Mock(side_effect=enqueue), payload=lambda _: payloads["payload"])
    rig.service._inter_bot_dispatcher = SimpleNamespace(store=store, wake=Mock(), _emit=AsyncMock())
    monkeypatch.setattr("llm_bawt.bots.BotManager", lambda _: SimpleNamespace(
        get_bot=lambda _: SimpleNamespace(agent_backend="claude-code")))
    req = request(message_id=str(uuid.uuid4()), message="/compact", session_id="other" if change == "selected" else "durable-A")
    if change == "binding":
        def memory(*_):
            with rig.engine.begin() as conn:
                conn.exec_driver_sql("UPDATE sessions SET user_id='foreign'")
            return rig.memory
        rig.service.get_memory_client = memory
    invoke = rig.enqueue_user(MaintenanceQueue(rig.service), req, copy(rig.turn))
    if change:
        with pytest.raises(HTTPException) as caught:
            asyncio.run(invoke)
        assert caught.value.status_code == 409
        store.enqueue.assert_not_called()
        rig.memory.add_message.assert_not_called()
    else:
        ack = asyncio.run(invoke)
        assert ack["session_id"] == payloads["payload"]["session_id"] == "durable-A"
        assert payloads["user_message_id"] == req.message_id
        assert payloads["payload"]["prefer_steer"] is False
        assert rig.memory.add_message.call_args.args[2] == "durable-A"
    rig.sub.send_steer.assert_not_awaited()


@pytest.mark.parametrize("lookup_fails", [False, True])
def test_active_list_keeps_bot_wide_busy_rows_with_nullable_owned_session(rig, monkeypatch, lookup_fails):
    from datetime import datetime, timezone
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from llm_bawt.service.routes import turn_logs
    from llm_bawt.service.turn_logs import TurnLog

    rows = [TurnLog(id=tid, bot_id="target", user_id="nick", path="/test", stream=True,
                    created_at=datetime.now(timezone.utc), status="streaming", user_prompt="working",
                    response_text="", trigger_message_id="trigger" if tid == "turn" else "unknown",
                    agent_session_key="target:nick") for tid in ("turn", "unknown-turn")]
    listing = Mock(return_value=(rows, 2))
    store = SimpleNamespace(engine=rig.engine, list_turns=listing,
                            changed_files_summaries_for_turns=lambda _: {})
    monkeypatch.setattr(turn_logs, "get_service", lambda: rig.service)
    monkeypatch.setattr(turn_logs, "get_turn_log_store", lambda: store)
    monkeypatch.setattr(turn_logs, "_live_tool_call_counts", lambda *_: {})
    monkeypatch.setattr(turn_logs, "_in_flight_steer_caps", lambda *_: {})
    if lookup_fails:
        with rig.engine.begin() as conn:
            conn.exec_driver_sql("DROP TABLE sessions")
    app = FastAPI()
    app.include_router(turn_logs.router)
    with TestClient(app) as client:
        response = client.get("/v1/turn-logs?bot_id=target&active_only=true&session_id=selected-other")
    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 2
    assert [(row["id"], row["session_id"]) for row in body["turns"]] == [
        ("turn", None if lookup_fails else "durable-A"), ("unknown-turn", None),
    ]
    assert listing.call_args.kwargs["bot_id"] == "target"
    assert listing.call_args.kwargs["active_only"] is True
    assert "session_id" not in listing.call_args.kwargs
