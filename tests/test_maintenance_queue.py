"""TASK-1072: isolated SQLite publication fence and durable user routing."""

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from agent_bridge.maintenance import is_compact_command
from claude_code_bridge.active_run import ClaudeActiveRun
from llm_bawt.maintenance_delivery import MaintenanceBarrier
from llm_bawt.service.maintenance_queue import MaintenanceQueue
from llm_bawt.service.routes.chat import ChatSteerRequest, steer_active_turn
from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher


@pytest.mark.parametrize("prompt,expected", [
    ("/compact", True), (" \n/compact\t保留 details\n", True),
    ("mention /compact", False), ("/compactness", False), (None, False),
])
def test_explicit_command_only(prompt, expected):
    assert is_compact_command(prompt) == expected


@pytest.mark.parametrize("priority", ["now", "next"])
@pytest.mark.parametrize("origin", ["user", "bot"])
def test_maintenance_never_accepts_sdk_injection(priority, origin):
    client = SimpleNamespace(query=AsyncMock(), interrupt=AsyncMock())
    run = ClaudeActiveRun(client=client, request_id="compact", maintenance=True)
    with pytest.raises(RuntimeError, match="maintenance_active"):
        asyncio.run(run.steer("go ahead", priority=priority, origin=origin))
    client.query.assert_not_called()
    client.interrupt.assert_not_called()


def test_incoming_compact_requires_own_turn():
    client = SimpleNamespace(query=AsyncMock(), interrupt=AsyncMock())
    run = ClaudeActiveRun(client=client, request_id="ordinary")
    with pytest.raises(RuntimeError, match="maintenance_command_requires_turn"):
        asyncio.run(run.steer("/compact\nkeep everything"))
    client.query.assert_not_called()
    client.interrupt.assert_not_called()


@pytest.fixture
def rig():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE turn_logs (id TEXT PRIMARY KEY, bot_id TEXT, user_id TEXT, "
                          "user_prompt TEXT, created_at INTEGER, ended_at TEXT, status TEXT, "
                          "response_text TEXT, trigger_message_id TEXT, assistant_message_id TEXT)"))
        conn.execute(text("CREATE TABLE inter_bot_deliveries (id TEXT PRIMARY KEY, target_bot_id TEXT, "
                          "ordinal INTEGER, status TEXT, turn_id TEXT, metadata_json TEXT, "
                          "delivery_mode TEXT, transport_accepted_at TEXT)"))
        conn.execute(text("CREATE TABLE messages (bot_id TEXT, id TEXT, session_id TEXT)"))
        conn.execute(text("INSERT INTO turn_logs VALUES ('compact','fixture','user','/compact',1,NULL,"
                          "'streaming','compaction result','trigger','assistant')"))
        conn.execute(text("INSERT INTO messages VALUES ('fixture','trigger','original-session')"))
        conn.execute(text("INSERT INTO inter_bot_deliveries VALUES "
                          "('one','fixture',1,'QUEUED','next','{}',NULL,NULL)"))
    subscriber = SimpleNamespace(publish_tool_event=AsyncMock(return_value="1-0"))
    store = SimpleNamespace(engine=engine, config=None)
    service = SimpleNamespace(_inter_bot_dispatcher=SimpleNamespace(store=store), _redis_subscriber=subscriber)
    queue = MaintenanceQueue(service)
    yield SimpleNamespace(engine=engine, queue=queue, subscriber=subscriber)
    engine.dispose()


def head(conn):
    return conn.execute(text("SELECT * FROM inter_bot_deliveries WHERE id='one'")).mappings().one()


def blocked(rig):
    with rig.engine.begin() as conn:
        return MaintenanceBarrier(conn, "fixture").prepare(head(conn)) is not None


def finish(rig, status="ok"):
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE turn_logs SET ended_at='finished',status=:status WHERE id='compact'"), {"status": status})


@pytest.mark.parametrize("status", ["ok", "error", "aborted"])
def test_terminal_status_alone_never_releases_before_publication(rig, status):
    assert blocked(rig)
    asyncio.run(rig.queue.publish_ready("fixture"))
    rig.subscriber.publish_tool_event.assert_not_called()
    finish(rig, status)
    assert blocked(rig)

    async def publish(bot, user, event):
        assert blocked(rig), "must remain fenced until publication ACK"
        assert event["response_text"] == "compaction result"
        assert event["session_id"] == "original-session"
        assert event["status"] == {"ok": "completed", "error": "error", "aborted": "cancelled"}[status]
        return "2-0"

    rig.subscriber.publish_tool_event.side_effect = publish
    asyncio.run(rig.queue.publish_ready("fixture"))
    assert not blocked(rig)
    asyncio.run(rig.queue.publish_ready("fixture"))
    assert rig.subscriber.publish_tool_event.await_count == 1


def test_receipt_survives_process_restart_before_successor_claim(rig):
    finish(rig)
    asyncio.run(rig.queue.publish_ready("fixture"))
    restarted = MaintenanceQueue(rig.queue.service)
    asyncio.run(restarted.publish_ready("fixture"))
    assert not blocked(rig)
    rig.subscriber.publish_tool_event.assert_awaited_once()


def test_crash_before_publication_retries_without_releasing_work(rig):
    finish(rig)
    rig.subscriber.publish_tool_event.side_effect = RuntimeError("crash before XADD")
    with pytest.raises(RuntimeError, match="before XADD"):
        asyncio.run(rig.queue.publish_ready("fixture"))
    assert blocked(rig)
    rig.subscriber.publish_tool_event.side_effect = None
    restarted = MaintenanceQueue(rig.queue.service)
    asyncio.run(restarted.publish_ready("fixture"))
    assert not blocked(rig)


def test_unacknowledged_publish_leaves_work_queued_for_retry(rig):
    finish(rig)
    rig.subscriber.publish_tool_event.return_value = None
    with pytest.raises(RuntimeError, match="successor remains queued"):
        asyncio.run(rig.queue.publish_ready("fixture"))
    assert blocked(rig)
    rig.subscriber.publish_tool_event.return_value = "2-0"
    asyncio.run(rig.queue.publish_ready("fixture"))
    assert not blocked(rig)


def test_crash_after_publish_replays_snapshot_not_work(rig, monkeypatch):
    finish(rig)
    release = rig.queue._release
    monkeypatch.setattr(rig.queue, "_release", Mock(side_effect=RuntimeError("crash")))
    with pytest.raises(RuntimeError, match="crash"):
        asyncio.run(rig.queue.publish_ready("fixture"))
    assert blocked(rig)
    monkeypatch.setattr(rig.queue, "_release", release)
    asyncio.run(rig.queue.publish_ready("fixture"))
    events = [c.args[2] for c in rig.subscriber.publish_tool_event.await_args_list]
    assert [e["turn_id"] for e in events] == ["compact", "compact"]
    assert [e["assistant_message_id"] for e in events] == ["assistant", "assistant"]
    assert not blocked(rig)


def test_fifo_heads_keep_independent_publication_receipts(rig):
    with rig.engine.begin() as conn:
        conn.execute(text("INSERT INTO inter_bot_deliveries VALUES "
                          "('two','fixture',2,'QUEUED','later','{}',NULL,NULL)"))
    finish(rig)
    assert rig.queue._pending("fixture")[0] == "one"
    asyncio.run(rig.queue.publish_ready("fixture"))
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE inter_bot_deliveries SET status='DISPATCHING' WHERE id='one'"))
    assert rig.queue._pending("fixture") is None, "cannot pass an unaccepted dispatch"
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE inter_bot_deliveries SET status='DELIVERED' WHERE id='one'"))
    assert rig.queue._pending("fixture")[0] == "two"
    asyncio.run(rig.queue.publish_ready("fixture"))
    with rig.engine.begin() as conn:
        rows = conn.execute(text("SELECT * FROM inter_bot_deliveries ORDER BY ordinal")).mappings().all()
        assert [r["id"] for r in rows] == ["one", "two"]
        assert all(json.loads(r["metadata_json"])["maintenance_barrier"]["published"] for r in rows)
    events = [call.args[2] for call in rig.subscriber.publish_tool_event.await_args_list]
    assert [event["assistant_message_id"] for event in events] == ["assistant", "assistant"]


def test_new_compaction_between_publish_and_claim_is_fenced(rig):
    finish(rig)
    asyncio.run(rig.queue.publish_ready("fixture"))
    with rig.engine.begin() as conn:
        conn.execute(text("INSERT INTO turn_logs VALUES ('second','fixture','user','/compact details',"
                          "2,NULL,'streaming','','trigger','second-assistant')"))
    assert blocked(rig)


def test_arrival_stamp_survives_newer_normal_turn(rig):
    with rig.engine.begin() as conn:
        metadata = MaintenanceBarrier(conn, "fixture").stamp_enqueue({"custom": "retained"})
        conn.execute(text("UPDATE inter_bot_deliveries SET metadata_json=:metadata"), {"metadata": metadata})
        conn.execute(text("INSERT INTO turn_logs VALUES ('newer','fixture','user','normal work',"
                          "2,NULL,'streaming','','trigger','second-assistant')"))
    finish(rig)
    assert blocked(rig)
    asyncio.run(rig.queue.publish_ready("fixture"))
    assert not blocked(rig)
    with rig.engine.connect() as conn:
        assert json.loads(head(conn)["metadata_json"])["custom"] == "retained"


@pytest.mark.parametrize("mode,accepted", [("steer", None), ("steer", "accepted"), ("turn", "accepted")])
def test_existing_accepted_or_ambiguous_work_keeps_recovery_path(rig, mode, accepted):
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE inter_bot_deliveries SET delivery_mode=:mode, transport_accepted_at=:accepted"),
                     {"mode": mode, "accepted": accepted})
    assert not blocked(rig)
    assert InterBotDeliveryDispatcher._steer_failure_action("maintenance_active", accepted=True) == "retry_same_steer"
    assert InterBotDeliveryDispatcher._steer_failure_action("maintenance_active", accepted=False) == "fallback"


@pytest.mark.parametrize("case,status", [
    ("queue_unavailable", 503), ("unsupported_backend", 409),
    ("missing_source_session", 503), ("invalid_uuid", 422), ("empty_text", 422),
])
def test_unsupported_user_submission_fails_visibly_before_acceptance(monkeypatch, case, status):
    from fastapi import HTTPException

    store = SimpleNamespace(engine=None if case == "queue_unavailable" else True, enqueue=Mock())
    client = SimpleNamespace(get_message_by_id=lambda _: (
        {} if case == "missing_source_session" else {"session_id": "original"}))
    service = SimpleNamespace(config=None, get_memory_client=lambda *a: client,
                              _inter_bot_dispatcher=SimpleNamespace(store=store))
    monkeypatch.setattr("llm_bawt.bots.BotManager", lambda _: SimpleNamespace(
        get_bot=lambda _: SimpleNamespace(agent_backend=(
            "unsupported" if case == "unsupported_backend" else "claude-code"))))
    request = SimpleNamespace(message="  " if case == "empty_text" else "keep this text",
                              message_id="bad" if case == "invalid_uuid" else str(uuid.uuid4()))
    turn = SimpleNamespace(bot_id="fixture", user_id="user", trigger_message_id="trigger")
    with pytest.raises(HTTPException) as error:
        asyncio.run(MaintenanceQueue(service).enqueue_user(request, turn))
    assert error.value.status_code == status
    assert error.value.detail
    store.enqueue.assert_not_called()


@pytest.mark.parametrize("priority", ["now", "next"])
@pytest.mark.parametrize("history_fails", [False, True])
@pytest.mark.parametrize("prompt,message", [("/compact", "go ahead\nKeep IDs"), ("normal work", "/compact\nhuge summary  ")])
def test_user_route_preserves_uuid_session_bytes_and_retry(monkeypatch, priority, history_fails, prompt, message):
    payloads, records, messages = {}, {}, {}
    original_id = str(uuid.uuid4())
    turn = SimpleNamespace(id="source-turn", bot_id="fixture", user_id="user", user_prompt=prompt,
                           ended_at=None, status="streaming", trigger_message_id="trigger")

    def enqueue(**kwargs):
        key = kwargs["idempotency_key"]
        if key in records:
            return records[key], True
        record = SimpleNamespace(id="delivery", turn_id="successor", user_message_id=kwargs["user_message_id"])
        records[key] = record
        payloads[record.id] = kwargs["payload"]
        assert kwargs["author"].entity_type == "user"
        return record, False

    client = SimpleNamespace(get_message_by_id=lambda _: {"session_id": "original-session"},
                             add_message=lambda role, content, session, _, mid, *a: messages.update({mid: (content, session)}))
    store = SimpleNamespace(engine=True, enqueue=Mock(side_effect=enqueue), payload=payloads.get)
    dispatcher = SimpleNamespace(store=store, wake=Mock(), _emit=AsyncMock())
    service = SimpleNamespace(_inter_bot_dispatcher=dispatcher, config=None,
                              _turn_log_store=SimpleNamespace(get_turn=lambda _: turn),
                              get_memory_client=lambda *a: client)
    monkeypatch.setattr("llm_bawt.bots.BotManager", lambda _: SimpleNamespace(
        get_bot=lambda _: SimpleNamespace(agent_backend="claude-code")))
    request = ChatSteerRequest(turn_id=turn.id, bot_id="fixture", user_id="user", message_id=original_id,
                               message=message, priority=priority)
    persist = client.add_message
    if history_fails:
        client.add_message = Mock(side_effect=RuntimeError("history temporarily unavailable"))
    first = asyncio.run(steer_active_turn(service, request))
    assert first.persisted is not history_fails
    client.add_message = persist
    turn.ended_at, turn.status = "ended", "ok"
    request.message = "/compact\nchanged retry must not replace original text"
    second = asyncio.run(steer_active_turn(service, request))
    assert first.queued and second.queued
    assert first.message_id == second.message_id == original_id
    assert first.turn_id == second.turn_id == "successor"
    assert len(records) == len(messages) == 1
    assert payloads["delivery"]["messages"] == [{"role": "user", "content": message}]
    assert payloads["delivery"]["prefer_steer"] is False
    assert messages[original_id] == (message, "original-session")
    assert not first.can_escalate and not first.can_cancel
    dispatcher._emit.assert_awaited_once()
