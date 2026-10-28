"""Isolated dispatcher ordering/error boundaries, with no live bot or database."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from llm_bawt.inter_bot_delivery import DeliveryRecord, InterBotDeliveryStore
from tests.test_inter_bot_dispatcher_supervision import _make_dispatcher


@pytest.mark.asyncio
@pytest.mark.parametrize("publication_fails", [False, True])
async def test_publication_finishes_before_claim_or_dispatch(monkeypatch, publication_fails):
    monkeypatch.setattr("llm_bawt.bots.BotManager", lambda _: SimpleNamespace(
        get_bot=lambda _: SimpleNamespace(agent_backend="claude-code")))
    entered, release = asyncio.Event(), asyncio.Event()
    order = []

    async def publish(_target):
        entered.set()
        await release.wait()
        if publication_fails:
            raise RuntimeError("unconfirmed")
        order.append("published")

    # Bind the fake as a normal method, retaining its explicit queue owner.
    async def bound_publish(_self, target):
        return await publish(target)
    monkeypatch.setattr("llm_bawt.service.maintenance_queue.MaintenanceQueue.publish_ready", bound_publish)
    dispatcher = _make_dispatcher()
    record = SimpleNamespace(id="one", status="DISPATCHING")
    dispatcher.store.claim_next = Mock(side_effect=lambda *a, **k: (order.append("claimed"), record)[1])
    dispatcher._emit = AsyncMock()
    dispatcher._dispatch = AsyncMock(side_effect=lambda _: order.append("dispatched"))
    task = asyncio.create_task(dispatcher._drain_target("fixture"))
    await asyncio.wait_for(entered.wait(), 1)
    dispatcher.store.claim_next.assert_not_called()
    dispatcher._dispatch.assert_not_called()
    release.set()
    if publication_fails:
        with pytest.raises(RuntimeError, match="unconfirmed"):
            await task
        dispatcher.store.claim_next.assert_not_called()
    else:
        assert await task is True
        await asyncio.gather(*dispatcher._dispatch_tasks)
        assert order == ["published", "claimed", "dispatched"]
        dispatcher._dispatch.assert_awaited_once_with(record)


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [False, True, "failure", "cancelled"])
async def test_only_actual_progress_immediately_wakes_dispatcher(result):
    dispatcher = _make_dispatcher()

    async def drain():
        if result == "failure":
            raise RuntimeError("publication unavailable")
        if result == "cancelled":
            raise asyncio.CancelledError()
        return result

    task = asyncio.create_task(drain())
    await asyncio.gather(task, return_exceptions=True)
    dispatcher._target_tasks["fixture"] = task
    dispatcher._target_done("fixture", task)
    assert not dispatcher._target_tasks
    assert dispatcher._wake_event.is_set() is (result is True)


@pytest.mark.asyncio
async def test_lifecycle_failure_does_not_reject_durable_acceptance():
    dispatcher = _make_dispatcher()
    dispatcher.service._redis_subscriber.publish_tool_event = AsyncMock(side_effect=RuntimeError("offline"))
    record = Mock(metadata={"maintenance_user_queue": True}, status="QUEUED",
                  target_bot_id="fixture", author_entity_id="user", user_message_id="message")
    await dispatcher._emit(record)
    dispatcher.service._redis_subscriber.publish_tool_event.assert_awaited_once()


def test_enqueue_snapshot_does_not_invert_scheduler_target_job_lock_order(monkeypatch):
    conn = MagicMock()
    def execute(sql, params):
        result = MagicMock()
        result.mappings.return_value.first.return_value = (
            {"id": "delivery"} if "INSERT INTO inter_bot_deliveries" in str(sql) else None
        )
        return result

    conn.execute.side_effect = execute
    engine = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    store = InterBotDeliveryStore.__new__(InterBotDeliveryStore)
    store.engine = engine
    monkeypatch.setattr(DeliveryRecord, "from_mapping", Mock(return_value="record"))
    frontend_id = "7efbf344-ee7d-44cd-852c-4cc9cbe97a61"
    assert store.enqueue(sender_bot_id="fixture-sender", target_bot_id="fixture",
                         message="work", payload={}, idempotency_key="key",
                         user_message_id=frontend_id) == ("record", False)
    statements = [str(call.args[0]) for call in conn.execute.call_args_list]
    locks = [sql for sql in statements if "pg_advisory" in sql]
    assert locks == ["SELECT pg_advisory_xact_lock(:namespace, :hashed)"] * 2
    pairs = [call.args[1] for call in conn.execute.call_args_list
             if "pg_advisory" in str(call.args[0])]
    assert pairs == sorted(pairs, key=lambda pair: (pair["namespace"], pair["hashed"]))
    assert any("SELECT id, user_prompt FROM turn_logs" in sql for sql in statements)
    inserted = next(call.args[1] for call in conn.execute.call_args_list
                    if "INSERT INTO inter_bot_deliveries" in str(call.args[0]))
    assert json.loads(inserted["metadata"])["_request_identity_producer"] == "bot"
    assert inserted["message_id"] == frontend_id
    assert json.loads(inserted["payload"])["user_message_id"] == frontend_id


@pytest.mark.asyncio
async def test_archived_source_dispatch_binds_original_thread_without_activation():
    from llm_bawt.agent_context import rotate_delivery_session
    from llm_bawt.service.session_lifecycle.bridge_reset import BridgeSessionResetMixin

    dispatcher = _make_dispatcher()
    frontend_id = "7efbf344-ee7d-44cd-852c-4cc9cbe97a61"
    payload = {"bot_id": "fixture", "user": "user", "session_id": "original-session",
               "messages": [{"role": "user", "content": "queued text"}],
               "user_message_id": frontend_id, "prefer_steer": False}
    conn = Mock()
    payload, old_session, new_session = rotate_delivery_session(
        conn, head={"session_policy": "continue"}, payload=payload,
        target_bot_id="fixture", user_id="user")
    conn.execute.assert_not_called()
    assert old_session is new_session is None
    record = SimpleNamespace(id="one", turn_id="successor", claim_token="claim")
    dispatcher.store.payload = Mock(return_value=payload)
    dispatcher.store.mark_transport_accepted = Mock(return_value=True)
    dispatcher.store.mark_delivered = Mock(return_value=None)
    dispatcher._heartbeat = AsyncMock()
    dispatcher.service._turn_log_store = SimpleNamespace(get_turn=lambda _: SimpleNamespace(
        ended_at="finished", status="ok", error_text=None, response_text="done", model="selected"))
    db = SimpleNamespace(
        get_session=Mock(return_value={"archived_at": "after-new",
            "session_metadata": {"agent_session_keys": {"claude_code": "original-sdk"}}}),
        get_active_session=Mock(side_effect=AssertionError("must not resolve the new active thread")),
        get_or_create_active_session=Mock(side_effect=AssertionError("must not activate a thread")),
    )
    llm = SimpleNamespace(bot=SimpleNamespace(agent_backend="claude-code", slug="fixture"),
                          user_id="user", client=SimpleNamespace(_bot_config={}),
                          history_manager=SimpleNamespace(_db_backend=db))
    seen = []

    async def stream(request):
        binding = BridgeSessionResetMixin()._bind_agent_thread(llm, request)
        assert binding == {"thread_session_id": "original-session",
                           "thread_resume_id": "original-sdk", "explicit_thread": True}
        assert request.user_message_id == frontend_id
        assert request.messages[-1].content == "queued text"
        seen.append(request.session_id)
        yield "done"

    dispatcher.service.chat_completion_stream = stream
    await dispatcher._dispatch(record)
    assert seen == ["original-session"]
    db.get_session.assert_called_once_with("original-session")
    db.get_active_session.assert_not_called()
    db.get_or_create_active_session.assert_not_called()
    dispatcher.store.mark_delivered.assert_called_once()
