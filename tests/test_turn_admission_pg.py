"""PostgreSQL admission lock and reservation fencing integration tests."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage
from llm_bawt.service.turn_admission import AdmissionClaim, AdmissionFenceLost, TargetBusy
from tests.pg_isolation import isolated_pg_store
from tests.pg_lock_wait import wait_blocked_by
from tests.test_turn_admission import _mcp_claim

pytestmark = pytest.mark.usefixtures("private_app_pg")


# ---------------------------------------------------------------------------
# PostgreSQL: the real lock, classification and reservation fence
#
# The shared guard refuses a non-*_test database BEFORE any store construction.
# A skipped integration test is not evidence of passing PostgreSQL behavior.
# ---------------------------------------------------------------------------


@pytest.fixture
def pg_turns():
    from llm_bawt.service.turn_logs import TurnLogStore

    store = isolated_pg_store(TurnLogStore)
    if store.engine is None:
        pytest.skip("PostgreSQL unavailable")
    bot = f"test-admission-{uuid.uuid4().hex}"
    yield store, bot
    with store.engine.begin() as conn:
        conn.execute(sa_text("DELETE FROM turn_logs WHERE bot_id=:bot"), {"bot": bot})


def _open(store, bot, status="streaming", turn_id=None):
    turn_id = turn_id or f"turn-{uuid.uuid4().hex}"
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id,
                                   status, user_prompt, response_text, ended_at)
            VALUES (:id, now(), '/test', true, :bot, 'nick', :status, '', '', NULL)
        """), {"id": turn_id, "bot": bot, "status": status})
    return turn_id


def _owner_claim(owner="owner-a", fence=None):
    return AdmissionClaim(owner=owner, fence=fence or (lambda conn: True))


def _save(store, bot, *, turn_id=None, admission=None, claim=None, connection=None):
    payload = {"messages": []}
    if admission:
        payload["admission"] = admission
    store.save_turn(
        turn_id=turn_id or f"turn-{uuid.uuid4().hex}", request_id="r",
        path="/v1/chat/completions", stream=True, model="m", bot_id=bot,
        user_id="nick", status="streaming", latency_ms=None, user_prompt="hi",
        request_payload=payload, response_text="", tool_calls=None,
        admission_claim=claim, connection=connection,
    )


def _reserve(store, bot, turn_id, *, owner="owner-a", fence=None):
    return store.reserve_idle_turn(
        turn_id=turn_id, bot_id=bot, user_id="nick", user_prompt="continuation",
        request_json={"admission": "idle_only"}, trigger_message_id=str(uuid.uuid4()),
        owner=owner, fence=fence,
    )


def _assert_waiting_on_holder(engine, holding_conn, worker, *, timeout=5.0):
    """Observe a real PG lock wait, not merely a thread that has not run yet."""
    holder_pid = holding_conn.execute(sa_text("SELECT pg_backend_pid()")).scalar_one()
    wait_blocked_by(engine, holder_pid, worker, timeout=timeout)


@pytest.mark.integration
def test_ordinary_agent_starts_enqueue_atomically_and_claim_fifo(pg_turns):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    from llm_bawt.service.ordinary_turn_queue import OrdinaryTurnQueue

    turns, bot = pg_turns
    deliveries = InterBotDeliveryStore(pg_turns[0].config)
    queue = OrdinaryTurnQueue(turns, deliveries)

    def send(message, *, message_id=None, assistant_id=None):
        request = ChatCompletionRequest(
            bot_id=bot, user="nick", model="test-model", stream=True,
            messages=[ChatMessage(role="user", content=message)],
            user_message_id=message_id or str(uuid.uuid4()),
            assistant_message_id=assistant_id or str(uuid.uuid4()),
            augment_memory=False, extract_memory=False,
        )
        result = queue.decide(
            request=request, bot_id=bot, user_id="nick", user_prompt=message,
        )
        return request, result

    try:
        first_req, first = send("first")
        assert first.state == "queued" and first.delivery_id
        assert turns.get_turn(first.turn_id) is None  # HTTP admission never opens a turn
        _, active_retry = send("first", message_id=first_req.user_message_id,
                               assistant_id=first_req.assistant_message_id)
        assert active_retry.delivery_id == first.delivery_id
        second_assistant = str(uuid.uuid4())
        second_req, second = send("second", assistant_id=second_assistant)
        assert second.state == "queued" and second.delivery_id
        assert turns.get_turn(second.turn_id) is None
        payload = deliveries.payload(second.delivery_id)
        assert payload["messages"] == [{"role": "user", "content": "second"}]
        assert payload["prefer_steer"] is False
        assert payload["assistant_message_id"] == second_assistant
        assert payload["augment_memory"] is False and payload["extract_memory"] is False
        _, duplicate = send("second", message_id=second_req.user_message_id,
                            assistant_id=second_assistant)
        assert duplicate.delivery_id == second.delivery_id
        from llm_bawt.request_identity import IdentityConflict
        with pytest.raises(IdentityConflict):
            send("changed", message_id=second_req.user_message_id,
                 assistant_id=second_assistant)
        claimed_first = deliveries.claim_next(bot, claim_owner="worker", steer_capable=True)
        assert claimed_first and claimed_first.id == first.delivery_id
        assert claimed_first.status == "DISPATCHING" and claimed_first.turn_id == first.turn_id
        assert turns.get_turn(first.turn_id).status == "reserved"
        assert deliveries.claim_next(bot, claim_owner="other", steer_capable=True) is None
        with turns.engine.begin() as conn:
            conn.execute(sa_text("UPDATE turn_logs SET status='ok', ended_at=now() WHERE id=:id"),
                         {"id": first.turn_id})
        deliveries.mark_delivered(first.delivery_id, claimed_first.claim_token,
                                  response_model=None, response_chars=0)
        _, terminal_retry = send("first", message_id=first_req.user_message_id,
                                 assistant_id=first_req.assistant_message_id)
        assert terminal_retry.delivery_id == first.delivery_id
        assert terminal_retry.receipt()["status"] == "DELIVERED"
        claimed = deliveries.claim_next(bot, claim_owner="worker", steer_capable=True)
        assert claimed and claimed.id == second.delivery_id
        assert claimed.status == "DISPATCHING" and claimed.turn_id == second.turn_id
        assert first_req.user_message_id == first.user_message_id
    finally:
        with deliveries.engine.begin() as conn:
            conn.execute(sa_text("DELETE FROM inter_bot_deliveries WHERE target_bot_id=:bot"),
                         {"bot": bot})


@pytest.mark.integration
def test_idle_http_ack_survives_dispatcher_crash_before_bridge_send(pg_turns):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    from llm_bawt.service.ordinary_turn_queue import OrdinaryTurnQueue

    turns, bot = pg_turns
    deliveries = InterBotDeliveryStore(pg_turns[0].config)
    request = ChatCompletionRequest(
        bot_id=bot, user="nick", model="test-model", stream=True,
        messages=[ChatMessage(role="user", content="recover my exact reply")],
        user_message_id=str(uuid.uuid4()), assistant_message_id=str(uuid.uuid4()),
    )
    message_id = request.user_message_id
    try:
        decision = OrdinaryTurnQueue(turns, deliveries).decide(
            request=request, bot_id=bot, user_id="nick", user_prompt="recover my exact reply",
        )
        assert decision.receipt()["object"] == "chat.completion.queued"
        assert turns.get_turn(decision.turn_id) is None
        first = deliveries.claim_next(bot, claim_owner="dead-worker", steer_capable=True)
        assert first and first.id == decision.delivery_id
        assert deliveries.mark_transport_accepted(first.id, first.claim_token)
        original = deliveries.payload(first.id)
        with turns.engine.begin() as conn:
            conn.execute(sa_text("""
                UPDATE inter_bot_deliveries SET lease_expires_at=now() - interval '1 second'
                WHERE id=:id
            """), {"id": first.id})
        recovered = deliveries.recover_expired()
        assert [row.id for row in recovered if row.id == first.id] == [first.id]
        assert deliveries.get(first.id).status == "QUEUED"
        second = deliveries.claim_next(bot, claim_owner="recovery-worker", steer_capable=True)
        assert second and second.id == first.id and second.turn_id == first.turn_id
        assert second.claim_token != first.claim_token
        assert deliveries.payload(second.id) == original
        identity = original["inter_bot_bridge_request_id"]
        assert identity.startswith("req_delivery_")
        assert not deliveries.validate_claim(
            delivery_id=first.id, claim_token=first.claim_token,
            target_bot_id=bot, turn_id=first.turn_id,
            user_message_id=message_id, bridge_request_id=identity,
        )
        assert deliveries.validate_claim(
            delivery_id=second.id, claim_token=second.claim_token,
            target_bot_id=bot, turn_id=second.turn_id,
            user_message_id=message_id, bridge_request_id=identity,
        )
        with turns.engine.connect() as conn:
            assert conn.execute(sa_text(
                "SELECT count(*) FROM inter_bot_deliveries WHERE target_bot_id=:bot"
            ), {"bot": bot}).scalar_one() == 1
    finally:
        with deliveries.engine.begin() as conn:
            conn.execute(sa_text("DELETE FROM inter_bot_deliveries WHERE target_bot_id=:bot"),
                         {"bot": bot})


@pytest.mark.integration
def test_terminal_takeover_blocks_bridge_binding_under_row_lock(pg_turns):
    from llm_bawt.service.turn_execution import TurnExecution, TurnExecutionCancelled

    turns, bot = pg_turns
    turn_id = _open(turns, bot)
    owner = TurnExecution(turn_id, threading.Event(), is_agent=True, store=turns)
    errors = []

    def bind():
        try:
            owner.bind_bridge(backend="claude-code", session_key=f"{bot}:nick", request_id="req-A")
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=bind)
    with turns.engine.begin() as holder:
        holder.execute(sa_text("SELECT id FROM turn_logs WHERE id=:id FOR UPDATE"), {"id": turn_id})
        worker.start()
        _assert_waiting_on_holder(turns.engine, holder, worker)
        holder.execute(sa_text("""
            UPDATE turn_logs SET status='error', ended_at=now() WHERE id=:id
        """), {"id": turn_id})
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], TurnExecutionCancelled)
    assert turns.get_turn(turn_id).agent_request_id is None


@pytest.mark.integration
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
def test_ordinary_agent_new_enqueues_without_http_execution(pg_turns, monkeypatch, stream, busy):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    from llm_bawt.service.ordinary_turn_queue import OrdinaryTurnQueue
    from tests.test_nonstream_turn_execution import _service

    turns, bot = pg_turns
    deliveries = InterBotDeliveryStore(pg_turns[0].config)
    running = _open(turns, bot) if busy else None
    service = _service(Mock(side_effect=AssertionError("queued request reached inference")))
    service._turn_log_store = turns
    service._ordinary_turn_queue = OrdinaryTurnQueue(turns, deliveries)
    service._inter_bot_dispatcher = SimpleNamespace(
        store=deliveries, announce_admitted=Mock(), wake=Mock(),
    )
    message_id, assistant_id = str(uuid.uuid4()), str(uuid.uuid4())
    request = ChatCompletionRequest(
        bot_id=bot, user="nick", model="test-model", stream=stream,
        messages=[ChatMessage(
            role="user", content="/new" if stream else [{"type": "text", "text": "/new"}],
        )],
        user_message_id=message_id, assistant_message_id=assistant_id,
    )
    if stream:
        monkeypatch.setattr(
            "llm_bawt.service.chat_streaming.prepare_stream_request_attachments",
            AsyncMock(return_value=("/new", [], [], None)),
        )
    try:
        if stream:
            async def consume():
                return [chunk async for chunk in service.chat_completion_stream(request)]
            chunks = asyncio.run(consume())
            payloads = [json.loads(chunk[6:]) for chunk in chunks if chunk.startswith("data: {")]
            assert chunks[-1] == "data: [DONE]\n\n"
            receipt = payloads[0]
        else:
            receipt = asyncio.run(service.chat_completion(request))
        assert receipt["object"] == "chat.completion.queued"
        assert receipt["user_message_id"] == message_id
        assert receipt["assistant_message_id"] == assistant_id
        assert deliveries.get(receipt["delivery_id"]).status == "QUEUED"
        active = turns.active_turn_for_bot(bot)
        assert (active.id if active else None) == running
        assert turns.get_turn(receipt["turn_id"]) is None
        service._persist_turn_log.assert_not_called()
        service._maybe_summarize_on_new.assert_not_called()
        service._maybe_rotate_agent_session.assert_not_called()
        service._inter_bot_dispatcher.announce_admitted.assert_called_once()
        assert service._inter_bot_dispatcher.announce_admitted.call_args.args[0].id == receipt["delivery_id"]
        service._inter_bot_dispatcher.wake.assert_not_called()
    finally:
        with deliveries.engine.begin() as conn:
            conn.execute(sa_text("DELETE FROM inter_bot_deliveries WHERE target_bot_id=:bot"),
                         {"bot": bot})


@pytest.mark.integration
def test_ordinary_start_can_commit_or_rollback_with_its_admission_connection(pg_turns):
    from llm_bawt.service.turn_admission import lock_target

    store, bot = pg_turns
    turn_id = f"turn-{uuid.uuid4().hex}"
    with pytest.raises(RuntimeError, match="roll back start"):
        with store.engine.begin() as conn:
            lock_target(conn, bot)
            _save(store, bot, turn_id=turn_id, connection=conn)
            assert conn.execute(sa_text(
                "SELECT status FROM turn_logs WHERE id=:id"
            ), {"id": turn_id}).scalar_one() == "streaming"
            raise RuntimeError("roll back start")
    assert store.get_turn(turn_id) is None
    with store.engine.begin() as conn:
        lock_target(conn, bot)
        _save(store, bot, turn_id=turn_id, connection=conn)
    assert store.get_turn(turn_id).status == "streaming"


@pytest.mark.integration
def test_ordinary_start_first_makes_continuation_reservation_lose(pg_turns):
    store, bot = pg_turns
    running = _open(store, bot)
    with pytest.raises(TargetBusy, match=running):
        _reserve(store, bot, f"turn-approval-{uuid.uuid4().hex}")


@pytest.mark.integration
def test_continuation_reservation_first_makes_ordinary_start_lose(pg_turns):
    store, bot = pg_turns
    reserved = f"turn-approval-{uuid.uuid4().hex}"
    _reserve(store, bot, reserved)
    with pytest.raises(TargetBusy, match="reserved"):
        _save(store, bot)
    # The reservation's own IDLE_ONLY save upserts it into the running turn.
    _save(store, bot, turn_id=reserved, admission="idle_only", claim=_owner_claim())
    turn = store.get_turn(reserved)
    assert turn.status == "streaming" and turn.ended_at is None
    with store.engine.connect() as conn:
        assert conn.execute(sa_text(
            "SELECT count(*) FROM turn_logs WHERE bot_id=:bot"), {"bot": bot}).scalar_one() == 1


@pytest.mark.integration
def test_release_drops_only_an_unstarted_reservation(pg_turns):
    store, bot = pg_turns
    reserved = f"turn-approval-{uuid.uuid4().hex}"
    _reserve(store, bot, reserved)
    assert store.release_reservation(reserved, owner="owner-a")
    _reserve(store, bot, reserved)
    _save(store, bot, turn_id=reserved, admission="idle_only", claim=_owner_claim())
    # running rows are never dropped
    assert not store.release_reservation(reserved, owner="owner-a")
    assert store.get_turn(reserved).status == "streaming"


@pytest.mark.integration
def test_idle_only_save_loses_to_any_open_turn_but_start_does_not(pg_turns):
    store, bot = pg_turns
    _open(store, bot)
    with pytest.raises(TargetBusy):
        _save(store, bot, admission="idle_only")
    # Transitional compatibility (slice 2 replaces this with backend queueing).
    _save(store, bot)


@pytest.mark.integration
def test_any_reservation_fences_start_even_when_not_newest(pg_turns):
    store, bot = pg_turns
    _open(store, bot, status="reserved")
    _open(store, bot, status="streaming")  # newer row is not the reservation
    with pytest.raises(TargetBusy, match="reserved"):
        _save(store, bot)


@pytest.mark.integration
def test_delivery_claim_waits_instead_of_steering_ambiguous_or_reserved_targets(pg_turns):
    from llm_bawt.inter_bot_delivery import STEERING, InterBotDeliveryStore

    store, bot = pg_turns
    deliveries = InterBotDeliveryStore(pg_turns[0].config)
    sender = f"test-admission-sender-{uuid.uuid4().hex}"
    try:
        record, _ = deliveries.enqueue(
            sender_bot_id=sender, target_bot_id=bot, message="callback",
            payload={"messages": [{"role": "user", "content": "callback"}],
                     "bot_id": bot, "user": "nick", "stream": True},
        )
        first = _open(store, bot)
        second = _open(store, bot)
        assert deliveries.claim_next(bot, claim_owner="o", steer_capable=True) is None

        with store.engine.begin() as conn:
            conn.execute(sa_text("UPDATE turn_logs SET ended_at=now(), status='ok' WHERE id IN (:a, :b)"),
                         {"a": first, "b": second})
        approval = f"turn-approval-{uuid.uuid4().hex}"
        _reserve(store, bot, approval)
        assert deliveries.claim_next(bot, claim_owner="o", steer_capable=True) is None

        _save(store, bot, turn_id=approval, admission="idle_only", claim=_owner_claim())
        claimed = deliveries.claim_next(bot, claim_owner="o", steer_capable=True)
        assert claimed.status == STEERING and claimed.target_turn_id == approval
        assert claimed.id == record.id
    finally:
        with deliveries.engine.begin() as conn:
            conn.execute(sa_text("DELETE FROM inter_bot_deliveries WHERE sender_bot_id=:s"),
                         {"s": sender})


@pytest.mark.integration
def test_delivery_consumption_fence_rejects_reclaimed_claim_on_postgres(pg_turns):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    from llm_bawt.service.inter_bot_claims import turn_admission_claim

    turns, bot = pg_turns
    deliveries = InterBotDeliveryStore(pg_turns[0].config)
    sender = f"test-admission-sender-{uuid.uuid4().hex}"
    record, _ = deliveries.enqueue(
        sender_bot_id=sender, target_bot_id=bot, message="callback",
        payload={"messages": [{"role": "user", "content": "callback"}],
                 "bot_id": bot, "user": "nick", "stream": True},
    )
    try:
        first = deliveries.claim_next(bot, claim_owner="worker-1", steer_capable=False)
        assert first and first.id == record.id
        payload = deliveries.payload(record.id)
        request = ChatCompletionRequest.model_validate({
            **payload, "inter_bot_claim_token": first.claim_token,
        })
        service = SimpleNamespace(_inter_bot_dispatcher=SimpleNamespace(store=deliveries))
        old_claim = turn_admission_claim(service, request)
        assert deliveries.requeue(record.id, first.claim_token, "reclaim", delay_seconds=0)
        successor = deliveries.claim_next(bot, claim_owner="worker-2", steer_capable=False)
        assert successor and successor.claim_token != first.claim_token

        with pytest.raises(AdmissionFenceLost):
            _save(turns, bot, turn_id=record.turn_id, admission="idle_only", claim=old_claim)
        with pytest.raises(AdmissionFenceLost):
            _save(turns, bot, turn_id=record.turn_id, admission="idle_only")
        assert turns.get_turn(record.turn_id).status == "reserved"

        request.inter_bot_claim_token = successor.claim_token
        _save(turns, bot, turn_id=record.turn_id, admission="idle_only",
              claim=turn_admission_claim(service, request))
        assert turns.get_turn(record.turn_id).status == "streaming"
    finally:
        with deliveries.engine.begin() as conn:
            conn.execute(sa_text("DELETE FROM inter_bot_deliveries WHERE sender_bot_id=:sender"),
                         {"sender": sender})


@pytest.mark.integration
def test_delivery_enqueue_can_share_the_admission_transaction(pg_turns):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    from llm_bawt.service.turn_admission import lock_target

    turns, bot = pg_turns
    deliveries = InterBotDeliveryStore(pg_turns[0].config)
    sender = f"test-admission-sender-{uuid.uuid4().hex}"
    assistant_id = str(uuid.uuid4())
    delivery_id = None
    try:
        with pytest.raises(RuntimeError, match="roll back admission"):
            with turns.engine.begin() as conn:
                lock_target(conn, bot)
                record, duplicate = deliveries.enqueue(
                    sender_bot_id=sender, target_bot_id=bot, message="wait",
                    payload={"messages": [{"role": "user", "content": "wait"}],
                             "bot_id": bot, "user": "nick", "stream": True},
                    assistant_message_id=assistant_id, connection=conn,
                )
                assert not duplicate
                delivery_id = record.id
                assert conn.execute(sa_text(
                    "SELECT status, payload_json->>'assistant_message_id' "
                    "FROM inter_bot_deliveries WHERE id=:id"
                ), {"id": delivery_id}).one() == ("QUEUED", assistant_id)
                raise RuntimeError("roll back admission")
        assert deliveries.get(delivery_id) is None
    finally:
        with deliveries.engine.begin() as conn:
            conn.execute(sa_text("DELETE FROM inter_bot_deliveries WHERE sender_bot_id=:sender"),
                         {"sender": sender})


@pytest.mark.integration
def test_delivery_claim_fence_locks_out_concurrent_requeue(pg_turns):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    from llm_bawt.service.inter_bot_claims import turn_admission_claim
    from llm_bawt.service.turn_admission import lock_target

    turns, bot = pg_turns
    deliveries = InterBotDeliveryStore(pg_turns[0].config)
    sender = f"test-admission-sender-{uuid.uuid4().hex}"
    record, _ = deliveries.enqueue(
        sender_bot_id=sender, target_bot_id=bot, message="callback",
        payload={"messages": [{"role": "user", "content": "callback"}],
                 "bot_id": bot, "user": "nick", "stream": True},
    )
    worker = None
    try:
        claimed = deliveries.claim_next(bot, claim_owner="worker", steer_capable=False)
        request = ChatCompletionRequest.model_validate({
            **deliveries.payload(record.id), "inter_bot_claim_token": claimed.claim_token,
        })
        service = SimpleNamespace(_inter_bot_dispatcher=SimpleNamespace(store=deliveries))
        claim = turn_admission_claim(service, request)
        outcomes = []
        with turns.engine.begin() as conn:
            lock_target(conn, bot)
            assert claim.fence(conn)
            worker = threading.Thread(target=lambda: outcomes.append(deliveries.requeue(
                record.id, claimed.claim_token, "reclaim", delay_seconds=0,
            )))
            worker.start()
            _assert_waiting_on_holder(turns.engine, conn, worker)
        worker.join(5)
        assert not worker.is_alive() and outcomes and outcomes[0].status == "QUEUED"
    finally:
        if worker and worker.is_alive():
            worker.join(5)
            assert not worker.is_alive()
        with deliveries.engine.begin() as conn:
            conn.execute(sa_text("DELETE FROM inter_bot_deliveries WHERE sender_bot_id=:sender"),
                         {"sender": sender})


@pytest.mark.integration
def test_release_is_fenced_by_reservation_owner(pg_turns):
    store, bot = pg_turns
    reserved = f"turn-approval-{uuid.uuid4().hex}"
    _reserve(store, bot, reserved, owner="owner-a")
    # A reclaimer re-arms the same deterministic id under its own token.
    _reserve(store, bot, reserved, owner="owner-b")
    assert not store.release_reservation(reserved, owner="owner-a")
    assert store.get_turn(reserved).status == "reserved"
    assert store.release_reservation(reserved, owner="owner-b")


@pytest.mark.integration
@pytest.mark.parametrize("finished", [False, True])
def test_reserve_never_rearms_a_started_or_finished_same_id_turn(pg_turns, finished):
    store, bot = pg_turns
    turn_id = f"turn-approval-{uuid.uuid4().hex}"
    _reserve(store, bot, turn_id)
    _save(store, bot, turn_id=turn_id, admission="idle_only", claim=_owner_claim())
    if finished:
        with store.engine.begin() as conn:
            conn.execute(sa_text(
                "UPDATE turn_logs SET status='ok', ended_at=now() WHERE id=:id"), {"id": turn_id})
    with pytest.raises(TargetBusy, match="already started"):
        _reserve(store, bot, turn_id, owner="owner-b")
    turn = store.get_turn(turn_id)
    assert turn.status == ("ok" if finished else "streaming")
    assert (turn.ended_at is not None) is finished


@pytest.mark.integration
def test_delivery_retry_still_rearms_its_ended_turn(pg_turns):
    """claim_next semantics are unchanged: a failed delivery turn is re-armed."""
    from llm_bawt.service.turn_admission import write_reservation

    store, bot = pg_turns
    turn_id = _open(store, bot)
    with store.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE turn_logs SET status='error', ended_at=now() WHERE id=:id"), {"id": turn_id})
        assert write_reservation(
            conn, turn_id=turn_id, bot_id=bot, user_id="nick", user_prompt="retry",
            request_json={}, trigger_message_id=None,
        )
    turn = store.get_turn(turn_id)
    assert turn.status == "reserved" and turn.ended_at is None


@pytest.fixture
def pg_approvals(pg_turns):
    from llm_bawt.approval_policies import ToolApprovalPolicyStore

    store, bot = pg_turns
    approvals = ToolApprovalPolicyStore(None, engine=store.engine)
    approvals._ensure_tables_exist()
    request_id = f"req-admission-{uuid.uuid4().hex}"
    yield store, bot, approvals, request_id
    with store.engine.begin() as conn:
        conn.execute(sa_text("DELETE FROM tool_approval_requests WHERE id=:id"), {"id": request_id})


@pytest.mark.integration
def test_superseded_claim_can_neither_reserve_nor_release(pg_approvals):
    from llm_bawt.service.turn_admission import AdmissionFenceLost

    store, bot, approvals, request_id = pg_approvals
    old = _mcp_claim(approvals, request_id=request_id)
    old_fence = approvals.continuation_claim_fence(old.id, old.continuation_claim_token)
    turn_id = f"turn-approval-{uuid.uuid4().hex}"
    _reserve(store, bot, turn_id, owner=old.continuation_claim_token, fence=old_fence)

    new = approvals.claim_continuation(old.id, lease_seconds=0)
    assert new.continuation_claim_token != old.continuation_claim_token
    with pytest.raises(AdmissionFenceLost):
        _reserve(store, bot, turn_id, owner=old.continuation_claim_token, fence=old_fence)
    assert not store.release_reservation(
        turn_id, owner=old.continuation_claim_token, fence=old_fence)
    assert store.get_turn(turn_id).status == "reserved"
    assert not approvals.claim_client_grant(
        old.id, continuation_claim_token=old.continuation_claim_token)

    new_fence = approvals.continuation_claim_fence(new.id, new.continuation_claim_token)
    _reserve(store, bot, turn_id, owner=new.continuation_claim_token, fence=new_fence)
    assert store.release_reservation(
        turn_id, owner=new.continuation_claim_token, fence=new_fence)


@pytest.mark.integration
def test_superseded_stream_cannot_consume_successor_reservation(pg_approvals):
    """save_turn fences reserved->executing with the trusted claim (TASK-1082)."""
    store, bot, approvals, request_id = pg_approvals
    old = _mcp_claim(approvals, request_id=request_id)
    old_claim = _owner_claim(old.continuation_claim_token, approvals.continuation_claim_fence(
        old.id, old.continuation_claim_token))
    turn_id = f"turn-approval-{uuid.uuid4().hex}"
    _reserve(store, bot, turn_id, owner=old_claim.owner, fence=old_claim.fence)

    new = approvals.claim_continuation(old.id, lease_seconds=0)
    new_claim = _owner_claim(new.continuation_claim_token, approvals.continuation_claim_fence(
        new.id, new.continuation_claim_token))
    # Old worker's own reservation, and then the successor's re-armed one.
    with pytest.raises(AdmissionFenceLost):
        _save(store, bot, turn_id=turn_id, admission="idle_only", claim=old_claim)
    assert store.release_reservation(turn_id, owner=old_claim.owner) is True
    _reserve(store, bot, turn_id, owner=new_claim.owner, fence=new_claim.fence)
    with pytest.raises(AdmissionFenceLost):
        _save(store, bot, turn_id=turn_id, admission="idle_only", claim=old_claim)
    with pytest.raises(AdmissionFenceLost):  # nor ordinary ingress without a claim
        _save(store, bot, turn_id=turn_id)
    turn = store.get_turn(turn_id)
    assert turn.status == "reserved"
    assert json.loads(turn.request_json)["reservation_owner"] == new_claim.owner

    _save(store, bot, turn_id=turn_id, admission="idle_only", claim=new_claim)
    assert store.get_turn(turn_id).status == "streaming"


@pytest.mark.integration
def test_claim_fence_serializes_a_concurrent_reclaim(pg_approvals):
    """FOR UPDATE: a reclaim cannot interleave between fence check and commit."""
    store, bot, approvals, request_id = pg_approvals
    row = _mcp_claim(approvals, request_id=request_id)
    fence = approvals.continuation_claim_fence(row.id, row.continuation_claim_token)
    reclaimed = []
    with store.engine.begin() as conn:
        assert fence(conn)
        worker = threading.Thread(
            target=lambda: reclaimed.append(approvals.claim_continuation(row.id, lease_seconds=0)))
        worker.start()
        _assert_waiting_on_holder(store.engine, conn, worker)
    worker.join(5)
    assert not worker.is_alive() and reclaimed
    assert reclaimed[0].continuation_claim_token != row.continuation_claim_token


def _claimed_delivery(store, bot, sender, *, metadata=None):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore

    deliveries = InterBotDeliveryStore(store.config)
    record, _ = deliveries.enqueue(
        sender_bot_id=sender, target_bot_id=bot, message="callback",
        payload={"messages": [{"role": "user", "content": "callback"}],
                 "bot_id": bot, "user": "nick", "stream": True},
        metadata=metadata,
    )
    claimed = deliveries.claim_next(bot, claim_owner="worker", steer_capable=False)
    assert claimed and claimed.id == record.id and claimed.delivery_mode == "turn"
    return deliveries, claimed


def _delivery_metadata(deliveries, delivery_id):
    with deliveries.engine.connect() as conn:
        return conn.execute(sa_text(
            "SELECT status, metadata_json FROM inter_bot_deliveries WHERE id=:id"
        ), {"id": delivery_id}).one()


def _drop_sender(deliveries, sender):
    with deliveries.engine.begin() as conn:
        conn.execute(sa_text("DELETE FROM inter_bot_deliveries WHERE sender_bot_id=:s"), {"s": sender})


@pytest.mark.integration
def test_requeue_metadata_merge_lands_with_the_queued_transition(pg_turns):
    """TASK-1251: the merge is shallow, keeps other keys, and rides the requeue UPDATE."""
    store, bot = pg_turns
    sender = f"test-admission-sender-{uuid.uuid4().hex}"
    deliveries, claimed = _claimed_delivery(store, bot, sender, metadata={"kept": "yes"})
    try:
        assert deliveries.requeue(claimed.id, claimed.claim_token, "rejected", delay_seconds=0,
                                  metadata_merge={"steer_generation": 1})
        status, metadata = _delivery_metadata(deliveries, claimed.id)
        assert status == "QUEUED"
        assert metadata["steer_generation"] == 1 and metadata["kept"] == "yes"
    finally:
        _drop_sender(deliveries, sender)


@pytest.mark.integration
def test_requeue_without_metadata_merge_leaves_metadata_untouched(pg_turns):
    store, bot = pg_turns
    sender = f"test-admission-sender-{uuid.uuid4().hex}"
    deliveries, claimed = _claimed_delivery(store, bot, sender, metadata={"kept": "yes"})
    try:
        _, before = _delivery_metadata(deliveries, claimed.id)
        assert deliveries.requeue(claimed.id, claimed.claim_token, "busy", delay_seconds=0)
        status, after = _delivery_metadata(deliveries, claimed.id)
        assert status == "QUEUED" and after == before
    finally:
        _drop_sender(deliveries, sender)


@pytest.mark.integration
def test_requeue_metadata_merge_is_ignored_when_stop_cancels(pg_turns):
    store, bot = pg_turns
    sender = f"test-admission-sender-{uuid.uuid4().hex}"
    deliveries, claimed = _claimed_delivery(store, bot, sender, metadata={"kept": "yes"})
    try:
        _, before = _delivery_metadata(deliveries, claimed.id)
        with store.engine.begin() as conn:
            conn.execute(sa_text("UPDATE turn_logs SET request_json=:rj WHERE id=:id"), {
                "id": claimed.turn_id,
                "rj": json.dumps({"abort_request": {"source": "chat_stop"}}),
            })
        cancelled = deliveries.requeue(claimed.id, claimed.claim_token, "rejected",
                                       delay_seconds=0, metadata_merge={"steer_generation": 1})
        assert cancelled and cancelled.status == "CANCELLED"
        status, after = _delivery_metadata(deliveries, claimed.id)
        assert status == "CANCELLED" and after == before
    finally:
        _drop_sender(deliveries, sender)


@pytest.mark.integration
def test_tool_start_does_not_wait_on_a_locked_parent_turn(pg_turns):
    """A tool-start FK can coexist with an app writer's non-key row lock."""
    from llm_bawt.service.tool_event_coordinator import ToolEventCoordinator
    from llm_bawt.service.turn_admission import fence_consumption

    turns, bot = pg_turns
    turn_id = _open(turns, bot)
    user_id = turns.get_turn(turn_id).user_id
    outcome = []
    done = threading.Event()

    def start():
        try:
            outcome.append(ToolEventCoordinator(turns.engine).start({
                "turn_id": turn_id, "bot_id": bot, "user_id": user_id,
                "call_id": "review-tool", "tool_name": "tasks_reviews_add", "event": "tool_start",
            }))
        except Exception as exc:
            outcome.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=start)
    try:
        with turns.engine.begin() as holder:
            # Exercise the real admission writer's lock, not a hand-written lock
            # that would still pass if fence_consumption regressed to FOR UPDATE.
            assert fence_consumption(holder, turn_id=turn_id, claim=None) == "streaming"
            worker.start()
            # timing-ok: a hang guard while the holder deliberately keeps its lock
            assert done.wait(10), "tool start waited for the turn-row lock"
            assert outcome and isinstance(outcome[0], dict)
            assert outcome[0]["_tool_persisted"] is True
            assert turns.get_turn(turn_id).ended_at is None
        worker.join(10)
        assert not worker.is_alive()
    finally:
        with turns.engine.begin() as conn:
            conn.execute(sa_text("DELETE FROM tool_call_records WHERE turn_id=:id"), {"id": turn_id})


@pytest.mark.integration
def test_reservation_delete_lock_excludes_concurrent_tool_start(pg_turns):
    """A deleting reservation retains its stronger lock until commit."""
    from sqlmodel import Session

    from llm_bawt.service.tool_call_store import ToolCallStore
    from llm_bawt.service.turn_admission import _LockedReservation, _lock_open_reservation

    turns, bot = pg_turns
    turn_id = f"turn-approval-{uuid.uuid4().hex}"
    _reserve(turns, bot, turn_id)
    outcome = []

    def start():
        try:
            outcome.append(ToolCallStore(turns.engine).save_start(
                turn_id=turn_id, bot_id=bot, user_id="nick", call_id="reserved-tool",
                tool_name="tasks_reviews_add",
            ))
        except Exception as exc:
            outcome.append(exc)

    worker = threading.Thread(target=start)
    try:
        with turns.engine.begin() as holder, Session(bind=holder) as session:
            assert isinstance(_lock_open_reservation(holder, session, turn_id), _LockedReservation)
            worker.start()
            _assert_waiting_on_holder(turns.engine, holder, worker)
        worker.join(10)
        assert not worker.is_alive() and len(outcome) == 1
        assert not isinstance(outcome[0], Exception)
    finally:
        if worker.is_alive():
            worker.join(10)
        with turns.engine.begin() as conn:
            conn.execute(sa_text("DELETE FROM tool_call_records WHERE turn_id=:id"), {"id": turn_id})


@pytest.mark.integration
def test_tool_start_first_is_interrupted_when_parent_commits_after_start(pg_turns, monkeypatch):
    """The terminal post-commit sweep catches a start paused before recheck."""
    from llm_bawt.service import turn_termination
    from llm_bawt.service.tool_call_store import ToolCallStore

    turns, bot = pg_turns
    turn_id = _open(turns, bot)
    user_id = turns.get_turn(turn_id).user_id
    started = threading.Event()
    resume = threading.Event()
    original = turn_termination.reconcile_finished_tool

    def paused_recheck(engine, *, turn_id, record_id=None):
        if record_id is not None:
            started.set()
            # timing-ok: a guarded handoff, not a wall-clock performance assertion
            assert resume.wait(10)
        return original(engine, turn_id=turn_id, record_id=record_id)

    monkeypatch.setattr(turn_termination, "reconcile_finished_tool", paused_recheck)
    worker = threading.Thread(target=lambda: ToolCallStore(turns.engine).save_start(
        turn_id=turn_id, bot_id=bot, user_id=user_id, call_id="review-tool",
        tool_name="tasks_reviews_add",
    ))
    try:
        worker.start()
        # timing-ok: a guarded handoff, not a wall-clock performance assertion
        assert started.wait(10)
        turns.update_turn(turn_id=turn_id, status="ok", end_reason="stop")
    finally:
        resume.set()
        worker.join(10)
        assert not worker.is_alive()
    try:
        calls = turns.get_tool_calls_for_turn(turn_id)
        assert len(calls) == 1 and calls[0].ended_at is not None
        assert calls[0].result_complete is False
        ended_at = calls[0].ended_at
        ToolCallStore(turns.engine).save_start(
            turn_id=turn_id, bot_id=bot, user_id=user_id, call_id="review-tool",
            tool_name="tasks_reviews_add",
        )
        assert turns.get_tool_calls_for_turn(turn_id)[0].ended_at == ended_at
    finally:
        with turns.engine.begin() as conn:
            conn.execute(sa_text("DELETE FROM tool_call_records WHERE turn_id=:id"), {"id": turn_id})


@pytest.mark.integration
def test_tool_start_after_parent_terminal_is_interrupted_by_recheck(pg_turns):
    """The start-side recheck catches a terminal before its insert."""
    from llm_bawt.service.tool_call_store import ToolCallStore

    turns, bot = pg_turns
    turn_id = _open(turns, bot)
    user_id = turns.get_turn(turn_id).user_id
    try:
        turns.update_turn(turn_id=turn_id, status="ok", end_reason="stop")
        ToolCallStore(turns.engine).save_start(
            turn_id=turn_id, bot_id=bot, user_id=user_id, call_id="late-review",
            tool_name="tasks_reviews_add",
        )
        calls = turns.get_tool_calls_for_turn(turn_id)
        assert len(calls) == 1 and calls[0].ended_at is not None
        assert calls[0].result_complete is False
    finally:
        with turns.engine.begin() as conn:
            conn.execute(sa_text("DELETE FROM tool_call_records WHERE turn_id=:id"), {"id": turn_id})
