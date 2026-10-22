"""Manual retry creates one new durable command only for failed bot callbacks."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import event, text

from llm_bawt.bot_identities import SYSTEM_BOT_ID
from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.inter_bot_delivery_retry import (
    DEFAULT_RESTART_RETRY_CONTINUE, RESTART_RETRY_OUTCOME, DeliveryRetryConflict,
    fail_claim_with_restart_retry, restart_abort_cause, restart_retry_delivery, retry_failed_delivery,
)
from llm_bawt.inter_bot_reply_channel import delivery_reply_channel_note
from llm_bawt.message_authorship import AuthorReference
from llm_bawt.service.turn_admission import write_reservation
from tests.pg_isolation import isolated_pg_store

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def delivery():
    store = isolated_pg_store(InterBotDeliveryStore)
    if store.engine is None:
        pytest.skip("PostgreSQL unavailable")
    slug = f"retry-test-{uuid.uuid4().hex}"
    yield store, slug
    with store.engine.begin() as conn:
        conn.execute(text("DELETE FROM inter_bot_deliveries WHERE sender_bot_id=:slug"), {"slug": slug})


def submit(store, slug, *, metadata=None, session_policy="continue", key="original", reply_mode="async"):
    return store.enqueue(
        sender_bot_id=slug, target_bot_id=slug, message="Please review this",
        payload={"messages": [{"role": "user", "content": "Please review this"}],
                 "bot_id": slug, "prefer_steer": reply_mode != "waited", "stream": False,
                 "reply_mode": reply_mode},
        idempotency_key=key, task_id="TASK-1111", message_kind="PROGRESS",
        metadata=metadata, session_policy=session_policy,
    )[0]


def test_manual_retry_and_cancel_failed_original_do_not_deadlock(delivery, monkeypatch):
    store, slug = delivery
    original = submit(store, slug)
    store.mark_failed(original.id, "transient error")
    retry_has_row = threading.Event()
    resume_retry = threading.Event()
    retry_result, cancel_result = [], []
    retry_pid = []
    enqueue = store.enqueue

    def paused_enqueue(**kwargs):
        conn = kwargs["connection"]
        conn.execute(text("SET LOCAL lock_timeout = '4s'"))
        retry_pid.append(conn.execute(text("SELECT pg_backend_pid()")).scalar_one())
        retry_has_row.set()  # retry holds target then original row before admission
        if not resume_retry.wait(timeout=3):
            raise TimeoutError("cancel never reached the original row")
        return enqueue(**kwargs)

    def set_lock_timeout(conn):
        conn.exec_driver_sql("SET LOCAL lock_timeout = '4s'")

    monkeypatch.setattr(store, "enqueue", paused_enqueue)
    event.listen(store.engine, "begin", set_lock_timeout)

    def retry():
        try:
            retry_result.append(retry_failed_delivery(store, original.id))
        except Exception as exc:
            retry_result.append(exc)

    def cancel():
        try:
            cancel_result.append(store.cancel(original.id))
        except Exception as exc:
            cancel_result.append(exc)

    retrier = threading.Thread(target=retry)
    canceller = threading.Thread(target=cancel)
    try:
        retrier.start()
        assert retry_has_row.wait(timeout=2)
        canceller.start()
        deadline = time.monotonic() + 2
        blocked = False
        with store.engine.connect() as observer:
            while time.monotonic() < deadline:
                blocked = observer.execute(text("""
                    SELECT EXISTS (SELECT 1 FROM pg_stat_activity
                    WHERE datname=current_database() AND wait_event_type='Lock'
                      AND :pid = ANY(pg_blocking_pids(pid)))
                """), {"pid": retry_pid[0]}).scalar_one()
                if blocked or not canceller.is_alive():
                    break
                time.sleep(0.02)
        assert blocked and canceller.is_alive(), "cancel never waited on target lock"
    finally:
        resume_retry.set()
        retrier.join(timeout=5)
        if canceller.ident is not None:
            canceller.join(timeout=5)
        event.remove(store.engine, "begin", set_lock_timeout)
    assert not retrier.is_alive() and not canceller.is_alive()
    assert len(retry_result) == 1 and isinstance(retry_result[0], tuple), retry_result
    child, duplicate = retry_result[0]
    assert not duplicate and child.id != original.id
    assert cancel_result == [None]  # FAILED original cannot be cancelled
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE inter_bot_deliveries SET status='FAILED' WHERE id=:id"),
                     {"id": child.id})
        assert conn.execute(text(
            "SELECT count(*) FROM inter_bot_deliveries WHERE sender_bot_id=:sender"
        ), {"sender": slug}).scalar_one() == 2


def test_failed_retry_is_new_idempotent_delivery_with_original_provenance(delivery):
    store, slug = delivery
    original = submit(store, slug, metadata={"source": "test", "maintenance_barrier": {"turn_id": "old-turn"}})
    store.mark_failed(original.id, "transient error")

    retried, duplicate = retry_failed_delivery(store, original.id)
    again, repeated = retry_failed_delivery(store, original.id)
    assert duplicate is False and repeated is True and again.id == retried.id
    assert retried.id != original.id and retried.user_message_id != original.user_message_id
    assert retried.turn_id != original.turn_id
    assert retried.status == "QUEUED" and store.get(original.id).status == "FAILED"
    assert retried.author == original.author
    assert retried.task_id == original.task_id and retried.message_kind == original.message_kind
    assert retried.metadata["manual_retry_of"] == original.id
    assert "maintenance_barrier" not in retried.metadata
    payload = store.payload(retried.id)
    assert payload["messages"] == store.payload(original.id)["messages"]
    assert payload["inter_bot_delivery_id"] == retried.id
    assert payload["inter_bot_bridge_request_id"] != store.payload(original.id)["inter_bot_bridge_request_id"]
    assert len(store.list(sender_bot_id=slug)) == 2


def test_manual_retry_preserves_waited_instruction_and_reply_mode(delivery):
    store, slug = delivery
    original = submit(store, slug, reply_mode="waited")
    store.mark_failed(original.id, "transient error")
    retried, duplicate = retry_failed_delivery(store, original.id)
    assert not duplicate
    assert retried.reply_mode == "waited"
    payload = store.payload(retried.id)
    assert payload["reply_mode"] == "waited"
    assert payload["prefer_steer"] is False
    assert store.get(retried.id).to_api()["reply_mode"] == "waited"


def test_only_failed_normal_bot_callbacks_can_be_retried(delivery):
    store, slug = delivery
    queued = submit(store, slug)
    with pytest.raises(DeliveryRetryConflict, match="Only failed"):
        retry_failed_delivery(store, queued.id)
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE inter_bot_deliveries SET status='DELIVERED' WHERE id=:id"), {"id": queued.id})
    with pytest.raises(DeliveryRetryConflict, match="Only failed"):
        retry_failed_delivery(store, queued.id)
    failed = submit(store, slug, key="failed-original")
    store.mark_failed(failed.id, "no transport")
    retry_failed_delivery(store, failed.id)
    for metadata in ({"ordinary_chat": {"session_id": None}},
                     {"prompt_schedule": {}}, {"maintenance_user_queue": True}):
        # Each row needs a distinct original key in the same sender namespace.
        other_slug = f"{slug}-{uuid.uuid4().hex[:6]}"
        row = submit(store, other_slug, metadata=metadata)
        store.mark_failed(row.id, "test")
        assert retry_failed_delivery(store, row.id) is None
        with store.engine.begin() as conn:
            conn.execute(text("DELETE FROM inter_bot_deliveries WHERE sender_bot_id=:slug"), {"slug": other_slug})
    assert retry_failed_delivery(store, "delivery-not-found") is None


def test_failed_session_reset_requires_explicit_new_send(delivery):
    store, slug = delivery
    original = submit(store, slug, session_policy="reset_retain_history")
    store.mark_failed(original.id, "test")
    with pytest.raises(DeliveryRetryConflict, match="Session-reset"):
        retry_failed_delivery(store, original.id)
    assert len(store.list(sender_bot_id=slug)) == 1


# --- TASK-1231B: restart retry (host shutdown) -------------------------------

def end_turn(store, record, *, status="aborted", audit=None):
    """Give ``record``'s turn a terminal row carrying ``audit`` request_json keys."""
    request = {"messages": [], **(audit or {})}
    with store.engine.begin() as conn:
        write_reservation(conn, turn_id=record.turn_id, bot_id=record.target_bot_id,
                          user_id="nick", user_prompt=record.message, request_json=request,
                          trigger_message_id=record.user_message_id)
        conn.execute(text("""
            UPDATE turn_logs SET status=:status, end_reason=:status, ended_at=now()
            WHERE id=:id
        """), {"id": record.turn_id, "status": status})


SHUTDOWN = {"abort_cause": {"cause": "host_shutdown", "at": "2026-10-22T00:26:22Z"}}


@pytest.fixture
def turn_cleanup(delivery):
    store, slug = delivery
    yield
    with store.engine.begin() as conn:
        conn.execute(text("DELETE FROM turn_logs WHERE bot_id=:slug OR bot_id LIKE :prefix"),
                     {"slug": slug, "prefix": f"{slug}-%"})


def test_restart_retry_continues_a_host_shutdown_turn(delivery, turn_cleanup):
    store, slug = delivery
    original = submit(store, slug, metadata={
        "ordinary_chat": {"session_id": "s-1", "stream": True, "digest": "d", "assistant_supplied": True},
    })
    store.mark_failed(original.id, "streamed durable turn ended with status=aborted")
    end_turn(store, original, audit=SHUTDOWN)

    child, duplicate = restart_retry_delivery(store, original.id)
    again, repeated = restart_retry_delivery(store, original.id)

    assert not duplicate and repeated and again.id == child.id
    assert child.id != original.id and child.turn_id != original.turn_id
    assert child.status == "QUEUED" and store.get(original.id).status == "FAILED"
    assert child.idempotency_key == f"restart-retry:{original.id}"
    assert child.metadata["restart_retry_of"] == original.id
    assert "manual_retry_of" not in child.metadata
    assert child.metadata["ordinary_chat"] == {"session_id": "s-1", "stream": True}
    # System-authored: never words in the original author's mouth.
    assert child.author == AuthorReference.bot(SYSTEM_BOT_ID) != original.author
    # The original request is always carried: a send the host killed before
    # it started must not shrink to a bare "continue" (Vex 1231B r1 B2).
    expected = f"{DEFAULT_RESTART_RETRY_CONTINUE}\n\nPlease review this"
    assert child.message == expected
    payload = store.payload(child.id)
    assert payload["messages"] == [{"role": "user", "content": expected}]
    assert payload["inter_bot_delivery_id"] == child.id


def test_restart_child_of_bot_request_replies_to_the_original_sender(delivery, turn_cleanup):
    """Vex 1231B r1 B1: a system-authored child routes its reply note to the bot sender."""
    store, slug = delivery
    original = submit(store, slug)
    assert original.author == AuthorReference.bot(slug)
    store.mark_failed(original.id, "aborted")
    end_turn(store, original, audit=SHUTDOWN)
    child, _ = restart_retry_delivery(store, original.id)

    assert child.author == AuthorReference.bot(SYSTEM_BOT_ID)
    assert child.metadata["retry_original_author"] == AuthorReference.bot(slug).to_dict()
    note = delivery_reply_channel_note(child)
    assert note is not None and f'target_bot_id="{slug}"' in note
    assert f'target_bot_id="{SYSTEM_BOT_ID}"' not in note


def test_forged_original_author_on_a_user_request_is_replaced(delivery, turn_cleanup):
    store, slug = delivery
    original = store.enqueue(
        sender_bot_id=slug, target_bot_id=slug, message="hi", author=AuthorReference.user("nick"),
        payload={"messages": [{"role": "user", "content": "hi"}], "bot_id": slug},
        idempotency_key="user-original",
        metadata={"retry_original_author": {"entity_type": "bot", "entity_id": "victim"}},
    )[0]
    store.mark_failed(original.id, "aborted")
    end_turn(store, original, audit=SHUTDOWN)
    child, _ = restart_retry_delivery(store, original.id)

    # The forged value is replaced by the real (user) author: no bot reply note.
    assert child.metadata["retry_original_author"] == AuthorReference.user("nick").to_dict()
    assert delivery_reply_channel_note(child) is None


@pytest.mark.parametrize("status, audit", [
    ("aborted", {**SHUTDOWN, "abort_request": {"source": "chat_stop"}}),  # user Stop wins
    ("aborted", {}),                                                       # plain abort
    ("error", SHUTDOWN),                                                   # not an abort
], ids=["user-stop", "no-cause", "error-status"])
def test_restart_retry_refuses_turns_not_killed_by_host_shutdown(delivery, turn_cleanup, status, audit):
    store, slug = delivery
    original = submit(store, slug)
    store.mark_failed(original.id, "failed")
    end_turn(store, original, status=status, audit=audit)
    assert restart_retry_delivery(store, original.id) is None
    assert len(store.list(sender_bot_id=slug)) == 1


def test_restart_retry_refuses_session_intents_resets_and_chains(delivery, turn_cleanup):
    store, slug = delivery
    for metadata in ({"new_session_intent": {}}, {"prompt_schedule": {}},
                     {"maintenance_user_queue": True}, {"restart_retry_of": "delivery-x"}):
        other = f"{slug}-{uuid.uuid4().hex[:6]}"
        row = submit(store, other, metadata=metadata)
        store.mark_failed(row.id, "failed")
        end_turn(store, row, audit=SHUTDOWN)
        assert restart_retry_delivery(store, row.id) is None, metadata
        with store.engine.begin() as conn:
            conn.execute(text("DELETE FROM inter_bot_deliveries WHERE sender_bot_id=:s"), {"s": other})
    reset = submit(store, slug, session_policy="reset_retain_history", key="reset")
    store.mark_failed(reset.id, "failed")
    end_turn(store, reset, audit=SHUTDOWN)
    with pytest.raises(DeliveryRetryConflict, match="Session-reset"):
        restart_retry_delivery(store, reset.id)


def test_one_retry_per_original_across_producers(delivery, turn_cleanup):
    store, slug = delivery
    restarted = submit(store, slug, key="restarted")
    store.mark_failed(restarted.id, "failed")
    end_turn(store, restarted, audit=SHUTDOWN)
    child, _ = restart_retry_delivery(store, restarted.id)
    with pytest.raises(DeliveryRetryConflict, match=child.id):
        retry_failed_delivery(store, restarted.id)

    manual = submit(store, slug, key="manual")
    store.mark_failed(manual.id, "failed")
    end_turn(store, manual, audit=SHUTDOWN)
    manual_child, _ = retry_failed_delivery(store, manual.id)
    with pytest.raises(DeliveryRetryConflict, match=manual_child.id):
        restart_retry_delivery(store, manual.id)
    assert json.loads(json.dumps(manual_child.metadata))["manual_retry_of"] == manual.id


def _claimed(store, record, token="claim-1231b"):
    with store.engine.begin() as conn:
        conn.execute(text("""
            UPDATE inter_bot_deliveries SET status='DISPATCHING', claim_token=:token,
                claim_owner='test', lease_expires_at=now() + interval '1 minute',
                delivery_mode='turn'
            WHERE id=:id
        """), {"id": record.id, "token": token})
    return token


def _children(store, original_id):
    with store.engine.connect() as conn:
        return conn.execute(text(
            "SELECT id FROM inter_bot_deliveries WHERE metadata_json->>'restart_retry_of'=:id"
        ), {"id": original_id}).scalars().all()


def test_fail_claim_with_restart_retry_commits_both(delivery, turn_cleanup):
    store, slug = delivery
    original = submit(store, slug)
    end_turn(store, original, audit=SHUTDOWN)
    token = _claimed(store, original)

    failed, child = fail_claim_with_restart_retry(store, original, token, "aborted")

    assert failed.status == "FAILED" and store.get(original.id).status == "FAILED"
    assert _children(store, original.id) == [child.id]
    # The outcome is durable on the FAILED row, written in the same transaction.
    outcome = {"outcome": "retried", "child_delivery_id": child.id}
    assert failed.metadata[RESTART_RETRY_OUTCOME] == outcome
    assert store.get(original.id).metadata[RESTART_RETRY_OUTCOME] == outcome


def test_failed_restart_insert_still_commits_failed(delivery, turn_cleanup, monkeypatch):
    """Vex 1231B r1 S1: a non-conflict error in the child insert rolls back
    only its SAVEPOINT; the claim's FAILED write commits."""
    store, slug = delivery
    original = submit(store, slug)
    end_turn(store, original, audit=SHUTDOWN)
    token = _claimed(store, original)
    real_enqueue = store.enqueue

    def crash_after_insert(*args, **kwargs):
        real_enqueue(*args, **kwargs)
        raise RuntimeError("crash after insert")

    monkeypatch.setattr(store, "enqueue", crash_after_insert)
    failed, child = fail_claim_with_restart_retry(store, original, token, "aborted")

    assert failed.status == "FAILED" and child is None
    assert store.get(original.id).status == "FAILED" and not _children(store, original.id)
    assert store.get(original.id).metadata[RESTART_RETRY_OUTCOME] == {
        "outcome": "error", "reason": "crash after insert"}


def test_lost_claim_writes_nothing(delivery, turn_cleanup):
    store, slug = delivery
    original = submit(store, slug)
    end_turn(store, original, audit=SHUTDOWN)
    _claimed(store, original)

    assert fail_claim_with_restart_retry(store, original, "stale-token", "aborted") == (None, None)
    assert store.get(original.id).status == "DISPATCHING" and not _children(store, original.id)
    assert RESTART_RETRY_OUTCOME not in store.get(original.id).metadata


@pytest.mark.parametrize("session_policy, audit", [
    ("continue", {"abort_request": {"source": "chat_stop"}}),
    # Vex 1231B DESIGN r2 C1: _retry refuses a reset before it reads the turn,
    # so the refusal must not be mistaken for a shutdown outcome.
    ("reset_retain_history", {"abort_request": {"source": "chat_stop"}}),
    ("reset_retain_history", None),
], ids=["user-stop", "session-reset-user-stop", "session-reset-plain-abort"])
def test_not_host_shutdown_fails_without_child(delivery, turn_cleanup, session_policy, audit):
    store, slug = delivery
    original = submit(store, slug, session_policy=session_policy)
    end_turn(store, original, audit=audit)
    token = _claimed(store, original)

    failed, child = fail_claim_with_restart_retry(store, original, token, "aborted")
    assert failed.status == "FAILED" and child is None and not _children(store, original.id)
    # Not a host shutdown: there was no restart to retry, so no outcome either.
    assert RESTART_RETRY_OUTCOME not in store.get(original.id).metadata


@pytest.mark.parametrize("options, reason", [
    ({"metadata": {"new_session_intent": {}}}, "not eligible"),
    ({"session_policy": "reset_retain_history"}, "Session-reset"),
], ids=["new-session-intent", "session-reset"])
def test_refused_restart_retry_is_recorded_on_the_failed_row(delivery, turn_cleanup, options, reason):
    """Snark 1231B: a shutdown-killed delivery that is not retried says why."""
    store, slug = delivery
    original = submit(store, slug, **options)
    end_turn(store, original, audit=SHUTDOWN)
    token = _claimed(store, original)

    failed, child = fail_claim_with_restart_retry(store, original, token, "aborted")

    assert failed.status == "FAILED" and child is None and not _children(store, original.id)
    recorded = store.get(original.id).metadata[RESTART_RETRY_OUTCOME]
    assert recorded["outcome"] == "refused" and reason in recorded["reason"]
    assert failed.metadata[RESTART_RETRY_OUTCOME] == recorded


def test_stale_host_shutdown_cause_does_not_qualify_the_next_attempt(delivery, turn_cleanup):
    """Vex 1231B r1 S2: attempt 1's cause is dropped by the re-arm, so an
    attempt 2 that merely aborted is not restart-retried."""
    from llm_bawt.service.turn_logs import TurnLogStore

    store, slug = delivery
    turns = TurnLogStore(store.config)
    original = submit(store, slug)

    def arm(attempt):
        with store.engine.begin() as conn:
            write_reservation(conn, turn_id=original.turn_id, bot_id=original.target_bot_id,
                              user_id="nick", user_prompt=original.message,
                              request_json={"messages": [], "attempt": attempt},
                              trigger_message_id=original.user_message_id)

    arm(1)
    turns.update_turn(turn_id=original.turn_id, status="aborted", end_reason="aborted",
                      abort_cause="host_shutdown")
    assert restart_abort_cause(turns.get_turn(original.turn_id)) == "host_shutdown"

    arm(2)  # the delivery re-arm starts attempt 2 on the same row
    turns.update_turn(turn_id=original.turn_id, status="aborted", end_reason="aborted")
    turn = turns.get_turn(original.turn_id)
    assert (turn.status, turn.ended_at is not None) == ("aborted", True)
    assert restart_abort_cause(turn) is None

    store.mark_failed(original.id, "streamed durable turn ended with status=aborted")
    assert restart_retry_delivery(store, original.id) is None
    assert not _children(store, original.id)


def test_restart_child_killed_again_stays_failed(delivery, turn_cleanup):
    """Depth one: the child is never restart-retried, so a second restart is visible."""
    store, slug = delivery
    original = submit(store, slug)
    store.mark_failed(original.id, "aborted")
    end_turn(store, original, audit=SHUTDOWN)
    child, _ = restart_retry_delivery(store, original.id)
    store.mark_failed(child.id, "aborted again")
    end_turn(store, child, audit=SHUTDOWN)
    assert restart_retry_delivery(store, child.id) is None
    assert store.get(child.id).status == "FAILED" and not _children(store, child.id)


# --- TASK-1231B: the dispatcher's aborted branch, end to end -----------------

class _Subscriber:
    def __init__(self):
        self.events = []

    async def publish_tool_event(self, bot_id, user_id, event):
        self.events.append((bot_id, user_id, event))


def _dispatch_aborted_turn(store, record):
    """Run the real ``_dispatch`` for a claimed delivery whose turn ended aborted."""
    from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher

    async def stream(_request):
        yield "data: [DONE]\n\n"

    turn = SimpleNamespace(status="aborted", ended_at=datetime.now(timezone.utc),
                           error_text="turn aborted", response_text="", model="m")
    subscriber = _Subscriber()
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.store = store
    dispatcher.service = SimpleNamespace(
        config=SimpleNamespace(DEFAULT_USER="default-user"), chat_completion_stream=stream,
        _turn_log_store=SimpleNamespace(get_turn=lambda _id: turn), _redis_subscriber=subscriber,
    )
    dispatcher._heartbeat = lambda *_a: asyncio.sleep(3600)
    asyncio.run(dispatcher._dispatch(store.get(record.id)))
    return [event for _bot, _user, event in subscriber.events]


def test_dispatcher_restart_retries_a_host_shutdown_abort(delivery, turn_cleanup):
    store, slug = delivery
    original = submit(store, slug)
    end_turn(store, original, audit=SHUTDOWN)
    _claimed(store, original)

    events = _dispatch_aborted_turn(store, original)

    [child_id] = _children(store, original.id)
    assert [(e["delivery_id"], e["status"]) for e in events] == [
        (original.id, "FAILED"), (child_id, "QUEUED")]
    assert events[1]["retry_of_delivery_id"] == original.id
    assert store.get(original.id).metadata[RESTART_RETRY_OUTCOME]["child_delivery_id"] == child_id


@pytest.mark.parametrize("audit, status", [
    ({**SHUTDOWN, "abort_request": {"source": "chat_stop"}}, "CANCELLED"),
    ({}, "FAILED"),
], ids=["user-stop", "plain-abort"])
def test_dispatcher_never_restart_retries_other_aborts(delivery, turn_cleanup, audit, status):
    store, slug = delivery
    original = submit(store, slug)
    end_turn(store, original, audit=audit)
    _claimed(store, original)

    events = _dispatch_aborted_turn(store, original)

    assert [(e["delivery_id"], e["status"]) for e in events] == [(original.id, status)]
    assert not _children(store, original.id)
    assert RESTART_RETRY_OUTCOME not in store.get(original.id).metadata


def test_restart_child_of_an_ordinary_chat_send_is_published_to_its_user(delivery, turn_cleanup):
    """The child is system-authored, but its lifecycle belongs to the user who sent it."""
    store, slug = delivery
    original = store.enqueue(
        sender_bot_id=slug, target_bot_id=slug, message="hi", author=AuthorReference.user("nick"),
        payload={"messages": [{"role": "user", "content": "hi"}], "bot_id": slug},
        idempotency_key="chat-original",
        metadata={"ordinary_chat": {"session_id": "s-1", "stream": True}},
    )[0]
    end_turn(store, original, audit=SHUTDOWN)
    _claimed(store, original)

    events = _dispatch_aborted_turn(store, original)

    child_event = events[-1]
    assert child_event["status"] == "QUEUED" and child_event["delivery_id"] != original.id
    assert child_event["user_id"] == "nick" and child_event["ordinary_chat"] is True
    assert child_event["bot_id"] == slug and child_event["session_id"] == "s-1"
