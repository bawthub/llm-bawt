"""Manual retry creates one new durable command only for failed bot callbacks."""

from __future__ import annotations

import threading
import time
import uuid

import pytest
from sqlalchemy import event, text

from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.inter_bot_delivery_retry import DeliveryRetryConflict, retry_failed_delivery
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
