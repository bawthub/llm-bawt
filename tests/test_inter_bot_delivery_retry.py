"""Manual retry creates one new durable command only for failed bot callbacks."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.inter_bot_delivery_retry import DeliveryRetryConflict, retry_failed_delivery
from llm_bawt.utils.config import Config

pytestmark = pytest.mark.integration


@pytest.fixture
def delivery():
    store = InterBotDeliveryStore(Config())
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
