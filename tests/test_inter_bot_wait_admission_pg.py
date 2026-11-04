"""PostgreSQL tests for one send-and-wait per target (TASK-1253)."""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.inter_bot_wait_admission import (
    TARGET_WAIT_IN_PROGRESS,
    WAIT_WOULD_DEADLOCK,
    WaitAdmissionRefused,
)
from llm_bawt.service.turn_admission import lock_target
from tests.pg_lock_wait import wait_blocked_by
from tests.test_turn_gates_pg import gated  # noqa: F401

pytestmark = pytest.mark.usefixtures("private_app_pg")


def _name(ctx, role):
    name = f"test-wait-{role}-{uuid.uuid4().hex[:10]}"
    ctx.senders.append(name)
    return name


def _send(ctx, sender, target, *, waited=True, connection=None):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore

    payload = {"messages": [{"role": "user", "content": "hi"}], "bot_id": target, "user": "nick"}
    if waited:
        payload["reply_mode"] = "waited"
    record, _ = InterBotDeliveryStore(ctx.store.config).enqueue(
        sender_bot_id=sender, target_bot_id=target, message="hi", payload=payload,
        connection=connection,
    )
    return record


def _finish(ctx, delivery_id):
    with ctx.store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE inter_bot_deliveries SET status='DELIVERED' WHERE id=:id"),
                     {"id": delivery_id})


def _count(ctx, target):
    with ctx.store.engine.connect() as conn:
        return conn.execute(sa_text(
            "SELECT count(*) FROM inter_bot_deliveries WHERE target_bot_id=:t"
        ), {"t": target}).scalar_one()


def test_second_waiter_is_refused_with_the_current_waiter(gated):
    a, b, target = _name(gated, "a"), _name(gated, "b"), _name(gated, "t")
    first = _send(gated, a, target)
    with pytest.raises(WaitAdmissionRefused) as refused:
        _send(gated, b, target)
    detail = refused.value.detail
    assert detail["error"] == TARGET_WAIT_IN_PROGRESS
    assert detail["current_waiter_bot_id"] == a
    assert detail["current_delivery_id"] == first.id
    assert detail["current_age_seconds"] >= 0
    assert _count(gated, target) == 1


def test_concurrent_waiters_admit_exactly_one(gated):
    a, b, target = _name(gated, "a"), _name(gated, "b"), _name(gated, "t")
    with ThreadPoolExecutor(max_workers=1) as pool:
        with gated.store.engine.begin() as conn:
            pid = conn.execute(sa_text("SELECT pg_backend_pid()")).scalar_one()
            lock_target(conn, target)
            _send(gated, a, target, connection=conn)
            worker = pool.submit(_send, gated, b, target)
            wait_blocked_by(gated.store.engine, pid, worker)
        with pytest.raises(WaitAdmissionRefused) as refused:
            worker.result(timeout=10)
    assert refused.value.detail["error"] == TARGET_WAIT_IN_PROGRESS
    assert _count(gated, target) == 1


def test_two_cycle_wait_is_refused_as_deadlock(gated):
    a, b = _name(gated, "a"), _name(gated, "b")
    first = _send(gated, a, b)
    with pytest.raises(WaitAdmissionRefused) as refused:
        _send(gated, b, a)
    assert refused.value.detail["error"] == WAIT_WOULD_DEADLOCK
    assert refused.value.detail["waiting_delivery_id"] == first.id
    assert _count(gated, a) == 0


def test_concurrent_reciprocal_waits_admit_exactly_one(gated):
    """A→B uncommitted while B→A arrives: distinct target locks, same pair lock."""
    a, b = _name(gated, "a"), _name(gated, "b")
    with ThreadPoolExecutor(max_workers=1) as pool:
        with gated.store.engine.begin() as conn:
            pid = conn.execute(sa_text("SELECT pg_backend_pid()")).scalar_one()
            lock_target(conn, b)
            first = _send(gated, a, b, connection=conn)
            worker = pool.submit(_send, gated, b, a)
            wait_blocked_by(gated.store.engine, pid, worker)
        with pytest.raises(WaitAdmissionRefused) as refused:
            worker.result(timeout=10)
    assert refused.value.detail["error"] == WAIT_WOULD_DEADLOCK
    assert refused.value.detail["waiting_delivery_id"] == first.id
    assert _count(gated, a) == 0


def test_async_send_during_a_wait_is_admitted(gated):
    a, b, target = _name(gated, "a"), _name(gated, "b"), _name(gated, "t")
    _send(gated, a, target)
    _send(gated, b, target, waited=False)
    _send(gated, target, a, waited=False)  # the reverse async callback too
    assert _count(gated, target) == 2


def test_new_wait_is_admitted_after_the_first_finishes(gated):
    a, b, target = _name(gated, "a"), _name(gated, "b"), _name(gated, "t")
    first = _send(gated, a, target)
    _finish(gated, first.id)
    second = _send(gated, b, target)
    assert second.id != first.id
