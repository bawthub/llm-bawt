"""PostgreSQL tests for a queued chat send's wait reason (TASK-1676).

A QUEUED ordinary chat row says why it waits: a live system gate (quiesce or
freeze) or the target's running turn. Every gate here is scoped to this
test's own bots.
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.service.delivery_wait_reason import annotate_rows, wait_reason
from llm_bawt.service.gate_change_republisher import next_expiry_wait, republish_waiting
from llm_bawt.service.inter_bot_delivery_events import lifecycle_event
from llm_bawt.service.turn_gates import GATE_CHANGED_CHANNEL, GateScope, release_gates
from llm_bawt.service.user_stop_hold import hold_spec
from tests.test_turn_gates_pg import _activate, _open, _spec, gated  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

HOLD = {"kind": "system_hold", "gate": "quiesce"}
FREEZE = {"kind": "system_hold", "gate": "freeze"}
TURN = {"kind": "turn_running", "gate": None}


def _reason(ctx, bot):
    with ctx.store.engine.connect() as conn:
        return wait_reason(conn, bot)


def _chat_row(ctx, bot):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    from llm_bawt.message_authorship import AuthorReference

    deliveries = InterBotDeliveryStore(ctx.store.config)
    ctx.senders.append(bot)
    record, _ = deliveries.enqueue(
        sender_bot_id=bot, target_bot_id=bot, author=AuthorReference.user("test-user"),
        message="hi", payload={"messages": [{"role": "user", "content": "hi"}], "bot_id": bot},
        user_message_id=str(uuid.uuid4()), metadata={"ordinary_chat": {"session_id": None}},
        producer="ordinary",
    )
    return deliveries, record


def test_idle_target_under_live_quiesce_is_a_system_hold(gated):
    bot = gated.bot()
    owner = gated.owner()
    _activate(gated.store, _spec(owner, GateScope.for_bots(bot)))
    assert _reason(gated, bot) == HOLD
    with gated.store.engine.begin() as conn:
        release_gates(conn, owner_id=owner)
    assert _reason(gated, bot) == TURN


def test_busy_target_with_no_gate_is_turn_running(gated):
    bot = gated.bot()
    _open(gated.store, bot)
    assert _reason(gated, bot) == TURN


def test_freeze_beats_quiesce_and_names_itself(gated):
    bot = gated.bot()
    _activate(gated.store, _spec(gated.owner(), GateScope.for_bots(bot)))
    _activate(gated.store, _spec(gated.owner(), GateScope.for_bots(bot), mode="deny", owner_kind="operator"))
    assert _reason(gated, bot) == FREEZE


def test_user_stop_hold_does_not_hold_the_users_own_chat(gated):
    bot = gated.bot()
    spec = hold_spec(bot, 600)
    gated.owners.append(spec.owner_id)
    _activate(gated.store, spec)
    assert _reason(gated, bot) == TURN


def test_hydrate_rows_and_lifecycle_event_carry_the_reason(gated):
    bot = gated.bot()
    deliveries, record = _chat_row(gated, bot)
    _activate(gated.store, _spec(gated.owner(), GateScope.for_bots(bot)))
    views = annotate_rows(deliveries.engine, [record], [record.to_api()])
    assert views[0]["wait_reason"] == HOLD
    event = asyncio.run(lifecycle_event(deliveries.engine, record, default_user="u"))
    assert event["wait_reason"] == HOLD


def _await_notifies(listener, payload: str, count: int) -> list[str]:
    """Read notifies until ``count`` carry ``payload``. A single poll() races the
    second NOTIFY still in flight on the socket (CI run 40895186485)."""
    import select

    while True:
        listener.poll()
        payloads = [n.payload for n in listener.notifies]
        if payloads.count(payload) >= count:
            return payloads
        # timing-ok: hang guard only; returning lets the caller's assert report the shortfall
        if not select.select([listener], [], [], 10)[0]:
            return payloads


def test_gate_open_and_release_notify_and_republish(gated):
    bot = gated.bot()
    deliveries, record = _chat_row(gated, bot)
    raw = gated.store.engine.raw_connection()
    listener = raw.driver_connection
    raw.detach()
    listener.autocommit = True
    try:
        with listener.cursor() as cursor:
            cursor.execute(f"LISTEN {GATE_CHANGED_CHANNEL}")
        owner = gated.owner()
        gate = _activate(gated.store, _spec(owner, GateScope.for_bots(bot), ttl=60)).gate
        assert next_expiry_wait(gated.store.engine) is not None
        with gated.store.engine.begin() as conn:
            release_gates(conn, owner_id=owner)
        payloads = _await_notifies(listener, str(gate.id), 2)
        assert payloads.count(str(gate.id)) == 2  # opened, then released
    finally:
        listener.close()
    announced, woke = [], []
    dispatcher = SimpleNamespace(store=deliveries, announce_lifecycle=announced.append,
                                 wake=lambda *a: woke.append(True))
    republish_waiting(dispatcher)
    assert record.id in [row.id for row in announced]
    assert woke


def test_rolled_back_gate_is_never_announced(gated):
    from llm_bawt.service.turn_admission import activate_hold

    bot = gated.bot()
    raw = gated.store.engine.raw_connection()
    listener = raw.driver_connection
    raw.detach()
    listener.autocommit = True
    try:
        with listener.cursor() as cursor:
            cursor.execute(f"LISTEN {GATE_CHANGED_CHANNEL}")
        conn = gated.store.engine.connect()
        tx = conn.begin()
        activate_hold(conn, _spec(gated.owner(), GateScope.for_bots(bot)))
        tx.rollback()
        conn.close()
        # Fence: a committed sentinel NOTIFY queues after any notify the hold
        # committed, so once it arrives a late gate NOTIFY cannot still be in flight.
        sentinel = f"fence-{uuid.uuid4()}"
        with gated.store.engine.begin() as probe:
            probe.execute(sa_text("SELECT pg_notify(:ch, :p)"), {"ch": GATE_CHANGED_CHANNEL, "p": sentinel})
        payloads = _await_notifies(listener, sentinel, 1)
        assert payloads == [sentinel]
    finally:
        listener.close()


def test_republish_pages_past_one_batch(gated, monkeypatch):
    """Codex r1 B2: every waiting row relabels, not just the first batch."""
    from llm_bawt.service import gate_change_republisher as rep

    bot = gated.bot()
    _, first = _chat_row(gated, bot)
    deliveries, second = _chat_row(gated, bot)
    monkeypatch.setattr(rep, "REPUBLISH_BATCH", 1)
    announced = []
    dispatcher = SimpleNamespace(store=deliveries, announce_lifecycle=announced.append, wake=lambda *a: None)
    rep.republish_waiting(dispatcher)
    ids = [row.id for row in announced]
    assert first.id in ids and second.id in ids
