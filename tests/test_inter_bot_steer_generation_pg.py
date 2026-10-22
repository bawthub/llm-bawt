"""Steer generations against the real claim, requeue and recovery (TASK-1251).

The dispatcher's ``_steer`` runs against the PostgreSQL store with the
caching fake bridge from test_inter_bot_steer_generation.py.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.inter_bot_delivery import DELIVERED, DISPATCHING, QUEUED, STEERING
from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
from llm_bawt.service.inter_bot_dispatcher_steer import SteerGeneration
from llm_bawt.service.turn_logs import TurnLogStore
from tests.test_inter_bot_delivery import _enqueue, delivery_store  # noqa: F401  (fixture)
from tests.test_inter_bot_steer_generation import _Bridge

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def rig(delivery_store, monkeypatch):  # noqa: F811
    store, sender, target = delivery_store
    bridge = _Bridge(registers_after=None)
    monkeypatch.setattr("llm_bawt.service.routes.chat.steer_active_turn", bridge.steer_active_turn)
    monkeypatch.setattr("llm_bawt.inter_bot_reply_channel.delivery_reply_channel_note", lambda _r: None)
    # Backoff is unit-tested; here each requeue is claimable at once.
    monkeypatch.setattr(SteerGeneration, "backoff_seconds", staticmethod(lambda _n: 0.0))
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.service = SimpleNamespace(config=store.config, _turn_log_store=TurnLogStore(store.config))
    dispatcher.store = store

    async def _emit(_record):
        return None

    dispatcher._emit = _emit
    first, _ = _enqueue(store, sender, target, "start work")
    second, _ = _enqueue(store, sender, target, "mid-turn reply")
    third, _ = _enqueue(store, sender, target, "after that")
    head = store.claim_next(target, claim_owner="owner", steer_capable=True)
    assert head.id == first.id and head.status == DISPATCHING
    assert store.mark_transport_accepted(first.id, head.claim_token or "")
    return SimpleNamespace(store=store, target=target, bridge=bridge, dispatcher=dispatcher,
                           first=first, second=second, third=third)


def _claim(rig):
    return rig.store.claim_next(rig.target, claim_owner="owner", steer_capable=True)


def _steer(rig, claimed):
    asyncio.run(rig.dispatcher._steer(claimed))
    return rig.store.get(claimed.id)


def _end_turn(rig, turn_id):
    with rig.store.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE turn_logs SET status='ok', ended_at=now() WHERE id=:id"
        ), {"id": turn_id})


def test_a_steer_into_a_fallback_turn_before_it_registers_waits_on_that_turn(rig):
    """Design test 4: N started a fallback turn, N+1 steers in before its run
    registered. N+1 keeps its target and FIFO place and then lands; N+2 is
    never claimed meanwhile."""
    rig.bridge.registers_after = 2
    steering = _claim(rig)
    assert steering.id == rig.second.id and steering.status == STEERING
    assert steering.target_turn_id == rig.first.turn_id

    row = _steer(rig, steering)
    assert row.status == QUEUED and row.delivery_mode == "steer"
    assert row.target_turn_id == rig.first.turn_id
    assert row.attempt_count == 0
    assert row.metadata["steer_generation"] == 1
    assert row.metadata["source"] == "test"  # merged, not replaced

    for _ in range(2):
        again = _claim(rig)
        assert again.id == rig.second.id and again.target_turn_id == rig.first.turn_id
        row = _steer(rig, again)
    assert row.status == DELIVERED
    base = SteerGeneration.of(steering).request_id
    assert rig.bridge.request_ids == [base, f"{base}_g1", f"{base}_g2"]
    assert row.attempt_count == 1

    successor = _claim(rig)
    assert successor.id == rig.third.id and successor.status == STEERING


def test_a_genuinely_ended_target_still_falls_back(rig):
    steering = _claim(rig)
    _end_turn(rig, rig.first.turn_id)
    row = _steer(rig, steering)
    assert row.status == QUEUED and row.delivery_mode == "steer_rejected"
    assert row.target_turn_id == rig.first.turn_id
    assert row.metadata["steer_generation"] == 1
    assert row.attempt_count == 0


def test_a_crash_before_the_requeue_replays_the_cached_generation_then_advances(rig):
    """Design test 8: the dispatcher died after the bridge answered g0 and
    before the requeue. Recovery replays g0, which re-reads the cached
    rejection; that requeue records the advance. The replay is no attempt."""
    crashed = _claim(rig)
    base = SteerGeneration.of(crashed).request_id
    rig.bridge.cache[base] = "no_active_run"  # the dead dispatcher's answer
    with rig.store.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE inter_bot_deliveries SET lease_expires_at=now() - interval '1 second' WHERE id=:id"
        ), {"id": crashed.id})
    recovered = [r for r in rig.store.recover_expired() if r.id == crashed.id]
    assert recovered and recovered[0].status == QUEUED
    assert recovered[0].delivery_mode == "steer"
    assert "steer_generation" not in recovered[0].metadata
    attempts_before = recovered[0].attempt_count

    replay = _claim(rig)
    assert replay.id == crashed.id and replay.target_turn_id == rig.first.turn_id
    row = _steer(rig, replay)

    assert rig.bridge.request_ids == [base]
    assert rig.bridge.fresh == 0  # a cache re-read, never a second command
    assert row.status == QUEUED and row.delivery_mode == "steer"
    assert row.metadata["steer_generation"] == 1
    assert row.attempt_count == attempts_before
