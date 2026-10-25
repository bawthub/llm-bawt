"""Steer generations against the real claim, requeue and recovery (TASK-1251).

The dispatcher's ``_steer`` runs against the PostgreSQL store with the
caching fake bridge from test_inter_bot_steer_generation.py.

TASK-1431 S1 adds ``steer_only`` claims (steer an executing turn, else settle
as not needed with no reservation) and the fan-out batch ledger: the per-sender
rate limit under its advisory lock, and keyed replay. The unit halves are in
test_inter_bot_fanout.py.
"""
from __future__ import annotations

import asyncio
import threading
import time
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


# ---- steer_only claims (TASK-1431) ------------------------------------------

def _steer_only(store, sender, target, message="only while you are running"):
    record, _ = store.enqueue(
        sender_bot_id=sender, target_bot_id=target, message=message,
        payload={"messages": [{"role": "user", "content": message}],
                 "bot_id": target, "stream": False, "steer_only": True},
    )
    return record


def _turn_rows(store, turn_id):
    with store.engine.connect() as conn:
        return conn.execute(sa_text(
            "SELECT count(*) FROM turn_logs WHERE id=:id"), {"id": turn_id}).scalar_one()


def _start_turn(store, sender, target, *, steer_capable):
    head, _ = _enqueue(store, sender, target, "start work")
    claimed = store.claim_next(target, claim_owner="owner", steer_capable=steer_capable)
    assert claimed.id == head.id and claimed.status == DISPATCHING
    assert store.mark_transport_accepted(head.id, claimed.claim_token or "")
    return claimed


def test_an_idle_target_settles_steer_only_as_not_needed_without_a_turn(delivery_store):  # noqa: F811
    store, sender, target = delivery_store
    row = _steer_only(store, sender, target)
    after, _ = _enqueue(store, sender, target, "an ordinary send behind it")
    settled = store.claim_next(target, claim_owner="owner", steer_capable=True)
    assert settled.id == row.id
    assert (settled.status, settled.delivery_mode, settled.last_error) == (
        "CANCELLED", "skipped", "not_needed:no_active_run")
    assert settled.attempt_count == row.attempt_count and settled.claim_token is None
    assert _turn_rows(store, row.turn_id) == 0  # never reserved, never started
    nxt = store.claim_next(target, claim_owner="owner", steer_capable=True)
    assert nxt.id == after.id and nxt.status == DISPATCHING  # the FIFO moves on


def test_an_executing_target_takes_a_steer_only_row_as_a_steer(delivery_store):  # noqa: F811
    store, sender, target = delivery_store
    running = _start_turn(store, sender, target, steer_capable=True)
    row = _steer_only(store, sender, target)
    steering = store.claim_next(target, claim_owner="owner", steer_capable=True)
    assert steering.id == row.id and steering.status == STEERING
    assert steering.target_turn_id == running.turn_id
    assert _turn_rows(store, row.turn_id) == 0


def test_a_backend_that_cannot_steer_settles_steer_only_even_while_busy(delivery_store):  # noqa: F811
    store, sender, target = delivery_store
    _start_turn(store, sender, target, steer_capable=False)
    row = _steer_only(store, sender, target)
    settled = store.claim_next(target, claim_owner="owner", steer_capable=False)
    assert settled.id == row.id
    assert (settled.status, settled.last_error) == ("CANCELLED", "not_needed:unsupported_backend")
    assert _turn_rows(store, row.turn_id) == 0


# ---- fan-out batch ledger (TASK-1431) ---------------------------------------

@pytest.fixture
def batches(delivery_store, monkeypatch):  # noqa: F811
    """The real batch loop and ledger; ``_admit_one`` reduced to its enqueue."""
    from llm_bawt import bot_tags, inter_bot_send as send
    from llm_bawt.bot_tags.facade import TagSnapshot
    from llm_bawt.message_authorship import AuthorReference

    store, sender, target = delivery_store
    others = [f"{target}-b", f"{target}-c"]
    bots = [SimpleNamespace(slug=slug, bot_type="agent", agent_backend="claude-code",
                            harness="claude-code")
            for slug in (sender, target, *others)]
    rows = [{"slug": bot.slug, "include_in_global_search": True} for bot in bots]

    class Tags:
        def __init__(self, _engine):
            pass

        def snapshot(self):
            return TagSnapshot.from_rows(rows)

    monkeypatch.setattr(bot_tags, "BotTags", Tags)
    monkeypatch.setattr(send, "BotManager", lambda _cfg: SimpleNamespace(list_bots=lambda: bots))

    async def admit_one(_self, body, _request=None):
        record, duplicate = store.enqueue(
            sender_bot_id=body.sender_bot_id, target_bot_id=body.target_bot_id,
            message=body.message, author=AuthorReference.bot(body.sender_bot_id),
            payload={"messages": [{"role": "user", "content": body.message}],
                     "bot_id": body.target_bot_id, "stream": False},
            idempotency_key=body.idempotency_key, metadata=body.metadata,
        )
        await asyncio.sleep(0.02)  # the real _admit_one awaits _emit and yields the loop
        return 202, {**record.to_api(duplicate=duplicate), "success": True}

    monkeypatch.setattr(send.DeliverySender, "_admit_one", admit_one)
    sender_api = send.DeliverySender(lambda: SimpleNamespace(config=store.config),
                                     dispatcher=lambda: SimpleNamespace(store=store))

    def body_for(key, targets=None):
        return send.BatchDeliveryCreate(sender_bot_id=sender, message="all hands",
                                        targets=targets or [target, *others],
                                        idempotency_key=key)

    def submit(key, targets=None):
        return asyncio.run(sender_api.submit(body_for(key, targets)))

    def on_one_loop(*keys):
        """Submit concurrently on ONE event loop, as one app process serves them."""
        async def each(key):
            try:
                return await sender_api.submit(body_for(key))
            except send.DeliveryRefused as exc:
                return exc.status_code, exc.detail

        async def gather():
            return await asyncio.gather(*(each(key) for key in keys))

        return asyncio.run(gather())

    def count():
        with store.engine.connect() as conn:
            return conn.execute(sa_text(
                "SELECT count(*) FROM inter_bot_deliveries WHERE sender_bot_id=:s"
            ), {"s": sender}).scalar_one()

    return SimpleNamespace(submit=submit, on_one_loop=on_one_loop, count=count, send=send,
                           targets=sorted([target, *others]), sender=sender, engine=store.engine)


def test_a_bots_fourth_batch_in_the_window_is_rate_limited(batches):
    for key in ("k1", "k2", "k3"):
        status, result = batches.submit(key)
        assert status == 202 and len(result["deliveries"]) == 3
    with pytest.raises(batches.send.DeliveryRefused) as refused:
        batches.submit("k4")
    assert refused.value.status_code == 429
    assert refused.value.detail["error"] == "fanout_rate_limited"
    assert batches.count() == 9  # the refused batch wrote nothing


def test_a_keyed_retry_replays_its_rows_and_is_not_a_new_batch(batches):
    status, first = batches.submit("k1")
    assert status == 202
    batches.submit("k2")
    batches.submit("k3")
    status, again = batches.submit("k1")  # the window is full; a retry is not new
    assert status == 202 and again["batch_id"] == first["batch_id"]
    assert [d["delivery_id"] for d in again["deliveries"]] == [
        d["delivery_id"] for d in first["deliveries"]]
    assert all(d["duplicate"] for d in again["deliveries"])
    assert sorted(d["target_bot_id"] for d in again["deliveries"]) == batches.targets
    assert batches.count() == 9


def test_a_reused_key_with_different_targets_is_a_conflict(batches):
    batches.submit("k1")
    with pytest.raises(batches.send.DeliveryRefused) as refused:
        batches.submit("k1", targets={"all_agents": True})
    assert refused.value.status_code == 409
    assert refused.value.detail["error"] == "batch_conflict"
    assert batches.count() == 3


def test_concurrent_batches_cannot_both_take_the_last_slot(batches):
    batches.submit("k1")
    batches.submit("k2")
    start = threading.Barrier(2)
    outcomes: list[int] = []

    def race(key):
        start.wait()
        try:
            outcomes.append(batches.submit(key)[0])
        except batches.send.DeliveryRefused as exc:
            outcomes.append(exc.status_code)

    threads = [threading.Thread(target=race, args=(key,)) for key in ("k3", "k4")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert sorted(outcomes) == [202, 429]
    assert batches.count() == 9


def test_same_sender_batches_on_one_loop_serialize_without_blocking_it(batches):
    """Loopy r1 B1: two submits on one event loop must not deadlock the loop."""
    batches.submit("k1")
    batches.submit("k2")
    started = time.monotonic()
    (first, accepted), (second, refused) = batches.on_one_loop("k3", "k4")
    # timing-ok: hang guard: a deadlocked loop never returns; a pass takes milliseconds
    assert time.monotonic() - started < 5
    assert first == 202 and len(accepted["deliveries"]) == 3
    assert second == 429 and refused["error"] == "fanout_rate_limited"
    assert batches.count() == 9


def test_a_keyed_retry_racing_its_original_on_one_loop_replays(batches):
    (first, original), (second, again) = batches.on_one_loop("k1", "k1")
    assert (first, second) == (202, 202)
    assert [d["delivery_id"] for d in again["deliveries"]] == [
        d["delivery_id"] for d in original["deliveries"]]
    assert all(d["duplicate"] for d in again["deliveries"])
    assert batches.count() == 3


def test_a_batch_held_by_another_process_is_refused_busy_not_waited_on(batches):
    with batches.engine.connect() as other, other.begin():
        other.execute(sa_text(
            "SELECT pg_advisory_xact_lock(hashtext('inter-bot-fanout'), hashtext(:s))"
        ), {"s": batches.sender})
        started = time.monotonic()
        with pytest.raises(batches.send.DeliveryRefused) as refused:
            batches.submit("k1")
        # timing-ok: hang guard on a real PG lock: waiting blocks while `other` holds it; a refusal is immediate
        assert time.monotonic() - started < 5
    assert refused.value.status_code == 429
    assert refused.value.detail["error"] == "fanout_busy"
    assert refused.value.detail["retryable"] is True
    assert batches.count() == 0
