"""PostgreSQL tests for TASK-1327: the user-Stop queue hold.

Gate primitives first (IR1): a user_stop decision made after a lock wait reads
the wall clock (``clock_timestamp()``), never the transaction-start ``now()``.
Then the hold itself (D1): open-or-extend, the user's own chat running past
it, Resume, Cancel all, and Stop opening it. Every gate is scoped to this
test's own bots, so parallel PG tests are never held by it.
"""

from __future__ import annotations

import threading

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.service.turn_admission import activate_hold, lock_target
from llm_bawt.service.turn_gates import (
    OWNER_OPERATOR,
    GateActor,
    GateScope,
    extend_gate,
    gate_blocks,
    gate_is_live,
    live_gate_for_owner,
    release_gates,
)
from tests.test_delivery_claim_gate_pg import _claim, _enqueue, _row
from tests.pg_lock_wait import wait_blocked_by
from tests.test_turn_gates_pg import _activate, _spec, gated  # noqa: F401  (gated is a fixture)

pytestmark = pytest.mark.usefixtures("private_app_pg")


def _user_stop_owner(gated, bot):
    owner_id = f"user_stop:{bot}"
    gated.owners.append(owner_id)
    return owner_id


def _after_lock_wait(gated, bot, gate_id, probe):
    """Run ``probe(conn)`` in a transaction that began, then waited on ``bot``'s lock
    while the lock holder moved the gate's expiry into the past (Loopy's IR1 repro)."""
    began = threading.Event()
    result, errors = [], []

    def waiter():
        try:
            with gated.store.engine.begin() as conn:
                conn.execute(sa_text("SET LOCAL statement_timeout='5s'"))
                conn.execute(sa_text("SELECT now()"))  # pins the transaction clock
                began.set()
                lock_target(conn, bot)
                result.append(probe(conn))
        except Exception as exc:  # surfaced by the assert below
            errors.append(repr(exc))

    with gated.store.engine.begin() as conn:
        lock_target(conn, bot)
        thread = threading.Thread(target=waiter)
        thread.start()
        assert began.wait(10)
        conn.execute(sa_text(
            "UPDATE system_gates SET expires_at=clock_timestamp()+interval '0.1 seconds' WHERE id=:id"
        ), {"id": gate_id})
        conn.execute(sa_text("SELECT pg_sleep(0.2)"))
    thread.join(6)
    assert not thread.is_alive()
    assert not errors, errors
    return result[0]


@pytest.mark.integration
def test_fresh_liveness_sees_expiry_that_passed_during_the_lock_wait(gated):
    bot = gated.bot()
    owner = _user_stop_owner(gated, bot)
    gate = _activate(gated.store, _spec(owner, GateScope.for_bots(bot), owner_kind=OWNER_OPERATOR)).gate

    stale, fresh, by_owner = _after_lock_wait(gated, bot, gate.id, lambda conn: (
        gate_is_live(conn, gate.id),
        gate_is_live(conn, gate.id, fresh=True),
        live_gate_for_owner(conn, owner, fresh=True),
    ))

    assert stale is True  # the transaction-start clock still sees it live (the defect IR1 names)
    assert fresh is False
    assert by_owner is None


@pytest.mark.integration
def test_fresh_extend_never_revives_a_gate_that_expired_during_the_lock_wait(gated):
    bot = gated.bot()
    owner = _user_stop_owner(gated, bot)
    gate = _activate(gated.store, _spec(owner, GateScope.for_bots(bot), owner_kind=OWNER_OPERATOR)).gate

    extended = _after_lock_wait(
        gated, bot, gate.id, lambda conn: extend_gate(conn, gate.id, owner_id=owner, ttl_seconds=600, fresh=True))

    assert extended is False
    with gated.store.engine.begin() as conn:
        assert gate_is_live(conn, gate.id, fresh=True) is False


@pytest.mark.integration
def test_fresh_extend_and_open_measure_ttl_from_the_wall_clock(gated):
    bot = gated.bot()
    owner = _user_stop_owner(gated, bot)
    with gated.store.engine.begin() as conn:
        conn.execute(sa_text("SELECT pg_sleep(0.5)"))  # the transaction clock falls behind
        gate = activate_hold(conn, _spec(owner, GateScope.for_bots(bot), owner_kind=OWNER_OPERATOR, ttl=1),
                             fresh_clock=True).gate
        opened_lead = conn.execute(sa_text(
            "SELECT EXTRACT(EPOCH FROM expires_at - now()) FROM system_gates WHERE id=:id"), {"id": gate.id}).scalar_one()
        conn.execute(sa_text("SELECT pg_sleep(0.5)"))
        assert extend_gate(conn, gate.id, owner_id=owner, ttl_seconds=1, fresh=True) is True
        extended_lead = conn.execute(sa_text(
            "SELECT EXTRACT(EPOCH FROM expires_at - now()) FROM system_gates WHERE id=:id"), {"id": gate.id}).scalar_one()

    # now() is the transaction start, so a wall-clock TTL lands ~0.5s / ~1.0s past now()+1.
    assert float(opened_lead) >= 1.4
    assert float(extended_lead) >= 1.9


@pytest.mark.integration
def test_default_clock_is_unchanged_for_other_gates(gated):
    bot = gated.bot()
    owner = gated.owner()
    with gated.store.engine.begin() as conn:
        conn.execute(sa_text("SELECT pg_sleep(0.3)"))
        gate = activate_hold(conn, _spec(owner, GateScope.for_bots(bot), ttl=1)).gate
        lead = conn.execute(sa_text(
            "SELECT EXTRACT(EPOCH FROM expires_at - now()) FROM system_gates WHERE id=:id"), {"id": gate.id}).scalar_one()
    assert abs(float(lead) - 1.0) < 0.05


@pytest.mark.integration
def test_live_gate_for_owner_matches_the_exact_owner_and_skips_released(gated):
    bot, other = gated.bot(), gated.bot()
    owner = _user_stop_owner(gated, bot)
    prefixed = f"{owner}-other"  # this owner id is a prefix of it: matching must be exact
    gated.owners.append(prefixed)
    other_gate = _activate(gated.store, _spec(prefixed, GateScope.for_bots(other),
                                              owner_kind=OWNER_OPERATOR)).gate
    with gated.store.engine.begin() as conn:
        assert live_gate_for_owner(conn, owner, fresh=True) is None
    gate = _activate(gated.store, _spec(owner, GateScope.for_bots(bot), owner_kind=OWNER_OPERATOR)).gate
    with gated.store.engine.begin() as conn:
        found = live_gate_for_owner(conn, owner, fresh=True)
        assert found is not None and found.id == gate.id and found.id != other_gate.id
        release_gates(conn, owner_id=owner)
        assert live_gate_for_owner(conn, owner, fresh=True) is None


@pytest.mark.integration
def test_gate_blocks_exclude_owner_drops_only_that_owner_before_choosing_the_strictest(gated):
    bot = gated.bot()
    owner = _user_stop_owner(gated, bot)
    user_stop = _activate(gated.store, _spec(owner, GateScope.for_bots(bot), owner_kind=OWNER_OPERATOR,
                                             ttl=600)).gate
    actor = GateActor("delivery_claim", None)
    with gated.store.engine.begin() as conn:
        lock_target(conn, bot)
        assert gate_blocks(conn, bot, actor).id == user_stop.id
        assert gate_blocks(conn, bot, actor, exclude_owner_id=owner) is None

    # A shorter job HOLD loses the strictest choice to user_stop, yet still blocks once user_stop is excluded.
    job = _activate(gated.store, _spec(gated.owner(), GateScope.for_bots(bot), ttl=60)).gate
    with gated.store.engine.begin() as conn:
        lock_target(conn, bot)
        assert gate_blocks(conn, bot, actor).id == user_stop.id
        assert gate_blocks(conn, bot, actor, exclude_owner_id=owner).id == job.id


# --- the hold (D1) ----------------------------------------------------------


def _hold(gated, bot, ttl=600):
    from llm_bawt.service.user_stop_hold import hold_owner, open_or_extend

    gated.owners.append(hold_owner(bot))
    with gated.store.engine.begin() as conn:
        return open_or_extend(conn, bot, ttl)


def _user_chat(gated, deliveries, bot, message="nick, while the bot is held"):
    """The user's own chat row as ``ordinary_turn_queue`` writes it: user-authored,
    self-addressed, ``ordinary_chat`` metadata."""
    from llm_bawt.message_authorship import AuthorReference

    gated.senders.append(bot)
    record, _ = deliveries.enqueue(
        sender_bot_id=bot, target_bot_id=bot, author=AuthorReference.user("nick"), message=message,
        payload={"messages": [{"role": "user", "content": message}], "bot_id": bot, "user": "nick",
                 "stream": True},
        metadata={"ordinary_chat": {"session_id": None, "stream": True}},
    )
    return record


def _finish(deliveries, record):
    """The claimed row's turn ran and ended."""
    with deliveries.engine.begin() as conn:
        conn.execute(sa_text("UPDATE inter_bot_deliveries SET status='DELIVERED' WHERE id=:id"),
                     {"id": record.id})
        conn.execute(sa_text("UPDATE turn_logs SET status='ok', ended_at=now() WHERE id=:id"),
                     {"id": record.turn_id})


def _live_hold(gated, bot):
    from llm_bawt.service.user_stop_hold import hold_owner

    with gated.store.engine.begin() as conn:
        return live_gate_for_owner(conn, hold_owner(bot), fresh=True)


def _stop(gated, turn_id, source="chat_stop"):
    import asyncio
    from types import SimpleNamespace

    from llm_bawt.service.turn_abort import abort_turn

    service = SimpleNamespace(_turn_log_store=gated.store, config=gated.store.config, _redis_subscriber=None)
    return asyncio.run(abort_turn(service, gated.store.get_turn(turn_id), source=source, peer=None))


@pytest.mark.integration
def test_user_chat_runs_past_the_hold_and_the_backlog_waits_for_resume(gated):
    from llm_bawt.service.user_stop_hold import resume

    bot = gated.bot()
    deliveries, held = _enqueue(gated, bot, "bot backlog")
    user = _user_chat(gated, deliveries, bot)
    _hold(gated, bot)
    before = _row(deliveries, held.id)

    claimed = _claim(deliveries, bot)

    assert claimed is not None and claimed.id == user.id
    after = _row(deliveries, held.id)
    assert after["status"] == "QUEUED" and after["attempt_count"] == before["attempt_count"]
    _finish(deliveries, claimed)
    assert _claim(deliveries, bot) is None  # the backlog is still held

    assert resume(gated.store.engine, bot)
    resumed = _claim(deliveries, bot)
    assert resumed is not None and resumed.id == held.id


@pytest.mark.integration
def test_a_delayed_bot_head_does_not_hide_a_ready_user_row(gated):
    """Loopy S1: the bypass head is chosen before the readiness return."""
    bot = gated.bot()
    deliveries, delayed = _enqueue(gated, bot, "retry backoff")
    with deliveries.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE inter_bot_deliveries SET available_at=now() + interval '1 hour' WHERE id=:id"
        ), {"id": delayed.id})
    user = _user_chat(gated, deliveries, bot)
    assert _claim(deliveries, bot) is None  # no hold: strict FIFO waits on the delayed head

    _hold(gated, bot)
    claimed = _claim(deliveries, bot)

    assert claimed is not None and claimed.id == user.id
    assert _row(deliveries, delayed.id)["status"] == "QUEUED"


@pytest.mark.integration
@pytest.mark.parametrize("mode,ttl", [("hold", 60), ("deny", 600)])
def test_every_other_gate_still_holds_the_user_row(gated, mode, ttl):
    """A shorter job HOLD loses the strictest choice to user_stop, yet still holds."""
    bot = gated.bot()
    deliveries, _ = _enqueue(gated, bot)
    user = _user_chat(gated, deliveries, bot)
    _hold(gated, bot)
    _activate(gated.store, _spec(gated.owner(), GateScope.for_bots(bot), mode=mode, ttl=ttl))

    assert _claim(deliveries, bot) is None
    assert _row(deliveries, user.id)["status"] == "QUEUED"


@pytest.mark.integration
def test_an_unaccepted_dispatch_fences_the_bypass(gated):
    """Claimed work that has not reached its turn is never jumped."""
    bot = gated.bot()
    deliveries, in_flight = _enqueue(gated, bot)
    user = _user_chat(gated, deliveries, bot)
    _hold(gated, bot)
    with deliveries.engine.begin() as conn:
        conn.execute(sa_text("UPDATE inter_bot_deliveries SET status='DISPATCHING' WHERE id=:id"),
                     {"id": in_flight.id})

    assert _claim(deliveries, bot) is None
    assert _row(deliveries, user.id)["status"] == "QUEUED"


@pytest.mark.integration
def test_a_hold_that_expires_during_the_lock_wait_takes_the_normal_path(gated):
    """IR1 at the claim: a lapsed hold grants no bypass, so the FIFO head is next."""
    bot = gated.bot()
    deliveries, head = _enqueue(gated, bot)
    user = _user_chat(gated, deliveries, bot)
    gate = _hold(gated, bot)
    result, errors = {}, []

    def worker_claim():
        try:
            result["claimed"] = _claim(deliveries, bot)
        except Exception as exc:  # surfaced by the assert below
            errors.append(repr(exc))

    with gated.store.engine.begin() as holder:
        lock_target(holder, bot)
        pid = holder.execute(sa_text("SELECT pg_backend_pid()")).scalar_one()
        worker = threading.Thread(target=worker_claim)
        worker.start()
        wait_blocked_by(gated.store.engine, pid, worker, detail=lambda: (result, errors))
        holder.execute(sa_text(
            "UPDATE system_gates SET expires_at=clock_timestamp()+interval '0.1 seconds' WHERE id=:id"
        ), {"id": gate.id})
        holder.execute(sa_text("SELECT pg_sleep(0.2)"))
    worker.join(10)
    assert not worker.is_alive() and not errors, errors

    claimed = result["claimed"]
    assert claimed is None or claimed.id == head.id  # never the bypass
    assert _row(deliveries, user.id)["status"] == "QUEUED"
    if claimed is None:  # gate_blocks reads the transaction-start clock, which still saw the gate
        claimed = _claim(deliveries, bot)
        assert claimed is not None and claimed.id == head.id


@pytest.mark.integration
def test_open_or_extend_never_stacks_and_a_released_hold_reopens_as_a_new_generation(gated):
    from llm_bawt.service.user_stop_hold import HOLD_ALLOW_ACTORS, HOLD_REASON, hold_owner, resume

    bot = gated.bot()
    first = _hold(gated, bot, ttl=60)
    again = _hold(gated, bot, ttl=600)
    with gated.store.engine.begin() as conn:
        rows = conn.execute(sa_text(
            "SELECT id, owner_kind, mode, reason, allow_actors FROM system_gates"
            " WHERE owner_id=:o AND released_at IS NULL"
        ), {"o": hold_owner(bot)}).mappings().all()
    assert [row["id"] for row in rows] == [first.id] and again.id == first.id
    assert again.expires_at > first.expires_at
    assert (rows[0]["owner_kind"], rows[0]["mode"], rows[0]["reason"]) == (OWNER_OPERATOR, "hold", HOLD_REASON)
    assert set(rows[0]["allow_actors"]) == HOLD_ALLOW_ACTORS

    assert resume(gated.store.engine, bot) == [first.id]
    assert resume(gated.store.engine, bot) == []
    assert _hold(gated, bot).id != first.id


@pytest.mark.integration
def test_cancel_all_cancels_only_the_held_rows_and_releases_the_hold(gated):
    from llm_bawt.service.user_stop_hold import HOLD_CANCEL_REASON, cancel_all, snapshot

    bot = gated.bot()
    deliveries, first = _enqueue(gated, bot, "one")
    _, second = _enqueue(gated, bot, "two")
    user = _user_chat(gated, deliveries, bot)
    _, scheduled = _enqueue(gated, bot, "scheduled prompt")
    with deliveries.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE inter_bot_deliveries SET metadata_json=CAST(:m AS jsonb) WHERE id=:id"
        ), {"id": scheduled.id, "m": '{"prompt_schedule": {"schedule_id": "s"}}'})
    assert snapshot(gated.store.engine, bot) is None
    assert cancel_all(gated.store.engine, bot) is None  # nothing to cancel without a live hold
    gate = _hold(gated, bot)

    held = snapshot(gated.store.engine, bot)
    assert held.gate_id == gate.id and held.summary()["queued_count"] == 2
    assert [row["id"] for row in held.queued] == [first.id, second.id]

    cancelled = cancel_all(gated.store.engine, bot)

    assert sorted(record.id for record in cancelled) == sorted([first.id, second.id])
    assert {(record.status, record.last_error) for record in cancelled} == {("CANCELLED", HOLD_CANCEL_REASON)}
    assert _row(deliveries, user.id)["status"] == "QUEUED"
    assert _row(deliveries, scheduled.id)["status"] == "QUEUED"
    assert snapshot(gated.store.engine, bot) is None
    assert cancel_all(gated.store.engine, bot) is None


@pytest.mark.integration
def test_stop_on_a_reservation_cancels_it_and_holds_the_rest(gated):
    from llm_bawt.service.user_stop_hold import hold_owner, snapshot

    bot = gated.bot()
    deliveries, stopped = _enqueue(gated, bot, "reserved, never started")
    _, waiting = _enqueue(gated, bot, "queued behind it")
    gated.owners.append(hold_owner(bot))
    assert _claim(deliveries, bot).id == stopped.id

    assert _stop(gated, stopped.turn_id) == "aborted:reserved"

    assert deliveries.get(stopped.id).status == "CANCELLED"
    assert _row(deliveries, waiting.id)["status"] == "QUEUED"
    held = snapshot(gated.store.engine, bot)
    assert held is not None and [row["id"] for row in held.queued] == [waiting.id]
    assert _claim(deliveries, bot) is None


@pytest.mark.integration
def test_stop_on_a_started_turn_holds_the_queue(gated):
    from llm_bawt.service.user_stop_hold import hold_owner

    bot = gated.bot()
    deliveries, running = _enqueue(gated, bot, "started")
    gated.owners.append(hold_owner(bot))
    assert _claim(deliveries, bot).id == running.id
    with gated.store.engine.begin() as conn:  # started, with no bridge binding to abort
        conn.execute(sa_text("UPDATE turn_logs SET status='streaming' WHERE id=:id"), {"id": running.turn_id})

    assert _stop(gated, running.turn_id, source="bot_list_stop") == "aborted:unbound"

    with gated.store.engine.begin() as conn:
        lock_target(conn, bot)
        block = gate_blocks(conn, bot, GateActor("delivery_claim", None))
    assert block is not None and block.owner_id == hold_owner(bot)


@pytest.mark.integration
@pytest.mark.parametrize("source,ttl", [("unknown", 600), ("chat_stop", 0)])
def test_no_hold_for_a_non_user_stop_or_a_zero_ttl(gated, monkeypatch, source, ttl):
    from llm_bawt import runtime_setting_resolution
    from llm_bawt.service.user_stop_hold import HOLD_SETTING, hold_owner

    real = runtime_setting_resolution.resolve_global_runtime_setting
    monkeypatch.setattr(runtime_setting_resolution, "resolve_global_runtime_setting",
                        lambda config, key, fallback=None: ttl if key == HOLD_SETTING else real(config, key, fallback))
    bot = gated.bot()
    deliveries, stopped = _enqueue(gated, bot)
    _, waiting = _enqueue(gated, bot)
    gated.owners.append(hold_owner(bot))
    assert _claim(deliveries, bot).id == stopped.id

    assert _stop(gated, stopped.turn_id, source=source) == "aborted:reserved"

    assert _live_hold(gated, bot) is None
    assert _claim(deliveries, bot).id == waiting.id


@pytest.mark.integration
def test_a_failed_hold_never_fails_the_stop(gated, monkeypatch):
    from llm_bawt.service import user_stop_hold

    def broken(*_args, **_kwargs):
        raise RuntimeError("hold store down")

    monkeypatch.setattr(user_stop_hold, "open_or_extend", broken)
    bot = gated.bot()
    deliveries, stopped = _enqueue(gated, bot)
    gated.owners.append(user_stop_hold.hold_owner(bot))
    assert _claim(deliveries, bot).id == stopped.id

    assert _stop(gated, stopped.turn_id) == "aborted:reserved"

    assert deliveries.get(stopped.id).status == "CANCELLED"
    assert _live_hold(gated, bot) is None
