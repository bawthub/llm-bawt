"""PostgreSQL tests for the TASK-1233 delivery claim gate (Loopy S3).

A live hold gate keeps a durable delivery QUEUED at claim time: no attempt is
spent, nothing is written, and the first sweep after release dispatches it.
The save_turn backstop (a gate that goes live after the claim) is deferred by
the dispatcher as a refunded, delayed requeue, never an attempt or a delay-0
spin. Every gate here is scoped to this test's own bots.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.service.inter_bot_claims import turn_admission_claim
from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
from llm_bawt.service.turn_admission import SystemGateRefused, TargetBusy, lock_target
from llm_bawt.service.turn_gates import GateScope, insert_gate, release_gates
from tests.pg_lock_wait import wait_blocked_by
from tests.test_turn_gates_pg import _activate, _spec, gated  # noqa: F401

pytestmark = pytest.mark.usefixtures("private_app_pg")

REFUSALS = 3


def _enqueue(ctx, bot, message="callback"):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore

    deliveries = InterBotDeliveryStore(ctx.store.config)
    sender = f"test-gate-sender-{uuid.uuid4().hex}"
    ctx.senders.append(sender)
    record, _ = deliveries.enqueue(
        sender_bot_id=sender, target_bot_id=bot, message=message,
        payload={"messages": [{"role": "user", "content": message}],
                 "bot_id": bot, "user": "nick", "stream": True},
    )
    return deliveries, record


def _claim(deliveries, bot):
    return deliveries.claim_next(bot, claim_owner="worker", steer_capable=False)


def _row(deliveries, delivery_id):
    with deliveries.engine.connect() as conn:
        return conn.execute(sa_text("""
            SELECT status, attempt_count, claim_token, delivery_mode, target_turn_id,
                   transport_accepted_at, available_at > now() AS deferred
            FROM inter_bot_deliveries WHERE id=:id
        """), {"id": delivery_id}).mappings().one()


def _make_due(deliveries, delivery_id):
    with deliveries.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE inter_bot_deliveries SET available_at=now(), next_retry_at=now() WHERE id=:id"
        ), {"id": delivery_id})


def _backstop_gate(ctx, bot):
    """A gate that goes live after the claim without revoking it (the save_turn backstop)."""
    spec = _spec(ctx.owner(), GateScope.for_bots(bot))
    with ctx.store.engine.begin() as conn:
        lock_target(conn, bot)
        insert_gate(conn, spec)
    return spec.owner_id


def _release(ctx, owner_id):
    with ctx.store.engine.begin() as conn:
        assert release_gates(conn, owner_id=owner_id)


class _Durable:
    """The dispatcher's real _dispatch over the real delivery and turn stores.

    The stream does what chat_completion_stream does for a durable claim: its
    first write is save_turn under the trusted admission claim, and a gate or
    busy refusal propagates (chat_streaming re-raises SystemGateRefused when
    the request holds an admission claim). A successful start finalizes ok.
    """

    def __init__(self, ctx, deliveries):
        self.ctx, self.deliveries = ctx, deliveries
        self.refusals: list[BaseException] = []
        self.emitted: list = []
        self.dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
        self.dispatcher.store = deliveries
        self.dispatcher.service = SimpleNamespace(
            config=deliveries.config, chat_completion_stream=self._stream,
            _turn_log_store=ctx.store, _inter_bot_dispatcher=self.dispatcher,
        )

        async def heartbeat(*_a):
            await asyncio.sleep(3600)

        async def emit(value):
            self.emitted.append(value)

        self.dispatcher._heartbeat = heartbeat
        self.dispatcher._emit = emit

    async def _stream(self, request):
        record = self.deliveries.get(request.inter_bot_delivery_id)
        claim = turn_admission_claim(self.dispatcher.service, request)
        assert claim is not None
        try:
            self.ctx.store.save_turn(
                turn_id=record.turn_id, request_id="r", path="/v1/chat/completions",
                stream=True, model="m", bot_id=record.target_bot_id, user_id="nick",
                status="streaming", latency_ms=None, user_prompt=record.message,
                request_payload={"messages": []}, response_text="", tool_calls=None,
                admission_claim=claim,
            )
        except TargetBusy as exc:
            self.refusals.append(exc)
            raise
        with self.ctx.store.engine.begin() as conn:
            conn.execute(sa_text("""
                UPDATE turn_logs SET status='ok', end_reason='stop', response_text='done',
                    ended_at=now() WHERE id=:id
            """), {"id": record.turn_id})
        yield "data: [DONE]\n\n"

    def dispatch(self, record):
        asyncio.run(self.dispatcher._dispatch(record))


# ---------------------------------------------------------------------------
# Claim gate: the held head stays QUEUED, spends nothing, dispatches on release
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_held_claim_stays_queued_and_claims_after_release(gated):
    store, bot = gated.store, gated.bot()
    deliveries, record = _enqueue(gated, bot)
    gate = _activate(store, _spec(gated.owner(), GateScope.for_bots(bot)))

    for _ in range(REFUSALS):
        assert _claim(deliveries, bot) is None
    row = _row(deliveries, record.id)
    assert (row["status"], row["attempt_count"], row["claim_token"], row["delivery_mode"]) == (
        "QUEUED", 0, None, None,
    )
    assert store.get_turn(record.turn_id) is None  # no reservation written under the gate

    with store.engine.begin() as conn:
        assert release_gates(conn, owner_id=gate.gate.owner_id) == [gate.gate.id]
    claimed = _claim(deliveries, bot)
    assert claimed and claimed.id == record.id
    assert (claimed.status, claimed.attempt_count) == ("DISPATCHING", 1)
    assert store.get_turn(record.turn_id).status == "reserved"


@pytest.mark.integration
def test_held_head_blocks_the_target_fifo(gated):
    store, bot = gated.store, gated.bot()
    deliveries, first = _enqueue(gated, bot, "first")
    _, second = _enqueue(gated, bot, "second")
    gate = _activate(store, _spec(gated.owner(), GateScope.for_bots(bot)))

    assert _claim(deliveries, bot) is None
    assert _row(deliveries, second.id)["status"] == "QUEUED"
    with store.engine.begin() as conn:
        release_gates(conn, owner_id=gate.gate.owner_id)
    claimed = _claim(deliveries, bot)
    assert claimed and claimed.id == first.id


@pytest.mark.integration
def test_other_bots_gate_does_not_hold_the_claim(gated):
    store, bot = gated.store, gated.bot()
    deliveries, record = _enqueue(gated, bot)
    _activate(store, _spec(gated.owner(), GateScope.for_bots(gated.bot())))

    claimed = _claim(deliveries, bot)
    assert claimed and claimed.id == record.id


@pytest.mark.integration
def test_explicit_stop_still_settles_under_a_gate(gated):
    store, bot = gated.store, gated.bot()
    deliveries, record = _enqueue(gated, bot)
    with store.engine.begin() as conn:  # the human Stop of this delivery's own turn
        conn.execute(sa_text("""
            INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status,
                                   user_prompt, response_text, ended_at, request_json)
            VALUES (:id, now(), '/test', true, :bot, 'nick', 'aborted', '', '', now(),
                    '{"abort_request": {"source": "chat_stop"}}')
        """), {"id": record.turn_id, "bot": bot})
    _activate(store, _spec(gated.owner(), GateScope.for_bots(bot)))

    assert _claim(deliveries, bot) is None
    assert _row(deliveries, record.id)["status"] == "CANCELLED"


@pytest.mark.integration
def test_claim_takes_the_shared_target_lock(gated):
    store, bot = gated.store, gated.bot()
    deliveries, record = _enqueue(gated, bot)
    result = {}
    with store.engine.connect() as holder:
        with holder.begin():
            lock_target(holder, bot)
            pid = holder.execute(sa_text("SELECT pg_backend_pid()")).scalar_one()
            worker = threading.Thread(target=lambda: result.update(claimed=_claim(deliveries, bot)))
            worker.start()
            wait_blocked_by(store.engine, pid, worker, detail=lambda: result)
            assert "claimed" not in result
    worker.join(timeout=10)
    assert result["claimed"] and result["claimed"].id == record.id


@pytest.mark.integration
def test_rearmed_accepted_member_reclaims_under_its_gate(gated):
    """C1: re-attaching to a turn the gate captured as accepted is exempt."""
    store, bot = gated.store, gated.bot()
    deliveries, record = _enqueue(gated, bot)
    claimed = _claim(deliveries, bot)
    with store.engine.begin() as conn:  # the bridge request was bound before chat.send
        conn.execute(sa_text("UPDATE turn_logs SET agent_request_id='req-bound' WHERE id=:id"),
                     {"id": claimed.turn_id})
    gate = _activate(store, _spec(gated.owner(), GateScope.for_bots(bot)))
    assert any(m.turn_id == claimed.turn_id and m.disposition == "accepted" for m in gate.members)
    with store.engine.begin() as conn:  # recover_expired's re-arm of an expired lease
        conn.execute(sa_text("""
            UPDATE inter_bot_deliveries SET status='QUEUED', claim_token=NULL, claim_owner=NULL,
                lease_expires_at=NULL, available_at=now(), next_retry_at=now() WHERE id=:id
        """), {"id": claimed.id})

    reclaimed = _claim(deliveries, bot)
    assert reclaimed and reclaimed.id == record.id and reclaimed.status == "DISPATCHING"


@pytest.mark.integration
def test_claim_calls_lock_target_not_a_private_lock(gated, monkeypatch):
    """B2: claim_next linearizes through the one per-bot admission lock function."""
    from llm_bawt.service import turn_admission

    bot = gated.bot()
    deliveries, _ = _enqueue(gated, bot)
    calls = []
    real = turn_admission.lock_target
    monkeypatch.setattr(turn_admission, "lock_target", lambda conn, b: calls.append(b) or real(conn, b))
    assert _claim(deliveries, bot) is not None
    assert calls[:1] == [bot]


# ---------------------------------------------------------------------------
# Backstop: save_turn refuses after the claim -> refunded, delayed requeue
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_backstop_refusals_keep_attempts_flat_then_dispatch_after_release(gated):
    store, bot = gated.store, gated.bot()
    deliveries, record = _enqueue(gated, bot)
    durable = _Durable(gated, deliveries)

    for cycle in range(REFUSALS):
        claimed = _claim(deliveries, bot)
        assert claimed and claimed.id == record.id and claimed.attempt_count == 1
        owner_id = _backstop_gate(gated, bot)
        durable.dispatch(claimed)

        assert isinstance(durable.refusals[-1], SystemGateRefused)
        row = _row(deliveries, record.id)
        assert (row["status"], row["attempt_count"], row["claim_token"]) == ("QUEUED", 0, None)
        # Never accepted, so a reset retry is not mistaken for an ambiguous replay.
        assert (row["delivery_mode"], row["target_turn_id"], row["transport_accepted_at"]) == (
            None, None, None,
        )
        assert row["deferred"], "a refused start must not requeue at delay 0"
        assert store.get_turn(record.turn_id) is None  # reservation released
        _release(gated, owner_id)
        assert _claim(deliveries, bot) is None  # deferred even with the gate gone: no spin
        _make_due(deliveries, record.id)
    assert len(durable.refusals) == REFUSALS

    claimed = _claim(deliveries, bot)
    assert claimed and claimed.attempt_count == 1
    durable.dispatch(claimed)
    final = deliveries.get(record.id)
    assert (final.status, final.attempt_count) == ("DELIVERED", 1)
    assert store.get_turn(record.turn_id).status == "ok"


@pytest.mark.integration
def test_busy_target_at_save_turn_is_refunded_and_delayed(gated):
    store, bot = gated.store, gated.bot()
    deliveries, record = _enqueue(gated, bot)
    durable = _Durable(gated, deliveries)
    claimed = _claim(deliveries, bot)
    with store.engine.begin() as conn:  # another producer's reservation won the target
        conn.execute(sa_text("""
            INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status,
                                   user_prompt, response_text, ended_at)
            VALUES (:id, now(), '/test', true, :bot, 'nick', 'reserved', '', '', NULL)
        """), {"id": f"turn-other-{uuid.uuid4().hex}", "bot": bot})
    durable.dispatch(claimed)

    refusal = durable.refusals[-1]
    assert isinstance(refusal, TargetBusy) and not isinstance(refusal, SystemGateRefused)
    row = _row(deliveries, record.id)
    assert (row["status"], row["attempt_count"], row["transport_accepted_at"]) == ("QUEUED", 0, None)
    assert row["deferred"]
    assert store.get_turn(record.turn_id) is None
