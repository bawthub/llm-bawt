"""PostgreSQL tests for the TASK-1233 gate edges: enqueue dispositions and steer.

Every gate is scoped to this test's own bots, so parallel PG tests are never
held by it. The claim-side hold is covered in test_delivery_claim_gate_pg.py.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.service.system_gate_admission import SteerGateRefused, check_steer
from llm_bawt.service.turn_admission import SystemGateRefused
from llm_bawt.service.turn_gates import GateScope
from tests.test_turn_gates_pg import _activate, _open, _spec, gated  # noqa: F401

pytestmark = pytest.mark.usefixtures("private_app_pg")


def _deliveries(ctx):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore

    return InterBotDeliveryStore(ctx.store.config)


def _enqueue(ctx, deliveries, bot, *, on_gate, key=None):
    sender = f"test-gate-sender-{uuid.uuid4().hex}"
    ctx.senders.append(sender)
    return deliveries.enqueue(
        sender_bot_id=sender, target_bot_id=bot, message="callback",
        payload={"messages": [{"role": "user", "content": "callback"}],
                 "bot_id": bot, "user": "nick", "stream": True},
        idempotency_key=key, on_gate=on_gate,
    )


def _rows(deliveries, bot):
    with deliveries.engine.connect() as conn:
        return conn.execute(sa_text(
            "SELECT id, status FROM inter_bot_deliveries WHERE target_bot_id=:b"
        ), {"b": bot}).mappings().all()


def _gate(ctx, bot, mode):
    return _activate(ctx.store, _spec(ctx.owner(), GateScope.for_bots(bot), mode=mode))


@pytest.mark.integration
def test_a_freeze_refuses_a_deny_producer_and_writes_nothing(gated):
    bot = gated.bot()
    deliveries = _deliveries(gated)
    _gate(gated, bot, "deny")
    with pytest.raises(SystemGateRefused) as refused:
        _enqueue(gated, deliveries, bot, on_gate="deny")
    assert (refused.value.code, refused.value.status_code) == ("system_frozen", 503)
    assert _rows(deliveries, bot) == []


@pytest.mark.integration
def test_a_freeze_admits_a_hold_producer_which_stays_queued(gated):
    bot = gated.bot()
    deliveries = _deliveries(gated)
    _gate(gated, bot, "deny")
    record, duplicate = _enqueue(gated, deliveries, bot, on_gate="hold")
    assert not duplicate
    assert [dict(r) for r in _rows(deliveries, bot)] == [{"id": record.id, "status": "QUEUED"}]


@pytest.mark.integration
@pytest.mark.parametrize("on_gate", ["deny", "hold"])
def test_a_quiesce_refuses_no_producer_at_enqueue(gated, on_gate):
    bot = gated.bot()
    deliveries = _deliveries(gated)
    _gate(gated, bot, "hold")
    record, duplicate = _enqueue(gated, deliveries, bot, on_gate=on_gate)
    assert not duplicate and record.status == "QUEUED"


@pytest.mark.integration
def test_another_bots_freeze_does_not_refuse(gated):
    bot, other = gated.bot(), gated.bot()
    deliveries = _deliveries(gated)
    _gate(gated, other, "deny")
    record, _ = _enqueue(gated, deliveries, bot, on_gate="deny")
    assert record.status == "QUEUED"


@pytest.mark.integration
def test_a_retry_accepted_before_the_freeze_still_gets_its_receipt(gated):
    bot = gated.bot()
    deliveries = _deliveries(gated)
    key = f"test-gate-key-{uuid.uuid4().hex}"
    sender = f"test-gate-sender-{uuid.uuid4().hex}"
    gated.senders.append(sender)

    def send():
        return deliveries.enqueue(
            sender_bot_id=sender, target_bot_id=bot, message="callback",
            payload={"messages": [{"role": "user", "content": "callback"}],
                     "bot_id": bot, "user": "nick", "stream": True},
            idempotency_key=key, on_gate="deny",
        )

    first, duplicate = send()
    assert not duplicate
    _gate(gated, bot, "deny")
    again, duplicate = send()
    assert duplicate and again.id == first.id
    assert len(_rows(deliveries, bot)) == 1


@pytest.mark.integration
@pytest.mark.parametrize(("mode", "code", "status"), [
    ("hold", "system_held", 409), ("deny", "system_frozen", 503),
])
def test_any_live_gate_refuses_a_steer(gated, mode, code, status):
    bot = gated.bot()
    turn_id = _open(gated.store, bot)
    _gate(gated, bot, mode)
    with pytest.raises(SteerGateRefused) as refused:
        check_steer(gated.store.engine, bot, turn_id=turn_id)
    assert (refused.value.code, refused.value.status_code) == (code, status)


@pytest.mark.integration
def test_a_steer_is_admitted_without_a_covering_gate(gated):
    bot, other = gated.bot(), gated.bot()
    turn_id = _open(gated.store, bot)
    assert check_steer(gated.store.engine, bot, turn_id=turn_id) is None
    _gate(gated, other, "deny")
    assert check_steer(gated.store.engine, bot, turn_id=turn_id) is None
