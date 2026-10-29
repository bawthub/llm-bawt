"""PostgreSQL tests for TASK-1758: a held head never blocks later steers.

A QUEUED head that must wait for the target's running turn to end
(``prefer_steer`` false) used to hold every later delivery at attempt 0 for
the whole turn. Later steer rows now steer into that run; the held head stays
QUEUED with a visible reason; wait-for-turn-end rows keep their own order.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.inter_bot_preclaim import WAITING_FOR_TURN_END
from tests.test_delivery_claim_gate_pg import _enqueue
from tests.test_turn_gates_pg import _open, gated  # noqa: F401  (gated is a fixture)

pytestmark = pytest.mark.usefixtures("private_app_pg")


def _wait_for_turn_end(deliveries, record):
    with deliveries.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE inter_bot_deliveries SET payload_json=payload_json || '{\"prefer_steer\": false}'::jsonb"
            " WHERE id=:id"), {"id": record.id})
    return record


def _state(deliveries, delivery_id):
    with deliveries.engine.connect() as conn:
        return conn.execute(sa_text(
            "SELECT status, attempt_count, last_error, target_turn_id FROM inter_bot_deliveries WHERE id=:id"
        ), {"id": delivery_id}).mappings().one()


def _claim(deliveries, bot):
    return deliveries.claim_next(bot, claim_owner="worker", steer_capable=True)


@pytest.mark.integration
def test_a_later_steer_row_steers_past_a_held_head(gated):
    bot = gated.bot()
    deliveries, held = _enqueue(gated, bot, "wait for your turn to end")
    _wait_for_turn_end(deliveries, held)
    _, steer = _enqueue(gated, bot, "steer me in")
    turn = _open(gated.store, bot)

    claimed = _claim(deliveries, bot)

    assert claimed is not None and claimed.id == steer.id
    assert claimed.status == "STEERING" and claimed.target_turn_id == turn
    head = _state(deliveries, held.id)
    assert head["status"] == "QUEUED" and head["attempt_count"] == 0
    assert head["last_error"] == WAITING_FOR_TURN_END


@pytest.mark.integration
def test_wait_rows_keep_order_and_the_head_runs_after_the_turn(gated):
    bot = gated.bot()
    deliveries, first = _enqueue(gated, bot, "first wait")
    _wait_for_turn_end(deliveries, first)
    _, second = _enqueue(gated, bot, "second wait")
    _wait_for_turn_end(deliveries, second)
    turn = _open(gated.store, bot)

    assert _claim(deliveries, bot) is None  # no steer row: nothing passes
    assert _state(deliveries, second.id)["attempt_count"] == 0
    assert _state(deliveries, first.id)["last_error"] == WAITING_FOR_TURN_END

    with gated.store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET status='ok', ended_at=now() WHERE id=:id"), {"id": turn})
    claimed = _claim(deliveries, bot)
    assert claimed is not None and claimed.id == first.id and claimed.status == "DISPATCHING"
    assert _state(deliveries, first.id)["last_error"] is None


@pytest.mark.integration
def test_an_unsteerable_run_lets_nothing_pass(gated):
    bot = gated.bot()
    deliveries, held = _enqueue(gated, bot, "wait")
    _wait_for_turn_end(deliveries, held)
    _enqueue(gated, bot, "would steer")
    _open(gated.store, bot)

    assert deliveries.claim_next(bot, claim_owner="worker", steer_capable=False) is None
    assert _state(deliveries, held.id)["last_error"] is None

@pytest.mark.integration
def test_a_sender_never_passes_its_own_earlier_waiting_row(gated):
    """Codex DESIGN r1 B1: one author's order is a causal boundary."""
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore

    bot = gated.bot()
    deliveries, held = _enqueue(gated, bot, "A: wait")
    _wait_for_turn_end(deliveries, held)
    later, _ = InterBotDeliveryStore(gated.store.config), None
    with deliveries.engine.connect() as conn:
        sender = conn.execute(sa_text(
            "SELECT sender_bot_id FROM inter_bot_deliveries WHERE id=:id"), {"id": held.id}).scalar_one()
    same, _ = later.enqueue(
        sender_bot_id=sender, target_bot_id=bot, message="B: same sender",
        payload={"messages": [{"role": "user", "content": "B"}], "bot_id": bot, "user": "nick", "stream": True},
    )
    _, other = _enqueue(gated, bot, "C: other sender")
    _open(gated.store, bot)

    claimed = _claim(deliveries, bot)

    assert claimed is not None and claimed.id == other.id
    assert _state(deliveries, same.id)["status"] == "QUEUED"


@pytest.mark.integration
def test_no_second_pass_while_a_steer_is_in_flight(gated):
    """Codex DESIGN r1 B2: a claimed steer fences the next jump."""
    bot = gated.bot()
    deliveries, held = _enqueue(gated, bot, "wait")
    _wait_for_turn_end(deliveries, held)
    _, first = _enqueue(gated, bot, "steer 1")
    _, second = _enqueue(gated, bot, "steer 2")
    _open(gated.store, bot)

    claimed = _claim(deliveries, bot)
    assert claimed is not None and claimed.id == first.id and claimed.status == "STEERING"
    assert _claim(deliveries, bot) is None
    assert _state(deliveries, second.id)["attempt_count"] == 0

