"""TASK-2590: a strict-stamped row is never claimed into another run.

The strict receipt stamp is validated only when the row starts its own turn.
Every claim lane must therefore refuse to steer, interrupt or inbox-claim a
stamped row into a running turn; it waits and then runs as its own turn.
"""
from __future__ import annotations

import json

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.inter_bot_inbox_claim import eligible
from llm_bawt.strict_reader_stamp import is_strict_stamped
from tests.test_delivery_claim_gate_pg import _enqueue
from tests.test_turn_gates_pg import _open, gated  # noqa: F401  (gated is a fixture)

pytestmark = pytest.mark.usefixtures("private_app_pg")

STAMP = {"bot": "fx-strict", "generation": "gen-fx", "layers_sha256": "0" * 64, "house": "fx-house"}


def _stamp(deliveries, record, *, voice: bool):
    payload = {"prefer_steer": True, **({"voice_input": True} if voice else {})}
    with deliveries.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE inter_bot_deliveries SET"
            " metadata_json = COALESCE(metadata_json, '{}'::jsonb) || jsonb_build_object('strict_reader', CAST(:s AS jsonb)),"
            " payload_json = payload_json || CAST(:p AS jsonb) WHERE id=:id"),
            {"s": json.dumps(STAMP), "p": json.dumps(payload), "id": record.id})


def _row(deliveries, delivery_id):
    with deliveries.engine.connect() as conn:
        return conn.execute(sa_text("SELECT * FROM inter_bot_deliveries WHERE id=:id"),
                            {"id": delivery_id}).mappings().one()


@pytest.mark.integration
@pytest.mark.parametrize("voice", [False, True])
def test_stamped_row_never_steers_into_a_running_turn(gated, voice):
    bot = gated.bot()
    deliveries, record = _enqueue(gated, bot, "strict while busy")
    _stamp(deliveries, record, voice=voice)
    turn = _open(gated.store, bot)

    assert deliveries.claim_next(bot, claim_owner="worker", steer_capable=True) is None
    row = _row(deliveries, record.id)
    assert row["status"] == "QUEUED" and row["delivery_mode"] is None and row["target_turn_id"] is None

    # When the run ends the stamped row starts its own turn, where drain
    # validates the stamp.
    with gated.store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET status='ok', ended_at=now() WHERE id=:id"), {"id": turn})
    claimed = deliveries.claim_next(bot, claim_owner="worker", steer_capable=True)
    assert claimed is not None and claimed.id == record.id
    assert claimed.status == "DISPATCHING" and claimed.delivery_mode == "turn"


@pytest.mark.integration
def test_unstamped_control_row_still_steers(gated):
    bot = gated.bot()
    deliveries, record = _enqueue(gated, bot, "ordinary while busy")
    turn = _open(gated.store, bot)
    claimed = deliveries.claim_next(bot, claim_owner="worker", steer_capable=True)
    assert claimed is not None and claimed.status == "STEERING" and claimed.target_turn_id == turn


def test_inbox_lane_refuses_stamped_rows():
    row = {"lane": "normal", "delivery_mode": None, "transport_accepted_at": None,
           "metadata_json": {"strict_reader": dict(STAMP)}, "payload_json": {"prefer_steer": True}}
    assert is_strict_stamped(row)
    assert eligible(row, user_id="fx-user", config=None) is False
