"""Unit tests for the queued chat wait reason's wire shape (TASK-1676)."""

from __future__ import annotations

from datetime import datetime, timezone

from llm_bawt.service.delivery_wait_reason import reason_for
from llm_bawt.service.ordinary_turn_queue import OrdinaryTurnDecision
from llm_bawt.service.turn_gates import GateBlock

NOW = datetime.now(timezone.utc)


def _block(mode):
    return GateBlock(id=1, mode=mode, owner_kind="job", owner_id="j", reason=None, expires_at=NOW)


def test_reason_for_names_the_gate_kind():
    assert reason_for(None) == {"kind": "turn_running", "gate": None}
    assert reason_for(_block("hold")) == {"kind": "system_hold", "gate": "quiesce"}
    assert reason_for(_block("deny")) == {"kind": "system_hold", "gate": "freeze"}


def test_queued_receipt_carries_the_wait_reason():
    reason = {"kind": "system_hold", "gate": "quiesce"}
    decision = OrdinaryTurnDecision("queued", "t", "u", "a", "d", "QUEUED", wait_reason=reason)
    assert decision.receipt()["wait_reason"] == reason
    assert "wait_reason" not in OrdinaryTurnDecision("queued", "t", "u", "a", "d", "QUEUED").receipt()
