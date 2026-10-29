"""PostgreSQL tests for the TASK-1748 quiesce escalation.

A held run whose control row sits DELIVERED (queued at the bridge behind one
long tool) past the grace gets exactly one escalation: the bridge is asked to
interrupt the tool and run that queued command now, inside the registry frame.
The attempt and its outcome are one ``gate_escalations`` row, and the job's
roster shows it.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from llm_bawt.service.gate_steer_escalation import KIND_QUIESCE_ESCALATE
from llm_bawt.service.quiesce_escalation import ESCALATED, QuiesceEscalator
from tests.test_ops_quiesce_pg import rig  # noqa: F401
from tests.test_quiesce_control_producer_pg import (  # noqa: F401
    _controls,
    _member,
    _message,
    _open,
    _producer,
    _steer,
    _turn,
)

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

FRAME = "[quiesce frame]\n{items}"


@pytest.fixture(autouse=True)
def _cleanup(rig):
    yield
    rig.sql("DELETE FROM gate_escalations WHERE bot_id=:b", b=rig.bot)


@pytest.fixture(autouse=True)
def _frame(monkeypatch):
    monkeypatch.setattr("llm_bawt.ops.quiesce_prompts.render_quiesce_interrupt_frame", lambda bot: FRAME)


class Bridge:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.sent = reply or {"ok": True}, error, []

    async def send_steer(self, **kwargs):
        self.sent.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.reply


def _queued(rig, monkeypatch, **turn):
    """A held running turn whose control row is DELIVERED; returns (turn, member, control)."""
    turn_id = _turn(rig, **turn)
    job_id = _open(rig)
    assert _producer(rig).run_pass().enqueued == 1
    _steer(rig, monkeypatch)
    [control] = _controls(rig)
    assert control["status"] == "DELIVERED"
    return job_id, turn_id, _member(rig, turn_id), control


def _age(rig, control_id, seconds=120):
    rig.sql("UPDATE inter_bot_deliveries SET delivered_at = now() - (:s * interval '1 second') WHERE id=:id",
            s=seconds, id=control_id)


def _records(rig, gate_id):
    return rig.sql("SELECT * FROM gate_escalations WHERE gate_id=:g AND kind=:k ORDER BY id",
                   g=gate_id, k=KIND_QUIESCE_ESCALATE).mappings().all()


def test_a_control_queued_past_the_grace_is_escalated_once_and_recorded(rig, monkeypatch):
    job_id, turn_id, member, control = _queued(rig, monkeypatch)
    escalator = QuiesceEscalator(rig.deliveries)
    bridge = Bridge()
    assert asyncio.run(escalator.run_pass(bridge)).attempted == 0  # still inside the grace
    _age(rig, control["id"])

    result = asyncio.run(escalator.run_pass(bridge))
    assert (result.attempted, result.escalated) == (1, 1)
    [sent] = bridge.sent
    # The bridge's held key is the control row's message id (a UUID kept as is).
    assert sent["message_id"] == str(control["user_message_id"])
    assert (sent["session_key"], sent["target_request_id"], sent["backend"]) == (
        "sess-held", f"req-{turn_id}", "claude-code")
    assert (sent["origin"], sent["escalate"], sent["interrupt_frame"]) == ("system", True, FRAME)
    [record] = _records(rig, member["gate_id"])
    assert (record["subject"], record["outcome"], record["actor"]) == (str(member["id"]), ESCALATED,
                                                                     "system:quiesce")

    assert asyncio.run(escalator.run_pass(bridge)).attempted == 0  # one attempt, never retried
    assert len(bridge.sent) == 1

    assert rig.driver.step(job_id).outcome == "holding"
    [row] = json.loads(rig.row(job_id)["quiesce_json"])["roster"]
    assert row["turn_id"] == turn_id and row["escalation"] == ESCALATED and row["escalated_at"]


@pytest.mark.parametrize("bridge, outcome", [
    (Bridge(reply={"ok": False, "error": "escalation_other_queued"}), "refused:escalation_other_queued"),
    (Bridge(error=TimeoutError("no reply")), "uncertain:no reply"),
], ids=["refused", "uncertain"])
def test_a_failed_escalation_is_recorded_and_never_retried(rig, monkeypatch, bridge, outcome):
    _, _, member, control = _queued(rig, monkeypatch)
    _age(rig, control["id"])
    escalator = QuiesceEscalator(rig.deliveries)
    assert asyncio.run(escalator.run_pass(bridge)).escalated == 0
    assert [r["outcome"] for r in _records(rig, member["gate_id"])] == [outcome]
    assert asyncio.run(escalator.run_pass(bridge)).attempted == 0 and len(bridge.sent) == 1


def test_a_run_pinned_without_escalate_is_recorded_unsupported_without_a_call(rig, monkeypatch):
    _, _, member, control = _queued(rig, monkeypatch, caps="next")
    _age(rig, control["id"])
    bridge = Bridge()
    assert asyncio.run(QuiesceEscalator(rig.deliveries).run_pass(bridge)).attempted == 0
    assert bridge.sent == []
    assert [r["outcome"] for r in _records(rig, member["gate_id"])] == ["refused:escalate_unsupported"]


def test_an_ended_turn_or_a_control_not_yet_delivered_is_never_escalated(rig, monkeypatch):
    turn_id = _turn(rig)
    _open(rig)
    assert _producer(rig).run_pass().enqueued == 1  # QUEUED, not yet steered
    [control] = _controls(rig)
    rig.sql("UPDATE inter_bot_deliveries SET delivered_at = now() - interval '1 hour' WHERE id=:id",
            id=control["id"])
    bridge = Bridge()
    escalator = QuiesceEscalator(rig.deliveries)
    assert asyncio.run(escalator.run_pass(bridge)).attempted == 0

    _steer(rig, monkeypatch)
    _age(rig, control["id"])
    rig.end(turn_id, exited=False)
    assert asyncio.run(escalator.run_pass(bridge)).attempted == 0 and bridge.sent == []
