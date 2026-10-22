"""PostgreSQL tests for TASK-1233 B1 source (a): the run terminal stamped at settle.

Gates are scoped to this test's own bots (the ``gated`` fixture), so parallel
PG tests are never held by them.
"""

from __future__ import annotations

import json
from threading import Event

import pytest
from sqlalchemy import text as sa_text

from agent_bridge.run_contract import RunTerminal, TeardownReport
from llm_bawt.service.turn_execution import TurnExecution
from llm_bawt.service.turn_gates import GateScope, execution_exit_record, stamp_member_exit
from tests.test_turn_gates_pg import (  # noqa: F401 — gated is a fixture
    _activate,
    _bind,
    _delete_turns,
    _drain,
    _members,
    _open,
    _spec,
    gated,
)

pytestmark = pytest.mark.usefixtures("private_app_pg")


def _terminal(exited, request_id="req-1", backend="claude-code"):
    return RunTerminal(request_id, backend, "boot-1", TeardownReport("ok", "none", exited, 9)).to_dict()


def _payload(store, turn_id):
    with store.engine.connect() as conn:
        return json.loads(conn.execute(sa_text("SELECT request_json FROM turn_logs WHERE id=:id"),
                                       {"id": turn_id}).scalar_one() or "{}")


def _held_bound_turn(gated, request_id="req-1"):  # noqa: F811 (pytest injects the imported fixture)
    """A streaming turn bound to ``request_id`` on claude-code, captured by a live hold."""
    store, bot = gated.store, gated.bot()
    turn = _open(store, bot)
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET request_json=:json WHERE id=:id"),
                     {"json": json.dumps({"messages": [{"role": "user", "content": "hi"}]}), "id": turn})
    _bind(store, turn, request_id)
    gate = _activate(store, _spec(gated.owner(), GateScope.for_bots(bot))).gate
    return store, turn, gate, TurnExecution(turn, Event(), is_agent=True, store=store)


@pytest.mark.integration
def test_a_confirmed_terminal_drains_the_member_and_survives_later_turn_writes(gated):  # noqa: F811 (pytest injects the imported fixture)
    store, turn, gate, execution = _held_bound_turn(gated)
    assert _drain(store, gate.id)[turn] == (False, "no_execution_exit")
    assert execution.record_exit(_terminal(True)) is True
    assert _drain(store, gate.id)[turn] == (True, "execution_exit:a")
    payload = _payload(store, turn)
    assert payload["messages"] == [{"role": "user", "content": "hi"}]  # the rest of the payload is kept
    assert payload["execution_exit"]["boot_id"] == "boot-1"

    store.update_turn(turn_id=turn, status="ok", end_reason="completed",
                      request_payload={"messages": [{"role": "user", "content": "settled"}]})
    assert _payload(store, turn)["execution_exit"]["exited"] is True  # a per-attempt key
    assert _drain(store, gate.id)[turn] == (True, "execution_exit:a")


@pytest.mark.integration
def test_an_unconfirmed_terminal_is_a_named_blocker_and_never_undoes_a_proof(gated, caplog):  # noqa: F811 (pytest injects the imported fixture)
    store, turn, gate, execution = _held_bound_turn(gated)
    assert execution.record_exit(_terminal(False)) is True
    assert _drain(store, gate.id)[turn] == (False, "exit_unconfirmed")
    assert execution.record_exit(_terminal(None)) is True
    assert _drain(store, gate.id)[turn] == (False, "exit_unconfirmed")
    assert execution.record_exit(_terminal(True)) is True
    assert execution.record_exit(_terminal(False)) is False  # a late false can't undo the proof
    assert execution.record_exit(_terminal(True)) is False  # idempotent per request
    assert _drain(store, gate.id)[turn] == (True, "execution_exit:a")
    assert "not stamped" not in caplog.text  # the idempotent no-op is quiet


@pytest.mark.integration
def test_a_terminal_for_another_request_or_backend_is_not_stamped(gated, caplog):  # noqa: F811 (pytest injects the imported fixture)
    store, turn, gate, execution = _held_bound_turn(gated)
    assert execution.record_exit(_terminal(True, request_id="req-other")) is False
    assert f"not stamped on turn {turn}: record request=req-other backend=claude-code, row request=req-1" in caplog.text
    assert execution.record_exit(_terminal(True, backend="codex")) is False
    assert "record request=req-1 backend=codex, row request=req-1 backend=claude-code" in caplog.text
    assert execution.record_exit({}) is False  # a pre-contract bridge proves nothing
    assert "execution_exit" not in _payload(store, turn)
    _delete_turns(store, turn)
    assert execution.record_exit(_terminal(True)) is False  # no row: the sweep's member stamp is the home
    assert caplog.text.count("not stamped") == 3 and "backend=claude-code, row missing" in caplog.text
    assert _drain(store, gate.id)[turn] == (False, "turn_missing:no_execution_exit")


@pytest.mark.integration
def test_the_member_stamp_never_undoes_a_proof_either(gated):  # noqa: F811 (pytest injects the imported fixture)
    store, turn, gate, _ = _held_bound_turn(gated)
    _delete_turns(store, turn)
    member = _members(store, gate.id)[turn]["id"]

    def record(exited):
        return execution_exit_record(source="b", request_id="req-1", backend="claude-code",
                                     boot_id="boot-2", exited=exited)

    with store.engine.begin() as conn:
        assert stamp_member_exit(conn, member, record(True)) is True
        assert stamp_member_exit(conn, member, record(False)) is False
        assert stamp_member_exit(conn, member, record(True)) is False
    assert _drain(store, gate.id)[turn] == (True, "execution_exit:b")
