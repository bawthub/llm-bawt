"""Heartbeat/detach tests drive events directly, never sleep for a lease."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt.service.turn_execution import TurnExecution, TurnExecutionDetached
from llm_bawt.service.turn_terminal_follower import TerminalFollower
from llm_bawt.service.turn_terminal_protocol import FollowerFenceLost, FollowerLease


def follower(*, agent=True):
    lease = FollowerLease("test-turn", "initial-admission-token", datetime.now(timezone.utc))
    execution = TurnExecution("test-turn", Event(), is_agent=agent, store=SimpleNamespace(engine=None))
    protocol = Mock()
    protocol.renew.side_effect = lambda current: FollowerLease(
        current.turn_id, current.token, current.expires_at + timedelta(seconds=60),
    )
    return TerminalFollower(protocol, lease, execution)


def test_start_renews_admission_token_instead_of_reclaiming_it():
    guard = follower()
    # The heartbeat reaches the stop condition immediately. There is no real
    # timer or narrow wall-clock assertion in this test.
    guard._stop.wait = Mock(return_value=True)
    original = guard.lease
    guard.start()
    assert guard.close()
    guard.protocol.renew.assert_called_once_with(original)
    guard.protocol.claim.assert_not_called()
    assert guard.lease.token == original.token
    assert guard.lease.expires_at > original.expires_at
    assert not guard.execution.detach_requested()
    assert not guard.execution.cancel_event.is_set()


def test_heartbeat_renews_while_setup_is_not_running_an_async_loop():
    guard = follower()
    guard._stop.wait = Mock(side_effect=[False, True])
    guard.start()
    assert guard.close()
    assert guard.protocol.renew.call_count == 2
    assert {call.args[0].token for call in guard.protocol.renew.call_args_list} == {
        "initial-admission-token",
    }
    guard.protocol.claim.assert_not_called()


@pytest.mark.parametrize("agent", [True, False])
@pytest.mark.parametrize("error", [FollowerFenceLost("expired owner"), ConnectionError("database unavailable")])
def test_renewal_failure_detaches_without_claiming_or_stopping_bridge(agent, error):
    guard = follower(agent=agent)
    guard.protocol.renew.side_effect = [guard.lease, error]
    guard._stop.wait = Mock(return_value=False)
    guard.start()
    assert guard.close()
    assert guard.execution.detach_requested()
    assert guard.execution.cancel_event.is_set() is (not agent)
    with pytest.raises(TurnExecutionDetached):
        guard.check()
    guard.protocol.claim.assert_not_called()
    assert guard.protocol.renew.call_count == 2


def test_initial_renewal_failure_aborts_setup_and_never_starts_a_heartbeat():
    guard = follower()
    guard.protocol.renew.side_effect = FollowerFenceLost("admission lease already lost")
    with pytest.raises(FollowerFenceLost):
        guard.start()
    assert guard._thread is None and guard.close()
    assert guard.execution.detach_requested()
    with pytest.raises(TurnExecutionDetached):
        guard.check()


def test_follower_rejects_a_different_execution_identity():
    guard = follower()
    guard.execution.turn_id = "different-turn"
    with pytest.raises(ValueError, match="does not belong"):
        TerminalFollower(guard.protocol, guard.lease, guard.execution)


def test_close_does_not_claim_worker_exit_and_cannot_restart():
    guard = follower()
    guard._stop.wait = Mock(return_value=True)
    guard.start()
    assert guard.close() and guard.close()
    assert not guard.execution.confirmed_stopped()
    with pytest.raises(RuntimeError, match="cannot be started"):
        guard.start()
    guard.protocol.claim.assert_not_called()


def test_external_process_detach_is_respected_before_provider_side_effects():
    guard = follower()
    guard.execution.detach()
    with pytest.raises(TurnExecutionDetached):
        guard.check()
    with pytest.raises(TurnExecutionDetached):
        guard.start()
    guard.protocol.renew.assert_not_called()
    assert guard.close()
