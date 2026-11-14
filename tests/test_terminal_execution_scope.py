"""Admission handoff and worker-lifetime cleanup, without timers or providers."""
from datetime import datetime, timezone
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import Mock
import asyncio

import pytest

from llm_bawt.service.turn_execution import TurnExecutionDetached, turn_executions
from llm_bawt.service.turn_lifecycle import TurnLifecycleMixin
from llm_bawt.service.turn_terminal_follower import TerminalFollower, TerminalFollowerScope
from llm_bawt.service.turn_terminal_protocol import FollowerFenceLost, FollowerLease


@pytest.fixture
def scope(monkeypatch):
    protocol = Mock()
    protocol.renew.side_effect = lambda lease: lease
    monkeypatch.setattr(
        "llm_bawt.service.turn_terminal_follower.TurnTerminalProtocol", lambda store: protocol,
    )
    original_start = TerminalFollower.start

    def start(follower):
        follower._stop.wait = Mock(return_value=True)
        return original_start(follower)

    monkeypatch.setattr(TerminalFollower, "start", start)
    result = TerminalFollowerScope(SimpleNamespace(engine=object()))
    yield result
    result._close()


def admit(scope, *, agent=True):
    lease = FollowerLease("scope-test", "insert-token", datetime.now(timezone.utc))
    event, closed = Event(), Mock()
    scope.admitted(
        lease, turn_id=lease.turn_id, cancel_event=event, is_agent=agent, on_close=closed,
    )
    return event, closed


class DeferredLoop:
    def run_in_executor(self, executor, worker):
        assert executor is None
        self.worker = worker
        return "worker-future"


def test_setup_cleanup_stops_the_heartbeat_without_claiming_provider_exit(scope):
    event, closed = admit(scope)
    follower = scope.follower
    assert turn_executions.get("scope-test") is follower.execution
    assert follower.execution.cancel_event is event
    scope.close_if_unlaunched()
    scope.close_if_unlaunched()
    assert follower._stop.is_set()
    assert turn_executions.get("scope-test") is None
    assert not follower.execution.confirmed_stopped()
    closed.assert_called_once()
    follower.protocol.claim.assert_not_called()


def test_request_exit_after_handoff_does_not_stop_a_still_running_worker(scope):
    _, closed = admit(scope)
    loop = DeferredLoop()
    follower = scope.follower
    assert scope.launch_worker(loop, lambda: "actual-response") == "worker-future"
    # This is also what happens when an HTTP task or SSE generator is cancelled.
    scope.close_if_unlaunched()
    assert not follower._stop.is_set()
    assert not follower.execution.confirmed_stopped()
    assert turn_executions.get("scope-test") is follower.execution
    closed.assert_not_called()
    assert loop.worker() == "actual-response"
    assert follower.execution.confirmed_stopped()
    assert follower._stop.is_set()
    assert turn_executions.get("scope-test") is None
    closed.assert_called_once()


def test_worker_exception_still_closes_heartbeat_after_real_worker_return(scope):
    _, closed = admit(scope)
    loop = DeferredLoop()

    def fail():
        raise ValueError("worker failed")

    scope.launch_worker(loop, fail)
    with pytest.raises(ValueError, match="worker failed"):
        loop.worker()
    assert scope.follower.execution.confirmed_stopped()
    assert turn_executions.get("scope-test") is None
    closed.assert_called_once()


def test_executor_refusal_cleans_up_without_inventing_worker_exit(scope):
    _, closed = admit(scope)
    loop = SimpleNamespace(run_in_executor=Mock(side_effect=RuntimeError("executor closed")))
    with pytest.raises(RuntimeError, match="executor closed"):
        scope.launch_worker(loop, Mock())
    assert scope.follower._stop.is_set()
    assert not scope.follower.execution.confirmed_stopped()
    assert turn_executions.get("scope-test") is None
    closed.assert_called_once()


def test_lost_observation_cannot_launch_a_provider(scope):
    admit(scope)
    scope.follower.execution.detach()
    loop = SimpleNamespace(run_in_executor=Mock())
    with pytest.raises(TurnExecutionDetached):
        scope.launch_worker(loop, Mock())
    loop.run_in_executor.assert_not_called()


def test_legacy_admission_does_not_start_a_protocol_follower(scope):
    scope.admitted(None, turn_id="legacy", cancel_event=Event(), is_agent=True)
    assert scope.follower is None
    loop = DeferredLoop()
    assert scope.launch_worker(loop, lambda: "legacy response") == "worker-future"
    assert loop.worker() == "legacy response"
    scope.close_if_unlaunched()


def test_pre_launch_failure_uses_original_fence_not_an_interrupt_or_reclaim(scope):
    admit(scope)
    scope.fail_setup("reset setup failed")
    scope.follower.protocol.commit.assert_called_once_with(
        scope.follower.lease, status="error", end_reason="setup_error", error_text="reset setup failed",
    )
    scope.follower.protocol.interrupt.assert_not_called()
    scope.follower.protocol.claim.assert_not_called()


def test_pre_launch_failure_cannot_override_a_new_owner(scope):
    admit(scope)
    scope.follower.protocol.commit.side_effect = FollowerFenceLost("new owner")
    scope.fail_setup("old setup failed")
    scope.follower.protocol.interrupt.assert_not_called()
    scope.follower.protocol.claim.assert_not_called()


def test_request_error_after_launch_does_not_terminalize_a_blocked_worker(scope):
    admit(scope)
    loop = DeferredLoop()
    scope.launch_worker(loop, lambda: "answer")
    scope.fail_setup("HTTP client disconnected")
    scope.follower.protocol.commit.assert_not_called()
    assert not scope.follower.execution.confirmed_stopped()
    assert loop.worker() == "answer"


def test_real_worker_can_commit_its_own_pre_inference_setup_failure(scope):
    admit(scope)
    loop = DeferredLoop()

    def worker():
        scope.fail_setup("worker reset failed", from_worker=True)
        return "failed setup"

    scope.launch_worker(loop, worker)
    assert loop.worker() == "failed setup"
    scope.follower.protocol.commit.assert_called_once()


def test_provider_error_is_not_misclassified_as_a_setup_failure(scope):
    admit(scope)
    loop = DeferredLoop()

    def worker():
        scope.begin_inference()
        scope.fail_setup("provider error", from_worker=True)

    scope.launch_worker(loop, worker)
    loop.worker()
    scope.follower.protocol.commit.assert_not_called()


def test_native_generation_keeps_early_stop_and_original_event_identity():
    service = TurnLifecycleMixin()
    service._cancel_lock = Lock()
    cancel, done = Event(), Event()
    cancel.set()
    returned = asyncio.run(service._start_generation("scope-test", cancel_event=cancel, done_event=done))
    assert returned == (cancel, done)
    assert service._gen_cancels["scope-test"] is cancel
    assert cancel.is_set()
    assert not done.is_set()
    service._end_generation(cancel, done, "scope-test")
    assert done.is_set() and not service._gen_cancels
