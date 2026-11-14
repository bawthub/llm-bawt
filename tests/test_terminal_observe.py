"""Deterministic lease/budget tests for read-only recovery observation."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from redis.exceptions import ConnectionError

from agent_bridge.terminal_replay import ReplayDisposition, RetainedRun
from llm_bawt.service.turn_execution import TurnExecutionDetached
from llm_bawt.service import turn_terminal_observe as mod


@pytest.fixture
def harness(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    # Replace this module's asyncio reference, never asyncio.sleep globally.
    async def sleep(seconds):
        clock[0] += seconds
    monkeypatch.setattr(mod, "asyncio", SimpleNamespace(timeout=asyncio.timeout, sleep=sleep))
    reader = AsyncMock()
    monkeypatch.setattr(mod, "read_retained_run", reader)
    follower = SimpleNamespace(check=Mock())
    redis = object()
    def run():
        return asyncio.run(mod.observe_bound_run(
            follower, redis, request_id="request", session_key="session",
            backend="bridge", boot_id="boot",
        ))
    return SimpleNamespace(run=run, clock=clock, reader=reader, follower=follower, redis=redis)


def test_transport_error_retries_without_turning_into_missing(harness):
    complete = RetainedRun(ReplayDisposition.COMPLETE, "complete")
    harness.reader.side_effect = [ConnectionError("offline"), complete]
    result = harness.run()
    assert result.replay is complete and result.incomplete_reason is None
    assert harness.clock == [1.0]
    assert harness.reader.await_count == 2
    harness.reader.assert_awaited_with(harness.redis, request_id="request", session_key="session",
                                     backend="bridge", boot_id="boot")


@pytest.mark.parametrize("disposition", [ReplayDisposition.MISSING, ReplayDisposition.TRIMMED,
                                       ReplayDisposition.INVALID, ReplayDisposition.LIMIT])
def test_conclusive_incomplete_evidence_returns_without_retry(harness, disposition):
    replay = RetainedRun(disposition, "specific_incomplete_reason")
    harness.reader.return_value = replay
    result = harness.run()
    assert result.replay is replay and result.incomplete_reason == replay.reason
    assert harness.clock == [0.0] and harness.reader.await_count == 1


@pytest.mark.parametrize("transport", [True, False])
def test_absolute_budget_is_never_reset_by_new_read_results(harness, transport):
    if transport:
        harness.reader.side_effect = ConnectionError("offline")
    else:
        harness.reader.return_value = RetainedRun(ReplayDisposition.PENDING, "still_running")
    result = harness.run()
    assert harness.clock == [60.0] and harness.reader.await_count == 60
    assert result.incomplete_reason == (
        "recovery_transport_unavailable" if transport else "recovery_run_not_terminal"
    )
    assert (result.replay is None) is transport


def test_fence_loss_before_read_has_no_transport_side_effect(harness):
    harness.follower.check.side_effect = TurnExecutionDetached()
    with pytest.raises(TurnExecutionDetached):
        harness.run()
    harness.reader.assert_not_awaited()


def test_fence_loss_after_complete_read_does_not_return_commit_authority(harness):
    harness.reader.return_value = RetainedRun(ReplayDisposition.COMPLETE, "complete")
    harness.follower.check.side_effect = [None, TurnExecutionDetached()]
    with pytest.raises(TurnExecutionDetached):
        harness.run()


def test_external_cancellation_is_not_converted_into_fallback(harness):
    harness.reader.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        harness.run()


def test_programming_error_is_not_misreported_as_transport_absence(harness):
    harness.reader.side_effect = RuntimeError("bug")
    with pytest.raises(RuntimeError, match="bug"):
        harness.run()


def test_outer_timeout_still_checks_fence_before_return(harness, monkeypatch):
    class Deadline:
        async def __aenter__(self):
            raise TimeoutError()
        async def __aexit__(self, *args):
            return False
    monkeypatch.setattr(mod.asyncio, "timeout", lambda seconds: Deadline())
    result = harness.run()
    assert result.replay is None and result.incomplete_reason == "recovery_observation_timeout"
    harness.follower.check.assert_called_once()
