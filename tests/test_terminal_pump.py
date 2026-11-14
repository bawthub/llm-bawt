"""Recovery pump is bounded, report-only by default, and cancellation-safe."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from llm_bawt.service import turn_terminal_pump as mod
from llm_bawt.service.turn_terminal_protocol import FollowerFenceLost


def test_report_only_never_claims_or_recovers(monkeypatch):
    protocol = SimpleNamespace(recoverable_ids=Mock(return_value=["a", "b"]))
    recover = AsyncMock()
    monkeypatch.setattr(mod, "recover_terminal_turn", recover)
    result = asyncio.run(mod.terminal_recovery_pass(protocol, object()))
    assert result.candidates == ("a", "b") and result.attempted == result.errors == 0
    protocol.recoverable_ids.assert_called_once_with(limit=16)
    recover.assert_not_awaited()


def test_one_failure_or_fence_loss_does_not_abandon_other_rows(monkeypatch):
    protocol = SimpleNamespace(recoverable_ids=Mock(return_value=["a", "b", "c"]))
    async def recover(protocol, turn, subscriber):
        if turn == "a":
            raise RuntimeError("temporary DB failure")
        if turn == "b":
            raise FollowerFenceLost("other owner")
    mock = AsyncMock(side_effect=recover)
    monkeypatch.setattr(mod, "recover_terminal_turn", mock)
    result = asyncio.run(mod.terminal_recovery_pass(protocol, object(), report_only=False))
    assert result.attempted == 3 and result.errors == 1
    assert mock.await_count == 3


def test_attempt_concurrency_is_bounded(monkeypatch):
    protocol = SimpleNamespace(recoverable_ids=Mock(return_value=list(map(str, range(16)))))
    active = [0]
    peak = [0]
    async def recover(*args):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        await asyncio.sleep(0)
        active[0] -= 1
    monkeypatch.setattr(mod, "recover_terminal_turn", recover)
    result = asyncio.run(mod.terminal_recovery_pass(protocol, object(), report_only=False))
    assert result.attempted == 16 and peak == [4] and active == [0]


@pytest.mark.parametrize("private_transport", [False, True])
def test_external_shutdown_cancels_every_inflight_attempt(monkeypatch, private_transport):
    protocol = SimpleNamespace(recoverable_ids=Mock(return_value=list(map(str, range(16)))))
    async def scenario():
        entered = asyncio.Event()
        cancelled = []
        started = []
        async def recover(protocol, turn, subscriber):
            started.append(turn)
            if len(started) == 4:
                entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(turn)
        monkeypatch.setattr(mod, "recover_terminal_turn", recover)
        subscriber = SimpleNamespace(close=AsyncMock())
        task = asyncio.create_task(mod.terminal_recovery_pass(
            protocol, None if private_transport else subscriber, report_only=False,
            subscriber_factory=lambda: subscriber,
        ))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sorted(cancelled) == sorted(started) and len(started) == 4
        assert subscriber.close.await_count == int(private_transport)
    asyncio.run(scenario())


def test_loop_exits_on_owner_stop_without_another_pass(monkeypatch):
    protocol = object()
    async def scenario():
        stop = asyncio.Event()
        async def recovery_pass(*args, **kwargs):
            stop.set()
            return mod.RecoveryPass((), 0, 0)
        run = AsyncMock(side_effect=recovery_pass)
        monkeypatch.setattr(mod, "terminal_recovery_pass", run)
        await mod.run_terminal_recovery_pump(protocol, object(), stop)
        run.assert_awaited_once()
    asyncio.run(scenario())


@pytest.mark.parametrize("mode, report_only", [("report", True), ("recover", False), ("unknown", True)])
def test_dynamic_mode_is_read_off_loop_and_unknown_reports_only(monkeypatch, mode, report_only):
    import threading

    async def scenario():
        stop = asyncio.Event()
        loop_thread = threading.get_ident()

        def resolve():
            assert threading.get_ident() != loop_thread
            return mode

        async def recovery_pass(*args, **kwargs):
            stop.set()
            return mod.RecoveryPass((), 0, 0)

        run = AsyncMock(side_effect=recovery_pass)
        monkeypatch.setattr(mod, "terminal_recovery_pass", run)
        await mod.run_terminal_recovery_pump("protocol", "subscriber", stop, mode_resolver=resolve)
        run.assert_awaited_once_with("protocol", "subscriber", report_only=report_only)

    asyncio.run(scenario())


@pytest.mark.parametrize("setting", ["off", RuntimeError("settings unavailable")])
def test_off_or_failed_settings_lookup_never_discovers_or_mutates(monkeypatch, setting):
    async def scenario():
        class StopAfterPass(asyncio.Event):
            async def wait(self):
                self.set()
                return True

        stop = StopAfterPass()
        run = AsyncMock()
        monkeypatch.setattr(mod, "terminal_recovery_pass", run)

        def resolve():
            if isinstance(setting, Exception):
                raise setting
            return setting

        await mod.run_terminal_recovery_pump(object(), object(), stop, mode_resolver=resolve)
        run.assert_not_awaited()

    asyncio.run(scenario())


def test_missing_startup_subscriber_retries_and_closes_only_private_transport(monkeypatch):
    protocol = SimpleNamespace(recoverable_ids=Mock(return_value=["turn"]))
    subscriber = SimpleNamespace(close=AsyncMock())
    factory = Mock(side_effect=[ConnectionError("startup outage"), subscriber])
    recover = AsyncMock()
    monkeypatch.setattr(mod, "recover_terminal_turn", recover)

    async def scenario():
        with pytest.raises(ConnectionError, match="startup outage"):
            await mod.terminal_recovery_pass(protocol, None, report_only=False, subscriber_factory=factory)
        recover.assert_not_awaited()
        result = await mod.terminal_recovery_pass(protocol, None, report_only=False, subscriber_factory=factory)
        assert result.attempted == 1 and result.errors == 0
        recover.assert_awaited_once_with(protocol, "turn", subscriber)
        subscriber.close.assert_awaited_once()

    asyncio.run(scenario())


@pytest.mark.parametrize("ids, report_only", [(["turn"], True), ([], False)])
def test_reporting_or_empty_discovery_never_opens_fallback_transport(monkeypatch, ids, report_only):
    protocol = SimpleNamespace(recoverable_ids=Mock(return_value=ids))
    factory = Mock(side_effect=AssertionError("must not connect"))
    result = asyncio.run(mod.terminal_recovery_pass(protocol, None, report_only=report_only,
                                                   subscriber_factory=factory))
    assert result.attempted == 0
    factory.assert_not_called()
