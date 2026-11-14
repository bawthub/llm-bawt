"""TASK-2280: recreation ends without relying on another Redis event."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from llm_bawt.house_watcher import DOWN, RECREATING, UP, HouseStates, HouseWatcher


def test_reseed_reconciles_a_failed_recreation():
    states = HouseStates()
    states.on_set("h1", "backend", "old")
    states.on_gone("h1", "backend", released=True)
    assert states.reseed({}) == [("h1", RECREATING, DOWN)]
    assert states.recreating_since == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("leases,expected", [({}, DOWN), ({("h1", "backend"): "new"}, UP)])
async def test_recreation_deadline_reads_leases_without_reconnect(monkeypatch, leases, expected):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr("llm_bawt.house_watcher.time.monotonic", lambda: clock.now)
    watcher = HouseWatcher.__new__(HouseWatcher)
    watcher.states = HouseStates()
    watcher._read_leases = AsyncMock(return_value=leases)
    watcher._emit = AsyncMock()
    watcher.states.on_gone("h1", "backend", released=True)
    clock.now = 59.0
    await watcher.reconcile_recreating(None)
    watcher._read_leases.assert_not_called()
    clock.now = 60.0
    await watcher.reconcile_recreating(None)
    assert watcher.states.state["h1"] == expected
    watcher._emit.assert_awaited_once_with([("h1", RECREATING, expected)])
    await watcher.reconcile_recreating(None)
    assert watcher._read_leases.await_count == 1


@pytest.mark.asyncio
async def test_reconcile_preserves_newer_events_and_failed_reads(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr("llm_bawt.house_watcher.time.monotonic", lambda: clock.now)
    watcher = HouseWatcher.__new__(HouseWatcher)
    watcher.states = HouseStates()
    watcher._emit = AsyncMock()
    watcher.states.on_gone("h1", "backend", released=True)
    clock.now = 60.0
    watcher._read_leases = AsyncMock(side_effect=RuntimeError("offline"))
    with pytest.raises(RuntimeError):
        await watcher.reconcile_recreating(None)
    assert watcher.states.state["h1"] == RECREATING
    async def newer_event(client):
        watcher.states.on_set("h1", "backend", "new")
        return {}
    watcher._read_leases = newer_event
    await watcher.reconcile_recreating(None)
    assert watcher.states.state["h1"] == UP
    watcher._emit.assert_awaited_once_with([])
