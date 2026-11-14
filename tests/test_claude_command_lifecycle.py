"""TASK-2268: startup recovery and fail-closed command-consumer health."""
from __future__ import annotations

import asyncio
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from redis.exceptions import ConnectionError, ResponseError

from agent_bridge.health_http import health_app
from agent_bridge.houses import HouseSlot
from claude_code_bridge.__main__ import _health_server, _wait_for_shutdown
from claude_code_bridge.bridge import ClaudeCodeBridge
from claude_code_bridge.command_ops import COMMANDS_STREAM


class FakeRedis:
    def __init__(self, fail=None, failures=1):
        self.fail = fail
        self.failures = failures
        self.calls = Counter()
        self.group = False
        self.group_ids = []
        self.consumed = asyncio.Event()
        self.reading = asyncio.Event()
        self.closed = False

    def step(self, name):
        self.calls[name] += 1
        if name == self.fail and self.calls[name] <= self.failures:
            raise ConnectionError(f"transient {name}")

    async def ping(self):
        self.step("ping")

    async def xgroup_create(self, stream, group, *, id, mkstream):
        self.step("group")
        self.group_ids.append(id)
        if self.group:
            raise ResponseError("BUSYGROUP already exists")
        self.group = True

    async def xinfo_groups(self, stream):
        return [{"name": "claude-code-bridge", "last-delivered-id": "0-0"}]

    async def incr(self, key):
        self.step("epoch")
        return self.calls["epoch"]

    async def set(self, *args, **kwargs):
        self.step("lease")

    async def delete(self, key):
        self.step("lease_cleanup")

    async def xreadgroup(self, group, consumer, streams, **kwargs):
        self.step("read")
        assert self.group
        if not self.consumed.is_set():
            return [(COMMANDS_STREAM, [("1-0", {"action": "unknown"})])]
        self.reading.set()
        await asyncio.Event().wait()

    async def xack(self, *args):
        self.consumed.set()

    async def aclose(self):
        self.calls["close"] += 1
        self.closed = True


def bridge_harness():
    bridge = ClaudeCodeBridge.__new__(ClaudeCodeBridge)
    bridge._publisher = Mock(connected=True)
    bridge._publisher._redis.connection_pool.connection_kwargs = {}
    bridge._backend_name = "claude-code"
    bridge._session_queue = Mock()
    bridge._command_task = None
    bridge._command_ready = False
    bridge._approval_reload_task = None
    bridge._approval_reload_listener = AsyncMock()
    return bridge


async def response(bridge):
    app = health_app(bridge._publisher, consumer_ready=lambda: bridge.command_consumer_ready)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        return await client.get("/health")


async def cleanup(bridge):
    bridge._command_task.cancel()
    await asyncio.gather(bridge._command_task, return_exceptions=True)
    slot = getattr(bridge, "_house_slot", None)
    tasks = [bridge._approval_reload_task, getattr(slot, "_task", None)]
    for task in tasks:
        if task is not None:
            task.cancel()
    await asyncio.gather(*(task for task in tasks if task is not None), return_exceptions=True)


@pytest.mark.parametrize("stage", ["ping", "group", "epoch", "lease", "lease_cleanup", "read"])
def test_transient_failure_recovers_and_health_tracks_consumer(monkeypatch, stage):
    async def run():
        redis = FakeRedis(stage)
        bridge = bridge_harness()
        delays = []

        async def backoff(delay):
            delays.append(delay)
            assert (await response(bridge)).status_code == 503

        monkeypatch.setenv("BAWT_HOUSE", "test-house")
        monkeypatch.delenv("BAWT_HOUSE_PREVIEW", raising=False)
        monkeypatch.setattr("redis.asyncio.Redis", lambda **kwargs: redis)
        monkeypatch.setattr("claude_code_bridge.command_ops.asyncio.sleep", backoff)
        monkeypatch.setattr(HouseSlot, "_heartbeat", AsyncMock(side_effect=lambda: None))
        assert (await response(bridge)).status_code == 503
        bridge._command_task = asyncio.create_task(bridge._command_listener())
        try:
            await redis.reading.wait()
            assert redis.consumed.is_set()
            assert delays == [2]
            assert redis.calls[stage] >= 2
            assert set(redis.group_ids) == {"0"}  # never skip startup backlog
            healthy = await response(bridge)
            assert healthy.status_code == 200
            assert healthy.json() == {"redis": True, "command_consumer": True}
            bridge._publisher.connected = False
            assert (await response(bridge)).status_code == 503
        finally:
            await cleanup(bridge)
        assert redis.calls["close"] == 1
        assert not bridge.command_consumer_ready
        assert (await response(bridge)).status_code == 503
    asyncio.run(asyncio.wait_for(run(), 5))


def test_startup_backoff_is_bounded_and_cancellation_closes_client(monkeypatch):
    async def run():
        redis = FakeRedis("ping", failures=99)
        bridge = bridge_harness()
        delays = []
        sleeping = asyncio.Event()

        async def backoff(delay):
            delays.append(delay)
            if len(delays) == 6:
                sleeping.set()
                await asyncio.Event().wait()

        monkeypatch.delenv("BAWT_HOUSE", raising=False)
        monkeypatch.setattr("redis.asyncio.Redis", lambda **kwargs: redis)
        monkeypatch.setattr("claude_code_bridge.command_ops.asyncio.sleep", backoff)
        bridge._command_task = asyncio.create_task(bridge._command_listener())
        await sleeping.wait()
        assert delays == [2, 4, 8, 16, 30, 30]
        assert (await response(bridge)).status_code == 503
        await cleanup(bridge)
        assert bridge._command_task.cancelled()
        assert redis.calls["close"] == 1
    asyncio.run(asyncio.wait_for(run(), 5))


@pytest.mark.parametrize("stage", ["ping", "group", "lease"])
def test_cancellation_during_startup_closes_client(monkeypatch, stage):
    async def run():
        redis = FakeRedis()
        bridge = bridge_harness()
        entered = asyncio.Event()

        async def blocked(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        method = {"ping": "ping", "group": "xgroup_create", "lease": "set"}[stage]
        monkeypatch.setattr(redis, method, blocked)
        monkeypatch.setenv("BAWT_HOUSE", "test-house")
        monkeypatch.setattr("redis.asyncio.Redis", lambda **kwargs: redis)
        bridge._command_task = asyncio.create_task(bridge._command_listener())
        await entered.wait()
        await cleanup(bridge)
        assert bridge._command_task.cancelled()
        assert redis.calls["close"] == 1
        assert not bridge.command_consumer_ready
    asyncio.run(asyncio.wait_for(run(), 5))


@pytest.mark.parametrize("exit_kind", ["error", "return", "cancel"])
def test_listener_exit_reaches_process_supervisor(exit_kind):
    async def run():
        bridge = bridge_harness()
        bridge.stop = AsyncMock()

        async def listener():
            if exit_kind == "error":
                raise ValueError("fatal listener")
            if exit_kind == "cancel":
                raise asyncio.CancelledError

        async def start():
            bridge._command_ready = True
            bridge._command_task = asyncio.create_task(listener())

        bridge.start = start
        supervisor = asyncio.create_task(bridge.run_forever())
        error = ValueError if exit_kind == "error" else RuntimeError
        with pytest.raises(error):
            await _wait_for_shutdown(supervisor, asyncio.Event())
        bridge.stop.assert_awaited_once()
        assert not bridge.command_consumer_ready
    asyncio.run(asyncio.wait_for(run(), 5))


def test_signal_shutdown_leaves_bridge_for_orderly_cancellation():
    async def run():
        bridge_task = asyncio.create_task(asyncio.Event().wait())
        event = asyncio.Event()
        event.set()
        await _wait_for_shutdown(bridge_task, event)
        assert not bridge_task.done()
        bridge_task.cancel()
        await asyncio.gather(bridge_task, return_exceptions=True)
    asyncio.run(asyncio.wait_for(run(), 5))


def test_health_server_wires_live_consumer_readiness(monkeypatch):
    async def run():
        serve = AsyncMock()
        monkeypatch.setattr("claude_code_bridge.__main__.serve_health", serve)
        bridge = SimpleNamespace(command_consumer_ready=False)
        publisher = Mock()
        await _health_server(bridge, publisher, 1234)
        callback = serve.call_args.kwargs["consumer_ready"]
        assert callback() is False
        bridge.command_consumer_ready = True
        assert callback() is True
    asyncio.run(run())
