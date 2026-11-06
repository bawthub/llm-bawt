"""TASK-2080: a vacated house restarts itself, never under a running turn."""
from __future__ import annotations

import asyncio

import pytest

from agent_bridge import houses
from agent_bridge.houses import (
    EPOCH_FIELD,
    RECYCLE_ACTION,
    HouseSlot,
    gate_send,
    is_recycle,
    lease_key,
    route_new_command,
)
from agent_bridge.publisher import MANSION_COMMANDS_STREAM

H1 = "agent:commands:house:h1"
GROUP = "codex-bridge"


class FakeRedis:
    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.streams: dict[str, list[dict]] = {}
        self.acks: list[tuple[str, str, str]] = []
        self.lag = 0

    async def get(self, key):
        return self.kv.get(key)

    async def set(self, key, value, *, nx=False, ex=None):
        if nx and key in self.kv:
            return None
        self.kv[key] = str(value)
        return True

    async def incr(self, key):
        self.kv[key] = str(int(self.kv.get(key, "0")) + 1)
        return int(self.kv[key])

    async def delete(self, *keys):
        for key in keys:
            self.kv.pop(key, None)

    async def expire(self, key, ttl):
        return key in self.kv

    async def eval(self, script, numkeys, *args):
        assert script == houses._RELEASE_LEASE_LUA
        (key,), (value,) = args[:numkeys], args[numkeys:]
        if self.kv.get(key) == value:
            del self.kv[key]
            return 1
        return 0

    async def scan_iter(self, match):
        for key in list(self.kv):
            if key.startswith(match.rstrip("*")):
                yield key

    async def xadd(self, stream, fields, **_):
        self.streams.setdefault(stream, []).append(dict(fields))
        return f"{len(self.streams[stream])}-0"

    async def xack(self, stream, group, msg_id):
        self.acks.append((stream, group, msg_id))

    async def xinfo_groups(self, stream):
        return [{"name": GROUP, "lag": self.lag}]


class Queue:
    def __init__(self):
        self.live: set[tuple[str, str]] = set()

    def request_phase(self, session_key, request_id):
        return "running" if (session_key, request_id) in self.live else None


class Publisher:
    def __init__(self):
        self.done = []

    def publish_run_done(self, request_id, teardown):
        self.done.append(request_id)


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    monkeypatch.setattr(houses, "RECYCLE_POLL_S", 0.001)


async def _house(redis, queue=None):
    slot = HouseSlot("h1", "codex", queue or Queue())
    await slot.start(redis)
    slot._task.cancel()
    return slot


def _control(slot, **over):
    return {"action": RECYCLE_ACTION, "backend": "codex", "house": "h1",
            EPOCH_FIELD: slot.epoch, **over}


class SyncRedis:
    def __init__(self, kv):
        self.kv = kv
        self.streams: dict[str, list[dict]] = {}

    def scan_iter(self, match):
        return [k for k in list(self.kv) if k.startswith(match.rstrip("*"))]

    def get(self, key):
        return self.kv.get(key)

    def xadd(self, stream, fields, **_):
        self.streams.setdefault(stream, []).append(dict(fields))


def test_app_asks_each_live_backend_with_its_epoch(monkeypatch):
    from llm_bawt import house_lease

    client = SyncRedis({lease_key("h1", "codex"): "7", lease_key("h10", "codex"): "3"})
    monkeypatch.setattr(house_lease, "_publish", lambda config, action: action(client))
    assert house_lease.request_recycle(object(), "h1") == ["codex"]
    assert client.streams == {H1: [{"action": RECYCLE_ACTION, "backend": "codex",
                                    "house": "h1", EPOCH_FIELD: "7"}]}
    # A dead house (no lease) and the mansion get nothing.
    assert house_lease.request_recycle(object(), "h3") == []
    assert house_lease.request_recycle(object(), "") == []
    assert set(client.streams) == {H1}


def test_idle_house_releases_its_lease_acks_and_exits():
    async def go():
        redis = FakeRedis()
        slot = await _house(redis)
        exits = []
        assert is_recycle(slot, _control(slot))
        assert await slot.recycle(_control(slot), "9-0", redis, group=GROUP,
                                  exit_process=lambda: exits.append(1))
        assert exits == [1] and redis.acks == [(H1, GROUP, "9-0")]
        assert lease_key("h1", "codex") not in redis.kv
        # New work for the house falls back while it restarts (T6 fallback).
        stream, epoch = await route_new_command(redis, house="h1", backend="codex", bot_id="b")
        assert (stream, epoch) == (MANSION_COMMANDS_STREAM, None)
        # The heartbeat of the exiting process never takes the lease back.
        await slot.refresh()
        assert lease_key("h1", "codex") not in redis.kv
    asyncio.run(go())


def test_in_flight_turn_is_never_evicted():
    async def go():
        redis, queue = FakeRedis(), Queue()
        slot = await _house(redis, queue)
        slot.GRACE_S = 0
        gate = await gate_send(slot, {"request_id": "r1", "session_key": "s",
                                      EPOCH_FIELD: slot.epoch}, "1-0", redis,
                               group=GROUP, publisher=Publisher())
        assert gate
        queue.live.add(("s", "r1"))
        exits = []
        task = asyncio.create_task(slot.recycle(
            _control(slot), "2-0", redis, group=GROUP, exit_process=lambda: exits.append(1)))
        for _ in range(20):
            await asyncio.sleep(0.002)
        assert not exits and not task.done() and redis.acks == []
        queue.live.clear()
        assert await asyncio.wait_for(task, 1)
        assert exits == [1]
    asyncio.run(go())


def test_unread_send_on_the_stream_holds_the_exit():
    async def go():
        redis = FakeRedis()
        slot = await _house(redis)
        redis.lag = 1
        exits = []
        task = asyncio.create_task(slot.recycle(
            _control(slot), "2-0", redis, group=GROUP, exit_process=lambda: exits.append(1)))
        for _ in range(10):
            await asyncio.sleep(0.002)
        assert not exits
        redis.lag = 0
        assert await asyncio.wait_for(task, 1)
    asyncio.run(go())


@pytest.mark.parametrize("over", [{EPOCH_FIELD: "0"}, {"backend": "claude-code"}])
def test_stale_or_foreign_control_is_acked_and_ignored(over):
    async def go():
        redis = FakeRedis()
        slot = await _house(redis)
        exits = []
        assert not await slot.recycle(_control(slot, **over), "3-0", redis, group=GROUP,
                                      exit_process=lambda: exits.append(1))
        assert exits == [] and redis.acks == [(H1, GROUP, "3-0")]
        assert redis.kv[lease_key("h1", "codex")] == slot.epoch
        assert not slot.recycling
    asyncio.run(go())


def test_restarted_house_ignores_its_predecessors_control():
    async def go():
        redis = FakeRedis()
        old = await _house(redis)
        control = _control(old)
        new = await _house(redis)  # Docker brought the house back: next epoch
        exits = []
        assert not await new.recycle(control, "4-0", redis, group=GROUP,
                                     exit_process=lambda: exits.append(1))
        assert exits == [] and redis.kv[lease_key("h1", "codex")] == new.epoch
    asyncio.run(go())


def test_mansion_never_handles_recycle():
    slot = HouseSlot("", "codex", Queue())
    assert not is_recycle(slot, {"action": RECYCLE_ACTION})
    assert not is_recycle(None, {"action": RECYCLE_ACTION})
