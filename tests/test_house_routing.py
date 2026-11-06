"""TASK-2077: per-house command routing, follow-up routing and the fence."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest

from agent_bridge import houses
from agent_bridge.command_publisher import CommandPublisherMixin
from agent_bridge.houses import (
    EPOCH_FIELD,
    HouseSlot,
    follow_up_stream,
    gate_send,
    lease_key,
    normalize_house,
    route_new_command,
)
from agent_bridge.publisher import MANSION_COMMANDS_STREAM
from agent_bridge.run_contract import REFUSED
from agent_bridge.subscriber import RedisSubscriber

MANSION = MANSION_COMMANDS_STREAM
H1 = "agent:commands:house:h1"


class FakeRedis:
    """The subset of redis.asyncio the routing code uses."""

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.ttl: dict[str, int] = {}
        self.streams: dict[str, list[dict]] = {}
        self.acks: list[tuple[str, str, str]] = []
        self.reads = 0

    async def get(self, key):
        self.reads += 1
        return self.kv.get(key)

    async def set(self, key, value, *, nx=False, ex=None):
        if nx and key in self.kv:
            return None
        self.kv[key] = str(value)
        if ex is not None:
            self.ttl[key] = ex
        return True

    async def incr(self, key):
        self.kv[key] = str(int(self.kv.get(key, "0")) + 1)
        return int(self.kv[key])

    async def delete(self, *keys):
        for key in keys:
            self.kv.pop(key, None)
            self.ttl.pop(key, None)

    async def expire(self, key, ttl):
        if key in self.kv:
            self.ttl[key] = ttl

    async def xadd(self, stream, fields, **_):
        self.streams.setdefault(stream, []).append(dict(fields))
        return f"{len(self.streams[stream])}-0"

    async def xack(self, stream, group, msg_id):
        self.acks.append((stream, group, msg_id))

    async def eval(self, script, numkeys, *args):
        keys, argv = args[:numkeys], args[numkeys:]
        if keys[0] in self.kv:
            return [0, self.kv[keys[0]]]
        start = 2 if "ARGV[2], 1" in script else 1  # chat.send carries request_id first
        flat = argv[start:]
        fields = dict(zip(flat[::2], flat[1::2]))
        stream_id = await self.xadd(keys[1], fields)
        self.kv[keys[0]] = stream_id
        return [1, stream_id]

    async def xread(self, streams, count=None, block=None):
        (key, _), = streams.items()
        return [(key, [("1-0", {"payload": json.dumps({"ok": True})})])]


class Publisher(CommandPublisherMixin):
    def __init__(self, redis):
        self._pub_redis = redis


def subscriber(redis) -> RedisSubscriber:
    sub = RedisSubscriber.__new__(RedisSubscriber)
    sub._pub_redis = redis
    sub._redis = redis
    return sub


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------ validation


@pytest.mark.parametrize("raw,expected", [(None, ""), ("", ""), ("  H1 ", "h1"), ("pool-2", "pool-2")])
def test_house_ids_normalize(raw, expected):
    assert normalize_house(raw) == expected


@pytest.mark.parametrize("raw", ["-h1", "h_1", "h 1", "a" * 32, "h1:x"])
def test_bad_house_ids_are_refused(raw):
    with pytest.raises(ValueError):
        normalize_house(raw)


def test_bridge_stream_follows_bawt_house():
    code = "import agent_bridge.publisher as p; print(p.COMMANDS_STREAM)"
    env_base = {"PYTHONPATH": "src"}
    out = subprocess.run([sys.executable, "-c", code], env={**env_base}, capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == MANSION
    out = subprocess.run([sys.executable, "-c", code], env={**env_base, "BAWT_HOUSE": "h1"},
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == H1
    bad = subprocess.run([sys.executable, "-c", code], env={**env_base, "BAWT_HOUSE": "H_1"},
                         capture_output=True, text=True)
    assert bad.returncode != 0


# ----------------------------------------------------------- zero houses


def test_zero_houses_is_byte_identical():
    plain, housed = FakeRedis(), FakeRedis()
    run(Publisher(plain).send_command("al:nick", "hi", "req_1", backend="claude-code", bot_id="al"))
    run(Publisher(housed).send_command("al:nick", "hi", "req_1", backend="claude-code", bot_id="al",
                                       house=None))
    assert plain.streams == housed.streams
    assert list(plain.streams) == [MANSION]
    assert EPOCH_FIELD not in plain.streams[MANSION][0]
    assert housed.reads == 0  # no lease or turn lookups for a mansion bot


def test_zero_houses_follow_ups_stay_in_the_mansion():
    redis = FakeRedis()
    sub = subscriber(redis)
    run(sub.send_tool_result("al:nick", "tu_1", "x", backend="claude-code", request_id="req_1"))
    run(sub.send_approval_grant(session_key="al:nick", grant_key="g", backend="claude-code"))
    run(sub.send_rpc("chat.abort", {"sessionKey": "al:nick", "requestId": "req_1"}, "abort_1",
                     backend="claude-code", bot_id="al"))
    assert list(redis.streams) == [MANSION]
    assert redis.streams[MANSION][2]["bot_id"] == "al"


# ----------------------------------------------------------- dispatch


def test_live_house_gets_new_commands_with_epoch_and_turn_record():
    redis = FakeRedis()
    redis.kv[lease_key("h1", "claude-code")] = "7"
    run(Publisher(redis).send_command("al:nick", "hi", "req_1", backend="claude-code", bot_id="al",
                                      house="h1"))
    assert list(redis.streams) == [H1]
    assert redis.streams[H1][0][EPOCH_FIELD] == "7"
    assert redis.kv[houses.turn_request_key("req_1")] == H1
    assert redis.kv[houses.turn_session_key("claude-code", "al:nick")] == H1


def test_lease_is_per_backend():
    redis = FakeRedis()
    redis.kv[lease_key("h1", "codex")] = "3"
    stream, epoch = run(route_new_command(redis, house="h1", backend="claude-code", bot_id="al"))
    assert (stream, epoch) == (MANSION, None)


def test_dead_house_falls_back_with_one_notice():
    redis, notices = FakeRedis(), []

    async def notify(house, backend, bot_id):
        notices.append((house, backend, bot_id))

    for _ in range(3):
        stream, epoch = run(route_new_command(redis, house="h1", backend="codex", bot_id="vex",
                                              notify=notify))
        assert (stream, epoch) == (MANSION, None)
    assert notices == [("h1", "codex", "vex")]


def test_follow_ups_follow_the_turn():
    redis = FakeRedis()
    redis.kv[lease_key("h1", "claude-code")] = "1"
    sub = subscriber(redis)
    run(sub.send_command("al:nick", "hi", "req_1", backend="claude-code", bot_id="al", house="h1"))
    run(sub.send_steer(session_key="al:nick", message="more", message_id="m1",
                       backend="claude-code", target_request_id="req_1", timeout_s=1))
    run(sub.send_tool_result("al:nick", "tu_1", "x", backend="claude-code", request_id="req_1"))
    run(sub.send_approval_grant(session_key="al:nick", grant_key="g", backend="claude-code",
                                request_id="req_1"))
    run(sub.send_rpc("chat.abort", {"sessionKey": "al:nick", "requestId": "req_1"}, "abort_1",
                     backend="claude-code", bot_id="al"))
    actions = [f["action"] for f in redis.streams[H1]]
    assert actions == ["chat.send", "chat.steer", "chat.tool_result", "approval.grant", "rpc.call"]
    assert MANSION not in redis.streams


# ------------------------------------------------------------- the fence


class Queue:
    def __init__(self):
        self.live: set[tuple[str, str]] = set()

    def request_phase(self, session_key, request_id):
        return "running" if (session_key, request_id) in self.live else None


class Done:
    def __init__(self):
        self.done: list[tuple[str, object]] = []

    def publish_run_done(self, request_id, *, teardown):
        self.done.append((request_id, teardown))


def test_mansion_slot_is_inert():
    async def go():
        redis = FakeRedis()
        slot = HouseSlot("", "claude-code", Queue())
        await slot.start(redis)
        assert slot.admit({"request_id": "r"})
        assert await gate_send(slot, {"request_id": "r", "session_key": "s"}, "1-0", redis,
                               group="g", publisher=Done())
        assert redis.kv == {} and redis.acks == []
    run(go())


def test_killed_house_never_double_executes():
    async def go():
        redis, queue, done = FakeRedis(), Queue(), Done()
        slot = HouseSlot("h1", "claude-code", queue)
        await slot.start(redis)
        assert redis.kv[lease_key("h1", "claude-code")] == "1"

        # A turn runs in the house, and another send is queued behind it.
        pub = Publisher(redis)
        await pub.send_command("al:nick", "a", "req_a", backend="claude-code", house="h1")
        await pub.send_command("al:nick", "b", "req_b", backend="claude-code", house="h1")
        sent_a, sent_b = redis.streams[H1]
        assert await gate_send(slot, sent_a, "1-0", redis, group="g", publisher=done)
        queue.live.add(("al:nick", "req_a"))
        slot._task.cancel()

        # The house dies: its lease expires. New work goes to the mansion with a
        # notice; the in-flight turn's follow-ups still go to the house.
        await redis.delete(lease_key("h1", "claude-code"))
        notices = []

        async def notify(*args):
            notices.append(args)

        await pub.send_command("al:nick", "c", "req_c", backend="claude-code", house="h1",
                               on_house_fallback=notify)
        assert [f["request_id"] for f in redis.streams[MANSION]] == ["req_c"]
        assert len(notices) == 1
        assert await follow_up_stream(redis, request_id="req_a") == H1

        # The house restarts with a new epoch: the send queued under the old
        # one is refused (terminal + ack), never run.
        reborn = HouseSlot("h1", "claude-code", Queue())
        await reborn.start(redis)
        assert reborn.epoch == "2"
        assert not await gate_send(reborn, sent_b, "2-0", redis, group="g", publisher=done)
        assert done.done == [("req_b", REFUSED)]
        assert (H1, "g", "2-0") in redis.acks
        reborn._task.cancel()

        # A republish of the same request id can't land anywhere new.
        await pub.send_command("al:nick", "b", "req_b", backend="claude-code", house="h1")
        assert len(redis.streams[H1]) == 2
    run(go())


def test_heartbeat_renews_live_turns_and_drops_finished_ones():
    async def go():
        redis, queue = FakeRedis(), Queue()
        slot = HouseSlot("h1", "codex", queue)
        await slot.start(redis)
        slot._task.cancel()
        slot.GRACE_S = 0
        slot.track({"request_id": "r1", "session_key": "s"})
        await asyncio.sleep(0)
        queue.live.add(("s", "r1"))
        await slot.refresh()
        assert redis.kv[houses.turn_request_key("r1")] == H1
        queue.live.clear()
        await slot.refresh()
        assert houses.turn_request_key("r1") not in redis.kv
        assert houses.turn_session_key("codex", "s") not in redis.kv
        assert redis.ttl[lease_key("h1", "codex")] == houses.LEASE_TTL_S
    run(go())
