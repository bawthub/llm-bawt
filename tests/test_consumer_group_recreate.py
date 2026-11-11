"""TASK-684: listeners recreate a Redis consumer group that disappears mid-loop."""

from __future__ import annotations

import asyncio

import pytest
from redis.exceptions import ResponseError

from agent_bridge.consumer_group import (
    RETIRED_UI_GROUP_PREFIX,
    GroupReader,
    ensure_group,
    is_missing_group,
)
from agent_bridge.publisher import HISTORY_STREAM
from agent_bridge.subscriber import RedisSubscriber
from agent_bridge.ui_stream import LEGACY_UI_GROUP_PREFIX


def _key(stream_id: str) -> tuple[int, int]:
    ms, _, seq = stream_id.partition("-")
    return int(ms), int(seq or 0)  # Redis accepts a bare "0"


class FakeStreams:
    """The slice of async Redis stream semantics the readers use."""

    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict]]] = {}
        self.groups: dict[tuple[str, str], str] = {}
        self.read_errors = 0
        self._seq = 0

    def add(self, stream: str, fields: dict) -> str:
        self._seq += 1
        entry_id = f"{self._seq}-0"
        self.streams.setdefault(stream, []).append((entry_id, fields))
        return entry_id

    def delete_group(self, stream: str, group: str) -> None:
        del self.groups[(stream, group)]

    def delete_stream(self, stream: str) -> None:
        self.streams.pop(stream, None)
        for key in [k for k in self.groups if k[0] == stream]:
            del self.groups[key]

    async def xgroup_create(self, stream, group, id="$", mkstream=False):
        if stream not in self.streams:
            if not mkstream:
                raise ResponseError("The XGROUP subcommand requires the key to exist.")
            self.streams[stream] = []
        if (stream, group) in self.groups:
            raise ResponseError("BUSYGROUP Consumer Group name already exists")
        entries = self.streams[stream]
        self.groups[(stream, group)] = (entries[-1][0] if entries else "0-0") if id == "$" else id

    async def xreadgroup(self, group, consumer, streams, count=None, block=None):
        out = []
        for stream in streams:
            if (stream, group) not in self.groups:
                self.read_errors += 1
                raise ResponseError(
                    f"NOGROUP No such key '{stream}' or consumer group '{group}' "
                    "in XREADGROUP with GROUP option"
                )
            last = _key(self.groups[(stream, group)])
            fresh = [e for e in self.streams[stream] if _key(e[0]) > last][:count]
            if fresh:
                self.groups[(stream, group)] = fresh[-1][0]
                out.append((stream, fresh))
        if not out:
            await asyncio.sleep(0)  # stands in for BLOCK so loops yield
        return out

    async def xack(self, stream, group, msg_id):
        return 1


def _ids(results) -> list[str]:
    return [msg_id for _stream, messages in results for msg_id, _ in messages]


def test_predicate_matches_nogroup_and_missing_key_only():
    assert is_missing_group(ResponseError("NOGROUP No such key 's' or consumer group 'g'"))
    assert is_missing_group(ResponseError("ERR no such key"))
    assert not is_missing_group(ResponseError("WRONGTYPE Operation against a key"))
    assert not is_missing_group(RuntimeError("NOGROUP"))


def test_ensure_group_tolerates_existing_and_raises_other_errors():
    async def run():
        redis = FakeStreams()
        await ensure_group(redis, "s", "g", start_id="0")
        await ensure_group(redis, "s", "g", start_id="0")  # BUSYGROUP swallowed

        class Broken(FakeStreams):
            async def xgroup_create(self, *a, **kw):
                raise ResponseError("READONLY You can't write against a read only replica.")

        with pytest.raises(ResponseError):
            await ensure_group(Broken(), "s", "g", start_id="0")

        # Retired per-window UI groups stay retired (TASK-1582).
        with pytest.raises(ValueError):
            await ensure_group(redis, "s", "ui:window", start_id="$")
        assert RETIRED_UI_GROUP_PREFIX == LEGACY_UI_GROUP_PREFIX

    asyncio.run(run())


def test_deleted_group_is_recreated_without_replay_or_loss():
    async def run():
        redis = FakeStreams()
        first = redis.add("cmd", {"n": "1"})
        reader = GroupReader(redis, "bridge", "worker-0", ["cmd"], initial_id="0")
        await reader.ensure()
        assert _ids(await reader.read(count=10, block=1)) == [first]

        redis.delete_group("cmd", "bridge")
        assert await reader.read(count=10, block=1) == []
        assert reader.recreations == 1

        later = redis.add("cmd", {"n": "2"})
        # Resumes after the last entry seen: "1" is not replayed, "2" arrives.
        assert _ids(await reader.read(count=10, block=1)) == [later]
        assert redis.read_errors == 1

    asyncio.run(run())


def test_deleted_stream_is_recreated():
    async def run():
        redis = FakeStreams()
        reader = GroupReader(redis, "bridge", "worker-0", ["cmd"], initial_id="$")
        await reader.ensure()
        redis.delete_stream("cmd")
        assert await reader.read(count=1, block=1) == []
        entry = redis.add("cmd", {"n": "1"})
        assert _ids(await reader.read(count=1, block=1)) == [entry]

    asyncio.run(run())


def test_recreate_before_any_read_starts_at_new_entries():
    async def run():
        redis = FakeStreams()
        redis.add("cmd", {"n": "old"})
        reader = GroupReader(redis, "bridge", "worker-0", ["cmd"], initial_id="0")
        # Group never created (e.g. startup creation failed) -> recovered at "$",
        # so a recreated group does not execute historical commands.
        assert await reader.read(count=10, block=1) == []
        new = redis.add("cmd", {"n": "new"})
        assert _ids(await reader.read(count=10, block=1)) == [new]

    asyncio.run(run())


def test_other_response_errors_still_raise():
    class Wrong(FakeStreams):
        async def xreadgroup(self, *a, **kw):
            raise ResponseError("WRONGTYPE Operation against a key holding the wrong kind of value")

    reader = GroupReader(Wrong(), "bridge", "worker-0", ["cmd"], initial_id="0")
    with pytest.raises(ResponseError):
        asyncio.run(reader.read(count=1, block=1))


def test_history_drain_survives_group_deleted_mid_loop():
    """The production listener loop keeps delivering after NOGROUP, no error spin."""

    async def run():
        redis = FakeStreams()
        sub = RedisSubscriber.__new__(RedisSubscriber)
        sub._redis = redis
        got: list[str] = []
        task = asyncio.create_task(
            sub.drain_history(lambda bot, role, content: got.append(content))
        )
        try:
            redis.add(HISTORY_STREAM, {"bot_id": "b", "role": "user", "content": "before"})
            for _ in range(50):
                await asyncio.sleep(0)
            assert got == ["before"]

            redis.delete_group(HISTORY_STREAM, "llm-bawt")
            redis.add(HISTORY_STREAM, {"bot_id": "b", "role": "user", "content": "after"})
            for _ in range(200):
                await asyncio.sleep(0)
                if got == ["before", "after"]:
                    break
            assert got == ["before", "after"]
            assert redis.read_errors == 1  # recovered on the first NOGROUP, no 2 s retry loop
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
