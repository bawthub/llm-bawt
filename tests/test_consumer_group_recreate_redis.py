"""TASK-684 against a real Redis server (opt-in).

Set ``LLM_BAWT_TEST_REDIS_URL`` to a DISPOSABLE Redis (never the live one):
each test uses its own random stream keys and deletes them afterwards, but it
does destroy groups and delete streams on purpose.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
import redis.asyncio as aioredis

from agent_bridge.consumer_group import GroupReader, ensure_group

REDIS_URL = os.environ.get("LLM_BAWT_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not REDIS_URL, reason="LLM_BAWT_TEST_REDIS_URL not set")

GROUP = "bridge"


def _ids(results) -> list[str]:
    return [msg_id for _stream, messages in results for msg_id, _ in messages]


def _run(scenario):
    async def wrapper():
        client = aioredis.from_url(REDIS_URL, decode_responses=True, socket_timeout=None)
        stream = f"test:task684:{uuid.uuid4().hex}"
        try:
            await scenario(client, stream)
        finally:
            await client.delete(stream)
            await client.aclose()

    asyncio.run(wrapper())


@pytest.mark.parametrize("loss", ["destroy_group", "delete_stream"])
def test_idle_loss_keeps_command_published_before_recreation(loss):
    async def scenario(client, stream):
        await client.xadd(stream, {"n": "history"})
        reader = GroupReader(client, GROUP, "worker-0", [stream], initial_id="$")
        await reader.ensure()
        assert await reader.read(count=10, block=50) == []

        if loss == "destroy_group":
            await client.xgroup_destroy(stream, GROUP)
        else:
            await client.delete(stream)
        fresh = await client.xadd(stream, {"n": "fresh"})

        assert await reader.read(count=10, block=50) == []
        assert reader.recreations == 1
        assert _ids(await reader.read(count=10, block=50)) == [fresh]

    _run(scenario)


def test_delivered_entries_are_not_replayed_after_group_loss():
    async def scenario(client, stream):
        reader = GroupReader(client, GROUP, "worker-0", [stream], initial_id="0")
        await reader.ensure()
        first = await client.xadd(stream, {"n": "1"})
        assert _ids(await reader.read(count=10, block=50)) == [first]

        await client.xgroup_destroy(stream, GROUP)
        second = await client.xadd(stream, {"n": "2"})
        assert await reader.read(count=10, block=50) == []
        assert _ids(await reader.read(count=10, block=50)) == [second]

    _run(scenario)


def test_existing_group_is_kept_and_busygroup_tolerated():
    async def scenario(client, stream):
        await ensure_group(client, stream, GROUP, start_id="0")
        await ensure_group(client, stream, GROUP, start_id="0")  # real BUSYGROUP
        entry = await client.xadd(stream, {"n": "1"})
        reader = GroupReader(client, GROUP, "worker-0", [stream], initial_id="$")
        await reader.ensure()  # existing group keeps its own position
        assert _ids(await reader.read(count=10, block=50)) == [entry]

    _run(scenario)


def test_blocked_read_wakes_after_recreation():
    async def scenario(client, stream):
        reader = GroupReader(client, GROUP, "worker-0", [stream], initial_id="$")
        await reader.ensure()
        await client.xgroup_destroy(stream, GROUP)
        assert await reader.read(count=1, block=50) == []  # recreated

        pending = asyncio.create_task(reader.read(count=1, block=5000))
        await asyncio.sleep(0.2)
        entry = await client.xadd(stream, {"n": "1"})
        assert _ids(await asyncio.wait_for(pending, 3)) == [entry]

    _run(scenario)
