"""Retained replay proof uses disposable Redis, never live run traffic."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import uuid

import pytest
import redis.asyncio as aioredis
from redis.exceptions import ConnectionError, ResponseError

from agent_bridge.events import AgentEvent, AgentEventKind
from agent_bridge.publisher import RUN_STREAM_PREFIX
from agent_bridge.run_contract import RunTerminal, TeardownReport
from agent_bridge.terminal_replay import ReplayDisposition, read_retained_run
from tests.test_terminal_publish import private_redis  # noqa: F401


@pytest.fixture
def replay_case(private_redis):  # noqa: F811
    def run(scenario, *, decode=True):
        async def wrapper():
            client = aioredis.Redis(unix_socket_path=private_redis, decode_responses=decode)
            identity = dict(request_id=uuid.uuid4().hex, session_key="fixture:user", backend="claude-code", boot_id="boot")
            key = RUN_STREAM_PREFIX + identity["request_id"]
            try:
                await scenario(client, identity, key)
            finally:
                await client.delete(key)
                await client.aclose()
        asyncio.run(wrapper())
    return run


def event(**overrides):
    values = dict(event_id=uuid.uuid4().hex, session_key="fixture:user", run_id="sdk-run",
                  kind=AgentEventKind.ASSISTANT_DELTA, origin="system", text="Hello 🙂")
    values.update(overrides)
    return AgentEvent(**values).to_dict()


def done(identity, **overrides):
    values = dict(request_id=identity["request_id"], backend=identity["backend"], boot_id="boot",
                  teardown=TeardownReport(exited=True))
    values.update(overrides)
    return RunTerminal(**values).to_fields()


@pytest.mark.parametrize("decode", [True, False])
def test_complete_replay_preserves_events_without_mutation(replay_case, decode):
    async def scenario(client, identity, key):
        original = event()
        await client.xadd(key, {"payload": json.dumps(original)})
        await client.xadd(key, done(identity))
        before = await client.xrange(key)
        result = await read_retained_run(client, **identity)
        assert result.disposition is ReplayDisposition.COMPLETE
        assert len(result.events) == 1 and result.events[0].text == original["text"]
        assert result.terminal.proves_exit
        assert await client.xrange(key) == before
        assert (await read_retained_run(client, **identity)).events == result.events
    replay_case(scenario, decode=decode)


def test_missing_is_distinct_from_active_stream(replay_case):
    async def scenario(client, identity, key):
        assert (await read_retained_run(client, **identity)).disposition is ReplayDisposition.MISSING
        await client.xadd(key, {"payload": json.dumps(event())})
        pending = await read_retained_run(client, **identity)
        assert pending.disposition is ReplayDisposition.PENDING and pending.events == ()
    replay_case(scenario)


@pytest.mark.parametrize("deletion", ["head", "middle", "all"])
def test_trim_and_delete_never_prove_complete(replay_case, deletion):
    async def scenario(client, identity, key):
        ids = [await client.xadd(key, {"payload": json.dumps(event())}) for _ in range(3)]
        terminal = await client.xadd(key, done(identity))
        if deletion == "head":
            await client.xtrim(key, maxlen=2, approximate=False)
        elif deletion == "middle":
            await client.xdel(key, ids[1])
        else:
            await client.xdel(key, *ids, terminal)
        result = await read_retained_run(client, **identity)
        assert result.disposition is ReplayDisposition.TRIMMED
        assert result.events == () and result.terminal is None
    replay_case(scenario)


@pytest.mark.parametrize("fault", ["bare_done", "wrong_request", "wrong_boot", "wrong_backend", "bad_done", "after_done",
                                   "unknown_kind", "wrong_session", "bad_json", "duplicate_event", "forged_terminal"])
def test_invalid_stream_falls_back_instead_of_inventing_success(replay_case, fault):
    async def scenario(client, identity, key):
        item = event()
        fields = done(identity)
        if fault == "wrong_session":
            item["session_key"] = "other:user"
        if fault == "unknown_kind":
            item["kind"] = "future_content_kind"
        if fault == "forged_terminal":
            item["kind"] = "run_terminal"
        await client.xadd(key, {"payload": "not JSON" if fault == "bad_json" else json.dumps(item)})
        if fault == "duplicate_event":
            await client.xadd(key, {"payload": json.dumps(item)})
        if fault == "bare_done":
            fields = {"done": "1"}
        elif fault == "wrong_request":
            fields["request_id"] = "other"
        elif fault == "wrong_boot":
            fields["boot_id"] = "other"
        elif fault == "wrong_backend":
            fields["backend"] = "other"
        elif fault == "bad_done":
            fields["done"] = "false"
        await client.xadd(key, fields)
        if fault == "after_done":
            await client.xadd(key, {"payload": json.dumps(event())})
        result = await read_retained_run(client, **identity)
        assert result.disposition is ReplayDisposition.INVALID
        assert result.events == ()
    replay_case(scenario)


@pytest.mark.parametrize("limit", [{"max_entries": 1}, {"max_payload_bytes": 1}])
def test_size_limits_are_explicit_not_truncated_success(replay_case, limit):
    async def scenario(client, identity, key):
        await client.xadd(key, {"payload": json.dumps(event())})
        await client.xadd(key, done(identity))
        result = await read_retained_run(client, **identity, **limit)
        assert result.disposition is ReplayDisposition.LIMIT and result.events == ()
    replay_case(scenario)


def test_acceptance_and_terminal_boot_must_match_even_without_pinned_boot(replay_case):
    async def scenario(client, identity, key):
        await client.xadd(key, {"payload": json.dumps(event(kind=AgentEventKind.RUN_ACCEPTED,
            raw={"request_id": identity["request_id"], "backend": "claude-code", "boot_id": "earlier"}))})
        await client.xadd(key, done(identity))
        identity["boot_id"] = None
        assert (await read_retained_run(client, **identity)).disposition is ReplayDisposition.INVALID
    replay_case(scenario)


def test_transport_failure_is_not_stream_absence():
    pipe = AsyncMock()
    pipe.__aenter__.return_value = pipe
    pipe.xinfo_stream = Mock()
    pipe.xrange = Mock()
    pipe.execute.side_effect = ConnectionError("test transport unavailable")
    client = SimpleNamespace(pipeline=Mock(return_value=pipe))
    with pytest.raises(ConnectionError):
        asyncio.run(read_retained_run(client, request_id="r", session_key="s", backend="b"))
    client.pipeline.assert_called_once_with(transaction=True)


def test_wrong_key_type_is_not_stream_absence(replay_case):
    async def scenario(client, identity, key):
        await client.set(key, "wrong-type")
        with pytest.raises(ResponseError):
            await read_retained_run(client, **identity)
    replay_case(scenario)
