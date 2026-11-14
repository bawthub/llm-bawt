"""Focused terminal publication contract and atomic Redis regressions."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import os
import shutil
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
import uuid

import pytest
import redis
import redis.asyncio as aioredis
from redis.exceptions import ConnectionError, ResponseError, TimeoutError

from agent_bridge.publisher import UNIFIED_EVENTS_PREFIX, UNIFIED_STREAM_MAXLEN
from agent_bridge.terminal_publish import (
    TERMINAL_DEDUPE_TTL_SECONDS,
    TerminalPublishResult,
    TerminalPublishStatus,
    _PUBLISH_TERMINAL_LUA,
    publish_terminal,
    terminal_publication_key,
)


@pytest.fixture
def publication():
    identity = {"bot_id": "fixture-bot", "user_id": "fixture-user", "turn_id": "fixture-turn"}
    return {
        **identity,
        "publication_deadline_epoch": 2_000_000_000.125,
        "completion": {
            **identity,
            "_type": "turn_complete",
            "assistant_message_id": "fixture-assistant",
            "session_id": "pinned-session",
            "status": "error",
            "end_reason": "error",
            "response_text": "Frozen café reply",
            "reasoning": "Frozen reasoning",
            "text_offset": 17,
            "attachments": [{"id": "frozen-attachment"}],
            "token_usage": {"input_tokens": 31, "output_tokens": 7},
            "changed_files": [{"path": "frozen.txt"}],
            "balance_update": {"remaining": 4.25},
            "animation": "idle",
            "question_id": "frozen-question",
            "approval_id": "frozen-approval",
            "error_text": "Frozen failure",
            "tts_text": "Frozen speech",
            "ts": 1_999_999_999.5,
        },
        "tts_tail": {
            **identity,
            "_type": "tts_delta",
            "tts_offset": 12,
            "delta": "Frozen speech tail",
            "ts": 1_999_999_999.25,
        },
    }


def test_one_atomic_eval_preserves_full_frozen_payloads_and_stream_convention(publication):
    client = SimpleNamespace(eval=AsyncMock(return_value=["published", "2000000000000-1"]))
    original = deepcopy(publication)

    result = asyncio.run(publish_terminal(client, **publication))

    assert result == TerminalPublishResult(TerminalPublishStatus.PUBLISHED, "2000000000000-1")
    assert publication == original
    client.eval.assert_awaited_once()
    script, key_count, stream, receipt, deadline, maxlen, ttl, completion, tail = (
        client.eval.await_args.args
    )
    assert script == _PUBLISH_TERMINAL_LUA
    assert key_count == 2
    assert stream == "events:fixture-bot:fixture-user"
    assert receipt == terminal_publication_key("fixture-bot", "fixture-user", "fixture-turn")
    assert float(deadline) == publication["publication_deadline_epoch"]
    assert maxlen == UNIFIED_STREAM_MAXLEN
    assert ttl == TERMINAL_DEDUPE_TTL_SECONDS == 604800
    assert json.loads(completion) == original["completion"]
    assert json.loads(tail) == original["tts_tail"]


@pytest.mark.parametrize(
    ("response", "status", "stream_id"),
    [
        ([b"published", b"100-1"], TerminalPublishStatus.PUBLISHED, "100-1"),
        ([b"already_published", b"100-1"], TerminalPublishStatus.ALREADY_PUBLISHED, "100-1"),
        ([b"expired", b""], TerminalPublishStatus.EXPIRED, None),
        (["expired", ""], TerminalPublishStatus.EXPIRED, None),
    ],
)
def test_decoded_and_binary_redis_responses(publication, response, status, stream_id):
    client = SimpleNamespace(eval=AsyncMock(return_value=response))
    publication["tts_tail"] = None

    result = asyncio.run(publish_terminal(client, **publication))

    assert result == TerminalPublishResult(status, stream_id)
    assert client.eval.await_args.args[-1] == ""


def test_serialization_finishes_before_redis_is_awaited(publication):
    original = deepcopy(publication)

    async def mutate_after_snapshot(*args):
        publication["completion"]["token_usage"]["output_tokens"] = 999
        publication["tts_tail"]["delta"] = "changed later"
        assert json.loads(args[-2]) == original["completion"]
        assert json.loads(args[-1]) == original["tts_tail"]
        return ["published", "100-1"]

    client = SimpleNamespace(eval=AsyncMock(side_effect=mutate_after_snapshot))
    asyncio.run(publish_terminal(client, **publication))


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("bot_id", None, TypeError),
        ("user_id", "", ValueError),
        ("turn_id", "  ", ValueError),
        ("bot_id", "\ud800", UnicodeEncodeError),
        ("publication_deadline_epoch", True, TypeError),
        ("publication_deadline_epoch", "2000000000", TypeError),
        ("publication_deadline_epoch", float("nan"), ValueError),
        ("publication_deadline_epoch", float("inf"), ValueError),
        ("publication_deadline_epoch", -1, ValueError),
        ("publication_deadline_epoch", 10**400, ValueError),
        ("completion", [], TypeError),
        ("tts_tail", "", TypeError),
    ],
)
def test_invalid_arguments_fail_before_redis(publication, field, value, error):
    client = SimpleNamespace(eval=AsyncMock())
    publication[field] = value

    with pytest.raises(error):
        asyncio.run(publish_terminal(client, **publication))

    client.eval.assert_not_awaited()


@pytest.mark.parametrize("event_name", ["completion", "tts_tail"])
@pytest.mark.parametrize("field", ["_type", "bot_id", "user_id", "turn_id"])
def test_event_identity_and_kind_must_match_before_redis(publication, event_name, field):
    client = SimpleNamespace(eval=AsyncMock())
    publication[event_name][field] = "wrong"

    with pytest.raises(ValueError, match=field):
        asyncio.run(publish_terminal(client, **publication))

    client.eval.assert_not_awaited()


@pytest.mark.parametrize("event_name", ["completion", "tts_tail"])
@pytest.mark.parametrize("invalid_value", [object(), float("nan"), float("inf"), "\ud800"])
def test_unserializable_event_fails_before_any_write(publication, event_name, invalid_value):
    client = SimpleNamespace(eval=AsyncMock())
    publication[event_name]["extra"] = invalid_value

    with pytest.raises((TypeError, ValueError)):
        asyncio.run(publish_terminal(client, **publication))

    client.eval.assert_not_awaited()


def test_circular_tail_cannot_leave_completion_or_tail_written(publication):
    client = SimpleNamespace(eval=AsyncMock())
    publication["tts_tail"]["cycle"] = publication["tts_tail"]

    with pytest.raises(ValueError, match="Circular reference"):
        asyncio.run(publish_terminal(client, **publication))

    client.eval.assert_not_awaited()


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("connection lost"),
        TimeoutError("acknowledgment lost"),
        ResponseError("WRONGTYPE"),
        asyncio.CancelledError("cancelled"),
    ],
)
def test_real_errors_and_cancellation_propagate_unchanged(publication, error):
    client = SimpleNamespace(eval=AsyncMock(side_effect=error))

    with pytest.raises(type(error)) as raised:
        asyncio.run(publish_terminal(client, **publication))

    assert raised.value is error


def test_receipt_identity_is_stable_scoped_and_delimiter_safe():
    original = terminal_publication_key("bot", "user", "turn")
    assert terminal_publication_key("bot", "user", "turn") == original
    assert len({
        original,
        terminal_publication_key("other-bot", "user", "turn"),
        terminal_publication_key("bot", "other-user", "turn"),
        terminal_publication_key("bot", "user", "other-turn"),
    }) == 4
    assert terminal_publication_key("a:b", "c", "turn") != terminal_publication_key(
        "a", "b:c", "turn"
    )


@pytest.fixture(scope="module")
def private_redis(tmp_path_factory):
    """Test-owned Redis only: no TCP port, inherited URL or persisted data."""
    executable = shutil.which("redis-server")
    directory = tmp_path_factory.mktemp("terminal-redis")
    socket = directory / "redis.sock"
    log = directory / "redis.log"
    container = None
    if executable is not None:
        command = [
            executable, "--port", "0", "--unixsocket", str(socket),
            "--unixsocketperm", "700", "--save", "", "--appendonly", "no",
            "--dir", str(directory), "--loglevel", "warning",
        ]
    else:
        if shutil.which("docker") is None:
            pytest.fail("terminal publication tests require redis-server or Docker")
        container = f"terminal-publish-test-{uuid.uuid4().hex}"
        command = [
            "docker", "run", "--rm", "--network", "none", "--name", container,
            "--user", f"{os.getuid()}:{os.getgid()}", "--memory", "128m", "--cpus", "1",
            "--mount", f"type=bind,source={directory},target=/data",
            "redis@sha256:8da19c7399dca900c6822a1cf30f9d2285585e2c4b457ff4095b0f4500054d3f",
            "redis-server", "--port", "0", "--unixsocket", "/data/redis.sock",
            "--unixsocketperm", "700", "--save", "", "--appendonly", "no",
            "--dir", "/data", "--loglevel", "warning",
        ]
    with log.open("wb") as output:
        process = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        probe = redis.Redis(
            unix_socket_path=str(socket),
            socket_timeout=0.1,
            socket_connect_timeout=0.1,
        )
        try:
            # Readiness polling, not an assertion about scheduler timing.
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    pytest.fail(f"private Redis exited: {log.read_text()}")
                try:
                    if probe.ping():
                        break
                except (ConnectionError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    pytest.fail(f"private Redis did not become ready: {log.read_text()}")
                time.sleep(0.01)
            yield str(socket)
        finally:
            probe.close()
            if container is not None:
                subprocess.run(
                    ["docker", "stop", "--time", "5", container],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
                )
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if container is not None:
                subprocess.run(
                    ["docker", "rm", "--force", container],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
                )


@pytest.fixture
def redis_case(private_redis, publication):
    """Run a scenario with fresh identities on the private Unix socket."""
    def run(scenario, *, decode_responses=True):
        async def wrapper():
            client = aioredis.Redis(
                unix_socket_path=private_redis,
                decode_responses=decode_responses,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            frozen = deepcopy(publication)
            identity = {
                "bot_id": f"terminal-test-{uuid.uuid4().hex}",
                "user_id": "private-user",
                "turn_id": uuid.uuid4().hex,
            }
            frozen.update(identity)
            frozen["completion"].update(identity)
            frozen["tts_tail"].update(identity)
            stream = f"{UNIFIED_EVENTS_PREFIX}{identity['bot_id']}:{identity['user_id']}"
            receipt = terminal_publication_key(**identity)
            try:
                seconds, microseconds = await client.time()
                frozen["publication_deadline_epoch"] = seconds + microseconds / 1_000_000 + 3600
                await scenario(client, frozen, stream, receipt)
            finally:
                try:
                    await client.delete(stream, receipt)
                finally:
                    await client.aclose()

        return asyncio.run(wrapper())

    return run


class TestAtomicRedisPublication:
    pytestmark = [pytest.mark.integration, pytest.mark.offline_integration]

    def test_tail_is_immediately_before_completion_and_receipt_has_seven_day_ttl(self, redis_case):
        async def scenario(client, frozen, stream, receipt):
            previous = await client.xadd(stream, {"payload": '{"_type":"text_delta"}'})
            original = deepcopy(frozen)

            result = await publish_terminal(client, **frozen)

            entries = await client.xrange(stream)
            assert [entry_id for entry_id, _ in entries][0] == previous
            assert len(entries) == 3
            assert [set(fields) for _, fields in entries] == [{"payload"}] * 3
            assert json.loads(entries[1][1]["payload"]) == original["tts_tail"]
            assert json.loads(entries[2][1]["payload"]) == original["completion"]
            assert result.status is TerminalPublishStatus.PUBLISHED
            assert result.completion_stream_id == entries[2][0]
            assert await client.get(receipt) == result.completion_stream_id
            assert TERMINAL_DEDUPE_TTL_SECONDS - 60 <= await client.ttl(receipt) <= 604800
            assert frozen == original

        redis_case(scenario)

    def test_completion_without_tail_works_with_binary_redis_responses(self, redis_case):
        async def scenario(client, frozen, stream, receipt):
            frozen["tts_tail"] = None

            result = await publish_terminal(client, **frozen)

            entries = await client.xrange(stream)
            assert len(entries) == 1
            assert json.loads(entries[0][1][b"payload"]) == frozen["completion"]
            assert result.completion_stream_id == entries[0][0].decode()
            assert await client.get(receipt) == entries[0][0]

        redis_case(scenario, decode_responses=False)

    @pytest.mark.parametrize("with_tail", [False, True])
    def test_retry_after_redis_commit_before_receipt_returns_original_id(self, redis_case, with_tail):
        async def scenario(client, frozen, stream, receipt):
            if not with_tail:
                frozen["tts_tail"] = None
            committed = []

            async def lose_acknowledgment(*args):
                committed.append(await client.eval(*args))
                raise TimeoutError("Redis committed, but acknowledgment was lost")

            lost_ack_client = SimpleNamespace(eval=lose_acknowledgment)
            with pytest.raises(TimeoutError, match="acknowledgment was lost"):
                await publish_terminal(lost_ack_client, **frozen)
            original_entries = await client.xrange(stream)
            # Shorten TTL to expose accidental retry refresh without sleeps.
            await client.pexpire(receipt, 120_000)

            retried = await publish_terminal(client, **frozen)

            assert retried.status is TerminalPublishStatus.ALREADY_PUBLISHED
            assert retried.completion_stream_id == committed[0][1] == original_entries[-1][0]
            assert await client.xrange(stream) == original_entries
            assert len(original_entries) == (2 if with_tail else 1)
            assert 0 < await client.pttl(receipt) <= 120_000

        redis_case(scenario)

    def test_concurrent_publishers_produce_one_ordered_pair(self, redis_case):
        async def scenario(client, frozen, stream, receipt):
            first, second = await asyncio.gather(
                publish_terminal(client, **frozen),
                publish_terminal(client, **frozen),
            )

            assert {first.status, second.status} == {
                TerminalPublishStatus.PUBLISHED,
                TerminalPublishStatus.ALREADY_PUBLISHED,
            }
            assert first.completion_stream_id == second.completion_stream_id
            entries = await client.xrange(stream)
            assert [json.loads(fields["payload"])["_type"] for _, fields in entries] == [
                "tts_delta", "turn_complete",
            ]
            assert await client.get(receipt) == entries[-1][0]

        redis_case(scenario)

    @pytest.mark.parametrize("existing_stream", [False, True])
    def test_expired_redis_deadline_writes_nothing_regardless_of_app_clock(
        self, redis_case, monkeypatch, existing_stream
    ):
        async def scenario(client, frozen, stream, receipt):
            if existing_stream:
                await client.xadd(stream, {"payload": '{"_type":"text_delta"}'})
            original = await client.xrange(stream)
            seconds, _ = await client.time()
            frozen["publication_deadline_epoch"] = float(seconds) - 1

            with monkeypatch.context() as patch:
                patch.setattr(time, "time", lambda: 0.0)
                result = await publish_terminal(client, **frozen)

            assert result == TerminalPublishResult(TerminalPublishStatus.EXPIRED, None)
            assert await client.xrange(stream) == original
            assert await client.exists(receipt) == 0
            assert bool(await client.exists(stream)) is existing_stream

        redis_case(scenario)

    def test_future_redis_deadline_ignores_application_clock_ahead(self, redis_case, monkeypatch):
        async def scenario(client, frozen, stream, receipt):
            with monkeypatch.context() as patch:
                patch.setattr(time, "time", lambda: 9_000_000_000_000.0)
                result = await publish_terminal(client, **frozen)

            assert result.status is TerminalPublishStatus.PUBLISHED
            assert await client.xlen(stream) == 2

        redis_case(scenario)

    def test_retained_receipt_wins_after_deadline_even_if_stream_is_gone(self, redis_case):
        async def scenario(client, frozen, stream, receipt):
            # Model an old acknowledged publication, now past its DB deadline
            # with the UI stream deleted/trimmed but its dedupe receipt retained.
            await client.set(receipt, "42-7", ex=120)
            frozen["publication_deadline_epoch"] = 0.0

            result = await publish_terminal(client, **frozen)

            assert result == TerminalPublishResult(TerminalPublishStatus.ALREADY_PUBLISHED, "42-7")
            assert await client.exists(stream) == 0
            assert 0 < await client.ttl(receipt) <= 120

        redis_case(scenario)

    @pytest.mark.parametrize(
        ("bad_key", "kind"),
        [
            ("stream", "string"),
            ("stream", "list"),
            ("stream", "hash"),
            ("receipt", "list"),
            ("receipt", "hash"),
            ("receipt", "stream"),
        ],
    )
    def test_wrong_key_types_raise_without_any_partial_write(self, redis_case, bad_key, kind):
        async def scenario(client, frozen, stream, receipt):
            key = stream if bad_key == "stream" else receipt
            if bad_key == "receipt":
                await client.xadd(stream, {"payload": '{"_type":"text_delta"}'})
            if kind == "string":
                await client.set(key, "wrong-type")
            elif kind == "list":
                await client.lpush(key, "wrong-type")
            elif kind == "hash":
                await client.hset(key, "field", "wrong-type")
            else:
                await client.xadd(key, {"field": "wrong-type"})
            before = (await client.dump(stream), await client.dump(receipt))

            with pytest.raises(ResponseError, match="WRONGTYPE"):
                await publish_terminal(client, **frozen)

            assert (await client.dump(stream), await client.dump(receipt)) == before

        # DUMP is binary regardless of the value's original encoding.
        redis_case(scenario, decode_responses=False)

    @pytest.mark.parametrize(
        ("index", "invalid"),
        [
            (0, "not-a-deadline"),
            (0, "-1"),
            (1, "1.5"),
            (1, "5000.0"),
            (1, "05000"),
            (1, "0"),
            (2, "604800.0"),
            (2, "1e3"),
            (2, "0604800"),
            (2, "0"),
            (3, '{"_type": "turn_complete"'),
            (3, "[]"),
            (4, '{"_type": "tts_delta"'),
            (4, '{"_type": "wrong"}'),
        ],
    )
    def test_lua_prevalidates_arguments_before_tail_or_completion(self, redis_case, index, invalid):
        async def scenario(client, frozen, stream, receipt):
            args = [
                repr(frozen["publication_deadline_epoch"]),
                str(UNIFIED_STREAM_MAXLEN),
                str(TERMINAL_DEDUPE_TTL_SECONDS),
                json.dumps(frozen["completion"]),
                json.dumps(frozen["tts_tail"]),
            ]
            args[index] = invalid

            with pytest.raises(ResponseError, match="terminal publication: invalid"):
                await client.eval(_PUBLISH_TERMINAL_LUA, 2, stream, receipt, *args)

            assert await client.exists(stream, receipt) == 0

        redis_case(scenario)

    @pytest.mark.parametrize("with_tail", [False, True])
    def test_stream_id_exhaustion_is_detected_before_partial_tail(self, redis_case, with_tail):
        async def scenario(client, frozen, stream, receipt):
            sequence = 18_446_744_073_709_551_614 if with_tail else 18_446_744_073_709_551_615
            await client.xadd(
                stream,
                {"payload": '{"_type":"text_delta"}'},
                id=f"18446744073709551615-{sequence}",
            )
            if not with_tail:
                frozen["tts_tail"] = None
            original = await client.xrange(stream)

            with pytest.raises(ResponseError, match="stream ID space exhausted"):
                await publish_terminal(client, **frozen)

            assert await client.xrange(stream) == original
            assert await client.exists(receipt) == 0

        redis_case(scenario)
