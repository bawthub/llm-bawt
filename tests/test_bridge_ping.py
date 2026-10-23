"""TASK-1297: the strict ``bridge.ping`` readiness RPC (TASK-1293 C1 / Loopy M2).

End to end against an in-memory Redis: :func:`probe_bridge` publishes on the
commands stream, a real claude-code bridge ``_handle_rpc`` answers through a
real :class:`RedisPublisher` (which stamps backend + boot id), and the probe
accepts only that exact, identified answer within one budget.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_bridge.abort_contract import process_boot_id
from agent_bridge.ping import (
    PING_METHOD,
    PingProbe,
    PingReplyError,
    answer_ping,
    parse_ping_reply,
    probe_bridge,
)
from agent_bridge.publisher import COMMANDS_STREAM, RedisPublisher
from tests.test_claude_code_send_boundaries import _SendHarness
from tests.test_other_bridge_abort_targeting import make_bridge

BUDGET = 0.8


class FakeRedis:
    """Streams in memory: async client for the probe, sync view for a publisher."""

    def __init__(self, on_command=None, *, xadd_delay: float = 0.0, delete_delay: float = 0.0,
                 xadd_error: Exception | None = None):
        self.streams: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        self.on_command = on_command
        self.xadd_delay = xadd_delay
        self.delete_delay = delete_delay
        self.xadd_error = xadd_error
        self.deleted: list[str] = []
        self.xdeleted: list[tuple[str, str]] = []
        self.closed = False
        self.consumer_groups: dict[str, dict[str, list[str]]] = {}
        self._seq = 0
        self._tasks: list[asyncio.Task] = []

    def append(self, stream: str, fields: dict) -> str:
        self._seq += 1
        msg_id = f"{self._seq}-0"
        self.streams[stream].append((msg_id, dict(fields)))
        return msg_id

    # -- async client surface used by probe_bridge ------------------------

    async def xadd(self, stream, fields, maxlen=None, approximate=None):
        if self.xadd_delay:
            await asyncio.sleep(self.xadd_delay)
        if self.xadd_error is not None:
            raise self.xadd_error
        msg_id = self.append(stream, fields)
        if stream == COMMANDS_STREAM and self.on_command is not None:
            self._tasks.append(asyncio.create_task(self.on_command(self, msg_id, dict(fields))))
        return msg_id

    async def xread(self, streams, count, block):
        (stream, cursor), = streams.items()
        loop = asyncio.get_running_loop()
        until = loop.time() + block / 1000
        while True:
            entries = self.streams.get(stream, [])
            ids = [msg_id for msg_id, _ in entries]
            start = 0 if cursor == "0-0" else ids.index(cursor) + 1
            batch = entries[start:start + count]
            if batch:
                return [(stream, batch)]
            if loop.time() >= until:
                return []
            await asyncio.sleep(0.005)

    async def delete(self, *names):
        if self.delete_delay:
            await asyncio.sleep(self.delete_delay)
        for name in names:
            self.deleted.append(name)
            self.streams.pop(name, None)

    async def xdel(self, stream, *ids):
        for msg_id in ids:
            self.xdeleted.append((stream, msg_id))
        self.streams[stream] = [e for e in self.streams[stream] if e[0] not in ids]

    async def aclose(self):
        self.closed = True

    # -- sync view for RedisPublisher -------------------------------------

    def sync_view(self):
        fake = self

        class _Sync:
            def xadd(self, stream, fields, maxlen=None):
                return fake.append(stream, fields)

            def expire(self, stream, ttl):
                return True

        return _Sync()


def _publisher(fake: FakeRedis, provider: str | None) -> RedisPublisher:
    publisher = RedisPublisher.__new__(RedisPublisher)
    publisher._redis = fake.sync_view()
    publisher._connected = True
    publisher._default_provider = provider
    return publisher


def _claude_bridge(*, stamp: str | None = "claude-code", delay: float = 0.0):
    """A live claude-code command consumer: the real ``_handle_rpc``."""
    bridge = _SendHarness()

    async def on_command(fake, msg_id, fields):
        bridge._publisher = _publisher(fake, stamp)
        if delay:
            await asyncio.sleep(delay)
        await bridge._handle_rpc(fields, msg_id, MagicMock(xack=AsyncMock()))

    return on_command


def _scripted(*replies, delay: float = 0.0):
    """A responder that writes raw reply entries (built from the command) to the reply stream."""

    async def on_command(fake, msg_id, fields):
        if delay:
            await asyncio.sleep(delay)
        params = json.loads(fields["params"])
        for build in replies:
            fake.append(f"agent:rpc:{fields['request_id']}", build(fields["request_id"], params))

    return on_command


def _stamped(payload: dict, *, backend: str = "claude-code", boot: str | None = None) -> dict:
    return {"payload": json.dumps(payload), "backend": backend, "boot_id": boot or process_boot_id()}


def _good(request_id: str, params: dict, /, **overrides) -> dict:
    payload = {
        "ok": True, "method": PING_METHOD, "v": 1, "backend": "claude-code",
        "instance": process_boot_id(), "request_id": request_id, "nonce": params["nonce"],
    }
    payload.update(overrides)
    return payload


def _probe(fake: FakeRedis, backend: str = "claude-code", budget: float = BUDGET):
    async def run():
        loop = asyncio.get_running_loop()
        started = loop.time()
        result = await probe_bridge(lambda: fake, backend, budget_s=budget)
        elapsed = loop.time() - started
        for task in fake._tasks:  # a broken responder must fail the test, not look like silence
            if task.done() and not task.cancelled() and task.exception() is not None:
                raise task.exception()
            task.cancel()
        return result, elapsed

    return asyncio.run(run())


# ---------------------------------------------------------------------------
# The four TASK-1297 readiness cases
# ---------------------------------------------------------------------------


def test_ready_when_the_live_claude_code_consumer_answers():
    fake = FakeRedis(_claude_bridge())
    result, elapsed = _probe(fake)

    assert result.ready is True
    assert result.backend == "claude-code"
    assert result.instance == process_boot_id()
    assert result.detail == "claude-code bridge answered ping v1"
    assert elapsed < BUDGET
    (command_id, command), = fake.streams[COMMANDS_STREAM]
    assert command["method"] == PING_METHOD and command["backend"] == "claude-code"
    assert command["request_id"].startswith("ping_")
    assert fake.deleted == [f"agent:rpc:{command['request_id']}"]
    assert fake.xdeleted == []  # answered: the bridge acked it, nothing to retract
    assert fake.closed


def test_not_ready_when_no_bridge_is_reading():
    fake = FakeRedis(on_command=None)
    result, elapsed = _probe(fake)

    assert result.ready is False
    assert result.instance is None
    assert result.detail == f"no reply from the claude-code bridge within {BUDGET:.1f}s"
    assert elapsed <= BUDGET + 0.1
    assert len(fake.xdeleted) == 1 and fake.xdeleted[0][0] == COMMANDS_STREAM
    assert fake.streams[COMMANDS_STREAM] == []  # unanswered probe retracted
    assert fake.closed


def test_not_ready_with_a_stale_consumer_record_and_no_live_subscriber():
    # The group still lists the dead bridge's consumer; only a live reader counts.
    fake = FakeRedis(on_command=None)
    fake.consumer_groups[COMMANDS_STREAM] = {"claude-code-bridge": ["claude-code-dead-host-1"]}
    result, elapsed = _probe(fake)

    assert result.ready is False
    assert result.detail.startswith("no reply from the claude-code bridge within")
    assert elapsed <= BUDGET + 0.1
    assert fake.streams[COMMANDS_STREAM] == []


def test_not_ready_when_only_another_backend_is_live():
    # A live codex bridge reads the same stream but leaves claude-code's ping alone.
    codex = make_bridge("codex")

    async def on_command(fake, msg_id, fields):
        codex._publisher = _publisher(fake, "codex")
        await codex._handle_rpc(fields, msg_id, MagicMock(xack=AsyncMock()))

    fake = FakeRedis(on_command)
    result, _ = _probe(fake)

    assert result.ready is False
    assert result.detail == f"no reply from the claude-code bridge within {BUDGET:.1f}s"


def test_wrong_backend_reply_is_ignored_not_accepted():
    fake = FakeRedis(_scripted(lambda rid, p: _stamped(_good(rid, p, backend="codex"), backend="codex")))
    result, _ = _probe(fake)

    assert result.ready is False
    assert result.detail.endswith("(ignored: reply from wrong backend 'codex')")


def test_wrong_backend_reply_then_the_owners_answer_is_ready():
    fake = FakeRedis(_scripted(
        lambda rid, p: _stamped(_good(rid, p, backend="codex"), backend="codex"),
        lambda rid, p: _stamped(_good(rid, p)),
    ))
    result, _ = _probe(fake)
    assert result.ready is True


def test_bridge_does_not_answer_a_ping_addressed_to_another_backend():
    fake = FakeRedis(_claude_bridge())
    result, _ = _probe(fake, backend="codex")

    assert result.ready is False
    assert [k for k in fake.streams if k.startswith("agent:rpc:")] == []


# ---------------------------------------------------------------------------
# Strictness: identity, freshness, version
# ---------------------------------------------------------------------------


def test_replayed_reply_with_another_nonce_is_ignored():
    fake = FakeRedis(_scripted(lambda rid, p: _stamped(_good(rid, p, nonce="0" * 32))))
    result, _ = _probe(fake)

    assert result.ready is False
    assert result.detail.endswith("(ignored: stale or replayed ping reply)")


def test_reply_for_another_request_is_ignored_then_ours_accepted():
    fake = FakeRedis(_scripted(
        lambda rid, p: _stamped(_good(rid, p, request_id="ping_old")),
        lambda rid, p: _stamped(_good(rid, p)),
    ))
    result, _ = _probe(fake)
    assert result.ready is True


def test_unstamped_legacy_reply_is_never_ready():
    # send_rpc accepts unstamped replies from pre-identity bridges; ping must not.
    fake = FakeRedis(_claude_bridge(stamp=None))
    result, _ = _probe(fake)

    assert result.ready is False
    assert "missing backend/boot identity" in result.detail


def test_bridge_predating_ping_fails_fast_with_its_refusal():
    budget = 3.0
    fake = FakeRedis(_scripted(
        lambda rid, p: _stamped({"ok": False, "error": "unknown method: bridge.ping"}),
    ))
    result, elapsed = _probe(fake, budget=budget)

    assert result.ready is False
    assert result.detail == "claude-code bridge rejected ping: unknown method: bridge.ping"
    assert elapsed < budget / 2


@pytest.mark.parametrize("build, reason", [
    (lambda rid, p: {"payload": "not json", "backend": "claude-code", "boot_id": process_boot_id()},
     "payload is not JSON"),
    (lambda rid, p: _stamped(_good(rid, p, instance="someone-else")), "instance disagrees"),
    (lambda rid, p: _stamped(_good(rid, p, v=2)), "unsupported version 2"),
    (lambda rid, p: _stamped(_good(rid, p, v=True)), "unsupported version True"),
    (lambda rid, p: _stamped(_good(rid, p, method="run.status")), "not a bridge.ping answer"),
    (lambda rid, p: _stamped(_good(rid, p, backend="codex")), "payload backend disagrees"),
])
def test_malformed_reply_from_the_addressed_backend_is_terminal(build, reason):
    fake = FakeRedis(_scripted(build))
    result, _ = _probe(fake)

    assert result.ready is False
    assert reason in result.detail


def test_every_probe_gets_a_fresh_request_id_and_nonce():
    a = PingProbe.new("claude-code", budget_s=5.0)
    b = PingProbe.new("claude-code", budget_s=5.0)
    assert a.request_id != b.request_id and a.nonce != b.nonce
    params = json.loads(a.command_fields()["params"])
    assert params == {"v": 1, "nonce": a.nonce, "expires_at": a.expires_at}


def test_parse_rejects_a_non_ok_reply_from_another_backend_as_non_terminal():
    probe = PingProbe.new("claude-code", budget_s=5.0)
    with pytest.raises(PingReplyError) as exc:
        parse_ping_reply(_stamped({"ok": False, "error": "x"}, backend="codex"), probe)
    assert exc.value.terminal is False


# ---------------------------------------------------------------------------
# Budget: connect + publish + read + cleanup all inside it
# ---------------------------------------------------------------------------


def test_slow_publish_counts_against_the_budget():
    # redis.asyncio connects lazily inside the first command, so a slow connect
    # surfaces here as a slow XADD.
    fake = FakeRedis(_claude_bridge(), xadd_delay=BUDGET * 3)
    result, elapsed = _probe(fake)

    assert result.ready is False
    assert result.detail == "Redis unavailable: timed out during publish"
    assert elapsed <= BUDGET + 0.1


def test_publish_and_reply_together_cannot_exceed_the_budget():
    fake = FakeRedis(_claude_bridge(delay=BUDGET * 0.6), xadd_delay=BUDGET * 0.6)
    result, elapsed = _probe(fake)

    assert result.ready is False
    assert elapsed <= BUDGET + 0.1


def test_slow_cleanup_does_not_lose_a_ready_answer():
    fake = FakeRedis(_claude_bridge(), delete_delay=BUDGET * 5)
    result, elapsed = _probe(fake)

    assert result.ready is True
    assert elapsed <= BUDGET + 0.1


def test_redis_connection_error_is_not_ready():
    fake = FakeRedis(xadd_error=ConnectionError("Connection refused"))
    result, _ = _probe(fake)

    assert result.ready is False
    assert result.detail == "Redis unavailable: ConnectionError: Connection refused"
    assert fake.closed


def test_factory_failure_is_not_ready():
    def factory():
        raise OSError("bad url")

    result = asyncio.run(probe_bridge(factory, "claude-code", budget_s=BUDGET))
    assert result.ready is False
    assert result.detail == "Redis unavailable: OSError: bad url"


# ---------------------------------------------------------------------------
# Bridge side
# ---------------------------------------------------------------------------


def _ping_fields(params: dict, backend: str = "claude-code") -> dict:
    return {"action": "rpc.call", "method": PING_METHOD, "backend": backend,
            "request_id": "ping_abc", "params": json.dumps(params)}


def test_bridge_answers_ping_through_the_stamping_publisher():
    bridge = _SendHarness()
    params = {"v": 1, "nonce": "n1", "expires_at": time.time() + 5}
    asyncio.run(bridge._handle_rpc(_ping_fields(params), "1-0", MagicMock(xack=AsyncMock())))

    bridge._publisher.publish_rpc_result.assert_called_once_with("ping_abc", {
        "ok": True, "method": PING_METHOD, "v": 1, "backend": "claude-code",
        "instance": process_boot_id(), "request_id": "ping_abc", "nonce": "n1",
    })


def test_bridge_stays_silent_on_an_expired_ping():
    bridge = _SendHarness()
    params = {"v": 1, "nonce": "n1", "expires_at": time.time() - 1}
    redis = MagicMock(xack=AsyncMock())
    asyncio.run(bridge._handle_rpc(_ping_fields(params), "1-0", redis))

    bridge._publisher.publish_rpc_result.assert_not_called()
    redis.xack.assert_awaited()


def test_bridge_answers_ping_while_a_session_turn_holds_its_lock():
    # Ping is dispatched by the command listener outside every session lock.
    bridge = _SendHarness()
    params = {"v": 1, "nonce": "n1", "expires_at": time.time() + 5}

    async def run():
        lock = bridge._session_queue.lock("agent:main:busy-session")
        async with lock:  # a turn is mid-flight on some session
            with patch.object(bridge._session_queue, "lock",
                              side_effect=AssertionError("ping took a session lock")):
                await asyncio.wait_for(
                    bridge._handle_rpc(_ping_fields(params), "1-0", MagicMock(xack=AsyncMock())), 1.0,
                )

    asyncio.run(run())
    bridge._publisher.publish_rpc_result.assert_called_once()


@pytest.mark.parametrize("params, error", [
    ("nope", "params must be an object"),
    ({"v": 1, "nonce": "n"}, "requires numeric expires_at"),
    ({"v": 1, "nonce": "n", "expires_at": True}, "requires numeric expires_at"),
    ({"v": 9, "nonce": "n", "expires_at": 10.0}, "unsupported bridge.ping version: 9"),
    ({"v": 1, "nonce": "", "expires_at": 10.0}, "requires nonce and request_id"),
])
def test_answer_ping_refuses_malformed_requests(params, error):
    reply = answer_ping("claude-code", "ping_abc", params, now=1.0)
    assert reply["ok"] is False and error in reply["error"]


# ---------------------------------------------------------------------------
# App-side wrapper
# ---------------------------------------------------------------------------


def test_check_agent_runtime_without_redis_is_not_ready():
    from llm_bawt.service.agent_runtime import check_agent_runtime

    config = MagicMock(REDIS_URL="")
    result = asyncio.run(check_agent_runtime(config))
    assert result.to_dict() == {"ready": False, "backend": "claude-code", "instance": None,
                                "detail": "Redis is not configured"}


def test_check_agent_runtime_probes_claude_code_with_the_5s_budget():
    from agent_bridge.ping import BridgeReadiness
    from llm_bawt.service import agent_runtime

    ready = BridgeReadiness(ready=True, backend="claude-code", instance="b", detail="ok")
    with patch.object(agent_runtime, "probe_bridge", AsyncMock(return_value=ready)) as probe:
        result = asyncio.run(agent_runtime.check_agent_runtime(MagicMock(REDIS_URL="redis://redis:6379/0")))

    assert result is ready
    _factory, backend = probe.await_args.args
    assert backend == "claude-code"
    assert probe.await_args.kwargs == {"budget_s": 5.0}
