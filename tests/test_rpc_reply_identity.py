"""TASK-1232 P1: an RPC is answered only by the backend it addresses.

Before this, a backend-less call to a method only one bridge serves was also
answered "unknown method" by every other bridge, and ``send_rpc`` returned
whichever reply landed first.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_bridge.abort_contract import process_boot_id
from agent_bridge.publisher import RedisPublisher
from agent_bridge.subscriber import RedisSubscriber
from tests.test_claude_code_send_boundaries import _SendHarness
from tests.test_other_bridge_abort_targeting import make_bridge


def _rpc(method, backend=None):
    fields = {"request_id": "rpc-1", "method": method, "params": json.dumps({"agentId": "main"})}
    if backend is not None:
        fields["backend"] = backend
    return fields


def _bridges():
    claude = _SendHarness()
    codex = make_bridge("codex")
    codex._bot_uses_codex = AsyncMock(return_value=True)
    return {"claude-code": claude, "codex": codex}


@pytest.mark.parametrize("name", ["claude-code", "codex"])
@pytest.mark.parametrize("backend", [None, "local"])
def test_unknown_method_not_addressed_here_is_left_to_its_owner(name, backend):
    bridge = _bridges()[name]
    redis = MagicMock(xack=AsyncMock())
    asyncio.run(bridge._handle_rpc(_rpc("other.method", backend), "msg-1", redis))
    bridge._publisher.publish_rpc_result.assert_not_called()
    redis.xack.assert_awaited_once()


@pytest.mark.parametrize("name", ["claude-code", "codex"])
def test_unknown_method_addressed_here_fails_fast(name):
    bridge = _bridges()[name]
    redis = MagicMock(xack=AsyncMock())
    asyncio.run(bridge._handle_rpc(_rpc("other.method", name), "msg-1", redis))
    bridge._publisher.publish_rpc_result.assert_called_once_with(
        "rpc-1", {"ok": False, "error": "unknown method: other.method"})


# -- reply identity -------------------------------------------------------


def _publisher(provider):
    publisher = RedisPublisher.__new__(RedisPublisher)
    publisher._redis = MagicMock()
    publisher._connected = True
    publisher._default_provider = provider
    return publisher


def test_rpc_reply_is_stamped_with_backend_and_boot_id_outside_the_payload():
    publisher = _publisher("codex")
    publisher.publish_rpc_result("rpc-1", {"ok": True})
    stream, fields = publisher._redis.xadd.call_args.args
    assert stream == "agent:rpc:rpc-1"
    assert fields == {"payload": json.dumps({"ok": True}), "backend": "codex",
                      "boot_id": process_boot_id()}


@pytest.mark.parametrize("request_id,ttl", [
    ("rpc-1", 60),
    ("steer_delivery_d1_g2", 604800),
    ("steer_quiesce_turn-1_req-1", 604800),
    ("steer_quiesce_turn-1_req-1_g3", 604800),
])
def test_retried_steer_results_live_as_long_as_their_dedupe_key(request_id, ttl):
    publisher = _publisher("claude-code")
    publisher.publish_rpc_result(request_id, {"ok": False, "error": "no_active_run"})
    publisher._redis.expire.assert_called_once_with(f"agent:rpc:{request_id}", ttl)


def test_publisher_without_identity_publishes_the_historical_shape():
    publisher = _publisher(None)
    publisher.publish_rpc_result("rpc-1", {"ok": True})
    assert publisher._redis.xadd.call_args.args[1] == {"payload": json.dumps({"ok": True})}


# -- send_rpc ---------------------------------------------------------------


class _RpcStream:
    """Minimal XREAD over one reply stream, honouring the last-id cursor."""

    def __init__(self, entries):
        self.entries = entries
        self.cursors = []

    async def xread(self, streams, count, block):
        (stream, cursor), = streams.items()
        self.cursors.append(cursor)
        ids = [entry_id for entry_id, _ in self.entries]
        start = 0 if cursor == "0-0" else ids.index(cursor) + 1
        batch = self.entries[start:start + count]
        if not batch:
            await asyncio.sleep(block / 1000)
            return []
        return [(stream, batch)]


def _subscriber(entries):
    subscriber = RedisSubscriber.__new__(RedisSubscriber)
    subscriber._pub_redis = MagicMock(xadd=AsyncMock())
    subscriber._redis = _RpcStream(entries)
    return subscriber


def _reply(payload, backend=None):
    fields = {"payload": json.dumps(payload)}
    if backend:
        fields.update(backend=backend, boot_id="b" * 32)
    return fields


def test_send_rpc_skips_a_foreign_reply_and_returns_the_owners():
    subscriber = _subscriber([
        ("1-0", _reply({"ok": False, "error": "unknown method: other.method"}, "claude-code")),
        ("2-0", _reply({"ok": True, "payload": {"content": "owner"}}, "local")),
    ])
    result = asyncio.run(subscriber.send_rpc(
        "other.method", {}, "rpc-1", timeout_s=2, backend="local"))
    assert result == {"ok": True, "payload": {"content": "owner"}}
    sent = subscriber._pub_redis.xadd.call_args.args[1]
    assert sent["backend"] == "local"


def test_send_rpc_advances_past_foreign_replies_across_reads():
    subscriber = _subscriber([
        ("1-0", _reply({"ok": False}, "codex")),
        ("2-0", _reply({"ok": True}, "local")),
    ])
    # One entry per read so the cursor must move for the owner's to be seen.
    real = subscriber._redis.xread
    subscriber._redis.xread = lambda streams, count, block: real(streams, 1, block)
    assert asyncio.run(subscriber.send_rpc(
        "other.method", {}, "rpc-1", timeout_s=2, backend="local")) == {"ok": True}
    assert subscriber._redis.cursors == ["0-0", "1-0"]


def test_send_rpc_times_out_when_only_foreign_replies_arrive():
    subscriber = _subscriber([("1-0", _reply({"ok": False}, "claude-code"))])
    with pytest.raises(TimeoutError):
        asyncio.run(subscriber.send_rpc(
            "other.method", {}, "rpc-1", timeout_s=0.3, backend="local"))


@pytest.mark.parametrize("backend", [None, "local"])
def test_send_rpc_accepts_an_unstamped_reply_from_a_pre_identity_bridge(backend):
    subscriber = _subscriber([("1-0", _reply({"ok": True}))])
    assert asyncio.run(subscriber.send_rpc(
        "other.method", {}, "rpc-1", timeout_s=2, backend=backend)) == {"ok": True}

