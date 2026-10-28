"""TASK-1232 P3: every bridge answers ``chat.abort`` with one v1 AbortReply."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_bridge.abort_contract import AbortReply, AbortState, process_boot_id
from agent_bridge.abort_rpc import abort_target, answer_abort, cancel_request_handler
from agent_bridge.session_queue import SessionQueue
from tests.test_claude_code_send_boundaries import SESSION, _SDKStream, _SendHarness
from tests.test_other_bridge_abort_targeting import abort, make_bridge

BACKENDS = ["claude-code", "codex", "local"]


def _bridge(kind):
    return _SendHarness() if kind == "claude-code" else make_bridge(kind)


async def _abort(bridge, target, session=SESSION):
    return await abort(bridge, target, SimpleNamespace(xack=AsyncMock()), session)


def _parse(reply, kind):
    # Strict path only: a v1 reply must never need the legacy adapter.
    parsed = AbortReply.from_wire(reply, backend=kind)
    assert parsed.contract == "v1"
    assert parsed.boot_id == process_boot_id()
    return parsed


@pytest.mark.parametrize("kind", BACKENDS)
def test_cancelling_a_registered_request_is_a_strict_v1_cancelled(kind):
    async def run():
        bridge = _bridge(kind)
        entered = asyncio.Event()

        async def hold():
            async with bridge._session_queue.active(SESSION, request_id="r"):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(hold())
        await entered.wait()
        reply = _parse(await _abort(bridge, "r"), kind)
        assert (reply.state, reply.request_id, reply.cancelled) == (AbortState.CANCELLED, "r", True)
        await asyncio.gather(task, return_exceptions=True)
        assert task.cancelled()
    asyncio.run(asyncio.wait_for(run(), timeout=5))


@pytest.mark.parametrize("kind", BACKENDS)
def test_unknown_request_is_a_strict_v1_tombstone(kind):
    # TASK-1232 P2: an abort for a send not received yet records the Stop.
    bridge = _bridge(kind)
    reply = _parse(asyncio.run(_abort(bridge, "missing")), kind)
    assert (reply.state, reply.request_id, reply.cancelled) == (AbortState.TOMBSTONED, "missing", False)
    assert bridge._session_queue.is_tombstoned(SESSION, "missing")


@pytest.mark.parametrize("kind", BACKENDS)
@pytest.mark.parametrize("session, target, request_id", [
    ("", "r", "r"),
    (SESSION, "", "unknown"),
    (SESSION, "  ", "unknown"),
])
def test_a_malformed_abort_is_a_strict_v1_error(kind, session, target, request_id):
    reply = _parse(asyncio.run(_abort(_bridge(kind), target, session)), kind)
    assert (reply.state, reply.request_id) == (AbortState.ERROR, request_id)
    assert "requires sessionKey and requestId" in reply.detail


def test_claude_running_request_reply_waits_for_and_reports_the_send_teardown(monkeypatch):
    # TASK-1232 P4: the abort never disconnects; the cancelled send owns one
    # teardown and the reply carries its outcome.
    async def run():
        harness = _SendHarness()
        sdk = _SDKStream()
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: sdk)
        send = harness.send()
        await sdk.waiting.wait()
        await harness.abort()
        reply = _parse(harness._publisher.publish_rpc_result.call_args.args[1], "claude-code")
        assert reply.state is AbortState.CANCELLED and reply.request_id == "turn-request"
        # The fake client has no CLI process, so exit can't be judged.
        assert reply.detail == "task_cancelled,teardown:disconnect=ok;signal=none;exited=unknown"
        assert sdk.disconnect_calls == 1
        await asyncio.gather(send, return_exceptions=True)
        assert harness._run_teardowns == {}
    asyncio.run(asyncio.wait_for(run(), timeout=5))


# -- answer_abort -----------------------------------------------------------


def test_answer_abort_reports_a_raising_handler_as_error_for_that_request():
    async def boom(session_key, request_id):
        raise RuntimeError("bridge exploded")

    reply = asyncio.run(answer_abort("codex", {"sessionKey": "s", "requestId": "r"}, boom))
    parsed = AbortReply.from_wire(reply, backend="codex")
    assert (parsed.state, parsed.request_id) == (AbortState.ERROR, "r")
    assert parsed.detail == "RuntimeError: bridge exploded"


def test_answer_abort_does_not_swallow_cancellation():
    async def run():
        started = asyncio.Event()

        async def hang(session_key, request_id):
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(answer_abort("codex", {"sessionKey": "s", "requestId": "r"}, hang))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(asyncio.wait_for(run(), timeout=5))


@pytest.mark.parametrize("params", [{}, {"sessionKey": "s"}, {"requestId": "r"},
                                    {"sessionKey": 1, "requestId": "r"}, "not-a-dict"])
def test_abort_target_requires_both_strings(params):
    with pytest.raises(ValueError):
        abort_target(params)


def test_answer_abort_handles_non_mapping_params():
    reply = asyncio.run(answer_abort("codex", "not-a-dict", cancel_request_handler(SessionQueue())))
    assert (reply["state"], reply["request_id"]) == ("error", "unknown")
