"""Blocking request setup runs off the event loop and stays owned (TASK-2541).

Ordinary-turn admission can wait on a per-target PostgreSQL advisory lock and
the turn-log persist writes the admission lease. Both run in a worker, so the
loop keeps serving other requests; a caller that leaves meanwhile must not
orphan what the worker commits afterwards. Ordering is checked with events,
never wall-clock thresholds.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt.service.owned_thread import finish_in_thread, run_in_thread
from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage
from llm_bawt.service.turn_terminal_follower import TerminalFollowerScope
from tests.test_turn_gates import _stream

WAIT_S = 5  # a safety bound only; every assertion is about order


class Gate:
    """A worker call that blocks until the test releases it."""

    def __init__(self, result=None):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.result = result
        self.returned = threading.Event()

    def __call__(self, *_a, **_k):
        self.entered.set()
        assert self.release.wait(WAIT_S), "worker was never released"
        self.returned.set()
        return self.result

    async def wait_entered(self):
        assert await asyncio.to_thread(self.entered.wait, WAIT_S)


async def _spin(n=20):
    for _ in range(n):
        await asyncio.sleep(0)


def test_run_in_thread_waits_for_the_worker_before_cancelling():
    async def main():
        gate = Gate(result="written")
        task = asyncio.create_task(run_in_thread(gate))
        await gate.wait_entered()
        task.cancel()
        await _spin()
        assert not task.done()  # still owns the in-flight worker
        gate.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert gate.returned.is_set()

    asyncio.run(main())


def test_finish_in_thread_returns_the_late_result_with_the_cancel_flag():
    async def main():
        gate = Gate(result="lease")
        task = asyncio.create_task(finish_in_thread(gate))
        await gate.wait_entered()
        task.cancel()
        task.cancel()  # repeated cancellation (anyio re-delivers) is absorbed
        await _spin()
        gate.release.set()
        assert await task == ("lease", True)

    asyncio.run(main())


def test_blocked_stream_admission_leaves_the_loop_free(monkeypatch):
    _, collect = _stream(monkeypatch, claim=None)
    gate = Gate(result=None)
    monkeypatch.setattr("llm_bawt.service.chat_streaming.admit_ordinary_agent_turn", gate)
    order: list[str] = []

    async def main():
        task = asyncio.create_task(collect())
        await gate.wait_entered()
        order.append("loop ran while admission was blocked")
        gate.release.set()
        await task
        order.append("request finished")

    asyncio.run(main())
    assert order == ["loop ran while admission was blocked", "request finished"]


def _record_scope(monkeypatch, order):
    def admitted(self, lease, **_k):
        order.append(("admitted", lease))
        self.follower = Mock()

    def abandon(self):
        order.append("abandoned")

    def close(self):
        order.append("closed")

    monkeypatch.setattr(TerminalFollowerScope, "admitted", admitted)
    monkeypatch.setattr(TerminalFollowerScope, "abandon_unlaunched", abandon)
    monkeypatch.setattr(TerminalFollowerScope, "close_if_unlaunched", close)


def test_stream_disconnect_during_persist_settles_the_late_lease(monkeypatch):
    service, collect = _stream(monkeypatch, claim=None)
    lease = SimpleNamespace(assistant_message_id=None)
    gate = Gate(result=lease)
    service._persist_turn_log = gate
    order: list = []
    _record_scope(monkeypatch, order)

    async def main():
        task = asyncio.create_task(collect())
        await gate.wait_entered()
        task.cancel()  # the client disconnected mid-write
        await _spin()
        assert not task.done() and order == []
        gate.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert order == [("admitted", lease), "abandoned", "closed"]
    service._get_llm_bawt.return_value.execute_llm_query.assert_not_called()


def test_nonstream_cancel_during_persist_settles_the_late_lease(monkeypatch):
    service, _ = _stream(monkeypatch, claim=None)
    monkeypatch.setattr("llm_bawt.service.chat_nonstream.admit_ordinary_agent_turn",
                        lambda *a, **k: None)
    monkeypatch.setattr("llm_bawt.service.chat_nonstream.turn_admission_claim", lambda *_: None)
    lease = SimpleNamespace(assistant_message_id=None)
    gate = Gate(result=lease)
    service._persist_turn_log = gate
    order: list = []
    _record_scope(monkeypatch, order)
    request = ChatCompletionRequest(
        bot_id="caid", user="fixture-user", model="test-model", stream=False,
        messages=[ChatMessage(role="user", content="hi")],
        user_message_id="00000000-0000-4000-8000-000000000002",
    )

    async def main():
        task = asyncio.create_task(service.chat_completion(request))
        await gate.wait_entered()
        task.cancel()
        await _spin()
        assert not task.done() and order == []
        gate.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert order == [("admitted", lease), "abandoned", "closed"]
    service._get_llm_bawt.return_value.execute_llm_query.assert_not_called()
