"""TASK-2395: blocking admission must not stall the shared event loop.

Ordinary-turn admission can wait on a per-target PostgreSQL advisory lock.
This simulates that wait with a sleeping admission and measures how late a
concurrent loop ticker fires while the stream request is being admitted.
"""

import asyncio
import time

from tests.test_turn_gates import _stream

BLOCK_S = 0.6


async def _max_loop_lag(collect):
    lags: list[float] = []
    stop = asyncio.Event()

    async def ticker():
        while not stop.is_set():
            start = time.monotonic()
            await asyncio.sleep(0.01)
            lags.append(time.monotonic() - start - 0.01)

    tick = asyncio.create_task(ticker())
    await asyncio.sleep(0.03)
    try:
        await collect()
    finally:
        stop.set()
        await tick
    return max(lags)


def test_blocked_stream_admission_does_not_stall_the_loop(monkeypatch):
    _, collect = _stream(monkeypatch, claim=None)

    def slow_admission(*_a, **_k):
        time.sleep(BLOCK_S)  # stands in for a contended pg_advisory_xact_lock
        return None

    monkeypatch.setattr(
        "llm_bawt.service.chat_streaming.admit_ordinary_agent_turn", slow_admission,
    )
    lag = asyncio.run(_max_loop_lag(collect))
    assert lag < BLOCK_S / 3, f"event loop stalled {lag:.3f}s during admission"
