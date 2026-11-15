"""TASK-595 r4 (Codex repro): a now-steer reply taken before steer() returns
still ends the run, with the override published before the write.
"""

import asyncio

import pytest

from tests.test_claude_active_run_background_tasks import _started
from tests.test_claude_active_run_finishing import _consume_until_result
from tests.test_claude_active_run_next_steer import (
    _FakeClient, _INIT, _result, _results, _run,
)


@pytest.mark.anyio
@pytest.mark.parametrize("answer_during_write", [False, True])
async def test_now_replacement_is_terminal_even_if_consumer_wakes_before_rpc(
    answer_during_write, monkeypatch,
):
    client = _FakeClient()
    run = _run(client, background_tasks=True)
    consumer = asyncio.create_task(_consume_until_result(run))
    for frame in (_INIT, _started("a"), _result("standing by")):
        client._query.frames.put_nowait(frame)
    try:
        for _ in range(50):
            if run._background_result is not None:
                break
            await asyncio.sleep(0)
        assert run._background_result is not None
        original = client.query

        async def answer_on_write(prompt):
            await original(prompt)
            if isinstance(prompt, str) and answer_during_write:
                client._query.frames.put_nowait(_result("replacement answer"))

        monkeypatch.setattr(client, "query", answer_on_write)
        await run.steer("stop now")
        assert run._steered_over_background
        assert run._active_tasks == {"a"}
        assert run._expected_interrupt_results == 0
        assert run._background_result is None, (
            "Already accepted replacement was re-held before override publication: "
            f"{run._background_result.result!r}"
        )
        if not answer_during_write:
            client._query.frames.put_nowait(_result("replacement answer"))
        assert _results(await asyncio.wait_for(consumer, 1)) == ["replacement answer"]
    finally:
        if not consumer.done():
            consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        run.mark_completed()
