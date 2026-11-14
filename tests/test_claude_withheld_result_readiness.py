"""TASK-2269: a terminal result cannot strand accepted next-steer commands."""
import asyncio

import pytest
from claude_agent_sdk import ResultMessage, UserMessage

from tests.test_claude_active_run_next_steer import CMD, CMD2, _FakeClient, _lifecycle, _next, _result, _results, _run


class Consumer:
    """Real message iterator with send-handler terminal semantics and frame barriers."""
    def __init__(self):
        self.client = _FakeClient()
        self.run = _run(self.client)
        self.messages = []
        self.barriers = {}
        self.task = None

    async def start(self):
        async def consume():
            async for msg in self.run.messages():
                self.messages.append(msg)
                if isinstance(msg, UserMessage) and isinstance(msg.content, str):
                    event = self.barriers.get(msg.content)
                    if event:
                        event.set()
                if isinstance(msg, ResultMessage) and not self.run.consume_replaced_result(msg):
                    break
        self.task = asyncio.create_task(consume())
        await self.send()  # explicit read barrier enables lifecycle support
        return self

    async def send(self, *frames):
        marker = f"barrier-{len(self.barriers)}"
        event = self.barriers[marker] = asyncio.Event()
        for frame in frames:
            self.client._query.frames.put_nowait(frame)
        self.client._query.frames.put_nowait({"type": "user", "message": {"role": "user", "content": marker}})
        waiter = asyncio.create_task(event.wait())
        try:
            # timing-ok: deadlock bound; assertions synchronize on frames, not elapsed time.
            done, _ = await asyncio.wait({waiter, self.task}, timeout=1, return_when=asyncio.FIRST_COMPLETED)
            assert waiter in done, f"premature terminal result: {_results(self.messages)}"
            assert not self.run._finishing
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    async def finish(self, *frames):
        for frame in frames:
            self.client._query.frames.put_nowait(frame)
        # timing-ok: deadlock bound for the synthetic frame consumer.
        await asyncio.wait_for(self.task, 1)
        return _results(self.messages)

    async def close(self):
        if not self.task.done():
            self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        if self.run._mcp_monitor:
            self.run._mcp_monitor.stop()


@pytest.mark.anyio
@pytest.mark.parametrize("origins", [("user", "user"), ("user", "bot"), ("bot", "user"), ("system", "user")])
@pytest.mark.parametrize("early_result", [True, False])
async def test_early_followup_result_waits_for_later_accepted_command(origins, early_result):
    c = await Consumer().start()
    try:
        await _next(c.run, "A", CMD, origin=origins[0])
        await c.send(_lifecycle(CMD, "started"), _result("A answer"))
        await _next(c.run, "B", CMD2, origin=origins[1])
        await c.send(_lifecycle(CMD, "completed"))
        await c.send(_lifecycle(CMD2, "started"))
        if early_result:
            answers = await c.finish(_result("B answer"), _lifecycle(CMD2, "completed"))
        else:
            await c.send(_lifecycle(CMD2, "completed"))
            answers = await c.finish(_result("B answer"))
        assert answers == ["B answer"]
        assert [frame["uuid"] for frame in c.client.written] == [CMD, CMD2]
    finally:
        await c.close()


@pytest.mark.anyio
@pytest.mark.parametrize("rejection", ["cancelled", "discarded", "refused"])
@pytest.mark.parametrize("completed_first", [False, True])
async def test_rejected_sibling_does_not_release_original_while_followup_started(rejection, completed_first):
    c = await Consumer().start()
    try:
        await _next(c.run, "A", CMD)
        await _next(c.run, "B", CMD2)
        await c.send(_result("original answer"))
        await c.send(_lifecycle(CMD, "started"))
        if completed_first:
            await c.send(_lifecycle(CMD, "completed"))
        await c.send(_lifecycle(CMD2, rejection))
        answers = await c.finish(_lifecycle(CMD, "completed"), _result("A answer"))
        assert answers == ["A answer"]
        assert [frame["uuid"] for frame in c.client.written] == [CMD, CMD2]
    finally:
        await c.close()


@pytest.mark.anyio
async def test_rejected_later_command_releases_valid_candidate_once():
    c = await Consumer().start()
    try:
        await _next(c.run, "A", CMD)
        await c.send(_lifecycle(CMD, "started"), _result("A answer"))
        await _next(c.run, "B", CMD2)
        await c.send(_lifecycle(CMD, "completed"))
        assert await c.finish(_lifecycle(CMD2, "cancelled")) == ["A answer"]
    finally:
        await c.close()


@pytest.mark.anyio
async def test_eof_valid_answer_settles_queued_command_as_uncertain_not_replayed():
    from types import SimpleNamespace

    c = await Consumer().start()
    events = []
    c.run.on_command_lifecycle = events.append
    try:
        await _next(c.run, "A", CMD)
        await c.send(_result("original answer"))
        assert await c.finish(None) == ["original answer"]
        c.run.mark_completed()
        assert c.run.release_unstarted(SimpleNamespace(exited=True)) == [CMD]
        assert (events[-1]["state"], events[-1]["proof"]) == ("uncertain", "teardown")
        assert len(c.client.written) == 1 and c.client.replacements == []
    finally:
        await c.close()


@pytest.mark.anyio
async def test_eof_does_not_promote_answer_while_captured_command_still_started():
    c = await Consumer().start()
    try:
        await _next(c.run, "A", CMD)
        await c.send(_lifecycle(CMD, "started"), _result("early answer"))
        with pytest.raises(RuntimeError, match="sdk_stream_ended_with_started_steers"):
            await c.finish(None)
        assert _results(c.messages) == [] and not c.run._finishing
        assert c.client.replacements == []
    finally:
        await c.close()


@pytest.mark.anyio
async def test_eof_never_restores_invalidated_candidate():
    c = await Consumer().start()
    try:
        await _next(c.run, "A", CMD)
        await c.send(_result("original answer"), _lifecycle(CMD, "started"), _lifecycle(CMD, "completed"))
        assert await c.finish(None) == []
        assert not c.run._finishing
    finally:
        await c.close()


@pytest.mark.anyio
async def test_now_steer_then_new_start_invalidates_without_spending_boundary():
    c = await Consumer().start()
    try:
        await _next(c.run, "A", CMD)
        await c.send(_result("original answer"))
        await c.run.steer("redirect", priority="now", origin="user", framed_message="NOW[redirect]")
        await c.send(_lifecycle(CMD, "started"))
        assert c.run._withheld_result is None
        assert (c.run._expected_interrupt_results, c.run._pending_now_interrupts) == (1, 1)
        await c.send(_result("interrupted"))
        assert (c.run._expected_interrupt_results, c.run._pending_now_interrupts) == (0, 0)
        assert await c.finish(_lifecycle(CMD, "completed"), _result("replacement answer")) == ["interrupted", "replacement answer"]
        assert c.client.replacements == ["NOW[redirect]"]
    finally:
        await c.close()


@pytest.mark.anyio
async def test_held_interrupt_boundary_can_drain_while_sibling_still_queued():
    c = await Consumer().start()
    try:
        await _next(c.run, "A", CMD)
        await c.send(_lifecycle(CMD, "started"), _result("A answer"))
        await _next(c.run, "B", CMD2)
        await c.run.steer("redirect", priority="now", origin="user", framed_message="NOW[redirect]")
        await c.send(_lifecycle(CMD, "completed"))
        assert c.run._commands[CMD2] == "queued"
        assert (c.run._expected_interrupt_results, c.run._pending_now_interrupts) == (0, 0)
        await c.send(_lifecycle(CMD2, "started"), _result("replacement answer"))
        assert await c.finish(_lifecycle(CMD2, "completed")) == ["replacement answer"]
    finally:
        await c.close()
