"""Unit tests for shared agent-bridge transport components.

Tests SessionQueue and ClaudeActiveRun — the per-session serialization
and active-run steering that the bridges rely on.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_bridge.session_queue import SessionQueue
from claude_code_bridge.active_run import ClaudeActiveRun
from tests import virtual_clock

# Virtual-clock hang guard for the steer tests: only a broken run reaches it.
HANG_GUARD_S = 30.0


# ---------------------------------------------------------------------------
# SessionQueue tests
# ---------------------------------------------------------------------------


class TestSessionQueue:
    def test_active_client_set_get_pop(self):
        queue = SessionQueue()
        client = object()

        queue.set_active_client("main", client)

        assert queue.get_active_client("main") is client
        assert queue.pop_active_client("main") is client
        assert queue.get_active_client("main") is None

    def test_active_client_pop_is_idempotent(self):
        queue = SessionQueue()
        client = object()

        queue.set_active_client("main", client)

        assert queue.pop_active_client("main") is client
        assert queue.pop_active_client("main") is None


class TestClaudeActiveRun:
    @pytest.mark.anyio
    async def test_steer_interrupts_then_queries_same_client(self):
        client = AsyncMock()
        run = ClaudeActiveRun(client=client, request_id="req-1")

        # TASK-1300: the replacement is app-rendered registry text, sent as is.
        framed = "[app-rendered chat.steer.interrupt_replacement]\nchange direction"
        await run.steer("change direction", origin="user", framed_message=framed)

        client.interrupt.assert_awaited_once_with()
        assert client.query.await_args.args[0] == framed
        interrupted = MagicMock(subtype="error_during_execution")
        assert run.consume_replaced_result(interrupted) is True
        assert run.consume_replaced_result(interrupted) is False

    def test_steer_now_does_not_wait_for_blocked_sdk_message_reader(self):
        """Bridge control can advance while its model-message reader is silent.

        This does not establish whether the live CLI acknowledges interrupt
        while its *own* upstream HTTP read is blocked; that requires a live probe.
        """
        async def main():
            entered = asyncio.Event()
            release = asyncio.Event()

            async def silent_reader():
                entered.set()
                await release.wait()
                yield MagicMock()

            client = AsyncMock()
            client.receive_messages = silent_reader
            # Production steer timeout, on the virtual clock.
            run = ClaudeActiveRun(client=client, request_id="req-1")
            reader = asyncio.create_task(anext(run.messages()))
            loop = asyncio.get_running_loop()
            try:
                await asyncio.wait_for(entered.wait(), timeout=HANG_GUARD_S)
                began = loop.time()
                await asyncio.wait_for(run.steer("resume unfinished work"), timeout=HANG_GUARD_S)
                # Steering finished without spending its own timeout, so it
                # never waited on the blocked reader.
                # timing-ok: virtual clock (run_timing): loop.time() is exact and never waits
                assert loop.time() - began < run._steer_timeout_s
                client.interrupt.assert_awaited_once_with()
                client.query.assert_awaited_once()
                assert not reader.done()
            finally:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)

        virtual_clock.run_timing(main())

    @pytest.mark.anyio
    async def test_steer_failure_does_not_hide_next_error_result(self):
        client = AsyncMock()
        client.query.side_effect = RuntimeError("closed")
        run = ClaudeActiveRun(client=client, request_id="req-1")

        with pytest.raises(RuntimeError, match="closed"):
            await run.steer("change direction")

        client.disconnect.assert_awaited_once_with()
        interrupted = MagicMock(subtype="error_during_execution")
        assert run.consume_replaced_result(interrupted) is False

    @pytest.mark.anyio
    async def test_steer_returns_no_active_run_when_turn_finishes_during_interrupt(self):
        interrupt_started = asyncio.Event()
        release_interrupt = asyncio.Event()

        async def hanging_interrupt():
            interrupt_started.set()
            await release_interrupt.wait()

        client = AsyncMock()
        client.interrupt.side_effect = hanging_interrupt
        run = ClaudeActiveRun(client=client, request_id="req-1")

        steer_task = asyncio.create_task(run.steer("change direction"))
        await interrupt_started.wait()
        run.mark_completed()

        with pytest.raises(RuntimeError, match="no_active_run"):
            await steer_task

        client.query.assert_not_awaited()
        assert run.consume_replaced_result(MagicMock(subtype="success")) is False

    @pytest.mark.anyio
    async def test_steer_rejects_already_completed_run_without_interrupting(self):
        client = AsyncMock()
        run = ClaudeActiveRun(client=client, request_id="req-1")
        run.mark_completed()

        with pytest.raises(RuntimeError, match="no_active_run"):
            await run.steer("change direction")

        client.interrupt.assert_not_awaited()
        client.query.assert_not_awaited()

    def test_steer_bounds_unacknowledged_interrupt_before_rpc_deadline(self):
        async def main():
            interrupt_started = asyncio.Event()
            release_interrupt = asyncio.Event()

            async def hanging_interrupt():
                interrupt_started.set()
                await release_interrupt.wait()

            client = AsyncMock()
            client.interrupt.side_effect = hanging_interrupt
            # Production steer timeout: the virtual clock jumps the full wait.
            run = ClaudeActiveRun(client=client, request_id="req-1")
            loop = asyncio.get_running_loop()
            began = loop.time()

            with pytest.raises(RuntimeError, match="steer_interrupt_timeout"):
                await run.steer("change direction")

            # One-sided: the full steer timeout was honored (per-step cost only
            # adds to it).
            # timing-ok: virtual clock (run_timing): loop.time() is exact and never waits
            assert loop.time() - began >= run._steer_timeout_s
            assert interrupt_started.is_set()
            client.query.assert_not_awaited()
            assert run.consume_replaced_result(MagicMock(subtype="success")) is False

        virtual_clock.run_timing(main())

    def test_steer_bounds_replacement_query_before_rpc_deadline(self):
        async def main():
            query_started = asyncio.Event()
            release_query = asyncio.Event()

            async def hanging_query(_message):
                query_started.set()
                await release_query.wait()

            client = AsyncMock()
            client.query.side_effect = hanging_query
            # Production steer timeout; interrupt and query share one deadline.
            run = ClaudeActiveRun(client=client, request_id="req-1")
            loop = asyncio.get_running_loop()
            began = loop.time()

            with pytest.raises(RuntimeError, match="steer_query_timeout"):
                await run.steer("change direction")

            # One-sided: the full steer timeout was honored (per-step cost only
            # adds to it).
            # timing-ok: virtual clock (run_timing): loop.time() is exact and never waits
            assert loop.time() - began >= run._steer_timeout_s
            assert query_started.is_set()
            client.disconnect.assert_awaited_once_with()
            assert run.consume_replaced_result(MagicMock(subtype="success")) is False

        virtual_clock.run_timing(main())

    def test_success_result_in_interrupt_race_is_still_replaced(self):
        run = ClaudeActiveRun(client=AsyncMock(), request_id="req-1")
        run._expected_interrupt_results = 1

        assert run.consume_replaced_result(MagicMock(subtype="success")) is True
        assert run.consume_replaced_result(MagicMock(subtype="success")) is False
