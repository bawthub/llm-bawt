"""TASK-1234 Fix 1: classify a durable delivery only after its turn row is final.

Incident a67418db (Nick -> Snark): the real run ended with an HTTP 429. The
stream worker hands an upstream failure to the chat stream (ending the
dispatcher's drain) *before* it writes ``status=error`` and finalizes. Each
attempt read the row while it still said ``streaming``, so the TASK-1095
"ended unsuccessfully" guard never fired and all 5 attempts were burned with
``did not finalize (status=streaming)`` instead of the 429.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from threading import Event
from types import SimpleNamespace

import pytest

from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
from llm_bawt.service.turn_execution import TurnExecution, turn_executions

REAL_CAUSE = "claude-code error: HTTP 429 rate_limit_error"


def _record(turn_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"delivery-{turn_id}", turn_id=turn_id, claim_token="claim",
        target_bot_id="snark", attempt_count=1, max_attempts=5,
        overflow_recovery_count=0, metadata={}, session_policy="continue",
        transport_accepted_at=None,
    )


def _payload(turn_id: str) -> dict:
    return {
        "messages": [{"role": "user", "content": "work"}],
        "bot_id": "snark",
        "user_message_id": "00000000-0000-4000-8000-0000000012a1",
        "assistant_message_id": "00000000-0000-4000-8000-0000000012a2",
        "inter_bot_delivery_id": f"delivery-{turn_id}",
        "inter_bot_turn_id": turn_id,
        "inter_bot_bridge_request_id": f"req_delivery_{turn_id}",
    }


class _Harness:
    """A turn row the stream worker finalizes *after* the drain has ended."""

    def __init__(self, turn_id: str, *, finalize_after: float | None, raise_in_stream: bool = False):
        self.turn_id = turn_id
        self.turn = SimpleNamespace(
            ended_at=None, status="streaming", error_text=None,
            response_text="", model="m",
        )
        self.failed: list[tuple] = []
        self.requeued: list[tuple] = []
        self.finalize_after = finalize_after
        self.raise_in_stream = raise_in_stream
        self.execution = TurnExecution(turn_id, Event(), is_agent=True, store=None)

    def _finalize(self) -> None:
        # Mirrors TurnStreamWorker: write the terminal row, then retire the
        # execution handle in TurnStreamFinalizer.finalize's finally block.
        self.turn.status = "error"
        self.turn.error_text = REAL_CAUSE
        self.turn.ended_at = datetime.now(timezone.utc)
        turn_executions.remove(self.turn_id)

    def dispatcher(self) -> InterBotDeliveryDispatcher:
        record = _record(self.turn_id)
        harness = self

        async def stream(_request):
            turn_executions.register(harness.execution)
            if harness.finalize_after is not None:
                asyncio.get_running_loop().call_later(harness.finalize_after, harness._finalize)
            yield "data: [DONE]\n\n"
            if harness.raise_in_stream:
                raise RuntimeError("stream_interrupted: upstream closed")

        store = SimpleNamespace(
            payload=lambda _id: _payload(self.turn_id),
            mark_transport_accepted=lambda *_a: True,
            get=lambda _id: record,
            turn_state=lambda _id: (self.turn.status, self.turn.ended_at),
            fail_claim=lambda *a: self.failed.append(a) or SimpleNamespace(status="FAILED"),
            requeue=lambda *a, **kw: self.requeued.append(a) or SimpleNamespace(status="QUEUED"),
            renew_lease=lambda *_a: False,
        )
        dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
        dispatcher.store = store
        dispatcher.service = SimpleNamespace(
            config=SimpleNamespace(),
            chat_completion_stream=stream,
            _turn_log_store=SimpleNamespace(get_turn=lambda _id: self.turn),
        )
        dispatcher._heartbeat = lambda *_a: asyncio.sleep(3600)
        dispatcher._emit = lambda _v: asyncio.sleep(0)
        return dispatcher

    def run(self, dispatcher: InterBotDeliveryDispatcher) -> None:
        try:
            asyncio.run(dispatcher._dispatch(_record(self.turn_id)))
        finally:
            turn_executions.remove(self.turn_id)


@pytest.mark.parametrize("raise_in_stream", [False, True], ids=["drained", "stream-raised"])
def test_terminal_error_replay_fails_on_first_attempt_with_real_cause(raise_in_stream):
    """The a67418db pattern: exactly one attempt, and last_error is the 429."""
    harness = _Harness(f"turn-1234-a67418db-{raise_in_stream}", finalize_after=0.05,
                       raise_in_stream=raise_in_stream)
    harness.run(harness.dispatcher())

    assert not harness.requeued, harness.requeued
    assert len(harness.failed) == 1
    assert "did not finalize" not in harness.failed[0][2]
    if not raise_in_stream:
        # A clean drain reports the finalized row's own terminal cause; a
        # raising stream keeps reporting the exception it raised.
        assert REAL_CAUSE in harness.failed[0][2]


def test_finalize_wait_is_bounded_and_keeps_prior_retry_behaviour():
    """A worker that never retires its handle cannot pin the claim forever."""
    harness = _Harness("turn-1234-wedged", finalize_after=None)
    dispatcher = harness.dispatcher()
    dispatcher.FINALIZE_WAIT_SECONDS = 0.1
    harness.run(dispatcher)

    assert not harness.failed
    assert len(harness.requeued) == 1
    assert "did not finalize (status=streaming)" in harness.requeued[0][2]


def test_no_wait_when_no_execution_is_registered():
    """Deliveries whose stream never registered a worker classify immediately."""
    harness = _Harness("turn-1234-unregistered", finalize_after=None)
    dispatcher = harness.dispatcher()
    dispatcher.FINALIZE_WAIT_SECONDS = 30.0

    async def stream(_request):
        yield "data: [DONE]\n\n"

    dispatcher.service.chat_completion_stream = stream

    async def timed() -> float:
        loop = asyncio.get_running_loop()
        started = loop.time()
        await dispatcher._dispatch(_record(harness.turn_id))
        return loop.time() - started

    assert asyncio.run(timed()) < 5.0
    assert len(harness.requeued) == 1
