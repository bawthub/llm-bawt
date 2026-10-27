"""Codex fresh threads open on the app's continuity seed (TASK-1579)."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_bridge.continuity import SEED_FRAMING_FIELD, SeedFraming, seed_receipt
from agent_bridge.events import AgentEventKind
from codex_bridge.app_server import (
    APP_SERVER_VERSION, METHOD_NOT_FOUND, AppServerError, AppServerVersionMismatch,
)
from codex_bridge.thread_seed import CodexSeedUnsupported, seed_items, seed_thread
from tests import virtual_clock
from tests.test_codex_reset_evidence import ANSWER, TASK, assert_no_ack, harness, send  # noqa: F401

SEED = [
    {"source_id": "s1", "role": "summary", "content": "[Previous conversation 2 days ago]\nPlanned the seed.",
     "timestamp": 1.0},
    {"source_id": "m1", "role": "user", "content": "The code word is PELICAN-42.", "timestamp": 2.0},
    {"source_id": "m2", "role": "assistant", "content": "Noted.", "timestamp": 3.0},
]
FRAMING = SeedFraming(briefing="BRIEFING-HEADER", ack="SEED-ACK")
PINNED = "PINNED-SYSTEM-PROMPT"


class FakeAppServer:
    """The app-server calls a seed makes, with an optional scripted failure."""

    def __init__(self, thread_id="seeded-provider", inject_error=None, block_inject=None,
                 block_close=None):
        self.thread_id = thread_id
        self.inject_error = inject_error
        self.block_inject = block_inject
        self.block_close = block_close
        self.calls: list[tuple[str, dict]] = []
        self.closed = 0
        self.reaped = False
        self.inject_entered = asyncio.Event()
        self.close_entered = asyncio.Event()
        self._close_task = None

    async def request(self, method, params, *, timeout=None):
        self.calls.append((method, params))
        if method == "thread/start":
            return {"thread": {"id": self.thread_id}}
        self.inject_entered.set()
        if self.block_inject is not None:
            await self.block_inject.wait()
        if self.inject_error is not None:
            raise self.inject_error
        return {}

    def close(self):
        """Like the real client: one teardown task, however often it is asked for."""
        self.closed += 1
        if self._close_task is None:
            async def teardown():
                self.close_entered.set()
                if self.block_close is not None:
                    await self.block_close.wait()
                self.reaped = True
                return "closed"
            self._close_task = asyncio.ensure_future(teardown())
        return self._close_task


def seeded(harness, app_server=None, *, framing=FRAMING):
    harness.fields["inject_messages"] = json.dumps(SEED)
    if framing is not None:
        harness.fields[SEED_FRAMING_FIELD] = framing.to_field()
    harness.app_server = app_server or FakeAppServer()
    harness.bridge._open_app_server = AsyncMock(return_value=harness.app_server)
    harness.provider_id = harness.app_server.thread_id
    harness.script[0].thread_id = harness.app_server.thread_id
    return harness


def bound_to(harness, provider):
    return [(
        "http://test-app/v1/sessions/successor-thread/agent-session-key?bot_id=review-bot",
        {"backend": "codex", "session_key": provider, "model": "codex-fixture"},
    )]


def restored(harness):
    return [body["restored"] for _url, body in harness.receipts]


def test_seed_items_are_one_briefing_turn_without_the_system_prompt():
    items = seed_items(SEED, FRAMING)
    assert [(item["role"], item["content"][0]["type"]) for item in items] == [
        ("user", "input_text"), ("assistant", "output_text")]
    briefing = items[0]["content"][0]["text"]
    assert briefing.startswith("BRIEFING-HEADER\n\n[Previous conversation 2 days ago]")
    assert "User: The code word is PELICAN-42." in briefing
    assert "Assistant (you): Noted." in briefing
    assert items[1]["content"][0]["text"] == "SEED-ACK"


def test_seeded_follow_up_resumes_the_bound_seed_thread_and_pastes_the_prompt_once(harness):
    seeded(harness)
    harness.fields.update(system_prompt=PINNED, system_prompt_delivery="cold_start")
    events = send(harness)
    methods = [method for method, _params in harness.app_server.calls]
    assert methods == ["thread/start", "thread/inject_items"]
    injected = json.dumps(harness.app_server.calls[1][1])
    assert PINNED not in injected and "PELICAN-42" in injected
    assert harness.app_server.closed == 1
    harness.sdk.start_thread.assert_not_called()
    assert harness.sdk.resume_thread.call_args.args[0] == "seeded-provider"
    # A thread this send seeded is fresh: the pinned prompt rides its first turn.
    assert len(harness.prompts) == 1 and PINNED in harness.prompts[0] and TASK in harness.prompts[0]
    assert harness.writes == bound_to(harness, "seeded-provider")
    assert restored(harness) == [True]
    assert harness.receipts[0][1]["payload_digest"] == seed_receipt(SEED)["seed_digest"]
    done = [event for event in events if event.kind == AgentEventKind.ASSISTANT_DONE]
    assert len(done) == 1 and done[0].text == ANSWER
    assert done[0].raw == {"new_session_ack": {
        "thread_session_id": "successor-thread", "provider_session_id": "seeded-provider",
        "clean_start": False, **seed_receipt(SEED),
    }}


def test_later_turn_on_the_seeded_thread_does_not_repaste_the_pinned_prompt(harness):
    harness.fields.pop("bridge_reset_action")
    harness.fields.update(thread_resume_id="seeded-provider", inject_messages="[]",
                          system_prompt=PINNED, system_prompt_delivery="cold_start",
                          message="later turn")
    harness.bridge._open_app_server = AsyncMock(side_effect=AssertionError("no reseed"))
    send(harness)
    assert harness.sdk.resume_thread.call_args.args[0] == "seeded-provider"
    assert harness.prompts == ["later turn"]


def test_seeded_standalone_reset_binds_the_seed_thread_without_a_turn(harness):
    seeded(harness)
    harness.fields.update(bridge_reset_action="reset_only", message="/new", thread_resume_id="")
    events = send(harness)
    harness.sdk.start_thread.assert_not_called()
    harness.sdk.resume_thread.assert_not_called()
    assert not harness.prompts
    assert harness.writes == bound_to(harness, "seeded-provider")
    assert restored(harness) == [True]
    assert len(events) == 1 and events[0].kind == AgentEventKind.ASSISTANT_DONE
    assert events[0].text.startswith("Session reset. Seeded 1 summary record + 2 recent messages")
    assert events[0].raw == {"new_session_ack": {
        "thread_session_id": "successor-thread", "provider_session_id": "seeded-provider",
        "clean_start": False, **seed_receipt(SEED),
    }}


def test_cold_start_turn_is_seeded_and_bound_without_a_reset_ack(harness):
    seeded(harness)
    harness.fields.pop("bridge_reset_action")
    harness.fields.update(thread_resume_id="", message="first turn")
    events = send(harness)
    assert harness.sdk.resume_thread.call_args.args[0] == "seeded-provider"
    harness.sdk.start_thread.assert_not_called()
    assert harness.writes == bound_to(harness, "seeded-provider")
    assert restored(harness) == [True]
    assert events[-1].kind == AgentEventKind.ASSISTANT_DONE and events[-1].raw == {}


def test_resumed_turn_never_reseeds(harness):
    seeded(harness)
    harness.fields.pop("bridge_reset_action")
    harness.fields["message"] = "ordinary"
    harness.provider_id = "previous-provider"
    send(harness)
    harness.bridge._open_app_server.assert_not_called()
    assert harness.sdk.resume_thread.call_args.args[0] == "previous-provider"
    assert not harness.receipts


@pytest.mark.parametrize("failure", ["unsupported", "inject-error", "version", "no-thread-id", "binding"])
def test_seed_failure_fails_the_send_and_never_runs_a_blank_thread(harness, failure):
    app_server = FakeAppServer(
        thread_id="" if failure == "no-thread-id" else "seeded-provider",
        inject_error={"unsupported": AppServerError("thread/inject_items", METHOD_NOT_FOUND, "nope"),
                      "inject-error": AppServerError("thread/inject_items", -32000, "boom")}.get(failure),
    )
    seeded(harness, app_server)
    if failure == "version":
        harness.bridge._open_app_server = AsyncMock(side_effect=AppServerVersionMismatch("0.1.0"))
    if failure == "binding":
        harness.put_status = 503
    send(harness)
    assert_no_ack(harness)
    assert harness.events[-1].kind == AgentEventKind.ERROR
    if failure == "unsupported":
        assert APP_SERVER_VERSION in harness.events[-1].text
    harness.sdk.start_thread.assert_not_called()
    harness.sdk.resume_thread.assert_not_called()
    assert not harness.prompts
    assert restored(harness) == [False]


def test_unframed_seed_is_refused_before_any_provider_work(harness):
    seeded(harness, framing=None)
    send(harness)
    assert_no_ack(harness)
    assert "framing" in harness.events[-1].text
    harness.bridge._open_app_server.assert_not_called()
    harness.bridge._ensure_codex.assert_not_called()


def test_session_error_on_a_seeded_thread_is_not_retried_blank(harness):
    seeded(harness)
    harness.fields.pop("bridge_reset_action")
    harness.fields.update(thread_resume_id="", message="first turn")
    harness.script = [RuntimeError("thread not found")]
    send(harness)
    assert harness.events[-1].kind == AgentEventKind.ERROR
    assert harness.sdk.resume_thread.call_count == 1
    harness.sdk.start_thread.assert_not_called()


@pytest.mark.parametrize("second_cancel", [False, True], ids=["cancel-in-inject", "cancel-again-in-close"])
def test_a_cancelled_seed_returns_only_after_the_app_server_is_reaped(second_cancel):
    """The send's session is released when seed_thread returns, so a cancel,
    even one landing mid-teardown, must not return before the child is reaped.

    The gate opens in the same step as the last cancel, so whichever runs
    first, the seed's own exit records whether the teardown had finished.
    """
    async def run():
        close = asyncio.Event()
        app_server = FakeAppServer(block_inject=asyncio.Event(), block_close=close)
        bridge = MagicMock()
        bridge._open_app_server = AsyncMock(return_value=app_server)
        reaped_at_exit = []

        async def seed():
            try:
                return await seed_thread(bridge, object(), (), model="m", cwd=None,
                                         messages=SEED, framing=FRAMING, timeout=1)
            finally:
                reaped_at_exit.append(app_server.reaped)

        task = asyncio.create_task(seed())
        await app_server.inject_entered.wait()
        task.cancel()
        await app_server.close_entered.wait()
        if second_cancel:
            task.cancel()
        close.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        return app_server, reaped_at_exit

    # The virtual-clock loop raises instead of hanging if an awaited event never fires.
    app_server, reaped_at_exit = virtual_clock.run(run())
    assert reaped_at_exit == [True], "seed_thread returned while the app-server was still being torn down"
    assert app_server.closed == 1


def test_method_not_found_names_the_pinned_app_server():
    async def run():
        bridge = MagicMock()
        bridge._open_app_server = AsyncMock(return_value=FakeAppServer(
            inject_error=AppServerError("thread/inject_items", METHOD_NOT_FOUND, "nope")))
        await seed_thread(bridge, object(), (), model="m", cwd="/w", messages=SEED,
                          framing=FRAMING, timeout=1)

    with pytest.raises(CodexSeedUnsupported, match=APP_SERVER_VERSION):
        asyncio.run(run())
