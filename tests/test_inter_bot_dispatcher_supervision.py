"""Supervision / self-healing contract for the inter-bot dispatcher (TASK-778).

The 2026-08-07 stuck-delivery incident was caused by an unhandled exception
propagating out of ``run()`` into the leadership task and silently killing it.
These tests pin the two invariants that closed that hole and its follow-ups:

1. ``_leadership_loop`` never dies from a crash inside ``run()`` — it logs and
   restarts the loop under the same leadership.
2. ``run()`` itself absorbs transient sweep failures in-place with exponential
   backoff, so a flapping DB pool does not tear down the task instance on
   every blip (the naked outer guard was observed hot-looping 5× in 25s).
3. ``_schedule_eligible_targets`` no longer silently no-ops when the Redis
   subscriber is offline and ripe deliveries exist — the invisibility was the
   whole reason the original stuck delivery went unnoticed for hours.

These are pure asyncio unit tests against stubbed collaborators; no DB.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher


class _FakeStore:
    """Minimal InterBotDeliveryStore stand-in for supervision tests."""

    def __init__(self):
        self.eligible = []
        self.recover_expired_calls = 0
        self.eligible_targets_calls = 0
        self.recover_raises: list[Exception] = []

    def recover_expired(self):  # noqa: D401 — mirror store API
        self.recover_expired_calls += 1
        if self.recover_raises:
            raise self.recover_raises.pop(0)

    def eligible_targets(self):
        self.eligible_targets_calls += 1
        return list(self.eligible)

    def acquire_dispatcher_lock(self, *_args, **_kwargs):  # pragma: no cover
        return object()

    def release_dispatcher_lock(self, *_args, **_kwargs):  # pragma: no cover
        return None


def _make_dispatcher(subscriber_connected: bool = True) -> InterBotDeliveryDispatcher:
    subscriber = SimpleNamespace(connected=subscriber_connected)
    service = SimpleNamespace(
        config=SimpleNamespace(DEFAULT_USER="nick"),
        _redis_subscriber=subscriber,
    )
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    # Bypass __init__ so we don't touch the real store; wire minimal state.
    dispatcher.service = service
    dispatcher.store = _FakeStore()
    dispatcher.recovery_interval_seconds = 0.05
    dispatcher._wake_event = asyncio.Event()
    dispatcher._stop_event = asyncio.Event()
    dispatcher._task = None
    dispatcher._leader_task = None
    dispatcher._loop = None
    dispatcher._target_tasks = {}
    dispatcher._dispatch_tasks = set()
    dispatcher.claim_owner = "test-dispatcher"
    dispatcher._dispatcher_lock = None
    dispatcher._subscriber_gap_logged = False
    return dispatcher


@pytest.mark.asyncio
async def test_run_absorbs_transient_sweep_error_and_backs_off_in_place():
    """A transient store failure must not tear down the run() task."""

    dispatcher = _make_dispatcher()
    # First sweep raises, second sweep succeeds. If run() tore the task down
    # on the first raise, recover_expired_calls would stop at 2 (init + one
    # sweep). We assert it reaches at least 3 (init + failed sweep + recovered
    # sweep) all inside a single run() invocation.
    dispatcher.store.recover_raises = [RuntimeError("pool invalidated")]
    # Skip the initial pre-loop call to make counting unambiguous.
    dispatcher.store.recover_expired_calls = 0

    async def run_briefly():
        task = asyncio.create_task(dispatcher.run())
        # Poke it repeatedly so it doesn't wait on the recovery timeout.
        for _ in range(6):
            dispatcher._wake_event.set()
            await asyncio.sleep(0.02)
        dispatcher._stop_event.set()
        dispatcher._wake_event.set()
        await asyncio.wait_for(task, timeout=1.0)

    await run_briefly()

    # Init call + one failed sweep + at least one recovered sweep.
    assert dispatcher.store.recover_expired_calls >= 3, (
        f"expected ≥3 recover_expired calls across the transient failure; "
        f"got {dispatcher.store.recover_expired_calls}"
    )


@pytest.mark.asyncio
async def test_talk_outbound_wait_never_blocks_agent_delivery_sweep():
    dispatcher = _make_dispatcher()
    dispatcher.service._turn_log_store = object()
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked_reply(_deliveries, _turns):
        entered.set()
        await release.wait()

    dispatcher.talk_replies = SimpleNamespace(drain=blocked_reply)
    sweeps = []
    dispatcher._schedule_eligible_targets = lambda: sweeps.append(True)
    loop_task = asyncio.create_task(dispatcher.run())
    try:
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        dispatcher._wake_event.set()
        for _ in range(100):
            if len(sweeps) >= 2:
                break
            await asyncio.sleep(0.01)
        assert len(sweeps) >= 2
        assert dispatcher._talk_reply_task is not None
    finally:
        release.set()
        dispatcher._stop_event.set()
        dispatcher._wake_event.set()
        await asyncio.wait_for(loop_task, timeout=1.0)


@pytest.mark.asyncio
async def test_leadership_loop_survives_run_crash(monkeypatch, caplog):
    """A crash escaping run() must be logged and the loop restarted."""

    dispatcher = _make_dispatcher()
    dispatcher.leadership_retry_delay_seconds = 0  # exercise restart without a real pause
    call_count = 0
    crash_seen = asyncio.Event()

    async def fake_run():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            crash_seen.set()
            raise RuntimeError("simulated hard crash")
        # Second invocation: exit cleanly so leadership loop can end.
        dispatcher._stop_event.set()

    dispatcher.run = fake_run  # type: ignore[assignment]

    # Bypass the advisory-lock call in _leadership_loop.
    dispatcher.store.acquire_dispatcher_lock = lambda *_a, **_k: object()

    caplog.set_level(logging.ERROR, logger="llm_bawt.service.inter_bot_dispatcher")
    task = asyncio.create_task(dispatcher._leadership_loop())
    try:
        await asyncio.wait_for(crash_seen.wait(), timeout=1.0)
        # Let the outer guard log and restart run().
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        dispatcher._stop_event.set()

    assert call_count >= 2, "leadership loop did not restart run() after crash"
    assert any(
        "dispatcher loop crashed" in rec.getMessage() for rec in caplog.records
    ), "expected the outer guard to log the run() crash"


@pytest.mark.asyncio
async def test_scheduler_warns_once_when_subscriber_offline_with_ripe_work(caplog):
    """Silent-dead-consumer detection: no more invisible no-op."""

    dispatcher = _make_dispatcher(subscriber_connected=False)
    dispatcher.store.eligible = ["target-a", "target-b"]

    caplog.set_level(logging.WARNING, logger="llm_bawt.service.inter_bot_dispatcher")
    dispatcher._schedule_eligible_targets()
    dispatcher._schedule_eligible_targets()  # second call must be throttled
    dispatcher._schedule_eligible_targets()  # third call still throttled

    warning_messages = [
        rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING
    ]
    matching = [m for m in warning_messages if "Redis subscriber offline" in m]
    assert len(matching) == 1, (
        f"expected exactly one throttled warning; got {len(matching)}: {matching}"
    )
    assert "target-a" in matching[0] and "target-b" in matching[0]


@pytest.mark.asyncio
async def test_scheduler_stays_silent_when_subscriber_offline_but_queue_empty(caplog):
    """No noise when the offline subscriber has nothing to dispatch anyway."""

    dispatcher = _make_dispatcher(subscriber_connected=False)
    dispatcher.store.eligible = []

    caplog.set_level(logging.WARNING, logger="llm_bawt.service.inter_bot_dispatcher")
    dispatcher._schedule_eligible_targets()

    assert not [
        rec for rec in caplog.records
        if rec.levelno >= logging.WARNING
        and "Redis subscriber offline" in rec.getMessage()
    ], "should not warn when the offline subscriber has no ripe work"


@pytest.mark.asyncio
async def test_scheduler_recovery_logs_when_subscriber_reconnects(caplog):
    """Gap-closed info log fires exactly once on reconnect."""

    dispatcher = _make_dispatcher(subscriber_connected=False)
    dispatcher.store.eligible = ["target-a"]

    caplog.set_level(logging.INFO, logger="llm_bawt.service.inter_bot_dispatcher")
    dispatcher._schedule_eligible_targets()  # triggers the offline warning
    dispatcher.service._redis_subscriber.connected = True
    dispatcher._schedule_eligible_targets()  # triggers the recovery info log
    dispatcher._schedule_eligible_targets()  # steady state — no second info log

    reconnect_messages = [
        rec.getMessage()
        for rec in caplog.records
        if rec.levelno == logging.INFO
        and "Redis subscriber back online" in rec.getMessage()
    ]
    assert len(reconnect_messages) == 1, (
        f"expected one reconnect log; got {reconnect_messages}"
    )


@pytest.mark.asyncio
async def test_ordinary_queue_lifecycle_routes_to_requesting_user():
    from unittest.mock import AsyncMock

    dispatcher = _make_dispatcher()
    publish = AsyncMock()
    dispatcher.service._redis_subscriber.publish_tool_event = publish
    record = SimpleNamespace(
        id="delivery-1", status="DISPATCHING", sender_bot_id="target",
        target_bot_id="target", author_entity_type="user", author_entity_id="alice",
        user_message_id="user-1", turn_id="turn-1", target_turn_id="turn-1",
        delivery_mode="turn", session_policy="continue", reset_status=None,
        reset_reason=None, old_session_id=None, new_session_id=None, reset_at=None,
        overflow_recovery_count=0, attempt_count=1, next_retry_at=None,
        last_error=None, project_id=None, task_id=None, message_kind=None,
        metadata={"ordinary_chat": {"session_id": "session-1"}},
    )
    await dispatcher._emit(record)
    publish.assert_awaited_once()
    args = publish.await_args.args
    assert args[0:2] == ("target", "alice")
    assert args[2]["user_message_id"] == "user-1"
    assert args[2]["session_id"] == "session-1"
    assert args[2]["ordinary_chat"] is True

    publish.reset_mock()
    record.metadata = {}
    record.author_entity_type = "bot"
    await dispatcher._emit(record)
    assert "ordinary_chat" not in publish.await_args.args[2]
    assert "session_id" not in publish.await_args.args[2]


@pytest.mark.asyncio
async def test_fallback_turn_does_not_hold_target_drain_open(monkeypatch):
    """TASK-1067: while a delivery-started turn streams, the target must stay
    drainable so the next delivery can steer into that turn."""

    import llm_bawt.bots as bots

    monkeypatch.setattr(bots, "BotManager", lambda _config: SimpleNamespace(
        get_bot=lambda _bot: SimpleNamespace(agent_backend="claude-code")
    ))
    from unittest.mock import AsyncMock
    monkeypatch.setattr("llm_bawt.service.maintenance_queue.MaintenanceQueue.publish_ready", AsyncMock())
    dispatcher = _make_dispatcher()
    record = SimpleNamespace(id="delivery-1", status="DISPATCHING")
    dispatcher.store.claim_next = lambda *_a, **_k: record
    release = asyncio.Event()
    started = asyncio.Event()

    async def fake_emit(_record):
        return None

    async def fake_dispatch(_record):
        started.set()
        await release.wait()

    dispatcher._emit = fake_emit
    dispatcher._dispatch = fake_dispatch

    await asyncio.wait_for(dispatcher._drain_target("byte"), timeout=1.0)
    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert len(dispatcher._dispatch_tasks) == 1

    dispatcher._wake_event.clear()
    release.set()
    await asyncio.sleep(0.01)
    assert not dispatcher._dispatch_tasks
    assert dispatcher._wake_event.is_set()
