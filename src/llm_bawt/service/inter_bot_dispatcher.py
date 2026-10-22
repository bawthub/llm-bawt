"""Application-owned dispatcher for durable inter-bot deliveries."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

import httpx

from ..inter_bot_delivery import (
    DELIVERED,
    DISPATCHING,
    FAILED,
    QUEUED,
    DeliveryRecord,
    InterBotDeliveryStore,
)
from ..inter_bot_delivery_retry import fail_claim_with_restart_retry, retry_original_author
from ..inter_bot_delivery_user import resolve_delivery_user
from .inter_bot_dispatcher_recovery import StartupCheck, StartupRecoveryRunner
from .inter_bot_dispatcher_steer import (
    APP_RUN_NOT_READY,
    BRIDGE_RUN_NOT_READY,
    DEFINITIVE_NOT_ACCEPTED,
    NOT_READY_BOUND,
    OPEN_TURN_STATUSES,
    SteerGeneration,
)
from .schemas import ChatCompletionRequest
from .turn_logs import DeliveryTargetBusy

logger = logging.getLogger(__name__)


class InterBotDeliveryDispatcher:
    """Drain one FIFO callback per target through the normal chat path.

    The event is the fast path (enqueue and terminal turn transitions call
    ``wake``); the timed sweep is only restart/missed-signal recovery.
    """

    # TASK-1234: bound on waiting for the stream worker to finalize the turn
    # row after the drain ends. The claim heartbeat keeps the lease meanwhile.
    FINALIZE_WAIT_SECONDS = 30.0
    FINALIZE_POLL_SECONDS = 0.05
    # A fallback turn refused at admission is deferred, never spun at delay 0.
    TARGET_BUSY_REQUEUE_SECONDS = 1.0

    def __init__(
        self, service: Any, *, recovery_interval_seconds: float = 15.0,
        leadership_retry_delay_seconds: float = 1.0,
        startup_recovery: StartupCheck | None = None,
    ):
        self.service = service
        self.store = InterBotDeliveryStore(service.config)
        from .talk_reply_outbox import TalkReplyOutbox
        self.talk_replies = TalkReplyOutbox(self.store.engine) if self.store.engine is not None else None
        from .quiesce_control import QuiesceControlRunner, build_pump
        # TASK-1233 B3: ask each quiesce-held running turn to stop (leader-only).
        self.quiesce_control = QuiesceControlRunner(lambda: build_pump(service, self.store.engine))
        self.recovery_interval_seconds = max(1.0, recovery_interval_seconds)
        self.leadership_retry_delay_seconds = max(0.0, leadership_retry_delay_seconds)
        self._wake_event = asyncio.Event()
        # TASK-1232: the startup recovery check, once per process, never on a
        # timer. Its end wakes the loop: a settled turn frees its target.
        self._recovery = (
            StartupRecoveryRunner(startup_recovery, on_done=self._wake_event.set)
            if startup_recovery is not None else None
        )
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._leader_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._target_tasks: dict[str, asyncio.Task] = {}
        self._dispatch_tasks: set[asyncio.Task] = set()
        self._announce_tasks: set[asyncio.Task] = set()
        self._talk_reply_task: asyncio.Task | None = None
        self._new_session_outbox = None
        self.claim_owner = f"dispatcher-{uuid.uuid4().hex}"
        self._dispatcher_lock = None
        self._subscriber_gap_logged = False

    @property
    def is_leader(self) -> bool:
        """Whether this process holds the dispatcher leadership lock now."""
        return self._dispatcher_lock is not None

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        if self._leader_task is None or self._leader_task.done():
            self._leader_task = asyncio.create_task(
                self._leadership_loop(), name="inter-bot-delivery-leadership"
            )

    async def _leadership_loop(self) -> None:
        passive_logged = False
        while not self._stop_event.is_set():
            if self._dispatcher_lock is None:
                self._dispatcher_lock = self.store.acquire_dispatcher_lock()
                if self._dispatcher_lock is None:
                    if not passive_logged:
                        logger.info(
                            "inter-bot dispatcher is passive; another process owns the lock"
                        )
                        passive_logged = True
                    try:
                        await asyncio.wait_for(self._stop_event.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        pass
                    continue
                passive_logged = False
                logger.info("inter-bot dispatcher acquired singleton leadership")
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(
                    self.run(), name="inter-bot-delivery-dispatcher"
                )
            try:
                await self._task
            except Exception:
                # An unhandled crash in run() must not kill leadership forever:
                # before this guard, one transient DB error here left QUEUED
                # deliveries with no consumer until the next app restart.
                logger.exception(
                    "inter-bot dispatcher loop crashed; restarting under leadership"
                )
            if not self._stop_event.is_set():
                logger.warning("inter-bot dispatcher loop exited; restarting under leadership")
                await asyncio.sleep(self.leadership_retry_delay_seconds)

    async def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        tasks = [
            task for task in (*self._target_tasks.values(), *self._dispatch_tasks,
                              getattr(self, "_talk_reply_task", None))
            if task is not None and not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        quiesce_control = getattr(self, "quiesce_control", None)
        if quiesce_control is not None:
            await quiesce_control.stop()
        if self._task:
            await asyncio.gather(self._task, return_exceptions=True)
        if self._leader_task:
            await asyncio.gather(self._leader_task, return_exceptions=True)
        recovery = getattr(self, "_recovery", None)
        if recovery is not None:
            # Leader-only work: stop it before the lock is released.
            await recovery.stop()
        self.store.release_dispatcher_lock(self._dispatcher_lock)
        self._dispatcher_lock = None

    def wake(self, target_bot_id: str | None = None) -> None:
        # target_bot_id is accepted for callers/documentation; eligible targets
        # are always read transactionally from PostgreSQL to avoid lost hints.
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._wake_event.set)
        else:
            self._wake_event.set()

    def announce_admitted(self, record: DeliveryRecord) -> None:
        """Publish a just-admitted row's QUEUED lifecycle, then wake.

        Sync-safe for admission code. Waking only after the publish keeps
        QUEUED ahead of this process's DISPATCHING for the same row, so every
        tab (not just the submitting one) shows the queued state and Cancel.
        """
        loop = self._loop
        if loop is None or not loop.is_running():
            self.wake(record.target_bot_id)
            return

        async def publish_then_wake() -> None:
            try:
                await self._emit(record)
            finally:
                self._wake_event.set()

        def schedule() -> None:
            task = loop.create_task(publish_then_wake(), name=f"inter-bot-announce:{record.id}")
            self._announce_tasks.add(task)
            task.add_done_callback(self._announce_tasks.discard)

        loop.call_soon_threadsafe(schedule)

    async def run(self) -> None:
        # Belt-and-suspenders around the outer leadership guard: transient
        # SQLAlchemy pool blips (e.g. InvalidatePoolError from a DB restart)
        # showed up in prod as 5 crashes in 25s because the outer guard tore
        # down and rebuilt the whole run() task each time. Guarding the sweep
        # in-place lets one task instance ride through pool recovery with
        # exponential backoff instead of hot-looping.
        recovery = getattr(self, "_recovery", None)
        try:
            self.store.recover_expired(live_owner=self.claim_owner)
        except Exception:
            logger.exception("initial recover_expired failed; continuing")
        if recovery is not None:
            # TASK-1232: the first run() entry is this process's startup under
            # held leadership; a crash re-entry never reruns the check. After
            # recover_expired: recovery steps defer claimed deliveries to it,
            # so the check must see the claims it released.
            await recovery.run_once(stop=self._stop_event)
        self._wake_event.set()
        consecutive_failures = 0
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._wake_event.wait(), timeout=self.recovery_interval_seconds
                )
            except asyncio.TimeoutError:
                pass
            self._wake_event.clear()
            try:
                self.store.recover_expired(live_owner=self.claim_owner)
                self._schedule_eligible_targets()
                if getattr(self.store, "engine", None) is not None:
                    from .session_lifecycle.outbox import NewSessionOutbox
                    from .session_lifecycle.receipt import NewSessionReceipt

                    settled = await asyncio.to_thread(
                        NewSessionReceipt.reconcile_typed, self.store.engine,
                    )
                    for delivery in settled:
                        await self._emit(delivery)
                    from .session_lifecycle.policy_receipt import BotPolicyReceipt

                    settled_policy = await asyncio.to_thread(
                        BotPolicyReceipt.reconcile, self.store.engine,
                    )
                    for delivery in settled_policy:
                        await self._emit(delivery)
                    if self._new_session_outbox is None:
                        self._new_session_outbox = NewSessionOutbox(self.store.engine)
                    await self._new_session_outbox.drain(
                        getattr(self.service, "_redis_subscriber", None),
                    )
                talk_replies = getattr(self, "talk_replies", None)
                running_reply = getattr(self, "_talk_reply_task", None)
                if talk_replies is not None and (running_reply is None or running_reply.done()):
                    reply_task = asyncio.create_task(
                        talk_replies.drain(self.store, self.service._turn_log_store),
                        name="talk-reply-drain",
                    )
                    reply_task.add_done_callback(self._talk_reply_done)
                    self._talk_reply_task = reply_task
                quiesce_control = getattr(self, "quiesce_control", None)
                if quiesce_control is not None:
                    quiesce_control.maybe_start()
            except Exception:
                consecutive_failures += 1
                backoff = min(30.0, float(2 ** min(consecutive_failures - 1, 5)))
                logger.exception(
                    "dispatcher sweep iteration failed (%d consecutive); "
                    "backing off %.1fs",
                    consecutive_failures,
                    backoff,
                )
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
            else:
                consecutive_failures = 0
        await asyncio.gather(
            *self._target_tasks.values(), *self._dispatch_tasks, return_exceptions=True
        )

    @staticmethod
    def _talk_reply_done(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception():
            logger.error("Talk reply sweep failed", exc_info=task.exception())

    def _schedule_eligible_targets(self) -> None:
        subscriber = getattr(self.service, "_redis_subscriber", None)
        if subscriber is None or not getattr(subscriber, "connected", False):
            # Silent-dead-consumer guard: if the Redis subscriber isn't
            # connected we cannot dispatch, but that must not be an invisible
            # no-op when ripe deliveries are sitting in the queue. Log once
            # per gap (throttled) so ops can see it in the same shape as the
            # 2026-08-07 stuck-delivery incident.
            try:
                eligible = list(self.store.eligible_targets())
            except Exception:
                logger.exception(
                    "eligible_targets probe failed while Redis subscriber offline"
                )
                return
            if eligible and not self._subscriber_gap_logged:
                logger.warning(
                    "Redis subscriber offline; %d target(s) with QUEUED deliveries "
                    "cannot be dispatched: %s",
                    len(eligible),
                    ", ".join(sorted(eligible)[:5]),
                )
                self._subscriber_gap_logged = True
            return
        if self._subscriber_gap_logged:
            logger.info("Redis subscriber back online; resuming dispatch sweeps")
            self._subscriber_gap_logged = False
        for target in self.store.eligible_targets():
            existing = self._target_tasks.get(target)
            if existing and not existing.done():
                continue
            task = asyncio.create_task(self._drain_target(target), name=f"inter-bot:{target}")
            self._target_tasks[target] = task
            task.add_done_callback(lambda done, t=target: self._target_done(t, done))

    def _target_done(self, target: str, task: asyncio.Task) -> None:
        if self._target_tasks.get(target) is task:
            self._target_tasks.pop(target, None)
        if not task.cancelled() and task.exception():
            logger.error("inter-bot target drain failed for %s", target, exc_info=task.exception())
        elif not task.cancelled() and task.result():
            self._wake_event.set()
        # No progress (busy/maintenance) or failed publication waits for a real
        # wake/recovery sweep; immediately waking here would hot-loop the fence.

    async def _drain_target(self, target_bot_id: str) -> bool:
        from ..bots import BotManager

        bot = BotManager(self.service.config).get_bot(target_bot_id)
        backend = (getattr(bot, "agent_backend", None) or "").strip()
        from .maintenance_queue import MaintenanceQueue
        await MaintenanceQueue(self.service, dispatcher=self).publish_ready(target_bot_id)
        record = self.store.claim_next(
            target_bot_id,
            claim_owner=self.claim_owner,
            steer_capable=(backend == "claude-code"),
        )
        if record is None:
            return False
        await self._emit(record)
        if record.status == "STEERING":
            await self._steer(record)
        else:
            # A fallback turn runs for its whole duration. Holding the target's
            # drain open for it would hide later deliveries until the turn ends,
            # so they could never steer into it (TASK-1067). The claim already
            # serializes the target; the stream runs as its own tracked task.
            task = asyncio.create_task(
                self._dispatch(record), name=f"inter-bot-dispatch:{record.id}"
            )
            self._dispatch_tasks.add(task)
            task.add_done_callback(self._dispatch_done)
        return True

    def _dispatch_done(self, task: asyncio.Task) -> None:
        self._dispatch_tasks.discard(task)
        if not task.cancelled() and task.exception():
            logger.error("inter-bot fallback dispatch failed", exc_info=task.exception())
        self._wake_event.set()

    async def _steer(self, record: DeliveryRecord) -> None:
        from ..inter_bot_reply_channel import delivery_reply_channel_note
        from .routes.chat import ChatSteerRequest, steer_active_turn
        from .system_gate_admission import SteerGateRefused

        claim_token = record.claim_token or ""
        payload = self.store.payload(record.id) or {}
        message = str((payload.get("messages") or [{}])[-1].get("content") or record.message)
        user_id = resolve_delivery_user(self.service.config, payload.get("user"))
        # TASK-1251: the claimed row names the steer RPC this attempt uses.
        steer = SteerGeneration.of(record)
        target_turn_id = record.target_turn_id or ""
        heartbeat = asyncio.create_task(
            self._heartbeat(record.id, claim_token),
            name=f"inter-bot-steer-heartbeat:{record.id}",
        )
        try:
            response = await steer_active_turn(
                self.service,
                ChatSteerRequest(
                    turn_id=target_turn_id,
                    message=message,
                    message_id=record.user_message_id,
                    bot_id=record.target_bot_id,
                    user_id=user_id,
                    # Bot deliveries never kill the target's running tools
                    # (TASK-934): queue for the next tool boundary.
                    priority="next",
                ),
                steer_request_id=steer.request_id,
                author=record.author,
                wire_suffix=delivery_reply_channel_note(record),
            )
            if response.ok:
                self.store.mark_transport_accepted(record.id, claim_token)
            if response.ok and response.persisted:
                delivered = self.store.mark_delivered(
                    record.id,
                    claim_token,
                    response_model=None,
                    response_chars=0,
                )
                if delivered:
                    await self._emit(delivered)
            else:
                queued = self.store.requeue(
                    record.id,
                    claim_token,
                    response.detail,
                    delay_seconds=1,
                )
                if queued:
                    await self._emit(queued)
        except SteerGateRefused as refused:
            # A live system gate refused before any RPC (TASK-1233). Not an
            # attempt, and the steer target is kept: the claim gate holds the
            # row until release, then the same target is steered or, if its
            # run ended meanwhile, the usual fallback applies.
            queued = self.store.requeue(
                record.id,
                claim_token,
                str(refused.detail),
                delay_seconds=self.TARGET_BUSY_REQUEUE_SECONDS,
                refund_attempt=True,
            )
            if queued:
                await self._emit(queued)
        except Exception as exc:
            error = str(getattr(exc, "detail", None) or exc or exc.__class__.__name__)
            current = self.store.get(record.id) or record
            accepted = current.transport_accepted_at is not None
            not_ready_count = steer.not_ready_count_for(target_turn_id)
            failure_action = self._steer_failure_action(
                error,
                accepted=accepted,
                target_open=(
                    not accepted and error in BRIDGE_RUN_NOT_READY
                    and self._steer_target_open(target_turn_id)
                ),
                not_ready_count=not_ready_count,
            )

            if failure_action == "wait_ready_next_generation":
                # TASK-1251: the target turn is still open but its run has not
                # registered with the bridge yet. Generation k definitively did
                # not reach a run, so the next claim steers the same target
                # under a fresh RPC id; retrying k would only re-read its
                # cached rejection. Not a delivery attempt.
                queued = self.store.requeue(
                    record.id,
                    claim_token,
                    f"steer target not ready: {error} (generation {steer.generation})",
                    delay_seconds=SteerGeneration.backoff_seconds(not_ready_count + 1),
                    refund_attempt=True,
                    metadata_merge=steer.advance(not_ready_target=target_turn_id),
                )
                if queued:
                    await self._emit(queued)
            elif failure_action == "fallback":
                # Steer RPC k proved it did not reach the target's run. Clear
                # that stale target exactly once; the next locked claim will
                # steer whichever run is current (under generation k+1, so the
                # bridge's cached rejection of k is never replayed) or reserve
                # one idle fallback turn. The failed probe is not an attempt.
                queued = self.store.requeue(
                    record.id,
                    claim_token,
                    error,
                    delay_seconds=1,
                    reject_steer_target=True,
                    refund_attempt=True,
                    metadata_merge=steer.advance(),
                )
                if queued:
                    await self._emit(queued)
            elif failure_action == "wait_ready":
                # Turn metadata registration races its first tool/model work.
                # Waiting for agent_session_key must not consume the bounded
                # delivery attempt budget or prematurely dead-letter the message.
                queued = self.store.requeue(
                    record.id,
                    claim_token,
                    error,
                    delay_seconds=1,
                    refund_attempt=True,
                )
                if queued:
                    await self._emit(queued)
            elif current.attempt_count >= current.max_attempts:
                failed = self.store.fail_claim(record.id, claim_token, error)
                if failed:
                    await self._emit(failed)
            else:
                # Timeouts and accepted-but-unpersisted steers retain the exact
                # target turn + steer generation, so the replay reuses the RPC id
                # the bridge may already have accepted. Replaying repairs message
                # persistence without issuing a second interrupt or idle turn.
                queued = self.store.requeue(
                    record.id,
                    claim_token,
                    error,
                    delay_seconds=min(30.0, float(2 ** max(0, current.attempt_count - 1))),
                )
                if queued:
                    await self._emit(queued)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    async def _dispatch(self, record: DeliveryRecord) -> None:
        claim_token = record.claim_token or ""
        payload = self.store.payload(record.id)
        if not payload:
            failed = self.store.fail_claim(record.id, claim_token, "delivery payload is missing")
            if failed:
                await self._emit(failed)
            return

        heartbeat = asyncio.create_task(
            self._heartbeat(record.id, claim_token),
            name=f"inter-bot-heartbeat:{record.id}",
        )
        finalize_awaited = False
        try:
            payload = {
                **payload,
                "inter_bot_claim_token": claim_token,
                "stream": True,
            }
            request = ChatCompletionRequest.model_validate(payload)
            if not self.store.mark_transport_accepted(record.id, claim_token):
                return
            async for _chunk in self.service.chat_completion_stream(request):
                # Durable fallback has no originating HTTP consumer. Draining the
                # canonical generator starts/joins TurnStreamWorker; live text,
                # reasoning, tools, offsets, persistence, and turn_complete are
                # published independently through the unified Redis stream.
                pass
            await self._await_turn_finalized(record.turn_id)
            finalize_awaited = True
            turn = self.service._turn_log_store.get_turn(record.turn_id)
            if turn is None or turn.ended_at is None:
                # TASK-1148: never mask a recorded terminal cause behind the
                # generic message; say which state the turn row was left in.
                raise RuntimeError(self._unfinalized_turn_error(record.turn_id, turn))
            if turn.status not in {"ok", "completed"} or turn.error_text:
                raise RuntimeError(
                    turn.error_text
                    or f"streamed durable turn ended with status={turn.status}"
                )
            content = turn.response_text or ""
            delivered = self.store.mark_delivered(
                record.id,
                claim_token,
                response_model=turn.model,
                response_chars=len(content),
            )
            if delivered:
                await self._finish_delivered(delivered)
        except DeliveryTargetBusy as exc:
            # save_turn, the turn's first write, refused it: another producer
            # won the target, or a system gate went live after the claim
            # (SystemGateRefused is a TargetBusy). Nothing reached the
            # transport, so this is no attempt and no acceptance; under a
            # gate the claim then holds the row until release (TASK-1233).
            queued = self.store.requeue(
                record.id, claim_token, str(exc),
                delay_seconds=self.TARGET_BUSY_REQUEUE_SECONDS,
                clear_delivery_mode=True, refund_attempt=True,
            )
            if queued:
                await self._emit(queued)
        except Exception as exc:  # classification is intentionally centralized
            error = str(exc) or exc.__class__.__name__
            if not finalize_awaited:
                await self._await_turn_finalized(record.turn_id)
            current = self.store.get(record.id) or record
            turn_status, turn_ended_at = self.store.turn_state(record.id)
            turn = self.service._turn_log_store.get_turn(record.turn_id)
            turn_error = getattr(turn, "error_text", None) if turn else None
            if turn_status in {"aborted", "cancelled", "cancelling"}:
                stopped = self.store.cancel_stopped_claim(record.id, claim_token)
                if stopped:
                    await self._emit(stopped)
                    return
                if turn_status in {"aborted", "cancelled"}:
                    # TASK-1231B: a turn its bridge host killed while shutting
                    # down gets one restart-retry child, committed with FAILED.
                    failed, child = fail_claim_with_restart_retry(
                        self.store, record, claim_token, error,
                    )
                    if failed:
                        await self._emit(failed)
                    if child:
                        await self._emit(child)
                    return
            from .session_lifecycle.recovery import NewSessionRecovery

            if current.transport_accepted_at is not None and NewSessionRecovery.is_reset({
                "session_policy": current.session_policy, "metadata_json": current.metadata,
                "payload_json": payload,
            }) and not (turn_ended_at is not None and turn_status in {"ok", "completed"} and not turn_error):
                with self.store.engine.begin() as conn:
                    ambiguous = NewSessionRecovery.mark_ambiguous(
                        conn, delivery_id=record.id, claim_token=claim_token,
                        reason=f"accepted reset outcome is ambiguous: {error}",
                    )
                if ambiguous:
                    await self._emit(ambiguous)
                return
            if self._is_context_overflow(error):
                from ..bots import BotManager

                bot = BotManager(self.service.config).get_bot(record.target_bot_id)
                backend = (
                    (getattr(bot, "agent_backend", None) or getattr(bot, "harness", None))
                    if bot else None
                )
                recovered = self.store.recover_context_overflow(
                    record.id,
                    claim_token,
                    backend=backend,
                    error=error,
                )
                if recovered:
                    await self._emit(recovered)
                    return
                reason = (
                    "context overflow persisted after the single controlled recovery"
                    if current.overflow_recovery_count >= 1
                    else "context overflow cannot be replayed safely after output or tool side effects"
                )
                failed = self.store.fail_claim(record.id, claim_token, reason)
                if failed:
                    await self._emit(failed)
                return
            # Automatic retries are safe before target acceptance (reserved/no
            # Automatic retries are safe before target acceptance (reserved/no
            # turn) and for deterministic agent runs, where Redis request-id
            # dedupe reattaches rather than launching a second SDK turn. Local
            # backends are capability-gated out of durable mode.
            deterministic_agent_run = (
                isinstance(payload.get("inter_bot_bridge_request_id"), str)
                and payload["inter_bot_bridge_request_id"].startswith("req_delivery_")
            )
            if (
                turn_ended_at is not None
                and turn_status in {"ok", "completed"}
                and not turn_error
            ):
                delivered = self.store.mark_delivered(
                    record.id,
                    claim_token,
                    response_model=None,
                    response_chars=0,
                )
                if delivered:
                    await self._finish_delivered(delivered)
            elif (
                self._is_terminal(exc)
                or current.attempt_count >= current.max_attempts
                # TASK-1095: reattaching to a deterministic run only helps while
                # it may still be live. Once the turn has ended unsuccessfully,
                # every retry re-reads the same finished Redis stream from the
                # start (re-broadcasting its whole history) and fails again.
                or turn_ended_at is not None
                or not (turn_status in (None, "reserved") or deterministic_agent_run)
            ):
                failed = self.store.fail_claim(record.id, claim_token, error)
                if failed:
                    await self._emit(failed)
            else:
                delay = min(60.0, float(2 ** max(0, current.attempt_count - 1)))
                queued = self.store.requeue(
                    record.id, claim_token, error, delay_seconds=delay
                )
                if queued:
                    await self._emit(queued)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    async def _finish_delivered(self, record: DeliveryRecord) -> None:
        """A delivered reset still needs an exact app-authored receipt."""
        from .session_lifecycle.receipt import NewSessionReceipt
        from .session_lifecycle.recovery import NewSessionRecovery

        if record.session_policy in {"reset_retain_history", "reset_without_history"}:
            from .session_lifecycle.policy_receipt import BotPolicyReceipt

            try:
                receipt = await asyncio.to_thread(
                    BotPolicyReceipt.ensure, self.store.engine, record,
                )
            except Exception:
                logger.exception("bot-policy reset receipt verification failed: %s", record.id)
                receipt = None
            if receipt is not None:
                await self._emit(self.store.get(record.id) or record)
                return
            ambiguous = await asyncio.to_thread(
                NewSessionRecovery.fail_missing_policy_receipt,
                self.store.engine, record.id,
                "reset follow-up finished without exact provider binding and persisted receipt",
            )
            if ambiguous is None:
                current = self.store.get(record.id)
                if current is None or not current.metadata.get("new_session_receipt"):
                    raise RuntimeError(f"bot-policy reset receipt could not be settled: {record.id}")
                ambiguous = current
            await self._emit(ambiguous)
            return
        try:
            intent = NewSessionReceipt.typed_intent(record)
        except (TypeError, ValueError) as exc:
            intent = None
            reason = f"typed reset provenance is invalid: {exc}"
        else:
            reason = "typed reset finished without provider binding and persisted receipt"
        if intent is not None:
            try:
                receipt = await asyncio.to_thread(
                    NewSessionReceipt.ensure, self.store.engine, record, intent,
                )
            except Exception:
                logger.exception("typed reset receipt verification failed: %s", record.id)
                receipt = None
            if receipt is not None:
                await self._emit(self.store.get(record.id) or record)
                return
        elif (not isinstance(record.metadata.get("new_session_intent"), dict)
              or record.metadata["new_session_intent"].get("origin") != "chat_command"):
            await self._emit(record)
            return
        ambiguous = await asyncio.to_thread(
            NewSessionRecovery.fail_missing_typed_receipt,
            self.store.engine, record.id, reason,
        )
        if ambiguous is None:
            # A concurrent receipt may have won the row lock; otherwise a DB
            # outage leaves this state unresolved for the next recovery sweep.
            current = self.store.get(record.id)
            if current is None or not current.metadata.get("new_session_receipt"):
                raise RuntimeError(f"typed reset receipt could not be settled: {record.id}")
            ambiguous = current
        await self._emit(ambiguous)

    async def _heartbeat(self, delivery_id: str, claim_token: str) -> None:
        while True:
            await asyncio.sleep(60)
            if not self.store.renew_lease(delivery_id, claim_token):
                return

    @staticmethod
    def _steer_failure_action(
        error: str, *, accepted: bool, target_open: bool = False, not_ready_count: int = 0,
    ) -> str:
        """Classify one steer failure without guessing acceptance.

        Once transport acceptance is recorded, no failure may convert the
        delivery into a fresh turn; retries must replay the same steer RPC and
        repair persistence. Before acceptance, missing run metadata only waits
        (no RPC was published). A bridge "no run" rejection while the target
        turn is still open means its run has not registered yet: steer the same
        target again under the next generation, up to ``NOT_READY_BOUND``
        times (TASK-1251). Any other definitive rejection clears stale steer
        intent.
        """
        if accepted:
            return "retry_same_steer"
        if error == APP_RUN_NOT_READY:
            return "wait_ready"
        if error in BRIDGE_RUN_NOT_READY and target_open and not_ready_count < NOT_READY_BOUND:
            return "wait_ready_next_generation"
        if error in DEFINITIVE_NOT_ACCEPTED:
            return "fallback"
        return "retry_same_steer"

    def _steer_target_open(self, turn_id: str) -> bool:
        """Whether ``turn_id`` may still register a bridge run (TASK-1251)."""
        if not turn_id:
            return False
        turn = self.service._turn_log_store.get_turn(turn_id)
        return bool(turn is not None and turn.ended_at is None and turn.status in OPEN_TURN_STATUSES)

    async def _await_turn_finalized(self, turn_id: str | None) -> None:
        """Wait (bounded) until the stream worker has finalized ``turn_id``.

        TASK-1234: the worker hands an upstream failure to the chat stream,
        which ends the drain, *before* it writes the terminal turn row; its
        execution handle is retired only after finalize. Classifying on the
        earlier read saw ``streaming``, so the TASK-1095 ended-turn guard never
        fired and each fast replay burned an attempt (incident a67418db).
        """
        if not turn_id:
            return
        from .turn_execution import turn_executions

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.FINALIZE_WAIT_SECONDS
        while turn_executions.get(turn_id) is not None:
            if loop.time() >= deadline:
                logger.warning(
                    "durable turn %s still finalizing after %.1fs; classifying anyway",
                    turn_id, self.FINALIZE_WAIT_SECONDS,
                )
                return
            await asyncio.sleep(self.FINALIZE_POLL_SECONDS)

    @staticmethod
    def _unfinalized_turn_error(turn_id: str | None, turn) -> str:
        """Describe a drained stream whose turn row never ended."""
        if turn is None:
            return f"streamed durable turn did not finalize: no turn log for {turn_id}"
        detail = f"status={getattr(turn, 'status', None) or 'unknown'}"
        error_text = getattr(turn, "error_text", None)
        if error_text:
            detail += f"; error: {error_text}"
        return f"streamed durable turn did not finalize ({detail})"

    @staticmethod
    def _is_context_overflow(error: str) -> bool:
        normalized = (error or "").lower()
        return (
            "exceeds the context window" in normalized
            or "input is too long for requested model" in normalized
            or "maximum context length" in normalized
        )

    @staticmethod
    def _is_terminal(exc: Exception) -> bool:
        if isinstance(exc, (ValueError, TypeError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return 400 <= exc.response.status_code < 500 and exc.response.status_code not in (408, 409, 425, 429)
        status_code = getattr(exc, "status_code", None)
        return isinstance(status_code, int) and 400 <= status_code < 500 and status_code not in (408, 409, 425, 429)

    async def _emit(self, record: DeliveryRecord) -> None:
        event = {
            "_type": "inter_bot_delivery",
            "delivery_id": record.id,
            "status": record.status,
            "sender_bot_id": record.sender_bot_id,
            "target_bot_id": record.target_bot_id,
            "bot_id": record.sender_bot_id,
            "user_id": self.service.config.DEFAULT_USER,
            "turn_id": record.turn_id,
            "target_turn_id": record.target_turn_id,
            "delivery_mode": record.delivery_mode,
            "reply_mode": record.reply_mode,
            "session_policy": record.session_policy,
            "reset_status": record.reset_status,
            "reset_reason": record.reset_reason,
            "old_session_id": record.old_session_id,
            "new_session_id": record.new_session_id,
            "reset_at": record.reset_at.isoformat() if record.reset_at else None,
            "overflow_recovery_count": record.overflow_recovery_count,
            "attempt_count": record.attempt_count,
            "next_retry_at": record.next_retry_at.isoformat() if record.next_retry_at else None,
            "last_error": record.last_error,
            "project_id": record.project_id,
            "task_id": record.task_id,
            "message_kind": record.message_kind,
            "retry_of_delivery_id": (record.metadata.get("manual_retry_of")
                                     or record.metadata.get("restart_retry_of")),
            "ts": time.time(),
        }
        if record.metadata.get("prompt_schedule"):
            event["user_id"] = record.author_entity_id
            event["bot_id"] = record.target_bot_id
            event["prompt_schedule"] = record.metadata["prompt_schedule"]
        chat_user = None
        if record.metadata.get("ordinary_chat"):
            # Ordinary queued sends belong to the selected user, not the
            # service's default user or the inter-bot sender namespace. A
            # system-authored restart child (TASK-1231B) belongs to the user
            # whose request it carries.
            if record.author_entity_type == "user":
                chat_user = record.author_entity_id
            else:
                original = retry_original_author(record)
                if original is not None and original.entity_type == "user":
                    chat_user = original.entity_id
        if chat_user:
            event.update(
                bot_id=record.target_bot_id, user_id=chat_user,
                ordinary_chat=True, user_message_id=record.user_message_id,
                session_id=record.metadata["ordinary_chat"].get("session_id"),
            )
        subscriber = getattr(self.service, "_redis_subscriber", None)
        if record.metadata.get("maintenance_user_queue"):
            event.update(bot_id=record.target_bot_id, user_id=record.author_entity_id)
        if subscriber is not None:
            try:
                if record.metadata.get("maintenance_user_queue"):
                    state = {QUEUED: "queued", DISPATCHING: "started", DELIVERED: "completed",
                             FAILED: "refused", "CANCELLED": "cancelled"}.get(record.status)
                    if state:
                        await subscriber.publish_tool_event(record.target_bot_id, record.author_entity_id, {
                            "_type": "steer_lifecycle", "message_id": record.user_message_id,
                            "bot_id": record.target_bot_id, "user_id": record.author_entity_id, "state": state,
                        })
                await subscriber.publish_tool_event(
                    event["bot_id"],
                    event["user_id"],
                    event,
                )
            except Exception:
                logger.warning("could not publish inter-bot lifecycle event", exc_info=True)
