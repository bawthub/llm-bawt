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
from .schemas import ChatCompletionRequest
from .turn_logs import DeliveryTargetBusy

logger = logging.getLogger(__name__)


class InterBotDeliveryDispatcher:
    """Drain one FIFO callback per target through the normal chat path.

    The event is the fast path (enqueue and terminal turn transitions call
    ``wake``); the timed sweep is only restart/missed-signal recovery.
    """

    def __init__(self, service: Any, *, recovery_interval_seconds: float = 15.0):
        self.service = service
        self.store = InterBotDeliveryStore(service.config)
        self.recovery_interval_seconds = max(1.0, recovery_interval_seconds)
        self._wake_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._leader_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._target_tasks: dict[str, asyncio.Task] = {}
        self.claim_owner = f"dispatcher-{uuid.uuid4().hex}"
        self._dispatcher_lock = None
        self._subscriber_gap_logged = False

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
                await asyncio.sleep(1)

    async def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        tasks = [task for task in self._target_tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._task:
            await asyncio.gather(self._task, return_exceptions=True)
        if self._leader_task:
            await asyncio.gather(self._leader_task, return_exceptions=True)
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

    async def run(self) -> None:
        # Belt-and-suspenders around the outer leadership guard: transient
        # SQLAlchemy pool blips (e.g. InvalidatePoolError from a DB restart)
        # showed up in prod as 5 crashes in 25s because the outer guard tore
        # down and rebuilt the whole run() task each time. Guarding the sweep
        # in-place lets one task instance ride through pool recovery with
        # exponential backoff instead of hot-looping.
        try:
            self.store.recover_expired()
        except Exception:
            logger.exception("initial recover_expired failed; continuing")
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
                self.store.recover_expired()
                self._schedule_eligible_targets()
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
        await asyncio.gather(*self._target_tasks.values(), return_exceptions=True)

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
        self._wake_event.set()

    async def _drain_target(self, target_bot_id: str) -> None:
        from ..bots import BotManager

        bot = BotManager(self.service.config).get_bot(target_bot_id)
        backend = (getattr(bot, "agent_backend", None) or "").strip()
        record = self.store.claim_next(
            target_bot_id,
            claim_owner=self.claim_owner,
            steer_capable=(backend == "claude-code"),
        )
        if record is None:
            return
        await self._emit(record)
        if record.status == "STEERING":
            await self._steer(record)
        else:
            await self._dispatch(record)

    async def _steer(self, record: DeliveryRecord) -> None:
        from .routes.chat import ChatSteerRequest, steer_active_turn

        claim_token = record.claim_token or ""
        payload = self.store.payload(record.id) or {}
        message = str((payload.get("messages") or [{}])[-1].get("content") or record.message)
        user_id = str(payload.get("user") or getattr(self.service.config, "DEFAULT_USER", "nick"))
        heartbeat = asyncio.create_task(
            self._heartbeat(record.id, claim_token),
            name=f"inter-bot-steer-heartbeat:{record.id}",
        )
        try:
            response = await steer_active_turn(
                self.service,
                ChatSteerRequest(
                    turn_id=record.target_turn_id or "",
                    message=message,
                    message_id=record.user_message_id,
                    bot_id=record.target_bot_id,
                    user_id=user_id,
                    # Bot deliveries never kill the target's running tools
                    # (TASK-934): queue for the next tool boundary.
                    priority="next",
                ),
                steer_request_id=(
                    f"steer_delivery_{record.id.removeprefix('delivery-')}"
                ),
                author=record.author,
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
        except Exception as exc:
            error = str(getattr(exc, "detail", None) or exc or exc.__class__.__name__)
            current = self.store.get(record.id) or record
            accepted = current.transport_accepted_at is not None
            failure_action = self._steer_failure_action(error, accepted=accepted)

            if failure_action == "fallback":
                # The cached deterministic steer RPC proved it did not reach the
                # old run. Clear that stale target exactly once; the next locked
                # claim will steer whichever run is current or reserve one idle
                # fallback turn. The failed steer probe is not a delivery attempt.
                queued = self.store.requeue(
                    record.id,
                    claim_token,
                    error,
                    delay_seconds=1,
                    reject_steer_target=True,
                    refund_attempt=True,
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
                # target turn + deterministic RPC ID. Replaying repairs message
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
            turn = self.service._turn_log_store.get_turn(record.turn_id)
            if turn is None or turn.ended_at is None:
                raise RuntimeError("streamed durable turn did not finalize")
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
                await self._emit(delivered)
        except DeliveryTargetBusy as exc:
            queued = self.store.requeue(record.id, claim_token, str(exc), delay_seconds=0)
            if queued:
                await self._emit(queued)
        except Exception as exc:  # classification is intentionally centralized
            error = str(exc) or exc.__class__.__name__
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
                    failed = self.store.fail_claim(record.id, claim_token, error)
                    if failed:
                        await self._emit(failed)
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
                    await self._emit(delivered)
            elif (
                self._is_terminal(exc)
                or current.attempt_count >= current.max_attempts
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

    async def _heartbeat(self, delivery_id: str, claim_token: str) -> None:
        while True:
            await asyncio.sleep(60)
            if not self.store.renew_lease(delivery_id, claim_token):
                return

    @staticmethod
    def _steer_failure_action(error: str, *, accepted: bool) -> str:
        """Classify one deterministic steer failure without guessing acceptance.

        Once transport acceptance is recorded, no failure may convert the
        delivery into a fresh turn; retries must replay the same steer RPC and
        repair persistence. Before acceptance, a definitive bridge rejection may
        clear stale steer intent, while missing run metadata only waits.
        """
        if accepted:
            return "retry_same_steer"
        if error == "Active bridge run is not ready":
            return "wait_ready"
        if error in {
            "no_active_run",
            "active_run_mismatch",
            "Turn is no longer active",
            "Turn not found",
        }:
            return "fallback"
        return "retry_same_steer"

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
            "user_id": getattr(self.service.config, "DEFAULT_USER", "nick"),
            "turn_id": record.turn_id,
            "target_turn_id": record.target_turn_id,
            "delivery_mode": record.delivery_mode,
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
            "ts": time.time(),
        }
        if record.metadata.get("prompt_schedule"):
            event["user_id"] = record.author_entity_id
            event["bot_id"] = record.target_bot_id
            event["prompt_schedule"] = record.metadata["prompt_schedule"]
        subscriber = getattr(self.service, "_redis_subscriber", None)
        if subscriber is not None:
            try:
                await subscriber.publish_tool_event(
                    event["bot_id"],
                    event["user_id"],
                    event,
                )
            except Exception:
                logger.warning("could not publish inter-bot lifecycle event", exc_info=True)
