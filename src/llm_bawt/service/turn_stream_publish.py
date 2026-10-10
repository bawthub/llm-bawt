"""TASK-622: per-turn Redis publish + turn-persistence helpers.

Extracted verbatim from ``chat_streaming.chat_completion_stream``. Each method
begins with a preamble that rebinds the captured closure variables from the
shared :class:`TurnStreamContext` (and, where the original body used ``self`` =
the BackgroundService, rebinds ``self = ctx.svc``) so the body below is byte-for-
byte identical to the original closure.
"""
from __future__ import annotations

import asyncio
import time

from ..approval_policies import ApprovalPersistError
from ..media.assets import MediaAssetStore
from ..media.serializers import enrich_attachments_for_messages
from .logging import get_service_logger
from .turn_stream_context import (
    _APPROVAL_PERSIST_BACKOFF_S,
    _APPROVAL_PERSIST_MAX_ATTEMPTS,
)

log = get_service_logger(__name__)


class TurnStreamPublishMixin:
    """Redis event publishing + durable approval/question persistence."""

    async def _publish_unified(self, event_dict):
        ctx = self.ctx
        _redis_sub = ctx._redis_sub
        bot_id = ctx.bot_id
        user_id = ctx.user_id
        """Single source of truth for publishing to the unified SSE stream.

        Race-free when awaited directly from the async generator (loop
        thread). Worker-thread callers must NOT call this directly —
        they go through _publish_event_direct, which marshals onto the
        loop with run_coroutine_threadsafe.
        """
        if not _redis_sub:
            return
        try:
            await _redis_sub.publish_tool_event(bot_id, user_id, event_dict)
        except Exception as pub_err:
            log.debug("Unified event publish failed: %s", pub_err)

    def _publish_event_direct(self, event_dict):
        ctx = self.ctx
        _publish_unified = self._publish_unified
        _redis_sub = ctx._redis_sub
        _tool_event_coordinator = ctx._tool_event_coordinator
        loop = ctx.loop
        """Publish to the unified stream FROM THE WORKER THREAD.

        Thin cross-thread wrapper over _publish_unified. ``run_coroutine_
        threadsafe`` is correct ONLY when called from a thread other than
        the loop's own — calling it from the loop thread schedules a future
        that races request teardown and is silently dropped (this was the
        TASK-305 approval-card/bell/gold-lock bug). Loop-thread callers must
        ``await _publish_unified(...)`` instead.

        Returns the cross-thread future so callers that require strict event
        ordering can wait for the Redis write before publishing a terminal
        event such as ``turn_complete``.
        """
        if event_dict.get("_type") == "tool_event":
            if event_dict.get("event") == "tool_start":
                event_dict = _tool_event_coordinator.start(event_dict)
            elif event_dict.get("event") == "tool_end":
                event_dict = _tool_event_coordinator.end(event_dict)
        if not _redis_sub:
            return None
        try:
            return asyncio.run_coroutine_threadsafe(
                _publish_unified(event_dict),
                loop,
            )
        except Exception as pub_err:
            log.debug("Direct event publish failed: %s", pub_err)
            return None

    def _enrich_attachment_refs(self, refs):
        """Expand tiny media refs into the canonical live/history envelope."""
        ctx = self.ctx
        svc = ctx.svc
        shell = [{"attachments": list(refs)}]
        enrich_attachments_for_messages(
            shell, MediaAssetStore(svc.config)
        )
        return shell[0].get("attachments") or []

    def _enrich_turn_start_attachments(self):
        return self._enrich_attachment_refs(self.ctx.attachments_to_persist)

    def _persist_publish_approval(self, chunk: dict) -> None:
        ctx = self.ctx
        _publish_event_direct = self._publish_event_direct
        _approval_handled = ctx._approval_handled
        approval_id_holder = ctx.approval_id_holder
        approval_persist_failed_holder = ctx.approval_persist_failed_holder
        bot_id = ctx.bot_id
        trigger_message_id = ctx.trigger_message_id
        turn_log_id = ctx.turn_log_id
        user_id = ctx.user_id
        self = ctx.svc
        """Durably record + surface one approval_required chunk (worker thread).

        Synchronous DB write with bounded retry; on success publishes the
        live unified ``tool_approval_required`` event, on a confirmed
        failure publishes ``tool_approval_persist_failed`` so the error
        reaches the user instead of vanishing.
        """
        req_id = chunk.get("tool_use_id") or ""
        if not req_id or req_id in _approval_handled:
            return
        _approval_handled.add(req_id)
        tool_args = chunk.get("arguments") or {}
        store = getattr(self, "_tool_approval_policy_store", None)

        persist_failure: dict | None = None
        if store is not None:
            committed = False
            last_err: Exception | None = None
            for _attempt in range(_APPROVAL_PERSIST_MAX_ATTEMPTS):
                try:
                    store.record_request(
                        request_id=req_id,
                        bot_id=bot_id,
                        user_id=user_id,
                        turn_id=turn_log_id,
                        backend=chunk.get("provider") or "claude-code",
                        tool_name=chunk.get("tool_name") or "",
                        tool_arguments=tool_args if isinstance(tool_args, dict) else {"value": tool_args},
                        subject=chunk.get("subject") or "",
                        grant_key=chunk.get("grant_key") or "",
                        policy_id=chunk.get("policy_id"),
                        severity=chunk.get("severity") or "medium",
                        prompt=chunk.get("prompt") or "",
                        trigger_message_id=trigger_message_id,
                        session_key=chunk.get("session_key") or None,
                        session_id=getattr(ctx.request, "session_id", None),
                    )
                    committed = True
                    break
                except ApprovalPersistError as _persist_err:
                    last_err = _persist_err
                    log.warning(
                        "approval persist attempt %d/%d failed id=%s: %s",
                        _attempt + 1, _APPROVAL_PERSIST_MAX_ATTEMPTS,
                        req_id, _persist_err,
                    )
                    if _attempt + 1 < _APPROVAL_PERSIST_MAX_ATTEMPTS:
                        time.sleep(_APPROVAL_PERSIST_BACKOFF_S * (_attempt + 1))
            if committed:
                approval_id_holder[0] = req_id
            else:
                persist_failure = {
                    "kind": "side_effect_failed",
                    "effect": "approval_request_persist",
                    "request_id": req_id,
                    "tool_name": chunk.get("tool_name") or "",
                    "reachable_to_user": False,
                    "error_class": type(last_err).__name__ if last_err else "ApprovalPersistError",
                    "detail": str(last_err) if last_err else "unknown",
                    "attempts": _APPROVAL_PERSIST_MAX_ATTEMPTS,
                }
        else:
            persist_failure = {
                "kind": "side_effect_failed",
                "effect": "approval_request_persist",
                "request_id": req_id,
                "tool_name": chunk.get("tool_name") or "",
                "reachable_to_user": False,
                "error_class": "NoApprovalStore",
                "detail": "approval policy store is not configured on this service",
                "attempts": 0,
            }

        if persist_failure is None:
            _publish_event_direct({
                "_type": "tool_approval_required",
                "turn_id": turn_log_id,
                "trigger_message_id": trigger_message_id,
                "bot_id": bot_id,
                "user_id": user_id,
                "request_id": req_id,
                "tool_name": chunk.get("tool_name", ""),
                "arguments": tool_args,
                "subject": chunk.get("subject", ""),
                "label": chunk.get("label", ""),
                "prompt": chunk.get("prompt", ""),
                "severity": chunk.get("severity", "medium"),
                "policy_id": chunk.get("policy_id"),
                "session_key": chunk.get("session_key", ""),
                "provider": chunk.get("provider", ""),
                "ts": time.time(),
            })
        else:
            approval_persist_failed_holder[0] = persist_failure
            log.error(
                "approval persist FAILED id=%s tool=%s — surfacing failure to user: %s",
                req_id, persist_failure["tool_name"], persist_failure["detail"],
            )
            _publish_event_direct({
                "_type": "tool_approval_persist_failed",
                "turn_id": turn_log_id,
                "trigger_message_id": trigger_message_id,
                "bot_id": bot_id,
                "user_id": user_id,
                "request_id": req_id,
                "tool_name": persist_failure["tool_name"],
                "detail": persist_failure["detail"],
                "ts": time.time(),
            })

    def _persist_publish_await(self, chunk: dict) -> None:
        ctx = self.ctx
        _publish_event_direct = self._publish_event_direct
        _await_handled = ctx._await_handled
        bot_id = ctx.bot_id
        question_id_holder = ctx.question_id_holder
        trigger_message_id = ctx.trigger_message_id
        turn_log_id = ctx.turn_log_id
        user_id = ctx.user_id
        self = ctx.svc
        """Durably record + surface one AskUserQuestion (worker thread).

        Mirrors _persist_publish_approval: persists the pending question and
        publishes the live unified ``tool_await_result`` event from a thread
        that survives async-gen teardown, and sets ``question_id_holder`` so
        the turn finalizes with end_reason="question" even if the async
        generator never drains the chunk.
        """
        tool_use_id = chunk.get("tool_use_id") or ""
        if not tool_use_id or tool_use_id in _await_handled:
            return
        _await_handled.add(tool_use_id)
        tool_args = chunk.get("arguments") or {}
        origin_harness = (chunk.get("provider") or "claude") or "claude"
        from datetime import datetime
        deadline = chunk.get("live_deadline")
        live_deadline = datetime.fromisoformat(deadline) if deadline and origin_harness == "claude-code" else None
        # Set BEFORE the persist so the turn ends as a question regardless of
        # a transient store failure (matches the prior inline semantics).
        question_id_holder[0] = tool_use_id
        try:
            # A turn that re-asks supersedes its earlier pending question so
            # only the latest stays answerable.
            self._pending_question_store.supersede_awaiting_for_turn(
                turn_log_id, keep=tool_use_id,
            )
            self._pending_question_store.upsert_awaiting(
                tool_use_id=tool_use_id,
                bot_id=bot_id,
                user_id=user_id,
                turn_id=turn_log_id,
                arguments=tool_args if isinstance(tool_args, dict) else {"value": tool_args},
                tool_name=chunk.get("tool_name") or "AskUserQuestion",
                trigger_message_id=trigger_message_id,
                session_key=chunk.get("session_key") or None,
                origin_harness=origin_harness,
                live_deadline=live_deadline,
            )
        except Exception as _persist_err:
            log.warning(
                "Failed to persist pending question tool_use_id=%s: %s",
                tool_use_id, _persist_err,
            )
        _publish_event_direct({
            "_type": "tool_await_result",
            "turn_id": turn_log_id,
            "trigger_message_id": trigger_message_id,
            "bot_id": bot_id,
            "user_id": user_id,
            "tool_use_id": tool_use_id,
            "tool_name": chunk.get("tool_name", ""),
            "arguments": tool_args,
            "session_key": chunk.get("session_key", ""),
            "provider": chunk.get("provider", ""),
            "live_deadline": deadline,
            "ts": time.time(),
        })
