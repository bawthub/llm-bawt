"""Terminal lifecycle coordinator for one streaming chat turn."""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from .chat_stream_worker import put_queue_item_threadsafe
from .logging import get_service_logger
from .turn_stream_context import TurnStreamContext

log = get_service_logger(__name__)


class TurnStreamFinalizer:
    """Persist and publish the terminal state for a streaming turn."""

    def __init__(
        self,
        ctx: TurnStreamContext,
        *,
        publish_event_direct: Callable[[dict[str, Any]], Any],
        enrich_attachment_refs: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    ) -> None:
        self.ctx = ctx
        self._publish_event_direct = publish_event_direct
        self._enrich_attachment_refs = enrich_attachment_refs

    def _turn_was_aborted(self) -> bool:
        try:
            current_turn = self.ctx.svc._turn_log_store.get_turn(self.ctx.turn_log_id)
            return current_turn is not None and current_turn.status in ("aborted", "cancelling")
        except Exception:
            return False

    def _turn_has_upstream_error(self) -> tuple[bool, str | None]:
        """TASK-714: was the turn already marked ``status="error"`` by an
        upstream-error signal from the bridge (agent path) or the finalize
        exception handler below (chat path)?

        The claude-code bridge ERROR event handler at
        ``chat_streaming_bridge.py:877`` already writes ``status="error"`` on
        the turn log BEFORE this finalize runs. Without this check, the
        subsequent ``_finalize_turn`` + ``turn_complete`` emit at line 253
        would clobber that honest error back to ``status="completed"`` /
        ``end_reason="stop"`` if any partial visible text was accumulated —
        the exact spec bug.

        Returns ``(had_error, error_text or None)`` so the classifier can also
        surface a stable ``end_reason`` and forward the error text on the wire.
        """
        try:
            current_turn = self.ctx.svc._turn_log_store.get_turn(self.ctx.turn_log_id)
            if current_turn is not None and current_turn.status == "error":
                return True, getattr(current_turn, "error_text", None)
        except Exception:
            pass
        return False, None

    def _usage_so_far(self) -> dict | None:
        ctx = self.ctx
        try:
            return ctx.svc._resolve_turn_token_usage(ctx.llm_bawt, ctx.token_usage_holder[0])
        except Exception:
            return ctx.token_usage_holder[0] if isinstance(ctx.token_usage_holder[0], dict) else None

    def finalize(self, *, prepared_messages: list[Any]) -> None:
        """No logger, publisher or persistence failure may strand the HTTP stream."""
        ctx = self.ctx
        try:
            self._finalize(prepared_messages=prepared_messages)
        finally:
            try:
                put_queue_item_threadsafe(ctx.loop, ctx.chunk_queue, None)
            finally:
                from .turn_execution import turn_executions
                turn_executions.remove(ctx.turn_log_id)
                if ctx.is_agent_backend:
                    ctx.done_event.set()
                else:
                    ctx.svc._end_generation(ctx.cancel_event, ctx.done_event, ctx.bot_id)

    def _finalize(self, *, prepared_messages: list[Any]) -> None:
        """Finalize persistence and publish the authoritative terminal outcome."""
        ctx = self.ctx
        svc = ctx.svc
        end_time = ctx.timing_holder[1] or time.time()
        start_time = ctx.timing_holder[0] or end_time
        elapsed_ms = (end_time - start_time) * 1000
        externally_aborted = ctx.cancelled_holder[0] or self._turn_was_aborted()
        if externally_aborted:
            ctx.cancelled_holder[0] = True

        # TASK-714: honor a prior upstream-error status BEFORE persistence so
        # we don't clobber it back to "completed" via _finalize_turn. Ordering:
        # aborted > upstream_error > (success/timeout branches below).
        upstream_error, upstream_error_text = (False, None)
        if not externally_aborted:
            upstream_error, upstream_error_text = self._turn_has_upstream_error()

        # A live answer continues this very turn; only an unanswered question
        # should mark the terminal as a deferred question.
        if ctx.question_id_holder[0]:
            row = svc._pending_question_store.get(ctx.question_id_holder[0])
            # Timeout clears live_deadline before any later continuation answer.
            # Only a live result clears the question terminal marker.
            if row is not None and row.live_deadline and row.status in ("answered", "skipped"):
                ctx.question_id_holder[0] = None

        # Wrap finalization so the sentinel, turn_complete event, and generation
        # cleanup still fire when persistence raises (for example, DB failure).
        try:
            if externally_aborted:
                # /v1/chat/abort owns this terminal state. Keep any partial reply
                # as a truncated assistant message without flipping the turn to ok.
                if ctx.full_response_holder[0]:
                    svc._finalize_turn(
                        llm_bawt=ctx.llm_bawt,
                        turn_id=ctx.turn_log_id,
                        response_text=ctx.full_response_holder[0],
                        tool_context=ctx.tool_context_holder[0],
                        tool_call_details=ctx.tool_call_details_holder,
                        prepared_messages=prepared_messages,
                        user_prompt=ctx.user_prompt,
                        model=ctx.model_alias,
                        bot_id=ctx.bot_id,
                        user_id=ctx.user_id,
                        elapsed_ms=elapsed_ms,
                        stream=True,
                        animation=ctx.animation_holder[0],
                        token_usage=ctx.token_usage_holder[0],
                        attachments=ctx.agent_attachments_holder or None,
                        reasoning=ctx.reasoning_holder[0] or None,
                        status="aborted",
                        end_reason="aborted",
                        assistant_message_id=ctx.assistant_message_id,
                    )
                else:
                    svc._update_turn_log(
                        turn_id=ctx.turn_log_id,
                        latency_ms=elapsed_ms,
                        tool_calls=ctx.tool_call_details_holder or None,
                        status="aborted",
                        end_reason="aborted",
                    )
            elif upstream_error:
                # TASK-714: an upstream error was already recorded on this turn
                # (bridge ERROR handler, or the finalize-exception handler below
                # on a prior partial). The honest terminal is "error".
                #
                # TASK-790: a failed turn must never end SILENT. Previously this
                # branch only updated the turn log — no assistant row was
                # persisted, so on reload the reply vanished entirely (user
                # message with no answer, no error, no retry hint — the exact
                # "silent death" Loopy hit on 2026-08-19). Now: append a visible
                # failure marker to any partial text and commit it to history as
                # the assistant message via _finalize_turn (same mechanism the
                # abort path uses to preserve partials), keeping status="error"
                # / end_reason="upstream_error". The mutated
                # full_response_holder also feeds the turn_complete wire's
                # response_text below, so live clients render the same marker
                # without a history refetch.
                _partial = ctx.full_response_holder[0] or ""
                # Guard: the legacy openclaw path (chat_streaming_bridge.py,
                # TASK-202) already appends its own visible "⚠️ … error" block
                # to the text before ERROR terminates — don't double-mark.
                if "⚠️" in _partial[-600:]:
                    _final_text = _partial
                else:
                    _err_note = (upstream_error_text or "upstream backend error").strip()
                    _marker = (
                        f"⚠️ **Turn failed** — {_err_note}\n\n"
                        "_This response ended with an error. Work already performed "
                        "is not rolled back; a transport failure does not confirm "
                        "the worker stopped. Check active work before retrying._"
                    )
                    _final_text = (
                        f"{_partial.rstrip()}\n\n{_marker}" if _partial.strip() else _marker
                    )
                ctx.full_response_holder[0] = _final_text
                svc._finalize_turn(
                    llm_bawt=ctx.llm_bawt,
                    turn_id=ctx.turn_log_id,
                    response_text=_final_text,
                    tool_context=ctx.tool_context_holder[0],
                    tool_call_details=ctx.tool_call_details_holder,
                    prepared_messages=prepared_messages,
                    user_prompt=ctx.user_prompt,
                    model=ctx.model_alias,
                    bot_id=ctx.bot_id,
                    user_id=ctx.user_id,
                    elapsed_ms=elapsed_ms,
                    stream=True,
                    animation=ctx.animation_holder[0],
                    token_usage=ctx.token_usage_holder[0] or self._usage_so_far(),
                    attachments=ctx.agent_attachments_holder or None,
                    reasoning=ctx.reasoning_holder[0] or None,
                    # Keep the honest terminal — _finalize_turn passes status
                    # through to the turn log; error_text set upstream survives
                    # (never passed here, so never overwritten).
                    status="error",
                    end_reason="upstream_error",
                    assistant_message_id=ctx.assistant_message_id,
                )
            elif ctx.full_response_holder[0]:
                # Commit the real outcome on the first terminal write; later
                # enrichment cannot rewrite a provisional "stop" outcome.
                deferred_reason = (
                    "question" if ctx.question_id_holder[0] else
                    "approval" if ctx.approval_id_holder[0] else
                    "approval_persist_failed" if ctx.approval_persist_failed_holder[0] else "stop"
                )
                svc._finalize_turn(
                    llm_bawt=ctx.llm_bawt,
                    turn_id=ctx.turn_log_id,
                    response_text=ctx.full_response_holder[0],
                    tool_context=ctx.tool_context_holder[0],
                    tool_call_details=ctx.tool_call_details_holder,
                    prepared_messages=prepared_messages,
                    user_prompt=ctx.user_prompt,
                    model=ctx.model_alias,
                    bot_id=ctx.bot_id,
                    user_id=ctx.user_id,
                    elapsed_ms=elapsed_ms,
                    stream=True,
                    animation=ctx.animation_holder[0],
                    token_usage=ctx.token_usage_holder[0],
                    attachments=ctx.agent_attachments_holder or None,
                    reasoning=ctx.reasoning_holder[0] or None,
                    assistant_message_id=ctx.assistant_message_id,
                    end_reason=deferred_reason,
                )
            else:
                # Persist any tool details captured before a failed follow-up.
                svc._update_turn_log(
                    turn_id=ctx.turn_log_id,
                    status="timeout",
                    latency_ms=elapsed_ms,
                    tool_calls=ctx.tool_call_details_holder or None,
                    token_usage=self._usage_so_far(),
                )
        except Exception as fin_err:
            upstream_error, upstream_error_text = True, f"finalize_error: {fin_err}"
            try:
                log.error("Finalization failed (turn %s): %s", ctx.turn_log_id, fin_err)
            except Exception:
                pass  # Diagnostics must not prevent terminal publication.
            try:
                svc._update_turn_log(
                    turn_id=ctx.turn_log_id,
                    status="error",
                    latency_ms=elapsed_ms,
                    response_text=ctx.full_response_holder[0] or None,
                    error_text=f"finalize_error: {fin_err}",
                    token_usage=self._usage_so_far(),
                )
            except Exception:
                pass

        # TASK-714 accounting fix: the classifier now considers upstream_error
        # ALONGSIDE aborted + full_response presence. Precedence (spec-locked):
        # cancelled/aborted > upstream_error > completed(has text) > timeout(empty).
        # Without this, an upstream-error turn with partial visible output would
        # persist status="completed"/end_reason="stop" — a lie that breaks
        # /v1/turn-logs review, task-context aggregation, and downstream
        # observability. See _turn_has_upstream_error() docstring.
        if ctx.cancelled_holder[0]:
            status = "cancelled"
        elif upstream_error:
            status = "error"
        elif ctx.full_response_holder[0]:
            status = "completed"
        else:
            status = "timeout"

        question_id = ctx.question_id_holder[0]
        approval_id = ctx.approval_id_holder[0]
        approval_persist_failed = ctx.approval_persist_failed_holder[0]
        if ctx.cancelled_holder[0]:
            end_reason = "aborted"
        elif question_id:
            end_reason = "question"
        elif approval_id:
            end_reason = "approval"
        elif approval_persist_failed:
            end_reason = "approval_persist_failed"
        elif upstream_error:
            # TASK-714: new terminal value for the turn_complete wire — additive
            # only; consumers that don't switch on it fall through to their
            # default handling (typically "error state" bucket).
            end_reason = "upstream_error"
        elif status == "timeout":
            end_reason = "error"
        else:
            end_reason = "stop"

        try:
            svc._turn_log_store.update_turn(
                turn_id=ctx.turn_log_id,
                end_reason=end_reason,
                question_id=question_id,
                error_text=(
                    json.dumps(approval_persist_failed)
                    if approval_persist_failed else None
                ),
                tts_scrubbed=ctx.tts_scrub,
            )
        except Exception as end_reason_err:
            log.debug(
                "update_turn end_reason failed for %s: %s",
                ctx.turn_log_id,
                end_reason_err,
            )

        # Cancellation may win while response persistence is running. Publish
        # the outcome that won the row lock, never a stale local success guess.
        try:
            current = svc._turn_log_store.get_turn(ctx.turn_log_id)
        except Exception:
            current = None
        if current is not None and current.status in {
            "aborted", "cancelled", "ok", "completed", "error", "timeout",
        }:
            status = {"aborted": "cancelled", "ok": "completed"}.get(current.status, current.status)
            end_reason = current.end_reason or end_reason
            if status == "error":
                upstream_error_text = current.error_text or upstream_error_text
                upstream_error = True

        # The TTS consumer closes input on turn_complete, so commit the final
        # scrubbed tail before publishing that terminal event.
        if ctx.tts_scrubber is not None and status not in ("cancelled", "aborted"):
            try:
                tts_tail = ctx.tts_scrubber.flush()
            except Exception:
                tts_tail = ""  # Optional speech cleanup cannot strand text clients.
            if tts_tail:
                tts_tail_future = self._publish_event_direct({
                    "_type": "tts_delta",
                    "turn_id": ctx.turn_log_id,
                    "bot_id": ctx.bot_id,
                    "user_id": ctx.user_id,
                    "delta": tts_tail,
                    "ts": time.time(),
                })
                if tts_tail_future is not None:
                    try:
                        tts_tail_future.result(timeout=5)
                    except Exception as order_err:
                        log.warning(
                            "Final tts_delta publish did not complete before "
                            "turn_complete for turn %s: %s",
                            ctx.turn_log_id,
                            order_err,
                        )

        try:
            changed_files = svc._turn_log_store.changed_files_summary(ctx.turn_log_id)
        except Exception:
            changed_files = None

        try:
            completed_attachments = (
                self._enrich_attachment_refs(ctx.agent_attachments_holder)
                if ctx.agent_attachments_holder else None
            )
        except Exception as attachment_err:
            completed_attachments = None
            log.warning("turn_complete attachment enrichment failed: %s", attachment_err)

        # TASK-709: session id for external-turn adoption. Read from the
        # request-local thread binding — the same source that populated
        # turn_start's session_id in chat_streaming.py. Falls back to null on
        # non-agent turns or when the turn was never thread-bound.
        _sid: str | None = None
        try:
            _binding = getattr(ctx, "thread_binding", None)
            if isinstance(_binding, dict):
                _rsid = str(_binding.get("thread_session_id") or "").strip()
                _sid = _rsid or None
        except Exception:
            _sid = None

        # TASK-779 + TASK-714: server-authoritative final text on the STREAMING
        # turn finalize path (symmetric with background_service.py's nonstream
        # emit). Streaming turns normally commit from the client-side partial
        # (accumulated text_delta), but if enough deltas drop the partial can
        # be empty at finalize and the reply vanishes. Carrying the full
        # server-side text lets commitServerOriginatedTurn fall back to the
        # same bytes the DB persisted. Additive optional field.
        #
        # TASK-714 extension: also carry the text on upstream-error terminals.
        # The claude-code bridge appends its visible error marker
        # ("⚠️ openclaw bridge error\n\n```\n...\n```") to full_response_holder
        # before ERROR is raised — preserving it here lets the client-side
        # assistant bubble still render the honest failure if the partial
        # buffer is empty. Cancellation carries the preserved partial too;
        # timeout without content remains None.
        _response_text = (
            ctx.full_response_holder[0]
            if status in ("completed", "error", "cancelled")
            else None
        )
        self._publish_event_direct({
            "_type": "turn_complete",
            "turn_id": ctx.turn_log_id,
            # TASK-784: symmetric with turn_start / background_service.py nonstream
            # emits — carry the assistant_message_id on the streaming success path
            # too, so commitServerOriginatedTurn on the client can adopt the reply
            # even if it never saw turn_start (mid-turn page join, HMR reconnect
            # after the run began). Additive optional field; consumers that ignore
            # it keep working.
            "assistant_message_id": ctx.assistant_message_id,
            "bot_id": ctx.bot_id,
            "user_id": ctx.user_id,
            # TASK-709: match turn_start's session id (see block above).
            "session_id": _sid,
            "status": status,
            "end_reason": end_reason,
            # TASK-790: raw upstream failure reason (e.g. "claude-code error:
            # No SDK messages for 300.0s — CLI may be hung"). Additive; only
            # populated on error terminals so clients can render WHY the turn
            # died and offer a retry, instead of a silent mid-turn stop.
            "error_text": upstream_error_text if upstream_error else None,
            "question_id": question_id,
            "approval_id": approval_id,
            "approval_persist_failed": approval_persist_failed,
            "animation": ctx.animation_holder[0],
            "token_usage": self._usage_so_far(),
            "changed_files": changed_files,
            "attachments": completed_attachments,
            "response_text": _response_text,
            "model": ctx.model_alias,
            "ts": time.time(),
        })
