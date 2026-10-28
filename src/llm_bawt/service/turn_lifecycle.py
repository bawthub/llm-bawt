"""Turn log persistence, generation cancellation, and SSE fan-out mixin.

Extracted from background_service.py — handles turn logging to DB,
generation cancel/done signalling, and in-memory SSE event buffering.
"""

from __future__ import annotations

import asyncio
import os
import threading
from datetime import datetime
from typing import Any

from ..clients.agent_backend_client import AgentBackendClient
from .logging import get_service_logger

log = get_service_logger(__name__)

# TASK-306 Section B: a bridge/agent turn that narrates this many or more
# (stripped) characters while recording ZERO tool calls is flagged as a
# possible fabricated work-claim. Below this it's treated as a trivial reply.
_MIN_NARRATION_CHARS_FOR_GUARD = 40


def _is_unbacked_work_claim(
    *,
    is_bridge_turn: bool,
    tool_call_details: list[dict] | None,
    response_text: str | None,
) -> bool:
    """TASK-306 Section B: True when a turn looks like a fabricated work-claim.

    A bridge/agent turn that finalized a non-trivial narration while recording
    ZERO tool calls is claiming work it may not have done. Pure predicate so the
    guard condition lives in exactly one place and is unit-testable without the
    full turn-lifecycle mixin. Only flags bridge turns: native-LLM chat replies
    legitimately produce text with no tools.
    """
    if not is_bridge_turn:
        return False
    if tool_call_details:
        return False
    return len((response_text or "").strip()) >= _MIN_NARRATION_CHARS_FOR_GUARD


def _message_to_dict(msg: Any) -> dict[str, Any]:
    """Normalize message objects for JSON logging."""
    if hasattr(msg, "to_dict"):
        value = msg.to_dict()
        if isinstance(value, dict):
            if not value.get("id") and value.get("db_id"):
                value = dict(value)
                value["id"] = value.get("db_id")
            return value
        return {"value": value}
    if hasattr(msg, "role"):
        return {
            "role": getattr(msg, "role", "unknown"),
            "content": getattr(msg, "content", ""),
            "timestamp": getattr(msg, "timestamp", 0),
            "id": getattr(msg, "db_id", None),
        }
    if isinstance(msg, dict):
        return msg
    return {"value": str(msg)}


def _normalize_tool_call_details(tool_calls: list[dict] | None) -> list[dict]:
    """Normalize tool-call details into request/response shape."""
    if not tool_calls:
        return []

    normalized: list[dict] = []
    for idx, item in enumerate(tool_calls, start=1):
        if not isinstance(item, dict):
            normalized.append({"index": idx, "name": "unknown", "arguments": {}, "result": str(item)})
            continue
        name = item.get("tool") or item.get("name") or "unknown"
        args = item.get("parameters") or item.get("arguments") or {}
        result = item.get("result") or item.get("response") or ""
        entry = {
                "index": idx,
                "iteration": item.get("iteration", 1),
                "name": name,
                "arguments": args if isinstance(args, dict) else {"raw": args},
                "result": result,
            }
        if item.get("call_id"):
            entry["call_id"] = item["call_id"]
        # SDK tool_use id + parent id. Without these on the persisted blob,
        # sub-agent nesting (child tool card under its Agent card) works live
        # via SSE but COLLAPSES on reload: the frontend's cardTuids set is built
        # from toolUseId, so an empty id makes every child render flat in the
        # timeline instead of inside its Agent card (TASK-344).
        if item.get("tool_use_id"):
            entry["tool_use_id"] = item["tool_use_id"]
        if item.get("parent_tool_use_id"):
            entry["parent_tool_use_id"] = item["parent_tool_use_id"]
        # Tool failure flag (SDK ToolResultBlock.is_error), threaded so a
        # recalled turn re-tints failed tool cards red without re-deriving
        # failure from result text. Only persist when true to keep the blob lean.
        if item.get("is_error"):
            entry["is_error"] = True
        # Interleave reconstruction: chars of assistant text before this tool.
        if item.get("text_offset") is not None:
            entry["text_offset"] = item["text_offset"]
        # Backend that produced this tool call ("codex", "claude-code",
        # "local"). Persisted so /v1/tool-calls recall can re-route to
        # the frontend's provider-aware renderer registry
        # (FileChangeBody / BashBody / …). Without this, codex turns
        # rehydrate from the DB without provider and the registry never
        # matches, falling back to GenericClaudeBody.
        if item.get("provider"):
            entry["provider"] = item["provider"]
        normalized.append(entry)
    return normalized


class TurnLifecycleMixin:
    @staticmethod
    def _resolve_turn_token_usage(llm_bawt: Any, explicit_usage: dict | None) -> dict | None:
        """Prefer streamed usage, then the agent backend's last partial/final result."""
        if isinstance(explicit_usage, dict) and explicit_usage:
            return explicit_usage
        client = getattr(llm_bawt, "client", None)
        get_token_usage = getattr(client, "get_token_usage", None)
        if isinstance(client, AgentBackendClient) and callable(get_token_usage):
            usage = get_token_usage()
            if isinstance(usage, dict) and usage:
                return usage
        return None

    """Mixin providing turn persistence and generation lifecycle for BackgroundService."""

    # ---- Generation cancellation ----

    async def _start_generation(self, bot_id: str | None = None) -> tuple[threading.Event, threading.Event]:
        """Start a new generation, cancelling and waiting for any in-progress one FOR THE SAME BOT.

        Returns:
            tuple of (cancel_event, done_event):
            - cancel_event: The generation should check this periodically and abort if set
            - done_event: The generation MUST set this when complete (in finally block)

        Generations are tracked per bot_id so different bots can run concurrently.
        Only a new request for the SAME bot cancels the previous generation for that bot.
        """
        loop = asyncio.get_event_loop()
        key = bot_id or "__global__"

        # Lazy-init per-bot dicts (set on BackgroundService.__init__ too, but
        # this is a mixin so we guard here).
        if not hasattr(self, "_gen_cancels"):
            self._gen_cancels: dict[str, threading.Event] = {}
            self._gen_dones: dict[str, threading.Event] = {}

        with self._cancel_lock:
            # Cancel any existing generation for this bot and wait for it
            prev_cancel = self._gen_cancels.get(key)
            if prev_cancel is not None:
                log.debug("Cancelling previous generation for bot %s", key)
                prev_cancel.set()

                prev_done = self._gen_dones.get(key)
                if prev_done is not None:
                    # Release lock while waiting to avoid deadlock
                    self._cancel_lock.release()
                    try:
                        await loop.run_in_executor(
                            None,
                            lambda: prev_done.wait(timeout=5.0)
                        )
                    finally:
                        self._cancel_lock.acquire()

            # Create new events for this generation
            cancel_event = threading.Event()
            done_event = threading.Event()
            self._gen_cancels[key] = cancel_event
            self._gen_dones[key] = done_event

            # Keep legacy single-slot fields in sync for any code that reads them
            self._current_generation_cancel = cancel_event
            self._generation_done = done_event
            return cancel_event, done_event

    def _end_generation(self, cancel_event: threading.Event, done_event: threading.Event, bot_id: str | None = None):
        """Mark a generation as complete."""
        done_event.set()
        key = bot_id or "__global__"

        # Some call paths (agent backends) intentionally bypass _start_generation
        # to allow concurrent requests; in that case the per-bot dicts may not
        # exist yet and there's nothing to clean up.
        if not hasattr(self, "_gen_cancels"):
            return

        with self._cancel_lock:
            if self._gen_cancels.get(key) is cancel_event:
                del self._gen_cancels[key]
                self._gen_dones.pop(key, None)
            # Keep legacy fields in sync
            if self._current_generation_cancel is cancel_event:
                self._current_generation_cancel = None
                self._generation_done = None

    # ---- Turn log persistence ----

    def _persist_turn_log(
        self,
        *,
        turn_id: str,
        request_id: str | None,
        path: str,
        stream: bool,
        model: str | None,
        bot_id: str,
        user_id: str,
        status: str,
        latency_ms: float | None,
        user_prompt: str,
        prepared_messages: list,
        response_text: str,
        tool_calls: list[dict] | None = None,
        error_text: str | None = None,
        agent_session_key: str | None = None,
        agent_request_id: str | None = None,
        trigger_message_id: str | None = None,
        assistant_message_id: str | None = None,
        parent_turn_id: str | None = None,
        request_extensions: dict | None = None,
        admission_claim=None,
        connection=None,
    ) -> None:
        """Persist one turn record to short-lived DB storage.

        This start-row write is the turn's mandatory admission, not optional
        telemetry: every failure propagates and the caller must not proceed.
        ``TargetBusy`` / ``AdmissionFenceLost`` / ``AdmissionUnavailable`` pass
        through typed; any other store failure becomes ``AdmissionUnavailable``.
        """
        try:
            request_payload = {
                "messages": [_message_to_dict(msg) for msg in prepared_messages],
            }
            if request_extensions:
                request_payload.update(request_extensions)
            self._turn_log_store.save_turn(
                turn_id=turn_id,
                request_id=request_id,
                path=path,
                stream=stream,
                model=model,
                bot_id=bot_id,
                user_id=user_id,
                status=status,
                latency_ms=latency_ms,
                user_prompt=user_prompt,
                request_payload=request_payload,
                response_text=response_text,
                tool_calls=_normalize_tool_call_details(tool_calls),
                error_text=error_text,
                agent_session_key=agent_session_key,
                agent_request_id=agent_request_id,
                trigger_message_id=trigger_message_id,
                assistant_message_id=assistant_message_id,
                parent_turn_id=parent_turn_id,
                admission_claim=admission_claim,
                connection=connection,
            )
        except Exception as e:
            from .turn_admission import AdmissionFenceLost, AdmissionUnavailable, TargetBusy
            if isinstance(e, (TargetBusy, AdmissionFenceLost, AdmissionUnavailable)):
                raise
            log.warning("Turn admission failed for %s: %s", turn_id, e)
            raise AdmissionUnavailable(f"turn admission failed: {e}") from e

    def _update_turn_log(
        self,
        *,
        turn_id: str,
        status: str | None = None,
        latency_ms: float | None = None,
        response_text: str | None = None,
        prepared_messages: list | None = None,
        tool_calls: list[dict] | None = None,
        error_text: str | None = None,
        assistant_message_id: str | None = None,
        agent_session_key: str | None = None,
        agent_request_id: str | None = None,
        animation: str | None = None,
        token_usage: dict | None = None,
        end_reason: str | None = None,
        abort_cause: str | None = None,
    ) -> None:
        """Update an existing turn log row with new data."""
        try:
            request_payload = None
            if prepared_messages is not None:
                request_payload = {
                    "messages": [_message_to_dict(msg) for msg in prepared_messages],
                }
            self._turn_log_store.update_turn(
                turn_id=turn_id,
                status=status,
                latency_ms=latency_ms,
                response_text=response_text,
                request_payload=request_payload,
                tool_calls=_normalize_tool_call_details(tool_calls) if tool_calls is not None else None,
                error_text=error_text,
                assistant_message_id=assistant_message_id,
                agent_session_key=agent_session_key,
                agent_request_id=agent_request_id,
                animation=animation,
                token_usage=token_usage,
                end_reason=end_reason,
                abort_cause=abort_cause,
            )
            if (status in {"ok", "completed", "error", "timeout", "cancelled", "aborted"} or end_reason is not None):
                dispatcher = getattr(self, "_inter_bot_dispatcher", None)
                if dispatcher is not None:
                    dispatcher.wake()
        except Exception as e:
            log.debug("Failed to update turn log: %s", e)

    def _extract_agent_backend_tool_calls(
        self,
        *,
        llm_bawt: Any,
    ) -> list[dict]:
        """Extract normalized tool-call details from agent backend clients."""
        if not isinstance(getattr(llm_bawt, "client", None), AgentBackendClient):
            return []

        extracted: list[dict] = []
        for tc in llm_bawt.client.get_tool_calls():
            result_payload = tc.get("result")
            if result_payload is None:
                result_payload = "Result not exposed by backend (see assistant response)."
            entry = {
                    "iteration": 1,
                    "tool": tc.get("display_name") or tc.get("name", "unknown"),
                    "parameters": tc.get("arguments", {}),
                    "result": result_payload,
                }
            if tc.get("call_id"):
                entry["call_id"] = tc["call_id"]
            extracted.append(entry)
        return extracted

    def _finalize_turn(
        self,
        *,
        llm_bawt,
        turn_id: str,
        response_text: str,
        tool_context: str,
        tool_call_details: list[dict],
        prepared_messages: list,
        user_prompt: str,
        model: str,
        bot_id: str,
        user_id: str,
        elapsed_ms: float,
        stream: bool,
        animation: str | None = None,
        token_usage: dict | None = None,
        attachments: list[dict] | None = None,
        reasoning: str | None = None,
        status: str = "ok",
        end_reason: str | None = None,
        assistant_message_id: str | None = None,
    ) -> None:
        """Finalize a turn in one place for both non-streaming and streaming paths.

        ``status``/``end_reason`` default to a normal ``ok`` completion. The abort
        path passes ``status="aborted"`` so a Stop still COMMITS the partial
        assistant text to history (instead of discarding the in-progress message)
        while preserving the aborted terminal state (TASK-286).
        """
        if not response_text:
            # Empty responses still end the turn. Without this, non-streaming
            # callbacks that are cancelled or legitimately return no text leave
            # ended_at NULL forever and block every queued successor.
            self._update_turn_log(
                turn_id=turn_id,
                status=status,
                latency_ms=elapsed_ms,
                prepared_messages=prepared_messages,
                response_text="",
                tool_calls=tool_call_details,
                assistant_message_id=assistant_message_id,
                animation=animation,
                token_usage=token_usage,
                end_reason=end_reason or ("aborted" if status in {"cancelled", "aborted"} else "stop"),
            )
            return

        # Safety-net output cleaning in case adapter-level cleanup did not run upstream.
        adapter = getattr(llm_bawt, "adapter", None)
        if adapter:
            cleaned = adapter.clean_output(response_text)
            if cleaned != response_text:
                log.info(
                    "Adapter '%s' cleaned response: %s -> %s chars",
                    adapter.name,
                    len(response_text),
                    len(cleaned),
                )
                response_text = cleaned

        token_usage = self._resolve_turn_token_usage(llm_bawt, token_usage)

        extracted_tool_calls = self._extract_agent_backend_tool_calls(llm_bawt=llm_bawt)
        if extracted_tool_calls and not tool_call_details:
            # Only use extracted calls when streaming didn't already capture them.
            # Streaming-captured calls have call_id for reliable dedup; extracted
            # calls from the agent backend API may lack it.
            tool_call_details.extend(extracted_tool_calls)
            # For agent backends (e.g. Claude Code), tools execute remotely so
            # tool_context is empty.  Synthesize it from the extracted calls
            # so that tool usage is persisted alongside the response.
            if not tool_context:
                parts = []
                for tc in extracted_tool_calls:
                    name = tc.get("tool") or tc.get("name", "unknown")
                    result = tc.get("result", "")
                    parts.append(f"[{name}]\n{result}")
                tool_context = "\n\n".join(parts)

        # TASK-306 Section B: turn-integrity guard. A bridge/agent turn that
        # produced a non-trivial narration but recorded ZERO tool calls is
        # claiming work it may not have done (the "claims work, no tool_use"
        # bug). We cannot yet reliably separate a legitimate conversational
        # reply from a fabricated work-claim without a task-bound signal (not
        # available in this turn path), so we FLAG (warn) rather than reject —
        # rejecting on this imperfect predicate would drop legitimate no-tool
        # replies, which violates the same honesty invariant in reverse.
        # Escalation to reject is gated on a reliable "claims work" predicate.
        is_bridge_turn = isinstance(getattr(llm_bawt, "client", None), AgentBackendClient)
        if _is_unbacked_work_claim(
            is_bridge_turn=is_bridge_turn,
            tool_call_details=tool_call_details,
            response_text=response_text,
        ):
            stripped = response_text.strip()
            log.warning(
                "TURN-INTEGRITY: bridge turn %s (bot=%s) finalized with "
                "non-trivial narration (%d chars) but ZERO tool_use — possible "
                "fabricated work claim. response_head=%r",
                turn_id, bot_id, len(stripped), stripped[:160],
            )

        llm_bawt.finalize_response(response_text, tool_context, attachments=attachments, reasoning=reasoning, message_id=assistant_message_id)

        self._update_turn_log(
            turn_id=turn_id,
            status=status,
            latency_ms=elapsed_ms,
            prepared_messages=prepared_messages,
            response_text=response_text,
            tool_calls=tool_call_details,
            assistant_message_id=assistant_message_id,
            animation=animation,
            token_usage=token_usage,
            end_reason=end_reason,
        )

        if stream:
            log.llm_response(response_text, elapsed_ms=elapsed_ms)

        if self.config.DEBUG_TURN_LOG or os.environ.get("LLM_BAWT_DEBUG_TURN_LOG"):
            _write_debug_turn_log(
                prepared_messages=prepared_messages,
                user_prompt=user_prompt,
                response=response_text,
                model=model,
                bot_id=bot_id,
                user_id=user_id,
                tool_calls=tool_call_details,
            )


def _write_debug_turn_log(
    prepared_messages: list,
    user_prompt: str,
    response: str,
    model: str,
    bot_id: str,
    user_id: str,
    tool_calls: list[dict] | None = None,
) -> None:
    """Write the current turn's request/response data to a debug log file.

    Called when debug logging is enabled. Overwrites the file on each turn
    to show the most recent request/response for review.
    """
    import json
    from ..utils.paths import resolve_log_dir

    try:
        logs_dir = resolve_log_dir()
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / "debug_turn.txt"

        # Build the log content
        lines = []
        lines.append("=" * 80)
        lines.append(f"DEBUG TURN LOG - {datetime.now().isoformat()}")
        lines.append(f"Model: {model}")
        lines.append(f"Bot: {bot_id}")
        lines.append(f"User: {user_id}")
        lines.append("=" * 80)
        lines.append("")

        # Request data - all context messages
        lines.append("─" * 40)
        lines.append("REQUEST MESSAGES")
        lines.append("─" * 40)
        for i, msg in enumerate(prepared_messages):
            role = msg.role if hasattr(msg, 'role') else msg.get('role', 'unknown')
            content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
            timestamp = msg.timestamp if hasattr(msg, 'timestamp') else msg.get('timestamp', 0)
            lines.append(f"\n[{i}] Role: {role}")
            lines.append(f"    Timestamp: {timestamp}")
            lines.append(f"    Content ({len(content)} chars):")
            lines.append("    " + "─" * 36)
            # Indent content for readability
            for content_line in str(content).split("\n"):
                lines.append(f"    {content_line}")
            lines.append("")

        # Tool calls section (between request and response)
        if tool_calls:
            total_calls = len(tool_calls)
            iterations = max((tc.get("iteration", 1) for tc in tool_calls), default=1)
            lines.append("─" * 40)
            lines.append(f"TOOL CALLS ({total_calls} call{'s' if total_calls != 1 else ''} across {iterations} iteration{'s' if iterations != 1 else ''})")
            lines.append("─" * 40)
            for idx, tc in enumerate(tool_calls, 1):
                lines.append("")
                lines.append(f"[{idx}] Tool: {tc.get('tool', 'unknown')}")
                params = tc.get('parameters', {})
                if isinstance(params, dict):
                    for pk, pv in params.items():
                        lines.append(f"    {pk}: {pv}")
                else:
                    lines.append(f"    Parameters: {params}")
                result = tc.get('result', '')
                lines.append(f"    Result ({len(result)} chars):")
                lines.append("    " + "─" * 36)
                for result_line in str(result)[:2000].split("\n"):
                    lines.append(f"    {result_line}")
            lines.append("")

        # Response data
        lines.append("─" * 40)
        lines.append("RESPONSE")
        lines.append("─" * 40)
        lines.append(f"Length: {len(response)} chars")
        lines.append("")
        lines.append(response)
        lines.append("")

        # Also dump as JSON for machine parsing
        lines.append("─" * 40)
        lines.append("JSON FORMAT (for parsing)")
        lines.append("─" * 40)

        json_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "bot_id": bot_id,
            "user_id": user_id,
            "request": [_message_to_dict(msg) for msg in prepared_messages],
            "tool_calls": _normalize_tool_call_details(tool_calls),
            "response": response,
        }
        lines.append(json.dumps(json_data, indent=2, ensure_ascii=False, default=str))

        # Write to file (overwrite)
        log_file.write_text("\n".join(lines), encoding="utf-8")
        log.debug(f"Debug turn log written to: {log_file}")

    except Exception as e:
        log.warning(f"Failed to write debug turn log: {e}")
