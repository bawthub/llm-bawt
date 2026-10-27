"""Chat-streaming bridge/session responsibilities (TASK-554).

Split out of ``chat_streaming.py``: the agent-bridge streaming path
(``_stream_via_bridge``) plus its conversation-offset and OpenClaw
session helpers. Kept as a base mixin that ``ChatStreamingMixin`` inherits,
so the composed ``BackgroundService`` exposes the same methods and every
``self.*`` reference (turn-log store, session bridge, gateway http, config,
finalize/persist/update turn) resolves across the mixin set unchanged.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator

from agent_bridge.abort_contract import abort_cause_of

from .logging import get_service_logger

log = get_service_logger(__name__)


from .session_lifecycle.bridge_reset import BridgeSessionResetMixin


class ChatStreamingBridgeMixin(BridgeSessionResetMixin):
    """Agent-bridge streaming and OpenClaw transport helpers."""

    def _is_openclaw_bot(self, model_alias: str) -> bool:
        """Check if this model alias maps to an openclaw agent backend."""
        model_def = self.config.resolve_model(model_alias, default={})
        if model_def.get("type") == "agent_backend" and model_def.get("backend") == "openclaw":
            return True
        return model_alias == "openclaw"

    def _get_openclaw_session_key(self, model_alias: str) -> str:
        """Get the session_key for an OpenClaw model from its bot_config."""
        model_def = self.config.resolve_model(model_alias, default={})
        return (
            (model_def.get("bot_config") or {}).get("session_key")
            or "agent:main:main"
        )

    async def _stream_via_bridge(
        self,
        *,
        user_prompt: str,
        session_key: str,
        response_id: str,
        created: int,
        model_alias: str,
        bot_id: str,
        user_id: str,
        ctx: Any,
        turn_log_id: str,
        llm_bawt: Any,
        trigger_message_id: str | None = None,
        assistant_message_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream an OpenClaw response via the WS SessionBridge."""
        bridge = self._session_bridge
        start_time = time.time()

        self._persist_turn_log(
            turn_id=turn_log_id,
            request_id=ctx.request_id,
            path=ctx.path,
            stream=True,
            model=model_alias,
            bot_id=bot_id,
            user_id=user_id,
            status="streaming",
            latency_ms=None,
            user_prompt=user_prompt,
            prepared_messages=[],
            response_text="",
        )

        # Accumulate response data outside try so the finally block can
        # always access them (including on GeneratorExit from client disconnect).
        from .tool_event_coordinator import ToolEventCoordinator
        from .tool_changed_files import ToolChangedFilesCoordinator
        tool_events = ToolEventCoordinator(self._turn_log_store.engine)
        changed_files = ToolChangedFilesCoordinator(self._turn_log_store.engine)
        full_text_parts: list[str] = []
        tool_call_details: list[dict] = []
        # Captures upstream SDK token usage (bridge sends it on ASSISTANT_DONE)
        # so the finalize path can persist it — mirrors the native stream path.
        token_usage_holder: list[dict | None] = [None]
        _finalized = False
        # Periodic flush: persist partial response text every N seconds so
        # clients reconnecting after a page refresh can show progress.
        _last_partial_flush = time.time()

        try:
            # Persist user message to bot history so it appears in /v1/history
            llm_bawt.history_manager.add_message("user", user_prompt)

            # Send user message and subscribe to events
            redis_sub = getattr(self, '_redis_subscriber', None)
            if redis_sub:
                # Redis mode: send via Gateway HTTP, subscribe via Redis
                # Record-first token, resolved per turn so a saved or rotated
                # token applies without a restart. An unusable record raises
                # and fails the turn; it never falls back to env.
                from llm_bawt.integrations.openclaw_credentials import gateway_token
                token = await asyncio.to_thread(gateway_token, self.config)
                gateway = getattr(self, '_gateway_http', None)
                if gateway is None or gateway.token != token:
                    from openclaw_bridge.gateway_http import GatewayHttpClient
                    gateway = GatewayHttpClient(self.config.OPENCLAW_GATEWAY_URL, token)
                    self._gateway_http = gateway
                run_id = await gateway.send_user_message(session_key, user_prompt)
                log.info("OpenClaw gateway HTTP: sent message, run_id=%s", run_id)
                event_source = redis_sub.subscribe(session_key, run_id=run_id)
            else:
                # In-process mode: send via WS bridge, subscribe via fanout
                run_id = await bridge.send_user_message(session_key, user_prompt)
                bridge._api_run_ids.add(run_id)
                log.info("OpenClaw bridge: sent message, run_id=%s", run_id)
                event_source = bridge._fanout.subscribe(session_key)

            from agent_bridge.events import AgentEventKind

            # Track tool call state for OpenAI-compat delta.tool_calls
            _tool_call_index = 0
            _in_tool_calls = False
            _last_call_id = ""

            async for event in event_source:
                if event.kind == AgentEventKind.ASSISTANT_DELTA:
                    delta = event.text or ""
                    if delta:
                        # If transitioning from tool calls to content, emit finish_reason first
                        if _in_tool_calls:
                            finish_data = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_alias,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                            }
                            yield f"data: {json.dumps(finish_data)}\n\n"
                            _in_tool_calls = False
                            _tool_call_index = 0
                            _last_call_id = ""
                            # Insert paragraph break between tool-call
                            # segment and resumed content so it persists
                            # to DB and survives page reload.
                            full_text_parts.append("\n\n")
                            delta = "\n\n" + delta

                        full_text_parts.append(delta)
                        data = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_alias,
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(data)}\n\n"

                        # Periodic flush of partial response to DB so resumed
                        # clients can show progress instead of empty bouncing dots.
                        now = time.time()
                        if now - _last_partial_flush >= 5.0:
                            _last_partial_flush = now
                            try:
                                self._update_turn_log(
                                    turn_id=turn_log_id,
                                    response_text="".join(full_text_parts),
                                )
                            except Exception:
                                pass  # non-critical

                elif event.kind == AgentEventKind.TOOL_START:
                    tool_name = event.tool_name or "unknown"
                    tool_args = event.tool_arguments or {}
                    call_id = f"call_{uuid.uuid4().hex[:8]}"
                    _last_call_id = call_id
                    # Emit as OpenAI delta.tool_calls
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_alias,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": _tool_call_index,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": json.dumps(tool_args, ensure_ascii=False) if isinstance(tool_args, dict) else str(tool_args),
                                    },
                                }],
                            },
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    _in_tool_calls = True
                    _tool_call_index += 1
                    tool_call_details.append({
                        "iteration": 1,
                        "tool": tool_name,
                        "parameters": tool_args,
                        "result": "",
                        "is_error": False,
                        "call_id": call_id,
                        # SDK tool_use id + parent id so sub-agent nesting (child
                        # card under its Agent card) survives reload from
                        # tool_calls_json, not just live rendering (TASK-344).
                        "tool_use_id": getattr(event, "tool_use_id", None),
                        "parent_tool_use_id": getattr(event, "parent_tool_use_id", None),
                    })
                    # Publish tool_start to unified event stream
                    if redis_sub:
                        try:
                            await redis_sub.publish_tool_event(bot_id, user_id, tool_events.start({
                                "_type": "tool_event",
                                "event": "tool_start",
                                "turn_id": turn_log_id,
                                "trigger_message_id": trigger_message_id,
                                "bot_id": bot_id,
                                "user_id": user_id,
                                "tool_name": tool_name,
                                "arguments": tool_args,
                                "call_id": call_id,
                                "tool_use_id": getattr(event, "tool_use_id", None),
                                "parent_tool_use_id": getattr(event, "parent_tool_use_id", None),
                                "iteration": _tool_call_index,
                                "provider": event.provider,
                                "ts": time.time(),
                            }))
                        except Exception:
                            pass

                elif event.kind == AgentEventKind.TOOL_END:
                    tool_result = str(event.tool_result or "")
                    tool_failed = bool(event.tool_error)
                    # Pair this tool_end with its tool_start. Prefer the stable
                    # bridge-stamped tool_use_id: tools do NOT always finish in
                    # LIFO order (parallel tool_use blocks in one message,
                    # sub-agents), so a blind stack pop can pair a tool_end with
                    # the WRONG tool_start — stamping a mismatched call_id. The
                    # frontend keys card-healing on call_id, so the real running
                    # card never flips and sticks on "running" forever while a
                    # sibling card wrongly absorbs this result (TASK-414).
                    end_call_id = _last_call_id or ""
                    end_tuid = getattr(event, "tool_use_id", None)
                    matched = None
                    if end_tuid:
                        for i in range(len(tool_call_details) - 1, -1, -1):
                            if tool_call_details[i].get("tool_use_id") == end_tuid:
                                matched = tool_call_details.pop(i)
                                break
                    if matched is None and tool_call_details:
                        # No tool_use_id (non-claude providers) or no id match:
                        # fall back to the original LIFO pairing.
                        matched = tool_call_details.pop()
                    if matched is not None:
                        matched["result"] = tool_result
                        matched["is_error"] = tool_failed
                        end_call_id = matched.get("call_id", end_call_id)
                    _tool_end_event = {
                        "_type": "tool_event",
                        "event": "tool_end",
                        "turn_id": turn_log_id,
                        "trigger_message_id": trigger_message_id,
                        "bot_id": bot_id,
                        "user_id": user_id,
                        "tool_name": event.tool_name or "unknown",
                        "arguments": (
                            matched.get("parameters", {}) if matched is not None
                            else (event.tool_arguments or {})
                        ),
                        "call_id": end_call_id,
                        "tool_use_id": getattr(event, "tool_use_id", None),
                        "parent_tool_use_id": getattr(event, "parent_tool_use_id", None),
                        "iteration": _tool_call_index,
                        "provider": event.provider,
                        "result": tool_result,
                        "tool_result_payload": getattr(event, "tool_result_payload", None),
                        "is_error": tool_failed,
                        "ts": time.time(),
                    }
                    # Publish tool_end to unified event stream, then its durable
                    # app-attributed file_changed companion when applicable.
                    if redis_sub:
                        try:
                            await redis_sub.publish_tool_event(
                                bot_id, user_id, tool_events.end(_tool_end_event)
                            )
                            if not tool_failed:
                                _metadata_change = changed_files.persist(_tool_end_event)
                                if _metadata_change is not None:
                                    await redis_sub.publish_tool_event(bot_id, user_id, {
                                        "_type": "file_changed",
                                        "turn_id": turn_log_id,
                                        "trigger_message_id": trigger_message_id,
                                        "bot_id": bot_id,
                                        "user_id": user_id,
                                        "tool_name": _tool_end_event["tool_name"],
                                        "tool_use_id": getattr(event, "tool_use_id", None),
                                        "file": _metadata_change["file"],
                                        "summary": _metadata_change["summary"],
                                        "ts": time.time(),
                                    })
                        except Exception:
                            pass

                elif event.kind == AgentEventKind.FILE_CHANGED:
                    file_data = event.raw.get("file") if isinstance(event.raw, dict) else None
                    persisted = changed_files.persist({
                        "turn_id": turn_log_id,
                        "bot_id": bot_id,
                        "user_id": user_id,
                        "trigger_message_id": trigger_message_id,
                        "tool_use_id": event.tool_use_id,
                        "file": file_data,
                    })
                    if redis_sub and persisted is not None:
                        try:
                            await redis_sub.publish_tool_event(bot_id, user_id, {
                                "_type": "file_changed",
                                "turn_id": turn_log_id,
                                "trigger_message_id": trigger_message_id,
                                "bot_id": bot_id,
                                "user_id": user_id,
                                "tool_name": event.tool_name,
                                "tool_use_id": event.tool_use_id,
                                "file": persisted["file"],
                                "summary": persisted["summary"],
                                "ts": time.time(),
                            })
                        except Exception:
                            pass

                elif event.kind == AgentEventKind.ASSISTANT_DONE:
                    if isinstance(event.raw, dict) and event.raw.get("end_reason") == "aborted":
                        # TASK-1231B: the cause is stamped by the same write that ends the row.
                        self._update_turn_log(turn_id=turn_log_id, status="aborted", end_reason="aborted",
                                              abort_cause=abort_cause_of(event.raw))
                    # ASSISTANT_DONE carries the complete response text.
                    # Yield any portion not already streamed as deltas.
                    # Capture token_usage from agent backends (bridge sends it
                    # on ASSISTANT_DONE, not as a separate token_usage event).
                    if event.token_usage and isinstance(event.token_usage, dict):
                        token_usage_holder[0] = event.token_usage
                    done_text = event.text or ""
                    if done_text:
                        accumulated = "".join(full_text_parts)
                        extra = ""
                        if done_text.startswith(accumulated):
                            extra = done_text[len(accumulated):]
                        elif len(done_text) > len(accumulated):
                            extra = done_text[len(accumulated):] if accumulated else done_text
                        if extra.strip():
                            if _in_tool_calls:
                                finish_data = {
                                    "id": response_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_alias,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                                }
                                yield f"data: {json.dumps(finish_data)}\n\n"
                                _in_tool_calls = False
                                full_text_parts.append("\n\n")
                                extra = "\n\n" + extra
                            full_text_parts.append(extra)
                            data = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_alias,
                                "choices": [{"index": 0, "delta": {"content": extra}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(data)}\n\n"

                elif event.kind == AgentEventKind.RUN_COMPLETED:
                    if event.run_id == run_id or not run_id:
                        break

                elif event.kind == AgentEventKind.ERROR:
                    # Surface the error as VISIBLE assistant content (TASK-202).
                    # Previously this only emitted a sidecar service.warning
                    # which the UI does not render — leaving the user with a
                    # blank message and no idea what went wrong. Now the error
                    # text shows up in the assistant bubble itself, in addition
                    # to the warning channel for tooling and the error status
                    # on the turn log.
                    err_text = (event.text or "").strip() or "(no error text from bridge)"
                    visible = f"⚠️ openclaw bridge error\n\n```\n{err_text}\n```"
                    full_text_parts.append(visible)
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_alias,
                        "choices": [
                            {"index": 0, "delta": {"content": visible}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    warning_data = {
                        "object": "service.warning",
                        "model": model_alias,
                        "warnings": [f"openclaw_error: {err_text}"],
                    }
                    yield f"data: {json.dumps(warning_data)}\n\n"
                    # Record the error on the turn log so it shows up in
                    # /v1/turn-logs review.
                    try:
                        elapsed_ms = (time.time() - start_time) * 1000
                        self._update_turn_log(
                            turn_id=turn_log_id,
                            status="error",
                            latency_ms=elapsed_ms,
                            error_text=err_text,
                        )
                    except Exception as _log_err:
                        log.warning(
                            "Failed to record openclaw bridge error on turn log %s: %s",
                            turn_log_id,
                            _log_err,
                        )
                    break

            # Final chunk
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_alias,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

            # Fetch tool calls from chat history (gateway doesn't stream them)
            # Only fetch if streaming didn't already capture tool calls —
            # otherwise this duplicates them (they lack call_id so dedup fails).
            if not tool_call_details:
                try:
                    if redis_sub and gateway:
                        history_msgs = await gateway.get_chat_history(session_key, limit=20)
                    elif bridge:
                        history_msgs = await bridge._ws_client.get_chat_history(session_key, limit=20)
                    else:
                        history_msgs = []
                    for msg in reversed(history_msgs):
                        if not isinstance(msg, dict):
                            continue
                        # Look for tool_calls in the most recent assistant message
                        if msg.get("role") == "assistant":
                            for tc in msg.get("tool_calls") or []:
                                fn = tc.get("function") or {}
                                tool_call_details.append({
                                    "iteration": 1,
                                    "tool": fn.get("name") or tc.get("name") or "unknown",
                                    "parameters": fn.get("arguments") or {},
                                    "result": "",
                                })
                            break
                        # Also capture tool results
                        if msg.get("role") == "tool":
                            # Will be matched to tool_call above
                            pass
                except Exception as e:
                    log.debug("Could not fetch tool calls from chat.history: %s", e)

            # Finalize turn
            full_text = "".join(full_text_parts)
            elapsed_ms = (time.time() - start_time) * 1000
            if full_text:
                self._finalize_turn(
                    llm_bawt=llm_bawt,
                    turn_id=turn_log_id,
                    response_text=full_text,
                    tool_context="",
                    tool_call_details=tool_call_details,
                    prepared_messages=[],
                    user_prompt=user_prompt,
                    model=model_alias,
                    bot_id=bot_id,
                    user_id=user_id,
                    elapsed_ms=elapsed_ms,
                    stream=True,
                    token_usage=token_usage_holder[0],
                    assistant_message_id=assistant_message_id,
                )
            else:
                self._update_turn_log(
                    turn_id=turn_log_id,
                    status="timeout",
                    latency_ms=elapsed_ms,
                )
            _finalized = True

        except Exception as e:
            # TASK-202: don't swallow the exception into a sidecar warning the
            # UI ignores — render it as VISIBLE assistant content. Include
            # exception type + message + a short traceback excerpt to help
            # diagnose what failed.
            import traceback as _tb
            log.exception("OpenClaw bridge stream failed: %s", e)
            elapsed_ms = (time.time() - start_time) * 1000
            tb_excerpt = _tb.format_exc(limit=4).strip()
            err_text = f"{type(e).__name__}: {e}"
            self._update_turn_log(
                turn_id=turn_log_id,
                status="error",
                latency_ms=elapsed_ms,
                error_text=err_text,
            )

            visible = (
                f"⚠️ bridge stream failed\n\n"
                f"```\n{err_text}\n\n{tb_excerpt}\n```"
            )
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_alias,
                "choices": [
                    {"index": 0, "delta": {"content": visible}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"
            full_text_parts.append(visible)

            warning_data = {
                "object": "service.warning",
                "model": model_alias,
                "warnings": [f"bridge_error: {err_text}"],
            }
            yield f"data: {json.dumps(warning_data)}\n\n"
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_alias,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

            # Persist the failed turn (partial response + visible error) to
            # bot history so it survives a page refresh. Without this the
            # user sees their prompt without any reply after reload — the
            # error bubble only existed on the SSE wire. Best-effort: any
            # persistence failure is logged and swallowed.
            try:
                full_text = "".join(full_text_parts)
                if full_text:
                    self._finalize_turn(
                        llm_bawt=llm_bawt,
                        turn_id=turn_log_id,
                        response_text=full_text,
                        tool_context="",
                        tool_call_details=tool_call_details,
                        prepared_messages=[],
                        user_prompt=user_prompt,
                        model=model_alias,
                        bot_id=bot_id,
                        user_id=user_id,
                        elapsed_ms=elapsed_ms,
                        stream=True,
                        token_usage=token_usage_holder[0],
                        assistant_message_id=assistant_message_id,
                    )
            except Exception as _persist_err:
                log.warning(
                    "Failed to persist errored turn %s to bot history: %s",
                    turn_log_id, _persist_err,
                )
            _finalized = True

        finally:
            # Ensure turn is finalized even on GeneratorExit (client disconnect).
            # GeneratorExit inherits from BaseException, not Exception, so the
            # except block above does not catch it.
            if not _finalized:
                full_text = "".join(full_text_parts)
                elapsed_ms = (time.time() - start_time) * 1000
                try:
                    if full_text:
                        self._finalize_turn(
                            llm_bawt=llm_bawt,
                            turn_id=turn_log_id,
                            response_text=full_text,
                            tool_context="",
                            tool_call_details=tool_call_details,
                            prepared_messages=[],
                            user_prompt=user_prompt,
                            model=model_alias,
                            bot_id=bot_id,
                            user_id=user_id,
                            elapsed_ms=elapsed_ms,
                            stream=True,
                            token_usage=token_usage_holder[0],
                            assistant_message_id=assistant_message_id,
                        )
                    else:
                        self._update_turn_log(
                            turn_id=turn_log_id,
                            status="disconnected",
                            latency_ms=elapsed_ms,
                            tool_calls=tool_call_details or None,
                        )
                except Exception as fin_err:
                    log.error("Bridge finalization failed (turn %s): %s", turn_log_id, fin_err)
                    try:
                        self._update_turn_log(
                            turn_id=turn_log_id,
                            status="error",
                            latency_ms=elapsed_ms,
                            error_text=f"finalize_error: {fin_err}",
                        )
                    except Exception:
                        pass
