"""TASK-737: the per-turn stream interceptor between the backend and the queue.

:class:`ToolEventInterceptor` wraps the backend stream inside the streaming
worker thread. It publishes tool, text, reasoning, usage, file-change and
sub-agent events straight to the unified Redis stream (so they survive SSE
generator cancellation), keeps the per-turn tool correlation and text/reasoning
offsets, and yields the items the downstream consumer still needs.

It was extracted unchanged from the ``_intercept_tool_events`` closure in
``turn_stream_worker.py``; every captured local is now an explicit
collaborator or a field of :class:`InterceptorState`.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

from .logging import get_service_logger
from .turn_stream_context import _TEXT_DELTA_FLUSH_CHARS
from .upstream_status import upstream_status_sse_payload

# Same logger name as the worker the code came from, so log filters and
# operators' searches keep matching.
log = get_service_logger("llm_bawt.service.turn_stream_worker")


def js_len(s: str) -> int:
    """len(s) in UTF-16 code units — JS ``String.length``."""
    return len(s) + sum(1 for ch in s if ord(ch) > 0xFFFF)


@dataclass
class InterceptorState:
    """Mutable per-stream state (formerly closure locals)."""

    # Track tool→content transitions / whether any text has been yielded.
    saw_tool: bool = False
    saw_text: bool = False
    # Cumulative count of assistant text yielded so far, in UTF-16 CODE
    # UNITS (JavaScript string-length semantics), NOT Python code points.
    # Stamped onto each tool_call as text_offset so the frontend can split
    # response_text at tool boundaries and render an interleaved transcript
    # (text → tool → text) that survives reload. A code-point count drifts
    # +1 per astral-plane char and orphans the pre-tool sentence's final "."
    # into the next bubble (TASK-799). MUST stay in lock-step with the strs
    # yielded to consume_stream_chunks (including injected "\n\n" breaks).
    text_chars: int = 0
    # Reasoning-lane offset counter, same UTF-16 units. Running tally (not
    # js_len of the whole lane per delta) so long lanes stay O(n) total.
    reasoning_units: int = 0
    # TASK-286/456: assistant text buffered for coalesced ``text_delta``
    # publication. Must stay in lock-step with ``text_chars`` so each
    # delta's ``text_offset`` (chars emitted before it) is exact.
    text_buf: list[str] = field(default_factory=list)


class ToolEventInterceptor:
    """Intercept one turn's backend stream; see the module docstring."""

    def __init__(
        self,
        ctx: Any,
        *,
        svc: Any,
        publish_event_direct: Callable[[dict], Any],
        persist_publish_approval: Callable[[dict], Any],
        persist_publish_await: Callable[[dict], Any],
        changed_files: Any,
        turn_usage: Any,
        pin_confirmer: Any,
        scheduled_notices: Any,
        tool_calls: Any,
        exec_stamps: Any,
        call_index: list[int],
        confirmed_start_reaped: list[bool],
    ) -> None:
        self.ctx = ctx
        self.svc = svc
        self.publish = publish_event_direct
        self.persist_publish_approval = persist_publish_approval
        self.persist_publish_await = persist_publish_await
        self.changed_files = changed_files
        self.turn_usage = turn_usage
        self.pin_confirmer = pin_confirmer
        self.scheduled_notices = scheduled_notices
        self.tool_calls = tool_calls
        self.exec_stamps = exec_stamps
        self.call_index = call_index
        self.confirmed_start_reaped = confirmed_start_reaped
        self.state = InterceptorState()
        # Per-turn identity, read once like the closure's captured locals.
        self.turn_id = ctx.turn_log_id
        self.trigger_message_id = ctx.trigger_message_id
        self.bot_id = ctx.bot_id
        self.user_id = ctx.user_id

    # ------------------------------------------------------------------ text

    def flush_text(self, min_chars: int = 0) -> None:
        st = self.state
        if not st.text_buf:
            return
        s = "".join(st.text_buf)
        if len(s) < min_chars:
            return
        st.text_buf.clear()
        self.ctx.stream_text_offset = st.text_chars
        self.publish({
            "_type": "text_delta",
            "turn_id": self.turn_id,
            "trigger_message_id": self.trigger_message_id,
            "bot_id": self.bot_id,
            "user_id": self.user_id,
            # UTF-16 units of assistant text emitted BEFORE this delta — lets
            # the client splice it at the right position and dedupe an
            # overlapping/replayed delta regardless of arrival order vs the
            # cold-reload partial fetch. JS-length semantics (TASK-799).
            "text_offset": st.text_chars - js_len(s),
            "delta": s,
            "ts": time.time(),
        })
        # Every turn may be spoken. Feed the SAME raw text through the
        # block-boundary scrubber for audio; text_delta above keeps the
        # original markdown for the visual stream.
        tts_scrubber = self.ctx.tts_scrubber
        if tts_scrubber is not None:
            tts_offset = tts_scrubber.offset
            tts_block = tts_scrubber.feed(s)
            if tts_block:
                self.publish({
                    "_type": "tts_delta",
                    "turn_id": self.turn_id,
                    "bot_id": self.bot_id,
                    "user_id": self.user_id,
                    "tts_offset": tts_offset,
                    "delta": tts_block,
                    "ts": time.time(),
                })

    # ------------------------------------------------------------- main loop

    def run(self, inner: Iterable[Any]) -> Iterator[Any]:
        ctx = self.ctx
        st = self.state
        for item in inner:
            if not ctx.is_agent_backend and ctx.cancel_event.is_set():
                break  # do not publish a post-Stop delta/tool start to Redis
            if isinstance(item, dict) and item.get("event") == "steer_lifecycle":
                self.flush_text()
                self.publish({
                    **{k: v for k, v in item.items() if k != "event"},
                    "_type": "steer_lifecycle", "turn_id": self.turn_id,
                    "bot_id": self.bot_id, "user_id": self.user_id,
                    "text_offset": st.text_chars, "ts": time.time(),
                })
                # A lifecycle control frame is not model output and must not
                # confirm a prompt pin or reap other turns.
                continue
            self.pin_confirmer.observe(item)  # TASK-1318: model output confirms a pin move
            self.scheduled_notices.observe(item)
            if not self.confirmed_start_reaped[0]:
                self.confirmed_start_reaped[0] = True
                self._reap_on_confirmed_start()

            # Inject paragraph break when transitioning from tool events back
            # to text content. This flows into full_response_holder (persisted
            # to DB) and the SSE stream, ensuring paragraph breaks survive
            # reload.
            if isinstance(item, str) and item.strip():
                if st.saw_tool and st.saw_text:
                    st.text_chars += 2
                    st.text_buf.append("\n\n")
                    yield "\n\n"
                st.saw_tool = False
                st.saw_text = True

            if isinstance(item, dict):
                evt = item.get("event")
                if evt == "approval_required":
                    # Persist + surface durably from this worker thread so it
                    # survives the async-gen teardown that drops the chunk on
                    # turn-end (TASK-292/306). Falls through to `yield item`;
                    # the async-gen branch dedups via _approval_handled.
                    self.persist_publish_approval(item)
                if evt == "await_tool_result":
                    # TASK-370: same durability for the deferred
                    # AskUserQuestion; falls through to `yield item`.
                    self.persist_publish_await(item)
                if evt == "reasoning":
                    yield from self._on_reasoning(item)
                    continue
                if evt == "turn_cancelled":
                    self.svc._update_turn_log(
                        turn_id=self.turn_id, status="aborted", end_reason="aborted",
                        abort_cause=item.get("abort_cause"),
                    )
                    continue
                if evt == "metadata":
                    if item.get("upstream_model"):
                        ctx._upstream_model[0] = item["upstream_model"]
                    continue
                if evt == "live_usage":
                    yield from self._on_live_usage(item)
                    continue
                if evt == "token_usage":
                    # Capture for turn_complete payload — frontend surfaces
                    # input/cache/output + context window in the bubble pill.
                    tu = item.get("token_usage")
                    if isinstance(tu, dict):
                        ctx.token_usage_holder[0] = tu
                    continue
                if evt == "attachments":
                    refs = item.get("attachments")
                    if isinstance(refs, list):
                        ctx.agent_attachments_holder.extend(
                            r for r in refs if isinstance(r, dict)
                        )
                    continue
                if evt == "upstream_status":
                    self.publish(upstream_status_sse_payload(
                        item,
                        turn_id=self.turn_id,
                        trigger_message_id=self.trigger_message_id,
                        bot_id=self.bot_id,
                        user_id=self.user_id,
                    ))
                    continue
                if evt == "file_changed":
                    self._on_file_changed(item)
                    continue
                if evt in ("subagent_started", "subagent_progress", "subagent_done"):
                    self._on_subagent(evt, item)
                    continue
                if evt == "tool_executing":
                    self.exec_stamps.on_executing(item)  # TASK-1095 timing only
                    continue
                if evt == "llm_response":
                    self._on_llm_response(item)
                    continue
                if evt in ("tool_call", "tool_result"):
                    st.saw_tool = True
                    # Flush buffered text BEFORE the tool event so the resume
                    # channel preserves text→tool order.
                    self.flush_text()
                if evt == "tool_call":
                    self._on_tool_call(item)
                elif evt == "tool_result":
                    self._on_tool_result(item)
            if isinstance(item, str):
                st.text_chars += js_len(item)
                st.text_buf.append(item)
                self.flush_text(min_chars=_TEXT_DELTA_FLUSH_CHARS)
            yield item

        # Flush trailing buffered text so the tail of the response reaches the
        # resume channel even if it never crossed the size threshold.
        self.flush_text()

    # -------------------------------------------------------------- handlers

    def _reap_on_confirmed_start(self) -> None:
        """Reap other still-open turns for this bot, once per turn.

        Runs on the SDK/bridge-CONFIRMED first output (not the optimistic
        save_turn insert or turn_start publish) and clears their indicators.
        """
        bot_id, user_id, turn_id = self.bot_id, self.user_id, self.turn_id
        try:
            reaped = self.svc._turn_log_store.reap_other_open_turns(
                bot_id=bot_id, current_turn_id=turn_id,
            )
            redis_sub = self.ctx._redis_sub
            for rt in reaped:
                ruid = rt.get("user_id") or user_id
                evt = {
                    "_type": "turn_complete",
                    "turn_id": rt["id"],
                    "bot_id": bot_id,
                    "user_id": ruid,
                    # TASK-709: no session_id on turn_logs; the client's
                    # fallback clears by turn_id on null session_id.
                    "session_id": None,
                    "balance_update": rt.get("balance_update"),
                    "status": "timeout",
                    "end_reason": "timeout",
                    "ts": time.time(),
                }
                # Publish to the reaped turn's OWN {bot_id}:{user_id} stream so
                # the right client clears (its user may differ).
                if redis_sub:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            redis_sub.publish_tool_event(bot_id, ruid, evt),
                            self.ctx.loop,
                        )
                    except Exception as pub_err:
                        log.debug(
                            "reap turn_complete publish failed for %s: %s",
                            rt["id"], pub_err,
                        )
            if reaped:
                log.info(
                    "Reaped %d stale open turn(s) for bot %s on confirmed start of %s",
                    len(reaped), bot_id, turn_id,
                )
        except Exception as reap_err:
            log.debug("confirmed-start reap failed for %s: %s", bot_id, reap_err)

    def _on_reasoning(self, item: dict) -> Iterator[dict]:
        # Model reasoning ("thinking"): published to the unified stream and
        # yielded downstream as reasoning_content. Dict-only, so it NEVER
        # enters full_response_holder / the saved message (TASK-301), and it
        # does not touch the answer-text offsets or tool transitions.
        rtext = item.get("text", "")
        if not rtext:
            return
        st = self.state
        # Absolute reasoning offset (units emitted BEFORE this delta) lets a
        # refreshed client splice each live delta against the cold-reload
        # seed (TASK-506); UTF-16 units (TASK-799).
        r_offset = st.reasoning_units
        st.reasoning_units += js_len(rtext)
        self.ctx.reasoning_holder[0] += rtext
        self.publish({
            "_type": "reasoning_delta",
            "turn_id": self.turn_id,
            "trigger_message_id": self.trigger_message_id,
            "bot_id": self.bot_id,
            "user_id": self.user_id,
            "delta": rtext,
            "reasoning_offset": r_offset,
            "ts": time.time(),
        })
        yield {"_type": "reasoning_delta", "delta": rtext}

    def _on_live_usage(self, item: dict) -> Iterator[dict]:
        usage_event = self.turn_usage.capture(
            turn_id=self.turn_id,
            trigger_message_id=self.trigger_message_id,
            bot_id=self.bot_id,
            user_id=self.user_id,
            # The turn's catalog ref (`model@access_path`), same as
            # turn_complete and the turn log — not the bridge's upstream name.
            model=self.ctx.model_alias or item.get("model"),
            token_usage=item.get("token_usage"),
        )
        if usage_event is not None:
            self.ctx.token_usage_holder[0] = usage_event["token_usage"]
            self.publish(usage_event)
            yield {"_type": "turn_usage", **self.turn_usage.http_chunk(usage_event)}

    def _on_file_changed(self, item: dict) -> None:
        # TASK-664: one completed Edit/Write/NotebookEdit. Persist before
        # publication so reconnect/history never sees an event without its row.
        trigger = item.get("trigger_message_id") or self.trigger_message_id
        persisted = self.changed_files.persist({
            "turn_id": self.turn_id,
            "bot_id": self.bot_id,
            "user_id": self.user_id,
            "trigger_message_id": trigger,
            "tool_use_id": item.get("tool_use_id"),
            "file": item.get("file"),
        })
        if persisted is not None:
            self.publish({
                "_type": "file_changed",
                "turn_id": self.turn_id,
                "trigger_message_id": trigger,
                "bot_id": self.bot_id,
                "user_id": self.user_id,
                "tool_name": item.get("tool_name"),
                "tool_use_id": item.get("tool_use_id"),
                "file": persisted["file"],
                "summary": persisted["summary"],
                "ts": time.time(),
            })

    def _on_subagent(self, evt: str, item: dict) -> None:
        # TASK-344: sub-agent lifecycle, published for cross-tab visibility
        # and not passed downstream (not part of the text/tool response).
        self.publish({
            "_type": evt,
            "turn_id": self.turn_id,
            "trigger_message_id": item.get("trigger_message_id") or self.trigger_message_id,
            "bot_id": self.bot_id,
            "user_id": self.user_id,
            "task_id": item.get("task_id", ""),
            "description": item.get("description", ""),
            "task_type": item.get("task_type"),
            "status": item.get("status"),
            "summary": item.get("summary"),
            "last_tool_name": item.get("last_tool_name"),
            "usage": item.get("usage"),
            # TASK-1148: bridge liveness timeout diagnostic.
            "failure": item.get("failure"),
            "tool_use_id": item.get("tool_use_id"),
            "provider": item.get("provider"),
            "ts": time.time(),
        })

    def _on_llm_response(self, item: dict) -> None:
        # TASK-2110: durable per-response record, upserted live (tool-less
        # responses included). Not part of the text/tool transcript.
        self.publish({
            "_type": "llm_response_observed",
            "turn_id": self.turn_id,
            "trigger_message_id": item.get("trigger_message_id") or self.trigger_message_id,
            "bot_id": self.bot_id,
            "user_id": self.user_id,
            "response_id": item.get("response_id"),
            "host_parent": item.get("host_parent"),
            "first_seen_seq": item.get("first_seen_seq"),
            "model": item.get("model"),
            "usage": item.get("usage"),
            "usage_state": item.get("usage_state"),
            "wait_ms": item.get("wait_ms"),
            "stream_ms": item.get("stream_ms"),
            "ts": time.time(),
        })

    def _on_tool_call(self, item: dict) -> None:
        text_chars = self.state.text_chars
        self.call_index[0] += 1
        cid = f"call_{uuid.uuid4().hex[:8]}"
        self.tool_calls.start(
            call_id=cid,
            tool_name=item.get("name", "unknown"),
            tool_use_id=item.get("tool_use_id"),
        )
        # Inject call_id so the queue consumer uses the same ID as the SSE
        # event (no double-ID).
        item["_call_id"] = cid
        # The AUTHORITATIVE text_offset for the live OpenAI-compat SSE wire, so
        # the client doesn't recompute it from its own accumulated length and
        # split bubbles mid-sentence live (TASK-367).
        item["_text_offset"] = text_chars
        self.ctx.tool_call_details_holder.append({
            'tool': item.get('name', 'unknown'),
            'parameters': item.get('arguments', {}),
            'call_id': cid,
            # Chars of assistant text before this tool — drives
            # interleaved-transcript reconstruction.
            'text_offset': text_chars,
            # Originating backend, so recall from tool_calls_json dispatches
            # the frontend's (provider, tool_name) renderer.
            'provider': item.get('provider'),
            # SDK tool_use id + parent id so sub-agent nesting survives reload
            # from tool_calls_json (TASK-344).
            'tool_use_id': item.get('tool_use_id'),
            'parent_tool_use_id': item.get('parent_tool_use_id'),
        })
        self.publish({
            "_type": "tool_event",
            "event": "tool_start",
            "turn_id": self.turn_id,
            # Prefer the bridge-stamped value (authoritative for that turn),
            # falling back to the request-scope value.
            "trigger_message_id": item.get("trigger_message_id") or self.trigger_message_id,
            "bot_id": self.bot_id,
            "user_id": self.user_id,
            "tool_name": item.get("name", "unknown"),
            "arguments": item.get("arguments", {}),
            "call_id": cid,
            # SDK tool_use id: the anchor a sub-agent's child cards match their
            # parent_tool_use_id against (TASK-344).
            "tool_use_id": item.get("tool_use_id"),
            "parent_tool_use_id": item.get("parent_tool_use_id"),
            "iteration": 1,
            "provider": item.get("provider"),
            "exec_tracked": bool(item.get("exec_tracked")),
            # TASK-2110: requesting LLM response (None = unknown).
            "response_id": item.get("response_id"),
            "ts": time.time(),
            "text_offset": text_chars,
        })
        self.exec_stamps.on_tool_started(item.get("tool_use_id"), cid)

    def _enrich_tool_end_attachments(self, item: dict) -> list | None:
        # TASK-483: enrich this call's screenshot refs into the canonical
        # envelope (the SAME shape /v1/history emits) so the live tool card
        # renders the thumbnail inline. Worker thread, so the sync SELECT is
        # fine. Failure degrades to no inline image.
        if not item.get("attachments"):
            return None
        try:
            from ..media.assets import MediaAssetStore
            from ..media.serializers import enrich_attachments_for_messages

            shell = [{"attachments": list(item["attachments"])}]
            enrich_attachments_for_messages(shell, MediaAssetStore(self.svc.config))
            return shell[0].get("attachments") or None
        except Exception as enrich_err:
            log.warning("TASK-483: tool_end attachment enrichment failed: %s", enrich_err)
            return None

    def _on_tool_result(self, item: dict) -> None:
        # Pair by the bridge's stable tool_use_id first; name/LIFO fallback
        # remains for providers that omit it.
        result_name = item.get("name") or ""
        end_cid = self.tool_calls.finish(
            tool_name=result_name,
            tool_use_id=item.get("tool_use_id"),
        )
        item["_call_id"] = end_cid
        tool_end_attachments = self._enrich_tool_end_attachments(item)
        tool_end_event = {
            "_type": "tool_event",
            "event": "tool_end",
            "turn_id": self.turn_id,
            "trigger_message_id": item.get("trigger_message_id") or self.trigger_message_id,
            "bot_id": self.bot_id,
            "user_id": self.user_id,
            "tool_name": result_name or "unknown",
            "arguments": item.get("arguments", {}),
            "call_id": end_cid,
            "result": item.get("result", ""),
            "tool_result_payload": item.get("tool_result_payload"),
            # Same ids as tool_start so a reconnect that only sees tool_end can
            # still place the card under its parent Agent (TASK-344).
            "tool_use_id": item.get("tool_use_id"),
            "parent_tool_use_id": item.get("parent_tool_use_id"),
            "iteration": 1,
            "provider": item.get("provider"),
            "ts": time.time(),
            # SDK failure flag, persisted so the red error ring survives reload.
            "is_error": item.get("is_error"),
            # Enriched screenshot refs for the live tool card (TASK-483).
            "attachments": tool_end_attachments,
        }
        tool_end_future = self.publish(tool_end_event)
        if not item.get("is_error"):
            metadata_change = self.changed_files.persist(tool_end_event)
            if metadata_change is not None:
                # Preserve public ordering: tool_end first, then file_changed.
                if tool_end_future is not None:
                    try:
                        tool_end_future.result(timeout=5)
                    except Exception as order_err:
                        log.debug(
                            "tool_end ordering wait failed turn=%s: %s",
                            self.turn_id, order_err,
                        )
                self.publish({
                    "_type": "file_changed",
                    "turn_id": self.turn_id,
                    "trigger_message_id": tool_end_event["trigger_message_id"],
                    "bot_id": self.bot_id,
                    "user_id": self.user_id,
                    "tool_name": tool_end_event["tool_name"],
                    "tool_use_id": item.get("tool_use_id"),
                    "file": metadata_change["file"],
                    "summary": metadata_change["summary"],
                    "ts": time.time(),
                })
        # Update the matching detail entry with result + failure flag so the
        # finalized tool_calls_json keeps the red error ring on reload.
        for td in reversed(self.ctx.tool_call_details_holder):
            if td.get("call_id") == end_cid:
                from agent_bridge.tool_results import payload_from_event

                payload = payload_from_event(
                    item.get("tool_result_payload"), item.get("result", "")
                )
                td["result"] = payload.preview
                td["is_error"] = item.get("is_error")
                break
