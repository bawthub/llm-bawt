"""TASK-622: the per-turn streaming worker thread.

``TurnStreamWorker`` selects the blocking stream, consumes/forwards events,
and finalizes the turn. Native chat-tool iteration is composed through
:class:`NativeToolStream`; agent dispatch stays request-local here. The shared
:class:`TurnStreamContext` carries per-turn state and the sibling publish helpers
are inherited from :class:`TurnStreamPublishMixin`.
"""
from __future__ import annotations

import time

from ..inter_bot_reply_channel import append_request_reply_channel_note
from ..model_switch_notice import apply_model_switch_notice
from ..media.serializers import build_agent_image_manifest
from .chat_stream_worker import consume_stream_chunks, put_queue_item_threadsafe
from .http_errors import client_error_text
from .logging import get_service_logger
from .pending_tool_calls import PendingToolCallCorrelator
from .system_prompt_pin import SessionPinConfirmer, apply_session_pin, pin_backend_kwargs
from .tool_event_interceptor import ToolEventInterceptor
from .tool_execution_stamps import ToolExecutionStamps
from .turn_stream_context import TurnStreamContext
from .turn_stream_finalize import TurnStreamFinalizer
from .turn_stream_publish import TurnStreamPublishMixin
from .turn_usage import TurnUsageCoordinator
from ..working_memory.turn import WorkingMemoryTurn

log = get_service_logger(__name__)


class TurnStreamWorker(TurnStreamPublishMixin):
    """Owns the worker-thread streaming + finalize for one chat turn."""

    def __init__(self, ctx: TurnStreamContext) -> None:
        self.ctx = ctx

    def _stream_to_queue(self):
        # The app already minted this exact conversation proof after reset and
        # binding. Keep it thread/request-local through history + finalization;
        # never attach it to the cached MemoryClient or forward agent headers.
        from ..bot_access.service import bind_service_turn

        ctx = self.ctx
        binding = ctx.thread_binding or {}
        # Only a row ALREADY fetched and owner-checked by SDK binding proves
        # ownership. A turn token minted around an arbitrary request.session_id
        # or a missing row is not enough. This marker is internal, not a field
        # copied from the chat request or stored on a cached client.
        owner = (binding.get("thread_session_id"), ctx.bot_id, ctx.user_id)
        capability = ctx.task_turn_capability if binding.get("_service_owner") == owner else None
        chat_reset = getattr(ctx, "chat_reset", None)
        if chat_reset is not None:
            # Native /new has no agent proof or SDK binding. Its owner-bound
            # rotation already committed; sign only that exact successor.
            capability = chat_reset.service_turn_capability(
                bot_id=ctx.bot_id, user_id=ctx.user_id, turn_id=ctx.turn_log_id,
                trigger_message_id=ctx.trigger_message_id,
            )
        with bind_service_turn(capability):
            return self._stream_to_queue_with_context()

    def _stream_to_queue_with_context(self):
        ctx = self.ctx
        _persist_publish_approval = self._persist_publish_approval
        _persist_publish_await = self._persist_publish_await
        _publish_event_direct = self._publish_event_direct
        from .tool_changed_files import ToolChangedFilesCoordinator
        _changed_files = ToolChangedFilesCoordinator(ctx.svc._turn_log_store.engine)
        _turn_usage = TurnUsageCoordinator(ctx.svc._turn_log_store)
        _enrich_attachment_refs = self._enrich_attachment_refs
        _redis_sub = ctx._redis_sub
        _upstream_model = ctx._upstream_model
        agent_attachments_holder = ctx.agent_attachments_holder
        animation_holder = ctx.animation_holder
        approval_id_holder = ctx.approval_id_holder
        approval_persist_failed_holder = ctx.approval_persist_failed_holder
        assistant_message_id = ctx.assistant_message_id
        attachments_to_persist = ctx.attachments_to_persist
        bot_id = ctx.bot_id
        cancel_event = ctx.cancel_event
        cancelled_holder = ctx.cancelled_holder
        chunk_queue = ctx.chunk_queue
        done_event = ctx.done_event
        full_response_holder = ctx.full_response_holder
        inject_seed_messages = ctx.inject_seed_messages
        is_agent_backend = ctx.is_agent_backend
        llm_bawt = ctx.llm_bawt
        loop = ctx.loop
        media_store = ctx.media_store
        model_alias = ctx.model_alias
        question_id_holder = ctx.question_id_holder
        reasoning_holder = ctx.reasoning_holder
        request = ctx.request
        timing_holder = ctx.timing_holder
        token_usage_holder = ctx.token_usage_holder
        tool_call_details_holder = ctx.tool_call_details_holder
        tool_context_holder = ctx.tool_context_holder
        trigger_message_id = ctx.trigger_message_id
        task_turn_capability = ctx.task_turn_capability
        tts_scrub = ctx.tts_scrub
        tts_scrubber = ctx.tts_scrubber
        turn_log_id = ctx.turn_log_id
        user_attachments = ctx.user_attachments
        user_author = ctx.user_author
        user_id = ctx.user_id
        user_prompt = ctx.user_prompt
        self = ctx.svc
        """Run streaming in a thread and push chunks to the async queue."""
        def _turn_was_aborted() -> bool:
            try:
                current_turn = self._turn_log_store.get_turn(turn_log_id)
                return current_turn is not None and current_turn.status in ("aborted", "cancelled", "cancelling")
            except Exception:
                return False

        def _usage_so_far() -> dict | None:
            return self._resolve_turn_token_usage(llm_bawt, token_usage_holder[0])

        def selected_history_context(phase):
            from contextlib import nullcontext

            selected = getattr(ctx, "selected_history", None)
            if selected is None:
                return nullcontext()  # existing agent/reset scope unchanged
            return selected.history_context(
                request, llm_bawt, phase=phase, bot_id=bot_id, user_id=user_id,
                turn_id=turn_log_id, trigger_message_id=trigger_message_id,
            )

        try:
            # Check if already cancelled before starting
            if cancel_event.is_set():
                cancelled_holder[0] = True
                return

            # Inject client-supplied system context (e.g. HA device list)
            llm_bawt._client_system_context = request.client_system_context
            llm_bawt._ha_mode = request.ha_mode
            # Bot's include_summaries=false overrides request default
            if hasattr(llm_bawt.bot, 'include_summaries') and not llm_bawt.bot.include_summaries:
                llm_bawt._include_summaries = False
            else:
                llm_bawt._include_summaries = request.include_summaries
            # Agent backends use a per-turn user-message prefix for
            # voice mode (chat.agent_voice_prefix), so tts_mode must
            # reflect the UI toggle only — bot.tts_mode is a profile
            # default for chatbots whose system prompt carries the TTS
            # instructions. Mixing both double-doses the constraint and
            # cripples agent tool chaining.
            _is_agent = llm_bawt.client.model_definition.get("type") in (
                "agent_backend", "claude-code",
            )
            if _is_agent:
                llm_bawt._tts_mode = request.tts_mode
            else:
                llm_bawt._tts_mode = request.tts_mode or llm_bawt.bot.tts_mode
            llm_bawt._inject_user_prefix = bool(request.inject_user_prefix)
            # TASK-251: explicit thread selection — set FRESH every turn so a
            # cached instance never leaks a prior turn's thread override into
            # a continuous (no-session_id) request.
            _sid = getattr(request, "session_id", None)
            llm_bawt._session_id_override = (
                _sid.strip() if isinstance(_sid, str) and _sid.strip() else None
            )

            # TASK-214: animations now arrive on the request payload from
            # bawthub (Prisma is the source of truth). llm-bawt no longer
            # owns the catalog. Caller pre-filters to enabled rows.
            #
            # TASK-215: animation selection no longer happens via an
            # injected tool call. The main LLM prompt is left untouched
            # (no "you MUST call trigger_animation" hack). Instead, we
            # run a local embedding classifier on the assistant's
            # response text *after* it finishes streaming — see the hook
            # below, before the sentinel is enqueued. The classifier
            # only runs when an avatar is actually visible so we don't
            # waste CPU on a response no one can see animated.
            _animations: list = list(request.animations or []) if llm_bawt._tts_mode else []
            _avatar_visible: bool = bool(request.avatar_visible) if request.avatar_visible is not None else False
            _run_classifier = bool(
                llm_bawt._tts_mode and _avatar_visible and _animations
            )
            llm_bawt._avatar_visible = _avatar_visible
            log.debug(
                "🎬 tts_mode=%s (req=%s bot=%s) avatar_visible=%s %d animations classifier=%s",
                llm_bawt._tts_mode, request.tts_mode, llm_bawt.bot.tts_mode,
                _avatar_visible, len(_animations), _run_classifier,
            )

            # Use llm_bawt.prepare_messages_for_query to get full context
            # (history from DB + memory + system prompt).
            #
            # TASK-225: ``attachments`` is the tiny ``{asset_id, kind}``
            # JSONB payload persisted on the user-message row.  Bytes
            # for the LLM live on ``user_attachments`` (separate
            # argument) and never touch the DB.
            # TASK-391: agent backends can *see* attached images inline but
            # their shell/tools can't fetch blob: URLs. Build a curlable
            # "Attached Images" manifest and hand it to the agent ONLY, via
            # context_suffix — it rides the outbound user message (like the
            # per-turn agent prefixes) and is NEVER persisted. The durable
            # record of what was attached is the tiny {asset_id, kind} refs
            # on the message row (``attachments`` below); the manifest is
            # just a per-turn, absolute-URL rendering of those same handles
            # so a sibling-container tool can curl them. user_prompt stays
            # clean in history + turn log; non-agent chat is unaffected.
            agent_context_suffix = None
            if is_agent_backend and attachments_to_persist and media_store is not None:
                try:
                    manifest = build_agent_image_manifest(
                        attachments_to_persist,
                        media_store.db,
                        origin=getattr(self.config, "AGENT_ORIGIN", "") or "",
                        public_origin_override=getattr(self.config, "PUBLIC_ORIGIN", None),
                    )
                except Exception as _manifest_err:
                    manifest = ""
                    log.warning(
                        "TASK-391: failed to build attachment manifest: %s",
                        _manifest_err,
                    )
                if manifest:
                    agent_context_suffix = manifest
            # Durable inter-bot delivery: tell the recipient whether its final
            # response reaches the sender. Instruction text, so it rides the
            # same outbound-only suffix and never reaches history or the
            # timeline. A reset follow-up keeps it: bridge_messages swaps only
            # the command span of this outbound message.
            if is_agent_backend:
                agent_context_suffix = append_request_reply_channel_note(
                    ctx.svc, request, agent_context_suffix,
                )
            # TASK-1281: this turn's working-memory section (set fresh; None
            # unless the policy allows the bot) and a scheduler turn's note.
            working_memory = WorkingMemoryTurn.begin(ctx)
            agent_context_suffix = working_memory.extend_suffix(agent_context_suffix)

            with selected_history_context("initial"):
                messages = llm_bawt.prepare_messages_for_query(
                    user_prompt,
                    user_attachments=user_attachments or None,
                    message_id=trigger_message_id,
                    attachments=attachments_to_persist or None,
                    context_suffix=agent_context_suffix,
                    author=user_author,
                )
            outbound_task = (
                ctx.bridge_prompt if ctx.bridge_prompt is not None else
                ctx.chat_reset.follow_up if ctx.chat_reset is not None else None
            )
            if outbound_task is not None:
                from .session_lifecycle.execution import NewSessionExecution

                messages = NewSessionExecution.bridge_messages(
                    messages, outbound_task, user_prompt,
                )

            # TASK-1318: send (and log) the session's pinned system prompt.
            system_prompt_pin = None
            if is_agent_backend:
                messages, system_prompt_pin = apply_session_pin(
                    ctx, llm_bawt.client, messages, turn_id=turn_log_id,
                    sections=working_memory.sections,
                )
                # TASK-1281: floor + notice, committed before the command publishes.
                messages = working_memory.after_pin(messages, system_prompt_pin)
                # TASK-1956: a kept pin may name the model from before a switch.
                messages = apply_model_switch_notice(ctx, llm_bawt.client, messages, system_prompt_pin)

            from .scheduled_run_notice import ScheduledRunNotices

            scheduled_notices = ScheduledRunNotices.prepare(
                self._turn_log_store.engine if is_agent_backend else None,
                bot_id=bot_id, user_id=user_id,
                session_id=(ctx.thread_binding or {}).get("thread_session_id"),
                turn_id=turn_log_id, service=self, request=request, user_prompt=user_prompt,
            )
            messages = scheduled_notices.apply(messages)

            # TASK-215: the "AVATAR ANIMATION: You MUST call trigger_animation"
            # prompt-injection hack has been removed. Animation selection
            # now runs as a post-hoc embedding classifier (see hook after
            # consume_stream_chunks below) — the main LLM call no longer
            # has to manage a tool call on every turn.

            # Backfill prepared messages into the turn log.
            # TASK-501: when the app injected a session seed for this turn,
            # splice it into the logged prompt so the turn log reflects what
            # the harness session actually received: [system, ...seed
            # history..., user]. Additive — nothing dropped. The seed dicts
            # and Message objects both serialize via _message_to_dict.
            _logged_messages = messages
            if inject_seed_messages:
                _system = [m for m in messages if getattr(m, "role", None) == "system"]
                _rest = [m for m in messages if getattr(m, "role", None) != "system"]
                _logged_messages = [*_system, *inject_seed_messages, *_rest]
            self._update_turn_log(
                turn_id=turn_log_id,
                prepared_messages=_logged_messages,
            )

            # Log what we're sending to the LLM (verbose mode)
            log.llm_context(messages)

            # Track when first token arrives
            timing_holder[0] = time.time()

            if cancel_event.is_set():
                cancelled_holder[0] = True
                return

            # Choose streaming method based on whether bot uses tools.
            # Agent-backend models (e.g. claude-code/codex) already handle
            # tool execution in their own bridge/runtime and may report
            # tool_format="none" at the llm-bawt model layer, so they must not
            # be forced through llm-bawt's text/native tool loop here.
            should_use_llm_bawt_tool_streaming = (
                not is_agent_backend
                and llm_bawt.bot.uses_tools
                and (llm_bawt.memory or llm_bawt.home_client or llm_bawt.ha_native_client)
            )
            if ctx.chat_reset is not None and ctx.chat_reset.follow_up is None:
                from .session_lifecycle.receipt import RESET_CONFIRMATION

                # Bare chat commands are app-authored; the normal stream worker
                # still persists their canonical user/assistant/turn rows.
                stream_iter = iter((RESET_CONFIRMATION,))
            elif should_use_llm_bawt_tool_streaming:
                # Check if client supports native streaming with tools (OpenAI)
                use_native_streaming = (
                    llm_bawt.client.supports_native_tools()
                    and llm_bawt.tool_format in ("native", "NATIVE_OPENAI")
                    and hasattr(llm_bawt.client, "stream_with_tools")
                )

                # Resolve per-bot generation parameters
                gen_kwargs = llm_bawt._get_generation_kwargs()

                if use_native_streaming:
                    # Native streaming with tools - streams content AND handles tool calls
                    from .native_tool_stream import NativeToolStream

                    stream_iter = NativeToolStream(
                        ctx, publish_event_direct=_publish_event_direct,
                    ).stream(messages, gen_kwargs)
                else:
                    # Fall back to text-based streaming for non-native models (GGUF, etc.)
                    from ..tools import stream_with_tools
                    log.debug(f"Using stream_with_tools for tool format: {llm_bawt.tool_format}")

                    adapter = getattr(llm_bawt, 'adapter', None)
                    if adapter:
                        log.debug(f"Passing adapter '{adapter.name}' to stream_with_tools")
                    else:
                        log.warning("No adapter found on llm_bawt instance")

                    def stream_fn(msgs, stop_sequences=None):
                        return llm_bawt.client.stream_raw(msgs, stop=stop_sequences, **gen_kwargs)

                    stream_iter = stream_with_tools(
                        messages=messages,
                        stream_fn=stream_fn,
                        memory_client=llm_bawt.memory,
                        profile_manager=llm_bawt.profile_manager,
                        search_client=llm_bawt.search_client,
                        home_client=llm_bawt.home_client,
                        ha_native_client=llm_bawt.ha_native_client,
                        model_lifecycle=llm_bawt.model_lifecycle,
                        config=llm_bawt.config,
                        user_id=llm_bawt.user_id,
                        bot_id=llm_bawt.bot_id,
                        tool_format=llm_bawt.tool_format,
                        adapter=adapter,
                        history_manager=llm_bawt.history_manager,
                        tool_call_details=tool_call_details_holder,
                    )
            else:
                # Resolve per-bot generation parameters
                gen_kwargs = llm_bawt._get_generation_kwargs()
                # Pass adapter stop sequences even without tools
                adapter = getattr(llm_bawt, 'adapter', None)
                adapter_stops = adapter.get_stop_sequences() if adapter else []
                extra_kwargs = {}
                if is_agent_backend:
                    extra_kwargs["turn_execution"] = ctx.execution
                if is_agent_backend and user_attachments:
                    extra_kwargs["attachments"] = user_attachments
                if is_agent_backend:
                    extra_kwargs["ordered_steer_events"] = True
                if is_agent_backend and trigger_message_id:
                    # Forward the frontend user-message UUID to the bridge
                    # so every emitted tool event carries it (frontend
                    # buckets tool activity by trigger_message_id).
                    extra_kwargs["trigger_message_id"] = trigger_message_id
                if is_agent_backend and inject_seed_messages is not None:
                    # [] is an explicit clean reset, distinct from an absent seed.
                    extra_kwargs["inject_messages"] = inject_seed_messages
                if is_agent_backend and ctx.thread_binding:
                    # TASK-252: request-local explicit-thread binding — rides
                    # the kwarg channel so it can never leak between turns.
                    extra_kwargs["thread_binding"] = ctx.thread_binding
                if is_agent_backend and ctx.bridge_reset_action:
                    extra_kwargs["bridge_reset_action"] = ctx.bridge_reset_action
                extra_kwargs.update(pin_backend_kwargs(ctx, system_prompt_pin, agent=is_agent_backend))
                if is_agent_backend and task_turn_capability:
                    # TASK-701: opaque current-turn authority. The client/backend
                    # forwards this request-locally; it is never model-visible.
                    extra_kwargs["task_turn_capability"] = task_turn_capability
                if is_agent_backend and request.inter_bot_bridge_request_id:
                    # Validated server-owned continuation identity. Dropping it
                    # creates a random bridge request that cannot consume the
                    # exact approval grant bound to this continuation.
                    extra_kwargs["bridge_request_id"] = request.inter_bot_bridge_request_id
                stream_iter = llm_bawt.client.stream_raw(
                    messages, stop=adapter_stops or None, **gen_kwargs, **extra_kwargs
                )

            # Wrap stream to publish tool events directly to Redis
            # from this thread (survives SSE generator cancellation).
            _oc_call_index = [0]
            # Stable correlator for in-flight agent tool calls. Claude's
            # tool_use_id is authoritative; name/LIFO fallback covers providers
            # that do not expose one.
            _oc_tool_calls = PendingToolCallCorrelator()
            _exec_stamps = ToolExecutionStamps(_oc_tool_calls, _publish_event_direct, {
                "turn_id": turn_log_id, "trigger_message_id": trigger_message_id,
                "bot_id": bot_id, "user_id": user_id,
            })

            # Fires once, on the SDK/bridge-CONFIRMED first output of this
            # turn, to reap any other still-open turns for this bot.
            _confirmed_start_reaped = [False]
            _pin_confirmer = SessionPinConfirmer(ctx, system_prompt_pin, turn_id=turn_log_id)

            _interceptor = ToolEventInterceptor(
                ctx,
                svc=self,
                publish_event_direct=_publish_event_direct,
                persist_publish_approval=_persist_publish_approval,
                persist_publish_await=_persist_publish_await,
                changed_files=_changed_files,
                turn_usage=_turn_usage,
                pin_confirmer=_pin_confirmer,
                scheduled_notices=scheduled_notices,
                tool_calls=_oc_tool_calls,
                exec_stamps=_exec_stamps,
                call_index=_oc_call_index,
                confirmed_start_reaped=_confirmed_start_reaped,
            )

            # Stream chunks to queue
            cancelled_holder[0] = consume_stream_chunks(
                _interceptor.run(stream_iter),
                cancel_event=cancel_event,
                loop=loop,
                chunk_queue=chunk_queue,
                full_response_holder=full_response_holder,
                stop_on_cancel=not is_agent_backend,
            )
            if cancelled_holder[0]:
                log.info("Generation cancelled - newer request received")

            # TASK-215: post-hoc embedding classifier. Runs only when
            # tts_mode is on, the avatar is actually visible, and we
            # have a non-empty response. Misses (no match above the
            # similarity threshold) leave animation_holder[0] as None —
            # the frontend then stays on its idle animation rather than
            # forcing an awkward fallback gesture.
            if _run_classifier and full_response_holder[0] and not animation_holder[0]:
                try:
                    from .animation_classifier import classify_animation
                    picked = classify_animation(full_response_holder[0], _animations)
                    if picked:
                        animation_holder[0] = picked
                except Exception as cls_err:
                    log.debug("animation classifier failed: %s", cls_err)

            timing_holder[1] = time.time()

        except Exception as e:
            execution = ctx.execution
            if execution is not None and execution.detach_requested():
                # TASK-1286: process shutdown detached this worker. The bridge
                # run is still going and the next process reattaches to it, so
                # write no error and queue nothing; the finalizer skips terminals.
                execution.mark_detached_exit()
                log.info("Turn %s detached for shutdown (%s); left open for reattach",
                         turn_log_id, type(e).__name__)
                return
            externally_aborted = _turn_was_aborted()
            if externally_aborted:
                cancelled_holder[0] = True
            elif not cancel_event.is_set():
                put_queue_item_threadsafe(loop, chunk_queue, e)
            elapsed_ms = (time.time() - timing_holder[0]) * 1000 if timing_holder[0] else None
            if not externally_aborted:
                # Feeds the persisted failure marker; driver text stays in the log.
                error_text = client_error_text(e, f"Turn {turn_log_id} stream failed", "Stream interrupted")
                if is_agent_backend:
                    # TASK-1385: never claim the run ended while the bridge
                    # keeps running it; stop it first and record the ack.
                    from .turn_abort import abort_bridge_run_before_app_error

                    bridge_state = abort_bridge_run_before_app_error(
                        self, turn_log_id, execution, loop)
                    error_text = f"{error_text} [bridge abort: {bridge_state}]"
                self._update_turn_log(
                    turn_id=turn_log_id,
                    status="error",
                    latency_ms=elapsed_ms,
                    response_text=full_response_holder[0] or None,
                    error_text=error_text,
                    token_usage=_usage_so_far(),
                )
        finally:
            if not is_agent_backend and "stream_iter" in locals():
                try:
                    close = getattr(stream_iter, "close", None)
                    if close is not None:
                        close()
                except Exception:
                    log.warning("Native stream close failed for %s", turn_log_id)
            prepared_messages = (
                _logged_messages
                if "_logged_messages" in locals()
                else (messages if "messages" in locals() else [])
            )
            with selected_history_context("final"):
                TurnStreamFinalizer(
                    ctx,
                    publish_event_direct=_publish_event_direct,
                    enrich_attachment_refs=_enrich_attachment_refs,
                ).finalize(prepared_messages=prepared_messages)
