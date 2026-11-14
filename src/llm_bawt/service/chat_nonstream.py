"""Non-streaming chat completion (``stream=false``) for BackgroundService.

Split out of ``background_service.py`` (monolith gate, TASK-1015) without
behavior change. Durable inter-bot fallbacks use the streaming path; this mixin
serves direct ``stream=false`` callers.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Any, AsyncIterator

from agent_bridge.maintenance import is_new_command

from ..bots import get_bot, strip_emotes
from ..inter_bot_reply_channel import append_request_reply_channel_note
from .chat_message_text import last_user_text, message_content_text
from .http_errors import safe_error_text
from .inter_bot_claims import continuation_author, turn_admission_claim, validate_inter_bot_claim
from .ordinary_turn_queue import admit_ordinary_agent_turn
from .turn_admission import turn_start_extensions
from .logging import RequestContext, generate_request_id, get_service_logger
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    UsageInfo,
)

log = get_service_logger(__name__)


class ChatNonStreamMixin:
    """OpenAI-compatible non-streaming completion over the shared turn runtime."""

    async def _publish_nonstream_turn_event(
        self,
        *,
        bot_id: str,
        user_id: str,
        event: dict[str, Any],
    ) -> None:
        """Publish one non-streaming turn event to the unified UI stream."""
        subscriber = getattr(self, "_redis_subscriber", None)
        if subscriber is None:
            return
        if event.get("_type") == "turn_complete":
            # TASK-1461: every turn_complete pushes the committed API balance.
            from .usage.balance import balance_update_for_turn_id

            engine = getattr(getattr(self, "_turn_log_store", None), "engine", None)
            event = {**event, "balance_update": await asyncio.to_thread(
                balance_update_for_turn_id, engine, event["turn_id"])}
        try:
            await subscriber.publish_tool_event(bot_id, user_id, event)
        except Exception as exc:
            log.debug("Non-streaming unified event publish failed: %s", exc)

    # ---- Non-streaming chat completion ----

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse | AsyncIterator:
        from .turn_terminal_follower import TerminalFollowerScope

        terminal_scope = TerminalFollowerScope(self._turn_log_store)
        try:
            return await self._chat_completion(request, terminal_scope=terminal_scope)
        except Exception as exc:
            terminal_scope.fail_setup(safe_error_text(exc, "Turn setup failed"))
            raise
        finally:
            terminal_scope.close_if_unlaunched()

    async def _chat_completion(
        self, request: ChatCompletionRequest, *, terminal_scope,
    ) -> ChatCompletionResponse | AsyncIterator:
        """
        Handle an OpenAI-compatible chat completion request.

        This is the main entry point for the API. It:
        1. Uses llm_bawt's internal history + memory for context
        2. Augments with bot system prompt and memory context (if enabled)
        3. Runs blocking LLM calls in a thread pool
        4. Stores messages and extracts memories for future use
        """
        inter_bot_author = validate_inter_bot_claim(self, request)

        # Create request context for logging
        if request.client_system_context is not None:
            req_path = f"/v1/botchat/{request.bot_id}/{request.user}/chat/completions"
        else:
            req_path = "/v1/chat/completions"
        ctx = RequestContext(
            request_id=generate_request_id(),
            method="POST",
            path=req_path,
            model=request.model,
            bot_id=request.bot_id,
            user_id=request.user,
            stream=False,
        )

        # Log incoming request (verbose mode will show the full payload)
        log.api_request(ctx, request.model_dump(exclude_none=True))

        bot_id = request.bot_id or self._default_bot
        user_id = request.user or self.config.DEFAULT_USER
        from ..message_authorship import AuthorReference
        user_author = inter_bot_author or continuation_author(request) or AuthorReference.user(user_id)
        local_mode = not request.augment_memory

        # Resolve model using shared bot/config logic
        # Agent-backend bots resolve to their virtual model (e.g. "claude-code")
        try:
            model_alias, model_warnings = self._resolve_request_model(request.model, bot_id, local_mode)
        except Exception as e:
            log.api_error(ctx, str(e), 400)
            raise

        ctx.model = model_alias

        # Debug: Show memory settings
        log.debug(
            f"Memory settings: augment_memory={request.augment_memory}, "
            f"local_mode={local_mode}, bot={bot_id}, user={user_id}"
        )

        # Get cached LLMBawt instance
        llm_bawt = self._get_llm_bawt(model_alias, bot_id, user_id, local_mode)

        # Use the same last-user/text-part extraction as reset receipts.
        user_prompt = last_user_text(request.messages) or ""

        if not user_prompt:
            raise ValueError("No user message found in request")

        # Start generation lifecycle.
        # Agent backends (claude-code / codex) can run independent requests
        # concurrently, so avoid the per-bot cancellation gate used for local
        # single-model execution.
        model_type = llm_bawt.client.model_definition.get("type", "")
        is_agent_backend = model_type in ("agent_backend", "claude-code")

        # Do not cancel the previous chat generation until this turn has been
        # admitted. Agent turns have independent execution handles.
        cancel_event = threading.Event()
        done_event = threading.Event()
        _reserved_turn_id = getattr(request, "inter_bot_turn_id", None)
        turn_log_id = (
            _reserved_turn_id.strip()
            if isinstance(_reserved_turn_id, str) and _reserved_turn_id.strip()
            else f"turn-{uuid.uuid4().hex}"
        )

        # TASK-303: Extract or generate a stable user-message id so the
        # persisted user message and the turn log share the same identity.
        # This mirrors the streaming path (chat_streaming.py:814).
        trigger_message_id = getattr(request, "user_message_id", None) or str(uuid.uuid4())
        # Canonical assistant-row id (frontend-minted) so live bubble == reloaded
        # row (single bubble). None on server-originated turns → server mints one.
        _amid = getattr(request, "assistant_message_id", None)
        assistant_message_id = (
            _amid.strip()
            if isinstance(_amid, str) and _amid.strip()
            else str(uuid.uuid4())
        )

        # Background/durable callers have no originating HTTP stream. Publish the
        # same canonical turn lifecycle the chat UI already consumes so the target
        # transcript is visible live instead of appearing only after a reload.
        publish_nonstream_lifecycle = inter_bot_author is not None
        # Persist turn log immediately so the user's prompt is recorded
        # even if the backend times out or errors before responding.
        admission_claim = turn_admission_claim(self, request)
        decision = admit_ordinary_agent_turn(
            self, llm_bawt.bot, request, is_agent_backend=is_agent_backend,
            admission_claim=admission_claim, bot_id=bot_id, user_id=user_id,
            user_prompt=user_prompt, trigger_message_id=trigger_message_id,
        )
        if decision is not None:
            return decision.receipt()
        from .turn_terminal_admission import direct_protocol_options
        protocol_options = await direct_protocol_options(
            self.config, bot_id, request, user_prompt, admission_claim,
        )
        admission_lease = self._persist_turn_log(
            turn_id=turn_log_id, request_id=ctx.request_id, path=ctx.path,
            stream=False, model=model_alias, bot_id=bot_id, user_id=user_id,
            status="pending", latency_ms=None, user_prompt=user_prompt,
            prepared_messages=[], response_text="", trigger_message_id=trigger_message_id,
            assistant_message_id=assistant_message_id, admission_claim=admission_claim,
            **protocol_options,
            request_extensions=turn_start_extensions(
                request, is_agent_backend=is_agent_backend, admission_claim=admission_claim,
            ),
        )
        if admission_lease is not None and admission_lease.assistant_message_id:
            assistant_message_id = admission_lease.assistant_message_id
        terminal_scope.admitted(
            admission_lease, turn_id=turn_log_id, cancel_event=cancel_event,
            is_agent=is_agent_backend,
            on_close=lambda: self._end_generation(cancel_event, done_event, bot_id),
        )
        if is_agent_backend:
            from .turn_task_attribution import stamp_turn_task
            await stamp_turn_task(
                self._turn_log_store, turn_id=turn_log_id, bot_id=bot_id, user_id=user_id,
            )
        chat_reset = None
        if not is_agent_backend:
            try:
                from .session_lifecycle.execution import NewSessionExecution

                chat_reset = NewSessionExecution.prepare_chat(
                    self, llm_bawt, request, bot_id=bot_id,
                    user_id=user_id, user_prompt=user_prompt,
                )
                if chat_reset is not None:
                    request = request.model_copy(update={"session_id": chat_reset.session_id})
            except Exception as exc:
                self._update_turn_log(
                    turn_id=turn_log_id, status="error",
                    error_text=f"Chat reset setup failed: {exc}",
                )
                raise
            if terminal_scope.follower is None:
                cancel_event, done_event = await self._start_generation(bot_id)
            else:
                cancel_event, done_event = await self._start_generation(
                    bot_id, cancel_event=cancel_event, done_event=done_event,
                )

        from .session_lifecycle.execution import NativeSelectedHistory

        selected_history = NativeSelectedHistory.take(
            request, llm_bawt, bot_id=bot_id, user_id=user_id,
            turn_id=turn_log_id, trigger_message_id=trigger_message_id,
            is_agent_backend=is_agent_backend, admission_claim=admission_claim,
        )

        # TASK-709: resolve the durable session id BEFORE publishing turn_start
        # so external clients can route the lifecycle stream to the correct
        # thread without a follow-up DB fetch.
        #
        # Priority:
        #   1. ``request.session_id`` — for reset-policy inter-bot deliveries
        #      the dispatcher wrote the NEW (target) session id there via
        #      ``agent_context.rotate_delivery_session``. For explicit thread
        #      selection from a chat client, that's also where the target
        #      lives. This is always the correct routing target.
        #   2. ``_resolve_active_thread_binding`` fallback for cases without
        #      an explicit selection.
        #
        # Do NOT use ``inter_bot_seed_session_id`` for routing — it names the
        # ARCHIVED (old) session that ``build_context_seed`` reads history
        # FROM. Routing turn_start there would send events to the dead thread
        # and the pane on the fresh target session would gate them out as
        # "wrong thread" (the exact cross-thread failure TASK-709 forbids).
        #
        # Best-effort: a resolution failure emits ``session_id: null`` for
        # backward-compat.
        # NB: nonlocal-mutable so post-rotation refresh inside ``_do_query``
        # can update it for the ``turn_complete`` emits (see F4 note there).
        resolved_session_id: str | None = chat_reset.session_id if chat_reset else None
        if is_agent_backend:
            _sid_req = getattr(request, "session_id", None)
            if isinstance(_sid_req, str) and _sid_req.strip():
                resolved_session_id = _sid_req.strip()
            else:
                try:
                    _binding = self._resolve_active_thread_binding(llm_bawt)
                    if _binding:
                        _rsid = str(_binding.get("thread_session_id") or "").strip()
                        resolved_session_id = _rsid or None
                except Exception as _sid_err:
                    log.warning(
                        "TASK-709: pre-emit session resolve failed for %s/%s: %s",
                        bot_id, user_id, _sid_err,
                    )

        if publish_nonstream_lifecycle:
            from ..entity_presentation import EntityPresentationResolver

            assistant_author = AuthorReference.bot(bot_id)
            author_presentations = EntityPresentationResolver(
                self._turn_log_store.engine
            ).resolve_many_safe([user_author, assistant_author])
            await self._publish_nonstream_turn_event(
                bot_id=bot_id,
                user_id=user_id,
                event={
                    "_type": "turn_start",
                    "turn_id": turn_log_id,
                    "trigger_message_id": trigger_message_id,
                    "assistant_message_id": assistant_message_id,
                    "bot_id": bot_id,
                    "user_id": user_id,
                    # TASK-709: see resolve block above.
                    "session_id": resolved_session_id,
                    "role": "user",
                    "content": user_prompt,
                    "author": author_presentations[
                        (user_author.entity_type, user_author.entity_id)
                    ],
                    "assistant_author": author_presentations[
                        (assistant_author.entity_type, assistant_author.entity_id)
                    ],
                    "attachments": [],
                    "ts": time.time(),
                },
            )

        # TASK-1015: register the request-local execution handle exactly like
        # the streaming path so Stop/steer/reaper see this live owner and the
        # bridge binds agent_session_key + agent_request_id before dispatch.
        turn_execution = terminal_scope.follower.execution if terminal_scope.follower is not None else None
        if is_agent_backend and turn_execution is None:
            from .turn_execution import TurnExecution, turn_executions

            turn_execution = TurnExecution(
                turn_log_id, cancel_event, is_agent=True, store=self._turn_log_store,
            )
            turn_executions.register(turn_execution)

        try:
            # Run the blocking query in single-thread executor
            loop = asyncio.get_event_loop()
            llm_start_time = time.time()
            cancelled = False
            terminal_snapshot = None
            terminal_finalizing = False
            from .turn_terminal_finalize import NonstreamTerminalMetadata

            terminal_metadata = NonstreamTerminalMetadata(
                self, request, turn_id=turn_log_id, trigger_message_id=trigger_message_id,
                bot_id=bot_id, user_id=user_id, loop=loop,
            ) if terminal_scope.follower is not None else None

            def commit_protocol_terminal(
                *, response=None, tool_context="", tool_calls=None,
                status="ok", end_reason="stop", error_text=None,
            ):
                nonlocal terminal_snapshot
                from .turn_terminal_finalize import (
                    commit_nonstream_terminal, project_execution_terminal_from_worker,
                )

                follower = terminal_scope.follower
                fields = terminal_metadata.frozen_fields()
                if status == "ok":
                    end_reason = (
                        "question" if fields["question_id"] else
                        "approval" if fields["approval_id"] else
                        "approval_persist_failed" if fields["approval_persist_failed"] else end_reason
                    )
                result = commit_nonstream_terminal(
                    self, llm_bawt, follower, response_text=response,
                    tool_context=tool_context, tool_call_details=tool_calls,
                    status=status, end_reason=end_reason, error_text=error_text,
                    token_usage=terminal_metadata.token_usage,
                    attachments=terminal_metadata.frozen_attachments(),
                    reasoning=terminal_metadata.reasoning or None,
                    completion_fields=fields,
                )
                terminal_snapshot = result.snapshot
                try:
                    project_execution_terminal_from_worker(
                        follower.protocol, turn_log_id, getattr(self, "_redis_subscriber", None), loop,
                        publish_lock=terminal_metadata.ctx.publish_lock,
                    )
                except Exception:
                    log.warning(
                        "Committed terminal %s awaits projection recovery", turn_log_id, exc_info=True,
                    )
                return result.snapshot["response_text"]

            bridge_event_callback = None
            if publish_nonstream_lifecycle:
                from .tool_event_coordinator import ToolEventCoordinator

                tool_events = ToolEventCoordinator(self._turn_log_store.engine)
                tool_call_ids: dict[str, str] = {}
                tool_iteration = 0

                def _bridge_event_callback(item: dict[str, Any]) -> None:
                    nonlocal tool_iteration
                    event_type = item.get("event")
                    if event_type == "llm_response":
                        # TASK-2110: same durable per-response record as streaming.
                        public = tool_events.response({
                            "turn_id": turn_log_id,
                            "trigger_message_id": trigger_message_id,
                            "bot_id": bot_id,
                            "user_id": user_id,
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
                        if public is not None:
                            asyncio.run_coroutine_threadsafe(
                                self._publish_nonstream_turn_event(
                                    bot_id=bot_id, user_id=user_id, event=public
                                ),
                                loop,
                            ).result(timeout=5)
                        return
                    if event_type not in {"tool_call", "tool_result"}:
                        return
                    tool_use_id = str(item.get("tool_use_id") or "").strip() or None
                    call_key = tool_use_id or f"iteration-{tool_iteration + 1}"
                    if event_type == "tool_call":
                        tool_iteration += 1
                        call_id = tool_use_id or f"call-{uuid.uuid4().hex}"
                        tool_call_ids[call_key] = call_id
                        public = tool_events.start({
                            "_type": "tool_event",
                            "event": "tool_start",
                            "turn_id": turn_log_id,
                            "trigger_message_id": trigger_message_id,
                            "bot_id": bot_id,
                            "user_id": user_id,
                            "tool_name": item.get("name") or "unknown",
                            "arguments": item.get("arguments") or {},
                            "call_id": call_id,
                            "tool_use_id": tool_use_id,
                            "parent_tool_use_id": item.get("parent_tool_use_id"),
                            "iteration": tool_iteration,
                            "provider": item.get("provider"),
                            "response_id": item.get("response_id"),
                            "ts": time.time(),
                        })
                    else:
                        call_id = tool_call_ids.get(call_key)
                        public = tool_events.end({
                            "_type": "tool_event",
                            "event": "tool_end",
                            "turn_id": turn_log_id,
                            "trigger_message_id": trigger_message_id,
                            "bot_id": bot_id,
                            "user_id": user_id,
                            "tool_name": item.get("name") or "unknown",
                            "arguments": item.get("arguments") or {},
                            "call_id": call_id,
                            "tool_use_id": tool_use_id,
                            "parent_tool_use_id": item.get("parent_tool_use_id"),
                            "iteration": tool_iteration or 1,
                            "provider": item.get("provider"),
                            "result": item.get("result", ""),
                            "tool_result_payload": item.get("tool_result_payload"),
                            "is_error": item.get("is_error"),
                            "ts": time.time(),
                        })
                    publish_future = asyncio.run_coroutine_threadsafe(
                        self._publish_nonstream_turn_event(
                            bot_id=bot_id, user_id=user_id, event=public
                        ),
                        loop,
                    )
                    # Preserve bridge event order: a fast final tool must reach
                    # Redis before the worker can publish turn_complete.
                    publish_future.result(timeout=5)

                bridge_event_callback = _bridge_event_callback

            def _do_query_with_context():
                nonlocal cancelled, terminal_finalizing
                if terminal_scope.follower is not None:
                    terminal_scope.follower.check()
                # Check if already cancelled before starting
                if cancel_event.is_set():
                    cancelled = True
                    return ""

                # Inject client-supplied system context (e.g. HA device list)
                llm_bawt._client_system_context = request.client_system_context
                llm_bawt._ha_mode = request.ha_mode
                llm_bawt._include_summaries = request.include_summaries
                # Agent backends use a per-turn user-message prefix for
                # voice mode — bot.tts_mode is a chatbot-only default.
                _is_agent = llm_bawt.client.model_definition.get("type") in (
                    "agent_backend", "claude-code",
                )
                if _is_agent:
                    llm_bawt._tts_mode = request.tts_mode
                else:
                    llm_bawt._tts_mode = request.tts_mode or llm_bawt.bot.tts_mode
                llm_bawt._inject_user_prefix = bool(request.inject_user_prefix)
                # TASK-251: explicit thread selection — set FRESH every turn
                # (cached instance; a stale override must never leak into a
                # continuous request).
                _sid = getattr(request, "session_id", None)
                llm_bawt._session_id_override = (
                    _sid.strip() if isinstance(_sid, str) and _sid.strip() else None
                )
                # TASK-1281: the section rides stream turns only (where the pin
                # and notice run); clear any render a stream turn left behind.
                llm_bawt._working_memory_section = None

                # Resolve the request-local thread binding before seed
                # assembly. Explicit selection wins; otherwise bind the current
                # ACTIVE thread so all resume decisions use canonical metadata.
                thread_binding = None
                delivery_new_id = None
                if _is_agent:
                    thread_binding = self._bind_agent_thread(llm_bawt, request)
                    if thread_binding is None:
                        thread_binding = self._resolve_active_thread_binding(llm_bawt)
                    from .agent_new_rotation import prepare_delivery_new_binding
                    thread_binding, delivery_new_id = prepare_delivery_new_binding(
                        self, llm_bawt, request, bot_id=bot_id, user_id=user_id,
                        user_prompt=user_prompt, thread_binding=thread_binding,
                        backend_name=str(getattr(llm_bawt.bot, "agent_backend", "") or ""),
                    )

                from contextlib import nullcontext
                from ..bot_access.service import bind_service_turn

                def history_context(binding, capability, *, phase):
                    if not _is_agent:
                        if selected_history is not None:
                            return selected_history.history_context(
                                request, llm_bawt, phase=phase,
                                bot_id=bot_id, user_id=user_id, turn_id=turn_log_id,
                                trigger_message_id=trigger_message_id,
                            )
                        return nullcontext()  # preserve the native reset scope
                    binding = binding or {}
                    owner = (binding.get("thread_session_id"), bot_id, user_id)
                    return bind_service_turn(
                        capability if binding.get("_service_owner") == owner else None
                    )

                # Nonstream user history precedes reset preparation and agent
                # proof minting. The SDK binding has ALREADY fetched/verified
                # this exact row; sign that provenance without moving rotation
                # or trusting a selected id. This early proof is history-only.
                initial_capability = None
                initial_binding = thread_binding or {}
                initial_sid = initial_binding.get("thread_session_id")
                if (_is_agent and initial_sid and initial_binding.get("_service_owner")
                        == (initial_sid, bot_id, user_id)):
                    from ..task_turn_context import mint_for_agent_turn

                    initial_capability = mint_for_agent_turn(
                        backend=str(getattr(llm_bawt.bot, "agent_backend", "") or ""),
                        session_id=initial_sid, turn_id=turn_log_id,
                        trigger_message_id=trigger_message_id, bot_id=bot_id, user_id=user_id,
                    )

                def prepare_user_messages(binding, capability):
                    # The reply note stays outbound-only, never in stored history.
                    with history_context(binding, capability, phase="initial"):
                        return llm_bawt.prepare_messages_for_query(
                            user_prompt, message_id=trigger_message_id, author=user_author,
                            context_suffix=(
                                append_request_reply_channel_note(self, request, None)
                                if _is_agent else None
                            ),
                            **({
                                "history_write_session_id": resolved_session_id,
                            } if terminal_scope.follower is not None else {}),
                        )

                prepared_messages = None
                if terminal_scope.follower is None:
                    # Legacy-null ordering stays exactly as before. Protocol
                    # writes must instead wait for final post-reset binding.
                    prepared_messages = prepare_user_messages(thread_binding, initial_capability)
                    log.llm_context(prepared_messages)

                from .session_lifecycle.execution import NewSessionExecution, PreparedNewSession

                reset_policy = getattr(request, "inter_bot_session_policy", None)
                prepared_reset = (
                    NewSessionExecution.prepare_agent(
                        self, llm_bawt, request, bot_id=bot_id,
                        user_prompt=user_prompt, model_alias=model_alias,
                        thread_binding=thread_binding, delivery_id=delivery_new_id,
                        backend_name=str(getattr(llm_bawt.bot, "agent_backend", "") or ""),
                    ) if _is_agent else PreparedNewSession(None, None, None)
                )
                inject_seed_messages = prepared_reset.seed_messages
                thread_binding = prepared_reset.thread_binding

                # TASK-709 F4: refresh the outer ``resolved_session_id`` from
                # the post-rotation binding so ``turn_complete`` emits point at
                # the CURRENT (post-``/new``-rotate, post-``_bind_agent_thread``)
                # session id, not the stale pre-emit value from turn_start.
                # ``turn_start`` staleness on ``/new`` is unavoidable (emit
                # happens before rotation); ``turn_complete`` accuracy is
                # cheap here and matches how ``turn_stream_finalize`` already
                # threads its session id.
                nonlocal resolved_session_id
                _final_binding = thread_binding
                if _is_agent and _final_binding:
                    _final_sid = str(
                        _final_binding.get("thread_session_id") or ""
                    ).strip()
                    if _final_sid:
                        resolved_session_id = _final_sid
                        if (
                            not reset_policy
                            and (_final_binding.get("explicit_thread")
                                 or not is_new_command(user_prompt))
                        ):
                            resumption = self._turn_log_store.accept_session_binding(
                                turn_id=turn_log_id, bot_id=bot_id, user_id=user_id,
                                session_id=_final_sid, admission_claim=admission_claim,
                                record_resumption=not is_new_command(user_prompt),
                            )
                            if isinstance(resumption, dict):
                                from .session_resumption import event as resumption_event

                                try:
                                    asyncio.run_coroutine_threadsafe(
                                        self._publish_nonstream_turn_event(
                                            bot_id=bot_id, user_id=user_id,
                                            event=resumption_event(resumption, bot_id=bot_id,
                                                user_id=user_id, trigger_message_id=trigger_message_id),
                                        ), loop,
                                    ).result(timeout=5)
                                except Exception as exc:
                                    # Optional delivery cannot undo acceptance or
                                    # stop inference; history owns the receipt.
                                    log.debug("Session-resumption event publish failed: %s", exc)

                from .session_continuity import record_session_continuity
                if _is_agent:
                    record_session_continuity(llm_bawt, thread_binding, inject_seed_messages,
                                              bot_id=bot_id, user_id=user_id,
                                              history_scope=prepared_reset.history_scope,
                                              report_empty=prepared_reset.report_empty_seed)
                task_turn_capability = None
                if _is_agent:
                    from ..task_turn_context import mint_for_agent_turn

                    task_turn_capability = mint_for_agent_turn(
                        backend=str(getattr(llm_bawt.bot, "agent_backend", "") or ""),
                        session_id=resolved_session_id,
                        turn_id=turn_log_id,
                        trigger_message_id=trigger_message_id,
                        bot_id=bot_id,
                        user_id=user_id,
                    )
                if terminal_scope.follower is not None:
                    from .turn_terminal_sessions import bind_terminal_session

                    resolved_session_id = bind_terminal_session(
                        terminal_scope.follower, llm_bawt, bot_id=bot_id, user_id=user_id,
                        resolved_session_id=(
                            resolved_session_id if _is_agent
                            else getattr(request, "session_id", None)
                        ),
                        is_agent=_is_agent,
                    )
                    prepared_messages = prepare_user_messages(thread_binding, task_turn_capability)
                    log.llm_context(prepared_messages)

                # Persist the original /new command; send only its parsed task
                # to the bridge, without mutating cached history or model text.
                outbound_task = (
                    prepared_reset.bridge_prompt if _is_agent else
                    chat_reset.follow_up if chat_reset else None
                )
                outbound_messages = NewSessionExecution.bridge_messages(
                    prepared_messages, outbound_task, user_prompt,
                )
                from .scheduled_run_notice import ScheduledRunNotices

                scheduled_notices = ScheduledRunNotices.prepare(
                    self._turn_log_store.engine if _is_agent else None,
                    bot_id=bot_id, user_id=user_id, session_id=resolved_session_id,
                    turn_id=turn_log_id, service=self, request=request, user_prompt=user_prompt,
                )
                outbound_messages = scheduled_notices.apply(outbound_messages)

                def notice_bridge_event(item):
                    scheduled_notices.observe(item)
                    if terminal_scope.follower is not None:
                        terminal_scope.follower.check()
                        terminal_metadata.observe(item, session_id=resolved_session_id)
                    if bridge_event_callback is not None:
                        bridge_event_callback(item)

                if chat_reset is not None and chat_reset.follow_up is None:
                    from .session_lifecycle.receipt import RESET_CONFIRMATION

                    response, tool_context, tool_call_details = RESET_CONFIRMATION, "", []
                else:
                    terminal_scope.begin_inference()
                    response, tool_context, tool_call_details = llm_bawt.execute_llm_query(
                        outbound_messages,
                        plaintext_output=True,
                        stream=False,
                        inject_messages=inject_seed_messages,
                        thread_binding=thread_binding,
                        bridge_reset_action=prepared_reset.bridge_reset_action,
                        bridge_request_id=getattr(request, "inter_bot_bridge_request_id", None),
                        bridge_timeout_seconds=getattr(request, "inter_bot_timeout_seconds", None),
                        bridge_event_callback=notice_bridge_event if _is_agent else bridge_event_callback,
                        turn_execution=turn_execution,
                        task_turn_capability=task_turn_capability,
                    )
                    if terminal_scope.follower is not None:
                        terminal_scope.follower.check()
                    scheduled_notices.observe(response)
                    # Agent bridges report their calls through the bridge
                    # event path; chat bots ran llm-bawt's own tool loop,
                    # which nothing else records (TASK-1775).
                    if tool_call_details and not _is_agent:
                        from .tool_event_coordinator import ToolEventCoordinator

                        try:
                            ToolEventCoordinator(self._turn_log_store.engine).record_completed(
                                tool_call_details,
                                turn_id=turn_log_id,
                                trigger_message_id=trigger_message_id,
                                bot_id=bot_id,
                                user_id=user_id,
                            )
                        except Exception:
                            log.exception("Failed to record tool calls for turn %s", turn_log_id)

                # Splice the injected seed into the logged prompt so the turn
                # log reflects what the harness session received: [system,
                # ...seed history..., user]. Additive — nothing dropped.
                _logged_messages = outbound_messages
                if inject_seed_messages:
                    _sys = [m for m in outbound_messages if getattr(m, "role", None) == "system"]
                    _rest = [m for m in outbound_messages if getattr(m, "role", None) != "system"]
                    _logged_messages = [*_sys, *inject_seed_messages, *_rest]

                # Check if cancelled during generation
                if cancel_event.is_set():
                    log.info("Generation cancelled - newer request received")
                    cancelled = True
                    if terminal_scope.follower is None:
                        return ""

                # Reset preparation may have rebound the session. Use ONLY
                # the final binding's verified owner and final agent proof;
                # never carry the initial history authority across rotation.
                if terminal_scope.follower is not None:
                    terminal_finalizing = True
                    return commit_protocol_terminal(
                        response=response, tool_context=tool_context, tool_calls=tool_call_details,
                        status="aborted" if cancelled or terminal_metadata.cancelled else "ok",
                        end_reason="aborted" if cancelled or terminal_metadata.cancelled else "stop",
                    )
                with history_context(thread_binding, task_turn_capability, phase="final"):
                    self._finalize_turn(
                        llm_bawt=llm_bawt,
                        turn_id=turn_log_id,
                        response_text=response,
                        tool_context=tool_context,
                        tool_call_details=tool_call_details,
                        prepared_messages=_logged_messages,
                        user_prompt=user_prompt,
                        model=model_alias,
                        bot_id=bot_id,
                        user_id=user_id,
                        elapsed_ms=(time.time() - llm_start_time) * 1000,
                        stream=False,
                        assistant_message_id=assistant_message_id,
                    )

                return response

            def _do_query():
                nonlocal terminal_finalizing
                # ContextVars must be bound INSIDE the executor thread. Native
                # reset provenance covers its query; agent history separately
                # scopes verified initial/final bindings, never a selected id.
                from ..bot_access.service import bind_service_turn

                capability = chat_reset.service_turn_capability(
                    bot_id=bot_id, user_id=user_id, turn_id=turn_log_id,
                    trigger_message_id=trigger_message_id,
                ) if chat_reset is not None else None
                try:
                    with bind_service_turn(capability):
                        result = _do_query_with_context()
                        if terminal_scope.follower is not None and terminal_snapshot is None:
                            # The worker observed cancellation before setup.
                            terminal_finalizing = True
                            return commit_protocol_terminal(status="aborted", end_reason="aborted")
                        return result
                except Exception as exc:
                    # Nonstream agent setup runs in this real worker. Only
                    # it can certify the provider has not been entered; an
                    # HTTP cancellation/error must never make that claim.
                    terminal_scope.fail_setup(
                        safe_error_text(exc, "Turn setup failed"), from_worker=True,
                    )
                    if (terminal_scope.follower is not None and terminal_snapshot is None
                            and not terminal_finalizing):
                        # A failed commit may have reached the DB. Reinterpretation
                        # as an empty error would discard a completed provider reply.
                        try:
                            if terminal_scope._inference_started:
                                commit_protocol_terminal(
                                    status="error", end_reason="upstream_error",
                                    error_text=safe_error_text(exc, "Turn failed"),
                                )
                            else:
                                # fail_setup already owns the commit. Only
                                # project it; never reinterpret its outcome.
                                from .turn_terminal_finalize import project_execution_terminal_from_worker

                                project_execution_terminal_from_worker(
                                    terminal_scope.follower.protocol, turn_log_id,
                                    getattr(self, "_redis_subscriber", None), loop,
                                    publish_lock=terminal_metadata.ctx.publish_lock,
                                )
                        except Exception:
                            log.exception("Protocol nonstream terminal failed for %s", turn_log_id)
                    raise

            try:
                # In-process GPU inference is gone (local models run in
                # local_model_bridge, TASK-276/278), so every model now streams
                # on the default thread pool — no single-worker serialization.
                response_text = await terminal_scope.launch_worker(loop, _do_query)
            except Exception as e:
                elapsed_ms = (time.time() - llm_start_time) * 1000
                if terminal_scope.follower is None:
                    self._update_turn_log(
                        turn_id=turn_log_id,
                        status="error",
                        latency_ms=elapsed_ms,
                        error_text=str(e),
                    )
                if publish_nonstream_lifecycle and terminal_scope.follower is None:
                    await self._publish_nonstream_turn_event(
                        bot_id=bot_id,
                        user_id=user_id,
                        event={
                            "_type": "turn_complete",
                            "turn_id": turn_log_id,
                            "assistant_message_id": assistant_message_id,
                            "bot_id": bot_id,
                            "user_id": user_id,
                            # TASK-709: match turn_start's session id so a
                            # receiving window can finalize the correct thread.
                            "session_id": resolved_session_id,
                            "status": "cancelled",
                            "end_reason": "error",
                            "model": model_alias,
                            "ts": time.time(),
                        },
                    )
                raise
            llm_elapsed_ms = (time.time() - llm_start_time) * 1000

            # If cancelled, return empty response (the new request will handle it)
            if cancelled and terminal_scope.follower is None:
                self._update_turn_log(
                    turn_id=turn_log_id,
                    status="cancelled",
                    latency_ms=llm_elapsed_ms,
                    response_text="",
                    end_reason="aborted",
                )
                if publish_nonstream_lifecycle:
                    await self._publish_nonstream_turn_event(
                        bot_id=bot_id,
                        user_id=user_id,
                        event={
                            "_type": "turn_complete",
                            "turn_id": turn_log_id,
                            "assistant_message_id": assistant_message_id,
                            "bot_id": bot_id,
                            "user_id": user_id,
                            # TASK-709: match turn_start's session id.
                            "session_id": resolved_session_id,
                            "status": "cancelled",
                            "end_reason": "aborted",
                            "model": model_alias,
                            "ts": time.time(),
                        },
                    )
                return ChatCompletionResponse(
                    model=model_alias,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=""),
                            finish_reason="cancelled",
                        )
                    ],
                    usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                )

            if publish_nonstream_lifecycle and terminal_scope.follower is None:
                token_usage = None
                get_token_usage = getattr(llm_bawt.client, "get_token_usage", None)
                if callable(get_token_usage):
                    token_usage = get_token_usage()
                if response_text:
                    await self._publish_nonstream_turn_event(
                        bot_id=bot_id,
                        user_id=user_id,
                        event={
                            "_type": "text_delta",
                            "turn_id": turn_log_id,
                            "trigger_message_id": trigger_message_id,
                            "bot_id": bot_id,
                            "user_id": user_id,
                            "delta": response_text,
                            "text_offset": 0,
                            "ts": time.time(),
                        },
                    )
                await self._publish_nonstream_turn_event(
                    bot_id=bot_id,
                    user_id=user_id,
                    event={
                        "_type": "turn_complete",
                        "turn_id": turn_log_id,
                        "assistant_message_id": assistant_message_id,
                        "bot_id": bot_id,
                        "user_id": user_id,
                        # TASK-709: match turn_start's session id.
                        "session_id": resolved_session_id,
                        "status": "completed",
                        "end_reason": "stop",
                        # TASK-779: server-authoritative final text. The
                        # nonstream inter-bot path emits a single aggregate
                        # text_delta immediately before turn_complete; if
                        # that delta never lands (packed flush, filter
                        # race, subscriber gap), the frontend has no live
                        # partial to commit and the reply vanishes at
                        # finalize. Carrying response_text here lets the
                        # commit fall back to the same bytes the DB
                        # persisted. Additive optional field — legacy
                        # consumers ignore it.
                        "response_text": response_text,
                        "token_usage": token_usage,
                        "model": model_alias,
                        "ts": time.time(),
                    },
                )
        finally:
            if turn_execution is not None and terminal_scope.follower is None:
                from .turn_execution import turn_executions

                turn_executions.remove(turn_log_id)
            if terminal_scope.follower is None:
                self._end_generation(cancel_event, done_event, bot_id)

        # Post-process for voice_optimized bots (strip emotes for TTS)
        bot = get_bot(bot_id)
        if bot and bot.voice_optimized:
            original_len = len(response_text)
            response_text = strip_emotes(response_text)
            if len(response_text) != original_len:
                log.debug(f"Stripped emotes for TTS: {original_len} -> {len(response_text)} chars")

        # Estimate token counts (rough approximation: 1 token ≈ 4 characters)
        prompt_text = " ".join(message_content_text(m.content) or "" for m in request.messages)
        prompt_tokens = len(prompt_text) // 4
        completion_tokens = len(response_text) // 4
        total_tokens = prompt_tokens + completion_tokens

        # Log response (verbose shows content summary with tokens/sec)
        log.llm_response(response_text, tokens=completion_tokens, elapsed_ms=llm_elapsed_ms)
        log.api_response(ctx, status=200, tokens=total_tokens)

        # Build response
        response = ChatCompletionResponse(
            model=model_alias,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason=(
                        "cancelled" if terminal_snapshot is not None
                        and terminal_snapshot["completion"]["status"] == "cancelled" else "stop"
                    ),
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )

        return response
