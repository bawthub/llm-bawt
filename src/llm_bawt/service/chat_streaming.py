"""Chat completion streaming mixin (native + OpenClaw bridge).

Extracted from background_service.py — handles SSE streaming for
chat completions including tool call handling and bridge integration.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from typing import Any, AsyncIterator

from ..approval_policies import ApprovalPersistError
from ..bots import (
    StreamingEmoteFilter,
    StreamingTTSScrubber,
    should_scrub_for_tts,
)
from ..media.serializers import build_agent_image_manifest
from .chat_stream_worker import consume_stream_chunks, put_queue_item_threadsafe
from .chat_streaming_bridge import ChatStreamingBridgeMixin
from .inter_bot_claims import validate_inter_bot_claim
from .logging import RequestContext, generate_request_id, get_service_logger
from .schemas import ChatCompletionRequest
from .stream_request_attachments import prepare_stream_request_attachments
from .tool_event_coordinator import ToolEventCoordinator
from .turn_stream_context import TurnStreamContext
from .turn_stream_worker import TurnStreamWorker

log = get_service_logger(__name__)

# TASK-622: the streaming coalescing (_TEXT_DELTA_FLUSH_CHARS) and approval-
# persist retry knobs (_APPROVAL_PERSIST_*) now live in turn_stream_context.py
# alongside the worker/publisher that use them.


class ChatStreamingMixin(ChatStreamingBridgeMixin):
    """Mixin providing streaming chat completions for BackgroundService.

    Bridge/session responsibilities (``_stream_via_bridge`` and the
    conversation-offset / OpenClaw session helpers) live on the inherited
    :class:`ChatStreamingBridgeMixin` (TASK-554).
    """


    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """Handle a streaming chat completion request.

        Uses llm_bawt's internal history + memory for context.
        Yields Server-Sent Events (SSE) formatted chunks.
        """
        # Flush an SSE comment IMMEDIATELY so the reverse proxy (Traefik) and
        # the Next.js /api/chat passthrough see bytes the instant the response
        # opens, before any of the slow setup below (model resolution, history
        # build, object-store attachment fetch, and — for agent bots — the
        # upstream image upload/analysis that delays the first real delta by
        # several seconds).  Without an early byte, a slow first token can trip
        # a gateway timeout and surface as a 502 on the client even though the
        # turn completes server-side.  Comment lines (": ...") are part of the
        # SSE spec and are ignored by every consumer — browser EventSource and
        # our own reader (ChatStreamContext skips lines starting with ":").
        yield ": connected\n\n"

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
            stream=True,
        )

        # Log incoming request (verbose mode will show the full payload)
        log.api_request(ctx, request.model_dump(exclude_none=True))

        bot_id = request.bot_id or self._default_bot
        user_id = request.user or self.config.DEFAULT_USER
        from ..message_authorship import AuthorReference
        user_author = inter_bot_author or AuthorReference.user(user_id)
        local_mode = not request.augment_memory

        # Resolve model using shared bot/config logic
        # Agent-backend bots resolve to their virtual model (e.g. "openclaw")
        from .prompt_execution import prompt_delivery_record, create_prompt_instance
        prompt_record = prompt_delivery_record(self, request)
        prompt_instance = None
        try:
            if prompt_record:
                model_alias, prompt_instance = create_prompt_instance(self, request)
                model_warnings = []
            else:
                model_alias, model_warnings = self._resolve_request_model(request.model, bot_id, local_mode)
        except Exception as e:
            log.api_error(ctx, str(e), 400)
            raise

        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # Get cached LLMBawt instance
        try:
            llm_bawt = prompt_instance or self._get_llm_bawt(model_alias, bot_id, user_id, local_mode)
        except Exception as e:
            log.api_error(ctx, str(e), 500)
            warning_data = {
                "object": "service.warning",
                "model": model_alias,
                "warnings": [f"request_failed: {e}"],
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
            return

        user_prompt, user_attachments, attachments_to_persist, media_store = (
            await prepare_stream_request_attachments(
                request,
                user_id=user_id,
                log=log,
            )
        )

        # Extract the user message ID so tool-call events and turn logs are
        # joinable with the frontend's history rendering.  Prefer the explicit
        # request.user_message_id (frontend-supplied UUID) and fall back to
        # scanning the trailing user message in request.messages.
        trigger_message_id = getattr(request, "user_message_id", None)
        if not trigger_message_id:
            for _m in reversed(request.messages):
                if _m.role != "user":
                    continue
                _mid = getattr(_m, "id", None) or getattr(_m, "message_id", None)
                if isinstance(_mid, str) and _mid.strip():
                    trigger_message_id = _mid.strip()
                    break
        # Every streaming turn needs one exact, durable trigger UUID before the
        # turn log and agent command are created. BawtHub already supplies one;
        # direct/server callers may not. Minting here keeps the history row,
        # turn_log, bridge events, and task association on the same join key.
        if not trigger_message_id:
            trigger_message_id = str(uuid.uuid4())

        # Frontend-minted UUID for the ASSISTANT reply row. Persisting the
        # assistant message under this id makes the live streaming bubble and the
        # reloaded history row share ONE id, so they merge into a single bubble
        # (closes the EPIC TASK-217 assistant-identity gap). None → history's
        # add_message mints a server UUID (server-originated turns).
        _amid = getattr(request, "assistant_message_id", None)
        assistant_message_id = _amid.strip() if isinstance(_amid, str) and _amid.strip() else None

        if not user_prompt:
            warning_data = {
                "object": "service.warning",
                "model": model_alias,
                "warnings": ["request_failed: No user message found in request"],
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
            return

        # OpenClaw bots now flow through the normal AgentBackendClient.stream_raw()
        # pipeline, which routes via Redis -> Bridge -> WS automatically.

        chunk_queue: asyncio.Queue = asyncio.Queue()
        full_response_holder = [""]  # Use list to allow mutation in nested function
        # TTS scrub decision — computed ONCE here (single source of truth via
        # should_scrub_for_tts), tracked on the turn, and shared by every
        # downstream consumer: the turn_start flag, the block-boundary scrubber
        # that emits `tts_delta` events, and the persisted `tts_scrubbed` column.
        tts_scrub = should_scrub_for_tts(llm_bawt.bot)
        tts_scrubber = StreamingTTSScrubber() if tts_scrub else None
        reasoning_holder = [""]  # TASK-301: accumulated model reasoning, persisted display-only
        animation_holder = [None]   # Populated by the embedding classifier (TASK-215)
        tool_context_holder = [""]  # Store tool context from native tool calls
        tool_call_details_holder: list[dict] = []
        timing_holder = [0.0, 0.0]  # [start_time, end_time]
        cancelled_holder = [False]  # Track if we were cancelled
        token_usage_holder: list[dict | None] = [None]  # Captures upstream SDK token usage for turn_complete
        # {asset_id, kind} refs for media the agent backend persisted during the
        # turn (e.g. Playwright screenshots) — attached to the assistant reply.
        agent_attachments_holder: list[dict] = []
        # TASK-269: tool_use_id of an AskUserQuestion the agent deferred this
        # turn.  Set when an await_tool_result chunk arrives; consumed at turn
        # completion to mark end_reason="question" + question_id on the turn log
        # and in the turn_complete event so the UI renders a QuestionMessage.
        question_id_holder: list[str | None] = [None]
        # TASK-292: request_id of an approval-gated tool call deferred this turn.
        # Set when an approval_required chunk arrives; consumed at turn
        # completion to mark end_reason="approval" so the UI keeps the card.
        approval_id_holder: list[str | None] = [None]
        # TASK-306 Section A: structured failure when a gated tool's approval
        # row could NOT be durably committed. When set, the turn ends honestly
        # (end_reason="approval_persist_failed") and the user is told the gate
        # did not reach them — never a silent success.
        approval_persist_failed_holder: list[dict | None] = [None]

        # Capture the event loop before entering the thread
        loop = asyncio.get_running_loop()

        # Start new generation.
        # For agent_backend (OpenClaw) bots the gateway queues messages as
        # separate runs, so we must NOT cancel a previous generation — each
        # request streams independently.  For native models we cancel the
        # previous generation so only the latest request runs.
        is_agent_backend = llm_bawt.client.model_definition.get("type") in ("agent_backend", "claude-code")

        # ---- /new command for CHAT bots --------------------------------
        # Agent bots already handle /new at the bridge (clears the SDK
        # session). Chat bots have no such concept, so /new here moves a
        # per-bot "conversation offset" marker to now: prior raw messages
        # drop out of the live context, but summaries + long-term memory are
        # kept (see HistoryManager/get_messages). Nothing is deleted — it's a
        # pointer move, and the dropped transcript still gets summarized by
        # the background job and reappears as a summary.
        # TASK-646: summarize → rotate → confirm lives in the SHARED helper
        # (chat_streaming_bridge.py) so this path and the non-streaming
        # chat_completion cannot drift.
        if not is_agent_backend:
            _confirm, user_prompt = self._maybe_handle_chat_new_command(
                llm_bawt, bot_id, user_prompt
            )
            if _confirm is not None:
                yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_alias, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': _confirm}, 'finish_reason': None}]})}\n\n"
                yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_alias, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
                return

        if is_agent_backend:
            # TASK-284 step 15: an agent /new also rotates the durable DB
            # thread (non-destructive). Rotation is deferred until AFTER the
            # session seed is assembled below — the seed must read the
            # OUTGOING thread's raw messages (session-scoped load), not the
            # fresh empty one. The bridge still owns the provider-side
            # reset+seed — the message passes through unchanged.
            cancel_event = threading.Event()
            done_event = threading.Event()
        else:
            cancel_event, done_event = await self._start_generation(bot_id)
        _reserved_turn_id = getattr(request, "inter_bot_turn_id", None)
        turn_log_id = (
            _reserved_turn_id.strip()
            if isinstance(_reserved_turn_id, str) and _reserved_turn_id.strip()
            else f"turn-{uuid.uuid4().hex}"
        )

        # Abort routing retains the ordinary per-bot/user bridge key; explicit
        # SDK transcript identity is carried separately by thread_binding.
        oc_session_key: str | None = None
        # TASK-501: seed history the app pre-assembles and pushes to the bridge
        # so it need not call back to /v1/history/context-seed. Stays None
        # unless this is a claude-code agent turn that will open a fresh session.
        inject_seed_messages: list | None = None
        # TASK-252: request-local explicit-thread binding (None = continuous).
        thread_binding: dict | None = None
        if is_agent_backend:
            bc = getattr(llm_bawt.bot, "agent_backend_config", None) or {}
            backend_name = getattr(llm_bawt.bot, "agent_backend", "")
            # TASK-276: the "local" backend routes the same way claude-code /
            # codex do — f"{bot_id}:{user_id}" — so chat.abort RPCs reach the
            # right active stream on the local-model-bridge.  Fall back to the
            # client's resolved backend when the bot has no agent_backend set
            # (local GPU bots are ordinary chat bots whose model_definition was
            # rewritten to type=agent_backend/backend=local in core.py).
            client_backend = ""
            try:
                client_backend = str(
                    llm_bawt.client.model_definition.get("backend") or ""
                ).strip()
            except Exception:
                client_backend = ""
            if backend_name in ("claude-code", "codex") or client_backend == "local":
                oc_session_key = f"{bot_id}:{user_id}"
            else:
                oc_session_key = bc.get("session_key")

            # Resolve the request-local thread binding BEFORE the seed
            # decision. Explicit selection wins; otherwise bind the current
            # ACTIVE thread so seed/resume decisions use canonical per-thread
            # metadata rather than the retired bot scalar.
            thread_binding = self._bind_agent_thread(llm_bawt, request)
            if prompt_record and (not thread_binding or not thread_binding.get("explicit_thread")):
                raise ValueError("Scheduled prompt could not bind its isolated thread")
            if thread_binding is None:
                thread_binding = self._resolve_active_thread_binding(llm_bawt)

            # Summarize the OUTGOING thread on /new BEFORE the seed builds
            # so the seed's summary bucket includes a fresh summary of the
            # ending conversation (TASK-641).
            self._maybe_summarize_on_new(
                llm_bawt, bot_id, user_prompt, thread_binding=thread_binding
            )
            from .routes.history import maybe_build_session_seed
            from .dependencies import get_service
            if prompt_record:
                from .prompt_execution import build_prompt_seed
                inject_seed_messages = build_prompt_seed(self, llm_bawt, thread_binding)
            else:
                inject_seed_messages = maybe_build_session_seed(
                    llm_bawt, bot_id, model_alias, user_prompt, get_service(),
                    thread_binding=thread_binding,
                )
            # Rotate the durable DB thread on /new — AFTER the seed is built
            # so the seed captured the outgoing thread's raw messages.
            self._maybe_rotate_agent_session(
                llm_bawt, bot_id, user_prompt, thread_binding=thread_binding
            )

            # A successful unscoped /new rotation invalidates the outgoing
            # binding. Resolve the fresh active thread (no SDK key yet) so the
            # bridge cold-starts and writes its minted key there.
            if (
                thread_binding
                and not thread_binding.get("explicit_thread")
                and (user_prompt or "").lstrip().startswith("/new")
            ):
                thread_binding = self._resolve_active_thread_binding(llm_bawt)

        task_turn_capability: str | None = None
        # Always bound: turn_start below publishes it for every backend, and
        # non-agent turns must send null rather than blow up on an unbound name.
        session_id: str = ""
        if is_agent_backend:
            session_id = (
                str(thread_binding.get("thread_session_id") or "").strip()
                if thread_binding else ""
            )
            from ..task_turn_context import mint_for_agent_turn

            task_turn_capability = mint_for_agent_turn(
                backend=str(getattr(llm_bawt.bot, "agent_backend", "") or ""),
                session_id=session_id,
                turn_id=turn_log_id,
                trigger_message_id=trigger_message_id,
                bot_id=bot_id,
                user_id=user_id,
            )

        # Persist turn log immediately so the user's prompt is recorded
        # even if the backend times out or errors before responding.
        # TASK-269: a continuation turn answering a deferred question carries
        # the awaiting turn as parent_turn_id (threads the chain + drives
        # cross-tab resolution via turn_start{parent_turn_id}).
        parent_turn_id = getattr(request, "parent_turn_id", None)
        answered_question_id = getattr(request, "answered_question_id", None)
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
            agent_session_key=oc_session_key,
            trigger_message_id=trigger_message_id,
            assistant_message_id=assistant_message_id,
            parent_turn_id=parent_turn_id,
            request_extensions={
                key: value
                for key, value in {
                    "inter_bot_delivery_id": getattr(
                        request, "inter_bot_delivery_id", None
                    ),
                    "inter_bot_turn_id": getattr(request, "inter_bot_turn_id", None),
                    "inter_bot_bridge_request_id": getattr(
                        request, "inter_bot_bridge_request_id", None
                    ),
                    "inter_bot_timeout_seconds": getattr(
                        request, "inter_bot_timeout_seconds", None
                    ),
                    "inter_bot_session_policy": getattr(
                        request, "inter_bot_session_policy", None
                    ),
                    "inter_bot_seed_session_id": getattr(
                        request, "inter_bot_seed_session_id", None
                    ),
                    "continuation_payload": (
                        request.continuation_payload.model_dump()
                        if getattr(request, "continuation_payload", None) else None
                    ),
                }.items()
                if value
            },
        )
        # Record which continuation turn carried the answer back to the agent.
        if answered_question_id:
            try:
                self._pending_question_store.set_answered_turn(
                    answered_question_id, turn_log_id,
                )
            except Exception as _link_err:
                log.debug("set_answered_turn failed for %s: %s", answered_question_id, _link_err)

        # Capture Redis subscriber for direct publish from the background
        # thread — ensures tool events (and turn_complete) reach Redis even
        # if the SSE generator is cancelled (client disconnect / page refresh).
        _redis_sub = getattr(self, "_redis_subscriber", None)

        _approval_handled: set[str] = set()
        _await_handled: set[str] = set()
        _upstream_model = [None]  # Actual model reported by agent backend
        _tool_event_coordinator = ToolEventCoordinator(self._turn_log_store.engine)
        _turn_ctx = TurnStreamContext(
            svc=self,
            request=request,
            llm_bawt=llm_bawt,
            loop=loop,
            bot_id=bot_id,
            user_id=user_id,
            model_alias=model_alias,
            user_prompt=user_prompt,
            user_attachments=user_attachments,
            user_author=user_author,
            attachments_to_persist=attachments_to_persist,
            media_store=media_store,
            trigger_message_id=trigger_message_id,
            assistant_message_id=assistant_message_id,
            turn_log_id=turn_log_id,
            is_agent_backend=is_agent_backend,
            inject_seed_messages=inject_seed_messages,
            thread_binding=thread_binding,
            task_turn_capability=task_turn_capability,
            cancel_event=cancel_event,
            done_event=done_event,
            chunk_queue=chunk_queue,
            tts_scrub=tts_scrub,
            tts_scrubber=tts_scrubber,
            full_response_holder=full_response_holder,
            reasoning_holder=reasoning_holder,
            animation_holder=animation_holder,
            tool_context_holder=tool_context_holder,
            tool_call_details_holder=tool_call_details_holder,
            timing_holder=timing_holder,
            cancelled_holder=cancelled_holder,
            token_usage_holder=token_usage_holder,
            agent_attachments_holder=agent_attachments_holder,
            question_id_holder=question_id_holder,
            approval_id_holder=approval_id_holder,
            approval_persist_failed_holder=approval_persist_failed_holder,
            _redis_sub=_redis_sub,
            _tool_event_coordinator=_tool_event_coordinator,
            _approval_handled=_approval_handled,
            _await_handled=_await_handled,
            _upstream_model=_upstream_model,
        )
        from .turn_execution import TurnExecution, turn_executions
        _turn_ctx.execution = TurnExecution(
            turn_log_id, cancel_event, is_agent=is_agent_backend, store=self._turn_log_store,
        )
        turn_executions.register(_turn_ctx.execution)
        _turn_worker = TurnStreamWorker(_turn_ctx)

        # Announce turn kickoff to the unified SSE stream BEFORE the worker
        # thread starts.  Other clients subscribed to this bot (other browser
        # tabs, the dashboard, CLI tools, scheduled jobs) flip their "bot is
        # active" indicators on the turn_start event — without it they'd only
        # find out a turn was running when the first tool_start arrived
        # (could be many seconds for non-tool turns) or on the next page
        # refresh via /api/chat/active-bots.  turn_complete (line ~1435)
        # is the matching teardown event.
        #
        # TASK-358: the payload also carries the user message itself
        # (``role``/``content``/``attachments``) so a SECOND window with no
        # prior bubble for this turn can render the USER BUBBLE live — not
        # just the "bot is active" rail. ``content`` is already rendered
        # client-side elsewhere (history + the originating tab's own optimistic
        # bubble), so this is not new exposure. ``attachments`` is enriched to
        # the SAME shape /v1/history emits ({asset_id, kind, mime_type, width,
        # height, urls}) via the canonical serializer, so the client can reuse
        # ``normalizeHistoryAttachment`` verbatim. Enrichment does a synchronous
        # media_assets SELECT, so it is offloaded to a thread and only when the
        # turn actually carries attachments (the common no-attachment path skips
        # all of it).
        turn_start_attachments: list[dict] = []
        if attachments_to_persist:
            try:
                turn_start_attachments = await asyncio.to_thread(
                    _turn_worker._enrich_turn_start_attachments
                )
            except Exception as _att_err:
                # A degraded MediaStore must never block the turn_start
                # announcement — fall back to no attachments (text bubble
                # still renders; images fill in on the receiving tab's next
                # history load).
                log.warning(
                    "TASK-358: turn_start attachment enrichment failed: %s",
                    _att_err,
                )
                turn_start_attachments = []

        from ..entity_presentation import EntityPresentationResolver
        assistant_author = AuthorReference.bot(bot_id)
        author_presentations = EntityPresentationResolver(
            self._turn_log_store.engine
        ).resolve_many_safe([user_author, assistant_author])

        await _turn_worker._publish_unified({
            "_type": "turn_start",
            "turn_id": turn_log_id,
            "trigger_message_id": trigger_message_id,
            # Canonical assistant row id (frontend-minted). Lets a non-originating
            # window key its assistant bubble on the SAME id the reload will carry,
            # so live→reload is a merge, not a duplicate.
            "assistant_message_id": assistant_message_id,
            "bot_id": bot_id,
            "user_id": user_id,
            # TASK-709: durable session id, so a receiving window can route this
            # event to the correct thread WITHOUT a follow-up DB fetch. Present
            # for agent-backend turns bound to an active/explicit thread; null
            # for continuous non-agent turns or on resolution failure — the
            # client's backward-compat path (adopt as before) still applies.
            "session_id": session_id or None,
            "parent_turn_id": parent_turn_id,
            "role": "user",
            "content": user_prompt,
            "author": author_presentations[
                (user_author.entity_type, user_author.entity_id)
            ],
            "assistant_author": author_presentations[
                (assistant_author.entity_type, assistant_author.entity_id)
            ],
            "attachments": turn_start_attachments,
            # TTS consumers (chat_tts_driver): when true, IGNORE raw text_delta
            # and synthesize the scrubbed `tts_delta` events instead. Markdown
            # is scrubbed at block boundaries (not per-token), so audio still
            # streams — just at paragraph granularity.
            "tts_scrubbed": tts_scrub,
            "ts": time.time(),
        })

        # ── Durable approval handling (TASK-292/306 race hardening) ──
        # The approval row + its surfacing MUST NOT depend on the SSE async
        # generator draining the chunk. A gated tool ends the turn and the
        # generator is torn down (client disconnect) before it dequeues the
        # already-delivered approval chunk — so the row never persisted and the
        # user saw nothing (a silent vacuum). We persist + publish from the
        # streaming WORKER THREAD (_intercept_tool_events), which runs to
        # completion regardless of client teardown. Idempotent: record_request
        # is keyed on tool_use_id and _approval_handled guards the async-gen
        # path from double-handling.

        # TASK-370: tool_use_ids of AskUserQuestion awaits already persisted +
        # surfaced by the streaming WORKER THREAD (_intercept_tool_events), which
        # runs to completion regardless of client/SSE teardown. Before this, a
        # deferred question was persisted INLINE in the async generator — which
        # can be torn down on turn-end before it drains the await chunk on a LONG
        # turn, silently dropping the question (turn ends end_reason="stop", no
        # chat_pending_questions row, bot believes it asked and never retries).
        # This is the exact durability fix approvals got in TASK-292/306.
        # _await_handled guards the async-gen path from double-handling.

        try:
            # Visual-stream emote filtering shares the SAME decision as TTS
            # scrubbing (should_scrub_for_tts), computed once above as tts_scrub.
            # NEVER for agent backends (TASK-799): their asterisks are markdown
            # (**bold** / *italic*), not roleplay emotes — the filter mangled
            # them AND removed wire chars that _text_chars/text_offset had
            # already counted, shifting every later tool anchor forward so
            # bubbles split mid-sentence live ("Before Taurus," / "Integration"
            # stranding). Persisted content is unfiltered, so offsets only map
            # onto an unfiltered wire. TTS scrubbing (tts_scrubber → tts_delta
            # in the worker) is a separate path and stays on.
            emote_filter = (
                StreamingEmoteFilter() if (tts_scrub and not is_agent_backend) else None
            )
            oc_tool_call_index = 0  # OpenClaw agent backend tool call index

            from .chat_stream_buffer import content_buffer
            _content_buf, _drain_content_buf = content_buffer(response_id, created, model_alias)

            # Send service warnings (e.g. model fallback) before content
            if model_warnings:
                warning_data = {
                    "object": "service.warning",
                    "model": model_alias,
                    "warnings": model_warnings,
                }
                yield f"data: {json.dumps(warning_data)}\n\n"

            # In-process GPU inference is gone (local models run in the
            # standalone local_model_bridge, TASK-276/278), so every model
            # streams on the default thread pool — multiple bots run
            # concurrently and nothing is serialized on a single GPU thread.
            loop.run_in_executor(None, _turn_worker._stream_to_queue)

            # Yield SSE chunks (with keepalive to prevent client timeout
            # during slow backends like vLLM first-inference)
            while True:
                try:
                    chunk = await asyncio.wait_for(chunk_queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Send SSE comment as keepalive to prevent client/proxy timeout
                    yield ": keepalive\n\n"
                    continue

                # DEBUG-292: trace every dict chunk through the async generator
                if isinstance(chunk, dict) and chunk.get("event") == "approval_required":
                    log.info(
                        "DEBUG-292 async_gen: approval_required chunk DEQUEUED "
                        "bot=%s turn=%s req_id=%s",
                        bot_id, turn_log_id,
                        chunk.get("tool_use_id", "?"),
                    )

                if chunk is None:
                    # Release the held text tail before closing (TASK-367).
                    sse = _drain_content_buf(flush_all=True)
                    if sse:
                        yield sse
                    # Stream complete - flush any buffered content from emote filter
                    if emote_filter:
                        final_chunk = emote_filter.flush()
                        if final_chunk:
                            data = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_alias,
                                "choices": [{"index": 0, "delta": {"content": final_chunk}, "finish_reason": None}],
                            }
                            yield (f"data: {json.dumps(data)}\n\n")

                    # Emit animation event (tts_mode only, normal completion only)
                    if animation_holder[0]:
                        anim_data = {"object": "service.animation", "animation": animation_holder[0]}
                        yield (f"data: {json.dumps(anim_data)}\n\n")

                    # Use upstream model name if available (e.g. "claude-opus-4-6[1m]")
                    final_model = _upstream_model[0] or model_alias

                    # Send final chunk
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": final_model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield (f"data: {json.dumps(data)}\n\n")
                    yield ("data: [DONE]\n\n")
                    break

                if isinstance(chunk, Exception):
                    # Upstream stream failed; close SSE cleanly so downstream clients
                    # don't see an incomplete chunked read protocol error.
                    log.error("Streaming backend failed: %s", chunk)
                    # Release the held text tail before the interruption notice
                    # so no buffered prose is lost on error (TASK-367).
                    sse = _drain_content_buf(flush_all=True)
                    if sse:
                        yield sse
                    warning_data = {
                        "object": "service.warning",
                        "model": model_alias,
                        "warnings": [f"stream_interrupted: {chunk}"],
                    }
                    yield (f"data: {json.dumps(warning_data)}\n\n")

                    # Flush any buffered content from emote filter.
                    if emote_filter:
                        final_chunk = emote_filter.flush()
                        if final_chunk:
                            data = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_alias,
                                "choices": [{"index": 0, "delta": {"content": final_chunk}, "finish_reason": None}],
                            }
                            yield (f"data: {json.dumps(data)}\n\n")

                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_alias,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield (f"data: {json.dumps(data)}\n\n")
                    yield ("data: [DONE]\n\n")
                    break

                # Past the None/Exception guards, a dict chunk is an event (tool
                # call, tool result, reasoning, terminal marker…) — NOT text.
                # Release the held text tail FIRST so buffered prose always
                # reaches the client before the event that follows it; this is
                # what anchors a tool card after its complete lead-in text
                # instead of mid-word (TASK-367). Guarded to dicts so a normal
                # str text chunk keeps accumulating in the boundary buffer below
                # instead of force-flushing every token.
                if isinstance(chunk, dict):
                    sse = _drain_content_buf(flush_all=True)
                    if sse:
                        yield sse

                # OpenAI-compatible delta.tool_calls streaming
                if isinstance(chunk, dict) and chunk.get("_type") == "tool_call_delta":
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_alias,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": chunk["index"],
                                    "id": chunk["id"],
                                    "type": "function",
                                    "function": {
                                        "name": chunk["name"],
                                        "arguments": chunk["arguments"],
                                    },
                                }],
                            },
                            "finish_reason": None,
                        }],
                    }
                    yield (f"data: {json.dumps(data)}\n\n")
                    continue

                if isinstance(chunk, dict) and chunk.get("_type") == "tool_event":
                    # Already published directly from the background thread
                    # via _publish_tool_event_direct — just skip.
                    continue

                if isinstance(chunk, dict) and chunk.get("_type") == "turn_usage":
                    usage_chunk = {
                        "id": response_id,
                        "created": created,
                        **{key: value for key, value in chunk.items() if key != "_type"},
                    }
                    yield f"data: {json.dumps(usage_chunk)}\n\n"
                    continue

                # Model reasoning ("thinking") → OpenAI-compat reasoning lane.
                # Emitted as choices[].delta.reasoning_content so it lands in a
                # separate UI channel, never in delta.content / the saved
                # message. The chunk is a dict, so consume_stream_chunks already
                # kept it out of full_response_holder (TASK-301).
                if isinstance(chunk, dict) and chunk.get("_type") == "reasoning_delta":
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_alias,
                        "choices": [{
                            "index": 0,
                            "delta": {"reasoning_content": chunk.get("delta", "")},
                            "finish_reason": None,
                        }],
                    }
                    yield (f"data: {json.dumps(data)}\n\n")
                    continue

                # OpenClaw agent backend tool events (from openclaw.py queue)
                if isinstance(chunk, dict) and chunk.get("event") == "tool_call":
                    tc_name = chunk.get("name", "unknown")
                    tc_args = chunk.get("arguments", {})
                    # Use the call_id injected by _intercept_tool_events so
                    # HTTP stream and SSE events share the same ID.
                    tc_id = chunk.get("_call_id") or f"call_{uuid.uuid4().hex[:8]}"
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_alias,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": oc_tool_call_index,
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {
                                        "name": tc_name,
                                        "arguments": json.dumps(tc_args, ensure_ascii=False) if isinstance(tc_args, dict) else str(tc_args),
                                    },
                                    # Authoritative char offset (chars of assistant
                                    # text before this tool), stamped by
                                    # _intercept_tool_events from the SAME _text_chars
                                    # counter persisted to tool_calls_json. Lets the
                                    # client anchor the tool at the byte-identical
                                    # position it will land on refresh, instead of its
                                    # own drifting local count (TASK-367). Non-standard
                                    # field — other OpenAI consumers ignore it.
                                    "text_offset": chunk.get("_text_offset"),
                                }],
                            },
                            "finish_reason": None,
                        }],
                    }
                    yield (f"data: {json.dumps(data)}\n\n")
                    oc_tool_call_index += 1
                    # Tool events already published from background thread
                    continue

                if isinstance(chunk, dict) and chunk.get("event") == "metadata":
                    if chunk.get("upstream_model"):
                        _upstream_model[0] = chunk["upstream_model"]
                    continue

                if isinstance(chunk, dict) and chunk.get("event") == "tool_result":
                    if oc_tool_call_index > 0:
                        data = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_alias,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                        }
                        yield (f"data: {json.dumps(data)}\n\n")
                        oc_tool_call_index = 0
                    continue

                if isinstance(chunk, dict) and chunk.get("event") == "await_tool_result":
                    # TASK-269/370 — deferred AskUserQuestion from the agent
                    # bridge. Durable persistence (chat_pending_questions),
                    # question_id_holder, and the unified tool_await_result event
                    # are now owned ENTIRELY by the streaming worker thread
                    # (_persist_publish_await), which runs to completion
                    # regardless of client/SSE teardown — the fix for a question
                    # emitted at the tail of a LONG turn being dropped when this
                    # async generator is torn down before draining the chunk
                    # (TASK-370). By the time this branch runs the worker has
                    # already persisted the row, set question_id_holder, and
                    # published the cross-tab event. This branch is now a pure
                    # sidecar: emit the originating client's OpenAI-compat SSE
                    # chunk so the live (still-connected) tab renders the question
                    # inline without waiting for a reload. (Mirrors the
                    # approval_required async-gen branch, likewise a dedup skip
                    # after _persist_publish_approval.)
                    tool_use_id = chunk.get("tool_use_id") or ""
                    tool_args = chunk.get("arguments") or {}
                    await_payload = {
                        "object": "tool.await_result",
                        "tool_use_id": tool_use_id,
                        "tool_name": chunk.get("tool_name", ""),
                        "arguments": tool_args,
                        "session_key": chunk.get("session_key", ""),
                        "provider": chunk.get("provider", ""),
                        "trigger_message_id": chunk.get("trigger_message_id"),
                        "bot_id": bot_id,
                        "turn_id": turn_log_id,
                    }
                    yield (f"data: {json.dumps(await_payload)}\n\n")
                    continue

                if isinstance(chunk, dict) and chunk.get("event") == "approval_required":
                    # TASK-292/306 — approval persistence + surfacing is owned
                    # ENTIRELY by the streaming worker thread
                    # (_persist_publish_approval). That path is the single source
                    # of truth: it commit-confirms the durable row (bounded retry),
                    # fails closed + informed on a confirmed write failure, and
                    # publishes the live event — all from a thread that runs to
                    # completion regardless of client/SSE teardown. By the time an
                    # approval_required chunk reaches this async generator the
                    # worker has already handled it, so this branch is a pure
                    # dedup skip. (Previously this branch carried a second, full
                    # copy of the persist/retry/emit logic — deleted: duplicated
                    # persistence is exactly the failure mode TASK-306 set out to
                    # remove.)
                    req_id = chunk.get("tool_use_id") or ""
                    if not req_id or req_id in _approval_handled:
                        continue
                    # Reaching here means the chunk bypassed _intercept_tool_events
                    # (a wiring bug) — the worker is wired unconditionally for every
                    # backend, so this should never happen. Log loudly rather than
                    # silently re-implementing persistence a second time.
                    log.error(
                        "approval_required reached async-gen UNHANDLED id=%s — "
                        "worker thread never persisted it; check "
                        "_intercept_tool_events wiring",
                        req_id,
                    )
                    continue

                if isinstance(chunk, dict) and chunk.get("event") == "tool_preapproved":
                    # TASK-305 — a previously approval-gated tool was re-attempted
                    # on this continuation turn and consumed a live one-shot grant.
                    # Publish to the unified stream so the tab(s) showing this turn
                    # mark that exact tool card as pre-approved (gold/lock). No HTTP
                    # yield needed: it's a live activity-card flag, and every chat
                    # tab is on the unified stream.
                    _preapproved_tuid = chunk.get("tool_use_id", "")
                    await _turn_worker._publish_unified({
                        "_type": "tool_preapproved",
                        "turn_id": turn_log_id,
                        "trigger_message_id": trigger_message_id,
                        "bot_id": bot_id,
                        "user_id": user_id,
                        "tool_use_id": _preapproved_tuid,
                        "tool_name": chunk.get("tool_name", ""),
                        "policy_id": chunk.get("policy_id"),
                        "severity": chunk.get("severity", "medium"),
                        "provider": chunk.get("provider", ""),
                        "ts": time.time(),
                    })
                    # Persist the preapproved flag on the tool_call_record so the
                    # gold badge survives a page reload (TASK-305 persistence).
                    if _preapproved_tuid and self._turn_log_store:
                        try:
                            self._turn_log_store.set_preapproved(_preapproved_tuid)
                        except Exception:
                            pass  # best-effort; live event is the primary surface
                    continue

                if isinstance(chunk, dict) and chunk.get("_type") == "tool_calls_finish":
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_alias,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                    }
                    yield (f"data: {json.dumps(data)}\n\n")
                    continue

                # Apply emote filter for voice_optimized bots
                if emote_filter:
                    chunk = emote_filter.process(chunk)
                    if not chunk:
                        # Chunk was filtered out or buffered
                        continue

                # Agent-backend text: hold the forming tail, release only at a
                # clean boundary (TASK-367). Every dict/terminal handler above
                # already flushed the buffer, so text→tool order is preserved.
                if is_agent_backend:
                    _content_buf.append(chunk)
                    sse = _drain_content_buf(flush_all=False)
                    if sse:
                        yield sse
                    continue

                # Normal chunk
                data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_alias,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                yield (f"data: {json.dumps(data)}\n\n")
        finally:
            # Generation lifecycle (done_event / _end_generation) is now
            # signalled from the worker thread's finally block so that it
            # fires only after _finalize_turn completes.  The worker runs
            # independently of this async generator, so a client disconnect
            # (GeneratorExit here) does not interrupt persistence.
            pass
