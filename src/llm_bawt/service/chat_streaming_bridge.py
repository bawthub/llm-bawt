"""Chat-streaming bridge/session responsibilities (TASK-554).

Split out of ``chat_streaming.py``: the agent-bridge streaming path
(``_stream_via_bridge``) plus its conversation-offset and OpenClaw
session helpers. Kept as a base mixin that ``ChatStreamingMixin`` inherits,
so the composed ``BackgroundService`` exposes the same methods and every
``self.*`` reference (turn-log store, session bridge, gateway http, config,
finalize/persist/update turn) resolves across the mixin set unchanged.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator

from .logging import get_service_logger

log = get_service_logger(__name__)


# TASK-646: confirmation text for a bare chat-bot /new — module-level so both
# dispatch paths emit byte-identical copy (streaming SSE + non-streaming JSON).
CHAT_NEW_CONFIRMATION = (
    "Fresh start. I've set down the recent back-and-forth — "
    "I still keep the longer-term summaries and what I know "
    "about you, just not this last thread. What's on your mind?"
)


class ChatStreamingBridgeMixin:
    """Agent-bridge streaming + session helpers for BackgroundService."""

    def _maybe_handle_chat_new_command(
        self, llm_bawt, bot_id: str, user_prompt: str,
    ) -> tuple[str | None, str]:
        """TASK-646: chat-bot ``/new`` interception, shared by BOTH dispatch
        paths (streaming ``chat_completion_stream`` and non-streaming
        ``chat_completion``) so they cannot drift — same pattern as
        ``_maybe_summarize_on_new`` / ``_maybe_rotate_agent_session``.

        Agent bots handle /new via session rotation at the bridge; chat bots
        have no bridge, so /new here summarizes the outgoing thread FIRST
        (TASK-641), then rotates the durable DB thread (TASK-284,
        non-destructive — failure just means the thread didn't advance).

        Returns ``(confirm_text, effective_prompt)``:
        - not a /new           → ``(None, user_prompt)`` — proceed unchanged.
        - ``/new <message>``   → ``(None, "<message>")`` — rotated; answer the
          remainder in the clean context.
        - bare ``/new``        → ``(CHAT_NEW_CONFIRMATION, "")`` — rotated;
          caller short-circuits with the confirmation, NO LLM round-trip
          (letting it reach the model makes it hallucinate a fake reset).
        """
        stripped = (user_prompt or "").lstrip()
        low = stripped.lower()
        if not (low == "/new" or low.startswith("/new ") or low.startswith("/new\n")):
            return None, user_prompt
        # TASK-641: summarize the outgoing thread BEFORE rotation so the next
        # turn's context (rebuilt every turn for chat bots) carries a summary
        # of the conversation that just ended.
        self._maybe_summarize_on_new(llm_bawt, bot_id, user_prompt)
        if not self._rotate_chat_session(llm_bawt, bot_id):
            log.error(
                "/new: session rotation failed for bot=%s — thread not advanced",
                bot_id,
            )
        remainder = stripped[len("/new"):].strip()
        if remainder:
            return None, remainder
        return CHAT_NEW_CONFIRMATION, ""

    def _rotate_chat_session(self, llm_bawt, bot_id: str) -> str | None:
        """TASK-284: non-destructive DB session rotation for chatbot /new.

        Closes the active durable thread and opens a fresh one (old rows
        untouched). The next turn's session-scoped load then sees only the new
        empty thread plus rolling summaries. Also clears the live in-memory
        transcript so a sub-TTL follow-up starts clean immediately.

        Returns the NEW session id on success (truthy — callers may treat it
        as a bool), None on failure (callers log; nothing is deleted either
        way, so a failed rotation just means the thread didn't advance).

        TASK-257: a successful rotation also publishes the deterministic
        ``thread_switched`` unified event so the UI's conversation list
        refreshes without text-matching "/new" or racing turn_complete.
        """
        slug = (bot_id or "").strip().lower()
        try:
            backend = getattr(llm_bawt.history_manager, "_db_backend", None)
            rotate = getattr(backend, "rotate_session", None) if backend else None
            if not callable(rotate):
                return None
            new_id = rotate()
            inv = getattr(llm_bawt, "invalidate_history_cache", None)
            if callable(inv):
                inv()
            try:
                hm = llm_bawt.history_manager
                hm.messages = [
                    m for m in hm.messages if getattr(m, "role", "") == "summary"
                ]
            except Exception:
                pass
            log.info("Rotated DB session for bot=%s -> %s (/new v2)", slug, new_id)
            if new_id:
                self._publish_thread_switched(
                    slug,
                    getattr(backend, "user_id", None),
                    str(new_id),
                    source="new_command",
                )
            return str(new_id) if new_id else None
        except Exception as e:
            log.error("Failed to rotate session for bot=%s: %s", slug, e)
            return None

    def _publish_thread_switched(
        self, bot_id: str, user_id: str | None, new_session_id: str, source: str,
    ) -> None:
        """TASK-257: emit ``thread_switched`` onto the unified event stream.

        Fire-and-forget from a daemon thread. Rotation runs in BOTH
        loop-thread (streaming dispatch) and executor-thread (non-streaming
        ``_do_query``) contexts, so the publish must not assume a loop —
        and per Gavel's recheck, socket timeouts alone do NOT bound a sync
        publish: DNS resolution of the Redis hostname happens BEFORE the
        connect timeout applies and can block arbitrarily. Offloading the
        entire body (client creation + XADD) to a short-lived daemon thread
        makes the caller's cost ~thread-spawn regardless of Redis/DNS
        health. /new is rare, so a thread per event is fine.
        Best-effort: never raises, a lost event only costs a list refresh.
        """
        try:
            import threading

            uid = (user_id or "").strip() or (
                getattr(self.config, "DEFAULT_USER", "") or ""
            ).strip() or "nick"
            threading.Thread(
                target=self._publish_thread_switched_blocking,
                args=(bot_id, uid, new_session_id, source),
                daemon=True,
                name=f"thread-switched-pub-{bot_id}",
            ).start()
        except Exception as e:
            log.warning("thread_switched publish failed for %s: %s", bot_id, e)

    def _publish_thread_switched_blocking(
        self, bot_id: str, uid: str, new_session_id: str, source: str,
    ) -> None:
        """Blocking half of the thread_switched publish (daemon thread only).

        Socket timeouts stay tight (1s connect + 1s command) so the daemon
        thread itself is short-lived; redis-py's connection pool is
        thread-safe, so the lazy singleton is shared across publishes.
        """
        try:
            client = getattr(self, "_thread_event_redis", None)
            if client is None:
                import redis as _redis

                client = _redis.Redis.from_url(
                    self.config.REDIS_URL,
                    decode_responses=True,
                    socket_timeout=1.0,
                    socket_connect_timeout=1.0,
                )
                self._thread_event_redis = client
            from agent_bridge.publisher import (
                UNIFIED_EVENTS_PREFIX,
                UNIFIED_STREAM_MAXLEN,
            )

            payload = json.dumps({
                "_type": "thread_switched",
                "bot_id": bot_id,
                "user_id": uid,
                "new_session_id": new_session_id,
                "source": source,
                "ts": time.time(),
            }, ensure_ascii=False)
            client.xadd(
                f"{UNIFIED_EVENTS_PREFIX}{bot_id}:{uid}",
                {"payload": payload},
                maxlen=UNIFIED_STREAM_MAXLEN,
                approximate=True,
            )
            log.info(
                "thread_switched published: bot=%s user=%s -> %s (%s)",
                bot_id, uid, new_session_id, source,
            )
        except Exception as e:
            log.warning("thread_switched publish failed for %s: %s", bot_id, e)

    def _bind_agent_thread(self, llm_bawt, request) -> dict | None:
        """Resolve the per-thread SDK binding for an explicit-thread turn.

        When a claude-code/codex agent turn carries an explicit ``session_id``
        (a thread the user opened in the UI), resolve THAT thread's stored SDK
        session key so the bridge resumes the thread's own transcript.

        Returns a REQUEST-LOCAL dict (never stored on the shared cached client
        — concurrent turns must not cross-bind):

        - ``thread_session_id``: the durable bawthub thread id;
        - ``thread_resume_id``: the SDK session id stored for it (absent when
          the thread has none — the bridge then cold-starts + seeds and writes
          the minted id back via PUT /v1/sessions/{id}/agent-session-key);
        - ``explicit_thread``: True — the user chose this thread in the UI.

        Returns None for unscoped turns and non-agent bots. Shared by
        BOTH dispatch paths so they cannot drift. Never raises.
        """
        try:
            backend_name = (getattr(llm_bawt.bot, "agent_backend", "") or "").strip()
            if backend_name not in ("claude-code", "codex"):
                return None
            sid = getattr(request, "session_id", None)
            sid = sid.strip() if isinstance(sid, str) and sid.strip() else None
            if not sid:
                return None
            binding: dict = {"thread_session_id": sid, "explicit_thread": True}
            return self._resolve_thread_sdk_key(llm_bawt, binding, backend_name)
        except Exception as e:
            log.warning("Agent thread binding failed: %s", e)
            return None

    def _resolve_active_thread_binding(self, llm_bawt) -> dict | None:
        """Resolve the ACTIVE thread's SDK binding for an unscoped turn.

        TASK-638: per-thread keys are canonical — every turn (not just
        explicit-thread ones) resolves its SDK session from the thread, never
        from the bot's scalar ``agent_backend_config.session_key``. Called
        AFTER ``_maybe_rotate_agent_session`` so a ``/new`` rotation has
        already landed and the active thread is the correct (possibly fresh)
        one.

        Returns a binding dict with ``thread_session_id`` +
        ``thread_resume_id`` (if the thread has one), or None on error.
        ``explicit_thread`` is absent (falsy) — the bridge uses this to gate
        ``/new`` processing (explicit threads never process ``/new``).
        """
        try:
            backend_name = (getattr(llm_bawt.bot, "agent_backend", "") or "").strip()
            if backend_name not in ("claude-code", "codex"):
                return None
            bot_id = getattr(llm_bawt.bot, "slug", None) or getattr(llm_bawt.bot, "name", "")
            user_id = getattr(llm_bawt, "user_id", None) or ""
            db_backend = getattr(llm_bawt.history_manager, "_db_backend", None)
            get_active = getattr(db_backend, "get_active_session", None) if db_backend else None
            if not callable(get_active):
                log.warning("Active-thread binding: no get_active_session on backend")
                return None
            row = get_active(bot_id=bot_id, user_id=user_id)
            if not row:
                # Binding is resolved before the worker persists the user row.
                # On a first-ever turn there is therefore no active thread yet,
                # but the bridge still needs a durable write-back target for the
                # SDK session it is about to mint. Create that target now.
                get_or_create = getattr(
                    db_backend, "get_or_create_active_session", None
                )
                if not callable(get_or_create):
                    log.warning(
                        "Active-thread binding: no active thread or creator for %s/%s",
                        bot_id, user_id,
                    )
                    return None
                sid = get_or_create(bot_id=bot_id, user_id=user_id)
                if not sid:
                    return None
                log.info(
                    "Active-thread binding: created fresh thread %s for %s/%s",
                    sid, bot_id, user_id,
                )
                return {"thread_session_id": str(sid)}
            sid = row.get("id")
            if not sid:
                return None
            binding: dict = {"thread_session_id": str(sid)}
            return self._resolve_thread_sdk_key(llm_bawt, binding, backend_name)
        except Exception as e:
            log.warning("Active-thread binding failed: %s", e)
            return None

    def _resolve_thread_sdk_key(
        self, llm_bawt, binding: dict, backend_name: str,
    ) -> dict:
        """Populate ``thread_resume_id`` on a binding dict from the thread's
        stored ``agent_session_keys``. Shared by explicit and active paths."""
        sid = binding["thread_session_id"]
        db_backend = getattr(llm_bawt.history_manager, "_db_backend", None)
        get_sess = getattr(db_backend, "get_session", None) if db_backend else None
        row = get_sess(sid) if callable(get_sess) else None
        if not row:
            log.warning("Thread SDK key resolve: no session row for %s (cold-start)", sid)
            return binding
        from .routes.sessions import resolve_agent_session_key

        metadata = row.get("session_metadata") or {}
        bc = getattr(getattr(llm_bawt, "client", None), "_bot_config", None) or {}
        resume = resolve_agent_session_key(
            metadata,
            backend_name,
            str(bc.get("model") or "").strip() or None,
        )
        if resume:
            binding["thread_resume_id"] = resume
        else:
            lifecycle = metadata.get("context_lifecycle")
            if isinstance(lifecycle, dict):
                action = str(lifecycle.get("action") or "")
                old_session_id = str(lifecycle.get("old_session_id") or "")
                if action in {"reset_retain_history", "reset_without_history"}:
                    binding["session_policy"] = action
                    if old_session_id:
                        binding["seed_session_id"] = old_session_id
        log.info(
            "Agent thread bound: thread=%s resume=%s explicit=%s",
            sid, resume or "none (cold-start)",
            bool(binding.get("explicit_thread")),
        )
        return binding

    def _maybe_summarize_on_new(
        self, llm_bawt, bot_id: str, user_prompt: str,
        thread_binding: dict | None = None, timeout_s: float = 8.0,
    ) -> bool:
        """TASK-641: summarize the OUTGOING thread before the /new seed builds.

        Runs on ``/new`` only (same gate as ``_maybe_rotate_agent_session``,
        shared by BOTH dispatch paths), BEFORE ``maybe_build_session_seed`` —
        so the seed's summary bucket includes a fresh summary of the
        conversation that just ended, via THE common per-thread unit
        (``HistorySummarizer.summarize_thread``, same function the background
        job calls).

        Gated on the bot's ``history_scope`` including summaries: with
        summaries off, summarizing would only STARVE the seed (flagged rows
        leave the raw pool and no summary bucket exists to carry the result).

        Bounded: the summarize (LLM attempt with short timeout -> heuristic
        fallback -> DB write) runs in a worker thread joined for at most
        ``timeout_s``. On timeout the turn proceeds without waiting — the
        summary commits moments later against the (by then archived) thread
        with its session_id stamp intact, and the seed degrades to today's
        raw-only behavior. Never raises; failure costs nothing.
        """
        try:
            if not (user_prompt or "").lstrip().startswith("/new"):
                return False
            if thread_binding and thread_binding.get("explicit_thread"):
                log.info("/new summarize skipped: explicit-thread turn (bot=%s)", bot_id)
                return False
            from ..utils.history import scope_flags
            try:
                scope = llm_bawt.config_resolver.resolve_config_setting(
                    "history_scope"
                ).value
            except Exception:
                scope = None
            include_history, include_summaries = scope_flags(scope)
            if not include_summaries:
                log.info(
                    "/new summarize skipped: scope=%r has no summaries (bot=%s)",
                    scope, bot_id,
                )
                return False

            backend = getattr(llm_bawt.history_manager, "_db_backend", None)
            if backend is None:
                log.info("/new summarize skipped: no db backend (bot=%s)", bot_id)
                return False
            session_id = str(
                (thread_binding or {}).get("thread_session_id")
                or getattr(backend, "_current_session_id", "")
                or ""
            )
            if not session_id:
                # MCP server mode: _db_backend is the _MCPShortTermManager
                # proxy, which keeps no local session cache — resolve the
                # active thread through its memory client, the SAME resolver
                # the session-scoped read path uses (load_session_scoped).
                mc = getattr(backend, "_memory_client", None)
                getter = getattr(mc, "get_active_session", None) if mc else None
                try:
                    active = getter() if callable(getter) else None
                except Exception:
                    active = None
                session_id = str((active or {}).get("id") or "")
            if not session_id:
                log.info("/new summarize skipped: no session id (bot=%s)", bot_id)
                return False
            log.info(
                "/new pre-seed summarize starting: bot=%s thread=%s",
                bot_id, session_id[:8],
            )

            import threading

            from ..memory.summarization import (
                HistorySummarizer,
                summarize_session_with_client,
            )

            # TASK-857: summarize with the GLOBAL background job model — the
            # same client the HISTORY_SUMMARIZATION job uses. This used to
            # POST /v1/llm/complete, which answers with "whatever client is
            # first in _client_cache" under a 6s timeout — on an all-agent
            # deployment that timed out every /new and the seed only ever got
            # the heuristic template. Time-bounding stays with the worker
            # thread join below.
            bg_client, bg_alias = self._get_background_client()
            if bg_client is None:
                log.warning(
                    "/new pre-seed summarize: no background job model client — "
                    "summary will be heuristic (bot=%s)", bot_id,
                )

            # TASK-858: same budget the route and the background job use —
            # derived from the job model's context window, not a 6000 floor.
            from ..memory.summarization_limits import resolve_summarization_limits
            seed_max_prompt_tokens, _seed_chunk = resolve_summarization_limits(
                self.config, bg_alias,
            )

            def _bounded_llm(session):
                return summarize_session_with_client(
                    session, bg_client, self.config,
                    max_prompt_tokens=seed_max_prompt_tokens,
                )

            def _work():
                try:
                    summarizer = HistorySummarizer(
                        self.config, bot_id=bot_id, summarize_fn=_bounded_llm,
                    )
                    # Mixed raw+summary seeds preserve the configured recent
                    # tail verbatim. Summary-only seeds have no raw fallback:
                    # summarize the complete outgoing thread and lower the job
                    # threshold so even a one-turn conversation survives /new.
                    summary_only = not include_history
                    result = summarizer.summarize_thread(
                        session_id,
                        protect_recent_turns=not summary_only,
                        min_messages=1 if summary_only else None,
                    )
                    log.info(
                        "/new pre-seed summarization: bot=%s thread=%s "
                        "created=%s messages=%s errors=%s",
                        bot_id, session_id[:8],
                        result.get("summaries_created"),
                        result.get("messages_summarized"),
                        result.get("errors") or None,
                    )
                except Exception as e:
                    log.warning(
                        "/new pre-seed summarization failed for %s: %s", bot_id, e
                    )

            worker = threading.Thread(
                target=_work, daemon=True, name=f"new-summarize-{bot_id}",
            )
            worker.start()
            worker.join(timeout=timeout_s)
            if worker.is_alive():
                log.warning(
                    "/new summarization still running after %.0fs for bot=%s — "
                    "proceeding without it (summary will land on the archived "
                    "thread)", timeout_s, bot_id,
                )
                return False
            return True
        except Exception as e:
            log.warning("/new summarize gate failed for %s: %s", bot_id, e)
            return False

    def _maybe_rotate_agent_session(
        self, llm_bawt, bot_id: str, user_prompt: str, thread_binding: dict | None = None,
    ) -> str | None | bool:
        """TASK-284 step 15: rotate the durable DB thread on an agent ``/new``.

        Agent ``/new`` reset+seed stays bridge-owned (provider hydration is
        untouched); this coordinates the DURABLE side so the fresh provider
        session maps onto a fresh DB thread — the /new turn's persisted
        messages then land on the new thread, and the bridge's subsequent
        session_key PATCH mirrors the new provider id onto it.

        Detection mirrors the bridge exactly (``lstrip().startswith("/new")``)
        so DB rotation fires iff the bridge will reset. Returns the new
        session id (truthy) only when a rotation happened. Shared by BOTH
        dispatch paths (streaming + non-streaming) so they cannot drift.
        """
        try:
            if not (user_prompt or "").lstrip().startswith("/new"):
                return False
            # A turn explicitly bound to a user-selected thread never rotates —
            # "/new" inside an opened old thread is treated as plain text.
            # Active-thread bindings are normal and MUST still rotate.
            if thread_binding and thread_binding.get("explicit_thread"):
                log.warning(
                    "Skipping /new rotation: turn is bound to thread %s",
                    thread_binding.get("thread_session_id"),
                )
                return False
        except Exception:
            return False
        return self._rotate_chat_session(llm_bawt, bot_id)

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
                gateway = getattr(self, '_gateway_http', None)
                if not gateway:
                    from openclaw_bridge.gateway_http import GatewayHttpClient
                    gateway = GatewayHttpClient(
                        self.config.OPENCLAW_GATEWAY_URL,
                        self.config.OPENCLAW_GATEWAY_TOKEN,
                    )
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
                        self._update_turn_log(turn_id=turn_log_id, status="aborted", end_reason="aborted")
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
