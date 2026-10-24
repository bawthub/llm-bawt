"""Session lifecycle helpers extracted from the streaming bridge (TASK-1188).

Keep the same service methods and ordering while the shared reset contract is
introduced. BackgroundService supplies cross-mixin dependencies through self.
"""
from __future__ import annotations

import json

from ..logging import get_service_logger

log = get_service_logger(__name__)


class BridgeSessionResetMixin:
    """Storage and provider primitives used by the shared reset executor.

    Command parsing, admission and confirmation belong to NewSessionExecution
    and NewSessionReceipt, not a pre-admission chat interceptor.
    """

    def _rotate_chat_session(
        self, llm_bawt, bot_id: str, *, working_memory_mode: str,
        idempotency_key: str | None = None,
    ) -> str | None:
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
            new_id = rotate(
                accept_pointer=True, working_memory_mode=working_memory_mode,
                **({"idempotency_key": idempotency_key} if idempotency_key else {}),
            )
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
            # Durable commands publish only from the receipt outbox. Rotation
            # alone (including a keyed replay) is not completed reset evidence.
            if new_id and not idempotency_key:
                self._publish_thread_switched(
                    slug,
                    getattr(backend, "user_id", None),
                    str(new_id),
                    source="new_command",
                )
            return str(new_id) if new_id else None
        except Exception as e:
            log.error("Failed to rotate session for bot=%s: %s", slug, e)
            if idempotency_key:
                raise  # A claimed delivery must not execute in the wrong thread.
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

            from ..runtime_switch_rotation import thread_switched_event

            payload = json.dumps(
                thread_switched_event(bot_id, uid, new_session_id, source),
                ensure_ascii=False,
            )
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
            # A claimed reset-policy delivery carries its server-owned successor
            # in request.session_id. It is not a user-selected archived thread:
            # treating it as explicit would suppress the provider reset and
            # reject the structured reset action at bridge preflight.
            policy_binding = bool(
                getattr(request, "inter_bot_delivery_id", None)
                and getattr(request, "inter_bot_session_policy", None)
                in {"reset_retain_history", "reset_without_history"}
            )
            binding: dict = {"thread_session_id": sid}
            if not policy_binding:
                binding["explicit_thread"] = True
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
        from ..routes.sessions import resolve_agent_session_key

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
        *, history_scope: str | None = None,
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
        ``history_scope`` is the same per-reset override the seed builder
        receives, so the gate and the seed can never disagree.

        Bounded: the summarize (LLM attempt with short timeout -> heuristic
        fallback -> DB write) runs in a worker thread joined for at most
        ``timeout_s``. On timeout the turn proceeds without waiting — the
        summary commits moments later against the (by then archived) thread
        with its session_id stamp intact, and the seed degrades to today's
        raw-only behavior. Never raises; failure costs nothing.
        """
        try:
            from agent_bridge.maintenance import is_new_command

            if not is_new_command(user_prompt):
                return False
            if thread_binding and thread_binding.get("explicit_thread"):
                log.info("/new summarize skipped: explicit-thread turn (bot=%s)", bot_id)
                return False
            from ...utils.history import effective_history_scope, scope_flags
            try:
                scope = effective_history_scope(llm_bawt.config_resolver, history_scope)
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

            from ...memory.summarization import (
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
            from ...memory.summarization_limits import resolve_summarization_limits
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
        *, delivery_id: str | None = None,
    ) -> str | None | bool:
        """TASK-284 step 15: rotate the durable DB thread on an agent ``/new``.

        Agent ``/new`` reset+seed stays bridge-owned (provider hydration is
        untouched); this coordinates the DURABLE side so the fresh provider
        session maps onto a fresh DB thread — the /new turn's persisted
        messages then land on the new thread, and the bridge's subsequent
        session_key PATCH mirrors the new provider id onto it.

        Detection shares the bridge's exact leading-token classifier, so
        /newspaper never rotates the durable thread. Returns the new
        session id (truthy) only when a rotation happened. Shared by BOTH
        dispatch paths (streaming + non-streaming) so they cannot drift.
        """
        try:
            from agent_bridge.maintenance import is_new_command

            if not is_new_command(user_prompt):
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
        from ...working_memory.session_state import PINNED
        from .execution import NewSessionExecution

        if not delivery_id:
            # An undelivered typed /new is a human /new: it pins (N1).
            successor = self._rotate_chat_session(llm_bawt, bot_id, working_memory_mode=PINNED)
            if not successor:
                raise RuntimeError("Durable /new session rotation did not return a successor")
            return successor
        # In-process fast path only. The backing session rotation is keyed and
        # atomic in the DB, so a process restart gets the same successor too.
        key = (bot_id, getattr(llm_bawt, "user_id", None), delivery_id)
        cache = getattr(self, "_agent_new_delivery_rotations", None)
        if cache is None:
            cache = {}
            self._agent_new_delivery_rotations = cache
        if key in cache:
            return cache[key]
        successor = self._rotate_chat_session(
            llm_bawt, bot_id, idempotency_key=delivery_id,
            working_memory_mode=NewSessionExecution.working_memory_mode(self, delivery_id),
        )
        if not successor:
            raise RuntimeError("Durable /new session rotation did not return a successor")
        cache[key] = successor
        if len(cache) > 200:
            cache.pop(next(iter(cache)))
        return successor
