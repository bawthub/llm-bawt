"""Agent-session history seed assembly and preview route."""

from fastapi import APIRouter, HTTPException, Query

from ..dependencies import get_service
from ..logging import get_service_logger

# Agent backends whose bridges can open a fresh session on an app-built seed:
# Claude Code via a synthetic transcript, Codex via app-server
# ``thread/inject_items`` (TASK-1579).
SEEDED_AGENT_BACKENDS = frozenset({"claude-code", "codex"})

router = APIRouter()
log = get_service_logger(__name__)

def cap_seed_contents(
    role_content_pairs: list[tuple[str, str]], cap: int
) -> tuple[list[str], int]:
    """Apply the per-message seed char cap (TASK-785). Pure, unit-tested.

    Truncates oversized ``user``/``assistant`` contents at ``cap`` chars with a
    marker pointing at durable chat history. Other roles (``summary`` — already
    the dense representation, bounded by summary_count) pass through untouched.
    ``cap <= 0`` disables (pre-785 behavior). Returns (contents, trimmed_count).
    """
    contents: list[str] = []
    trimmed = 0
    for role, content in role_content_pairs:
        if cap <= 0 or len(content) <= cap or role not in ("user", "assistant"):
            contents.append(content)
            continue
        dropped = len(content) - cap
        contents.append(
            content[:cap].rstrip()
            + f"\n[... trimmed {dropped} chars for session seed — "
            "full text is in durable chat history (messages_get)]"
        )
        trimmed += 1
    return contents, trimmed


def build_context_seed(
    bot_id: str, model: str | None, service, session_id: str | None = None,
    *, llm_bawt=None, history_scope: str | None = None,
    user_id: str | None = None,
) -> dict:
    """Build the context-seed payload for a fresh Claude Code SDK session.

    Extracted from the ``/v1/history/context-seed`` handler (TASK-501) so the
    app can assemble the seed IN-PROCESS at dispatch time and push it to the
    bridge via ``inject_messages`` — no HTTP round-trip. The bridge no longer
    has to call back for it.

    Reuses ``HistoryManager.get_context_messages()`` — the exact function the
    chat path uses — so the seed matches what a normal chatbot sees: rolling
    summaries of older sessions PLUS the most recent turns, token-budgeted
    against the model's context window. System rows are dropped (the SDK
    carries its own system prompt).

    Returns ``{bot_id, model, budget_tokens, messages:[{role,content,timestamp}],
    stats:{...}}``. Raises on failure (callers decide whether to swallow).

    ``history_scope`` is the scope a /new already resolved for this one seed
    (the bot setting, or a scheduled reset's continuity parameter); ``None``
    reads the bot setting here.
    """
    effective_bot_id = bot_id or service._default_bot
    # Pass the caller's model through AS-IS (usually None for a preview). Do NOT
    # coalesce to the service default here: _resolve_request_model already
    # resolves None to the BOT's own default_model (priority 3) before the
    # service default (priority 4). Forcing the service default made it an
    # "explicit request" (priority 2), which overrode the bot's real model —
    # e.g. seeding chat-harness 'mira' (grok-4.3) with the service default
    # grok-4.5@xai-responses, a responses-only endpoint incompatible with
    # harness=chat → hard failure. TASK-620.
    if llm_bawt is None:
        # TASK-1774: a seed is one user's history. Without the caller's user
        # there is nothing safe to show; never fall back to DEFAULT_USER.
        if not user_id:
            raise ValueError("context seed requires the requesting user_id")
        model_alias, _ = service._resolve_request_model(
            model, effective_bot_id, local_mode=False
        )
        llm_bawt = service._get_llm_bawt(
            model_alias, effective_bot_id, user_id
        )
    else:
        # Automation already owns an isolated, owner-scoped instance. Never
        # resolve DEFAULT_USER or touch the personal model/cache during seeding.
        model_alias = llm_bawt.resolved_model_alias
    # Load fresh so we never serve a stale in-memory transcript.
    # ``session_id`` (TASK-252): scoped seed for hydrating an explicitly
    # opened thread — pool becomes that thread's raw + rolling summaries.
    llm_bawt.history_manager.load_history(session_id=session_id)

    # Same budget the agent path uses (base.py) — the ONE Tier-2 authority.
    # 0 => no budget (return everything). TASK-609. Capture the full
    # decomposition (window -> reserve -> prompt budget) so the context-management
    # UI can paint the whole ceiling picture, not just the prompt budget.
    context_window, effective_reserve, budget = service.config.resolve_context_budget(
        model_alias
    )

    # TASK-493/518: seed content comes from the ONE shared handler
    # (HistoryManager.build_context_payload) — the same function the chat turn
    # path uses. maybe_build_session_seed already decided we should seed (scope
    # != "none"); history_scope decodes into two INDEPENDENT flags so the seed
    # can be recent-only, summary-only (dense), or both. delivery="seed" ->
    # system rows dropped (the SDK injects its own system prompt every turn).
    from ...utils.history import effective_history_scope, scope_flags

    if history_scope is not None:
        # An invalid resolved scope is a caller bug; never degrade it to "both on".
        scope = effective_history_scope(None, history_scope)
    else:
        try:
            scope = effective_history_scope(llm_bawt.config_resolver)
        except Exception:
            scope = "inline+summaries"
    include_history, include_summaries = scope_flags(scope)
    payload = llm_bawt.history_manager.build_context_payload(
        include_history=include_history,
        include_summaries=include_summaries,
        delivery="seed",
        max_tokens=budget,
    )
    seed_msgs = payload.seed_messages

    # TASK-785: cap per-message size in the seed. Inter-bot task-report
    # deliveries run 12-16k chars each and were seeded verbatim, tripling the
    # fresh-session base cost. Oversized user/assistant rows are truncated with
    # a marker pointing at durable chat history; summaries are exempt (they are
    # already the dense representation and capped by summary_count). Applied
    # here — the single choke point for BOTH live delivery
    # (maybe_build_session_seed) and the /v1/history/context-seed preview — so
    # preview and delivery can never disagree. 0 = no cap.
    try:
        seed_cap = int(
            llm_bawt.config_resolver.resolve_config_setting(
                "seed_message_max_chars"
            ).value
        )
    except Exception:
        from ...setting_definitions import setting_default

        seed_cap = int(setting_default("seed_message_max_chars", 3000) or 0)

    seed_contents, trimmed_count = cap_seed_contents(
        [(m.role, m.content or "") for m in seed_msgs], seed_cap
    )

    summary_count = sum(1 for m in seed_msgs if m.role == "summary")
    convo_count = sum(1 for m in seed_msgs if m.role in ("user", "assistant"))
    # Same per-message estimate as estimate_messages_tokens, on trimmed content.
    approx_tokens = sum(len(c) // 4 + 4 for c in seed_contents)
    ts_values = [
        float(getattr(m, "timestamp", 0.0) or 0.0)
        for m in seed_msgs
        if getattr(m, "timestamp", 0.0)
    ]
    oldest = min(ts_values) if ts_values else None
    newest = max(ts_values) if ts_values else None

    # Effective context-sizing knobs that shape the ladder, resolved with the
    # SAME resolver the assembler uses, each tagged with its provenance
    # (code_default / global / bot) so the UI can mark default-vs-override
    # without a second call. TASK-620.
    def _eff(key: str, fallback):
        try:
            rv = llm_bawt.config_resolver.resolve_config_setting(key)
            return {"value": rv.value, "source": rv.source}
        except Exception:
            return {"value": fallback, "source": "code_default"}

    return {
        "bot_id": effective_bot_id,
        "model": model_alias,
        "budget_tokens": budget,
        # Full Tier-2 decomposition: window - reserve = prompt budget.
        "budget": {
            "context_window": context_window,
            "effective_reserve": effective_reserve,
            "prompt_budget": budget,
        },
        # The sizing knobs that bound each ladder bucket, with provenance.
        "sizing": {
            "history_scope": (
                {"value": scope, "source": "reset"} if history_scope is not None
                else _eff("history_scope", "inline+summaries")
            ),
            "history_tokens": _eff("history_tokens", 12000),
            "history_max_age_hours": _eff("history_max_age_hours", 0),
            "summary_count": _eff("summary_count", 5),
            "seed_message_max_chars": _eff("seed_message_max_chars", 3000),
        },
        "messages": [
            {
                "role": m.role,
                "content": content,
                "timestamp": float(getattr(m, "timestamp", 0.0) or 0.0),
                "source_id": str(m.db_id) if getattr(m, "db_id", None) else None,
                "shortened": content != (m.content or ""),
            }
            for m, content in zip(seed_msgs, seed_contents)
        ],
        "stats": {
            "summary_count": summary_count,
            "message_count": convo_count,
            "total_count": len(seed_msgs),
            "trimmed_count": trimmed_count,
            "approx_tokens": approx_tokens,
            "oldest_timestamp": oldest,
            "newest_timestamp": newest,
        },
    }


def maybe_build_session_seed(
    llm_bawt, bot_id, model, user_prompt, service, thread_binding: dict | None = None,
    *, history_scope: str | None = None,
) -> list | None:
    """Decide whether to seed a fresh SDK session and, if so, build it (TASK-501).

    SINGLE source of truth shared by BOTH dispatch paths (streaming
    ``chat_streaming`` and non-streaming ``background_service.chat_completion``)
    so they can never drift — the reason non-streaming was a gap in the first
    place was two copy-pasted dispatch paths.

    Returns a list of ``{role, content, timestamp}`` seed messages to push to
    the bridge via ``inject_messages``, or None when no seed should attach:
    unseeded agent backend, continuity off, or a warm session that isn't a
    ``/new``. Never raises — any failure yields None. This is the SOLE seed
    payload authority (TASK-615/501 Phase 2): the app decides what continuity
    context is injected; the bridge only hydrates the fresh SDK transcript.

    ``history_scope`` is the scope this ``/new`` resolved once (the bot
    setting, or a scheduled reset's continuity parameter), shared with the
    summarize gate. It applies to the active-thread carry below, never to
    explicit-thread hydration or policy resets.
    """
    try:
        if (getattr(llm_bawt.bot, "agent_backend", "") or "") not in SEEDED_AGENT_BACKENDS:
            return None
        _binding = thread_binding or {}
        _thread_id = str(_binding.get("thread_session_id") or "").strip()
        _resume_id = str(_binding.get("thread_resume_id") or "").strip()
        _explicit = bool(_binding.get("explicit_thread"))
        from agent_bridge.maintenance import is_new_command

        _is_new = is_new_command(user_prompt)
        _reset_policy = str(_binding.get("session_policy") or "")
        _seed_session_id = str(_binding.get("seed_session_id") or "").strip()

        # A manual context reset is recorded on the fresh durable thread. Until
        # its provider key exists, honor the receipt exactly once: retained mode
        # seeds from the archived predecessor; clean mode explicitly seeds [].
        if not _resume_id and _reset_policy == "reset_without_history":
            return []
        if not _resume_id and _reset_policy == "reset_retain_history":
            seed = build_context_seed(
                bot_id, model, service, session_id=_seed_session_id or None, llm_bawt=llm_bawt,
            )
            return seed.get("messages", [])

        # A user-opened historical thread is hydration, not cross-session
        # carry. Resume its SDK transcript when present; otherwise seed from
        # that thread regardless of the active conversation's continuity
        # setting.
        if _explicit:
            if _resume_id:
                return None
            if not _thread_id:
                return None
            seed = build_context_seed(bot_id, model, service, session_id=_thread_id, llm_bawt=llm_bawt)
            return seed.get("messages") or None

        # Derive continuity from history_scope directly — NOT from the
        # separate session_memory_continuity mirror. The UI writes
        # history_scope FIRST; reading the mirror (written second) created
        # a race where /new arrived between the two writes and saw the old
        # "false" value. scope != "none" is the canonical truth; the mirror
        # is kept for legacy readers but no longer gates the seed.
        try:
            from ...utils.history import effective_history_scope, scope_flags
            _scope = effective_history_scope(llm_bawt.config_resolver, history_scope)
            _inc_hist, _inc_summ = scope_flags(_scope)
            continuity_on = _inc_hist or _inc_summ
        except Exception:
            continuity_on = False
        if not continuity_on:
            return [] if _thread_id and (_is_new or not _resume_id) else None

        # Warm active thread: the bridge resumes its canonical per-thread SDK
        # transcript, so injecting history again would duplicate context.
        if _resume_id and not _is_new:
            return None

        # Cold active thread (first turn/model switch) or /new: seed from the
        # bot's configured eligible history pool — the same unscoped assembly
        # exposed by the /new context preview. Restricting this to _thread_id
        # collapses back-to-back /new continuity to only the previous reset
        # exchange and makes live delivery disagree with the preview. Explicit
        # historical-thread hydration and reset_retain_history remain scoped in
        # their dedicated branches above.
        seed = build_context_seed(
            bot_id, model, service, llm_bawt=llm_bawt, history_scope=history_scope,
        )
        return seed.get("messages", [])
    except Exception:
        return None


@router.get("/v1/history/context-seed", tags=["History"])
def get_context_seed(
    bot_id: str = Query(..., description="Bot ID to build a session seed for"),
    model: str | None = Query(None, description="Optional model alias for context-window sizing"),
    user_id: str = Query(..., min_length=1, max_length=128, description="Requesting user; the seed is that user's history (TASK-1774)"),
):
    """Return the context payload a chat bot would receive, for seeding a fresh
    Claude Code SDK session (TASK-445). Thin HTTP wrapper over
    ``build_context_seed`` — kept for the bridge's legacy fallback path and any
    external callers.
    """
    service = get_service()
    effective_bot_id = bot_id or service._default_bot
    from ...model_catalog import ModelResolutionError
    try:
        return build_context_seed(bot_id, model, service, user_id=user_id)
    except ModelResolutionError as e:
        # Not a server fault: this bot's model can't be resolved for a seed
        # (e.g. a chat-harness bot whose model falls back to a responses-only
        # endpoint). Return a graceful "not previewable" payload so the
        # context-management UI degrades instead of surfacing a 500. TASK-620.
        log.info(f"Context seed not previewable for {effective_bot_id}: {e}")
        return {
            "bot_id": effective_bot_id,
            "model": model,
            "unavailable": True,
            "reason": str(e),
            "budget_tokens": None,
            "messages": [],
            "stats": {
                "summary_count": 0,
                "message_count": 0,
                "total_count": 0,
                "approx_tokens": 0,
                "oldest_timestamp": None,
                "newest_timestamp": None,
            },
        }
    except Exception as e:
        log.error(f"Failed to build context seed for {effective_bot_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
