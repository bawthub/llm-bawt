"""Capture the already-selected seed after the destination thread is resolved."""
from __future__ import annotations

import logging
from ..memory.session_continuity import SessionContinuity
from ..mcp_server.storage import get_storage

log = logging.getLogger(__name__)


def record_session_continuity(llm_bawt, binding, messages, *, bot_id, user_id, history_scope=None):
    # None is no assembly decision (including a warm resume), not an empty seed.
    session_id = (binding or {}).get("thread_session_id")
    if messages is None or not session_id:
        return
    try:
        settings = {}
        for key in ("history_scope", "history_tokens", "history_max_age_hours", "summary_count",
                    "seed_message_max_chars", "compact_context"):
            value = llm_bawt.config_resolver.resolve_config_setting(key)
            settings[key] = value.value
        if history_scope is not None:
            # Record the scope the seed was actually built with, not a re-read.
            settings["history_scope"] = history_scope
        manifest = SessionContinuity.manifest(messages, settings=settings,
                                              model=llm_bawt.resolved_model_alias)
        SessionContinuity(get_storage().engine).record(session_id, bot_id, user_id, manifest)
    except Exception:
        # Observability must not turn a valid reset into a failed conversation.
        log.warning("Unable to record continuity for %s/%s", bot_id, session_id, exc_info=True)
