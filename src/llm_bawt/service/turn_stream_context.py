"""TASK-622: shared per-turn streaming state (extracted from chat_streaming.py).

``TurnStreamContext`` carries every local the extracted turn-worker / publisher
methods used to capture as closures. It is threaded BY REFERENCE, so the mutable
holder lists (``*_holder``) and the approval/await guard sets stay shared with
``chat_completion_stream`` exactly as they were when these were nested closures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# TASK-286: coalesce assistant text deltas into >= this many chars before
# publishing a ``text_delta`` to the unified Redis stream.
_TEXT_DELTA_FLUSH_CHARS = 80
# TASK-306 Section A: bounded retry for the confirmed approval-row commit.
_APPROVAL_PERSIST_MAX_ATTEMPTS = 3
_APPROVAL_PERSIST_BACKOFF_S = 0.25


@dataclass
class TurnStreamContext:
    """Per-turn state shared between the coordinator and the turn worker."""

    svc: Any  # the composed BackgroundService (was ``self`` in the closures)
    _approval_handled: Any
    _await_handled: Any
    _redis_sub: Any
    _tool_event_coordinator: Any
    _upstream_model: Any
    agent_attachments_holder: Any
    animation_holder: Any
    approval_id_holder: Any
    approval_persist_failed_holder: Any
    assistant_message_id: Any
    attachments_to_persist: Any
    bot_id: Any
    cancel_event: Any
    cancelled_holder: Any
    chunk_queue: Any
    done_event: Any
    full_response_holder: Any
    inject_seed_messages: Any
    is_agent_backend: Any
    llm_bawt: Any
    loop: Any
    media_store: Any
    model_alias: Any
    question_id_holder: Any
    reasoning_holder: Any
    request: Any
    timing_holder: Any
    token_usage_holder: Any
    tool_call_details_holder: Any
    tool_context_holder: Any
    trigger_message_id: Any
    tts_scrub: Any
    tts_scrubber: Any
    turn_log_id: Any
    user_attachments: Any
    user_author: Any
    user_id: Any
    user_prompt: Any
    # TASK-252: request-local explicit-thread SDK binding
    # ({thread_session_id, thread_resume_id?}) or None for continuous turns.
    thread_binding: Any = None
    # TASK-701: opaque, signed current-turn capability forwarded only to
    # harnesses that support request-local MCP headers (Claude SDK today).
    task_turn_capability: str | None = None
    execution: Any = None
