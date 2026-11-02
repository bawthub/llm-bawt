"""Tell an agent its model changed while its session was kept (TASK-1956).

A Claude Code session keeps the system prompt it started with: the CLI
ignores a new prompt on resume, and llm-bawt pins those bytes per session
(TASK-1318). A model switch deliberately keeps the SDK transcript, so the
pinned ``<runtime-context>`` block keeps naming the old model until the next
session boundary.

When a kept pin names a different model from the one now configured, the
turn gets the ``agents.model_switch_notice`` registry prompt appended to its
user message. The session row records the last model noticed in this pin
episode, so every later switch (including a return to the pinned model) is
noticed once and an unchanged model is not. A boundary turn sends a fresh
render that already names the right model, so it gets nothing.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

MODEL_SWITCH_NOTICE_KEY = "agents.model_switch_notice"
#: ``sessions.session_metadata`` key recording the last notice:
#: ``{"pin_turn_id", "model"}``.
STATE_KEY = "model_switch_notice"

_RUNTIME_MODEL = re.compile(r"<runtime-context>\s*model:[ \t]*(\S+)")


def pinned_model(prompt: str | None) -> str | None:
    """The model named by a prompt's ``<runtime-context>`` block, if any."""
    match = _RUNTIME_MODEL.search(prompt or "")
    return match.group(1) if match else None


def told_model(marker: Any, decision: Any) -> str | None:
    """The last model the retained transcript was told it runs on.

    A notice in this pin episode overrides the pinned block; otherwise the
    pinned ``<runtime-context>`` is the latest word.
    """
    if isinstance(marker, dict) and marker.get("pin_turn_id") == decision.pin_turn_id:
        model = marker.get("model")
        if isinstance(model, str) and model:
            return model
    return pinned_model(decision.text)


def notice_model(marker: Any, decision: Any, current_model: str | None) -> str | None:
    """The current model when the transcript was last told a different one."""
    current = str(current_model or "").strip()
    if decision is None or decision.is_boundary or not current:
        return None
    if pinned_model(decision.text) is None:  # no block to correct
        return None
    told = told_model(marker, decision)
    return None if told == current else current


def render_notice(model: str, bot_id: str | None) -> str:
    from .prompt_registry import PromptResolver, get_prompt_resolver

    try:
        body = get_prompt_resolver().render(
            key=MODEL_SWITCH_NOTICE_KEY, variables={"model": model},
            scope_type="bot" if bot_id else "global", scope_id=bot_id,
        )
        if body.strip():
            return body
    except Exception as error:  # a DB failure falls back to the registered default
        logger.warning("model switch notice resolve failed, using its default: %s", error)
    definition = PromptResolver.definition_for(MODEL_SWITCH_NOTICE_KEY)
    if definition is None:
        raise RuntimeError(f"{MODEL_SWITCH_NOTICE_KEY} is not registered")
    return definition.default_body().format(model=model)


def _claim(engine, session_id: str, decision: Any, current_model: str | None) -> str | None:
    """Under the session row lock, record and return the model to notice, if any."""
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT session_metadata FROM sessions WHERE id=:sid FOR UPDATE"),
            {"sid": session_id},
        ).first()
        if row is None:
            return None
        meta = row[0] if isinstance(row[0], dict) else {}
        model = notice_model(meta.get(STATE_KEY), decision, current_model)
        if model is None:
            return None
        conn.execute(
            text("""
                UPDATE sessions SET session_metadata=jsonb_set(
                    COALESCE(session_metadata, '{}'::jsonb),
                    '{model_switch_notice}', CAST(:mark AS jsonb)
                ) WHERE id=:sid
            """),
            {"mark": json.dumps({"pin_turn_id": decision.pin_turn_id, "model": model}),
             "sid": session_id},
        )
    return model


def model_switch_notice(engine, *, bot_id: str | None, session_id: str | None,
                        decision: Any, current_model: str | None,
                        eligible: bool = True) -> str | None:
    """The notice line for this turn, or None. Never raises.

    ``eligible`` is the caller's turn-kind check: a maintenance command's
    leading token must reach the bridge unchanged.
    """
    if engine is None or not session_id or not eligible:
        return None
    try:
        if decision is None or decision.is_boundary or pinned_model(decision.text) is None:
            return None  # cheap exits before the row lock
        model = _claim(engine, session_id, decision, current_model)
        if model is None:
            return None
        logger.info("Session %s was last told another model; noticing %s", session_id, model)
        return render_notice(model, bot_id)
    except Exception:
        logger.warning("model switch notice failed for session %s; the turn continues without it",
                       session_id, exc_info=True)
        return None


def apply_model_switch_notice(ctx: Any, client: Any, messages: list, decision: Any) -> list:
    """``messages`` with the notice on the last user message when it fires."""
    from .working_memory.turn import append_to_last_user, is_maintenance_command

    notice = model_switch_notice(
        ctx.svc._turn_log_store.engine,
        bot_id=ctx.bot_id,
        session_id=str((ctx.thread_binding or {}).get("thread_session_id") or "").strip(),
        decision=decision,
        current_model=(getattr(client, "_bot_config", None) or {}).get("model"),
        eligible=not is_maintenance_command(ctx.user_prompt),
    )
    return append_to_last_user(messages, notice) if notice else messages


__all__ = ["apply_model_switch_notice", "MODEL_SWITCH_NOTICE_KEY", "STATE_KEY", "model_switch_notice", "notice_model", "pinned_model", "told_model"]
