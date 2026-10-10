"""Parsed representation of a ``chat.send`` command (TASK-623).

Extracted verbatim from ``ClaudeSendMixin._handle_send`` so the field
parsing / validation / normalization seam lives on its own. Behavior is
unchanged: ``SendRequest.from_fields`` performs exactly the same decoding,
default-picking, and warning logging the monolith did inline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from ._bridge_helpers import _bot_slug_from_session_key
from .proxy.transport_policy import normalize_responses_transport

# Keep the original logger name so log lines are byte-identical to the
# pre-split monolith (which logged under ``claude_code_bridge.bridge``).
logger = logging.getLogger("claude_code_bridge.bridge")

_ALLOWED_EFFORT = {"low", "medium", "high", "xhigh", "max"}


@dataclass
class SendRequest:
    """Normalized fields for a single ``chat.send`` dispatch."""

    request_id: str
    session_key: str
    bot_slug: str
    message: str
    system_prompt: str | None
    model: str
    inject_messages: list | None
    trigger_message_id: str | None
    bot_effort: str | None
    bot_max_turns: int | None
    subagent_model: str | None
    bot_context_window: int | None
    responses_transport: str | None
    mcp_tool_timeout_ms: int | None
    question_answer_wait_seconds: int
    configured_disallowed_tools: object
    attachments: list[dict] = field(default_factory=list)
    # Per-thread SDK binding. The app resolves the durable thread (explicit
    # user selection or active thread) and its stored SDK session id.
    # The bridge resumes/persists the SDK session PER THREAD.
    thread_session_id: str | None = None
    thread_resume_id: str | None = None
    # True when the user explicitly selected this thread in the UI.
    # Controls /new gating: explicit threads never process /new.
    explicit_thread: bool = False
    # TASK-701: opaque app-signed current-turn authority, forwarded as an MCP
    # HTTP header. The bridge never opens or logs the payload.
    task_turn_capability: str | None = None
    skill_bundle: str | None = None
    # App-resolved Claude Code auto-compact window (None = CLI default).
    compact_threshold: int | None = None

    @classmethod
    def from_fields(cls, fields: dict) -> "SendRequest":
        """Decode a raw Redis command ``fields`` dict into a SendRequest.

        This is a straight move of the parsing block that used to open
        ``_handle_send`` — same defaults, same validation warnings.
        """
        request_id = fields.get("request_id", "")
        session_key = fields.get("session_key", "")
        bot_slug = (fields.get("bot_id", "") or "").strip() or _bot_slug_from_session_key(session_key)
        message = fields.get("message", "")
        system_prompt = fields.get("system_prompt") or None
        model = (fields.get("model") or "").strip()
        # TASK-501: history the app pre-assembled for a fresh-session seed.
        # When present, the bridge seeds from THIS instead of calling back to
        # /v1/history/context-seed. JSON-encoded list of {role, content} dicts.
        inject_messages = None
        _raw_inject = fields.get("inject_messages")
        if _raw_inject is not None:
            try:
                inject_messages = json.loads(_raw_inject)
            except (ValueError, TypeError) as _e:
                logger.warning("Failed to parse inject_messages: %s", _e)
        # Frontend-supplied user-message UUID; stamped on every emitted event
        # so the frontend can bucket tool activity under the originating user
        # message without falling back to turn_id heuristics.
        trigger_message_id = (fields.get("trigger_message_id") or "").strip() or None

        # Per-bot ClaudeAgentOptions tuning (TASK: bot config -> SDK).
        # ``effort`` constrains thinking depth; ``max_turns`` caps the
        # autonomous tool-loop length per dispatch. Both default to None
        # (SDK default) when the bot doesn't override.
        effort_raw = (fields.get("effort") or "").strip().lower() or None
        bot_effort = effort_raw if effort_raw in _ALLOWED_EFFORT else None
        if effort_raw and bot_effort is None:
            logger.warning(
                "Ignoring invalid effort=%r for %s (allowed: %s)",
                effort_raw, bot_slug, sorted(_ALLOWED_EFFORT),
            )
        max_turns_raw = (fields.get("max_turns") or "").strip()
        bot_max_turns: int | None = None
        if max_turns_raw:
            try:
                mt = int(max_turns_raw)
                bot_max_turns = mt if mt > 0 else None
            except ValueError:
                logger.warning(
                    "Ignoring invalid max_turns=%r for %s (must be positive int)",
                    max_turns_raw, bot_slug,
                )

        # TASK-546: Per-bot subagent/background model override. When set,
        # the bridge injects it as ANTHROPIC_SMALL_FAST_MODEL and
        # CLAUDE_CODE_SUBAGENT_MODEL on proxy-routed turns so Claude Code's
        # internal background Haiku calls and subagent model resolution use
        # a provider-qualified model the proxy accepts instead of bare
        # Anthropic IDs (claude-haiku-4-5-...) that the proxy rejects.
        subagent_model = (fields.get("subagent_model") or "").strip() or None

        # TASK-609: app-resolved catalog context window. The app (single Tier-2
        # authority) resolves the true per-model window and sends it down; the
        # bridge consumes this scalar to report the real window for
        # proxy-routed models the Claude CLI defaults to 200k. No bridge-side
        # window table. Absent/invalid -> None -> defer to the SDK's own view.
        cw_raw = (fields.get("context_window") or "").strip()
        bot_context_window: int | None = None
        if cw_raw:
            try:
                cw = int(cw_raw)
                bot_context_window = cw if cw > 0 else None
            except ValueError:
                logger.warning(
                    "Ignoring invalid context_window=%r for %s (must be positive int)",
                    cw_raw, bot_slug,
                )

        # App-resolved Claude Code auto-compact window (catalog per-model value,
        # else the global agent_compact_threshold). Absent/invalid -> None ->
        # the CLI's own default.
        ct_raw = (fields.get("compact_threshold") or "").strip()
        compact_threshold: int | None = None
        if ct_raw:
            try:
                ct = int(ct_raw)
                compact_threshold = ct if ct > 0 else None
            except ValueError:
                logger.warning(
                    "Ignoring invalid compact_threshold=%r for %s (must be positive int)",
                    ct_raw, bot_slug,
                )

        # TASK-896: app-resolved endpoint transport policy. The model catalog is
        # authoritative; absent/invalid values fall back to ordinary supervised
        # SSE in the adapter rather than guessing from the model name.
        transport_raw = fields.get("responses_transport")
        responses_transport = normalize_responses_transport(transport_raw)
        if transport_raw and responses_transport is None:
            logger.warning(
                "Ignoring invalid responses_transport=%r for %s",
                transport_raw, bot_slug,
            )

        # TASK-618: app-resolved DB setting plus 30s headroom, expressed in the
        # milliseconds the Claude CLI expects. Invalid/absent -> CLI default.
        mcp_timeout_raw = (fields.get("mcp_tool_timeout_ms") or "").strip()
        mcp_tool_timeout_ms: int | None = None
        if mcp_timeout_raw:
            try:
                parsed_timeout = int(mcp_timeout_raw)
                mcp_tool_timeout_ms = parsed_timeout if parsed_timeout > 0 else None
            except ValueError:
                logger.warning(
                    "Ignoring invalid mcp_tool_timeout_ms=%r for %s",
                    mcp_timeout_raw, bot_slug,
                )

        # Missing or invalid payload disables the wait; the bridge never chooses
        # a setting default (that belongs to the app's SettingDefinition).
        try:
            question_answer_wait_seconds = max(0, int(fields.get("question_answer_wait_seconds", "0")))
        except (TypeError, ValueError):
            logger.warning("Invalid question_answer_wait_seconds in chat.send")
            question_answer_wait_seconds = 0

        # TASK-593: app-resolved, DB-managed base SDK tool policy. Keep the raw
        # field until proxy routing is known; the pure resolver below decodes it,
        # falls back safely, and adds proxy-only exclusions.
        configured_disallowed_tools = fields.get("disallowed_tools")

        # Per-thread SDK binding. A resume value containing ":" is a routing
        # key, never an SDK session id — drop it.
        thread_session_id = (fields.get("thread_session_id") or "").strip() or None
        thread_resume_id = (fields.get("thread_resume_id") or "").strip() or None
        if thread_resume_id and ":" in thread_resume_id:
            logger.warning(
                "Ignoring routing-key thread_resume_id for %s: %s",
                bot_slug, thread_resume_id,
            )
            thread_resume_id = None
        explicit_thread = (fields.get("explicit_thread") or "").strip() in ("1", "true")
        task_turn_capability = (
            fields.get("task_turn_capability") or ""
        ).strip() or None

        attachments_raw = fields.get("attachments", "")
        attachments: list[dict] = []
        if attachments_raw:
            try:
                attachments = json.loads(attachments_raw)
            except json.JSONDecodeError:
                pass

        return cls(
            request_id=request_id,
            session_key=session_key,
            bot_slug=bot_slug,
            message=message,
            system_prompt=system_prompt,
            model=model,
            inject_messages=inject_messages,
            trigger_message_id=trigger_message_id,
            bot_effort=bot_effort,
            bot_max_turns=bot_max_turns,
            subagent_model=subagent_model,
            bot_context_window=bot_context_window,
            compact_threshold=compact_threshold,
            responses_transport=responses_transport,
            mcp_tool_timeout_ms=mcp_tool_timeout_ms,
            question_answer_wait_seconds=question_answer_wait_seconds,
            configured_disallowed_tools=configured_disallowed_tools,
            attachments=attachments,
            thread_session_id=thread_session_id,
            thread_resume_id=thread_resume_id,
            explicit_thread=explicit_thread,
            task_turn_capability=task_turn_capability,
            skill_bundle=fields.get("skill_bundle"),
        )
