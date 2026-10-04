from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AgentEventKind(str, Enum):
    ASSISTANT_DELTA = "assistant_delta"
    ASSISTANT_DONE = "assistant_done"
    # Provider-reported usage snapshot for the current model/API iteration.
    # Partial by definition; ASSISTANT_DONE remains the terminal authority.
    USAGE_UPDATE = "usage_update"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    USER_MESSAGE = "user_message"
    # CLI command_lifecycle state for a user-origin mid-turn message.
    # raw: {"message_id": <user message UUID>, "state": <CLI state>}.
    STEER_LIFECYCLE = "steer_lifecycle"
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    SYSTEM_NOTE = "system_note"
    # Transient provider/proxy liveness state. Structured metadata lives in
    # ``raw`` and must never be appended to persisted assistant text.
    UPSTREAM_STATUS = "upstream_status"
    ERROR = "error"
    # Emitted whenever a bridge clears its server-side session — via /new
    # in _handle_send or via the session.reset RPC.  Lets the frontend
    # clear its visible message buffer deterministically instead of
    # racing turn_complete timing.  Payload uses raw={"bot_id", "session_key",
    # "target", "had_session"} so the UI can scope the clear.
    SESSION_RESET = "session_reset"
    # TASK-257: emitted by the APP layer (not the bridges) when a /new — or an
    # explicit session.reset — rotates a bot's durable DB thread. Rotation is
    # app-owned (TASK-284), so a single app-side emit covers claude-code,
    # codex, openclaw AND chat bots for free. Payload (raw dict on the unified
    # stream): {_type:"thread_switched", bot_id, user_id, new_session_id,
    # source:"new_command"|"session_reset"}. Lets the UI refresh its
    # conversation list deterministically instead of text-matching "/new" or
    # racing turn_complete.
    THREAD_SWITCHED = "thread_switched"
    # Emitted when a bridge has paused the SDK on an interactive tool call
    # (currently only the Claude Agent SDK's built-in ``AskUserQuestion``).
    # The bridge holds an asyncio.Future keyed by ``tool_use_id``; the run
    # cannot continue until the app POSTs a chat.tool_result Redis command
    # carrying the user's answer.  Payload uses ``tool_name``,
    # ``tool_arguments`` (the original tool input — e.g. ``{questions: [...]}``
    # for AskUserQuestion), and the new ``tool_use_id`` field on AgentEvent
    # so the UI can echo it back when the user picks an answer.
    AWAIT_TOOL_RESULT = "await_tool_result"
    # Streamed model reasoning ("thinking") text, surfaced on a separate channel
    # from ASSISTANT_DELTA so the UI can render a collapsible "Thinking…" lane
    # without the reasoning ever entering the final assistant message body.
    # Emitted by the claude-code bridge when the proxy/native SDK yields Anthropic
    # ``thinking_delta`` frames (TASK-301). ``text`` carries the reasoning chunk.
    # Additive: consumers that don't know this kind ignore it (see from_dict).
    REASONING_DELTA = "reasoning_delta"
    # Emitted when an approval-gated tool policy matched a tool the model tried
    # to run (TASK-292). The bridge denies the call, emits this, and ends the
    # turn cleanly (same deferred/continuation model as AWAIT_TOOL_RESULT). The
    # app persists a tool_approval_requests row, the UI renders an Approve/Deny
    # card, and on Approve the app sends an approval.grant Redis command back to
    # the bridge + dispatches a continuation turn so the model re-issues the now
    # allow-listed call. Payload: ``tool_name``, ``tool_arguments`` (the original
    # tool input), ``tool_use_id`` (the SDK tool_use id), and ``raw`` carries
    # {policy_id, severity, subject, prompt, grant_key, action}.
    APPROVAL_REQUIRED = "approval_required"
    # Redacted evaluation audit, not an execution receipt or approval grant.
    # App commits this from the run stream independently of UI delivery.
    APPROVAL_DECISION = "approval_decision"
    # Emitted on the *continuation* turn when a previously approval-gated tool is
    # re-attempted and a live one-shot grant is consumed (TASK-305). The bridge
    # allows the call and emits this so the UI can mark that specific tool card
    # as having run with prior authorization (gold/lock affordance). This is the
    # single source of truth for "this execution was pre-approved" — the client
    # must never re-derive it (that would duplicate the subject→grant_key hashing
    # the bridge already owns). Payload: ``tool_use_id`` (the re-attempt's SDK
    # id, distinct from the original gated call) and ``raw`` carries
    # {policy_id, grant_key, severity}. Additive: unknown to older consumers,
    # which degrade it to SYSTEM_NOTE (see from_dict).
    TOOL_PREAPPROVED = "tool_preapproved"
    # Sub-agent lifecycle events (TASK-344). Emitted when the Claude Agent SDK
    # spawns sub-agents via the Agent/Workflow tool. The bridge detects
    # TaskStartedMessage, TaskProgressMessage, TaskUpdatedMessage, and
    # TaskNotificationMessage and translates them into these three kinds.
    # ``raw`` carries {task_id, description, task_type, usage, status, ...};
    # ``tool_use_id`` links back to the parent Agent tool call in the UI.
    SUBAGENT_STARTED = "subagent_started"
    SUBAGENT_PROGRESS = "subagent_progress"
    SUBAGENT_DONE = "subagent_done"
    # TASK-664: emitted immediately after a completed Edit/Write/NotebookEdit.
    # Attribution comes from that exact tool call; ``raw.file`` carries the
    # targeted path plus optional before/after bytes captured around the tool.
    # The app persists it before forwarding a lightweight public file_changed
    # event, so abrupt turn termination cannot lose already-completed edits.
    FILE_CHANGED = "file_changed"


@dataclass
class AgentEvent:
    event_id: str
    session_key: str
    run_id: str | None
    kind: AgentEventKind
    origin: str  # "user" | "system" | "heartbeat" | "cron" | "subagent"
    text: str | None = None
    tool_name: str | None = None
    tool_arguments: dict | None = None
    tool_result: Any | None = None
    # Versioned, lossless serialization of the exact provider-delivered result.
    # It travels only on the short-lived bridge run stream; the app persists it
    # before publishing a bounded preview to the public unified event stream.
    tool_result_payload: dict | None = None
    # Whether the tool call failed (the SDK's ToolResultBlock.is_error). Stamped
    # on TOOL_END events so the UI can tint a failed tool card red without
    # re-deriving failure from the result text (which is unreliable across
    # providers). None on non-tool events / bridges that don't surface it.
    tool_error: bool | None = None
    model: str | None = None
    seq: int | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw: dict = field(default_factory=dict)
    db_id: int | None = None  # populated after store
    # Per-turn token accounting from the upstream SDK. Successful turns carry
    # final ResultMessage-derived usage on ASSISTANT_DONE; interrupted/error
    # terminals may carry the latest provider-reported iteration snapshot with
    # usage_status="partial" (TASK-304). Shape: {input_tokens,
    # cache_read_tokens, cache_creation_tokens, output_tokens, context_window,
    # total_cost_usd, ...}.
    token_usage: dict | None = None
    # Identifies which agent backend produced this event ("claude-code",
    # "codex", "openclaw"). Lets the UI dispatch tool rendering by
    # (provider, tool_name) so each harness can show its own native tool
    # shapes (e.g. codex file_change carries only path+kind, no diff).
    provider: str | None = None
    # Frontend-supplied user-message UUID (or "local-user-*" placeholder)
    # that triggered this run.  Stamped on TOOL_START / TOOL_END events by
    # every bridge so the frontend can bucket tool activity under the
    # originating user message without relying on turn_id heuristics.
    # Optional because passive subscription paths (e.g. CLI sessions on
    # the OpenClaw gateway) have no originating frontend message.
    trigger_message_id: str | None = None
    # SDK-supplied tool_use id for the *specific* tool call this event refers
    # to.  Stamped on TOOL_START / TOOL_END / AWAIT_TOOL_RESULT events by the
    # claude-code bridge so the app can pair a user-supplied answer (via the
    # chat.tool_result Redis command) back to the paused SDK Future.  Other
    # bridges leave it unset.
    tool_use_id: str | None = None
    # SDK-supplied id of the *parent* tool call this event nests under. The
    # Claude Agent SDK stamps ``parent_tool_use_id`` on the AssistantMessage /
    # UserMessage / StreamEvent frames it emits for a sub-agent's *inner*
    # activity — its value is the ``tool_use_id`` of the Agent/Workflow tool
    # call that spawned the sub-agent. Threading it lets the UI nest a
    # sub-agent's tool cards (e.g. the Bash it ran) under the parent Agent card
    # instead of rendering them as flat siblings (TASK-344). ``None`` for
    # top-level activity — the overwhelming common case — so consumers that
    # ignore it degrade to the existing flat rendering with no regression.
    parent_tool_use_id: str | None = None
    # Media refs ({"asset_id": "ma_...", "kind": "image"}) produced *during*
    # this turn and already persisted to the media store — e.g. Playwright
    # screenshots the claude-code bridge offloaded instead of leaving the inline
    # base64 in the model context.  Stamped on the terminal ASSISTANT_DONE event
    # so the app can attach them to the bot's reply message (browsable per turn).
    attachments: list[dict] | None = None

    def to_dict(self) -> dict:
        """Serialize for Redis/JSON transport."""
        return {
            "event_id": self.event_id,
            "session_key": self.session_key,
            "run_id": self.run_id,
            "kind": self.kind.value,
            "origin": self.origin,
            "text": self.text,
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments,
            "tool_result": self.tool_result,
            "tool_result_payload": self.tool_result_payload,
            "tool_error": self.tool_error,
            "model": self.model,
            "seq": self.seq,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "db_id": self.db_id,
            "raw": self.raw,
            "token_usage": self.token_usage,
            "provider": self.provider,
            "trigger_message_id": self.trigger_message_id,
            "tool_use_id": self.tool_use_id,
            "parent_tool_use_id": self.parent_tool_use_id,
            "attachments": self.attachments,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AgentEvent:
        """Deserialize from Redis/JSON transport."""
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif ts is None:
            ts = datetime.now(timezone.utc)
        # Tolerant kind decode: an additive kind emitted by a newer producer must
        # not crash an older consumer (rolling deploy). Unknown values degrade to
        # SYSTEM_NOTE — a benign kind every consumer already ignores for display —
        # instead of raising ValueError (TASK-301).
        raw_kind = data.get("kind")
        try:
            kind = AgentEventKind(raw_kind)
        except ValueError:
            kind = AgentEventKind.SYSTEM_NOTE
        return cls(
            event_id=data["event_id"],
            session_key=data["session_key"],
            run_id=data.get("run_id"),
            kind=kind,
            origin=data.get("origin", "system"),
            text=data.get("text"),
            tool_name=data.get("tool_name"),
            tool_arguments=data.get("tool_arguments"),
            tool_result=data.get("tool_result"),
            tool_result_payload=data.get("tool_result_payload"),
            tool_error=data.get("tool_error"),
            model=data.get("model"),
            seq=data.get("seq"),
            timestamp=ts,
            db_id=data.get("db_id"),
            raw=data.get("raw", {}),
            token_usage=data.get("token_usage"),
            provider=data.get("provider"),
            trigger_message_id=data.get("trigger_message_id"),
            tool_use_id=data.get("tool_use_id"),
            parent_tool_use_id=data.get("parent_tool_use_id"),
            attachments=data.get("attachments"),
        )


def synthesize_event_id(session_key: str, event_type: str, data: dict, stream_seq: int) -> str:
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    payload = f"{session_key}:{event_type}:{canonical}:{stream_seq}"
    return hashlib.sha256(payload.encode("utf-8", errors="surrogateescape")).hexdigest()
