"""Shared agent-bridge backend — generic base for bridge-style agents.

All bridge backends (OpenClaw, Codex, Claude Code) speak the same
Redis command/event protocol; only the upstream transport differs:

  Main App → Redis command → Bridge → upstream (WS / SDK / process)
  Upstream → Bridge → Redis run stream → Main App

Concrete backends subclass :class:`AgentBridgeBackend` and override
``name`` (registry key) plus ``_resolve_session_key`` if they need
backend-specific session routing. Everything else — command publish,
run subscription, tool-call assembly, token-usage forwarding — is
inherited.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

from .base import AgentBackend
from ..utils.config import Config

logger = logging.getLogger(__name__)

# Module-level subscriber singleton — set by service startup
_subscriber = None


def set_agent_subscriber(subscriber) -> None:
    """Set the Redis subscriber shared by all AgentBridgeBackend instances."""
    global _subscriber
    _subscriber = subscriber


def get_agent_subscriber():
    """Get the current Redis subscriber."""
    return _subscriber


# ---------------------------------------------------------------------------
# Structured result returned by chat_full()
# ---------------------------------------------------------------------------

@dataclass
class AgentToolCall:
    """One tool invocation extracted from an agent-bridge response."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    display_name: str = ""  # friendly name for UI (e.g. first word of exec command)
    result: Any | None = None
    tool_use_id: str | None = None

    def __post_init__(self):
        if not self.display_name:
            self.display_name = _friendly_tool_name(self.name, self.arguments)


@dataclass
class AgentResult:
    """Full structured result from an agent-bridge run."""

    text: str
    model: str = ""
    provider: str = ""
    duration_ms: int = 0
    session_id: str = ""
    tool_calls: list[AgentToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def _friendly_tool_name(name: str, arguments: dict[str, Any]) -> str:
    if name == "exec":
        cmd = arguments.get("command", "")
        if cmd:
            first_word = cmd.split()[0] if cmd.split() else name
            first_word = first_word.rsplit("/", 1)[-1]
            return first_word
    return name


class AgentBridgeBackend(AgentBackend):
    """Agent backend that sends messages through the agent-bridge transport.

    Subclassed by ``OpenClawBackend``, ``CodexBackend``, ``ClaudeCodeBackend``.
    Each subclass overrides ``name`` (registry key, also stamped onto every
    event as ``provider``) and may override ``_resolve_session_key`` to
    define backend-specific session routing.

    Configuration keys (``agent_backend_config`` in bot DB profile):
        session_key: routing key (semantics defined by the subclass)
        timeout_seconds: Max wait for response (default: 600)
    """

    name = "agent-bridge"

    def __init__(self) -> None:
        self._config = Config()
        # Per-request state is stored in threading.local to avoid races
        # between concurrent stream_raw() calls (e.g. different bots).
        self._thread_local = threading.local()
        self._active_request_id: str | None = None

    def _resolve_session_key(self, config: dict) -> str:
        explicit = str(config.get("session_key") or "").strip()
        if explicit:
            return explicit
        return "main"

    async def _persist_approval_decision(self, event, config: dict) -> bool:
        """App persistence boundary, independent of SSE/consumer iteration.

        Commit before draining the next run event. Three bounded attempts; a DB
        outage never retroactively changes a tool decision. No durable retry
        outbox exists here: exhaustion/app death can lose audit events.
        """
        from ..approval_policies import ToolApprovalPolicyStore

        def commit():
            store = getattr(self, "_approval_audit_store", None)
            if store is None or store.engine is None:
                store = ToolApprovalPolicyStore(self._config)
                self._approval_audit_store = store
            store.record_bridge_decision(
                event, bot_id=config.get("bot_id"), user_id=config.get("user_id"),
                session_id=config.get("thread_session_id"), turn_id=config.get("turn_id"),
            )

        for attempt in range(3):
            try:
                await asyncio.to_thread(commit)
                return True
            except Exception as exc:
                # Never log SQL parameters or the payload on persistence errors.
                logger.warning("Bridge decision audit commit failed request=%s event=%s attempt=%d error=%s",
                               event.run_id, event.event_id, attempt + 1, type(exc).__name__)
                if attempt < 2:
                    await asyncio.sleep(0.1 * (attempt + 1))
        logger.error("Bridge decision audit LOST request=%s event=%s; best-effort retries exhausted",
                     event.run_id, event.event_id)
        return False

    def stream_raw(
        self,
        prompt: str,
        config: dict,
        attachments: list | None = None,
        trigger_message_id: str | None = None,
    ) -> Iterator[str | dict[str, Any]]:
        """Stream OpenClaw response deltas and tool events via the bridge.

        Sends a command to the bridge via Redis, then subscribes to the
        per-run response stream. Yields text deltas and tool event dicts.

        ``trigger_message_id`` is forwarded to the bridge so every emitted
        tool event carries the originating frontend user-message UUID.
        """
        subscriber = get_agent_subscriber()
        if not subscriber:
            raise RuntimeError(
                "Agent-bridge subscriber not initialized. "
                "Ensure REDIS_URL is set and the bridge is running."
            )

        session_key = self._resolve_session_key(config)
        timeout = int(config.get("timeout_seconds", 600))
        configured_request_id = str(config.get("request_id") or "").strip()
        request_id = configured_request_id or f"req_{uuid.uuid4().hex}"
        self._active_request_id = request_id  # diagnostics only; never abort authority
        execution = config.get("turn_execution")
        if execution is not None:
            execution.bind_bridge(backend=self.name, session_key=session_key, request_id=request_id)
        # Fall back to config so callers that thread it through agent_backend_config
        # (rather than as a kwarg) still propagate.
        if not trigger_message_id:
            tmid = config.get("trigger_message_id")
            if isinstance(tmid, str) and tmid.strip():
                trigger_message_id = tmid.strip()

        request_started = time.time()
        first_delta_at: float | None = None
        tool_calls: list[AgentToolCall] = []
        text_parts: list[str] = []
        upstream_model: str = ""
        token_usage: dict[str, Any] = {}
        termination_reason = "completed"
        event_callback = config.get("event_callback")

        def _notify_event(item: dict[str, Any]) -> None:
            if not callable(event_callback):
                return
            try:
                event_callback(item)
            except Exception:
                logger.debug("agent event callback failed", exc_info=True)

        import queue as queue_mod

        result_queue: queue_mod.Queue[str | dict | None] = queue_mod.Queue()
        error_holder: list[Exception] = []

        try:
            # Schedule async work on the running event loop from this sync context.
            # We use a separate thread with its own Redis client to avoid
            # event loop cross-contamination.
            import threading

            def _run_in_thread():
                async def _worker():
                    from agent_bridge.events import AgentEventKind
                    # Create a fresh async Redis client for this thread's loop
                    redis_url = subscriber._redis.connection_pool.connection_kwargs.get(
                        "url", None
                    )
                    if redis_url is None:
                        # Reconstruct URL from connection kwargs
                        ck = subscriber._redis.connection_pool.connection_kwargs
                        host = ck.get("host", "localhost")
                        port = ck.get("port", 6379)
                        db = ck.get("db", 0)
                        redis_url = f"redis://{host}:{port}/{db}"

                    from agent_bridge.subscriber import RedisSubscriber
                    local_sub = RedisSubscriber(redis_url)
                    await local_sub.connect()
                    pending_error: RuntimeError | None = None
                    try:
                        if execution is not None:
                            execution.check_cancelled()
                        await local_sub.send_command(
                            session_key=session_key,
                            message=prompt,
                            request_id=request_id,
                            attachments=attachments or [],
                            system_prompt=config.get("system_prompt"),
                            model=config.get("model"),
                            backend=self.name,
                            bot_id=str(config.get("bot_id") or "").strip() or None,
                            trigger_message_id=trigger_message_id,
                            effort=(
                                str(config.get("effort")).strip().lower()
                                if config.get("effort") else None
                            ),
                            max_turns=(
                                int(config.get("max_turns"))
                                if config.get("max_turns") is not None else None
                            ),
                            skill_bundle=config.get("skill_bundle"),
                            subagent_model=(
                                str(config.get("subagent_model")).strip()
                                if config.get("subagent_model") else None
                            ),
                            disallowed_tools=(
                                config.get("disallowed_tools")
                                if isinstance(config.get("disallowed_tools"), list)
                                else None
                            ),
                            inject_messages=config.get("inject_messages"),
                            context_window=(
                                int(config.get("context_window"))
                                if config.get("context_window") else None
                            ),
                            compact_threshold=(
                                int(config.get("compact_threshold"))
                                if config.get("compact_threshold") else None
                            ),
                            responses_transport=(
                                str(config.get("responses_transport")).strip()
                                if config.get("responses_transport") else None
                            ),
                            mcp_tool_timeout_ms=(
                                int(config.get("mcp_tool_timeout_ms"))
                                if config.get("mcp_tool_timeout_ms") else None
                            ),
                            thread_session_id=(
                                str(config.get("thread_session_id") or "").strip()
                                or None
                            ),
                            thread_resume_id=(
                                str(config.get("thread_resume_id") or "").strip()
                                or None
                            ),
                            explicit_thread=bool(config.get("explicit_thread")),
                            task_turn_capability=(
                                str(config.get("task_turn_capability") or "").strip()
                                or None
                            ),
                        )
                        if execution is not None and execution.dispatched():
                            # Stop won while send_command was in flight. Publish
                            # AFTER chat.send so cancellation cannot overtake it.
                            result = await local_sub.send_rpc(
                                "chat.abort", {"sessionKey": session_key, "requestId": request_id},
                                f"abort_{uuid.uuid4().hex}", timeout_s=25, backend=self.name,
                            )
                            if not (result.get("cancelled") is True or result.get("aborted") is True):
                                logger.warning("Dispatch-time abort unconfirmed request=%s", request_id)
                        logger.info(
                            "%s request via bridge: session=%s request_id=%s",
                            self.name,
                            session_key, request_id,
                        )

                        nonlocal first_delta_at
                        async for event in local_sub.subscribe_run(
                            request_id, timeout_s=timeout,
                        ):
                            if event.kind == AgentEventKind.ASSISTANT_DELTA:
                                delta = event.text or ""
                                if delta:
                                    if first_delta_at is None:
                                        first_delta_at = time.time()
                                        logger.info(
                                            "%s first-token latency: %.1fms request_id=%s",
                                        self.name,
                                            (first_delta_at - request_started) * 1000,
                                            request_id,
                                        )
                                    text_parts.append(delta)
                                    result_queue.put(delta)

                            elif event.kind == AgentEventKind.USAGE_UPDATE:
                                if event.token_usage:
                                    result_queue.put({
                                        "event": "live_usage",
                                        "token_usage": event.token_usage,
                                        "model": event.model,
                                    })

                            elif event.kind == AgentEventKind.UPSTREAM_STATUS:
                                meta = event.raw.get("upstream_status", {}) if isinstance(event.raw, dict) else {}
                                result_queue.put({
                                    "event": "upstream_status",
                                    "text": event.text or "",
                                    "state": meta.get("state"),
                                    "attempt": meta.get("attempt"),
                                    "next_attempt": meta.get("next_attempt"),
                                    "max_attempts": meta.get("max_attempts"),
                                    "transport": meta.get("transport"),
                                    "fallback_transport": meta.get("fallback_transport"),
                                    "stall_phase": meta.get("stall_phase"),
                                    "productive_idle_seconds": meta.get("productive_idle_seconds"),
                                    "elapsed_seconds": meta.get("elapsed_seconds"),
                                    "provider": meta.get("provider") or event.provider,
                                    "trigger_message_id": event.trigger_message_id,
                                })

                            elif event.kind == AgentEventKind.ASSISTANT_DONE:
                                if isinstance(event.raw, dict) and event.raw.get("end_reason") == "aborted":
                                    result_queue.put({"event": "turn_cancelled"})
                                # Capture actual upstream model if provided
                                if event.model:
                                    nonlocal upstream_model
                                    upstream_model = event.model
                                    result_queue.put({
                                        "event": "metadata",
                                        "upstream_model": upstream_model,
                                    })
                                # Forward token usage / context window so the
                                # SSE generator can surface it on turn_complete
                                # for the chat UI's per-bubble usage pill.
                                if event.token_usage:
                                    token_usage.clear()
                                    token_usage.update(event.token_usage)
                                    result_queue.put({
                                        "event": "token_usage",
                                        "token_usage": event.token_usage,
                                    })
                                # Media the bridge persisted during the turn
                                # (e.g. Playwright screenshots) — forward the
                                # {asset_id, kind} refs so the app can attach
                                # them to this assistant reply.
                                if event.attachments:
                                    result_queue.put({
                                        "event": "attachments",
                                        "attachments": event.attachments,
                                    })
                                # ASSISTANT_DONE carries the complete response
                                # text.  Yield any portion not already streamed
                                # as deltas — this is critical for tool-heavy
                                # turns where the final synthesis text only
                                # appears in the done event, not as individual
                                # ASSISTANT_DELTA events.
                                done_text = event.text or ""
                                if done_text:
                                    accumulated = "".join(text_parts)
                                    if done_text.startswith(accumulated):
                                        extra = done_text[len(accumulated):]
                                        if extra.strip():
                                            text_parts.append(extra)
                                            result_queue.put(extra)
                                            logger.info(
                                                "ASSISTANT_DONE: supplemented %d chars "
                                                "(accumulated=%d, done=%d) request_id=%s",
                                                len(extra), len(accumulated),
                                                len(done_text), request_id,
                                            )
                                    elif len(done_text) > len(accumulated):
                                        # Deltas don't prefix-match done text;
                                        # yield the full done text so the
                                        # response is at least complete for
                                        # persistence.  May duplicate some
                                        # content in the SSE stream, but the
                                        # DB record will be correct on reload.
                                        logger.warning(
                                            "ASSISTANT_DONE: prefix mismatch, "
                                            "yielding full done_text (done=%d, "
                                            "accumulated=%d) request_id=%s",
                                            len(done_text), len(accumulated),
                                            request_id,
                                        )
                                        text_parts.clear()
                                        text_parts.append(done_text)
                                        result_queue.put(
                                            done_text[len(accumulated):]
                                            if accumulated
                                            else done_text
                                        )

                            elif event.kind == AgentEventKind.TOOL_START:
                                tc = AgentToolCall(
                                    name=event.tool_name or "unknown",
                                    arguments=event.tool_arguments or {},
                                    tool_use_id=event.tool_use_id,
                                )
                                tool_calls.append(tc)
                                logger.info("%s tool event: %s", self.name, tc.display_name)
                                tool_item = {
                                    "event": "tool_call",
                                    "name": tc.display_name,
                                    "arguments": tc.arguments,
                                    "provider": event.provider,
                                    # Forward bridge-stamped originating user-msg UUID
                                    # so the SSE emitter doesn't have to fall back to
                                    # request-scope state.
                                    "trigger_message_id": event.trigger_message_id,
                                    # Forward the SDK tool_use id (toolu_…) so the
                                    # unified tool_start event can stamp it on the
                                    # live activity card.  AskUserQuestion's inline
                                    # picker pairs the card to its pending-question
                                    # row by this id; without it the live card can't
                                    # match and the picker only appears on reload.
                                    "tool_use_id": event.tool_use_id,
                                    # Parent Agent/Workflow tool_use_id when this call
                                    # is a sub-agent's inner activity — lets the UI nest
                                    # it under the parent card (TASK-344). None otherwise.
                                    "parent_tool_use_id": event.parent_tool_use_id,
                                }
                                _notify_event(tool_item)
                                result_queue.put(tool_item)

                            elif event.kind == AgentEventKind.TOOL_END:
                                if tool_calls:
                                    # Gateway may not include actual result —
                                    # use meta/summary or a placeholder so the
                                    # SSE handler always sees a non-None result.
                                    result = event.tool_result if event.tool_result is not None else "(completed)"
                                    # Match completion to its start by the stable SDK
                                    # tool_use id. Parallel calls need not finish LIFO.
                                    matched_call = tool_calls[-1]
                                    if event.tool_use_id:
                                        for candidate in reversed(tool_calls):
                                            if getattr(candidate, "tool_use_id", None) == event.tool_use_id:
                                                matched_call = candidate
                                                break
                                    matched_call.result = result
                                    tool_item = {
                                        "event": "tool_result",
                                        "name": matched_call.display_name,
                                        "arguments": matched_call.arguments,
                                        "result": result,
                                        "tool_result_payload": event.tool_result_payload,
                                        "provider": event.provider,
                                        "trigger_message_id": event.trigger_message_id,
                                        "tool_use_id": event.tool_use_id,
                                        # Carry the SDK failure flag (bridge-stamped
                                        # tool_error) downstream so the published
                                        # tool_end can persist is_error. Without this
                                        # the red error ring never lights for agent
                                        # turns — the flag died here.
                                        "is_error": event.tool_error,
                                        # Parent Agent/Workflow tool_use_id for nesting
                                        # sub-agent tool results under the parent card
                                        # (TASK-344). None for top-level calls.
                                        "parent_tool_use_id": event.parent_tool_use_id,
                                        # Media refs the bridge persisted for THIS
                                        # tool call (Playwright screenshots) so the
                                        # UI can render them inline in the tool card
                                        # immediately (TASK-483). The end-of-turn
                                        # ASSISTANT_DONE attachments flush is
                                        # unchanged and still backs history/reload.
                                        "attachments": event.attachments or None,
                                    }
                                    _notify_event(tool_item)
                                    result_queue.put(tool_item)

                            elif event.kind == AgentEventKind.AWAIT_TOOL_RESULT:
                                # Claude-code bridge has paused the SDK on an
                                # interactive tool call (AskUserQuestion).  The
                                # frontend renders a picker and POSTs the user's
                                # answer to /v1/chat/tool-result, which fans out
                                # a chat.tool_result Redis command back to the
                                # bridge — that resolves the pending Future and
                                # the SDK turn continues.  No state to mutate
                                # here; just forward the event downstream.
                                result_queue.put({
                                    "event": "await_tool_result",
                                    "tool_name": event.tool_name or "",
                                    "tool_use_id": event.tool_use_id or "",
                                    "arguments": event.tool_arguments or {},
                                    "session_key": event.session_key,
                                    "provider": event.provider,
                                    "trigger_message_id": event.trigger_message_id,
                                })

                            elif event.kind == AgentEventKind.APPROVAL_DECISION:
                                # Persist in this app-owned run consumer, including
                                # chat_full() and detached/disconnected UI requests.
                                await self._persist_approval_decision(event, config)

                            elif event.kind == AgentEventKind.APPROVAL_REQUIRED:
                                logger.info(
                                    "DEBUG-292 agent_bridge: APPROVAL_REQUIRED received tool=%s tuid=%s raw_keys=%s",
                                    event.tool_name, event.tool_use_id,
                                    list((event.raw or {}).keys()),
                                )
                                # TASK-292: an approval-gated tool policy matched.
                                # The bridge denied the call and ended the turn;
                                # the app persists a tool_approval_requests row and
                                # surfaces an Approve/Deny card. The user's decision
                                # comes back via /v1/chat/approvals/{id}/resolve,
                                # which grants the bridge a one-shot allow and the
                                # client dispatches a continuation turn. Forward the
                                # event with its policy metadata (event.raw).
                                meta = event.raw if isinstance(event.raw, dict) else {}
                                result_queue.put({
                                    "event": "approval_required",
                                    "tool_name": event.tool_name or "",
                                    "tool_use_id": event.tool_use_id or "",
                                    "arguments": event.tool_arguments or {},
                                    "session_key": event.session_key,
                                    "provider": event.provider,
                                    "trigger_message_id": event.trigger_message_id,
                                    "policy_id": meta.get("policy_id"),
                                    "severity": meta.get("severity") or "medium",
                                    "category": meta.get("category"),
                                    "subject": meta.get("subject") or "",
                                    "prompt": meta.get("prompt") or "",
                                    "grant_key": meta.get("grant_key") or "",
                                })

                            elif event.kind == AgentEventKind.TOOL_PREAPPROVED:
                                # TASK-305: a previously gated tool was re-attempted
                                # on this continuation turn and consumed a live
                                # one-shot grant. Forward so the UI can mark that
                                # specific tool card as pre-approved (gold/lock).
                                meta = event.raw if isinstance(event.raw, dict) else {}
                                result_queue.put({
                                    "event": "tool_preapproved",
                                    "tool_name": event.tool_name or "",
                                    "tool_use_id": event.tool_use_id or "",
                                    "provider": event.provider,
                                    "trigger_message_id": event.trigger_message_id,
                                    "policy_id": meta.get("policy_id"),
                                    "severity": meta.get("severity") or "medium",
                                    "grant_key": meta.get("grant_key") or "",
                                })

                            elif event.kind == AgentEventKind.REASONING_DELTA:
                                # Model reasoning ("thinking"). Forward as a dict
                                # item — NOT a str — so downstream text
                                # accumulation (full_response_holder) never sees
                                # it and it cannot leak into the saved message.
                                # Deliberately does not touch text_parts or
                                # first_delta_at (reasoning is not the answer and
                                # must not register as first-token latency).
                                # TASK-301.
                                reasoning = event.text or ""
                                if reasoning:
                                    result_queue.put({
                                        "event": "reasoning",
                                        "text": reasoning,
                                        "provider": event.provider,
                                    })

                            elif event.kind == AgentEventKind.SUBAGENT_STARTED:
                                # TASK-344: sub-agent spawned by Agent/Workflow tool.
                                meta = event.raw if isinstance(event.raw, dict) else {}
                                result_queue.put({
                                    "event": "subagent_started",
                                    "task_id": meta.get("task_id", ""),
                                    "description": meta.get("description", ""),
                                    "task_type": meta.get("task_type"),
                                    "tool_use_id": event.tool_use_id,
                                    "provider": event.provider,
                                    "trigger_message_id": event.trigger_message_id,
                                })

                            elif event.kind == AgentEventKind.SUBAGENT_PROGRESS:
                                meta = event.raw if isinstance(event.raw, dict) else {}
                                result_queue.put({
                                    "event": "subagent_progress",
                                    "task_id": meta.get("task_id", ""),
                                    "description": meta.get("description", ""),
                                    "last_tool_name": event.tool_name,
                                    "status": meta.get("status"),
                                    "usage": meta.get("usage"),
                                    "tool_use_id": event.tool_use_id,
                                    "provider": event.provider,
                                    "trigger_message_id": event.trigger_message_id,
                                })

                            elif event.kind == AgentEventKind.SUBAGENT_DONE:
                                meta = event.raw if isinstance(event.raw, dict) else {}
                                result_queue.put({
                                    "event": "subagent_done",
                                    "task_id": meta.get("task_id", ""),
                                    "status": meta.get("status", "completed"),
                                    "summary": event.text or "",
                                    "usage": meta.get("usage"),
                                    "tool_use_id": event.tool_use_id,
                                    "provider": event.provider,
                                    "trigger_message_id": event.trigger_message_id,
                                })

                            elif event.kind == AgentEventKind.FILE_CHANGED:
                                meta = event.raw if isinstance(event.raw, dict) else {}
                                file_data = meta.get("file")
                                if isinstance(file_data, dict):
                                    result_queue.put({
                                        "event": "file_changed",
                                        "file": file_data,
                                        "tool_name": event.tool_name,
                                        "arguments": event.tool_arguments or {},
                                        "tool_use_id": event.tool_use_id,
                                        "trigger_message_id": event.trigger_message_id,
                                    })

                            elif event.kind == AgentEventKind.ERROR:
                                # Terminal failures can still carry a provider-reported
                                # last-iteration usage snapshot (TASK-304). Preserve it
                                # before raising so the app can persist usage-so-far.
                                if event.token_usage:
                                    token_usage.clear()
                                    token_usage.update(event.token_usage)
                                    result_queue.put({
                                        "event": "token_usage",
                                        "token_usage": event.token_usage,
                                    })
                                # Do not abort the run subscription immediately.
                                # Drain through run_done so any already-completed
                                # tool/file events queued before the error persist.
                                pending_error = RuntimeError(
                                    f"{self.name} error: {event.text}"
                                )
                    except TimeoutError:
                        if self.name == "claude-code":
                            try:
                                receipt = await local_sub.send_rpc(
                                    "chat.cancel",
                                    {"sessionKey": session_key, "requestId": request_id},
                                    request_id=f"cancel_{request_id}",
                                    backend=self.name,
                                )
                                if not receipt.get("ok"):
                                    logger.error("Bridge cancellation rejected: %s", receipt)
                            except Exception:
                                logger.exception(
                                    "Could not confirm cancellation of timed-out request %s",
                                    request_id,
                                )
                        raise
                    finally:
                        await local_sub.close()
                    if pending_error is not None:
                        raise pending_error

                try:
                    asyncio.run(_worker())
                except Exception as e:
                    error_holder.append(e)
                finally:
                    result_queue.put(None)  # sentinel

            t = threading.Thread(target=_run_in_thread, daemon=True)
            t.start()

            # Drain the queue, yielding items to the caller
            while True:
                try:
                    item = result_queue.get(timeout=2.0)
                except queue_mod.Empty:
                    if not t.is_alive():
                        break
                    continue
                if item is None:
                    break
                yield item

            t.join(timeout=5.0)

            if error_holder:
                termination_reason = "error"
                raise error_holder[0]

        except Exception as exc:
            termination_reason = "error"
            raise exc
        finally:
            end_time = time.time()
            logger.info(
                "%s stream termination: reason=%s total_latency_ms=%.1f request_id=%s",
                self.name,
                termination_reason,
                (end_time - request_started) * 1000,
                request_id,
            )
            full_text = "".join(text_parts)
            result = AgentResult(
                text=full_text,
                model=upstream_model or self.name,
                provider=self.name,
                duration_ms=int((end_time - request_started) * 1000),
                tool_calls=tool_calls,
                usage=dict(token_usage),
                raw={"request_id": request_id, "termination_reason": termination_reason},
            )
            self._thread_local.last_stream_result = result

    def get_last_stream_result(self) -> AgentResult | None:
        return getattr(self._thread_local, "last_stream_result", None)

    # ----- public API -----------------------------------------------------

    async def chat(
        self,
        prompt: str,
        config: dict,
        stream: bool = False,
    ) -> str:
        """Return just the text response (backward compat)."""
        result = await self.chat_full(prompt, config)
        return result.text

    async def chat_full(
        self,
        prompt: str,
        config: dict,
    ) -> AgentResult:
        """Run the agent and return a structured result with tool call data."""
        # Collect all stream results
        text_parts: list[str] = []
        tool_calls: list[AgentToolCall] = []

        for item in self.stream_raw(prompt, config):
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if item.get("event") == "tool_call":
                    if item.get("result") is not None and tool_calls:
                        # TOOL_END: update last tool call with result
                        tool_calls[-1].result = item["result"]
                    else:
                        # TOOL_START: new tool call
                        tool_calls.append(AgentToolCall(
                            name=item.get("name", "unknown"),
                            arguments=item.get("arguments", {}),
                        ))

        result = self.get_last_stream_result()
        if result:
            return result

        return AgentResult(
            text="".join(text_parts),
            tool_calls=tool_calls,
            provider=self.name,
        )

    async def health_check(self, config: dict) -> bool:
        """Check if bridge subscriber is connected."""
        subscriber = get_agent_subscriber()
        return subscriber is not None and subscriber.connected
