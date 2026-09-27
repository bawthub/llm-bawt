from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from claude_agent_sdk import ClaudeSDKClient, StreamEvent
from claude_agent_sdk.types import (
    AssistantMessage,
    MirrorErrorMessage,
    ResultMessage,
    SystemMessage,
    UserMessage,
)

from agent_bridge.events import AgentEventKind
from agent_bridge.publisher import COMMANDS_STREAM
from claude_code_bridge.tool_policy import effective_disallowed_tools

from ._bridge_helpers import (
    _is_cli_crash,
    _is_auth_failure,
)
from .active_run import STEER_INTERRUPTED_TOOL_RESULT, ClaudeActiveRun
from .send_boundaries import (
    publish_run_done_once,
    separator_before_new_block,
    should_read_native_context_usage,
)
from .send_errors import (
    AuthRetryPolicy,
    TerminalSDKResultError,
    classify_terminal_error,
    result_message_error,
)
from .send_preflight import prepare_send
from .send_result import ClaudeResultMixin
from .send_stream import ClaudeStreamMixin
from .send_usage import ClaudeUsageMixin, LiveUsagePublisher
from .turn_watchdog import TurnWatchdog, TurnWatchdogTimeout

logger = logging.getLogger("claude_code_bridge.bridge")


class ClaudeSendMixin(ClaudeStreamMixin, ClaudeUsageMixin, ClaudeResultMixin):
    """Claude agent send-path (TASK-555 quarantine; see TASK-622-style follow-up).

    Split out of ``ClaudeCodeBridge`` (TASK-555); composed back via
    inheritance so ``self.*`` state and sibling-mixin methods resolve
    on the assembled instance.
    """

    async def _handle_send(
        self, fields: dict, msg_id: str, async_redis,
    ) -> None:
        preflight = await prepare_send(self, fields, msg_id, async_redis)
        if preflight is None:
            return
        req = preflight.request
        message = preflight.message
        reset_requested = preflight.reset_requested
        request_id = req.request_id
        session_key = req.session_key
        bot_slug = req.bot_slug
        system_prompt = req.system_prompt
        model = req.model
        inject_messages = req.inject_messages
        bot_effort = req.bot_effort
        bot_max_turns = req.bot_max_turns
        subagent_model = req.subagent_model
        bot_context_window = req.bot_context_window
        responses_transport = req.responses_transport
        mcp_tool_timeout_ms = req.mcp_tool_timeout_ms
        configured_disallowed_tools = req.configured_disallowed_tools
        attachments = req.attachments
        thread_session_id = req.thread_session_id
        task_turn_capability = req.task_turn_capability

        queue_seq = 0

        def _publish_queue_heartbeat() -> None:
            nonlocal queue_seq
            queue_seq += 1
            self._publish_event(
                request_id,
                session_key,
                queue_seq,
                kind=AgentEventKind.SYSTEM_NOTE,
                extra_raw={"turn_health": {"phase": "queued"}},
            )

        if self._session_queue.is_busy(session_key):
            logger.info(
                "Session %s busy — queuing send request_id=%s",
                session_key, request_id,
            )

        async with self._session_queue.active(
            session_key,
            on_wait=_publish_queue_heartbeat,
            request_id=request_id,
        ):
            logger.info(
                "Handling send: request_id=%s session=%s model=%s system_prompt=%s msg=%.60s...",
                request_id, session_key, model,
                f"{len(system_prompt)} chars" if system_prompt else "none",
                message,
            )

            seq = 0
            text_parts: list[str] = []
            reasoning_tail = ""
            current_tool_name: str | None = None
            current_tool_input: str = ""
            actual_model: str = model  # updated from SystemMessage if available
            # Map tool_use_id -> tool name (the SDK's ToolResultBlock doesn't echo
            # the name, only the id) so we can recognise a Playwright screenshot
            # result and persist its image instead of letting the inline base64
            # ride in the model context forever.
            tool_names_by_id: dict[str, str] = {}
            tool_arguments_by_id: dict[str, dict] = {}
            # {asset_id, kind} refs for screenshots persisted to the media store
            # during this turn; stamped onto the terminal ASSISTANT_DONE event so
            # the app can attach them to the bot's reply message.
            turn_screenshot_assets: list[dict] = []
            # Canonical upload-response envelopes keyed by SDK tool_use_id. The
            # PostToolUse hook fills this before the model resumes so it receives
            # curlable Garage URLs; the later UserMessage observer reuses the same
            # upload for TOOL_END/history instead of uploading duplicate bytes.
            screenshot_artifacts_by_tool_use_id: dict[str, list[dict]] = {}
            direct_anthropic = not (
                self._proxy_base_url is not None
                and self._model_provider_prefix(model) is not None
            )
            if not direct_anthropic:
                self._proxy_request_sessions[request_id] = session_key
            interrupted_usage: dict | None = None
            run_done_published = False
            try:
                # Inject MCP tool context so Claude passes the right identifiers.
                # Body comes from the registry (TASK-490) with a byte-identical
                # local fallback; separator added here.
                if system_prompt and self._mcp_servers and bot_slug:
                    mcp_ctx = await self._get_mcp_tool_context(bot_slug)
                    system_prompt += f"\n\n{mcp_ctx}"

                # Session resolution: the app resolved the thread's stored
                # SDK session id (model-checked). None → cold-start.
                resume_id = None if reset_requested else req.thread_resume_id
                if resume_id:
                    logger.info(
                        "Resuming thread: thread=%s resume=%s",
                        thread_session_id or "(active)", resume_id,
                    )
                else:
                    logger.info(
                        "No resume key for thread=%s (cold-start)",
                        thread_session_id or "(active)",
                    )

                # Cold start with no session to resume — first-ever run,
                # post-model-switch, or post-/new. Seed from the app-injected
                # history so the new SDK session opens with continuity.
                if resume_id is None:
                    cold_seed = await self._seed_new_session(
                        bot_slug, model, injected=inject_messages,
                        thread_session_id=thread_session_id,
                    )
                    if cold_seed and cold_seed.get("seeded"):
                        resume_id = cold_seed["session_id"]
                        logger.info(
                            "Cold-start seeded session for %s: %s (%s summaries, %s msgs)",
                            bot_slug, resume_id,
                            cold_seed.get("summary_count"), cold_seed.get("message_count"),
                        )

                # Resolve settings file path
                settings_path = str(Path.home() / ".claude" / "settings.json")
                if not Path(settings_path).exists():
                    settings_path = None

                # Build prompt — multimodal if attachments present
                if attachments:
                    content: list[dict] = [{"type": "text", "text": message}]
                    for att in attachments:
                        mime = att.get("mimeType", "image/png")
                        data = att.get("content", "")
                        if data:
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime,
                                    "data": data,
                                },
                            })
                    logger.info("Multimodal prompt: %d text + %d images", 1, len(attachments))
                    user_content: str | list[dict] = content
                else:
                    user_content = message

                auth_retry = AuthRetryPolicy()
                fresh_session_retry = False
                # Pull (or create) the cooperative cancel event for this session so a
                # chat.abort that arrives mid-loop is observed at the next message
                # boundary — without waiting for `task.cancel()` to fire CancelledError
                # at the next `await` (which can be tens of seconds inside a tool call).
                cancel_event = (
                    self._session_queue.cancel_event(session_key) if session_key else None
                )
                if cancel_event is not None and cancel_event.is_set():
                    # Stale signal from a previous run — clear so this turn can proceed.
                    cancel_event.clear()
                while True:
                    interrupted_usage = None
                    # An async generator is single-use and the auth/session retry
                    # paths below re-enter this loop, so build a fresh prompt +
                    # completion gate per attempt.  turn_done releases the prompt
                    # generator (closing SDK input) only once the turn finishes —
                    # see _make_prompt_input for why it must stay open until then.
                    prompt_input, turn_done = self._make_prompt_input(user_content)
                    stderr_lines: list[str] = []
                    # Per-attempt: the auth/session retry paths re-run the turn
                    # from scratch, so reset screenshot tracking to avoid double-
                    # counting an earlier attempt's uploads on the final DONE event.
                    tool_names_by_id.clear()
                    tool_arguments_by_id.clear()
                    turn_screenshot_assets.clear()
                    screenshot_artifacts_by_tool_use_id.clear()

                    def _log_stderr(line: str) -> None:
                        line = line.rstrip()
                        stderr_lines.append(line)
                        logger.warning("CLI stderr: %s", line)

                    # TASK-270: route this turn to the in-process Anthropic-compat
                    # proxy when the model name carries a known provider prefix
                    # (e.g. "openai_chatgpt/gpt-5.4"). The proxy reads ChatGPT
                    # OAuth from ~/.codex/auth.json and forwards to OpenAI's
                    # Responses API. Otherwise fall through to Anthropic-direct.
                    use_proxy = (
                        self._proxy_base_url is not None
                        and self._model_provider_prefix(model) is not None
                    )

                    # TASK-593: the app resolves the DB-managed base list on each
                    # dispatch. Normalize again at this trust boundary and preserve
                    # the proxy transport rule: Anthropic server-side WebSearch /
                    # WebFetch cannot execute through non-Anthropic upstreams.
                    disallowed_tools = effective_disallowed_tools(
                        configured_disallowed_tools,
                        use_proxy=use_proxy,
                    )

                    # TASK-623: proxy-vs-direct SDK env construction extracted
                    # into _build_sdk_env (behavior-identical).
                    sdk_env = self._build_sdk_env(
                        use_proxy=use_proxy,
                        model=model,
                        subagent_model=subagent_model,
                        context_window=bot_context_window,
                        responses_transport=responses_transport,
                        force_refresh=auth_retry.attempted,
                        bot_id=bot_slug,
                        session_key=session_key,
                        thread_session_id=thread_session_id,
                        request_id=request_id,
                    )
                    if mcp_tool_timeout_ms:
                        # TASK-618: DB-backed app policy, delivered per turn so
                        # runtime-setting edits require no bridge recreation.
                        sdk_env["MCP_TOOL_TIMEOUT"] = str(mcp_tool_timeout_ms)

                    # Pass `seq` to the can_use_tool factory by reference so it
                    # can keep the AWAIT_TOOL_RESULT event ordered in the same
                    # sequence as the surrounding ASSISTANT_DELTA / TOOL_START
                    # events.  Tuple-wrapped in a single-element list so the
                    # closure can mutate it without rebinding.
                    seq_holder = [seq]
                    can_use_tool_cb = self._make_can_use_tool(
                        request_id=request_id,
                        session_key=session_key,
                        seq_holder=seq_holder,
                    )
                    # TASK-292: the approval gate lives in a PreToolUse hook, NOT
                    # can_use_tool. Under permission_mode="bypassPermissions" (our
                    # standing config) the SDK auto-approves regular tools and
                    # never calls can_use_tool for them, so a can_use_tool-based
                    # gate is dead code. PreToolUse hooks are a separate control
                    # plane that fires regardless of permission_mode (verified
                    # live: hook fires + "deny" blocks under bypass). Shares
                    # seq_holder with can_use_tool — turns are sequential, so no
                    # concurrent mutation.
                    pre_tool_use_cb = self._make_pre_tool_use_hook(
                        request_id=request_id,
                        session_key=session_key,
                        seq_holder=seq_holder,
                        task_turn_capability=task_turn_capability,
                    )
                    post_tool_use_cb = self._make_post_tool_use_hook(
                        session_key=session_key,
                        screenshot_artifacts_by_tool_use_id=(
                            screenshot_artifacts_by_tool_use_id
                        ),
                    )

                    # TASK-288 observability: log the system_prompt value AS SENT
                    # to the SDK, paired with resume state. This is the only place
                    # the resume-gate decision is visible — the earlier "Handling
                    # send" log prints the pre-gate request value and cannot prove
                    # whether persona actually reached the agent on a resumed turn.
                    logger.info(
                        "SDK call: resume=%s system_prompt_sent=%s",
                        bool(resume_id),
                        f"{len(system_prompt)} chars" if system_prompt else "none",
                    )

                    # TASK-623: ClaudeAgentOptions construction moved to
                    # _build_agent_options (behavior-identical).
                    from agent_bridge.skill_selection import guard_resumed_bundle
                    guard_resumed_bundle(req.skill_bundle, thread=req.thread_session_id, resume=resume_id)
                    options = self._build_agent_options(
                        model=model,
                        system_prompt=system_prompt,
                        disallowed_tools=disallowed_tools,
                        resume_id=resume_id,
                        sdk_env=sdk_env,
                        settings_path=settings_path,
                        bot_effort=bot_effort,
                        bot_max_turns=bot_max_turns,
                        can_use_tool_cb=can_use_tool_cb,
                        pre_tool_use_cb=pre_tool_use_cb,
                        post_tool_use_cb=post_tool_use_cb,
                        stderr=_log_stderr,
                        task_turn_capability=task_turn_capability,
                        skill_bundle=req.skill_bundle,
                    )

                    session_persisted = False
                    aborted = False
                    # Track the latest AssistantMessage.usage to surface "current
                    # context fullness" to the UI. ResultMessage.usage is
                    # cumulative across all internal API iterations in a turn,
                    # so cache_read_input_tokens can exceed contextWindow on
                    # multi-tool-use turns and produce >100% counters. The last
                    # AssistantMessage's usage reflects the actual final API
                    # call's view of the context.
                    latest_assistant_usage: dict | None = None
                    # One request-local publisher merges sparse message_delta
                    # frames and de-duplicates AssistantMessage echoes.
                    live_usage = LiveUsagePublisher(
                        self, request_id, session_key, actual_model, model,
                        bot_context_window,
                    )
                    # /compact lifecycle tracking for this turn. The SDK reports
                    # compaction via SystemMessage(subtype="status") — never a
                    # compact_boundary on the wire — so we watch the status
                    # payload to (a) give immediate feedback (a /compact can be
                    # ~50s of otherwise-silent work the UI reads as "hung") and
                    # (b) report the new resident size, since the /compact
                    # ResultMessage.usage is all-zeros.
                    compact_announced = False
                    compact_status: str | None = None  # None | "success" | "failed"
                    compact_error_msg: str | None = None
                    turn_session_id: str | None = resume_id
                    # Some providers / SDK paths terminate after an
                    # AssistantMessage snapshot with text/tool_use content and
                    # NEVER emit a trailing ResultMessage. If we only finalize
                    # on ResultMessage, the bridge logs "Send completed" but
                    # the app receives no ASSISTANT_DONE and the turn is saved
                    # as an empty timeout. Capture the latest assistant text
                    # snapshot so we can publish a fallback DONE on clean EOF.
                    assistant_snapshot_text: str = ""
                    assistant_done_emitted = False
                    # Track upstream API retries so we can (a) show live
                    # status in the UI ("z.ai overloaded, retrying…") and
                    # (b) include the error in the final DONE when all
                    # retries are exhausted and the turn ends empty.
                    api_retry_count = 0
                    api_last_error: str | None = None
                    api_retry_surfaced = False  # True once we've pushed a delta
                    # Retry safety is based on model/tool side effects, not status
                    # text emitted by the bridge itself (e.g. api_retry notices).
                    model_side_effects = False

                    def _publish_partial(text: str, *, attachments=None) -> None:
                        nonlocal seq, interrupted_usage
                        seq, interrupted_usage = self._publish_interrupted_done(
                            request_id=request_id, session_key=session_key, seq=seq,
                            text=text, actual_model=actual_model, model=model,
                            bot_context_window=bot_context_window,
                            latest_assistant_usage=latest_assistant_usage,
                            latest_stream_usage=live_usage.stream_usage,
                            attachments=attachments,
                        )

                    sdk_client = None
                    active_run = None
                    msg_stream = None
                    try:
                        sdk_client = ClaudeSDKClient(options=options)
                        await sdk_client.connect(prompt_input)
                        active_run = ClaudeActiveRun(
                            client=sdk_client,
                            request_id=request_id,
                        )
                        msg_stream = active_run.messages()

                        def _publish_turn_health(health: dict) -> None:
                            nonlocal seq
                            seq += 1
                            self._publish_event(
                                request_id,
                                session_key,
                                seq,
                                kind=AgentEventKind.SYSTEM_NOTE,
                                extra_raw={"turn_health": health},
                            )

                        watchdog = TurnWatchdog(
                            self._request_timeout,
                            on_health=_publish_turn_health,
                        )
                        # Register one live run controller for both control paths:
                        # chat.abort calls disconnect(), while chat.steer performs
                        # SDK interrupt -> replacement query on this same client.
                        # A single response consumer remains here in _handle_send.
                        if session_key:
                            self._session_queue.set_active_client(session_key, active_run)
                        while True:
                            # Cooperative abort check — runs before every SDK
                            # `__anext__`, so an abort signalled by chat.abort is
                            # observed even if the previous `await` was already
                            # past the cancel injection point.
                            if cancel_event is not None and cancel_event.is_set():
                                logger.info(
                                    "chat.abort signalled, halting SDK iteration: session=%s request=%s",
                                    session_key, request_id,
                                )
                                aborted = True
                                break
                            try:
                                msg = await watchdog.receive_next(
                                    msg_stream,
                                    request_id=request_id,
                                    session_key=session_key,
                                )
                            except StopAsyncIteration:
                                break

                            # Session-mirror write failure. MirrorErrorMessage is a
                            # SystemMessage subclass the SDK emits when its
                            # SessionStore.append fails — i.e. a turn's frame did not
                            # get persisted to the on-disk transcript. Left unhandled
                            # it slips into the generic SystemMessage branch below,
                            # matches none of its data conditions, and vanishes — so a
                            # persistence failure that can later wedge resume/replay
                            # goes completely unsignalled. Surface it as a structured
                            # warning (operational, not user-facing — no chat bubble).
                            if isinstance(msg, MirrorErrorMessage):
                                logger.warning(
                                    "SDK session-mirror append failed: key=%s error=%s session=%s",
                                    getattr(msg, "key", None),
                                    getattr(msg, "error", None),
                                    session_key,
                                )
                            if isinstance(msg, SystemMessage):
                                data = getattr(msg, "data", {}) or {}
                                # Capture session_id + actual model from the first
                                # SystemMessage (the init), then persist once.
                                if not session_persisted:
                                    if data.get("model"):
                                        actual_model = data["model"]
                                        live_usage.actual_model = actual_model
                                        logger.info("Actual model: %s", actual_model)
                                    if not resume_id:
                                        sid = data.get("session_id")
                                        if sid and thread_session_id:
                                            await self._set_thread_session(
                                                thread_session_id,
                                                bot_slug or session_key,
                                                sid, model,
                                            )
                                    session_persisted = True
                                # Track the session_id for this turn regardless of
                                # resume state — used to read the compaction result
                                # back from the transcript below.
                                if data.get("session_id"):
                                    turn_session_id = data["session_id"]
                                # Compaction lifecycle. A /compact turn emits
                                # SystemMessage(subtype="status"): first
                                # status="compacting", then a payload carrying
                                # compact_result ("success"/"failed") and, on
                                # failure, compact_error. There is NO
                                # compact_boundary on the wire. Surface the start
                                # immediately so the turn doesn't read as "hung",
                                # and record the outcome for the ResultMessage.
                                if data.get("status") == "compacting" and not compact_announced:
                                    compact_announced = True
                                    seq += 1
                                    note = "🗜️ Compacting conversation to free up context…"
                                    text_parts.append(note)
                                    self._publish_event(
                                        request_id, session_key, seq,
                                        kind=AgentEventKind.ASSISTANT_DELTA,
                                        text=note,
                                    )
                                cr = data.get("compact_result")
                                if cr == "success":
                                    compact_status = "success"
                                elif cr == "failed":
                                    compact_status = "failed"
                                    compact_error_msg = (
                                        data.get("compact_error") or "unknown error"
                                    )
                                # API retry lifecycle. The SDK CLI retries on
                                # upstream errors (429, 500, 529, etc.). Surface
                                # the retry to the UI as a live status delta so
                                # the user sees feedback instead of a dead bubble.
                                if data.get("subtype") == "api_retry":
                                    (
                                        seq,
                                        api_retry_count,
                                        api_last_error,
                                        api_retry_surfaced,
                                    ) = self._on_api_retry_status(
                                        data,
                                        request_id=request_id,
                                        session_key=session_key,
                                        seq=seq,
                                        text_parts=text_parts,
                                        already_surfaced=api_retry_surfaced,
                                    )
                                # TASK-623: sub-agent (Task*) lifecycle event
                                # emission extracted into _emit_subagent_task_events.
                                seq = self._emit_subagent_task_events(
                                    msg, request_id=request_id,
                                    session_key=session_key, seq=seq,
                                )
                            msg_type = type(msg).__name__
                            if not isinstance(msg, (StreamEvent, SystemMessage)):
                                content = getattr(msg, "content", [])
                                logger.debug(
                                    "SDK msg: %s blocks=%d content_types=%s",
                                    msg_type, len(content) if isinstance(content, list) else 0,
                                    [getattr(b, "type", type(b).__name__) for b in content] if isinstance(content, list) else "n/a",
                                )

                            if isinstance(msg, StreamEvent):
                                event = msg.event
                                event_type = event.get("type", "")
                                if event_type == "message_delta":
                                    seq = live_usage.publish(
                                        seq,
                                        assistant_usage=latest_assistant_usage,
                                        stream_usage=event.get("usage"),
                                    )

                                if event_type == "content_block_delta":
                                    delta = event.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        text = delta.get("text", "")
                                        if text:
                                            model_side_effects = True
                                            seq += 1
                                            text_parts.append(text)
                                            self._publish_event(
                                                request_id, session_key, seq,
                                                kind=AgentEventKind.ASSISTANT_DELTA,
                                                text=text,
                                            )
                                    elif delta.get("type") == "thinking_delta":
                                        # Model reasoning ("thinking"). Surface on
                                        # the REASONING_DELTA channel for the UI's
                                        # collapsible lane. Deliberately NOT
                                        # appended to text_parts — reasoning must
                                        # never enter the final assistant message
                                        # body (TASK-301).
                                        thinking = delta.get("thinking", "")
                                        if thinking:
                                            model_side_effects = True
                                            seq += 1
                                            reasoning_tail = thinking
                                            self._publish_event(
                                                request_id, session_key, seq,
                                                kind=AgentEventKind.REASONING_DELTA,
                                                text=thinking,
                                            )
                                    elif delta.get("type") == "signature_delta":
                                        # Opaque reasoning signature — no display
                                        # value; drop it.
                                        pass
                                    elif delta.get("type") == "input_json_delta":
                                        current_tool_input += delta.get("partial_json", "")

                                elif event_type == "content_block_start":
                                    block = event.get("content_block", {})
                                    if block.get("type") == "thinking":
                                        separator = separator_before_new_block(reasoning_tail)
                                        if separator:
                                            seq += 1
                                            reasoning_tail += separator
                                            self._publish_event(
                                                request_id, session_key, seq,
                                                kind=AgentEventKind.REASONING_DELTA,
                                                text=separator,
                                            )
                                    elif block.get("type") == "tool_use":
                                        model_side_effects = True
                                        current_tool_name = block.get("name", "unknown")
                                        current_tool_input = ""
                                    elif block.get("type") == "text":
                                        # A NEW text block. Narrating agents emit one
                                        # text block per iteration ("Looking up X." →
                                        # tool → "Grabbing coords." → tool → answer).
                                        # `text_parts` is joined with "", so without a
                                        # boundary those blocks glue into
                                        # "…Palworld.Grabbing exact coords…" — one run-on
                                        # paragraph in the persisted message body.
                                        # Emit the separator as a real ASSISTANT_DELTA so
                                        # the live stream, the persisted content, and the
                                        # tool `textOffset` char counts all stay in sync.
                                        tail = text_parts[-1] if text_parts else ""
                                        if tail and not tail.endswith("\n"):
                                            seq += 1
                                            text_parts.append("\n\n")
                                            self._publish_event(
                                                request_id, session_key, seq,
                                                kind=AgentEventKind.ASSISTANT_DELTA,
                                                text="\n\n",
                                            )

                                elif event_type == "content_block_stop":
                                    if current_tool_name:
                                        current_tool_name = None
                                        current_tool_input = ""

                            elif isinstance(msg, AssistantMessage):
                                # TASK-623: AssistantMessage tool_use / snapshot
                                # handling extracted into _on_assistant_message.
                                prior_seq = seq
                                prior_snapshot = assistant_snapshot_text
                                seq, latest_assistant_usage, assistant_snapshot_text = (
                                    self._on_assistant_message(
                                        msg,
                                        request_id=request_id,
                                        session_key=session_key,
                                        seq=seq,
                                        tool_names_by_id=tool_names_by_id,
                                        tool_arguments_by_id=tool_arguments_by_id,
                                        latest_assistant_usage=latest_assistant_usage,
                                        assistant_snapshot_text=assistant_snapshot_text,
                                    )
                                )
                                if seq != prior_seq or assistant_snapshot_text != prior_snapshot:
                                    model_side_effects = True
                                seq = live_usage.publish(
                                    seq,
                                    assistant_usage=latest_assistant_usage,
                                )

                            elif isinstance(msg, UserMessage):
                                # TASK-623: UserMessage tool_result (TOOL_END)
                                # handling extracted into _on_user_message_tool_results.
                                seq = await self._on_user_message_tool_results(
                                    msg,
                                    request_id=request_id,
                                    session_key=session_key,
                                    seq=seq,
                                    tool_names_by_id=tool_names_by_id,
                                    tool_arguments_by_id=tool_arguments_by_id,
                                    turn_screenshot_assets=turn_screenshot_assets,
                                    screenshot_artifacts_by_tool_use_id=(
                                        screenshot_artifacts_by_tool_use_id
                                    ),
                                )

                            elif isinstance(msg, ResultMessage):
                                if (
                                    active_run is not None
                                    and active_run.consume_replaced_result(msg)
                                ):
                                    # Tools still open at the interrupt boundary
                                    # will never receive SDK tool_result blocks.
                                    # Close their UI cards honestly before the
                                    # replacement query starts emitting events.
                                    for tool_use_id, tool_name in list(
                                        tool_names_by_id.items()
                                    ):
                                        seq += 1
                                        self._publish_event(
                                            request_id,
                                            session_key,
                                            seq,
                                            kind=AgentEventKind.TOOL_END,
                                            tool_name=tool_name,
                                            tool_arguments=tool_arguments_by_id.get(
                                                tool_use_id, {}
                                            ),
                                            tool_use_id=tool_use_id,
                                            tool_result=STEER_INTERRUPTED_TOOL_RESULT,
                                            tool_error=True,
                                        )
                                    tool_names_by_id.clear()
                                    tool_arguments_by_id.clear()
                                    logger.info(
                                        "Drained steer interrupt boundary: request_id=%s "
                                        "session=%s",
                                        request_id,
                                        session_key,
                                    )
                                    continue

                                terminal_error = result_message_error(
                                    msg,
                                    fallback=api_last_error,
                                )
                                if terminal_error is not None:
                                    raise terminal_error

                                native_context_usage = None
                                if should_read_native_context_usage(use_proxy=use_proxy):
                                    native_context_usage = await self._read_native_context_usage(
                                        sdk_client
                                    )
                                seq = await self._finalize_result_message(
                                    msg,
                                    request_id=request_id,
                                    session_key=session_key,
                                    seq=seq,
                                    text_parts=text_parts,
                                    assistant_snapshot_text=assistant_snapshot_text,
                                    api_retry_count=api_retry_count,
                                    api_last_error=api_last_error,
                                    api_retry_surfaced=api_retry_surfaced,
                                    actual_model=actual_model,
                                    model=model,
                                    bot_context_window=bot_context_window,
                                    latest_assistant_usage=latest_assistant_usage,
                                    latest_stream_usage=live_usage.stream_usage,
                                    native_context_usage=native_context_usage,
                                    compact_status=compact_status,
                                    compact_error_msg=compact_error_msg,
                                    turn_session_id=turn_session_id,
                                    turn_screenshot_assets=turn_screenshot_assets,
                                )
                                assistant_done_emitted = True
                                # ASSISTANT_DONE is now terminal for the app-side
                                # subscription. Publish the run sentinel immediately
                                # so persistence/UI finalization does not wait for the
                                # SDK subprocess's graceful-shutdown window. The
                                # session lock remains held through disconnect below,
                                # so a successor turn still cannot overlap cleanup.
                                run_done_published = publish_run_done_once(
                                    self._publisher,
                                    request_id,
                                    already_published=run_done_published,
                                )
                                # Turn complete — release the prompt generator so
                                # the SDK closes its input stream.  Kept open until
                                # now so the can_use_tool control channel survived
                                # any AskUserQuestion pause earlier in the turn.
                                turn_done.set()
                                # ResultMessage is terminal for this send (one user
                                # message -> one assistant turn), so stop iterating
                                # NOW instead of looping back to await a trailing
                                # StopAsyncIteration. After a deferred
                                # AskUserQuestion the streaming-input session stays
                                # alive — heartbeat/stream events keep re-arming the
                                # per-message timeout — so that await can block
                                # indefinitely while STILL holding the per-session
                                # lock, deadlocking the next continuation turn on the
                                # same session (TASK-269). The `finally` below closes
                                # the stream and kills the subprocess cleanly.
                                break
                        if aborted:
                            # Cooperative abort fired — publish the partial text +
                            # provider-reported usage accumulated before cancellation.
                            _publish_partial("".join(text_parts))
                            assistant_done_emitted = True
                        elif not assistant_done_emitted:
                            # Clean EOF without a ResultMessage. z.ai / GLM via
                            # the Claude SDK can end a turn after an
                            # AssistantMessage snapshot only (text and/or tool
                            # uses already captured above). Publish a fallback
                            # terminal DONE so the app persists the reply instead
                            # of timing out with response_chars=0.
                            full_text = "".join(text_parts)
                            if assistant_snapshot_text:
                                if not full_text:
                                    full_text = assistant_snapshot_text
                                elif assistant_snapshot_text.startswith(full_text):
                                    full_text = assistant_snapshot_text
                            # Surface retry errors when the turn ends empty
                            if not full_text and api_retry_count > 0:
                                error_note = (
                                    f"\n\n❌ Upstream error after {api_retry_count} "
                                    f"retries: {api_last_error or 'unknown'}. "
                                    f"Try again in a moment."
                                )
                                if api_retry_surfaced:
                                    full_text = "".join(text_parts) + error_note
                                else:
                                    full_text = error_note.lstrip()
                            # Always publish — even if full_text is empty.
                            # An empty ASSISTANT_DONE is far better than no
                            # DONE at all, which causes timeout + vanishing bubble.
                            _publish_partial(
                                full_text, attachments=turn_screenshot_assets or None,
                            )
                            assistant_done_emitted = True
                            logger.info(
                                "EOF fallback ASSISTANT_DONE: chars=%d request_id=%s session=%s",
                                len(full_text), request_id, session_key,
                            )
                        if assistant_done_emitted:
                            run_done_published = publish_run_done_once(
                                self._publisher,
                                request_id,
                                already_published=run_done_published,
                            )
                        break
                    except asyncio.CancelledError:
                        # task.cancel() arrived from elsewhere (legacy path /
                        # belt-and-suspenders fallback). Make sure the run is
                        # finalized before we re-raise so the frontend doesn't
                        # see a stuck `streaming` turn.
                        logger.info(
                            "Send cancelled via task.cancel: request_id=%s session=%s",
                            request_id, session_key,
                        )
                        try:
                            _publish_partial("".join(text_parts))
                            assistant_done_emitted = True
                        except Exception:
                            logger.debug("Failed to publish ASSISTANT_DONE on cancel", exc_info=True)
                        try:
                            run_done_published = publish_run_done_once(
                                self._publisher,
                                request_id,
                                already_published=run_done_published,
                            )
                        except Exception:
                            logger.debug("Failed to publish run_done on cancel", exc_info=True)
                        raise
                    except Exception as e:
                        if cancel_event is not None and cancel_event.is_set():
                            logger.info(
                                "Abort teardown surfaced %r; treating as abort: session=%s",
                                e, session_key,
                            )
                            aborted = True
                            _publish_partial("".join(text_parts))
                            assistant_done_emitted = True
                            run_done_published = publish_run_done_once(
                                self._publisher,
                                request_id,
                                already_published=run_done_published,
                            )
                            break

                        interrupted_usage = self._compute_interrupted_usage(
                            actual_model=actual_model,
                            model=model,
                            bot_context_window=bot_context_window,
                            latest_assistant_usage=latest_assistant_usage,
                            latest_stream_usage=live_usage.stream_usage,
                        )

                        # 1) Legacy direct path only: the OAuth gateway already
                        #    owns request-level recovery; never replay its turn.
                        auth_failure = (
                            e.credential_error
                            if isinstance(e, TerminalSDKResultError)
                            else _is_auth_failure(e, stderr_lines)
                        )
                        if auth_retry.claim(
                            is_auth_failure=auth_failure,
                            direct_anthropic=not use_proxy and not self._uses_claude_oauth_gateway(),
                            model_side_effects=model_side_effects,
                        ):
                            logger.warning(
                                "Auth failure for %s; force-fetching broker token and retrying once",
                                request_id,
                            )
                            # The first attempt may have emitted only a bridge-owned
                            # retry notice. It is not assistant content and must not
                            # survive into the successful retry's final message.
                            text_parts.clear()
                            continue

                        # 2) CLI crash or timeout before any text streamed →
                        #    clear stale session and retry once fresh
                        retryable_timeout = (
                            isinstance(e, TimeoutError)
                            and not isinstance(e, TurnWatchdogTimeout)
                        ) or (
                            isinstance(e, TurnWatchdogTimeout)
                            and e.retry_fresh_session
                        )
                        if (
                            not fresh_session_retry
                            and not text_parts
                            and (_is_cli_crash(e) or retryable_timeout)
                        ):
                            fresh_session_retry = True
                            reason = "timeout" if isinstance(e, TimeoutError) else "CLI crash (exit code 1)"
                            logger.warning(
                                "%s for %s; clearing session and retrying fresh",
                                reason, request_id,
                            )
                            if stderr_lines:
                                logger.warning("Captured stderr before retry: %s", stderr_lines)
                            # Retry without the wedged SDK session. The next init
                            # persists its replacement to the same durable thread.
                            resume_id = None
                            continue

                        raise
                    finally:
                        # Guarantee the prompt generator is released on EVERY exit
                        # path (StopAsyncIteration, the ResultMessage break above,
                        # abort, or exception). If it stays parked on
                        # `await done_event.wait()` the SDK input stream never closes
                        # and this session's lock can never be reacquired — the
                        # TASK-269 deadlock. Event.set() is idempotent.
                        turn_done.set()
                        # Wake any concurrent steer that is waiting for the SDK
                        # CLI to acknowledge interrupt. A turn can finish in the
                        # narrow window after the bridge accepted the steer
                        # command but before the CLI handles its control request.
                        if active_run is not None:
                            active_run.mark_completed()
                        # Always deregister this iteration's client so a
                        # subsequent chat.abort doesn't try to disconnect()
                        # something that's already finished. We pop only if
                        # the registry still points at our client — concurrent
                        # aborts may have already popped it to disconnect().
                        if session_key and sdk_client is not None:
                            current = self._session_queue.get_active_client(session_key)
                            if current is active_run:
                                self._session_queue.pop_active_client(session_key)
                            try:
                                await asyncio.wait_for(sdk_client.disconnect(), timeout=20.0)
                            except (asyncio.TimeoutError, asyncio.CancelledError):
                                pass
                            except Exception:
                                logger.debug(
                                    "sdk_client.disconnect() raised", exc_info=True,
                                )

                run_done_published = publish_run_done_once(
                    self._publisher,
                    request_id,
                    already_published=run_done_published,
                )
                if aborted:
                    logger.info(
                        "Send aborted via chat.abort: request_id=%s session=%s",
                        request_id, session_key,
                    )
                else:
                    logger.info("Send completed: request_id=%s session=%s", request_id, session_key)

            except asyncio.CancelledError:
                # Already handled inside the inner try (we published run_done
                # before re-raising). Suppress here so the asyncio task ends
                # cleanly without the "Task was destroyed but it is pending"
                # noise.
                logger.debug("Send cancellation propagated past inner handler")
                raise
            except Exception as e:
                logger.exception("Send failed: request_id=%s", request_id)
                error_text, error_raw = classify_terminal_error(
                    e,
                    direct_anthropic=direct_anthropic,
                    proxy_provider=(
                        None if direct_anthropic
                        else self._model_provider_prefix(model)
                    ),
                )
                seq += 1
                self._publish_event(
                    request_id, session_key, seq,
                    kind=AgentEventKind.ERROR,
                    text=error_text,
                    token_usage=interrupted_usage,
                    extra_raw=error_raw,
                )
                self._publisher.publish_run_done(request_id)
            finally:
                self._discard_changed_file_request(request_id)
                self._proxy_request_sessions.pop(request_id, None)
                # Drop the per-run trigger_message_id mapping so we don't leak.
                self._trigger_message_ids.pop(request_id, None)
                await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
