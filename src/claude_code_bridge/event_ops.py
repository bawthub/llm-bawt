from __future__ import annotations

import logging
from datetime import datetime, timezone


from agent_bridge.events import AgentEvent, AgentEventKind, synthesize_event_id
from agent_bridge.tool_images import ToolImageMixin
from agent_bridge.user_scope import session_user


logger = logging.getLogger("claude_code_bridge.bridge")

# TASK-1689: only Bash tool cards get ~ display shortening.
SHELL_TOOL_NAMES = frozenset({"Bash"})


class ClaudeEventMixin(ToolImageMixin):
    """Claude output persistence + tool/event emission (TASK-555).

    Split out of ``ClaudeCodeBridge`` (TASK-555); composed back via
    inheritance so ``self.*`` state and sibling-mixin methods resolve
    on the assembled instance.
    """

    @staticmethod
    def _tool_response_content(tool_response: object) -> list:
        """Return MCP content blocks from a PostToolUse response payload."""
        if isinstance(tool_response, list):
            return tool_response
        if isinstance(tool_response, dict):
            content = tool_response.get("content")
            if isinstance(content, list):
                return content
        return []

    def _make_post_tool_use_hook(
        self,
        *,
        session_key: str,
        screenshot_artifacts_by_tool_use_id: dict[str, list[dict]],
        worker_liveness=None,
    ):
        """Persist Playwright images before the SDK resumes model reasoning.

        ``additionalContext`` is appended to the original MCP tool result, so the
        model keeps the inline image for vision and also receives durable Garage
        URLs it can curl later in the same turn.

        TASK-1148: a SendMessage whose resumed worker the bridge stopped gets a
        note telling the parent model the worker timed out and why.
        """

        async def post_tool_use(input_data, tool_use_id, context):
            del context
            tool_name = input_data.get("tool_name") or ""
            worker_stop = (
                worker_liveness.stop_for_tool(tool_use_id or input_data.get("tool_use_id"))
                if worker_liveness is not None else None
            )
            if worker_stop is not None:
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": worker_stop.parent_note(),
                    }
                }
            if tool_name.split("__")[-1] != "browser_take_screenshot":
                return {}
            tuid = (tool_use_id or input_data.get("tool_use_id") or "").strip()
            if not tuid:
                return {}
            cached = screenshot_artifacts_by_tool_use_id.get(tuid)
            if cached is None:
                content = self._tool_response_content(input_data.get("tool_response"))
                if not any(self._extract_image_block(block) for block in content):
                    return {}
                cached = await self._persist_screenshot_artifacts(
                    content, session_key, tuid,
                )
                screenshot_artifacts_by_tool_use_id[tuid] = cached
            additional_context = self._format_screenshot_artifacts(cached)
            if not additional_context:
                additional_context = (
                    "[Playwright Screenshot Artifact] The screenshot is visible "
                    "inline, but durable artifact persistence failed; no Garage URL "
                    "is available for this screenshot."
                )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": additional_context,
                }
            }

        return post_tool_use

    def _publish_event(
        self,
        request_id: str,
        session_key: str,
        seq: int,
        *,
        kind: AgentEventKind,
        text: str | None = None,
        tool_name: str | None = None,
        tool_arguments: dict | None = None,
        tool_result: str | None = None,
        tool_result_payload: dict | None = None,
        tool_error: bool | None = None,
        model: str | None = None,
        token_usage: dict | None = None,
        tool_use_id: str | None = None,
        parent_tool_use_id: str | None = None,
        attachments: list[dict] | None = None,
        extra_raw: dict | None = None,
    ) -> None:
        # TASK-1689: ~ shortening is a Bash tool-card display nicety only.
        # Prose and other tools' args/results must round-trip byte-identical.
        # Approval arguments are executable replay payloads, never shortened.
        if tool_name in SHELL_TOOL_NAMES:
            tool_result = self._shorten_paths(tool_result)
            if kind != AgentEventKind.APPROVAL_REQUIRED:
                tool_arguments = self._shorten_paths_in_dict(tool_arguments)
        event_id = synthesize_event_id(
            session_key, kind.value,
            {"text": text, "tool": tool_name, "seq": seq},
            seq,
        )
        event = AgentEvent(
            event_id=event_id,
            session_key=session_key,
            run_id=request_id,
            kind=kind,
            origin="system",
            text=text,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            tool_result=tool_result,
            tool_result_payload=tool_result_payload,
            tool_error=tool_error,
            model=model,
            seq=seq,
            timestamp=datetime.now(timezone.utc),
            # APPROVAL_REQUIRED stashes policy metadata (policy_id, severity,
            # subject, prompt, grant_key) in raw so the app can persist the
            # request row and the UI can render the Approve/Deny card.
            raw=dict(extra_raw) if extra_raw else {},
            token_usage=token_usage,
            provider=self._backend_name,
            # Inherit from the active run so every event for this request
            # (TOOL_START, TOOL_END, ASSISTANT_*, RUN_*) carries the same
            # originating user-message UUID without each call site needing
            # to remember to thread it through.
            trigger_message_id=self._trigger_message_ids.get(request_id),
            tool_use_id=tool_use_id,
            parent_tool_use_id=parent_tool_use_id,
            attachments=attachments,
        )
        self._publish_run_event_with_changed_file(request_id, event)

    def publish_proxy_egress(self, request_id: str, capture: dict) -> None:
        """Ship one provider-bound proxy body to the app's turn log (TASK-502).

        Serialized now, because adapters may keep mutating the dict after
        send; posted in the background so capture never delays the model call.
        """
        if not self._app_api_url or not self._proxy_request_sessions.get(request_id):
            return
        import asyncio
        import json

        seqs = self.__dict__.setdefault("_proxy_egress_seq", {})
        seqs[request_id] = seq = seqs.get(request_id, 0) + 1
        payload = {
            **{k: v for k, v in capture.items() if k != "body"},
            "request_id": request_id,
            "seq": seq,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "body_json": json.dumps(capture.get("body"), ensure_ascii=False, default=str),
        }
        task = asyncio.get_running_loop().create_task(self._post_proxy_egress(payload))
        pending = self.__dict__.setdefault("_proxy_egress_tasks", set())
        pending.add(task)
        task.add_done_callback(pending.discard)

    async def _post_proxy_egress(self, payload: dict) -> None:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self._app_api_url}/v1/turn-logs/egress", json=payload,
                )
            if resp.status_code >= 400:
                logger.warning(
                    "Proxy egress capture rejected request_id=%s: %s %s",
                    payload["request_id"], resp.status_code, resp.text[:200],
                )
        except Exception:
            logger.warning(
                "Proxy egress capture failed request_id=%s", payload["request_id"],
                exc_info=True,
            )

    def publish_proxy_status(self, request_id: str, status: dict) -> None:
        """Bridge an in-process proxy status onto the active Redis run."""
        session_key = self._proxy_request_sessions.get(request_id)
        if not session_key:
            logger.debug(
                "Dropping proxy status for inactive request_id=%s", request_id
            )
            return
        cause = status.get("terminal_tool_error")
        if isinstance(cause, str) and cause:
            if not hasattr(self, "_proxy_tool_errors"):
                self._proxy_tool_errors = {}
            self._proxy_tool_errors[request_id] = cause
        # TASK-1074: the request's phase tracker is the single owner of turn
        # phase; it dedups and stamps phase/since on the same event kind.
        tracker = getattr(self, "_turn_phase_trackers", {}).get(request_id)
        if tracker is not None:
            tracker.on_proxy_status(status)
            return
        if status.get("state") == "waiting":
            # TASK-1079: only a phase tracker can use "a new request began";
            # alone it is a no-op signal and would overwrite real status.
            return
        attempt = status.get("attempt")
        seq = int(attempt) if isinstance(attempt, int) else 0
        self._publish_event(
            request_id,
            session_key,
            seq,
            kind=AgentEventKind.UPSTREAM_STATUS,
            text=str(status.get("message") or "Upstream status changed."),
            extra_raw={"upstream_status": dict(status)},
        )

    def _publish_session_reset_unified(
        self,
        bot_id: str,
        session_key: str,
        *,
        had_session: bool = False,
        user_id: str | None = None,
    ) -> None:
        """Publish a SESSION_RESET event on the unified SSE stream.

        Deterministic signal for the frontend to clear its visible message
        buffer for ``bot_id``.  Carries the confirmation text so the UI can
        render it as a synthetic assistant message in one place instead of
        racing the HTTP run stream's ASSISTANT_DONE.  See TASK-249.
        """
        if not bot_id:
            return
        user_id = session_user(session_key, user_id)
        if not user_id:
            logger.warning("session_reset for %s has no user; not published", bot_id)
            return
        try:
            self._publisher.publish_unified_event(bot_id, user_id, {
                "_type": "session_reset",
                "bot_id": bot_id,
                "user_id": user_id,
                "session_key": session_key,
                "had_session": bool(had_session),
                "text": "Session reset. Ready for a new conversation.",
                "provider": self._backend_name,
                "ts": datetime.now(timezone.utc).timestamp(),
            })
        except Exception:
            logger.exception(
                "Failed to publish unified session_reset for bot=%s session=%s",
                bot_id, session_key,
            )

    @staticmethod
    def _shorten_paths(s: str | None) -> str | None:
        """Replace container home dir with ~ to save space in persisted logs."""
        if s is None:
            return None
        return s.replace("/home/bridge/", "~/").replace("/home/bridge", "~")

    @classmethod
    def _shorten_paths_in_dict(cls, d: dict | None) -> dict | None:
        if d is None:
            return None
        out = {}
        for k, v in d.items():
            if isinstance(v, str):
                out[k] = cls._shorten_paths(v)
            elif isinstance(v, dict):
                out[k] = cls._shorten_paths_in_dict(v)
            else:
                out[k] = v
        return out
