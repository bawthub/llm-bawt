from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterable
from datetime import datetime, timezone
from pathlib import Path

import httpx
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, StreamEvent
from claude_agent_sdk.types import (
    AssistantMessage,
    HookMatcher,
    MirrorErrorMessage,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SystemMessage,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from agent_bridge.approval import (
    ApprovalDecision,
    PolicyAction,
    PolicyBundle,
    evaluate as evaluate_policies,
)
from agent_bridge.events import AgentEvent, AgentEventKind, synthesize_event_id
from agent_bridge.publisher import COMMANDS_STREAM, RedisPublisher
from agent_bridge.session_queue import SessionQueue
from claude_code_bridge.tool_events import normalize_tool_result
from claude_code_bridge.tool_policy import effective_disallowed_tools

from ._bridge_helpers import (
    SESSION_PREFIX,
    MCP_TOOL_CONTEXT_KEY,
    _MCP_TOOL_CONTEXT_FALLBACK,
    _SEED_CLI_VERSION,
    _SEED_SANITIZE_RE,
    _XAI_RATES,
    _XAI_DEFAULT_RATES,
    _REFRESH_BUFFER_MS,
    _bot_slug_from_session_key,
    _fmt_tokens,
    _usage_input_total,
    _estimate_proxy_cost_usd,
    _pick_iteration_usage,
    _read_latest_compact_metadata,
    _token_expired_or_stale,
    _get_fresh_oauth_token,
    _is_cli_crash,
    _is_auth_failure,
)

logger = logging.getLogger("claude_code_bridge.bridge")


class ClaudeEventMixin:
    """Claude output persistence + tool/event emission (TASK-555).

    Split out of ``ClaudeCodeBridge`` (TASK-555); composed back via
    inheritance so ``self.*`` state and sibling-mixin methods resolve
    on the assembled instance.
    """

    @classmethod
    def _is_image_result_tool(cls, tool_name: str | None) -> bool:
        """True for tools whose result carries an image we should persist
        (namespaced ``mcp__x__name`` or bare)."""
        if not tool_name:
            return False
        return tool_name.split("__")[-1] in cls._IMAGE_RESULT_TOOL_TAILS

    @staticmethod
    def _extract_image_block(block: object) -> tuple[str, str] | None:
        """Pull (base64_data, mime) from a tool-result image content block.

        Handles both the MCP shape ``{type:image, data, mimeType}`` and the
        Anthropic API shape ``{type:image, source:{data, media_type}}``.
        """
        if not isinstance(block, dict) or block.get("type") != "image":
            return None
        data = block.get("data")
        mime = block.get("mimeType") or block.get("mime_type")
        if not data:
            src = block.get("source")
            if isinstance(src, dict):
                data = src.get("data")
                mime = mime or src.get("media_type")
        if not data:
            return None
        return data, (mime or "image/png")

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

    @staticmethod
    def _artifact_refs(artifacts: list[dict]) -> list[dict]:
        """Reduce canonical upload envelopes to persisted attachment refs."""
        refs: list[dict] = []
        seen: set[str] = set()
        for artifact in artifacts:
            asset_id = artifact.get("asset_id") if isinstance(artifact, dict) else None
            if isinstance(asset_id, str) and asset_id and asset_id not in seen:
                seen.add(asset_id)
                refs.append({"asset_id": asset_id, "kind": "image"})
        return refs

    def _absolute_artifact_url(self, path: object) -> str | None:
        if not isinstance(path, str) or not path:
            return None
        if path.startswith(("http://", "https://")):
            return path
        if not self._app_api_url:
            return path
        return f"{self._app_api_url.rstrip('/')}/{path.lstrip('/')}"

    def _format_screenshot_artifacts(self, artifacts: list[dict]) -> str:
        """Render canonical upload envelopes as model-visible curlable URLs."""
        lines = [
            "[Playwright Screenshot Artifact] The screenshot is stored durably. "
            "Use these URLs with curl if you need the raw artifact; paste the "
            "`public` link if the user should be able to open it:",
        ]
        rendered = 0
        for artifact in artifacts:
            if not isinstance(artifact, dict) or not artifact.get("asset_id"):
                continue
            rendered += 1
            meta = [
                f"asset_id={artifact['asset_id']}",
                f"type={artifact.get('mime_type') or 'image'}",
            ]
            if artifact.get("width") and artifact.get("height"):
                meta.append(f"{artifact['width']}x{artifact['height']}")
            lines.append(f"{rendered}. " + "  ".join(meta))
            urls = artifact.get("urls") if isinstance(artifact.get("urls"), dict) else {}
            for label in ("original", "preview", "thumb"):
                absolute = self._absolute_artifact_url(urls.get(label))
                if absolute:
                    lines.append(f"   {label}: {absolute}")
            # TASK-847: the user-facing BawtHub link. Only this one is
            # clickable by the user — the URLs above are internal.
            public_url = artifact.get("public_url")
            if isinstance(public_url, str) and public_url:
                lines.append(f"   public (share with user): {public_url}")
        return "\n".join(lines) if rendered else ""

    async def _persist_screenshot_artifacts(
        self, content: list, session_key: str, tool_use_id: str | None,
    ) -> list[dict]:
        """Upload image blocks and return canonical upload-response envelopes."""
        artifacts: list[dict] = []
        if not self._app_api_url:
            return artifacts
        user_id = session_key.split(":", 1)[1] if ":" in session_key else "nick"
        for block in content:
            img = self._extract_image_block(block)
            if img is None:
                if isinstance(block, dict) and block.get("type") == "image":
                    logger.warning(
                        "Screenshot image block in unrecognised shape (keys=%s)",
                        list(block.keys()),
                    )
                continue
            data_b64, mime = img
            artifact = await self._upload_data_url(
                f"data:{mime};base64,{data_b64}", user_id, tool_use_id,
            )
            if artifact:
                artifacts.append(artifact)
        return artifacts

    async def _persist_screenshot_blocks(
        self, content: list, session_key: str, tool_use_id: str | None,
    ) -> list[dict]:
        """Compatibility wrapper returning tiny refs for UI/history rails."""
        artifacts = await self._persist_screenshot_artifacts(
            content, session_key, tool_use_id,
        )
        return self._artifact_refs(artifacts)

    async def _upload_data_url(
        self, data_url: str, user_id: str, tool_use_id: str | None,
    ) -> dict | None:
        """POST image bytes and return the canonical upload-response envelope."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{self._app_api_url}/v1/uploads",
                    params={"source": "agent_attachment"},
                    headers={"X-Entity-Id": user_id},
                    json={
                        "data_url": data_url,
                        "filename": f"image-{tool_use_id or 'shot'}.png",
                    },
                )
                if resp.status_code >= 400:
                    logger.warning(
                        "Screenshot upload failed: %s %s",
                        resp.status_code, resp.text[:200],
                    )
                    return None
                payload = resp.json() or {}
                if not isinstance(payload, dict) or not payload.get("asset_id"):
                    logger.warning("Screenshot upload returned no asset_id")
                    return None
                return payload
        except Exception:
            logger.warning("Screenshot upload error", exc_info=True)
            return None

    def _make_post_tool_use_hook(
        self,
        *,
        session_key: str,
        screenshot_artifacts_by_tool_use_id: dict[str, list[dict]],
    ):
        """Persist Playwright images before the SDK resumes model reasoning.

        ``additionalContext`` is appended to the original MCP tool result, so the
        model keeps the inline image for vision and also receives durable Garage
        URLs it can curl later in the same turn.
        """

        async def post_tool_use(input_data, tool_use_id, context):
            del context
            tool_name = input_data.get("tool_name") or ""
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
        text = self._shorten_paths(text)
        tool_result = self._shorten_paths(tool_result)
        # Approval arguments are executable replay payloads, not display text.
        # Prettifying paths changes the exact invocation authorized by the grant.
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

    def publish_proxy_status(self, request_id: str, status: dict) -> None:
        """Bridge an in-process proxy status onto the active Redis run."""
        session_key = self._proxy_request_sessions.get(request_id)
        if not session_key:
            logger.debug(
                "Dropping proxy status for inactive request_id=%s", request_id
            )
            return
        # TASK-1074: the request's phase tracker is the single owner of turn
        # phase; it dedups and stamps phase/since on the same event kind.
        tracker = getattr(self, "_turn_phase_trackers", {}).get(request_id)
        if tracker is not None:
            tracker.on_proxy_status(status)
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
        user_id: str = "nick",
    ) -> None:
        """Publish a SESSION_RESET event on the unified SSE stream.

        Deterministic signal for the frontend to clear its visible message
        buffer for ``bot_id``.  Carries the confirmation text so the UI can
        render it as a synthetic assistant message in one place instead of
        racing the HTTP run stream's ASSISTANT_DONE.  See TASK-249.
        """
        if not bot_id:
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
