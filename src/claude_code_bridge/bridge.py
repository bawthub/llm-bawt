"""Claude Code bridge: Redis command listener + Agent SDK event translator."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

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

from agent_bridge.changed_file_lifecycle import ChangedFileLifecycleMixin
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
from .session_ops import ClaudeSessionMixin
from .approval_ops import ClaudeApprovalMixin
from .event_ops import ClaudeEventMixin
from .command_ops import ClaudeCommandMixin
from .send_handler import ClaudeSendMixin

logger = logging.getLogger(__name__)


class ClaudeCodeBridge(
    ChangedFileLifecycleMixin,
    ClaudeSessionMixin,
    ClaudeApprovalMixin,
    ClaudeEventMixin,
    ClaudeCommandMixin,
    ClaudeSendMixin,
):
    """Reads chat.send commands from Redis, runs them through the Claude Agent
    SDK, and publishes AgentEvent-formatted results back to Redis."""

    # Default timeout for a single query() call (seconds).
    # The CLI's internal API_TIMEOUT_MS is 600s; we cut shorter to fail fast.
    DEFAULT_REQUEST_TIMEOUT = 300

    # TASK-269: tool_result fed back when an AskUserQuestion is NOT answered
    # within the live window (TASK-1043). It's a deferral, not an answer, so it
    # goes out via PermissionResultDeny.message; real answers ALLOW the tool
    # with native ``answers`` instead. The model acknowledges and ends its
    # turn; the user's answer arrives later as a continuation turn.
    _DEFERRED_ACK = (
        "[The question has been delivered to the user, who will answer it in a "
        "separate later message. Do not wait and do not guess an answer. Briefly "
        "acknowledge that you've asked, then end your turn — you'll receive the "
        "user's answer as a new message and can act on it then.]"
    )

    def __init__(
        self,
        publisher: RedisPublisher,
        *,
        backend_name: str = "claude-code",
        app_api_url: str = "",
        cwd: str = "/app",
        permission_mode: str = "bypassPermissions",
        add_dirs: list[str] | None = None,
        request_timeout: float | None = None,
    ) -> None:
        self._publisher = publisher
        self._backend_name = backend_name
        self._app_api_url = app_api_url
        self._cwd = cwd
        self._permission_mode = permission_mode
        self._add_dirs = add_dirs or []
        self._request_timeout = request_timeout or self.DEFAULT_REQUEST_TIMEOUT
        self._command_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._redis = None  # set in _command_listener
        # Shared session queue — serializes sends per session and tracks
        # active tasks for abort support.
        self._session_queue = SessionQueue()
        # request_id → frontend user-message UUID, populated in _handle_send
        # and read by _publish_event so every emitted AgentEvent (tool_*,
        # assistant_*, etc.) carries the originating message id.  Cleared on
        # _handle_send finally.
        self._trigger_message_ids: dict[str, str] = {}
        # Proxy status callbacks run in-process and resolve request_id back to
        # the active Redis run without exposing the session key in HTTP headers.
        self._proxy_request_sessions: dict[str, str] = {}
        # Tool-use id -> Future for a live AskUserQuestion on this process.
        # Only the matching command may resolve it; cleaned up on timeout/abort.
        # Result: dict = native answers (allow), str = deny text (dismissal).
        self._pending_question_futures: dict[str, tuple[str, asyncio.Future[str | dict]]] = {}
        # Load MCP servers from settings file
        self._mcp_servers = self._load_mcp_servers()
        # TASK-270: Anthropic-compat proxy URL set by __main__ after the
        # ProxyServer binds. None when proxy is disabled. When set and the
        # request's model starts with a known provider prefix (see
        # ``proxy.adapters.REGISTRY``), the SDK subprocess is spawned with
        # ANTHROPIC_BASE_URL pointing here so its outbound /v1/messages
        # call lands on the proxy instead of api.anthropic.com.
        self._proxy_base_url: str | None = None
        self._proxy_cancel_request: Callable[[str], None] | None = None
        self._proxy_clear_request: Callable[[str], None] | None = None
        # ---- Approval-gated tool policies (TASK-291 / TASK-292) ----
        # Compiled bundle fetched from the app, TTL-cached. Grants are one-shot
        # allows keyed by grant_key, populated by approval.grant commands when a
        # user approves a gated call, consumed when the model re-issues it.
        self._policy_bundle: PolicyBundle | None = None
        self._policy_bundle_fetched_at: float = 0.0
        self._policy_bundle_ttl: float = float(
            os.getenv("CLAUDE_CODE_APPROVAL_BUNDLE_TTL", "15")
        )
        # When the app is unreachable, fail OPEN by default (allow tools) so a
        # transient app blip never halts every agent. Set 1/true to fail CLOSED
        # (deny gated tools when policies can't be fetched). Non-gated tools are
        # unaffected either way — only tools a (cached) policy gates are denied.
        self._approval_fail_closed: bool = os.getenv(
            "CLAUDE_CODE_APPROVAL_FAIL_CLOSED", ""
        ).strip().lower() in ("1", "true", "yes")
        self._approval_grants: dict[str, float] = {}  # grant_key -> expiry (monotonic)
        self._approval_reload_task: asyncio.Task | None = None
        # Tracks whether the last bundle fetch reached the app. Only consulted
        # when fail-closed is on: a failing fetch then denies all tools.
        self._policy_fetch_ok: bool = True

    def set_proxy_base_url(self, url: str) -> None:
        """Wire up the in-process Anthropic-compat proxy. Called by __main__
        after ``ProxyServer.start()`` binds its ephemeral port."""
        self._proxy_base_url = url
        logger.info("Bridge proxy base_url set: %s", url)

    def set_proxy_request_controls(
        self,
        cancel: Callable[[str], None],
        clear: Callable[[str], None],
    ) -> None:
        """Wire the request tombstone shared with the in-process proxy."""
        self._proxy_cancel_request = cancel
        self._proxy_clear_request = clear

    def _cancel_proxy_sampling(self, request_id: str) -> None:
        if self._proxy_cancel_request is not None:
            self._proxy_cancel_request(request_id)

    def _clear_proxy_sampling_cancel(self, request_id: str) -> None:
        if self._proxy_clear_request is not None:
            self._proxy_clear_request(request_id)

    @staticmethod
    def _model_provider_prefix(model: str) -> str | None:
        """Return the provider prefix if ``model`` is a proxy-routed name
        (``"<provider>/<upstream>"`` where provider is registered), else None."""
        if not model or "/" not in model:
            return None
        provider, _, upstream = model.partition("/")
        if not provider or not upstream:
            return None
        # Local import keeps the proxy subpackage off the bridge's import
        # graph until needed (it pulls in openai, fastapi, etc.).
        from .proxy.adapters import REGISTRY

        return provider if provider in REGISTRY else None

    def _load_mcp_servers(self) -> dict:
        """Load MCP server configs from ~/.claude/settings.json."""
        settings_path = Path.home() / ".claude" / "settings.json"
        try:
            if settings_path.exists():
                data = json.loads(settings_path.read_text())
                servers = data.get("mcpServers", {})
                if servers:
                    logger.info("Loaded MCP servers from settings: %s", list(servers.keys()))
                    return servers
        except Exception as e:
            logger.warning("Failed to load MCP servers from %s: %s", settings_path, e)
        return {}

    # ----- Periodic cache cleanup -----

    _CLEANUP_INTERVAL = 6 * 3600  # every 6 hours
    _CACHE_DIRS = ("shell-snapshots", "file-history", "debug", "paste-cache")
    _CACHE_MAX_AGE = 24 * 3600  # delete files older than 24h

    async def _periodic_cache_cleanup(self) -> None:
        """Periodically prune ~/.claude/ cache dirs to prevent unbounded growth."""
        import shutil

        claude_dir = Path.home() / ".claude"
        while True:
            try:
                await asyncio.sleep(self._CLEANUP_INTERVAL)
                now = time.time()
                total_removed = 0

                for dirname in self._CACHE_DIRS:
                    cache_path = claude_dir / dirname
                    if not cache_path.is_dir():
                        continue
                    for entry in cache_path.iterdir():
                        try:
                            age = now - entry.stat().st_mtime
                            if age > self._CACHE_MAX_AGE:
                                if entry.is_dir():
                                    shutil.rmtree(entry)
                                else:
                                    entry.unlink()
                                total_removed += 1
                        except OSError:
                            pass

                # NEVER prune ~/.claude/projects/**/*.jsonl here. Those are the
                # SDK transcripts that `resume=` replays — they ARE each
                # thread's memory (see reference/agent-system-prompt.md). A
                # 24h sweep here silently lobotomized every thread idle for a
                # day ("No conversation found with session ID" → cold-start
                # with no seed). The CLI's own `cleanupPeriodDays` (30d
                # default) is the only sanctioned transcript retention.

                if total_removed:
                    logger.info("Cache cleanup: removed %d stale entries", total_removed)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Cache cleanup error")

    # ----- Session persistence via bot profile API -----




    # ----- TASK-445: seed a fresh SDK session with chat summary history -----










    async def start(self) -> None:
        self._command_task = asyncio.create_task(self._command_listener())
        self._cleanup_task = asyncio.create_task(self._periodic_cache_cleanup())
        logger.info(
            "ClaudeCodeBridge started (backend=%s)",
            self._backend_name,
        )

    async def stop(self) -> None:
        for task in (self._command_task, self._cleanup_task, self._approval_reload_task):
            if task:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._publisher.close()
        logger.info("ClaudeCodeBridge stopped")

    async def run_forever(self) -> None:
        await self.start()
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    # ----- Redis command listener -----


    # ----- Handle a single chat.send command -----


    # ----- Event publishing -----


    # ----- Approval-gated tool policies (TASK-291 / TASK-292) -----

    # Synthetic tool result fed back to the model when a call is gated. Mirrors
    # _DEFERRED_ACK: the turn ends cleanly and the decision returns as a brand
    # new continuation turn (approve → re-issue; deny → drop it).
    _APPROVAL_PENDING_ACK = (
        "[This action requires user approval and has been sent to the user. Do "
        "NOT retry it now and do not work around it. Briefly acknowledge that "
        "you've requested approval, then end your turn — you'll receive the "
        "user's decision as a new message and can act on it then.]"
    )












    # ----- SDK can_use_tool callback -----





    # ----- Event publishing -----



    # ----- Screenshot persistence (Playwright -> media store) -----

    #: Tool tails whose image-bearing results we persist to the media store so
    #: the app can attach them to the reply (browsable per turn + inline card).
    #: - browser_take_screenshot: Playwright screenshots.
    #: - generate_image: Grok Imagine output (TASK-599). The tool already stored
    #:   the identical raw bytes, so this re-upload dedups to the same asset.
    _IMAGE_RESULT_TOOL_TAILS = frozenset({"browser_take_screenshot", "generate_image"})




