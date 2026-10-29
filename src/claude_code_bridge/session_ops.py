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

from agent_bridge.continuity import (
    SeedFraming,
    format_seed_ack,
    post_continuity_receipt,
    render_seed_briefing,
    seed_receipt,
    stats_from_messages,
)
from agent_bridge.approval import (
    ApprovalDecision,
    PolicyAction,
    PolicyBundle,
    evaluate as evaluate_policies,
)
from agent_bridge.events import AgentEvent, AgentEventKind, synthesize_event_id
from agent_bridge.maintenance import is_new_command
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


class ClaudeSessionMixin:
    """Claude session persistence + /new seed briefing (TASK-555).

    Split out of ``ClaudeCodeBridge`` (TASK-555); composed back via
    inheritance so ``self.*`` state and sibling-mixin methods resolve
    on the assembled instance.
    """

    @staticmethod
    def _seed_usage_estimate(
        stats: dict | None,
        *,
        context_window: int | None,
    ) -> dict:
        """Represent reset context as a seed estimate or explicit clean zero."""
        seeded = bool(stats and stats.get("seeded"))
        resident_tokens = max(0, int((stats or {}).get("approx_tokens") or 0))
        return {
            "input_tokens": resident_tokens,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "output_tokens": 0,
            "resident_tokens": resident_tokens,
            "resident_source": (
                "session_seed_char_estimate" if seeded else "clean_session_reset"
            ),
            "context_window": context_window,
            # Seeding writes a synthetic transcript; bare /new makes no provider
            # request, so its exact incremental cost is zero even though the
            # future resident-context size is estimated.
            "total_cost_usd": 0.0,
            "usage_status": "estimated" if seeded else "reset",
        }

    async def _preprocess_new_command(
        self,
        message: str,
        *,
        explicit_thread: bool,
        reset_action: str | None = None,
        bot_slug: str,
        session_key: str,
        request_id: str,
        model: str,
        context_window: int | None,
        inject_messages: list | None,
        thread_session_id: str | None,
        msg_id: str,
        async_redis,
        scope=None,
        inject_seed_framing=None,
        reset_skills: dict | None = None,
    ) -> str | None:
        """Handle bridge-owned ``/new`` setup and return the remaining prompt.

        ``None`` means a bare ``/new`` was fully acknowledged and the caller
        must stop. ``/new <message>`` returns only the trailing message; its
        normal cold-start path creates exactly one SDK session. Explicit-thread
        turns pass through unchanged because opening an old thread must never
        rotate it.
        """
        if explicit_thread or (not reset_action and not is_new_command(message)):
            return message

        self._publish_session_reset_unified(
            bot_slug or session_key, session_key, had_session=True,
        )
        remaining = (
            message if reset_action == "reset_follow_up"
            else "" if reset_action == "reset_only"
            else message.lstrip().removeprefix("/new").strip()
        )
        if remaining:
            return remaining

        # Bare /new has no normal send path to create the fresh SDK session,
        # so seed it here before acknowledging.
        seed_stats = await self._seed_new_session(
            bot_slug, model, injected=inject_messages,
            thread_session_id=thread_session_id, framing=inject_seed_framing,
        )
        if isinstance(seed_stats, dict) and seed_stats.get("reason") and not seed_stats.get("clean_start"):
            raise RuntimeError(f"new-session seed failed: {seed_stats['reason']}")
        self._publish_event(
            request_id, session_key, 1,
            kind=AgentEventKind.ASSISTANT_DONE,
            text=self._format_seed_ack(seed_stats, reset_skills),
            model=model,
            token_usage=self._seed_usage_estimate(
                seed_stats, context_window=context_window,
            ),
            extra_raw={"new_session_ack": {
                "thread_session_id": thread_session_id,
                "provider_session_id": (seed_stats or {}).get("session_id"),
                "clean_start": bool((seed_stats or {}).get("clean_start")),
                # TASK-1579: binds the app receipt to the seed actually applied.
                **seed_receipt(inject_messages),
            }},
        )
        if scope is not None:
            # TASK-1232 P2: a Stop landing on the xack must not add a second terminal.
            scope.terminal_published = True
        await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
        return None

    async def _set_thread_session(
        self, thread_session_id: str, bot_id: str, sdk_session_id: str, model: str,
    ) -> None:
        """Persist an SDK session id onto its bawthub thread.

        Writes to ``sessions.session_metadata.agent_session_keys`` via the
        app's ``PUT /v1/sessions/{id}/agent-session-key`` endpoint. This is
        the ONLY session-key write path (TASK-638); the scalar
        ``agent_backend_config.session_key`` is retired.
        """
        if not self._app_api_url:
            logger.warning(
                "No API URL — thread session not persisted for %s/%s",
                bot_id, thread_session_id,
            )
            return
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.put(
                    f"{self._app_api_url}/v1/sessions/{thread_session_id}/agent-session-key",
                    params={"bot_id": bot_id},
                    json={
                        "backend": "claude-code",
                        "session_key": sdk_session_id,
                        "model": model,
                    },
                )
                resp.raise_for_status()
            logger.info(
                "Thread session persisted: %s thread=%s -> %s",
                bot_id, thread_session_id, sdk_session_id,
            )
        except Exception as e:
            logger.warning(
                "Failed to persist thread session for %s/%s: %s",
                bot_id, thread_session_id, e,
            )

    async def _get_mcp_tool_context(self, bot_slug: str) -> str:
        """Return the MCP tool context block for a bot (TASK-490).

        Fetches the (bot-overridable) template body from the app registry via
        GET /v1/prompts/{key}, cached per bot for the process, and falls back to
        a byte-identical local copy if the app is unreachable — so behavior is
        preserved regardless. Returns the block WITHOUT the leading separator;
        the caller prepends ``\\n\\n``.
        """
        cache = getattr(self, "_mcp_ctx_cache", None)
        if cache is None:
            cache = {}
            self._mcp_ctx_cache = cache
        if bot_slug in cache:
            return cache[bot_slug]

        body = _MCP_TOOL_CONTEXT_FALLBACK
        if self._app_api_url:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(
                        f"{self._app_api_url}/v1/prompts/{MCP_TOOL_CONTEXT_KEY}",
                        params={"scope_type": "bot", "scope_id": bot_slug},
                    )
                    resp.raise_for_status()
                    fetched = (resp.json() or {}).get("body")
                    if fetched:
                        body = fetched
            except Exception as e:
                logger.warning(
                    "MCP tool context fetch failed for %s (%s); using local fallback",
                    bot_slug, e,
                )
        # Bridge-side substitution (robust against stray braces in overrides).
        rendered = body.replace("{bot_slug}", bot_slug)
        cache[bot_slug] = rendered
        return rendered

    def _project_slug(self, cwd: str) -> str:
        """Reproduce the SDK's project-dir sanitization (non-alnum -> '-')."""
        return _SEED_SANITIZE_RE.sub("-", cwd or "")

    def _render_seed_briefing(self, messages: list[dict], briefing: str) -> str:
        """The shared text-only briefing (``agent_bridge.continuity``)."""
        return render_seed_briefing(messages, briefing)

    def _write_seed_transcript(self, session_id: str, messages: list[dict],
                               framing: SeedFraming) -> Path:
        """Write a synthetic two-entry Claude Code transcript (user briefing +
        assistant ack) so the SDK can ``resume`` it into a fresh session.

        Entry shape verified against real on-disk transcripts. The assistant
        entry uses ``model: "<synthetic>"`` — Claude Code's own marker for a
        fabricated turn (the SDK writes these for injected errors and resumes
        past them cleanly). Ends on the assistant so the leaf is clean."""
        slug = self._project_slug(self._cwd)
        proj_dir = Path.home() / ".claude" / "projects" / slug
        proj_dir.mkdir(parents=True, exist_ok=True)
        path = proj_dir / f"{session_id}.jsonl"

        briefing = self._render_seed_briefing(messages, framing.briefing)
        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        user_uuid = str(uuid.uuid4())
        asst_uuid = str(uuid.uuid4())

        common = {
            "isSidechain": False,
            "sessionId": session_id,
            "cwd": self._cwd,
            "version": _SEED_CLI_VERSION,
            "gitBranch": "",
            "userType": "external",
        }
        user_entry = {
            "parentUuid": None,
            "type": "user",
            "message": {"role": "user", "content": briefing},
            "uuid": user_uuid,
            "timestamp": ts,
            **common,
        }
        asst_entry = {
            "parentUuid": user_uuid,
            "type": "assistant",
            "uuid": asst_uuid,
            "timestamp": ts,
            "message": {
                "role": "assistant",
                "model": "<synthetic>",
                "type": "message",
                "stop_reason": "stop_sequence",
                "content": [{"type": "text", "text": framing.ack}],
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
            **common,
        }
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(user_entry) + "\n")
            f.write(json.dumps(asst_entry) + "\n")
        logger.info(
            "Seed transcript written: %s (%d context msgs)",
            path, len(messages),
        )
        return path

    @staticmethod
    def _stats_from_messages(messages: list[dict]) -> dict:
        """Shared /new-ack stats (``agent_bridge.continuity``)."""
        return stats_from_messages(messages)

    async def _record_continuity_receipt(self, thread_id, bot_id, messages, *, restored):
        """Confirm the app's reference manifest only after the transcript write."""
        await post_continuity_receipt(self._app_api_url, thread_id, bot_id, messages,
                                      restored=restored, logger=logger)

    async def _seed_new_session(
        self,
        bot_id: str,
        model: str,
        injected: list | None = None,
        thread_session_id: str | None = None,
        framing: SeedFraming | None = None,
    ) -> dict | None:
        """Seed a brand-new SDK session for ``bot_id`` from app-injected history.
        Writes the synthetic transcript, persists the minted session id, and
        returns a stats dict for the /new ack.

        ``injected`` (TASK-501): messages the app pre-assembled and pushed in the
        dispatch. TASK-508 (policy-vs-mechanics) + TASK-615/501 Phase 2: llm-bawt
        (``maybe_build_session_seed``) is the SOLE seed authority — it already
        evaluated continuity + ``/new`` + session state and only emits messages
        when a seed is warranted. So the injected path is ungated, and its
        ABSENCE means no seed: the bridge holds NO independent seed policy and no
        self-fetch fallback (that dual-authority path is gone).

        Returns:
            None      -> no seed (app pushed nothing)
            {seeded: False, reason} -> a seed was warranted but nothing to seed / error
            {seeded: True, session_id, summary_count, message_count,
             approx_tokens, oldest_timestamp, newest_timestamp}
        """
        if injected is None:
            return None
        # llm-bawt authorized this seed by sending it. An explicit empty list is
        # a clean-session decision: keep resume=None so the real first query mints
        # the provider session, then persist that actual SDK id through write-back.
        messages = injected
        stats = self._stats_from_messages(messages)
        if not messages:
            return {**stats, "seeded": False, "clean_start": True}
        if framing is None:
            # TASK-1579: the briefing and reply are registry text the app
            # resolves; a command without them is refused, never defaulted.
            logger.warning("Seed refused for %s: no registry framing", bot_id)
            await self._record_continuity_receipt(thread_session_id, bot_id, messages, restored=False)
            return {"seeded": False, "reason": "seed framing missing"}
        try:
            session_id = str(uuid.uuid4())
            self._write_seed_transcript(session_id, messages, framing)
            # Persist the minted SDK session id onto the thread.
            if thread_session_id:
                await self._set_thread_session(
                    thread_session_id, bot_id, session_id, model,
                )
        except Exception as e:
            logger.warning("Seed write/persist failed for %s: %s", bot_id, e)
            await self._record_continuity_receipt(thread_session_id, bot_id, messages, restored=False)
            return {"seeded": False, "reason": f"seed write failed: {e}"}
        await self._record_continuity_receipt(thread_session_id, bot_id, messages, restored=True)
        stats = dict(stats)
        stats["seeded"] = True
        stats["clean_start"] = False
        stats["session_id"] = session_id
        return stats

    @staticmethod
    def _format_seed_ack(stats: dict | None, skills: dict | None = None) -> str:
        """Human-facing /new acknowledgement (``agent_bridge.continuity``)."""
        return format_seed_ack(stats, skills)
