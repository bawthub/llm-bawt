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


class ClaudeCommandMixin:
    """Claude Redis command listener + tool-result/RPC handlers (TASK-555).

    Split out of ``ClaudeCodeBridge`` (TASK-555); composed back via
    inheritance so ``self.*`` state and sibling-mixin methods resolve
    on the assembled instance.
    """

    async def _command_listener(self) -> None:
        import redis.asyncio as aioredis

        conn_kwargs = self._publisher._redis.connection_pool.connection_kwargs
        host = conn_kwargs.get("host", "localhost")
        port = conn_kwargs.get("port", 6379)
        db = conn_kwargs.get("db", 0)
        # socket_timeout=None: redis-py 8.0 defaults to 5s, which races our
        # blocking XREADGROUP(block=5000) reads. Bound only the connect.
        async_redis = aioredis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_timeout=None,
            socket_connect_timeout=5,
        )
        await async_redis.ping()
        self._redis = async_redis

        try:
            await async_redis.xgroup_create(
                COMMANDS_STREAM, "claude-code-bridge", id="0", mkstream=True,
            )
        except Exception:
            pass

        logger.info("Command listener started on %s (group=claude-code-bridge)", COMMANDS_STREAM)

        # TASK-291: subscribe to approval-policy reload broadcasts so admin
        # edits drop the cached bundle without a restart. Best-effort.
        if self._approval_reload_task is None or self._approval_reload_task.done():
            reload_url = f"redis://{host}:{port}/{db}"
            self._approval_reload_task = asyncio.create_task(
                self._approval_reload_listener(reload_url)
            )

        while True:
            try:
                results = await async_redis.xreadgroup(
                    "claude-code-bridge",
                    "worker-0",
                    {COMMANDS_STREAM: ">"},
                    count=1,
                    block=5000,
                )
                if not results:
                    continue

                for _stream, messages in results:
                    for msg_id, fields in messages:
                        action = fields.get("action", "")
                        backend = fields.get("backend", "")

                        if action == "chat.send" and backend != self._backend_name:
                            # Not our chat command — ACK so it doesn't pile up
                            await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
                            continue

                        if action == "chat.send":
                            asyncio.create_task(
                                self._handle_send(fields, msg_id, async_redis)
                            )
                        elif action == "rpc.call":
                            asyncio.create_task(
                                self._handle_rpc(fields, msg_id, async_redis)
                            )
                        elif action == "chat.steer":
                            # Side-channel into the live ClaudeSDKClient. Never
                            # take the per-session send lock: _handle_send owns it
                            # for the entire run and is the sole response consumer.
                            asyncio.create_task(
                                self._handle_steer(fields, msg_id, async_redis)
                            )
                        elif action == "chat.tool_result":
                            # Resolves a pending AskUserQuestion Future so the
                            # paused SDK turn can continue.  Deliberately NOT
                            # wrapped in the session lock — _handle_send is
                            # holding that lock while awaiting the Future, so
                            # taking it here would deadlock the run forever.
                            asyncio.create_task(
                                self._handle_tool_result(fields, msg_id, async_redis)
                            )
                        elif action == "approval.grant":
                            # TASK-292: user approved a gated tool. Store a
                            # one-shot allow so the re-issued call on the
                            # continuation turn sails through.
                            if backend and backend != self._backend_name:
                                await async_redis.xack(
                                    COMMANDS_STREAM, "claude-code-bridge", msg_id
                                )
                                continue
                            asyncio.create_task(
                                self._handle_approval_grant(fields, msg_id, async_redis)
                            )
                        else:
                            await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)

            except asyncio.CancelledError:
                await async_redis.aclose()
                raise
            except Exception:
                logger.exception("Command listener error")
                await asyncio.sleep(2)

    async def _handle_steer(
        self, fields: dict, msg_id: str, async_redis,
    ) -> None:
        """Steer one active Claude SDK run in place.

        ``priority`` ``now`` (default, user redirection) interrupts running
        tools; ``next`` (bot deliveries) waits for the next tool boundary.
        """
        backend = (fields.get("backend") or "").strip()
        if backend and backend != self._backend_name:
            await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
            return

        request_id = (fields.get("request_id") or "").strip()
        session_key = (fields.get("session_key") or "").strip()
        message = (fields.get("message") or "").strip()
        priority = (fields.get("priority") or "now").strip().lower()
        try:
            if not session_key or not message:
                raise ValueError("chat.steer requires session_key and message")
            active_run = self._session_queue.get_active_client(session_key)
            if active_run is None or not hasattr(active_run, "steer"):
                raise RuntimeError("no_active_run")
            active_request_id = getattr(active_run, "request_id", "")
            target_request_id = (fields.get("target_request_id") or "").strip()
            if target_request_id and target_request_id != active_request_id:
                raise RuntimeError("active_run_mismatch")

            message_id = (fields.get("message_id") or "").strip()
            await active_run.steer(
                message, priority=priority, message_id=message_id or None
            )
            # A steer adds a separate user message to this run; it does not
            # change the originating turn's trigger. All run events (including
            # tools and terminal usage) retain that canonical owner.
            logger.info(
                "chat.steer accepted: session=%s active_request=%s priority=%s chars=%d",
                session_key,
                active_request_id or "?",
                priority,
                len(message),
            )
            if request_id:
                self._publisher.publish_rpc_result(
                    request_id,
                    {
                        "ok": True,
                        "detail": "steered",
                        "active_request_id": active_request_id or None,
                    },
                )
        except Exception as exc:
            logger.warning(
                "chat.steer failed: session=%s request=%s error=%s",
                session_key,
                request_id,
                exc,
            )
            if request_id:
                self._publisher.publish_rpc_result(
                    request_id,
                    {"ok": False, "error": str(exc)},
                )
        finally:
            await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)

    async def _handle_tool_result(
        self, fields: dict, msg_id: str, async_redis,
    ) -> None:
        """Deprecated no-op (TASK-269).

        AskUserQuestion no longer blocks on an in-process Future, so there is
        nothing to resolve here.  The answer now arrives as a brand-new
        continuation turn (chat.send carrying the user's answer).  We keep this
        handler only to drain/ACK any stray chat.tool_result commands a stale
        client might still emit during a deploy window.
        """
        tool_use_id = (fields.get("tool_use_id") or "").strip()
        logger.info(
            "chat.tool_result is deprecated and ignored (tool_use_id=%s) — "
            "answers are delivered as continuation turns now",
            tool_use_id,
        )
        try:
            await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
        except Exception:
            pass

    async def _handle_rpc(
        self, fields: dict, msg_id: str, async_redis,
    ) -> None:
        """Handle RPC commands (e.g. session.reset)."""
        backend = fields.get("backend", "")
        if backend and backend != self._backend_name:
            await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
            return
        import json as _json

        request_id = fields.get("request_id", "")
        method = fields.get("method", "")
        params_raw = fields.get("params", "{}")
        try:
            params = _json.loads(params_raw) if isinstance(params_raw, str) else params_raw
        except (_json.JSONDecodeError, TypeError):
            params = {}

        try:
            if method == "session.reset":
                session_key = params.get("sessionKey", "")
                # The app handles thread rotation; the bridge just publishes
                # the SESSION_RESET event so the frontend clears its buffer.
                # The scalar agent_backend_config.session_key is retired
                # (TASK-638); no _clear_session call needed.
                reset_user = (params.get("userId") or "").strip() or "nick"
                target = _bot_slug_from_session_key(session_key) or session_key
                if target:
                    logger.info("Session reset: %s", target)
                    self._publish_session_reset_unified(
                        target, session_key or target,
                        had_session=True, user_id=reset_user,
                    )
                    self._publisher.publish_rpc_result(
                        request_id,
                        {
                            "ok": True,
                            "reset": target,
                            "had_session": True,
                        },
                    )
                else:
                    self._publisher.publish_rpc_result(
                        request_id, {"ok": False, "error": "missing sessionKey"}
                    )
            elif method == "chat.cancel":
                # Transport failures belong to one request, not the whole
                # session: a queued caller must never abort its running sibling.
                session_key = params.get("sessionKey", "")
                target_request_id = params.get("requestId", "")
                cancelled = bool(target_request_id) and self._session_queue.cancel_request(
                    session_key, target_request_id,
                )
                self._publisher.publish_rpc_result(
                    request_id,
                    {"ok": True, "cancelled": cancelled, "request_id": target_request_id},
                )
            elif method == "chat.abort":
                session_key = params.get("sessionKey", "")
                target_request_id = params.get("requestId", "")
                if not session_key or not target_request_id:
                    raise ValueError("chat.abort requires sessionKey and requestId")
                # Claim and cancel only this request before yielding. Disconnect
                # may outlive the run and a successor may own the session by then.
                client = self._session_queue.get_active_client(session_key)
                cancelled = self._session_queue.cancel_request(session_key, target_request_id)
                if cancelled and getattr(client, "request_id", None) == target_request_id:
                    client = self._session_queue.pop_active_client(session_key)
                else:
                    client = None
                client_disconnected = False
                if client is not None:
                    try:
                        await asyncio.wait_for(client.disconnect(), timeout=20.0)
                        client_disconnected = True
                    except asyncio.TimeoutError:
                        logger.warning(
                            "disconnect() timed out for session %s — subprocess may be stuck",
                            session_key,
                        )
                    except Exception:
                        logger.debug(
                            "disconnect() raised on chat.abort for session %s",
                            session_key,
                            exc_info=True,
                        )
                detail_parts: list[str] = []
                if cancelled:
                    detail_parts.append("task_cancelled")
                if client_disconnected:
                    detail_parts.append("client_disconnected")
                if not detail_parts:
                    detail_parts.append("no_active_task")
                logger.info(
                    "chat.abort: session=%s detail=%s",
                    session_key,
                    ",".join(detail_parts),
                )
                self._publisher.publish_rpc_result(
                    request_id,
                    {
                        "ok": True,
                        "aborted": session_key,
                        "cancelled": cancelled,
                        "request_id": target_request_id,
                        "detail": ",".join(detail_parts),
                    },
                )
            else:
                # Unknown RPC — return ok so callers don't hang
                self._publisher.publish_rpc_result(
                    request_id, {"ok": False, "error": f"unknown method: {method}"}
                )
        except Exception as e:
            logger.warning("RPC %s failed: %s", method, e)
            self._publisher.publish_rpc_result(
                request_id, {"ok": False, "error": str(e)}
            )
        finally:
            await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
