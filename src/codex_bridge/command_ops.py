from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from agent_bridge.events import AgentEvent, AgentEventKind, synthesize_event_id
from agent_bridge.publisher import COMMANDS_STREAM, RedisPublisher
from agent_bridge.session_queue import SessionQueue

from .transport import CodexTransport, validate_auth_json
from .mcp_context import codex_mcp_environment
from .local_plugins import CodexMcpContextConfigError
from ._bridge_helpers import (
    CONSUMER_GROUP,
    CONSUMER_NAME,
    MCP_TOOL_CONTEXT_KEY,
    _MCP_TOOL_CONTEXT_FALLBACK,
    _ModelInfoCache,
    _bot_slug_from_session_key,
    _is_auth_failure,
    _is_codex_session_error,
)

logger = logging.getLogger("codex_bridge.bridge")


class CodexCommandMixin:
    """Codex Redis command listener + send/RPC handlers (TASK-555).

    Split out of ``CodexBridge`` (TASK-555). Composed back on via
    inheritance, so ``self.*`` state set in ``CodexBridge.__init__`` and
    methods on sibling mixins all resolve on the assembled instance.
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
            # id="$" so a brand-new codex-bridge group skips historical
            # RPCs (which were authored before the codex bridge existed
            # and target other backends). Once the group is established,
            # this id is no longer consulted — xreadgroup uses the
            # group's last delivered id, so restart-on-codex-RPC works.
            await async_redis.xgroup_create(
                COMMANDS_STREAM, CONSUMER_GROUP, id="$", mkstream=True,
            )
        except Exception:
            pass

        logger.info(
            "Command listener started on %s (group=%s)",
            COMMANDS_STREAM, CONSUMER_GROUP,
        )

        while True:
            try:
                results = await async_redis.xreadgroup(
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
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
                            await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)
                            continue
                        # Filter RPCs that explicitly target a different backend.
                        # Legacy callers send RPCs without `backend` — we still
                        # accept those (they may target any agent backend).
                        if action == "rpc.call" and backend and backend != self._backend_name:
                            await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)
                            continue

                        if action == "chat.send":
                            asyncio.create_task(
                                self._handle_send(fields, msg_id, async_redis)
                            )
                        elif action == "rpc.call":
                            asyncio.create_task(
                                self._handle_rpc(fields, msg_id, async_redis)
                            )
                        else:
                            await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)

            except asyncio.CancelledError:
                await async_redis.aclose()
                raise
            except Exception:
                logger.exception("Command listener error")
                await asyncio.sleep(2)

    async def _handle_send(
        self, fields: dict, msg_id: str, async_redis,
    ) -> None:
        request_id = fields.get("request_id", "")
        session_key = fields.get("session_key", "")
        bot_slug = (fields.get("bot_id", "") or "").strip() or _bot_slug_from_session_key(session_key)
        message = fields.get("message", "")
        system_prompt = fields.get("system_prompt") or None
        model = fields.get("model") or self._default_model
        task_turn_capability = (fields.get("task_turn_capability") or "").strip()
        # Frontend-supplied user-message UUID; stamped on every emitted event
        # so the frontend can bucket tool activity under the originating user
        # message without falling back to turn_id heuristics.
        trigger_message_id = (fields.get("trigger_message_id") or "").strip() or None
        attachments_raw = fields.get("attachments", "")
        attachments: list[dict] = []
        if attachments_raw:
            try:
                attachments = json.loads(attachments_raw)
            except json.JSONDecodeError:
                pass

        if not request_id or not message or (task_turn_capability and not session_key):
            logger.warning(
                "Invalid send command: missing request_id/message or contextual session_key"
            )
            await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)
            return

        if trigger_message_id:
            self._trigger_message_ids[request_id] = trigger_message_id

        # Parse per-thread binding fields (TASK-638).
        thread_session_id = (fields.get("thread_session_id") or "").strip() or None
        thread_resume_id = (fields.get("thread_resume_id") or "").strip() or None
        if thread_resume_id and ":" in thread_resume_id:
            logger.warning("Ignoring routing-key thread_resume_id for %s: %s", bot_slug, thread_resume_id)
            thread_resume_id = None
        explicit_thread = (fields.get("explicit_thread") or "").strip() in ("1", "true")

        # /new resets the session — strip and start fresh. Explicit-thread
        # turns never process /new (message passes through as literal text).
        reset_requested = (
            not explicit_thread and message.lstrip().startswith("/new")
        )
        if reset_requested:
            self._publish_session_reset_unified(
                bot_slug or session_key, session_key, had_session=True,
            )
            message = message.lstrip().removeprefix("/new").strip()
            if not message:
                self._publish_event(
                    request_id, session_key, 1,
                    kind=AgentEventKind.ASSISTANT_DONE,
                    text="Session reset. Ready for a new conversation.",
                    model=model,
                )
                self._publisher.publish_run_done(request_id)
                await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)
                return

        if self._session_queue.is_busy(session_key):
            logger.info(
                "Session %s busy — queuing send request_id=%s",
                session_key, request_id,
            )

        cleanup_complete = False

        async def on_cancel() -> None:
            # Includes cancellation while queued or during pre-SDK setup.
            self._publisher.publish_run_done(request_id)
            if not cleanup_complete:
                self._discard_changed_file_request(request_id)
                self._trigger_message_ids.pop(request_id, None)
                await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)

        async with self._session_queue.active(
            session_key, request_id=request_id, on_cancel=on_cancel,
        ):
            logger.info(
                "Handling send: request_id=%s session=%s model=%s system_prompt=%s msg=%.60s...",
                request_id, session_key, model,
                f"{len(system_prompt)} chars" if system_prompt else "none",
                message,
            )

            seq = 0
            text_parts: list[str] = []
            actual_model: str = model
            tmp_image_paths: list[str] = []

            try:
                # MCP context injection (matches claude bridge). Body from the
                # registry (TASK-490) with a byte-identical local fallback.
                if system_prompt and bot_slug:
                    mcp_ctx = await self._get_mcp_tool_context(bot_slug)
                    system_prompt += f"\n\n{mcp_ctx}"

                # Session resolution: the app resolved the thread's stored
                # SDK session id (model-checked). None → cold-start.
                resume_id: str | None = (
                    None if reset_requested else thread_resume_id
                )
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

                # Cooperative cancel event for chat.abort (TASK-205)
                cancel_event = (
                    self._session_queue.cancel_event(session_key) if session_key else None
                )
                fresh_session_retry = False
                while True:
                    try:
                        from agent_bridge.skill_selection import guard_resumed_bundle
                        guard_resumed_bundle(fields.get("skill_bundle"), thread=thread_session_id, resume=resume_id, harness="codex")
                        codex_environment: dict[str, str] | None = None
                        if fields.get("skill_bundle") is not None:
                            from agent_bridge.skill_codex import codex_skill_env
                            from .local_plugins import ensure_bawthub_mcp_config
                            codex_environment = codex_skill_env(fields["skill_bundle"])
                            ensure_bawthub_mcp_config(
                                logger=logger,
                                codex_home=Path(codex_environment["CODEX_HOME"]),
                            )
                        if task_turn_capability:
                            codex_environment = codex_mcp_environment(
                                codex_environment or os.environ,
                                capability=task_turn_capability,
                                agent_request_id=request_id,
                                session_key=session_key,
                                backend=self._backend_name,
                            )
                        codex = (
                            self._transport.codex_for_environment(codex_environment)
                            if codex_environment is not None
                            else self._ensure_codex()
                        )
                    except CodexMcpContextConfigError as context_error:
                        seq += 1
                        self._publish_event(
                            request_id, session_key, seq,
                            kind=AgentEventKind.ERROR,
                            text=f"Codex MCP context setup failed — {context_error}",
                            model=model,
                        )
                        self._publisher.publish_run_done(request_id)
                        return
                    except RuntimeError as auth_err:
                        # auth.json missing/invalid at startup — surface to user
                        seq += 1
                        self._publish_event(
                            request_id, session_key, seq,
                            kind=AgentEventKind.ERROR,
                            text=f"Codex OAuth failed — {auth_err}",
                            model=model,
                        )
                        self._publisher.publish_run_done(request_id)
                        return

                    # Build prompt input fresh each retry attempt — temp files
                    # for images get rebuilt too (the inner loop may clear and
                    # rebuild after a recoverable session error).
                    self._cleanup_tmp_files(tmp_image_paths)
                    # TASK-288 observability: log the system_prompt value AS SENT
                    # (prepended to the user message by _build_prompt_input),
                    # paired with resume state — the only place the resume-gate
                    # decision is visible.
                    logger.info(
                        "Codex prompt: resume=%s system_prompt_sent=%s",
                        bool(resume_id),
                        f"{len(system_prompt)} chars" if system_prompt else "none",
                    )
                    prompt_input, tmp_image_paths = self._build_prompt_input(
                        message,
                        attachments,
                        # TASK-288: inject the system prompt on EVERY turn, resume
                        # included. Previously gated to fresh threads only, which
                        # meant the persona decayed to nothing on long-lived codex
                        # sessions (the steady state, since the thread is resumed
                        # every turn). The prompt is now byte-stable (temporal +
                        # response-style moved off it in core/base.py), so re-
                        # sending keeps persona alive. Codex prepends it to the
                        # user message rather than a cached system param, so this
                        # adds the (stable) persona to each turn's input.
                        system_prompt=system_prompt,
                    )

                    aborted = False
                    item_text: dict[str, str] = {}  # last seen text per agent_message item
                    item_buffers: dict[str, dict[str, Any]] = {}
                    final_usage: Any = None
                    session_persisted = bool(resume_id)

                    # Per-turn AbortController so chat.abort can interrupt the
                    # codex subprocess server-side via the SDK's signal plumbing.
                    from openai_codex_sdk import AbortController, AbortError

                    controller = AbortController()
                    if session_key:
                        self._session_queue.set_active_client(session_key, controller)

                    try:
                        # Open or resume thread (sync API)
                        thread_options = {
                            "model": model,
                            "working_directory": self._cwd,
                            "approval_policy": "never",
                            "sandbox_mode": "danger-full-access",
                            "skip_git_repo_check": True,
                        }
                        if resume_id:
                            thread = codex.resume_thread(resume_id, thread_options)
                        else:
                            thread = codex.start_thread(thread_options)

                        streamed = await thread.run_streamed(
                            prompt_input,
                            {"signal": controller.signal},
                        )

                        async for event in self._iter_events(streamed):
                            if cancel_event is not None and cancel_event.is_set():
                                logger.info(
                                    "chat.abort signalled, aborting Codex turn: session=%s request=%s",
                                    session_key, request_id,
                                )
                                aborted = True
                                try:
                                    controller.abort("chat.abort")
                                except Exception:
                                    logger.debug("controller.abort raised", exc_info=True)
                                break

                            (
                                seq,
                                persisted_now,
                                usage_now,
                            ) = await self._handle_event(
                                event,
                                bot_slug=bot_slug,
                                session_key=session_key,
                                request_id=request_id,
                                seq=seq,
                                text_parts=text_parts,
                                item_text=item_text,
                                model=model,
                                actual_model_ref=[actual_model],
                                resume_id=resume_id,
                                session_persisted=session_persisted,
                                item_buffers=item_buffers,
                                thread=thread,
                                thread_session_id=thread_session_id,
                            )
                            if persisted_now:
                                session_persisted = True
                            if usage_now is not None:
                                final_usage = usage_now

                        # On a fresh thread, capture the id once the stream is
                        # done in case we never saw a typed thread.started event.
                        if (
                            not session_persisted
                            and not resume_id
                            and bot_slug
                            and thread_session_id
                            and getattr(thread, "id", None)
                        ):
                            await self._set_thread_session(
                                thread_session_id, bot_slug,
                                str(thread.id), model,
                            )
                            session_persisted = True

                        if not aborted and not self._already_sent_done(seq, item_buffers):
                            token_usage = self._extract_token_usage(
                                final_usage,
                                actual_model_or_default=actual_model,
                            )
                            token_usage = self._sanitize_context_usage(
                                await self._merge_model_info(
                                    token_usage, actual_model,
                                )
                            )
                            seq += 1
                            self._publish_event(
                                request_id, session_key, seq,
                                kind=AgentEventKind.ASSISTANT_DONE,
                                text="".join(text_parts),
                                model=actual_model,
                                token_usage=token_usage,
                            )
                        break

                    except asyncio.CancelledError:
                        logger.info(
                            "Send cancelled via task.cancel: request_id=%s session=%s",
                            request_id, session_key,
                        )
                        # The app persists partial text and aborted status.
                        # Do not emit a successful assistant completion.
                        try:
                            controller.abort("task_cancelled")
                        except Exception:
                            pass
                        raise

                    except AbortError:
                        # Triggered by controller.abort(); the app owns partial
                        # persistence, so only publish the transport terminal.
                        logger.info(
                            "Codex turn aborted: request_id=%s session=%s",
                            request_id, session_key,
                        )
                        aborted = True
                        break

                    except Exception as e:
                        # Auth failure → publish ERROR, drop SDK handle, no retry.
                        if _is_auth_failure(e):
                            logger.warning(
                                "Codex auth failure for %s: %s — resetting SDK handle",
                                request_id, e,
                            )
                            self._supervisor_teardown()
                            seq += 1
                            self._publish_event(
                                request_id, session_key, seq,
                                kind=AgentEventKind.ERROR,
                                text=(
                                    "[credential_expired:codex] "
                                    "Codex OAuth failed — check the app-owned OpenAI ChatGPT "
                                    "provider connection and bridge broker materialization; "
                                    "host CLI credentials are a separate path"
                                ),
                                model=model,
                            )
                            self._publisher.publish_run_done(request_id)
                            return

                        # Recoverable session error → retry once with fresh thread.
                        if (
                            not fresh_session_retry
                            and not text_parts
                            and _is_codex_session_error(e)
                        ):
                            fresh_session_retry = True
                            logger.warning(
                                "Recoverable session error for %s: %s — retrying fresh",
                                request_id, e,
                            )
                            resume_id = None
                            continue

                        raise
                    finally:
                        if session_key:
                            current = self._session_queue.get_active_client(session_key)
                            if current is controller:
                                self._session_queue.pop_active_client(session_key)

                self._publisher.publish_run_done(request_id)
                if aborted:
                    logger.info(
                        "Send aborted via chat.abort: request_id=%s session=%s",
                        request_id, session_key,
                    )
                else:
                    logger.info(
                        "Send completed: request_id=%s session=%s",
                        request_id, session_key,
                    )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("Send failed: request_id=%s", request_id)
                seq += 1
                self._publish_event(
                    request_id, session_key, seq,
                    kind=AgentEventKind.ERROR,
                    text=str(e),
                    model=model,
                )
                self._publisher.publish_run_done(request_id)
            finally:
                self._discard_changed_file_request(request_id)
                self._cleanup_tmp_files(tmp_image_paths)
                # Drop the per-run trigger_message_id mapping so we don't leak.
                self._trigger_message_ids.pop(request_id, None)
                await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)
                cleanup_complete = True

    async def _handle_rpc(
        self, fields: dict, msg_id: str, async_redis,
    ) -> None:
        request_id = fields.get("request_id", "")
        method = fields.get("method", "")
        message_backend = (fields.get("backend") or "").strip()
        params_raw = fields.get("params", "{}")
        try:
            params = json.loads(params_raw) if isinstance(params_raw, str) else params_raw
        except (json.JSONDecodeError, TypeError):
            params = {}

        # Defensive bot-ownership check: when the message has no `backend`
        # field (legacy callers), look up the bot and skip silently if it
        # isn't ours. The command-listener already filtered explicit
        # backend=other; this catches the no-hint case.
        if not message_backend:
            session_key = (params.get("sessionKey") or "") if isinstance(params, dict) else ""
            target = _bot_slug_from_session_key(session_key) or session_key
            if target and not await self._bot_uses_codex(target):
                await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)
                return

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
                        request_id, {"ok": False, "error": "missing sessionKey"},
                    )
            elif method == "chat.abort":
                session_key = params.get("sessionKey", "")
                target_request_id = params.get("requestId", "")
                if (
                    not isinstance(session_key, str) or not session_key.strip()
                    or not isinstance(target_request_id, str) or not target_request_id.strip()
                ):
                    raise ValueError("chat.abort requires sessionKey and requestId")
                # The targeted send owns its AbortController and interrupts it
                # in its cancellation handler; never take a sibling's handle.
                cancelled = self._session_queue.cancel_request(session_key, target_request_id)
                self._publisher.publish_rpc_result(
                    request_id,
                    {"ok": True, "aborted": cancelled, "requestId": target_request_id,
                     "detail": "task_cancelled" if cancelled else "no_matching_request"},
                )
            else:
                self._publisher.publish_rpc_result(
                    request_id, {"ok": False, "error": f"unknown method: {method}"},
                )
        except Exception as e:
            logger.warning("RPC %s failed: %s", method, e)
            self._publisher.publish_rpc_result(
                request_id, {"ok": False, "error": str(e)},
            )
        finally:
            await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)
