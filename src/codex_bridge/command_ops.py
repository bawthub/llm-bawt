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
from agent_bridge.abort_rpc import answer_abort, cancel_request_handler
from agent_bridge.run_status import RUN_STATUS_METHOD, answer_run_status, queue_run_state
from agent_bridge.send_admission import admit_send
from agent_bridge.events import AgentEvent, AgentEventKind, synthesize_event_id
from agent_bridge.maintenance import is_new_command
from agent_bridge.publisher import COMMANDS_STREAM, RedisPublisher
from agent_bridge.session_queue import SessionQueue

from .auth_broker import materialize_auth_from_broker
from .process_ledger import ProcessLedger, bind_ledger
from .transport import CodexTransport, auth_path, validate_auth_json
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
    system_prompt_for_attempt,
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
        # TASK-1232 P2: registered at receipt, before any await.
        # TASK-1233: admit_send publishes the run's one done, with the
        # ledger's report on whether every codex subprocess was reaped.
        ledger = ProcessLedger()
        bind_ledger(ledger)
        await admit_send(
            self._session_queue, fields.get("session_key", ""), fields.get("request_id", ""),
            msg_id, async_redis, consumer_group=CONSUMER_GROUP, publisher=self._publisher,
            run=lambda: self._run_send(fields, msg_id, async_redis),
            exit_probe=ledger.report,
        )

    async def _run_send(
        self, fields: dict, msg_id: str, async_redis,
    ) -> None:
        request_id = fields.get("request_id", "")
        session_key = fields.get("session_key", "")
        bot_slug = (fields.get("bot_id", "") or "").strip() or _bot_slug_from_session_key(session_key)
        message = fields.get("message", "")
        system_prompt = fields.get("system_prompt") or None
        system_prompt_delivery = (fields.get("system_prompt_delivery") or "").strip() or None
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

        from .reset_evidence import CodexResetEvidence

        try:
            reset_evidence = CodexResetEvidence.from_fields(
                fields, thread_session_id=thread_session_id,
                explicit_thread=explicit_thread, thread_resume_id=thread_resume_id,
            )
        except ValueError as exc:
            self._publish_event(request_id, session_key, 1,
                                kind=AgentEventKind.ERROR, text=str(exc), model=model)
            self._trigger_message_ids.pop(request_id, None)
            await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)
            return

        # Structured reset-follow-up preserves the supplied task byte-for-byte.
        # Its app-owned receipt is published only after verified completion.
        # Legacy /new parsing remains unchanged for non-structured commands.
        reset_requested = bool(reset_evidence) or (
            not explicit_thread and is_new_command(message)
        )
        if reset_requested and reset_evidence is None:
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
                if reset_evidence is not None and reset_evidence.reset_only:
                    # Codex has no process-global conversation: continuity is
                    # solely the app-supplied resume ID. A clean standalone reset
                    # acknowledges explicit no-resume under the session queue;
                    # the app verifies its empty successor before committing the
                    # receipt. Do not invent a task merely to mint a provider ID.
                    self._publish_event(
                        request_id, session_key, 1,
                        kind=AgentEventKind.ASSISTANT_DONE,
                        text="Session reset. Ready for a new conversation.",
                        model=model,
                        extra_raw={"new_session_ack": reset_evidence.acknowledgment()},
                    )
                    return

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
                        return

                    # Build prompt input fresh each retry attempt — temp files
                    # for images get rebuilt too (the inner loop may clear and
                    # rebuild after a recoverable session error).
                    self._cleanup_tmp_files(tmp_image_paths)
                    # TASK-1318: a session-pinned prompt is pasted only when it
                    # takes effect (fresh thread, or the app moved the pin);
                    # the resumed thread's history already carries it. Without
                    # a delivery mode it is pasted every turn (TASK-288).
                    attempt_system_prompt = system_prompt_for_attempt(
                        system_prompt, system_prompt_delivery, resuming=bool(resume_id),
                    )
                    # TASK-288 observability: log the system_prompt value AS SENT
                    # (prepended to the user message by _build_prompt_input),
                    # paired with resume state — the only place the resume-gate
                    # decision is visible.
                    logger.info(
                        "Codex prompt: resume=%s delivery=%s system_prompt_sent=%s",
                        bool(resume_id), system_prompt_delivery or "every_turn",
                        f"{len(attempt_system_prompt)} chars" if attempt_system_prompt else "none",
                    )
                    prompt_input, tmp_image_paths = self._build_prompt_input(
                        message,
                        attachments,
                        system_prompt=attempt_system_prompt,
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
                                reset_evidence=reset_evidence,
                            )
                            if persisted_now:
                                session_persisted = True
                            if usage_now is not None:
                                final_usage = usage_now

                        # On a fresh thread, capture the id once the stream is
                        # done in case we never saw a typed thread.started event.
                        if (
                            reset_evidence is None
                            and not session_persisted
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
                            if reset_evidence is not None:
                                raise RuntimeError("Codex reset has no verified turn.completed event")
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
                        # Never replay a possibly started turn after auth failure.
                        # Refresh the local CLI bundle from the app owner for the
                        # NEXT request, including after an account reconnect.
                        if _is_auth_failure(e):
                            logger.warning(
                                "Codex auth failure for %s: %s — refreshing broker bundle",
                                request_id, e,
                            )
                            refreshed = await asyncio.to_thread(
                                materialize_auth_from_broker, auth_path(),
                            )
                            if not refreshed:
                                logger.warning("Codex broker bundle refresh unavailable")
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
                            return

                        # Recoverable session error → retry once with fresh thread.
                        if (
                            reset_evidence is None
                            and not fresh_session_retry
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
                # The targeted send owns its AbortController and interrupts it
                # in its cancellation handler; never take a sibling's handle.
                self._publisher.publish_rpc_result(
                    request_id,
                    await answer_abort(self._backend_name, params,
                                       cancel_request_handler(self._session_queue)),
                )
            elif method == RUN_STATUS_METHOD and message_backend == self._backend_name:
                # Answered only when addressed to us: see agent_bridge.run_status.
                self._publisher.publish_rpc_result(
                    request_id,
                    answer_run_status(
                        self._backend_name, params,
                        lambda sk, rid: queue_run_state(self._session_queue, sk, rid),
                    ),
                )
            elif message_backend == self._backend_name:
                # Unknown RPC addressed to us. A backend-less unknown method
                # may belong to another bridge (OpenClaw's agents.files.*);
                # answering it would race the owner's reply.
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
