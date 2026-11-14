from __future__ import annotations

import asyncio
import json
import logging


from agent_bridge.abort_contract import ABORT_TEARDOWN_BOUND_S, AbortState
from agent_bridge.user_scope import session_user
from agent_bridge.abort_rpc import answer_abort
from agent_bridge.consumer_group import GroupReader
from agent_bridge.houses import HouseSlot, gate_send, house_from_env, is_recycle, start_recycle
from agent_bridge.ping import PING_METHOD, answer_ping
from agent_bridge.run_status import RUN_STATUS_METHOD, answer_run_status, queue_run_state
from agent_bridge.steer_contract import (
    ATTACHMENTS_FIELD,
    parse_steer_attachments,
    FRAMED_MESSAGE_FIELD,
    INTERRUPT_FRAME_FIELD,
    INTERRUPT_ITEM_FIELD,
)
from agent_bridge.publisher import COMMANDS_STREAM

from ._bridge_helpers import (
    _bot_slug_from_session_key,
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
        self._command_ready = False
        try:
            await self._consume_commands(async_redis, host, port, db)
        finally:
            self._command_ready = False
            await async_redis.aclose()

    async def _consume_commands(self, async_redis, host, port, db) -> None:
        reader = GroupReader(
            async_redis, "claude-code-bridge", "worker-0", [COMMANDS_STREAM], initial_id="0",
        )
        # Configuration errors are fatal; transient Redis initialization errors
        # recover on this same client (also used by the house heartbeat).
        if getattr(self, "_house_slot", None) is None:
            self._house_slot = HouseSlot(house_from_env(), self._backend_name, self._session_queue)
        delay = 2
        while True:
            try:
                await async_redis.ping()
                await reader.ensure(strict=True)
                await self._house_slot.start(async_redis)
                break
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Command listener startup failed; retrying in %ss", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)
        self._redis = async_redis
        self._command_ready = True
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
                results = await reader.read(count=1, block=5000)
                self._command_ready = True
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
                            if not await gate_send(
                                getattr(self, "_house_slot", None), fields, msg_id, async_redis,
                                group="claude-code-bridge", publisher=self._publisher,
                            ):
                                continue
                            asyncio.create_task(
                                self._handle_send(fields, msg_id, async_redis)
                            )
                        elif is_recycle(getattr(self, "_house_slot", None), fields):
                            # TASK-2080: drain, then exit for a fresh container.
                            start_recycle(self._house_slot, fields, msg_id, async_redis,
                                          group="claude-code-bridge")
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
                raise
            except Exception:
                self._command_ready = False
                logger.exception("Command listener error")
                await asyncio.sleep(2)

    async def _handle_steer(
        self, fields: dict, msg_id: str, async_redis,
    ) -> None:
        """Steer one active Claude SDK run in place.

        ``priority`` ``next`` queues at the next tool boundary; explicit
        ``now`` interrupts running tools. Escalation atomically cancels an
        already-queued user message before replacing it with an interrupt.
        """
        backend = (fields.get("backend") or "").strip()
        if backend and backend != self._backend_name:
            await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
            return

        request_id = (fields.get("request_id") or "").strip()
        session_key = (fields.get("session_key") or "").strip()
        message = (fields.get("message") or "").strip()
        priority = (fields.get("priority") or "next").strip().lower()
        escalate = fields.get("escalate") == "1" or fields.get("steer_action") == "escalate"
        cancel = fields.get("steer_action") == "cancel"
        try:
            if not session_key:
                raise ValueError("chat.steer requires session_key")
            if not (escalate or cancel) and not message:
                raise ValueError("chat.steer requires message")
            if escalate and cancel:
                raise ValueError("chat.steer cannot escalate and cancel together")
            active_run = self._session_queue.get_active_client(session_key)
            if active_run is None or not hasattr(active_run, "steer"):
                raise RuntimeError("no_active_run")
            active_request_id = getattr(active_run, "request_id", "")
            target_request_id = (fields.get("target_request_id") or "").strip()
            if target_request_id and target_request_id != active_request_id:
                raise RuntimeError("active_run_mismatch")

            message_id = (fields.get("message_id") or "").strip()
            if escalate or cancel:
                action = "escalation" if escalate else "cancellation"
                if message:
                    raise ValueError(f"chat.steer {action} must not include message")
                if not message_id or not target_request_id:
                    raise ValueError(f"chat.steer {action} requires message_id and target_request_id")
                method = "escalate" if escalate else "cancel_queued"
                if not hasattr(active_run, method):
                    raise RuntimeError(f"{action}_unsupported_sdk")
                if escalate:
                    # TASK-1300: the app-rendered interruption frame; the run
                    # fills in every command the sweep removes.
                    detail = await active_run.escalate(
                        message_id, interrupt_frame=fields.get(INTERRUPT_FRAME_FIELD) or None,
                    )
                else:
                    detail = await active_run.cancel_queued(message_id)
            else:
                # TASK-1300: app-rendered registry text, plus this RPC's id so
                # a lifecycle event names the delivery generation it settles.
                images = parse_steer_attachments(fields.get(ATTACHMENTS_FIELD))
                await active_run.steer(
                    message, priority=priority, message_id=message_id or None,
                    origin=(fields.get("origin") or "system").strip().lower(),
                    framed_message=fields.get(FRAMED_MESSAGE_FIELD) or None,
                    interrupt_item=fields.get(INTERRUPT_ITEM_FIELD) or None,
                    steer_rpc_id=request_id or None,
                    **({"images": images} if images else {}),
                )
                detail = "steered"
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
                        "detail": detail,
                        "active_request_id": active_request_id or None,
                        "can_escalate": (
                            not escalate and not cancel
                            and "interrupt_cancel_queued_v1" in getattr(active_run, "_cli_capabilities", set())
                        ),
                        "can_cancel": not escalate and not cancel and hasattr(active_run, "cancel_queued"),
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
        """Resolve a live AskUserQuestion without taking the session send lock."""
        tool_use_id = (fields.get("tool_use_id") or "").strip()
        entry = self._pending_question_futures.get(tool_use_id)
        if (fields.get("backend") or self._backend_name) == self._backend_name and entry:
            session_key, future = entry
            if fields.get("session_key") == session_key and not future.done():
                # dict = native answers (allow); str = deny text (dismissal).
                answers = None
                if fields.get("answers_json"):
                    try:
                        answers = json.loads(fields["answers_json"])
                    except (TypeError, ValueError):
                        logger.warning("Invalid answers_json for tool_use_id=%s", tool_use_id)
                future.set_result(
                    answers if isinstance(answers, dict) and answers
                    else fields.get("result", "")
                )
        await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)

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
                reset_user = session_user(session_key, params.get("userId"))
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
                if target_request_id:
                    cancel_proxy = getattr(self, "_cancel_proxy_sampling", None)
                    if cancel_proxy is not None:
                        cancel_proxy(target_request_id)
                # TASK-1232 P2: stop_request also tombstones the id, so a
                # send for it that hasn't been received yet is refused.
                cancelled = bool(target_request_id) and self._session_queue.stop_request(
                    session_key, target_request_id,
                )
                if target_request_id and not cancelled:
                    clear_proxy = getattr(self, "_clear_proxy_sampling_cancel", None)
                    if clear_proxy is not None:
                        clear_proxy(target_request_id)
                self._publisher.publish_rpc_result(
                    request_id,
                    {"ok": True, "cancelled": cancelled, "request_id": target_request_id},
                )
            elif method == "chat.abort":
                self._publisher.publish_rpc_result(
                    request_id,
                    await answer_abort(self._backend_name, params, self._abort_request),
                )
            elif method == PING_METHOD and backend == self._backend_name:
                # Readiness probe (TASK-1297). Answered on this live command
                # consumer, outside every session lock: it proves the consumer
                # that receives chat.send is reading, nothing more.
                reply = answer_ping(self._backend_name, request_id, params)
                if reply is not None:
                    self._publisher.publish_rpc_result(request_id, reply)
            elif method == RUN_STATUS_METHOD and backend == self._backend_name:
                # Answered only when addressed to us: see agent_bridge.run_status.
                self._publisher.publish_rpc_result(
                    request_id,
                    answer_run_status(
                        self._backend_name, params,
                        lambda sk, rid: queue_run_state(self._session_queue, sk, rid),
                    ),
                )
            elif backend == self._backend_name:
                # Unknown RPC addressed to us: fail fast so the caller doesn't
                # hang. A backend-less unknown method may belong to another
                # bridge; answering it would race the owner's reply, since send_rpc takes the first one.
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

    async def _abort_request(self, session_key: str, target_request_id: str) -> tuple[AbortState, str]:
        """Cancel exactly ``target_request_id``; :func:`answer_abort` replies.

        TASK-1232 P4: the abort never disconnects the client itself. The
        cancelled send owns one bounded teardown (``run_teardown``); the reply
        waits for its outcome, up to ``ABORT_TEARDOWN_BOUND_S``, and carries it
        in ``detail`` so the app's ``user_stop`` audit records whether the CLI
        is confirmed gone.
        """
        # Look up and cancel only this request before yielding; a successor
        # may own the session by the time the teardown finishes.
        teardown = getattr(self, "_run_teardowns", {}).get(target_request_id)
        cancel_proxy = getattr(self, "_cancel_proxy_sampling", None)
        if cancel_proxy is not None:
            cancel_proxy(target_request_id)
        cancelled = self._session_queue.stop_request(session_key, target_request_id)
        if not cancelled:
            clear_proxy = getattr(self, "_clear_proxy_sampling_cancel", None)
            if clear_proxy is not None:
                clear_proxy(target_request_id)
        if not cancelled:
            detail = "no_active_task"
        elif teardown is None:
            # Cancelled before the SDK client existed: nothing to tear down.
            detail = "task_cancelled"
        else:
            outcome = await teardown.wait(ABORT_TEARDOWN_BOUND_S)
            detail = "task_cancelled," + (
                outcome.detail() if outcome is not None else "teardown:pending"
            )
        logger.info("chat.abort: session=%s detail=%s", session_key, detail)
        if not cancelled:
            return AbortState.TOMBSTONED, detail
        return AbortState.CANCELLED, detail
