"""The codex send's ``app-server`` turn, its steers and its stop (TASK-1243).

``agent_backend_config.codex_transport = "app_server"`` runs a send's turn on
``codex app-server`` (:mod:`codex_bridge.app_server`) instead of
``codex exec``. That turn can take a ``priority=next`` steer while it runs
(``turn/steer``), and ``chat.abort`` stops it with ``turn/interrupt`` before
anything is killed. Notifications go through the exec event translation
(:class:`~codex_bridge.app_server_events.AppServerEventAdapter`), so the app
sees the same events from either transport.

Steer lifecycle: a steer is ``started`` once app-server reports the user
message carrying its ``clientUserMessageId``. One accepted but never seen
before the turn ended is ``uncertain`` (``agent_bridge.steer_contract``):
the app fences it and never re-queues it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from types import SimpleNamespace
from typing import Any

from agent_bridge.abort_contract import ABORT_TEARDOWN_BOUND_S, AbortState
from agent_bridge.abort_rpc import cancel_request_handler
from agent_bridge.events import AgentEventKind
from agent_bridge.publisher import COMMANDS_STREAM
from agent_bridge.steer_contract import (
    FRAMED_MESSAGE_FIELD,
    LIFECYCLE_UNCERTAIN,
    STEER_PRIORITY_UNSUPPORTED,
    STEER_UNFRAMED,
    STEER_UNSUPPORTED_TRANSPORT,
)

from ._bridge_helpers import CONSUMER_GROUP, _is_auth_failure, _is_codex_session_error
from .app_server import (
    CONTROL_RPC_TIMEOUT_S,
    GRACEFUL_STOPS,
    AppServerVersionMismatch,
    CodexAppServerRun,
    JsonRpcStdioClient,
    spawn_app_server,
)
from .app_server_events import ABORTED, AppServerEventAdapter, app_server_input

logger = logging.getLogger("codex_bridge.bridge")

APP_SERVER_TRANSPORT = "app_server"
CODEX_TRANSPORT_FIELD = "codex_transport"
# chat.abort: how long the interrupt may take before the send is cancelled.
# It and the teardown share ABORT_TEARDOWN_BOUND_S, the app's abort budget.
ABORT_INTERRUPT_BOUND_S = 10.0

_CANCELLED = object()


def wants_app_server(fields: dict) -> bool:
    return (fields.get(CODEX_TRANSPORT_FIELD) or "").strip() == APP_SERVER_TRANSPORT


class CodexAppServerMixin:
    """Composed onto ``CodexBridge`` through ``CodexCommandMixin``."""

    _app_server_unsupported: str | None = None

    async def _open_app_server(self, codex: Any, overrides: tuple[str, ...]) -> JsonRpcStdioClient:
        """A ready app-server for this attempt; failures raise into the send.

        A binary that is not the pinned app-server version fails the send,
        and is remembered for this process (the binary doesn't change under
        it). It is never run on exec instead: the app pinned this run's steer
        capabilities from the bot's ``app_server`` opt-in, and an exec run
        would be offered steers it can't take.
        """
        if self._app_server_unsupported:
            raise AppServerVersionMismatch(self._app_server_unsupported)
        from openai_codex_sdk.exec import CodexExecArgs

        executor = codex._exec
        try:
            return await spawn_app_server(
                executor.executable_path,
                overrides=overrides,
                env=executor._build_env(CodexExecArgs(input="")),
                cwd=None,
                timeout=self._request_timeout,
            )
        except AppServerVersionMismatch as exc:
            self._app_server_unsupported = (
                f"{exc}; codex_transport=app_server sends fail until the supported codex "
                "is installed or the bot is set to codex_transport=exec"
            )
            logger.error("%s", self._app_server_unsupported)
            raise AppServerVersionMismatch(self._app_server_unsupported) from exc

    async def _run_app_server_turn(
        self,
        client: JsonRpcStdioClient,
        *,
        prompt_input: Any,
        resume_id: str | None,
        model: str,
        cancel_event: asyncio.Event | None,
        request_id: str,
        session_key: str,
        bot_slug: str,
        seq: int,
        text_parts: list[str],
        item_text: dict[str, str],
        item_buffers: dict[str, dict[str, Any]],
        session_persisted: bool,
        thread_session_id: str | None,
        reset_evidence: Any,
    ) -> tuple[int, bool, bool, Any, Any]:
        """Run one turn on ``client`` and tear it down.

        Returns ``(seq, aborted, session_persisted, final_usage, thread)``,
        the values the exec path leaves for the send's shared completion.
        """
        run: CodexAppServerRun | None = None
        cancelled = False
        aborted = False
        final_usage: Any = None
        thread: Any = None
        try:
            options: dict[str, Any] = {
                "model": model, "approvalPolicy": "never", "sandbox": "danger-full-access",
            }
            if self._cwd:
                options["cwd"] = self._cwd
            if resume_id:
                opened = await client.request(
                    "thread/resume", {"threadId": resume_id, "excludeTurns": True, **options},
                    timeout=self._request_timeout,
                )
            else:
                opened = await client.request("thread/start", options, timeout=self._request_timeout)
            thread_id = str(((opened or {}).get("thread") or {}).get("id") or "")
            if not thread_id:
                raise RuntimeError("codex app-server opened a thread without an id")
            thread = SimpleNamespace(id=thread_id)
            seq, persisted, _ = await self._app_server_event(
                SimpleNamespace(type="thread.started", thread_id=thread_id),
                bot_slug=bot_slug, session_key=session_key, request_id=request_id, seq=seq,
                text_parts=text_parts, item_text=item_text, model=model, resume_id=resume_id,
                session_persisted=session_persisted, item_buffers=item_buffers, thread=thread,
                thread_session_id=thread_session_id, reset_evidence=reset_evidence,
            )
            session_persisted = session_persisted or persisted

            run = CodexAppServerRun(client, request_id=request_id, thread_id=thread_id)
            started = await client.request(
                "turn/start", {"threadId": thread_id, "input": app_server_input(prompt_input)},
                timeout=self._request_timeout,
            )
            turn_id = str(((started or {}).get("turn") or {}).get("id") or "")
            if not turn_id:
                raise RuntimeError("codex app-server started a turn without an id")
            run.set_turn(turn_id)
            if session_key:
                # The steer and stop target, from here until the turn is left.
                self._session_queue.set_active_client(session_key, run)

            adapter = AppServerEventAdapter()
            while True:
                note = await self._next_app_server_note(client, cancel_event)
                if note is _CANCELLED:
                    logger.info("chat.abort signalled, stopping codex app-server turn: session=%s request=%s",
                                session_key, request_id)
                    aborted = True
                    break
                if note is None:
                    if run.stopping:
                        aborted = True
                        break
                    tail = client.stderr_tail()
                    raise RuntimeError("codex app-server exited mid-turn" + (f": {tail}" if tail else ""))
                params = note.get("params") or {}
                method = note.get("method")
                if method == "turn/completed" and ((params.get("turn") or {}).get("id")) != run.turn_id:
                    continue
                if method in ("item/started", "item/completed"):
                    seq = self._note_steer_started(run, params.get("item"), session_key, seq)
                event = adapter.adapt(note)
                if event is None:
                    continue
                terminal = event.type in ("turn.completed", "turn.failed", ABORTED)
                if terminal and (run.stopping or event.type == ABORTED):
                    # Stopped: the app finalizes the turn as aborted, so no
                    # completion or failure is published for it.
                    aborted = True
                    break
                if event.type == "turn.failed":
                    message = str(event.error.get("message") or "")
                    if _is_auth_failure(RuntimeError(message)) or _is_codex_session_error(RuntimeError(message)):
                        # The send's own recovery: broker refresh, or one fresh thread.
                        raise RuntimeError(message)
                seq, persisted, usage = await self._app_server_event(
                    event, bot_slug=bot_slug, session_key=session_key, request_id=request_id,
                    seq=seq, text_parts=text_parts, item_text=item_text, model=model,
                    resume_id=resume_id, session_persisted=session_persisted,
                    item_buffers=item_buffers, thread=thread,
                    thread_session_id=thread_session_id, reset_evidence=reset_evidence,
                )
                session_persisted = session_persisted or persisted
                if usage is not None:
                    final_usage = usage
                if terminal:
                    break
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            if run is not None:
                if session_key and self._session_queue.get_active_client(session_key) is run:
                    self._session_queue.pop_active_client(session_key)
                seq = await self._settle_steers(run, session_key, seq, wait=not (cancelled or run.stopping))
            close_task = client.close()
            if run is not None:
                run.track_close(close_task)
            detail = await asyncio.shield(close_task)
            logger.info("codex app-server closed: request=%s %s", request_id, detail)
        return seq, aborted, session_persisted, final_usage, thread

    async def _app_server_event(self, event: Any, **kwargs: Any) -> tuple[int, bool, Any]:
        model = kwargs["model"]
        return await self._handle_event(event, actual_model_ref=[model], **kwargs)

    async def _next_app_server_note(self, client: JsonRpcStdioClient, cancel_event: asyncio.Event | None) -> Any:
        """The next notification, ``None`` once app-server is gone, or
        ``_CANCELLED`` when the send's cooperative cancel is raised.

        Raises ``TimeoutError`` after ``_request_timeout`` without one, as the
        exec event stream does, so the send's stuck-thread retry applies.
        """
        if cancel_event is not None and cancel_event.is_set():
            return _CANCELLED
        note_task = asyncio.ensure_future(client.next_notification(self._request_timeout))
        waiters: set[asyncio.Future] = {note_task}
        cancel_task = None
        if cancel_event is not None:
            cancel_task = asyncio.ensure_future(cancel_event.wait())
            waiters.add(cancel_task)
        try:
            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in waiters:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)
        if note_task in done:
            try:
                return note_task.result()
            except TimeoutError:
                raise TimeoutError(
                    f"No Codex events for {self._request_timeout}s — codex app-server may be hung"
                ) from None
        return _CANCELLED

    def _note_steer_started(self, run: CodexAppServerRun, item: Any, session_key: str, seq: int) -> int:
        if not isinstance(item, dict) or item.get("type") != "userMessage":
            return seq
        pending = run.pending_steers.get(str(item.get("clientId") or ""))
        if pending is None or pending.started:
            return seq
        pending.started = True
        return self._publish_steer_lifecycle(run, str(item["clientId"]), "started", session_key, seq)

    async def _settle_steers(self, run: CodexAppServerRun, session_key: str, seq: int, *, wait: bool) -> int:
        """Publish ``uncertain`` for every possibly delivered steer the turn never showed.

        That is one app-server accepted, one whose reply was lost, and one
        still in flight: a definite non-delivery has already left
        ``pending_steers``. On a normal end the in-flight steer RPCs are given
        their bounded answer first; a stopped or cancelled send doesn't wait.
        """
        unresolved = [p.resolved.wait() for p in run.pending_steers.values() if not p.resolved.is_set()]
        if wait and unresolved:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.gather(*unresolved), CONTROL_RPC_TIMEOUT_S)
        for message_id, pending in list(run.pending_steers.items()):
            if (pending.accepted or not pending.resolved.is_set()) and not pending.started:
                pending.started = True  # settled; publish once
                seq = self._publish_steer_lifecycle(run, message_id, LIFECYCLE_UNCERTAIN, session_key, seq)
        return seq

    def _publish_steer_lifecycle(self, run: CodexAppServerRun, message_id: str, state: str,
                                 session_key: str, seq: int) -> int:
        pending = run.pending_steers.get(message_id)
        seq += 1
        self._publish_event(
            run.request_id, session_key, seq, kind=AgentEventKind.STEER_LIFECYCLE,
            extra_raw={
                "message_id": message_id,
                "state": state,
                "origin": pending.origin if pending else None,
                "steer_rpc_id": pending.steer_rpc_id if pending else None,
                "run_request_id": run.request_id,
            },
        )
        return seq

    async def _handle_steer(self, fields: dict, msg_id: str, async_redis: Any) -> None:
        """Steer the session's live app-server turn (``priority=next`` only).

        Never takes the send lock: the send holds it for the whole run.
        """
        rpc_id = (fields.get("request_id") or "").strip()
        session_key = (fields.get("session_key") or "").strip()
        try:
            if not session_key:
                raise ValueError("chat.steer requires session_key")
            run = self._session_queue.get_active_client(session_key)
            if run is None:
                raise RuntimeError("no_active_run")
            active_request_id = getattr(run, "request_id", "")
            target_request_id = (fields.get("target_request_id") or "").strip()
            if target_request_id and target_request_id != active_request_id:
                raise RuntimeError("active_run_mismatch")
            if not isinstance(run, CodexAppServerRun):
                # An exec run: nothing can reach the model until it ends.
                raise RuntimeError(STEER_UNSUPPORTED_TRANSPORT)
            priority = (fields.get("priority") or "next").strip().lower()
            if priority != "next" or fields.get("steer_action") or fields.get("escalate") == "1":
                raise RuntimeError(STEER_PRIORITY_UNSUPPORTED)
            origin = (fields.get("origin") or "system").strip().lower()
            text = fields.get(FRAMED_MESSAGE_FIELD) or None
            if text is None:
                if origin != "system":
                    raise RuntimeError(STEER_UNFRAMED)
                text = (fields.get("message") or "").strip()
            if not text:
                raise ValueError("chat.steer requires message")
            # An unconfirmed steer is still ``ok``: it may be in the turn, so
            # the caller must not send it again. Its lifecycle settles it.
            detail = await run.steer(
                text, message_id=(fields.get("message_id") or "").strip() or None,
                origin=origin, steer_rpc_id=rpc_id or None,
            )
            logger.info("chat.steer %s: session=%s active_request=%s chars=%d",
                        detail, session_key, active_request_id or "?", len(text))
            if rpc_id:
                self._publisher.publish_rpc_result(rpc_id, {
                    "ok": True, "detail": detail, "active_request_id": active_request_id or None,
                    "can_escalate": False, "can_cancel": False,
                })
        except Exception as exc:
            logger.warning("chat.steer failed: session=%s request=%s error=%s", session_key, rpc_id, exc)
            if rpc_id:
                self._publisher.publish_rpc_result(rpc_id, {"ok": False, "error": str(exc)})
        finally:
            await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)

    async def _abort_request(self, session_key: str, request_id: str) -> tuple[AbortState, str]:
        """``chat.abort`` for one request: interrupt an app-server turn first.

        The interrupt gets :data:`ABORT_INTERRUPT_BOUND_S`. If the turn ends
        in time the send finishes on its own and nothing is killed; otherwise
        the send is cancelled and its teardown kills app-server. Either way the
        reply carries the teardown, within ``ABORT_TEARDOWN_BOUND_S``. Any
        other send (exec, queued, not started yet) is cancelled as before.
        """
        queue = self._session_queue
        run = queue.get_active_client(session_key)
        if not isinstance(run, CodexAppServerRun) or run.request_id != request_id:
            return await cancel_request_handler(queue)(session_key, request_id)
        deadline = asyncio.get_running_loop().time() + ABORT_TEARDOWN_BOUND_S
        queue.tombstone(session_key, request_id)
        reason = await run.stop(ABORT_INTERRUPT_BOUND_S)
        parts = [reason]
        if reason not in GRACEFUL_STOPS:
            queue.cancel_request(session_key, request_id)
            parts.append("task_cancelled")
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        parts.append(await run.wait_closed(remaining))
        detail = ",".join(parts)
        logger.info("chat.abort: session=%s request=%s detail=%s", session_key, request_id, detail)
        return AbortState.CANCELLED, detail


__all__ = ["ABORT_INTERRUPT_BOUND_S", "APP_SERVER_TRANSPORT", "CODEX_TRANSPORT_FIELD",
           "CodexAppServerMixin", "wants_app_server"]
