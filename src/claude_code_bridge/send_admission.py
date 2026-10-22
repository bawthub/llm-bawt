"""Request-scope admission for one Claude bridge send (TASK-1232 P2).

A ``chat.send`` registers its task with the session queue at receipt,
synchronously and before any await, so a ``chat.abort`` for the same request
either cancels it or, if the abort was read first, leaves a tombstone that
refuses the send here. Nothing a send does (``/new`` reset, seed, the run
itself) happens outside that registration. Exactly one terminal event is
published per request:

* refused (tombstoned): the cancelled terminal, published here;
* Stopped during preflight: the cancelled terminal, published here, unless
  preflight already published its own terminal;
* Stopped or finished after preflight: :meth:`_run_send` owns the terminal,
  as before.

Run contract (TASK-1233): an accepted send publishes ``RUN_ACCEPTED`` here,
and its one ``done`` sentinel is published here too, through a
:class:`~claude_code_bridge.run_terminal.RunTerminalLatch`, after
:meth:`_run_send` (including its run teardown) has finished. Only a host
shutdown publishes earlier, from inside :meth:`_run_send`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from agent_bridge.abort_contract import aborted_terminal_raw
from agent_bridge.events import AgentEventKind
from agent_bridge.publisher import COMMANDS_STREAM
from agent_bridge.run_contract import UNATTEMPTED
from agent_bridge.session_queue import Admission

from .run_terminal import RunTerminalLatch
from .send_preflight import prepare_parsed_send
from .send_request import SendRequest

logger = logging.getLogger("claude_code_bridge.bridge")


@dataclass(slots=True)
class PreflightScope:
    """Whether preflight already published this request's terminal events."""

    terminal_published: bool = False


class ClaudeSendAdmissionMixin:
    """Owns ``_handle_send``: admission, preflight, then :meth:`_run_send`."""

    async def _handle_send(self, fields: dict, msg_id: str, async_redis) -> None:
        request = SendRequest.from_fields(fields)
        if not request.request_id or not request.message:
            logger.warning("Invalid send command: missing request_id or message")
            await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
            return
        session_key, request_id = request.session_key, request.request_id
        # No await before accept(): see the module docstring.
        admission = self._session_queue.accept(session_key, request_id)
        if admission is Admission.TOMBSTONED:
            logger.info(
                "Refusing stopped send: request_id=%s session=%s", request_id, session_key,
            )
            if request.trigger_message_id:
                # Stamp the terminal like any other; popped when it publishes.
                self._trigger_message_ids[request_id] = request.trigger_message_id
            await self._publish_request_cancelled(
                request_id, session_key, 1, msg_id, async_redis,
            )
            self._publisher.publish_run_done(request_id, teardown=UNATTEMPTED)
            return
        if admission is Admission.DUPLICATE:
            logger.warning(
                "Dropping duplicate send for a live request: request_id=%s session=%s",
                request_id, session_key,
            )
            await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
            return
        self._publisher.publish_run_accepted(request_id, session_key)
        terminal = RunTerminalLatch(self._publisher, request_id, owner=asyncio.current_task())
        self._run_terminals[request_id] = terminal
        cancelled = False
        try:
            scope = PreflightScope()
            try:
                preflight = await prepare_parsed_send(
                    self, request, msg_id, async_redis, scope=scope,
                )
            except asyncio.CancelledError:
                if not scope.terminal_published:
                    await self._publish_request_cancelled(
                        request_id, session_key, 1, msg_id, async_redis,
                    )
                raise
            if preflight is None:
                return
            await self._run_send(preflight, msg_id, async_redis, terminal)
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            report = terminal.report(
                phase=self._session_queue.request_phase(session_key, request_id),
                cancelled=cancelled,
            )
            self._session_queue.release(session_key, request_id)
            terminal.publish(report)
            if self._run_terminals.get(request_id) is terminal:
                del self._run_terminals[request_id]

    async def _publish_request_cancelled(
        self, request_id: str, session_key: str, seq: int, msg_id: str, async_redis,
    ) -> None:
        """Publish the cancelled terminal event, clean up, and xack.

        The ``done`` sentinel is ``_handle_send``'s (TASK-1233).
        """
        self._publish_event(
            request_id, session_key, seq,
            kind=AgentEventKind.ASSISTANT_DONE, text="",
            # TASK-1231B: a host-shutdown cancel is stamped so the app can
            # retry it; an abort/cancel keeps the historical shape.
            extra_raw=aborted_terminal_raw(
                self._session_queue.cancel_cause(session_key, request_id)
            ),
        )
        self._discard_changed_file_request(request_id)
        self._trigger_message_ids.pop(request_id, None)
        if request_id not in self._proxy_request_sessions:
            self._clear_proxy_sampling_cancel(request_id)
        self._proxy_request_sessions.pop(request_id, None)
        await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)


__all__ = ["ClaudeSendAdmissionMixin", "PreflightScope"]
