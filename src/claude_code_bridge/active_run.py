"""Live Claude SDK run control for mid-turn steering.

Two delivery priorities map onto the Claude Code CLI's command queue:

``now``
    SDK ``interrupt()`` + replacement query. Kills in-flight tools. Used for
    explicit user redirection.
``next``
    The message is written to the CLI's stdin with our uuid and
    ``priority: "next"``. The CLI folds it into the next model call once the
    running tools return; if the turn is already producing its final answer the
    CLI runs it as a follow-up turn instead. Nothing is interrupted. Used for
    bot-originated deliveries (TASK-934).

The CLI reports each uuid-tagged command's fate via ``command_lifecycle``
frames (queued -> started -> completed|cancelled|discarded|refused). The
Python SDK's parser drops that frame type, so :meth:`ClaudeActiveRun.messages`
reads the raw frame stream, tracks our commands, and withholds a ResultMessage
while an injected command has not started yet — the follow-up turn's own
result then ends the llm-bawt request.
"""

from __future__ import annotations

import asyncio
import logging
import uuid as uuid_mod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

STEER_PRIORITIES = frozenset({"now", "next"})

# Lifecycle states (Claude Code CLI ``command_lifecycle`` frames).
_UNSTARTED = frozenset({"sent", "queued"})
_ACCEPTED = frozenset({"queued", "started", "completed"})
_REJECTED = frozenset({"cancelled", "discarded", "refused"})

# The CLI's canned tool_result for tools killed by an interrupt. Recording it
# verbatim made steer interruptions look like user permission denials.
_CLI_REJECTION_PREFIX = "The user doesn't want to proceed with this tool use"
STEER_INTERRUPTED_TOOL_RESULT = (
    "Interrupted by mid-turn steering — this tool call was cancelled before it "
    "completed (not a permission denial)."
)


def _command_uuid(message_id: str | None) -> str:
    """Use the caller's message id when it is a UUID, else mint one."""
    if message_id:
        try:
            return str(uuid_mod.UUID(str(message_id)))
        except ValueError:
            pass
    return str(uuid_mod.uuid4())


def _sdk_query_handle(client: Any) -> Any | None:
    """Return the SDK's raw frame ``Query`` for a real ``ClaudeSDKClient``."""
    try:
        from claude_agent_sdk._internal.query import Query
    except Exception:  # pragma: no cover - SDK layout changed
        return None
    handle = getattr(client, "_query", None)
    return handle if isinstance(handle, Query) else None


@dataclass
class ClaudeActiveRun:
    """Own the control plane for one live ``ClaudeSDKClient`` run.

    ``now`` steering is an SDK interrupt followed by a new query on the same
    client. The CLI emits one ``error_during_execution`` ResultMessage for the
    interrupted request before it starts the replacement query; the single
    bridge response consumer must drain that boundary rather than finalizing
    the llm-bawt turn on it.
    """

    client: Any
    request_id: str
    _steer_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _expected_interrupt_results: int = 0
    _completed: asyncio.Event = field(default_factory=asyncio.Event)
    _steer_timeout_s: float = field(default=8.0, repr=False)
    # command_uuid -> last lifecycle state for ``next`` injections.
    _commands: dict[str, str] = field(default_factory=dict)
    _command_events: dict[str, asyncio.Event] = field(default_factory=dict)
    _user_commands: set[str] = field(default_factory=set)
    _user_prompts: dict[str, str] = field(default_factory=dict)
    _escalations: dict[str, str] = field(default_factory=dict)
    _escalated_pending: set[str] = field(default_factory=set)
    _cli_capabilities: set[str] = field(default_factory=set)
    on_user_lifecycle: Callable[[str, str], None] | None = None
    _deferred_result: Any = None
    _pending_user_result: Any = None
    lifecycle_supported: bool = False

    async def steer(
        self,
        message: str,
        *,
        priority: str = "now",
        message_id: str | None = None,
        origin: str = "system",
    ) -> None:
        """Deliver ``message`` into the live run at ``priority``."""
        text = message.strip()
        if not text:
            raise ValueError("Steering message must not be empty")
        if priority not in STEER_PRIORITIES:
            raise ValueError(f"Unknown steer priority: {priority!r}")
        if priority == "next":
            if origin == "user" and (not message_id or _command_uuid(message_id) != message_id):
                raise ValueError("User next steer requires a UUID message_id")
            await self._steer_next(text, message_id, origin=origin)
            return
        await self._steer_now(text)

    async def _steer_next(self, text: str, message_id: str | None, *, origin: str) -> None:
        """Queue ``text`` for the next tool boundary without interrupting."""
        if not self.lifecycle_supported:
            # Without lifecycle frames the consumer cannot tell a folded message
            # from one that needs a follow-up turn; refusing lets the durable
            # delivery retry instead of risking a silently dropped message.
            raise RuntimeError("next_unsupported")
        command_uuid = _command_uuid(message_id)
        prompt = (
            "[Mid-turn message — delivered at a tool boundary; your running "
            "work was not interrupted]\n"
            f"{text}\n\n"
            "Address it as needed, then continue the unfinished task."
        )
        async with self._steer_lock:
            if self._completed.is_set():
                raise RuntimeError("no_active_run")
            if command_uuid in self._escalations:
                raise RuntimeError("steer_already_escalated")
            state = self._commands.get(command_uuid)
            if state is None or state in _REJECTED:
                self._commands[command_uuid] = "sent"
                event = self._command_events[command_uuid] = asyncio.Event()
                if origin == "user":
                    self._user_commands.add(command_uuid)
                    self._user_prompts[command_uuid] = text

                async def _frames():
                    yield {
                        "type": "user",
                        "message": {"role": "user", "content": prompt},
                        "parent_tool_use_id": None,
                        "uuid": command_uuid,
                        "priority": "next",
                    }

                await asyncio.wait_for(
                    self.client.query(_frames()), timeout=self._steer_timeout_s
                )
            else:
                event = self._command_events[command_uuid]
            try:
                await asyncio.wait_for(event.wait(), timeout=self._steer_timeout_s)
            except asyncio.TimeoutError:
                # Written to stdin but not yet acknowledged. The CLI reads stdin
                # in order, and the consumer keeps the run open while the command
                # is unstarted, so treat it as accepted rather than re-sending.
                logger.warning(
                    "next steer not yet acknowledged by CLI: request=%s uuid=%s",
                    self.request_id,
                    command_uuid,
                )
                return
            final = self._commands.get(command_uuid)
            if final in _REJECTED:
                raise RuntimeError(f"steer_{final}")

    async def escalate(self, message_id: str) -> str:
        """Atomically replace one queued user steer with an interrupt-now turn.

        The CLI's ``interrupt(cancel_queued=True)`` sweeps even a message
        mid-fold. We only use it when this is the sole unstarted bridge command:
        the sweep is session-wide, not scoped to a uuid. An unconfirmed receipt
        is never replayed. Retried escalation requests cannot send a second
        replacement query, even when the first control request times out.
        """
        async with self._steer_lock:
            prior = self._escalations.get(message_id)
            if prior == "escalated":
                return prior
            if prior is not None:
                raise RuntimeError(prior)
            if self._completed.is_set():
                raise RuntimeError("no_active_run")
            if message_id not in self._user_prompts:
                raise RuntimeError("unknown_user_steer")
            if self._commands.get(message_id) not in _UNSTARTED:
                raise RuntimeError("steer_already_started")
            if any(
                uuid != message_id and state in _UNSTARTED
                for uuid, state in self._commands.items()
            ):
                raise RuntimeError("escalation_other_queued_commands")
            if "interrupt_cancel_queued_v1" not in self._cli_capabilities:
                raise RuntimeError("escalation_unsupported_cli")
            handle = _sdk_query_handle(self.client)
            if handle is None:
                raise RuntimeError("escalation_unsupported_sdk")

            # The old turn's result must not end the send handler while the
            # interrupt is in flight. This is a one-shot attempt: a timeout may
            # mean the CLI processed the request without delivering its receipt.
            self._escalations[message_id] = "escalation_uncertain"
            self._expected_interrupt_results += 1
            interrupted = False
            try:
                receipt = await asyncio.wait_for(
                    handle._send_control_request(
                        {"subtype": "interrupt", "cancel_queued": True},
                        timeout=self._steer_timeout_s,
                    ),
                    timeout=self._steer_timeout_s,
                )
                interrupted = True
                cancelled = receipt.get("cancelled")
                survivors = receipt.get("still_queued")
                if not isinstance(cancelled, list) or not isinstance(survivors, list):
                    raise RuntimeError("escalation_missing_receipt")
                if message_id not in cancelled:
                    # A command that already started can have its content in
                    # the interrupted turn. Never replay it speculatively.
                    raise RuntimeError("escalation_not_cancelled")
                if survivors:
                    # A hosted CLI can retain in-flight sends. Do not add an
                    # identical turn while their delivery remains uncertain.
                    raise RuntimeError("escalation_survivors")
                if self._completed.is_set():
                    raise RuntimeError("no_active_run")
                await asyncio.wait_for(
                    self.client.query(_replacement_prompt(self._user_prompts[message_id])),
                    timeout=self._steer_timeout_s,
                )
                self._escalations[message_id] = "escalated"
                self._escalated_pending.add(message_id)
                self._publish_user_lifecycle(message_id, "started")
                return "escalated"
            except BaseException:
                # If no replacement was accepted, do not wait indefinitely for
                # one after consuming the aborted turn's ResultMessage.
                self._expected_interrupt_results = max(0, self._expected_interrupt_results - 1)
                if interrupted:
                    try:
                        await self.client.disconnect()
                    except BaseException:
                        pass
                raise

    def note_lifecycle(self, command_uuid: str, state: str) -> None:
        """Record a CLI ``command_lifecycle`` frame for our injected commands."""
        if command_uuid not in self._commands:
            return
        previous = self._commands[command_uuid]
        if previous == state:
            return
        self._commands[command_uuid] = state
        if command_uuid in self._user_commands and not (
            state == "cancelled" and command_uuid in self._escalations
        ):
            self._publish_user_lifecycle(command_uuid, state)
        event = self._command_events.get(command_uuid)
        if event is not None and (state in _ACCEPTED or state in _REJECTED):
            event.set()

    def _publish_user_lifecycle(self, command_uuid: str, state: str) -> None:
        if self.on_user_lifecycle is not None:
            try:
                self.on_user_lifecycle(command_uuid, state)
            except Exception:
                logger.exception("Failed to publish user steer lifecycle for %s", command_uuid)

    def has_unstarted_command(self) -> bool:
        """True while an injected ``next`` command has not drained into a turn."""
        return any(state in _UNSTARTED for state in self._commands.values())

    async def messages(self) -> AsyncIterator[Any]:
        """Yield parsed SDK messages, handling ``next``-steer turn boundaries.

        Replaces ``client.receive_messages()`` for the single response consumer.
        """
        handle = _sdk_query_handle(self.client)
        if handle is None:
            self.lifecycle_supported = False
            async for msg in self.client.receive_messages():
                yield msg
            return

        from claude_agent_sdk import ResultMessage, UserMessage
        from claude_agent_sdk._internal.message_parser import parse_message

        self.lifecycle_supported = True
        async for data in handle.receive_messages():
            if isinstance(data, dict) and data.get("type") == "system" and data.get("subtype") == "init":
                capabilities = data.get("capabilities")
                self._cli_capabilities = set(capabilities) if isinstance(capabilities, list) else set()
            if isinstance(data, dict) and data.get("type") == "command_lifecycle":
                self.note_lifecycle(
                    str(data.get("command_uuid") or ""), str(data.get("state") or "")
                )
                deferred = self._deferred_result
                if deferred is not None and not self.has_unstarted_command():
                    self._deferred_result = None
                    if data.get("state") in _REJECTED:
                        # An escalation consumes the aborted turn's result;
                        # ordinary rejection releases the withheld result.
                        if self._expected_interrupt_results > 0:
                            self._expected_interrupt_results -= 1
                        else:
                            yield deferred
                    # Otherwise it started a follow-up turn whose own
                    # ResultMessage will end the run; drop the stale one.
                if self._pending_user_result is not None and not any(
                    self._commands.get(cmd) == "started" for cmd in self._user_commands
                ):
                    result = self._pending_user_result
                    self._pending_user_result = None
                    if self._expected_interrupt_results > 0:
                        self._expected_interrupt_results -= 1
                    else:
                        yield result
                continue
            msg = parse_message(data)
            if msg is None:
                continue
            if isinstance(msg, UserMessage) and self._expected_interrupt_results > 0:
                _relabel_interrupted_tool_results(msg)
            if isinstance(msg, ResultMessage) and self._expected_interrupt_results <= 0:
                if self.has_unstarted_command():
                    logger.info(
                        "Holding result for queued next-steer follow-up: request=%s",
                        self.request_id,
                    )
                    self._deferred_result = msg
                    continue
                if any(self._commands.get(cmd) == "started" for cmd in self._user_commands):
                    # A late follow-up can emit its ResultMessage before its
                    # completed lifecycle. The send handler stops at result, so
                    # retain it until the lifecycle is published on the run.
                    self._pending_user_result = msg
                    continue
            yield msg
        if self._pending_user_result is not None:
            yield self._pending_user_result
            self._pending_user_result = None

    async def _steer_now(self, text: str) -> None:
        """Interrupt current work and inject ``text`` into the same SDK run."""
        # ``interrupt()`` terminates the SDK's current task; ``query()`` starts a
        # replacement task on the same session. The session retains all prior
        # context, but a bare status question ("how's it going?") naturally gets
        # answered and then ends. Frame the replacement as an interruption of
        # unfinished work so it resumes by default. Explicit stop/pause/abandon
        # requests remain authoritative — the model must honor them instead.
        replacement_prompt = _replacement_prompt(text)

        async with self._steer_lock:
            if self._completed.is_set():
                raise RuntimeError("no_active_run")

            self._expected_interrupt_results += 1
            interrupted = False
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._steer_timeout_s
            interrupt_task = asyncio.create_task(self.client.interrupt())
            completed_task = asyncio.create_task(self._completed.wait())
            try:
                done, _pending = await asyncio.wait(
                    {interrupt_task, completed_task},
                    timeout=max(0.0, deadline - loop.time()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if completed_task in done:
                    raise RuntimeError("no_active_run")
                if interrupt_task not in done:
                    raise RuntimeError("steer_interrupt_timeout")
                await interrupt_task
                interrupted = True
                if self._completed.is_set():
                    raise RuntimeError("no_active_run")
                remaining_s = deadline - loop.time()
                if remaining_s <= 0:
                    raise RuntimeError("steer_interrupt_timeout")
                try:
                    await asyncio.wait_for(
                        self.client.query(replacement_prompt),
                        timeout=remaining_s,
                    )
                except asyncio.TimeoutError as exc:
                    raise RuntimeError("steer_query_timeout") from exc
            except BaseException:
                # If the replacement query was not accepted, do not hide the
                # interrupted result as though a continuation were coming. Once
                # interrupt succeeded the old task is already dead, so tear down
                # the client rather than leaving _handle_send parked forever.
                self._expected_interrupt_results = max(
                    0, self._expected_interrupt_results - 1
                )
                if interrupted:
                    try:
                        await self.client.disconnect()
                    except BaseException:
                        pass
                raise
            finally:
                for task in (interrupt_task, completed_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(
                    interrupt_task,
                    completed_task,
                    return_exceptions=True,
                )

    def mark_completed(self) -> None:
        """Signal that the response consumer reached the run's terminal edge."""
        self._completed.set()

    def consume_replaced_result(self, result_message: Any) -> bool:
        """Consume one terminal boundary superseded by a steering message.

        Normally the SDK labels it ``error_during_execution``. If the original
        request finishes in the narrow race between our counter increment and
        the interrupt control request, its success result is still the replaced
        boundary; the already-accepted correction owns the next result.
        """
        if self._expected_interrupt_results <= 0:
            for command_uuid in tuple(self._escalated_pending):
                self._publish_user_lifecycle(command_uuid, "completed")
                self._escalated_pending.remove(command_uuid)
            return False
        self._expected_interrupt_results -= 1
        return True

    async def disconnect(self) -> None:
        """Preserve SessionQueue's duck-typed abort-handle contract."""
        await self.client.disconnect()


def _replacement_prompt(text: str) -> str:
    return (
        "[Mid-turn user interruption]\n"
        f"{text}\n\n"
        "Respond to this interruption as needed, then continue the unfinished "
        "task from the interrupted turn, incorporating the user's new direction. "
        "Do not conclude the turn merely because this was a status question. "
        "If the user explicitly asks you to stop, pause, abandon, or only report "
        "status without continuing, honor that instruction instead."
    )


def _relabel_interrupted_tool_results(msg: Any) -> None:
    """Replace the CLI's canned rejection text on steer-killed tool results."""
    for block in getattr(msg, "content", None) or []:
        content = getattr(block, "content", None)
        if getattr(block, "tool_use_id", None) is None:
            continue
        if isinstance(content, str) and content.startswith(_CLI_REJECTION_PREFIX):
            block.content = STEER_INTERRUPTED_TOOL_RESULT
            block.is_error = True
        elif isinstance(content, list) and any(
            isinstance(part, dict)
            and str(part.get("text", "")).startswith(_CLI_REJECTION_PREFIX)
            for part in content
        ):
            block.content = STEER_INTERRUPTED_TOOL_RESULT
            block.is_error = True
