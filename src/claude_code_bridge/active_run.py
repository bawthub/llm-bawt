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

Every model-facing word is app-rendered prompt-registry text
(:mod:`agent_bridge.steer_contract`); this module only joins it.

Lifecycle publication (TASK-1300). Every held command, whatever its origin,
publishes its CLI lifecycle through ``on_command_lifecycle``. A rejection
carries a ``proof`` only when the bridge knows the command can never run:

``cli_terminal``
    The CLI's own cancelled/discarded/refused frame, with no started frame
    before it on the ordered stream.
``sweep_receipt``
    An escalation's interrupt receipt listed the command as cancelled, the
    ordered receipt barrier completed with no started frame for it, and no
    replacement query was ever written.

When the bridge cannot know, it publishes ``uncertain`` and never a
rejection. The app fences an uncertain command and never re-queues it.

Finishing (TASK-1344). Once :meth:`ClaudeActiveRun.messages` yields the
ResultMessage that ends the run, the send handler finalizes and tears down.
No steer can be served after that point: the CLI would run it as a follow-up
nobody renders. ``now`` and ``next`` steers are refused there with
:data:`STEER_RUN_FINISHING` before anything is written, so the caller sends the
message as a fresh turn. A command written earlier is settled from the CLI
frames the SDK reader buffered but the consumer never took; those are read
from a clone of the SDK's receive stream taken when the consumer stops, so
the teardown's ``close_receive_stream()`` does not discard them.
"""

from __future__ import annotations

import asyncio
import logging
import uuid as uuid_mod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import anyio

from agent_bridge.maintenance import is_compact_command
from agent_bridge.steer_contract import (
    LIFECYCLE_REJECTED,
    LIFECYCLE_STARTED,
    PROOF_CLI_TERMINAL,
    PROOF_SWEEP_RECEIPT,
    STEER_ITEMS_SLOT,
    STEER_RUN_FINISHING,
    STEER_UNFRAMED,
    render_interrupt,
)

from .mcp_reconnect import McpReconnectMonitor
from .run_teardown import RunTeardown

logger = logging.getLogger(__name__)

STEER_PRIORITIES = frozenset({"now", "next"})

# Lifecycle states (Claude Code CLI ``command_lifecycle`` frames).
_UNSTARTED = frozenset({"sent", "queued"})
_ACCEPTED = frozenset({"queued", "started", "completed"})
_REJECTED = LIFECYCLE_REJECTED
_STARTED = LIFECYCLE_STARTED

# ``proof`` values on a published lifecycle (see the module docstring); the
# two proven-not-run proofs are shared with the app (steer_contract).
PROOF_REPLAYED = "replayed"
PROOF_BARRIER_TIMEOUT = "barrier_timeout"
PROOF_MISSING_RECEIPT = "missing_receipt"
PROOF_STARTED_IN_SWEEP = "started_in_sweep"
PROOF_REPLACEMENT_WRITE_UNCERTAIN = "replacement_write_uncertain"
PROOF_TEARDOWN = "teardown"

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


def _nonhuman_result_origin(message: Any) -> str | None:
    """Return the SDK origin kind when a result belongs to an injected message.

    The SDK may drain an older background notification's result before it
    starts this run's own prompt. Only an explicit non-human origin is stale;
    the run's own result has no origin or ``kind=human``.
    """
    origin = getattr(message, "origin", None)
    kind = origin.get("kind") if isinstance(origin, dict) else None
    return kind if kind not in (None, "human") else None


def _buffered_frames(handle: Any) -> int | None:
    """Frames the SDK reader has queued but ``messages()`` has not taken yet.

    The SDK has one stdout reader. It awaits every non-control frame into this
    memory stream before it reads the next line, and a ``control_response``
    only wakes its waiter. So when a control receipt resolves, every frame the
    CLI wrote before it has either been taken or is counted here.
    """
    try:
        return int(handle._message_receive.statistics().current_buffer_used)
    except Exception:
        return None


def _clone_receive(handle: Any) -> Any | None:
    """A second reader on the SDK's frame buffer, or ``None``.

    ``disconnect()`` closes the SDK's receive side. anyio keeps a shared
    buffer readable while any clone is open, so frames the reader buffered
    before it was cancelled survive the teardown.
    """
    try:
        return handle._message_receive.clone()
    except Exception:
        return None


@dataclass(frozen=True)
class HeldCommand:
    """A ``next`` command written to the CLI, kept so a sweep can replay it."""

    uuid: str
    origin: str
    item: str
    steer_rpc_id: str | None = None


@dataclass
class _SweepWindow:
    """Frames for the escalation's eligible commands, held until it settles."""

    eligible: frozenset[str]
    buffered: list[tuple[str, str]] = field(default_factory=list)


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
    # Arm alongside the now-steer interrupt, before the CLI can emit its
    # result; the replacement query may not have returned yet (TASK-1879).
    _pending_now_interrupts: int = 0
    _injected_notification_active: bool = False
    _completed: asyncio.Event = field(default_factory=asyncio.Event)
    _steer_timeout_s: float = field(default=8.0, repr=False)
    # command_uuid -> last lifecycle state for ``next`` injections.
    _commands: dict[str, str] = field(default_factory=dict)
    _command_events: dict[str, asyncio.Event] = field(default_factory=dict)
    _user_commands: set[str] = field(default_factory=set)
    # Every ``next`` command, in send order (dict insertion order).
    _held: dict[str, HeldCommand] = field(default_factory=dict)
    # Permanent started evidence for the current send of each uuid.
    _started_ever: set[str] = field(default_factory=set)
    # Swept commands carried by an accepted replacement query.
    _replayed: set[str] = field(default_factory=set)
    # Commands the bridge itself settled (cancelled/uncertain); later CLI
    # frames for them are not republished.
    _settled: dict[str, str] = field(default_factory=dict)
    _sweep: _SweepWindow | None = None
    _frames_taken: int = 0
    _frame_progress: asyncio.Event = field(default_factory=asyncio.Event)
    _escalations: dict[str, str] = field(default_factory=dict)
    _cancellations: dict[str, str] = field(default_factory=dict)
    # Replayed commands awaiting the replacement's completion, in send order.
    _escalated_pending: dict[str, None] = field(default_factory=dict)
    _cli_capabilities: set[str] = field(default_factory=set)
    on_command_lifecycle: Callable[[dict[str, Any]], None] | None = None
    _deferred_result: Any = None
    _pending_user_result: Any = None
    lifecycle_supported: bool = False
    maintenance: bool = False
    # Once steering may have reached the CLI, replaying the original prompt
    # loses user input even if no model text/tool block ever arrived.
    steering_attempted: bool = False
    # The run's one bounded teardown, shared with the send (TASK-1232 S1).
    teardown: RunTeardown | None = None
    # Set when ``messages()`` yields the run's terminal ResultMessage
    # (TASK-1344); steers are refused from then on.
    _finishing: bool = False
    # Clone of the SDK frame buffer, taken when the consumer stops.
    _teardown_frames: Any = None
    _mcp_monitor: McpReconnectMonitor | None = field(default=None, repr=False)

    async def steer(
        self,
        message: str,
        *,
        priority: str = "now",
        message_id: str | None = None,
        origin: str = "system",
        framed_message: str | None = None,
        interrupt_item: str | None = None,
        steer_rpc_id: str | None = None,
    ) -> None:
        """Deliver ``message`` into the live run at ``priority``.

        ``framed_message`` is the exact model-facing text and
        ``interrupt_item`` is this message's replay item. Both are rendered by
        the app (:mod:`agent_bridge.steer_contract`). A ``system`` steer is
        already final registry text and goes verbatim.
        """
        # The app retains these as durable, separate turns. Never inject them
        # into compaction or reinterpret a new command as ordinary steering.
        if self.maintenance:
            raise RuntimeError("maintenance_active")
        if is_compact_command(message):
            raise RuntimeError("maintenance_command_requires_turn")
        text = message.strip()
        if not text:
            raise ValueError("Steering message must not be empty")
        if priority not in STEER_PRIORITIES:
            raise ValueError(f"Unknown steer priority: {priority!r}")
        if origin == "system":
            framed = framed_message if (framed_message or "").strip() else text
            item = interrupt_item if (interrupt_item or "").strip() else text
        else:
            framed = framed_message or ""
            item = interrupt_item or ""
            if not framed.strip() or (priority == "next" and not item.strip()):
                raise RuntimeError(STEER_UNFRAMED)
        if priority == "next":
            if origin == "user" and (not message_id or _command_uuid(message_id) != message_id):
                raise ValueError("User next steer requires a UUID message_id")
            await self._steer_next(
                framed, item, message_id, origin=origin, steer_rpc_id=steer_rpc_id,
            )
            return
        await self._steer_now(framed)

    async def _steer_next(
        self,
        framed: str,
        item: str,
        message_id: str | None,
        *,
        origin: str,
        steer_rpc_id: str | None,
    ) -> None:
        """Queue ``framed`` for the next tool boundary without interrupting."""
        if not self.lifecycle_supported:
            # Without lifecycle frames the consumer cannot tell a folded message
            # from one that needs a follow-up turn; refusing lets the durable
            # delivery retry instead of risking a silently dropped message.
            raise RuntimeError("next_unsupported")
        command_uuid = _command_uuid(message_id)
        async with self._steer_lock:
            if self._completed.is_set():
                raise RuntimeError("no_active_run")
            if command_uuid in self._escalations:
                raise RuntimeError("steer_already_escalated")
            if command_uuid in self._cancellations:
                raise RuntimeError("steer_already_cancelled")
            state = self._commands.get(command_uuid)
            if state is None or state in _REJECTED:
                # Only a new CLI command is refused here: a replay of one
                # already accepted keeps its answer, so it can never become a
                # second, fresh-turn delivery.
                self._refuse_new_command()
                self._commands[command_uuid] = "sent"
                event = self._command_events[command_uuid] = asyncio.Event()
                # A re-send is a new CLI command: drop the old send's evidence.
                self._held.pop(command_uuid, None)
                self._held[command_uuid] = HeldCommand(
                    uuid=command_uuid, origin=origin, item=item, steer_rpc_id=steer_rpc_id,
                )
                self._started_ever.discard(command_uuid)
                self._replayed.discard(command_uuid)
                self._settled.pop(command_uuid, None)
                if origin == "user":
                    self._user_commands.add(command_uuid)

                async def _frames():
                    yield {
                        "type": "user",
                        "message": {"role": "user", "content": framed},
                        "parent_tool_use_id": None,
                        "uuid": command_uuid,
                        "priority": "next",
                    }

                self.steering_attempted = True
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
                    "next steer not yet acknowledged by CLI: request=%s uuid=%s origin=%s",
                    self.request_id,
                    command_uuid,
                    origin,
                )
                return
            final = self._commands.get(command_uuid)
            if final in _REJECTED:
                raise RuntimeError(f"steer_{final}")

    async def cancel_queued(self, message_id: str) -> str:
        """Withdraw one user message without interrupting the running tool.

        ``cancel_async_message`` is per UUID and returns false if the CLI has
        already folded the message into a turn. Never report success without
        its positive receipt, and never turn a timeout into a speculative retry.
        """
        async with self._steer_lock:
            prior = self._cancellations.get(message_id)
            if prior == "cancelled":
                return prior
            if prior is not None:
                raise RuntimeError(prior)
            if self._completed.is_set():
                raise RuntimeError("no_active_run")
            if message_id not in self._user_commands:
                raise RuntimeError("unknown_user_steer")
            if message_id in self._escalations:
                raise RuntimeError("steer_already_escalated")
            if self._commands.get(message_id) not in _UNSTARTED:
                raise RuntimeError("steer_already_started")
            handle = _sdk_query_handle(self.client)
            if handle is None:
                raise RuntimeError("cancel_unsupported_sdk")
            self._cancellations[message_id] = "cancellation_uncertain"
            receipt = await asyncio.wait_for(
                handle._send_control_request(
                    {"subtype": "cancel_async_message", "message_uuid": message_id},
                    timeout=self._steer_timeout_s,
                ),
                timeout=self._steer_timeout_s,
            )
            if receipt.get("cancelled") is not True:
                # A false receipt means it was no longer removable. Its
                # lifecycle frame will reveal whether it started or ended.
                self._cancellations.pop(message_id)
                raise RuntimeError("steer_already_started")
            self._cancellations[message_id] = "cancelled"
            self.note_lifecycle(message_id, "cancelled")
            return "cancelled"

    async def escalate(self, message_id: str, *, interrupt_frame: str | None = None) -> str:
        """Interrupt now, replaying every queued command the sweep removed.

        ``interrupt(cancel_queued=True)`` sweeps the CLI's whole queue, not one
        uuid, so one replacement query carries every swept command we hold, in
        send order (TASK-1300). The rules that keep this exactly-once:

        * eligible commands are frozen before the interrupt;
        * the replay set is computed only after the ordered receipt barrier,
          and never includes a command with started evidence;
        * a failure before any replacement write releases the swept commands
          as provably cancelled only when the barrier completed and none
          started; every other failure publishes them ``uncertain``;
        * an unconfirmed outcome is never retried.
        """
        async with self._steer_lock:
            prior = self._escalations.get(message_id)
            if prior == "escalated":
                return prior
            if prior is not None:
                raise RuntimeError(prior)
            if self._completed.is_set():
                raise RuntimeError("no_active_run")
            held = self._held.get(message_id)
            # TASK-1748: a system steer is a quiesce control row; bot steers
            # never interrupt a running tool (TASK-934).
            if held is None or held.origin not in ("user", "system"):
                raise RuntimeError("unknown_user_steer")
            if self._commands.get(message_id) not in _UNSTARTED or message_id in self._started_ever:
                raise RuntimeError("steer_already_started")
            if not interrupt_frame or STEER_ITEMS_SLOT not in interrupt_frame:
                raise RuntimeError(STEER_UNFRAMED)
            if "interrupt_cancel_queued_v1" not in self._cli_capabilities:
                raise RuntimeError("escalation_unsupported_cli")
            handle = _sdk_query_handle(self.client)
            if handle is None or _buffered_frames(handle) is None:
                raise RuntimeError("escalation_unsupported_sdk")

            eligible = tuple(
                uuid for uuid in self._held
                if self._commands.get(uuid) in _UNSTARTED
                and uuid not in self._started_ever
                and uuid not in self._settled
                and uuid not in self._replayed
            )
            if held.origin == "system" and eligible != (message_id,):
                # TASK-1748: the sweep takes every queued command and marks it
                # started; a quiesce that then ends the turn would drop them.
                # Refuse before the interrupt, so nothing is swept.
                raise RuntimeError("escalation_other_queued")
            window = _SweepWindow(eligible=frozenset(eligible))
            self._sweep = window
            # The old turn's result must not end the send handler while the
            # interrupt is in flight. This is a one-shot attempt: a timeout may
            # mean the CLI processed the request without delivering its receipt.
            self._escalations[message_id] = "escalation_uncertain"
            self.steering_attempted = True
            self._expected_interrupt_results += 1
            interrupted = False
            barrier_done = False
            replacement_attempted = False
            receipt_valid = False
            swept: list[str] = []
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
                receipt_valid = True
                swept_ids = {str(uuid) for uuid in cancelled}
                swept = [uuid for uuid in eligible if uuid in swept_ids]
                foreign = sorted(swept_ids - set(self._held))
                logger.info(
                    "escalation receipt: request=%s clicked=%s cancelled=%s still_queued=%s",
                    self.request_id, message_id,
                    self._describe(swept), [str(uuid) for uuid in survivors],
                )
                if foreign:
                    logger.warning(
                        "escalation swept commands the bridge does not hold: request=%s uuids=%s",
                        self.request_id, foreign,
                    )
                await self._await_frames(self._frames_taken + (_buffered_frames(handle) or 0))
                barrier_done = True
                started = [uuid for uuid in swept if uuid in self._started_ever]
                if started:
                    # The receipt says these were still queued, but the ordered
                    # stream says they started: never replay or release them.
                    raise RuntimeError(f"escalation_started_in_sweep: {self._describe(started)}")
                if message_id not in swept_ids:
                    # A command that already started can have its content in
                    # the interrupted turn. Never replay it speculatively.
                    raise RuntimeError("escalation_not_cancelled")
                if held.origin == "system" and swept_ids != {message_id}:
                    # TASK-1748 (impl r1 B1): the CLI swept a command beyond the
                    # quiesce control (one the bridge doesn't hold, or a late
                    # arrival). Replaying only the control would silently drop
                    # it, so the escalation fails visibly instead of succeeding.
                    raise RuntimeError(
                        "escalation_other_cancelled: "
                        + self._describe(sorted(swept_ids - {message_id}))
                    )
                if survivors:
                    # A hosted CLI can retain in-flight sends. Do not add an
                    # identical turn while their delivery remains uncertain.
                    raise RuntimeError(
                        "escalation_survivors: "
                        + self._describe([str(uuid) for uuid in survivors])
                    )
                if self._completed.is_set():
                    raise RuntimeError("no_active_run")
                prompt = render_interrupt(
                    interrupt_frame, [self._held[uuid].item for uuid in swept]
                )
                # Set before the write: ``client.query`` only awaits the
                # transport write, so once it is attempted the CLI may have the
                # text even if the await fails.
                replacement_attempted = True
                await asyncio.wait_for(self.client.query(prompt), timeout=self._steer_timeout_s)
                self._replayed.update(swept)
                self._escalations[message_id] = "escalated"
                self._escalated_pending.update(dict.fromkeys(swept))
                for uuid in swept:
                    self._publish(uuid, "started", proof=PROOF_REPLAYED)
                logger.info(
                    "escalation replayed: request=%s commands=%s",
                    self.request_id, self._describe(swept),
                )
                return "escalated"
            except BaseException as exc:
                # If no replacement was accepted, do not wait indefinitely for
                # one after consuming the aborted turn's ResultMessage.
                self._expected_interrupt_results = max(0, self._expected_interrupt_results - 1)
                if interrupted:
                    self._settle_failed_sweep(
                        exc,
                        eligible=eligible,
                        swept=swept,
                        receipt_valid=receipt_valid,
                        barrier_done=barrier_done,
                        replacement_attempted=replacement_attempted,
                    )
                    try:
                        await self._tear_down()
                    except BaseException:
                        pass
                raise
            finally:
                self._close_sweep(window)

    def _settle_failed_sweep(
        self,
        exc: BaseException,
        *,
        eligible: tuple[str, ...],
        swept: list[str],
        receipt_valid: bool,
        barrier_done: bool,
        replacement_attempted: bool,
    ) -> None:
        """Publish the fate of commands a failed escalation may have swept.

        The run is torn down next, so anything the CLI swept will not run
        unless the replacement carried it. A command is released as provably
        cancelled only when all four hold: the barrier completed, the receipt
        is valid and lists it, it has no started evidence, and no replacement
        write was attempted. Anything else is ``uncertain``.
        """
        if replacement_attempted:
            targets, state, proof = swept, "uncertain", PROOF_REPLACEMENT_WRITE_UNCERTAIN
        elif not receipt_valid:
            targets, state, proof = list(eligible), "uncertain", PROOF_MISSING_RECEIPT
        elif not barrier_done:
            targets, state, proof = swept, "uncertain", PROOF_BARRIER_TIMEOUT
        elif any(uuid in self._started_ever for uuid in swept):
            targets, state, proof = swept, "uncertain", PROOF_STARTED_IN_SWEEP
        else:
            targets, state, proof = swept, "cancelled", PROOF_SWEEP_RECEIPT
        targets = [uuid for uuid in targets if uuid not in self._started_ever]
        log = logger.info if state == "cancelled" else logger.warning
        log(
            "escalation failed after interrupt: request=%s error=%s %s=%s proof=%s",
            self.request_id, exc, state, self._describe(targets), proof,
        )
        for uuid in targets:
            self._settled[uuid] = state
            self._publish(uuid, state, proof=proof)

    def _close_sweep(self, window: _SweepWindow) -> None:
        """Publish the frames held during the escalation, in arrival order."""
        if self._sweep is window:
            self._sweep = None
        for uuid, state in window.buffered:
            self._publish_frame(uuid, state)

    async def _await_frames(self, target: int) -> None:
        """Wait until ``messages()`` has taken ``target`` frames (the barrier)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._steer_timeout_s
        while self._frames_taken < target:
            self._frame_progress.clear()
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise RuntimeError("escalation_barrier_timeout")
            try:
                await asyncio.wait_for(self._frame_progress.wait(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise RuntimeError("escalation_barrier_timeout") from exc

    def release_unstarted(self, outcome: Any = None) -> list[str]:
        """Publish ``uncertain`` for held commands a finished run never settled.

        Runs once the run's teardown has finished. Frames the SDK reader had
        already buffered are applied first (TASK-1344), so a command the CLI
        started is published started, not uncertain. A confirmed CLI exit does
        not prove a command never started (its frame may be unread), so this
        never publishes a rejection without the CLI's own frame.
        """
        self._reconcile_buffered_frames()
        released: list[str] = []
        for uuid in self._held:
            if (
                uuid in self._started_ever
                or uuid in self._settled
                or uuid in self._replayed
                or self._commands.get(uuid) not in _UNSTARTED
            ):
                continue
            self._settled[uuid] = "uncertain"
            released.append(uuid)
            self._publish(
                uuid, "uncertain", proof=PROOF_TEARDOWN,
                exited=getattr(outcome, "exited", None),
            )
        if released:
            logger.warning(
                "run ended with unsettled steers: request=%s uncertain=%s",
                self.request_id, self._describe(released),
            )
        return released

    def _reconcile_buffered_frames(self) -> None:
        """Apply the lifecycle frames the consumer never took, then drop the clone.

        Only ``command_lifecycle`` frames matter here: the run is over, so
        nothing else in the buffer would be rendered.
        """
        stream, self._teardown_frames = self._teardown_frames, None
        if stream is None:
            return
        applied = 0
        try:
            while True:
                try:
                    data = stream.receive_nowait()
                except (anyio.WouldBlock, anyio.EndOfStream, anyio.ClosedResourceError):
                    break
                if isinstance(data, dict) and data.get("type") == "command_lifecycle":
                    applied += 1
                    self.note_lifecycle(
                        str(data.get("command_uuid") or ""), str(data.get("state") or "")
                    )
        finally:
            stream.close()
        if applied:
            logger.info(
                "reconciled buffered lifecycle frames at teardown: request=%s frames=%d",
                self.request_id, applied,
            )

    def _refuse_new_command(self) -> None:
        """Refuse a new CLI write the run can no longer serve; call under the steer lock."""
        if self._completed.is_set():
            raise RuntimeError("no_active_run")
        if self._finishing:
            raise RuntimeError(STEER_RUN_FINISHING)

    def _finish(self, result: Any) -> Any:
        """Mark ``result`` as the run's terminal result and return it."""
        self._finishing = True
        return result

    def _injected_interrupt_boundary(self, result: Any, origin_kind: str) -> bool:
        """Recognize a CLI-interrupted background notification, not a stale completion.

        The installed CLI can interrupt an injected notification after a host
        query is queued. Its non-human error result is the now-steer's missing
        boundary; the subsequent human result must finish the replacement.
        Require the notification's system frame and a pending now-steer intent.
        With missing provenance or multiple interrupts the result is not
        attributable, so fail visibly rather than guess which one it consumed.
        """
        if getattr(result, "subtype", None) != "error_during_execution" or not self._pending_now_interrupts:
            return False
        if origin_kind != "task-notification":
            raise RuntimeError("unsupported_injected_interrupt_origin")
        if not self._injected_notification_active:
            raise RuntimeError("unattributed_injected_interrupt_boundary")
        if self._expected_interrupt_results != 1 or self._pending_now_interrupts != 1:
            raise RuntimeError("ambiguous_injected_interrupt_boundary")
        logger.info("Draining interrupted injected notification: request=%s", self.request_id)
        return True

    def note_lifecycle(self, command_uuid: str, state: str) -> None:
        """Record a CLI ``command_lifecycle`` frame for our injected commands."""
        if command_uuid not in self._commands:
            logger.debug(
                "command_lifecycle for unknown uuid: request=%s uuid=%s state=%s",
                self.request_id, command_uuid, state,
            )
            return
        previous = self._commands[command_uuid]
        if previous == state:
            return
        self._commands[command_uuid] = state
        held = self._held.get(command_uuid)
        logger.info(
            "command_lifecycle: request=%s uuid=%s origin=%s %s->%s",
            self.request_id, command_uuid, held.origin if held else "?", previous, state,
        )
        if state in _STARTED:
            self._started_ever.add(command_uuid)
        window = self._sweep
        if window is not None and command_uuid in window.eligible:
            window.buffered.append((command_uuid, state))
        else:
            self._publish_frame(command_uuid, state)
        event = self._command_events.get(command_uuid)
        if event is not None and (state in _ACCEPTED or state in _REJECTED):
            event.set()

    def _publish_frame(self, command_uuid: str, state: str) -> None:
        """Publish one CLI lifecycle state unless the bridge already settled it."""
        if command_uuid in self._replayed or command_uuid in self._settled:
            return
        proof = None
        if state in _REJECTED and command_uuid not in self._started_ever:
            proof = PROOF_CLI_TERMINAL
        self._publish(command_uuid, state, proof=proof)

    def _publish(self, command_uuid: str, state: str, *, proof: str | None = None, **extra: Any) -> None:
        if self.on_command_lifecycle is None:
            return
        held = self._held.get(command_uuid)
        payload: dict[str, Any] = {
            "message_id": command_uuid,
            "state": state,
            "origin": held.origin if held else None,
            "steer_rpc_id": held.steer_rpc_id if held else None,
            "run_request_id": self.request_id,
        }
        if proof is not None:
            payload["proof"] = proof
        payload.update(extra)
        try:
            self.on_command_lifecycle(payload)
        except Exception:
            logger.exception("Failed to publish steer lifecycle for %s", command_uuid)

    def _describe(self, uuids: list[str]) -> str:
        """``uuid(origin), …`` for logs and refusal details."""
        parts = []
        for uuid in uuids:
            held = self._held.get(uuid)
            parts.append(f"{uuid}({held.origin if held else 'unknown'})")
        return ", ".join(parts) or "none"

    def has_unstarted_command(self) -> bool:
        """True while an injected ``next`` command has not drained into a turn."""
        return any(state in _UNSTARTED for state in self._commands.values())

    async def messages(self) -> AsyncIterator[Any]:
        """Yield parsed SDK messages, handling ``next``-steer turn boundaries.

        Replaces ``client.receive_messages()`` for the single response consumer.
        Every ResultMessage it yields while no interrupt boundary is expected
        ends the run, so steering closes as it is yielded (TASK-1344).
        """
        from claude_agent_sdk import ResultMessage, UserMessage
        from claude_agent_sdk.types import TaskNotificationMessage

        if self._mcp_monitor is None:
            self._mcp_monitor = McpReconnectMonitor(self.client)
            self._mcp_monitor.start()
        handle = _sdk_query_handle(self.client)
        if handle is None:
            self.lifecycle_supported = False
            async for msg in self.client.receive_messages():
                if isinstance(msg, TaskNotificationMessage):
                    self._injected_notification_active = True
                if isinstance(msg, ResultMessage):
                    stale_kind = _nonhuman_result_origin(msg)
                    if stale_kind is not None:
                        boundary = self._injected_interrupt_boundary(msg, stale_kind)
                        if stale_kind == "task-notification":
                            self._injected_notification_active = False
                        if boundary:
                            yield msg  # the response consumer drains this boundary
                        else:
                            logger.info(
                                "Ignoring injected SDK result: request=%s origin=%s",
                                self.request_id, stale_kind,
                            )
                        continue
                    if self._expected_interrupt_results <= 0:
                        msg = self._finish(msg)
                yield msg
            return

        from claude_agent_sdk._internal.message_parser import parse_message

        self.lifecycle_supported = True
        async for data in handle.receive_messages():
            # Counted before anything awaits: the escalation barrier relies on
            # a taken lifecycle frame having been recorded already.
            self._frames_taken += 1
            self._frame_progress.set()
            if isinstance(data, dict) and data.get("type") == "system":
                if data.get("subtype") == "init":
                    capabilities = data.get("capabilities")
                    self._cli_capabilities = set(capabilities) if isinstance(capabilities, list) else set()
                elif data.get("subtype") == "task_notification":
                    self._injected_notification_active = True
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
                            yield self._finish(deferred)
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
                        yield self._finish(result)
                continue
            msg = parse_message(data)
            if msg is None:
                continue
            if isinstance(msg, ResultMessage):
                stale_kind = _nonhuman_result_origin(msg)
                if stale_kind is not None:
                    boundary = self._injected_interrupt_boundary(msg, stale_kind)
                    if stale_kind == "task-notification":
                        self._injected_notification_active = False
                    if boundary:
                        yield msg  # the response consumer drains this boundary
                    else:
                        logger.info(
                            "Ignoring injected SDK result: request=%s origin=%s",
                            self.request_id, stale_kind,
                        )
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
                msg = self._finish(msg)
            yield msg
        if self._pending_user_result is not None:
            result, self._pending_user_result = self._pending_user_result, None
            yield self._finish(result)

    async def _steer_now(self, replacement_prompt: str) -> None:
        """Interrupt current work and inject ``replacement_prompt`` into the run.

        ``interrupt()`` terminates the SDK's current task; ``query()`` starts a
        replacement task on the same session. The app-rendered prompt
        (``chat.steer.interrupt_replacement``) frames it as an interruption of
        unfinished work, so the run resumes by default.
        """
        async with self._steer_lock:
            self._refuse_new_command()

            self.steering_attempted = True
            self._expected_interrupt_results += 1
            self._pending_now_interrupts += 1
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
                self._pending_now_interrupts = max(0, self._pending_now_interrupts - 1)
                if interrupted:
                    try:
                        await self._tear_down()
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
        """Signal that the response consumer reached the run's terminal edge.

        The consumer has stopped reading, so this is where the frames it never
        took start to matter: keep a reader on the SDK buffer for
        :meth:`release_unstarted`. A run torn down earlier has nothing left
        to clone.
        """
        self._completed.set()
        if self._mcp_monitor is not None:
            self._mcp_monitor.stop()
        if self._teardown_frames is None:
            handle = _sdk_query_handle(self.client)
            if handle is not None:
                self._teardown_frames = _clone_receive(handle)

    def consume_replaced_result(self, result_message: Any) -> bool:
        """Consume one terminal boundary superseded by a steering message.

        Normally the SDK labels it ``error_during_execution``. If the original
        request finishes in the narrow race between our counter increment and
        the interrupt control request, its success result is still the replaced
        boundary; the already-accepted correction owns the next result.
        """
        if self._expected_interrupt_results <= 0:
            for command_uuid in tuple(self._escalated_pending):
                self._publish(command_uuid, "completed", proof=PROOF_REPLAYED)
                del self._escalated_pending[command_uuid]
            return False
        self._expected_interrupt_results -= 1
        if self._pending_now_interrupts:
            self._pending_now_interrupts -= 1
        return True

    async def _tear_down(self) -> None:
        """End a run whose interrupted task can't continue.

        Goes through the run's one bounded ``RunTeardown`` (TASK-1232 S1): a
        direct ``disconnect()`` would null the transport, so the send's own
        teardown could no longer see the CLI pid or kill its descendants.
        Shielded, so a cancelled steer can't cut the teardown short.
        """
        if self.teardown is None:
            self.teardown = RunTeardown(self.client)
        await asyncio.shield(self.teardown.start())


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
