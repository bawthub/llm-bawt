"""One fenced recovery attempt: durable evidence → observe → commit → project.

No runtime scheduling/admission is enabled by this module. Never starts a model
run or replays the live pipeline. Legacy NULL-marker rows are rejected by the
protocol itself. Losing a lease detaches without granting fallback authority.
"""
from __future__ import annotations

import asyncio
import logging
from threading import Event

from redis.exceptions import RedisError

from agent_bridge.run_contract import RunContractError, RunTerminal, run_terminal_key
from .recovery.orphan_turns import BridgeRunProbe
from .turn_execution import TurnExecution, turn_executions
from .turn_terminal_evidence import load_recovery_evidence, recovery_candidate
from .turn_terminal_fold import IncompleteReplay, fold_retained_run
from .turn_terminal_follower import TerminalFollower
from .turn_terminal_observe import observe_bound_run
from .turn_terminal_projection import project_committed_terminal
from .turn_termination import reconcile_finished_tool_safely

log = logging.getLogger(__name__)
TERMINAL_READ_SECONDS = 5.0


async def retained_terminal(redis, row, boot_id):
    """The compact terminal hash can outlive a missing/expired run stream."""
    if not row.agent_request_id or not row.agent_backend:
        return None
    try:
        async with asyncio.timeout(TERMINAL_READ_SECONDS):
            fields = await redis.hgetall(run_terminal_key(row.agent_request_id))
        fields = {(key.decode() if isinstance(key, bytes) else key):
                  (value.decode() if isinstance(value, bytes) else value) for key, value in fields.items()}
        if not fields:
            return None
        terminal = RunTerminal.from_fields(fields)
        if terminal is None or terminal.request_id != row.agent_request_id or terminal.backend != row.agent_backend:
            return None
        if boot_id is not None and terminal.boot_id != boot_id:
            return None
        return terminal
    except (RedisError, TimeoutError, RunContractError, UnicodeError):
        # Unknown is not exit proof, and never invents a successful response.
        return None


async def recover_terminal_turn(protocol, turn_id, subscriber, *, rows_for_bot=None):
    """Recover only an expired protocol owner, or finish an existing projection.

    Blocking database/heartbeat work stays off the API loop. The follower is
    private to this attempt; it never replaces a real worker's registry handle
    or pretends that worker returned. Existing execution-exit gates stay intact.
    """
    lease = await asyncio.to_thread(protocol.claim, turn_id)
    if lease is None:
        return await project_committed_terminal(
            protocol, turn_id, subscriber._pub_redis, rows_for_bot=rows_for_bot,
        )
    execution = TurnExecution(turn_id, Event(), is_agent=True, store=protocol.turn_store)
    follower = TerminalFollower(protocol, lease, execution)
    try:
        await asyncio.to_thread(follower.start)
        old = turn_executions.get(turn_id)
        if old is not None:
            old.detach()
            if not old.is_agent:
                old.request_cancel()
        evidence = await asyncio.to_thread(load_recovery_evidence, protocol, follower.lease)
        row = evidence.row
        content = None
        terminal = None
        incomplete = "recovery_no_resumable_run"
        bound = bool(row.agent_request_id and row.agent_session_key and row.agent_backend)
        if bound and row.terminal_protocol_state == "open":
            observation = await observe_bound_run(
                follower, subscriber._redis, request_id=row.agent_request_id,
                session_key=row.agent_session_key, backend=row.agent_backend,
                boot_id=evidence.boot_id,
            )
            incomplete = observation.incomplete_reason
            if observation.replay is not None:
                terminal = observation.replay.terminal
                if incomplete is None:
                    try:
                        content = fold_retained_run(observation.replay)
                    except IncompleteReplay:
                        incomplete = "recovery_aggregate_unavailable"
        follower.check()
        if bound and incomplete == "recovery_run_not_terminal" and row.status != "cancelling":
            # The 60 s budget bounds one pass, not a healthy run's lifetime.
            # Only a current backend/request/boot-matched positive liveness
            # answer defers; missing/trimmed/transport failures still fall back.
            probe = BridgeRunProbe(lambda: subscriber, timeout_seconds=5.0)
            reply, _ = await probe.status(backend=row.agent_backend, session_key=row.agent_session_key,
                                          request_id=row.agent_request_id, label=turn_id)
            follower.check()
            if (reply is not None and reply.request_id == row.agent_request_id
                    and reply.backend == row.agent_backend
                    and (evidence.boot_id is None or reply.boot_id == evidence.boot_id)
                    and reply.state.value in {"running", "queued", "preflight"}):
                await asyncio.to_thread(protocol.renew, follower.lease)
                # Leave its observation row intact. The recurring recovery pump
                # revisits after this lease expires; no inference is restarted.
                return None
        # A missing stream may be a silent live tool whose 900 s TTL elapsed.
        # Use retained exit evidence, otherwise bounded exact-request abort.
        if terminal is None or not terminal.proves_exit:
            terminal = await retained_terminal(subscriber._redis, row, evidence.boot_id)
        follower.check()
        stop_state = "not_needed" if terminal is not None and terminal.proves_exit else "unconfirmed"
        if bound and stop_state != "not_needed":
            # Validate durable authority immediately before a remote side effect,
            # not merely the heartbeat's last in-memory observation.
            await asyncio.to_thread(protocol.renew, follower.lease)
            follower.check()
            probe = BridgeRunProbe(lambda: subscriber)
            reply = await probe.abort(backend=row.agent_backend, session_key=row.agent_session_key,
                                      request_id=row.agent_request_id)
            follower.check()
            stop_state = reply.state.value if reply is not None else "unconfirmed"
            terminal = await retained_terminal(subscriber._redis, row, evidence.boot_id)
        follower.check()
        if terminal is not None:
            # Existing request/backend validation preserves truthful independent
            # execution-exit evidence; this is not a terminal-content rewrite.
            await asyncio.to_thread(execution.record_exit, terminal.to_dict())
        # Re-read canonical partial/interaction/tool state after the bounded
        # waits. A competing Stop/commit causes fence loss instead of overwrite.
        evidence = await asyncio.to_thread(load_recovery_evidence, protocol, follower.lease)
        candidate = recovery_candidate(evidence, content, incomplete_reason=incomplete)
        candidate["audit"].update(
            bridge_stop=stop_state,
            execution_exit_confirmed=bool(terminal and terminal.proves_exit),
        )
        try:
            candidate["completion_fields"]["changed_files"] = await asyncio.to_thread(
                protocol.turn_store.changed_files_summary, turn_id,
            )
        except Exception:
            candidate["completion_fields"]["changed_files"] = None
        follower.check()
        result = await asyncio.to_thread(protocol.commit, follower.lease, **candidate)
        await asyncio.to_thread(reconcile_finished_tool_safely, protocol.engine, turn_id=turn_id)
    finally:
        await asyncio.to_thread(follower.close)
    # A committed winner is replayed from its own bytes, including when Stop
    # won the final race. History and publication are separately idempotent.
    await project_committed_terminal(protocol, turn_id, subscriber._pub_redis, rows_for_bot=rows_for_bot)
    return result
