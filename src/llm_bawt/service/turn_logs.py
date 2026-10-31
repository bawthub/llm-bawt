"""Persistent turn logs for debugging and UI inspection. Retained indefinitely by default."""

from __future__ import annotations

import json
import logging
import time
from contextlib import nullcontext
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from sqlalchemy import exists as sa_exists, func, text as sa_text
from sqlmodel import Session, delete, select

from ..utils.config import Config, has_database_credentials
from ..utils.schema import SchemaBootstrapGuard
from .tool_call_store import ToolCallRecord, ToolCallStore
from .turn_log_schema import (  # noqa: F401 — re-exported for existing importers
    TERMINAL_TURN_STATUSES,
    TurnLog,
    _extract_trigger_id,
    _is_terminal,
    _terminal_ended_at,
    backfill_trigger_message_ids,
    bootstrap_turn_log_schema,
)
from .turn_usage import encode_token_usage
from .turn_admission import (
    RESERVED,
    AdmissionClaim,
    AdmissionUnavailable,
    SystemGateRefused,
    TargetBusy,
    TurnAdmission,
    classify_target,
    fence_consumption,
    lock_target,
    peek_target,
    release_reservation,
    reserve_turn,
)
from .turn_gates import GateActor, ensure_gate_schema, gate_blocks, pin_member_request
from .turn_termination import reconcile_finished_tool_safely

logger = logging.getLogger(__name__)


# Compat name for existing importers; the admission authority owns the type.
DeliveryTargetBusy = TargetBusy


class TurnLogStore:
    """DB access helper for persistent turn logs."""

    config: Config | None = None
    _last_cleanup_at: float = 0.0
    _cleanup_interval_seconds: float = 300.0
    _backfill_done: bool = False
    _schema_guard = SchemaBootstrapGuard()
    # Admission fails closed unless the deployment genuinely has no database
    # (unknown -> assume configured). See ``save_turn``.
    database_configured: bool = True

    def __init__(self, config: Config, ttl_hours: int | None = None):
        self.config = config
        # None disables expiry entirely — turn logs are retained forever.
        self.ttl_hours = max(1, int(ttl_hours)) if ttl_hours is not None else None
        self.engine = None
        try:
            self.database_configured = has_database_credentials(config)
        except Exception:
            self.database_configured = True  # unknown: fail closed
        try:
            from ..utils.db import get_shared_engine
            self.engine = get_shared_engine(config)  # TASK-202: shared pool
            if self.engine is None:
                return
            self._ensure_tables_exist()
            self._backfill_trigger_message_ids()
        except Exception as e:
            self.engine = None
            logger.warning("Turn logs DB unavailable: %s", e)

    def _ensure_tables_exist(self) -> None:
        if self.engine is None:
            return
        # Keep orchestration here while each nested store remains independently
        # guarded for direct callers.
        ToolCallStore(self.engine).ensure_schema()
        from .changed_files_store import ChangedFilesStore
        ChangedFilesStore(self.engine).ensure_schema()

        self._schema_guard.run(self.engine, "turn-log-store", bootstrap_turn_log_schema)
        ensure_gate_schema(self.engine)  # every admission here asks the system gate

    def _backfill_trigger_message_ids(self) -> None:
        """One-time backfill: populate trigger_message_id for existing rows."""
        if self.engine is None or TurnLogStore._backfill_done:
            return
        TurnLogStore._backfill_done = True
        backfill_trigger_message_ids(self.engine)

    def _cleanup_expired_if_due(self, force: bool = False) -> None:
        if self.engine is None or self.ttl_hours is None:
            return
        now = time.time()
        if not force and (now - self.__class__._last_cleanup_at) < self.__class__._cleanup_interval_seconds:
            return

        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.ttl_hours)
        try:
            with Session(self.engine) as session:
                session.exec(delete(TurnLog).where(TurnLog.created_at < cutoff))
                session.commit()
            self.__class__._last_cleanup_at = now
        except Exception as e:
            logger.debug("Turn log TTL cleanup skipped: %s", e)

    def save_turn(
        self,
        *,
        turn_id: str,
        request_id: str | None,
        path: str,
        stream: bool,
        model: str | None,
        bot_id: str | None,
        user_id: str | None,
        status: str,
        latency_ms: float | None,
        user_prompt: str | None,
        request_payload: dict | None,
        response_text: str | None,
        tool_calls: list[dict] | None,
        error_text: str | None = None,
        trigger_message_id: str | None = None,
        assistant_message_id: str | None = None,
        agent_session_key: str | None = None,
        agent_request_id: str | None = None,
        animation: str | None = None,
        token_usage: dict | None = None,
        parent_turn_id: str | None = None,
        admission_claim: AdmissionClaim | None = None,
        connection=None,
    ) -> None:
        """Persist one turn entry (and run TTL cleanup if a TTL is configured).

        ``admission_claim`` is the producer's trusted claim over a fenced
        reservation (approval continuations); see ``fence_consumption``.

        This write IS the turn's admission: it raises ``AdmissionUnavailable``
        rather than no-op when a configured database is unreachable, or when a
        durable producer (claim / IDLE_ONLY) has no authority to admit it. Only
        a database-less deployment (no shared targets, no durable producers)
        starts ordinary turns without a row.
        """
        if self.engine is None:
            if (self.database_configured or admission_claim is not None
                    or TurnAdmission.from_payload(request_payload) is TurnAdmission.IDLE_ONLY):
                raise AdmissionUnavailable("turn admission database unavailable; turn not started")
            return

        self._cleanup_expired_if_due()

        # Auto-extract trigger message ID from request if not provided.
        if trigger_message_id is None and request_payload:
            trigger_message_id = _extract_trigger_id(request_payload)

        created_at = datetime.now(timezone.utc)
        token_usage_json = self._encode_token_usage(token_usage, model)

        # TASK-1082: the one admission decision for every producer (stream,
        # nonstream, durable delivery, approval continuation). It runs under the
        # same per-target advisory lock as the delivery outbox claim. IDLE_ONLY
        # requests lose to ANY other open turn; every request loses to ANY
        # fenced reservation (not merely the newest row). The loser receives
        # typed ``TargetBusy`` and nothing is started. Reusing a reserved id is
        # an atomic upsert, so target acceptance cannot create two turn rows.
        admission = TurnAdmission.from_payload(request_payload)
        if connection is not None and (
            connection.engine is not self.engine or bot_id is None
            or admission is not TurnAdmission.START or admission_claim is not None
        ):
            raise ValueError("shared turn admission connection requires an ordinary start on this engine")
        if bot_id:
            busy: TargetBusy | None = None
            with (nullcontext(connection) if connection is not None else self.engine.begin()) as conn:
                lock_target(conn, bot_id)
                # Consumption is fenced in this same transaction as creation and
                # release are: a superseded claim, or a caller that is not the
                # reservation's owner, changes nothing (AdmissionFenceLost).
                existing_status = fence_consumption(
                    conn, turn_id=turn_id, claim=admission_claim,
                )
                # TASK-1233 backstop: a live gate refuses like a busy target (C1 re-attach exempt).
                block = gate_blocks(conn, bot_id, GateActor("save_turn", turn_id))
                state = classify_target(conn, bot_id, exclude_turn_id=turn_id)
                if block is not None or not state.admits(admission):
                    if admission is TurnAdmission.IDLE_ONLY and existing_status == RESERVED:
                        # Release this request's own reservation (ownership
                        # proven above, under the same lock).
                        conn.execute(
                            sa_text("DELETE FROM turn_logs WHERE id=:id AND status='reserved'"),
                            {"id": turn_id},
                        )
                    busy = SystemGateRefused(block) if block is not None else TargetBusy(
                        f"target '{bot_id}' is {state.describe()}"
                    )
                else:
                    if existing_status is not None:
                        if existing_status != "reserved":
                            raise RuntimeError(f"turn id already exists: {turn_id}")
                        # A reservation is not a started turn. Stamp admission
                        # time so the new unknown gauge supersedes turns that
                        # finished while this request was queued.
                        from .turn_termination import request_payload_with_audit
                        conn.execute(
                            sa_text(
                                "UPDATE turn_logs SET created_at=:created_at, request_id=:request_id, path=:path,"
                                " stream=:stream, model=:model, bot_id=:bot_id, user_id=:user_id,"
                                " status=:status, latency_ms=:latency_ms, user_prompt=:user_prompt,"
                                " request_json=:request_json, response_text=:response_text,"
                                " error_text=:error_text, trigger_message_id=:trigger_message_id,"
                                " assistant_message_id=:assistant_message_id,"
                                " agent_session_key=:agent_session_key,"
                                " agent_request_id=:agent_request_id, animation=:animation,"
                                " token_usage_json=:token_usage_json, parent_turn_id=:parent_turn_id"
                                " WHERE id=:turn_id"
                            ),
                            {
                                "turn_id": turn_id,
                                "created_at": created_at,
                                "request_id": request_id,
                                "path": path,
                                "stream": stream,
                                "model": model,
                                "bot_id": bot_id,
                                "user_id": user_id,
                                "status": status,
                                "latency_ms": latency_ms,
                                "user_prompt": user_prompt,
                                # r2-M1: consuming the reservation keeps its audits
                                # (row locked by fence_consumption above).
                                "request_json": request_payload_with_audit(conn.execute(
                                    sa_text("SELECT request_json FROM turn_logs WHERE id=:id"),
                                    {"id": turn_id},
                                ).scalar_one_or_none(), request_payload or None, rearm=True),
                                "response_text": response_text,
                                "error_text": error_text,
                                "trigger_message_id": trigger_message_id,
                                "assistant_message_id": assistant_message_id,
                                "agent_session_key": agent_session_key,
                                "agent_request_id": agent_request_id,
                                "animation": animation,
                                "token_usage_json": token_usage_json,
                                "parent_turn_id": parent_turn_id,
                            },
                        )
                        from .turn_context_snapshot import record_turn_context
                        record_turn_context(conn, SimpleNamespace(
                            id=turn_id, bot_id=bot_id, user_id=user_id,
                            trigger_message_id=trigger_message_id,
                            created_at=created_at, ended_at=None,
                            token_usage_json=None, agent_session_key=agent_session_key,
                        ))
                        return

                    # Ordinary turn insert stays inside the advisory-lock transaction
                    # so it cannot cross a delivery claim between check and commit.
                    finished_at = (_terminal_ended_at(created_at, latency_ms)
                                   if _is_terminal(status, None) else None)
                    conn.execute(
                        TurnLog.__table__.insert().values(
                            id=turn_id,
                            created_at=created_at,
                            request_id=request_id,
                            path=path,
                            stream=stream,
                            model=model,
                            bot_id=(bot_id or None),
                            user_id=(user_id or None),
                            status=status,
                            latency_ms=latency_ms,
                            user_prompt=user_prompt,
                            request_json=json.dumps(request_payload, ensure_ascii=False, default=str) if request_payload else None,
                            response_text=response_text,
                            error_text=error_text,
                            trigger_message_id=trigger_message_id,
                            assistant_message_id=assistant_message_id,
                            agent_session_key=agent_session_key,
                            agent_request_id=agent_request_id,
                            animation=animation,
                            token_usage_json=token_usage_json,
                            parent_turn_id=parent_turn_id,
                            ended_at=finished_at,
                        )
                    )
                    from .turn_context_snapshot import record_turn_context
                    record_turn_context(conn, SimpleNamespace(
                        id=turn_id, bot_id=bot_id, user_id=user_id,
                        trigger_message_id=trigger_message_id,
                        created_at=created_at, ended_at=finished_at,
                        token_usage_json=token_usage_json,
                        agent_session_key=agent_session_key,
                    ))
                    return
            raise busy  # after the commit: the own-reservation release must persist

        row = TurnLog(
            id=turn_id,
            created_at=created_at,
            request_id=request_id,
            path=path,
            stream=stream,
            model=model,
            bot_id=(bot_id or None),
            user_id=(user_id or None),
            status=status,
            latency_ms=latency_ms,
            user_prompt=user_prompt,
            request_json=json.dumps(request_payload, ensure_ascii=False, default=str) if request_payload else None,
            response_text=response_text,
            # TASK-364: tool_calls_json retired — tool_call_records is canonical.
            # Column left in place (tombstoned) but no longer written or read.
            error_text=error_text,
            trigger_message_id=trigger_message_id,
            agent_session_key=agent_session_key,
            agent_request_id=agent_request_id,
            animation=animation,
            token_usage_json=token_usage_json,
            parent_turn_id=parent_turn_id,
            # Rare: a turn created already-terminal (e.g. errored before any
            # processing) should not look perpetually in-flight.
            ended_at=(
                _terminal_ended_at(created_at, latency_ms)
                if _is_terminal(status, None) else None
            ),
        )

        with Session(self.engine) as session:
            session.add(row)
            session.commit()

    def _encode_token_usage(self, token_usage: dict | None, model: str | None) -> str | None:
        """Usage JSON with the serving access path frozen in (see turn_usage)."""
        return encode_token_usage(token_usage, model_ref=model, catalog=getattr(self.config, "model_catalog", None))

    def accept_session_binding(
        self, *, turn_id: str, bot_id: str, user_id: str, session_id: str,
        admission_claim: AdmissionClaim | None = None,
    ) -> None:
        """Fenced bot-wide pointer update after resolving the accepted thread."""
        from .session_lifecycle.admission import accept_session_binding

        accept_session_binding(
            self.engine, turn_id=turn_id, bot_id=bot_id, user_id=user_id,
            session_id=session_id, claim=admission_claim,
        )

    def bind_agent_bridge(self, *, turn_id: str, backend: str, session_key: str, request_id: str,
                          steer_caps: str | None = None) -> None:
        """Bind before publication, refusing a turn finalized by another owner.

        A worker may outlive its HTTP request or lose a race to cancellation.
        Updating a terminal row used to silently do nothing while the caller
        still sent chat.send; the row lock makes the decision atomic with the
        persisted bridge identity. This is not an orphan-replay authority.
        """
        from .turn_execution import TurnExecutionCancelled

        if self.engine is None:
            raise TurnExecutionCancelled("turn admission database is unavailable")
        with Session(self.engine) as session:
            row = session.exec(select(TurnLog).where(TurnLog.id == turn_id).with_for_update(key_share=True)).first()
            if (row is None or row.ended_at is not None
                    or row.status not in {"pending", "streaming"}
                    or (row.agent_request_id is not None and row.agent_request_id != request_id)):
                raise TurnExecutionCancelled(f"turn '{turn_id}' no longer permits bridge publication")
            row.agent_backend = backend
            row.agent_session_key = session_key
            row.agent_request_id = request_id
            row.agent_steer_caps = steer_caps  # TASK-1243: None = recorded before caps existed
            session.add(row)
            pin_member_request(session.connection(), turn_id, request_id, steer_caps)  # TASK-1233 test_Z (a)
            session.commit()

    def update_turn(
        self,
        *,
        turn_id: str,
        status: str | None = None,
        latency_ms: float | None = None,
        response_text: str | None = None,
        request_payload: dict | None = None,
        tool_calls: list[dict] | None = None,
        error_text: str | None = None,
        assistant_message_id: str | None = None,
        agent_session_key: str | None = None,
        agent_request_id: str | None = None,
        animation: str | None = None,
        token_usage: dict | None = None,
        end_reason: str | None = None,
        question_id: str | None = None,
        parent_turn_id: str | None = None,
        tts_scrubbed: bool | None = None,
        abort_cause: str | None = None,
    ) -> None:
        """Update an existing turn log row with new data."""
        if self.engine is None:
            return
        with Session(self.engine) as session:
            row = session.exec(select(TurnLog).where(TurnLog.id == turn_id).with_for_update(key_share=True)).first()
            if row is None:
                logger.debug("update_turn: no row with id=%s", turn_id)
                return
            prior_status = row.status
            prior_end_reason = row.end_reason
            from .turn_termination import stamp_abort_cause
            stamp_abort_cause(row, abort_cause)  # TASK-1231B; judged on the prior state
            if prior_status == "cancelling":
                if _is_terminal(status, end_reason):
                    status, end_reason = "aborted", "aborted"
                else:
                    status = None  # a late streaming update cannot revoke intent
            # Terminal outcomes are monotonic. Preserve the existing stale-reap
            # repair exception, but never let EOF/late errors undo cancellation.
            repair_timeout = (prior_status == "timeout" and prior_end_reason == "timeout"
                              and status in ("ok", "completed") and end_reason not in (None, "timeout"))
            if row.ended_at is not None and not repair_timeout:
                # Preserve diagnostic enrichment for the same outcome (notably
                # approval_persist_failed), but reject a conflicting late error.
                if status not in (None, prior_status) or (
                    prior_end_reason is not None and end_reason not in (None, prior_end_reason)
                ):
                    error_text = None
                status = None
                if row.end_reason is not None:
                    end_reason = None
            if status is not None:
                row.status = status
            if latency_ms is not None:
                row.latency_ms = latency_ms
            if response_text is not None:
                row.response_text = response_text
            if request_payload is not None:
                from .turn_termination import request_payload_with_audit
                row.request_json = request_payload_with_audit(row.request_json, request_payload)
                # Backfill trigger_message_id if it wasn't set on initial persist
                # (common when prepared_messages were empty at creation time).
                if not row.trigger_message_id:
                    tid = _extract_trigger_id(request_payload)
                    if tid:
                        row.trigger_message_id = tid
            # TASK-364: tool_calls_json retired (tombstoned) — no longer written.
            if error_text is not None:
                row.error_text = error_text
            if assistant_message_id is not None:
                row.assistant_message_id = assistant_message_id
            if agent_session_key is not None:
                row.agent_session_key = agent_session_key
            if agent_request_id is not None:
                row.agent_request_id = agent_request_id
            if animation is not None:
                row.animation = animation
            if token_usage is not None:
                row.token_usage_json = self._encode_token_usage(token_usage, row.model)
            if tts_scrubbed is not None:
                row.tts_scrubbed = tts_scrubbed
            if end_reason is not None:
                row.end_reason = end_reason
            if question_id is not None:
                row.question_id = question_id
            if parent_turn_id is not None:
                row.parent_turn_id = parent_turn_id
            # Stamp completion once: streaming, finalization and abort writers
            # all funnel through update_turn.
            if row.ended_at is None and _is_terminal(status, end_reason):
                row.ended_at = _terminal_ended_at(row.created_at, latency_ms)
            elif (
                prior_status == "timeout"
                and prior_end_reason == "timeout"
                and status in ("ok", "completed")
                and end_reason not in (None, "timeout")
            ):
                # A queued agent turn can be incorrectly stamped timeout by a
                # stale-turn reap, then later complete normally when the bridge
                # drains the queue. In that repair path, the successful
                # finalization is authoritative and should also repair ended_at.
                row.ended_at = _terminal_ended_at(row.created_at, latency_ms)
            session.add(row)
            finished = row.ended_at is not None
            if finished:
                from .turn_terminal_effects import record_terminal_effects
                record_terminal_effects(session, row)
            session.commit()
        if finished:
            reconcile_finished_tool_safely(self.engine, turn_id=turn_id)

    def update_partial_response(
        self, *, turn_id: str, response_text: str, reasoning: str | None = None
    ) -> None:
        """Write in-flight partial assistant text (and, TASK-360/P4, partial
        reasoning) without clobbering a finalized turn (TASK-286).

        Conditional on ``status='streaming'`` so once ``_finalize_turn`` flips
        the row to ``ok`` (with the cleaned, authoritative text) this becomes a
        no-op — a late drain write can never overwrite the final response. This
        is the text analogue of the incremental ``tool_call_records`` writes:
        it gives a COLD reload (resumeScan → /v1/turn-logs) the partial text
        that was streamed before a refresh, and survives client disconnect
        because the drain consumer runs server-side, not in the request thread.

        ``reasoning`` is flushed the same way so a mid-turn cold reload can
        recover already-produced thinking; when ``None`` the reasoning column is
        left untouched.
        """
        if self.engine is None or not turn_id:
            return
        try:
            with self.engine.begin() as conn:
                if reasoning is not None:
                    conn.execute(
                        sa_text(
                            "UPDATE turn_logs SET response_text = :t, reasoning = :r"
                            " WHERE id = :id AND status = 'streaming'"
                        ),
                        {"t": response_text, "r": reasoning, "id": turn_id},
                    )
                else:
                    conn.execute(
                        sa_text(
                            "UPDATE turn_logs SET response_text = :t"
                            " WHERE id = :id AND status = 'streaming'"
                        ),
                        {"t": response_text, "id": turn_id},
                    )
        except Exception as e:
            logger.debug("update_partial_response failed for %s: %s", turn_id, e)

    def reap_other_open_turns(
        self, *, bot_id: str, current_turn_id: str
    ) -> list[dict]:
        """Close older still-open turns for ``bot_id`` as a timeout.

        The single source of truth for "a turn began" is the SDK/bridge
        CONFIRMING its first real output (see chat_streaming's confirmed-start
        hook) — not the optimistic ``save_turn`` insert or ``turn_start``
        publish, both of which fire before the backend produces anything. When
        that confirmed signal lands for a new turn, this reaps prior turn rows
        for the same bot that are still ``ended_at IS NULL`` (zombies that never
        terminated — dropped bridge, aborted-without-finalize, server restart
        mid-turn). They are stamped ``status='timeout'``,
        ``end_reason='timeout'`` and ``ended_at=now`` atomically.

        Registered live/queued workers, fresh cancellation intent and ordinary
        admitted agent starts are excluded: another turn's first output cannot
        prove their bridge work stopped. Ordinary orphan recovery requires a
        separate, fenced publication authority; until then the open row blocks
        admission rather than being closed by a guessed timeout. Unresolved
        tools of genuinely reaped legacy turns settle in the same transaction.

        Returns ``[{"id", "user_id", "balance_update"}]`` so the caller can
        emit a ``turn_complete`` per row on the correct ``{bot_id}:{user_id}``
        stream, clearing its indicator and pushing its balance. Excludes ``current_turn_id``
        so the turn that triggered the reap is never closed by it. Newer open
        rows are also preserved: agent session queues can let an older turn
        reach confirmed-start after a newer user message has already been
        inserted, and that newer queued turn must not be reaped as stale.
        """
        if self.engine is None or not bot_id or not current_turn_id:
            return []
        try:
            from .turn_termination import reap_unowned_turns
            return reap_unowned_turns(self, bot_id, current_turn_id)
        except Exception as e:
            logger.debug("reap_other_open_turns failed for %s: %s", bot_id, e)
            return []

    def get_turn(self, turn_id: str) -> TurnLog | None:
        """Get one turn by id."""
        if self.engine is None:
            return None
        self._cleanup_expired_if_due()
        with Session(self.engine) as session:
            return session.exec(select(TurnLog).where(TurnLog.id == turn_id)).first()

    def reserve_idle_turn(self, **kwargs) -> bool:
        """Fenced idle reservation (see ``turn_admission.reserve_turn``).

        Returns False when no database is configured; raises ``TargetBusy``
        when the target is not idle. Taken BEFORE external side effects.
        """
        if self.engine is None:
            return False
        reserve_turn(self.engine, **kwargs)
        return True

    def peek_target_state(self, bot_id: str, *, exclude_turn_id: str | None = None):
        """Advisory unlocked target state (``turn_admission.peek_target``), or None."""
        if self.engine is None or not bot_id:
            return None
        return peek_target(self.engine, bot_id, exclude_turn_id=exclude_turn_id)

    def release_reservation(self, turn_id: str, *, owner: str, fence=None) -> bool:
        if self.engine is None or not turn_id or not owner:
            return False
        return release_reservation(self.engine, turn_id, owner=owner, fence=fence)

    def active_turn_for_bot(self, bot_id: str) -> TurnLog | None:
        """Return the bot's most recent in-flight turn, or None if idle.

        TASK-1231C: derived from the one busy predicate (``peek_target`` →
        ``classify_target``), so this readout and admission can never disagree.
        There is no age window: an open orphan blocks admission, so it is
        reported busy until it is settled (TASK-1232), not silently aged out.
        An open reservation is not an executing/steerable turn.
        """
        state = self.peek_target_state(bot_id)
        if state is None:
            return None
        running = next((row for row in state.rows if row["status"] != RESERVED), None)
        return self.get_turn(str(running["id"])) if running is not None else None

    def list_turns(
        self,
        *,
        bot_id: str | None = None,
        user_id: str | None = None,
        model: str | None = None,
        request_id: str | None = None,
        status: str | None = None,
        active_only: bool = False,
        stream: bool | None = None,
        has_tools: bool | None = None,
        trigger_message_ids: set[str] | None = None,
        after: float | None = None,
        before: float | None = None,
        since_hours: int = 24,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[TurnLog], int]:
        """List turns with filters and pagination."""
        if self.engine is None:
            return [], 0
        self._cleanup_expired_if_due()

        conditions: list = []
        if after is not None:
            conditions.append(TurnLog.created_at >= datetime.fromtimestamp(after, tz=timezone.utc))
        elif before is None and not active_only:
            # Active turns remain active regardless of when they started.
            # The recency window is only for historical turn browsing.
            since_cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, int(since_hours)))
            conditions.append(TurnLog.created_at >= since_cutoff)
        if before is not None:
            conditions.append(TurnLog.created_at <= datetime.fromtimestamp(before, tz=timezone.utc))
        if bot_id:
            conditions.append(TurnLog.bot_id == bot_id.strip().lower())
        if user_id:
            conditions.append(TurnLog.user_id == user_id.strip().lower())
        if model:
            conditions.append(TurnLog.model == model.strip())
        if request_id:
            conditions.append(TurnLog.request_id == request_id.strip())
        if status:
            conditions.append(TurnLog.status == status.strip().lower())
        if active_only:
            # Only accepted turns can be steered. Reservations are admission
            # fences, not executing turns, even though they have no ended_at.
            conditions.append(TurnLog.ended_at.is_(None))
            conditions.append(TurnLog.status != RESERVED)
        if stream is not None:
            conditions.append(TurnLog.stream.is_(stream))
        if has_tools is not None:
            # TASK-364: presence of tools is derived from the canonical
            # tool_call_records table (correlated EXISTS), not the retired
            # tool_calls_json blob — so the filter stays correct once the blob
            # is no longer written.
            tool_exists = sa_exists().where(ToolCallRecord.turn_id == TurnLog.id)
            conditions.append(tool_exists if has_tools else ~tool_exists)
        if trigger_message_ids:
            # Include rows matching the given IDs OR rows with NULL trigger
            # (agent-backend turns that need post-processing via turn_id fallback).
            conditions.append(
                TurnLog.trigger_message_id.in_(trigger_message_ids)
                | TurnLog.trigger_message_id.is_(None)
            )

        statement = select(TurnLog).where(*conditions).order_by(TurnLog.created_at.desc())
        count_statement = select(func.count()).select_from(TurnLog).where(*conditions)

        with Session(self.engine) as session:
            count_result = session.exec(count_statement).one()
            try:
                total_count = int(count_result or 0)
            except (TypeError, ValueError):
                total_count = int(count_result[0] or 0)
            rows = session.exec(statement.offset(offset).limit(limit)).all()

        return rows, total_count

    def recent_turns_by_bot(
        self,
        *,
        user_id: str | None,
        bot_ids: list[str] | None = None,
        since_hours: int = 168,
    ) -> list[TurnLog]:
        """Return the most recent TurnLog per bot_id for a user — one row each.

        Uses Postgres ``DISTINCT ON (bot_id)`` so the entire fan-out is a single
        index-backed query. Designed for "what's been happening with each bot"
        dashboard views — replaces the N round-trip per-bot history poll.

        Args:
            user_id: Scope to a user (lowercased). ``None`` returns the latest
                turn per bot across all users (admin/global view).
            bot_ids: Optional whitelist to restrict the result. ``None`` returns
                all bots the user has talked to in the window.
            since_hours: Only consider turns from the last N hours (default 7d).
        """
        if self.engine is None:
            return []
        self._cleanup_expired_if_due()

        since_cutoff = datetime.now(timezone.utc) - timedelta(
            hours=max(1, int(since_hours))
        )
        conditions: list = [
            TurnLog.bot_id.is_not(None),
            TurnLog.created_at >= since_cutoff,
        ]
        if user_id:
            conditions.append(TurnLog.user_id == user_id.strip().lower())
        if bot_ids:
            normalized = [b.strip().lower() for b in bot_ids if b and b.strip()]
            if normalized:
                conditions.append(TurnLog.bot_id.in_(normalized))

        # DISTINCT ON (bot_id) requires bot_id to be the FIRST column in
        # ORDER BY; the secondary key picks the row to keep per partition.
        statement = (
            select(TurnLog)
            .where(*conditions)
            .order_by(TurnLog.bot_id, TurnLog.created_at.desc())
            .distinct(TurnLog.bot_id)
        )

        with Session(self.engine) as session:
            rows = session.exec(statement).all()
        return list(rows)

    # ---- Tool call records (event-based persistence) ----

    def save_tool_call(
        self,
        *,
        turn_id: str | None,
        bot_id: str | None,
        user_id: str | None,
        call_id: str | None,
        tool_name: str,
        arguments: dict | None = None,
        result: str | None = None,
        iteration: int = 1,
        started_at: float | None = None,
        ended_at: float | None = None,
        text_offset: int | None = None,
        is_error: bool | None = None,
        tool_use_id: str | None = None,
        parent_tool_use_id: str | None = None,
    ) -> None:
        """Persist a single tool call record."""
        if self.engine is None:
            return
        store = ToolCallStore(self.engine)
        store.save_start(
            turn_id=turn_id,
            bot_id=bot_id,
            user_id=user_id,
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            iteration=iteration,
            started_at=started_at,
            text_offset=text_offset,
            tool_use_id=tool_use_id,
            parent_tool_use_id=parent_tool_use_id,
        )
        if result is not None:
            from agent_bridge.tool_results import ToolResultPayload
            store.save_result(
                turn_id=turn_id,
                call_id=call_id,
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                bot_id=bot_id,
                user_id=user_id,
                payload=ToolResultPayload.from_value(result),
                ended_at=ended_at,
                is_error=is_error,
                iteration=iteration,
                parent_tool_use_id=parent_tool_use_id,
            )

    def update_tool_call_result(
        self,
        *,
        call_id: str,
        result: str,
        ended_at: float | None = None,
        is_error: bool | None = None,
    ) -> None:
        """Compatibility delegate for legacy preview-only callers."""
        if self.engine is None or not call_id:
            return
        from agent_bridge.tool_results import ToolResultPayload
        ToolCallStore(self.engine).save_result(
            turn_id=None,
            call_id=call_id,
            tool_use_id=None,
            tool_name="unknown",
            bot_id=None,
            user_id=None,
            payload=ToolResultPayload.from_value(result),
            ended_at=ended_at,
            is_error=is_error,
        )

    # ---- Changed files (TASK-661) ----

    def _changed_files_store(self):
        from .changed_files_store import ChangedFilesStore
        return ChangedFilesStore(self.engine)

    def save_changed_files(
        self,
        *,
        turn_id: str,
        bot_id: str | None,
        user_id: str | None,
        trigger_message_id: str | None,
        files: list,
    ) -> int:
        """Persist a turn's changed files (list of ChangedFileInput). Returns count."""
        if self.engine is None:
            return 0
        return self._changed_files_store().save_turn_files(
            turn_id=turn_id,
            bot_id=bot_id,
            user_id=user_id,
            trigger_message_id=trigger_message_id,
            files=files,
        )

    def changed_files_summary(self, turn_id: str) -> dict:
        if self.engine is None:
            from .changed_files_store import build_turn_summary
            return build_turn_summary(turn_id, [])
        return self._changed_files_store().summary_for_turn(turn_id)

    def changed_files_summaries_for_triggers(self, message_ids: list[str]) -> dict:
        if self.engine is None:
            return {}
        return self._changed_files_store().summaries_for_triggers(message_ids)

    def changed_files_summaries_for_turns(self, turn_ids: list[str]) -> dict:
        if self.engine is None:
            return {}
        return self._changed_files_store().summaries_for_turns(turn_ids)

    def uncommitted_changed_files_summaries(
        self,
        *,
        session_id: str | None,
        bot_id: str,
        user_id: str,
        anchor_turn_id: str | None = None,
    ) -> tuple[str | None, dict, dict]:
        if self.engine is None:
            from .changed_files_store import build_uncommitted_summary
            empty = build_uncommitted_summary([])
            return None, empty, empty
        return self._changed_files_store().uncommitted_summaries_for_session(
            session_id=session_id,
            anchor_turn_id=anchor_turn_id,
            bot_id=bot_id,
            user_id=user_id,
        )

    def uncommitted_changed_files_summary(
        self,
        *,
        session_id: str | None,
        bot_id: str,
        user_id: str,
        anchor_turn_id: str | None = None,
    ) -> tuple[str | None, dict]:
        resolved, summary, _ = self.uncommitted_changed_files_summaries(
            session_id=session_id,
            anchor_turn_id=anchor_turn_id,
            bot_id=bot_id,
            user_id=user_id,
        )
        return resolved, summary

    def mark_changed_files_commit_requested(
        self,
        *,
        session_id: str | None,
        bot_id: str,
        user_id: str,
        turn_ids: list[str],
    ) -> tuple[str | None, int]:
        if self.engine is None:
            return None, 0
        return self._changed_files_store().mark_commit_requested(
            session_id=session_id,
            bot_id=bot_id,
            user_id=user_id,
            turn_ids=turn_ids,
        )

    def changed_file_content(self, *, turn_id: str, repo_key: str, path: str):
        if self.engine is None:
            return None
        return self._changed_files_store().get_file_content(
            turn_id=turn_id, repo_key=repo_key, path=path
        )

    def get_tool_calls_for_turn(self, turn_id: str) -> list[ToolCallRecord]:
        """Get all tool call records for a turn."""
        if self.engine is None:
            return []
        try:
            with Session(self.engine) as session:
                return list(session.exec(
                    select(ToolCallRecord)
                    .where(ToolCallRecord.turn_id == turn_id)
                    .order_by(ToolCallRecord.created_at)
                ).all())
        except Exception:
            return []

    def set_approval_status(
        self,
        tool_use_id: str,
        approval_request_id: str,
        approval_status: str,
    ) -> bool:
        """Stamp a tool_call_record with its approval resolution.

        Called when an approval request is resolved (approved/denied/cancelled/
        responded). The tool_use_id of the gated call == the approval request id,
        so we match on that. Returns True if a row was updated.
        """
        if self.engine is None or not tool_use_id:
            return False
        try:
            with Session(self.engine) as session:
                stmt = (
                    sa_text(
                        "UPDATE tool_call_records"
                        " SET approval_request_id = :req_id,"
                        "     approval_status = :status"
                        " WHERE tool_use_id = :tuid"
                    )
                )
                result = session.execute(stmt, {
                    "req_id": approval_request_id,
                    "status": approval_status,
                    "tuid": tool_use_id,
                })
                session.commit()
                return result.rowcount > 0  # type: ignore[union-attr]
        except Exception:
            logger.exception("Failed to set approval status for tool_use_id=%s", tool_use_id)
            return False

    def set_preapproved(self, tool_use_id: str) -> bool:
        """Mark a tool_call_record as pre-approved (ran with a live grant).

        Called from the tool_preapproved event handler so the gold "Approved"
        badge survives a page reload. Returns True if a row was updated.
        """
        if self.engine is None or not tool_use_id:
            return False
        try:
            with Session(self.engine) as session:
                result = session.execute(
                    sa_text(
                        "UPDATE tool_call_records SET preapproved = TRUE"
                        " WHERE tool_use_id = :tuid"
                    ),
                    {"tuid": tool_use_id},
                )
                session.commit()
                return result.rowcount > 0  # type: ignore[union-attr]
        except Exception:
            logger.exception("Failed to set preapproved for tool_use_id=%s", tool_use_id)
            return False
