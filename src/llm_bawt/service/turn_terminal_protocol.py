"""DB-owned follower leases and immutable terminal snapshots (TASK-1242).

This store never adopts a legacy row, invokes a model or publishes an event.
Admission supplies ``protocol_insert_fields`` in the initial INSERT. History
and Redis projection consume only the winning snapshot after commit.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from typing import Any
import uuid

from sqlalchemy import or_, text
from sqlmodel import Session, select

from agent_bridge.terminal_publish import TerminalPublishResult, TerminalPublishStatus

from .turn_log_schema import TERMINAL_TURN_STATUSES, TurnLog

LEASE_SECONDS = 60
RENEW_SECONDS = 20
PUBLICATION_HORIZON = timedelta(days=7)
PROJECTION_STATES = ("pending", "projected", "publication_expired")
OBSERVATION_STATES = ("binding", "open")


class FollowerFenceLost(RuntimeError):
    """The caller no longer owns this protocol row."""


def protocol_insert_fields(
    *, connection, assistant_message_id: str, history_session_id: str | None = None,
) -> dict:
    """Initial INSERT fields; session setup must not precede turn admission.

    An absent session explicitly starts ``binding``, never a legacy-null row.
    Only a fenced, owner-validated bind can enable inference/history writes.
    """
    if (not assistant_message_id or not assistant_message_id.strip()
            or len(assistant_message_id) > 36 or (
                history_session_id is not None
                and (not history_session_id.strip() or len(history_session_id) > 36)
            )):
        raise ValueError("protocol admission requires assistant and history session identities")
    return {
        "assistant_message_id": assistant_message_id,
        "history_session_id": history_session_id,
        "terminal_protocol_state": "binding" if history_session_id is None else "open",
        # Own the row from its very first committed visibility. Recovery cannot
        # steal healthy admission work between INSERT and an executor's claim.
        "follower_owner": uuid.uuid4().hex,
        "follower_lease_until": _connection_now(connection) + timedelta(seconds=LEASE_SECONDS),
    }


def protocol_reservation_fields(*, assistant_message_id: str | None) -> dict:
    """Mark a fresh, unconsumed reservation without pretending it is executing.

    Producer admission fences own this state. Follower discovery never claims
    it, and its assistant UUID survives pre-start retries and consumption.
    """
    identity = assistant_message_id
    if not isinstance(identity, str) or not identity.strip() or len(identity) > 36:
        raise ValueError("protocol reservation requires a pinned assistant identity")
    return {"terminal_protocol_state": "reserved", "assistant_message_id": identity,
            "history_session_id": None, "follower_owner": None, "follower_lease_until": None}


def consume_protocol_reservation(connection, *, turn_id: str, bot_id: str, user_id: str):
    """Consume after the producer's admission fence, in that same transaction.

    Legacy NULL rows stay legacy. A consumed protocol ID can never be rearmed.
    """
    from .turn_admission import AdmissionFenceLost

    row = connection.execute(select(TurnLog.__table__).where(TurnLog.id == turn_id)
                             .with_for_update()).mappings().one()
    if row["terminal_protocol_state"] is None:
        return None
    if (row["terminal_protocol_state"] != "reserved" or row["status"] != "reserved"
            or row["bot_id"] != bot_id or row["user_id"] != user_id
            or any(row[key] is not None for key in (
                "ended_at", "history_session_id", "agent_request_id", "agent_session_key",
                "agent_backend", "terminal_snapshot_json", "follower_owner", "follower_lease_until",
            )) or row["response_text"] or row["reasoning"]):
        raise AdmissionFenceLost("protocol reservation is consumed or has conflicting execution evidence")
    fields = protocol_insert_fields(connection=connection, assistant_message_id=row["assistant_message_id"])
    connection.execute(TurnLog.__table__.update().where(TurnLog.id == turn_id).values(**fields))
    return FollowerLease(turn_id, fields["follower_owner"], fields["follower_lease_until"],
                         fields["assistant_message_id"])


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _connection_now(connection) -> datetime:
    sql = "SELECT clock_timestamp()" if connection.dialect.name == "postgresql" else "SELECT CURRENT_TIMESTAMP"
    value = connection.execute(text(sql)).scalar_one()
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return _aware(value)


def _db_now(session: Session) -> datetime:
    # Read after acquiring a row lock: a waiter must not use a pre-wait timestamp.
    return _connection_now(session.connection())


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


@dataclass(frozen=True)
class FollowerLease:
    turn_id: str
    token: str
    expires_at: datetime
    assistant_message_id: str | None = None


@dataclass(frozen=True)
class TerminalCommitResult:
    snapshot: dict
    won: bool


@dataclass(frozen=True)
class TerminalProjection:
    snapshot: dict
    publication_deadline_epoch: float
    history_projected: bool


class TurnTerminalProtocol:
    def __init__(self, turn_store) -> None:
        self.turn_store = turn_store
        self.engine = turn_store.engine

    @staticmethod
    def _row(session: Session, turn_id: str) -> TurnLog:
        row = session.exec(
            select(TurnLog).where(TurnLog.id == turn_id).with_for_update()
        ).first()
        if row is None or row.terminal_protocol_state is None:
            raise FollowerFenceLost("missing or legacy turn is not protocol recovery work")
        return row

    @staticmethod
    def _fence(row: TurnLog, lease: FollowerLease, now: datetime) -> None:
        if (row.id != lease.turn_id or row.follower_owner != lease.token
                or row.follower_lease_until is None
                or _aware(row.follower_lease_until) <= now):
            raise FollowerFenceLost(f"follower no longer owns turn {lease.turn_id}")

    def claim(self, turn_id: str, *, projection: bool = False) -> FollowerLease | None:
        """Claim with a fresh token. An expired owner cannot renew itself."""
        states = PROJECTION_STATES if projection else OBSERVATION_STATES
        with Session(self.engine) as session:
            row = self._row(session, turn_id)
            now = _db_now(session)
            if row.terminal_protocol_state not in states:
                return None
            if row.follower_owner and row.follower_lease_until and _aware(row.follower_lease_until) > now:
                return None
            if projection and row.terminal_history_projected_at and row.terminal_protocol_state == "publication_expired":
                return None
            token = uuid.uuid4().hex
            until = now + timedelta(seconds=LEASE_SECONDS)
            row.follower_owner, row.follower_lease_until = token, until
            session.add(row)
            session.commit()
            return FollowerLease(turn_id, token, until)

    def bind_history_session(self, lease: FollowerLease, *, session_id: str) -> None:
        """Pin one real owner-bound session, before provider/user history writes.

        A cached history-manager ID (which may be a fallback UUID) is not
        existence or ownership evidence. This lookup must succeed in the DB.
        Stop/recovery revokes binding just as it revokes observation.
        """
        if not session_id or session_id != session_id.strip() or len(session_id) > 36:
            raise ValueError("history binding requires a canonical session ID")
        from .turn_admission import lock_target

        with Session(self.engine) as session:
            bot_id = session.exec(select(TurnLog.bot_id).where(TurnLog.id == lease.turn_id)).first()
            if not bot_id:
                raise FollowerFenceLost("history binding requires a turn owner")
            # Keep the target→turn→session ordering used by session admission.
            connection = session.connection()
            lock_target(connection, bot_id)
            row = self._row(session, lease.turn_id)
            self._fence(row, lease, _db_now(session))
            if (row.terminal_protocol_state != "binding" or row.ended_at is not None
                    or row.history_session_id is not None):
                raise FollowerFenceLost("turn no longer accepts its initial history binding")
            lock = " FOR SHARE" if connection.dialect.name == "postgresql" else ""
            owned = connection.execute(text(
                "SELECT id FROM sessions WHERE id=:session AND bot_id=:bot"
                " AND user_id=:user AND status <> 'deleted'" + lock
            ), {"session": session_id, "bot": row.bot_id, "user": row.user_id}).first()
            if owned is None:
                raise ValueError("history session does not exist or has the wrong owner")
            # Session lifecycle may have held its row lock across the lease
            # boundary. Do not bind using the pre-wait database timestamp.
            self._fence(row, lease, _db_now(session))
            row.history_session_id = session_id
            row.terminal_protocol_state = "open"
            session.add(row)
            session.commit()

    def renew(self, lease: FollowerLease) -> FollowerLease:
        with Session(self.engine) as session:
            row = self._row(session, lease.turn_id)
            now = _db_now(session)
            self._fence(row, lease, now)
            if row.terminal_protocol_state == "published":
                raise FollowerFenceLost("published turn no longer needs a follower")
            until = now + timedelta(seconds=LEASE_SECONDS)
            row.follower_lease_until = until
            session.add(row)
            session.commit()
            return FollowerLease(lease.turn_id, lease.token, until, lease.assistant_message_id)

    def write_partial(self, lease: FollowerLease, *, text_value: str, reasoning: str | None = None) -> None:
        with Session(self.engine) as session:
            row = self._row(session, lease.turn_id)
            self._fence(row, lease, _db_now(session))
            if row.terminal_protocol_state != "open" or row.ended_at is not None:
                raise FollowerFenceLost("terminal content is immutable")
            row.response_text = text_value
            if reasoning is not None:
                row.reasoning = reasoning
            session.add(row)
            session.commit()

    def commit(self, lease: FollowerLease, **candidate) -> TerminalCommitResult:
        """Commit once under the current owner, or return the existing winner."""
        return self._commit(lease.turn_id, lease=lease, **candidate)

    def interrupt(self, turn_id: str, *, status: str, **candidate) -> TerminalCommitResult:
        """Stop/reaper authority; never a success or timeout-repair back door."""
        if status not in {"aborted", "cancelled", "timeout", "error"}:
            raise ValueError("interruption must be a failure or cancellation")
        return self._commit(turn_id, lease=None, status=status, **candidate)

    def _commit(
        self, turn_id: str, *, lease: FollowerLease | None,
        status: str, end_reason: str, response_text: str | None = None,
        reasoning: str | None = None, tool_context: str = "",
        attachments: list[dict] | None = None, token_usage: dict | None = None,
        error_text: str | None = None, completion_fields: dict | None = None,
        tts_tail: dict | None = None, audit_key: str | None = None,
        audit: dict | None = None, abort_cause: str | None = None,
        connection=None,
    ) -> TerminalCommitResult:
        if status not in TERMINAL_TURN_STATUSES or not end_reason:
            raise ValueError("terminal commit requires an explicit terminal outcome")
        if connection is not None and (connection.engine is not self.engine or not connection.in_transaction()):
            raise ValueError("terminal commit requires this engine's active outer transaction")
        # Reservation producers already own an admission transaction. A session
        # joined in rollback-only mode may flush, never commit that outer unit.
        with Session(bind=connection if connection is not None else self.engine,
                     join_transaction_mode="rollback_only") as session:
            row = self._row(session, turn_id)
            if row.terminal_snapshot_json is not None:
                return TerminalCommitResult(json.loads(row.terminal_snapshot_json), won=False)
            now = _db_now(session)
            if lease is not None:
                self._fence(row, lease, now)
            if row.terminal_protocol_state not in (*OBSERVATION_STATES, "reserved") or row.ended_at is not None:
                raise FollowerFenceLost("protocol row has no writable open outcome")
            if row.terminal_protocol_state == "reserved" and (
                lease is not None or row.status != "reserved" or row.agent_request_id
                or row.agent_session_key or row.agent_backend or row.follower_owner
                or row.follower_lease_until
            ):
                raise FollowerFenceLost("protocol reservation already has execution ownership")
            unbound = row.terminal_protocol_state in {"binding", "reserved"}
            if not row.assistant_message_id or (not unbound and not row.history_session_id):
                raise ValueError("protocol row has no pinned history identity")
            if unbound and (
                row.history_session_id is not None or status not in {"aborted", "cancelled", "timeout", "error"}
                or row.response_text or row.reasoning or response_text or reasoning
                or tool_context or attachments or token_usage or tts_tail
            ):
                raise ValueError("unbound turn may only commit a pre-inference failure without content")
            if abort_cause is not None and status in {"aborted", "cancelled"}:
                from .turn_termination import stamp_abort_cause
                stamp_abort_cause(row, abort_cause)
            if row.status == "cancelling":
                status, end_reason = "aborted", "aborted"
            if audit_key is not None:
                from .turn_termination import merge_audit
                row.request_json = merge_audit(
                    row.request_json, audit_key, {**(audit or {}), "settled_at": now.isoformat()},
                )
            row.status, row.end_reason, row.ended_at = status, end_reason, now
            if response_text is not None:
                row.response_text = response_text
            if reasoning is not None:
                row.reasoning = reasoning
            if token_usage is not None:
                row.token_usage_json = self.turn_store._encode_token_usage(token_usage, row.model)
            if error_text is not None:
                row.error_text = error_text
            fields = dict(completion_fields or {})
            row.question_id = fields.get("question_id")
            row.animation = fields.get("animation", row.animation)
            row.latency_ms = max(0, (now - _aware(row.created_at)).total_seconds() * 1000)
            from .turn_terminal_effects import record_terminal_effects
            record_terminal_effects(session, row)
            from .usage.balance import balance_update_for_turn
            balance = balance_update_for_turn(
                self.engine, row.token_usage_json, connection=session.connection(),
            )
            completion = {
                **fields, "_type": "turn_complete", "turn_id": row.id,
                "assistant_message_id": row.assistant_message_id,
                "bot_id": row.bot_id, "user_id": row.user_id,
                "session_id": row.history_session_id,
                "status": {"ok": "completed", "aborted": "cancelled"}.get(status, status),
                "end_reason": end_reason, "error_text": row.error_text,
                "response_text": row.response_text or "",
                "token_usage": json.loads(row.token_usage_json) if row.token_usage_json else None,
                "animation": row.animation, "question_id": row.question_id,
                "attachments": attachments, "balance_update": balance,
                "model": row.model, "ts": now.timestamp(),
            }
            tail = None
            if tts_tail is not None and status not in {"cancelled", "aborted"}:
                tail = {
                    **tts_tail, "_type": "tts_delta", "turn_id": row.id,
                    "bot_id": row.bot_id, "user_id": row.user_id,
                }
            snapshot = {
                "version": 1, "turn_id": row.id,
                "assistant_message_id": row.assistant_message_id,
                # messages.id is VARCHAR(36); a literal ":tools" suffix on an
                # assistant UUID would fail on PostgreSQL.
                "tools_message_id": str(uuid.uuid5(
                    uuid.NAMESPACE_URL, f"llm-bawt:terminal-tools:{row.assistant_message_id}",
                )),
                "history_session_id": row.history_session_id,
                "history_disposition": "unbound" if unbound else "bound",
                "history_timestamp": now.timestamp(),
                "response_text": row.response_text or "",
                "reasoning": row.reasoning, "attachments": attachments,
                "tool_context": tool_context, "completion": completion, "tts_tail": tail,
            }
            row.terminal_snapshot_json = _json(snapshot)
            row.terminal_protocol_state = "pending"
            row.terminal_publication_deadline = now + PUBLICATION_HORIZON
            # Terminal commit revokes observation; projection requires a new claim.
            row.follower_owner = row.follower_lease_until = None
            session.add(row)
            session.commit()
            return TerminalCommitResult(json.loads(_json(snapshot)), won=True)

    def load_projection(self, lease: FollowerLease) -> TerminalProjection:
        """Read committed bytes under the projection owner's current fence."""
        with Session(self.engine) as session:
            row = self._row(session, lease.turn_id)
            self._fence(row, lease, _db_now(session))
            if (
                row.terminal_protocol_state not in PROJECTION_STATES
                or row.terminal_snapshot_json is None or row.terminal_publication_deadline is None
            ):
                raise FollowerFenceLost("turn has no committed projection work")
            snapshot = json.loads(row.terminal_snapshot_json)
            completion = snapshot["completion"]
            if (
                snapshot["turn_id"] != row.id or completion["turn_id"] != row.id
                or completion["bot_id"] != row.bot_id or completion["user_id"] != row.user_id
            ):
                raise ValueError("terminal snapshot does not match its durable owner")
            return TerminalProjection(
                snapshot=snapshot,
                publication_deadline_epoch=_aware(row.terminal_publication_deadline).timestamp(),
                history_projected=row.terminal_history_projected_at is not None,
            )

    def mark_history_projected(self, lease: FollowerLease) -> None:
        with Session(self.engine) as session:
            row = self._row(session, lease.turn_id)
            now = _db_now(session)
            self._fence(row, lease, now)
            if not row.terminal_snapshot_json:
                raise ValueError("history cannot precede terminal commit")
            row.terminal_history_projected_at = row.terminal_history_projected_at or now
            if row.terminal_protocol_state == "pending":
                row.terminal_protocol_state = "projected"
            session.add(row)
            session.commit()

    def record_publication(self, lease: FollowerLease, *, result: TerminalPublishResult) -> None:
        """A receipt or expiry never substitutes for the durable history write."""
        with Session(self.engine) as session:
            row = self._row(session, lease.turn_id)
            now = _db_now(session)
            self._fence(row, lease, now)
            if not row.terminal_snapshot_json or row.terminal_publication_deadline is None:
                raise ValueError("publication cannot precede terminal commit")
            if result.status in {TerminalPublishStatus.PUBLISHED, TerminalPublishStatus.ALREADY_PUBLISHED}:
                if not result.completion_stream_id:
                    raise ValueError("published result has no acknowledged stream ID")
                if not row.terminal_history_projected_at:
                    raise ValueError("completion publication must follow history projection")
                row.terminal_published_stream_id = result.completion_stream_id
                row.terminal_published_at = now
                row.terminal_protocol_state = "published"
            elif result.status is TerminalPublishStatus.EXPIRED:
                # Lua's Redis TIME is the publication clock. Rechecking expiry
                # against the DB clock would merely retry on clock skew.
                row.terminal_protocol_state = "publication_expired"
            else:
                raise ValueError("unknown terminal publication result")
            row.follower_owner = row.follower_lease_until = None
            session.add(row)
            session.commit()

    def recoverable_ids(self, *, limit: int = 50) -> list[str]:
        """Bounded discovery only; every returned row still needs its own claim."""
        if not 1 <= limit <= 1000:
            raise ValueError("recovery batch limit must be between 1 and 1000")
        with Session(self.engine) as session:
            now = _db_now(session)
            return list(session.exec(select(TurnLog.id).where(
                TurnLog.terminal_protocol_state.in_((*OBSERVATION_STATES, *PROJECTION_STATES)),
                or_(TurnLog.follower_lease_until.is_(None), TurnLog.follower_lease_until <= now),
                or_(TurnLog.terminal_protocol_state != "publication_expired",
                    TurnLog.terminal_history_projected_at.is_(None)),
            # Never-attempted and least-recently-leased work first. A broken
            # old row must not monopolize every bounded batch after its retry.
            ).order_by(TurnLog.follower_lease_until.asc().nulls_first(),
                       TurnLog.created_at, TurnLog.id).limit(limit)).all())
