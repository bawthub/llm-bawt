"""Durable /new successor links, separate from ordinary archived task threads."""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from sqlalchemy import text


class SessionResetLineage:
    @staticmethod
    def delivery_successor_id(delivery_id: str) -> str:
        """One deterministic id for rotation and claimed-delivery lookup."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"delivery-new:{delivery_id}"))

    @staticmethod
    def rotate(
        engine, *, bot_id: str, user_id: str | None, new_id: str,
        working_memory_mode: str,
        idempotency_key: str | None = None, connection=None,
        context_lifecycle: dict | None = None, accept_pointer: bool = False,
    ) -> bool:
        """Archive + successor link in one transaction; keyed retries reuse the successor.

        The delivery key is stamped on the successor in the same transaction as
        archiving the predecessor. A crash cannot leave a rotated thread without
        its retry proof. An unkeyed /new retains its original unique-id behavior.

        ``working_memory_mode`` (``pinned`` or ``suppressed``, no default) is
        written to the successor's ``session_metadata.working_memory`` in the
        same INSERT (docs/WORKING_MEMORY.md 10.1). Each caller passes its
        origin's mode. A keyed replay keeps the mode the successor was born with.
        """
        from ..working_memory.session_state import KEY, successor_metadata

        working_memory = successor_metadata(working_memory_mode)
        reset_at = time.time()
        with (nullcontext(connection) if connection is not None else engine.begin()) as conn:
            postgres = conn.dialect.name == "postgresql"
            if postgres:
                from ..service.turn_admission import lock_target

                lock_target(conn, bot_id)
                conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:bot), hashtext(:user))"),
                             {"bot": bot_id, "user": user_id or ""})
            params = {"bot": bot_id, "user": user_id}
            if idempotency_key:
                existing = conn.execute(text("""
                    SELECT bot_id, user_id, status, session_metadata FROM sessions WHERE id=:id
                """ + (" FOR UPDATE" if postgres else "")), {"id": new_id}).mappings().first()
                if existing is not None:
                    metadata = SessionResetLineage.metadata(existing["session_metadata"])
                    if (existing["bot_id"] != bot_id or existing["user_id"] != user_id
                            or existing["status"] == "deleted"
                            or metadata.get("rotation_delivery_id") != idempotency_key):
                        raise ValueError("Delivery rotation successor belongs to another owner")
                    return False
            rows = conn.execute(text("""
                SELECT id, session_metadata FROM sessions
                WHERE bot_id=:bot AND status='active' AND ended_at IS NULL
                  AND (user_id=:user OR (:user IS NULL AND user_id IS NULL))
            """ + (" FOR UPDATE" if postgres else "")), params).mappings().all()
            meta_expr = "CAST(:meta AS JSONB)" if postgres else ":meta"
            for row in rows:
                metadata = SessionResetLineage.metadata(row["session_metadata"])
                metadata["reset_successor_id"] = new_id
                # Absolute epoch (sessions timestamps are naive DB-local) so
                # callers can tell "reset while pending" from "work started in
                # an already-archived thread".
                metadata["reset_at"] = reset_at
                conn.execute(text(f"""
                    UPDATE sessions SET ended_at=CURRENT_TIMESTAMP,
                        archived_at=CURRENT_TIMESTAMP, status='archived',
                        session_metadata={meta_expr} WHERE id=:id
                """), {"id": row["id"], "meta": json.dumps(metadata)})
            metadata = {"reset_predecessor_ids": [row["id"] for row in rows], KEY: working_memory}
            if idempotency_key:
                metadata["rotation_delivery_id"] = idempotency_key
            if context_lifecycle is not None:
                metadata["context_lifecycle"] = context_lifecycle
            conn.execute(text(f"""
                INSERT INTO sessions (id, bot_id, user_id, started_at, status, session_metadata)
                VALUES (:id, :bot, :user, CURRENT_TIMESTAMP, 'active', {meta_expr})
            """), {**params, "id": new_id, "meta": json.dumps(metadata)})
            if accept_pointer:
                from ..bot_session_state import BotSessionState

                BotSessionState.accept(conn, bot_id=bot_id, session_id=new_id)
            return True

    @staticmethod
    def metadata(value) -> dict:
        if isinstance(value, str):
            value = json.loads(value)
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("Invalid session reset metadata")
        return dict(value)

    @classmethod
    def resolve(cls, read_session: Callable, origin: str, *, bot_id: str, user_id: str,
                started_at: float | None = None) -> str:
        """Follow explicit /new lineage only; never pick an arbitrary active thread.

        ``started_at`` (epoch) is when the work bound to ``origin`` began. If the
        origin was already reset before then, the user deliberately reopened the
        archived thread, so it stays the target instead of jumping to /new.
        """
        current = origin
        seen: set[str] = set()
        previous = None
        while current not in seen:
            seen.add(current)
            row = read_session(current)
            if not row or row.get("bot_id") != bot_id or row.get("user_id") != user_id:
                raise ValueError("Approval continuation session is missing or belongs to another owner")
            if row.get("status") == "deleted":
                raise ValueError("Approval continuation session was deleted")
            metadata = cls.metadata(row.get("session_metadata"))
            if previous and previous not in metadata.get("reset_predecessor_ids", []):
                raise ValueError("Approval continuation reset lineage is inconsistent")
            successor = metadata.get("reset_successor_id")
            if not successor:
                return current
            reset_at = metadata.get("reset_at")
            if (previous is None and started_at is not None
                    and isinstance(reset_at, (int, float)) and reset_at <= started_at):
                return current
            if not isinstance(successor, str):
                raise ValueError("Invalid approval continuation reset successor")
            previous, current = current, successor
        raise ValueError("Approval continuation reset lineage contains a cycle")
