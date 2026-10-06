"""Durable /new successor links, separate from ordinary archived task threads."""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from sqlalchemy import text


class SessionResetLineage:
    @staticmethod
    def rotate(engine, *, bot_id: str, user_id: str | None, new_id: str) -> None:
        """Archive + successor link + fresh thread commit as one transaction."""
        reset_at = time.time()
        with engine.begin() as conn:
            postgres = conn.dialect.name == "postgresql"
            if postgres:
                conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:bot), hashtext(:user))"),
                             {"bot": bot_id, "user": user_id or ""})
            params = {"bot": bot_id, "user": user_id}
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
            metadata = {"reset_predecessor_ids": [row["id"] for row in rows]}
            conn.execute(text(f"""
                INSERT INTO sessions (id, bot_id, user_id, started_at, status, session_metadata)
                VALUES (:id, :bot, :user, CURRENT_TIMESTAMP, 'active', {meta_expr})
            """), {**params, "id": new_id, "meta": json.dumps(metadata)})

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
