"""Owner-scoped message mutations; semantic memories have a separate lifetime."""
from __future__ import annotations

from sqlalchemy import text

from .short_term_message_reads import history_owner_clause


class ShortTermMessageMutationsMixin:
    def _message_mutation_owner(self, user_id: str | None) -> str:
        owner = self.user_id if user_id is None else user_id
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("History mutation requires user_id")
        return owner

    def clear_messages(self, *, user_id: str | None = None) -> int:
        """Delete only this user's messages, including their rolling summaries."""
        owner = self._message_mutation_owner(user_id)
        params: dict = {}
        clause = history_owner_clause(params, bot_id=self.bot_id, user_id=owner)
        with self._backend.engine.begin() as conn:
            result = conn.execute(text(
                f"DELETE FROM {self._backend._messages_table_name} WHERE {clause}"
            ), params)
            return result.rowcount

    def clear(self, *, user_id: str | None = None) -> bool:
        """Clear conversation rows, never the backend's distilled memories."""
        return self.clear_messages(user_id=user_id) >= 0

    def remove_last_message_if_partial(self, role: str, *, user_id: str | None = None) -> bool:
        """Remove the user's newest message only if its role matches.

        Filter before choosing the newest row, and bind ownership again on the
        deletion. One statement avoids a select/delete role race. Session-less
        legacy rows are not assigned to an arbitrary caller.
        """
        owner = self._message_mutation_owner(user_id)
        params: dict = {"role": role}
        clause = history_owner_clause(params, bot_id=self.bot_id, user_id=owner)
        table = self._backend._messages_table_name
        with self._backend.engine.begin() as conn:
            row = conn.execute(text(f"""
                DELETE FROM {table}
                WHERE id = (SELECT id FROM {table} WHERE {clause}
                            ORDER BY timestamp DESC, id DESC LIMIT 1)
                  AND role = :role AND {clause}
                RETURNING id
            """), params).first()
            return row is not None
