"""Server-mode short-term (history) manager adapter for the memory client (TASK-1222).

``client.py`` re-exports ``_MCPShortTermManager``; import it from there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_bawt.mcp_server.client import MemoryClient


class _MCPShortTermManager:
    """Adapter matching PostgreSQLShortTermManager but routed via MCP tools."""

    def __init__(self, memory_client: MemoryClient):
        self._memory_client = memory_client

    def add_message(
        self,
        role: str,
        content: str,
        timestamp: float | None = None,
        message_id: str | None = None,
        attachments: list[dict] | None = None,
        reasoning: str | None = None,
        session_id: str | None = None,
        author=None,
    ) -> str:
        # ``reasoning`` (TASK-301) is forwarded through the MCP write chain so
        # server-mode deployments persist it too (display-only on the row).
        # ``session_id`` (TASK-251): explicit thread override — None keeps the
        # server-side active-thread resolution (the primary mode).
        msg = self._memory_client.add_message(
            role=role,
            content=content,
            timestamp=timestamp,
            message_id=message_id,
            attachments=attachments,
            reasoning=reasoning,
            session_id=session_id,
            author_entity_type=(author.entity_type if author else None),
            author_entity_id=(author.entity_id if author else None),
        )
        return msg.id

    def get_messages(
        self,
        since_minutes: int | None = None,
    ) -> list:
        # NOTE: despite name, since_minutes is seconds for backward compatibility.
        from llm_bawt.models.message import Message

        rows = self._memory_client.get_messages(since_seconds=since_minutes)
        return [
            Message(
                role=r.get("role", ""),
                content=r.get("content", ""),
                timestamp=r.get("timestamp", 0.0),
                db_id=r.get("id"),
                session_id=r.get("session_id"),  # TASK-284: preserve thread id
                author_entity_type=(r.get("author") or {}).get("entity_type"),
                author_entity_id=(r.get("author") or {}).get("entity_id"),
            )
            for r in rows
        ]

    def load_session_scoped(
        self, since_minutes: int | None = None, session_id: str | None = None
    ) -> list | None:
        """Session-scoped history read (TASK-284 step 12 / TASK-251).

        Loads ONE durable thread's raw bubbles composed with the bot's rolling
        summary husks (continuity) — the same content shape the legacy
        summary-aware path produced, but with raw scoped to a single thread.

        ``session_id`` (TASK-251): the EXPLICITLY selected thread — this is the
        only caller-facing mode now; a request carrying session_id views or
        continues that thread. When None, falls back to the active thread
        (legacy step-12 behaviour; no longer used as a default context source
        anywhere — sessions are metadata, not a context filter).

        Returns a chronologically-ordered ``list[Message]``, or ``None`` when
        no thread can be resolved (the caller falls back to the continuous
        load — never a silent empty context).
        """
        from llm_bawt.models.message import Message

        if not session_id:
            active = self._memory_client.get_active_session()
            if not active or not active.get("id"):
                return None
            session_id = active["id"]

        raw_rows = self._memory_client.get_messages(
            session_id=session_id, since_seconds=since_minutes,
            exclude_summarized=True,
        )
        summary_rows = self._memory_client.get_messages(summaries_only=True)

        merged = [
            Message(
                role=r.get("role", ""),
                content=r.get("content", ""),
                timestamp=r.get("timestamp", 0.0) or 0.0,
                db_id=r.get("id"),
                session_id=r.get("session_id"),
                author_entity_type=(r.get("author") or {}).get("entity_type"),
                author_entity_id=(r.get("author") or {}).get("entity_id"),
            )
            for r in (list(raw_rows) + list(summary_rows))
        ]
        merged.sort(key=lambda m: m.timestamp or 0.0)
        return merged

    def get_session(self, session_id: str) -> dict | None:
        """Return one session row by id for per-thread SDK-key resolution."""
        return self._memory_client.get_session(session_id)

    def get_active_session(
        self, bot_id: str | None = None, user_id: str | None = None,
    ) -> dict | None:
        """Return this adapter's active thread.

        Optional arguments keep parity with the embedded manager; ownership is
        already bound into ``MemoryClient`` and mismatched overrides are not
        used by the app's request-local history manager.
        """
        return self._memory_client.get_active_session()

    def get_or_create_active_session(
        self, bot_id: str | None = None, user_id: str | None = None,
    ) -> str:
        """Return/create this adapter's active thread for SDK write-back."""
        return self._memory_client.get_or_create_active_session()

    def rotate_session(
        self, idempotency_key: str | None = None, *, accept_pointer: bool = False,
    ) -> str | None:
        """Rotate the active thread; accepted /new stamps the bot-wide pointer."""
        return self._memory_client.rotate_session(
            accept_pointer=accept_pointer,
            **({"idempotency_key": idempotency_key} if idempotency_key else {})
        )

    def clear(self) -> bool:
        deleted = self._memory_client.clear_messages()
        return deleted >= 0

    def remove_last_message_if_partial(self, role: str) -> bool:
        return bool(self._memory_client.remove_last_message_if_partial(role=role))

    def get_messages_for_summary(self, summary_id: str) -> list:
        from llm_bawt.models.message import Message

        rows = self._memory_client.get_messages_for_summary(summary_id)
        return [
            Message(
                role=row.get("role", ""),
                content=row.get("content", ""),
                timestamp=row.get("timestamp", 0.0),
                db_id=row.get("id"),
                session_id=row.get("session_id"),  # TASK-284: preserve thread id
            )
            for row in rows
        ]

    def mark_messages_recalled(self, message_ids: list[str]) -> int:
        return int(self._memory_client.mark_messages_recalled(message_ids))
