"""Session forwarding for MemoryStorage.

Behavior-preserving extraction; the facade owns manager caching and configuration.
"""

from __future__ import annotations

from llm_bawt.bot_access import BotAccessError
from llm_bawt.bot_access.context import current_bot_access
from llm_bawt.bot_access.service import SERVICE_SELF_STORAGE_OPS, is_service_self


class StorageSessionsMixin:
    """Shared-session operations using the facade's short-term manager."""

    @staticmethod
    def _session_owner_scope(bot_id: str, operation: str) -> dict:
        access = current_bot_access()
        if access is None:
            return {}
        target = access.require_target(bot_id)
        service_filter = operation in SERVICE_SELF_STORAGE_OPS and is_service_self(access, target)
        if access.caller is not None and access.caller.user_id is None and not service_filter:
            raise BotAccessError("access_denied")
        user = None
        if target in access.private_bots and not service_filter:
            user = access.caller.user_id
            if not user:
                raise BotAccessError("caller_unavailable")
        return {"bot_id": target, "user_id": user}

    async def close_session(
        self,
        session_id: str,
        bot_id: str = "default",
    ) -> bool:
        """Mark a session row archived (sets ended_at + status='archived').

        Returns True iff a row was updated. Idempotent: closing an already-
        closed session returns False without error.
        """
        owner = self._session_owner_scope(bot_id, "close_session")
        manager = self.get_short_term_manager(bot_id, provision_schema=True)
        return manager.close_session(session_id, **owner)

    async def get_session(
        self,
        session_id: str,
        bot_id: str = "default",
    ) -> dict | None:
        """Read one session; guarded calls bind bot and applicable user in SQL."""
        owner = self._session_owner_scope(bot_id, "get_session")
        manager = self.get_short_term_manager(bot_id)
        return manager.get_session(session_id, **owner)

    async def list_sessions(
        self,
        bot_id: str = "default",
        since: float | str | None = None,
        status: str | None = None,
        limit: int = 50,
        user_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[dict]:
        """List sessions for a bot, newest first.

        Pass `bot_id=""` to query across all bots. Pass ``user_id`` (TASK-284)
        to scope to one user's threads; ``None`` leaves the user dimension
        unfiltered. Soft-deleted sessions are excluded unless
        ``include_deleted=True`` or an explicit ``status='deleted'`` filter is
        given (TASK-250).
        """
        # Use a default manager engine, but pass the requested bot through
        # to the query (manager is bot-scoped, the table is not).
        access = current_bot_access()
        scoped = {}
        manager_bot = bot_id or "default"
        if access is not None:
            service_filter = "list_sessions" in SERVICE_SELF_STORAGE_OPS and is_service_self(access, bot_id)
            if access.caller is not None and access.caller.user_id is None and not service_filter:
                raise BotAccessError("access_denied")
            scope = access.aggregate_scope(bot_id or None)
            if not scope.partitions:
                return []
            scoped["allowed_bot_ids"] = frozenset(p.bot_id for p in scope.partitions)
            manager_bot = scope.partitions[0].bot_id
            if not service_filter and any(p.authority == "caller" for p in scope.partitions):
                if not scope.user_id:
                    raise BotAccessError("caller_unavailable")
                if user_id is not None and user_id != scope.user_id:
                    raise BotAccessError("access_denied")
                user_id = scope.user_id
        engine_manager = self.get_short_term_manager(manager_bot)
        return engine_manager.list_sessions(
            bot_id=bot_id,
            since=since,
            status=status,
            limit=limit,
            user_id=user_id,
            include_deleted=include_deleted,
            **scoped,
        )

    async def get_active_session(
        self,
        bot_id: str = "default",
        user_id: str | None = None,
    ) -> dict | None:
        """Return the most-recent active session for a (bot, user), or None.

        TASK-284: ``user_id`` scopes to that user's active thread; ``None``
        preserves the legacy bot-only lookup.
        """
        manager = self.get_short_term_manager(bot_id)
        return manager.get_active_session(bot_id=bot_id, user_id=user_id)

    async def rotate_session(
        self,
        bot_id: str = "default",
        user_id: str | None = None,
        idempotency_key: str | None = None,
        accept_pointer: bool = False,
        *, working_memory_mode: str,
    ) -> str:
        """Close the active thread for (bot, user) and open a new one.

        TASK-284: the non-destructive ``/new`` primitive — old rows untouched,
        the previous session row flips to ``status='archived'``. Returns the
        new active session id. Close+open run in one transaction. The caller
        passes the successor's working-memory mode (no default).
        """
        manager = self.get_short_term_manager(bot_id, provision_schema=True)
        return manager.rotate_session(
            bot_id=bot_id, user_id=user_id, accept_pointer=accept_pointer,
            working_memory_mode=working_memory_mode,
            **({"idempotency_key": idempotency_key} if idempotency_key else {}),
        )

    async def create_inactive_session(
        self,
        bot_id: str = "default",
        user_id: str | None = None,
        session_metadata: dict | None = None,
    ) -> str:
        """Mint a session row that is born archived (TASK-702).

        Background dispatch uses this to open a dedicated task thread
        without rotating (closing) the user's currently active
        conversation. The one-active-per-(bot,user) invariant is untouched.
        The returned session is fully owned and reachable via explicit
        ``session_id`` continuation.
        """
        manager = self.get_short_term_manager(bot_id, provision_schema=True)
        return manager.create_inactive_session(
            bot_id=bot_id,
            user_id=user_id,
            session_metadata=session_metadata,
        )

    async def activate_session(
        self,
        session_id: str,
        bot_id: str = "default",
        user_id: str | None = None,
    ) -> bool:
        """Make an existing thread the active one for (bot, user).

        TASK-284: the thread-switch primitive. Returns ``False`` if the target
        doesn't exist or belongs to a different (bot, user).
        """
        manager = self.get_short_term_manager(bot_id, provision_schema=True)
        return manager.activate_session(session_id, bot_id=bot_id, user_id=user_id)

    async def get_or_create_active_session(
        self,
        bot_id: str = "default",
        user_id: str | None = None,
    ) -> str:
        """Return the active thread id for (bot, user), creating one if none.

        TASK-284 step 15: used by the provider↔thread mirror so a freshly
        minted provider session always has a durable DB thread to map onto.
        """
        manager = self.get_short_term_manager(bot_id, provision_schema=True)
        return manager.get_or_create_active_session(bot_id=bot_id, user_id=user_id)

    async def set_session_status(
        self,
        session_id: str,
        status: str,
        bot_id: str = "default",
    ) -> bool:
        """Set a session's lifecycle status: 'archived' or 'deleted' (TASK-250).

        Soft-delete keeps the messages; the row just leaves default listings
        and its deep-links return 410. Reactivation goes through
        :meth:`activate_session`, never here.
        """
        manager = self.get_short_term_manager(bot_id, provision_schema=True)
        return manager.set_session_status(session_id, status)

    async def update_session_metadata(
        self,
        session_id: str,
        patch: dict,
        bot_id: str = "default",
    ) -> bool:
        """Shallow-merge ``patch`` into a session row's ``session_metadata``.

        Records canonical per-backend SDK key/model maps on the durable thread.
        """
        manager = self.get_short_term_manager(bot_id, provision_schema=True)
        return manager.update_session_metadata(session_id, patch)
