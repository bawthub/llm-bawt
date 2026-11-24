"""Participant-scoped receipt reads; ordinary LAN REST remains operator-owned."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from llm_bawt.bot_access import BotAccessError
from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.inter_bot_delivery_query import IN_FLIGHT_STATUSES, parse_statuses
from llm_bawt.utils.bounded_worker import BoundedWorker
from llm_bawt.utils.config import Config

_WORKER = BoundedWorker(4, "delivery-reads")


class ScopedDeliveryReader:
    def __init__(self, access, store=None):
        self.allowed = frozenset(p.bot_id for p in access.aggregate_scope().partitions)
        self._store = store

    @property
    def store(self):
        # Construct only inside the bounded worker and sanitized error boundary.
        if self._store is None:
            self._store = InterBotDeliveryStore(Config(), provision_schema=False)
        if self._store.engine is None:
            raise BotAccessError("source_unavailable")
        return self._store

    async def get(self, delivery_id: str) -> dict:
        try:
            record = await _WORKER.run(self._get, delivery_id)
            return record.to_api() if record else {"delivery_id": delivery_id, "error": "not found"}
        except Exception:
            raise BotAccessError("source_unavailable") from None

    def _get(self, delivery_id):
        return self.store.get(delivery_id, allowed_bot_ids=self.allowed)

    async def list(self, *, bot_id, sender_bot_id, target_bot_id, status, active_only,
                   task_id, since_minutes, include_chat, limit) -> dict:
        if active_only and status:
            return {"error": "Use either active_only or status, not both"}
        try:
            statuses = IN_FLIGHT_STATUSES if active_only else parse_statuses(status)
        except ValueError as error:
            return {"error": str(error)}
        since = (datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
                 if since_minutes is not None and since_minutes > 0 else None)
        try:
            rows = await _WORKER.run(self._list, dict(
                bot_id=bot_id, sender_bot_id=sender_bot_id, target_bot_id=target_bot_id,
                status=statuses, task_id=task_id, since=since, limit=limit,
                exclude_prompt_origin=True, exclude_user_chat=not include_chat,
                allowed_bot_ids=self.allowed,
            ))
            return {"deliveries": [row.to_compact() for row in rows], "total": len(rows)}
        except Exception:
            raise BotAccessError("source_unavailable") from None

    def _list(self, kwargs):
        return self.store.list(**kwargs)
