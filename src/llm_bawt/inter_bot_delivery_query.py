"""SQL filter for durable delivery listings (TASK-1239).

Every filter is applied in SQL before ``LIMIT`` so excluded rows can never
crowd selected rows out of a capped page.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

DELIVERY_STATUSES = frozenset({
    "QUEUED", "STEERING", "DISPATCHING", "DELIVERED", "FAILED", "CANCELLED",
})
IN_FLIGHT_STATUSES = ("QUEUED", "STEERING", "DISPATCHING")

# Metadata keys that mark a row as private to another producer or as a user's
# own chat traffic routed through the delivery queue rather than bot-to-bot.
# ``exclude_user_chat`` also drops every self-addressed row (sender == target).
PROMPT_ORIGIN_KEYS = ("prompt_schedule",)
USER_CHAT_KEYS = ("ordinary_chat", "maintenance_user_queue")


def parse_statuses(value: str | Iterable[str] | None) -> tuple[str, ...]:
    """Normalize one status, a comma-separated string, or an iterable.

    Unknown values raise ``ValueError`` instead of silently matching nothing.
    """
    if value is None:
        return ()
    parts = value.split(",") if isinstance(value, str) else list(value)
    statuses: list[str] = []
    for part in parts:
        status = str(part).strip().upper()
        if not status:
            continue
        if status not in DELIVERY_STATUSES:
            raise ValueError(
                f"unknown delivery status {status!r}; expected one of "
                + ", ".join(sorted(DELIVERY_STATUSES))
            )
        if status not in statuses:
            statuses.append(status)
    return tuple(statuses)


def _slug(value: str | None) -> str | None:
    cleaned = (value or "").strip().lower()
    return cleaned or None


@dataclass(frozen=True)
class DeliveryListFilter:
    sender_bot_id: str | None = None
    target_bot_id: str | None = None
    bot_id: str | None = None  # sender OR target
    statuses: tuple[str, ...] = ()
    idempotency_key: str | None = None
    task_id: str | None = None
    since: datetime | None = None
    exclude_prompt_origin: bool = False
    exclude_user_chat: bool = False
    allowed_bot_ids: frozenset[str] | None = None

    def where(self, dialect: str) -> tuple[str, dict[str, Any]]:
        clauses: list[str] = []
        params: dict[str, Any] = {}

        if self.allowed_bot_ids is not None:
            names = []
            for index, bot in enumerate(sorted(self.allowed_bot_ids)):
                name = f"allowed_bot{index}"
                names.append(f":{name}")
                params[name] = bot
            allowed = ", ".join(names)
            clauses.append(
                f"sender_bot_id IN ({allowed}) AND target_bot_id IN ({allowed})"
                if names else "1=0"
            )

        excluded = []
        if self.exclude_prompt_origin:
            excluded.extend(PROMPT_ORIGIN_KEYS)
        if self.exclude_user_chat:
            excluded.extend(USER_CHAT_KEYS)
        for key in excluded:
            if dialect == "postgresql":
                clauses.append(f"NOT (COALESCE(metadata_json, '{{}}'::jsonb) ? '{key}')")
            else:
                clauses.append(f"json_type(metadata_json, '$.{key}') IS NULL")
        if self.exclude_user_chat:
            # A bot's own queue (user chat, self_model continuations, anything
            # untagged) is never bot-to-bot traffic, whatever its metadata.
            clauses.append("sender_bot_id <> target_bot_id")

        if sender := _slug(self.sender_bot_id):
            clauses.append("sender_bot_id=:sender")
            params["sender"] = sender
        if target := _slug(self.target_bot_id):
            clauses.append("target_bot_id=:target")
            params["target"] = target
        if either := _slug(self.bot_id):
            clauses.append("(sender_bot_id=:either OR target_bot_id=:either)")
            params["either"] = either
        if self.statuses:
            names = []
            for index, status in enumerate(self.statuses):
                names.append(f":status{index}")
                params[f"status{index}"] = status
            clauses.append(f"status IN ({', '.join(names)})")
        if self.idempotency_key and self.idempotency_key.strip():
            clauses.append("idempotency_key=:idem")
            params["idem"] = self.idempotency_key.strip()
        if self.task_id and self.task_id.strip():
            clauses.append("task_id=:task_id")
            params["task_id"] = self.task_id.strip()
        if self.since is not None:
            clauses.append("created_at>=:since")
            params["since"] = self.since
        return (" AND ".join(clauses) or "1=1"), params
