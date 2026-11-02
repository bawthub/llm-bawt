"""Trusted failure-notice lineage and bounded error codes (TASK-1946).

Only admission writes ORIGIN from a verified turn capability. Neither a bridge
session key nor caller metadata is authority for the callback's destination.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .task_turn_context import TaskTurnContext

ORIGIN = "_failure_origin"
DECISION = "_failure_notice"
NOTICE_KIND = "DELIVERY_FAILED"
NOTICE_KEY_PREFIX = "failure-notice:"


@dataclass(frozen=True)
class FailureOrigin:
    bot_id: str
    user_id: str
    session_id: str
    turn_id: str
    trigger_message_id: str

    @classmethod
    def capture(cls, caller: TaskTurnContext) -> dict[str, Any]:
        return {"version": 1, **{name: getattr(caller, name) for name in cls.__dataclass_fields__}}

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> FailureOrigin | None:
        raw = (row.get("metadata_json") or {}).get(ORIGIN)
        if not isinstance(raw, dict) or type(raw.get("version")) is not int or raw["version"] != 1:
            return None
        fields = {name: raw.get(name) for name in cls.__dataclass_fields__}
        if any(not isinstance(value, str) or not value or len(value) > 128 for value in fields.values()):
            return None
        origin = cls(**fields)
        metadata = row.get("metadata_json") or {}
        # Existing restart children are system-authored, with server-stamped
        # retry provenance. Preserve their original initiator, not 'system'.
        from .bot_identities import SYSTEM_BOT_ID

        retry_author = metadata.get("retry_original_author")
        author_matches = row.get("author_entity_id") == origin.bot_id or (
            row.get("author_entity_id") == SYSTEM_BOT_ID
            and metadata.get("_request_identity_producer") == "retry"
            and retry_author == {"entity_type": "bot", "entity_id": origin.bot_id, "status": "canonical"}
        )
        if (origin.bot_id != row["sender_bot_id"] or origin.turn_id != row.get("sender_turn_id")
                or row.get("author_entity_type") != "bot" or not author_matches):
            return None
        return origin


def failure_code(error: str | None) -> str:
    """Classify, never quote: provider errors can contain secrets/tool arguments.

    Even the fallback is a fixed code, not the first line/class-like token of an
    arbitrary provider body. The delivery/turn IDs allow authorized drilldown.
    """
    lowered = (error or "").lower()
    if "chatgpteventtimeout" in lowered or "event timeout" in lowered:
        return "upstream_event_timeout"
    if "context" in lowered and any(part in lowered for part in ("overflow", "too long", "window")):
        return "context_limit"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "recovery window" in lowered or "ambiguous" in lowered:
        return "outcome_unknown"
    return "worker_error"
