"""Durable tool-event coordination for live chat turns."""

from __future__ import annotations

from typing import Any

from agent_bridge.tool_results import ToolResultPayload, payload_from_event

from ..media.serializers import attachment_refs
from .tool_call_store import ToolCallStore


class ToolEventCoordinator:
    """Persist canonical tool rows before shaping bounded public events."""

    def __init__(self, engine) -> None:
        self.store = ToolCallStore(engine)

    def start(self, event: dict[str, Any]) -> dict[str, Any]:
        record_id = self.store.save_start(
            turn_id=event.get("turn_id"),
            bot_id=event.get("bot_id"),
            user_id=event.get("user_id"),
            call_id=event.get("call_id"),
            tool_name=event.get("tool_name") or "unknown",
            arguments=event.get("arguments") if isinstance(event.get("arguments"), dict) else {},
            iteration=int(event.get("iteration") or 1),
            started_at=event.get("ts"),
            text_offset=event.get("text_offset"),
            tool_use_id=event.get("tool_use_id"),
            parent_tool_use_id=event.get("parent_tool_use_id"),
        )
        public = dict(event)
        if record_id is not None:
            public["record_id"] = record_id
        public["_tool_persisted"] = True
        return public

    def end(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = payload_from_event(event.get("tool_result_payload"), event.get("result"))
        record_id, available = self.store.save_result(
            turn_id=event.get("turn_id"),
            call_id=event.get("call_id"),
            tool_use_id=event.get("tool_use_id"),
            tool_name=event.get("tool_name") or "unknown",
            bot_id=event.get("bot_id"),
            user_id=event.get("user_id"),
            payload=payload,
            ended_at=event.get("ts"),
            is_error=event.get("is_error"),
            iteration=int(event.get("iteration") or 1),
            parent_tool_use_id=event.get("parent_tool_use_id"),
            # TASK-977: the live event carries enriched envelopes; persist
            # only the refs so /v1/tool-calls can re-enrich on reload.
            attachments=attachment_refs(event.get("attachments")) or None,
        )
        public = dict(event)
        public.pop("tool_result_payload", None)
        public["result"] = payload.preview
        public["result_meta"] = payload.result_meta(record_id=record_id, available=available)
        public["_tool_persisted"] = True
        return public
