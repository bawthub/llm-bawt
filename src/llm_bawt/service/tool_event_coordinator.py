"""Durable tool-event coordination for live chat turns."""

from __future__ import annotations

import time
import uuid
from typing import Any

from agent_bridge.tool_results import payload_from_event

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

    def executing(self, event: dict[str, Any]) -> dict[str, Any]:
        self.store.mark_executing(
            turn_id=event.get("turn_id"),
            call_id=event.get("call_id"),
            tool_use_id=event.get("tool_use_id"),
            exec_started_at=event.get("ts"),
        )
        public = dict(event)
        public["_tool_persisted"] = True
        return public

    def end(self, event: dict[str, Any]) -> dict[str, Any]:
        raw_payload = event.get("tool_result_payload")
        payload = payload_from_event(raw_payload, event.get("result"))
        skill_tokens = (
            raw_payload.get("skill_context_tokens")
            if event.get("tool_name") == "Skill" and not event.get("is_error")
            and isinstance(raw_payload, dict)
            else None
        )
        if not isinstance(skill_tokens, int) or isinstance(skill_tokens, bool) or skill_tokens <= 0:
            skill_tokens = None
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
            skill_context_tokens=skill_tokens,
        )
        public = dict(event)
        public.pop("tool_result_payload", None)
        public["result"] = payload.preview
        public["result_meta"] = payload.result_meta(record_id=record_id, available=available)
        if skill_tokens is not None:
            public["result_meta"]["skill_context_tokens"] = skill_tokens
        public["_tool_persisted"] = True
        return public

    def record_completed(
        self,
        details: list[dict[str, Any]],
        *,
        turn_id: str | None,
        trigger_message_id: str | None,
        bot_id: str | None,
        user_id: str | None,
    ) -> None:
        """Persist calls an in-process tool loop already ran (TASK-1775).

        Non-streaming chat turns run ``query_with_tools`` to completion and
        only hand back per-call details; replay each as start + end so they
        land through the same writer as live calls.
        """
        for detail in details:
            call_id = detail.get("call_id") or f"call-{uuid.uuid4().hex}"
            arguments = detail.get("parameters", detail.get("arguments"))
            base = {
                "_type": "tool_event",
                "turn_id": turn_id,
                "trigger_message_id": trigger_message_id,
                "bot_id": bot_id,
                "user_id": user_id,
                "tool_name": detail.get("tool") or "unknown",
                "arguments": arguments if isinstance(arguments, dict) else {},
                "call_id": call_id,
                "iteration": int(detail.get("iteration") or 1),
                "ts": time.time(),
            }
            self.start({**base, "event": "tool_start"})
            self.end({**base, "event": "tool_end", "result": detail.get("result") or ""})
