"""Durable tool-event coordination for live chat turns."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from agent_bridge.tool_results import payload_from_event

from ..media.serializers import attachment_refs
from .llm_response_store import LlmResponseStore
from .tool_call_store import ToolCallStore
from .worker_lifecycle_store import WorkerLifecycleStore

logger = logging.getLogger(__name__)


class ToolEventCoordinator:
    """Persist canonical tool rows before shaping bounded public events."""

    def __init__(self, engine) -> None:
        self.store = ToolCallStore(engine)
        self.responses = LlmResponseStore(engine)
        self.workers = WorkerLifecycleStore(engine)

    def worker(self, event: dict[str, Any]) -> dict[str, Any]:
        """Persist one sub-agent lifecycle event (TASK-2169).

        Publishes the raw event plus the resolved spawning ``tool_use_id``, so
        status frames that lack it still reach the right worker live. A
        persistence failure never drops the live event.
        """
        try:
            record = self.workers.record(event)
        except Exception:
            logger.exception("Worker lifecycle upsert failed turn=%s", event.get("turn_id"))
            return event
        if record is None:
            return event
        public = dict(event)
        if not public.get("tool_use_id") and record.get("tool_use_id"):
            public["tool_use_id"] = record["tool_use_id"]
        return public

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
            exec_tracked=event.get("exec_tracked") is True,
            text_offset=event.get("text_offset"),
            tool_use_id=event.get("tool_use_id"),
            parent_tool_use_id=event.get("parent_tool_use_id"),
            response_id=event.get("response_id"),
        )
        public = dict(event)
        if record_id is not None:
            public["record_id"] = record_id
        public["_tool_persisted"] = True
        return public

    def response(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Upsert one LLM response record; publish the merged state (TASK-2110).

        The public event carries the persisted merge, not the raw observation,
        so replays and duplicate deliveries converge on the same record.
        """
        try:
            record = self.responses.upsert(
                turn_id=event.get("turn_id"),
                response_id=event.get("response_id"),
                first_seen_seq=event.get("first_seen_seq"),
                host_parent=event.get("host_parent"),
                model=event.get("model"),
                usage=event.get("usage"),
                usage_state=event.get("usage_state"),
                wait_ms=event.get("wait_ms"),
                stream_ms=event.get("stream_ms"),
                bot_id=event.get("bot_id"),
                user_id=event.get("user_id"),
            )
        except Exception:
            logger.exception("LLM response record upsert failed turn=%s", event.get("turn_id"))
            return None
        if record is None:
            return None
        return {
            "_type": "llm_response",
            "turn_id": event.get("turn_id"),
            "trigger_message_id": event.get("trigger_message_id"),
            "bot_id": event.get("bot_id"),
            "user_id": event.get("user_id"),
            "ts": event.get("ts"),
            **record,
        }

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
            is_error = detail.get("is_error")
            self.end({
                **base,
                "event": "tool_end",
                "result": detail.get("result") or "",
                "is_error": is_error if isinstance(is_error, bool) else None,
            })
