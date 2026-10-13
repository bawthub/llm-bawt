"""Bounded text recovery with exact-prefix validation (TASK-1064).

Appending partial assistant text alone is NOT continuation: live ChatGPT probes
repeated the entire answer. Request exact replay, discard only a byte-identical
visible prefix, and fail closed before forwarding any new output on mismatch.
Native reasoning is retained in order, just as the SDK will replay it next turn.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

MAX_CAPTURE_CHARS = 1_000_000
MAX_PENDING_EVENTS = 2048


class ContinuationMismatchError(ValueError):
    """Cannot prove a safe splice. Never retry this as a network failure."""


@dataclass
class CommittedOutput:
    items: list[dict] = field(default_factory=list)
    text: str = ""
    size: int = 0
    blocked: bool = False

    def _reserve(self, size: int) -> bool:
        if self.blocked:
            return False
        self.size += size
        if self.size > MAX_CAPTURE_CHARS:
            self.blocked = True
            self.items.clear()
            self.text = ""
            return False
        return True

    def add_text(self, text: str) -> None:
        if not self._reserve(len(text)):
            return
        self.text += text
        if self.items and self.items[-1].get("role") == "assistant":
            self.items[-1]["content"] += text
        else:
            self.items.append({"role": "assistant", "content": text})

    def add_reasoning(self, item: dict) -> None:
        if self._reserve(len(json.dumps(item))):
            self.items.append(copy.deepcopy(item))

    @property
    def can_continue(self) -> bool:
        return bool(self.text) and not self.blocked

    def request(self, original: dict) -> dict:
        if not self.can_continue:
            raise ContinuationMismatchError("No complete replayable text checkpoint")
        body = copy.deepcopy(original)
        body["input"] += copy.deepcopy(self.items)
        body["input"].append({"role": "developer", "content": (
            "Transport recovery: the preceding assistant output was interrupted. "
            "Resume the original task. Begin your visible output by reproducing "
            "the following already-displayed text EXACTLY, with no preamble or "
            "formatting changes, then continue from its end. The transport will "
            "validate and remove that exact prefix; do not call any tools until "
            "you have reproduced it. Already-displayed text (JSON string): "
            + json.dumps(self.text)
        )})
        return body


async def validated_replay(stream, prefix: str):
    """Withhold ALL new blocks until exact text replay is proven; bounded RAM.

    Deliberately no fuzzy/semantic deduplication. A tool before the matched prefix
    is rejected before its tool_use start reaches the SDK. Once matched, ordinary
    translator/tool-commit guards own the rest of the stream.
    """
    matched = 0
    pending = []
    pending_size = 0
    async for event in stream:
        if matched == len(prefix):
            yield event
            continue
        kind = getattr(event, "type", "")
        if kind == "response.output_text.delta":
            delta = getattr(event, "delta", "") or ""
            take = min(len(delta), len(prefix) - matched)
            if delta[:take] != prefix[matched:matched + take]:
                raise ContinuationMismatchError("Continuation did not reproduce the visible prefix")
            matched += take
            if matched == len(prefix):
                for buffered in pending:
                    yield buffered
                pending.clear()
                if delta[take:]:
                    suffix = copy.copy(event)
                    suffix.delta = delta[take:]
                    yield suffix
            continue
        item = getattr(event, "item", None)
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", "")
        if (kind == "response.refusal.delta" or kind in {
            "response.completed", "response.incomplete", "response.failed", "error", "response.error",
        } or (kind == "response.output_item.added" and item_type not in {"reasoning", "message"})):
            raise ContinuationMismatchError("Continuation ended or requested a tool before prefix validation")
        # Measure opaque reasoning without logging it; cap silent buffered output.
        pending_size += len(repr(event))
        if len(pending) >= MAX_PENDING_EVENTS or pending_size > MAX_CAPTURE_CHARS:
            raise ContinuationMismatchError("Continuation validation buffer exceeded")
        pending.append(event)
    if matched != len(prefix):
        raise ContinuationMismatchError("Continuation closed before prefix validation")
