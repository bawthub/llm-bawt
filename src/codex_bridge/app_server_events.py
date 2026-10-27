"""Map codex ``app-server`` notifications onto exec-style events (TASK-1243 P2).

The bridge's event translation (``event_ops._handle_event``) was written for
the ``codex exec --experimental-json`` stream: ``thread.started``,
``item.started|updated|completed`` with snake_case item types, and
``turn.completed`` carrying usage. app-server reports the same turn as
JSON-RPC notifications with camelCase names. :class:`AppServerEventAdapter`
converts one into the other so a single translation serves both transports.

Events and items are ``SimpleNamespace`` because ``_handle_event`` reads
agent-message text with ``getattr``. Nested values stay plain dicts, so tool
arguments reach the UI as the dicts they are.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

# exec reports a stopped turn this way; the send treats it as an abort.
ABORTED = "aborted"

_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")

# app-server camelCase item fields the exec translation reads in snake_case.
_ITEM_ALIASES = {
    "aggregatedOutput": "aggregated_output",
    "exitCode": "exit_code",
    "durationMs": "duration_ms",
}

_USAGE_FIELDS = {
    "inputTokens": "input_tokens",
    "cachedInputTokens": "cached_input_tokens",
    "cacheWriteInputTokens": "cache_write_input_tokens",
    "outputTokens": "output_tokens",
    "reasoningOutputTokens": "reasoning_output_tokens",
    "totalTokens": "total_tokens",
}


def snake_item_type(item_type: str) -> str:
    """``commandExecution`` → ``command_execution``."""
    return _CAMEL.sub("_", item_type).lower()


def usage_from_token_usage(token_usage: Any) -> dict | None:
    """The thread's running total in the exec ``usage`` shape.

    exec's ``turn.completed`` usage is the thread total, and the rollout
    resolver (``rollout_usage``) reads it that way, so ``total`` is the
    matching app-server figure, not ``last``.
    """
    total = token_usage.get("total") if isinstance(token_usage, Mapping) else None
    if not isinstance(total, Mapping):
        return None
    return {snake: total[camel] for camel, snake in _USAGE_FIELDS.items() if camel in total}


class AppServerEventAdapter:
    """Stateful: agent-message deltas accumulate per item, and the newest
    token usage rides on ``turn.completed``. One adapter per turn."""

    def __init__(self) -> None:
        self._agent_text: dict[str, str] = {}
        self._usage: dict | None = None

    def adapt(self, note: Mapping[str, Any]) -> SimpleNamespace | None:
        """The exec-style event for one notification, or ``None`` to skip it."""
        method = note.get("method")
        params = note.get("params") or {}
        if not isinstance(params, Mapping):
            return None
        if method in ("item/started", "item/completed"):
            item = params.get("item")
            if not isinstance(item, Mapping):
                return None
            converted = self._item(item)
            if method == "item/completed" and converted.type == "agent_message":
                self._agent_text.pop(converted.id, None)
            phase = "item.started" if method == "item/started" else "item.completed"
            return SimpleNamespace(type=phase, item=converted)
        if method == "item/agentMessage/delta":
            item_id = str(params.get("itemId") or "")
            text = self._agent_text.get(item_id, "") + str(params.get("delta") or "")
            self._agent_text[item_id] = text
            return SimpleNamespace(
                type="item.updated",
                item=SimpleNamespace(type="agent_message", id=item_id, text=text),
            )
        if method == "thread/tokenUsage/updated":
            usage = usage_from_token_usage(params.get("tokenUsage"))
            if usage is not None:
                self._usage = usage
            return None
        if method == "turn/completed":
            turn = params.get("turn") or {}
            status = turn.get("status") if isinstance(turn, Mapping) else None
            if status == "completed":
                return SimpleNamespace(type="turn.completed", usage=self._usage)
            if status == "interrupted":
                return SimpleNamespace(type=ABORTED)
            error = turn.get("error") if isinstance(turn, Mapping) else None
            message = error.get("message") if isinstance(error, Mapping) else None
            return SimpleNamespace(
                type="turn.failed",
                error={"message": str(message or f"Codex turn {status or 'failed'}")},
            )
        if method == "error":
            # A retryable stream error app-server is handling itself; the turn
            # still ends with turn/completed, which carries a real failure.
            error = params.get("error")
            logger.info("codex app-server error notification (willRetry=%s): %.300s",
                        params.get("willRetry"), error)
            return None
        return None

    @staticmethod
    def _item(item: Mapping[str, Any]) -> SimpleNamespace:
        fields: dict[str, Any] = dict(item)
        fields["type"] = snake_item_type(str(item.get("type") or ""))
        for camel, snake in _ITEM_ALIASES.items():
            if camel in item:
                fields[snake] = item[camel]
        if fields["type"] == "file_change" and isinstance(item.get("changes"), list):
            fields["changes"] = [_flat_change(change) for change in item["changes"]]
        if fields["type"] == "agent_message":
            fields.setdefault("text", "")
        fields.setdefault("id", "")
        return SimpleNamespace(**fields)


def _flat_change(change: Any) -> Any:
    """``{"kind": {"type": "add"}}`` → ``{"kind": "add"}``, as exec reports it."""
    if not isinstance(change, Mapping):
        return change
    flat = dict(change)
    kind = change.get("kind")
    if isinstance(kind, Mapping):
        flat["kind"] = kind.get("type") or "update"
    return flat


def app_server_input(prompt_input: Any) -> list[dict]:
    """``turn/start`` input from the exec prompt (a string, or text and image parts)."""
    if isinstance(prompt_input, str):
        return [_text(prompt_input)]
    parts: list[dict] = []
    for part in prompt_input:
        kind = _field(part, "type")
        if kind == "text":
            parts.append(_text(str(_field(part, "text") or "")))
        elif kind == "local_image":
            parts.append({"type": "localImage", "path": str(_field(part, "path") or "")})
        else:
            raise ValueError(f"unsupported codex input part: {kind!r}")
    return parts


def _text(text: str) -> dict:
    return {"type": "text", "text": text, "text_elements": []}


def _field(part: Any, name: str) -> Any:
    if isinstance(part, Mapping):
        return part.get(name)
    return getattr(part, name, None)


__all__ = ["ABORTED", "AppServerEventAdapter", "app_server_input", "snake_item_type",
           "usage_from_token_usage"]
