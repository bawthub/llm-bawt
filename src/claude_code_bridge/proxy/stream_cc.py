"""OpenAI Chat Completions SSE → Anthropic Messages SSE.

Kimi For Coding exposes only ``/chat/completions``. Its stream carries visible
text in ``choices[].delta.content``, reasoning in ``reasoning_content``, and
fragmented function calls in ``tool_calls``. This module converts that dialect
to the strict Anthropic block lifecycle expected by the Claude Agent SDK.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from .stream import _sse
from .tool_sanitizers import recover_trailing_json, sanitize_tool_arguments

logger = logging.getLogger(__name__)

_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


def _required_args(tools: list[dict] | None) -> dict[str, frozenset[str]]:
    required_by_tool: dict[str, frozenset[str]] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        schema = tool.get("input_schema")
        if not isinstance(name, str) or not isinstance(schema, dict):
            continue
        required_by_tool[name] = frozenset(
            value for value in schema.get("required") or [] if isinstance(value, str)
        )
    return required_by_tool


def _clean_tool_arguments(
    raw: str,
    *,
    tool_name: str,
    required_by_tool: dict[str, frozenset[str]],
    allowed_skill_names: tuple[str, ...] = (),
) -> str:
    """Remove invalid optional empty strings, strip leaked reasoning tokens."""
    raw = raw or "{}"
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Try to recover valid JSON with trailing leaked reasoning.
        recovered = recover_trailing_json(raw)
        if recovered is not None:
            try:
                parsed = json.loads(recovered)
            except (json.JSONDecodeError, TypeError):
                return raw
        else:
            return raw
    if not isinstance(parsed, dict):
        return json.dumps(parsed, separators=(",", ":"))

    required = required_by_tool.get(tool_name)
    if required is not None:
        parsed = {
            key: value
            for key, value in parsed.items()
            if value != "" or key in required
        }
    if parsed.get("isolation") == "worktree":
        del parsed["isolation"]
    # Tool-specific sanitizers (JS trailing-token strip, etc.)
    parsed = sanitize_tool_arguments(
        parsed, tool_name, allowed_skill_names=allowed_skill_names
    )
    return json.dumps(parsed, separators=(",", ":"))


def _usage(payload: dict[str, Any]) -> tuple[int, int, int]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, 0
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    details = usage.get("prompt_tokens_details")
    cached = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
    # Kimi also exposes cached_tokens at the usage top level on some responses.
    cached = max(cached, int(usage.get("cached_tokens") or 0))
    return prompt, completion, cached


async def chat_completions_to_anthropic_sse(
    lines: AsyncIterator[str],
    *,
    anthropic_model: str,
    tool_schemas: list[dict] | None = None,
    allowed_skill_names: tuple[str, ...] = (),
) -> AsyncIterator[bytes]:
    """Translate one Chat Completions SSE stream into Anthropic SSE bytes."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": anthropic_model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        },
    )

    next_index = 0
    open_block: tuple[str, int] | None = None
    tool_calls: dict[int, dict[str, str]] = {}
    required_by_tool = _required_args(tool_schemas)
    finish_reason: str | None = None
    prompt_tokens = completion_tokens = cached_tokens = 0

    def start_frame(kind: str, index: int) -> bytes:
        if kind == "thinking":
            block = {"type": "thinking", "thinking": "", "signature": ""}
        else:
            block = {"type": "text", "text": ""}
        return _sse(
            "content_block_start",
            {"type": "content_block_start", "index": index, "content_block": block},
        )

    def stop_frame(index: int) -> bytes:
        return _sse(
            "content_block_stop", {"type": "content_block_stop", "index": index}
        )

    async def close_open(*, sign_thinking: bool = True) -> AsyncIterator[bytes]:
        nonlocal open_block
        if open_block is None:
            return
        kind, index = open_block
        if kind == "thinking" and sign_thinking:
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "signature_delta",
                        "signature": f"reasoning:{message_id}",
                    },
                },
            )
        yield stop_frame(index)
        open_block = None

    try:
        async for line in lines:
            line = line.strip()
            if not line or line.startswith(("event:", ":")):
                continue
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                break
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed Chat Completions SSE data line")
                continue

            error = payload.get("error")
            if isinstance(error, dict):
                async for frame in close_open():
                    yield frame
                yield _sse(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "type": error.get("type") or "upstream_error",
                            "message": error.get("message") or "Kimi upstream error",
                        },
                    },
                )
                return

            p_prompt, p_completion, p_cached = _usage(payload)
            if p_prompt or p_completion or p_cached:
                prompt_tokens = p_prompt
                completion_tokens = p_completion
                cached_tokens = p_cached

            choices = payload.get("choices") or []
            if not choices:
                continue
            choice = choices[0] if isinstance(choices[0], dict) else {}
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue

            reasoning = delta.get("reasoning_content") or ""
            if reasoning:
                if open_block is None or open_block[0] != "thinking":
                    async for frame in close_open():
                        yield frame
                    open_block = ("thinking", next_index)
                    next_index += 1
                    yield start_frame(*open_block)
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": open_block[1],
                        "delta": {"type": "thinking_delta", "thinking": reasoning},
                    },
                )

            content = delta.get("content") or ""
            if content:
                if open_block is None or open_block[0] != "text":
                    async for frame in close_open():
                        yield frame
                    open_block = ("text", next_index)
                    next_index += 1
                    yield start_frame(*open_block)
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": open_block[1],
                        "delta": {"type": "text_delta", "text": content},
                    },
                )

            for part in delta.get("tool_calls") or []:
                if not isinstance(part, dict):
                    continue
                try:
                    tool_index = int(part.get("index", 0) or 0)
                except (TypeError, ValueError):
                    tool_index = 0
                call = tool_calls.setdefault(
                    tool_index, {"id": "", "name": "", "arguments": ""}
                )
                if part.get("id"):
                    call["id"] = str(part["id"])
                function = part.get("function") or {}
                if isinstance(function, dict):
                    if function.get("name"):
                        call["name"] = str(function["name"])
                    if function.get("arguments"):
                        call["arguments"] += str(function["arguments"])

        async for frame in close_open():
            yield frame

        for tool_index in sorted(tool_calls):
            call = tool_calls[tool_index]
            call_id = call["id"] or f"tool_{uuid.uuid4().hex[:20]}"
            index = next_index
            next_index += 1
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": call_id,
                        "name": call["name"],
                        "input": {},
                    },
                },
            )
            arguments = _clean_tool_arguments(
                call["arguments"],
                tool_name=call["name"],
                required_by_tool=required_by_tool,
                allowed_skill_names=allowed_skill_names,
            )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": arguments,
                    },
                },
            )
            yield stop_frame(index)

        stop_reason = (
            "tool_use"
            if tool_calls
            else _STOP_REASON_MAP.get(finish_reason or "", "end_turn")
        )
        uncached = max(prompt_tokens - cached_tokens, 0)
        logger.info(
            "Chat Completions usage: input=%d cached=%d uncached=%d output=%d cache_hit=%.1f%%",
            prompt_tokens,
            cached_tokens,
            uncached,
            completion_tokens,
            (100.0 * cached_tokens / prompt_tokens) if prompt_tokens else 0.0,
        )
        yield _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {
                    "input_tokens": uncached,
                    "output_tokens": completion_tokens,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": cached_tokens,
                },
            },
        )
        yield _sse("message_stop", {"type": "message_stop"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat Completions stream translation failed")
        async for frame in close_open():
            yield frame
        yield _sse(
            "error",
            {
                "type": "error",
                "error": {
                    "type": "proxy_stream_error",
                    "message": f"Chat Completions stream translation failed: {exc}",
                },
            },
        )
