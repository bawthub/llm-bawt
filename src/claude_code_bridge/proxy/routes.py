"""FastAPI routes for the Anthropic-compatible proxy.

Only one substantive endpoint right now: ``POST /v1/messages``. The
Anthropic Messages API spec also defines ``/v1/messages/count_tokens`` and
some metadata endpoints; the Claude Agent SDK doesn't currently call them
in the workflows we care about, so they're not implemented yet.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .adapters import lookup
from .errors import anthropic_error, auth_failed_error
from .heartbeat import DEFAULT_INTERVAL, with_heartbeat
from .observability import OBSERVABILITY
from .request_context import (
    BOT_HEADER,
    CONVERSATION_HEADER,
    REQUEST_HEADER,
    RESPONSES_TRANSPORT_HEADER,
    SKILL_NAMES_HEADER,
    ProxyRequestContext,
    valid_conversation_identity,
    valid_skill_names,
)
from .transport_policy import normalize_responses_transport

# Seconds of upstream silence before the proxy injects a keepalive ping.
# Override with CLAUDE_CODE_BRIDGE_PROXY_PING_INTERVAL (0 disables).
try:
    _PING_INTERVAL = float(
        os.getenv("CLAUDE_CODE_BRIDGE_PROXY_PING_INTERVAL", DEFAULT_INTERVAL)
    )
except ValueError:
    _PING_INTERVAL = DEFAULT_INTERVAL

logger = logging.getLogger(__name__)

router = APIRouter()


def _split_model(model_field: str) -> tuple[str, str]:
    """Split ``"<provider>/<upstream>"`` → ``(provider, upstream)``.

    Forward-slash is the separator because LiteLLM/OpenRouter already use it
    and the Claude SDK passes the model field through unchanged.
    """
    if "/" not in model_field:
        raise anthropic_error(
            400,
            f"Proxy expects model='<provider>/<upstream>' (e.g. "
            f"'openai_chatgpt/gpt-5.4'); got {model_field!r}.",
            error_type="invalid_request_error",
        )
    provider, _, upstream = model_field.partition("/")
    if not provider or not upstream:
        raise anthropic_error(
            400,
            f"Invalid namespaced model {model_field!r}.",
            error_type="invalid_request_error",
        )
    return provider, upstream


def _error_status(error_type: str | None) -> int:
    return {
        "invalid_request_error": 400,
        "authentication_error": 401,
        "permission_error": 403,
        "not_found_error": 404,
        "rate_limit_error": 429,
    }.get((error_type or "").strip(), 500)


def _upstream_error_type(message: str, status_code: int | None) -> str:
    lowered = message.lower()
    if status_code in (401, 403) or "auth" in lowered:
        return "authentication_error"
    if status_code == 429 or "rate" in lowered:
        return "rate_limit_error"
    if (
        (status_code is not None and 500 <= status_code <= 599)
        or "overloaded" in lowered
    ):
        return "overloaded_error"
    return "api_error"


async def _proxy_iter(
    adapter,
    body: dict[str, Any],
    upstream_model: str,
    *,
    provider: str,
    context: ProxyRequestContext,
) -> AsyncIterator[bytes]:
    """Run one proxy request, surfacing adapter failures as Anthropic SSE."""
    started = time.perf_counter()
    account_scope = adapter.account_hash()
    context.account_hash = account_scope
    context.active_provider, context.active_account = OBSERVABILITY.begin(
        provider, account_scope
    )
    status_code: int | None = None
    try:
        # Heartbeat-wrap every adapter so neither the OpenAI translate path
        # nor the Z.AI passthrough can go silent long enough to read as a
        # dead connection (covers the reasoning-window stall).
        async for chunk in with_heartbeat(
            adapter.call(body, upstream_model, context), interval=_PING_INTERVAL
        ):
            yield chunk
    except (asyncio.CancelledError, GeneratorExit):
        # Cancellation is not a successful model response. No error frame can
        # be sent to a departed consumer, but telemetry must retain the outcome.
        status_code = 499
        raise
    except RuntimeError as e:
        # Adapter failures (auth errors, upstream HTTP errors, etc.)
        # surface as RuntimeError. Classify them so the SDK handles
        # retries appropriately instead of treating all as auth errors.
        msg = str(e)
        status_match = re.search(r"\b([45]\d\d)\b", msg)
        status_code = int(status_match.group(1)) if status_match else None
        if status_code is None:
            status_code = 500
        OBSERVABILITY.record_error(provider, status_code)
        etype = _upstream_error_type(msg, status_code)
        logger.warning("Adapter error (type=%s): %s", etype, e)
        err_payload = {
            "type": "error",
            "error": {
                "type": etype,
                "message": msg,
            },
        }
        yield (
            b"event: error\n"
            b"data: " + json.dumps(err_payload).encode() + b"\n\n"
        )
    except Exception as e:  # noqa: BLE001
        status_code = getattr(e, "status_code", None) or 500
        OBSERVABILITY.record_error(provider, status_code)
        logger.exception("Unexpected adapter failure")
        etype = _upstream_error_type(str(e), status_code)
        err_payload = {
            "type": "error",
            "error": {"type": etype, "message": f"Proxy error: {e}"},
        }
        yield (
            b"event: error\n"
            b"data: " + json.dumps(err_payload).encode() + b"\n\n"
        )
    finally:
        OBSERVABILITY.end(provider, account_scope)
        errors_429, errors_5xx = OBSERVABILITY.error_totals(provider)
        logger.info(
            "proxy_stream_complete request_id=%s provider=%s account=%s bot=%s "
            "session_hash=%s active_provider=%d active_account=%d queue_ms=%.1f "
            "setup_ms=%s upstream_ttfb_ms=%s stream_ms=%.1f input=%d cached=%d "
            "output=%d cache_hit=%.1f%% status=%s errors_429_total=%d "
            "errors_5xx_total=%d upstream_model=%s",
            context.request_id,
            provider,
            context.account_hash,
            context.bot_id or "unknown",
            context.session_hash,
            context.active_provider,
            context.active_account,
            context.queue_wait_ms,
            f"{context.local_setup_ms:.1f}" if context.local_setup_ms is not None else "na",
            f"{context.upstream_ttfb_ms:.1f}" if context.upstream_ttfb_ms is not None else "na",
            (time.perf_counter() - started) * 1000,
            context.input_tokens,
            context.cached_tokens,
            context.output_tokens,
            context.cache_hit_pct,
            status_code or 200,
            errors_429,
            errors_5xx,
            upstream_model,
        )


async def _anthropic_sse_to_message(
    source: AsyncIterator[bytes],
    *,
    fallback_model: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Buffer an Anthropic SSE stream into one Messages API response object.

    Claude Code occasionally issues ``stream=false`` requests (confirmed on the
    z.ai / GLM path during the first user turn). Our adapters are stream-native,
    so for the non-streaming route we coerce the upstream call to streaming,
    then rebuild the final ``Message`` JSON body from the emitted SSE frames.

    Returns ``(message_payload, error_payload)`` where ``error_payload`` is the
    Anthropic ``{"type": "error", ...}`` object if the stream surfaced an
    error frame.
    """
    buf = ""
    message: dict[str, Any] | None = None
    blocks: dict[int, dict[str, Any]] = {}
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: dict[str, Any] | None = None
    error_payload: dict[str, Any] | None = None

    def _finalize_tool_inputs() -> None:
        for block in blocks.values():
            raw = block.pop("__input_json_buffer", None)
            if raw is None:
                continue
            if raw == "":
                block["input"] = {}
                continue
            try:
                block["input"] = json.loads(raw)
            except json.JSONDecodeError:
                block["input"] = raw

    async for chunk in source:
        if not chunk:
            continue
        buf += chunk.decode("utf-8", "ignore")
        while "\n\n" in buf:
            block_text, buf = buf.split("\n\n", 1)
            if not block_text.strip():
                continue

            data_lines: list[str] = []
            event_name = ""
            for line in block_text.splitlines():
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if not data_lines:
                continue

            try:
                payload = json.loads("\n".join(data_lines))
            except json.JSONDecodeError:
                logger.debug("Proxy non-stream buffer skipped malformed SSE block")
                continue

            ptype = payload.get("type") or event_name
            if ptype == "ping":
                continue
            if ptype == "error":
                error_payload = payload
                break
            if ptype == "message_start":
                started = payload.get("message") or {}
                message = dict(started)
                if isinstance(started.get("usage"), dict):
                    usage = started["usage"]
                continue
            if ptype == "content_block_start":
                idx = int(payload.get("index", 0) or 0)
                content_block = payload.get("content_block") or {}
                blocks[idx] = dict(content_block)
                if blocks[idx].get("type") == "tool_use":
                    blocks[idx]["__input_json_buffer"] = ""
                continue
            if ptype == "content_block_delta":
                idx = int(payload.get("index", 0) or 0)
                block = blocks.setdefault(idx, {"type": "text", "text": ""})
                delta = payload.get("delta") or {}
                dtype = delta.get("type")
                if dtype == "text_delta":
                    block["text"] = f"{block.get('text', '')}{delta.get('text', '')}"
                elif dtype == "thinking_delta":
                    block["thinking"] = f"{block.get('thinking', '')}{delta.get('thinking', '')}"
                elif dtype == "signature_delta":
                    block["signature"] = f"{block.get('signature', '')}{delta.get('signature', '')}"
                elif dtype == "input_json_delta":
                    block["__input_json_buffer"] = (
                        f"{block.get('__input_json_buffer', '')}{delta.get('partial_json', '')}"
                    )
                continue
            if ptype == "message_delta":
                delta = payload.get("delta") or {}
                stop_reason = delta.get("stop_reason")
                stop_sequence = delta.get("stop_sequence")
                if isinstance(payload.get("usage"), dict):
                    usage = payload["usage"]
                continue
            if ptype == "message_stop":
                _finalize_tool_inputs()
                continue

        if error_payload is not None:
            break

    _finalize_tool_inputs()
    if message is None:
        message = {
            "id": f"msg_proxy_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": fallback_model,
            "stop_reason": stop_reason,
            "stop_sequence": stop_sequence,
            "usage": usage
            or {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
    message["content"] = [blocks[i] for i in sorted(blocks)]
    message["model"] = message.get("model") or fallback_model
    message["stop_reason"] = (
        stop_reason if stop_reason is not None else message.get("stop_reason")
    )
    message["stop_sequence"] = (
        stop_sequence if stop_sequence is not None else message.get("stop_sequence")
    )
    if usage is not None:
        message["usage"] = usage
    return message, error_payload


@router.post("/v1/messages", response_model=None)
async def messages(request: Request) -> JSONResponse | StreamingResponse:
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        raise anthropic_error(400, f"Body is not valid JSON: {e}",
                              error_type="invalid_request_error")

    if not isinstance(body, dict):
        raise anthropic_error(400, "Body must be a JSON object.",
                              error_type="invalid_request_error")

    model_field = (body.get("model") or "").strip()
    if not model_field:
        raise anthropic_error(400, "Missing required field 'model'.",
                              error_type="invalid_request_error")

    provider, upstream_model = _split_model(model_field)
    adapter = lookup(provider)
    if adapter is None:
        raise anthropic_error(
            400,
            f"Unknown proxy provider {provider!r}. Registered: "
            f"{sorted((lookup.__globals__.get('REGISTRY') or {}).keys())}",
            error_type="invalid_request_error",
        )

    # The SDK CLI sends stream=true on the first init call but can omit the
    # field (stream=None) or set stream=false on internal calls (title gen,
    # non-streaming tool-loop iterations).  In both cases the CLI expects a
    # JSON Messages response, not SSE.  Our adapters are stream-native, so
    # the non-streaming path buffers upstream SSE via _anthropic_sse_to_message
    # and returns the assembled JSON body.
    stream_requested = bool(body.get("stream"))

    request_headers = getattr(request, "headers", {})
    context = ProxyRequestContext(
        request_id=(request_headers.get(REQUEST_HEADER) or uuid.uuid4().hex).strip(),
        provider=provider,
        bot_id=(request_headers.get(BOT_HEADER) or "").strip() or None,
        conversation_id=valid_conversation_identity(
            request_headers.get(CONVERSATION_HEADER)
        ),
        responses_transport=normalize_responses_transport(
            request_headers.get(RESPONSES_TRANSPORT_HEADER)
        ),
        skill_names=valid_skill_names(request_headers.get(SKILL_NAMES_HEADER)),
        status_callback=getattr(
            getattr(getattr(request, "app", None), "state", None),
            "proxy_status_callback",
            None,
        ),
    )

    cancellations = getattr(
        getattr(getattr(request, "app", None), "state", None),
        "proxy_cancellations",
        None,
    )
    if cancellations is not None and cancellations.is_cancelled(context.request_id):
        logger.info("proxy_request_rejected_cancelled request_id=%s", context.request_id)
        raise anthropic_error(499, "Request cancelled", error_type="api_error")

    logger.info(
        "proxy_stream_start request_id=%s provider=%s bot=%s session_hash=%s "
        "upstream_model=%s messages=%d tools=%d",
        context.request_id,
        provider,
        context.bot_id or "unknown",
        context.session_hash,
        upstream_model,
        len(body.get("messages") or []),
        len(body.get("tools") or []),
    )

    # Some Claude Agent SDK flows issue a non-streaming Anthropic Messages call
    # on the first turn. Our adapters are stream-native, so emulate
    # non-streaming by forcing the upstream to stream and buffering the SSE back
    # into one Messages API JSON response.
    body_for_adapter = dict(body)
    body_for_adapter["stream"] = True

    if not stream_requested:
        message, error_payload = await _anthropic_sse_to_message(
            _proxy_iter(
                adapter,
                body_for_adapter,
                upstream_model,
                provider=provider,
                context=context,
            ),
            fallback_model=body.get("model") or upstream_model,
        )
        if error_payload is not None:
            err = error_payload.get("error") if isinstance(error_payload, dict) else {}
            return JSONResponse(error_payload, status_code=_error_status(err.get("type")))
        return JSONResponse(message)

    return StreamingResponse(
        _proxy_iter(
            adapter,
            body_for_adapter,
            upstream_model,
            provider=provider,
            context=context,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable any proxy buffering en route
        },
    )


# Auth-related router exports kept here for future expansion.
__all__ = ["router", "auth_failed_error"]
