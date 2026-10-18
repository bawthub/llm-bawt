"""Public chat response adapter for already-admitted ordinary turns.

The service's internal stream can carry an admission receipt and UI sidecars.
The public completion routes observe that exact turn without resubmitting it.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import HTTPException

from .delivery_reply import DeliveryReplyMismatch, is_accepted_receipt, wait_for_receipt_reply
from .schemas import (
    ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage,
)

REPLY_TIMEOUT_SECONDS = 300.0
_PENDING = frozenset({"QUEUED", "STEERING", "DISPATCHING", "IN_FLIGHT"})


def _model(request: ChatCompletionRequest) -> str:
    return request.model or request.bot_id or "agent"


def _reply_error(reply: dict[str, Any]) -> tuple[int, str]:
    status = str(reply.get("status") or "UNRESOLVED")
    detail = str(reply.get("error") or (
        f"Reply not ready ({status}); original turn remains admitted"
        if status in _PENDING else f"Original turn result unavailable ({status})"
    ))
    return (504 if status in _PENDING else 502), detail


async def observe_receipt(service: Any, request: ChatCompletionRequest, receipt: dict) -> dict:
    """Read one original result; loss of evidence is never permission to replay."""
    dispatcher = getattr(service, "_inter_bot_dispatcher", None)
    deliveries = getattr(dispatcher, "store", None)
    if receipt.get("delivery_id") and deliveries is None:
        raise DeliveryReplyMismatch("Original delivery store unavailable")
    return await wait_for_receipt_reply(
        deliveries, service._turn_log_store, receipt,
        bot_id=receipt.get("bot_id") or request.bot_id or service._default_bot,
        user_id=receipt.get("user_id") or request.user or service.config.DEFAULT_USER,
        timeout_seconds=REPLY_TIMEOUT_SECONDS,
    )


async def completion_response(service: Any, request: ChatCompletionRequest):
    """Return a standard completed response, never an HTTP 200/202 receipt."""
    response = await service.chat_completion(request)
    if not is_accepted_receipt(response):
        return response
    try:
        reply = await observe_receipt(service, request, response)
    except DeliveryReplyMismatch as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if reply.get("status") != "DELIVERED":
        status, detail = _reply_error(reply)
        raise HTTPException(status_code=status, detail=detail)
    return ChatCompletionResponse(
        model=_model(request),
        choices=[ChatCompletionChoice(
            index=0, finish_reason="stop",
            message=ChatMessage(role="assistant", content=reply.get("response_text") or ""),
        )],
    )


def _data(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _stream_error(detail: str, status: int = 502) -> str:
    # OpenAI streaming errors are a terminal error event, NOT a successful
    # stop chunk or [DONE]. In particular, a timeout cannot masquerade as an
    # empty assistant answer while the original turn continues independently.
    return _data({"error": {"message": detail, "type": "server_error", "code": status}})


async def _wait_stream_reply(service: Any, request: ChatCompletionRequest, receipt: dict) -> AsyncIterator[str]:
    task = asyncio.create_task(observe_receipt(service, request, receipt))
    try:
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=15.0)
            if not done:
                yield ": waiting for admitted turn\n\n"
        reply = await task
    finally:
        if not task.done():
            task.cancel()
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created, model = int(time.time()), _model(request)
    if reply.get("status") != "DELIVERED":
        status, detail = _reply_error(reply)
        yield _stream_error(detail, status)
        return
    yield _data({
        "id": response_id, "object": "chat.completion.chunk", "created": created,
        "model": model, "choices": [{"index": 0, "finish_reason": None,
                               "delta": {"role": "assistant", "content": reply.get("response_text") or ""}}],
    })
    yield _data({
        "id": response_id, "object": "chat.completion.chunk", "created": created,
        "model": model, "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}],
    })
    yield "data: [DONE]\n\n"


async def completion_stream(
    service: Any, request: ChatCompletionRequest, *, admission_comment: bool = False,
) -> AsyncIterator[str]:
    """Preserve live direct chunks; turn a queued receipt into its exact reply."""
    try:
        async for event in service.chat_completion_stream(request):
            if not event.startswith("data: "):
                yield event
                continue
            try:
                payload = json.loads(event[6:])
            except ValueError:
                yield event
                continue
            if not isinstance(payload, dict):
                yield event
                continue
            if is_accepted_receipt(payload):
                if admission_comment:
                    # BawtHub's Hono proxy translates this SSE comment for its
                    # existing UI. OpenAI clients ignore comments and observe the
                    # same admitted turn through the normal completion stream.
                    yield f": llm-bawt-admission {json.dumps(payload)}\n\n"
                async for reply_event in _wait_stream_reply(service, request, payload):
                    yield reply_event
                return  # Discard the internal receipt's [DONE].
            if payload.get("object") == "service.warning":
                failures = [str(item) for item in payload.get("warnings", [])
                            if str(item).startswith(("request_failed:", "stream_interrupted:"))]
                if failures:
                    yield _stream_error("; ".join(failures))
                    return  # A failed turn must never emit a successful stop/DONE.
            yield event
    except Exception as exc:
        # Once StreamingResponse has sent headers, HTTPException cannot change
        # status. A concrete error event is the only honest in-band outcome.
        yield _stream_error(str(exc))
