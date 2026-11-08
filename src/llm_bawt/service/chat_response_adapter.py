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
# TASK-1418: how long a public stream may hold its headers waiting for the
# first real frame, so a failure before it can still set the HTTP status.
# Bounded well under gateway timeouts; after it, headers go out as before.
PRE_HEADER_SECONDS = 5.0
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
    if reply.get("status") != "DELIVERED":
        status, detail = _reply_error(reply)
        yield _stream_error(detail, status)
        return
    for frame in _reply_frames(request, reply.get("response_text") or ""):
        yield frame


def _reply_frames(request: ChatCompletionRequest, text: str):
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created, model = int(time.time()), _model(request)
    yield _data({
        "id": response_id, "object": "chat.completion.chunk", "created": created,
        "model": model, "choices": [{"index": 0, "finish_reason": None,
                               "delta": {"role": "assistant", "content": text}}],
    })
    yield _data({
        "id": response_id, "object": "chat.completion.chunk", "created": created,
        "model": model, "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}],
    })
    yield "data: [DONE]\n\n"


EXTENSION_COMMENT = ": llm-bawt-event "

# Fields of OpenAI's ChatCompletionChunk / chunk Choice / ChoiceDelta. Anything
# else (e.g. delta.reasoning_content) is an llm-bawt extension.
_CHUNK_KEYS = frozenset({"id", "object", "created", "model", "choices",
                         "system_fingerprint", "service_tier", "usage"})
_CHOICE_KEYS = frozenset({"index", "delta", "finish_reason", "logprobs"})
_DELTA_KEYS = frozenset({"role", "content", "refusal"})
# Every tool in this stream already ran on the server. These fields mean "the
# client must execute this call and reply", so they never reach OpenAI clients.
_CLIENT_TOOL_DELTA = ("tool_calls", "function_call")
_CLIENT_TOOL_FINISH = frozenset({"tool_calls", "function_call"})


def _split_chunk(payload: dict) -> tuple[dict | None, dict | None]:
    """Split one frame into (standard OpenAI chunk, llm-bawt extension).

    Either side may be None. A frame that is not a chunk at all (warnings,
    usage, animation, ask-user) or that carries a server-run tool call is
    entirely an extension.
    """
    choices = payload.get("choices")
    if (payload.get("object") != "chat.completion.chunk" or not isinstance(choices, list)
            or not choices or not all(isinstance(choice, dict) for choice in choices)):
        return None, payload
    deltas = [choice.get("delta") if isinstance(choice.get("delta"), dict) else {} for choice in choices]
    if any(choice.get("finish_reason") in _CLIENT_TOOL_FINISH for choice in choices) or any(
            delta.get(key) for delta in deltas for key in _CLIENT_TOOL_DELTA):
        return None, payload
    extra_top = {key: value for key, value in payload.items() if key not in _CHUNK_KEYS}
    standard_choices, extension_choices = [], []
    for choice, delta in zip(choices, deltas):
        standard_choices.append({
            **{key: value for key, value in choice.items() if key in _CHOICE_KEYS},
            "delta": {key: value for key, value in delta.items() if key in _DELTA_KEYS},
        })
        extra_choice = {key: value for key, value in choice.items() if key not in _CHOICE_KEYS}
        extra_delta = {key: value for key, value in delta.items() if key not in _DELTA_KEYS}
        if extra_choice or extra_delta:
            extension_choices.append({"index": choice.get("index", 0), **extra_choice,
                                      "delta": extra_delta, "finish_reason": None})
    if not extra_top and not extension_choices:
        return payload, None
    head = {key: payload[key] for key in ("id", "object", "created", "model") if key in payload}
    extension = {**head, **extra_top, "choices": extension_choices}
    standard = {**{key: value for key, value in payload.items() if key in _CHUNK_KEYS},
                "choices": standard_choices}
    if not any(choice["delta"] or choice.get("finish_reason") for choice in standard_choices):
        standard = None  # e.g. a reasoning-only chunk: nothing standard left.
    return standard, extension


def _openai_frames(event: str, payload: dict, extensions: str) -> list[str]:
    """Encode one frame for the caller's negotiated extension mode."""
    standard, extension = _split_chunk(payload)
    if extension is None:
        return [event]  # Already a pure standard chunk: bytes unchanged.
    if extensions == "data":
        return [event]  # Pre-negotiation BawtHub: keep the frames it consumes.
    frames = [] if standard is None else [_data(standard)]
    if extensions == "comment":
        # SSE comments are ignored by every OpenAI client; BawtHub's proxy
        # turns this JSON back into the data frame its UI already handles.
        frames.append(f"{EXTENSION_COMMENT}{json.dumps(extension)}\n\n")
    return frames  # extensions == "none": a stock client gets standard chunks only.


async def completion_stream(
    service: Any, request: ChatCompletionRequest, *, admission_comment: bool = False,
    extensions: str = "none",
) -> AsyncIterator[str]:
    """Preserve live direct chunks; turn a queued receipt into its exact reply.

    ``extensions`` is ``comment`` (caller translates ``: llm-bawt-event``
    comments), ``data`` (legacy caller keeps non-OpenAI data frames) or
    ``none`` (stock OpenAI client: such frames are dropped).
    """
    reader = None
    from .voice_step import VOICE_STEP_TIMEOUT_SECONDS, observe_voice_step

    voice_deadline = asyncio.get_running_loop().time() + VOICE_STEP_TIMEOUT_SECONDS
    if request.voice_input:
        subscriber = getattr(service, "_redis_subscriber", None)
        if subscriber is not None:
            try:
                # A unique cursor cannot supersede a chat tab or another call.
                reader = subscriber.ui_stream(
                    request.bot_id or service._default_bot,
                    request.user or service.config.DEFAULT_USER,
                    f"voice-step-{uuid.uuid4().hex}",
                )
                async with asyncio.timeout_at(voice_deadline):
                    await reader.claim()  # Must precede admission/dispatch.
            except Exception:
                reader = None  # Admission still happens once; observation fails closed.
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
                if request.voice_input:
                    # Route on the durable outcome, not just the advisory busy
                    # snapshot: another admission can start before this claim.
                    text = None
                    busy = payload.get("busy") is True
                    if not busy or payload.get("steer_capable") is True:
                        deadline = voice_deadline if busy else (
                            voice_deadline + REPLY_TIMEOUT_SECONDS - VOICE_STEP_TIMEOUT_SECONDS
                        )
                        observer = asyncio.create_task(observe_voice_step(
                            service, request, payload, reader, deadline=deadline,
                        ))
                        try:
                            while not observer.done():
                                done, _ = await asyncio.wait({observer}, timeout=15.0)
                                if not done:
                                    yield ": waiting for voice reply\n\n"
                            text = await observer
                        finally:
                            if not observer.done():
                                observer.cancel()
                                await asyncio.gather(observer, return_exceptions=True)
                    if text is None:
                        warning = {"object": "service.warning", "warnings": ["voice_step_unavailable"]}
                        for frame in _openai_frames(_data(warning), warning, extensions):
                            yield frame
                        yield "data: [DONE]\n\n"
                    else:
                        for frame in _reply_frames(request, text):
                            yield frame
                else:
                    async for reply_event in _wait_stream_reply(service, request, payload):
                        yield reply_event
                return  # Discard the internal receipt's [DONE].
            if payload.get("object") == "service.warning":
                failures = [str(item) for item in payload.get("warnings", [])
                            if str(item).startswith(("request_failed:", "stream_interrupted:"))]
                if failures:
                    yield _stream_error("; ".join(failures))
                    return  # A failed turn must never emit a successful stop/DONE.
            for frame in _openai_frames(event, payload, extensions):
                yield frame
    except Exception as exc:
        # Once StreamingResponse has sent headers, HTTPException cannot change
        # status. A concrete error event is the only honest in-band outcome;
        # open_stream turns it into the real status if no output preceded it.
        yield _stream_error(str(exc), _exception_status(exc))


def _exception_status(exc: Exception) -> int:
    """The status the non-stream route gives the same failure."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    return 400 if isinstance(exc, ValueError) else 502


def _early_error(frame: str) -> HTTPException | None:
    """An error frame as an HTTP error, when it is the stream's first output."""
    if not frame.startswith("data: "):
        return None
    try:
        error = json.loads(frame[6:]).get("error")
    except (ValueError, AttributeError):
        return None
    if not isinstance(error, dict):
        return None
    status = error.get("code")
    status = status if isinstance(status, int) and 400 <= status <= 599 else 502
    return HTTPException(status_code=status, detail=str(error.get("message") or "stream failed"))


async def open_stream(
    frames: AsyncIterator[str], *, timeout: float | None = None,
) -> AsyncIterator[str]:
    """Hold a stream's headers until its first real frame (TASK-1418).

    Leading SSE comments are buffered until a data frame, an admission
    comment, the end of the stream or ``timeout``. An error frame that arrives
    first raises HTTPException with its real status, so a stock OpenAI client
    sees a non-2xx error instead of an HTTP 200 stream carrying one. After
    that, the remaining frames pass through unchanged.
    """
    iterator = frames.__aiter__()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + (PRE_HEADER_SECONDS if timeout is None else timeout)
    held: list[str] = []
    pending: asyncio.Future | None = None
    finished = False
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())
            done, _ = await asyncio.wait({pending}, timeout=remaining)
            if not done:
                break  # Keep the in-flight step; the body resumes it.
            step, pending = pending, None
            try:
                frame = step.result()
            except StopAsyncIteration:
                finished = True
                break
            error = _early_error(frame)
            if error is not None:
                finished = True
                raise error
            held.append(frame)
            if frame.startswith("data: ") or frame.startswith(": llm-bawt-admission"):
                break
    except BaseException:
        await _drop(pending)
        await _aclose(iterator)
        raise

    async def body() -> AsyncIterator[str]:
        try:
            for frame in held:
                yield frame
            if finished:
                return
            if pending is not None:
                try:
                    yield await pending
                except StopAsyncIteration:
                    return
            async for frame in iterator:
                yield frame
        finally:
            if pending is not None and not pending.done():
                await _drop(pending)
            await _aclose(iterator)

    return body()


async def _drop(pending: asyncio.Future | None) -> None:
    # The step must settle before aclose(); a generator mid-step cannot close.
    if pending is not None:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


async def _aclose(iterator: Any) -> None:
    aclose = getattr(iterator, "aclose", None)
    if aclose is not None:
        try:
            await aclose()
        except Exception:
            pass
