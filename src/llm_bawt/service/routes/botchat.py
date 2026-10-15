"""Bot-specific OpenAI-compatible endpoints.

Provides /v1/botchat/{bot_id}/{user_id}/chat/completions and /models
so clients that can't pass extra params can use a base URL like:
    http://host:8642/v1/botchat/nova/nick
and point their OpenAI-compatible client at it.
"""

import json
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from ..chat_streaming import request_failed_events, stream_stop_events
from ..delivery_reply import DeliveryReplyMismatch, is_accepted_receipt, wait_for_receipt_reply
from ..dependencies import get_service
from ..logging import get_service_logger
from ..schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelInfo,
    ModelsResponse,
    UsageInfo,
)

router = APIRouter()
log = get_service_logger(__name__)


_HA_CONFLICTING_LINES = frozenset(
    {
        # HA's default assistant prompt tells the LLM to describe actions instead
        # of executing them, which fights the tool-calling system.
        "do not execute service without user's confirmation.",
        "use execute_services function only for requested action, not for current states.",
        # Redundant state restatement instruction — Nova handles this via tools.
        "do not restate or appreciate what user says, rather make a quick inquiry.",
    }
)


def _sanitize_client_context(text: str) -> str:
    """Strip HA boilerplate from client context.

    The HA prompt instructions conflict with our tool-calling system,
    so we remove those lines. The device CSV list is kept intact so
    the LLM can identify devices by entity ID and friendly name.
    """
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        stripped = line.strip().lower()

        # Skip known conflicting HA instructions
        if stripped in _HA_CONFLICTING_LINES:
            continue

        # Skip HA prompt boilerplate that conflicts with our tool system
        if any(phrase in stripped for phrase in (
            "i want you to act as smart home manager",
            "i will provide information of smart home",
            "available devices:",
            "the current state of devices",
            "use execute_services function",
        )):
            continue

        cleaned.append(line)

    result = "\n".join(cleaned).strip()
    # Remove excessive blank lines
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")
    return result


def _extract_client_system_context(request: ChatCompletionRequest) -> None:
    """Extract system messages from the request and set them as client context.

    Pulls out all role='system' messages, concatenates their content into
    ``request.client_system_context``, and removes them from the message list
    so only user/assistant messages remain.

    HA-specific boilerplate that conflicts with tool-calling is stripped.
    """
    system_parts: list[str] = []
    non_system: list = []

    for msg in request.messages:
        if msg.role == "system" and msg.content:
            system_parts.append(msg.content)
        else:
            non_system.append(msg)

    if system_parts:
        combined = "\n\n".join(system_parts)
        request.client_system_context = _sanitize_client_context(combined)
        request.ha_mode = True
        request.messages = non_system


@router.post(
    "/v1/botchat/{bot_id}/{user_id}/chat/completions",
    tags=["Bot-Specific OpenAI Compatible"],
)
async def botchat_completions(bot_id: str, user_id: str, request: ChatCompletionRequest):
    """OpenAI-compatible chat completion with bot_id and user_id baked into the URL.

    Clients set their base URL to /v1/botchat/{bot_id}/{user_id} and call
    /chat/completions as normal — no extra body params needed.

    Any system messages in the request are extracted and injected as a
    "Client Context" section in the prompt builder, alongside the bot's
    own system prompt, user profile, memory, etc.
    """
    # Override from path — ignore whatever the client sent
    request.bot_id = bot_id
    request.user = user_id

    # Pull system messages out of the message list and into client_system_context
    _extract_client_system_context(request)

    service = get_service()
    log.debug(
        f"botchat request: bot={bot_id} user={user_id} model={request.model}"
        f" client_context={'yes' if request.client_system_context else 'no'}"
    )

    if request.stream:
        return StreamingResponse(
            _stream_with_reply(service, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    try:
        response = await service.chat_completion(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("botchat completion failed")
        raise HTTPException(status_code=500, detail=str(e))
    if not is_accepted_receipt(response):
        return response
    reply = await _await_reply(service, request, response)
    status = reply.get("status")
    if status == "DELIVERED":
        return ChatCompletionResponse(
            model=_reply_model(request),
            choices=[ChatCompletionChoice(
                index=0, finish_reason="stop",
                message=ChatMessage(role="assistant", content=reply.get("response_text") or ""),
            )],
            usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
    if status in _PENDING_REPLY:
        # Still running at the deadline: hand back the exact receipt.
        return JSONResponse(content={**response, "status": status}, status_code=202)
    raise HTTPException(status_code=502, detail=reply.get("error") or str(status).lower())


# OpenAI-compatible clients cannot follow a queued receipt, so botchat waits
# for the exact admitted turn instead of bypassing the shared admission queue.
BOTCHAT_REPLY_TIMEOUT_SECONDS = 300.0
_PENDING_REPLY = frozenset({"QUEUED", "STEERING", "DISPATCHING", "IN_FLIGHT"})


def _reply_model(request: ChatCompletionRequest) -> str:
    return request.model or request.bot_id or "agent"


async def _await_reply(service, request: ChatCompletionRequest, receipt: dict) -> dict:
    """Bounded wait on the receipt's own delivery/turn; never a second submit."""
    dispatcher = getattr(service, "_inter_bot_dispatcher", None)
    deliveries = getattr(dispatcher, "store", None)
    if deliveries is None and receipt.get("delivery_id"):
        # Cannot observe the outbox from here; report it as still pending.
        return {"status": str(receipt.get("status") or "QUEUED")}
    try:
        return await wait_for_receipt_reply(
            deliveries, service._turn_log_store, receipt,
            bot_id=receipt.get("bot_id") or request.bot_id,
            user_id=receipt.get("user_id") or request.user,
            timeout_seconds=BOTCHAT_REPLY_TIMEOUT_SECONDS,
        )
    except DeliveryReplyMismatch as exc:
        log.warning("botchat receipt could not be resolved: %s", exc)
        return {"status": "UNRESOLVED", "error": str(exc)}


def _accepted_receipt_from_event(event: str) -> dict | None:
    if not event.startswith("data: ") or "chat.completion." not in event:
        return None
    try:
        payload = json.loads(event[6:])
    except ValueError:
        return None
    return payload if is_accepted_receipt(payload) else None


async def _stream_with_reply(service, request: ChatCompletionRequest):
    """Pass the stream through; replace an admission receipt with its reply."""
    async for event in service.chat_completion_stream(request):
        receipt = _accepted_receipt_from_event(event)
        if receipt is None:
            yield event
            continue
        reply = await _await_reply(service, request, receipt)
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created, model = int(time.time()), _reply_model(request)
        if reply.get("status") != "DELIVERED":
            status = str(reply.get("status") or "unresolved")
            error = RuntimeError(reply.get("error") or (
                f"reply not ready ({status}); delivery {receipt.get('delivery_id') or receipt.get('turn_id')}"
                if status in _PENDING_REPLY else status.lower()))
            for failed in request_failed_events(response_id, created, model, error):
                yield failed
            return
        chunk = {
            "id": response_id, "object": "chat.completion.chunk", "created": created,
            "model": model, "choices": [{"index": 0, "finish_reason": None, "delta": {
                "role": "assistant", "content": reply.get("response_text") or ""}}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        for trailer in stream_stop_events(response_id, created, model):
            yield trailer
        return


@router.get(
    "/v1/botchat/{bot_id}/{user_id}/models",
    response_model=ModelsResponse,
    tags=["Bot-Specific OpenAI Compatible"],
)
def botchat_models(bot_id: str, user_id: str):
    """List available models (OpenAI-compatible) — scoped under the botchat path."""
    service = get_service()
    models = [ModelInfo(id=alias) for alias in service._available_models]
    return ModelsResponse(data=models)
