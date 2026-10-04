"""OpenAI-compatible chat completion route."""

import asyncio
import json
import time
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..dependencies import get_service
from ..logging import get_service_logger
from ..schemas import ChatCompletionRequest

router = APIRouter()
log = get_service_logger(__name__)


def _internal_inter_bot_sender(http_request: Request) -> str | None:
    sender = (http_request.headers.get("X-LLM-Bawt-Inter-Bot-Sender") or "").strip()
    if not sender:
        return None
    client_host = http_request.client.host if http_request.client else ""
    if client_host not in {"127.0.0.1", "::1"}:
        raise HTTPException(
            status_code=403,
            detail="internal inter-bot sender requires loopback",
        )
    return sender


class ChatAbortRequest(BaseModel):
    """Request to abort an in-flight turn."""
    turn_id: str = Field(..., description="Turn log ID to abort")
    source: Literal["unknown", "chat_stop", "bot_list_stop"] = "unknown"


class ChatAbortResponse(BaseModel):
    ok: bool
    detail: str | None = None
    turn_id: str | None = None


@router.post("/v1/chat/abort", tags=["OpenAI Compatible"])
async def chat_abort(request: ChatAbortRequest, http_request: Request) -> ChatAbortResponse:
    """Persist cancellation before requesting teardown of the exact bridge run."""
    service = get_service()
    store = service._turn_log_store
    turn = store.get_turn(request.turn_id)
    if not turn:
        raise HTTPException(status_code=404, detail="Turn not found")

    from ..turn_abort import abort_turn

    detail = await abort_turn(
        service, turn, source=request.source,
        peer=http_request.client.host if http_request.client else None,
    )
    return ChatAbortResponse(ok=True, detail=detail, turn_id=turn.id)

class ChatSteerRequest(BaseModel):
    """Redirect an in-flight Claude agent turn without creating another turn."""

    turn_id: str = Field(..., description="Active llm-bawt turn log ID")
    message: str = Field(..., min_length=1, description="User correction/instruction")
    message_id: str = Field(..., description="Frontend-generated user message UUID")
    content_offset: int | None = Field(None, ge=0, description="Assistant characters visible when steering")
    bot_id: str = Field(..., description="Bot slug owning the active turn")
    user_id: str = Field("nick", description="User namespace owning the turn")
    priority: Literal["now", "next"] = Field(
        "next", description="Queue until the next model call without interrupting running tools"
    )


class ChatSteerResponse(BaseModel):
    ok: bool
    detail: str
    turn_id: str
    message_id: str
    persisted: bool = False
    can_escalate: bool = False
    can_cancel: bool = False


def _persist_interrupt_anchor(
    service,
    *,
    bot_id: str,
    message_id: str,
    source_message_id: str,
    content_offset: int,
) -> bool:
    """Persist a steer anchor through embedded or MCP-backed memory modes."""
    memory_client = service.get_memory_client(bot_id)
    if memory_client is not None:
        manager = memory_client.get_short_term_manager()
        backend = getattr(manager, "_backend", None)
        if backend is not None:
            return backend.set_interrupt_anchor(
                message_id,
                source_message_id,
                content_offset,
            )

    store = getattr(service, "_turn_log_store", None)
    engine = getattr(store, "engine", None)
    if engine is None:
        return False

    from sqlalchemy import text

    with engine.begin() as conn:
        result = conn.execute(
            text(
                "UPDATE messages "
                "SET interrupt_source_message_id=:source_message_id, "
                "interrupt_content_offset=:content_offset "
                "WHERE bot_id=:bot_id AND id=:message_id"
            ),
            {
                "bot_id": bot_id,
                "message_id": message_id,
                "source_message_id": source_message_id,
                "content_offset": content_offset,
            },
        )
    return bool(result.rowcount)


async def steer_active_turn(
    service,
    request: ChatSteerRequest,
    *,
    steer_request_id: str | None = None,
    author=None,
) -> ChatSteerResponse:
    """Shared steer implementation for public chat and durable inter-bot work."""
    turn = service._turn_log_store.get_turn(request.turn_id)
    if turn is None:
        raise HTTPException(status_code=404, detail="Turn not found")
    if (
        turn.ended_at is not None or turn.status not in ("streaming", "pending")
    ) and not steer_request_id:
        raise HTTPException(status_code=409, detail="Turn is no longer active")

    bot_id = request.bot_id.strip().lower()
    user_id = request.user_id.strip().lower() or "nick"
    if turn.bot_id != bot_id or (turn.user_id or "").lower() != user_id:
        raise HTTPException(status_code=409, detail="Turn ownership mismatch")
    if not turn.agent_session_key:
        # A deterministic delivery retry may inspect the original target after it
        # ended. If the run never registered a bridge session, no steer RPC could
        # possibly have been submitted; report a definitive rejection so the
        # dispatcher can clear stale steer intent and choose one safe fallback.
        if steer_request_id and turn.ended_at is not None:
            raise HTTPException(status_code=409, detail="no_active_run")
        raise HTTPException(status_code=409, detail="Active bridge run is not ready")

    from ...agent_backends.agent_bridge import get_agent_subscriber
    from ...bots import BotManager

    bot = BotManager(service.config).get_bot(bot_id)
    backend_name = (getattr(bot, "agent_backend", None) or "").strip()
    if backend_name != "claude-code":
        raise HTTPException(
            status_code=409,
            detail="Mid-turn steering is currently supported only by claude-code",
        )

    message = request.message.strip()
    subscriber = get_agent_subscriber()
    if subscriber is None:
        raise HTTPException(status_code=503, detail="Agent bridge unavailable")

    result: dict = {"ok": False, "error": "no_active_run"}
    try:
        for attempt in range(8):
            result = await subscriber.send_steer(
                session_key=turn.agent_session_key,
                message=message,
                message_id=request.message_id,
                backend=backend_name,
                target_request_id=turn.agent_request_id,
                request_id=steer_request_id,
                priority=request.priority,
                origin="user" if steer_request_id is None else "bot",
            )
            if result.get("ok") or result.get("error") != "no_active_run":
                break
            if attempt < 7:
                await asyncio.sleep(0.25)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    if not result.get("ok"):
        detail = str(result.get("error") or result.get("detail") or "steer failed")
        status = 409 if detail in {"no_active_run", "active_run_mismatch"} else 502
        raise HTTPException(status_code=status, detail=detail)

    persisted = False
    try:
        memory_client = service.get_memory_client(bot_id, user_id)
        if memory_client is None:
            raise RuntimeError("Memory client unavailable")
        if not turn.trigger_message_id:
            raise RuntimeError("Active turn has no trigger message id")
        trigger = await asyncio.to_thread(
            memory_client.get_message_by_id,
            turn.trigger_message_id,
        )
        if not isinstance(trigger, dict):
            raise RuntimeError("Active turn trigger message was not found")
        session_id = trigger.get("session_id")
        if not session_id:
            raise RuntimeError("Active turn trigger message has no session id")
        await asyncio.to_thread(
            memory_client.add_message,
            "user",
            message,
            session_id,
            None,
            request.message_id,
            None,
            None,
            (author.entity_type if author else None),
            (author.entity_id if author else None),
        )
        if turn.trigger_message_id is not None and request.content_offset is not None:
            anchored = await asyncio.to_thread(
                _persist_interrupt_anchor,
                service,
                bot_id=bot_id,
                message_id=request.message_id,
                source_message_id=turn.trigger_message_id,
                content_offset=request.content_offset,
            )
            if not anchored:
                raise RuntimeError("Persisted steer message was not found for anchor update")
        persisted = True
    except Exception:
        log.exception(
            "chat.steer accepted but persistence failed: turn=%s message=%s",
            turn.id,
            request.message_id,
        )

    return ChatSteerResponse(
        ok=True,
        detail="steered" if persisted else "steered_persistence_failed",
        turn_id=turn.id,
        message_id=request.message_id,
        persisted=persisted,
        can_escalate=bool(result.get("can_escalate")) and request.priority == "next",
        can_cancel=bool(result.get("can_cancel")) and request.priority == "next",
    )


@router.post("/v1/chat/steer", tags=["Agent Backends"])
async def chat_steer(request: ChatSteerRequest) -> ChatSteerResponse:
    """Interrupt and redirect one active Claude SDK run in place.

    This is deliberately not another chat completion: the existing bridge
    subprocess, response stream, assistant bubble, and turn log remain active.
    """
    return await steer_active_turn(get_service(), request)


class ChatSteerEscalateRequest(BaseModel):
    turn_id: str
    message_id: str
    bot_id: str
    user_id: str = "nick"


@router.post("/v1/chat/steer/escalate", tags=["Agent Backends"])
async def chat_steer_escalate(request: ChatSteerEscalateRequest) -> dict:
    """Escalate a queued user message, only on its original active run."""
    service = get_service()
    turn = service._turn_log_store.get_turn(request.turn_id)
    if turn is None:
        raise HTTPException(status_code=404, detail="Turn not found")
    if (turn.ended_at is not None or turn.status not in ("streaming", "pending")
            or turn.bot_id != request.bot_id.strip().lower()
            or (turn.user_id or "").lower() != request.user_id.strip().lower()):
        raise HTTPException(status_code=409, detail="Turn is no longer active or ownership mismatch")
    if not turn.agent_session_key or not turn.agent_request_id:
        raise HTTPException(status_code=409, detail="Active bridge run is not ready")
    from ...agent_backends.agent_bridge import get_agent_subscriber
    from ...bots import BotManager

    backend = (getattr(BotManager(service.config).get_bot(turn.bot_id), "agent_backend", None) or "").strip()
    if backend != "claude-code":
        raise HTTPException(status_code=409, detail="Escalation requires claude-code")
    subscriber = get_agent_subscriber()
    if subscriber is None:
        raise HTTPException(status_code=503, detail="Agent bridge unavailable")
    try:
        result = await subscriber.send_steer(
            session_key=turn.agent_session_key, message="", message_id=request.message_id,
            backend=backend, target_request_id=turn.agent_request_id,
            request_id=f"escalate_{request.turn_id}_{request.message_id}",
            origin="user", escalate=True,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail=str(result.get("error") or "escalation rejected"))
    return {"ok": True, "detail": result.get("detail", "escalated"), "message_id": request.message_id}


@router.post("/v1/chat/steer/cancel", tags=["Agent Backends"])
async def chat_steer_cancel(request: ChatSteerEscalateRequest) -> dict:
    """Withdraw one still-queued user steer; never interrupt active work."""
    service = get_service()
    turn = service._turn_log_store.get_turn(request.turn_id)
    if turn is None:
        raise HTTPException(status_code=404, detail="Turn not found")
    if (turn.ended_at is not None or turn.status not in ("streaming", "pending")
            or turn.bot_id != request.bot_id.strip().lower()
            or (turn.user_id or "").lower() != request.user_id.strip().lower()):
        raise HTTPException(status_code=409, detail="Turn is no longer active or ownership mismatch")
    if not turn.agent_session_key or not turn.agent_request_id:
        raise HTTPException(status_code=409, detail="Active bridge run is not ready")
    from ...agent_backends.agent_bridge import get_agent_subscriber
    from ...bots import BotManager

    backend = (getattr(BotManager(service.config).get_bot(turn.bot_id), "agent_backend", None) or "").strip()
    if backend != "claude-code":
        raise HTTPException(status_code=409, detail="Cancellation requires claude-code")
    subscriber = get_agent_subscriber()
    if subscriber is None:
        raise HTTPException(status_code=503, detail="Agent bridge unavailable")
    try:
        result = await subscriber.send_steer(
            session_key=turn.agent_session_key, message="", message_id=request.message_id,
            backend=backend, target_request_id=turn.agent_request_id,
            request_id=f"cancel_{request.turn_id}_{request.message_id}",
            origin="user", cancel=True,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail=str(result.get("error") or "cancellation rejected"))
    return {"ok": True, "detail": result.get("detail", "cancelled"), "message_id": request.message_id}


class ToolResultRequest(BaseModel):
    """Answer to a paused AskUserQuestion tool call.

    Routed through Redis (chat.tool_result action) to the claude-code bridge,
    which resolves the pending asyncio.Future keyed by tool_use_id.  The SDK
    turn resumes immediately on its own — this endpoint does not stream the
    continuation; the originating SSE connection (or its resume path) does.
    """
    bot_id: str = Field(..., description="Bot slug whose paused turn this answers")
    user_id: str = Field("nick", description="User id (scopes the routing key)")
    tool_use_id: str = Field(..., description="SDK tool_use id from the AWAIT_TOOL_RESULT event")
    result: str = Field(..., description="User's answer (label, free text, or JSON-encoded multi-pick)")


class ToolResultResponse(BaseModel):
    ok: bool
    detail: str | None = None


@router.post("/v1/chat/tool-result", tags=["Agent Backends"], deprecated=True)
async def chat_tool_result(request: ToolResultRequest) -> ToolResultResponse:
    """DEPRECATED (TASK-269).  Use ``POST /v1/chat/questions/{id}/answer``.

    The blocking pause/resume model is gone: AskUserQuestion no longer holds
    the SDK turn open, so there is no Future to resolve.  Answers are now
    recorded canonically and delivered to the agent as a continuation turn.
    We still record the answer here so a stale client doesn't silently drop
    it, but the continuation must be dispatched via the new endpoint/flow.
    """
    service = get_service()
    pq_store = getattr(service, "_pending_question_store", None)
    if pq_store is not None and request.tool_use_id:
        try:
            pq_store.record_answer(
                tool_use_id=request.tool_use_id, answer=request.result,
            )
        except Exception:
            log.debug("deprecated tool-result record_answer failed", exc_info=True)
    raise HTTPException(
        status_code=410,
        detail=(
            "POST /v1/chat/tool-result is deprecated; the answer was recorded "
            "but use POST /v1/chat/questions/{question_id}/answer to deliver it "
            "as a continuation turn."
        ),
    )


# ---------------------------------------------------------------------------
# TASK-269 — canonical question/answer + continuation
# ---------------------------------------------------------------------------

class QuestionResponseItem(BaseModel):
    """One sub-question's answer in the canonical structured form."""
    question_id: str | None = Field(default=None, description="Sub-question id/header")
    selected: list[str] = Field(default_factory=list, description="Chosen option labels")
    other: str | None = Field(default=None, description="Free text for an 'Other' choice")


class QuestionAnswerRequest(BaseModel):
    """Answer to a deferred AskUserQuestion (TASK-269)."""
    bot_id: str = Field(..., description="Bot the continuation turn should run on")
    user_id: str = Field("nick", description="User id (scopes routing + events)")
    responses: list[QuestionResponseItem] | None = Field(
        default=None, description="Canonical structured answer, one entry per sub-question",
    )
    free_text: str | None = Field(default=None, description="Optional extra free text")
    # Pre-formatted flat answer; if given it wins over `responses` formatting.
    result: str | None = Field(default=None, description="Pre-formatted answer text")
    # Skip/dismiss: resolve the question WITHOUT sending anything back to the
    # agent.  Marks it "skipped" (leaves the awaiting set, never reappears on
    # reload) and returns an empty continuation_prompt so the client dispatches
    # no continuation turn.
    dismiss: bool = Field(default=False, description="Dismiss without answering")


class QuestionAnswerResponse(BaseModel):
    ok: bool
    detail: str | None = None
    question_id: str
    bot_id: str
    origin_harness: str = "claude"
    # The text the client should send as the continuation user message.
    continuation_prompt: str
    # The awaiting turn this answers — pass as parent_turn_id on the continuation.
    parent_turn_id: str | None = None
    already_answered: bool = False


def _format_continuation_prompt(
    question_args: dict | None,
    responses: list[QuestionResponseItem] | None,
    free_text: str | None,
    result: str | None,
) -> str:
    """Render a user's structured answer into a natural continuation message.

    The resumed agent reads this as a normal user message; the spike proved a
    clearly-framed answer ("My answer to your question: …") lets the model pick
    up coherently without any transcript surgery (same-bot resume).
    """
    if result and result.strip():
        return result.strip()
    # Terse + natural: this string is BOTH the agent's continuation input and
    # the user's visible chat bubble.  The agent already has the question in its
    # session (it asked it), so a concise "Header: pick" reads naturally in the
    # transcript and resumes coherently — no "My answer to your question:" frame.
    lines: list[str] = []
    for r in (responses or []):
        picked = ", ".join([s for s in (r.selected or []) if s])
        if r.other:
            picked = f"{picked}, {r.other}" if picked else r.other
        if not picked:
            continue
        header = (r.question_id or "").strip()
        lines.append(f"{header}: {picked}" if header else picked)
    body = "; ".join(lines)
    if free_text and free_text.strip():
        body = f"{body} — {free_text.strip()}" if body else free_text.strip()
    return body or "(no answer provided)"


@router.post("/v1/chat/questions/{question_id}/answer", tags=["Agent Backends"])
async def answer_question(question_id: str, request: QuestionAnswerRequest) -> QuestionAnswerResponse:
    """Record a user's answer to a deferred AskUserQuestion (TASK-269).

    Persists the canonical answer, flips the question to ``answered``, and
    fans out a ``question_answered`` unified event so every tab converges.
    Returns the ``continuation_prompt`` + ``parent_turn_id`` the client uses to
    dispatch the continuation turn (a normal streaming /v1/chat/completions
    call carrying the answer back to the agent).  Idempotent — a re-POST of an
    already-answered question returns the recorded answer with
    ``already_answered=true`` and does not double-record.
    """
    service = get_service()
    pq_store = getattr(service, "_pending_question_store", None)
    if pq_store is None:
        raise HTTPException(status_code=503, detail="Question store unavailable")

    row = pq_store.get(question_id)
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"No question recorded for id={question_id}",
        )

    try:
        question_args = json.loads(row.arguments_json) if row.arguments_json else {}
    except Exception:
        question_args = {}

    async def _fanout_resolved(answer_text: str) -> None:
        """Fan out a question_answered event so every open tab clears its
        picker (used by both the answer and the dismiss paths)."""
        try:
            from ...agent_backends.agent_bridge import get_agent_subscriber
            sub = get_agent_subscriber()
            if sub is not None:
                await sub._redis.xadd(  # type: ignore[attr-defined]
                    f"events:{request.bot_id}:{request.user_id}",
                    {"payload": json.dumps({
                        "_type": "question_answered",
                        "bot_id": request.bot_id,
                        "user_id": request.user_id,
                        "question_id": question_id,
                        "turn_id": row.turn_id,
                        "answer": answer_text,
                        "ts": time.time(),
                    }, ensure_ascii=False, default=str)},
                    maxlen=5000,
                    approximate=True,
                )
        except Exception:
            log.debug("failed to publish question_answered for id=%s", question_id, exc_info=True)

    already = row.status == "answered"
    if already:
        prompt = row.answer or _format_continuation_prompt(
            question_args, request.responses, request.free_text, request.result,
        )
        return QuestionAnswerResponse(
            ok=True, detail="already_answered", question_id=question_id,
            bot_id=row.bot_id, origin_harness=getattr(row, "origin_harness", "claude"),
            continuation_prompt=prompt, parent_turn_id=row.turn_id, already_answered=True,
        )

    if request.dismiss:
        # Skip = dismiss without resuming the agent.  Resolve the row as
        # "skipped" (drops it from the awaiting set so it never reappears on
        # reload), clear every tab's picker, and return an empty continuation
        # so the client dispatches no turn.
        pq_store.mark(tool_use_id=question_id, status="skipped")
        await _fanout_resolved("")
        log.info(
            "Question dismissed (skipped): id=%s bot=%s parent_turn=%s — no continuation",
            question_id, request.bot_id, row.turn_id,
        )
        return QuestionAnswerResponse(
            ok=True, detail="dismissed", question_id=question_id,
            bot_id=request.bot_id, origin_harness=getattr(row, "origin_harness", "claude"),
            continuation_prompt="", parent_turn_id=row.turn_id,
        )

    prompt = _format_continuation_prompt(
        question_args, request.responses, request.free_text, request.result,
    )
    responses_json = [r.model_dump() for r in (request.responses or [])]
    updated = pq_store.record_answer(
        tool_use_id=question_id,
        answer=prompt,
        answer_json=responses_json or None,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Question disappeared during answer")

    # Fan out so other tabs flip the QuestionMessage to its answered state.
    await _fanout_resolved(prompt)

    log.info(
        "Question answered: id=%s bot=%s parent_turn=%s — client will dispatch continuation",
        question_id, request.bot_id, row.turn_id,
    )
    return QuestionAnswerResponse(
        ok=True, detail="recorded", question_id=question_id,
        bot_id=request.bot_id, origin_harness=getattr(row, "origin_harness", "claude"),
        continuation_prompt=prompt, parent_turn_id=row.turn_id,
    )


class PendingQuestionItem(BaseModel):
    """Single row of the pending-questions API."""
    tool_use_id: str
    tool_name: str
    origin_harness: str = "claude"
    bot_id: str
    user_id: str
    turn_id: str
    trigger_message_id: str | None = None
    session_key: str | None = None
    arguments: dict
    status: str
    answer: str | None = None
    answer_json: list | dict | None = None
    answered_turn_id: str | None = None
    created_at: str | None = None
    answered_at: str | None = None


class PendingQuestionList(BaseModel):
    data: list[PendingQuestionItem]


@router.get("/v1/chat/pending-questions", tags=["Agent Backends"])
async def list_pending_questions(
    bot_id: str | None = None,
    user_id: str | None = None,
    limit: int = 50,
    include_resolved: bool = False,
) -> PendingQuestionList:
    """Return AskUserQuestion rows for a bot/user scope.

    By default returns only currently-awaiting questions (used to hydrate any
    open pickers on page load).  With ``include_resolved=true`` it returns
    recent questions of ALL statuses so the UI can render resolved
    (answered/skipped) questions as a read-only record — and, crucially, so it
    can tell a resolved question apart from one whose awaiting row simply
    hasn't loaded yet (which must keep showing the live picker, not a stale
    read-only card).
    """
    service = get_service()
    pq_store = getattr(service, "_pending_question_store", None)
    if pq_store is None:
        return PendingQuestionList(data=[])
    capped = max(1, min(int(limit or 50), 200))
    rows = (
        pq_store.list_recent(bot_id=bot_id, user_id=user_id, limit=capped)
        if include_resolved
        else pq_store.list_awaiting(bot_id=bot_id, user_id=user_id, limit=capped)
    )
    return PendingQuestionList(
        data=[PendingQuestionItem(**pq_store.row_to_dict(r)) for r in rows],
    )


class SessionResetRequest(BaseModel):
    """Request to reset an agent backend session."""
    bot_id: str = Field(..., description="Bot slug to reset session for")
    # TASK-257 (Gavel finding): rotation + thread_switched emit must target
    # the CALLER's user context, not blindly DEFAULT_USER — otherwise a reset
    # issued for another user rotates the default user's thread and publishes
    # on their stream. Optional for backward compatibility (single-user
    # deployments omit it and get DEFAULT_USER exactly as before).
    user_id: str | None = Field(
        None, description="User whose thread rotates; defaults to DEFAULT_USER"
    )


class SessionResetResponse(BaseModel):
    ok: bool
    bot_id: str
    session_key: str | None = None
    detail: str | None = None


@router.post("/v1/chat/session/reset", tags=["Agent Backends"])
async def session_reset(request: SessionResetRequest) -> SessionResetResponse:
    """Reset the agent backend session for a bot.

    Sends a session.reset RPC for frontend/provider reset signaling, then
    rotates the durable DB thread. Claude/Codex continuity is per-thread, so
    the fresh thread naturally cold-starts on the next message; OpenClaw keeps
    its gateway-owned routing session behavior.
    """
    service = get_service()
    from ...bots import BotManager

    bot = BotManager(service.config).get_bot(request.bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail=f"Bot '{request.bot_id}' not found")

    backend_name = getattr(bot, "agent_backend", None)
    if not backend_name:
        raise HTTPException(status_code=400, detail=f"Bot '{request.bot_id}' is not an agent backend bot")

    bc = getattr(bot, "agent_backend_config", {}) or {}
    # Claude/Codex route RPCs by bot:user and persist SDK continuity on the
    # durable thread, not in agent_backend_config. The bot slug is sufficient
    # for reset signaling; OpenClaw still uses its gateway routing key.
    if backend_name in ("claude-code", "codex"):
        session_key = request.bot_id
    else:
        session_key = bc.get("session_key", "")

    from ...agent_backends.agent_bridge import get_agent_subscriber

    subscriber = get_agent_subscriber()
    if not subscriber:
        raise HTTPException(status_code=503, detail="Redis subscriber not available")

    # Resolve the caller before the RPC so reset signaling lands on that
    # user's unified stream. ``clearProviderSession`` remains meaningful only
    # for gateway-owned OpenClaw sessions; Claude/Codex reset by thread rotation.
    default_user = (getattr(service.config, "DEFAULT_USER", "") or "").strip()
    user_id = (request.user_id or "").strip() or default_user
    clear_provider = backend_name == "openclaw"

    rpc_req_id = f"reset_{uuid.uuid4().hex}"
    try:
        await subscriber.send_rpc(
            "session.reset",
            {
                "sessionKey": session_key,
                "userId": user_id,
                "clearProviderSession": clear_provider,
            },
            rpc_req_id,
            timeout_s=10,
            backend=backend_name,
        )
    except Exception as e:
        log.warning("session.reset RPC failed for bot %s: %s", request.bot_id, e)
        # Even if the bridge fails to clear gateway-side context, we still
        # return ok — every bridge (claude-code, codex, openclaw) now
        # intercepts session.reset and emits a SESSION_RESET unified event
        # on a best-effort path, so the frontend-visible reset still
        # happens.  See TASK-249.

    # TASK-284: an explicit session reset also rotates the durable DB thread
    # (non-destructive) so the provider's fresh session maps onto a fresh
    # thread — same coordination as agent /new. Best-effort: a rotation
    # failure never fails the reset.
    rotated_thread: str | None = None
    try:
        if user_id:
            from ...mcp_server.storage import get_storage

            rotated_thread = await get_storage().rotate_session(
                bot_id=request.bot_id, user_id=user_id
            )
            log.info(
                "Rotated durable thread on session reset: bot=%s -> %s",
                request.bot_id, rotated_thread,
            )
            # TASK-257: deterministic thread_switched event for the UI's
            # conversation list (same emit as the /new path; source differs).
            if rotated_thread:
                service._publish_thread_switched(
                    request.bot_id, user_id, str(rotated_thread),
                    source="session_reset",
                )
    except Exception as e:
        log.warning("Thread rotation on reset failed for %s: %s", request.bot_id, e)

    log.info("Session reset requested: bot=%s session=%s", request.bot_id, session_key)
    return SessionResetResponse(
        ok=True,
        bot_id=request.bot_id,
        session_key=session_key,
        detail="reset",
    )


@router.post("/v1/chat/completions", tags=["OpenAI Compatible"])
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """
    Create a chat completion (OpenAI-compatible).
    
    Supports all standard OpenAI parameters plus llm_bawt extensions:
    - `bot_id`: Bot personality to use (default: nova)
    - `augment_memory`: Whether to include memory context (default: true)
    - `extract_memory`: Whether to extract memories from response (default: true)
    - `include_summaries`: Whether to inject conversation summary records into context (default: true)
    - `tts_mode`: Whether to append TTS output formatting instructions to the system prompt (default: false)
    """
    from fastapi.responses import StreamingResponse
    
    service = get_service()

    internal_sender = _internal_inter_bot_sender(http_request)
    if internal_sender:
        request._internal_inter_bot_sender_id = internal_sender

    # Log request BEFORE validation so we can debug failures
    log.debug(f"Request payload: {request.model_dump(exclude_none=True)}")

    # TASK-251: explicit thread selection — validate ownership/lifecycle up
    # front (404 for missing/cross-user, 410 for deleted) so a bad thread id
    # fails loudly instead of silently persisting the turn to a dead thread.
    # Absent session_id = continuous default; no validation, no DB read.
    if getattr(request, "session_id", None):
        from ..dependencies import get_effective_bot_id
        from .sessions import _owned_session_or_404, _resolve_user

        await _owned_session_or_404(
            request.session_id.strip(),
            get_effective_bot_id(request.bot_id),
            _resolve_user(request.user),
        )

    if request.stream:
        # Streaming response
        try:
            return StreamingResponse(
                service.chat_completion_stream(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log.exception("Streaming chat completion failed")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Non-streaming
    try:
        response = await service.chat_completion(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Chat completion failed")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------------
# Task Management Endpoints
# -------------------------------------------------------------------------
