"""OpenAI-compatible chat completion route."""

import asyncio
import json
import time
from typing import TYPE_CHECKING, Literal

from agent_bridge.steer_contract import STEER_CAPABILITY_REFUSALS, STEER_RUN_FINISHING
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...message_authorship import AuthorReference
from ...request_identity import AdmissionUnavailable, IdentityConflict, InvalidIdentity
from ..dependencies import get_service
from ..gate_steer_escalation import SendNow
from ..logging import get_service_logger
from ..schemas import ChatCompletionRequest

if TYPE_CHECKING:
    from ...inter_bot_control_lane import DeliveryControlGrant

router = APIRouter()
log = get_service_logger(__name__)
_DISMISSED_QUESTION_RESULT = "[Question dismissed by user.]"


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
    # A user Stop's queue hold (TASK-1327): {gate_id, expires_at, queued_count}.
    queue_hold: dict | None = None


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
    return ChatAbortResponse(ok=True, detail=detail, turn_id=turn.id,
                             queue_hold=await _queue_hold_summary(store.engine, turn.bot_id, request.source))


async def _queue_hold_summary(engine, bot_id: str | None, source: str) -> dict | None:
    from ..user_stop_hold import USER_STOP_SOURCES, snapshot

    if source not in USER_STOP_SOURCES or not bot_id:
        return None
    try:
        hold = await asyncio.to_thread(snapshot, engine, bot_id)
    except Exception:  # the Stop already happened; the strip rehydrates from GET
        log.exception("Queue hold snapshot failed for %s", bot_id)
        return None
    return hold.summary() if hold is not None else None


class ChatSteerRequest(BaseModel):
    """Redirect an in-flight Claude agent turn without creating another turn."""

    turn_id: str = Field(..., description="Active llm-bawt turn log ID")
    message: str = Field(..., min_length=1, description="User correction/instruction")
    message_id: str = Field(..., description="Frontend-generated user message UUID")
    content_offset: int | None = Field(None, ge=0, description="Assistant characters visible when steering")
    bot_id: str = Field(..., description="Bot slug owning the active turn")
    user_id: str = Field("nick", description="User namespace owning the turn")
    allow_queue: bool = Field(True, description="Allow maintenance to queue a separate turn; false for same-turn recovery")
    priority: Literal["now", "next"] = Field(
        "next", description="Queue until the next model call without interrupting running tools"
    )


# Steer errors that are a definite answer about the run, never a transport fault.
STEER_DEFINITE_409 = STEER_CAPABILITY_REFUSALS | {"no_active_run", "active_run_mismatch", STEER_RUN_FINISHING}


def _run_steer_caps(service, turn) -> tuple[str, frozenset[str]]:
    """``(backend, steer capabilities)`` of the turn's bound run (TASK-1243).

    The bot's current backend is read only for a row that never recorded one.
    """
    from ...steer_capabilities import turn_steer_caps

    bot_backend = ""
    if not (getattr(turn, "agent_backend", None) or "").strip():
        from ...bots import BotManager

        bot = BotManager(service.config).get_bot(turn.bot_id)
        bot_backend = getattr(bot, "agent_backend", None) or ""
    return turn_steer_caps(turn, bot_backend=bot_backend)


class ChatSteerResponse(BaseModel):
    ok: bool
    detail: str
    turn_id: str
    message_id: str
    persisted: bool = False
    can_escalate: bool = False
    can_cancel: bool = False
    queued: bool = False
    delivery_id: str | None = None


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


def _steer_framing(
    message: str, *, origin: str, priority: str, bot_id: str, user_id: str,
) -> dict[str, str]:
    """The registry-rendered ``chat.steer`` framing fields for this steer (TASK-1300).

    A ``system`` steer is already final registry text and goes verbatim. A
    ``next`` steer also carries its interruption item, which the bridge replays
    if "Interrupt now" later sweeps it.
    """
    if origin == "system":
        return {}
    from ...steer_prompts import interrupt_item, render_next, render_now

    item = interrupt_item(message, origin=origin, user_id=user_id)
    if priority == "next":
        return {"framed_message": render_next(message, bot_id=bot_id), "interrupt_item": item}
    return {"framed_message": render_now(item, bot_id=bot_id)}


async def steer_active_turn(
    service,
    request: ChatSteerRequest,
    *,
    steer_request_id: str | None = None,
    author: AuthorReference | None = None,
    wire_suffix: str | None = None,
    control: "DeliveryControlGrant | None" = None,
    send_now_delivery_id: str | None = None,
) -> ChatSteerResponse:
    """Shared steer implementation for public chat and durable inter-bot work.

    ``wire_suffix`` is internal-only, model-visible instruction text (the
    inter-bot reply-channel note). It is appended to what the bridge receives
    and never persisted or published: history and the timeline keep
    ``request.message``. The public steer route cannot set it.

    Under a live system gate every steer is refused with ``SteerGateRefused``
    before any bridge RPC (TASK-1233). ``control`` (a claimed control-lane
    delivery only, TASK-1431; the public route cannot set it) is admitted on
    its delivery claim and linking member, and then addresses only the pinned
    run: its request id, backend and session, never the bot's current ones.
    """
    if control is not None and not steer_request_id:
        raise HTTPException(status_code=409, detail="control steer requires a steer_request_id")
    turn = service._turn_log_store.get_turn(request.turn_id)
    if turn is None:
        raise HTTPException(status_code=404, detail="Turn not found")
    bot_id = request.bot_id.strip().lower()
    user_id = request.user_id.strip().lower() or "nick"
    if turn.bot_id != bot_id or (turn.user_id or "").lower() != user_id:
        raise HTTPException(status_code=409, detail="Turn ownership mismatch")
    # Inter-bot steers pass their sender; a composer steer is the turn owner's.
    author = author or AuthorReference.user(user_id)
    from agent_bridge.maintenance import is_compact_command
    if not steer_request_id and (
        is_compact_command(getattr(turn, "user_prompt", None)) or is_compact_command(request.message)
    ):
        if not request.allow_queue:
            # Exact recovery must never become a separately queued turn.
            # Ordinary composer messages retain maintenance queue behavior.
            raise HTTPException(status_code=409, detail="maintenance_requires_separate_turn")
        from ..maintenance_queue import MaintenanceQueue
        return ChatSteerResponse(**await MaintenanceQueue(service).enqueue_user(request, turn))
    if (
        turn.ended_at is not None or turn.status not in ("streaming", "pending")
    ) and not steer_request_id:
        raise HTTPException(status_code=409, detail="Turn is no longer active")
    # A maintenance-time message above is an enqueue and asks the gate there.
    from ..system_gate_admission import check_steer
    engine = getattr(service._turn_log_store, "engine", None)
    target = None
    if control is not None:
        target = await asyncio.to_thread(check_steer, engine, bot_id, turn_id=turn.id, grant=control)
        session_key, backend_name = target.session_key, (target.backend or "").strip()
        caps = target.steer_caps
    else:
        forced = {"send_now": send_now_delivery_id} if send_now_delivery_id else {}
        await asyncio.to_thread(check_steer, engine, bot_id, turn_id=turn.id, **forced)
        session_key, backend_name, caps = turn.agent_session_key, None, None
    if not session_key:
        # A deterministic delivery retry may inspect the original target after it
        # ended. If the run never registered a bridge session, no steer RPC could
        # possibly have been submitted; report a definitive rejection so the
        # dispatcher can clear stale steer intent and choose one safe fallback.
        if steer_request_id and turn.ended_at is not None:
            raise HTTPException(status_code=409, detail="no_active_run")
        raise HTTPException(status_code=409, detail="Active bridge run is not ready")

    from ...agent_backends.agent_bridge import get_agent_subscriber
    from ...steer_capabilities import CANCEL_QUEUED, ESCALATE, steer_refusal
    # TASK-1243: the run's own pinned capabilities, never the bot's current backend.
    if caps is None:
        backend_name, caps = _run_steer_caps(service, turn)
    request.priority = SendNow.priority() if send_now_delivery_id else request.priority  # TASK-1749
    refusal = steer_refusal(caps, request.priority)
    if refusal is not None:
        raise HTTPException(status_code=409, detail=refusal)

    message = request.message.strip()
    wire_message = f"{message}\n\n{wire_suffix}" if wire_suffix else message
    origin = "user" if steer_request_id is None or send_now_delivery_id else "system" if control else "bot"
    from ...steer_prompts import SteerPromptUnavailable
    try:
        framing = await asyncio.to_thread(
            _steer_framing, wire_message, origin=origin, priority=request.priority,
            bot_id=bot_id, user_id=user_id,
        )
    except SteerPromptUnavailable as exc:
        # Never sent unframed: refuse before any bridge RPC.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    subscriber = get_agent_subscriber()
    if subscriber is None:
        raise HTTPException(status_code=503, detail="Agent bridge unavailable")

    result: dict = {"ok": False, "error": "no_active_run"}
    # The bridge dedupes a steer on its request id and replays the first
    # result, so re-sending a caller-chosen id only re-reads that result. A
    # caller that names the id owns its retry (TASK-1251: the durable
    # dispatcher retries under a new steer generation). Composer steers pass
    # no id and keep their retry loop unchanged.
    attempts = 1 if steer_request_id else 8
    try:
        for attempt in range(attempts):
            result = await subscriber.send_steer(
                session_key=session_key,
                message=wire_message,
                message_id=request.message_id,
                backend=backend_name,
                target_request_id=target.request_id if target is not None else turn.agent_request_id,
                request_id=steer_request_id,
                priority=request.priority,
                origin=origin,
                **framing,
            )
            if result.get("ok") or result.get("error") != "no_active_run":
                break
            if attempt < attempts - 1:
                await asyncio.sleep(0.25)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    if not result.get("ok"):
        detail = str(result.get("error") or result.get("detail") or "steer failed")
        # TASK-1344: ``run_finishing`` is definite too; the run already gave its
        # final result, so callers send the message as a fresh turn.
        status = 409 if detail in STEER_DEFINITE_409 else 502
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
        from ..steer_timeline import SteerTimeline

        timeline = SteerTimeline(
            service, bot_id=bot_id, user_id=user_id,
            turn_id=turn.id, message_id=request.message_id,
        )
        content_offset = await timeline.content_offset(subscriber, request.content_offset)
        await asyncio.to_thread(
            memory_client.add_message,
            "user",
            message,
            session_id,
            None,
            request.message_id,
            None,
            None,
            author.entity_type,
            author.entity_id,
        )
        if content_offset is not None:
            anchored = await asyncio.to_thread(
                _persist_interrupt_anchor,
                service,
                bot_id=bot_id,
                message_id=request.message_id,
                source_message_id=turn.trigger_message_id,
                content_offset=content_offset,
            )
            if not anchored:
                raise RuntimeError("Persisted steer message was not found for anchor update")
        persisted = True
        await timeline.publish(subscriber)
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
        can_escalate=bool(result.get("can_escalate")) and request.priority == "next" and ESCALATE in caps,
        can_cancel=bool(result.get("can_cancel")) and request.priority == "next" and CANCEL_QUEUED in caps,
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
    """Escalate a queued user message, only on its original active run.

    Escalation interrupts the running tool, so a live system gate refuses it
    like any steer (TASK-1233). ``/steer/cancel`` only withdraws a queued
    steer and stays open.
    """
    service = get_service()
    turn = service._turn_log_store.get_turn(request.turn_id)
    if turn is None:
        raise HTTPException(status_code=404, detail="Turn not found")
    if (turn.ended_at is not None or turn.status not in ("streaming", "pending")
            or turn.bot_id != request.bot_id.strip().lower()
            or (turn.user_id or "").lower() != request.user_id.strip().lower()):
        raise HTTPException(status_code=409, detail="Turn is no longer active or ownership mismatch")
    from ..system_gate_admission import check_steer

    await asyncio.to_thread(
        check_steer, getattr(service._turn_log_store, "engine", None), turn.bot_id, turn_id=turn.id,
    )
    if not turn.agent_session_key or not turn.agent_request_id:
        raise HTTPException(status_code=409, detail="Active bridge run is not ready")
    from ...agent_backends.agent_bridge import get_agent_subscriber
    from ...steer_capabilities import ESCALATE, steer_refusal

    backend, caps = _run_steer_caps(service, turn)
    refusal = steer_refusal(caps, ESCALATE)
    if refusal is not None:
        raise HTTPException(status_code=409, detail=refusal)
    subscriber = get_agent_subscriber()
    if subscriber is None:
        raise HTTPException(status_code=503, detail="Agent bridge unavailable")
    from ...steer_prompts import SteerPromptUnavailable, render_interrupt_frame

    # TASK-1300: the bridge fills this frame with every command the sweep removes.
    try:
        interrupt_frame = await asyncio.to_thread(render_interrupt_frame, turn.bot_id)
    except SteerPromptUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        result = await subscriber.send_steer(
            session_key=turn.agent_session_key, message="", message_id=request.message_id,
            backend=backend, target_request_id=turn.agent_request_id,
            request_id=f"escalate_{request.turn_id}_{request.message_id}",
            origin="user", escalate=True, interrupt_frame=interrupt_frame,
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
    from ...steer_capabilities import CANCEL_QUEUED, steer_refusal

    backend, caps = _run_steer_caps(service, turn)
    refusal = steer_refusal(caps, CANCEL_QUEUED)
    if refusal is not None:
        raise HTTPException(status_code=409, detail=refusal)
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


class QuestionTimeoutRequest(BaseModel):
    session_key: str


@router.post("/v1/chat/questions/{question_id}/timeout", tags=["Agent Backends"])
async def timeout_question(question_id: str, request: QuestionTimeoutRequest) -> dict:
    """CAS the live hold into durable deferral; report the answer if it won."""
    store = get_service()._pending_question_store
    row = store.get(question_id)
    if row is None or row.session_key != request.session_key:
        raise HTTPException(status_code=404, detail="Question not found for session")
    expired = store.expire_live(question_id)
    current = store.get(question_id)
    answered = current is not None and current.status == "answered"
    dismissed = current is not None and current.status == "skipped"
    return {
        "expired": expired or (current is not None and current.status == "awaiting"),
        # Native AskUserQuestion input; the bridge allows the tool with it.
        "answers": store.tool_answers(current) if answered else None,
        # Deny text for a dismissal (never an answer).
        "message": _DISMISSED_QUESTION_RESULT if dismissed else None,
        "dismissed": dismissed,
    }


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
    answer: str | None = None
    # The awaiting turn this answers — pass as parent_turn_id on the continuation.
    parent_turn_id: str | None = None
    already_answered: bool = False
    live: bool = False


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
    """Record the answer and atomically choose live delivery or continuation.

    A live CAS winner sends chat.tool_result to the held SDK turn. Once the
    timeout CAS wins, return the continuation prompt as before. Repeated
    answers are idempotent and never start a second continuation.
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
    if row.bot_id != request.bot_id or row.user_id != request.user_id:
        raise HTTPException(status_code=404, detail="Question not found for bot/user")

    async def _deliver_live(result: str, answers: dict[str, str] | None = None) -> None:
        """``answers`` → bridge allows the tool natively; otherwise ``result`` is deny text."""
        from ...agent_backends.agent_bridge import get_agent_subscriber
        sub = get_agent_subscriber()
        if sub is None or not row.session_key:
            raise HTTPException(status_code=503, detail="Question bridge unavailable")
        await sub.send_tool_result(
            row.session_key, question_id, result, backend="claude-code", answers=answers,
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
        return QuestionAnswerResponse(
            ok=True, detail="already_answered", question_id=question_id,
            bot_id=row.bot_id, origin_harness=getattr(row, "origin_harness", "claude"),
            continuation_prompt="", answer=row.answer,
            parent_turn_id=row.turn_id, already_answered=True,
        )

    if request.dismiss:
        # Skip = dismiss without resuming the agent.  Resolve the row as
        # "skipped" (drops it from the awaiting set so it never reappears on
        # reload), clear every tab's picker, and return an empty continuation
        # so the client dispatches no turn.
        live = row.status == "awaiting_live" and pq_store.dismiss_live(question_id)
        if not live:
            pq_store.expire_live(question_id)
            if not pq_store.dismiss_deferred(question_id):
                current = pq_store.get(question_id)
                if current is not None and current.status == "answered":
                    return QuestionAnswerResponse(
                        ok=True, detail="already_answered", question_id=question_id,
                        bot_id=row.bot_id, origin_harness=row.origin_harness,
                        continuation_prompt="", answer=current.answer,
                        parent_turn_id=row.turn_id, already_answered=True,
                    )
                if current is None or current.status != "skipped":
                    raise HTTPException(status_code=409, detail="Question cannot be dismissed")
        if live:
            try:
                await _deliver_live(_DISMISSED_QUESTION_RESULT)
            except Exception:
                log.exception("Live question dismissal delivery failed for %s", question_id)
        await _fanout_resolved("")
        log.info(
            "Question dismissed (skipped): id=%s bot=%s parent_turn=%s — no continuation",
            question_id, request.bot_id, row.turn_id,
        )
        return QuestionAnswerResponse(
            ok=True, detail="dismissed", question_id=question_id,
            bot_id=request.bot_id, origin_harness=getattr(row, "origin_harness", "claude"),
            continuation_prompt="", parent_turn_id=row.turn_id, live=live,
        )

    prompt = _format_continuation_prompt(
        question_args, request.responses, request.free_text, request.result,
    )
    responses_json = [r.model_dump() for r in (request.responses or [])]
    live = row.status == "awaiting_live" and pq_store.claim_live_answer(
        question_id, prompt, responses_json or None,
    )
    if live:
        # Persist first; if Redis is unavailable the bridge's timeout
        # arbitration can still read the winning answer from the DB.
        try:
            claimed_row = pq_store.get(question_id)
            await _deliver_live(
                prompt, answers=pq_store.tool_answers(claimed_row) if claimed_row else None,
            )
        except Exception:
            log.exception("Live question delivery failed for %s", question_id)
    else:
        pq_store.expire_live(question_id)
        claimed = pq_store.claim_deferred_answer(question_id, prompt, responses_json or None)
        if not claimed:
            updated = pq_store.get(question_id)
            if updated is None:
                raise HTTPException(status_code=404, detail="Question disappeared during answer")
            if updated.status != "answered":
                raise HTTPException(status_code=409, detail=f"Question is {updated.status}")
            return QuestionAnswerResponse(
                ok=True, detail="already_answered", question_id=question_id,
                bot_id=row.bot_id, origin_harness=row.origin_harness,
                continuation_prompt="", answer=updated.answer,
                parent_turn_id=row.turn_id, already_answered=True,
            )

    # Fan out so other tabs flip the QuestionMessage to its answered state.
    await _fanout_resolved(prompt)

    log.info(
        "Question answered: id=%s bot=%s parent_turn=%s — client will dispatch continuation",
        question_id, request.bot_id, row.turn_id,
    )
    return QuestionAnswerResponse(
        ok=True, detail="recorded", question_id=question_id,
        bot_id=request.bot_id, origin_harness=getattr(row, "origin_harness", "claude"),
        continuation_prompt="" if live else prompt, answer=prompt,
        parent_turn_id=row.turn_id, live=live,
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
    live_deadline: str | None = None
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
    idempotency_key: str | None = Field(default=None, max_length=256)
    include_working_memory: bool | None = Field(
        default=None,
        description=(
            "false starts the new session with working memory suppressed (no "
            "section, no change notice) for its lifetime; omitted or true pins it"
        ),
    )


class SessionResetResponse(BaseModel):
    ok: bool
    bot_id: str
    user_id: str
    reset: bool
    outcome: str
    old_session_id: str | None = None
    new_session_id: str
    receipt_message_id: str
    delivery_id: str | None = None
    turn_id: str | None = None


@router.post("/v1/chat/session/reset", tags=["Agent Backends"])
async def session_reset(request: SessionResetRequest) -> SessionResetResponse:
    """Submit one clean reset; only a provider-backed receipt is success."""
    service = get_service()
    from ...bots import BotManager
    from ..session_lifecycle.api import NewSessionAPI
    from ..session_lifecycle.intent import Continuity, Origin

    bot = BotManager(service.config).get_bot(request.bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail=f"Bot '{request.bot_id}' not found")
    if not getattr(bot, "agent_backend", None):
        raise HTTPException(status_code=400, detail=f"Bot '{request.bot_id}' is not an agent backend bot")
    user_id = (request.user_id or "").strip() or (
        getattr(service.config, "DEFAULT_USER", "") or ""
    ).strip()
    result = await NewSessionAPI.submit(
        service, bot_id=bot.slug, user_id=user_id,
        continuity=Continuity.CLEAN, origin=Origin.SESSION_API,
        idempotency_key=request.idempotency_key,
        include_working_memory=request.include_working_memory,
    )
    return SessionResetResponse(**result)


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

    Every caller gets the OpenAI contract: an admitted agent turn is observed
    to its reply, never returned as a queued receipt. Work that must be
    tracked before it completes submits /v1/inter-bot-deliveries instead.
    """
    from fastapi.responses import StreamingResponse
    from ..chat_response_adapter import completion_response, completion_stream

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
        admission_comment = http_request.headers.get("X-LLM-Bawt-Admission-Events") == "comment"
        # Non-OpenAI frames (server-run tool calls, warnings, usage, sidecars):
        # comments for a caller that translates them; data frames for a BawtHub
        # proxy built before that negotiation existed; dropped for a stock
        # OpenAI client, which would misread them.
        if http_request.headers.get("X-LLM-Bawt-Extension-Events") == "comment":
            extensions = "comment"
        else:
            extensions = "data" if admission_comment else "none"
        # Streaming response
        try:
            return StreamingResponse(
                completion_stream(
                    service, request, admission_comment=admission_comment, extensions=extensions,
                ),
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
        return await completion_response(service, request)
    except HTTPException:
        raise
    except (IdentityConflict, AdmissionUnavailable, InvalidIdentity) as e:
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log.exception("Chat completion failed")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------------
# Task Management Endpoints
# -------------------------------------------------------------------------
