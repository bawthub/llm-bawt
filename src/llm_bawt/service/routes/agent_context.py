"""Agent context-health inspection and safe idle reset API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ...agent_context import AgentContextStore, normalize_session_policy
from ...bots import BotManager
from ...model_catalog import bot_model_ref
from ..dependencies import get_service

router = APIRouter()


class ContextResetRequest(BaseModel):
    bot_id: str
    user: str | None = None
    session_policy: str = Field(
        description="reset_retain_history or reset_without_history"
    )
    reason: str = Field(default="agent-requested context maintenance", max_length=1000)
    idempotency_key: str | None = Field(default=None, max_length=256)
    include_working_memory: bool | None = Field(
        default=None,
        description=(
            "false starts the new session with working memory suppressed (no "
            "section, no change notice) for its lifetime; omitted or true pins it"
        ),
    )


def _resolve(bot_id: str, user: str | None):
    service = get_service()
    slug = (bot_id or "").strip().lower()
    if not slug:
        raise HTTPException(status_code=422, detail="bot_id is required")
    bot = BotManager(service.config).get_bot(slug)
    if bot is None:
        raise HTTPException(status_code=404, detail="Bot not found")
    user_id = (user or getattr(service.config, "DEFAULT_USER", "nick") or "nick").strip()
    backend = bot.agent_backend or bot.harness
    model_ref = bot_model_ref(service.config, bot)
    ceiling = service.config.get_model_context_window(model_ref) if model_ref else None
    return service, bot, user_id, backend, ceiling


@router.get("/v1/agent-context/health", tags=["Agent Context"])
def get_agent_context_health(
    bot_id: str = Query(...),
    user: str | None = Query(None),
):
    service, bot, user_id, backend, ceiling = _resolve(bot_id, user)
    return AgentContextStore(service.config).health(
        bot_id=bot.slug,
        user_id=user_id,
        backend=backend,
        configured_ceiling=ceiling,
    )


@router.post("/v1/agent-context/reset", tags=["Agent Context"])
async def reset_agent_context(body: ContextResetRequest):
    service, bot, user_id, _backend, _ = _resolve(body.bot_id, body.user)
    from ..session_lifecycle.api import NewSessionAPI
    from ..session_lifecycle.intent import Continuity, Origin

    try:
        policy = normalize_session_policy(body.session_policy)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    continuity = {
        "reset_retain_history": Continuity.RETAIN,
        "reset_without_history": Continuity.CLEAN,
    }.get(policy.value)
    if continuity is None:
        raise HTTPException(status_code=422, detail="continue does not reset a session")
    return await NewSessionAPI.submit(
        service, bot_id=bot.slug, user_id=user_id, continuity=continuity,
        origin=Origin.CONTEXT_API, idempotency_key=body.idempotency_key,
        reason=body.reason.strip() or "agent-requested context maintenance",
        include_working_memory=body.include_working_memory,
    )
