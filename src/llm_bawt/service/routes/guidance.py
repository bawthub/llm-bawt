"""User guidance approval routes (TASK-893 slice 5).

BawtHub's server is the only caller (service token). It refuses bots and
scheduled jobs and binds ``user_id`` to the signed-in person; the decider is
always that ``user_id``, never a body field. ``guidance.decide`` additionally
refuses bot identities and stale bases. Nothing here runs history or drafts.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool

from llm_bawt.approval_mcp_store import _SENTINEL_IDENTITIES
from llm_bawt.approval_models import BOT_RESOLVER_PREFIX
from llm_bawt.memory import guidance, guidance_view
from llm_bawt.memory.guidance_run import resolve_protected
from llm_bawt.service import state_events
from llm_bawt.service.dependencies import get_service
from llm_bawt.utils.db import get_shared_engine

from .service_auth import require_service_token

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_service_token)], tags=["Guidance"])


def _engine():
    engine = get_shared_engine(get_service().config)
    if engine is None:
        raise HTTPException(503, "guidance_store_unavailable")
    return engine


def _person(user_id: str) -> str:
    u = (user_id or "").strip()
    if not u or u.lower() in _SENTINEL_IDENTITIES or u.lower().startswith(BOT_RESOLVER_PREFIX):
        raise HTTPException(403, "a signed-in person is required")
    return u


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str
    approve: bool


@router.get("/v1/guidance")
async def get_guidance(user_id: str):
    user_id = _person(user_id)

    def read():
        with _engine().connect() as conn:
            protected = resolve_protected(conn)
            if not protected:
                raise HTTPException(503, "protected bots unresolved")
            return guidance_view.dumps(guidance_view.view(conn, user_id=user_id, protected=protected))
    return await run_in_threadpool(read)


@router.post("/v1/guidance/proposals/{proposal_id}/decide")
async def decide_proposal(proposal_id: int, body: DecideRequest):
    user_id = _person(body.user_id)

    def run():
        engine = _engine()
        with engine.connect() as conn:
            if conn.execute(text("SELECT to_regclass(:t)"), {"t": guidance.PROPOSALS}).scalar() is None:
                raise HTTPException(404, "no such proposal")
            owner = conn.execute(text(f"SELECT user_id FROM {guidance.PROPOSALS} WHERE id = :id"),
                                 {"id": proposal_id}).scalar()
        if owner is None:
            raise HTTPException(404, "no such proposal")
        if owner != user_id:
            raise HTTPException(403, "proposal owner mismatch")
        try:
            return guidance.decide(engine, proposal_id=proposal_id, decided_by=user_id,
                                   approve=body.approve)
        except PermissionError as e:
            raise HTTPException(403, str(e)) from e
        except ValueError as e:
            raise HTTPException(409, str(e)) from e

    result = await run_in_threadpool(run)
    try:
        await state_events.publish_state_changed(get_service(), [state_events.GUIDANCE])
    except Exception:  # the decision is committed; refresh is best effort
        logger.warning("guidance state_changed publish failed", exc_info=True)
    return result
