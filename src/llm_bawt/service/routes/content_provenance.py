"""Bridge pre-dispatch exposure registration; never accepts an asserted actor."""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from agent_bridge.content_provenance import external_source
from ...content_provenance import ContentTurn
from ...task_turn_context import (
    TASK_TURN_CONTEXT_HEADER, TaskTurnContextError, open_task_turn_context,
)
from ...utils.bounded_worker import BoundedWorker
from ...utils.config import Config
from ..dependencies import get_content_provenance_store

router = APIRouter(prefix="/v1/agent-content", tags=["agent-content"])
_WORKER = BoundedWorker(4, "content-exposure")


class ExposureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: str


@router.post("/exposure")
async def register_exposure(body: ExposureRequest, request: Request):
    try:
        context = open_task_turn_context(request.headers.get(TASK_TURN_CONTEXT_HEADER))
    except TaskTurnContextError:
        raise HTTPException(403, "Verified turn capability required") from None
    source = external_source(body.tool_name)
    if source is None or source.startswith("bawthub:"):
        raise HTTPException(422, "Not a covered remote external tool")
    turn = ContentTurn(
        context.bot_id, context.user_id, context.turn_id, context.trigger_message_id,
    )
    try:
        await _WORKER.run(_expose, turn)
    except Exception:
        raise HTTPException(503, "Content provenance unavailable; do not execute") from None
    return {"registered": True, "source_turn": turn.turn_id, "untrusted_seen": True}


def _expose(turn):
    get_content_provenance_store(Config()).expose(turn)
