"""Typed, owner-scoped message routing API (LAN trust boundary)."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from ..dependencies import get_service
from ..message_routing import MessageRouter, RouteConflict, RouteRequest

router = APIRouter(prefix="/v1/message-routing", tags=["Message routing"])


class RouteOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=128)
    candidate_bot_ids: list[str] | None = None
    model_endpoint_id: int | None = Field(default=None, ge=1)
    session_policy: Literal["continue", "reset_retain_history", "reset_without_history"] = "continue"
    scope: Literal["personal", "schedule"] = "personal"

    def request(self) -> RouteRequest:
        return RouteRequest(**self.model_dump())


class RouteMessage(RouteOptions):
    message: str = Field(min_length=1, max_length=100000)
    idempotency_key: str = Field(min_length=1, max_length=256)
    project_id: str | None = Field(default=None, max_length=128)
    task_id: str | None = Field(default=None, max_length=128)

    def request(self) -> RouteRequest:
        return RouteRequest(**self.model_dump(include=set(RouteOptions.model_fields)))


class ConfigureCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=128)
    bot_id: str = Field(min_length=1, max_length=128)
    enabled: bool


def _router():
    try:
        return MessageRouter(get_service())
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/config")
@router.get("/candidates")
def candidates(user_id: str = Query(..., min_length=1, max_length=128)):
    try:
        return _router().candidates(RouteRequest(user_id=user_id))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.put("/config")
def configure_candidate(body: ConfigureCandidate):
    try:
        return _router().set_enabled(body.user_id, body.bot_id, body.enabled)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/preview")
def preview(body: RouteOptions):
    try:
        return _router().preview(body.request())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/route", status_code=202)
def route(body: RouteMessage, request: Request):
    provenance = None
    actor = None
    from ...task_turn_context import (
        TASK_TURN_CONTEXT_HEADER, TaskTurnContextError, open_task_turn_context,
    )

    capability = request.headers.get(TASK_TURN_CONTEXT_HEADER)
    if capability:
        try:
            caller = open_task_turn_context(capability)
        except TaskTurnContextError as exc:
            raise HTTPException(403, str(exc)) from exc
        if caller.user_id != body.user_id.strip().lower():
            raise HTTPException(403, "Signed caller cannot route for another owner")
        from ...message_authorship import AuthorReference

        actor = AuthorReference.bot(caller.bot_id)
        provenance = {"origin_bot_id": caller.bot_id,
                      "origin_turn_id": caller.turn_id}
    try:
        return _router().route(body.request(), message=body.message,
                               idempotency_key=body.idempotency_key,
                               project_id=body.project_id, task_id=body.task_id,
                               metadata=provenance, actor=actor)
    except RouteConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
