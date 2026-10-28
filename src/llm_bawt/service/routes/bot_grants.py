"""Grant tag routes (TASK-1609): ``/v1/bot-grants``.

* ``GET /v1/bot-grants``: every current grant with who made it and when.
* ``PUT /v1/bot-grants/{tag}/bots/{slug}``: grant or revoke one grant tag.
  ``{granted, reason}``; idempotent, and a change writes a
  ``bot_profile_changes`` row.

The PUT takes one of two callers, checked before the body is read:

* a bot, by its signed turn capability (``X-LLM-Bawt-Task-Turn-Context``,
  forwarded by the ``bots_grant_set`` MCP tool). The actor is ``bot:<id>``
  and ``bot_tags.grants`` decides whether that bot manages grants;
* a person, by the TASK-1472 service bearer that only BawtHub's server holds,
  with the signed-in person as ``actor``.

Grant semantics live in ``bot_tags.grants``.
"""
from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from ...bot_tags import grants
from ...bot_tags.errors import BotTagError, GrantRefused
from ...task_turn_context import (
    TASK_TURN_CONTEXT_HEADER,
    TaskTurnContextError,
    open_task_turn_context,
)
from ..dependencies import get_runtime_settings_store, get_service
from . import settings as profile_routes
from .bot_tags import bot_tag_store
from .service_auth import NO_STORE, read_body, require_service_token

router = APIRouter(tags=["System"])


class GrantWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    granted: bool
    reason: str = Field(min_length=1, max_length=grants.REASON_MAX)
    #: The signed-in person; service-bearer callers only.
    actor: str | None = Field(default=None, max_length=grants.ACTOR_MAX)


class GrantResponse(BaseModel):
    tag: str
    bot: str
    granted: bool
    granted_by: str | None
    granted_at: datetime | None
    changed: bool = False


class GrantListResponse(BaseModel):
    data: list[GrantResponse]


def _response(record: grants.GrantRecord) -> GrantResponse:
    return GrantResponse(tag=record.tag, bot=record.bot, granted=record.granted,
                         granted_by=record.assigned_by, granted_at=record.assigned_at,
                         changed=record.changed)


def _manager_capability() -> str | None:
    store = get_runtime_settings_store(get_service().config)
    return grants.manager_capability(store.get_scope_settings("global", "*"))


async def _actor(request: Request, authorization: str | None) -> tuple[grants.GrantActor, GrantWrite]:
    capability = request.headers.get(TASK_TURN_CONTEXT_HEADER)
    if capability:
        try:
            caller = open_task_turn_context(capability)
        except TaskTurnContextError as exc:
            raise HTTPException(403, str(exc), headers=NO_STORE) from exc
        body = await read_body(request, GrantWrite)
        if body.actor is not None:
            raise HTTPException(422, "a bot's actor comes from its signed turn, not the body",
                                headers=NO_STORE)
        return grants.GrantActor.for_bot(caller.bot_id), body
    require_service_token(authorization)
    body = await read_body(request, GrantWrite)
    return grants.GrantActor.for_person(body.actor), body


@router.get("/v1/bot-grants", response_model=GrantListResponse)
async def list_bot_grants():
    """Every current grant tag assignment with its granter and time."""
    engine = bot_tag_store().engine

    def read():
        with engine.connect() as conn:
            return grants.list_grants(conn)

    return GrantListResponse(data=[_response(r) for r in await asyncio.to_thread(read)])


@router.put("/v1/bot-grants/{tag}/bots/{slug}", response_model=GrantResponse)
async def put_bot_grant(tag: str, slug: str, request: Request,
                        authorization: str | None = Header(default=None)):
    """Grant or revoke ``tag`` on bot ``slug``. 403 says why a bot was refused."""
    try:
        actor, body = await _actor(request, authorization)
    except BotTagError as exc:
        raise HTTPException(exc.status_code, str(exc), headers=NO_STORE) from exc
    engine = bot_tag_store().engine

    def write():
        capability = _manager_capability() if actor.bot is not None else None
        with engine.begin() as conn:
            return grants.write_grant(conn, tag, slug, granted=body.granted,
                                      reason=body.reason, actor=actor, capability=capability)

    try:
        record = await asyncio.to_thread(write)
    except GrantRefused as refused:
        raise HTTPException(refused.status_code, {"code": refused.code, "message": str(refused)},
                            headers=NO_STORE) from refused
    except BotTagError as exc:
        raise HTTPException(exc.status_code, str(exc), headers=NO_STORE) from exc
    if record.changed:
        # Cached Bot objects carry tag slugs; refresh them and tell the bot's UI.
        await asyncio.to_thread(profile_routes._reload_bot_registry)
        await profile_routes._publish_bot_profile_changed(get_service(), record.bot)
    return _response(record)
