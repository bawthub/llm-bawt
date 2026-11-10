"""A bot's own house tenancy (TASK-2118): ``/v1/houses/self``.

* ``GET  /v1/houses/self``          houses, their live harnesses and tenants
* ``POST /v1/houses/self/move-in``  ``{"house": "h1"}`` or ``{}`` for any free one
* ``POST /v1/houses/self/move-out`` vacate the caller's own house

The caller is the bot in the signed turn capability
(``X-LLM-Bawt-Task-Turn-Context``, forwarded by the ``self_house`` MCP tool);
nothing in the request names a bot. Rules live in :mod:`llm_bawt.house_tenancy`.
A refusal is ``{"detail": {"code", "message", ...}}``.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from ... import house_tenancy
from ...task_turn_context import TASK_TURN_CONTEXT_HEADER, TaskTurnContextError, open_task_turn_context
from ..dependencies import get_bot_profile_store, get_service
from .service_auth import NO_STORE, read_body

logger = logging.getLogger(__name__)
router = APIRouter(tags=["System"])


class MoveIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    house: str | None = Field(default=None, max_length=64)


def _caller(request: Request) -> str:
    try:
        return open_task_turn_context(request.headers.get(TASK_TURN_CONTEXT_HEADER)).bot_id.strip().lower()
    except TaskTurnContextError as exc:
        raise HTTPException(403, {"code": "unauthorized", "message": str(exc)},
                            headers=NO_STORE) from exc


def _state() -> house_tenancy.HouseState:
    """Read live routing state; an unreadable Redis refuses (503), never guesses."""
    import redis

    client = redis.Redis.from_url(get_service().config.REDIS_URL, decode_responses=True,
                                  socket_timeout=2.0, socket_connect_timeout=2.0)
    from ...house_pool_config import configured_pool

    try:
        return house_tenancy.read_house_state(client, configured_pool())
    finally:
        client.close()


async def _run(fn, *args) -> Any:
    try:
        return await asyncio.to_thread(fn, *args)
    except house_tenancy.HouseMoveRefused as refused:
        raise HTTPException(refused.status_code, refused.detail(), headers=NO_STORE) from refused
    except HTTPException:
        raise
    except Exception as exc:
        from ...bot_profile_changes import ProfileWriteConflict

        if isinstance(exc, ProfileWriteConflict):
            raise HTTPException(409, {"code": "profile_changed",
                                      "message": "your profile changed meanwhile; list and retry"},
                                headers=NO_STORE) from exc
        logger.exception("house self-move failed")
        raise HTTPException(503, {"code": "unavailable",
                                  "message": "house state is unavailable; nothing was changed"},
                            headers=NO_STORE) from exc


def _store():
    return get_bot_profile_store(get_service().config)


def _after_move(slug: str, result: house_tenancy.MoveResult) -> None:
    """Make the move authoritative for the next turn; recycle an emptied house."""
    from ... import house_lease

    service = get_service()
    house_lease._refresh_bot(service, None, slug)
    if result.vacated and result.vacated_empty:
        house_lease.request_recycle(service.config, result.vacated)


def _previews(houses: list[str]) -> dict[str, dict[str, Any]]:
    """Each house's preview URL and what it serves now (TASK-2175).

    ``preview_host`` comes from the pool file (house ids and preview hosts need
    not share a number); ``preview`` is the published value (branch, commit,
    up), null when none runs, ``"unknown"`` when Redis can't be read.
    """
    import redis

    from ...house_pool_config import configured_pool
    from ...house_watcher import parse_preview

    hosts = {h["id"]: h["preview_host"] for h in configured_pool() or []}
    out = {house: {"preview_host": hosts.get(house), "preview": "unknown"} for house in houses}
    try:
        client = redis.Redis.from_url(get_service().config.REDIS_URL, decode_responses=True,
                                      socket_timeout=2.0, socket_connect_timeout=2.0)
        try:
            for house in houses:
                out[house]["preview"] = parse_preview(client.get(f"agent:house:{house}:preview"))
        finally:
            client.close()
    except Exception:
        logger.warning("house preview read failed", exc_info=True)
    return out


async def _with_previews(houses: list[str]) -> dict[str, dict[str, Any]]:
    try:
        return await asyncio.to_thread(_previews, houses)
    except Exception:
        logger.warning("house preview hosts unavailable", exc_info=True)
        return {}


@router.get("/v1/houses/self")
async def list_houses(request: Request):
    slug = _caller(request)
    listing = await _run(house_tenancy.list_houses, _store(), slug, _state)
    previews = await _with_previews([row["house"] for row in listing["houses"]])
    for row in listing["houses"]:
        row.update(previews.get(row["house"], {}))
    return listing


@router.post("/v1/houses/self/move-in")
async def move_in(request: Request):
    slug = _caller(request)
    body = await read_body(request, MoveIn)
    result = await _run(house_tenancy.move_in, _store(), slug, body.house, _state)
    if result.changed:
        await asyncio.to_thread(_after_move, slug, result)
    out = result.as_dict()
    if result.house:
        out.update((await _with_previews([result.house])).get(result.house, {}))
    return out


@router.post("/v1/houses/self/move-out")
async def move_out(request: Request):
    slug = _caller(request)
    result = await _run(house_tenancy.move_out, _store(), slug)
    if result.changed:
        await asyncio.to_thread(_after_move, slug, result)
    return result.as_dict()
