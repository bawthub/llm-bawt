"""Bot tag catalog and assignment routes (TASK-1329 S5, BOT_TAGS.md §8).

Tags are operator data: there is no MCP write tool, and like the rest of
``/v1/*`` this surface is LAN-trusted. Assignments are set through
``PATCH /v1/bots/{slug}/profile`` (``tags``); this module edits role tags and
removes single assignments, including those of a bot whose profile is gone.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from ...bot_tags.errors import BotTagError
from ...bot_tags.store import BotTagStore, CatalogTag
from ..dependencies import get_bot_profile_store, get_service
from . import settings as profile_routes
from ..schemas import (
    BotTagAssignmentRemovedResponse,
    BotTagCatalogResponse,
    BotTagCreateRequest,
    BotTagInstructionsRequest,
    BotTagPatchRequest,
    BotTagResponse,
    BotTagUse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def bot_tag_store() -> BotTagStore:
    """The tag store on the profile store's engine (its bootstrap made the tables)."""
    engine = get_bot_profile_store(get_service().config).engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Bot profiles DB unavailable")
    return BotTagStore(engine)


def _response(tag: CatalogTag) -> BotTagResponse:
    return BotTagResponse(
        slug=tag.slug, name=tag.name, description=tag.description, kind=tag.kind,
        prompt_key=tag.prompt_key, capabilities=list(tag.capabilities), system=tag.system,
        used_by=[BotTagUse(bot=u.bot, has_profile=u.has_profile) for u in tag.used_by],
        created_at=tag.created_at, updated_at=tag.updated_at)


async def _run(fn, *args):
    """A store call off the event loop, with its typed failure as the HTTP status."""
    try:
        return await asyncio.to_thread(fn, *args)
    except BotTagError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/v1/bot-tags", response_model=BotTagCatalogResponse, tags=["System"])
async def list_bot_tags():
    """Every catalog tag with the bots that carry it (profile-less ones included)."""
    store = bot_tag_store()
    return BotTagCatalogResponse(data=[_response(t) for t in await _run(store.catalog)])


@router.post("/v1/bot-tags", response_model=BotTagResponse, status_code=201, tags=["System"])
async def create_bot_tag(request: BotTagCreateRequest):
    """Create a role tag. 409 when the slug exists (behaviour slugs always do)."""
    store = bot_tag_store()
    tag = await _run(lambda: store.create_role(
        request.slug, request.name, request.description, request.prompt_key,
        request.capabilities))
    return _response(tag)


async def _announce_assignments_changed(tag: CatalogTag) -> None:
    """Cached Bot objects carry tag slugs; refresh them and tell each holder's UI."""
    service = get_service()
    await asyncio.to_thread(profile_routes._reload_bot_registry)
    for use in tag.used_by:
        await profile_routes._publish_bot_profile_changed(service, use.bot)


@router.patch("/v1/bot-tags/{slug}", response_model=BotTagResponse, tags=["System"])
async def patch_bot_tag(slug: str, request: BotTagPatchRequest):
    """Edit a role tag; a new ``slug`` renames it with its assignments. System rows → 409."""
    store = bot_tag_store()
    changes = {name: getattr(request, name) for name in request.model_fields_set}
    tag = await _run(store.update_role, slug, changes)
    if tag.slug != slug:
        await _announce_assignments_changed(tag)
    return _response(tag)


@router.put("/v1/bot-tags/{slug}/instructions", response_model=BotTagResponse, tags=["System"])
async def put_bot_tag_instructions(slug: str, request: BotTagInstructionsRequest):
    """Save a role tag's instructions as a versioned global prompt.

    The first save allocates the tag's ``prompt_key``; later saves add versions.
    """
    store = bot_tag_store()
    return _response(await _run(store.save_instructions, slug, request.body, request.updated_by))


@router.delete("/v1/bot-tags/{slug}/instructions", response_model=BotTagResponse,
               tags=["System"])
async def delete_bot_tag_instructions(slug: str):
    """Detach a role tag's instructions; the registry prompt and its history stay."""
    store = bot_tag_store()
    return _response(await _run(store.clear_instructions, slug))


@router.delete("/v1/bot-tags/{slug}", tags=["System"])
async def delete_bot_tag(slug: str):
    """Delete an unassigned role tag. 409 for a system tag or one still assigned."""
    store = bot_tag_store()
    await _run(store.delete_role, slug)
    return {"deleted": slug}


@router.delete("/v1/bot-tags/{slug}/assignments/{bot}",
               response_model=BotTagAssignmentRemovedResponse, tags=["System"])
async def delete_bot_tag_assignment(slug: str, bot: str):
    """Remove one assignment; works for a bot whose profile row is gone (BOT_TAGS.md §6)."""
    service = get_service()
    store = bot_tag_store()
    removed = await _run(store.unassign, slug, bot)
    # Cached Bot objects carry tags; the resolver reads fresh either way.
    await asyncio.to_thread(profile_routes._reload_bot_registry)
    await profile_routes._publish_bot_profile_changed(service, removed.bot)
    return BotTagAssignmentRemovedResponse(tag=removed.tag, bot=removed.bot,
                                           has_profile=removed.has_profile)
