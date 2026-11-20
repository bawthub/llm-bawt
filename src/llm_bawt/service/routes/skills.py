"""Skill catalog and exception routes (TASK-1309).

Skill content is read-only here (it is authored in git). These routes read it
and edit the per-bot and per-tag exceptions to "every skill enabled". Like
the rest of ``/v1/*`` this surface is LAN-trusted. Every rule lives in
``llm_bawt.skills``.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...skills.service import SkillService, SkillServiceError
from ..dependencies import get_bot_profile_store, get_service
from .service_auth import require_service_token

router = APIRouter()


class AssignmentRequest(BaseModel):
    scope_type: str
    scope_id: str
    skill_ref: str
    mode: str
    position: int = Field(default=0)
    updated_by: str | None = None


def skill_service() -> SkillService:
    config = get_service().config
    engine = get_bot_profile_store(config).engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Bot profiles DB unavailable")
    return SkillService.from_config(engine, config)


async def _run(fn, *args):
    """A service call off the event loop, with its typed failure as the HTTP status."""
    try:
        return await asyncio.to_thread(fn, *args)
    except SkillServiceError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc


@router.get("/v1/skills", tags=["Skills"])
async def list_skills():
    """Every skill the source offers, with size, file count and assignment counts."""
    return {"data": await _run(skill_service().catalog)}


@router.get("/v1/skills/harnesses", tags=["Skills"])
async def list_harnesses():
    """What each agent harness can do with assigned skills."""
    return {"data": skill_service().harnesses()}


@router.get("/v1/skills/catalog/{ref}/tree", tags=["Skills"])
async def skill_tree(ref: str):
    """A skill's files and where its bytes come from."""
    return await _run(skill_service().tree, ref)


@router.get("/v1/skills/catalog/{ref}/file", tags=["Skills"])
async def skill_file(ref: str, path: str = Query(...)):
    """One text file inside a skill, confined to its directory."""
    return await _run(skill_service().file, ref, path)


@router.get("/v1/skills/assignments", tags=["Skills"])
async def list_assignments(scope_type: str | None = None, scope_id: str | None = None):
    return {"data": await _run(skill_service().assignments, scope_type, scope_id)}


@router.put("/v1/skills/assignments", tags=["Skills"])
async def put_assignment(request: AssignmentRequest):
    """Create or change one row. 409 with ``over_cap`` details when it would exceed a cap."""
    service = skill_service()
    return await _run(service.put, request.scope_type, request.scope_id, request.skill_ref,
                      request.mode, request.position, request.updated_by)


@router.delete("/v1/skills/assignments", tags=["Skills"])
async def delete_assignment(scope_type: str = Query(...), scope_id: str = Query(...),
                            skill_ref: str = Query(...)):
    await _run(skill_service().remove, scope_type, scope_id, skill_ref)
    return {"deleted": {"scope_type": scope_type, "scope_id": scope_id, "skill_ref": skill_ref}}


@router.get("/v1/skills/bots", tags=["Skills"])
async def list_bot_skills():
    """Per-bot summary: locked, harness, row and disable counts, injected total against the caps."""
    return await _run(skill_service().bots_summary)


@router.get("/v1/skills/bots/{slug}", tags=["Skills"])
async def bot_skills(slug: str):
    """One bot's effective skills, with why any assigned skill does not apply."""
    return await _run(skill_service().effective, slug)


# -- Skill sources (TASK-2534) ------------------------------------------------


class SourceRequest(BaseModel):
    label: str
    url: str
    ref: str | None = None
    subdir: str | None = None
    credential: str = "none"
    priority: int = Field(default=100, ge=0, le=10_000)


class SourceActionRequest(BaseModel):
    revision: str | None = None


# Source writes change the skills every bot gets, so they need BawtHub's
# service bearer (added by its proxy for admins only, TASK-2370). ``user_id``
# is bound to the signed-in person by that proxy and recorded as the actor.
_SERVICE = [Depends(require_service_token)]


def source_service():
    from ...skills.source_config import source_service_for

    config = get_service().config
    engine = get_bot_profile_store(config).engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Bot profiles DB unavailable")
    return source_service_for(engine, config)


async def _run_source(fn, *args, **kwargs):
    from ...skills.source_store import SourceError

    try:
        return await asyncio.to_thread(lambda: fn(*args, **kwargs))
    except SourceError as exc:
        raise HTTPException(status_code=exc.status, detail={"code": exc.code, "message": exc.detail}) from exc


@router.get("/v1/skills/sources", tags=["Skills"])
async def list_sources():
    """Configured skill sources, their pinned revisions and any pending update."""
    return await _run_source(source_service().list)


@router.post("/v1/skills/sources", tags=["Skills"], dependencies=_SERVICE)
async def add_source(request: SourceRequest, user_id: str | None = None):
    """Fetch, validate and activate a new Git source. Nothing is saved if it can't be read."""
    service = source_service()
    return await _run_source(service.add, label=request.label, url=request.url, ref=request.ref,
                             subdir=request.subdir, credential=request.credential,
                             priority=request.priority, updated_by=user_id)


@router.post("/v1/skills/sources/{source_id}/check", tags=["Skills"], dependencies=_SERVICE)
async def check_source(source_id: str):
    """Stage the remote revision and report which skills it adds, changes or removes."""
    return await _run_source(source_service().check, source_id)


@router.post("/v1/skills/sources/{source_id}/update", tags=["Skills"], dependencies=_SERVICE)
async def update_source(source_id: str, request: SourceActionRequest, user_id: str | None = None):
    """Activate exactly the checked revision the owner reviewed."""
    if not request.revision:
        raise HTTPException(status_code=400, detail={"code": "revision_required",
                                                     "message": "Check for updates and confirm a revision."})
    return await _run_source(source_service().update, source_id, revision=request.revision,
                             updated_by=user_id)


@router.post("/v1/skills/sources/{source_id}/rollback", tags=["Skills"], dependencies=_SERVICE)
async def rollback_source(source_id: str, user_id: str | None = None):
    """Re-activate the previous pinned revision."""
    return await _run_source(source_service().rollback, source_id, updated_by=user_id)


@router.delete("/v1/skills/sources/{source_id}", tags=["Skills"], dependencies=_SERVICE)
async def remove_source(source_id: str):
    """Remove a source and its files. Other sources, the bundle and assignments are untouched."""
    return await _run_source(source_service().remove, source_id)
