"""Skill catalog and assignment routes (TASK-1309).

Skill content is read-only here (it is authored in git); these routes read it
and edit which bots get which skills, and how. Like the rest of ``/v1/*`` this
surface is LAN-trusted. Every rule lives in ``llm_bawt.skills``.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ...skills.service import SkillService, SkillServiceError
from ..dependencies import get_bot_profile_store, get_service

router = APIRouter()


class AssignmentRequest(BaseModel):
    scope_type: str
    scope_id: str
    skill_ref: str
    mode: str
    position: int = Field(default=0)
    updated_by: str | None = None


class AdoptRequest(BaseModel):
    adopted_by: str | None = None


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
    """Per-bot summary: managed, locked, harness, injected total against the caps."""
    return await _run(skill_service().bots_summary)


@router.get("/v1/skills/bots/{slug}", tags=["Skills"])
async def bot_skills(slug: str):
    """One bot's effective skills, with why any assigned skill does not apply."""
    return await _run(skill_service().effective, slug)


@router.post("/v1/skills/bots/{slug}/adopt", tags=["Skills"])
async def adopt_bot(slug: str, request: AdoptRequest | None = None):
    """Start managing a bot's skills; every current skill is seeded as available."""
    service = skill_service()
    return await _run(service.adopt, slug, request.adopted_by if request else None)


@router.delete("/v1/skills/bots/{slug}/managed", tags=["Skills"])
async def release_bot(slug: str):
    """Stop managing a bot's skills and drop its rows; it returns to the ambient set."""
    await _run(skill_service().release, slug)
    return {"released": slug}
