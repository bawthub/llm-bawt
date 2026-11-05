"""Read-only purge inventory; its digest is what the guarded execution confirms."""
from fastapi import APIRouter, HTTPException, Query, Response
from sqlalchemy.exc import SQLAlchemyError

from ...bot_purge_preview import PreviewUnavailable, normalize_slug, preview_bot_purge
from ...utils.db import get_shared_engine
from ..dependencies import get_service

router = APIRouter()
_NO_STORE = {"Cache-Control": "no-store"}


@router.get("/v1/bots/{slug}/purge-preview", tags=["System"])
def bot_purge_preview(slug: str, response: Response,
                      delete_profile: bool = Query(False, description="Preview deleting the profile and tags too")):
    """Describe what a confirmed purge would delete and retain; reads only."""
    response.headers.update(_NO_STORE)
    try:
        slug = normalize_slug(slug)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc), headers=_NO_STORE) from exc
    try:
        return preview_bot_purge(get_shared_engine(get_service().config), slug, delete_profile=delete_profile)
    except (PreviewUnavailable, SQLAlchemyError) as exc:
        raise HTTPException(status_code=503, detail="Purge preview unavailable; no writes were attempted.",
                            headers=_NO_STORE) from exc
