"""Advisory read-only purge inventory. No destructive/confirmation API here."""
from fastapi import APIRouter, HTTPException, Response
from sqlalchemy.exc import SQLAlchemyError

from ...bot_purge_preview import PreviewUnavailable, normalize_slug, preview_bot_purge
from ...utils.db import get_shared_engine
from ..dependencies import get_service

router = APIRouter()
_NO_STORE = {"Cache-Control": "no-store"}


@router.get("/v1/bots/{slug}/purge-preview", tags=["System"])
def bot_purge_preview(slug: str, response: Response):
    """Describe current standalone purge selection/retention, never authorize it."""
    response.headers.update(_NO_STORE)
    try:
        slug = normalize_slug(slug)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc), headers=_NO_STORE) from exc
    try:
        return preview_bot_purge(get_shared_engine(get_service().config), slug)
    except (PreviewUnavailable, SQLAlchemyError) as exc:
        raise HTTPException(status_code=503, detail="Purge preview unavailable; no writes were attempted.",
                            headers=_NO_STORE) from exc
