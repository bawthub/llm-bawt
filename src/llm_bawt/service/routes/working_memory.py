"""Owner REST path for working memory (docs/WORKING_MEMORY.md 6.2; LAN trust boundary).

Protected bots are refused. A bot that isn't opted in stays readable and
editable here, so Nick can clean up after disabling it (7). There is no create.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from ...utils.db import get_shared_engine
from ...working_memory.contracts import WorkingMemoryError
from ...working_memory.service import WorkingMemoryService
from ..dependencies import get_service

router = APIRouter(prefix="/v1/working-memory", tags=["Working memory"])
logger = logging.getLogger(__name__)

_STATUS = {
    "invalid_request": 422,
    "cap_exceeded": 422,
    "protected_bot": 403,
    "not_found": 404,
    "removed": 410,
    "revision_conflict": 409,
    "unavailable": 503,
}


class ItemRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str
    id: str
    label: str | None = None
    excerpt: str | None = None


class ItemPatch(BaseModel):
    """An explicit ``null`` clears a nullable field; an omitted field is unchanged."""

    model_config = ConfigDict(extra="forbid")
    expected_revision: int
    objective: str | None = None
    status: Literal["planned", "active", "blocked", "done", "cancelled"] | None = None
    next_action: str | None = None
    blockers: list[str] | None = None
    checkpoint: str | None = None
    priority: int | None = None
    refs: list[ItemRef] | None = None

    def changes(self) -> dict[str, Any]:
        fields = self.model_fields_set - {"expected_revision"}
        return self.model_dump(include=fields, exclude_none=False)


def _run(operation) -> Any:
    try:
        return operation(WorkingMemoryService(get_shared_engine(get_service().config)))
    except WorkingMemoryError as error:
        return JSONResponse(error.as_dict(), status_code=_STATUS.get(error.code, 400))


@router.get("/health")
def health():
    """Per opted-in bot: counts, versions, counters and behind-pin sessions; no item text (14)."""
    from ...working_memory.health import health_report

    try:
        return health_report(get_shared_engine(get_service().config))
    except WorkingMemoryError as error:
        return JSONResponse(error.as_dict(), status_code=_STATUS.get(error.code, 400))
    except Exception:
        logger.warning("working memory health report failed", exc_info=True)
        return JSONResponse({"status": "unavailable", "bots": []}, status_code=503)


@router.get("/{bot_id}/{user_id}")
def list_items(bot_id: str, user_id: str):
    """Every item, archived and removed included; hard-deleted items appear content-free under ``purged``."""
    return _run(lambda service: service.owner_list(bot_id, user_id))


@router.patch("/{bot_id}/{user_id}/{item_id}")
def update_item(bot_id: str, user_id: str, item_id: str, body: ItemPatch):
    return _run(lambda service: service.owner_update(
        bot_id, user_id, item_id, body.expected_revision, body.changes(),
    ))


@router.delete("/{bot_id}/{user_id}/{item_id}")
def delete_item(bot_id: str, user_id: str, item_id: str, hard: bool = Query(False)):
    """Soft delete is final; ``hard=true`` purges. Repeats are idempotent."""
    return _run(lambda service: service.owner_delete(bot_id, user_id, item_id, hard=hard))
