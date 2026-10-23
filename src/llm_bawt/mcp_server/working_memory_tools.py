"""Trusted-capability MCP tools for working memory (docs/WORKING_MEMORY.md 6.1).

No tool takes a bot or user: identity comes only from the signed turn
capability, and a missing, expired or malformed one is refused, reads included.
Errors come back as ``{"error": code, ...}`` JSON, never as an empty result.
"""
from __future__ import annotations

import asyncio
import json
import logging
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from llm_bawt.task_turn_context import TaskTurnContextError
from llm_bawt.working_memory.contracts import WorkingMemoryError
from llm_bawt.working_memory.service import WorkingMemoryService

from .registry import mcp
from .task_association import current_task_turn_context

OpenStatus = Literal["planned", "active", "blocked"]
Status = Literal["planned", "active", "blocked", "done", "cancelled"]
ClearableField = Literal["next_action", "checkpoint", "priority"]

logger = logging.getLogger(__name__)


class Ref(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str
    id: str
    label: str | None = None
    excerpt: str | None = None


@lru_cache(maxsize=1)
def _service() -> WorkingMemoryService:
    from llm_bawt.utils.config import Config
    from llm_bawt.utils.db import get_shared_engine
    return WorkingMemoryService(get_shared_engine(Config()))


def _dump(value: Any) -> str:
    # A string avoids FastMCP's second structured-output copy.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _refs(refs: list[Ref] | None) -> list[dict[str, Any]] | None:
    return None if refs is None else [ref.model_dump(exclude_none=True) for ref in refs]


async def _call(operation) -> str:
    try:
        context = current_task_turn_context()
    except TaskTurnContextError:
        return _dump({"error": "unauthorized", "message": "a trusted turn capability is required"})
    try:
        return _dump(await asyncio.to_thread(operation, _service(), context))
    except WorkingMemoryError as error:
        return _dump(error.as_dict())
    except Exception:
        logger.exception("working memory tool failed")
        return _dump({"error": "unavailable", "message": "working memory unavailable"})


@mcp.tool(name="working_memory_list")
async def working_memory_list(include_archived: bool = False) -> str:
    """List your working-memory items for this user; open items unless include_archived."""
    return await _call(lambda service, context: service.bot_list(context, include_archived=include_archived))


@mcp.tool(name="working_memory_create")
async def working_memory_create(
    objective: str,
    idempotency_key: str,
    next_action: str | None = None,
    status: OpenStatus = "planned",
    priority: int | None = None,
    refs: list[Ref] | None = None,
    kind: Literal["personal", "initiative_link"] = "personal",
    initiative_task_id: str | None = None,
) -> str:
    """Create one working-memory item; idempotency_key is a fresh UUID per create."""
    fields = {
        "objective": objective, "next_action": next_action, "status": status, "priority": priority,
        "refs": _refs(refs), "kind": kind, "initiative_task_id": initiative_task_id,
    }
    return await _call(lambda service, context: service.bot_create(context, fields, idempotency_key))


@mcp.tool(name="working_memory_update")
async def working_memory_update(
    item_id: str,
    expected_revision: int,
    objective: str | None = None,
    status: Status | None = None,
    next_action: str | None = None,
    blockers: list[str] | None = None,
    checkpoint: str | None = None,
    priority: int | None = None,
    refs: list[Ref] | None = None,
    clear: list[ClearableField] | None = None,
) -> str:
    """CAS-update one item; omitted fields stay, clear nulls the named fields."""
    changes: dict[str, Any] = {
        name: value for name, value in (
            ("objective", objective), ("status", status), ("next_action", next_action),
            ("blockers", blockers), ("checkpoint", checkpoint), ("priority", priority), ("refs", _refs(refs)),
        ) if value is not None
    }
    clear = clear or []
    both = sorted(set(clear) & set(changes))
    if both:
        return _dump({"error": "invalid_request", "message": f"cannot both set and clear {', '.join(both)}"})
    changes.update({name: None for name in clear})
    return await _call(
        lambda service, context: service.bot_update(context, item_id, expected_revision, changes)
    )
