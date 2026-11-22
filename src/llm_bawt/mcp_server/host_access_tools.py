"""Bot-facing inventory for the server's installation, not an SSH executor."""
from __future__ import annotations

import asyncio
from typing import Literal

from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from ..bot_access import BotAccessError
from ..ops.host_access import INVENTORY_LIMIT, HostInventoryEntry, read_host_inventory
from ..utils.config import Config
from ..utils.db import get_shared_engine
from .bot_access_identity import trusted_caller
from .registry import mcp


class HostAccessInventory(BaseModel):
    status: Literal["ok", "unavailable", "unauthorized"]
    hosts: list[HostInventoryEntry] = Field(default_factory=list, max_length=INVENTORY_LIMIT)
    truncated: bool = False


def _read_inventory() -> HostAccessInventory:
    try:
        # Use the same installation engine as OpsStore, without creating an
        # OpsStore (its constructor may run bootstrap DDL).
        engine = get_shared_engine(Config())
        if engine is not None:
            hosts, truncated = read_host_inventory(engine)
            return HostAccessInventory(status="ok", hosts=hosts, truncated=truncated)
    except (SQLAlchemyError, ValueError):
        # Connection errors and corrupt records can contain operator data.
        pass
    return HostAccessInventory(status="unavailable")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
async def host_access_list() -> HostAccessInventory:
    """List registered SSH destinations for this installation and their recorded login outcomes."""
    # A turn token is signed by this installation. Neither OAuth/service
    # identity nor caller-provided arguments can select a scope or substitute
    # for a trusted turn. No database access before verifying the caller.
    try:
        caller = trusted_caller("host_access_list", {})
    except BotAccessError:
        caller = None
    if caller is None or caller.kind != "turn":
        return HostAccessInventory(status="unauthorized")
    return await asyncio.to_thread(_read_inventory)
