"""``self_house``: a bot lists houses, moves itself into a free one, or out (TASK-2118).

A thin forwarder of the signed turn capability to ``/v1/houses/self``
(``service/routes/houses.py``), which decides everything. There is no bot
argument: the caller can only move itself, and an occupied house is refused.
"""
from __future__ import annotations

from typing import Literal

from .bot_grant_tools import _error, _forward
from .registry import mcp


@mcp.tool(name="self_house")
async def self_house(action: Literal["list", "move_in", "move_out"] = "list",
                     house: str | None = None) -> str:
    """List houses, move yourself into a free one, or move out of your own.

    move_in takes ``house`` or, omitted, any free one; an occupied house is
    refused. Applies from your next turn. Never pins.

    Args:
        action: list | move_in | move_out.
        house: move_in only: the house id, or omit for any free house.
    """
    if action == "list":
        return await _forward("GET", "/v1/houses/self", None,
                              unknown="house state is unavailable; try again")
    if action == "move_in":
        return await _forward("POST", "/v1/houses/self/move-in",
                              {"house": house.strip()} if house and house.strip() else {},
                              unknown="the move's outcome is unknown; check action=list")
    if action == "move_out":
        if house:
            return _error("invalid_request", "move_out takes no house; it leaves your own")
        return await _forward("POST", "/v1/houses/self/move-out", None,
                              unknown="the move's outcome is unknown; check action=list")
    return _error("invalid_request", "action must be list, move_in or move_out")
