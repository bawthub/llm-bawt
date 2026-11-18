"""Capability-holder discovery over HTTP (TASK-1712).

The read behind the MCP ``bots_list_available`` tool, for a service such as
BawtHub's monitors that must pick a bot by capability without reimplementing
the tag mapping. Anonymous: no caller is excluded and provider comparison is
null. Same filters, policy and failure behavior as the tool.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


@router.get("/v1/bots/available", tags=["Bots"])
async def bots_available(
    capability: str | None = Query(default=None),
    tag: str | None = Query(default=None),
):
    """Listable bots with their tags, capabilities and live turn state."""
    from ...mcp_server.inter_bot_tools import list_available_bots

    try:
        bots = await list_available_bots(tag=tag, capability=capability, include_state=True, detail="full")
    except ValueError as error:  # unknown tag or capability
        raise HTTPException(status_code=400, detail=str(error)) from None
    except Exception:  # policy/profile unreadable: never a partial list
        raise HTTPException(status_code=503, detail="Bot availability unavailable") from None
    return {"bots": bots}
