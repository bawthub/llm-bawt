"""``usage_utility_rollup``: utility-model token use per day and caller (TASK-1538).

A thin reader of ``GET /v1/usage/utility``. Aggregates only: no bot, prompt or
content appears in the result.
"""
from __future__ import annotations

from .inter_bot_tools import _get_json
from .registry import mcp


@mcp.tool(name="usage_utility_rollup")
async def usage_utility_rollup(days: int = 1) -> dict:
    """Utility-model calls (recall, titles, summaries, ...) per UTC day and caller.

    Args:
        days: UTC days back, today included (1-90).
    """
    days = min(max(int(days), 1), 90)
    return await _get_json(f"/v1/usage/utility?days={days}") or {"days": days, "rows": []}
