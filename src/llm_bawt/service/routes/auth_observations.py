"""GET /v1/admin/auth-observations: the API-auth log phase's caller counts (TASK-2380).

Rows are ``{"key": [kind, who..., method, route], "count", "first", "last"}``,
bounded and aged by the middleware's ``Observations``. Service bearer only;
nothing here holds a key.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from starlette.responses import JSONResponse

from llm_bawt.api_auth import mcp, wiring

from .service_auth import NO_STORE, require_service_token

router = APIRouter(prefix="/v1/admin", dependencies=[Depends(require_service_token)],
                   tags=["admin"])


def _mcp_server_url() -> str | None:
    try:
        from ..dependencies import get_service

        return get_service().config.MCP_SERVER_URL or None
    except Exception:  # noqa: BLE001 - no service: report this process's view
        return None


@router.get("/auth-observations")
async def get_auth_observations() -> JSONResponse:
    # TASK-2381: the MCP server (:8001) has its own mode and counts, usually in
    # its own process, so they come from that server.
    mcp_view = await mcp.fetch_observations(_mcp_server_url())
    return JSONResponse({"mode": wiring.RESOLVER.mode(), "rows": wiring.OBSERVATIONS.snapshot(),
                         "mcp": mcp_view},
                        headers=NO_STORE)
