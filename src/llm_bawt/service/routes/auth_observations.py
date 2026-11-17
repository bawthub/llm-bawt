"""GET /v1/admin/auth-observations: the API-auth log phase's caller counts (TASK-2380).

Rows are ``{"key": [kind, who..., method, route], "count", "first", "last"}``,
bounded and aged by the middleware's ``Observations``. Service bearer only;
nothing here holds a key.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from starlette.responses import JSONResponse

from llm_bawt.api_auth import wiring

from .service_auth import NO_STORE, require_service_token

router = APIRouter(prefix="/v1/admin", dependencies=[Depends(require_service_token)],
                   tags=["admin"])


@router.get("/auth-observations")
async def get_auth_observations() -> JSONResponse:
    return JSONResponse({"mode": wiring.RESOLVER.mode(), "rows": wiring.OBSERVATIONS.snapshot()},
                        headers=NO_STORE)
