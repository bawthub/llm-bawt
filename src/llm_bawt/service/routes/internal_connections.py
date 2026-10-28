"""Service-token-gated read of stored connection secrets for BawtHub (TASK-1476).

``GET /v1/internal/connections/{kind}/{id}`` returns ``{kind, id, status,
meta, secret}`` for a connection on :data:`BAWTHUB_READABLE`, limited to the
fields its spec names. The bearer is checked before anything else.

* 404 ``not_configured``: no record, or not readable by BawtHub. The only
  status on which BawtHub may use its legacy env fallback.
* 409 ``not_connected`` / 503 ``unreadable`` / 503 ``store_unavailable``: a
  record exists but can't be used; BawtHub must not fall back to env.

There is no list or write route, and nothing here logs a value.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from llm_bawt.service.connection_secrets import (
    BAWTHUB_READABLE,
    ConnectionUnavailable,
    read_connection,
)
from llm_bawt.service.dependencies import get_service

from .service_auth import NO_STORE, require_service_token

PREFIX = "/v1/internal/connections"
_STATUS = {"not_connected": 409, "unreadable": 503, "store_unavailable": 503}

router = APIRouter(prefix=PREFIX, dependencies=[Depends(require_service_token)],
                   tags=["Internal"])


def get_config():
    return get_service().config


@router.get("/{kind}/{connection_id}")
async def get_connection(kind: str, connection_id: str, config=Depends(get_config)):
    spec = BAWTHUB_READABLE.get((kind, connection_id))
    if spec is None:
        raise HTTPException(404, "not_configured", headers=NO_STORE)
    try:
        found = await run_in_threadpool(read_connection, config, kind, connection_id, spec)
    except ConnectionUnavailable as exc:
        raise HTTPException(_STATUS[exc.code], exc.code, headers=NO_STORE) from None
    if found is None:
        raise HTTPException(404, "not_configured", headers=NO_STORE)
    return JSONResponse(
        {"kind": kind, "id": connection_id, "status": "connected",
         "meta": found.meta, "secret": found.secret},
        headers=NO_STORE,
    )
