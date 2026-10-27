"""Gated reads of stored connection secrets for trusted consumers (TASK-1476).

Two routes, one contract, each with its own gate and allowlist:

* ``GET /v1/internal/connections/{kind}/{id}``: BawtHub's server, gated by
  the BawtHub service token, allowlist :data:`BAWTHUB_READABLE`.
* ``GET /v1/internal/bridge/connections/{kind}/{id}``: agent bridges, gated
  by the ``X-Bridge-Token`` broker gate, allowlist :data:`BRIDGE_READABLE`.

Both return ``{kind, id, status, meta, secret}`` limited to the fields the
spec names. The gate is checked before anything else.

* 404 ``not_configured``: no record, or not on that caller's allowlist. The
  only status on which a consumer may use its legacy env fallback.
* 409 ``not_connected`` / 503 ``unreadable`` / 503 ``store_unavailable``: a
  record exists but can't be used; the consumer must not fall back to env.

There is no list or write route, and nothing here logs a value.
"""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from llm_bawt.service import bridge_gate
from llm_bawt.service.connection_secrets import (
    BAWTHUB_READABLE,
    BRIDGE_READABLE,
    ConnectionUnavailable,
    FieldSpec,
    read_connection,
)
from llm_bawt.service.dependencies import get_service

from .service_auth import NO_STORE, require_service_token

PREFIX = "/v1/internal/connections"
BRIDGE_PREFIX = "/v1/internal/bridge/connections"
_STATUS = {"not_connected": 409, "unreadable": 503, "store_unavailable": 503}

router = APIRouter(prefix=PREFIX, dependencies=[Depends(require_service_token)],
                   tags=["Internal"])
bridge_router = APIRouter(prefix=BRIDGE_PREFIX, tags=["Internal"])


def get_config():
    return get_service().config


async def _serve(readable: Mapping[tuple[str, str], FieldSpec], kind: str, connection_id: str,
                 config) -> JSONResponse:
    spec = readable.get((kind, connection_id))
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


@router.get("/{kind}/{connection_id}")
async def get_connection(kind: str, connection_id: str, config=Depends(get_config)):
    return await _serve(BAWTHUB_READABLE, kind, connection_id, config)


@bridge_router.get("/{kind}/{connection_id}")
async def get_bridge_connection(request: Request, kind: str, connection_id: str):
    # The gate runs before the config dependency or any store access.
    bridge_gate.check(request, bridge_gate.BROKER)
    return await _serve(BRIDGE_READABLE, kind, connection_id, get_config())
