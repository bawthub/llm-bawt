"""Service-token-gated named secrets for BawtHub auth (TASK-1467 §10, TASK-1472).

``GET/PUT /v1/internal/secrets/bawthub-auth/{name}`` where ``name`` is
``session-secret`` or ``oidc/<providerId>``. The bearer is checked before
anything else, so an unauthenticated caller learns nothing about which names
exist. There is no list route and a PUT never echoes the value.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from llm_bawt.bawthub_auth import TokenCheck, service_token_source
from llm_bawt.bawthub_auth.secret_store import (
    MAX_VALUE_CHARS,
    BawtHubAuthSecretStore,
    SecretUnreadable,
    is_create_only,
    valid_secret_name,
)
from llm_bawt.service.dependencies import get_service
from llm_bawt.utils.db import get_shared_engine

logger = logging.getLogger(__name__)
PREFIX = "/v1/internal/secrets/bawthub-auth"
NO_STORE = {"Cache-Control": "no-store"}


def require_service_token(authorization: str | None = Header(default=None)) -> None:
    result = service_token_source().check(authorization)
    if result is TokenCheck.OK:
        return
    if result is TokenCheck.UNCONFIGURED:
        raise HTTPException(503, "service_token_unconfigured", headers=NO_STORE)
    raise HTTPException(401, "unauthorized", headers={**NO_STORE, "WWW-Authenticate": "Bearer"})


def get_secret_store() -> BawtHubAuthSecretStore:
    engine = get_shared_engine(get_service().config)
    if engine is None:
        raise HTTPException(503, "secret_store_unavailable", headers=NO_STORE)
    return BawtHubAuthSecretStore(engine)


router = APIRouter(prefix=PREFIX, dependencies=[Depends(require_service_token)],
                   tags=["Internal"])


class SecretWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str = Field(min_length=1, max_length=MAX_VALUE_CHARS)


def _known(name: str) -> str:
    if not valid_secret_name(name):
        raise HTTPException(404, "not_found", headers=NO_STORE)
    return name


@router.get("/{name:path}")
async def get_secret(name: str, store: BawtHubAuthSecretStore = Depends(get_secret_store)):
    name = _known(name)
    try:
        value = await run_in_threadpool(store.get, name)
    except SecretUnreadable:
        logger.error("bawthub-auth secret %s is stored but can't be decrypted", name)
        raise HTTPException(503, "secret_unreadable", headers=NO_STORE) from None
    if value is None:
        raise HTTPException(404, "not_found", headers=NO_STORE)
    return JSONResponse({"name": name, "value": value}, headers=NO_STORE)


@router.put("/{name:path}")
async def put_secret(body: SecretWrite, name: str,
                     store: BawtHubAuthSecretStore = Depends(get_secret_store)):
    name = _known(name)
    if is_create_only(name):
        if not await run_in_threadpool(store.create_if_absent, name, body.value):
            raise HTTPException(409, "exists", headers=NO_STORE)
        logger.info("bawthub-auth secret %s created", name)
        return JSONResponse({"name": name, "created": True}, status_code=201, headers=NO_STORE)
    created = await run_in_threadpool(store.put, name, body.value)
    logger.info("bawthub-auth secret %s %s", name, "created" if created else "replaced")
    return JSONResponse({"name": name, "created": created}, headers=NO_STORE)
