"""Operator freeze API (TASK-1233): ``/v1/admin/gates``.

* ``GET /v1/admin/gates``: every live freeze (``frozen`` is true when any is).
* ``POST /v1/admin/gates``: freeze. ``actor``, ``reason`` and ``ttl_seconds``
  are required; ``scope`` defaults to every agent bot.
* ``POST /v1/admin/gates/{gate_id}/thaw``: thaw. ``override`` thaws another
  actor's freeze.

Every route takes the TASK-1472 service bearer, checked before the body is
read: BawtHub's server is the only caller, and bots don't hold the token. The
person is the ``actor`` BawtHub signed in; BawtHub decides who may override.
None of these is an admission edge, so a freeze never blocks its own thaw.
Semantics live in ``operator_freeze``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from llm_bawt.service.dependencies import get_service
from llm_bawt.utils.db import get_shared_engine

from ..operator_freeze import (
    MAX_FREEZE_TTL_SECONDS,
    MAX_REASON_CHARS,
    MIN_FREEZE_TTL_SECONDS,
    FreezeConflict,
    FreezeUnavailable,
    NotFreezeOwner,
    freeze,
    live_freezes,
    thaw,
    valid_actor,
)
from ..turn_gates import GateScope, GateScopeError
from .service_auth import NO_STORE, require_service_token
from .service_auth import read_body as _read

PREFIX = "/v1/admin/gates"

router = APIRouter(prefix=PREFIX, dependencies=[Depends(require_service_token)],
                   tags=["Admin"])


def get_gate_engine():
    return get_shared_engine(get_service().config)


def _check_actor(value: str) -> str:
    value = value.strip()
    if not valid_actor(value):
        raise ValueError("actor must be 1-96 characters: letters, digits and . _ @ + : -")
    return value


Actor = Annotated[str, AfterValidator(_check_actor)]


class FreezeScope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    all_agents: bool = False
    bots: list[str] = Field(default_factory=list, max_length=200)
    backends: list[str] = Field(default_factory=list, max_length=50)

    def gate_scope(self) -> GateScope:
        # Normalized as GateScope.for_bots/on_backends do; GateScope itself
        # refuses a scope naming none or more than one of the three.
        def names(values: list[str]) -> frozenset[str]:
            return frozenset(v.strip().lower() for v in values if v and v.strip())

        return GateScope(all_agents=self.all_agents, bots=names(self.bots),
                         backends=names(self.backends))


class FreezeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actor: Actor
    reason: str = Field(max_length=MAX_REASON_CHARS)
    ttl_seconds: int = Field(ge=MIN_FREEZE_TTL_SECONDS, le=MAX_FREEZE_TTL_SECONDS)
    scope: FreezeScope = Field(default_factory=lambda: FreezeScope(all_agents=True))

    @field_validator("reason")
    @classmethod
    def check_reason(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("a freeze needs a reason")
        return value

    @model_validator(mode="after")
    def check_scope(self) -> "FreezeRequest":
        try:
            self.scope.gate_scope()
        except GateScopeError as error:
            raise ValueError(str(error)) from None
        return self


class ThawRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actor: Actor
    override: bool = False


def _unavailable() -> HTTPException:
    return HTTPException(503, "gates_unavailable", headers=NO_STORE)


def get_dispatch_wake():
    """Wakes the delivery dispatcher, so held deliveries go out as soon as a freeze lifts."""
    dispatcher = getattr(get_service(), "_inter_bot_dispatcher", None)
    return dispatcher.wake if dispatcher is not None else (lambda: None)


@router.get("")
async def get_gates(engine=Depends(get_gate_engine)):
    freezes = await run_in_threadpool(live_freezes, engine)
    return JSONResponse({
        "frozen": bool(freezes), "freezes": [f.to_api() for f in freezes],
        "max_ttl_seconds": MAX_FREEZE_TTL_SECONDS,
    }, headers=NO_STORE)


@router.post("")
async def post_freeze(request: Request, engine=Depends(get_gate_engine)):
    body = await _read(request, FreezeRequest)
    try:
        frozen = await run_in_threadpool(
            lambda: freeze(engine, actor=body.actor, reason=body.reason,
                           scope=body.scope.gate_scope(), ttl_seconds=body.ttl_seconds))
    except FreezeUnavailable:
        raise _unavailable() from None
    except FreezeConflict as conflict:
        return JSONResponse({"detail": "already_frozen", "freeze": conflict.existing.to_api()},
                            status_code=409, headers=NO_STORE)
    return JSONResponse({"freeze": frozen.to_api()}, status_code=201, headers=NO_STORE)


@router.post("/{gate_id}/thaw")
async def post_thaw(gate_id: int, request: Request, engine=Depends(get_gate_engine),
                    wake=Depends(get_dispatch_wake)):
    body = await _read(request, ThawRequest)
    try:
        result = await run_in_threadpool(
            lambda: thaw(engine, gate_id, actor=body.actor, override=body.override))
    except FreezeUnavailable:
        raise _unavailable() from None
    except NotFreezeOwner:
        raise HTTPException(403, "not_owner", headers=NO_STORE) from None
    if result is None:
        raise HTTPException(404, "not_found", headers=NO_STORE)
    if result.released:
        wake()
    return JSONResponse({
        "freeze": result.freeze.to_api(), "released": result.released, "override": result.override,
    }, headers=NO_STORE)
