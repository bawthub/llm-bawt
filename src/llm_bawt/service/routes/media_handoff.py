"""Operator offer/confirm contract for forward GPU handoff (TASK-926).

Network-trusted, like the existing operator ops API. Agent tool callers must
use their existing approval path; this is the future Studio confirmation API.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from ...media.gpu_handoff import GpuHandoff
from ...media.gpu_handoff_adapter import LocalHandoffAdapter
from ...media.gpu_handoff_store import GpuHandoffStore, HandoffConflict
from ...ops.service import OpsDispatchError
from ...utils.db import get_shared_engine
from ..dependencies import get_ops_service, get_service, get_tool_approval_policy_store
from .ops import DISPATCH_ERROR_STATUS
from .service_auth import NO_STORE, read_body, require_service_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/media/local-video/handoff", tags=["Media"])


class OfferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    user: str = Field(min_length=1, max_length=128)
    expected_generation: int = Field(ge=0)
    calibration: bool = False


class ConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    user: str = Field(min_length=1, max_length=128)
    expected_generation: int = Field(ge=0)
    token: str = Field(min_length=1, max_length=128)
    approved: bool


@asynccontextmanager
async def _controller():
    from .media import _get_video_client

    config = get_service().config
    store = await asyncio.to_thread(GpuHandoffStore, get_shared_engine(config))
    ops = await asyncio.to_thread(get_ops_service, config)
    policies = await asyncio.to_thread(get_tool_approval_policy_store, config)
    async with httpx.AsyncClient() as http:
        adapter = LocalHandoffAdapter(config=config, http=http, video=_get_video_client("local-video"),
                                      ops=ops, policies=policies)
        yield GpuHandoff(store, adapter)


async def _run(method, *, coded=False, **kwargs):
    """``coded`` answers a conflict as ``{code, message}`` (the TTS restart
    contract); the older handoff routes keep their plain-string detail."""
    try:
        async with _controller() as controller:
            return await getattr(controller, method)(**kwargs)
    except HandoffConflict as exc:
        if not coded:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise HTTPException(status_code=400 if exc.code == "args_invalid" else 409,
                            detail={"code": exc.code, "message": str(exc)}, headers=NO_STORE) from exc
    except OpsDispatchError as exc:
        raise HTTPException(status_code=DISPATCH_ERROR_STATUS.get(exc.code, 500),
                            detail={"code": exc.code, "message": str(exc)}) from exc
    except Exception as exc:
        logger.exception("GPU handoff %s unavailable", method)
        raise HTTPException(status_code=503, detail=(
            f"GPU switch could not reach a required service ({type(exc).__name__}). "
            "The GPU panel shows what, if anything, changed.")) from exc


@router.post("/offer")
async def offer(body: OfferRequest):
    return await _run("offer", **body.model_dump())


@router.post("/confirm")
async def confirm(body: ConfirmRequest):
    return await _run("confirm", **body.model_dump())


class RestoreOfferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    user: str = Field(min_length=1, max_length=128)
    expected_generation: int = Field(ge=0)


@router.post("/resume-video")
async def resume_video(body: RestoreOfferRequest):
    """Side-effect free: re-verify live state and, if it is still video's, clear the fence."""
    return await _run("resume_video", **body.model_dump())


class ResolveVideoClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    user: str = Field(min_length=1, max_length=128)
    job_id: str = Field(min_length=1, max_length=32)
    expected_generation: int = Field(ge=0)
    outcome: Literal["completed", "failed", "unknown"]
    evidence: str = Field(min_length=20, max_length=1000)
    approved: bool


@router.get("/claims/resolved/{job_id}")
async def claim_resolution(job_id: str):
    return await _run("claim_resolution", job_id=job_id)


@router.post("/claims/resolve")
async def resolve_video_claim(body: ResolveVideoClaimRequest):
    """Manual, audited disposition after inspecting the named job and restoring voice.

    This does not stop/start a service or delete a video/media record. The GPU
    claim becomes an immutable resolution record only after live checks pass.
    """
    try:
        from .media import _get_store
        row = await asyncio.to_thread(_get_store().get_by_provider_job_id, body.job_id)
    except Exception as exc:
        logger.exception("GPU claim media evidence unavailable")
        raise HTTPException(status_code=503, detail="Media evidence unavailable; leave the claim fenced") from exc
    media = ({key: row.get(key) for key in ("id", "provider", "status", "file_path", "error")}
             if row else None)
    return await _run("resolve_video_claim", **body.model_dump(), media_observation=media)


# Restore waits minutes for Moshi to load, longer than proxies should hold a
# request open. Consent is consumed synchronously; progress is the GPU status.
_restores: set[asyncio.Task] = set()


async def _execute_restore(user: str, generation: int) -> None:
    try:
        async with _controller() as controller:
            await controller.execute_restore(user=user, generation=generation)
    except Exception:
        # The executor already recorded recovery_required with the reason.
        logger.exception("GPU restore to voice failed (generation %s)", generation)


@router.post("/restore/offer")
async def restore_offer(body: RestoreOfferRequest):
    return await _run("offer_restore", **body.model_dump())


@router.post("/restore/confirm")
async def restore_confirm(body: ConfirmRequest):
    generation = await _run("accept_restore", **body.model_dump())
    task = asyncio.create_task(_execute_restore(body.user, generation))
    _restores.add(task)
    task.add_done_callback(_restores.discard)
    return {"accepted": True, "generation": generation}


# ── TTS restart (TASK-1171) ─────────────────────────────────────────────────
# BawtHub's TTS admin restart. Service-bearer only: the restart takes the GPU
# fence, so no bot reaches it; ``ops_run`` refuses the operation for anyone
# but this controller.
tts_router = APIRouter(prefix="/tts-restart", dependencies=[Depends(require_service_token)])


class TtsRestartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")


@tts_router.post("")
async def tts_restart(request: Request):
    """Restart TTS once while voice owns the GPU; a replayed ``request_id`` reads the same job."""
    body = await read_body(request, TtsRestartRequest)
    return await _run("tts_restart", coded=True, request_id=body.request_id)


@tts_router.get("/{request_id}")
async def tts_restart_status(request_id: str):
    """``outcome``: in_progress | held | released | refused. Reading it is what releases the fence."""
    return await _run("tts_restart_status", coded=True, request_id=request_id)


router.include_router(tts_router)
