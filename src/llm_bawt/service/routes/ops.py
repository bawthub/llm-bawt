"""HTTP admin surface for the ops catalog + job ledger (TASK-639).

CRUD for :class:`OpsOperation`, the seed bootstrap, and read-only job
listing / status endpoints. Also exposes a direct-dispatch endpoint so the
BawtHub operations page can trigger a job without going through the MCP
approval flow (still policy-gated at the HTTP layer per operator setup).

BawtHub can proxy every route here via its existing ``/api/chat/proxy/v1/*``
path — no new frontend server prefix is required.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..dependencies import get_ops_service, get_ops_store, get_service

log = logging.getLogger(__name__)
router = APIRouter()


def _store():
    service = get_service()
    store = get_ops_store(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Ops store database unavailable")
    return store


def _service():
    service = get_service()
    ops = get_ops_service(service.config)
    return ops


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class OperationUpsert(BaseModel):
    model_config = {"extra": "forbid"}
    slug: str | None = None
    title: str | None = None
    description: str | None = None
    enabled: bool | None = None
    executor_kind: str | None = None
    target_host: str | None = None
    run_as_user: str | None = None
    working_directory: str | None = None
    command_script: str | None = None
    args_schema_json: str | None = None
    args_defaults_json: str | None = None
    timeout_seconds: int | None = None
    start_delay_seconds: int | None = None
    max_output_bytes: int | None = None
    max_concurrent: int | None = None
    risk_level: str | None = None
    category: str | None = None
    approval_prompt_prefix: str | None = None

    actor: str | None = Field(default=None, max_length=128)

    def writable(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True, exclude={"actor"})


class EnableRequest(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    enabled: bool = True
    actor: str | None = Field(default=None, max_length=128)


class DispatchRequest(BaseModel):
    model_config = {"extra": "forbid"}
    operation: str = Field(..., description="Operation slug")
    args: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, max_length=128)
    actor: str | None = Field(default=None, max_length=128)
    caller_user_id: str | None = Field(default=None, max_length=128)
    caller_bot_id: str | None = Field(default=None, max_length=128)
    caller_turn_id: str | None = Field(default=None, max_length=128)
    caller_session_key: str | None = Field(default=None, max_length=128)


# ---------------------------------------------------------------------------
# Operation CRUD
# ---------------------------------------------------------------------------

@router.get("/v1/ops/operations", tags=["Ops"])
def list_operations(include_disabled: bool = False, include_soft_deleted: bool = False,
                    limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    store = _store()
    filters = dict(include_disabled=include_disabled, include_soft_deleted=include_soft_deleted)
    rows = store.list_operations(**filters, limit=limit, offset=offset)
    return {"operations": [r.to_api(include_script=False) for r in rows],
            "total": store.count_operations(**filters), "limit": limit, "offset": offset}


@router.get("/v1/ops/operations/{slug}", tags=["Ops"])
def get_operation(slug: str, include_script: bool = True):
    store = _store()
    row = store.get_operation_by_slug(slug)
    if row is None:
        raise HTTPException(status_code=404, detail=f"operation not found: {slug}")
    return row.to_api(include_script=include_script)


@router.get("/v1/ops/operations/{slug}/revisions", tags=["Ops"])
def list_revisions(slug: str, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    store = _store()
    if store.get_operation_by_slug(slug) is None:
        raise HTTPException(status_code=404, detail=f"operation not found: {slug}")
    rows, total = store.list_revisions(slug, limit=limit, offset=offset)
    return {"revisions": [row.to_api() for row in rows], "total": total, "limit": limit, "offset": offset}


@router.post("/v1/ops/operations", tags=["Ops"], status_code=201)
def create_operation(body: OperationUpsert):
    store = _store()
    try:
        row = store.create_operation(body.writable(), actor=body.actor or "api")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return row.to_api(include_script=True)


@router.patch("/v1/ops/operations/{slug}", tags=["Ops"])
def update_operation(slug: str, body: OperationUpsert):
    store = _store()
    try:
        row = store.update_operation(slug, body.writable(), actor=body.actor or "api")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(status_code=404, detail=f"operation not found: {slug}")
    return row.to_api(include_script=True)


@router.post("/v1/ops/operations/{slug}/soft-delete", tags=["Ops"])
def soft_delete_operation(slug: str, actor: str | None = Query(None, max_length=128)):
    store = _store()
    try:
        deleted = store.soft_delete_operation(slug, actor=actor or "api")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"operation not found: {slug}")
    return {"ok": True, "slug": slug}


@router.post("/v1/ops/operations/{slug}/enable", tags=["Ops"])
def enable_operation(slug: str, body: EnableRequest | None = None):
    body = body or EnableRequest()
    store = _store()
    try:
        row = store.update_operation(slug, {"enabled": body.enabled}, actor=body.actor or "api")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(status_code=404, detail=f"operation not found: {slug}")
    return row.to_api(include_script=False)


@router.post("/v1/ops/seed-defaults", tags=["Ops"])
def seed_defaults():
    """Insert every canonical seed row that isn't already present.

    Never overwrites operator edits — a slug that already exists is skipped
    whether or not the seed dict has drifted.
    """
    from ...ops.seeds import seed_all

    store = _store()
    inserted, skipped = seed_all(store)
    return {"inserted": inserted, "skipped": skipped, "total_inserted": len(inserted)}


# ---------------------------------------------------------------------------
# Durable release lifecycle (read-only)
# ---------------------------------------------------------------------------

@router.get("/v1/ops/releases", tags=["Ops"])
def list_releases(limit: int = Query(25, ge=1, le=200), offset: int = Query(0, ge=0)):
    releases = _service().releases
    if releases.engine is None:
        raise HTTPException(status_code=503, detail="Release store database unavailable")
    return {"releases": [row.to_api() for row in releases.list(limit=limit, offset=offset)],
            "total": releases.count(), "limit": limit, "offset": offset}


@router.get("/v1/ops/releases/{release_id}", tags=["Ops"])
def get_release(release_id: str):
    releases = _service().releases
    if releases.engine is None:
        raise HTTPException(status_code=503, detail="Release store database unavailable")
    row = releases.get(release_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"release not found: {release_id}")
    result = row.to_api(events=releases.events(row.id))
    if row.state == "deployed" and row.deploy_job_id:
        from ...ops.image_deploy import parse_deployment

        job = _service().store.get_job(row.deploy_job_id)
        record = parse_deployment(job.output_tail) if job and job.state == "succeeded" else None
        baked = ((record or {}).get("health") or {}).get("release") or {}
        deployed = (record or {}).get("deployed") or {}
        if (baked.get("version") == row.version and baked.get("sourceSha") == row.source_sha
                and str(baked.get("workflowRunId")) == str(row.github_run_id)
                and (record or {}).get("health", {}).get("status") == "healthy"
                and deployed.get("image_ref") == f"{row.image_repository}@{row.digest}"):
            # Historical verified identity; never claim this is a live health read.
            result["production_identity"] = {
                "version": row.version, "source_sha": row.source_sha, "digest": row.digest,
                "workflow_run_id": row.github_run_id,
                "verified_at": result["finished_at"],
            }
    return result


# ---------------------------------------------------------------------------
# Job listing + status
# ---------------------------------------------------------------------------

@router.get("/v1/ops/jobs", tags=["Ops"])
def list_jobs(operation: str | None = None, state: str | None = None,
              limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    store = _store()
    filters = dict(operation_slug=operation, state=state)
    rows = store.list_jobs(**filters, limit=limit, offset=offset)
    return {"jobs": [r.to_api(include_output=False) for r in rows],
            "total": store.count_jobs(**filters), "limit": limit, "offset": offset}


@router.get("/v1/ops/jobs/{job_id}", tags=["Ops"])
def get_job(job_id: str, output_tail_bytes: int | None = None):
    ops = _service()
    result = ops.get_job_status(
        job_id,
        output_tail_bytes=output_tail_bytes,
        reconcile_if_active=True,
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return result


# ---------------------------------------------------------------------------
# Direct dispatch (HTTP surface for the operator UI)
# ---------------------------------------------------------------------------

@router.post("/v1/ops/jobs", tags=["Ops"], status_code=202)
def dispatch_job(body: DispatchRequest):
    """Directly dispatch a job via HTTP.

    Same semantics as the ``ops_run`` MCP tool — validation + idempotency
    + executor dispatch — but callable from BawtHub's operations page or
    from a terminal (``curl``). Approval gating for this endpoint stays a
    frontend concern; the store enforces disabled/soft-deleted status.
    """
    from ...ops.service import OpsDispatchError

    ops = _service()
    # No explicit key means a genuinely new invocation, not permanent dedup.
    idem = (body.idempotency_key or "").strip() or None
    try:
        result = ops.dispatch_job(
            operation_slug=body.operation,
            args=body.args or {},
            idempotency_key=idem,
            caller_actor=body.actor or "api",
            caller_user_id=body.caller_user_id,
            caller_bot_id=body.caller_bot_id,
            caller_turn_id=body.caller_turn_id,
            caller_session_key=body.caller_session_key,
            caller_backend="http-operator",
        )
    except OpsDispatchError as exc:
        status = {
            "operation_not_found": 404,
            "operation_disabled": 409,
            "args_invalid": 400,
            "idempotency_conflict": 409,
            "snapshot_invalid": 409,
            "target_unavailable": 409,
            "executor_unavailable": 503,
            "dispatch_failed": 502,
            "executor_kind_unknown": 500,
        }.get(exc.code, 500)
        raise HTTPException(status_code=status, detail={"code": exc.code, "message": str(exc)}) from exc
    return result
