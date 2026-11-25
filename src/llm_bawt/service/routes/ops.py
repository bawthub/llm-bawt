"""HTTP admin surface for the ops catalog + job ledger (TASK-639).

CRUD for :class:`OpsOperation`, the seed bootstrap, and read-only job
listing / status endpoints. Also exposes a direct-dispatch endpoint so the
BawtHub operations page can trigger a job without going through the MCP
approval flow (still policy-gated at the HTTP layer per operator setup).

BawtHub can proxy every route here via its existing ``/api/chat/proxy/v1/*``
path — no new frontend server prefix is required.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ...ops.quiesce_status import attach_members, live_holds
from ...ops.release_status import attach_release_status
from ..dependencies import get_ops_service, get_ops_store, get_service
from .service_auth import NO_STORE, read_body, require_service_token

log = logging.getLogger(__name__)
router = APIRouter()
# BawtHub-server only (TASK-2370): the bridges and the LAN never hold this bearer.
_SERVICE = [Depends(require_service_token)]


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
    approval_required: bool | None = Field(default=None, strict=True)
    approval_prompt_prefix: str | None = None
    # TASK-1233 quiesce settings; the store validates them with the whole row.
    quiesce_enabled: bool | None = None
    quiesce_timeout_seconds: int | None = None
    quiesce_scope_json: str | None = None
    # TASK-1169 S4: where the worker runs; the store validates it per executor.
    execution_mode: str | None = None
    # Inert gate-only grant configuration; no gate admission route is enabled.
    execution_tag: str | None = Field(default=None, strict=True, max_length=64)
    # TASK-1183: bell notifications for this operation's job start and finish.
    notify_bell: bool | None = Field(default=None, strict=True)

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


@router.post("/v1/ops/operations", tags=["Ops"], status_code=201, dependencies=_SERVICE)
def create_operation(body: OperationUpsert):
    store = _store()
    try:
        row = store.create_operation(body.writable(), actor=body.actor or "api")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return row.to_api(include_script=True)


@router.patch("/v1/ops/operations/{slug}", tags=["Ops"], dependencies=_SERVICE)
def update_operation(slug: str, body: OperationUpsert):
    store = _store()
    try:
        row = store.update_operation(slug, body.writable(), actor=body.actor or "api")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(status_code=404, detail=f"operation not found: {slug}")
    return row.to_api(include_script=True)


@router.post("/v1/ops/operations/{slug}/soft-delete", tags=["Ops"], dependencies=_SERVICE)
def soft_delete_operation(slug: str, actor: str | None = Query(None, max_length=128)):
    store = _store()
    try:
        deleted = store.soft_delete_operation(slug, actor=actor or "api")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"operation not found: {slug}")
    return {"ok": True, "slug": slug}


@router.post("/v1/ops/operations/{slug}/enable", tags=["Ops"], dependencies=_SERVICE)
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


# TASK-2630: bots propose catalog edits (MCP ops_propose_catalog_edit); a
# person decides through BawtHub's admin check, which adds the bearer.

class ProposalDecision(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    approve: bool
    actor: str = Field(..., min_length=1, max_length=128)


@router.get("/v1/ops/catalog-proposals", tags=["Ops"])
def list_catalog_proposals(status: Literal["pending", "approved", "rejected", "stale", "partial"] | None = "pending",
                           limit: int = Query(50, ge=1, le=200)):
    from ...ops import catalog_proposals
    store = _store()
    if status == "partial":
        # Mixed batches have no single row status; filter after grouping.
        rows = [b for b in catalog_proposals.list_batches(store, status=None, limit=limit)
                if b["status"] == "partial"]
    else:
        rows = catalog_proposals.list_batches(store, status=status, limit=limit)
    return {"batches": rows}


@router.post("/v1/ops/catalog-proposals/{batch_id}/decide", tags=["Ops"], dependencies=_SERVICE)
def decide_catalog_proposal(batch_id: str, body: ProposalDecision):
    from ...ops import catalog_proposals
    store = _store()
    try:
        return catalog_proposals.decide(store, batch_id=batch_id, decided_by=body.actor,
                                        approve=body.approve)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/v1/ops/catalog-drift", tags=["Ops"])
def catalog_drift():
    """Where live catalog rows differ from the code seeds. Read-only (TASK-1861)."""
    from ...ops.catalog_drift import report

    return report(_store())


@router.post("/v1/ops/seed-defaults", tags=["Ops"], dependencies=_SERVICE)
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
    jobs = attach_members(store.engine, attach_release_status(
        store.engine, [r.to_api(include_output=False) for r in rows]))
    return {"jobs": jobs,
            "total": store.count_jobs(**filters), "limit": limit, "offset": offset}


@router.get("/v1/ops/jobs/live", tags=["Ops"])
def live_jobs():
    """All nonterminal jobs plus fresh outcomes, in one unpaginated ledger read.

    Global banners poll this while a job runs; recent outcomes make completion
    visible without a per-job status fanout or stale page-1 assumptions.
    """
    store = _store()
    # Keep outcomes beyond an idle poll interval so short bot-initiated jobs
    # cannot finish and disappear between two global banner reads.
    since = datetime.now(timezone.utc) - timedelta(seconds=60)
    rows = store.find_active_jobs(limit=None, recent_terminal_since=since)
    return {"jobs": attach_members(store.engine, attach_release_status(
        store.engine, [r.to_api(include_output=False) for r in rows]))}


@router.get("/v1/ops/jobs/{job_id}", tags=["Ops"])
def get_job(job_id: str, output_tail_bytes: int | None = None):
    # A read only (TASK-1169 S1d): the ops reconciler is the one writer of job
    # state; observed_at and last_reconcile_at say how fresh this read is.
    ops = _service()
    result = ops.get_job_status(
        job_id,
        output_tail_bytes=output_tail_bytes,
        reconcile_if_active=False,
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return result


class CancelRequest(BaseModel):
    model_config = {"extra": "forbid"}
    actor: str = Field(..., min_length=1, max_length=128,
                       description="Operator label; self-asserted, recorded as claimed")
    reason: str | None = Field(default=None, max_length=500)


@router.post("/v1/ops/jobs/{job_id}/cancel", tags=["Ops"], dependencies=_SERVICE)
def cancel_job(job_id: str, body: CancelRequest):
    """Cancel a quiesce job that has not dispatched (TASK-1233 (f)).

    Releases the job's holds and records ``cancelled`` with no action
    submitted. A job that already dispatched (or is not a quiesce job) is 409
    ``not_cancellable``: its action may already be running."""
    from ...ops.quiesce import QuiesceUnavailable

    try:
        outcome, job = _service().cancel_quiesce_job(job_id, actor=body.actor, reason=body.reason)
    except QuiesceUnavailable as exc:
        raise HTTPException(status_code=503, detail={"code": "quiesce_unavailable", "message": str(exc)}) from exc
    if outcome == "missing":
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    if outcome == "not_cancellable":
        raise HTTPException(status_code=409, detail={
            "code": "not_cancellable", "state": job["state"],
            "message": "only a quiescing job that has not dispatched can be cancelled"})
    if outcome == "already":
        return {"job": job, "cancelled": False, "already": True}
    return {"job": job, "cancelled": True}


@router.get("/v1/ops/holds", tags=["Ops"])
def list_holds():
    """Every live hold/deny gate. ``job`` is null for operator-owned gates."""
    return {"holds": live_holds(_store().engine)}


# ---------------------------------------------------------------------------
# Direct dispatch (HTTP surface for the operator UI)
# ---------------------------------------------------------------------------

#: HTTP status per ``OpsDispatchError`` code, shared by every route that dispatches.
DISPATCH_ERROR_STATUS = {
    "operation_not_found": 404,
    "operation_disabled": 409,
    # TASK-1474: an approval-card-only operation; the operator Run button can't start it.
    "approval_required": 409,
    "args_invalid": 400,
    "idempotency_conflict": 409,
    "snapshot_invalid": 409,
    "target_unavailable": 409,
    # TASK-1171: only the GPU handoff may start or recreate a Moshi container.
    "gpu_handoff_required": 409,
    "gpu_guard_unverifiable": 409,
    # TASK-2100: a house action refuses while the house is occupied.
    "house_occupied": 409,
    "house_occupancy_unverifiable": 409,
    "executor_unavailable": 503,
    "dispatch_failed": 502,
    "executor_kind_unknown": 500,
    # TASK-1169 S3: the app prepares through the ops service; nothing was created.
    "ops_unavailable": 503,
    # An admission-only service was asked to run an executor: a code defect.
    "ops_admission_only": 500,
}


@router.post("/v1/ops/jobs", tags=["Ops"], status_code=202, dependencies=_SERVICE)
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
            require_catalog_free=True,
        )
    except OpsDispatchError as exc:
        status = DISPATCH_ERROR_STATUS.get(exc.code, 500)
        raise HTTPException(status_code=status, detail={"code": exc.code, "message": str(exc)}) from exc
    return result


class OperatorDispatchRequest(BaseModel):
    model_config = {"extra": "forbid", "str_strip_whitespace": True}
    operation: str = Field(min_length=1, max_length=128, strict=True)
    args: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = Field(min_length=1, max_length=128, strict=True)
    # Assertion by authenticated BawtHub after its verified-person manager guard,
    # never accepted directly from the browser or the legacy LAN dispatch path.
    operator_user_id: str = Field(min_length=1, max_length=128, strict=True)
    operator_run: Literal[True]


@router.post("/v1/ops/operator-jobs", tags=["Ops"], status_code=202,
             dependencies=[Depends(require_service_token)])
async def dispatch_operator_job(request: Request):
    from ...ops.operator_run import dispatch_operator_run
    from ...ops.service import OpsDispatchError
    from .approval_policies import _store as approval_store

    body = await read_body(request, OperatorDispatchRequest)
    try:
        return await asyncio.to_thread(
            dispatch_operator_run, _service(), approval_store(),
            user_id=body.operator_user_id, operation=body.operation,
            args=body.args, idempotency_key=body.idempotency_key)
    except OpsDispatchError as exc:
        raise HTTPException(DISPATCH_ERROR_STATUS.get(exc.code, 500),
                            {"code": exc.code, "message": str(exc)}, headers=NO_STORE) from exc


# ---------------------------------------------------------------------------
# Host access records (TASK-2430): the destinations host_key jobs may target.
# Writes are owner-only; jobs run through the approved ops catalog.
# ---------------------------------------------------------------------------

class HostAccessCreate(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    label: str = Field(..., max_length=64)
    host: str = Field(..., max_length=253)
    port: int = 22
    login_user: str = Field(..., max_length=32)
    actor: str | None = Field(default=None, max_length=128)


class HostAccessPatch(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    label: str | None = Field(default=None, max_length=64)
    host: str | None = Field(default=None, max_length=253)
    port: int | None = None
    login_user: str | None = Field(default=None, max_length=32)


@router.get("/v1/ops/host-access", tags=["Ops"])
def list_host_access():
    from ...ops import host_access
    from ...ops.service import OpsDispatchError
    hosts = host_access.list_hosts(_store().engine)
    try:
        bot_key = _service().bot_public_key()
    except OpsDispatchError as exc:
        bot_key = {"public_key": None, "fingerprint": None, "error": str(exc)}
    return {"hosts": hosts, "bot_key": bot_key}


@router.post("/v1/ops/host-access", tags=["Ops"], status_code=201, dependencies=_SERVICE)
def create_host_access(body: HostAccessCreate):
    from ...ops import host_access
    try:
        return host_access.create_host(_store().engine, body.model_dump(exclude={"actor"}), actor=body.actor)
    except host_access.HostAccessError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.patch("/v1/ops/host-access/{host_id}", tags=["Ops"], dependencies=_SERVICE)
def update_host_access(host_id: int, body: HostAccessPatch):
    from ...ops import host_access
    try:
        return host_access.update_host(_store().engine, host_id, body.model_dump(exclude_unset=True))
    except KeyError as exc:
        raise HTTPException(404, "host record not found") from exc
    except host_access.HostAccessError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/v1/ops/host-access/{host_id}", tags=["Ops"], dependencies=_SERVICE)
def delete_host_access(host_id: int):
    from ...ops import host_access
    if not host_access.delete_host(_store().engine, host_id):
        raise HTTPException(404, "host record not found")
    return {"deleted": host_id}


class HostKeyConfirm(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    fingerprints: list[str] = Field(..., min_length=1, max_length=8)


@router.post("/v1/ops/host-access/{host_id}/confirm-host-key", tags=["Ops"], dependencies=_SERVICE)
def confirm_host_access_key(host_id: int, body: HostKeyConfirm):
    """TASK-2520: pin the scanned host keys the operator confirmed by fingerprint."""
    from ...ops import host_access
    try:
        return host_access.confirm_host_keys(_store().engine, host_id, body.fingerprints)
    except KeyError as exc:
        raise HTTPException(404, "host record not found") from exc
    except host_access.HostAccessError as exc:
        raise HTTPException(400, str(exc)) from exc
