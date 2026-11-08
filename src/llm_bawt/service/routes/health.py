"""Health and service status routes."""

import dataclasses

from fastapi import APIRouter

from ..dependencies import get_service
from ..schemas import HealthResponse, ServiceStatusResponse, SystemStatusResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check endpoint."""
    return HealthResponse()


@router.get("/status", response_model=ServiceStatusResponse, tags=["System"])
def get_status():
    """Get detailed service status."""
    return get_service().get_status()


@router.get("/v1/health/checks", tags=["System"])
def get_health_checks():
    """Internal capability checks (DB, Redis, ops executors, ops queue).

    Always 200 when the service answers: per-check status is the payload, and
    the BawtHub aggregator treats an unreachable service as its own failure.
    """
    from ..dependencies import get_ops_service, get_ops_store
    from ..health_checks import collect

    service = get_service()
    config = service.config
    return collect(config, get_ops_store(config), get_ops_service(config), service=service)


CHANGES_HOLD_SECONDS = 25.0
# Each waiter holds a threadpool thread (shared with every sync route) for the
# hold; beyond the cap, 503 so the client backs off rather than spins.
MAX_CHANGE_WAITERS = 8
_change_waiters = {"n": 0}  # event-loop only, so no lock


@router.get("/v1/health/changes/{seen}", tags=["System"])
async def get_health_changes(seen: str):
    """Long-poll the app health change version (TASK-2105 S3).

    Returns at once when the version differs from ``seen`` (use 0 to read
    it), else after a change or the hold. BawtHub reruns its llm-bawt and
    provider checks when it moves.
    """
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import JSONResponse

    from ..health_changes import health_changes

    if not seen.isdigit() or len(seen) > 18:
        return JSONResponse({"error": "not found"}, status_code=404)
    if _change_waiters["n"] >= MAX_CHANGE_WAITERS:
        return JSONResponse({"error": "busy"}, status_code=503)
    _change_waiters["n"] += 1
    try:
        version = await run_in_threadpool(health_changes().wait_for_change, int(seen), CHANGES_HOLD_SECONDS)
    finally:
        _change_waiters["n"] -= 1
    return {"schemaVersion": 1, "version": version}


@router.get("/v1/health/maintenance", tags=["System"])
def get_maintenance_health():
    """Per-job outcome of botless maintenance/utility calls (TASK-1704).

    Always 200: ``{"checks": [{name, status, reason}]}``, one row per job that
    has run since start. A failing job is ``down`` with its error class,
    provider, credential name and failure count; its next success clears it.

    Backup-model state (slice B) adds at most one ``degraded`` row per provider:
    ``maintenance_fallback:<p>`` while on the backup, then
    ``maintenance_fallback_recovered:<p>`` for 24h after switch-back.
    It is read from the durable Redis record on every poll, so a switch is
    never lost to a restart; Redis down is reported by the state row instead.
    """
    from ..maintenance_fallback import (
        StateUnavailable, default_state_health, default_store, fallback_checks,
    )
    from ..maintenance_health import maintenance_health

    try:
        fallback_rows = fallback_checks(default_store(get_service().config))
    except StateUnavailable as error:
        default_state_health().unavailable(str(error))
        fallback_rows = []
    payload = maintenance_health().checks_payload()
    payload["checks"].extend(fallback_rows)
    return payload


@router.post("/v1/maintenance/primary-available", tags=["System"])
def announce_maintenance_primary_available():
    """A local maintenance primary announces it is back (TASK-1716).

    Push-based recovery: ends the cooldown of every provider switched to the
    backup as ``unreachable``, so the next maintenance call probes the primary.
    Takes no input and clears nothing itself: only that real call switches back,
    so the call is safe without credentials. Idempotent.
    """
    from fastapi.responses import JSONResponse

    from ..maintenance_fallback import (
        StateUnavailable, announce_available, default_state_health, default_store,
    )

    try:
        providers = announce_available(default_store(get_service().config))
    except StateUnavailable as error:
        default_state_health().unavailable(str(error))
        return JSONResponse({"error": "fallback_state_unavailable"}, status_code=503)
    return {"expedited": providers}


@router.get("/v1/status", response_model=SystemStatusResponse, tags=["System"])
def get_system_status():
    """Get full system status (config, service, memory, dependencies)."""
    from llm_bawt.core.status import collect_system_status

    service = get_service()
    # Collect local status (local_only=True avoids recursive self-call)
    status = collect_system_status(service.config, local_only=True)
    status_dict = dataclasses.asdict(status)

    # Override service info with real running-service data.
    # When collect_system_status runs local_only=True the ServiceInfo is empty
    # (available=False) because it deliberately skips the network check.
    # We are *inside* the service right now, so we can populate it directly.
    svc = service.get_status()
    status_dict["service"] = {
        "available": True,
        "healthy": svc.healthy,
        "uptime_seconds": svc.uptime_seconds,
        "current_model": svc.current_model,
        "default_model": svc.default_model,
        "default_bot": svc.default_bot,
        "tasks_processed": svc.tasks_processed,
        "tasks_pending": svc.tasks_pending,
    }

    # local_only=True also sets mode="direct"; correct it to "service".
    status_dict["config"]["mode"] = "service"

    return SystemStatusResponse(**status_dict)


# -------------------------------------------------------------------------
# Nextcloud Talk Webhook
# -------------------------------------------------------------------------
