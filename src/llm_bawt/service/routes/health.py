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

    config = get_service().config
    return collect(config, get_ops_store(config), get_ops_service(config))


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
