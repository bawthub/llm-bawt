"""Preview network topology: one source for the compose guard and the live probe (TASK-1802).

``tests/test_preview_compose.py`` checks docker-compose.yml against these
tables; ``probes.inventory`` checks the running containers against the same
tables, so the declared and the live topology cannot drift apart.
"""

from __future__ import annotations

# Compose network key -> fixed Docker network name (set with ``name:``).
NETWORK_NAMES: dict[str, str] = {
    "preview": "llm-bawt-preview",
    "preview-broker-net": "llm-bawt-preview-broker",
    "preview-credential-net": "llm-bawt-preview-credential",
}

# Preview services and the exact networks each joins. A preview service not
# listed here joins ``preview`` alone. The credential gateway bridges the prod
# app (broker side) and the preview bridge (consumer side) and is on nothing
# else: never ``default`` (prod Redis) and never ``preview``.
SERVICE_NETWORKS: dict[str, frozenset[str]] = {
    "preview-credential-gateway": frozenset({"preview-broker-net", "preview-credential-net"}),
}
DEFAULT_SERVICE_NETWORKS = frozenset({"preview"})

# Non-preview services allowed onto a preview network, and which.
PROD_JOINERS: dict[str, frozenset[str]] = {"app": frozenset({"preview-broker-net"})}

# Long-running preview services the probe requires to be up.
REQUIRED_RUNNING = frozenset({"preview-postgres", "preview-redis"})

SERVICE_PREFIX = "preview-"


def is_preview_service(name: str) -> bool:
    return name.startswith(SERVICE_PREFIX)


def networks_for(service: str) -> frozenset[str]:
    return SERVICE_NETWORKS.get(service, DEFAULT_SERVICE_NETWORKS)
