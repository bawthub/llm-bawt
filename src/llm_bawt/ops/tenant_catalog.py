"""Point the restart rows of the ops catalog at a tenant install (TASK-2128).

The seeds name echo's containers and ship disabled. A tenant install
(``bawthub/deploy/bootstrap.sh``) runs one Compose project whose services have
explicit ``container_name`` values. After ``up`` the installer feeds this
module the project and its running ``service -> container`` map; it rewrites
only the restart rows whose targets exist there and enables them. Every run
still needs an approval card (the ``ops_run`` approval policy).

Idempotent and operator-safe: a row is rewritten only while it still holds the
seed's ``command_script``. Once classified (or edited by an operator) it is
left alone, so a rerun changes nothing and never re-enables a row an operator
turned off. Config-reconcile rows stay disabled: their env/mount inventory is
install-owned and fails closed (TASK-2122).

Usage (inside the app container)::

    python -m llm_bawt.ops.tenant_catalog --project NAME < services.json

``services.json`` is ``{"<compose service>": "<container name>", ...}``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

from .seeds import SEEDS, seed_all

ACTOR = "tenant-install"
#: The actor ``OpsStore.seed_operation_if_missing`` records on a fresh seed row.
SEED_ACTOR = "system-seed"
_HOUSE = re.compile(r"^house-[a-z0-9][a-z0-9-]{0,30}$")
#: Optional single-container services the aux restart may target.
AUX_SERVICES = ("pocket-tts", "voice-backend", "local-model-bridge")
_BRIDGE_HEALTH = {"health_url": "http://127.0.0.1:8681/health", "health_field": "redis",
                  "health_value": True, "health_timeout_seconds": 120}


def _schema(props: dict[str, Any], required: list[str]) -> str:
    body = {"type": "object", "additionalProperties": False, "properties": props}
    if required:
        body["required"] = required
    return json.dumps(body, ensure_ascii=False)


def _spec(**fields: Any) -> str:
    return json.dumps(fields, ensure_ascii=False)


def plan(project: str, services: dict[str, str]) -> dict[str, dict[str, Any]]:
    """The tenant form of each applicable restart row, keyed by slug."""
    out: dict[str, dict[str, Any]] = {}
    if "app" in services:
        out["llm-bawt.restart-app"] = {"command_script": _spec(
            action="restart", container_name=services["app"],
            health_url="http://127.0.0.1:8642/health", health_field="status", health_value="ok",
            health_timeout_seconds=150)}
    if "agent-bridge" in services:
        out["llm-bawt.restart-bridge"] = {
            "description": "Restart the agent bridge. HIGH risk: kills active agent sessions on it.",
            "command_script": _spec(action="restart", container_name=services["agent-bridge"],
                                    **_BRIDGE_HEALTH),
            "args_schema_json": _schema({}, []),
            # One fixed bridge, so the drain scope is static (the seed's scope
            # keyed off the removed `bridge` argument). Off until an operator
            # turns it on, as on echo.
            "quiesce_enabled": False,
            "quiesce_scope_json": json.dumps({"backends": ["claude-code"]}),
        }
    if "frontend" in services:
        out["bawthub.restart-frontend"] = {
            "title": "Restart bawthub frontend",
            "description": "Restart the bawthub frontend container.",
            "command_script": _spec(
                action="restart", container_name=services["frontend"],
                health_url="http://127.0.0.1:3000/api/health", health_field="status",
                health_value="healthy", health_timeout_seconds=150)}
    if "redis" in services:
        out["llm-bawt.restart-redis"] = {"command_script": _spec(
            action="restart", container_name=services["redis"])}
    aux = [services[s] for s in AUX_SERVICES if s in services]
    if aux:
        out["llm-bawt.restart-aux"] = {
            "description": "Restart a tenant voice or embedding service.",
            "command_script": _spec(action="restart", container_name_from_arg="service"),
            "args_schema_json": _schema({"service": {"type": "string", "enum": aux}}, ["service"]),
        }
    if any(_HOUSE.fullmatch(s) for s in services):
        out["llm-bawt.restart-house"] = {"command_script": _spec(
            action="restart", compose_project=project, compose_service_from_arg="house",
            **_BRIDGE_HEALTH)}
    return out


def classify(store, project: str, services: dict[str, str]) -> dict[str, list[str]]:
    seeds = {s["slug"]: s for s in SEEDS}
    result: dict[str, list[str]] = {"classified": [], "unchanged": [], "missing": []}
    for slug, fields in plan(project, services).items():
        row = store.get_operation_by_slug(slug)
        if row is None or row.soft_deleted_at is not None:
            result["missing"].append(slug)
            continue
        # Untouched seed only: last written by the seeder with the seed's
        # script. After classification (or any operator edit) updated_by
        # changes, so a rerun never re-enables a row even when the tenant
        # target string equals the seed's (e.g. a project named llm-bawt).
        if row.updated_by != SEED_ACTOR or row.command_script != seeds[slug]["command_script"]:
            result["unchanged"].append(slug)
            continue
        written = store.update_operation(slug, {**fields, "enabled": True}, actor=ACTOR,
                                         expect_version=row.version)
        result["classified" if written is not None else "unchanged"].append(slug)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", required=True)
    args = parser.parse_args(argv)
    services = json.load(sys.stdin)
    if not isinstance(services, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and v for k, v in services.items()):
        parser.error("stdin must be a JSON object of service -> container name")
    from ..utils.config import Config
    from .store import OpsStore

    store = OpsStore(Config())
    if store.engine is None:
        print("ops store unavailable", file=sys.stderr)
        return 2
    # A fresh tenant has no catalog rows until first-run setup seeds them, so
    # seed here first. Insert-if-missing: existing rows and edits are kept.
    seed_all(store)
    result = classify(store, args.project, services)
    print(json.dumps(result, sort_keys=True))
    return 3 if result["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
