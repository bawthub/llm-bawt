"""Canonical operation catalog seeds (TASK-639).

Every row here is inserted per-slug via
:meth:`OpsStore.seed_operation_if_missing` — an existing row with the same
slug is preserved as-is, never overwritten. Operators own the catalog after
first insert; seeds only bootstrap or add net-new slugs.

Seeds ship **disabled** so nothing dangerous is invocable until an operator
explicitly enables the row in the BawtHub UI. This is a deliberate safety
choice — the TASK-639 spec explicitly warns against "a safety feature that
ships disabled protects nothing", but that guidance applies to *approval
policies*, not to the executable ops catalog itself. Restart-your-own-bridge
scripts should require a human enable.

Argument schemas use ``additionalProperties: false`` so unknown args are
rejected before any script runs.

Every op targets the local Docker daemon via the mounted socket. The
``command_script`` column holds a JSON spec that
:class:`~llm_bawt.ops.executor.DockerExecutor` parses:

    {"action": "restart", "container_name": "..."}
    {"action": "restart", "compose_project": "...", "compose_service": "..."}
    {"action": "restart", "compose_project": "...",
     "compose_service_from_arg": "<arg-key>"}

A restart/start may add ``health_url``/``health_field``/``health_value``/
``health_timeout_seconds`` (TASK-1697, :mod:`.restart_health`): the job then
fails unless the service answers healthy afterwards.
"""

from __future__ import annotations

import json
from typing import Any

from .models import (
    EXECUTOR_DOCKER,
    EXECUTOR_RELEASE,
    RISK_CRITICAL,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
)
from .probe_contract import DESTINATION_PROBES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Compose project as seen in the docker labels (com.docker.compose.project).
_LLM_BAWT_PROJECT = "llm-bawt"
_BAWTHUB_PROJECT = "bawthub"


def _schema(props: dict[str, Any] | None = None, required: list[str] | None = None) -> str:
    body: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": props or {},
    }
    if required:
        body["required"] = required
    return json.dumps(body, ensure_ascii=False)


def _no_args_schema() -> str:
    return _schema({})


def _defaults(d: dict[str, Any] | None = None) -> str:
    return json.dumps(d or {}, ensure_ascii=False)


def _spec(**fields: Any) -> str:
    return json.dumps(fields, ensure_ascii=False)


# ---------------------------------------------------------------------------
# llm-bawt service restarts
# ---------------------------------------------------------------------------

_LLM_BAWT_SEEDS: list[dict[str, Any]] = [
    {
        "slug": "llm-bawt.restart-app",
        "title": "Restart llm-bawt app",
        "description": (
            "Reload Python source in the `app` container. Safe for `.py` edits — "
            "does not rebuild image. Executor forces a start_delay so the "
            "response has time to drain before Docker kills us."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_DOCKER,
        "target_host": "",
        "working_directory": None,
        # The production app has a stable explicit container_name. Prefer it to
        # Compose labels here: an ops worker image can inherit Compose labels
        # from its base image and otherwise impersonate the target service.
        "command_script": _spec(
            action="restart",
            container_name="llm-bawt-app",
            # TASK-1697: the job fails unless the app answers healthy again.
            health_url="http://127.0.0.1:8642/health", health_field="status", health_value="ok",
            health_timeout_seconds=150,
        ),
        "args_schema_json": _no_args_schema(),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 240,
        "start_delay_seconds": 5,
        "risk_level": RISK_MEDIUM,
        "category": "restart",
        "approval_prompt_prefix": "Restart llm-bawt app container",
    },
    {
        "slug": "llm-bawt.restart-aux",
        "title": "Restart llm-bawt aux service",
        "description": (
            "Restart a non-bridge llm-bawt service (crawl4ai, playwright-mcp, "
            "local-model-bridge). Bridges + redis are excluded — those need "
            "explicit dedicated ops."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_DOCKER,
        "target_host": "",
        "working_directory": None,
        "command_script": _spec(
            action="restart",
            compose_project=_LLM_BAWT_PROJECT,
            compose_service_from_arg="service",
        ),
        "args_schema_json": _schema(
            {
                "service": {
                    "type": "string",
                    "enum": ["crawl4ai", "playwright-mcp", "local-model-bridge"],
                }
            },
            required=["service"],
        ),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 180,
        "start_delay_seconds": 0,
        "risk_level": RISK_LOW,
        "category": "restart",
        "approval_prompt_prefix": "Restart llm-bawt aux service",
    },
    {
        "slug": "llm-bawt.restart-bridge",
        "title": "Restart an agent bridge",
        "description": (
            "Restart claude-code-bridge or codex-bridge. HIGH "
            "risk: kills active agent sessions hosted by that bridge. Requires "
            "explicit operator approval each time."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_DOCKER,
        "target_host": "",
        "working_directory": None,
        "command_script": _spec(
            action="restart",
            compose_project=_LLM_BAWT_PROJECT,
            compose_service_from_arg="bridge",
        ),
        "args_schema_json": _schema(
            {
                "bridge": {
                    "type": "string",
                    "enum": ["claude-code-bridge", "codex-bridge"],
                }
            },
            required=["bridge"],
        ),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 180,
        "start_delay_seconds": 10,
        "risk_level": RISK_HIGH,
        "category": "restart",
        "approval_prompt_prefix": "Restart bridge (KILLS ACTIVE AGENTS)",
        # TASK-1233: off until an operator enables it. When on, the restart
        # holds the named bridge's agent turns and waits for them to drain.
        "quiesce_enabled": False,
        "quiesce_timeout_seconds": 600,
        "quiesce_scope_json": json.dumps({
            "backend_from_arg": "bridge",
            "map": {"claude-code-bridge": "claude-code", "codex-bridge": "codex"},
        }, sort_keys=True),
    },
    {
        "slug": "llm-bawt.restart-redis",
        "title": "Restart Redis",
        "description": (
            "Restart the shared Redis instance. CRITICAL risk: cuts the agent "
            "bridge event bus. Every in-flight agent turn dies. Requires "
            "explicit operator approval each time."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_DOCKER,
        "target_host": "",
        "working_directory": None,
        "command_script": _spec(
            action="restart",
            compose_project=_LLM_BAWT_PROJECT,
            compose_service="redis",
        ),
        "args_schema_json": _no_args_schema(),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 180,
        "start_delay_seconds": 10,
        "risk_level": RISK_CRITICAL,
        "category": "restart",
        "approval_prompt_prefix": "Restart Redis (KILLS ALL AGENTS)",
    },
]


# ---------------------------------------------------------------------------
# BawtHub restarts (rebuild-prod dropped — that's a `make` shell command, not
# a docker action; add back with a future SSH executor if needed).
# ---------------------------------------------------------------------------

_BAWTHUB_SEEDS: list[dict[str, Any]] = [
    # TASK-915: narrowly scoped voice controls. Seeds stay disabled until the
    # operator reviews/enables each one; no generic container action is exposed.
    *[
        {
            "slug": f"bawthub.{action}-moshi-{service}",
            "title": f"{action.title()} Moshi {service.upper()}",
            "description": (
                f"{action.title()} the dedicated Moshi {service.upper()} container for an explicitly "
                "consented GPU handoff. Voice admission and active calls must be checked separately."
            ),
            "enabled": False,
            "executor_kind": EXECUTOR_DOCKER,
            "target_host": "",
            "working_directory": None,
            "command_script": _spec(action=action, compose_project=_BAWTHUB_PROJECT, compose_service=service),
            "args_schema_json": _no_args_schema(),
            "args_defaults_json": _defaults(),
            "timeout_seconds": 120,
            "start_delay_seconds": 0,
            "risk_level": RISK_HIGH,
            "category": "gpu-handoff",
            "approval_prompt_prefix": f"{action.title()} Moshi {service.upper()} for a GPU handoff",
        }
        for action in ("stop", "start") for service in ("stt", "tts")
    ],
    {
        "slug": "bawthub.restart-backend",
        "title": "Restart bawthub backend",
        "description": "Restart the bawthub Python voice/agent backend container.",
        "enabled": False,
        "executor_kind": EXECUTOR_DOCKER,
        "target_host": "",
        "working_directory": None,
        "command_script": _spec(
            action="restart",
            compose_project=_BAWTHUB_PROJECT,
            compose_service="backend",
        ),
        "args_schema_json": _no_args_schema(),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 120,
        "start_delay_seconds": 0,
        "risk_level": RISK_MEDIUM,
        "category": "restart",
        "approval_prompt_prefix": "Restart bawthub backend",
    },
    {
        # TASK-1171: the TTS admin page's restart. It must be dispatched only
        # through the GPU handoff controller, under its fence. Seeded disabled;
        # enable it only once that fence is deployed (TASK-1173).
        "slug": "bawthub.restart-tts",
        "title": "Restart bawthub TTS",
        "description": (
            "Restart the Moshi TTS container. Only the GPU handoff controller can dispatch it, "
            "and only while voice owns the GPU."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_DOCKER,
        "target_host": "",
        "working_directory": None,
        "command_script": _spec(
            action="restart",
            compose_project=_BAWTHUB_PROJECT,
            compose_service="tts",
        ),
        "args_schema_json": _no_args_schema(),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 120,
        "start_delay_seconds": 0,
        "risk_level": RISK_HIGH,
        "category": "gpu-handoff",
        "approval_prompt_prefix": "Restart bawthub TTS",
    },
    {
        "slug": "bawthub.restart-frontend",
        "title": "Restart bawthub frontend (HMR)",
        "description": "Restart the bawthub Vite/Hono HMR container.",
        "enabled": False,
        "executor_kind": EXECUTOR_DOCKER,
        "target_host": "",
        "working_directory": None,
        "command_script": _spec(
            action="restart",
            compose_project=_BAWTHUB_PROJECT,
            compose_service="frontend",
            # TASK-1697: probe the Hono API through Vite's /api proxy, so a
            # dead API (not just a dead Vite) fails the job.
            health_url="http://127.0.0.1:5173/api/health", health_field="status", health_value="healthy",
            health_timeout_seconds=150,
        ),
        "args_schema_json": _no_args_schema(),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 240,
        "start_delay_seconds": 0,
        "risk_level": RISK_LOW,
        "category": "restart",
        "approval_prompt_prefix": "Restart bawthub frontend HMR",
    },
]


# ---------------------------------------------------------------------------
# BawtHub production image deploy / rollback (TASK-997).
#
# Fixed target, fixed GHCR repository, fixed release workflow. Arguments only
# NAME a release that the app verifies against GitHub before approval; the
# worker re-verifies the pulled digest + labels + /api/health. Both rows ship
# disabled and are covered by dedicated require_approval policies.
# ---------------------------------------------------------------------------

# TASK-1518: the Compose service the target is declared by. A deploy refuses
# when the running container differs from it; the reconcile op below repairs
# that at the same image. The project directory comes from the container's own
# Compose labels, so only file basenames live here. The renderer is the
# official Docker CLI image, pinned by digest (it ships the Compose plugin).
_PROD_COMPOSE_CONFIG = {
    "compose_files": ["docker-compose.yml", "docker-compose.image.yml"],
    "renderer_image": (
        "docker.io/library/docker@sha256:625d9431a9f54c5a2bc90f24f0e1c3d55b1349fd857dd85035f98c2c9acbdd4d"
    ),
    "image_variable": "BAWTHUB_FRONTEND_IMAGE",
}

# TASK-1801: the frontend's own health endpoint, in-container (the worker
# stays network-less). Data on the row, never wired into the executor.
_FRONTEND_PROBE = {"type": "http_exec", "port": 3000, "path": "/api/health"}

_PROD_TARGET = dict(
    container_name="bawthub-frontend-prod-1",
    compose_project=_BAWTHUB_PROJECT,
    compose_service="frontend-prod",
    image_repository="ghcr.io/bawthub/frontend",
    github_repository="bawthub/bawthub",
    workflow_path=".github/workflows/release-frontend.yml",
    canonical_branch="main",
    stop_grace_seconds=20,
    health_timeout_seconds=180,
    compose_config=_PROD_COMPOSE_CONFIG,
    health_probe=_FRONTEND_PROBE,
)

_IMAGE_SEEDS: list[dict[str, Any]] = [
    {
        "slug": "bawthub.deploy-prod-image",
        "title": "Deploy BawtHub production image",
        "description": (
            "Replace frontend-prod with a GitHub Actions release image pinned by digest. "
            "The run, receipt, tag and digest are verified before approval; the previous "
            "container is restored automatically if health/release verification fails."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_DOCKER,
        "target_host": "",
        "working_directory": None,
        "command_script": _spec(action="deploy_image", **_PROD_TARGET),
        "args_schema_json": _schema(
            {
                "release_run_id": {"type": "string", "pattern": "^[0-9a-f]{32}$"},
            },
            required=["release_run_id"],
        ),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 900,
        "start_delay_seconds": 0,
        "max_output_bytes": 65536,
        "max_concurrent": 1,
        "risk_level": RISK_HIGH,
        "category": "deploy",
        "approval_prompt_prefix": "Deploy BawtHub production image",
    },
    {
        "slug": "bawthub.rollback-prod-image",
        "title": "Roll back BawtHub production image",
        "description": (
            "Restore the last-known-good image recorded by a succeeded deploy job. "
            "Never pulls; refuses if the target no longer runs that deploy's image."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_DOCKER,
        "target_host": "",
        "working_directory": None,
        "command_script": _spec(action="rollback_image", **_PROD_TARGET),
        "args_schema_json": _schema(
            {"deploy_job_id": {"type": "string", "pattern": "^[0-9a-f]{32}$"}},
            required=["deploy_job_id"],
        ),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 600,
        "start_delay_seconds": 0,
        "max_output_bytes": 65536,
        "max_concurrent": 1,
        "risk_level": RISK_CRITICAL,
        "category": "deploy",
        "approval_prompt_prefix": "Roll back BawtHub production image",
    },
]


# TASK-1179: configuration-only repair; never rebuild/pull or change privileges.
_ROUTING_SEEDS: list[dict[str, Any]] = [{
    "slug": "bawthub.reconcile-prod-routing",
    "title": "Repair production frontend health routing",
    "description": (
        "Add /api/system to the existing production API router using the SAME pinned image. "
        "Approval binds the current container/configuration. No build, pull, environment, mount "
        "or permission changes. Verifies the hostname-routed detailed report and restores the "
        "previous container on verification failure. Brief frontend interruption."
    ),
    "enabled": False,
    "executor_kind": EXECUTOR_DOCKER,
    "target_host": "",
    "working_directory": None,
    "command_script": _spec(
        action="reconcile_frontend_routing", container_name="bawthub-frontend-prod-1",
        compose_project=_BAWTHUB_PROJECT, compose_service="frontend-prod",
        image_repository="ghcr.io/bawthub/frontend", router_name="frontprod-api",
        route_service="frontend-prod", hostname="app.bawthub.com",
        add_prefixes=["/api/system"], stop_grace_seconds=20, health_timeout_seconds=180,
        health_probe=_FRONTEND_PROBE,
    ),
    "args_schema_json": _schema({}),
    "args_defaults_json": _defaults(),
    "timeout_seconds": 900,
    "start_delay_seconds": 0,
    "max_output_bytes": 65536,
    "max_concurrent": 1,
    "risk_level": RISK_HIGH,
    "category": "configuration",
    "approval_prompt_prefix": "Repair production health routing with the same image (brief interruption)",
}]


# TASK-1518: recreate the target from its rendered Compose service at the SAME
# pinned image. Approval shows drift names, mounts and closed-enum values only.
_CONFIG_SEEDS: list[dict[str, Any]] = [{
    "slug": "bawthub.reconcile-prod-config",
    "title": "Reconcile production frontend configuration",
    "description": (
        "Recreate frontend-prod from its declared Compose service using the SAME pinned image. "
        "No build or pull. Approval binds the running container and the rendered configuration "
        "and shows changed names only, never values. Verifies health and release identity and "
        "restores the previous container on failure. Brief frontend interruption."
    ),
    "enabled": False,
    "executor_kind": EXECUTOR_DOCKER,
    "target_host": "",
    "working_directory": None,
    "command_script": _spec(
        action="reconcile_frontend_config", container_name="bawthub-frontend-prod-1",
        compose_project=_BAWTHUB_PROJECT, compose_service="frontend-prod",
        image_repository="ghcr.io/bawthub/frontend", compose_config=_PROD_COMPOSE_CONFIG,
        display_values={"BAWTHUB_AUTH_SESSIONS": ["on", "off"]},
        stat_paths=["/run/bawthub-service-token/token"],
        stop_grace_seconds=20, health_timeout_seconds=180, health_probe=_FRONTEND_PROBE,
    ),
    "args_schema_json": _schema({}),
    "args_defaults_json": _defaults(),
    "timeout_seconds": 900,
    "start_delay_seconds": 0,
    "max_output_bytes": 65536,
    "max_concurrent": 1,
    "risk_level": RISK_HIGH,
    "category": "configuration",
    "approval_prompt_prefix": "Reconcile production configuration with the same image (brief interruption)",
}, {
    # TASK-1474: the same for the llm-bawt app, whose image Compose builds.
    # The app serves the approval itself, so the worker waits start_delay
    # for the reply to drain; the ops reconciler records the outcome.
    "slug": "llm-bawt.reconcile-app-config",
    "title": "Reconcile llm-bawt app configuration",
    "description": (
        "Recreate the llm-bawt app from its declared Compose service using the SAME running image. "
        "No build or pull; the image is bound by ID, never by tag. Approval shows changed names, "
        "ports and dependencies only, never values. Dependencies must already be running and are "
        "never restarted. Refuses unless the bind-mounted source is exactly the published branch "
        "head. Verifies health and restores the previous container on failure. "
        "Cuts in-flight chat turns and MCP calls."
    ),
    "enabled": False,
    "executor_kind": EXECUTOR_DOCKER,
    "target_host": "",
    "working_directory": None,
    "command_script": _spec(
        action="reconcile_app_config", container_name="llm-bawt-app",
        compose_project=_LLM_BAWT_PROJECT, compose_service="app",
        compose_config={"renderer_image": _PROD_COMPOSE_CONFIG["renderer_image"]},
        display_values={}, stat_paths=[], stop_grace_seconds=10, health_timeout_seconds=180,
        health_url="http://127.0.0.1:8642/health", health_field="status", health_value="ok",
        # The app runs its source from bind mounts of the checkout, so the
        # source must be exactly the published branch head. Every container
        # mount is named here as source or data; anything else refuses.
        # Destinations are the product's own Compose files' container paths.
        source_repository="bawthub/llm-bawt", source_branch="master",
        source_mounts=["/app/src", "/app/server.sh"],
        data_mounts=["/app/.logs", "/app/models", "/app/storage", "/root/.config/llm-bawt",
                     "/root/.cache/huggingface", "/root/.cache/llm-bawt/models", "/var/lib/llm-bawt/media",
                     "/var/lib/llm-bawt-ops", "/var/run/docker.sock", "/home/bridge/dev/agent-skills",
                     "/run/bawthub-service-token", "/run/llm-bawt-bridge-tokens",
                     "/run/llm-bawt-crawl4ai-token"],
        # Every environment name the container carries beyond the product's
        # own runtime variables is app data, opaque to the approver: the
        # install's operator lists its names here (never values). Empty fails
        # closed: a reconcile refuses until each name is classified.
        env_names=[],
    ),
    "args_schema_json": _schema({}),
    "args_defaults_json": _defaults(),
    "timeout_seconds": 900,
    "start_delay_seconds": 5,
    "max_output_bytes": 65536,
    "max_concurrent": 1,
    "risk_level": RISK_HIGH,
    "category": "configuration",
    "approval_prompt_prefix": "Reconcile llm-bawt app configuration with the same image (cuts in-flight turns)",
}, {
    # TASK-1474 step 4: the BawtHub backend, also Compose-built, recreated at
    # the same image. Not a self-target (the app serves the approval), so no
    # start delay. Same refusals and trust block as the app row.
    "slug": "bawthub.reconcile-backend-config",
    "title": "Reconcile BawtHub backend configuration",
    "description": (
        "Recreate the BawtHub backend from its declared Compose service using the SAME running image. "
        "No build or pull; the image is bound by ID, never by tag. Approval shows changed names, "
        "ports and security options only, never environment values. Refuses unless the bind-mounted "
        "source is exactly the published branch head. Verifies health and restores the previous "
        "container on failure. Cuts active voice sessions and in-flight backend requests."
    ),
    "enabled": False,
    "executor_kind": EXECUTOR_DOCKER,
    "target_host": "",
    "working_directory": None,
    "command_script": _spec(
        action="reconcile_app_config", container_name="bawthub-backend-1",
        compose_project=_BAWTHUB_PROJECT, compose_service="backend",
        compose_config={"renderer_image": _PROD_COMPOSE_CONFIG["renderer_image"]},
        display_values={},
        # Mode/owner of the service token by fixed argv, never its contents.
        stat_paths=["/run/bawthub-service-token/token"],
        stop_grace_seconds=10, health_timeout_seconds=120,
        health_url="http://127.0.0.1:80/v1/health", health_field="ok", health_value=True,
        # Destinations are the product's own Compose file's container paths.
        source_repository="bawthub/bawthub", source_branch="main",
        source_mounts=["/app/bawthub"],
        data_mounts=["/app/services/moshi-server/configs", "/run/bawthub-service-token"],
        # The backend's own Compose environment names (never values): data it
        # reads (URLs, keys, ids, voice names); none is executed.
        env_names=["AZURE_SPEECH_ENDPOINT", "AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "AZURE_VOICE_FILTER",
                   "BAWTHUB_LLM_MODEL", "BAWTHUB_LLM_URL",
                   "BAWTHUB_SERVICE_TOKEN_FILE", "BAWTHUB_STT_URL", "BAWTHUB_TTS_DEFAULT_PROVIDER",
                   "BAWTHUB_TTS_URL", "DATABASE_URL", "GROK_TTS_DEFAULT_VOICE",
                   "GROK_TTS_LANGUAGE", "GROK_TTS_LATENCY_MODE", "GROK_TTS_TEXT_NORMALIZATION",
                   "HOME_ASSISTANT_TOKEN", "HOME_ASSISTANT_URL", "HOME_ASSISTANT_WEATHER_ENTITY_ID",
                   "XAI_API_KEY"],
    ),
    "args_schema_json": _schema({}),
    "args_defaults_json": _defaults(),
    "timeout_seconds": 900,
    "start_delay_seconds": 0,
    "max_output_bytes": 65536,
    "max_concurrent": 1,
    "risk_level": RISK_HIGH,
    "category": "configuration",
    "approval_prompt_prefix": "Reconcile BawtHub backend configuration with the same image (cuts voice sessions)",
}, {
    # TASK-1649: the BawtHub hot-reload DEV frontend, Compose-built and
    # recreated at the same image. It serves its live working tree, so there is
    # no source guard; it is never a production target.
    "slug": "bawthub.reconcile-dev-frontend-config",
    "title": "Reconcile BawtHub dev frontend configuration",
    "description": (
        "Recreate the BawtHub hot-reload dev frontend from its declared Compose service using the SAME "
        "running image. No build or pull; the image is bound by ID, never by tag. Approval shows changed "
        "names, mounts, ports and the memory cap only, never environment values. Verifies health and "
        "restores the previous container on failure. Drops open dev-server connections; production is "
        "untouched."
    ),
    "enabled": False,
    "executor_kind": EXECUTOR_DOCKER,
    "target_host": "",
    "working_directory": None,
    "command_script": _spec(
        action="reconcile_dev_config", container_name="bawthub-frontend-1",
        compose_project=_BAWTHUB_PROJECT, compose_service="frontend",
        compose_config={"renderer_image": _PROD_COMPOSE_CONFIG["renderer_image"]},
        display_values={},
        # Mode/owner of the service token by fixed argv, never its contents.
        stat_paths=["/run/bawthub-service-token/token"],
        stop_grace_seconds=10, health_timeout_seconds=180,
        health_url="http://127.0.0.1:5173/api/health", health_field="status", health_value="healthy",
    ),
    "args_schema_json": _schema({}),
    "args_defaults_json": _defaults(),
    "timeout_seconds": 900,
    "start_delay_seconds": 0,
    "max_output_bytes": 65536,
    "max_concurrent": 1,
    "risk_level": RISK_MEDIUM,
    "category": "configuration",
    "approval_prompt_prefix": "Reconcile the BawtHub dev frontend configuration with the same image (dev only)",
}, {
    # TASK-1476: Crawl4AI runs a public image by tag; recreated at the SAME
    # local image (never pulled). A cutover plan target.
    "slug": "llm-bawt.reconcile-crawl4ai-config",
    "title": "Reconcile Crawl4AI configuration",
    "description": (
        "Recreate Crawl4AI from its declared Compose service using the SAME local image the running "
        "container uses. No pull; a moved tag is reported, never followed. Approval shows changed names, "
        "mounts and dependencies only, never values. Verifies health and restores the previous container "
        "on failure. Cuts in-flight page fetches."
    ),
    "enabled": False,
    "executor_kind": EXECUTOR_DOCKER,
    "target_host": "",
    "working_directory": None,
    "command_script": _spec(
        action="reconcile_tagged_config", container_name="llm-bawt-crawl4ai",
        compose_project=_LLM_BAWT_PROJECT, compose_service="crawl4ai", image_tag="unclecode/crawl4ai:latest",
        compose_config={"renderer_image": _PROD_COMPOSE_CONFIG["renderer_image"]},
        display_values={}, stat_paths=[], stop_grace_seconds=10, health_timeout_seconds=180,
        health_url="http://127.0.0.1:11235/health", health_field="status", health_value="ok",
    ),
    "args_schema_json": _schema({}),
    "args_defaults_json": _defaults(),
    "timeout_seconds": 900,
    "start_delay_seconds": 0,
    "max_output_bytes": 65536,
    "max_concurrent": 1,
    "risk_level": RISK_HIGH,
    "category": "configuration",
    "approval_prompt_prefix": "Reconcile Crawl4AI configuration with the same image (cuts page fetches)",
}, *(
    # TASK-1476: the agent bridges, Compose-built from the shared bridge image
    # and recreated at the same image. Same refusals and trust block as the
    # app row. The source guard binds the bridge's OWN source; the agents it
    # hosts run code from the mounted workspace by design.
    {
        "slug": f"llm-bawt.reconcile-{bridge}-config",
        "title": f"Reconcile {bridge} configuration",
        "description": (
            f"Recreate {bridge} from its declared Compose service using the SAME running image. No build "
            "or pull; the image is bound by ID, never by tag. Approval shows changed names, mounts and "
            "dependencies only, never environment values. Refuses unless the bind-mounted source is exactly "
            "the published branch head. Verifies health and restores the previous container on failure. "
            "KILLS every agent session the bridge hosts."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_DOCKER,
        "target_host": "",
        "working_directory": None,
        "command_script": _spec(
            action="reconcile_app_config", container_name=f"llm-bawt-{bridge}",
            compose_project=_LLM_BAWT_PROJECT, compose_service=bridge,
            compose_config={"renderer_image": _PROD_COMPOSE_CONFIG["renderer_image"]},
            display_values={}, stat_paths=[], stop_grace_seconds=10, health_timeout_seconds=180,
            health_url=f"http://127.0.0.1:{port}/health", health_field="redis", health_value=True,
            source_repository="bawthub/llm-bawt", source_branch="master", source_mounts=["/app/src"],
            # Destinations are the product's own Compose files' container paths.
            data_mounts=data_mounts,
            # The bridge's own Compose environment names (never values): URLs,
            # keys, log levels, token FILE paths and workspace paths it reads.
            env_names=env_names,
        ),
        "args_schema_json": _schema({}),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 900,
        "start_delay_seconds": 0,
        "max_output_bytes": 65536,
        "max_concurrent": 1,
        "risk_level": RISK_CRITICAL,
        "category": "configuration",
        "approval_prompt_prefix": f"Reconcile {bridge} configuration with the same image (KILLS ACTIVE AGENTS)",
    }
    for bridge, port, data_mounts, env_names in (
        ("codex-bridge", 8682,
         ["/app/.logs", "/home/bridge/dev", "/home/bridge/.codex/config.toml", "/home/bridge/.codex/sessions",
          "/home/bridge/.ssh", "/opt/agent-skills/bin", "/run/llm-bawt-bridge-tokens"],
         ["BRIDGE_BROKER_TOKEN_FILE", "BRIDGE_GIT_CREDENTIAL_TOKEN_FILE", "CODEX_BACKEND_NAME",
          "CODEX_BAWTHUB_MCP_URL", "CODEX_BRIDGE_CWD", "CODEX_BRIDGE_HEALTH_PORT", "CODEX_BRIDGE_LOG_LEVEL",
          "CODEX_BRIDGE_REQUEST_TIMEOUT", "CODEX_CRAWL4AI_MCP_URL", "CODEX_DEV_ROOT", "CODEX_HOME",
          "CODEX_LOCAL_PLUGINS_ENABLED", "CODEX_LOCAL_PLUGINS_SRC", "CODEX_MODEL", "CRAWL4AI_API_TOKEN_FILE",
          "LLM_BAWT_API_URL", "LLM_BAWT_SERVICE_HOST", "REDIS_URL"]),
        ("claude-code-bridge", 8681,
         ["/app/.logs", "/home/bridge/dev", "/home/bridge/.claude", "/home/bridge/.claude.json",
          "/home/bridge/.ssh", "/etc/ssh/ssh_config", "/opt/agent-skills/bin", "/run/llm-bawt-bridge-tokens"],
         ["ANTHROPIC_API_KEY", "BRIDGE_BROKER_TOKEN_FILE", "BRIDGE_GIT_CREDENTIAL_TOKEN_FILE",
          "CLAUDE_CODE_BAWTHUB_MCP_URL", "CLAUDE_CODE_BRIDGE_LOG_LEVEL", "CLAUDE_CODE_CRAWL4AI_MCP_URL",
          "CLAUDE_CODE_CWD", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_PERMISSION_MODE",
          "CLAUDE_CODE_PLAYWRIGHT_MCP_URL", "CLAUDE_CODE_SKILLS_PATH",
          "CRAWL4AI_API_TOKEN_FILE", "ENABLE_TOOL_SEARCH", "LLM_BAWT_API_URL", "LLM_BAWT_REDIS_URL",
          "LLM_BAWT_SERVICE_HOST", "LLM_BAWT_XAI_API_KEY", "LOCAL_ANTHROPIC_API_KEY", "LOCAL_ANTHROPIC_BASE_URL",
          "OPENAI_API_KEY", "REDIS_URL", "XAI_API_KEY"]),
    )
)]


# The env_import destinations a plan may accept losing (closed list).
_DESTINATIONS = sorted(DESTINATION_PROBES)

# ---------------------------------------------------------------------------
# TASK-1801: the optional CPU Pocket TTS worker, deployed from the image its CI
# publish workflow pushes (no release row, no build executor). Arguments name
# the publish run, its source SHA and the digest; the app verifies the run and
# its publish receipt before approval. The image is versioned ``sha-<sha>``.
# The pinned-hardened profile also compares the worker's caps, CPU, PIDs and
# read-only root, so a deploy can never quietly drop them. The target was
# created from the local-build overlay; ``adopt_files`` lets the first deploy
# (or the reconcile below) adopt it and relabel it with the image overlay.
# ---------------------------------------------------------------------------

_POCKET_COMPOSE_CONFIG = {
    "compose_files": ["docker-compose.yml", "docker-compose.pocket.yml", "docker-compose.pocket-image.yml"],
    "adopt_files": ["docker-compose.yml", "docker-compose.pocket.yml"],
    "renderer_image": _PROD_COMPOSE_CONFIG["renderer_image"],
    "image_variable": "BAWTHUB_POCKET_TTS_IMAGE",
    "profile": "pinned-hardened",
}

_POCKET_TARGET = dict(
    container_name="bawthub-pocket-tts-1",
    compose_project=_BAWTHUB_PROJECT,
    compose_service="pocket-tts",
    image_repository="ghcr.io/bawthub/pocket-tts",
    github_repository="bawthub/bawthub",
    workflow_path=".github/workflows/publish-tenant-voice.yml",
    canonical_branch="main",
    stop_grace_seconds=10,
    # The model loads at start (healthcheck start_period 180s).
    health_timeout_seconds=300,
    compose_config=_POCKET_COMPOSE_CONFIG,
    # Its own Docker healthcheck is the probe; identity is checked on the image.
    health_probe={"type": "docker_health"},
)

_POCKET_SEEDS: list[dict[str, Any]] = [
    {
        "slug": "bawthub.deploy-pocket-tts",
        "title": "Deploy the Pocket TTS worker image",
        "description": (
            "Replace the Pocket TTS worker with the image its CI publish run pushed, pinned by digest. "
            "The run, its publish receipt, the source SHA and the digest are verified before approval; "
            "the previous container is restored automatically if health or release verification fails. "
            "Interrupts Pocket voice synthesis while the model loads."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_DOCKER,
        "target_host": "",
        "working_directory": None,
        "command_script": _spec(action="deploy_image", binding="ci_publish",
                                trigger_workflow_path=".github/workflows/ci.yml", **_POCKET_TARGET),
        "args_schema_json": _schema(
            {
                "workflow_run_id": {"type": "string", "pattern": "^[0-9]{1,20}$"},
                "source_sha": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                "digest": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
            },
            required=["workflow_run_id", "source_sha", "digest"],
        ),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 1200,
        "start_delay_seconds": 0,
        "max_output_bytes": 65536,
        "max_concurrent": 1,
        "risk_level": RISK_HIGH,
        "category": "deploy",
        "approval_prompt_prefix": "Deploy Pocket TTS worker image",
    },
    {
        "slug": "bawthub.rollback-pocket-tts",
        "title": "Roll back the Pocket TTS worker image",
        "description": (
            "Restore the last-known-good image recorded by a succeeded Pocket TTS deploy job. "
            "Never pulls; refuses if the worker no longer runs that deploy's image."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_DOCKER,
        "target_host": "",
        "working_directory": None,
        "command_script": _spec(action="rollback_image", **_POCKET_TARGET),
        "args_schema_json": _schema(
            {"deploy_job_id": {"type": "string", "pattern": "^[0-9a-f]{32}$"}},
            required=["deploy_job_id"],
        ),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 900,
        "start_delay_seconds": 0,
        "max_output_bytes": 65536,
        "max_concurrent": 1,
        "risk_level": RISK_CRITICAL,
        "category": "deploy",
        "approval_prompt_prefix": "Roll back Pocket TTS worker image",
    },
    {
        "slug": "bawthub.reconcile-pocket-tts-config",
        "title": "Reconcile Pocket TTS worker configuration",
        "description": (
            "Recreate the Pocket TTS worker from its declared Compose service using the SAME pinned image. "
            "No build or pull. Approval shows changed names only, never values. Verifies health and "
            "release identity and restores the previous container on failure. Interrupts Pocket voice "
            "synthesis while the model loads."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_DOCKER,
        "target_host": "",
        "working_directory": None,
        "command_script": _spec(
            action="reconcile_frontend_config", container_name=_POCKET_TARGET["container_name"],
            compose_project=_BAWTHUB_PROJECT, compose_service="pocket-tts",
            image_repository=_POCKET_TARGET["image_repository"], compose_config=_POCKET_COMPOSE_CONFIG,
            display_values={}, stat_paths=[], stop_grace_seconds=10, health_timeout_seconds=300,
            health_probe=_POCKET_TARGET["health_probe"],
        ),
        "args_schema_json": _schema({}),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 1200,
        "start_delay_seconds": 0,
        "max_output_bytes": 65536,
        "max_concurrent": 1,
        "risk_level": RISK_HIGH,
        "category": "configuration",
        "approval_prompt_prefix": "Reconcile Pocket TTS worker configuration with the same image",
    },
]


# TASK-1476: the one-window cutover. Recreates each target in order, each
# only after the previous one is verified, then probes every credential the
# app had; any failure reverts every recreated target in reverse. Every
# target is its own catalog row and must itself be enabled.
_PLAN_SEEDS: list[dict[str, Any]] = [{
    "slug": "llm-bawt.cutover-plan",
    "title": "Cut over llm-bawt services to their declared configuration",
    "description": (
        "Recreate the app, Crawl4AI, the codex bridge and the claude-code bridge in that order, each at "
        "its SAME image and each only after the previous one is verified. Approval shows every target's "
        "own card plus how each stored credential is carried over (migrated, kept in env, unconfigured or "
        "an accepted loss); a credential the app would lose without a stored record refuses unless named "
        "in accept_loss. Probes run after the last target; any failure reverts every recreated target in "
        "reverse. KILLS every agent session and cuts in-flight chat turns."
    ),
    "enabled": False,
    "executor_kind": EXECUTOR_DOCKER,
    "target_host": "",
    "working_directory": None,
    "command_script": _spec(
        action="reconcile_service_plan",
        targets=["llm-bawt.reconcile-app-config", "llm-bawt.reconcile-crawl4ai-config",
                 "llm-bawt.reconcile-codex-bridge-config", "llm-bawt.reconcile-claude-code-bridge-config"],
        records_target="llm-bawt.reconcile-app-config",
        probes=[{"id": "crawl4ai_bearer", "target": "llm-bawt.reconcile-app-config"},
                {"id": "web_fetch", "target": "llm-bawt.reconcile-app-config"},
                {"id": "anthropic_models", "target": "llm-bawt.reconcile-app-config"},
                {"id": "broker", "target": "llm-bawt.reconcile-claude-code-bridge-config"}],
    ),
    "args_schema_json": _schema({
        "accept_loss": {"type": "array", "maxItems": len(_DESTINATIONS),
                        "items": {"type": "string", "enum": _DESTINATIONS}},
    }),
    "args_defaults_json": _defaults({"accept_loss": []}),
    "timeout_seconds": 1800,
    "start_delay_seconds": 5,
    "max_output_bytes": 65536,
    "max_concurrent": 1,
    "risk_level": RISK_CRITICAL,
    "category": "configuration",
    "approval_prompt_prefix": "Cut over llm-bawt services (KILLS ACTIVE AGENTS, cuts chat turns)",
    # Off until an operator enables it: holds every agent's turns first.
    "quiesce_enabled": False,
    "quiesce_timeout_seconds": 600,
    "quiesce_scope_json": json.dumps({"all_agents": True}),
}]


# ---------------------------------------------------------------------------
# One-command release: a durable coordinator, never a shell or local clone.
# A release approval delegates deploy only after build and target verification.
# ---------------------------------------------------------------------------

_RELEASE_SEEDS: list[dict[str, Any]] = [
    {
        "slug": "bawthub.release-prod",
        "title": "Build and deploy BawtHub production release",
        "description": (
            "Build a GitHub Actions release from an approved remote commit, recover safe partial "
            "attempts on the same run, and deploy its verified digest if the approved production "
            "baseline is unchanged. Rollback requires its own approval."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_RELEASE,
        "target_host": "",
        "working_directory": None,
        "command_script": _spec(
            action="release_orchestrate",
            github_repository="bawthub/bawthub",
            workflow_path=".github/workflows/release-frontend.yml",
            canonical_branch="main",
            image_repository="ghcr.io/bawthub/frontend",
            deploy_operation="bawthub.deploy-prod-image",
        ),
        "args_schema_json": _schema({
            "release_task": {"type": "string", "pattern": "^TASK-[0-9]+$"},
            "bump": {"type": "string", "enum": ["patch", "minor", "major"]},
        }, required=["release_task"]),
        "args_defaults_json": _defaults({"bump": "patch"}),
        "timeout_seconds": 86400,
        "start_delay_seconds": 0,
        "max_output_bytes": 65536,
        "max_concurrent": 1,
        "risk_level": RISK_HIGH,
        "category": "deploy",
        "approval_prompt_prefix": "Build and deploy verified BawtHub production release",
    },
]


# ---------------------------------------------------------------------------
# Standalone container restarts (not managed by a compose file the app owns).
# ---------------------------------------------------------------------------

_STANDALONE_CONTAINER_SEEDS: list[dict[str, Any]] = [
    {
        "slug": "container.restart",
        "title": "Restart an allowlisted container",
        "description": (
            "Restart a specific container by name. Only the allowlist below is "
            "runnable — every other name is rejected by the args schema."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_DOCKER,
        "target_host": "",
        "working_directory": None,
        "command_script": _spec(
            action="restart",
            container_name_from_arg="container",
        ),
        "args_schema_json": _schema(
            {
                "container": {
                    "type": "string",
                    # Only containers on the executor's own Docker host (echo).
                    # NginxProxyManager (Unraid) and the retired
                    # bawthub-public-site were dropped in TASK-1002: the
                    # Docker-only executor cannot reach them.
                    "enum": [
                        "bawthub-frontend-1",
                        "bawthub-nocodb-1",
                        "llm-bawt-local-model-bridge",
                    ],
                }
            },
            required=["container"],
        ),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 120,
        "start_delay_seconds": 0,
        "risk_level": RISK_MEDIUM,
        "category": "restart",
        "approval_prompt_prefix": "Restart container",
    },
]


# ---------------------------------------------------------------------------
# Public seed set
# ---------------------------------------------------------------------------

SEEDS: list[dict[str, Any]] = [
    *_LLM_BAWT_SEEDS,
    *_BAWTHUB_SEEDS,
    *_STANDALONE_CONTAINER_SEEDS,
    *_IMAGE_SEEDS,
    *_ROUTING_SEEDS,
    *_CONFIG_SEEDS,
    *_POCKET_SEEDS,
    *_PLAN_SEEDS,
    *_RELEASE_SEEDS,
]


def seed_all(store) -> tuple[list[str], list[str]]:
    """Insert every seed row that doesn't already exist.

    Returns ``(inserted_slugs, skipped_slugs)``. Preserves operator edits —
    any slug already present is skipped even if the seed dict differs.
    """
    inserted: list[str] = []
    skipped: list[str] = []
    for seed in SEEDS:
        row = store.seed_operation_if_missing(seed)
        if row is None:
            skipped.append(seed["slug"])
        else:
            inserted.append(seed["slug"])
    return inserted, skipped


__all__ = ["SEEDS", "seed_all"]
