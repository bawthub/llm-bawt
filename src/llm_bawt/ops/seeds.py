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
            "Restart claude-code-bridge. HIGH "
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
                    "enum": ["claude-code-bridge"],
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
            "map": {"claude-code-bridge": "claude-code"},
        }, sort_keys=True),
    },
    {
        # TASK-2100: the operator's restart of one pool house (e.g. a stuck
        # one). Routine recycling needs no ops: a vacated house restarts
        # itself (TASK-2080). Recreate/image repin stays a deployment step.
        "slug": "llm-bawt.restart-house",
        "title": "Restart a house",
        "description": (
            "Restart one agent house (Compose service house-<id>); never a mansion bridge, volume or "
            "worktree. Runs only from an approval card. Refuses while the house has a tenant, a running "
            "turn or an unread send, checked when the card is prepared and again before dispatch, unless "
            "override_occupied is set (that kills the house's running turns). The house comes back on "
            "its own container with empty scratch, and the job fails unless its bridge answers healthy."
        ),
        "enabled": False,
        "executor_kind": EXECUTOR_DOCKER,
        "target_host": "",
        "working_directory": None,
        "command_script": _spec(
            action="restart",
            compose_project=_LLM_BAWT_PROJECT,
            compose_service_from_arg="house",
            health_url="http://127.0.0.1:8681/health", health_field="redis", health_value=True,
            health_timeout_seconds=120,
        ),
        "args_schema_json": _schema(
            {
                "house": {"type": "string", "pattern": r"^house-[a-z0-9][a-z0-9-]{0,30}$"},
                "override_occupied": {"type": "boolean"},
            },
            required=["house"],
        ),
        "args_defaults_json": _defaults({"override_occupied": False}),
        "timeout_seconds": 240,
        "start_delay_seconds": 0,
        "max_concurrent": 1,
        "risk_level": RISK_HIGH,
        "category": "restart",
        "approval_prompt_prefix": "Restart house",
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


# The official Docker CLI image, pinned by digest (it ships the Compose
# plugin); config reconciles render the declared Compose service with it.
_RENDERER_IMAGE = (
    "docker.io/library/docker@sha256:625d9431a9f54c5a2bc90f24f0e1c3d55b1349fd857dd85035f98c2c9acbdd4d"
)


# TASK-1518: recreate the target from its rendered Compose service at the SAME
# pinned image. Approval shows drift names, mounts and closed-enum values only.
_CONFIG_SEEDS: list[dict[str, Any]] = [{
    # TASK-1474: the same for the llm-bawt app, whose image Compose builds.
    # The app serves the approval itself, so the worker waits start_delay
    # for the reply to drain; the ops reconciler records the outcome.
    "slug": "llm-bawt.reconcile-app-config",
    "title": "Reconcile llm-bawt app configuration",
    "description": (
        "Recreate the llm-bawt app from its declared Compose service using the SAME running image. "
        "Always recreates to load the approved bind-mounted source, even when configuration matches. "
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
        compose_config={"renderer_image": _RENDERER_IMAGE},
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
                     "/run/bawthub-service-token", "/run/llm-bawt-bridge-tokens", "/run/llm-bawt-house-pool.json",
                     "/run/llm-bawt-crawl4ai-token",
                     # Read-only pin list the weekly harness check reads (TASK-1975); never executed.
                     "/app/docker/agent-bridge-constraints.txt"],
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
    # TASK-1844: the GPU local-model bridge, Compose-built from the CUDA image.
    # ``gpu: true`` selects the native-CDI GPU profile; the job reserves the
    # GPU handoff row from insert until its outcome is known.
    "slug": "llm-bawt.reconcile-local-model-bridge-config",
    "title": "Reconcile local-model-bridge configuration",
    "description": (
        "Recreate the GPU local-model bridge from its declared Compose service using the SAME running image. "
        "Always recreates to load the approved bind-mounted source. No build or pull; the image is bound by "
        "ID. Native CDI only: refuses unless the daemon offers every declared CDI device and runc, and the "
        "created container carries exactly the declared runtime, IPC, devices and security options. Holds "
        "the GPU handoff for the whole job; refuses while a handoff, video ownership or render is active. "
        "Verifies health and restores the previous container on failure. Unloads local models and cuts "
        "in-flight local inference and embeddings."
    ),
    "enabled": False,
    "executor_kind": EXECUTOR_DOCKER,
    "target_host": "",
    "working_directory": None,
    "command_script": _spec(
        action="reconcile_app_config", container_name="llm-bawt-local-model-bridge",
        compose_project=_LLM_BAWT_PROJECT, compose_service="local-model-bridge", gpu=True,
        compose_config={"renderer_image": _RENDERER_IMAGE},
        display_values={}, stat_paths=[], stop_grace_seconds=30, health_timeout_seconds=180,
        health_url="http://127.0.0.1:8683/health", health_field="redis", health_value=True,
        source_repository="bawthub/llm-bawt", source_branch="master", source_mounts=["/app/src"],
        # Destinations are the product's own Compose file's container paths.
        data_mounts=["/app/.logs", "/app/models", "/root/.config/llm-bawt", "/root/.cache/huggingface",
                     "/root/.cache/llm-bawt/models"],
        # The service's own Compose environment names (never values).
        env_names=["FLASHINFER_CUDA_ARCH_LIST", "HF_HUB_DISABLE_PROGRESS_BARS", "HF_HUB_DISABLE_XET", "HF_TOKEN",
                   "LLAMA_CPP_N_GPU_LAYERS", "LLM_BAWT_API_URL", "LLM_BAWT_DEFAULT_MODEL_ALIAS",
                   "LLM_BAWT_DEFAULT_USER", "LLM_BAWT_LLAMA_CPP_FLASH_ATTN", "LLM_BAWT_LLAMA_CPP_N_BATCH",
                   "LLM_BAWT_LLAMA_CPP_N_CTX", "LLM_BAWT_LLAMA_CPP_N_GPU_LAYERS", "LLM_BAWT_LOG_DIR",
                   "LLM_BAWT_LOG_PREFIX", "LLM_BAWT_MODEL_CACHE_DIR", "LLM_BAWT_POSTGRES_DATABASE",
                   "LLM_BAWT_POSTGRES_HOST", "LLM_BAWT_POSTGRES_PASSWORD", "LLM_BAWT_POSTGRES_PORT",
                   "LLM_BAWT_POSTGRES_USER", "LLM_BAWT_REDIS_URL", "LLM_BAWT_SERVICE_HOST",
                   "LOCAL_MODEL_BACKEND_NAME", "LOCAL_MODEL_BRIDGE_EMBED_PORT", "LOCAL_MODEL_BRIDGE_HEALTH_PORT",
                   "LOCAL_MODEL_BRIDGE_LOG_LEVEL", "LOCAL_MODEL_BRIDGE_REQUEST_TIMEOUT", "LOCAL_MODEL_EMBED_DEVICE",
                   "LOCAL_MODEL_VIDEO_PORT", "POSTGRES_DATABASE", "POSTGRES_HOST", "POSTGRES_PASSWORD",
                   "POSTGRES_PORT", "POSTGRES_USER", "PYTORCH_CUDA_ALLOC_CONF", "REDIS_URL",
                   "TOKENIZERS_PARALLELISM", "TRANSFORMERS_NO_ADVISORY_WARNINGS", "TRANSFORMERS_VERBOSITY",
                   "VLLM_USE_FLASHINFER_SAMPLER"],
    ),
    "args_schema_json": _schema({}),
    "args_defaults_json": _defaults(),
    "timeout_seconds": 900,
    "start_delay_seconds": 0,
    "max_output_bytes": 65536,
    "max_concurrent": 1,
    "risk_level": RISK_HIGH,
    "category": "configuration",
    "approval_prompt_prefix": "Reconcile local-model-bridge configuration with the same image (unloads local models)",
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
        compose_config={"renderer_image": _RENDERER_IMAGE},
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
}, {
    # TASK-1831: Redis carries the agent bridge transport and runs a public
    # tagged image with no HTTP endpoint, so its own Docker healthcheck is the
    # probe. Shared infrastructure: the Compose source guard binds the rendered
    # service to the published branch head and a clean tree.
    "slug": "llm-bawt.reconcile-redis-config",
    "title": "Reconcile Redis configuration",
    "description": (
        "Recreate Redis from its declared Compose service using the SAME local image the running "
        "container uses. No pull; a moved tag is reported, never followed. Refuses unless the checkout "
        "is clean and exactly the published branch head. Approval shows changed names, ports and mounts "
        "only, never values. Verifies the container's own Docker healthcheck and restores the previous "
        "container on failure. The data volume is kept. Interrupts every bridge's Redis connection: "
        "in-flight agent turns may fail."
    ),
    "enabled": False,
    "executor_kind": EXECUTOR_DOCKER,
    "target_host": "",
    "working_directory": None,
    "command_script": _spec(
        action="reconcile_tagged_config", container_name="llm-bawt-redis",
        compose_project=_LLM_BAWT_PROJECT, compose_service="redis", image_tag="redis:7-alpine",
        compose_config={"renderer_image": _RENDERER_IMAGE},
        display_values={}, stat_paths=[], stop_grace_seconds=10, health_timeout_seconds=120,
        health_probe={"type": "docker_health"},
        source_repository="bawthub/llm-bawt", source_branch="master",
    ),
    "args_schema_json": _schema({}),
    "args_defaults_json": _defaults(),
    "timeout_seconds": 900,
    "start_delay_seconds": 0,
    "max_output_bytes": 65536,
    "max_concurrent": 1,
    "risk_level": RISK_CRITICAL,
    "category": "configuration",
    "approval_prompt_prefix": "Reconcile Redis configuration with the same image (interrupts the bridge transport)",
}, *(
    # TASK-1476: the agent bridges, Compose-built from the shared bridge image
    # and recreated at the same image. Same refusals and trust block as the
    # app row. The source guard binds the bridge's OWN source; the agents it
    # hosts run code from the mounted workspace by design.
    {
        "slug": f"llm-bawt.reconcile-{bridge}-config",
        "title": f"Reconcile {bridge} configuration",
        "description": (
            f"Recreate {bridge} from its declared Compose service using the SAME running image. "
            "Always recreates to load the approved bind-mounted source, even when configuration matches. No build "
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
            compose_config={"renderer_image": _RENDERER_IMAGE},
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
        ("claude-code-bridge", 8681,
         ["/app/.logs", "/home/bridge/dev", "/home/bridge/.claude", "/home/bridge/.claude.json",
          "/home/bridge/.ssh", "/etc/ssh/ssh_config", "/opt/agent-skills/bin", "/run/llm-bawt-bridge-tokens"],
         ["ANTHROPIC_API_KEY", "BAWT_WT_DEV_ROOT", "BRIDGE_BROKER_TOKEN_FILE", "BRIDGE_GIT_CREDENTIAL_TOKEN_FILE",
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

# TASK-1476: the one-window cutover. Recreates each target in order, each
# only after the previous one is verified, then probes every credential the
# app had; any failure reverts every recreated target in reverse. Every
# target is its own catalog row and must itself be enabled.
_PLAN_SEEDS: list[dict[str, Any]] = [{
    "slug": "llm-bawt.cutover-plan",
    "title": "Cut over llm-bawt services to their declared configuration",
    "description": (
        "Recreate the app, Crawl4AI and the claude-code bridge in that order, each at "
        "its SAME image and each only after the previous one is verified. Approval shows every target's "
        "own card plus how each stored credential is carried over (migrated, kept in env, unconfigured or "
        "an accepted loss). Source-mounted app/bridge targets always recreate even when configuration matches; "
        "a credential the app would lose without a stored record refuses unless named "
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
                 "llm-bawt.reconcile-claude-code-bridge-config"],
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


# TASK-1263: one approval and worker for a bounded, caller-ordered sequence.
# These existing rows all accept no arguments. Each child must independently
# be enabled and pass its normal admission checks; this seed enables nothing.
_SEQUENCE_OPERATIONS = [
    "llm-bawt.restart-app",
    "bawthub.restart-frontend",
    "llm-bawt.reconcile-app-config",
    "llm-bawt.reconcile-crawl4ai-config",
    "llm-bawt.reconcile-claude-code-bridge-config",
]

_SEQUENCE_SEEDS: list[dict[str, Any]] = [{
    "slug": "llm-bawt.operation-sequence",
    "title": "Run an ordered service operation sequence",
    "description": (
        "Run one to six allowlisted service operations in the requested order, each only after "
        "the previous target is healthy. One approval and terminal callback; all-agent quiesce "
        "is held for the entire window. Stops on the first failure without undoing earlier "
        "successful steps. An interrupted mutation requires inspection, never automatic retry. "
        "Every child must already be enabled; duplicate containers, nested plans, executor/GPU "
        "targets and houses are refused. May kill agent sessions and cut in-flight chat turns."
    ),
    "enabled": False,
    "executor_kind": EXECUTOR_DOCKER,
    "target_host": "",
    "working_directory": None,
    "command_script": _spec(action="operation_sequence", operations=_SEQUENCE_OPERATIONS),
    "args_schema_json": _schema({
        "steps": {
            "type": "array", "minItems": 1, "maxItems": 6,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["operation", "args"],
                "properties": {
                    "operation": {"type": "string", "enum": _SEQUENCE_OPERATIONS},
                    "args": {"type": "object", "additionalProperties": False, "properties": {}},
                },
            },
        },
    }, required=["steps"]),
    "args_defaults_json": _defaults(),
    # Covers up to six seeded children's delay + execution + health budgets.
    "timeout_seconds": 5400,
    "start_delay_seconds": 5,
    "max_output_bytes": 65536,
    "max_concurrent": 1,
    "risk_level": RISK_CRITICAL,
    "category": "sequence",
    "approval_prompt_prefix": "Run ordered service operations (KILLS ACTIVE AGENTS, cuts chat turns)",
    # Mandatory whenever this disabled row is explicitly enabled by an operator.
    "quiesce_enabled": True,
    "quiesce_timeout_seconds": 600,
    "quiesce_scope_json": json.dumps({"all_agents": True}),
}]


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
                    # Only shipped containers on the executor's own Docker
                    # host. An install allowlists its extra containers by
                    # editing this row in its own catalog.
                    "enum": [
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
    *_CONFIG_SEEDS,
    *_PLAN_SEEDS,
    *_SEQUENCE_SEEDS,
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
