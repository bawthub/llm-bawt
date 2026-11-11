"""Install-specific ops catalog rows, kept as test fixtures (TASK-2194).

Product seeds (:mod:`llm_bawt.ops.seeds`) ship only rows for the shipped
Compose stack. These rows target containers that exist only in one
installation's extra Compose services (a production image frontend, a dev
backend/frontend, an optional TTS worker and its release). That install keeps
them as rows in its own ops database; ``seed_all`` is insert-only, so moving
the definitions here changes nothing there. The executor tests still exercise
these action shapes, so the definitions live on as fixtures.
"""

from __future__ import annotations

from typing import Any

from llm_bawt.ops import seeds as _product
from llm_bawt.ops.models import EXECUTOR_DOCKER, EXECUTOR_RELEASE, RISK_CRITICAL, RISK_HIGH, RISK_MEDIUM
from llm_bawt.ops.seeds import _BAWTHUB_PROJECT, _RENDERER_IMAGE, _defaults, _schema, _spec

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
    "renderer_image": _RENDERER_IMAGE,
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
    # TASK-1535: start the new release unrouted, hand over the network
    # aliases once healthy, then stop the old one; no serving gap.
    cutover="overlap",
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


# Install-only reconcile rows split out of the product _CONFIG_SEEDS list.
_CONFIG_ROWS: list[dict[str, Any]] = [{
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
    # TASK-1474 step 4: the BawtHub backend, also Compose-built, recreated at
    # the same image. Not a self-target (the app serves the approval), so no
    # start delay. Same refusals and trust block as the app row.
    "slug": "bawthub.reconcile-backend-config",
    "title": "Reconcile BawtHub backend configuration",
    "description": (
        "Recreate the BawtHub backend from its declared Compose service using the SAME running image. "
        "Always recreates to load the approved bind-mounted source, even when configuration matches. "
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
        compose_config={"renderer_image": _RENDERER_IMAGE},
        display_values={},
        # Mode/owner of the service token by fixed argv, never its contents.
        stat_paths=["/run/bawthub-service-token/token"],
        stop_grace_seconds=10, health_timeout_seconds=120,
        health_url="http://127.0.0.1:80/v1/health", health_field="ok", health_value=True,
        # Destinations are the product's own Compose file's container paths.
        source_repository="bawthub/bawthub", source_branch="main",
        source_mounts=["/app/bawthub"],
        # TASK-1709: /voice-library is the read-only local voice library
        # (BawtHub TASK-2133); voice files, never code.
        data_mounts=["/app/services/moshi-server/configs", "/run/bawthub-service-token", "/voice-library"],
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
        compose_config={"renderer_image": _RENDERER_IMAGE},
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
}]


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
    "renderer_image": _RENDERER_IMAGE,
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


INSTALL_ROWS: list[dict[str, Any]] = [
    *_IMAGE_SEEDS,
    *_ROUTING_SEEDS,
    *_CONFIG_ROWS,
    *_POCKET_SEEDS,
    *_RELEASE_SEEDS,
]

#: Product seeds plus the install rows, as that installation's catalog holds them.
SEEDS: list[dict[str, Any]] = [*_product.SEEDS, *INSTALL_ROWS]


def seed_all(store) -> tuple[list[str], list[str]]:
    """``llm_bawt.ops.seeds.seed_all`` over :data:`SEEDS`."""
    inserted: list[str] = []
    skipped: list[str] = []
    for seed in SEEDS:
        (skipped if store.seed_operation_if_missing(seed) is None else inserted).append(seed["slug"])
    return inserted, skipped
