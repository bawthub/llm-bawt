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
        ),
        "args_schema_json": _no_args_schema(),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 120,
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
            "Restart claude-code-bridge, openclaw-bridge, or codex-bridge. HIGH "
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
                    "enum": ["claude-code-bridge", "openclaw-bridge", "codex-bridge"],
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
        ),
        "args_schema_json": _no_args_schema(),
        "args_defaults_json": _defaults(),
        "timeout_seconds": 120,
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


# ---------------------------------------------------------------------------
# One-command release: a durable coordinator, never a shell or local clone.
# A separate human approval is required after the build before deploy.
# ---------------------------------------------------------------------------

_RELEASE_SEEDS: list[dict[str, Any]] = [
    {
        "slug": "bawthub.release-prod",
        "title": "Build and approve BawtHub production release",
        "description": (
            "Build a GitHub Actions release from remote branch heads, recover partial "
            "attempts on the same run, then ask for a separate production deploy "
            "approval. No local checkout, shell command, or automatic deploy."
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
        "approval_prompt_prefix": "Build BawtHub production release (deploy requires separate approval)",
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
