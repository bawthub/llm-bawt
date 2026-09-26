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
                    "enum": [
                        "bawthub-frontend-1",
                        "NginxProxyManager",
                        "bawthub-nocodb-1",
                        "llm-bawt-local-model-bridge",
                        "bawthub-public-site",
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
