"""First-run tenant setup surface (TASK-355, agent-first since TASK-1293/1297).

The wizard uses the same code paths admins do; these routes only add what
the admin surface has no equivalent for. Flow the wizard drives:

1. ``POST /v1/setup/seed``: idempotent infrastructure seed (catalog schema,
   standard access paths, prompt defaults). Must run before anything that
   reads the catalog on a fresh tenant. Also the installer / operator
   fallback (``tenant.sh seed``).
2. ``GET  /v1/models/catalog/access-paths?agent_backend=claude-code``: the
   admin list of routes a ``claude-code`` agent can run on, each with its
   provider ``connection``.
3. The admin Add Model flow (catalog ``PUT`` routes) adds the chosen model and
   endpoint. It is the only writer of model rows.
4. ``GET  /v1/setup/agent-runtime``: whether the ``claude-code`` bridge's
   command consumer answers a strict ping right now (:mod:`agent_bridge.ping`).
5. ``GET  /v1/setup/first-bot``: wizard defaults, including the starting
   prompt from the registry (``setup.first_agent_prompt``).
6. ``POST /v1/setup/first-bot``: refuse unless the runtime is ready, then
   create a ``claude-code`` agent on the chosen endpoint through the admin
   bot-profile writer, which resolves the endpoint and derives the harness.

Chat bots stay creatable from Settings; the wizard creates agents only.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from ...prompt_registry import get_prompt_resolver
from ...seeding import FIRST_BOT_AGENT_BACKEND, seed_tenant
from ..agent_runtime import check_agent_runtime
from ..dependencies import get_runtime_settings_store, get_service
from .settings import _persist_bot_profile

log = logging.getLogger(__name__)
router = APIRouter()

FIRST_AGENT_PROMPT_KEY = "setup.first_agent_prompt"

_DOMAIN_RE = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
_SLUG_RE = r"^[a-z0-9][a-z0-9_-]{1,63}$"


class TenantPreferences(BaseModel):
    exposure: str | None = Field(default=None, pattern=r"^(private|public)$")
    domains: list[str] | None = Field(default=None, max_length=10)
    voice_enabled: bool | None = None

    @field_validator("domains")
    @classmethod
    def validate_domains(cls, domains: list[str] | None) -> list[str] | None:
        if domains is None:
            return None
        normalized = [domain.strip().lower() for domain in domains if domain.strip()]
        invalid = [domain for domain in normalized if not _DOMAIN_RE.fullmatch(domain)]
        if invalid:
            raise ValueError("Invalid domains: " + ", ".join(invalid))
        return list(dict.fromkeys(normalized))


class TenantSeedRequest(TenantPreferences):
    """Infrastructure-only seed (installer / operator fallback)."""


class FirstBotRequest(TenantPreferences):
    """Create the tenant's first agent on an existing catalog endpoint.

    The harness is never accepted from the client: the bot-profile writer
    derives it from the endpoint's access path.
    """

    endpoint_id: int = Field(gt=0)
    slug: str = Field(pattern=_SLUG_RE)
    name: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=2000)
    system_prompt: str = Field(min_length=1)


def _store_setup_preferences(service, body: TenantPreferences | None) -> dict[str, object]:
    if body is None:
        return {}
    values = {
        "tenant.exposure": body.exposure,
        "tenant.domains": body.domains,
        "tenant.voice_enabled": body.voice_enabled,
    }
    selected = {key: value for key, value in values.items() if value is not None}
    if not selected:
        return {}
    store = get_runtime_settings_store(service.config)
    if store.engine is None:
        raise RuntimeError("Runtime settings DB unavailable")
    for key, value in selected.items():
        store.set_value("global", "*", key, value)
    return selected


def _first_agent_prompt(service) -> str:
    resolved = get_prompt_resolver(service.config).resolve(FIRST_AGENT_PROMPT_KEY)
    if resolved is None:  # the key is a built-in definition; absence is a code bug
        raise RuntimeError(f"Prompt '{FIRST_AGENT_PROMPT_KEY}' is not registered")
    return resolved.body


def _http_error(exc: Exception, *, what: str) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, RuntimeError):
        log.warning("%s unavailable: %s", what, exc)
        return HTTPException(status_code=503, detail=str(exc))
    log.exception("%s failed", what)
    return HTTPException(status_code=500, detail=f"{what} failed")


@router.post("/v1/setup/seed", tags=["Setup"])
async def apply_tenant_seed(body: TenantSeedRequest | None = None):
    """Apply the idempotent infrastructure seed and return the report."""
    service = get_service()
    try:
        report = await run_in_threadpool(seed_tenant, service.config)
        preferences = await run_in_threadpool(_store_setup_preferences, service, body)
    except Exception as exc:  # noqa: BLE001 — mapped to HTTP below
        raise _http_error(exc, what="Tenant seed") from exc
    return {"ok": True, "report": report, "preferences": preferences}


@router.get("/v1/setup/agent-runtime", tags=["Setup"])
async def get_agent_runtime():
    """Whether the agent bridge answers a strict ping now. Always 200; see ``ready``."""
    service = get_service()
    return (await check_agent_runtime(service.config)).to_dict()


@router.get("/v1/setup/first-bot", tags=["Setup"])
async def get_first_bot_defaults():
    """What the wizard pre-fills for the first agent."""
    service = get_service()
    try:
        prompt = await run_in_threadpool(_first_agent_prompt, service)
    except Exception as exc:  # noqa: BLE001 — mapped to HTTP below
        raise _http_error(exc, what="First bot defaults") from exc
    return {
        "default_system_prompt": prompt,
        "bot_type": "agent",
        "agent_backend": FIRST_BOT_AGENT_BACKEND,
    }


@router.post("/v1/setup/first-bot", tags=["Setup"])
async def create_first_bot(body: FirstBotRequest):
    """Create the tenant's first agent.

    Refuses with 503 unless the agent runtime answers a ping. Then the admin
    bot-profile writer validates and creates the bot: 409 if the slug exists,
    422 if the endpoint is unknown or can't run a ``claude-code`` agent.
    Preferences are stored last.
    """
    service = get_service()
    try:
        runtime = await check_agent_runtime(service.config)
        if not runtime.ready:
            raise HTTPException(status_code=503, detail=f"Agent runtime not ready: {runtime.detail}")
        bot = await _persist_bot_profile(
            {
                "slug": body.slug,
                "name": body.name,
                "description": body.description,
                "system_prompt": body.system_prompt,
                "requires_memory": True,
                "voice_optimized": False,
                "tts_mode": False,
                "include_summaries": True,
                "include_in_global_search": True,
                "uses_tools": True,
                "uses_search": True,
                "uses_home_assistant": False,
                "bot_type": "agent",
                "agent_backend": FIRST_BOT_AGENT_BACKEND,
                # The writer heals this to ``claude-proxy`` for non-Anthropic endpoints.
                "harness": FIRST_BOT_AGENT_BACKEND,
                "endpoint_id": body.endpoint_id,
            },
            create_only=True,
        )
        preferences = await run_in_threadpool(_store_setup_preferences, service, body)
    except Exception as exc:  # noqa: BLE001 — mapped to HTTP below
        raise _http_error(exc, what="First bot setup") from exc

    return {
        "ok": True,
        "bot": bot,
        "runtime": runtime.to_dict(),
        "preferences": preferences,
    }
