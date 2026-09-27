"""Provider-connect routes — /v1/providers.

Drives the reusable "connect a provider" flow (device OAuth / api key) used by
the first-run wizard and settings UI. Phase 1 ships GitHub (device flow) + the
internal git-credential endpoint that feeds the tenant bridge's git helper.

Adapter network calls are sync (httpx); we run them in a threadpool so the async
event loop isn't blocked.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from ..dependencies import get_service
from ..providers.base import AUTH_API_KEY, AUTH_CLI_OAUTH, AUTH_DEVICE_OAUTH
from ..providers.github import GitHubAdapter, GitHubConfigError
from ..providers.registry import all_adapters, get_adapter

logger = logging.getLogger(__name__)

router = APIRouter()


class PollRequest(BaseModel):
    device_code: str
    user_code: str = ""


class CliLoginCompleteRequest(BaseModel):
    session_id: str
    code: str


class SelectReposRequest(BaseModel):
    installation_id: int
    repos: list[str] | None = None


class ApiKeyRequest(BaseModel):
    api_key: str


def _adapter_or_404(provider_id: str):
    service = get_service()
    adapter = get_adapter(service.config, provider_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{provider_id}'")
    return adapter


@router.get("/v1/providers")
async def list_providers():
    service = get_service()
    return {"providers": [a.descriptor() for a in all_adapters(service.config)]}


@router.get("/v1/providers/health")
async def providers_health():
    """Cheap aggregate credential health for UI polling (TASK-637).

    No secrets, no PTY, no upstream calls — file/DB reads only. Registered
    BEFORE the ``{provider_id}`` route so "health" isn't captured as an id.
    """
    service = get_service()

    def _collect() -> list[dict]:
        out = []
        for a in all_adapters(service.config):
            try:
                h = a.health()
            except Exception as e:  # noqa: BLE001 — one bad adapter must not hide the rest
                logger.warning("health probe failed for %s: %s", a.id, e)
                h = {"state": "unknown", "detail": str(e)[:200]}
            out.append({"id": a.id, "label": a.label, **h})
        return out

    return {"providers": await run_in_threadpool(_collect)}


@router.get("/v1/providers/{provider_id}")
async def get_provider(provider_id: str):
    return _adapter_or_404(provider_id).descriptor()


@router.post("/v1/providers/{provider_id}/connect/start")
async def connect_start(provider_id: str):
    adapter = _adapter_or_404(provider_id)
    if not adapter.supports(AUTH_DEVICE_OAUTH):
        raise HTTPException(status_code=400, detail=f"{provider_id} has no device flow")
    try:
        start = await run_in_threadpool(adapter.start_device_flow)
    except GitHubConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.warning("device flow start failed for %s: %s", provider_id, e)
        raise HTTPException(status_code=502, detail="provider error")
    return {
        "user_code": start.user_code,
        "verification_uri": start.verification_uri,
        "device_code": start.device_code,
        "interval": start.interval,
        "expires_in": start.expires_in,
    }


@router.post("/v1/providers/{provider_id}/connect/poll")
async def connect_poll(provider_id: str, body: PollRequest):
    adapter = _adapter_or_404(provider_id)
    if not adapter.supports(AUTH_DEVICE_OAUTH):
        raise HTTPException(status_code=400, detail=f"{provider_id} has no device flow")
    try:
        result = await run_in_threadpool(adapter.poll_device_flow, body.device_code, body.user_code)
    except GitHubConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.warning("device flow poll failed for %s: %s", provider_id, e)
        raise HTTPException(status_code=502, detail="provider error")
    out = {"status": result.status, "detail": result.detail}
    if result.record is not None:
        out["connection"] = result.record.public()
    return out


# --- CLI-driven OAuth (e.g. Claude subscription login) ----------------------
@router.post("/v1/providers/{provider_id}/login/start")
async def cli_login_start(provider_id: str):
    """Spawn the provider's login CLI under a PTY and return the authorize URL.

    The live process is held server-side keyed by ``session_id`` until the user
    submits the pasted code via ``/login/complete``.
    """
    adapter = _adapter_or_404(provider_id)
    if not adapter.supports(AUTH_CLI_OAUTH):
        raise HTTPException(status_code=400, detail=f"{provider_id} has no cli login")
    try:
        start = await run_in_threadpool(adapter.start_cli_login)
    except Exception as e:  # noqa: BLE001
        logger.warning("cli login start failed for %s: %s", provider_id, e)
        raise HTTPException(status_code=502, detail="login error")
    return {
        "session_id": start.session_id,
        "verification_uri": start.verification_uri,
        "instructions": start.instructions,
    }


@router.post("/v1/providers/{provider_id}/login/complete")
async def cli_login_complete(provider_id: str, body: CliLoginCompleteRequest):
    """Submit the pasted code into the live login session; persist on success."""
    adapter = _adapter_or_404(provider_id)
    if not adapter.supports(AUTH_CLI_OAUTH):
        raise HTTPException(status_code=400, detail=f"{provider_id} has no cli login")
    result = await run_in_threadpool(
        adapter.complete_cli_login, body.session_id, body.code
    )
    if result.status != "connected":
        raise HTTPException(status_code=400, detail=result.detail or "login failed")
    out = {"status": result.status, "detail": result.detail}
    if result.record is not None:
        out["connection"] = result.record.public()
    return out


# --- API-key connect (openrouter, any future AUTH_API_KEY adapter) ----------
@router.post("/v1/providers/{provider_id}/connect/api-key")
async def connect_api_key(provider_id: str, body: ApiKeyRequest):
    """Validate and persist an API key for a key-auth provider.

    The adapter probes the key upstream before saving; a bad key never lands
    in the CredentialStore. The key is encrypted at rest and never echoed back.
    """
    adapter = _adapter_or_404(provider_id)
    if not adapter.supports(AUTH_API_KEY):
        raise HTTPException(status_code=400, detail=f"{provider_id} has no api-key auth")
    try:
        result = await run_in_threadpool(adapter.set_api_key, body.api_key)
    except Exception as e:  # noqa: BLE001
        logger.warning("api-key connect failed for %s: %s", provider_id, e)
        raise HTTPException(status_code=502, detail="provider error")
    if not result.ok:
        raise HTTPException(status_code=400, detail=result.detail or "invalid API key")
    record = adapter.store.load(provider_id)
    return {
        "status": "connected",
        "account": result.account,
        "connection": record.public() if record else None,
    }


# --- OpenRouter key broker (TASK-822) ----------------------------------------
@router.get("/v1/providers/openrouter/token")
async def openrouter_api_key(request: Request):
    """Internal: hand a reader the stored OpenRouter API key.

    The claude-code bridge proxy adapter calls this instead of reading env
    vars — the DB-backed CredentialStore is the sole source of truth. Static
    key, so no refresh chain; ``expires_at`` is always null. Same trust model
    as the Claude/ChatGPT brokers above: internal network only, optional
    ``X-Bridge-Token`` defense in depth.
    """
    expected = os.getenv("BRIDGE_CLAUDE_TOKEN_SECRET")
    if expected and request.headers.get("X-Bridge-Token") != expected:
        raise HTTPException(status_code=401, detail="bad bridge token")

    adapter = _adapter_or_404("openrouter")
    key = await run_in_threadpool(adapter.api_key)
    if not key:
        raise HTTPException(status_code=503, detail="no OpenRouter credential installed")
    return {"access_token": key, "expires_at": None, "state": "ok"}


@router.delete("/v1/providers/{provider_id}")
async def disconnect(provider_id: str):
    adapter = _adapter_or_404(provider_id)
    ok = await run_in_threadpool(adapter.disconnect)
    return {"disconnected": ok}


# --- Claude token broker (TASK-635) -----------------------------------------
@router.get("/v1/providers/claude/token")
async def claude_access_token(
    request: Request,
    force: bool = False,
    rejected_token_sha256: str | None = Query(default=None, pattern=r"^[0-9a-f]{64}$"),
):
    """Internal: hand a reader the current app-owned Claude access token.

    The claude-code bridge reads this per native inference request. After a
    401 it supplies ``force=true`` and ``rejected_token_sha256`` so an already
    rotated token is reused. The app is the SOLE refresher of the rotate-on-use
    chain — readers never refresh. Same trust model as git credentials: internal
    network only; if ``BRIDGE_CLAUDE_TOKEN_SECRET`` is set we additionally
    require it via the ``X-Bridge-Token`` header for defense in depth.
    """
    from ..usage.claude_oauth import get_access_token  # noqa: PLC0415

    expected = os.getenv("BRIDGE_CLAUDE_TOKEN_SECRET")
    if expected and request.headers.get("X-Bridge-Token") != expected:
        raise HTTPException(status_code=401, detail="bad bridge token")

    result = await run_in_threadpool(
        get_access_token, force_refresh=force, rejected_token_sha256=rejected_token_sha256,
    )
    if result.token is None:
        raise HTTPException(status_code=503, detail="no Claude credential installed")
    return {
        "access_token": result.token,
        "expires_at": result.expires_at,
        "state": result.state,
    }


# --- ChatGPT/codex token broker (TASK-636 Phase 2) -----------------------
@router.get("/v1/providers/openai_chatgpt/token")
async def chatgpt_access_token(request: Request, force: bool = False):
    """Internal: hand a reader the current app-owned ChatGPT/codex access token.

    ``expires_at`` is the access-token JWT ``exp`` as Unix epoch **seconds**.
    Consumers must compare it directly with ``time.time()``; it is not epoch
    milliseconds. This matches :class:`CodexToken` and ``_bundle_expires_at``.

    The proxy adapter (claude-code-bridge) and the codex-bridge call this to
    resolve tokens without reading a bind-mounted file. The app is the SOLE
    refresher — readers never refresh. Same trust model as the Claude broker
    above: internal network only; ``X-Bridge-Token`` defense in depth.
    """
    from ..usage.codex_oauth import get_access_token  # noqa: PLC0415

    expected = os.getenv("BRIDGE_CLAUDE_TOKEN_SECRET")
    if expected and request.headers.get("X-Bridge-Token") != expected:
        raise HTTPException(status_code=401, detail="bad bridge token")

    result = await run_in_threadpool(get_access_token, force_refresh=force)
    if result.token is None:
        raise HTTPException(status_code=503, detail="no ChatGPT credential installed")
    return {
        "access_token": result.token,
        "account_id": result.account_id,
        "expires_at": result.expires_at,
        "state": result.state,
    }


@router.get("/v1/providers/openai_chatgpt/bundle")
async def chatgpt_auth_bundle(request: Request):
    """Internal: hand the codex-bridge the full OAuth bundle for materialization.

    The codex CLI binary reads ``~/.codex/auth.json`` directly (it cannot call
    a broker endpoint). The codex-bridge calls this at startup to materialize
    a container-local auth.json from the DB-backed bundle — no bind mount
    needed (TASK-636 Phase 2c).

    Same trust model as the token broker above. Returns the full bundle dict
    suitable for writing verbatim to auth.json.
    """
    from ..usage.codex_oauth import _load  # noqa: PLC0415

    expected = os.getenv("BRIDGE_CLAUDE_TOKEN_SECRET")
    if expected and request.headers.get("X-Bridge-Token") != expected:
        raise HTTPException(status_code=401, detail="bad bridge token")

    _, bundle = await run_in_threadpool(_load)
    if not bundle:
        raise HTTPException(status_code=503, detail="no ChatGPT credential installed")
    return bundle


# --- GitHub-specific --------------------------------------------------------
@router.get("/v1/providers/github/installations")
async def github_installations():
    service = get_service()
    adapter = get_adapter(service.config, "github")
    if not isinstance(adapter, GitHubAdapter):
        raise HTTPException(status_code=404, detail="github not available")
    return {"installations": await run_in_threadpool(adapter.list_installations)}


@router.post("/v1/providers/github/repos")
async def github_select_repos(body: SelectReposRequest):
    service = get_service()
    adapter = get_adapter(service.config, "github")
    if not isinstance(adapter, GitHubAdapter):
        raise HTTPException(status_code=404, detail="github not available")
    try:
        record = await run_in_threadpool(
            adapter.select_installation, body.installation_id, body.repos
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))
    return {"connection": record.public()}


@router.get("/v1/providers/github/git-credential")
async def github_git_credential(request: Request):
    """Internal: mint a fresh installation token in git-credential format.

    The tenant bridge configures ``credential.helper`` to curl this endpoint.
    Reachable only within the deployment's internal network: the app talks to
    other services by compose service name, and only the frontend (:3000) is
    forwarded to the LAN — the app port is not routed to the internet. If
    ``BRIDGE_GIT_CREDENTIAL_TOKEN`` is set, we additionally require it via the
    ``X-Bridge-Token`` header for defense in depth.
    """
    expected = os.getenv("BRIDGE_GIT_CREDENTIAL_TOKEN")
    if expected and request.headers.get("X-Bridge-Token") != expected:
        raise HTTPException(status_code=401, detail="bad bridge token")

    service = get_service()
    adapter = get_adapter(service.config, "github")
    if not isinstance(adapter, GitHubAdapter):
        raise HTTPException(status_code=404, detail="github not available")
    try:
        token = await run_in_threadpool(adapter.mint_installation_token)
    except GitHubConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.warning("git-credential mint failed: %s", e)
        raise HTTPException(status_code=502, detail="git-credential mint failed")

    # git credential helper protocol: key=value lines on stdout.
    body = f"username=x-access-token\npassword={token}\n"
    return Response(content=body, media_type="text/plain")


# --- Generic API-key broker (TASK-825) ---------------------------------------
# Registered LAST so the specific claude / openai_chatgpt / openrouter routes
# above always match first (FastAPI resolves in registration order).
@router.get("/v1/providers/{provider_id}/token")
async def api_key_token(provider_id: str, request: Request):
    """Internal: hand a reader the stored API key for any api-key provider.

    Serves xai / openai-api / anthropic-api (and any future ApiKeyAdapter)
    from the DB-backed CredentialStore — bridges and internal consumers call
    this instead of reading env vars. Static keys, so ``expires_at`` is null.
    Same trust model as the brokers above: internal network only, optional
    ``X-Bridge-Token`` defense in depth.
    """
    from ..providers.api_key import ApiKeyAdapter  # noqa: PLC0415

    expected = os.getenv("BRIDGE_CLAUDE_TOKEN_SECRET")
    if expected and request.headers.get("X-Bridge-Token") != expected:
        raise HTTPException(status_code=401, detail="bad bridge token")

    adapter = _adapter_or_404(provider_id)
    if not isinstance(adapter, ApiKeyAdapter):
        raise HTTPException(status_code=404, detail=f"{provider_id} has no api-key broker")
    key = await run_in_threadpool(adapter.api_key)
    if not key:
        raise HTTPException(status_code=503, detail=f"no {provider_id} credential installed")
    return {"access_token": key, "expires_at": None, "state": "ok"}
