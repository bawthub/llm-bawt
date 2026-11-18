"""Authenticate the MCP server (:8001) per caller (TASK-2381).

Reuses the TASK-2380 middleware with its own mode, ``mcp_auth_mode``
(``off|log|enforce``, env default ``LLM_BAWT_MCP_AUTH_MODE``, operator
override ``LLM_BAWT_MCP_AUTH_MODE_FORCE``), so :8001 can be rolled out and
rolled back independently of ``/v1``.

Accepted credentials:

* a bot's signed turn capability (bridges send it on every MCP call);
* ``X-LLM-Bawt-Key`` for service principals only: the app's own MCP client,
  plus any principal named in ``LLM_BAWT_MCP_SERVICE_PRINCIPALS`` that also
  has a key export in ``LLM_BAWT_API_KEY_EXPORTS`` (TASK-2426);
* an OAuth bearer from BawtHub for allowed human users (TASK-2426,
  ``mcp_oauth.py``; dormant unless configured).

Bot keys are not accepted here, and a turn capability presented with a key is
refused as ambiguous. Which bot a call acts for is still decided per tool by
the existing bot-access policy; this gate only refuses calls with no verified
caller at all.
"""
from __future__ import annotations

from .middleware import ApiAuthMiddleware, Observations
from .mode import ModeResolver
from .wiring import SETTINGS, turn_capability

SETTING_KEY = "mcp_auth_mode"
MODE_ENV = "LLM_BAWT_MCP_AUTH_MODE"
FORCE_ENV = "LLM_BAWT_MCP_AUTH_MODE_FORCE"
APP_PRINCIPAL = "llm-bawt-app"
SERVICE_PRINCIPALS_ENV = "LLM_BAWT_MCP_SERVICE_PRINCIPALS"


def service_principals(environ=None) -> frozenset[str]:
    """The app plus each named principal the app also exports a key for.

    A name without an export is not a service and is ignored (logged), so the
    MCP gate never admits a key principal that has no export policy.
    """
    import logging
    import os

    from .export import EXPORTS_ENV, parse

    env = os.environ if environ is None else environ
    wanted = {p.strip() for p in env.get(SERVICE_PRINCIPALS_ENV, "").split(",") if p.strip()}
    if not wanted:
        return frozenset({APP_PRINCIPAL})
    try:
        exported = {spec.principal for spec in parse(env.get(EXPORTS_ENV))}
    except ValueError:
        exported = set()
    missing = wanted - exported - {APP_PRINCIPAL}
    if missing:
        logging.getLogger(__name__).error(
            "MCP auth: %s names %s with no key export; not admitted",
            SERVICE_PRINCIPALS_ENV, ", ".join(sorted(missing)))
    return frozenset({APP_PRINCIPAL} | (wanted & exported))


RESOLVER = ModeResolver(SETTINGS, setting_key=SETTING_KEY, mode_env=MODE_ENV,
                        force_env=FORCE_ENV)
OBSERVATIONS = Observations()


def _path(scope) -> str:
    return scope.get("path") or "<none>"


def ensure_store(make_store=None) -> None:
    """Attach the runtime settings store when no app startup did (standalone).

    Inside the app, startup attaches it first. A store that can't be built
    leaves the mode unresolved, so the gate answers 503 rather than opening.
    """
    if SETTINGS.store is not None:
        return
    try:
        if make_store is None:
            from ..runtime_settings import RuntimeSettingsStore
            from ..utils.config import Config

            def make_store():
                return RuntimeSettingsStore(Config())
        SETTINGS.store = make_store()
    except Exception as error:  # noqa: BLE001 - fail closed: stays unresolved
        import logging

        logging.getLogger(__name__).error(
            "MCP auth: runtime settings store unavailable (%s); :8001 answers 503 "
            "until it is (or set %s)", type(error).__name__, FORCE_ENV)


def wrap(app):
    """The MCP ASGI app behind the :8001 gate. Nothing on :8001 is public."""
    gate = ApiAuthMiddleware(
        app, resolver=RESOLVER, checkers=(turn_capability,), observations=OBSERVATIONS,
        allowed_paths=frozenset(), route_template=_path,
        key_principal_allowed=service_principals().__contains__, refuse_turn_with_key=True)
    from .mcp_oauth import McpOAuthLayer, load_config

    config = load_config()  # TASK-2426: dormant unless configured
    outer = McpOAuthLayer(gate, config) if config else gate
    return _ObservationsEndpoint(gate, outer)


# The MCP server usually runs as its own process, so its counts live there.
# It serves them to the app's service key, whatever the mode, and the app's
# /v1/admin/auth-observations fetches them from here.
OBSERVATIONS_PATH = "/_auth/observations"


class _ObservationsEndpoint:
    """Serves the gate's own counts at OBSERVATIONS_PATH; everything else goes to the gate."""

    def __init__(self, gate: ApiAuthMiddleware, outer=None):
        self.gate = gate
        self.outer = outer or gate

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("path") != OBSERVATIONS_PATH:
            return await self.outer(scope, receive, send)
        import json

        from . import keys

        header = keys.HEADER.encode()
        presented = [v for k, v in scope.get("headers") or () if k.lower() == header]
        allowed = False
        if scope.get("method") == "GET" and len(presented) == 1:
            result = self.gate._verify_key(presented[0].decode("latin-1"))
            allowed = result.ok and self.gate._key_allowed(result.principal)
        status, payload = (
            (200, {"mode": self.gate.resolver.mode(), "rows": self.gate.observations.snapshot()})
            if allowed else (401, {"detail": "service_key_required"}))
        body = json.dumps(payload).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"cache-control", b"no-store"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


async def fetch_observations(server_url: str | None) -> dict:
    """The MCP server's own mode and counts; this process's if it can't be reached."""
    if server_url:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(server_url.rstrip("/") + OBSERVATIONS_PATH,
                                            headers=app_key_header())
            if response.status_code == 200:
                return {**response.json(), "source": "mcp_server"}
            reason = f"http_{response.status_code}"
        except Exception as error:  # noqa: BLE001 - report, never fail the admin view
            reason = type(error).__name__
    else:
        reason = "no_server_url"
    return {"mode": RESOLVER.mode(), "rows": OBSERVATIONS.snapshot(),
            "source": "in_process", "remote_error": reason}


def app_key_header() -> dict[str, str]:
    """The app's own service key header, derived in process; ``{}`` if unusable."""
    from . import keys
    from .client import HEADER_NAME

    try:
        return {HEADER_NAME: keys.header_value(APP_PRINCIPAL)}
    except Exception:  # noqa: BLE001 - no master key: send nothing, server decides
        return {}
