"""Authenticate the MCP server (:8001) per caller (TASK-2381).

Reuses the TASK-2380 middleware with its own mode, ``mcp_auth_mode``
(``off|log|enforce``, env default ``LLM_BAWT_MCP_AUTH_MODE``, operator
override ``LLM_BAWT_MCP_AUTH_MODE_FORCE``), so :8001 can be rolled out and
rolled back independently of ``/v1``.

Accepted credentials:

* a bot's signed turn capability (bridges send it on every MCP call);
* ``X-LLM-Bawt-Key`` for service principals only (the app's own MCP client).

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
SERVICE_PRINCIPALS = frozenset({APP_PRINCIPAL})

RESOLVER = ModeResolver(SETTINGS, setting_key=SETTING_KEY, mode_env=MODE_ENV,
                        force_env=FORCE_ENV)
OBSERVATIONS = Observations()


def _path(scope) -> str:
    return scope.get("path") or "<none>"


def wrap(app):
    """The MCP ASGI app behind the :8001 gate. Nothing on :8001 is public."""
    return ApiAuthMiddleware(
        app, resolver=RESOLVER, checkers=(turn_capability,), observations=OBSERVATIONS,
        allowed_paths=frozenset(), route_template=_path,
        key_principal_allowed=SERVICE_PRINCIPALS.__contains__, refuse_turn_with_key=True)


def app_key_header() -> dict[str, str]:
    """The app's own service key header, derived in process; ``{}`` if unusable."""
    from . import keys
    from .client import HEADER_NAME

    try:
        return {HEADER_NAME: keys.header_value(APP_PRINCIPAL)}
    except Exception:  # noqa: BLE001 - no master key: send nothing, server decides
        return {}
