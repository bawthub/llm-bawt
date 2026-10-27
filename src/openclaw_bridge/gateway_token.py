"""The OpenClaw gateway token, record first (TASK-1476).

The token lives in the app's encrypted ``integration_connection:openclaw``
record (Integrations → OpenClaw gateway). This bridge holds no root key, so it
asks the app through the bridge-gated read route
``GET /v1/internal/bridge/connections/integration/openclaw`` on every connect;
a saved or rotated token applies at the next (re)connect without a restart.

* 200: the record's token.
* 404 ``not_configured``: no record. The deprecated ``OPENCLAW_GATEWAY_TOKEN``
  env applies (warned once, by name only), else no token.
* Anything else (409 disconnected, 503 unreadable or store down, 401, network
  error, malformed body): :class:`GatewayTokenUnavailable`. A record that
  exists but can't be used never falls back to env.

With no ``LLM_BAWT_API_URL`` there is no app to ask, so the legacy env is the
only source. Values are never logged.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

import httpx

from agent_bridge import bridge_token

logger = logging.getLogger(__name__)

LEGACY_ENV = "OPENCLAW_GATEWAY_TOKEN"
ROUTE = "/v1/internal/bridge/connections/integration/openclaw"
TIMEOUT_S = 10.0

_warned: set[str] = set()


class GatewayTokenUnavailable(RuntimeError):
    """The stored gateway token exists but can't be read. Carries a code, never a value."""

    def __init__(self, code: str):
        super().__init__(f"OpenClaw gateway token unavailable: {code}")
        self.code = code


def _legacy(env: Mapping[str, str]) -> str:
    value = (env.get(LEGACY_ENV) or "").strip()
    if value and LEGACY_ENV not in _warned:
        _warned.add(LEGACY_ENV)
        logger.warning("OpenClaw gateway token read from deprecated %s because no connection "
                       "is stored; save it under Integrations instead", LEGACY_ENV)
    return value


def _code(response: httpx.Response) -> str:
    try:
        detail = (response.json() or {}).get("detail")
    except ValueError:
        detail = None
    return detail if isinstance(detail, str) and detail.isidentifier() else f"http_{response.status_code}"


def gateway_token(environ: Mapping[str, str] | None = None, *,
                  client: httpx.Client | None = None) -> str:
    """The token to present, ``""`` for none, or :class:`GatewayTokenUnavailable`."""
    env = os.environ if environ is None else environ
    api_url = (env.get("LLM_BAWT_API_URL") or "").strip().rstrip("/")
    if not api_url:
        return _legacy(env)
    try:
        if client is None:
            response = httpx.get(f"{api_url}{ROUTE}", headers=bridge_token.headers(environ=env),
                                 timeout=TIMEOUT_S)
        else:
            response = client.get(f"{api_url}{ROUTE}", headers=bridge_token.headers(environ=env))
    except httpx.HTTPError as exc:
        raise GatewayTokenUnavailable(f"unreachable:{type(exc).__name__}") from None
    if response.status_code == 404:
        return _legacy(env)
    if response.status_code != 200:
        raise GatewayTokenUnavailable(_code(response))
    try:
        token = ((response.json() or {}).get("secret") or {}).get("token")
    except (ValueError, AttributeError):
        token = None
    if not isinstance(token, str) or not token.strip():
        raise GatewayTokenUnavailable("malformed")
    return token.strip()


__all__ = ["GatewayTokenUnavailable", "LEGACY_ENV", "ROUTE", "gateway_token"]
