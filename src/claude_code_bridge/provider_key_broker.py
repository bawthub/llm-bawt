"""Read a provider's API key from the app's generic API-key broker (TASK-1295).

The app keeps connected provider keys encrypted in its CredentialStore and
serves them at ``GET /v1/providers/<provider_id>/token`` (TASK-825). Bridge
consumers ask on every use, so rotation and disconnects are visible
immediately; nothing is cached here.

``BrokerKeyNotInstalled`` (HTTP 503) is the only outcome a caller may treat
as "no DB connection exists"; every other failure is a distinct error so it
is never masked by a fallback credential.
"""

from __future__ import annotations

import logging
import os

import httpx

from agent_bridge import bridge_token

logger = logging.getLogger(__name__)

BROKER_TIMEOUT_S = 15.0


class BrokerKeyError(RuntimeError):
    """The broker could not supply a key (unreachable, refused, malformed)."""


class BrokerKeyNotInstalled(BrokerKeyError):
    """The broker answered 503: no key is connected for this provider."""


def fetch_provider_api_key(provider_id: str) -> str:
    """Return the connected key for ``provider_id`` or raise ``BrokerKeyError``."""
    api_url = (os.environ.get("LLM_BAWT_API_URL") or "").rstrip("/")
    if not api_url:
        raise BrokerKeyError(
            f"LLM_BAWT_API_URL not set — cannot reach the {provider_id} API-key broker"
        )
    headers = bridge_token.headers()
    try:
        response = httpx.get(
            f"{api_url}/v1/providers/{provider_id}/token",
            headers=headers,
            timeout=BROKER_TIMEOUT_S,
        )
    except httpx.HTTPError as exc:
        raise BrokerKeyError(f"{provider_id} API-key broker unreachable: {exc}") from exc
    if response.is_error:
        detail = (response.text or "")[:200]
        if response.status_code == 503:
            raise BrokerKeyNotInstalled(f"no {provider_id} API key is connected ({detail})")
        raise BrokerKeyError(
            f"{provider_id} API-key broker returned {response.status_code}: {detail}"
        )
    try:
        key = (response.json() or {}).get("access_token")
    except ValueError as exc:
        raise BrokerKeyError(f"{provider_id} API-key broker returned invalid JSON") from exc
    if not key:
        raise BrokerKeyError(f"{provider_id} API-key broker returned no key")
    logger.info("Fetched %s API key from app broker", provider_id)
    return str(key)
