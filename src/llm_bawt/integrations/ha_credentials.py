"""Home Assistant URL and token, from the encrypted store first (TASK-1476).

The ``integration_connection:home-assistant`` record (the Integrations
credentials form, shared with BawtHub's weather card) wins. Only when no record
exists are the legacy ``HA_NATIVE_MCP_URL``/``HA_NATIVE_MCP_TOKEN`` settings
read, with a one-time deprecation warning naming them, never a value.

A record that exists but can't be used (disconnected, undecryptable, store
down) raises :class:`HomeAssistantUnavailable` and never falls back to env: a
disconnect must not bring an old token back. Disconnect therefore keeps a
credential-free ``disconnected`` record (``keeps_disconnected_record``).

Resolve per operation, not once per process, so a saved or rotated token is
used without a restart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from llm_bawt.service.connection_secrets import ConnectionUnavailable, integration_spec, read_connection

logger = logging.getLogger(__name__)

INTEGRATION_ID = "home-assistant"
_MCP_PATH = "/api/mcp"
_env_fallback_warned = False


class HomeAssistantUnavailable(RuntimeError):
    """A stored connection exists (or the store is down) but can't be used.

    ``code`` is ``not_connected``, ``unreadable`` or ``store_unavailable``.
    """

    def __init__(self, code: str):
        super().__init__(f"Home Assistant connection unavailable: {code}")
        self.code = code


@dataclass(frozen=True)
class HACredentials:
    base_url: str  # without a trailing slash or /api/mcp
    token: str

    @property
    def mcp_url(self) -> str:
        return self.base_url + _MCP_PATH

    def __repr__(self) -> str:  # never show the token
        return f"HACredentials(base_url={self.base_url!r}, token=<set>)"


def _base_url(url: str) -> str:
    return url.strip().rstrip("/").removesuffix(_MCP_PATH).rstrip("/")


def _stored(config):
    """The usable record, ``None`` when none exists; raises when unusable."""
    try:
        return read_connection(config, "integration", INTEGRATION_ID, integration_spec(INTEGRATION_ID))
    except ConnectionUnavailable as error:
        raise HomeAssistantUnavailable(error.code) from None


def check_stored_connection(config) -> bool:
    """Whether a usable record exists (``False`` when none does).

    Raises :class:`HomeAssistantUnavailable` for a record that exists but
    can't be used, so env-only clients (the legacy MCP server) stay off after
    a disconnect, the same as the native readers.
    """
    return _stored(config) is not None


def resolve_ha_credentials(config) -> HACredentials | None:
    """The Home Assistant credentials to use now, or ``None`` when none are set."""
    global _env_fallback_warned
    found = _stored(config)
    if found is not None:
        return HACredentials(_base_url(found.meta["url"]), found.secret["token"].strip())
    url = _base_url(getattr(config, "HA_NATIVE_MCP_URL", "") or "")
    token = (getattr(config, "HA_NATIVE_MCP_TOKEN", "") or "").strip()
    if not url or not token:
        return None
    if not _env_fallback_warned:
        _env_fallback_warned = True
        logger.warning(
            "Home Assistant credentials read from deprecated settings "
            "LLM_BAWT_HA_NATIVE_MCP_URL/LLM_BAWT_HA_NATIVE_MCP_TOKEN; save them under Integrations instead"
        )
    return HACredentials(url, token)


__all__ = [
    "HACredentials",
    "HomeAssistantUnavailable",
    "INTEGRATION_ID",
    "check_stored_connection",
    "resolve_ha_credentials",
]
