"""Search-tool credentials, record-first and fail-closed (TASK-1476).

API keys live in ``provider_connection:<id>``; Reddit's OAuth app credentials
live in ``integration_connection:reddit``. The encrypted record wins. The legacy env vars,
then the ``Config`` field, are read only when no record exists, with one
warning per provider that names the source and never a value. A record that
exists but can't be used (disconnected, undecryptable, store down) never falls
back to env: that provider is treated as unconfigured and :attr:`SearchKey.problem`
says why.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .base import SearchProvider

if TYPE_CHECKING:
    from ..service.providers.search_keys import SearchKeyAdapter
    from ..utils.config import Config

logger = logging.getLogger(__name__)

#: Warnings already logged: ``<id>`` for a legacy fallback, ``<id>:<code>``
#: for an unusable record. One each per process.
_legacy_warned: set[str] = set()


@dataclass(frozen=True)
class SearchKey:
    """A resolved key, or why there is none (``problem`` is a ConnectionUnavailable code)."""

    value: str | None = None
    problem: str | None = None

    def __bool__(self) -> bool:
        return bool(self.value)


def _adapter_cls(provider_id: str) -> type[SearchKeyAdapter]:
    from ..service.providers.search_keys import SEARCH_KEY_ADAPTERS  # noqa: PLC0415 — import cycle

    for cls in SEARCH_KEY_ADAPTERS:
        if cls.id == provider_id:
            return cls
    raise KeyError(provider_id)


def _warn_unusable(label: str, connection_id: str, code: str) -> None:
    if f"{connection_id}:{code}" not in _legacy_warned:
        _legacy_warned.add(f"{connection_id}:{code}")
        logger.warning("%s credentials are unavailable (%s); %s is skipped until it is reconnected.",
                       label, code, label)


def resolve_key(config: Config, provider_id: str) -> SearchKey:
    """The ``brave``, ``tavily`` or ``newsapi`` key for this call."""
    from ..service.connection_secrets import (  # noqa: PLC0415 — import cycle
        ConnectionUnavailable,
        FieldSpec,
        has_connection_store,
        read_connection,
    )

    cls = _adapter_cls(provider_id)
    if has_connection_store(config):
        try:
            found = read_connection(config, "provider", provider_id, FieldSpec(secret=("api_key",)))
        except ConnectionUnavailable as exc:
            _warn_unusable(cls.label, provider_id, exc.code)
            return SearchKey(problem=exc.code)
        if found is not None:
            return SearchKey(value=found.secret["api_key"])

    legacy = [(var, os.getenv(var)) for var in cls.env_vars]
    legacy.append((f"Config.{cls.config_attr}", getattr(config, cls.config_attr, None)))
    for source, value in legacy:
        if isinstance(value, str) and value.strip():
            if provider_id not in _legacy_warned:
                _legacy_warned.add(provider_id)
                logger.warning(
                    "Using %s for %s because no connection is stored (deprecated; "
                    "connect the key in Settings → Providers). Checked: %s.",
                    source, cls.label, ", ".join(name for name, _ in legacy),
                )
            return SearchKey(value=value.strip())
    return SearchKey()


_PROVIDER_IDS = {SearchProvider.BRAVE: "brave", SearchProvider.TAVILY: "tavily"}


def search_key(config: Config, provider: SearchProvider) -> SearchKey:
    """The key for a keyed web-search provider (Brave or Tavily)."""
    return resolve_key(config, _PROVIDER_IDS[provider])


@dataclass(frozen=True)
class RedditCredentials:
    """Reddit OAuth app credentials, or why there are none (``problem`` as in :class:`SearchKey`)."""

    client_id: str | None = None
    client_secret: str | None = field(default=None, repr=False)
    problem: str | None = None

    def __bool__(self) -> bool:
        return bool(self.client_id and self.client_secret)


REDDIT_INTEGRATION_ID = "reddit"
# The env names behind Config.REDDIT_CLIENT_ID/SECRET, for the deprecation warning.
_REDDIT_LEGACY_NAMES = ("LLM_BAWT_REDDIT_CLIENT_ID", "LLM_BAWT_REDDIT_CLIENT_SECRET")


def reddit_credentials(config: Config) -> RedditCredentials:
    """Reddit credentials for this call: the ``integration_connection:reddit`` record first."""
    from ..service.connection_secrets import (  # noqa: PLC0415 — import cycle
        ConnectionUnavailable,
        resolve_integration,
    )

    legacy = {"client_id": getattr(config, "REDDIT_CLIENT_ID", None),
              "client_secret": getattr(config, "REDDIT_CLIENT_SECRET", None)}
    try:
        found = resolve_integration(config, REDDIT_INTEGRATION_ID, legacy, legacy_names=_REDDIT_LEGACY_NAMES)
    except ConnectionUnavailable as exc:
        _warn_unusable("Reddit", REDDIT_INTEGRATION_ID, exc.code)
        return RedditCredentials(problem=exc.code)
    if found is None:
        return RedditCredentials()
    return RedditCredentials(client_id=found.meta["client_id"], client_secret=found.secret["client_secret"])


def client_is_current(config: Config, client: object) -> bool:
    """Whether a cached search client still holds the credentials stored now.

    A long-lived client must follow a disconnect or rotation, so its holder
    rebuilds it when this is False. Clients without credentials are always current.
    """
    provider = getattr(client, "PROVIDER", None)
    if provider in _PROVIDER_IDS:
        key = search_key(config, provider)
        return bool(key) and key.value == getattr(client, "_api_key", None)
    if provider == SearchProvider.REDDIT:
        creds = reddit_credentials(config)
        return bool(creds) and (creds.client_id, creds.client_secret) == (
            getattr(client, "_client_id", None), getattr(client, "_client_secret", None))
    return True


PROVIDERS_SETTINGS = "Settings → Providers"
INTEGRATIONS_SETTINGS = "Settings → Integrations"


def unusable_reason(label: str, key: SearchKey | RedditCredentials,
                    where: str = PROVIDERS_SETTINGS) -> str | None:
    """User-facing text for a stored credential that can't be used, else None."""
    if key.problem == "not_connected":
        return f"{label} is disconnected. Connect it in {where}."
    if key.problem == "unreadable":
        return f"The stored {label} key can't be read. Reconnect it in {where}."
    if key.problem:
        return f"The credential store is unavailable, so {label} can't be used right now."
    return None


__all__ = [
    "INTEGRATIONS_SETTINGS",
    "PROVIDERS_SETTINGS",
    "RedditCredentials",
    "SearchKey",
    "client_is_current",
    "reddit_credentials",
    "resolve_key",
    "search_key",
    "unusable_reason",
]
