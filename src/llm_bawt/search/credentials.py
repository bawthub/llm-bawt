"""Search-tool API keys, record-first and fail-closed (TASK-1476).

The encrypted ``provider_connection:<id>`` record wins. The legacy env vars,
then the ``Config`` field, are read only when no record exists, with one
warning per provider that names the source and never a value. A record that
exists but can't be used (disconnected, undecryptable, store down) never falls
back to env: that provider is treated as unconfigured and :attr:`SearchKey.problem`
says why.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
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


def resolve_key(config: Config, provider_id: str) -> SearchKey:
    """The ``brave``, ``tavily`` or ``newsapi`` key for this call."""
    from ..service.connection_secrets import (  # noqa: PLC0415 — import cycle
        ConnectionUnavailable,
        FieldSpec,
        read_connection,
    )

    from ..utils.config import has_database_credentials  # noqa: PLC0415

    cls = _adapter_cls(provider_id)
    # Without database credentials (a CLI-only install) no record can exist,
    # so the legacy sources are the only ones. A configured but unreachable
    # database is a store outage and stays fail-closed. Search configs are
    # duck-typed, so a config without the field counts as no database.
    if hasattr(config, "POSTGRES_PASSWORD") and has_database_credentials(config):
        try:
            found = read_connection(config, "provider", provider_id, FieldSpec(secret=("api_key",)))
        except ConnectionUnavailable as exc:
            if f"{provider_id}:{exc.code}" not in _legacy_warned:
                _legacy_warned.add(f"{provider_id}:{exc.code}")
                logger.warning("%s key is unavailable (%s); %s is skipped until it is reconnected.",
                               cls.label, exc.code, cls.label)
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


def unusable_reason(label: str, key: SearchKey) -> str | None:
    """User-facing text for a stored key that can't be used, else None."""
    if key.problem == "not_connected":
        return f"{label} is disconnected. Connect it in Settings → Providers."
    if key.problem == "unreadable":
        return f"The stored {label} key can't be read. Reconnect it in Settings → Providers."
    if key.problem:
        return f"The credential store is unavailable, so {label} can't be used right now."
    return None


__all__ = ["SearchKey", "resolve_key", "search_key", "unusable_reason"]
