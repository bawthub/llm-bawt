"""The stored API key a usage card reads (TASK-1476).

Usage adapters for the bridge-proxy providers (Z.AI, Kimi For Coding,
Moonshot) authenticate their quota/balance reads with the same key the bridge
uses for inference. That key lives encrypted as ``provider_connection:<id>``;
the legacy env vars apply only while no record exists.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from ..canonical import STATUS_ERROR, STATUS_UNAUTHORIZED


@dataclass(frozen=True)
class StoredKey:
    """Either ``key``, or the ``status``/``error`` a card shows instead."""

    key: str | None = None
    status: str | None = None
    error: str | None = None


def _unusable(label: str, code: str) -> StoredKey:
    if code == "store_unavailable":
        return StoredKey(status=STATUS_ERROR, error=f"Couldn't read the stored {label} key.")
    if code == "not_connected":
        return StoredKey(status=STATUS_UNAUTHORIZED,
                         error=f"{label} is disconnected. Connect it in Settings → Providers.")
    return StoredKey(status=STATUS_UNAUTHORIZED,
                     error=f"The stored {label} key can't be read. Reconnect it in Settings → Providers.")


async def stored_api_key(provider_id: str, label: str, env_vars: Sequence[str]) -> StoredKey:
    """The key for ``provider_id``, read off the event loop.

    No record and no env gives ``StoredKey(status=unauthorized)``. A record
    that can't be used never falls back to env.
    """
    from ...connection_secrets import ConnectionUnavailable  # noqa: PLC0415
    from ...dependencies import get_service  # noqa: PLC0415
    from ...providers.api_key import resolve_stored_api_key  # noqa: PLC0415

    def read() -> str | None:
        return resolve_stored_api_key(get_service().config, provider_id, env_vars)

    try:
        key = await asyncio.to_thread(read)
    except ConnectionUnavailable as exc:
        return _unusable(label, exc.code)
    if not key:
        return StoredKey(status=STATUS_UNAUTHORIZED,
                         error=f"No {label} key connected. Connect it in Settings → Providers.")
    return StoredKey(key=key)


__all__ = ["StoredKey", "stored_api_key"]
