"""Shared base for Sonarr/Radarr/SABnzbd async clients.

Pulls common concerns into one place:

- credentials from the encrypted ``integration_connection:<service>`` record,
  with the legacy env vars only as a deprecated fallback when no record exists
- one ``httpx.AsyncClient`` per service (lazy, rebuilt when credentials change)
- consistent error wrapping (``ArrAPIError``) so MCP tools can return
  a structured failure instead of leaking httpx tracebacks
- a small ``api_key`` header injection helper for Sonarr/Radarr
  (SABnzbd takes its key as a query param, so it overrides)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 20.0


class ArrAPIError(RuntimeError):
    """Raised when an *arr or SAB API returns a non-2xx or invalid payload."""

    def __init__(self, service: str, status: int | None, message: str):
        super().__init__(f"[{service}] HTTP {status}: {message}")
        self.service = service
        self.status = status
        self.message = message


class ArrNotConfigured(RuntimeError):
    """Raised when the env vars for a given service are missing."""

    def __init__(self, service: str, missing: list[str]):
        super().__init__(
            f"[{service}] not configured — missing env vars: {', '.join(missing)}"
        )
        self.service = service
        self.missing = missing


class _BaseArrClient:
    """Common HTTP plumbing for Sonarr/Radarr v3-style APIs.

    Subclasses set ``service`` (used in errors/logs), ``base_url``, and
    ``api_key``, then call ``_get``/``_post``/``_put``/``_delete``.
    """

    service: str = "arr"

    def __init__(self, base_url: str, api_key: str, timeout: float = _DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers={"X-Api-Key": api_key, "Accept": "application/json"},
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json: Any = None,
    ) -> Any:
        try:
            resp = await self._client.request(method, path, params=params, json=json)
        except httpx.HTTPError as e:
            raise ArrAPIError(self.service, None, f"transport: {e!r}") from e
        if resp.status_code >= 400:
            # Try to surface the API's own error body — *arr usually returns
            # a list of {errorMessage, propertyName} on 400.
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:200]
            raise ArrAPIError(self.service, resp.status_code, str(detail))
        if not resp.content:
            return None
        try:
            return resp.json()
        except Exception as e:
            raise ArrAPIError(self.service, resp.status_code, f"non-JSON body: {e!r}") from e

    async def _get(self, path: str, **params: Any) -> Any:
        return await self._request("GET", path, params=params or None)

    async def _post(self, path: str, json: Any = None) -> Any:
        return await self._request("POST", path, json=json)

    async def _put(self, path: str, json: Any = None) -> Any:
        return await self._request("PUT", path, json=json)

    async def _delete(self, path: str, **params: Any) -> Any:
        return await self._request("DELETE", path, params=params or None)

    async def aclose(self) -> None:
        await self._client.aclose()


def _read_env_pair(service: str, url_var: str, key_var: str) -> tuple[str, str]:
    """Read a (url, api_key) pair from env, raising ArrNotConfigured if missing."""
    url = os.getenv(url_var)
    key = os.getenv(key_var)
    missing = [n for n, v in [(url_var, url), (key_var, key)] if not v]
    if missing:
        raise ArrNotConfigured(service, missing)
    return url, key  # type: ignore[return-value]


class ArrUnavailable(RuntimeError):
    """A stored connection exists (or the store is down) but can't be used.

    ``code`` is ``not_connected``, ``unreadable`` or ``store_unavailable``. The
    legacy env vars are never consulted in this case: a broken record must
    surface, not be silently replaced by an old value.
    """

    def __init__(self, service: str, code: str):
        super().__init__(f"[{service}] stored connection unavailable: {code}")
        self.service = service
        self.code = code


_ENV_FALLBACK_WARNED: set[str] = set()


def resolve_credentials(service: str, url_var: str, key_var: str, *, config=None) -> tuple[str, str]:
    """``(url, api_key)`` for one service, from its encrypted connection first.

    The ``integration_connection:<service>`` record (the Integrations
    credentials form) wins. Only when no record exists are the legacy env vars
    read, with a one-time deprecation warning naming the vars, never a value.
    """
    from llm_bawt.service.connection_secrets import (
        ConnectionUnavailable,
        integration_spec,
        read_connection,
    )

    if config is None:
        from llm_bawt.utils.config import Config

        config = Config()
    try:
        found = read_connection(config, "integration", service, integration_spec(service))
    except ConnectionUnavailable as e:
        raise ArrUnavailable(service, e.code) from None
    if found is not None:
        return found.meta["url"], found.secret["api_key"]
    url, key = _read_env_pair(service, url_var, key_var)
    if service not in _ENV_FALLBACK_WARNED:
        _ENV_FALLBACK_WARNED.add(service)
        logger.warning(
            "%s credentials read from deprecated env vars %s/%s; save them under "
            "Integrations instead", service, url_var, key_var,
        )
    return url, key


class ClientSlot:
    """One process-wide client per service, rebuilt when its credentials change.

    Credentials are resolved on every call, so a key saved or rotated in the
    Integrations form takes effect without a restart.
    """

    def __init__(self, build):
        self._build = build
        self._client = None
        self._pair: tuple[str, str] | None = None

    def get(self, pair: tuple[str, str]):
        if self._client is None or self._pair != pair:
            old = self._client
            self._client, self._pair = self._build(*pair), pair
            if old is not None:
                _close_later(old)
        return self._client

    def reset(self) -> None:
        self._client, self._pair = None, None


def _close_later(client) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no loop: nothing in flight, let it be collected
    task = loop.create_task(client.aclose())
    _CLOSING.add(task)
    task.add_done_callback(_CLOSING.discard)


_CLOSING: set = set()


def arr_timeout() -> float:
    return float(os.getenv("LLM_BAWT_ARR_TIMEOUT", str(_DEFAULT_TIMEOUT)))
