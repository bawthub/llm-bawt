"""Async client for SABnzbd.

SABnzbd's API is one endpoint (``/api``) where you pass ``mode=<verb>``,
``apikey=...``, ``output=json``. We wrap the few operations the media
tools actually need: version check, queue, history, pause/resume.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from .base import ArrAPIError, ClientSlot, arr_timeout, resolve_credentials

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 20.0


class SabnzbdClient:
    """Minimal SABnzbd async client.

    Doesn't inherit ``_BaseArrClient`` because SAB doesn't use the
    X-Api-Key header pattern — every request takes ``apikey`` as a query
    arg, and there's no path-based REST.
    """

    service = "sabnzbd"

    def __init__(self, base_url: str, api_key: str, timeout: float = _DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

    async def _call(self, mode: str, **extra: Any) -> dict:
        params: dict[str, Any] = {
            "mode": mode,
            "output": "json",
            "apikey": self.api_key,
        }
        params.update(extra)
        try:
            resp = await self._client.get("/api", params=params)
        except httpx.HTTPError as e:
            raise ArrAPIError(self.service, None, f"transport: {e!r}") from e
        if resp.status_code >= 400:
            raise ArrAPIError(self.service, resp.status_code, resp.text[:200])
        try:
            data = resp.json()
        except Exception as e:
            raise ArrAPIError(
                self.service, resp.status_code, f"non-JSON body: {e!r}"
            ) from e
        # SAB returns {"status": false, "error": "..."} on auth failures.
        if isinstance(data, dict) and data.get("status") is False and "error" in data:
            raise ArrAPIError(self.service, resp.status_code, data["error"])
        return data

    # ---- info ------------------------------------------------------------

    async def version(self) -> dict:
        return await self._call("version")

    async def queue(self, *, limit: int = 50) -> dict:
        """Current download queue.

        Returns the ``queue`` sub-object Sab gives us — keys include
        ``paused``, ``speed``, ``size``, ``sizeleft``, ``timeleft``,
        ``diskspace1`` (free GB on download dir), and ``slots`` (list of
        in-flight downloads).
        """
        data = await self._call("queue", start=0, limit=limit)
        return data.get("queue", data)

    async def history(self, *, limit: int = 20) -> dict:
        """Recent SAB history (completed/failed jobs)."""
        data = await self._call("history", start=0, limit=limit)
        return data.get("history", data)

    async def pause(self) -> dict:
        return await self._call("pause")

    async def resume(self) -> dict:
        return await self._call("resume")

    async def aclose(self) -> None:
        await self._client.aclose()


_slot = ClientSlot(lambda url, key: SabnzbdClient(url, key, timeout=arr_timeout()))


async def get_sabnzbd_client() -> SabnzbdClient:
    """The process-wide SABnzbd client for the current credentials (see Sonarr)."""
    pair = await asyncio.to_thread(
        resolve_credentials, "sabnzbd", "LLM_BAWT_SABNZBD_URL", "LLM_BAWT_SABNZBD_API_KEY",
    )
    return _slot.get(pair)
