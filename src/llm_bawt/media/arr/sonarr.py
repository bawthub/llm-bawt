"""Async client for Sonarr v3 API.

Methods focus on the moves a bot actually needs:

- find a series (``lookup``)
- add it to the library and trigger a search (``add_series``)
- inspect the download queue (``queue``)
- recent history of grabs/imports/failures (``history``)
- helpers for quality profiles, root folders, and stats

The full Sonarr v3 spec is huge; this client only covers what the
``media_*`` MCP tools consume. Extend incrementally as new use cases
appear.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import ClientSlot, _BaseArrClient, arr_timeout, resolve_credentials

logger = logging.getLogger(__name__)


class SonarrClient(_BaseArrClient):
    """Sonarr v3 async client."""

    service = "sonarr"
    api_prefix = "/api/v3"

    # ---- discovery -------------------------------------------------------

    async def system_status(self) -> dict:
        return await self._get(f"{self.api_prefix}/system/status")

    async def lookup(self, term: str) -> list[dict]:
        """Search for a series by title/tvdb id (``tvdb:NNNN``) etc.

        Returns the raw Sonarr lookup objects; entries already in the
        library have ``id > 0``.
        """
        return await self._get(f"{self.api_prefix}/series/lookup", term=term) or []

    async def list_quality_profiles(self) -> list[dict]:
        return await self._get(f"{self.api_prefix}/qualityprofile") or []

    async def list_root_folders(self) -> list[dict]:
        return await self._get(f"{self.api_prefix}/rootfolder") or []

    async def list_language_profiles(self) -> list[dict]:
        # Sonarr v4 removed languageProfile; tolerate either.
        try:
            return await self._get(f"{self.api_prefix}/languageprofile") or []
        except Exception:
            return []

    # ---- library ---------------------------------------------------------

    async def list_series(self) -> list[dict]:
        return await self._get(f"{self.api_prefix}/series") or []

    async def get_series(self, series_id: int) -> dict | None:
        return await self._get(f"{self.api_prefix}/series/{series_id}")

    async def add_series(
        self,
        lookup_result: dict,
        *,
        quality_profile_id: int,
        root_folder_path: str,
        monitored: bool = True,
        season_folder: bool = True,
        search_now: bool = True,
        monitor: str = "all",
    ) -> dict:
        """Add a series to Sonarr.

        Pass a fresh ``lookup_result`` dict (from ``lookup``) — this avoids
        the brittle "construct the right shape from scratch" path. We
        overlay user-controlled fields on top so Sonarr accepts it.
        ``monitor`` corresponds to Sonarr's monitoring option
        (``all``, ``future``, ``missing``, ``existing``, ``firstSeason``,
        ``latestSeason``, ``none``).
        """
        payload = dict(lookup_result)
        payload.update(
            {
                "qualityProfileId": quality_profile_id,
                "rootFolderPath": root_folder_path,
                "monitored": monitored,
                "seasonFolder": season_folder,
                "addOptions": {
                    "ignoreEpisodesWithFiles": False,
                    "ignoreEpisodesWithoutFiles": False,
                    "searchForMissingEpisodes": search_now,
                    "monitor": monitor,
                },
            }
        )
        return await self._post(f"{self.api_prefix}/series", json=payload)

    async def trigger_series_search(self, series_id: int) -> dict:
        return await self._post(
            f"{self.api_prefix}/command",
            json={"name": "SeriesSearch", "seriesId": series_id},
        )

    # ---- queue / history -------------------------------------------------

    async def queue(
        self,
        *,
        page_size: int = 50,
        include_unknown_series_items: bool = True,
    ) -> dict:
        return await self._get(
            f"{self.api_prefix}/queue",
            pageSize=page_size,
            includeUnknownSeriesItems=str(include_unknown_series_items).lower(),
            includeSeries=str(True).lower(),
            includeEpisode=str(True).lower(),
        ) or {"records": [], "totalRecords": 0}

    async def history(
        self,
        *,
        page_size: int = 20,
        event_type: int | None = None,
    ) -> dict:
        """Sonarr history.

        ``event_type`` corresponds to Sonarr's enum:
          1 = grabbed, 2 = seriesFolderImported, 3 = downloadFolderImported,
          4 = downloadFailed, 5 = episodeFileDeleted, 6 = episodeFileRenamed,
          7 = downloadIgnored.
        """
        params: dict[str, Any] = {
            "page": 1,
            "pageSize": page_size,
            "sortKey": "date",
            "sortDirection": "descending",
        }
        if event_type is not None:
            params["eventType"] = event_type
        return await self._get(f"{self.api_prefix}/history", **params) or {
            "records": [],
            "totalRecords": 0,
        }

    async def missing(self, *, page_size: int = 20, monitored: bool = True) -> dict:
        return await self._get(
            f"{self.api_prefix}/wanted/missing",
            page=1,
            pageSize=page_size,
            sortKey="airDateUtc",
            sortDirection="descending",
            monitored=str(monitored).lower(),
            includeSeries=str(True).lower(),
        ) or {"records": [], "totalRecords": 0}


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_slot = ClientSlot(lambda url, key: SonarrClient(url, key, timeout=arr_timeout()))


def _credentials() -> tuple[str, str]:
    return resolve_credentials("sonarr", "LLM_BAWT_SONARR_URL", "LLM_BAWT_SONARR_API_KEY")


def sonarr_client():
    """Hold the process-wide Sonarr client for one operation.

    ``async with sonarr_client() as sonarr:``. Credentials are resolved on
    entry; it raises ``ArrNotConfigured`` when there is no stored connection
    and no legacy env, ``ArrUnavailable`` when the stored connection can't be
    used. Callers in the MCP tool layer surface both as a structured error.
    """
    return _slot.lease(_credentials)
