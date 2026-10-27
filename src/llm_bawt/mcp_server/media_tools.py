"""MCP tools for the media library (Sonarr / Radarr / SABnzbd).

Surface area (six tools — kept tight on purpose):

    media_lookup           — search for new content (movies or series)
    media_add              — add to library, optionally trigger a search
    media_queue            — combined queue across Sonarr/Radarr/SAB
    media_history          — recent grabs / imports / failures
    media_pipeline_status  — trace one title end-to-end through the stack
    media_library_stats    — quick snapshot of library + downloader state

Imported by ``server.py`` so registration happens on startup. Each tool
returns plain dicts/lists that serialize as JSON. Configuration errors
(no stored connection, or one that can't be used) surface as
``{"error": "...", "service": ...}``
rather than raised exceptions so bots can react.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_bawt.media.arr import (
    ArrAPIError,
    ArrNotConfigured,
    ArrUnavailable,
    radarr_client,
    sabnzbd_client,
    sonarr_client,
)

from .server import mcp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _err(e: Exception) -> dict:
    """Convert client errors into a structured response."""
    if isinstance(e, ArrNotConfigured):
        return {"error": str(e), "service": e.service, "configured": False}
    if isinstance(e, ArrUnavailable):
        return {"error": str(e), "service": e.service, "configured": True, "code": e.code}
    if isinstance(e, ArrAPIError):
        return {
            "error": e.message,
            "service": e.service,
            "status": e.status,
            "configured": True,
        }
    return {"error": f"{type(e).__name__}: {e}"}


def _shorten(s: Any, n: int = 240) -> str:
    if not s:
        return ""
    text = str(s)
    return text if len(text) <= n else text[: n - 1] + "…"


def _condense_lookup_movie(m: dict) -> dict:
    return {
        "kind": "movie",
        "title": m.get("title"),
        "year": m.get("year"),
        "tmdb_id": m.get("tmdbId"),
        "imdb_id": m.get("imdbId"),
        "in_library": (m.get("id") or 0) > 0,
        "library_id": m.get("id") or None,
        "monitored": m.get("monitored"),
        "status": m.get("status"),
        "runtime": m.get("runtime"),
        "studio": m.get("studio"),
        "overview": _shorten(m.get("overview")),
        "remote_poster": m.get("remotePoster"),
    }


def _condense_lookup_series(s: dict) -> dict:
    return {
        "kind": "series",
        "title": s.get("title"),
        "year": s.get("year"),
        "tvdb_id": s.get("tvdbId"),
        "imdb_id": s.get("imdbId"),
        "in_library": (s.get("id") or 0) > 0,
        "library_id": s.get("id") or None,
        "monitored": s.get("monitored"),
        "status": s.get("status"),
        "seasons": len(s.get("seasons") or []),
        "network": s.get("network"),
        "runtime": s.get("runtime"),
        "overview": _shorten(s.get("overview")),
        "remote_poster": s.get("remotePoster"),
    }


def _condense_sonarr_queue_item(r: dict) -> dict:
    size = r.get("size") or 0
    sizeleft = r.get("sizeleft") or 0
    progress = round((1 - sizeleft / size) * 100, 1) if size else 0.0
    series = r.get("series") or {}
    return {
        "kind": "series",
        "title": r.get("title") or series.get("title"),
        "series_title": series.get("title"),
        "status": r.get("status"),
        "tracked_status": r.get("trackedDownloadStatus"),
        "tracked_state": r.get("trackedDownloadState"),
        "progress_pct": progress,
        "size_mb": round(size / 1e6, 1) if size else 0,
        "timeleft": r.get("timeleft"),
        "estimated_completion": r.get("estimatedCompletionTime"),
        "download_client": r.get("downloadClient"),
        "id": r.get("id"),
    }


def _condense_radarr_queue_item(r: dict) -> dict:
    size = r.get("size") or 0
    sizeleft = r.get("sizeleft") or 0
    progress = round((1 - sizeleft / size) * 100, 1) if size else 0.0
    movie = r.get("movie") or {}
    return {
        "kind": "movie",
        "title": r.get("title") or movie.get("title"),
        "movie_title": movie.get("title"),
        "status": r.get("status"),
        "tracked_status": r.get("trackedDownloadStatus"),
        "tracked_state": r.get("trackedDownloadState"),
        "progress_pct": progress,
        "size_mb": round(size / 1e6, 1) if size else 0,
        "timeleft": r.get("timeleft"),
        "estimated_completion": r.get("estimatedCompletionTime"),
        "download_client": r.get("downloadClient"),
        "id": r.get("id"),
    }


def _condense_sab_slot(s: dict) -> dict:
    return {
        "kind": "sab_slot",
        "title": s.get("filename") or s.get("name"),
        "status": s.get("status"),
        "category": s.get("cat"),
        "percentage": _safe_float(s.get("percentage")),
        "size": s.get("size"),
        "sizeleft": s.get("sizeleft"),
        "timeleft": s.get("timeleft"),
        "nzo_id": s.get("nzo_id"),
        "priority": s.get("priority"),
    }


def _safe_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


# Resolve "HD-1080p" → profile id; pull the highest-matching one.
async def _resolve_quality_profile(
    list_fn, requested: str | None
) -> tuple[int | None, str | None]:
    profiles = await list_fn()
    if not profiles:
        return None, None
    if requested:
        name_lower = requested.lower()
        for p in profiles:
            if p["name"].lower() == name_lower:
                return p["id"], p["name"]
        # Soft fallback: substring
        for p in profiles:
            if name_lower in p["name"].lower():
                return p["id"], p["name"]
    # Default: prefer HD-1080p, then Any, then first one
    for preferred in ("HD-1080p", "Any"):
        for p in profiles:
            if p["name"] == preferred:
                return p["id"], p["name"]
    return profiles[0]["id"], profiles[0]["name"]


async def _pick_root_folder(list_fn) -> str | None:
    folders = await list_fn()
    return folders[0]["path"] if folders else None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool(name="media_lookup")
async def media_lookup(query: str, kind: str = "auto", limit: int = 5) -> dict:
    """Search Sonarr (TV) and/or Radarr (movies) for content to download.

    Use this BEFORE ``media_add`` — the result tells you whether the
    title is already in the library and surfaces a stable external id.

    Args:
        query: Title or external id reference. Sonarr accepts forms like
            ``tvdb:NNNN`` and ``imdb:ttNNNN``; Radarr accepts
            ``tmdb:NNN``/``imdb:ttNNN``.
        kind: ``"auto"`` (search both), ``"movie"``, or ``"series"``.
        limit: Max results per kind.

    Returns:
        ``{"results": [{kind, title, year, tmdb_id/tvdb_id, in_library, ...}],
           "errors": {service: msg}}``
    """
    logger.debug("MCP tool invoked: media_lookup query=%r kind=%s", query, kind)
    results: list[dict] = []
    errors: dict[str, str] = {}

    kinds = {"series", "movie"} if kind == "auto" else {kind}

    if "series" in kinds:
        try:
            async with sonarr_client() as sonarr:
                raw = await sonarr.lookup(query)
                results.extend(_condense_lookup_series(s) for s in raw[:limit])
        except Exception as e:
            errors["sonarr"] = str(e)

    if "movie" in kinds:
        try:
            async with radarr_client() as radarr:
                raw = await radarr.lookup(query)
                results.extend(_condense_lookup_movie(m) for m in raw[:limit])
        except Exception as e:
            errors["radarr"] = str(e)

    return {"query": query, "kind": kind, "results": results, "errors": errors}


@mcp.tool(name="media_add")
async def media_add(
    kind: str,
    query: str | None = None,
    external_id: str | None = None,
    quality_profile: str | None = None,
    monitored: bool = True,
    search_now: bool = True,
    monitor: str = "all",
    minimum_availability: str = "released",
) -> dict:
    """Add a movie or series to the library and (optionally) trigger a search.

    Resolves the title via lookup, picks the first match, then submits
    it to Sonarr/Radarr. If the title is already in the library, returns
    an ``already_in_library`` flag instead of an error.

    Args:
        kind: ``"movie"`` or ``"series"``.
        query: Free-text title (or ``tmdb:NNN`` / ``tvdb:NNN`` / ``imdb:ttNNN``).
            Either ``query`` or ``external_id`` must be supplied.
        external_id: Exact external id, e.g. ``tmdb:693134`` — preferred
            over ``query`` because it skips disambiguation.
        quality_profile: Profile name (``"HD-1080p"``, ``"Ultra-HD"``,
            ``"Any"``, etc.). Defaults to ``HD-1080p`` if available, else
            the first profile.
        monitored: Whether the item is monitored for downloads.
        search_now: If True, kick off an immediate indexer search.
        monitor: (series only) ``"all"`` (default), ``"future"``,
            ``"missing"``, ``"existing"``, ``"firstSeason"``,
            ``"latestSeason"``, ``"none"``.
        minimum_availability: (movie only) one of ``"announced"``,
            ``"inCinemas"``, ``"released"``, ``"preDB"``.

    Returns:
        ``{success, kind, library_id, title, year, profile, message}``
        — or ``{error, ...}`` on failure.
    """
    if kind not in {"movie", "series"}:
        return {"error": "kind must be 'movie' or 'series'"}
    if not query and not external_id:
        return {"error": "supply either query or external_id"}

    term = external_id or query
    logger.info("media_add kind=%s term=%r profile=%r", kind, term, quality_profile)

    try:
        if kind == "movie":
            async with radarr_client() as client:
                matches = await client.lookup(term)
                if not matches:
                    return {"error": f"no Radarr match for {term!r}"}
                chosen = matches[0]
                if (chosen.get("id") or 0) > 0:
                    return {
                        "success": True,
                        "already_in_library": True,
                        "kind": "movie",
                        "library_id": chosen["id"],
                        "title": chosen.get("title"),
                        "year": chosen.get("year"),
                        "message": "Movie already in Radarr library",
                    }
                profile_id, profile_name = await _resolve_quality_profile(
                    client.list_quality_profiles, quality_profile
                )
                root = await _pick_root_folder(client.list_root_folders)
                if not profile_id or not root:
                    return {
                        "error": "missing quality profile or root folder in Radarr",
                        "profile_id": profile_id,
                        "root": root,
                    }
                added = await client.add_movie(
                    chosen,
                    quality_profile_id=profile_id,
                    root_folder_path=root,
                    monitored=monitored,
                    minimum_availability=minimum_availability,
                    search_now=search_now,
                )
                return {
                    "success": True,
                    "kind": "movie",
                    "library_id": added.get("id"),
                    "title": added.get("title"),
                    "year": added.get("year"),
                    "profile": profile_name,
                    "root_folder": root,
                    "monitored": added.get("monitored"),
                    "search_triggered": search_now,
                    "message": f"Added {added.get('title')} ({added.get('year')}) to Radarr",
                }
        else:  # series
            async with sonarr_client() as client:
                matches = await client.lookup(term)
                if not matches:
                    return {"error": f"no Sonarr match for {term!r}"}
                chosen = matches[0]
                if (chosen.get("id") or 0) > 0:
                    return {
                        "success": True,
                        "already_in_library": True,
                        "kind": "series",
                        "library_id": chosen["id"],
                        "title": chosen.get("title"),
                        "year": chosen.get("year"),
                        "message": "Series already in Sonarr library",
                    }
                profile_id, profile_name = await _resolve_quality_profile(
                    client.list_quality_profiles, quality_profile
                )
                root = await _pick_root_folder(client.list_root_folders)
                if not profile_id or not root:
                    return {
                        "error": "missing quality profile or root folder in Sonarr",
                        "profile_id": profile_id,
                        "root": root,
                    }
                added = await client.add_series(
                    chosen,
                    quality_profile_id=profile_id,
                    root_folder_path=root,
                    monitored=monitored,
                    search_now=search_now,
                    monitor=monitor,
                )
                return {
                    "success": True,
                    "kind": "series",
                    "library_id": added.get("id"),
                    "title": added.get("title"),
                    "year": added.get("year"),
                    "profile": profile_name,
                    "root_folder": root,
                    "monitor_mode": monitor,
                    "monitored": added.get("monitored"),
                    "search_triggered": search_now,
                    "message": f"Added {added.get('title')} ({added.get('year')}) to Sonarr",
                }
    except Exception as e:
        return _err(e)


@mcp.tool(name="media_queue")
async def media_queue(limit_per_service: int = 25) -> dict:
    """Combined download queue across Sonarr, Radarr, and SABnzbd.

    Args:
        limit_per_service: Cap records per service so the response stays
            small for chat models.

    Returns:
        ``{sonarr: {total, items: []}, radarr: {...},
           sabnzbd: {paused, speed, free_gb, items: []},
           errors: {service: msg}}``
    """
    out: dict[str, Any] = {"sonarr": None, "radarr": None, "sabnzbd": None, "errors": {}}

    try:
        async with sonarr_client() as sonarr:
            s_queue = await sonarr.queue(page_size=limit_per_service)
            out["sonarr"] = {
                "total": s_queue.get("totalRecords", 0),
                "items": [_condense_sonarr_queue_item(r) for r in s_queue.get("records", [])],
            }
    except Exception as e:
        out["errors"]["sonarr"] = str(e)

    try:
        async with radarr_client() as radarr:
            r_queue = await radarr.queue(page_size=limit_per_service)
            out["radarr"] = {
                "total": r_queue.get("totalRecords", 0),
                "items": [_condense_radarr_queue_item(r) for r in r_queue.get("records", [])],
            }
    except Exception as e:
        out["errors"]["radarr"] = str(e)

    try:
        async with sabnzbd_client() as sab:
            q = await sab.queue(limit=limit_per_service)
            slots = q.get("slots", []) or []
            out["sabnzbd"] = {
                "paused": q.get("paused"),
                "speed": q.get("speed"),
                "size_left": q.get("sizeleft"),
                "time_left": q.get("timeleft"),
                "free_gb": _safe_float(q.get("diskspace1")),
                "total_free_gb": _safe_float(q.get("diskspacetotal1")),
                "active_slots": len(slots),
                "items": [_condense_sab_slot(s) for s in slots],
            }
    except Exception as e:
        out["errors"]["sabnzbd"] = str(e)

    return out


@mcp.tool(name="media_history")
async def media_history(
    kind: str = "all", limit: int = 20, event: str | None = None
) -> dict:
    """Recent activity: grabs, imports, failures.

    Args:
        kind: ``"movie"``, ``"series"``, or ``"all"``.
        limit: Records per service (default 20).
        event: ``"grabbed"``, ``"imported"``, or ``"failed"`` to filter,
            or None for all events.

    Returns:
        ``{sonarr: [{eventType, source_title, date, ...}],
           radarr: [...], errors: {service: msg}}``
    """
    # Map event strings to enum values (different between services!)
    sonarr_event_map = {"grabbed": 1, "imported": 3, "failed": 4}
    radarr_event_map = {"grabbed": 1, "imported": 2, "failed": 3}
    out: dict[str, Any] = {"sonarr": None, "radarr": None, "errors": {}}

    if kind in ("series", "all"):
        try:
            async with sonarr_client() as sonarr:
                evt = sonarr_event_map.get(event) if event else None
                h = await sonarr.history(page_size=limit, event_type=evt)
                out["sonarr"] = [
                    {
                        "title": r.get("sourceTitle"),
                        "event": r.get("eventType"),
                        "date": r.get("date"),
                        "series_id": r.get("seriesId"),
                        "episode_id": r.get("episodeId"),
                        "quality": (r.get("quality") or {}).get("quality", {}).get("name"),
                        "download_id": r.get("downloadId"),
                    }
                    for r in (h.get("records") or [])
                ]
        except Exception as e:
            out["errors"]["sonarr"] = str(e)

    if kind in ("movie", "all"):
        try:
            async with radarr_client() as radarr:
                evt = radarr_event_map.get(event) if event else None
                h = await radarr.history(page_size=limit, event_type=evt)
                out["radarr"] = [
                    {
                        "title": r.get("sourceTitle"),
                        "event": r.get("eventType"),
                        "date": r.get("date"),
                        "movie_id": r.get("movieId"),
                        "quality": (r.get("quality") or {}).get("quality", {}).get("name"),
                        "download_id": r.get("downloadId"),
                    }
                    for r in (h.get("records") or [])
                ]
        except Exception as e:
            out["errors"]["radarr"] = str(e)

    return out


@mcp.tool(name="media_pipeline_status")
async def media_pipeline_status(query: str, kind: str = "auto") -> dict:
    """Trace one title end-to-end across Sonarr/Radarr → SAB → library.

    Useful for "what's happening with X?" — reports library status,
    matching queue items, and recent history events in one payload.

    Args:
        query: Title to trace.
        kind: ``"auto"``, ``"movie"``, or ``"series"``.

    Returns:
        ``{query, library: {...}, queue: {sonarr|radarr|sab matches},
           recent_history: [...], errors: {service: msg}}``
    """
    needle = query.lower().strip()
    out: dict[str, Any] = {
        "query": query,
        "library": [],
        "queue": {"sonarr": [], "radarr": [], "sabnzbd": []},
        "recent_history": [],
        "errors": {},
    }

    # 1) library lookup
    lk = await media_lookup(query, kind=kind, limit=3)  # type: ignore[misc]
    out["library"] = [r for r in lk.get("results", []) if r.get("in_library")]
    out["errors"].update(lk.get("errors") or {})

    # 2) queue scan (only check the services that may match the kind)
    check_series = kind in ("series", "auto")
    check_movie = kind in ("movie", "auto")

    if check_series:
        try:
            async with sonarr_client() as sonarr:
                q = await sonarr.queue(page_size=100)
                for r in q.get("records", []):
                    title = (r.get("title") or "") + " " + ((r.get("series") or {}).get("title") or "")
                    if needle in title.lower():
                        out["queue"]["sonarr"].append(_condense_sonarr_queue_item(r))
        except Exception as e:
            out["errors"]["sonarr"] = str(e)

    if check_movie:
        try:
            async with radarr_client() as radarr:
                q = await radarr.queue(page_size=100)
                for r in q.get("records", []):
                    title = (r.get("title") or "") + " " + ((r.get("movie") or {}).get("title") or "")
                    if needle in title.lower():
                        out["queue"]["radarr"].append(_condense_radarr_queue_item(r))
        except Exception as e:
            out["errors"]["radarr"] = str(e)

    # 3) SAB slot scan
    try:
        async with sabnzbd_client() as sab:
            sq = await sab.queue(limit=100)
            for s in sq.get("slots", []) or []:
                if needle in (s.get("filename") or "").lower():
                    out["queue"]["sabnzbd"].append(_condense_sab_slot(s))
    except Exception as e:
        out["errors"]["sabnzbd"] = str(e)

    # 4) recent history hits
    hist = await media_history(kind=kind, limit=50)  # type: ignore[misc]
    if hist.get("sonarr"):
        out["recent_history"].extend(
            {"source": "sonarr", **h}
            for h in hist["sonarr"]
            if needle in (h.get("title") or "").lower()
        )
    if hist.get("radarr"):
        out["recent_history"].extend(
            {"source": "radarr", **h}
            for h in hist["radarr"]
            if needle in (h.get("title") or "").lower()
        )

    # Stage hint for quick consumption
    if out["queue"]["sabnzbd"]:
        out["stage"] = "downloading_in_sab"
    elif out["queue"]["sonarr"] or out["queue"]["radarr"]:
        out["stage"] = "queued_in_arr"
    elif out["library"]:
        out["stage"] = "in_library"
    elif any(out["queue"].values()):
        out["stage"] = "in_flight"
    else:
        out["stage"] = "not_found"

    return out


@mcp.tool(name="media_library_stats")
async def media_library_stats() -> dict:
    """Quick snapshot of library + downloader state.

    Returns:
        ``{series: {total, queue, missing}, movies: {total, queue, missing},
           sabnzbd: {paused, free_gb, active}, errors: {service: msg}}``
    """
    out: dict[str, Any] = {"series": None, "movies": None, "sabnzbd": None, "errors": {}}

    try:
        async with sonarr_client() as sonarr:
            all_series = await sonarr.list_series()
            s_queue = await sonarr.queue(page_size=1)
            s_missing = await sonarr.missing(page_size=1)
            out["series"] = {
                "total": len(all_series),
                "monitored": sum(1 for s in all_series if s.get("monitored")),
                "queue": s_queue.get("totalRecords", 0),
                "missing": s_missing.get("totalRecords", 0),
            }
    except Exception as e:
        out["errors"]["sonarr"] = str(e)

    try:
        async with radarr_client() as radarr:
            all_movies = await radarr.list_movies()
            r_queue = await radarr.queue(page_size=1)
            r_missing = await radarr.missing(page_size=1)
            out["movies"] = {
                "total": len(all_movies),
                "monitored": sum(1 for m in all_movies if m.get("monitored")),
                "have_file": sum(1 for m in all_movies if m.get("hasFile")),
                "queue": r_queue.get("totalRecords", 0),
                "missing": r_missing.get("totalRecords", 0),
            }
    except Exception as e:
        out["errors"]["radarr"] = str(e)

    try:
        async with sabnzbd_client() as sab:
            q = await sab.queue(limit=1)
            out["sabnzbd"] = {
                "paused": q.get("paused"),
                "speed": q.get("speed"),
                "free_gb": _safe_float(q.get("diskspace1")),
                "active_slots": q.get("noofslots", 0),
            }
    except Exception as e:
        out["errors"]["sabnzbd"] = str(e)

    return out
