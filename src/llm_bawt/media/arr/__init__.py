"""Async clients for the *arr media stack.

Wraps Sonarr (TV), Radarr (movies), and SABnzbd (download client) HTTP
APIs behind small, typed async clients. The MCP server's ``media_*``
tools layer on top of these to give bots a unified media library
interface.

Credentials (URL + API key) come from each service's encrypted
``integration_connection:<service>`` record, saved under Integrations.
Only when no record exists are the deprecated env vars read:

    LLM_BAWT_SONARR_URL / LLM_BAWT_SONARR_API_KEY
    LLM_BAWT_RADARR_URL / LLM_BAWT_RADARR_API_KEY
    LLM_BAWT_SABNZBD_URL / LLM_BAWT_SABNZBD_API_KEY

``await get_*_client()`` resolves credentials on every call and reuses one
``httpx.AsyncClient`` per service until they change.
"""

from .base import ArrAPIError, ArrNotConfigured, ArrUnavailable
from .radarr import RadarrClient, get_radarr_client
from .sabnzbd import SabnzbdClient, get_sabnzbd_client
from .sonarr import SonarrClient, get_sonarr_client

__all__ = [
    "ArrAPIError",
    "ArrNotConfigured",
    "ArrUnavailable",
    "RadarrClient",
    "SabnzbdClient",
    "SonarrClient",
    "get_radarr_client",
    "get_sabnzbd_client",
    "get_sonarr_client",
]
