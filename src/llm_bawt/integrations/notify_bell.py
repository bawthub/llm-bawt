"""Bell history for phone pushes (TASK-2002).

A push that reached Home Assistant (``sent`` or ``partial:*``) is also
recorded as one BawtHub notification-center entry, so the bell is the push
history. Recording is best-effort: the push already happened, so a BawtHub
failure never changes the push result. The internal route upserts on the
push id, so a retry can't add a second entry. Logs carry the push id and an
HTTP status or error type, never the pushed text.
"""
from __future__ import annotations

import logging
import os
from urllib.parse import urlsplit, urlunsplit

import httpx

logger = logging.getLogger(__name__)

BELL_PATH = "/internal/notifications/push"
_TIMEOUT = httpx.Timeout(5.0)


def bell_base_url() -> str:
    return (os.getenv("BAWTHUB_NOTIFICATIONS_INTERNAL_URL")
            or os.getenv("BAWTHUB_TASK_ASSOCIATION_INTERNAL_URL")
            or "http://frontend-prod:3002").rstrip("/")


def bell_http() -> httpx.AsyncClient:
    from llm_bawt.bawthub_auth import service_auth
    return httpx.AsyncClient(base_url=bell_base_url(), timeout=_TIMEOUT, auth=service_auth())


def delivered(result: str) -> bool:
    """Whether at least one phone received the push."""
    return result == "sent" or result.startswith("partial:")


def bell_link(url: str | None) -> str | None:
    """The app path of an already-validated push link, for in-app navigation."""
    if not url:
        return None
    if url.startswith("/") and not url.startswith("//"):
        return url
    parts = urlsplit(url)
    return urlunsplit(("", "", parts.path or "/", parts.query, parts.fragment))


async def record_push_bell(http: httpx.AsyncClient, *, push_id: str, bot_id: str,
                           user_id: str, title: str, message: str, urgency: str,
                           link: str | None, result: str) -> bool:
    """POST one bell entry for a delivered push; False (logged) on any failure."""
    body = {"pushId": push_id, "botId": bot_id, "userId": user_id, "title": title,
            "message": message, "urgency": urgency, "link": bell_link(link),
            "result": result}
    try:
        response = await http.post(BELL_PATH, json=body)
    except httpx.HTTPError as error:
        logger.warning("notify bell %s not recorded: %s", push_id, type(error).__name__)
        return False
    if not response.is_success:
        logger.warning("notify bell %s not recorded: HTTP %s", push_id, response.status_code)
        return False
    return True
