"""Bell entry for a new Playground artifact; best-effort.

BawtHub's internal route upserts on the artifact id, so a retry can't add a
second entry. Logs carry the artifact id and a status, never the title.
"""
from __future__ import annotations

import logging

import httpx

from ..integrations import notify_bell

logger = logging.getLogger(__name__)

BELL_PATH = "/internal/notifications/playground"
LINK = "/bots/playground"  # BawtHub pins the bell link to this page regardless


async def record_artifact_bell(http: httpx.AsyncClient, *, artifact_id: str, bot_id: str,
                               user_id: str, sender: str, title: str) -> bool:
    body = {"artifactId": artifact_id, "botId": bot_id, "userId": user_id,
            "title": f"{sender} sent you an artifact", "message": title, "link": LINK}
    try:
        response = await http.post(BELL_PATH, json=body)
    except httpx.HTTPError as error:
        logger.warning("playground bell %s not recorded: %s", artifact_id, type(error).__name__)
        return False
    if not response.is_success:
        logger.warning("playground bell %s not recorded: HTTP %s", artifact_id,
                       response.status_code)
        return False
    return True


def bell_http() -> httpx.AsyncClient:
    return notify_bell.bell_http()
