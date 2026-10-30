"""Short-lived BawtHub browser sessions for bot UI checks (TASK-1659).

With BawtHub sign-in sessions on, a bot's Playwright browser lands on the
sign-in page. Bot bridges hold no service token, so this tool asks BawtHub's
internal port to mint a session for its reserved, non-admin "browser-check"
account on the calling bot's behalf. The bot id comes from the trusted turn
capability, never from a tool argument, so every session is attributed.

The returned cookies are a one-hour bearer for that account only. Inject
them into the browser (Playwright ``context.addCookies``); don't paste them
into chat or files.

Configuration (deployment env, not source):
- ``BAWTHUB_BROWSER_CHECK_INTERNAL_URL``: the BawtHub internal API base whose
  deployment has sign-in on (the frontend's INTERNAL_PORT).
- ``BAWTHUB_BROWSER_CHECK_ORIGIN``: the origin the browser signs in at; the
  cookies are scoped to its host.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

import httpx

from ..bawthub_auth import service_auth
from .registry import mcp
from .task_association import current_task_turn_context

_MINT_PATH = "/internal/auth/browser-check-session"

# TASK-1794: with sign-in sessions off there is nothing to mint, and the UI
# needs no session. A 401 then means "no user selected", not "sign in".
_SESSIONS_OFF = (
    "no browser-check session is needed: sign-in sessions are off on this "
    "deployment. Open the UI directly (a 401 code=user_required means no user "
    "is selected, not a login); for curl, send the user-subject cookie or header."
)


def _config() -> tuple[str, str] | None:
    base = os.getenv("BAWTHUB_BROWSER_CHECK_INTERNAL_URL", "").strip().rstrip("/")
    origin = os.getenv("BAWTHUB_BROWSER_CHECK_ORIGIN", "").strip().rstrip("/")
    if not base or not origin:
        return None
    return base, origin


def _host(origin: str) -> str | None:
    parts = urlsplit(origin)
    return parts.hostname if parts.scheme in ("http", "https") else None


@mcp.tool(name="browser_check_session")
async def browser_check_session() -> dict[str, Any]:
    """Mint a 1h non-admin BawtHub browser session for this bot's UI checks.

    Returns Playwright-ready cookies for the configured sign-in origin. Pass
    them to ``context.addCookies`` before navigating; never echo them.
    """
    config = _config()
    if config is None:
        return {"error": f"browser checks are not configured; {_SESSIONS_OFF}"}
    base, origin = config
    host = _host(origin)
    if not host:
        return {"error": "BAWTHUB_BROWSER_CHECK_ORIGIN is not an http(s) origin"}
    try:
        bot_id = current_task_turn_context().bot_id
    except Exception:
        return {"error": "a browser-check session needs a trusted bot turn"}
    if not bot_id:
        return {"error": "a browser-check session needs a trusted bot turn"}
    try:
        async with httpx.AsyncClient(base_url=base, timeout=10.0, auth=service_auth()) as client:
            response = await client.post(_MINT_PATH, headers={"X-Agent-Bot-Id": bot_id})
            response.raise_for_status()
            minted = response.json()
    except httpx.HTTPStatusError as error:
        if error.response.status_code == 404:
            # The mint route answers 404 whenever sessions are off.
            return {"error": _SESSIONS_OFF, "status": 404}
        try:
            payload = error.response.json()
        except Exception:
            payload = None
        message = payload.get("error") if isinstance(payload, dict) else None
        return {"error": str(message or error), "status": error.response.status_code}
    except httpx.HTTPError as error:
        return {"error": f"BawtHub is unreachable: {type(error).__name__}"}
    cookies = [
        {
            "name": c["name"],
            "value": c["value"],
            "domain": host,
            "path": c.get("path") or "/",
            "expires": c["expires"],
            "httpOnly": bool(c.get("httpOnly", True)),
            "secure": bool(c.get("secure", origin.startswith("https:"))),
            "sameSite": c.get("sameSite") or "Lax",
        }
        for c in minted.get("cookies") or []
    ]
    return {
        "origin": origin,
        "entityId": minted.get("entityId"),
        "expiresAt": minted.get("expiresAt"),
        "cookies": cookies,
    }
