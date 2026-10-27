"""The bearer token our services send to the Crawl4AI container (TASK-1476).

The token is an internal shared secret, never something an operator types:
the app derives it from the one bootstrap root and writes it to shared files
(``llm_bawt.service.crawl4ai_token_export``). Crawl4AI reads its copy at boot
and every consumer (the app's web_fetch tool, the claude-code and codex
bridges' MCP registrations) reads ``CRAWL4AI_API_TOKEN_FILE``.

Reader rule, one place for every consumer:

* ``CRAWL4AI_API_TOKEN_FILE`` set and readable with a value: that value.
* Otherwise the legacy ``CRAWL4AI_API_TOKEN`` env value, with one warning per
  process naming the variable (never the value).
* Otherwise no token, and the consumer sends no ``Authorization`` header.

With no file mounted this is exactly the pre-1476 behaviour, so the code
lands before the compose change that switches Crawl4AI and its consumers to
the derived token together. The file is read on every call: the app rewrites
it when the root rotates. Values are never logged.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

logger = logging.getLogger(__name__)

FILE_ENV = "CRAWL4AI_API_TOKEN_FILE"
LEGACY_ENV = "CRAWL4AI_API_TOKEN"

_warned: set[str] = set()


def _warn_once(key: str, message: str, *args: object) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message, *args)


def _read_file(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip() or None
    except (OSError, UnicodeDecodeError) as exc:
        _warn_once(f"file:{path}", "Crawl4AI token file %s unreadable (%s); using fallback",
                   path, type(exc).__name__)
        return None


def api_token(environ: Mapping[str, str] | None = None) -> str | None:
    """The token to send to Crawl4AI for this call, or ``None`` for none."""
    env = os.environ if environ is None else environ
    path = (env.get(FILE_ENV) or "").strip()
    if path:
        value = _read_file(path)
        if value:
            return value
    legacy = (env.get(LEGACY_ENV) or "").strip()
    if not legacy:
        return None
    _warn_once("legacy", "Crawl4AI token read from deprecated env %s; mount the derived token "
               "and set %s instead", LEGACY_ENV, FILE_ENV)
    return legacy


def auth_headers(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """``{"Authorization": "Bearer <token>"}``, or ``{}`` when there is no token."""
    value = api_token(environ)
    return {"Authorization": f"Bearer {value}"} if value else {}


__all__ = ["FILE_ENV", "LEGACY_ENV", "api_token", "auth_headers"]
