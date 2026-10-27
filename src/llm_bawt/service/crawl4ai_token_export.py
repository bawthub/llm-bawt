"""Derive the Crawl4AI bearer token and hand it to its readers (TASK-1476).

Crawl4AI checks a static bearer token, and every service that calls it needs
the same value. It is an internal shared secret, so it is **derived** from
the one bootstrap root (``crypto.derive_key(LABEL)``, hex), never stored or
typed into env. The app writes it at startup to each configured export file;
consumers read it through ``agent_bridge.crawl4ai_token``.

Two targets, because the readers run as different uids and the file is 0600:

* ``CRAWL4AI_API_TOKEN_EXPORT_FILE``, chowned to ``CRAWL4AI_API_TOKEN_EXPORT_UID``:
  the Crawl4AI container's copy, read by its entrypoint at boot.
* ``CRAWL4AI_API_TOKEN_BRIDGE_EXPORT_FILE``, chowned to ``BRIDGE_TOKEN_EXPORT_UID``:
  the agent bridges' copy.

An unset path disables that target. With no derivable root the stale file is
removed, so no reader keeps a token the app can no longer produce. Never
raises; never logs a value.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from llm_bawt.bawthub_auth.token_export import publish

logger = logging.getLogger(__name__)

# Versioned so a later scheme can't collide with this one.
LABEL = "crawl4ai-api-token/v1"


@dataclass(frozen=True)
class Target:
    slug: str
    path_env: str
    uid_env: str


TARGETS: tuple[Target, ...] = (
    Target("crawl4ai", "CRAWL4AI_API_TOKEN_EXPORT_FILE", "CRAWL4AI_API_TOKEN_EXPORT_UID"),
    Target("bridges", "CRAWL4AI_API_TOKEN_BRIDGE_EXPORT_FILE", "BRIDGE_TOKEN_EXPORT_UID"),
)


def _derive() -> str:
    from .providers import crypto  # noqa: PLC0415 - loads the root on demand

    return crypto.derive_key(LABEL).hex()


def export_crawl4ai_token(environ: Mapping[str, str] | None = None, *,
                          derive: Callable[[], str] = _derive) -> dict[str, str]:
    """Write (or withdraw) every configured export. Returns ``{target: result}``."""
    env = os.environ if environ is None else environ
    results: dict[str, str] = {}
    configured = [(t, (env.get(t.path_env) or "").strip()) for t in TARGETS]
    if not any(path for _, path in configured):
        return {t.slug: "disabled" for t in TARGETS}
    try:
        token: str | None = derive()
    except Exception as error:  # noqa: BLE001 - any key problem means "no derived token"
        logger.warning("No derived Crawl4AI token: the master key can't derive one (%s)",
                       type(error).__name__)
        token = None
    for target, raw in configured:
        if not raw:
            results[target.slug] = "disabled"
            continue
        try:
            results[target.slug] = publish(Path(raw), token, env, uid_env=target.uid_env).value
        except (OSError, ValueError) as error:
            logger.error("Could not export the Crawl4AI token for %s to %s (%s: %s)",
                         target.slug, raw, type(error).__name__, error)
            results[target.slug] = "failed"
    return results


__all__ = ["LABEL", "TARGETS", "Target", "export_crawl4ai_token"]
