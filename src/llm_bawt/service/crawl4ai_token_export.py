"""Derive the Crawl4AI bearer token and hand it to its readers (TASK-1476).

Crawl4AI checks a static bearer token, and every service that calls it needs
the same value. It is an internal shared secret, so it is **derived** from
the one bootstrap root with a Crawl4AI-only versioned label, never typed
into env. The nonsecret global ``crawl4ai.token_generation`` adds an independent
rotation domain; zero/absent retains the legacy label. The app writes it at
startup to each configured export file;
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

import json
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from llm_bawt.bawthub_auth.token_export import publish

logger = logging.getLogger(__name__)

# Versioned so a later scheme can't collide with this one.
LABEL = "crawl4ai-api-token/v1"
GENERATION_KEY = "crawl4ai.token_generation"
MAX_GENERATION = 2**31 - 1


@dataclass(frozen=True)
class Target:
    slug: str
    path_env: str
    uid_env: str


TARGETS: tuple[Target, ...] = (
    Target("crawl4ai", "CRAWL4AI_API_TOKEN_EXPORT_FILE", "CRAWL4AI_API_TOKEN_EXPORT_UID"),
    Target("bridges", "CRAWL4AI_API_TOKEN_BRIDGE_EXPORT_FILE", "BRIDGE_TOKEN_EXPORT_UID"),
)


def validate_generation(value: object) -> int:
    """A public rotation counter, never a bearer or a root-key override."""
    if type(value) is not int or not 0 <= value <= MAX_GENERATION:
        raise ValueError("Crawl4AI generation must be an integer in 0..2147483647")
    return value


def current_generation() -> int:
    """Read exactly the global row, without resolver caches or bot overrides.

    An absent row means legacy generation zero. A configured but unavailable DB,
    malformed row or duplicate is an error, never a fallback to an exposed key.
    Startup initializes runtime_settings before calling the exporter.
    """
    from sqlalchemy import text  # noqa: PLC0415

    from llm_bawt.utils.config import Config, has_database_credentials  # noqa: PLC0415
    from llm_bawt.utils.db import get_shared_engine  # noqa: PLC0415

    config = Config()
    if not has_database_credentials(config):
        return 0  # standalone, unconfigured dev use retains legacy behavior
    engine = get_shared_engine(config)
    if engine is None:
        raise RuntimeError("Crawl4AI generation store unavailable")
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT value_json FROM runtime_settings "
            "WHERE scope_type='global' AND scope_id='*' AND key=:key LIMIT 2"),
            {"key": GENERATION_KEY}).fetchall()
    if not rows:
        return 0
    if len(rows) != 1:
        raise ValueError("Crawl4AI generation row ambiguous")
    return validate_generation(json.loads(rows[0][0]))


def derive_for_generation(generation: int) -> str:
    from .providers import crypto  # noqa: PLC0415 - loads the root on demand

    generation = validate_generation(generation)
    label = LABEL if generation == 0 else f"{LABEL}/generation/{generation}"
    return crypto.derive_key(label).hex()


def _derive() -> str:
    return derive_for_generation(current_generation())


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
        logger.warning("Crawl4AI token derivation unavailable (%s)",
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
