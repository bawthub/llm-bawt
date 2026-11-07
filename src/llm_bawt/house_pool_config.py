"""The deployment's configured house pool, read by the app (TASK-2118/2120).

``BAWT_HOUSE_POOL_FILE`` points at the deployment's ``houses.json``
(docs/HOUSES.md), mounted read-only. Only id, harness, preview origin and the
optional mansion origin leave this module. Unset, empty (the compose default
mounts ``/dev/null``) or unreadable means no pool is configured: callers that
hand out houses refuse, listings fall back to what they observe.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

POOL_FILE_ENV = "BAWT_HOUSE_POOL_FILE"


def _validated_pool() -> dict | None:
    path = os.environ.get(POOL_FILE_ENV, "").strip()
    if not path:
        return None
    try:
        from agent_bridge.house_pool import validate_pool

        with open(path, encoding="utf-8") as handle:
            raw = handle.read()
        if not raw.strip():
            return None
        return validate_pool(json.loads(raw))
    except Exception:
        logger.warning("house pool file %s unreadable", path, exc_info=True)
        return None


def configured_pool() -> list[dict] | None:
    """``[{id, harness, preview_host}]`` from the pool file; None when not set or unreadable."""
    pool = _validated_pool()
    if pool is None:
        return None
    return [{"id": h["id"], "harness": h["harness"], "preview_host": h["origin"]}
            for h in pool["houses"]]


def mansion_origin() -> str | None:
    """The shared dev site's origin (pool ``mansion_origin``), if configured (TASK-2131)."""
    pool = _validated_pool()
    return pool.get("mansion_origin") if pool else None


__all__ = ["POOL_FILE_ENV", "configured_pool", "mansion_origin"]
