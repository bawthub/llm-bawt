"""The deployment's configured house pool, read by the app (TASK-2118/2120).

``BAWT_HOUSE_POOL_FILE`` points at the deployment's ``houses.json``
(docs/HOUSES.md), mounted read-only. Only id, harness and preview origin leave
this module. Unset, empty (the compose default mounts ``/dev/null``) or
unreadable means no pool is configured: callers that hand out houses refuse,
listings fall back to what they observe.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

POOL_FILE_ENV = "BAWT_HOUSE_POOL_FILE"


def configured_pool() -> list[dict] | None:
    """``[{id, harness, preview_host}]`` from the pool file; None when not set or unreadable."""
    path = os.environ.get(POOL_FILE_ENV, "").strip()
    if not path:
        return None
    try:
        from agent_bridge.house_pool import validate_pool

        with open(path, encoding="utf-8") as handle:
            raw = handle.read()
        if not raw.strip():
            return None
        pool = validate_pool(json.loads(raw))
    except Exception:
        logger.warning("house pool file %s unreadable", path, exc_info=True)
        return None
    return [{"id": h["id"], "harness": h["harness"], "preview_host": h["origin"]}
            for h in pool["houses"]]


__all__ = ["POOL_FILE_ENV", "configured_pool"]
