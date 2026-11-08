"""Per-call usage reporting for app-owned clients (TASK-1733).

The background client factory sets ``client.usage_observer`` so every billed
maintenance call reaches the spend ledger; chat turns are ledgered from their
turn logs instead, so their clients carry no observer. A plain function, not a
base-class method: client modules load lazily into a running app, and a call
site must not depend on a base class imported before it changed (TASK-1281).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def report_usage(client: Any, response: Any) -> None:
    """Hand a completed non-streaming response to ``client.usage_observer``.

    Never raises: a spend-recording failure must not fail the call that paid.
    """
    if response is None:
        return
    try:
        # TASK-1538: the open utility call (any path) counts this response too.
        from ..service.usage.utility_usage import note_response
        note_response(response)
    except Exception:  # noqa: BLE001
        logger.warning("utility usage capture failed", exc_info=True)
    observer = getattr(client, "usage_observer", None)
    if observer is None:
        return
    try:
        observer(response)
    except Exception:  # noqa: BLE001
        logger.warning("usage observer failed for %s", getattr(client, "model", "?"), exc_info=True)
