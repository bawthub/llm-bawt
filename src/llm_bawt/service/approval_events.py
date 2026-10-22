"""The ``approval_resolved`` event clients finalize an approval card from.

One publisher for every writer that resolves a request: the decide route
(``routes/approval_policies.py``) and the operator purge
(``recovery/purge_ops.py``).
"""

from __future__ import annotations

import json
import logging
import time

log = logging.getLogger(__name__)


async def publish_approval_resolved(
    subscriber,
    bot_id,
    user_id,
    request_id,
    turn_id,
    status,
    *,
    tool_use_id=None,
    result=None,
    is_error=None,
    execution_state=None,
    has_result=True,
) -> None:
    """Fan out resolution and, for MCP, its terminal original-call result.

    ``has_result=False`` (an approved ops_run awaiting admission) omits the
    ``result`` key entirely: a client that sees ``result: null`` would
    finalize the card as a null success (TASK-1220).
    """
    if subscriber is None:
        return
    event = {
        "_type": "approval_resolved",
        "bot_id": bot_id,
        "user_id": user_id,
        "request_id": request_id,
        "turn_id": turn_id,
        "status": status,
        "tool_use_id": tool_use_id,
        "result": result,
        "is_error": is_error,
        "execution_state": execution_state,
        "ts": time.time(),
    }
    if not has_result:
        del event["result"], event["is_error"]
    try:
        await subscriber._redis.xadd(
            f"events:{bot_id}:{user_id}",
            {"payload": json.dumps(event, ensure_ascii=False, default=str)},
            maxlen=5000,
            approximate=True,
        )
    except Exception:  # noqa: BLE001
        log.debug("failed to publish approval_resolved for %s", request_id, exc_info=True)
