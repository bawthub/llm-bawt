"""Transport facade for already-encoded bot/user UI events.

The caller owns JSON encoding, client lifetime and best-effort error reporting.
Transport errors propagate to that boundary. Returning XADD directly preserves
both synchronous writes and awaited asynchronous writes without scheduling work.
"""

from __future__ import annotations

from .publisher import UNIFIED_EVENTS_PREFIX, UNIFIED_STREAM_MAXLEN


def publish_user_event(client, bot_id: str, user_id: str, payload: str):
    """Append exactly one payload using the shared user-stream contract."""
    return client.xadd(
        f"{UNIFIED_EVENTS_PREFIX}{bot_id}:{user_id}",
        {"payload": payload},
        maxlen=UNIFIED_STREAM_MAXLEN,
        approximate=True,
    )
