"""TASK-1501: stamp a finished turn with the account quota cached for its route.

The reading is the usage report already cached for the turn's frozen access
path (``token_usage_json.access_path``); no provider call is made. Its
``fetched_at`` can predate the turn by up to the usage cache TTL, so it shows a
trend over time, not an exact per-turn delta. With no cached report the column
stays NULL rather than holding an invented value.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

# A terminal replay long after the turn ended (late usage, recovery) must not
# stamp it with a reading taken well after the fact.
LATE_REPLAY_SECONDS = 600


def _frozen_access_path(token_usage_json: str | None) -> str | None:
    try:
        usage = json.loads(token_usage_json) if token_usage_json else None
    except (TypeError, ValueError):
        return None
    frozen = usage.get("access_path") if isinstance(usage, dict) else None
    key = frozen.get("key") if isinstance(frozen, dict) else None
    return key if isinstance(key, str) and key else None


def quota_snapshot_for_path(access_path: str | None) -> dict[str, Any] | None:
    """The cached report for one access path, reduced to stored quota facts."""
    from .usage import get_cached, usage_adapter_for_access_path

    adapter = usage_adapter_for_access_path(access_path)
    snap = get_cached(adapter.provider) if adapter is not None else None
    if snap is None or not snap.available or snap.fetched_at is None:
        return None
    return {
        "provider": snap.provider,
        "display_name": snap.display_name,
        "kind": snap.kind,
        "fetched_at": snap.fetched_at,
        "limits": [limit.model_dump() for limit in snap.limits],
        "balance": snap.balance.model_dump() if snap.balance is not None else None,
    }


def record_turn_quota(row, *, now: datetime | None = None) -> None:
    """Fill ``row.quota_snapshot_json`` once, on the turn's terminal write."""
    if getattr(row, "quota_snapshot_json", None) is not None or row.ended_at is None:
        return
    ended = row.ended_at if row.ended_at.tzinfo else row.ended_at.replace(tzinfo=timezone.utc)
    if ((now or datetime.now(timezone.utc)) - ended).total_seconds() > LATE_REPLAY_SECONDS:
        return
    try:
        snapshot = quota_snapshot_for_path(_frozen_access_path(row.token_usage_json))
    except Exception:  # noqa: BLE001 - telemetry must never fail the turn's terminal write
        log.warning("quota snapshot skipped for turn %s", row.id, exc_info=True)
        return
    if snapshot is not None:
        row.quota_snapshot_json = json.dumps(snapshot, ensure_ascii=False, default=str)
