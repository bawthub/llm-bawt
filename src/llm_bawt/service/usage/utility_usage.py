"""One usage row per utility-model completion (TASK-1538).

``complete_utility`` (recall synthesis, session titles, media prompt expansion,
``/v1/llm/complete`` summaries and consolidation) writes one ``utility_usage``
row per call on every access path, subscription and local included. Spend on
pay-per-token paths stays in ``api_balance_ledger`` (TASK-1733); this table
answers "how many tokens do utility calls burn, and for whom".

Token counts come from the provider response that the client hands to
``report_usage``: :func:`capture` opens a per-call box and :func:`note_response`
fills it. A call whose provider reported nothing is stored with an estimate and
``estimated=true``. No prompt, query, evidence or completion text is stored.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

from sqlalchemy import text

from .background_spend import normalize_usage

logger = logging.getLogger(__name__)

TOKEN_KEYS = ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens")
MAX_ROLLUP_DAYS = 90

_CALL: ContextVar[dict | None] = ContextVar("utility_usage_call", default=None)


@contextmanager
def capture() -> Iterator[dict]:
    """Collect provider usage reported during one utility call."""
    box: dict = {}
    token = _CALL.set(box)
    try:
        yield box
    finally:
        _CALL.reset(token)


def note_response(response: Any) -> None:
    """Add a completed response's provider usage to the open call, if any.

    Summed, so a client that makes more than one request for a call counts all
    of them. Never raises.
    """
    box = _CALL.get()
    if box is None or response is None:
        return
    try:
        usage = normalize_usage(response)
    except Exception:  # noqa: BLE001
        return
    if not usage or "input_tokens" not in usage:
        return
    totals = box.setdefault("usage", dict.fromkeys(TOKEN_KEYS, 0))
    for key in TOKEN_KEYS:
        totals[key] += int(usage.get(key) or 0)


def bootstrap_utility_usage_schema(conn) -> None:
    if getattr(getattr(conn, "dialect", None), "name", None) != "postgresql":
        return
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS utility_usage (
            id BIGSERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL,
            caller VARCHAR(64) NOT NULL,
            bot_id VARCHAR(64),
            model VARCHAR(128),
            input_tokens INTEGER NOT NULL DEFAULT 0,
            cache_read_tokens INTEGER NOT NULL DEFAULT 0,
            cache_write_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            estimated BOOLEAN NOT NULL,
            elapsed_ms INTEGER NOT NULL,
            outcome VARCHAR(32) NOT NULL
        )
    """))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_utility_usage_created ON utility_usage (created_at)"
    ))


def build_row(*, caller: str, bot_id: str | None, model: str | None, box: dict,
              estimate: dict[str, int] | None, elapsed_ms: float, outcome: str) -> dict:
    """The stored row: provider usage when reported, else the flagged estimate."""
    usage = box.get("usage")
    estimated = usage is None
    tokens = usage if usage is not None else {**dict.fromkeys(TOKEN_KEYS, 0), **(estimate or {})}
    return {
        "created_at": datetime.now(timezone.utc), "caller": caller[:64],
        "bot_id": bot_id[:64] if bot_id else None, "model": model[:128] if model else None,
        **{key: int(tokens.get(key) or 0) for key in TOKEN_KEYS},
        "estimated": estimated, "elapsed_ms": max(int(elapsed_ms), 0), "outcome": outcome[:32],
    }


def _default_engine():
    from ..dependencies import get_service

    return get_service()._turn_log_store.engine


def record(row: dict, *, engine: Any = None) -> None:
    """Insert one row. Never raises: recording must not fail the call that ran."""
    try:
        engine = engine if engine is not None else _default_engine()
        if engine is None or engine.dialect.name != "postgresql":
            return
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO utility_usage (created_at, caller, bot_id, model, input_tokens,
                    cache_read_tokens, cache_write_tokens, output_tokens, estimated, elapsed_ms, outcome)
                VALUES (:created_at, :caller, :bot_id, :model, :input_tokens, :cache_read_tokens,
                    :cache_write_tokens, :output_tokens, :estimated, :elapsed_ms, :outcome)
            """), row)
    except Exception:  # noqa: BLE001
        logger.warning("utility usage row not recorded (%s)", row.get("caller"), exc_info=True)


def rollup(conn, *, days: int, now: datetime | None = None) -> list[dict]:
    """Per UTC day x caller totals for the last ``days`` days, newest first."""
    days = min(max(int(days), 1), MAX_ROLLUP_DAYS)
    now = now or datetime.now(timezone.utc)
    since = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    rows = conn.execute(text("""
        SELECT (created_at AT TIME ZONE 'UTC')::date AS day, caller,
               COUNT(*) AS calls,
               COALESCE(SUM(input_tokens), 0) AS input_tokens,
               COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
               COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
               COALESCE(SUM(output_tokens), 0) AS output_tokens,
               COUNT(*) FILTER (WHERE estimated) AS estimated_calls,
               COUNT(*) FILTER (WHERE outcome <> 'ok') AS failed_calls
        FROM utility_usage WHERE created_at >= :since
        GROUP BY 1, 2 ORDER BY 1 DESC, 2
    """), {"since": since}).mappings().all()
    return [{**{k: int(v) for k, v in row.items() if k not in ("day", "caller")},
             "day": row["day"].isoformat(), "caller": row["caller"]} for row in rows]
