"""Working-memory health report (docs/WORKING_MEMORY.md 14).

Per bot that has ``working_memory_enabled`` stored (protected bots never
appear): whether the policy currently allows it, item counts, namespace
versions, the process-local counters, and the active sessions whose pinned
section is behind the store. Counts and versions only, never item text.

Served at ``GET /v1/working-memory/health`` rather than inside ``/health``,
which is the cheap liveness probe and must not grow per-bot queries.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import text

from . import metrics
from .notice import entry_version
from .policy import SETTING_KEY, WorkingMemoryPolicy, _enabled, is_protected
from .store import WorkingMemoryStore

_PIN_KEY = "system_prompt_pin"  # TASK-1318 pointer (service/system_prompt_pin.py PIN_KEY)


def _opted_in_bots(conn) -> list[str]:
    rows = conn.execute(text(
        "SELECT scope_id, value_json FROM runtime_settings"
        " WHERE scope_type = 'bot' AND key = :key ORDER BY scope_id"
    ), {"key": SETTING_KEY}).all()
    return [str(slug) for slug, raw in rows if _enabled(raw) and not is_protected(str(slug))]


def _counts(conn, bot_id: str) -> dict[str, int]:
    row = conn.execute(text(
        "SELECT count(*) FILTER (WHERE removed_at IS NULL AND archived_at IS NULL) AS open,"
        " count(*) FILTER (WHERE removed_at IS NULL AND archived_at IS NOT NULL) AS archived,"
        " count(*) FILTER (WHERE removed_at IS NOT NULL) AS removed"
        " FROM working_memory_items WHERE bot_id = :bot"
    ), {"bot": bot_id}).mappings().one()
    return {key: int(row[key]) for key in ("open", "archived", "removed")}


def _behind(conn, bot_id: str) -> dict[str, Any]:
    """Active pinned sessions whose section version is below their namespace version.

    A pinned session with no section version (absent, unavailable or
    suppressed) is counted separately: it has no lag to measure.
    """
    versions = dict(conn.execute(text(
        "SELECT user_id, version FROM working_memory_namespaces WHERE bot_id = :bot"
    ), {"bot": bot_id}).all())
    sessions = conn.execute(text(
        "SELECT user_id, session_metadata -> :pin AS pin FROM sessions"
        " WHERE bot_id = :bot AND status = 'active' AND session_metadata ? :pin"
    ), {"bot": bot_id, "pin": _PIN_KEY}).all()
    behind, unversioned, oldest = 0, 0, 0
    for user_id, pin in sessions:
        pinned = entry_version((pin or {}).get("sections") if isinstance(pin, dict) else None)
        if pinned is None:
            unversioned += 1
            continue
        lag = int(versions.get(user_id, 0)) - pinned
        if lag > 0:
            behind += 1
            oldest = max(oldest, lag)
    return {
        "pinned_sessions": len(sessions),
        "behind": behind,
        "oldest_lag": oldest,
        "without_section_version": unversioned,
    }


def bot_report(conn, policy: WorkingMemoryPolicy, bot_id: str) -> dict[str, Any]:
    described = policy.describe(bot_id, conn)
    report: dict[str, Any] = {"bot_id": bot_id, "allowed": described.tools_allowed}
    if not described.tools_allowed:
        report["reason"] = "not_found" if not described.exists else "backend_not_supported"
    namespace = conn.execute(text(
        "SELECT count(*), coalesce(max(version), 0) FROM working_memory_namespaces WHERE bot_id = :bot"
    ), {"bot": bot_id}).one()
    report.update({
        "items": _counts(conn, bot_id),
        "namespaces": int(namespace[0]),
        "max_namespace_version": int(namespace[1]),
        "sessions": _behind(conn, bot_id),
        "since_start": metrics.snapshot(bot_id),
    })
    if not described.tools_allowed:
        report["warning"] = "working_memory_enabled is stored but the policy refuses this bot"
    return report


def health_report(engine) -> dict[str, Any]:
    """The whole report in one read-only transaction; storage failures surface as ``unavailable``."""
    WorkingMemoryStore(engine).ensure_schema()
    policy = WorkingMemoryPolicy(engine)
    with engine.connect() as conn:
        bots = [bot_report(conn, policy, slug) for slug in _opted_in_bots(conn)]
    return {"status": "ok", "bots": bots}


__all__ = ["bot_report", "health_report"]
