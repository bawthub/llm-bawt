"""The working-memory system-prompt section (docs/WORKING_MEMORY.md 8.1, 12).

``render_section`` produces this turn's section text and its content-free
metadata ``{"v": 1, "version", "state"}``. The metadata goes onto TASK-1318's
pin pointer as ``sections.working_memory``; the notice (9.2) reads it back.

Items are JSON data, one line each, so item text can't pass as instructions
or break the layout. Active then blocked items render in full (at most
``MAX_FULL``), the rest as titles, all within ``CHAR_BUDGET``; whatever doesn't
fit is counted as omitted. The bytes are a function of (items, policy, mode,
render-time age bucket) only, so two renders with no change in between match.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from . import metrics
from .policy import WorkingMemoryPolicy
from .prompts import section_empty_text, section_text, section_unavailable_text
from .session_state import SUPPRESSED, session_mode
from .store import WorkingMemoryStore

logger = logging.getLogger(__name__)

SECTION_NAME = "working_memory"
META_VERSION = 1
OK, EMPTY, UNAVAILABLE = "ok", "empty", "unavailable"
STATES = (OK, EMPTY, UNAVAILABLE, SUPPRESSED)
MAX_FULL = 8
CHAR_BUDGET = 6000  # about 1.5K tokens (12)
FULL_STATUSES = ("active", "blocked")

_AGE_BUCKETS = ((3600, "<1h"), (86400, "<1d"), (7 * 86400, "<7d"))


@dataclass(frozen=True)
class SectionRender:
    """One render: the prompt text (None when nothing renders) and its pin metadata."""

    text: str | None
    version: int | None
    state: str
    meta: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "meta", {"v": META_VERSION, "version": self.version, "state": self.state})


def age_bucket(updated_at: str | None, now: datetime) -> str:
    try:
        then = datetime.fromisoformat(str(updated_at))
    except (TypeError, ValueError):
        return "unknown"
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    seconds = (now - then).total_seconds()
    for limit, label in _AGE_BUCKETS:
        if seconds < limit:
            return label
    return ">=7d"


def _line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def _common(item: dict[str, Any], session_id: str | None, now: datetime) -> dict[str, Any]:
    out: dict[str, Any] = {"age": age_bucket(item.get("updated_at"), now)}
    if item.get("last_session_id") != session_id:
        out["other_session"] = True
    return out


def task_pointer(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "initiative_task_id": item.get("canonical_task_id") or item.get("initiative_task_id"),
        "task_short_id": item.get("task_short_id"),
        "task_title": item.get("task_title"),
        "task_status": item.get("task_status"),
    }


def full_line(item: dict[str, Any], session_id: str | None, now: datetime) -> str:
    value: dict[str, Any] = {
        "id": item["id"], "revision": item["revision"], "status": item["status"],
        "objective": item["objective"],
    }
    if item.get("kind") == "initiative_link":
        value.update(task_pointer(item))
    for key in ("priority", "next_action", "checkpoint"):
        if item.get(key) is not None:
            value[key] = item[key]
    for key in ("blockers", "refs"):
        if item.get(key):
            value[key] = item[key]
    value.update(_common(item, session_id, now))
    return _line(value)


def title_line(item: dict[str, Any], session_id: str | None, now: datetime) -> str:
    value = {"id": item["id"], "status": item["status"], "objective": item["objective"]}
    if item.get("kind") == "initiative_link":
        value.update(task_pointer(item))
    value.update(_common(item, session_id, now))
    return _line(value)


def render_items(items: list[dict[str, Any]], session_id: str | None, now: datetime) -> str:
    """Item lines in store order (active, blocked, planned; by priority), within budget."""
    lines: list[str] = []
    used = 0
    full = 0
    omitted = 0
    for item in items:
        if omitted:
            omitted += 1
            continue
        candidates = []
        if item["status"] in FULL_STATUSES and full < MAX_FULL:
            candidates.append((full_line(item, session_id, now), True))
        candidates.append((title_line(item, session_id, now), False))
        for text, is_full in candidates:
            if used + len(text) + 1 <= CHAR_BUDGET:
                lines.append(text)
                used += len(text) + 1
                full += int(is_full)
                break
        else:
            omitted = 1
    if omitted:
        lines.append(_line({"omitted": omitted}))
    return "\n".join(lines)


def _bot_policy_allows(engine, bot_id: str) -> bool:
    try:
        return WorkingMemoryPolicy(engine).describe(bot_id).tools_allowed
    except Exception:
        # Unreadable policy fails closed: no section, so the prompt stays as before.
        logger.warning("working memory policy read failed for %s; no section", bot_id, exc_info=True)
        return False


def render_section(engine, *, bot_id: str, user_id: str, session_id: str | None,
                   now: datetime | None = None) -> SectionRender | None:
    """This turn's section, or None when the policy excludes the bot (7, 8.1).

    Protected bots are refused before any read. A suppressed session renders
    nothing but still reports its state. A storage failure renders the
    unavailable line, never the empty one, with ``version`` None.
    """
    if not _bot_policy_allows(engine, bot_id):  # tags first: private reads no row
        return None
    try:
        if session_mode(engine, session_id) == SUPPRESSED:
            return SectionRender(None, None, SUPPRESSED)
        from .service import WorkingMemoryService

        WorkingMemoryService(engine).reconcile(bot_id, user_id)
        snapshot = WorkingMemoryStore(engine).render_snapshot(bot_id, user_id)
    except Exception as error:  # WorkingMemoryError("unavailable") or a session-row read failure
        metrics.record_render_failure(bot_id)
        logger.warning("working memory render failed for %s: %s", bot_id, error)
        return SectionRender(section_unavailable_text(bot_id), None, UNAVAILABLE)
    version = snapshot["version"]
    if not snapshot["items"]:
        return SectionRender(section_empty_text(bot_id), version, EMPTY)
    lines = render_items(snapshot["items"], session_id, now or datetime.now(timezone.utc))
    return SectionRender(section_text(bot_id, lines), version, OK)


__all__ = [
    "CHAR_BUDGET", "EMPTY", "MAX_FULL", "OK", "SECTION_NAME", "STATES", "UNAVAILABLE", "SectionRender",
    "age_bucket", "full_line", "render_items", "render_section", "title_line",
]
