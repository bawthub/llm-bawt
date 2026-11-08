"""Stale pinned agent prompts (TASK-1771).

A session keeps the system prompt pinned at its last boundary
(:mod:`.system_prompt_pin`). When a later turn's fresh render differs, the
session is running on superseded instructions. This module records that on
``sessions.session_metadata[STALE_KEY]`` so coordinators can see it
(``agent_context_health``, ``bots_list_available``) and the idle-context job
can refresh the session at a safe boundary.

The marker is content-free::

    {"since": iso, "changed_at": iso, "rendered_sha256": hex,
     "pinned_sha256": hex, "sections": [heading, ...]}

``since`` is the first drifting turn; ``changed_at`` moves only when the
render changes again, so a burst of edits can settle before one reset.

Clearing needs evidence that the harness now holds the current render:

* a confirmed verified boundary (cold start, manual compaction) — the
  unverified fallbacks (missing/corrupt pin, legacy adopt) never clear it,
  because the harness may still hold the old snapshot;
* a kept turn on a *verified* pin whose bytes equal the render again
  (an edit rolled back).

Bookkeeping is optional: it runs in a savepoint and only logs on failure,
so it can never abort the pin transaction or change the pin decision.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from sqlalchemy import text

logger = logging.getLogger(__name__)

STALE_KEY = "system_prompt_stale"
#: Section name for text before the first heading.
PREAMBLE = "(preamble)"
MAX_SECTIONS = 20
_HEADING = re.compile(r"^(#{1,3}) +(\S.*?)\s*$")


def _sections(prompt: str) -> dict[str, str]:
    """Heading -> content digest; repeated headings get a ``#n`` suffix."""
    out: dict[str, str] = {}
    name, lines = PREAMBLE, []

    def close() -> None:
        body = "\n".join(lines)
        if name == PREAMBLE and not body.strip():
            return
        key, n = name, 2
        while key in out:
            key, n = f"{name} #{n}", n + 1
        out[key] = hashlib.sha256(body.encode("utf-8")).hexdigest()

    for line in prompt.splitlines():
        match = _HEADING.match(line)
        if match:
            close()
            name, lines = match.group(2)[:120], []
        else:
            lines.append(line)
    close()
    return out


def changed_sections(pinned: str, rendered: str) -> list[str]:
    """Names (never content) of sections that differ, were added or removed."""
    old, new = _sections(pinned), _sections(rendered)
    names = [key for key in new if old.get(key) != new[key]]
    names += [key for key in old if key not in new]
    if not names and pinned != rendered:
        names = [PREAMBLE]  # whitespace-only or heading-order change
    return names[:MAX_SECTIONS]


def marker_value(value: Any) -> dict | None:
    return value if isinstance(value, dict) and value.get("since") else None


def public_view(value: Any) -> dict | None:
    """``{since, changed_at, sections}`` for health/state readers, or None."""
    marker = marker_value(value)
    if marker is None:
        return None
    return {"since": marker.get("since"), "changed_at": marker.get("changed_at"),
            "sections": list(marker.get("sections") or [])}


def _parse(value: Any) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def settled(marker: Any, *, now: datetime, settle_minutes: int) -> bool:
    """True when the session is stale and its render stopped changing long enough ago."""
    marker = marker_value(marker if not isinstance(marker, str) else _loads(marker))
    if marker is None:
        return False
    changed = _parse(marker.get("changed_at") or marker.get("since"))
    return changed is not None and now - changed >= timedelta(minutes=settle_minutes)


def _loads(value: str) -> Any:
    try:
        return json.loads(value)
    except ValueError:
        return None


def _write(conn, session_id: str, marker: dict | None) -> None:
    if marker is None:
        conn.execute(text(
            "UPDATE sessions SET session_metadata=session_metadata - CAST(:key AS text) WHERE id=:sid"
        ), {"key": STALE_KEY, "sid": session_id})
    else:
        conn.execute(text("""
            UPDATE sessions SET session_metadata=jsonb_set(
                COALESCE(session_metadata, '{}'::jsonb), '{system_prompt_stale}', CAST(:m AS jsonb)
            ) WHERE id=:sid
        """), {"m": json.dumps(marker), "sid": session_id})


def _guarded(conn, session_id: str, action) -> None:
    try:
        with conn.begin_nested():
            action()
    except Exception:
        logger.warning("prompt staleness bookkeeping failed for session %s", session_id, exc_info=True)


def reconcile_kept(conn, *, session_id: str, metadata: Mapping[str, Any], pinned_text: str,
                   pinned_sha: str, rendered: str, rendered_sha: str, verified: bool,
                   now: datetime | None = None) -> None:
    """Update the marker after a kept-pin turn (inside the pin transaction)."""
    existing = marker_value(metadata.get(STALE_KEY))

    def act() -> None:
        if rendered_sha == pinned_sha:
            if existing is not None and verified:
                _write(conn, session_id, None)
            return
        if existing is not None and existing.get("rendered_sha256") == rendered_sha \
                and existing.get("pinned_sha256") == pinned_sha:
            return
        stamp = (now or datetime.now(timezone.utc)).isoformat()
        _write(conn, session_id, {
            "since": (existing or {}).get("since") or stamp,
            "changed_at": stamp if (existing or {}).get("rendered_sha256") != rendered_sha
            else existing.get("changed_at") or stamp,
            "rendered_sha256": rendered_sha,
            "pinned_sha256": pinned_sha,
            "sections": changed_sections(pinned_text, rendered),
        })

    _guarded(conn, session_id, act)


def clear_on_verified_boundary(conn, *, session_id: str, metadata: Mapping[str, Any],
                               staged: Mapping[str, Any]) -> None:
    """A confirmed boundary the harness really adopted clears the marker.

    Unverified fallback promotions keep it: the harness may hold old bytes.
    """
    if marker_value(metadata.get(STALE_KEY)) is None or staged.get("snapshot_unverified"):
        return
    _guarded(conn, session_id, lambda: _write(conn, session_id, None))


__all__ = ["STALE_KEY", "changed_sections", "clear_on_verified_boundary",
           "marker_value", "reconcile_kept", "settled"]
