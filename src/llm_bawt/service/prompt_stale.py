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

For markers with ``comparison: "instructions"``, the digests and section
names exclude the managed working-memory snapshot. The pin's actual bytes
and digest are independent of this comparison. ``since`` is the first drifting
turn; ``changed_at`` moves only when the instructions change again.

Clearing needs evidence that the harness now holds the current render:

* a confirmed verified boundary (cold start, manual compaction) — the
  unverified fallbacks (missing/corrupt pin, legacy adopt) never clear it,
  because the harness may still hold the old snapshot;
* a kept turn on a *verified* pin whose instructions equal the render again
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

from ..working_memory.prompts import (
    DEFAULT_SECTION, DEFAULT_SECTION_EMPTY, DEFAULT_SECTION_UNAVAILABLE, ITEMS_PLACEHOLDER,
)
from ..working_memory.render import SECTION_NAME

logger = logging.getLogger(__name__)

STALE_KEY = "system_prompt_stale"
#: Section name for text before the first heading.
PREAMBLE = "(preamble)"
MAX_SECTIONS = 20
_HEADING = re.compile(r"^(#{1,3}) +(\S.*?)\s*$")
_INSTRUCTIONS = "instructions"
_LEGACY_MEMORY_HEADING = "Working Memory"
_SEPARATOR = "\n\n"


def working_memory_span(base: str, composed: str, memory: str | None) -> tuple[int, int] | None:
    """Locate the builder's managed tail section in the exact composed bytes.

    Codex leaves the base intact. Claude wraps ``base.rstrip()`` with runtime
    and optional provider instructions; keep those instructions on both sides.
    The section text is supplied by the renderer, never inferred from a heading.
    """
    if not memory:
        return None
    wrapped = composed != base
    if wrapped:
        base, memory = base.rstrip(), memory.rstrip()
    if base == memory:
        preceding = ""
    elif base.endswith(_SEPARATOR + memory):
        preceding = base[:-len(_SEPARATOR + memory)]
    else:
        return None
    offset = composed.find(base)
    if offset < 0 or composed.find(base, offset + 1) >= 0:
        return None  # don't guess if a backend duplicates or rewrites the base
    end = offset + len(base)
    if preceding:
        return offset + len(preceding.rstrip() if wrapped else preceding), end
    if offset and composed[:offset].endswith(_SEPARATOR):
        return offset - len(_SEPARATOR), end
    if composed.startswith(_SEPARATOR, end):
        end += len(_SEPARATOR)
    return offset, end


def legacy_memory_span(prompt: str, sections: Mapping[str, Any] | None) -> tuple[int, int] | None:
    """Recover a pre-upgrade pin's default snapshot, when provably managed.

    Old pin metadata had the section's state but not its bounds. Require the
    default framing for that state, not just a shared Markdown heading. A
    legacy custom template with no identifiable framing is left alone.
    """
    meta = (sections or {}).get(SECTION_NAME)
    if not isinstance(meta, Mapping) or meta.get("v") != 1:
        return None
    state = meta.get("state")
    prefix = {
        "empty": DEFAULT_SECTION_EMPTY,
        "unavailable": DEFAULT_SECTION_UNAVAILABLE,
        "ok": DEFAULT_SECTION.removesuffix(ITEMS_PLACEHOLDER),
    }.get(state) if isinstance(state, str) else None
    if prefix is None:
        return None
    start = prompt.find(prefix)
    if start < 0 or prompt.find(prefix, start + 1) >= 0:
        return None
    end = start + len(prefix)
    if state == "ok":
        end = prompt.find(_SEPARATOR, end)
        if end < 0:
            end = len(prompt)
        try:
            rows = [json.loads(line) for line in prompt[start + len(prefix):end].split("\n")]
        except ValueError:
            return None
        if not all(isinstance(row, dict) and ("id" in row or "omitted" in row) for row in rows):
            return None
    elif end != len(prompt) and not prompt.startswith(_SEPARATOR, end):
        return None
    if start and prompt[:start].endswith(_SEPARATOR):
        return start - len(_SEPARATOR), end
    if start == 0:
        return start, end + (len(_SEPARATOR) if prompt.startswith(_SEPARATOR, end) else 0)
    return None


def _without_snapshot(prompt: str, span: Any) -> str:
    if (isinstance(span, (tuple, list)) and len(span) == 2
            and all(type(offset) is int for offset in span) and 0 <= span[0] < span[1] <= len(prompt)):
        return prompt[:span[0]] + prompt[span[1]:]
    return prompt


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
    if not isinstance(value, dict) or not value.get("since"):
        return None
    # Old markers carry only heading names. Ignore the old default memory-only
    # signature at read time, including on idle sessions that never turn again.
    # Retain mixed/unknown markers and do not apply this heuristic to new ones.
    if "comparison" not in value and value.get("sections") == [_LEGACY_MEMORY_HEADING]:
        return None
    return value


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
                   pinned_span: Any = None, rendered_span: tuple[int, int] | None = None,
                   legacy_sections: Mapping[str, Any] | None = None,
                   now: datetime | None = None) -> None:
    """Update the marker after a kept-pin turn (inside the pin transaction)."""
    raw_marker = metadata.get(STALE_KEY)
    existing = marker_value(raw_marker)

    def act() -> None:
        span = pinned_span if pinned_span is not None else legacy_memory_span(pinned_text, legacy_sections)
        pinned = _without_snapshot(pinned_text, span)
        fresh = _without_snapshot(rendered, rendered_span)
        if rendered_sha == pinned_sha or fresh == pinned:
            if isinstance(raw_marker, dict) and raw_marker.get("since") and (verified or existing is None):
                _write(conn, session_id, None)
            return
        fresh_sha = _digest(fresh)
        held_sha = _digest(pinned)
        previous = existing or {}
        comparable = previous.get("comparison") == _INSTRUCTIONS
        old_sha = previous.get("rendered_sha256")
        same_render = old_sha == (fresh_sha if comparable else rendered_sha)
        if comparable and same_render and previous.get("pinned_sha256") == held_sha:
            return
        stamp = (now or datetime.now(timezone.utc)).isoformat()
        _write(conn, session_id, {
            "comparison": _INSTRUCTIONS,
            "since": previous.get("since") or stamp,
            "changed_at": (previous.get("changed_at") or previous.get("since") or stamp)
            if same_render else stamp,
            "rendered_sha256": fresh_sha,
            "pinned_sha256": held_sha,
            "sections": changed_sections(pinned, fresh),
        })

    _guarded(conn, session_id, act)


def clear_on_verified_boundary(conn, *, session_id: str, metadata: Mapping[str, Any],
                               staged: Mapping[str, Any]) -> None:
    """A confirmed boundary the harness really adopted clears the marker.

    Unverified fallback promotions keep it: the harness may hold old bytes.
    """
    if STALE_KEY not in metadata or staged.get("snapshot_unverified"):
        return
    _guarded(conn, session_id, lambda: _write(conn, session_id, None))


__all__ = ["STALE_KEY", "changed_sections", "clear_on_verified_boundary",
           "marker_value", "reconcile_kept", "settled"]
