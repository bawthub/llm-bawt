"""Live bot location and house state events for the Campus view (TASK-2120).

Two frames on the global unified stream (``events:system:__config__``), both
stamped from one Redis counter that the ``/v1/campus`` snapshot also reads, so
a client drops events with ``seq <= snapshot.seq`` and refetches once on a gap:

* ``bot_location_changed {event_id, seq, ts, kind, bot_id, from, to, reason}``;
  ``from``/``to`` are :mod:`bot_location` next-turn routes (null for a housed
  profile created or deleted). ``kind`` is ``assignment`` (a committed
  house/pin write) or ``route`` (the house's lease came or went; no write).
  Neither moves a turn already running: that stays where it was admitted.
  ``version`` is the bot's committed location version (see :func:`version_of`):
  frames can arrive out of commit order, so a client applies a frame only when
  its ``version`` is at least the one it holds for that bot (snapshot included).
* ``house_state_changed {event_id, seq, ts, house_id, from, to, tenants}``;
  states are ``up`` / ``down`` / ``recreating``.

Tenancy changes are captured at the one place a bot's ``house`` is written,
``BotProfileStore.upsert``, via :func:`note_tenancy_change`. The frame is sent
from the session's ``after_commit``, so it fires for borrowed sessions too
(whoever commits) and a rollback sends nothing. House state transitions come
from :mod:`house_watcher`. Publishing is best effort: a lost frame shows up as
a ``seq`` gap and the client refetches the snapshot. Private and inactive bots
are never named (same policy as ``bots_list_available``); a visibility change
arrives as ``bot_profile_changed``, on which the client refetches.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import event

log = logging.getLogger(__name__)

SEQ_KEY = "campus:seq"
BOT_LOCATION_CHANGED = "bot_location_changed"
HOUSE_STATE_CHANGED = "house_state_changed"
#: ``kind``: a committed change to the bot's assigned house/pin, or a change of
#: its next-turn route with no profile write (the house's lease came or went).
ASSIGNMENT, ROUTE = "assignment", "route"
ASSIGNMENT_REASONS = frozenset({"move_in", "move_out", "new_session", "idle_evict",
                                "pinned_change"})
ROUTE_REASONS = frozenset({"house_down_fallback", "house_up"})
#: Optional upsert payload key naming why the tenancy changed.
REASON_KEY = "_location_reason"

_PENDING = "campus_location_changes"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redis(config: Any = None):
    import redis

    if config is None:
        from .utils.config import Config
        config = Config()
    url = getattr(config, "REDIS_URL", None)
    if not url:
        return None
    return redis.Redis.from_url(url, decode_responses=True,
                                socket_timeout=2.0, socket_connect_timeout=2.0)


def publish(client: Any, frames: list[dict]) -> None:
    """Stamp each frame with the next ``seq`` and append it to the global stream."""
    from agent_bridge.publisher import GLOBAL_CONFIG_STREAM, UNIFIED_STREAM_MAXLEN

    for frame in frames:
        frame = {"event_id": uuid.uuid4().hex, "seq": int(client.incr(SEQ_KEY)),
                 "ts": _now(), **frame}
        client.xadd(GLOBAL_CONFIG_STREAM, {"payload": json.dumps(frame)},
                    maxlen=UNIFIED_STREAM_MAXLEN, approximate=True)


def publish_frames(frames: list[dict], config: Any = None) -> None:
    if not frames:
        return
    try:
        client = _redis(config)
        if client is None:
            return
        try:
            publish(client, frames)
        finally:
            client.close()
    except Exception:
        log.warning("campus event publish failed (%d frames)", len(frames), exc_info=True)


def infer_reason(before: dict, after: dict) -> str:
    """Reason for a tenancy write that didn't name one."""
    if (before.get("house") or None) != (after.get("house") or None):
        return "move_in" if after.get("house") else "move_out"
    return "pinned_change"


def version_of(row: Any) -> int | None:
    """A bot's committed location version: its ``updated_at`` in microseconds.

    Writes to one bot serialize under its tag lock, so a later commit carries a
    larger version; a client keeps the frame with the highest version per bot.
    """
    stamp = getattr(row, "updated_at", None)
    if stamp is None:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return int(stamp.timestamp() * 1_000_000)


def visibility(config: Any = None):
    """``slug -> bool``: the bot may be shown (not private or inactive).

    The same policy as ``bots_list_available``; an unreadable policy hides all.
    """
    from .bot_tags.facade import BotTags

    engine = None
    try:
        from .utils.db import get_shared_engine

        if config is None:
            from .utils.config import Config
            config = Config()
        engine = get_shared_engine(config)
    except Exception:
        log.warning("campus visibility: no engine", exc_info=True)
    snapshot = BotTags(engine).snapshot()
    return lambda slug: snapshot.excluded_reason(slug) is None


def location_frame(bot_id: str, before: dict, after: dict, reason: str,
                   live: dict[str, bool | None], version: int | None = None) -> dict | None:
    """The frame for one profile write; None when the location didn't change.

    ``before``/``after`` hold ``bot_type, agent_backend, house, house_pinned``;
    ``live`` maps a lease key to its liveness.
    """
    from agent_bridge.houses import lease_key

    from .bot_location import location_of, tenancy

    def loc(fields: dict) -> dict:
        routed = tenancy(fields.get("bot_type"), fields.get("agent_backend"), fields.get("house"))
        lease = live.get(lease_key(routed[1], routed[0])) if routed and routed[1] else None
        return location_of(fields.get("bot_type"), fields.get("agent_backend"),
                           fields.get("house"), fields.get("house_pinned"), lease)

    def frame(old: dict | None, new: dict | None, why: str) -> dict:
        return {"_type": BOT_LOCATION_CHANGED, "kind": ASSIGNMENT, "bot_id": bot_id,
                "from": old, "to": new, "reason": why, "version": version}

    # A profile created into, or deleted from, a house; mansion bots appear and
    # disappear through the ordinary bot_profile_changed frame.
    if not before:
        new = loc(after)
        return frame(None, new, "move_in") if new.get("house") else None
    if not after:
        old = loc(before)
        return frame(old, None, "move_out") if old.get("house") else None
    old, new = loc(before), loc(after)
    if old == new:
        return None
    return frame(old, new, reason if reason in ASSIGNMENT_REASONS else infer_reason(before, after))


TENANCY_FIELDS = ("bot_type", "agent_backend", "house", "house_pinned")


def tenancy_fields(row: Any) -> dict:
    return {name: getattr(row, name, None) for name in TENANCY_FIELDS} if row is not None else {}


def note_tenancy_change(session: Any, bot_id: str, before: dict, after: dict,
                        reason: str | None, version: int | None = None) -> None:
    """Queue a location frame for this write; it is sent after ``session`` commits."""
    if {k: before.get(k) for k in TENANCY_FIELDS} == {k: after.get(k) for k in TENANCY_FIELDS}:
        return
    # Listeners can't be removed while SQLAlchemy runs them, so each session
    # gets them once and they do nothing when no change is pending.
    if not event.contains(session, "after_commit", _flush):
        event.listen(session, "after_commit", _flush)
        event.listen(session, "after_rollback", _drop)
    session.info.setdefault(_PENDING, []).append(
        (bot_id, dict(before), dict(after), reason or "", version))


def _drop(session: Any) -> None:
    session.info.pop(_PENDING, None)


def _flush(session: Any) -> None:
    pending = session.info.pop(_PENDING, None) or []
    if pending:
        emit_tenancy_changes(pending)


def emit_tenancy_changes(pending: list[tuple], config: Any = None) -> None:
    from agent_bridge.houses import lease_key

    from .bot_location import tenancy

    try:
        client = _redis(config)
        if client is None:
            return
        try:
            live: dict[str, bool | None] = {}
            for _, before, after, *_ in pending:
                for fields in (before, after):
                    routed = tenancy(fields.get("bot_type"), fields.get("agent_backend"),
                                     fields.get("house"))
                    if routed and routed[1]:
                        key = lease_key(routed[1], routed[0])
                        if key not in live:
                            live[key] = bool(client.get(key))
            shown = visibility(config)
            frames = [f for bot_id, before, after, reason, version in pending
                      if shown(bot_id)
                      and (f := location_frame(bot_id, before, after, reason, live, version))]
            publish(client, frames)
        finally:
            client.close()
    except Exception:
        log.warning("campus location publish failed for %s",
                    [p[0] for p in pending], exc_info=True)
