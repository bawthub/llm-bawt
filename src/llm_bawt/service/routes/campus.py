"""``GET /v1/campus``: where every bot runs and the house pool, now (TASK-2120).

The Campus view loads this once, then applies ``bot_location_changed`` and
``house_state_changed`` frames from the global stream whose ``seq`` is greater
than this snapshot's (see :mod:`llm_bawt.campus_events`). ``seq`` is read
first, so an event racing the read is applied twice (idempotent), never lost.

Houses are the configured pool (``BAWT_HOUSE_POOL_FILE``, the deployment's
``houses.json``; only id, harness and preview origin are exposed) plus any
house holding a lease or a tenant that the file doesn't list
(``configured: false``). Without a pool file ``pool_configured`` is false and
the list is only what was observed. A Redis failure makes lease-derived fields
``unknown``, never ``down``.

Per bot: ``assigned``/``pinned`` (the profile), ``route`` (where its next turn
would run, TASK-2116) and ``active`` (where its running turn executes: null
idle, ``unknown`` when that can't be established) and ``version`` (see
:func:`llm_bawt.campus_events.version_of`). Private and inactive bots are left
out; a house counts them in ``hidden_tenants``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter

from ...house_pool_config import POOL_FILE_ENV, configured_pool  # noqa: F401

log = logging.getLogger(__name__)
router = APIRouter()

def house_rows(pool: list[dict] | None, leases: dict[str, set[str]] | None,
               watched: dict[str, str], tenants: dict[str, list[str]],
               hidden: dict[str, int] | None = None) -> list[dict]:
    """The house list; ``leases`` None means Redis couldn't be read."""
    from ...house_watcher import DOWN, UNKNOWN, UP

    configured = {h["id"]: h for h in pool or []}
    hidden = hidden or {}
    observed = set(tenants) | set(hidden) | set(leases or ()) | set(watched)
    rows = []
    for house in sorted(set(configured) | observed):
        conf = configured.get(house)
        if leases is None:
            state = UNKNOWN
        elif house in leases:
            state = UP
        else:
            # The watcher knows a release (recreating) from an expiry (down).
            state = watched.get(house) if watched.get(house) not in (None, UP) else DOWN
        harness = conf["harness"] if conf else (sorted(leases.get(house, ()))[0]
                                                 if leases and leases.get(house) else None)
        rows.append({
            "id": house, "harness": harness, "state": state,
            "configured": conf is not None,
            "tenants": sorted(tenants.get(house, [])),
            "hidden_tenants": hidden.get(house, 0),
            "preview_host": conf["preview_host"] if conf else None,
        })
    return rows


async def _active(client: Any, rows: list[Any], turns: dict[str, Any] | None
                  ) -> dict[str, Any]:
    """Where each bot's running turn executes.

    ``turns`` maps a bot to its admission state (``kind``, ``request_id`` of
    the one executing turn). That exact request's route record
    (``agent:turn:req:<id>``, written with its publication) names the stream
    it went to. Only housed bots' turns are recorded, so with no record the
    turn is on its backend's bridge only when the bot was unhoused when the
    turn was admitted (no profile write since); otherwise ``unknown``.
    """
    from agent_bridge.houses import HOUSE_STREAM_PREFIX, turn_request_key

    from ...bot_location import _bridge, tenancy

    out: dict[str, Any] = {}
    for row in rows:
        routed = tenancy(row.bot_type, row.agent_backend, row.house)
        if routed is None:
            out[row.slug] = None
            continue
        turn = (turns or {}).get(row.slug) or {}
        kind = turn.get("kind")
        if kind in ("idle", "reserved"):  # reserved: admitted, not yet running
            out[row.slug] = None
            continue
        request_id = turn.get("request_id")
        if kind != "executing" or client is None or not request_id:
            out[row.slug] = "unknown"
            continue
        try:
            stream = await client.get(turn_request_key(str(request_id)))
        except Exception:
            out[row.slug] = "unknown"
            continue
        if stream and stream.startswith(HOUSE_STREAM_PREFIX):
            out[row.slug] = {"runtime": "house", "house": stream[len(HOUSE_STREAM_PREFIX):]}
        elif stream or (not row.house and _unhoused_since_admission(row, turn)):
            out[row.slug] = {"runtime": _bridge(routed[0])}
        else:
            out[row.slug] = "unknown"
    return out


def _unhoused_since_admission(row: Any, turn: dict) -> bool:
    """No profile write since the turn was admitted, so its route saw this (empty) house."""
    from ...campus_events import version_of

    version = version_of(row)
    admitted = version_of(SimpleNamespace(updated_at=turn.get("created_at")))
    return version is not None and admitted is not None and version < admitted


def _turns(engine: Any, slugs: list[str]) -> dict[str, dict] | None:
    """``{bot: {kind, request_id}}`` from turn admission; None if unreadable."""
    try:
        from ..turn_admission import peek_targets

        out = {}
        for slug, state in peek_targets(engine, slugs).items():
            executing = state.executing
            out[slug] = {"kind": state.kind,
                         "request_id": executing.get("agent_request_id") if executing else None,
                         "created_at": executing.get("created_at") if executing else None}
        return out
    except Exception:
        log.warning("campus: turn state unavailable", exc_info=True)
        return None


@router.get("/v1/campus", tags=["System"])
async def campus_snapshot():
    import redis.asyncio as aioredis

    from ...bot_location import resolve_locations
    from ...campus_events import SEQ_KEY, version_of, visibility
    from ...house_watcher import LEASE_RE
    from ..dependencies import get_bot_profile_store, get_service

    service = get_service()
    store = get_bot_profile_store(service.config)
    client = None
    seq: int | None = None
    leases: dict[str, set[str]] | None = None
    try:
        client = aioredis.from_url(service.config.REDIS_URL, decode_responses=True,
                                   socket_timeout=2.0, socket_connect_timeout=2.0)
        seq = int(await client.get(SEQ_KEY) or 0)
    except Exception:
        log.warning("campus: redis unavailable", exc_info=True)
        client = None
    ts = datetime.now(timezone.utc).isoformat()
    try:
        every = sorted(await asyncio.to_thread(store.list_all), key=lambda r: r.slug)
        shown = await asyncio.to_thread(visibility, service.config)
        rows = [r for r in every if shown(r.slug)]
        if client is not None:
            try:
                found: dict[str, set[str]] = {}
                async for key in client.scan_iter(match="agent:house:*:lease:*", count=200):
                    match = LEASE_RE.match(key)
                    if match:
                        found.setdefault(match.group(1), set()).add(match.group(2))
                leases = found
            except Exception:
                log.warning("campus: lease read failed", exc_info=True)
        routes = await resolve_locations(rows, client.get if client is not None else None)
        turns = await asyncio.to_thread(_turns, store.engine, [r.slug for r in rows])
        active = await _active(client, rows, turns)
    finally:
        if client is not None:
            await client.aclose()

    watcher = getattr(service, "_house_watcher", None)
    watched = dict(watcher.states.state) if watcher is not None else {}
    tenants: dict[str, list[str]] = {}
    hidden: dict[str, int] = {}
    for row in every:
        if row.house:
            if shown(row.slug):
                tenants.setdefault(row.house, []).append(row.slug)
            else:
                hidden[row.house] = hidden.get(row.house, 0) + 1
    pool = configured_pool()
    return {
        "seq": seq, "ts": ts, "pool_configured": pool is not None,
        "bots": [{"bot_id": row.slug, "assigned": row.house or None,
                  "pinned": bool(row.house_pinned), "route": routes[row.slug],
                  "active": active[row.slug], "version": version_of(row)} for row in rows],
        "houses": house_rows(pool, leases, watched, tenants, hidden),
    }
