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
idle, ``unknown`` when that can't be established).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter

log = logging.getLogger(__name__)
router = APIRouter()

POOL_FILE_ENV = "BAWT_HOUSE_POOL_FILE"


def configured_pool() -> list[dict] | None:
    """``[{id, harness, preview_host}]`` from the pool file; None when not set or unreadable."""
    path = os.environ.get(POOL_FILE_ENV, "").strip()
    if not path:
        return None
    try:
        from agent_bridge.house_pool import validate_pool

        with open(path, encoding="utf-8") as handle:
            raw = handle.read()
        if not raw.strip():
            return None  # the compose default mounts /dev/null: no pool configured
        pool = validate_pool(json.loads(raw))
    except Exception:
        log.warning("house pool file %s unreadable", path, exc_info=True)
        return None
    return [{"id": h["id"], "harness": h["harness"], "preview_host": h["origin"]}
            for h in pool["houses"]]


def house_rows(pool: list[dict] | None, leases: dict[str, set[str]] | None,
               watched: dict[str, str], tenants: dict[str, list[str]]) -> list[dict]:
    """The house list; ``leases`` None means Redis couldn't be read."""
    from ...house_watcher import DOWN, UNKNOWN, UP

    configured = {h["id"]: h for h in pool or []}
    observed = set(tenants) | set(leases or ()) | set(watched)
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
            "preview_host": conf["preview_host"] if conf else None,
        })
    return rows


async def _active(client: Any, rows: list[Any], kinds: dict[str, str] | None
                  ) -> dict[str, Any]:
    """Where each bot's running turn executes.

    House bridges record every admitted turn (``agent:turn:sess:<backend>:
    <bot>:*`` -> house stream); mansion turns are not recorded, so an executing
    bot with no house record runs on its backend's bridge.
    """
    from agent_bridge.houses import HOUSE_STREAM_PREFIX

    from ...bot_location import _bridge, tenancy

    out: dict[str, Any] = {}
    for row in rows:
        routed = tenancy(row.bot_type, row.agent_backend, row.house)
        if routed is None:
            out[row.slug] = None
            continue
        kind = (kinds or {}).get(row.slug)
        if kind in ("idle", "reserved"):  # reserved: admitted, not yet running
            out[row.slug] = None
            continue
        if kind != "executing" or client is None:
            out[row.slug] = "unknown"
            continue
        backend = routed[0]
        try:
            house = None
            async for key in client.scan_iter(match=f"agent:turn:sess:{backend}:{row.slug}:*",
                                              count=100):
                stream = await client.get(key) or ""
                if stream.startswith(HOUSE_STREAM_PREFIX):
                    house = stream[len(HOUSE_STREAM_PREFIX):]
                    break
        except Exception:
            out[row.slug] = "unknown"
            continue
        out[row.slug] = ({"runtime": "house", "house": house} if house
                         else {"runtime": _bridge(backend)})
    return out


def _turn_kinds(engine: Any, slugs: list[str]) -> dict[str, str] | None:
    """``{bot: idle|reserved|executing|ambiguous}`` from turn admission; None if unreadable."""
    try:
        from ..turn_admission import peek_targets

        return {slug: state.kind for slug, state in peek_targets(engine, slugs).items()}
    except Exception:
        log.warning("campus: turn state unavailable", exc_info=True)
        return None


@router.get("/v1/campus", tags=["System"])
async def campus_snapshot():
    import redis.asyncio as aioredis

    from ...bot_location import resolve_locations
    from ...campus_events import SEQ_KEY
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
        rows = sorted(await asyncio.to_thread(store.list_all), key=lambda r: r.slug)
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
        kinds = await asyncio.to_thread(_turn_kinds, store.engine, [r.slug for r in rows])
        active = await _active(client, rows, kinds)
    finally:
        if client is not None:
            await client.aclose()

    watcher = getattr(service, "_house_watcher", None)
    watched = dict(watcher.states.state) if watcher is not None else {}
    tenants: dict[str, list[str]] = {}
    for row in rows:
        if row.house:
            tenants.setdefault(row.house, []).append(row.slug)
    pool = configured_pool()
    return {
        "seq": seq, "ts": ts, "pool_configured": pool is not None,
        "bots": [{"bot_id": row.slug, "assigned": row.house or None,
                  "pinned": bool(row.house_pinned), "route": routes[row.slug],
                  "active": active[row.slug]} for row in rows],
        "houses": house_rows(pool, leases, watched, tenants),
    }
