"""House guard: a manual restart/start/stop of a house never cuts a tenant (TASK-2100).

A house (TASK-2077) is a pool bridge container: Compose service ``house-<id>``,
container ``<project>-house-<id>``. Normal recycling needs no ops (the house
restarts itself when vacated, TASK-2080). This guard covers the operator's
deliberate action on one, e.g. a stuck house:

* a container action on a house runs only through an approval card;
* it refuses while the house is occupied: a profile is assigned to it
  (``bot_profiles.house``), a turn admitted there is live (its
  ``agent:turn:req:*`` record names the house stream), or a send on the house
  stream is still unread. The approver may override with the operation's
  ``override_occupied`` argument, which the card shows;
* occupancy is checked when the card is prepared and again right before the
  worker is dispatched. Anything unreadable counts as occupied (fail closed).

Mansion bridges never match: their services and containers carry no
``house-`` segment.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from sqlalchemy import text

from .worker import target_selector

#: Container actions on a house. ``pull`` touches no container.
GUARDED_ACTIONS = frozenset({"start", "restart", "stop"})
OVERRIDE_ARG = "override_occupied"
_HOUSE_ID = r"[a-z0-9][a-z0-9-]{0,30}"
HOUSE_SERVICE_RE = re.compile(rf"house-({_HOUSE_ID})")
_HOUSE_ID_RE = re.compile(_HOUSE_ID)
_PROJECT_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
_CONTAINER_SEP = "-house-"
_TENANTS_SQL = text("SELECT slug FROM bot_profiles WHERE house = :house ORDER BY slug")
_TURN_RECORDS = "agent:turn:req:*"


class HouseOccupancyUnverifiable(Exception):
    """Occupancy could not be read; the guard fails closed."""


def _container_houses(name: str) -> tuple[str, ...]:
    """Every house id a ``<project>-house-<id>`` container name can denote.

    Both the project and the id may contain ``-house-`` themselves (e.g.
    ``llm-bawt-house-dev-house-h1`` is house ``dev-house-h1`` *or* ``h1`` in
    project ``llm-bawt-house-dev``), so a name is not split greedily: every
    legal reading counts and the guard checks all of them (fail closed).
    """
    found: list[str] = []
    start = name.find(_CONTAINER_SEP)
    while start != -1:
        project, house = name[:start], name[start + len(_CONTAINER_SEP):]
        if _PROJECT_RE.fullmatch(project) and _HOUSE_ID_RE.fullmatch(house):
            found.append(house)
        start = name.find(_CONTAINER_SEP, start + 1)
    return tuple(found)


def house_targets(spec: dict, args: dict) -> tuple[str, ...]:
    """The house id(s) a container spec may act on; empty when not a house.

    A Compose service is unambiguous (``house-<id>``); a literal container
    name may have more than one legal reading, and every one is returned.
    """
    if not isinstance(spec, dict) or spec.get("action") not in GUARDED_ACTIONS:
        return ()
    try:
        name, _filters = target_selector(spec, args)
        if name is not None:
            return _container_houses(name)
        match = HOUSE_SERVICE_RE.fullmatch(
            str(spec.get("compose_service") or args[spec["compose_service_from_arg"]]))
    except (KeyError, TypeError):
        return ()
    return (match.group(1),) if match else ()


def override_requested(resolved_args: dict) -> bool:
    return isinstance(resolved_args, dict) and resolved_args.get(OVERRIDE_ARG) is True


def read_occupancy(engine: Any, redis_client: Any, house: str) -> list[str]:
    """Why ``house`` is occupied (empty when free). Raises HouseOccupancyUnverifiable."""
    from agent_bridge.houses import commands_stream

    stream = commands_stream(house)
    reasons: list[str] = []
    try:
        with engine.connect() as conn:
            tenants = list(conn.execute(_TENANTS_SQL, {"house": house}).scalars())
    except Exception as exc:
        raise HouseOccupancyUnverifiable(f"cannot read the house's tenants: {type(exc).__name__}") from exc
    if tenants:
        reasons.append(f"assigned to {', '.join(tenants)}")
    try:
        groups = redis_client.xinfo_groups(stream) if redis_client.exists(stream) else []
        unread = 0
        for group in groups:
            lag = group.get("lag")
            # Redis can't always compute lag; unknown is never "drained".
            unread += 1 if lag is None else int(lag)
        live = sum(1 for key in redis_client.scan_iter(_TURN_RECORDS, count=500)
                   if redis_client.get(key) == stream)
    except Exception as exc:
        raise HouseOccupancyUnverifiable(f"cannot read the house's turns: {type(exc).__name__}") from exc
    if live:
        reasons.append(f"{live} turn(s) running")
    if unread:
        reasons.append(f"{unread} unread send(s) on its stream")
    return reasons


def occupancy_reader(config: Any, engine: Any) -> Callable[[str], list[str]]:
    """``house -> reasons`` over the ops store's DB and the app's Redis."""
    def read(house: str) -> list[str]:
        url = getattr(config, "REDIS_URL", None) if config is not None else None
        if engine is None or not url:
            raise HouseOccupancyUnverifiable("no database or Redis to read occupancy from")
        import redis

        client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2.0,
                                      socket_connect_timeout=2.0)
        try:
            return read_occupancy(engine, client, house)
        finally:
            client.close()
    return read


__all__ = ["GUARDED_ACTIONS", "OVERRIDE_ARG", "HouseOccupancyUnverifiable", "house_targets",
           "occupancy_reader", "override_requested", "read_occupancy"]
