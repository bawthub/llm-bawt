"""A bot moves itself into a free house, or out of its own (TASK-2118).

Tenancy is the existing ``bot_profiles.house`` field; routing, leases and
fencing are unchanged (``agent_bridge.houses``). This module only decides who
may take which house:

* The caller is always the bot named by its signed turn; there is no target
  argument, so a bot can never set or clear another bot's house here.
* Move-in takes a house that holds a live lease for the bot's backend and has
  no tenant. An occupied house is refused, always: there is no force path.
* Self-move-in never pins. A pin stays a coordinator decision, so the idle
  janitor and ``/new`` still vacate a self-claimed house.
* A pinned bot cannot move itself out; the pin's owner unpins it first.

Every self-move runs under one transaction-scoped advisory lock and rechecks
the tenants inside it, so two bots racing for one house get one winner and one
``occupied`` refusal. The profile write itself goes through
:meth:`BotProfileStore.upsert` with ``expected_fields``, which audits it and
refuses a concurrent edit of the same bot.

A move takes effect at the bot's next turn; the current turn finishes where it
was admitted.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from agent_bridge.houses import normalize_house

logger = logging.getLogger(__name__)

#: Harnesses whose bridges run in houses today. Other agent backends are
#: refused with :data:`UNSUPPORTED_MESSAGE` until their house bridge exists.
SELF_MOVE_BACKENDS = frozenset({"claude-code"})
UNSUPPORTED_MESSAGE = ("houses support claude-code bots only for now; "
                       "this bot's harness ({backend}) stays in the mansion")

_LOCK_SQL = text("SELECT pg_advisory_xact_lock(2118, 1)")
_HOUSED_SQL = text(
    "SELECT slug, house, house_pinned FROM bot_profiles "
    "WHERE house IS NOT NULL AND house <> '' ORDER BY slug")
_LEASE_PATTERN = "agent:house:*:lease:*"

#: ``scan(pattern) -> keys``. Raises when Redis can't be read.
LeaseScanner = Callable[[str], Iterable[str]]


class HouseMoveRefused(Exception):
    """A self-move was refused; nothing was written."""

    def __init__(self, code: str, message: str, status_code: int = 409, **extra: Any):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.extra = extra

    def detail(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), **self.extra}


@dataclass(frozen=True)
class MoveResult:
    bot: str
    house: str | None
    changed: bool
    vacated: str | None = None
    #: The vacated house has no tenant left, so it may recycle.
    vacated_empty: bool = False

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"bot": self.bot, "house": self.house, "changed": self.changed}
        if self.vacated:
            out["vacated"] = self.vacated
        if self.changed:
            out["effective"] = "next turn; this turn finishes where it is"
        return out


def live_leases(scan: LeaseScanner) -> dict[str, set[str]]:
    """``{house: {backend, ...}}`` for every house holding a lease now."""
    out: dict[str, set[str]] = {}
    for key in scan(_LEASE_PATTERN):
        parts = str(key).split(":")
        # agent:house:<id>:lease:<backend>
        if len(parts) != 5 or parts[3] != "lease" or not parts[4]:
            continue
        try:
            house = normalize_house(parts[2])
        except ValueError:
            continue
        if house:
            out.setdefault(house, set()).add(parts[4])
    return out


def _tenants(conn: Any) -> dict[str, list[tuple[str, bool]]]:
    out: dict[str, list[tuple[str, bool]]] = {}
    for slug, house, pinned in conn.execute(_HOUSED_SQL):
        out.setdefault(str(house), []).append((str(slug), bool(pinned)))
    return out


def _lock(conn: Any) -> None:
    if conn.dialect.name == "postgresql":
        conn.execute(_LOCK_SQL)


def _caller_row(store: Any, slug: str) -> Any:
    row = store.get(slug)
    if row is None:
        raise HouseMoveRefused("not_found", f"bot {slug!r} has no profile", status_code=404)
    return row


def _backend(row: Any) -> str:
    backend = (getattr(row, "agent_backend", None) or "").strip().lower()
    if getattr(row, "bot_type", None) != "agent" or backend not in SELF_MOVE_BACKENDS:
        raise HouseMoveRefused(
            "unsupported_harness",
            UNSUPPORTED_MESSAGE.format(backend=backend or getattr(row, "bot_type", None) or "chat"),
            status_code=403)
    return backend


def list_houses(store: Any, slug: str, scan: LeaseScanner) -> dict[str, Any]:
    """Every known house with its live harnesses, tenants and whether it is free for ``slug``."""
    row = _caller_row(store, slug)
    backend = (getattr(row, "agent_backend", None) or "").strip().lower()
    leases = live_leases(scan)
    with store.engine.connect() as conn:
        tenants = _tenants(conn)
    houses = []
    for house in sorted(set(leases) | set(tenants)):
        live = sorted(leases.get(house, ()))
        held = tenants.get(house, [])
        houses.append({
            "house": house,
            "live": live,
            "tenants": [{"bot": s, **({"pinned": True} if p else {})} for s, p in held],
            "free": not held and backend in live,
        })
    try:
        _backend(row)
        eligible: dict[str, Any] = {"eligible": True}
    except HouseMoveRefused as refused:
        eligible = {"eligible": False, "reason": str(refused)}
    return {"you": {"bot": row.slug, "house": row.house or None,
                    **({"pinned": True} if row.house_pinned else {}), **eligible},
            "houses": houses}


def move_in(store: Any, slug: str, house: str | None, scan: LeaseScanner) -> MoveResult:
    """Claim ``house`` (or any free one) for ``slug``; refuse an occupied one."""
    from sqlmodel import Session

    try:
        wanted = normalize_house(house)
    except ValueError as exc:
        raise HouseMoveRefused("invalid_house", str(exc), status_code=400) from exc
    _backend(_caller_row(store, slug))  # refuse early, before taking the lock
    with Session(store.engine) as session:
        conn = session.connection()
        _lock(conn)
        row = _caller_row(store, slug)  # reread under the lock
        backend = _backend(row)
        current = (row.house or "").strip() or None
        if current and (not wanted or wanted == current):
            return MoveResult(slug, current, changed=False)
        if current:
            raise HouseMoveRefused(
                "already_housed", f"you live in {current}; move out first", house=current)
        tenants = _tenants(conn)
        live = {h for h, backends in live_leases(scan).items() if backend in backends}
        if wanted:
            if tenants.get(wanted):
                raise HouseMoveRefused(
                    "occupied", f"{wanted} is occupied; houses are never taken from a tenant",
                    house=wanted, tenants=[s for s, _ in tenants[wanted]])
            if wanted not in live:
                raise HouseMoveRefused(
                    "house_unavailable",
                    f"{wanted} has no live {backend} bridge; pick a free house from the list",
                    house=wanted)
            target = wanted
        else:
            free = sorted(live - set(tenants))
            if not free:
                raise HouseMoveRefused("no_free_house", "every live house is occupied")
            target = free[0]
        store.upsert({
            "slug": slug, "house": target, "house_pinned": False,
            "expected_fields": {"house": row.house},
            "_changed_by": f"bot:{slug}", "change_note": "moved itself into a free house",
        }, session=session, require_existing=True)
        session.commit()
    logger.info("bot %s moved itself into house %s", slug, target)
    return MoveResult(slug, target, changed=True)


def move_out(store: Any, slug: str) -> MoveResult:
    """Vacate ``slug``'s own house. A pinned tenancy is the coordinator's to end."""
    from sqlmodel import Session

    with Session(store.engine) as session:
        conn = session.connection()
        _lock(conn)
        row = _caller_row(store, slug)
        current = (row.house or "").strip() or None
        if not current:
            return MoveResult(slug, None, changed=False)
        if row.house_pinned:
            raise HouseMoveRefused(
                "pinned", f"your tenancy of {current} is pinned; ask a coordinator to unpin it",
                house=current)
        store.upsert({
            "slug": slug, "house": None, "house_pinned": False,
            "expected_fields": {"house": current, "house_pinned": False},
            "_changed_by": f"bot:{slug}", "change_note": "moved itself out of its house",
        }, session=session, require_existing=True)
        remaining = [s for s, _ in _tenants(conn).get(current, []) if s != slug]
        session.commit()
    logger.info("bot %s moved itself out of house %s", slug, current)
    return MoveResult(slug, None, changed=True, vacated=current, vacated_empty=not remaining)
