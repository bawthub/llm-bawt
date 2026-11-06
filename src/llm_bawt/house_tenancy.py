"""A bot moves itself into a free house, or out of its own (TASK-2118).

Tenancy is the existing ``bot_profiles.house`` field; routing, leases and
fencing are unchanged (``agent_bridge.houses``). This module only decides who
may take which house:

* The caller is always the bot named by its signed turn; there is no target
  argument, so a bot can never set or clear another bot's house here.
* Eligibility is the bot's runtime (an agent on a backend in
  :data:`SELF_MOVE_BACKENDS`), never its name, model or provider.
* Move-in takes only a ``free`` house: one in the deployment's configured pool
  for the bot's harness, holding a live lease for it, not asked to recycle at
  that lease's epoch, with no tenant, no admitted turn still running there and
  no send on its stream unread or unacked (a released bot's work finishes where
  it was sent). Anything else is refused; there is no force path.
* Recycling an emptied house rechecks for tenants and sets its marker under the
  same lock (:func:`llm_bawt.house_lease.request_recycle`), so a claim either
  lands first and the recycle is skipped, or sees ``recycling`` and is refused.
* A bot already housed elsewhere is refused (``already_housed``) and keeps its
  house: it moves out first, so a failed move never strands it.
* Self-move-in never pins. Setting a pin stays a coordinator decision, so
  ``/new`` and the idle janitor still vacate a self-claimed house. A pinned
  bot may still leave voluntarily; that ends its tenancy and the pin with it.

Every write of ``bot_profiles.house`` (self-moves here, a profile PATCH, the
``/new`` eviction, the janitor) takes :data:`HOUSE_LOCK_SQL` inside
:meth:`BotProfileStore.upsert`, and a self-move takes it before reading, so
two bots racing for one house get one winner and one ``occupied`` refusal. The
write goes through ``upsert`` with ``expected_fields``, which audits it.

A move takes effect at the bot's next turn; the current turn finishes where it
was admitted.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text

from agent_bridge.houses import commands_stream, normalize_house

logger = logging.getLogger(__name__)

#: Harnesses whose bridges run in houses today. Other agent backends are
#: refused with :data:`UNSUPPORTED_MESSAGE` until their house bridge exists.
SELF_MOVE_BACKENDS = frozenset({"claude-code"})
UNSUPPORTED_MESSAGE = ("houses support claude-code bots only for now; "
                       "this bot's harness ({backend}) stays in the mansion")

#: One lock for every house assignment write; reentrant within a transaction.
HOUSE_LOCK_SQL = text("SELECT pg_advisory_xact_lock(2118, 1)")
RECYCLE_MARKER_TTL_S = 300
_HOUSED_SQL = text(
    "SELECT slug, house, house_pinned FROM bot_profiles "
    "WHERE house IS NOT NULL AND house <> '' ORDER BY slug")
PRIVATE_TENANT = "(private)"


def recycle_marker_key(house: str, backend: str) -> str:
    """Set to the lease epoch when the app asks that house to recycle."""
    return f"agent:house:{house}:recycle:{backend}"


def lock_house_assignments(conn: Any) -> None:
    if conn.dialect.name == "postgresql":
        conn.execute(HOUSE_LOCK_SQL)


@dataclass(frozen=True)
class HouseState:
    """One read of the configured pool and the routing state in Redis."""

    #: ``{house: {backend: epoch}}`` for every live lease.
    leases: dict[str, dict[str, str]] = field(default_factory=dict)
    #: ``{(house, backend): epoch}`` a recycle was requested at.
    recycling: dict[tuple[str, str], str] = field(default_factory=dict)
    #: Command streams with a turn running, or a send unread or unacked.
    busy_streams: frozenset[str] = frozenset()
    #: ``{house: harness}`` from the deployment's pool; None when none is configured.
    pool: dict[str, str] | None = None

    def status(self, house: str, backend: str, tenants: list) -> str:
        """``free`` | ``occupied`` | ``unconfigured`` | ``incompatible`` | ``down``
        | ``recycling`` | ``draining``."""
        if tenants:
            return "occupied"
        if self.pool is None or house not in self.pool:
            return "unconfigured"
        if self.pool[house] != backend:
            return "incompatible"
        epoch = self.leases.get(house, {}).get(backend)
        if not epoch:
            return "down"
        if self.recycling.get((house, backend)) == epoch:
            return "recycling"
        if commands_stream(house) in self.busy_streams:
            return "draining"
        return "free"


def _house_part(key: str, kind: str) -> tuple[str, str] | None:
    parts = str(key).split(":")  # agent:house:<id>:<kind>:<backend>
    if len(parts) != 5 or parts[3] != kind or not parts[4]:
        return None
    try:
        house = normalize_house(parts[2])
    except ValueError:
        return None
    return (house, parts[4]) if house else None


def _stream_busy(client: Any, stream: str) -> bool:
    """A send nobody has read, or read but not acked (no turn record yet), holds the house.

    Same rule as the ops house guard: lag Redis can't compute is never drained.
    """
    if not client.exists(stream):
        return False
    for group in client.xinfo_groups(stream):
        lag = group.get("lag")
        if lag is None or int(lag) > 0 or int(group.get("pending") or 0) > 0:
            return True
    return False


def read_house_state(client: Any, pool: list[dict] | None = None) -> HouseState:
    """Pool, leases, recycle markers and stream occupancy; raises if Redis is unreadable."""
    leases: dict[str, dict[str, str]] = {}
    recycling: dict[tuple[str, str], str] = {}
    for key in client.scan_iter(match="agent:house:*:lease:*"):
        part = _house_part(key, "lease")
        epoch = client.get(key) if part else None
        if part and epoch:
            leases.setdefault(part[0], {})[part[1]] = str(epoch)
    for key in client.scan_iter(match="agent:house:*:recycle:*"):
        part = _house_part(key, "recycle")
        epoch = client.get(key) if part else None
        if part and epoch:
            recycling[part] = str(epoch)
    busy = set()
    for key in client.scan_iter(match="agent:turn:req:*"):
        stream = client.get(key)
        if stream:
            busy.add(str(stream))
    for house in leases:
        stream = commands_stream(house)
        if stream not in busy and _stream_busy(client, stream):
            busy.add(stream)
    configured = None if pool is None else {h["id"]: h["harness"] for h in pool}
    return HouseState(leases, recycling, frozenset(busy), configured)


StateReader = Callable[[], HouseState]


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


def _tenants(conn: Any) -> dict[str, list[tuple[str, bool]]]:
    out: dict[str, list[tuple[str, bool]]] = {}
    for slug, house, pinned in conn.execute(_HOUSED_SQL):
        out.setdefault(str(house), []).append((str(slug), bool(pinned)))
    return out


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


def _private_bots(store: Any) -> frozenset[str]:
    from .bot_tags import BotTags

    return BotTags(store.engine).snapshot().private_bots  # unavailable: everyone private


def list_houses(store: Any, slug: str, read_state: StateReader,
                private_bots: frozenset[str] | None = None) -> dict[str, Any]:
    """Every known house: live harnesses, tenants, and its status for ``slug``'s harness."""
    row = _caller_row(store, slug)
    backend = (getattr(row, "agent_backend", None) or "").strip().lower()
    state = read_state()
    with store.engine.connect() as conn:
        tenants = _tenants(conn)
    hidden = _private_bots(store) if private_bots is None else private_bots
    houses = []
    for house in sorted(set(state.pool or ()) | set(state.leases) | set(tenants)):
        held = tenants.get(house, [])
        houses.append({
            "house": house,
            "configured": state.pool is not None and house in state.pool,
            "harness": (state.pool or {}).get(house),
            "live": sorted(state.leases.get(house, {})),
            "tenants": [{"bot": s if s == slug or s not in hidden else PRIVATE_TENANT,
                         **({"pinned": True} if p else {})} for s, p in held],
            "status": state.status(house, backend, held),
        })
    try:
        _backend(row)
        eligible: dict[str, Any] = {"eligible": True}
    except HouseMoveRefused as refused:
        eligible = {"eligible": False, "reason": str(refused)}
    return {"you": {"bot": row.slug, "house": row.house or None,
                    **({"pinned": True} if row.house_pinned else {}), **eligible},
            "pool_configured": state.pool is not None,
            "houses": houses}


_REFUSALS = {
    "occupied": "{house} is occupied; houses are never taken from a tenant",
    "unconfigured": "{house} is not in the configured house pool",
    "incompatible": "{house} is configured for another harness than {backend}",
    "down": "{house} has no live {backend} bridge; pick a free house from the list",
    "recycling": "{house} is restarting for a fresh container; try again shortly or pick another",
    "draining": "{house} is still finishing a turn; try again shortly or pick another",
}


def move_in(store: Any, slug: str, house: str | None, read_state: StateReader) -> MoveResult:
    """Claim ``house`` (or any free one) for ``slug``; refuse anything not free."""
    from sqlmodel import Session

    try:
        wanted = normalize_house(house)
    except ValueError as exc:
        raise HouseMoveRefused("invalid_house", str(exc), status_code=400) from exc
    _backend(_caller_row(store, slug))  # refuse early, before taking the lock
    with Session(store.engine) as session:
        conn = session.connection()
        lock_house_assignments(conn)
        row = _caller_row(store, slug)  # reread under the lock
        backend = _backend(row)
        current = (row.house or "").strip() or None
        if current and (not wanted or wanted == current):
            return MoveResult(slug, current, changed=False)
        if current:
            raise HouseMoveRefused(
                "already_housed", f"you live in {current}; move out first", house=current)
        tenants = _tenants(conn)
        state = read_state()
        if state.pool is None:
            raise HouseMoveRefused("no_pool", "no house pool is configured; nothing to move into",
                                   status_code=503)
        if wanted:
            status = state.status(wanted, backend, tenants.get(wanted, []))
            if status != "free":
                raise HouseMoveRefused(
                    "house_unavailable" if status == "down" else status,
                    _REFUSALS[status].format(house=wanted, backend=backend), house=wanted)
            target = wanted
        else:
            free = sorted(h for h in state.pool
                          if state.status(h, backend, tenants.get(h, [])) == "free")
            if not free:
                raise HouseMoveRefused("no_free_house", f"no {backend} house is free right now")
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
    """Vacate ``slug``'s own house (and the pin bound to it); idempotent in the mansion."""
    from sqlmodel import Session

    with Session(store.engine) as session:
        conn = session.connection()
        lock_house_assignments(conn)
        row = _caller_row(store, slug)
        _backend(row)  # same eligibility as move-in; others' tenancy is a coordinator's
        current = (row.house or "").strip() or None
        if not current:
            return MoveResult(slug, None, changed=False)
        # Leaving voluntarily ends the tenancy and the pin bound to it, as any
        # move out of a house already does; nobody else's pin is touched.
        store.upsert({
            "slug": slug, "house": None, "house_pinned": False,
            "expected_fields": {"house": current, "house_pinned": bool(row.house_pinned)},
            "_changed_by": f"bot:{slug}", "change_note": "moved itself out of its house",
        }, session=session, require_existing=True)
        remaining = [s for s, _ in _tenants(conn).get(current, []) if s != slug]
        session.commit()
    logger.info("bot %s moved itself out of house %s", slug, current)
    return MoveResult(slug, None, changed=True, vacated=current, vacated_empty=not remaining)
