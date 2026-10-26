"""Operator freeze: a TTL-bounded deny gate opened and thawed by a person (TASK-1233).

A freeze is an ordinary operator DENY gate (``turn_gates``) owned by exactly
``freeze:<actor>``, where ``actor`` is the person BawtHub signed in. Every
admission edge already refuses under it with 503 ``system_frozen``; nothing
here adds a check to those edges.

* Opened through ``activate_hold``, so it is linearized with every admission
  by the per-bot target locks, and a reservation that never started is
  returned to its producer (``revoked_unstarted``). Runs already executing
  are not cancelled: a freeze stops new work, it does not kill turns.
* One live freeze per actor. A second one while the first is live is refused
  (``FreezeConflict``), so a double-click can't leave a gate behind that the
  actor's Thaw didn't release.
* Ends on Thaw or on expiry. The lease is required and capped at
  ``MAX_FREEZE_TTL_SECONDS``; liveness is the gate predicate on the DB clock,
  so a forgotten freeze lifts on its own with no sweeper running.
* Thaw: the owner thaws their own freeze. An override thaws another actor's
  freeze; who may override is BawtHub's role check, and llm-bawt logs it.
  An override never reaches a job gate (an ops quiesce owns its lifecycle and
  is cancelled through the ops API) or a user-Stop queue hold (Resume).

Freeze, thaw and status are routes, not admission edges, so a frozen system
can always be read and thawed.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import text as sa_text

from .turn_admission import activate_hold
from .turn_gates import (
    DENY,
    OWNER_OPERATOR,
    GateScope,
    GateSpec,
    ensure_gate_schema,
    live_gate_for_owner,
    release_gates,
)

log = logging.getLogger(__name__)

OWNER_PREFIX = "freeze:"
# A freeze covers an operator window: a deploy, a migration, an incident.
# Twelve hours spans one overnight hold; anything longer has been forgotten,
# and the family must be back the next day without anyone remembering to thaw.
MAX_FREEZE_TTL_SECONDS = 12 * 60 * 60
# Shorter than a minute is a typo, not a window.
MIN_FREEZE_TTL_SECONDS = 60
MAX_REASON_CHARS = 500
# The actor is BawtHub's user id (or another stable sign-in id), stored inside
# owner_id (VARCHAR(128)) after the prefix.
_ACTOR = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@+:-]{0,95}")
# Two-key advisory locks are a separate key space from ``lock_target``'s
# single-key ones, so this never collides with a bot's admission lock.
_OWNER_LOCK = "SELECT pg_advisory_xact_lock(hashtext('operator_freeze'), hashtext(:owner))"

_FREEZE_SQL = """
    SELECT id, owner_id, scope, reason, created_at, expires_at, released_at,
           (released_at IS NULL AND expires_at > now()) AS live
    FROM system_gates
    WHERE owner_kind = :kind AND mode = :mode AND starts_with(owner_id, :prefix)
"""


class FreezeUnavailable(RuntimeError):
    """Gates are PostgreSQL-only; there is nothing to freeze on another engine."""


class FreezeConflict(Exception):
    """The actor already has a live freeze."""

    def __init__(self, existing: "Freeze"):
        super().__init__(f"freeze {existing.gate_id} is live until {existing.expires_at.isoformat()}")
        self.existing = existing


class NotFreezeOwner(Exception):
    """Thaw of another actor's freeze without an override."""


def valid_actor(actor: str) -> bool:
    return bool(_ACTOR.fullmatch(actor or ""))


def freeze_owner(actor: str) -> str:
    if not valid_actor(actor):
        raise ValueError("invalid freeze actor")
    return OWNER_PREFIX + actor


@dataclass(frozen=True)
class Freeze:
    gate_id: int
    actor: str
    reason: str | None
    scope: dict[str, Any]
    created_at: datetime
    expires_at: datetime

    @classmethod
    def from_row(cls, row) -> "Freeze":
        scope = json.loads(GateScope.from_json(row["scope"]).to_json())
        return cls(
            gate_id=int(row["id"]), actor=row["owner_id"].removeprefix(OWNER_PREFIX),
            reason=row["reason"], scope=scope,
            created_at=row["created_at"], expires_at=row["expires_at"],
        )

    def to_api(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id, "actor": self.actor, "reason": self.reason,
            "scope": self.scope, "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }


@dataclass(frozen=True)
class ThawResult:
    freeze: Freeze
    released: bool  # False: already released or lapsed before this call
    override: bool  # the thawing actor is not the owner


def _require_postgres(engine) -> None:
    if engine is None or engine.dialect.name != "postgresql":
        raise FreezeUnavailable("system gates require PostgreSQL")
    ensure_gate_schema(engine)


def _params(**extra: Any) -> dict[str, Any]:
    return {"kind": OWNER_OPERATOR, "mode": DENY, "prefix": OWNER_PREFIX, **extra}


def _one(conn, gate_id: int):
    return conn.execute(sa_text(_FREEZE_SQL + " AND id = :id"), _params(id=gate_id)).mappings().first()


def freeze(engine, *, actor: str, reason: str, scope: GateScope, ttl_seconds: int) -> Freeze:
    """Open ``actor``'s freeze; ``FreezeConflict`` when one is already live."""
    if not MIN_FREEZE_TTL_SECONDS <= ttl_seconds <= MAX_FREEZE_TTL_SECONDS:
        raise ValueError(f"ttl_seconds must be {MIN_FREEZE_TTL_SECONDS}..{MAX_FREEZE_TTL_SECONDS}")
    owner = freeze_owner(actor)
    _require_postgres(engine)
    spec = GateSpec(owner_kind=OWNER_OPERATOR, owner_id=owner, mode=DENY, scope=scope,
                    ttl_seconds=ttl_seconds, reason=reason)
    with engine.begin() as conn:
        # Lock order: this owner lock, then the sorted target locks
        # (activate_hold). No other path takes the owner lock.
        conn.execute(sa_text(_OWNER_LOCK), {"owner": owner})
        live = live_gate_for_owner(conn, owner, fresh=True)
        if live is not None:
            raise FreezeConflict(Freeze.from_row(_one(conn, live.id)))
        gate = activate_hold(conn, spec, fresh_clock=True).gate
        frozen = Freeze.from_row(_one(conn, gate.id))
    log.warning("System frozen by %s (gate %s, scope %s, until %s): %s",
                actor, frozen.gate_id, frozen.scope, frozen.expires_at.isoformat(), reason)
    return frozen


def thaw(engine, gate_id: int, *, actor: str, override: bool = False) -> ThawResult | None:
    """Release freeze ``gate_id``; None when it is no freeze at all.

    ``NotFreezeOwner`` when ``actor`` doesn't own it and ``override`` is off.
    Idempotent: a freeze already released or lapsed returns ``released=False``.
    """
    if not valid_actor(actor):
        raise ValueError("invalid freeze actor")
    _require_postgres(engine)
    with engine.begin() as conn:
        row = _one(conn, gate_id)
        if row is None:
            return None
        frozen = Freeze.from_row(row)
        foreign = frozen.actor != actor
        if foreign and not override:
            raise NotFreezeOwner(f"freeze {gate_id} belongs to {frozen.actor}")
        released = bool(release_gates(conn, owner_id=row["owner_id"], gate_id=gate_id)) and row["live"]
    if released:
        log.warning("System thawed by %s (gate %s%s)", actor, gate_id,
                    f", override of {frozen.actor}" if foreign else "")
    return ThawResult(freeze=frozen, released=released, override=foreign)


def live_freezes(engine) -> list[Freeze]:
    """Every live freeze, oldest first (empty off PostgreSQL)."""
    if engine is None or engine.dialect.name != "postgresql":
        return []
    ensure_gate_schema(engine)
    with engine.connect() as conn:
        rows = conn.execute(sa_text(
            _FREEZE_SQL + " AND released_at IS NULL AND expires_at > now() ORDER BY id"
        ), _params()).mappings().all()
    return [Freeze.from_row(row) for row in rows]


__all__ = [
    "MAX_FREEZE_TTL_SECONDS", "MIN_FREEZE_TTL_SECONDS", "MAX_REASON_CHARS", "Freeze",
    "FreezeConflict", "FreezeUnavailable", "NotFreezeOwner", "ThawResult",
    "freeze", "freeze_owner", "live_freezes", "thaw", "valid_actor",
]
